#!/usr/bin/env python3
"""
probe_n8_bridge_jitter.py
=========================

Diagnostic probe script for the OCCT *silent-drop* bug observed when
generating the biomimetic enamel lattice at the new pipeline defaults
(``DEFAULT_BRIDGE_RATIO = 0.8``, ``DEFAULT_PLATE_CLEARANCE_MM = 0.5``;
ROD_DIAMETER ≈ 2.128 mm, BRIDGE_DIAMETER ≈ 1.702 mm) with
``N_BRIDGE_LAYERS = 8``.

Background
----------
At the new defaults, N = 2, 4, 6, 7, 9 all generate clean watertight
lattices via :mod:`generators.cad_runner` → ``lattice_cad.py``.  N = 8
*looks* clean — STL writes successfully, ``is_watertight`` is True — but
silently drops all 8 horizontal bridge layers from the OCCT BOOLEAN fuse,
yielding ~12,053 mm³ of rods + plates only (vs ~14,300 mm³ expected with
bridges). The bug manifests only at specific positions of the bridge
elevations against the helically twisted rods: a *tangent resonance* in
OCCT's boolean-fusion algorithm where the bridge cylinders graze the rod
surface and disappear from the output without raising an error.

Prior empirical probes (this session, 2026-05-27) established the
bracket:

    +0.01 mm uniform Z-shift  →  STILL BROKEN  (STL byte-identical to default)
    +0.50 mm uniform Z-shift  →  FIXED          (8 of 8 bridges present)

so the minimum jitter that escapes the resonance lies somewhere in
(0.01, 0.50] mm. This script bisects that interval one value at a time
so we can pick the smallest *deterministic* offset to bake into
``mapping/bridge_mappers.compute_safe_bridge_elevations()`` — small
enough to leave the rest of the lattice geometry essentially unchanged,
large enough to robustly avoid the OCCT tangent valley.

Diagnostic
----------
A "clean" bridge layer cross-section at z = bridge elevation contains
the union of (91 rod ellipses) ∪ (84 inter-rod horizontal bridges):
the rod ellipses alone contribute ~323 mm² at z = 0.6 mm (rod-only
baseline below the bottom bridge), while a successfully-fused bridge
layer is ~3,179 mm² (≈ +2,856 mm² from the bridges themselves). The
silent-drop signature is a bridge-elevation slice that returns the
rod-only area (~323 mm²) instead of the full ~3,179 mm². We flag any
layer below 1,000 mm² as MISSING.

Usage
-----
    conda run -n base python scripts/probe_n8_bridge_jitter.py --jitter-mm 0.1

Args
----
    --jitter-mm : float, required. Uniform additive Z-shift in mm
                  applied to *every* default N = 8 bridge elevation.
                  Must remain ≤ ~0.3 mm so the shifted top elevation
                  17.649 + Δ stays within the safe band below the top
                  plate.

Outputs
-------
    runs/n8_jitter_<jitter_µm>um_probe/
        ├── cad_params.json
        ├── compound_enamel_lattice.stl
        └── compound_enamel_lattice.step

    stdout: per-layer cross-section area, count of surviving bridges
            (≤ 8), STL volume + watertight flag, run time.

Exit code 0 on completion; non-zero only on import or CAD-runner errors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Resolve the biomimetic_pipeline package root (this script lives in
# <root>/scripts/) and prepend it to sys.path so the local
# ``mapping`` and ``generators`` packages import cleanly when this
# script is invoked from any working directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from biomimetic_pipeline.generators import cad_runner  # noqa: E402
from biomimetic_pipeline.mapping import feature_to_cad  # noqa: E402 — sys.path manipulated above

# ---------------------------------------------------------------------
# Cross-section geometry helpers
# ---------------------------------------------------------------------


def polygon_area(xy: np.ndarray) -> float:
    """Shoelace area (mm²) of a closed 2-D polygon.

    Parameters
    ----------
    xy : np.ndarray, shape (n_pts, 2)
        Ordered vertex coordinates of a single closed planar loop.

    Returns
    -------
    float
        Absolute polygon area in mm² (unsigned — orientation-independent).
    """
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1])))


def section_area(mesh: trimesh.Trimesh, z: float) -> float:
    """Sum of all closed-polygon areas (mm²) where ``mesh`` intersects
    the horizontal plane ``z = const``.

    Returns 0.0 when the intersection is empty or contains only
    degenerate (< 3 vertex) loops. This is exactly the diagnostic
    quantity for bridge presence: a fused bridge layer produces large
    closed loops (rod ellipses + bridge rectangles), a missing bridge
    produces only the rod ellipses.
    """
    sect = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if sect is None:
        return 0.0
    total = 0.0
    for poly in sect.discrete:
        if poly.shape[0] >= 3:
            total += polygon_area(poly[:, :2])
    return total


# ---------------------------------------------------------------------
# Probe entry point
# ---------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--jitter-mm",
        type=float,
        required=True,
        help=(
            "Uniform additive Z-shift in mm applied to every default "
            "N=8 bridge elevation. Sane range: 0.0 to ~0.3 mm."
        ),
    )
    ap.add_argument(
        "--shrink-safe-band",
        action="store_true",
        help=(
            "Use the IMPLEMENTATION pattern (--jitter-mm is also reserved "
            "as headroom by reducing safe_z_max by the same amount before "
            "computing uniform elevations, then shifting up by --jitter-mm). "
            "This preserves the original top-bridge plate clearance while "
            "still injecting the OCCT-escape offset. Without this flag, the "
            "script just adds --jitter-mm to every default elevation (the "
            "exploratory pattern used to bracket the threshold)."
        ),
    )
    args = ap.parse_args()
    jitter = float(args.jitter_mm)

    # ------------------------------------------------------------------
    # Default N=8 BRIDGE_Z_OFFSETS at the post-fix pipeline defaults.
    #
    # These come from
    #   mapping.bridge_mappers.compute_safe_bridge_elevations(
    #       n_bridge_layers=8,
    #       rod_diameter_mm=2.127955,
    #       bridge_diameter_mm=1.702364,
    #       enamel_thickness_mm=20.0,
    #       plate_overlap_mm=0.5,
    #       junction_sphere_factor=0.0,
    #       clearance_mm=DEFAULT_PLATE_CLEARANCE_MM (=0.5),
    #   )
    # i.e. safe_z_min = plate_overlap + max(bridge_half, sphere_r) + clearance
    #                 = 0.5 + 0.851 + 0.5 = 1.851 mm
    #      safe_z_max = (enamel - plate_overlap) - plate_overlap - margin
    #                 = 19.5 - 0.5 - 1.351 = 17.649 mm
    #      dz = (17.649 - 1.851) / 7 = 2.257 mm
    #
    # Caching them here avoids re-deriving on every probe and makes the
    # shifted comparison reproducible across CAD-runner invocations.
    # ------------------------------------------------------------------
    default_n8 = [1.851, 4.108, 6.365, 8.622, 10.878, 13.135, 15.392, 17.649]

    # ------------------------------------------------------------------
    # Two probe patterns. The exploratory pattern (default) simply adds
    # jitter to every default elevation; it's the cheapest way to bracket
    # the OCCT-resonance threshold but the top elevation drifts above
    # safe_z_max = 17.649 mm, eating into the plate-underside clearance.
    #
    # The implementation pattern (--shrink-safe-band) instead reserves
    # the jitter headroom by reducing safe_z_max by the same amount,
    # then shifting all elevations up; this preserves the original
    # 0.5 mm top-bridge plate clearance and matches what the patched
    # mapping.bridge_mappers.compute_safe_bridge_elevations() will emit.
    # ------------------------------------------------------------------
    safe_z_min, safe_z_max = 1.851, 17.649
    if args.shrink_safe_band:
        safe_z_max_adj = safe_z_max - jitter
        if safe_z_max_adj <= safe_z_min:
            ap.error(f"--jitter-mm {jitter} too large; safe band collapses.")
        elev_pre = [safe_z_min + i * (safe_z_max_adj - safe_z_min) / (8 - 1) for i in range(8)]
        shifted = [round(z + jitter, 6) for z in elev_pre]
        pattern_label = "IMPLEMENTATION (shrink-safe-band + uniform shift)"
    else:
        shifted = [round(z + jitter, 6) for z in default_n8]
        pattern_label = "exploratory (uniform shift, no band reduction)"

    print(f"=== N=8 bridge jitter probe: +{jitter:.4f} mm [{pattern_label}] ===")
    print(f"  default N=8 offsets: {default_n8}")
    print(f"  shifted offsets:     {shifted}")

    # ------------------------------------------------------------------
    # Build the CAD-parameter dict through the standard feature-to-CAD
    # mapper so that ROD_DIAMETER, BRIDGE_DIAMETER, CENTER_SPACING,
    # RING_ROTATION, etc. all come from the canonical morphometrics
    # path — only N_BRIDGE_LAYERS is overridden through the mapper.
    #
    # CRITICAL GOTCHA: feature_to_cad.map_morphometrics() unconditionally
    # *recomputes* BRIDGE_Z_OFFSETS via compute_safe_bridge_elevations()
    # at lines 199-225, AFTER the extra_overrides merge. So passing
    # BRIDGE_Z_OFFSETS via extra_overrides has no effect — the override
    # is silently clobbered back to the uniformly-spaced default for
    # the given N. We therefore override BRIDGE_Z_OFFSETS on the returned
    # dict, AFTER the mapper finishes, so the shifted elevations actually
    # reach cad_runner.run() → lattice_cad.py.
    # ------------------------------------------------------------------
    morph_path = ROOT / "runs/live_001/morphometrics.json"
    morph = json.load(open(morph_path))
    params = feature_to_cad.map_morphometrics(
        morph,
        morphometrics_source=morph_path,
        extra_overrides={
            "N_BRIDGE_LAYERS": 8,
        },
    )

    # Post-mapper override: replace the recomputed default offsets with
    # our jittered list. This is the only override channel that survives
    # the mapper's internal recomputation, so it must occur *after*
    # map_morphometrics() returns and *before* cad_runner.run() consumes
    # the dict.
    params["BRIDGE_Z_OFFSETS"] = shifted

    # Per-probe artifact directory; jitter rounded to micrometres
    # so directory names sort lexicographically and are unambiguous.
    jitter_um = int(round(jitter * 1000))
    # Suffix _impl distinguishes implementation-pattern probes (shrunk band)
    # from exploratory uniform-shift probes at the same jitter magnitude;
    # without this, a +0.15 mm probe in either mode would overwrite the other.
    pattern_suffix = "_impl" if args.shrink_safe_band else ""
    out = ROOT / "runs" / f"n8_jitter_{jitter_um:04d}um_probe{pattern_suffix}"
    out.mkdir(parents=True, exist_ok=True)
    feature_to_cad.save(params, out / "cad_params.json")

    # ------------------------------------------------------------------
    # Drive the cad_runner subprocess (CadQuery via cad_env). This is
    # the same code path that the full pipeline takes, so any tangent-
    # resonance bug reproduces here identically.
    # ------------------------------------------------------------------
    t0 = time.time()
    result = cad_runner.run(params, export_dir=out)
    dt = time.time() - t0
    print(f"  CAD complete in {dt:.1f} s")

    stl_path = Path(result.stl_path)
    print(f"  STL: {stl_path}")

    # ------------------------------------------------------------------
    # Cross-section bridge-presence diagnostic. The STL is sliced at:
    #   (a) z = 0.6 mm   — above the bottom plate (plate top = 0.5 mm)
    #                      but well below the first bridge (1.851 mm),
    #                      i.e. *only* rods are present; gives the
    #                      rod-only baseline area for this geometry.
    #   (b) z = each shifted bridge elevation — clean layers should
    #                      add ~2,856 mm² of bridge material above the
    #                      baseline (total ~3,179 mm²); a layer below
    #                      THRESHOLD_AREA mm² is flagged MISSING.
    # ------------------------------------------------------------------
    mesh = trimesh.load(stl_path)
    print(f"  STL volume: {mesh.volume:.1f} mm³, watertight: {mesh.is_watertight}")

    baseline_z = 0.6
    baseline_area = section_area(mesh, baseline_z)
    print(f"  rod-only baseline at z={baseline_z} mm: {baseline_area:.1f} mm²")
    print()

    THRESHOLD_AREA = 1000.0  # mm² — sits well above rod-only (~323) and below clean (~3179)
    n_clean = 0
    for i, z in enumerate(shifted):
        a = section_area(mesh, z)
        status = "OK" if a > THRESHOLD_AREA else "MISSING"
        if a > THRESHOLD_AREA:
            n_clean += 1
        print(f"  layer {i}: z={z:.4f} mm, area={a:8.1f} mm²  [{status}]")

    print()
    print(f"  RESULT: {n_clean} / 8 bridges fused at jitter +{jitter:.4f} mm")
    if n_clean == 8:
        print("  ==> JITTER ESCAPES OCCT TANGENT RESONANCE")
    else:
        print(f"  ==> STILL BROKEN ({8 - n_clean} bridge layers silently dropped)")


if __name__ == "__main__":
    main()
