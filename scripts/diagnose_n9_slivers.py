#!/usr/bin/env python
"""
diagnose_n9_slivers.py
======================

Purpose
-------
Empirically pin WHY thick N=9 at the canonical rod (3.167 mm) fails to
volume-mesh (gmsh returns 0 tets) while thick N=8 @ 3.167 and thick N=9 @
2.75 mesh fine. It loads each lattice STL *unprocessed* (so near-zero-area
"sliver" faces are NOT silently cleaned away), characterises the degenerate
face population (count, area, how thin), and locates each sliver in Z
against the bridge-layer centres -- the decisive test of whether the
slivers sit at the bridge<->rod junctions (clustered at bridge Z's) or at
the rod-rod near-tangencies (spread uniformly along Z). Three cases are
compared in a 2x2-style isolation so the rod gap and the bridge spacing
can each be tested as the cause.

Why this exists
---------------
runs/sweep_layers_v2_thick/WHY_N9_THICK_NEEDS_OWN_ROD.txt asserts a cause
("bridge-Z-spacing x fat rod") but mixes three different length scales --
the 0.025 mm rod-rod gap, the 0.210 mm bridge-surface gap, and the
8.7e-9 mm^2 "zero-area" slivers -- without showing which one actually
produces the failing faces. A 0.21 mm gap is finite and far larger than a
8.7e-9 mm^2 face, so the narrative is internally inconsistent: a tight gap
(an under-resolution story) and an exact degeneracy (a CAD-topology story)
are different failure modes. This script gathers the geometry evidence
needed to write a single, self-consistent diagnosis and to settle whether
"0.21 mm < 0.5 mm mesh size" is even the right framing.

Isolation design (defaults below):
  A = archive .../trial_006 N=9 @ rod 3.167  (FAILS) -- rod gap 0.025, bridge gap 0.21
  B = runs    .../trial_002 N=8 @ rod 3.167  (meshes) -- rod gap 0.025, bridge gap 0.48
  C = runs    .../trial_006 N=9 @ rod 2.75   (meshes) -- rod gap 0.442, bridge gap 0.21
  A vs B  -> isolates BRIDGE spacing (rod gap held at 0.025).
  A vs C  -> isolates the ROD       (bridge spacing held at the N=9 value).

Inputs (CLI)
------------
  --stl PATH    : repeatable; lattice STL(s) to analyse. If omitted, uses the
                  three references above. For each STL the sibling
                  cad_params.json (at <stl>/../../cad_params.json) is read for
                  N_BRIDGE_LAYERS, ROD/BRIDGE/CENTER_SPACING, BRIDGE_Z_OFFSETS.
  --sliver-area : area (mm^2) below which a face is called a degenerate
                  sliver (default 1e-6 mm^2 = 1 um^2; a healthy 0.5 mm-mesh
                  face is ~0.1 mm^2, i.e. ~1e5 um^2, so 1e-6 mm^2 is ~11
                  orders of magnitude smaller -> unambiguously degenerate).

Outputs
-------
  Per-case report to stdout: face count, bbox, area histogram (counts below a
  ladder of thresholds), sliver shape (area / min-altitude / slenderness),
  and the Z-distribution of slivers vs the bridge-layer centres. Then a
  cross-case comparison table. Writes NO files (pure read-only analysis).

Side effects / non-obvious behaviour
------------------------------------
  * Loads STLs with process=False ON PURPOSE: trimesh's default processing
    merges vertices and drops degenerate faces, which would ERASE the very
    slivers we are hunting. We therefore see the raw triangulation exactly as
    gmsh's STEP tessellation produced it.
  * Each ~90-100 MB STL (~1.8 M triangles) is loaded one at a time and freed
    before the next, to stay within the 16 GB box.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent

# Default isolation set (see module docstring). Paths are repo-relative.
DEFAULT_STLS = [
    ROOT
    / "archive/thick_gap010_backup_20260601_231014/sweep_layers_v2_thick"
    / "trial_006_N_BRIDGE_LAYERS_9/cad/compound_enamel_lattice.stl",  # A: FAILS
    ROOT
    / "runs/sweep_layers_v2_thick/trial_002_N_BRIDGE_LAYERS_8"
    / "cad/compound_enamel_lattice.stl",  # B: meshes
    ROOT
    / "runs/sweep_layers_v2_thick/trial_006_N_BRIDGE_LAYERS_9"
    / "cad/compound_enamel_lattice.stl",  # C: meshes
]

# Ladder of area thresholds (mm^2) for the histogram. Spans true degeneracy
# (1e-9) up to a small-but-healthy face (1e-2). Lets the reader SEE the
# population shape rather than trust one cut.
AREA_LADDER_MM2 = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def _load_params(stl: Path) -> dict:
    """Read the trial's cad_params.json (sits two dirs up from the STL).

    Returns {} if absent so the script still runs (geometry-only), but the
    Z-bucketing against bridge layers needs BRIDGE_Z_OFFSETS, so we warn.
    """
    cad_p = stl.parents[1] / "cad_params.json"  # <trial>/cad/x.stl -> <trial>/cad_params.json
    if not cad_p.is_file():
        print(f"  WARN: no cad_params.json at {cad_p} -- Z-bucketing skipped")
        return {}
    return json.loads(cad_p.read_text())


def analyse(stl: Path, sliver_area_mm2: float) -> dict:
    """Analyse one STL; print a detailed report; return a summary dict."""
    label = stl.parents[1].name  # e.g. trial_006_N_BRIDGE_LAYERS_9
    params = _load_params(stl)
    N = params.get("N_BRIDGE_LAYERS")
    rod = params.get("ROD_DIAMETER")
    bridge_d = params.get("BRIDGE_DIAMETER")
    cs = params.get("CENTER_SPACING")
    offsets = sorted(params.get("BRIDGE_Z_OFFSETS") or [])  # bridge-layer Z centres (mm)

    print("=" * 92)
    print(f"CASE  {label}")
    print(f"  params: N={N}  ROD={rod}  BRIDGE={bridge_d}  CS={cs}")
    if rod is not None and cs is not None:
        print(f"  rod-rod gap (CS - ROD) = {cs - rod:.4f} mm")  # horizontal rod spacing margin
    if len(offsets) >= 2:
        dzs = np.diff(offsets)  # consecutive bridge-layer spacings (mm)
        dz_min = float(dzs.min())
        # bridge SURFACE gap = centre spacing - bridge diameter (mm); the
        # vertical clearance between adjacent bridge cylinders.
        b_gap = dz_min - (bridge_d or 0.0)
        print(
            f"  bridge layers: {len(offsets)}  dz(min)={dz_min:.4f} mm  "
            f"-> bridge surface gap = {b_gap:.4f} mm"
        )
        print(f"  bridge Z centres: {[round(o,3) for o in offsets]}")

    # --- load raw triangulation (NO processing -> keep degenerate faces) ---
    mesh = trimesh.load(stl, process=False)
    areas = mesh.area_faces  # (F,) mm^2 per triangle
    F = len(areas)
    lo, hi = mesh.bounds  # bbox corners (mm)
    print(
        f"  faces: {F:,}   bbox (mm): x[{lo[0]:.2f},{hi[0]:.2f}] "
        f"y[{lo[1]:.2f},{hi[1]:.2f}] z[{lo[2]:.2f},{hi[2]:.2f}]"
    )
    print(
        f"  face area (mm^2): min={areas.min():.3e}  median={np.median(areas):.3e}  max={areas.max():.3e}"
    )

    # --- area histogram: how many faces below each ladder threshold --------
    print("  faces below area threshold:")
    for t in AREA_LADDER_MM2:
        print(f"      < {t:.0e} mm^2 : {int((areas < t).sum())}")

    # --- sliver subset + shape ---------------------------------------------
    sliver_idx = np.flatnonzero(areas < sliver_area_mm2)
    n_sliver = len(sliver_idx)
    print(f"  SLIVERS (area < {sliver_area_mm2:.0e} mm^2): {n_sliver}")

    summary = {
        "label": label,
        "N": N,
        "rod": rod,
        "cs": cs,
        "bridge_d": bridge_d,
        "faces": F,
        "n_sliver": n_sliver,
        "rod_gap": (cs - rod) if (rod is not None and cs is not None) else None,
    }

    if n_sliver:
        tris = mesh.triangles[sliver_idx]  # (n,3,3) vertex coords (mm)
        # Edge lengths of each sliver triangle (mm).
        e0 = np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1)
        e1 = np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1)
        e2 = np.linalg.norm(tris[:, 0] - tris[:, 2], axis=1)
        edges = np.stack([e0, e1, e2], axis=1)  # (n,3)
        emax = edges.max(axis=1)
        emin = edges.min(axis=1)
        a = areas[sliver_idx]
        # Minimum altitude (mm) = 2*area / longest edge: how THIN the triangle
        # is. A true sliver has a long base but a near-zero height.
        min_alt = 2.0 * a / np.clip(emax, 1e-30, None)
        # Slenderness = longest edge / shortest altitude. ~1 for healthy,
        # >>1 for slivers/needles.
        slender = emax / np.clip(min_alt, 1e-30, None)
        print(
            f"    edge len (mm): shortest {emin.min():.2e}..{emin.max():.2e}  "
            f"longest {emax.min():.2e}..{emax.max():.2e}"
        )
        print(
            f"    min altitude (mm): {min_alt.min():.2e}..{min_alt.max():.2e}  "
            f"(median {np.median(min_alt):.2e})  <- how thin the slivers are"
        )
        print(
            f"    slenderness (longest_edge/min_alt): median {np.median(slender):.1f}  max {slender.max():.1f}"
        )

        # --- locate slivers: centroid x,y,z and radius r -------------------
        cen = tris.mean(axis=1)  # (n,3) centroid (mm)
        cz = cen[:, 2]
        r = np.hypot(cen[:, 0], cen[:, 1])
        print(
            f"    centroid Z (mm): {cz.min():.2f}..{cz.max():.2f}   "
            f"radius r (mm): {r.min():.2f}..{r.max():.2f}"
        )

        # --- THE decisive test: are sliver Z's clustered at bridge layers? -
        if offsets:
            offs = np.asarray(offsets)
            # Distance from each sliver to its NEAREST bridge-layer centre (mm).
            d_near = np.abs(cz[:, None] - offs[None, :]).min(axis=1)
            half_h = (bridge_d or 0.0) / 2.0  # bridge cylinder half-height (mm)
            on_layer = int((d_near <= half_h).sum())  # sliver sits within a bridge band
            # How many DISTINCT bridge layers have >=1 nearby sliver:
            nearest_layer = np.abs(cz[:, None] - offs[None, :]).argmin(axis=1)
            distinct = len(set(nearest_layer.tolist()))
            print(
                f"    Z vs bridge layers: {on_layer}/{n_sliver} within +/-{half_h:.2f} mm "
                f"of a bridge centre; touch {distinct}/{len(offs)} distinct layers"
            )
            print(
                f"    dist-to-nearest-bridge (mm): median {np.median(d_near):.3f}  max {d_near.max():.3f}"
            )
            # Uniform-in-Z control: if slivers were rod-rod tangencies they would
            # spread across the full bridge band, not cluster. Report Z spread vs band.
            band = offs.max() - offs.min()
            print(
                f"    (bridge band height {band:.2f} mm; sliver Z spread {cz.max()-cz.min():.2f} mm)"
            )
            summary.update(on_layer=on_layer, distinct_layers=distinct, n_layers=len(offs))
        summary.update(
            min_alt_med=float(np.median(min_alt)),
            slender_med=float(np.median(slender)),
            sliver_z=(float(cz.min()), float(cz.max())),
            sliver_r=(float(r.min()), float(r.max())),
        )

    # free the big mesh before the next case (16 GB box)
    del mesh, areas
    gc.collect()
    return summary


def main() -> int:
    # --- 1. Parse CLI ----------------------------------------------------
    ap = argparse.ArgumentParser(description="Diagnose thick N=9 mesh-failure slivers.")
    ap.add_argument(
        "--stl",
        type=Path,
        action="append",
        default=None,
        help="lattice STL to analyse (repeatable); default = 3 references",
    )
    ap.add_argument(
        "--sliver-area",
        type=float,
        default=1e-6,
        help="area (mm^2) below which a face is a degenerate sliver",
    )
    args = ap.parse_args()
    stls = args.stl or DEFAULT_STLS

    # --- 2. Analyse each case -------------------------------------------
    summaries = []
    for s in stls:
        s = Path(s).resolve()
        if not s.is_file():
            print(f"MISSING: {s}")
            continue
        summaries.append(analyse(s, args.sliver_area))

    # --- 3. Cross-case comparison table ---------------------------------
    print("=" * 92)
    print("CROSS-CASE COMPARISON  (the isolation result)")
    print(
        f"  {'case':40} {'N':>2} {'rod':>7} {'rod_gap':>8} {'#faces':>10} {'#slivers':>9} "
        f"{'on-layer':>9} {'thin(med um)':>12}"
    )
    for s in summaries:
        rg = f"{s['rod_gap']:.3f}" if s.get("rod_gap") is not None else "?"
        rod = f"{s['rod']:.3f}" if isinstance(s.get("rod"), (int, float)) else "?"
        onl = f"{s.get('on_layer','-')}/{s.get('n_sliver',0)}" if s.get("n_sliver") else "-"
        thin = f"{s['min_alt_med']*1e3:.2e}" if s.get("min_alt_med") is not None else "-"  # mm->um
        print(
            f"  {s['label']:40} {str(s['N']):>2} {rod:>7} {rg:>8} "
            f"{s['faces']:>10,} {s['n_sliver']:>9} {onl:>9} {thin:>12}"
        )
    print()
    print("Read: if ONLY the failing case (A) carries a sliver population that the two")
    print("meshable controls (B,C) lack, then BOTH the tight N=9 bridge spacing AND the")
    print("fat 3.167 rod are necessary. If the slivers cluster on the bridge layers")
    print("(high on-layer fraction), they are bridge<->rod junction slivers, not rod-rod")
    print("tangencies. 'thin(med um)' is the median sliver height: if ~nm-um it is a true")
    print("degenerate face (a CAD-topology failure), NOT an under-resolution gap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
