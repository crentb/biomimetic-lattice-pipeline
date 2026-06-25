#!/usr/bin/env python
"""
diagnose_h24_watertight.py
==========================

Purpose
-------
Characterise WHY thick lattices at ENAMEL_THICKNESS=24 mm report STL
"watertight=False" for sparse N (N=4/5) at every OCCT jitter, even though the
STEP solid meshes fine (verified: N=4 @ H=24 STEP -> 1.55 M tets). It loads
each STL with vertex MERGING (process=True, the correct watertight test),
confirms watertightness, counts the OPEN boundary edges (edges used by only one
face = holes in the surface), measures how LONG those open edges are (micron
slivers vs a real gap), and locates them in (z, radius). This decides whether
the non-watertightness is (a) tiny tessellation slivers spread along the
near-tangent rod-rod contacts -- an STL-export artifact that the integrity gate
should NOT hard-fail on -- or (b) a real, localised CAD hole.

Why this exists
---------------
generators/cad_integrity.py gates the pipeline on STL watertightness and ABORTS
before meshing, but mesh_runner meshes the STEP. Taller near-tangent rods
(24 mm, 0.025 mm gap) are exactly where OCCT's STL tessellation leaves
sub-tolerance sliver gaps while the BREP solid stays closed. This script is the
evidence that the gate is a FALSE NEGATIVE for these geometries (so N=4..8 @
H=24 are recoverable), and it quantifies the slivers so the fix can be sized
(e.g. a watertight-merge tolerance, or gate on STEP/mesh instead of STL).

Isolation set (defaults):
  A = N=4 @ H=24  (FAIL, non-watertight)  runs/_h24_hunt/thick_N4_j0.0
  B = N=9 @ H=24  (PASS, watertight)      runs/sweep_H24_thick/trial_006...
  C = N=4 @ H=20  (PASS, watertight)      runs/sweep_layers_v2_thick/trial_003...
  A vs C  -> isolates the HEIGHT effect on watertightness (same N=4).
  A vs B  -> isolates the N effect (same H=24).

Inputs / Outputs
----------------
  Default STL trio (override with --stl PATH, repeatable). Read-only; prints a
  per-case report + a cross-case table. Writes nothing.

Side effects / non-obvious behaviour
------------------------------------
  * Loads each ~70-105 MB STL with process=True (vertex merge -- REQUIRED for a
    valid watertight test; process=False reads every solid as non-watertight).
    One at a time, freed before the next (16 GB box).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent

# Default isolation trio (see docstring). Repo-relative.
DEFAULT_STLS = [
    ROOT / "runs/_h24_hunt/thick_N4_j0.0/cad/compound_enamel_lattice.stl",  # A FAIL
    ROOT
    / "runs/sweep_H24_thick/trial_006_N_BRIDGE_LAYERS_9/cad/compound_enamel_lattice.stl",  # B PASS
    ROOT
    / "runs/sweep_layers_v2_thick/trial_003_N_BRIDGE_LAYERS_4/cad/compound_enamel_lattice.stl",  # C PASS
]


def _params(stl: Path) -> dict:
    """Read the trial's cad_params.json (two dirs up from the STL)."""
    p = stl.parents[1] / "cad_params.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def analyse(stl: Path) -> dict:
    label = stl.parents[1].name
    pr = _params(stl)
    N = pr.get("N_BRIDGE_LAYERS")
    H = pr.get("ENAMEL_THICKNESS")
    rod = pr.get("ROD_DIAMETER")
    cs = pr.get("CENTER_SPACING")
    offs = sorted(pr.get("BRIDGE_Z_OFFSETS") or [])

    print("=" * 92)
    print(
        f"CASE {label}   N={N}  H={H}  rod={rod}  rod-rod gap={None if (rod is None or cs is None) else round(cs-rod,4)}"
    )

    # process=True merges coincident vertices so a CLOSED solid reads watertight
    # (matches generators/cad_integrity's check). This is the make-or-break flag.
    m = trimesh.load(stl, process=True)
    F = len(m.faces)
    wt = bool(m.is_watertight)
    print(f"  faces={F:,}  watertight={wt}  euler_number={m.euler_number}")

    # --- open boundary edges = undirected edges used by exactly ONE face -------
    # In a closed manifold every edge is shared by 2 faces; a count of 1 marks a
    # hole rim, a count >2 marks a non-manifold edge.
    es = np.sort(m.edges, axis=1)  # (3F,2) undirected
    uniq, counts = np.unique(es, axis=0, return_counts=True)
    open_e = uniq[counts == 1]  # (n_open,2) boundary edges
    nonman = int((counts > 2).sum())
    n_open = len(open_e)
    print(f"  OPEN boundary edges (hole rims): {n_open:,}   non-manifold edges: {nonman}")

    summary = {
        "label": label,
        "N": N,
        "H": H,
        "faces": F,
        "watertight": wt,
        "n_open": n_open,
        "nonman": nonman,
    }

    if n_open:
        v = m.vertices
        p0 = v[open_e[:, 0]]
        p1 = v[open_e[:, 1]]
        elen = np.linalg.norm(p1 - p0, axis=1)  # open-edge length (mm)
        mid = 0.5 * (p0 + p1)  # midpoint (mm)
        z = mid[:, 2]
        r = np.hypot(mid[:, 0], mid[:, 1])
        print(
            f"    open-edge LENGTH (mm): min {elen.min():.2e}  median {np.median(elen):.2e}  max {elen.max():.2e}"
        )
        print("      (micron-scale => tessellation slivers; ~mesh-size => a real gap)")
        print(
            f"    open-edge Z (mm): {z.min():.2f}..{z.max():.2f}   radius r (mm): {r.min():.2f}..{r.max():.2f}"
        )
        # Spread across the full rod Z-span => slivers along the rods; a narrow
        # cluster => a localised hole. Show a 10-bin Z histogram.
        hist, _ = np.histogram(z, bins=10)
        print(f"    open-edge Z histogram (10 bins): {hist.tolist()}")
        # Are they at the bridge layers, or between rods (away from bridges)?
        if offs:
            offa = np.asarray(offs)
            dz_near = np.abs(z[:, None] - offa[None, :]).min(axis=1)
            half_h = pr.get("BRIDGE_DIAMETER", 1.702) / 2.0
            on_bridge = int((dz_near <= half_h).sum())
            print(
                f"    open edges within a bridge half-height ({half_h:.2f} mm) of a bridge layer: "
                f"{on_bridge}/{n_open}  (low => the holes are NOT at the bridges)"
            )
        summary.update(
            open_len_med=float(np.median(elen)),
            open_len_max=float(elen.max()),
            zspan=(float(z.min()), float(z.max())),
            rspan=(float(r.min()), float(r.max())),
        )

    del m, es, uniq, counts
    gc.collect()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Characterise H=24 thick STL non-watertightness (slivers vs hole)."
    )
    ap.add_argument(
        "--stl",
        type=Path,
        action="append",
        default=None,
        help="STL to analyse (repeatable); default = the 3-case isolation trio",
    )
    args = ap.parse_args()
    stls = args.stl or DEFAULT_STLS

    out = []
    for s in stls:
        s = Path(s).resolve()
        if not s.is_file():
            print(f"MISSING: {s}")
            continue
        out.append(analyse(s))

    print("=" * 92)
    print("CROSS-CASE  (A=N4@H24 FAIL, B=N9@H24 PASS, C=N4@H20 PASS)")
    print(
        f"  {'case':42} {'N':>2} {'H':>5} {'watertight':>10} {'open_edges':>11} "
        f"{'len_med(mm)':>12} {'len_max(mm)':>12}"
    )
    for s in out:
        lm = f"{s['open_len_med']:.2e}" if s.get("open_len_med") is not None else "-"
        lx = f"{s['open_len_max']:.2e}" if s.get("open_len_max") is not None else "-"
        print(
            f"  {s['label']:42} {str(s['N']):>2} {str(s['H']):>5} {str(s['watertight']):>10} "
            f"{s['n_open']:>11,} {lm:>12} {lx:>12}"
        )
    print()
    print("Read: A vs C isolates HEIGHT (same N=4); A vs B isolates N (same H=24).")
    print("If A's open edges are micron-scale and spread along the rod Z-span (not at the")
    print("bridges), the non-watertightness is OCCT STL-tessellation slivers at the 0.025 mm")
    print("near-tangent rod contacts -- benign (the STEP meshes), so the cad_integrity STL")
    print("gate is a FALSE NEGATIVE and should tolerate it / gate on the STEP instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
