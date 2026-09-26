#!/usr/bin/env python
"""Strain at which neighbouring rods would first touch (linear-elastic estimate).

Purpose
-------
The near-touching design arm leaves a 0.025-mm gap between neighbouring rod
surfaces. The finite-element model has no contact, so once the relative
motion of two facing surfaces exceeds the gap its results no longer describe
the physical lattice. Because the solution is linear, the closing rate of
every gap is proportional to the applied strain, and the strain at which a
gap first closes is

    eps_contact = min over facing surface-node pairs of  g_ij / c_ij,

where g_ij is the current separation of the pair and c_ij its closing per
unit nominal strain. This script estimates eps_contact for each run.

Method
------
1. Extract the boundary surface of the tetrahedral mesh with its outward
   point normals and nodal displacements u.
2. Facing pairs: for each surface node p, candidate nodes q within
   --max-gap-mm whose separation vector d = x_q - x_p points along p's
   outward normal (d . n_p > 0.8 |d|) and whose own normal points back
   (n_q . n_p < -0.8). Such pairs sit on opposite sides of a narrow gap
   between two different surfaces; nodes on the same surface are excluded
   by the normal test.
3. Closing per unit strain: c = -(u_q - u_p) . d/|d| / eps_nominal, with
   eps_nominal the solver's nominal axial strain (top-plate displacement /
   specimen height, from the load-path JSON). Pairs that open (c <= 0)
   never touch.
4. eps_contact = min g/c; the 1st percentile of g/c is also reported
   because the minimum is set by a single node pair.
5. The 0.5-mm mesh facets the rod surfaces, so meshed node-pair gaps
   (~0.07-0.08 mm) exceed the 0.025-mm design gap. The design-gap estimate
   eps_design = design_gap / max(c) (--design-gap-mm) is therefore also
   reported; it is the physically relevant onset for the printed design.

Inputs
------
--runs DIR [DIR ...]   run folders with fea/final/compound_enamel_lattice.vtk and
                       runs/_load_path_analysis/<key>.json (nominal strain)
--max-gap-mm           largest separation treated as a gap (default 0.10 mm)

Outputs
-------
runs/_load_path_analysis/contact_onset.csv   one row per run (others kept)

Side effects: read-only on run directories.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

BIO = Path(__file__).resolve().parent.parent
RUNS = (
    BIO / "biomimetic_pipeline" / "runs"
)  # public layout: run directories live inside the package (RunContext default)
LPA = RUNS / "_load_path_analysis"
OUT = LPA / "contact_onset.csv"
FIELDS = [
    "run",
    "n_pairs",
    "gap_median_mm",
    "eps_contact_min_pct",
    "eps_contact_p1_pct",
    "closing_max_mm_per_pct",
    "eps_contact_design_gap_pct",
]


def onset(run_dir: Path, max_gap: float, design_gap: float) -> dict:
    """Contact-onset estimate for one solved model (see module docstring)."""
    key = str(run_dir.relative_to(RUNS))
    eps = json.loads((LPA / f"{key.replace('/', '__')}.json").read_text())["top_plate"][
        "nominal_axial_strain"
    ]
    g = pv.read(run_dir / "fea" / "final" / "compound_enamel_lattice.vtk")

    # --- 1. Boundary surface with outward normals and displacements ---------------------
    surf = g.extract_surface().compute_normals(
        point_normals=True, cell_normals=False, auto_orient_normals=True, consistent_normals=True
    )
    X = np.asarray(surf.points)
    N = np.asarray(surf.point_data["Normals"])
    U = np.asarray(surf.point_data["u"])

    # --- 2. Facing node pairs across narrow gaps ------------------------------------------
    tree = cKDTree(X)
    pairs = tree.query_pairs(max_gap, output_type="ndarray")
    if len(pairs) == 0:
        return {"run": key, "n_pairs": 0}
    p, q = pairs[:, 0], pairs[:, 1]
    d = X[q] - X[p]
    dist = np.linalg.norm(d, axis=1)
    ok = dist > 1e-9
    e = np.zeros_like(d)
    e[ok] = d[ok] / dist[ok, None]
    facing = ok & ((e * N[p]).sum(1) > 0.8) & ((N[q] * N[p]).sum(1) < -0.8)
    p, q, e, dist = p[facing], q[facing], e[facing], dist[facing]
    if len(p) == 0:
        return {"run": key, "n_pairs": 0}

    # --- 3. Closing per unit nominal strain (linear solution -> proportional) --------------
    closing = -((U[q] - U[p]) * e).sum(1) / eps  # mm per unit strain; > 0 = gap closes
    closes = closing > 0
    ratio = dist[closes] / closing[closes]  # nominal strain at contact
    return {
        "run": key,
        "n_pairs": int(len(p)),
        "gap_median_mm": round(float(np.median(dist)), 4),
        "eps_contact_min_pct": round(100.0 * float(ratio.min()), 4) if len(ratio) else float("inf"),
        "eps_contact_p1_pct": (
            round(100.0 * float(np.percentile(ratio, 1)), 4) if len(ratio) else float("inf")
        ),
        "closing_max_mm_per_pct": round(float(closing.max()) / 100.0, 5),
        # design gap / largest closing per 1 % strain -> onset strain in %
        "eps_contact_design_gap_pct": (
            round(design_gap / (float(closing.max()) / 100.0), 3)
            if closing.max() > 0
            else float("inf")
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--max-gap-mm", type=float, default=0.10)
    ap.add_argument(
        "--design-gap-mm",
        type=float,
        default=0.025,
        help="design surface gap of the near-touching arm (CENTER_SPACING - ROD_DIAMETER)",
    )
    args = ap.parse_args()
    rows = {}
    if OUT.exists():
        with open(OUT) as fh:
            rows = {r["run"]: r for r in csv.DictReader(fh)}
    for pat in args.runs:
        for rd in sorted(glob.glob(pat)) or [pat]:
            r = onset(Path(rd).resolve(), args.max_gap_mm, args.design_gap_mm)
            rows[r["run"]] = r
            print(r, flush=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows.values():
            w.writerow({k: r.get(k, "") for k in FIELDS})


if __name__ == "__main__":
    main()
