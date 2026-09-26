#!/usr/bin/env python
"""Robust stress-concentration factors from stored FEA fields.

Purpose
-------
The stock metrics script (cad_modeling/.../extract_metrics.py, line 206)
defines SCF = max(von Mises) / mean(von Mises): the stress in the single most
stressed element over the mean. Linear-elastic solutions are singular at the
sharp re-entrant rod-bridge and rod-plate junctions of the lattice CAD (no
fillets), so the peak element stress depends on the local mesh and does not
converge under refinement; independent meshes of the same geometry at the
same nominal size already scatter by ~3 % in this quantity. This script
recomputes, for every run, volume-weighted percentile-based factors that
exclude the singular corner elements:
    SCF_p99  = P99(VM) / mean(VM)     (volume-weighted percentile and mean)
    SCF_p999 = P99.9(VM) / mean(VM)
    SCF_max  = max(VM) / mean(VM)     (the stock definition, for comparison)
The v1 manuscript described the stock quantity as a 99th-percentile ratio,
which it is not.

Inputs
------
--runs DIR [DIR ...]  (glob patterns allowed); each needs
                      fea/final/compound_enamel_lattice.vtk
Outputs
-------
runs/_load_path_analysis/robust_scf.csv   one row per run (replaced by key)

Side effects: read-only on run directories.
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import pyvista as pv

BIO = Path(__file__).resolve().parent.parent
RUNS = (
    BIO / "biomimetic_pipeline" / "runs"
)  # public layout: run directories live inside the package (RunContext default)
OUT = RUNS / "_load_path_analysis" / "robust_scf.csv"


def weighted_percentile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Percentile q (0-100) of x with weights w (element volumes)."""
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(np.interp(q / 100.0 * cw[-1], cw, x[o]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument(
        "--fea-subdir",
        default="final",
        help="fea/<subdir> holding the VTK (default final; e.g. iter_1)",
    )
    args = ap.parse_args()
    rows = {}
    if OUT.exists():
        with open(OUT) as fh:
            rows = {r["run"]: r for r in csv.DictReader(fh)}
    for pat in args.runs:
        for rd in sorted(glob.glob(pat)) or [pat]:
            rd = Path(rd).resolve()
            vtk = rd / "fea" / args.fea_subdir / "compound_enamel_lattice.vtk"
            if not vtk.exists():
                continue
            g = pv.read(vtk)
            tet = np.asarray(g.celltypes) == pv.CellType.TETRA
            vm = np.asarray(g.cell_data["von_mises"])[tet]
            vol = np.abs(
                g.extract_cells(np.where(tet)[0])
                .compute_cell_sizes(length=False, area=False, volume=True)
                .cell_data["Volume"]
            )
            mean = float((vm * vol).sum() / vol.sum())
            key = str(rd.relative_to(RUNS))
            rows[key] = {
                "run": key,
                "n_tet": int(tet.sum()),
                "vm_mean_MPa": round(mean, 4),
                "SCF_p99": round(weighted_percentile(vm, vol, 99.0) / mean, 5),
                "SCF_p999": round(weighted_percentile(vm, vol, 99.9) / mean, 5),
                "SCF_max": round(float(vm.max()) / mean, 5),
            }
            r = rows[key]
            print(
                f"{key:55s} n={r['n_tet']:8d} SCF_p99={r['SCF_p99']:.3f} p99.9={r['SCF_p999']:.3f} max={r['SCF_max']:.3f}",
                flush=True,
            )
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["run", "n_tet", "vm_mean_MPa", "SCF_p99", "SCF_p999", "SCF_max"]
        )
        w.writeheader()
        for r in rows.values():
            w.writerow(r)


if __name__ == "__main__":
    main()
