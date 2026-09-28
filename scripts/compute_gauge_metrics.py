#!/usr/bin/env python
"""Plate-free (gauge-only) normalisation, stress-concentration and solid fraction.

Purpose
-------
Every FEA model is a rod lattice between two solid loading plates. Three
quantities in the v2 manuscript were computed over the WHOLE model, plates
included:
  * sigma_bar  -- the volume-averaged von Mises stress used to normalise the
                  stress maps (the solver scales the load so that it is
                  200 MPa over the whole model);
  * SCF_p99    -- P99(VM) / mean(VM) (compute_robust_scf.py);
  * phi        -- the upstream solid fraction V_total / V_bounding_box
                  (extract_metrics.py), which counts the solid plates.
The plates are nearly uniformly and lightly stressed and fully solid, so
they dilute the stress mean and inflate phi. A review asked for the same
quantities restricted to the lattice. This script recomputes them over the
gauge (the rod/bridge region between the plates, found exactly as in
analyze_load_paths.gauge_bounds):
  vm_mean_gauge   volume-weighted mean VM over the gauge window, i.e. the
                  gauge minus 1 mm next to each plate interface (the same
                  window as the load-partition metrics, so the singular
                  rod-plate corners are excluded);
  SCF_p99_gauge   P99(VM) / mean(VM) within that window (volume-weighted);
  SCF_p999_gauge  P99.9(VM) / mean(VM) within that window;
  sigma_ratio     vm_mean_gauge / vm_mean_all -- multiply a whole-model-
                  normalised stress by 1/sigma_ratio to re-normalise it by
                  the gauge mean;
  phi_gauge       solid volume between the plate interfaces divided by
                  footprint x gauge height (footprint = plate x-y extent);
  phi_all         the upstream definition, recomputed as a check.

Inputs
------
--runs DIR [DIR ...]   run folders (glob patterns allowed); each needs
                       fea/<subdir>/compound_enamel_lattice.vtk
--fea-subdir NAME      default "final" (e.g. "iter_1" for the 0.4-mm mesh)

Outputs
-------
runs/_load_path_analysis/gauge_metrics.csv   one row per run (rows for other
                                             runs already in the file are kept)

Side effects: read-only on run directories.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

# --- 1. Shared helpers (gauge detection identical to the load-path analysis) --
BIO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BIO / "scripts"))
from analyze_load_paths import gauge_bounds  # noqa: E402
from compute_robust_scf import weighted_percentile  # noqa: E402

RUNS = (
    BIO / "biomimetic_pipeline" / "runs"
)  # public layout: run directories live inside the package (RunContext default)
OUT = RUNS / "_load_path_analysis" / "gauge_metrics.csv"
WINDOW_INSET_MM = 1.0  # same exclusion next to each plate as the partition metrics
FIELDS = [
    "run",
    "gauge_lo_mm",
    "gauge_hi_mm",
    "vm_mean_all_MPa",
    "vm_mean_gauge_MPa",
    "sigma_ratio",
    "SCF_p99_all",
    "SCF_p99_gauge",
    "SCF_p999_gauge",
    "phi_all",
    "phi_gauge",
]


def metrics(vtk: Path) -> dict:
    """Gauge-only quantities for one solved model."""
    g = pv.read(vtk)
    tet_ids = np.where(np.asarray(g.celltypes) == pv.CellType.TETRA)[0]
    tets = g.extract_cells(tet_ids)
    vm = np.asarray(g.cell_data["von_mises"])[tet_ids]
    vol = np.abs(tets.compute_cell_sizes(length=False, area=False, volume=True).cell_data["Volume"])
    cent = np.asarray(tets.cell_centers().points)
    z_lo, z_hi = gauge_bounds(cent)

    # --- 2. Stress normalisation and percentile SCF -------------------------------
    win = (cent[:, 2] > z_lo + WINDOW_INSET_MM) & (cent[:, 2] < z_hi - WINDOW_INSET_MM)
    mean_all = float((vm * vol).sum() / vol.sum())
    mean_g = float((vm[win] * vol[win]).sum() / vol[win].sum())

    # --- 3. Solid fractions --------------------------------------------------------
    # Footprint from the model's x-y extent (the plates are the widest parts).
    x0, x1, y0, y1, z0, z1 = g.bounds
    foot = (x1 - x0) * (y1 - y0)
    gauge = (cent[:, 2] >= z_lo) & (cent[:, 2] <= z_hi)
    return {
        "gauge_lo_mm": round(float(z_lo), 3),
        "gauge_hi_mm": round(float(z_hi), 3),
        "vm_mean_all_MPa": round(mean_all, 4),
        "vm_mean_gauge_MPa": round(mean_g, 4),
        "sigma_ratio": round(mean_g / mean_all, 5),
        "SCF_p99_all": round(weighted_percentile(vm, vol, 99.0) / mean_all, 5),
        "SCF_p99_gauge": round(weighted_percentile(vm[win], vol[win], 99.0) / mean_g, 5),
        "SCF_p999_gauge": round(weighted_percentile(vm[win], vol[win], 99.9) / mean_g, 5),
        "phi_all": round(float(vol.sum() / (foot * (z1 - z0))), 5),
        "phi_gauge": round(float(vol[gauge].sum() / (foot * (z_hi - z_lo))), 5),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--fea-subdir", default="final")
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
                print(f"[skip] {rd.name}: no VTK in fea/{args.fea_subdir}", flush=True)
                continue
            key = str(rd.relative_to(RUNS))
            rows[key] = {"run": key, **metrics(vtk)}
            r = rows[key]
            print(
                f"{key:58s} SCF_p99 all={r['SCF_p99_all']:.3f} gauge={r['SCF_p99_gauge']:.3f}  "
                f"sigma_ratio={r['sigma_ratio']:.3f}  phi all={r['phi_all']:.3f} gauge={r['phi_gauge']:.3f}",
                flush=True,
            )
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows.values():
            w.writerow(r)


if __name__ == "__main__":
    main()
