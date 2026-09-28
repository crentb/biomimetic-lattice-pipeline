#!/usr/bin/env python
"""Figure S1 + table: mesh convergence of the reference lattice.

Purpose
-------
The reference lattice (f = 1, N = 2, H = 20 mm) was solved at Gmsh target
element sizes of 0.8, 0.65, 0.5 and 0.4 mm with identical geometry (the
0.5-mm level is the original May-2026 run; runs/live_001_digital_twin_v2).
This script tabulates, per level, the tetrahedron count and every mechanical
quantity reported in the paper -- apparent modulus, the percentile stress
ratio kappa_99 (volume-weighted P99/mean von Mises over the gauge window;
revision 2026-09-25, gauge_metrics.csv), tension-dominated volume fraction,
field inclination of the dominant principal direction, and top-plate
rotation per 1 % strain -- and plots each normalised by its finest-mesh
value, together with the stock peak-element ratio (singular; not used).

Inputs
------
runs/mesh_convergence_v1/trial_00{0,1,2}_mesh_*mm/   (0.8, 0.65, 0.4 mm)
runs/live_001_digital_twin_v2/                        (0.5 mm)
runs/_load_path_analysis/*.json                       (partition, coupling)

Outputs
-------
figures/figureS1.{pdf,png,tif}; data/mesh_convergence.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv

HERE = Path(__file__).resolve().parent
MANU = HERE.parent
BIO = MANU.parents[1]
sys.path.insert(0, str(HERE))
import _style_v2 as S  # noqa: E402

RUNS = BIO / "runs"
LEVELS = [
    (0.8, "mesh_convergence_v1/trial_000_mesh_0p8mm"),
    (0.65, "mesh_convergence_v1/trial_001_mesh_0p65mm"),
    (0.5, "live_001_digital_twin_v2"),
    (0.4, "mesh_convergence_v1/trial_002_mesh_0p4mm"),
]
OUT = MANU / "figures" / "figureS1"
VALUES = MANU / "data" / "mesh_convergence.json"


def fea_dir(run: str) -> Path | None:
    """Folder holding the run's fields: fea/final, else fea/iter_1.

    The 0.4-mm level exceeded the workstation's memory in the second strain-
    solve iteration (SuperLU fill-in); its first iteration is a complete
    linear solution at a smaller displacement, and every quantity used here
    (stiffness F/delta, percentile ratios, tension fraction, inclination,
    rotation per strain) is invariant to that linear scaling.
    """
    for sub in ("final", "iter_1"):
        d = RUNS / run / "fea" / sub
        if (d / "compound_enamel_lattice.vtk").exists() and (
            d / "global_results_compression.csv"
        ).exists():
            return d
    return None


def metrics(run: str) -> dict | None:
    d = fea_dir(run)
    lp = RUNS / "_load_path_analysis" / (run.replace("/", "__") + ".json")
    if d is None or not lp.exists():
        return None
    g = pv.read(d / "compound_enamel_lattice.vtk")
    n_tet = int((np.asarray(g.celltypes) == pv.CellType.TETRA).sum())
    gr = pd.read_csv(d / "global_results_compression.csv").iloc[0]
    k = abs(float(gr.force_N) / float(gr.disp_mm))  # axial stiffness, N/mm
    j = json.loads(lp.read_text())
    rs = pd.read_csv(RUNS / "_load_path_analysis/robust_scf.csv").set_index("run")
    gm = pd.read_csv(RUNS / "_load_path_analysis/gauge_metrics.csv").set_index("run")
    if run not in rs.index or run not in gm.index:
        return None
    return {
        "n_tet": n_tet,
        "fea_dir": d.name,
        "k_N_per_mm": k,
        "SCF": float(gm.loc[run, "SCF_p99_gauge"]),  # kappa_99: gauge P99 / gauge mean
        "SCF_whole": float(rs.loc[run, "SCF_p99"]),  # whole-model P99 / mean (superseded)
        "SCF_peak": float(rs.loc[run, "SCF_max"]),  # stock peak-element / whole-model mean
        "tens": j["load_partition"]["tensile_volume_frac"],
        "incl": j["load_partition"]["inclination_field_deg"],
        "rot": j["top_plate"]["rotation_deg_per_pct_strain"],
    }


def main() -> None:
    S.apply()
    rows = []
    for h, run in LEVELS:
        m = metrics(run)
        if m:
            rows.append({"h_mm": h, "run": run, **m})
    t = pd.DataFrame(rows).sort_values("h_mm", ascending=False)
    ref05 = json.loads((RUNS / "live_001_digital_twin_v2/metrics.json").read_text())
    e_app_05 = float(ref05["E_effective_MPa"]) * float(ref05["solid_fraction"])
    k05 = float(t[t.h_mm == 0.5].k_N_per_mm.iloc[0])
    t["E_app"] = e_app_05 * t.k_N_per_mm / k05  # same geometry -> E_app proportional to F/delta
    ref = t.iloc[-1]  # finest available level
    keys = [
        ("E_app", "apparent modulus"),
        ("SCF", r"$\kappa_{99}$ (gauge P99/mean)"),
        ("tens", "tension-dominated fraction"),
        ("incl", "principal-stress inclination"),
        ("rot", "platen rotation"),
    ]
    fig, ax = plt.subplots(figsize=(S.WIDTH_1COL, 62 * S.MM))
    for (k, lab), c, mk in zip(keys, S.CATEGORICAL + ["#6E6E6A"], "osD^v"):
        ax.plot(t.n_tet / 1e6, t[k] / ref[k], marker=mk, color=c, ms=3.4, lw=0.9, label=lab)
    ax.plot(
        t.n_tet / 1e6,
        t["SCF_peak"] / ref["SCF_peak"],
        marker="x",
        color=S.INK,
        ms=3.4,
        lw=0.7,
        ls=(0, (2, 1.5)),
        label="peak/mean (stock; singular)",
    )
    ax.axhline(1.0, color=S.RULE, lw=0.5)
    ax.set_xlabel("tetrahedra (millions)")
    ax.set_ylabel("value / finest-mesh value")
    # Mesh-size labels: vertical, in the empty band above the x axis, so
    # neighbouring levels (0.8 and 0.65 mm are close in element count) cannot collide.
    for _, r in t.iterrows():
        ax.text(
            r.n_tet / 1e6,
            0.946,
            f"h = {r.h_mm:g} mm",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=5.2,
            color=S.MUTED,
        )
    ax.set_ylim(0.94, 1.045)  # room below the curves for the mesh-size labels
    fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.19)  # keep the x label inside
    ax.set_xlim(0.5, 1.15)
    ax.legend(fontsize=5.2, loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.4)
    S.recessive_grid(ax)
    S.save(fig, OUT)
    t.to_json(VALUES, orient="records", indent=1)
    print(t.drop(columns="run").round(4).to_string(index=False))
    print("peak/mean by level:", t.SCF_peak.round(3).tolist())
    print(
        "relative change, 0.5 mm -> finest:",
        {k: round(float(t[t.h_mm == 0.5][k].iloc[0] / ref[k] - 1) * 100, 2) for k, _ in keys},
    )


if __name__ == "__main__":
    main()
