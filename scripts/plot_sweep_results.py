#!/usr/bin/env python
"""Graphical summary of the biomimetic_pipeline FEA sweep runs.

Reads `sweep_log.csv` from each completed sweep under `runs/` and plots the key
mechanical metrics against the swept design parameter. This is a results-
inspection figure -- distinct from the manuscript Fig 5 (`make_fig5_sweep.py`)
-- produced to check the reruns at the SLA-resin modulus (3000 MPa, 200 MPa
von-Mises strain-solve target).

For each sweep it draws two stacked panels:
  - top:    effective modulus E_eff (navy, left axis) + stress-concentration
            factor SCF (orange, right axis)
  - bottom: specific toughness (navy, left axis) + crack-deflection streamline
            tortuosity, 90th percentile (orange, right axis)

Output: runs/sweep_results_summary.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

# Manuscript NavyBlue palette (matches manuscript/matter_v1/scripts/_style.py).
NAVY, ORANGE, INK, MUTED = "#1F3A5F", "#C97B27", "#1A1A1A", "#6B6B6B"

# Each sweep: (run-dir name, x-source, human-readable x-axis label).
# x-source "label:" parses the requested sweep value from the CSV `label`
# column. We deliberately use the label, NOT the param_* columns: the param_*
# columns hold the post-clamp cad_params value, which can differ from what was
# requested.
#
# NOTE: bridge thickness is swept via `bridge_ratio` (BRIDGE_DIAMETER =
# bridge_ratio x ROD_DIAMETER), NOT via a direct BRIDGE_DIAMETER override --
# feature_to_cad re-derives BRIDGE_DIAMETER from ROD_DIAMETER, so a direct
# `--param BRIDGE_DIAMETER` sweep would produce identical lattices.
SWEEPS = [
    ("sweep_bridge_ratio_resin", "label:", "Bridge-to-rod ratio"),
    ("sweep_layers_resin", "label:", "Bridge layers (count)"),
    ("sweep_decussation_resin", "label:", "Decussation scale factor"),
]


def load_rows(run_name: str):
    """Return the list of CSV rows for a sweep, or None if it has not run."""
    path = RUNS / run_name / "sweep_log.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    return rows or None


def x_value(row: dict, x_source: str) -> float:
    """Extract the swept-parameter value for one trial."""
    if x_source == "label:":
        # label looks like "decussation:factor=0.5"
        return float(row.get("label", "=nan").split("=")[-1])
    return float(row[x_source])


def num(row: dict, key: str):
    """Float-or-None accessor (sweep_log leaves missing metrics blank)."""
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def main() -> None:
    # Keep only sweeps that have produced at least one logged trial.
    available = [(n, x, lab, load_rows(n)) for (n, x, lab) in SWEEPS]
    available = [(n, x, lab, r) for (n, x, lab, r) in available if r]
    if not available:
        print("[plot_sweep_results] no sweep_log.csv found yet -- nothing to plot.")
        return

    ncol = len(available)
    fig, axes = plt.subplots(2, ncol, figsize=(5.2 * ncol, 8.0), squeeze=False, facecolor="white")
    fig.suptitle(
        "FEA sweep results -- SLA-resin modulus (3000 MPa), " "200 MPa von-Mises target",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )

    for col, (run_name, x_source, x_label, rows) in enumerate(available):
        # Sort trials by the swept parameter so the lines read left-to-right.
        rows = sorted(rows, key=lambda r: x_value(r, x_source))
        xs = [x_value(r, x_source) for r in rows]

        e_eff = [num(r, "E_effective_MPa") for r in rows]
        scf = [num(r, "SCF") for r in rows]
        tough = [num(r, "specific_toughness_mJ_per_MPa") for r in rows]
        tort = [num(r, "crack_deflection_tortuosity_p90") for r in rows]

        # --- top panel: effective modulus + stress-concentration factor -----
        ax = axes[0][col]
        ax.set_title(f"{run_name}  (n={len(rows)})", fontsize=10, color=MUTED)
        ax.plot(xs, e_eff, "o-", color=NAVY, lw=2, label="E_eff (MPa)")
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel("Effective modulus E_eff (MPa)", color=NAVY, fontsize=9)
        ax.tick_params(axis="y", labelcolor=NAVY)
        for x, y in zip(xs, e_eff):
            if y is not None:
                ax.annotate(
                    f"{y:.0f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=7,
                    color=NAVY,
                )
        axr = ax.twinx()
        axr.plot(xs, scf, "s--", color=ORANGE, lw=2, label="SCF")
        axr.set_ylabel("Stress-concentration factor", color=ORANGE, fontsize=9)
        axr.tick_params(axis="y", labelcolor=ORANGE)

        # --- bottom panel: specific toughness + crack-deflection tortuosity --
        ax2 = axes[1][col]
        ax2.plot(xs, tough, "o-", color=NAVY, lw=2)
        ax2.set_xlabel(x_label, fontsize=9)
        ax2.set_ylabel("Specific toughness (mJ/MPa)", color=NAVY, fontsize=9)
        ax2.tick_params(axis="y", labelcolor=NAVY)
        ax2r = ax2.twinx()
        ax2r.plot(xs, tort, "s--", color=ORANGE, lw=2)
        ax2r.set_ylabel("Crack-deflection tortuosity (p90)", color=ORANGE, fontsize=9)
        ax2r.tick_params(axis="y", labelcolor=ORANGE)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = RUNS / "sweep_results_summary.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"[plot_sweep_results] wrote {out}")
    # Echo a compact table to stdout for the run log.
    for run_name, x_source, x_label, rows in available:
        rows = sorted(rows, key=lambda r: x_value(r, x_source))
        print(f"\n{run_name}:")
        for r in rows:
            print(
                f"  {x_label}={x_value(r, x_source):<6g}  "
                f"E_eff={num(r,'E_effective_MPa')}  SCF={num(r,'SCF')}  "
                f"toughness={num(r,'specific_toughness_mJ_per_MPa')}  "
                f"tortuosity_p90={num(r,'crack_deflection_tortuosity_p90')}"
            )


if __name__ == "__main__":
    main()
