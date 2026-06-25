"""Append-only CSV log for sweep and optimize runs.

Schema is a superset of the stock `optimization_sweep_log.csv` used by
`cad_modeling/.../progress_tracker.py`, with extra columns for our new metrics
(crack deflection, critical strain, biomimicry score) so existing tooling
keeps working while our new analyses land in the same file.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

STOCK_KEY_METRICS = [
    "E_effective_MPa",
    "solid_fraction",
    "SCF",
    "VM_max_MPa",
    "VM_P95_MPa",
    "VM_P99_MPa",
    "BB_hotspot_frac",
    "RPJ_VM_max_MPa",
    "BB_VM_max_MPa",
    "RMS_VM_mean_MPa",
    "max_SED_MPa",
    "SED_mean_MPa",
    "triaxiality_mean",
    "energy_absorption_mJ",
    "total_elastic_energy_mJ",
    "specific_toughness_mJ_per_MPa",
    "resilience_index",
    "toughness_uniformity",
]

NEW_KEY_METRICS = [
    "crack_deflection_tortuosity_mean",
    "crack_deflection_tortuosity_p90",
    "crack_deflection_tortuosity_max",
    "critical_strain_at_200MPa",
    "critical_disp_mm_at_200MPa",
    "biomimicry_score",
    "_objective_score",
    "avg_von_mises_MPa",
]

STOCK_KEY_PARAMS = [
    "ROD_DIAMETER",
    "CENTER_SPACING",
    "BRIDGE_DIAMETER",
    "N_BRIDGE_LAYERS",
    "FILLET_RADIUS",
    "CHAMFER_SIZE",
    "JUNCTION_SPHERE_FACTOR",
    "ROD_TAPER_FACTOR",
    "TWIST_TYPE",
    "PLATE_THICKNESS",
    "PLATE_OVERLAP",
    "mesh_size",
    "junction_refinement_factor",
]


def columns(include_new_metrics: bool = True) -> List[str]:
    metric_cols = list(STOCK_KEY_METRICS)
    if include_new_metrics:
        metric_cols += NEW_KEY_METRICS
    return (
        ["iteration", "timestamp", "label", "run_name", "objective", "notes"]
        + [f"param_{p}" for p in STOCK_KEY_PARAMS]
        + metric_cols
        + ["total_time_s", "strain_solve_accepted", "strain_solve_iters"]
    )


def ensure_header(csv_path: Path) -> None:
    csv_path = Path(csv_path)
    if csv_path.exists():
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns())


def append(
    csv_path: Path,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    run_name: str,
    objective: Optional[str] = None,
    label: str = "",
    notes: str = "",
    total_time_s: float = 0.0,
) -> int:
    csv_path = Path(csv_path)
    ensure_header(csv_path)
    iteration = _next_iteration(csv_path)
    row = {
        "iteration": iteration,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "run_name": run_name,
        "objective": objective or "",
        "notes": notes,
        "total_time_s": total_time_s,
        "strain_solve_accepted": metrics.get("strain_solve_accepted", ""),
        "strain_solve_iters": metrics.get("strain_solve_iters", ""),
    }
    for p in STOCK_KEY_PARAMS:
        row[f"param_{p}"] = params.get(p, "")
    for m in STOCK_KEY_METRICS + NEW_KEY_METRICS:
        v = metrics.get(m)
        if isinstance(v, float) and v != v:  # NaN
            row[m] = ""
        else:
            row[m] = "" if v is None else v
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns())
        writer.writerow(row)
    return iteration


def _next_iteration(csv_path: Path) -> int:
    rows = 0
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        for _ in reader:
            rows += 1
    return max(0, rows - 1)  # subtract header
