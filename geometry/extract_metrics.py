#!/usr/bin/env python3
# Copyright 2026 Cameron B. Renteria
# SPDX-License-Identifier: Apache-2.0
"""
extract_metrics.py
==================
Parses FEA output CSVs from compression simulations and computes
optimization-relevant scalar metrics for decussated lattice structures.

Expected files in results_dir:
    - element_results_compression.csv   (per-element stress/strain/geometry)
    - global_results_compression.csv    (summary row)
    - force_displacement_compression.csv
    - reaction_force_bottom_z_compression.txt

Usage:
    python extract_metrics.py --results-dir . --output optimization_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# NumPy 2.0 renamed trapz → trapezoid; support both for forward/backward compat
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ---------------------------------------------------------------------------
# Region classification
# ---------------------------------------------------------------------------

def classify_regions(
    cz: np.ndarray,
    bridge_elevations: Sequence[float] = (1.25, 7.08, 12.92, 18.75),
    bb_half_width: float = 0.5,
    plate_overlap: float = 0.5,
    cut_top_z: float = 20.0,
) -> np.ndarray:
    """Classify each element into a structural region based on z-coordinate.

    Regions (priority order):
        BP       Bottom Plate:             cz < 0
        RPJ-B    Rod-Plate Jcn Bottom:     0 <= cz < plate_overlap
        BB       Bridge Band:              within bb_half_width of bridge elevation
        RPJ-T    Rod-Plate Jcn Top:        cz > (cut_top_z - plate_overlap) (and <= cut_top_z)
        TP       Top Plate:                cz > cut_top_z
        RMS      Rod Mid-span:             everything else
    """
    rpj_top_start = cut_top_z - plate_overlap
    labels = np.full(cz.shape, "RMS", dtype="<U5")
    labels[cz < 0.0] = "BP"
    mask_rpjb = (cz >= 0.0) & (cz < plate_overlap)
    labels[mask_rpjb] = "RPJ-B"
    for z_bridge in bridge_elevations:
        mask_bb = np.abs(cz - z_bridge) <= bb_half_width
        labels[mask_bb] = "BB"
    labels[cz > rpj_top_start] = "RPJ-T"
    labels[cz > cut_top_z] = "TP"
    return labels


# ---------------------------------------------------------------------------
# Volume-weighted percentile helper
# ---------------------------------------------------------------------------

def volume_weighted_percentiles(
    values: np.ndarray,
    volumes: np.ndarray,
    percentiles: Sequence[float],
) -> np.ndarray:
    """Compute volume-weighted percentiles of *values*."""
    order = np.argsort(values)
    sorted_vals = values[order]
    sorted_vols = volumes[order]
    cum_vol = np.cumsum(sorted_vols)
    total_vol = cum_vol[-1]
    cum_frac = cum_vol / total_vol

    results = np.empty(len(percentiles), dtype=np.float64)
    for i, p in enumerate(percentiles):
        target = p / 100.0
        idx = np.searchsorted(cum_frac, target, side="left")
        idx = min(idx, len(sorted_vals) - 1)
        results[i] = sorted_vals[idx]
    return results


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_metrics(
    results_dir: str,
    specimen_height: float = 22.4,
    bridge_elevations: Sequence[float] = (1.25, 7.08, 12.92, 18.75),
    bb_half_width: float = 0.5,
    plate_overlap: float = 0.5,
    cut_top_z: float = 20.0,
    chunk_size: int = 200_000,
) -> Dict[str, float]:
    """Extract optimisation-relevant scalar metrics from FEA output files.

    Parameters
    ----------
    results_dir : str
        Directory containing the four FEA output files.
    specimen_height : float
        Specimen height in mm (z_max - z_min of model).
    bridge_elevations : sequence of float
        Z-coordinates of bridge band centres.
    bb_half_width : float
        Half-width of bridge band region in mm.
    plate_overlap : float
        Plate overlap into lattice in mm.
    cut_top_z : float
        Z-height where rods are trimmed (top of lattice).

    Returns
    -------
    dict
        Dictionary mapping metric names to float values.
    """
    rdir = Path(results_dir)

    # ------------------------------------------------------------------
    # 1. Read small helper files
    # ------------------------------------------------------------------
    rf_path = rdir / "reaction_force_bottom_z_compression.txt"
    with open(rf_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            reaction_force = float(line)
            break

    fd = pd.read_csv(rdir / "force_displacement_compression.csv")
    displacement_mm = float(fd["displacement_mm"].iloc[-1])

    # F-d curve metrics (work with absolute values; compression is negative)
    disp_arr  = fd["displacement_mm"].values
    force_arr = fd["force_N"].values
    # Area under |F|–|δ| curve = elastic energy absorbed (N·mm = mJ)
    energy_absorption_mJ = float(np.abs(_trapz(force_arr, disp_arr)))
    # Initial stiffness: slope of |F| vs |δ| via linear fit (skip the zero point)
    nz = np.abs(disp_arr) > 0
    if nz.sum() >= 2:
        fd_stiffness_N_per_mm = float(
            np.polyfit(np.abs(disp_arr[nz]), np.abs(force_arr[nz]), 1)[0]
        )
    else:
        fd_stiffness_N_per_mm = float(np.abs(force_arr[-1]) / np.abs(displacement_mm))

    # ------------------------------------------------------------------
    # 2. Chunked read of element results
    # ------------------------------------------------------------------
    elem_path = rdir / "element_results_compression.csv"
    needed_cols = [
        "sxx_MPa", "syy_MPa", "szz_MPa",
        "von_mises_MPa", "energy_MPa", "volume_mm3",
        "cx_mm", "cy_mm", "cz_mm",
    ]

    chunks = []
    for chunk in pd.read_csv(elem_path, chunksize=chunk_size, usecols=needed_cols):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    vm = df["von_mises_MPa"].values
    vol = df["volume_mm3"].values
    energy = df["energy_MPa"].values
    sxx = df["sxx_MPa"].values
    syy = df["syy_MPa"].values
    szz = df["szz_MPa"].values
    cz = df["cz_mm"].values
    cx = df["cx_mm"].values
    cy = df["cy_mm"].values

    total_volume = vol.sum()

    # ------------------------------------------------------------------
    # 3. Stiffness metrics
    # ------------------------------------------------------------------
    cross_section_area = total_volume / specimen_height
    applied_strain = abs(displacement_mm) / specimen_height
    E_effective = abs(reaction_force) / (cross_section_area * applied_strain)

    bb_vol = (
        (cx.max() - cx.min())
        * (cy.max() - cy.min())
        * (cz.max() - cz.min())
    )
    solid_fraction = total_volume / bb_vol if bb_vol > 0 else np.nan

    # ------------------------------------------------------------------
    # 4. Stress-concentration metrics
    # ------------------------------------------------------------------
    vm_mean = np.average(vm, weights=vol)
    vm_max = vm.max()
    SCF = vm_max / vm_mean if vm_mean > 0 else np.nan

    pctiles = volume_weighted_percentiles(vm, vol, [50, 90, 95, 99])
    vm_p50, vm_p90, vm_p95, vm_p99 = pctiles

    # ------------------------------------------------------------------
    # 5. Failure-prediction metrics
    # ------------------------------------------------------------------
    max_SED = energy.max()
    SED_mean = np.average(energy, weights=vol)

    hydrostatic = (sxx + syy + szz) / 3.0
    with np.errstate(divide="ignore", invalid="ignore"):
        triax = np.where(vm > 0, hydrostatic / vm, 0.0)
    triax_mean = np.average(triax, weights=vol)
    triax_var = np.average((triax - triax_mean) ** 2, weights=vol)
    triax_std = np.sqrt(triax_var)

    # ------------------------------------------------------------------
    # 6. Hotspot metrics
    # ------------------------------------------------------------------
    hotspot_threshold = vm_p95
    hotspot_mask = vm >= hotspot_threshold
    hotspot_volume_frac = vol[hotspot_mask].sum() / total_volume

    top200_idx = np.argpartition(vm, -200)[-200:]
    hotspot_count_top200 = len(top200_idx)

    regions = classify_regions(cz, bridge_elevations, bb_half_width, plate_overlap, cut_top_z)
    top200_regions = regions[top200_idx]
    bb_hotspot_frac = np.sum(top200_regions == "BB") / len(top200_idx)

    # ------------------------------------------------------------------
    # 7. Regional metrics
    # ------------------------------------------------------------------
    bb_mask = regions == "BB"
    rpj_mask = (regions == "RPJ-B") | (regions == "RPJ-T")
    rms_mask = regions == "RMS"

    BB_VM_max = vm[bb_mask].max() if bb_mask.any() else np.nan
    RPJ_VM_max = vm[rpj_mask].max() if rpj_mask.any() else np.nan
    RMS_VM_mean = np.average(vm[rms_mask], weights=vol[rms_mask]) if rms_mask.any() else np.nan

    # ------------------------------------------------------------------
    # 8. Toughness & damage-tolerance metrics
    # ------------------------------------------------------------------
    # Total elastic strain energy: MPa * mm³ = N/mm² * mm³ = N·mm = mJ
    total_elastic_energy_mJ = float(SED_mean * total_volume)

    # Specific toughness: energy absorbed per unit of worst-case stress.
    # Higher = more energy stored before reaching the peak stress — the
    # primary damage-tolerance objective.
    specific_toughness = total_elastic_energy_mJ / vm_max if vm_max > 0 else np.nan

    # Resilience index: effective stiffness per unit of stress concentration.
    # Higher = stiffer structure with more uniform load distribution.
    resilience_index = E_effective / SCF if SCF > 0 else np.nan

    # Toughness uniformity: mean SED / max SED.
    # 1.0 = perfectly uniform energy absorption (ideal damage tolerance).
    # Lower values indicate localised energy concentration (weak points).
    toughness_uniformity = SED_mean / max_SED if max_SED > 0 else np.nan

    # Region-wise mean strain energy density
    BB_SED_mean = (
        float(np.average(energy[bb_mask], weights=vol[bb_mask]))
        if bb_mask.any() else np.nan
    )
    RPJ_SED_mean = (
        float(np.average(energy[rpj_mask], weights=vol[rpj_mask]))
        if rpj_mask.any() else np.nan
    )
    RMS_SED_mean = (
        float(np.average(energy[rms_mask], weights=vol[rms_mask]))
        if rms_mask.any() else np.nan
    )

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    metrics: Dict[str, float] = {
        # --- Stiffness ---
        "E_effective_MPa": float(E_effective),
        "solid_fraction": float(solid_fraction),
        "reaction_force_N": float(reaction_force),
        # --- Stress concentration ---
        "SCF": float(SCF),
        "VM_mean_MPa": float(vm_mean),
        "VM_P50_MPa": float(vm_p50),
        "VM_P90_MPa": float(vm_p90),
        "VM_P95_MPa": float(vm_p95),
        "VM_P99_MPa": float(vm_p99),
        "VM_max_MPa": float(vm_max),
        # --- Energy ---
        "max_SED_MPa": float(max_SED),
        "SED_mean_MPa": float(SED_mean),
        # --- Failure prediction ---
        "triaxiality_mean": float(triax_mean),
        "triaxiality_std": float(triax_std),
        # --- Hotspot distribution ---
        "hotspot_volume_frac": float(hotspot_volume_frac),
        "hotspot_count_top200": int(hotspot_count_top200),
        "BB_hotspot_frac": float(bb_hotspot_frac),
        # --- Regional stress ---
        "BB_VM_max_MPa": float(BB_VM_max),
        "RPJ_VM_max_MPa": float(RPJ_VM_max),
        "RMS_VM_mean_MPa": float(RMS_VM_mean),
        # --- Force-displacement ---
        "energy_absorption_mJ": float(energy_absorption_mJ),
        "fd_stiffness_N_per_mm": float(fd_stiffness_N_per_mm),
        # --- Toughness / damage-tolerance ---
        "total_elastic_energy_mJ": float(total_elastic_energy_mJ),
        "specific_toughness_mJ_per_MPa": float(specific_toughness),
        "resilience_index": float(resilience_index),
        "toughness_uniformity": float(toughness_uniformity),
        "BB_SED_mean_MPa": float(BB_SED_mean),
        "RPJ_SED_mean_MPa": float(RPJ_SED_mean),
        "RMS_SED_mean_MPa": float(RMS_SED_mean),
    }
    return metrics


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
    """Save metrics dictionary to a JSON file."""
    with open(output_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Metrics saved to {output_path}")


def append_to_sweep_log(
    metrics: Dict[str, float],
    params: Dict[str, float],
    log_path: str,
) -> None:
    """Append one design-evaluation row to a CSV sweep log.

    Creates header row if the log file does not yet exist.
    """
    row = {**params, **metrics}
    file_exists = os.path.isfile(log_path)

    with open(log_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Row appended to {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(metrics: Dict[str, float]) -> None:
    """Pretty-print a summary table of all metrics."""
    sections = {
        "Force-Displacement": [
            "energy_absorption_mJ", "fd_stiffness_N_per_mm",
        ],
        "Stiffness": [
            "E_effective_MPa", "solid_fraction", "reaction_force_N",
        ],
        "Stress Concentration": [
            "SCF", "VM_mean_MPa", "VM_P50_MPa", "VM_P90_MPa",
            "VM_P95_MPa", "VM_P99_MPa", "VM_max_MPa",
        ],
        "Failure Prediction": [
            "max_SED_MPa", "SED_mean_MPa", "triaxiality_mean", "triaxiality_std",
        ],
        "Hotspot": [
            "hotspot_volume_frac", "hotspot_count_top200", "BB_hotspot_frac",
        ],
        "Regional Stress": [
            "BB_VM_max_MPa", "RPJ_VM_max_MPa", "RMS_VM_mean_MPa",
        ],
        "Toughness / Damage Tolerance": [
            "total_elastic_energy_mJ",
            "specific_toughness_mJ_per_MPa",
            "resilience_index",
            "toughness_uniformity",
            "BB_SED_mean_MPa",
            "RPJ_SED_mean_MPa",
            "RMS_SED_mean_MPa",
        ],
    }

    width = 60
    print("=" * width)
    print("  FEA OPTIMISATION METRICS SUMMARY")
    print("=" * width)
    for section, keys in sections.items():
        print(f"\n  [{section}]")
        print("-" * width)
        for k in keys:
            v = metrics.get(k, float("nan"))
            if isinstance(v, int):
                print(f"    {k:<30s} {v:>16d}")
            elif abs(v) >= 1e5 or (abs(v) < 1e-2 and v != 0):
                print(f"    {k:<30s} {v:>16.6e}")
            else:
                print(f"    {k:<30s} {v:>16.4f}")
    print("=" * width)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract optimisation metrics from FEA compression results."
    )
    parser.add_argument("--results-dir", default=".",
                        help="Directory containing FEA output files.")
    parser.add_argument("--output", default="optimization_metrics.json",
                        help="Output JSON file path.")
    parser.add_argument("--height", type=float, default=18.9,
                        help="Specimen height in mm.")
    parser.add_argument("--params-json", default=None,
                        help="Path to lattice_params.json written by lattice_cad.py. "
                             "Auto-detected in results-dir if not specified.")
    args = parser.parse_args()

    # Auto-detect lattice_params.json if not given explicitly
    bridge_elevations = None
    plate_overlap = 0.5
    cut_top_z = 20.0
    specimen_height = args.height  # CLI default; overridden by lattice_params.json if available
    rdir = Path(args.results_dir)
    params_path = Path(args.params_json) if args.params_json else rdir / "lattice_params.json"
    if params_path.is_file():
        with open(params_path) as fh:
            lp = json.load(fh)
        bridge_elevations = lp.get("bridge_elevations")
        plate_overlap = lp.get("plate_overlap", plate_overlap)
        cut_top_z = lp.get("cut_top_z", cut_top_z)
        # Use actual model height from sidecar unless user explicitly provided --height
        if args.height == 18.9:  # default sentinel: user didn't override
            auto_h = lp.get("specimen_height")
            if auto_h is None:
                z_min = lp.get("model_z_min")
                z_max = lp.get("model_z_max")
                if z_min is not None and z_max is not None:
                    auto_h = z_max - z_min
            if auto_h is not None:
                specimen_height = auto_h
                print(f"[extract_metrics] specimen_height from lattice_params.json: {specimen_height:.3f} mm")
        if bridge_elevations:
            print(f"[extract_metrics] Using bridge_elevations from {params_path.name}: "
                  f"{[f'{z:.2f}' for z in bridge_elevations]}")

    kwargs = dict(specimen_height=specimen_height, plate_overlap=plate_overlap, cut_top_z=cut_top_z)
    if bridge_elevations:
        kwargs["bridge_elevations"] = bridge_elevations

    metrics = extract_metrics(args.results_dir, **kwargs)
    _print_summary(metrics)
    save_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
