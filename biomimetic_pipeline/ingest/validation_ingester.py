"""Ingest per-slice rod-level morphometry from output_validation_100/slices/
(full stack, preferred) or output_validation/slices/.

Produces `rod_slice_stats` (per-slice means + std of diameter, eccentricity,
circularity, solidity, boundary fraction) plus depth-binned diameter and
eccentricity depth profiles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from biomimetic_pipeline.orchestration.run_context import MICROCT_ROOT


def default_validation_dir() -> Path:
    pri = MICROCT_ROOT / "output_validation_100" / "slices"
    if pri.exists():
        return pri
    return MICROCT_ROOT / "output_validation" / "slices"


def ingest(directory: Optional[Path] = None, slice_thickness_um: float = 0.345) -> Dict[str, Any]:
    try:
        import numpy as np  # noqa: F401 - availability guard; numpy must import for downstream use
        import pandas as pd
    except ImportError:
        return {"_missing": True, "_reason": "pandas/numpy not available"}

    d = Path(directory) if directory else default_validation_dir()
    if not d.exists():
        return {"_missing": True, "_source": str(d)}

    slice_files = sorted(d.glob("slice_*_rods.csv"))
    if not slice_files:
        return {"_missing": True, "_source": str(d), "_reason": "no slice csvs"}

    rod_slice_stats: List[Dict[str, Any]] = []
    depth_um_arr: List[float] = []
    diam_mean: List[float] = []
    diam_std: List[float] = []
    ecc_mean: List[float] = []
    ecc_std: List[float] = []

    for f in slice_files:
        try:
            slice_idx = int(f.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        df = pd.read_csv(f)
        if df.empty:
            continue
        z_um = 10.0 + (slice_idx - 1) * slice_thickness_um
        interior = df[~df["is_boundary"]] if "is_boundary" in df.columns else df
        n_rods = int(len(df))
        n_boundary = int(df.get("is_boundary", pd.Series(False, index=df.index)).sum())
        entry: Dict[str, Any] = {"z_um": z_um, "n_rods": n_rods}
        if "equiv_diam_um" in df.columns and not interior.empty:
            entry["equiv_diam_um_mean"] = float(interior["equiv_diam_um"].mean())
            entry["equiv_diam_um_std"] = float(interior["equiv_diam_um"].std(ddof=0))
            diam_mean.append(entry["equiv_diam_um_mean"])
            diam_std.append(entry["equiv_diam_um_std"])
        if "eccentricity" in df.columns and not interior.empty:
            entry["eccentricity_mean"] = float(interior["eccentricity"].mean())
            entry["eccentricity_std"] = float(interior["eccentricity"].std(ddof=0))
            ecc_mean.append(entry["eccentricity_mean"])
            ecc_std.append(entry["eccentricity_std"])
        if "circularity" in df.columns:
            entry["circularity_mean"] = float(interior["circularity"].mean())
        if "solidity" in df.columns:
            entry["solidity_mean"] = float(interior["solidity"].mean())
        entry["boundary_frac"] = float(n_boundary / n_rods) if n_rods else 0.0
        rod_slice_stats.append(entry)
        depth_um_arr.append(z_um)

    out: Dict[str, Any] = {
        "_source": str(d),
        "rod_slice_stats": rod_slice_stats,
        "depth_profiles": {
            "depth_um": depth_um_arr,
            "rod_diameter_um_mean": diam_mean,
            "rod_diameter_um_std": diam_std,
            "eccentricity_mean": ecc_mean,
            "eccentricity_std": ecc_std,
        },
    }
    return out


if __name__ == "__main__":
    import json
    import sys

    json.dump(ingest(), sys.stdout, indent=2, default=str)
