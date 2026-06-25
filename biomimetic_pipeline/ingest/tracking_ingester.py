"""Ingest rod-angle depth profiles from output_tracking_100/ (full stack, preferred)
falling back to output_tracking/ (20-slice).

Produces `depth_profiles.pitch/yaw/tilt_signed_deg` and `_abs_deg` sampled at
track-specific depth bins. instantaneous_angles.csv gives per-slice deltas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from biomimetic_pipeline.orchestration.run_context import MICROCT_ROOT


def default_tracking_dir() -> Path:
    pri = MICROCT_ROOT / "output_tracking_100"
    if (pri / "track_summary.csv").exists():
        return pri
    return MICROCT_ROOT / "output_tracking"


def ingest(directory: Optional[Path] = None) -> Dict[str, Any]:
    try:
        import numpy as np  # noqa: F401 - availability guard; numpy must import for downstream use
        import pandas as pd
    except ImportError:
        return {"_missing": True, "_reason": "pandas/numpy not available"}

    d = Path(directory) if directory else default_tracking_dir()
    inst_path = d / "instantaneous_angles.csv"
    if not inst_path.exists():
        return {"_missing": True, "_source": str(inst_path)}

    inst = pd.read_csv(inst_path)
    depth = inst.get("depth_um")
    pitch_signed = inst.get("inst_pitch_deg")
    yaw_signed = inst.get("inst_yaw_deg")

    bins = _bin_depths(depth.to_numpy(dtype=float)) if depth is not None else None
    out: Dict[str, Any] = {"_source": str(inst_path), "depth_profiles": {}}

    if bins is not None and pitch_signed is not None:
        depth_um, pitch_mean_signed, pitch_mean_abs = _aggregate_by_depth(
            bins, depth.to_numpy(dtype=float), pitch_signed.to_numpy(dtype=float)
        )
        out["depth_profiles"]["depth_um"] = depth_um.tolist()
        out["depth_profiles"]["pitch_signed_deg"] = pitch_mean_signed.tolist()
        out["depth_profiles"]["pitch_abs_deg"] = pitch_mean_abs.tolist()

    if bins is not None and yaw_signed is not None:
        _, yaw_signed_mean, yaw_abs_mean = _aggregate_by_depth(
            bins, depth.to_numpy(dtype=float), yaw_signed.to_numpy(dtype=float)
        )
        out["depth_profiles"]["yaw_signed_deg"] = yaw_signed_mean.tolist()
        out["depth_profiles"]["yaw_abs_deg"] = yaw_abs_mean.tolist()

    summary_path = d / "track_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        out["_n_tracks_summary"] = int(len(summary))
        tilt_col = (
            "tilt_deg"
            if "tilt_deg" in summary.columns
            else ("fit_tilt_deg" if "fit_tilt_deg" in summary.columns else None)
        )
        if tilt_col is not None:
            out["_tilt_from_summary_mean_deg"] = float(summary[tilt_col].mean())
    return out


def _bin_depths(depth: Any, n_bins: int = 20) -> Any:
    import numpy as np

    if depth.size == 0:
        return None
    edges = np.linspace(depth.min(), depth.max() + 1e-6, n_bins + 1)
    return edges


def _aggregate_by_depth(edges: Any, depth: Any, values: Any):
    import numpy as np

    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_signed = np.zeros_like(centers, dtype=float)
    mean_abs = np.zeros_like(centers, dtype=float)
    idx = np.searchsorted(edges, depth, side="right") - 1
    idx = np.clip(idx, 0, centers.size - 1)
    for i in range(centers.size):
        mask = idx == i
        if mask.any():
            mean_signed[i] = float(values[mask].mean())
            mean_abs[i] = float(np.abs(values[mask]).mean())
    return centers, mean_signed, mean_abs


if __name__ == "__main__":
    import json
    import sys

    json.dump(ingest(), sys.stdout, indent=2, default=str)
