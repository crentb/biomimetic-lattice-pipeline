"""Ingest smoothed 3D rod trajectories and yaw/pitch stats from output_3d_v2/.

Output_3d_v2 carries smoothed per-track summaries and a separate angle-stats CSV.
We use this as a secondary source (lower priority than HSB) for angle_stats and
as a cross-check for rod_tracks_summary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from biomimetic_pipeline.orchestration.run_context import MICROCT_ROOT


def default_dir() -> Path:
    return MICROCT_ROOT / "output_3d_v2"


def ingest(directory: Optional[Path] = None) -> Dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return {"_missing": True, "_reason": "pandas not available"}

    d = Path(directory) if directory else default_dir()
    out: Dict[str, Any] = {"_source": str(d)}

    ys_path = d / "yaw_pitch_stats.csv"
    if ys_path.exists():
        df = pd.read_csv(ys_path)
        out["angle_stats_from_smoothed3d"] = _parse_angle_stats(df)

    ts_path = d / "track_summary_smoothed.csv"
    if ts_path.exists():
        ts = pd.read_csv(ts_path)
        summary: Dict[str, Any] = {"n_tracks": int(len(ts))}
        if "mean_diam_um" in ts.columns:
            summary["mean_diam_um"] = float(ts["mean_diam_um"].mean())
        if "mean_radius_um" in ts.columns:
            summary["mean_radius_um"] = float(ts["mean_radius_um"].mean())
        if "mean_ecc" in ts.columns:
            summary["mean_ecc"] = float(ts["mean_ecc"].mean())
        out["smoothed3d_summary"] = summary

    if "angle_stats_from_smoothed3d" not in out and "smoothed3d_summary" not in out:
        out["_missing"] = True
    return out


def _parse_angle_stats(df: Any) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    if df is None or df.empty:
        return stats
    label_col = None
    for cand in ("angle", "name", "label", "variable"):
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        label_col = df.columns[0]
    for _, row in df.iterrows():
        label = str(row[label_col]).strip().lower()
        if label not in ("pitch", "yaw", "tilt"):
            continue
        st: Dict[str, float] = {}
        for k in (
            "n",
            "mean",
            "median",
            "std",
            "sem",
            "iqr",
            "skew",
            "kurt",
            "p5",
            "p95",
            "min",
            "max",
        ):
            if k in row and row[k] == row[k]:  # not NaN
                try:
                    st[k] = float(row[k])
                except (TypeError, ValueError):
                    pass
        if "n" in st:
            st["n"] = int(st["n"])
        stats[label] = st
    return stats


if __name__ == "__main__":
    import json
    import sys

    json.dump(ingest(), sys.stdout, indent=2, default=str)
