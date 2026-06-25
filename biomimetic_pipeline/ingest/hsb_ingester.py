"""Ingest HSB descriptive angle stats from output_hsb/ and output_hsb_roi/.

Each folder has a `piv_stats.csv` with descriptive stats for Pitch, Yaw, Tilt.
Prefers the ROI folder when both exist (tighter spatial resolution).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from biomimetic_pipeline.orchestration.run_context import MICROCT_ROOT

EXPECTED_ANGLES = ("pitch", "yaw", "tilt")
STAT_FIELDS = (
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
)
STAT_ALIASES = {"skewness": "skew", "kurtosis": "kurt"}


def default_hsb_paths() -> List[Path]:
    return [MICROCT_ROOT / "output_hsb_roi", MICROCT_ROOT / "output_hsb"]


def ingest(paths: Optional[List[Path]] = None) -> Dict[str, Any]:
    candidate_dirs = paths or default_hsb_paths()
    out: Dict[str, Any] = {"_source": None, "angle_stats": {}}

    for d in candidate_dirs:
        csv_path = Path(d) / "piv_stats.csv"
        if csv_path.exists():
            angle_stats = _parse_piv_stats(csv_path)
            if angle_stats:
                out["_source"] = str(csv_path)
                out["angle_stats"] = angle_stats
                break

    if not out["angle_stats"]:
        out["_missing"] = True

    return out


def _parse_piv_stats(csv_path: Path) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        return stats

    # The actual file is long-format: angle,statistic,value
    fieldnames = list(rows[0].keys())
    if "statistic" in fieldnames and "value" in fieldnames:
        angle_col = next(
            (c for c in fieldnames if c != "statistic" and c != "value"), fieldnames[0]
        )
        for row in rows:
            label = (row.get(angle_col) or "").strip().lower()
            if label not in EXPECTED_ANGLES:
                continue
            stat_name = (row.get("statistic") or "").strip().lower()
            stat_name = STAT_ALIASES.get(stat_name, stat_name)
            if stat_name not in STAT_FIELDS:
                continue
            try:
                v = float(row["value"])
            except (TypeError, ValueError):
                continue
            stats.setdefault(label, {})[stat_name] = v
        for label in list(stats.keys()):
            if "n" in stats[label]:
                stats[label]["n"] = int(stats[label]["n"])
        return stats

    # Fall back to wide-format (angle row with one column per stat).
    label_col = _find_label_column(fieldnames)
    if label_col is None:
        return stats
    for row in rows:
        label = (row.get(label_col) or "").strip().lower()
        if label not in EXPECTED_ANGLES:
            continue
        stat: Dict[str, float] = {}
        for key in STAT_FIELDS:
            if key in row and row[key] not in ("", None):
                try:
                    stat[key] = float(row[key])
                except ValueError:
                    continue
        if "n" in stat:
            stat["n"] = int(stat["n"])
        stats[label] = stat
    return stats


def _find_label_column(fieldnames: Any) -> Optional[str]:
    if not fieldnames:
        return None
    for cand in ("angle", "name", "label", "variable", ""):
        if cand in fieldnames:
            return cand
    return list(fieldnames)[0] if fieldnames else None


if __name__ == "__main__":
    import json
    import sys

    json.dump(ingest(), sys.stdout, indent=2)
