"""Ingest rod-track centerlines from output_piv/track_centerlines_piv.parquet.

Produces `rod_tracks_summary` (aggregate stats) plus tortuosity aggregates.
Centerline arrays are packed as semicolon-delimited strings in the parquet; we
only need their lengths + start/end points to derive arc/chord.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from biomimetic_pipeline.orchestration.run_context import MICROCT_ROOT


def default_piv_path() -> Path:
    return MICROCT_ROOT / "output_piv" / "track_centerlines_piv.parquet"


def ingest(path: Optional[Path] = None) -> Dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return {"_missing": True, "_reason": "pandas not available", "rod_tracks_summary": {}}

    src = Path(path) if path else default_piv_path()
    if not src.exists():
        return {"_missing": True, "_source": str(src), "rod_tracks_summary": {}}

    df = pd.read_parquet(src)
    n_tracks = int(len(df))
    if "is_complete" in df.columns:
        n_complete = int(df["is_complete"].sum())
    else:
        n_complete = n_tracks

    arc_um = _compute_arc_lengths(df) if "cx_um" in df.columns else None
    chord_um = _compute_chord_lengths(df)

    if arc_um is not None and chord_um is not None:
        import numpy as np

        mask = chord_um > 1e-6
        tort = arc_um[mask] / chord_um[mask]
        tort_mean = float(tort.mean()) if tort.size else 0.0
        tort_p90 = float(np.percentile(tort, 90)) if tort.size else 0.0
        mean_arc = float(arc_um[mask].mean()) if mask.any() else 0.0
        mean_chord = float(chord_um[mask].mean()) if mask.any() else 0.0
    else:
        tort_mean = tort_p90 = mean_arc = mean_chord = 0.0

    return {
        "_source": str(src),
        "rod_tracks_summary": {
            "n_tracks": n_tracks,
            "n_complete": n_complete,
            "mean_arc_length_um": mean_arc,
            "mean_chord_length_um": mean_chord,
            "tortuosity_mean": tort_mean,
            "tortuosity_p90": tort_p90,
        },
        "_column_set": list(df.columns),
    }


def _unpack_semicolon_array(s: Any):
    import numpy as np

    if s is None or (isinstance(s, float) and s != s):  # NaN
        return np.array([], dtype=float)
    if isinstance(s, (list, tuple)):
        return np.asarray(s, dtype=float)
    try:
        return np.fromstring(str(s).replace(",", " ").replace(";", " "), sep=" ")
    except Exception:
        return np.array([], dtype=float)


def _compute_arc_lengths(df) -> Any:
    import numpy as np

    arcs = np.zeros(len(df), dtype=float)
    cx_col = df["cx_um"].values
    cy_col = df["cy_um"].values
    cz_col = df["cz_um"].values
    for i in range(len(df)):
        cx = _unpack_semicolon_array(cx_col[i])
        cy = _unpack_semicolon_array(cy_col[i])
        cz = _unpack_semicolon_array(cz_col[i])
        if cx.size >= 2 and cx.size == cy.size == cz.size:
            dx = np.diff(cx)
            dy = np.diff(cy)
            dz = np.diff(cz)
            arcs[i] = float(np.sqrt(dx * dx + dy * dy + dz * dz).sum())
    return arcs


def _compute_chord_lengths(df) -> Any:
    import numpy as np

    if {"dx_um", "dy_um", "dz_um"}.issubset(df.columns):
        dx = df["dx_um"].to_numpy(dtype=float)
        dy = df["dy_um"].to_numpy(dtype=float)
        dz = df["dz_um"].to_numpy(dtype=float)
        return np.sqrt(dx * dx + dy * dy + dz * dz)
    return None


if __name__ == "__main__":
    import json
    import sys

    json.dump(ingest(), sys.stdout, indent=2, default=str)
