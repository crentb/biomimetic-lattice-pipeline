"""Combine ingester outputs into a canonical morphometrics.json payload.

Reconciles overlapping sources (HSB angle_stats is primary, smoothed3d is a
fallback), hashes source files for provenance, and validates the result against
morphometrics.schema.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from biomimetic_pipeline.ingest import (
    hsb_ingester,
    piv_ingester,
    smoothed3d_ingester,
    som_ingester,
    tracking_ingester,
    validation_ingester,
)
from biomimetic_pipeline.orchestration.run_context import (
    PIPELINE_VERSION,
    hash_existing,
    now_iso,
)

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "morphometrics.schema.json"


def merge_all(specimen_id: str) -> Dict[str, Any]:
    som = som_ingester.ingest()
    hsb = hsb_ingester.ingest()
    piv = piv_ingester.ingest()
    tracking = tracking_ingester.ingest()
    validation = validation_ingester.ingest()
    smoothed3d = smoothed3d_ingester.ingest()

    depth_profiles = _merge_depth_profiles(tracking, validation)
    angle_stats = hsb.get("angle_stats") or smoothed3d.get("angle_stats_from_smoothed3d") or {}

    source_paths: Dict[str, str] = {
        "som": som.get("_source", ""),
        "hsb": hsb.get("_source", ""),
        "piv": piv.get("_source", ""),
        "tracking": tracking.get("_source", ""),
        "validation": validation.get("_source", ""),
        "smoothed3d": smoothed3d.get("_source", ""),
    }
    source_paths = {k: v for k, v in source_paths.items() if v}

    source_hashes = hash_existing({k: Path(v) for k, v in source_paths.items()})

    payload: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "specimen_id": specimen_id,
        "units": {"length": "um", "angle": "deg", "stress": "MPa"},
        "provenance": {
            "source_paths": source_paths,
            "source_hashes": source_hashes,
            "ingestion_timestamp_iso": now_iso(),
            "coordinate_frame": {
                "z_origin": "DEJ+10um",
                "z_axis": "+into_enamel",
                "xy_units": "um",
                "z_units": "um",
            },
            "pipeline_version": PIPELINE_VERSION,
            "scale_clamps": [],
        },
        "depth_profiles": depth_profiles,
        "angle_stats": angle_stats,
        "bands": som.get("bands", []),
        "band_boundaries": som.get("band_boundaries", {}),
        "periodicity": som.get("periodicity", {}),
        "rod_slice_stats": validation.get("rod_slice_stats", []),
        "rod_tracks_summary": piv.get("rod_tracks_summary", {}),
    }
    return payload


def _merge_depth_profiles(tracking: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, Any]:
    """Combine tracking depth profiles (pitch/yaw) with validation (diameter/ecc).

    Depth axis is taken from validation when present (denser), otherwise tracking.
    Values from the other source are linearly interpolated onto that axis.
    """
    try:
        import numpy as np
    except ImportError:
        return tracking.get("depth_profiles", {}) or validation.get("depth_profiles", {})

    dp_t = tracking.get("depth_profiles", {}) or {}
    dp_v = validation.get("depth_profiles", {}) or {}

    depth_t = np.asarray(dp_t.get("depth_um", []), dtype=float)
    depth_v = np.asarray(dp_v.get("depth_um", []), dtype=float)

    if depth_v.size and depth_t.size:
        axis = depth_v
    elif depth_v.size:
        axis = depth_v
    elif depth_t.size:
        axis = depth_t
    else:
        return {}

    out: Dict[str, Any] = {"depth_um": axis.tolist()}

    def resample(src_axis: Any, src_vals: Any) -> List[float]:
        if src_axis.size == 0 or src_vals.size == 0:
            return []
        return np.interp(axis, src_axis, src_vals).tolist()

    for k in ("pitch_signed_deg", "pitch_abs_deg", "yaw_signed_deg", "yaw_abs_deg"):
        if k in dp_t:
            out[k] = resample(depth_t, np.asarray(dp_t[k], dtype=float))

    for k in (
        "rod_diameter_um_mean",
        "rod_diameter_um_std",
        "eccentricity_mean",
        "eccentricity_std",
    ):
        if k in dp_v:
            out[k] = resample(depth_v, np.asarray(dp_v[k], dtype=float))

    return out


def validate(payload: Dict[str, Any]) -> None:
    """Validate payload against morphometrics.schema.json if jsonschema is available."""
    try:
        import jsonschema
    except ImportError:
        return
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(payload, schema)


def save(payload: Dict[str, Any], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def ingest_and_save(specimen_id: str, out_path: Path, validate_schema: bool = True) -> Path:
    payload = merge_all(specimen_id)
    if validate_schema:
        validate(payload)
    return save(payload, out_path)
