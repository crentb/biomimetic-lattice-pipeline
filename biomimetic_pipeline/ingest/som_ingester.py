"""Ingest SOM band morphometrics from som_approach/output_som_morphometrics/morphometrics.json.

Returns a dict fragment containing `bands`, `band_boundaries`, `periodicity` in the
canonical morphometrics schema units (lengths in um, angles in deg).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from biomimetic_pipeline.orchestration.run_context import MICROCT_ROOT


def default_som_path() -> Path:
    return MICROCT_ROOT / "som_approach" / "output_som_morphometrics" / "morphometrics.json"


def ingest(path: Optional[Path] = None) -> Dict[str, Any]:
    src = Path(path) if path else default_som_path()
    if not src.exists():
        return {
            "_missing": True,
            "_source": str(src),
            "bands": [],
            "band_boundaries": {},
            "periodicity": {},
        }

    raw = json.loads(src.read_text())
    out: Dict[str, Any] = {"_source": str(src)}

    bands_raw = raw.get("bands", {})
    bands_out: List[Dict[str, Any]] = []
    for key in sorted(bands_raw.keys()):
        b = bands_raw[key]
        try:
            band_id = int(str(key).split("_")[-1])
        except ValueError:
            band_id = len(bands_out)
        bw_um = b.get("band_width_um", {})
        bands_out.append(
            {
                "band_id": band_id,
                "area_fraction_pct": float(b.get("area_fraction_pct", 0.0)),
                "mean_direction_deg": float(b.get("mean_direction_deg", 0.0)),
                "circular_variance": float(b.get("circular_variance", 0.0)),
                "mean_displacement_mag_um": float(b.get("mean_displacement_mag_um", 0.0)),
                "band_width_um": {
                    "mean": float(bw_um.get("mean", 0.0)),
                    "std": float(bw_um.get("std", 0.0)),
                    "min": float(bw_um.get("min", 0.0)),
                    "max": float(bw_um.get("max", 0.0)),
                },
            }
        )
    out["bands"] = bands_out

    boundaries: Dict[str, float] = {}
    for k, v in raw.get("boundaries", {}).items():
        if isinstance(v, dict):
            mismatch = v.get("angular_mismatch_deg") or v.get("mean_direction_diff_deg")
            if mismatch is not None:
                boundaries[k] = float(mismatch)
        elif isinstance(v, (int, float)):
            boundaries[k] = float(v)
    out["band_boundaries"] = boundaries

    periodicity = raw.get("periodicity", {}) or {}
    out["periodicity"] = {
        "dominant_wavelength_um_mean": float(periodicity.get("dominant_wavelength_um_mean", 0.0)),
        "dominant_wavelength_um_std": float(periodicity.get("dominant_wavelength_um_std", 0.0)),
    }

    out["_pixel_size_um"] = float(raw.get("pixel_size_um", 0.0))
    image_size_um = raw.get("image_size_um") or [0.0, 0.0]
    out["_image_size_um"] = [float(x) for x in image_size_um]
    out["_n_slices"] = int(raw.get("n_slices", 0))
    return out


if __name__ == "__main__":
    import sys

    result = ingest()
    json.dump(result, sys.stdout, indent=2)
