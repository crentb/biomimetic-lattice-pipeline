"""Map rod-level morphometry to rod geometry parameters.

Outputs: ROD_DIAMETER, ROD_TAPER_FACTOR, optional per_ring_diameter,
N_RINGS, and optional rod_cross_section elliptical flag.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from biomimetic_pipeline.mapping.scale import (
    DEFAULT_BIOLOGY_SCALE_FACTOR,
    DEFAULT_SLA_MIN_FEATURE_MM,
    clamp_min_feature,
    um_to_mm_scaled,
)

DEFAULT_N_RINGS_MIN = 3
DEFAULT_N_RINGS_MAX = 7
ECCENTRICITY_ELLIPSE_THRESHOLD = 0.2


def map_rod(
    morphometrics: Dict[str, Any],
    biology_scale_factor: float = DEFAULT_BIOLOGY_SCALE_FACTOR,
    sla_min_feature_mm: float = DEFAULT_SLA_MIN_FEATURE_MM,
    clamps: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    import numpy as np

    dp = morphometrics.get("depth_profiles", {}) or {}
    diam_um_arr = np.asarray(dp.get("rod_diameter_um_mean", []), dtype=float)

    if diam_um_arr.size == 0:
        slice_stats = morphometrics.get("rod_slice_stats", []) or []
        diam_um_arr = np.asarray(
            [
                s.get("equiv_diam_um_mean", 0.0)
                for s in slice_stats
                if s.get("equiv_diam_um_mean") is not None
            ],
            dtype=float,
        )

    mean_diam_um = float(diam_um_arr.mean()) if diam_um_arr.size else 4.0
    rod_diameter_mm = clamp_min_feature(
        um_to_mm_scaled(mean_diam_um, biology_scale_factor),
        "ROD_DIAMETER",
        sla_min_feature_mm,
        clamps,
    )

    taper = 0.0
    per_ring_diameter: Optional[Dict[int, float]] = None
    use_graded = False
    if diam_um_arr.size >= 3:
        diam_top = float(diam_um_arr[-1])
        diam_bot = float(diam_um_arr[0])
        diam_mid = float(diam_um_arr[diam_um_arr.size // 2]) or 1.0
        taper = float((diam_top - diam_bot) / max(diam_mid, 1e-6))

        monotone_violations = _count_monotone_violations(diam_um_arr)
        if monotone_violations / max(diam_um_arr.size - 1, 1) > 0.15:
            use_graded = True

    ecc_mean = _mean_eccentricity(morphometrics)
    rod_cross_section = None
    if ecc_mean > ECCENTRICITY_ELLIPSE_THRESHOLD:
        aspect = 1.0 / max(1.0 - ecc_mean, 0.01)
        rod_cross_section = {"shape": "ellipse", "aspect_ratio": float(aspect)}

    n_rings = _infer_n_rings(morphometrics, rod_diameter_mm)

    # lattice_cad.py validates 0.0 <= ROD_TAPER_FACTOR <= 1.0; clamp negative
    # taper (biology sometimes gets narrower at plates) to 0 and log the clamp.
    taper_clamped = max(0.0, min(1.0, taper))
    if taper_clamped != taper and clamps is not None:
        clamps.append(
            {
                "field": "ROD_TAPER_FACTOR",
                "raw_value": float(taper),
                "clamped_value": float(taper_clamped),
                "reason": "lattice_cad requires 0.0 <= taper <= 1.0",
            }
        )

    out: Dict[str, Any] = {
        "ROD_DIAMETER": float(rod_diameter_mm),
        "ROD_TAPER_FACTOR": float(taper_clamped),
        "N_RINGS": int(n_rings),
    }
    if use_graded and diam_um_arr.size >= 3:
        per_ring_diameter = {}
        ring_z = _linspace(0.0, 1.0, n_rings + 1)
        src_z = _linspace(0.0, 1.0, diam_um_arr.size)
        for i, zf in enumerate(ring_z):
            val_um = _linear_interp(zf, src_z, diam_um_arr)
            per_ring_diameter[i] = clamp_min_feature(
                um_to_mm_scaled(val_um, biology_scale_factor),
                f"per_ring_diameter[{i}]",
                sla_min_feature_mm,
                clamps,
            )
        out["per_ring_diameter"] = per_ring_diameter
    if rod_cross_section is not None:
        out["rod_cross_section"] = rod_cross_section
    return out


def _count_monotone_violations(arr: Any) -> int:
    import numpy as np

    diffs = np.diff(arr)
    if diffs.size == 0:
        return 0
    sign = np.sign(diffs[diffs != 0])
    if sign.size == 0:
        return 0
    return int(np.sum(sign != sign[0]))


def _mean_eccentricity(morphometrics: Dict[str, Any]) -> float:
    dp = morphometrics.get("depth_profiles", {}) or {}
    ecc_arr = dp.get("eccentricity_mean", [])
    if ecc_arr:
        vals = [float(x) for x in ecc_arr if x is not None]
        if vals:
            return sum(vals) / len(vals)
    slice_stats = morphometrics.get("rod_slice_stats", []) or []
    vals = [
        float(s.get("eccentricity_mean", 0.0))
        for s in slice_stats
        if s.get("eccentricity_mean") is not None
    ]
    return (sum(vals) / len(vals)) if vals else 0.0


def _infer_n_rings(morphometrics: Dict[str, Any], rod_diameter_mm: float) -> int:
    # Use SOM image_size_um as proxy for specimen radius, else default to 5
    som_source = None
    # There's no direct image_size here after merge; rely on band-width derived spacing
    # Default fall-through to 5 preserves lattice_cad.DEFAULTS.
    n = 5
    return max(DEFAULT_N_RINGS_MIN, min(DEFAULT_N_RINGS_MAX, n))


def _linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + step * i for i in range(n)]


def _linear_interp(x: float, xs: List[float], ys: Any) -> float:
    if len(xs) == 0:
        return 0.0
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = float(ys[i - 1]), float(ys[i])
            t = (x - x0) / max(x1 - x0, 1e-12)
            return y0 + t * (y1 - y0)
    return float(ys[-1])
