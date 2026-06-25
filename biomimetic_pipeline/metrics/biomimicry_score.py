"""Distance between a generated lattice's implied morphometrics and the
measured morphometrics used to seed it.

Phase 2 computes a small z-scored Euclidean distance over a fixed feature
vector — enough to close the biomimicry loop. More features can be added
without changing the interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from biomimetic_pipeline.mapping.scale import DEFAULT_BIOLOGY_SCALE_FACTOR


@dataclass
class BiomimicryResult:
    score: float  # smaller = closer to biology
    feature_pairs: List[Dict[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "biomimicry_score": float(self.score),
            "biomimicry_n_features": len(self.feature_pairs),
            "biomimicry_feature_pairs": self.feature_pairs,
        }


def compute(
    morphometrics: Dict[str, Any],
    cad_params: Dict[str, Any],
    fea_metrics: Optional[Dict[str, Any]] = None,
) -> BiomimicryResult:
    biology_scale_factor = float(
        cad_params.get("biology_scale_factor", DEFAULT_BIOLOGY_SCALE_FACTOR)
    )

    pairs: List[Dict[str, Any]] = []

    # Rod diameter
    measured_diam_um = _measured_mean_rod_diameter_um(morphometrics)
    if measured_diam_um > 0:
        measured_diam_mm = measured_diam_um * 1e-3 * biology_scale_factor
        pairs.append(
            _feature_pair(
                "rod_diameter_mm", measured_diam_mm, float(cad_params.get("ROD_DIAMETER", 0.0))
            )
        )

    # Band width / center spacing
    measured_band_width_um = _measured_mean_band_width_um(morphometrics)
    if measured_band_width_um > 0:
        expected_spacing_mm = measured_band_width_um * 1e-3 * biology_scale_factor
        pairs.append(
            _feature_pair(
                "center_spacing_mm",
                expected_spacing_mm,
                float(cad_params.get("CENTER_SPACING", 0.0)),
            )
        )

    # Dominant periodicity -> bridge layer count
    periodicity = (morphometrics.get("periodicity") or {}).get("dominant_wavelength_um_mean", 0.0)
    enamel_thickness_mm = float(cad_params.get("ENAMEL_THICKNESS", 20.0))
    if periodicity and periodicity > 0:
        expected_bridge_layers = enamel_thickness_mm / max(
            periodicity * 1e-3 * biology_scale_factor, 1e-6
        )
        pairs.append(
            _feature_pair(
                "n_bridge_layers_equivalent",
                min(max(expected_bridge_layers, 2.0), 8.0),
                float(cad_params.get("N_BRIDGE_LAYERS", 0)),
            )
        )

    # Band direction anchor
    bands = morphometrics.get("bands") or []
    if bands:
        ring_rotation = cad_params.get("RING_ROTATION", {}) or {}
        ring0 = float(ring_rotation.get("0", ring_rotation.get(0, 0.0)))
        pairs.append(
            _feature_pair(
                "band0_direction_deg", float(bands[0].get("mean_direction_deg", 0.0)), ring0
            )
        )

    # Tortuosity vs crack deflection (if available from FEA)
    measured_tort = _lookup(morphometrics, "rod_tracks_summary", "tortuosity_mean") or 1.0
    if fea_metrics and "crack_deflection_tortuosity_mean" in fea_metrics:
        pairs.append(
            _feature_pair(
                "tortuosity",
                float(measured_tort),
                float(fea_metrics["crack_deflection_tortuosity_mean"]),
            )
        )

    if not pairs:
        return BiomimicryResult(score=float("nan"), feature_pairs=[])

    # z-scored Euclidean over (measured - generated) / measured-magnitude
    import math

    sq = 0.0
    for p in pairs:
        denom = abs(p["measured"]) if abs(p["measured"]) > 1e-9 else 1.0
        z = (p["generated"] - p["measured"]) / denom
        p["z"] = z
        sq += z * z
    score = math.sqrt(sq / len(pairs))
    return BiomimicryResult(score=score, feature_pairs=pairs)


def _feature_pair(name: str, measured: float, generated: float) -> Dict[str, Any]:
    return {"feature": name, "measured": float(measured), "generated": float(generated)}


def _measured_mean_rod_diameter_um(morphometrics: Dict[str, Any]) -> float:
    dp = (morphometrics.get("depth_profiles") or {}).get("rod_diameter_um_mean") or []
    if dp:
        vals = [float(v) for v in dp if v is not None]
        if vals:
            return sum(vals) / len(vals)
    stats = morphometrics.get("rod_slice_stats") or []
    vals = [
        float(s.get("equiv_diam_um_mean", 0.0))
        for s in stats
        if s.get("equiv_diam_um_mean") is not None
    ]
    return sum(vals) / len(vals) if vals else 0.0


def _measured_mean_band_width_um(morphometrics: Dict[str, Any]) -> float:
    bands = morphometrics.get("bands") or []
    widths = [float(b.get("band_width_um", {}).get("mean", 0.0)) for b in bands]
    widths = [w for w in widths if w > 0]
    return sum(widths) / len(widths) if widths else 0.0


def _lookup(d: Dict[str, Any], *keys: str) -> Optional[float]:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None
