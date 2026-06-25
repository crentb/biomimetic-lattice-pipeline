"""Biology→print scale conversion and SLA manufacturability clamps.

Single source of truth: every other mapping module calls `um_to_mm_scaled` and
`clamp_min_feature` from here so all clamps are logged consistently into the
morphometrics provenance block.
"""

from __future__ import annotations

from typing import Dict, List

# Biology-to-print scale factor. Measured rod diameter in real enamel is
# ~4 um; the stock continuous-twist reference geometry uses ROD_DIAMETER=2 mm,
# which is ~500x biology. We keep that as the default so mapped lattices sit
# in the same SLA-printable regime as the stock model.
DEFAULT_BIOLOGY_SCALE_FACTOR = 500.0
DEFAULT_SLA_MIN_FEATURE_MM = 0.3

# Bridge-to-plate clearance required so the top bridge layer does not visually
# intersect, fuse with, or merge into the loading plate during CAD or print.
# Sized for FDM at 0.4 mm nozzle (the most permissive target manufacturing
# process in this project): 0.4 mm nozzle floor + 0.1 mm safety margin =
# 0.5 mm. SLA at 0.05-0.1 mm tolerance is far inside this envelope, so this
# value is also safe for the SLA path. The historical 0.05 mm value was sized
# only for SLA min-feature and was rediscovered (2026-05-26) to leave a
# functionally-fused top-bridge geometry on every sweep_layers trial. See
# project_bridge_plate_clearance memory.
DEFAULT_PLATE_CLEARANCE_MM = 0.5

# Bridge-diameter / rod-diameter thickness ratio. Real enamel is dense and
# space-filling: rods occupy ~75-85% of inter-rod center spacing, with
# protein-rich matrix filling the rest. The historical 0.5 default produced
# bridges at 50% of rod thickness, leaving the structure quite open
# (rod/spacing ≈ 67%, packing fraction ≈ 40%). 0.8 brings bridges closer to
# space-filling territory while preserving BRIDGE_DIAMETER < ROD_DIAMETER
# (the invariant lattice_cad.py enforces). Tuned 2026-05-26 to better
# approximate enamel density. Override per-call via cad['bridge_ratio'].
DEFAULT_BRIDGE_RATIO = 0.8


def um_to_mm_scaled(
    value_um: float, biology_scale_factor: float = DEFAULT_BIOLOGY_SCALE_FACTOR
) -> float:
    return float(value_um) * 1e-3 * biology_scale_factor


def clamp_min_feature(
    value_mm: float,
    field: str,
    sla_min_feature_mm: float = DEFAULT_SLA_MIN_FEATURE_MM,
    clamps: List[Dict] = None,
) -> float:
    if value_mm < sla_min_feature_mm:
        if clamps is not None:
            clamps.append(
                {
                    "field": field,
                    "raw_value": float(value_mm),
                    "clamped_value": float(sla_min_feature_mm),
                    "reason": f"SLA min feature {sla_min_feature_mm} mm",
                }
            )
        return sla_min_feature_mm
    return float(value_mm)


def scale_and_clamp(
    value_um: float,
    field: str,
    biology_scale_factor: float = DEFAULT_BIOLOGY_SCALE_FACTOR,
    sla_min_feature_mm: float = DEFAULT_SLA_MIN_FEATURE_MM,
    clamps: List[Dict] = None,
) -> float:
    mm = um_to_mm_scaled(value_um, biology_scale_factor)
    return clamp_min_feature(mm, field, sla_min_feature_mm, clamps)
