"""Map SOM band morphometrics to bridge geometry and ring anchoring.

Uses:
  - periodicity.dominant_wavelength_um_mean  -> N_BRIDGE_LAYERS
  - bands[0].mean_direction_deg              -> anchor for RING_ROTATION[0]
  - band_width_um.mean                        -> CENTER_SPACING
"""

from __future__ import annotations

from typing import Any, Dict, List

from biomimetic_pipeline.mapping.scale import (
    DEFAULT_BIOLOGY_SCALE_FACTOR,
    DEFAULT_PLATE_CLEARANCE_MM,
    DEFAULT_SLA_MIN_FEATURE_MM,
    um_to_mm_scaled,
)


def map_bridges(
    morphometrics: Dict[str, Any],
    enamel_thickness_mm: float,
    biology_scale_factor: float = DEFAULT_BIOLOGY_SCALE_FACTOR,
) -> Dict[str, Any]:
    """Return a fragment with N_BRIDGE_LAYERS (BRIDGE_Z_OFFSETS is set later
    by feature_to_cad once ROD_DIAMETER, BRIDGE_DIAMETER, and junction-sphere
    factor are all known — the safe zone depends on all three)."""
    period = (morphometrics.get("periodicity") or {}).get("dominant_wavelength_um_mean", 0.0)
    if period and period > 0:
        wavelength_mm = um_to_mm_scaled(period, biology_scale_factor)
        raw_n = round(enamel_thickness_mm / max(wavelength_mm, 1e-6))
    else:
        raw_n = 4
    n_bridge_layers = max(2, min(8, int(raw_n)))

    return {"N_BRIDGE_LAYERS": n_bridge_layers, "BRIDGE_Z_OFFSETS": None}


def compute_safe_bridge_elevations(
    n_bridge_layers: int,
    rod_diameter_mm: float,
    bridge_diameter_mm: float,
    enamel_thickness_mm: float,
    plate_overlap_mm: float,
    junction_sphere_factor: float,
    clearance_mm: float = DEFAULT_PLATE_CLEARANCE_MM,
) -> List[float]:
    """Place bridge elevations so that (a) the bridge cylinder does not
    intersect the top or bottom loading plate, AND (b) the junction sphere
    (diameter = junction_sphere_factor * rod_diameter) added at each bridge-rod
    junction also stays clear of the plates.

    Stock `lattice_cad.py:472-481` only subtracts `bridge_half` from the plate
    faces when auto-placing bridges — if JUNCTION_SPHERE_FACTOR > 0 the junction
    spheres at bridge ends can still reach into the plates, producing degenerate
    OCC topology. This helper includes the sphere radius in the clearance.

    Layout (mm, z-axis):
        z = 0 .................... plate_top_face (bottom plate)
        z = plate_overlap ........ safe_z_min = plate_overlap + max(bridge_half, sphere_radius) + clearance
        ...
        z = safe_z_max .......... enamel_thickness - plate_overlap - max(bridge_half, sphere_radius) - clearance
        z = enamel_thickness ..... cut_top_z = enamel_thickness - plate_overlap is top of rod region
                                   top plate sits at cut_top_z..enamel_thickness
    """
    bridge_half = 0.5 * float(bridge_diameter_mm)
    sphere_radius = 0.5 * float(junction_sphere_factor) * float(rod_diameter_mm)
    margin = max(bridge_half, sphere_radius) + float(clearance_mm)

    cut_top_z = float(enamel_thickness_mm) - float(plate_overlap_mm)
    safe_z_min = float(plate_overlap_mm) + margin
    safe_z_max = cut_top_z - float(plate_overlap_mm) - margin

    if safe_z_max <= safe_z_min:
        raise ValueError(
            f"No valid bridge elevation band: enamel_thickness={enamel_thickness_mm} "
            f"plate_overlap={plate_overlap_mm} margin={margin:.3f} leaves "
            f"[{safe_z_min:.3f}, {safe_z_max:.3f}]. Reduce junction_sphere_factor, "
            f"bridge_diameter, or increase enamel_thickness / plate_overlap."
        )

    n = max(1, int(n_bridge_layers))
    if n == 1:
        return [0.5 * (safe_z_min + safe_z_max)]
    return [safe_z_min + i * (safe_z_max - safe_z_min) / (n - 1) for i in range(n)]


def anchor_ring_rotation(
    ring_rotation: Dict[int, float],
    morphometrics: Dict[str, Any],
) -> Dict[int, float]:
    """Rotate the whole RING_ROTATION dict so ring 0 lines up with bands[0]."""
    bands = morphometrics.get("bands") or []
    if not bands:
        return ring_rotation
    anchor_deg = float(bands[0].get("mean_direction_deg", 0.0))
    return {i: float(v) + anchor_deg for i, v in ring_rotation.items()}


def map_center_spacing(
    morphometrics: Dict[str, Any],
    rod_diameter_mm: float,
    biology_scale_factor: float = DEFAULT_BIOLOGY_SCALE_FACTOR,
    sla_min_feature_mm: float = DEFAULT_SLA_MIN_FEATURE_MM,
    clamps: List[Dict] = None,
    max_spacing_ratio: float = 1.5,
) -> float:
    """CENTER_SPACING from mean band_width_um; clamped so rods neither overlap
    (floor = 1.05 x rod diameter) nor become so sparse that the bridges
    cross un-meshable distances (ceiling = `max_spacing_ratio` x rod diameter).
    """
    bands = morphometrics.get("bands") or []
    floor_from_rod = rod_diameter_mm * 1.05
    floor_from_sla = 2 * sla_min_feature_mm
    floor = max(floor_from_rod, floor_from_sla)
    ceiling = rod_diameter_mm * max_spacing_ratio

    if not bands:
        return max(floor, min(ceiling, rod_diameter_mm * 1.2))

    widths = [float(b.get("band_width_um", {}).get("mean", 0.0)) for b in bands]
    widths = [w for w in widths if w > 0]
    if not widths:
        return max(floor, min(ceiling, rod_diameter_mm * 1.2))

    mean_width_um = sum(widths) / len(widths)
    raw_spacing_mm = um_to_mm_scaled(mean_width_um, biology_scale_factor)

    spacing_mm = raw_spacing_mm
    if spacing_mm < floor:
        if clamps is not None:
            clamps.append(
                {
                    "field": "CENTER_SPACING",
                    "raw_value": float(raw_spacing_mm),
                    "clamped_value": float(floor),
                    "reason": f"below floor max(rod*1.05={floor_from_rod:.3f}, 2*sla_min={floor_from_sla:.3f})",
                }
            )
        spacing_mm = floor
    elif spacing_mm > ceiling:
        if clamps is not None:
            clamps.append(
                {
                    "field": "CENTER_SPACING",
                    "raw_value": float(raw_spacing_mm),
                    "clamped_value": float(ceiling),
                    "reason": f"above ceiling rod*{max_spacing_ratio} (prevents un-meshable bridge spans)",
                }
            )
        spacing_mm = ceiling
    return float(spacing_mm)
