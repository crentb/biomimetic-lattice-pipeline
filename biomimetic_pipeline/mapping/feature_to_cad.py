"""Top-level feature → CAD-parameter orchestrator.

Composes twist_mappers, bridge_mappers, rod_mappers, and scale into a single
cad_parameters.json that honors the cad_parameters.schema.json contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from biomimetic_pipeline.mapping import bridge_mappers, rod_mappers, twist_mappers
from biomimetic_pipeline.mapping.scale import (
    DEFAULT_BIOLOGY_SCALE_FACTOR,
    DEFAULT_BRIDGE_RATIO,
    DEFAULT_PLATE_CLEARANCE_MM,
    DEFAULT_SLA_MIN_FEATURE_MM,
)
from biomimetic_pipeline.orchestration.run_context import PIPELINE_VERSION, now_iso

LATTICE_CAD_STOCK_DEFAULTS = {
    "ADD_HORIZONTAL_BRIDGES": True,
    "CONTINUOUS_TWIST": True,
    "CUT_FLAT": True,
    "ENAMEL_THICKNESS": 20.0,
    "PLATE_OVERHANG": 2.0,
    "PLATE_THICKNESS": 1.2,
    "PLATE_OVERLAP": 0.5,
    "Z_SAMPLES": 50,
    "FILLET_RADIUS": 0.0,
    "CHAMFER_SIZE": 0.0,
    "JUNCTION_SPHERE_FACTOR": 0.8,
    "BRIDGE_DIAMETER": 1.0,
    "STL_TOLERANCE": 0.01,
    "STL_ANGULAR_TOLERANCE": 0.1,
}


def map_morphometrics(
    morphometrics: Dict[str, Any],
    *,
    morphometrics_source: Optional[Path] = None,
    biology_scale_factor: float = DEFAULT_BIOLOGY_SCALE_FACTOR,
    sla_min_feature_mm: float = DEFAULT_SLA_MIN_FEATURE_MM,
    enamel_thickness_mm: Optional[float] = None,
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    clamps: List[Dict[str, Any]] = []

    cad: Dict[str, Any] = dict(LATTICE_CAD_STOCK_DEFAULTS)
    cad["biology_scale_factor"] = biology_scale_factor
    cad["sla_min_feature_mm"] = sla_min_feature_mm

    if enamel_thickness_mm is not None:
        cad["ENAMEL_THICKNESS"] = float(enamel_thickness_mm)

    # Rod geometry
    rod_frag = rod_mappers.map_rod(
        morphometrics,
        biology_scale_factor=biology_scale_factor,
        sla_min_feature_mm=sla_min_feature_mm,
        clamps=clamps,
    )
    cad.update(rod_frag)

    # Bridge geometry
    bridge_frag = bridge_mappers.map_bridges(
        morphometrics,
        enamel_thickness_mm=cad["ENAMEL_THICKNESS"],
        biology_scale_factor=biology_scale_factor,
    )
    cad.update(bridge_frag)

    # Center spacing (depends on rod diameter already decided)
    cad["CENTER_SPACING"] = bridge_mappers.map_center_spacing(
        morphometrics,
        rod_diameter_mm=cad["ROD_DIAMETER"],
        biology_scale_factor=biology_scale_factor,
        sla_min_feature_mm=sla_min_feature_mm,
        clamps=clamps,
    )

    # Twist: needs final N_RINGS
    twist_frag = twist_mappers.map_twist(
        morphometrics,
        n_rings=cad["N_RINGS"],
        enamel_thickness_mm=cad["ENAMEL_THICKNESS"],
    )
    cad.update(twist_frag)

    # Anchor ring rotations to band[0] direction
    cad["RING_ROTATION"] = bridge_mappers.anchor_ring_rotation(cad["RING_ROTATION"], morphometrics)
    # Serialise int keys to strings for JSON round-trip safety
    cad["RING_ROTATION"] = {str(k): float(v) for k, v in cad["RING_ROTATION"].items()}

    if "per_ring_diameter" in cad and cad["per_ring_diameter"] is not None:
        cad["per_ring_diameter"] = {str(k): float(v) for k, v in cad["per_ring_diameter"].items()}

    # --- Apply caller overrides, REMEMBERING which keys were explicitly set --
    # CRITICAL (2026-05-31 value-flow audit): the BRIDGE_DIAMETER and
    # CENTER_SPACING derivations below run AFTER this merge and used to OVERWRITE
    # caller values unconditionally. That silently inflated thick-variant bridges
    # (override ROD=3.167 -> BRIDGE re-derived 0.80*3.167 = 2.53 instead of the
    # intended 1.702) and bumped CENTER_SPACING to 1.05*ROD. We now record the
    # overridden key set and HONOR explicit BRIDGE_DIAMETER / CENTER_SPACING
    # overrides instead of re-deriving over them.
    overridden = set(extra_overrides) if extra_overrides else set()
    if extra_overrides:
        cad.update(extra_overrides)

    # Coupling-footgun guard: overriding ROD alone would silently re-derive
    # BRIDGE (=bridge_ratio*ROD) and CENTER_SPACING (1.05*ROD floor) from the new
    # ROD. For the thick-rod variant the intent is to GROW ROD while KEEPING the
    # biomimetic BRIDGE/CS, so a lone ROD override almost always means the caller
    # forgot to pin them. Fail loudly rather than emit unintended geometry.
    # (Thick variants should be built via scripts/make_thick_rod_variant.py, which
    # sets ROD on the param dict directly and never routes through this derivation.)
    if "ROD_DIAMETER" in overridden and "BRIDGE_DIAMETER" not in overridden:
        raise ValueError(
            "ROD_DIAMETER overridden without BRIDGE_DIAMETER: the derivation would "
            "silently re-scale BRIDGE_DIAMETER = bridge_ratio x ROD_DIAMETER (this "
            "shipped inflated 2.47 mm thick bridges). Co-specify BRIDGE_DIAMETER "
            "(and usually CENTER_SPACING) in extra_overrides, or build thick "
            "variants via scripts/make_thick_rod_variant.py."
        )

    rod_mm = float(cad["ROD_DIAMETER"])
    # Bridge-to-rod thickness ratio used by the DERIVED path (BRIDGE = ratio*ROD).
    # Overridable for parametric sweeps; default is the documented design constant.
    bridge_ratio = float(cad.get("bridge_ratio", DEFAULT_BRIDGE_RATIO))

    # --- BRIDGE_DIAMETER: honor an explicit override; otherwise derive -------
    if "BRIDGE_DIAMETER" in overridden:
        # Caller pinned BRIDGE explicitly (e.g. thick variant keeps biomimetic
        # 1.702 mm). Keep it verbatim; only enforce the lattice_cad invariant.
        bridge_val = float(cad["BRIDGE_DIAMETER"])
        if bridge_val >= rod_mm:
            raise ValueError(
                f"Overridden BRIDGE_DIAMETER ({bridge_val}) must be < ROD_DIAMETER "
                f"({rod_mm}) per the lattice_cad invariant."
            )
        cad["BRIDGE_DIAMETER"] = bridge_val
        clamps.append(
            {
                "field": "BRIDGE_DIAMETER",
                "raw_value": bridge_val,
                "clamped_value": bridge_val,
                "reason": "explicit caller override honored (NOT re-derived from ROD)",
            }
        )
    else:
        # Derived path: BRIDGE = max(SLA min, bridge_ratio*ROD). If the SLA floor
        # would push BRIDGE >= ROD, bump ROD so a sub-rod bridge still fits.
        target_bridge = max(sla_min_feature_mm, bridge_ratio * rod_mm)
        if target_bridge >= rod_mm:
            new_rod = max(rod_mm, 2.0 * sla_min_feature_mm + 0.05)
            clamps.append(
                {
                    "field": "ROD_DIAMETER",
                    "raw_value": rod_mm,
                    "clamped_value": new_rod,
                    "reason": "bumped so BRIDGE_DIAMETER can stay strictly below ROD per lattice_cad invariant",
                }
            )
            cad["ROD_DIAMETER"] = new_rod
            rod_mm = new_rod
            target_bridge = bridge_ratio * rod_mm
        prev_bridge = float(cad.get("BRIDGE_DIAMETER", 1.0))
        if abs(prev_bridge - target_bridge) > 1e-6:
            # Record which term of the max() actually won (audit low-pri fix:
            # the old reason hardcoded "clamped to SLA min" even when it didn't).
            won = (
                "SLA min floor"
                if sla_min_feature_mm >= bridge_ratio * rod_mm
                else f"{bridge_ratio:.4f} x ROD_DIAMETER"
            )
            clamps.append(
                {
                    "field": "BRIDGE_DIAMETER",
                    "raw_value": prev_bridge,
                    "clamped_value": float(target_bridge),
                    "reason": f"derived = max(SLA_min={sla_min_feature_mm}, {bridge_ratio:.4f}*ROD); {won} operative",
                }
            )
        cad["BRIDGE_DIAMETER"] = float(target_bridge)

    # --- CENTER_SPACING: honor an explicit override; else enforce > 1.05*ROD -
    if "CENTER_SPACING" in overridden:
        cs_val = float(cad["CENTER_SPACING"])
        if cs_val <= rod_mm:
            raise ValueError(
                f"Overridden CENTER_SPACING ({cs_val}) must be > ROD_DIAMETER ({rod_mm}) "
                "so rods cannot overlap."
            )
        cad["CENTER_SPACING"] = cs_val
        clamps.append(
            {
                "field": "CENTER_SPACING",
                "raw_value": cs_val,
                "clamped_value": cs_val,
                "reason": "explicit caller override honored (NOT bumped to 1.05 x ROD)",
            }
        )
    else:
        # rods can't overlap; keep spacing strictly above ROD (re-check after any
        # ROD bump above).
        min_spacing = rod_mm * 1.05
        if cad["CENTER_SPACING"] < min_spacing:
            clamps.append(
                {
                    "field": "CENTER_SPACING",
                    "raw_value": float(cad["CENTER_SPACING"]),
                    "clamped_value": float(min_spacing),
                    "reason": "bumped to keep spacing > 1.05 * ROD_DIAMETER after ROD bump",
                }
            )
            cad["CENTER_SPACING"] = float(min_spacing)

    # Junction sphere sanity: sphere_radius = 0.5 * factor * ROD_DIAMETER must
    # leave room between bridges and plates. Stock lattice_cad's auto bridge-
    # placement only accounts for bridge_half, not the sphere radius — so a
    # large JUNCTION_SPHERE_FACTOR + tight plate_overlap can leave spheres
    # intersecting plates. We cap the factor so spheres fit inside the safe
    # zone with both plates and bridges, and then we emit BRIDGE_Z_OFFSETS
    # ourselves using the sphere-aware safe zone.
    plate_overlap = float(cad.get("PLATE_OVERLAP", 0.5))
    enamel = float(cad["ENAMEL_THICKNESS"])
    bridge_mm = float(cad["BRIDGE_DIAMETER"])
    jsf = float(cad.get("JUNCTION_SPHERE_FACTOR", 0.0))
    if jsf > 0:
        # Sphere must fit within plate_overlap margin; otherwise reduce factor.
        # margin available = plate_overlap - bridge_half - clearance
        clearance = 0.05
        max_sphere_radius = plate_overlap - 0.5 * bridge_mm - clearance
        # Additionally require a tiny positive margin so OCC booleans stay clean.
        if max_sphere_radius <= 0.02:
            # Can't fit any sphere -- disable the feature.
            new_factor = 0.0
            clamps.append(
                {
                    "field": "JUNCTION_SPHERE_FACTOR",
                    "raw_value": jsf,
                    "clamped_value": 0.0,
                    "reason": (
                        f"plate_overlap ({plate_overlap}) - bridge_half ({0.5*bridge_mm:.3f}) "
                        f"- clearance ({clearance}) <= 0; no room for junction sphere, disabling."
                    ),
                }
            )
            cad["JUNCTION_SPHERE_FACTOR"] = 0.0
            jsf = 0.0
        else:
            max_factor = 2.0 * max_sphere_radius / rod_mm
            if jsf > max_factor:
                clamps.append(
                    {
                        "field": "JUNCTION_SPHERE_FACTOR",
                        "raw_value": jsf,
                        "clamped_value": float(max_factor),
                        "reason": (
                            f"reduced so junction sphere radius ({jsf*rod_mm/2:.3f} mm) fits "
                            f"inside plate_overlap ({plate_overlap}) - bridge_half "
                            f"({0.5*bridge_mm:.3f}) - clearance ({clearance})"
                        ),
                    }
                )
                cad["JUNCTION_SPHERE_FACTOR"] = float(max_factor)
                jsf = float(max_factor)

    # Emit explicit BRIDGE_Z_OFFSETS using a sphere-aware safe zone so bridges
    # (and their junction spheres) never intersect the top or bottom plates.
    from biomimetic_pipeline.mapping.bridge_mappers import compute_safe_bridge_elevations

    try:
        bridge_elevs = compute_safe_bridge_elevations(
            n_bridge_layers=int(cad["N_BRIDGE_LAYERS"]),
            rod_diameter_mm=rod_mm,
            bridge_diameter_mm=bridge_mm,
            enamel_thickness_mm=enamel,
            plate_overlap_mm=plate_overlap,
            junction_sphere_factor=jsf,
            # Use the project-wide FDM-safe clearance so the top bridge
            # never fuses with the plate underside on print. Historical
            # value here was 0.05 mm, sized for SLA, and produced a
            # functionally-fused top bridge on every sweep_layers trial.
            clearance_mm=DEFAULT_PLATE_CLEARANCE_MM,
        )
        cad["BRIDGE_Z_OFFSETS"] = [float(z) for z in bridge_elevs]
    except ValueError as exc:
        # Safe zone collapsed; fall back to stock auto-placement (None) and log.
        clamps.append(
            {
                "field": "BRIDGE_Z_OFFSETS",
                "raw_value": None,
                "clamped_value": None,
                "reason": f"safe-zone computation failed, delegating to stock auto: {exc}",
            }
        )
        cad["BRIDGE_Z_OFFSETS"] = None

    cad["provenance"] = _build_provenance(morphometrics, morphometrics_source)
    cad["provenance"]["scale_clamps"] = clamps

    return cad


def _build_provenance(
    morphometrics: Dict[str, Any], morphometrics_source: Optional[Path]
) -> Dict[str, Any]:
    p: Dict[str, Any] = {
        "mapper_version": PIPELINE_VERSION,
        "generated_timestamp_iso": now_iso(),
        "morphometrics_specimen_id": morphometrics.get("specimen_id", ""),
    }
    if morphometrics_source is not None:
        src = Path(morphometrics_source)
        p["morphometrics_source"] = str(src)
        if src.exists():
            h = hashlib.sha256()
            with open(src, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            p["morphometrics_sha256"] = h.hexdigest()
    return p


def save(cad_params: Dict[str, Any], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cad_params, indent=2))
    return out_path


def validate(cad_params: Dict[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError:
        return
    schema_path = Path(__file__).resolve().parent.parent / "schemas" / "cad_parameters.schema.json"
    schema = json.loads(schema_path.read_text())
    jsonschema.validate(cad_params, schema)
