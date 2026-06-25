"""Map SOM band morphometrics to bridge geometry and ring anchoring.

Uses:
  - periodicity.dominant_wavelength_um_mean  -> N_BRIDGE_LAYERS
  - bands[0].mean_direction_deg              -> anchor for RING_ROTATION[0]
  - band_width_um.mean                        -> CENTER_SPACING
"""

from __future__ import annotations

import os
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


# ------------------------------------------------------------------
# OCCT tangent-resonance jitter
# ------------------------------------------------------------------
# Empirical OpenCASCADE/CadQuery bug discovered on 2026-05-27/28 at the
# post-clearance-fix pipeline defaults (DEFAULT_BRIDGE_RATIO = 0.8,
# DEFAULT_PLATE_CLEARANCE_MM = 0.5). At N_BRIDGE_LAYERS = 8 with the
# uniformly-spaced default elevations
#   [1.851, 4.108, 6.365, 8.622, 10.878, 13.135, 15.392, 17.649] mm,
# OCCT's boolean-fusion algorithm hits a tangent geometry against the
# Z_SAMPLES=50 helical rod tessellation and SILENTLY drops every
# horizontal bridge from the output BREP — the STL is still watertight
# (rods + plates only) but 0 of 8 expected bridges fuse. The bug is
# tangent-position-specific to N=8's dz = 2.257 mm spacing; N=2/4/6/7/9
# at the same defaults all fuse cleanly.
#
# WHY THE JITTER WORKS (mechanism). A boolean UNION stitches two solids
# along the curve where their surfaces CROSS transversally. When two
# surfaces instead sit TANGENT (graze along a line) or COPLANAR, that
# intersection degenerates to a zero-width point/line/area: OCCT either
# emits a zero-area sliver face (-> a tiny gap = NON-WATERTIGHT STL) or
# fails outright and returns a Null TopoDS_Shape (-> hard crash, usually
# at the bridge-union or top-plate-union step). The "bad" bridge
# elevations are the Z-heights where a bridge surface grazes a neighbour
# -- an adjacent bridge layer, or (with fat rods) the top-plate region.
# The stock CAD fights the same enemy with its own epsilon offsets
# ("avoid coplanar faces" / "avoid tangent coplanarity", lattice_cad.py).
# A small uniform Z-shift slides every bridge off the grazing position
# into a clean crossing/clearance and OCCT fuses normally. This is a
# POSITION resonance, NOT crowding: jitter 0.50 mm gives SLIGHTLY TIGHTER
# layer spacing than 0.15 mm yet builds clean, and raising ENAMEL_THICKNESS
# (more vertical room) was hit-or-miss -- only the absolute Z-position of
# each bridge relative to its neighbours matters.
#
# THRESHOLD HISTORY.
#   2026-05-28: N=8 thin-rod silent-drop bracketed via CAD-only probes
#     (scripts/probe_n8_bridge_jitter.py): +0.05 mm broken, +0.075..+0.15
#     all 8/8 fused. 0.15 mm chosen (~2x margin over the 0.075 threshold).
#   2026-05-29: the canonical sweep was extended to N=4..9 x {biomimetic,
#     thick-rod (ROD_DIAMETER = CENTER_SPACING - 0.025 = 3.167 mm)}. Three
#     high-density cases FAILED at 0.15 mm -- N=9 biomimetic (non-watertight),
#     N=8 thick and N=9 thick (Null-shape crash). 0.30 mm was still
#     insufficient; 0.50 mm makes ALL THREE watertight (CAD-only probes
#     scripts/probe_n8_thick_offsets.py + scripts/probe_taller_lattice.py,
#     re-verified with a MERGED-VERTEX watertight check -- see methodology
#     note). Denser/fatter geometries pack more surfaces close, so more
#     bridges risk grazing and a larger nudge is needed. Bumped 0.15 ->
#     0.50 mm; still only ~3% of the ~15.8 mm bridge band, and the top-plate
#     clearance is preserved EXACTLY (see implementation note below).
#   2026-05-29 (cont.): NO single jitter is clean for ALL 12 trials -- the
#     resonance is PER-GEOMETRY. 0.50 mm fixes the dense three but BREAKS
#     N=7 thick (which is watertight at 0.15). The sweep therefore uses
#     MIXED jitter. Each trial's actual offsets are recorded in its
#     cad_params.json, so reproducibility holds.
#   2026-06-01: the resonance ALSO manifests downstream as a gmsh VOLUME-mesh
#     failure, not only as a CAD bridge-drop. With the override-inflation bug
#     fixed (bridges correctly 1.702 mm), thick N=5 and N=9 build watertight
#     with all bridges present yet gmsh returns 0 tetrahedra with every 3D
#     algorithm (HXT, Delaunay, Frontal) at mesh-size 0.5 mm: the bridge<->rod
#     junctions across the 0.025 mm rod-rod gap form near-degenerate slivers
#     no tet can fill. Because the failure is N-dependent (thick N=4,6,8 mesh
#     fine; bio N=9 with thinner rods meshes fine) it tracks BRIDGE POSITION,
#     so the same Z-jitter is the lever to test against it -- a different
#     jitter may move the junctions off the unmeshable slivers.
#
# PER-TRIAL OVERRIDE (no longer "edit the constant"). The EFFECTIVE jitter is
# resolved at call time from the OCCT_TANGENT_JITTER_MM *environment variable*
# (falling back to the module default below). The sweep driver exports
# OCCT_TANGENT_JITTER_MM=<value> before each trial's CAD subprocess; because
# env vars inherit into the cad_env child, that trial builds at its own
# tangent-escape value with no source edit. This RETIRES the old footgun
# where rebuilding a mid-N thick trial while the constant sat at 0.50 would
# silently break it -- set the env var per trial instead.
#
# METHODOLOGY NOTE (this cost a wrong turn on 2026-05-29). Watertightness
# MUST be tested on a VERTEX-MERGED mesh: trimesh.load(path) with the
# DEFAULT process=True, exactly as metrics.cad_integrity does.
# trimesh.load(path, process=False) keeps every triangle's 3 vertices
# separate (a "triangle soup") so NO edge is ever shared and EVERY mesh --
# even a perfect solid -- reads non-watertight. That false negative made
# the working 0.50 mm fix look like a failure during probing until the
# checks were redone with merged vertices.
#
# IMPLEMENTATION. To preserve the top-bridge plate-underside clearance the
# safe-zone calculation builds in, the jitter REDUCES safe_z_max by the
# jitter before computing uniformly-spaced elevations, then ADDS the jitter
# back to every elevation; the top elevation lands exactly at the original
# safe_z_max so plate clearance stays at DEFAULT_PLATE_CLEARANCE_MM.
#
# Companion runtime check: every CAD run is sliced at each emitted
# BRIDGE_Z_OFFSETS value (cross-section area vs a baseline,
# metrics.cad_integrity.verify_bridge_presence) AND its STL watertightness
# is asserted; the pipeline aborts before meshing if either fails, so no
# FEA compute is wasted on broken geometry. scripts/probe_n8_bridge_jitter.py
# and scripts/probe_n8_thick_offsets.py remain as CLI diagnostics.
# See memory project_occt_bridge_tangent_volume_loss.md for the bug story.
# ------------------------------------------------------------------
OCCT_TANGENT_JITTER_MM: float = 0.50


def _effective_jitter_mm() -> float:
    """Resolve the tangent-escape jitter (mm) to use for THIS build.

    Precedence: the OCCT_TANGENT_JITTER_MM environment variable (per-trial
    override exported by the sweep driver and inherited by the cad_env
    subprocess) wins; otherwise the module default OCCT_TANGENT_JITTER_MM.
    A missing/blank/malformed env value falls back to the default rather
    than aborting the CAD stage -- a typo in a driver export must not crash
    a multi-hour sweep, it should just build at the documented default.
    """
    raw = os.environ.get("OCCT_TANGENT_JITTER_MM")
    if raw is None or raw.strip() == "":
        return OCCT_TANGENT_JITTER_MM
    try:
        return float(raw)
    except ValueError:
        return OCCT_TANGENT_JITTER_MM


def compute_safe_bridge_elevations(
    n_bridge_layers: int,
    rod_diameter_mm: float,
    bridge_diameter_mm: float,
    enamel_thickness_mm: float,
    plate_overlap_mm: float,
    junction_sphere_factor: float,
    clearance_mm: float = DEFAULT_PLATE_CLEARANCE_MM,
) -> List[float]:
    """Place bridge elevations so that:
        (a) the bridge cylinder does not intersect the top or bottom plate,
        (b) the junction sphere (diameter = junction_sphere_factor * rod_diameter)
            at each bridge-rod junction also stays clear of the plates,
        (c) no bridge centerline lands at a default uniformly-spaced position
            (which triggers the OCCT silent-drop bug at N=8 — see the
            OCCT_TANGENT_JITTER_MM comment block above).

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

    Implementation note on the OCCT-escape jitter:
        - We reserve OCCT_TANGENT_JITTER_MM of headroom at the TOP of the
          safe band by computing safe_z_max_adj = safe_z_max - jitter,
        - place N uniformly-spaced elevations in [safe_z_min, safe_z_max_adj],
        - and then shift every elevation up by `jitter`.
        The result is that elev[0] = safe_z_min + jitter and
        elev[-1] = safe_z_max (the original top, unmodified) — preserving
        the same top-plate clearance as before the patch while breaking
        the OCCT-tangent regular-spacing resonance.
    """
    # Resolve the effective tangent-escape jitter ONCE (env override or module
    # default) and use the local everywhere below, so a single build is
    # internally consistent even if the env changes mid-process.
    jitter = _effective_jitter_mm()

    bridge_half = 0.5 * float(bridge_diameter_mm)
    sphere_radius = 0.5 * float(junction_sphere_factor) * float(rod_diameter_mm)
    margin = max(bridge_half, sphere_radius) + float(clearance_mm)

    cut_top_z = float(enamel_thickness_mm) - float(plate_overlap_mm)
    safe_z_min = float(plate_overlap_mm) + margin
    safe_z_max = cut_top_z - float(plate_overlap_mm) - margin

    # Reserve OCCT-jitter headroom at the top before checking band feasibility.
    safe_z_max_adj = safe_z_max - jitter
    if safe_z_max_adj <= safe_z_min:
        raise ValueError(
            f"No valid bridge elevation band after reserving OCCT-jitter headroom: "
            f"enamel_thickness={enamel_thickness_mm} plate_overlap={plate_overlap_mm} "
            f"margin={margin:.3f} OCCT_jitter={jitter} leaves "
            f"[{safe_z_min:.3f}, {safe_z_max_adj:.3f}]. Reduce junction_sphere_factor, "
            f"bridge_diameter, or increase enamel_thickness / plate_overlap."
        )

    n = max(1, int(n_bridge_layers))
    if n == 1:
        # Single bridge: place at the center of the (shrunk) band, then shift up.
        return [0.5 * (safe_z_min + safe_z_max_adj) + jitter]
    # Uniform spacing within the SHRUNK band, then uniform Z-shift on every
    # elevation. Equivalent to: bottom moves up by JITTER, top stays at the
    # unmodified safe_z_max, with linear interpolation between them.
    elevs_uniform = [safe_z_min + i * (safe_z_max_adj - safe_z_min) / (n - 1) for i in range(n)]
    return [z + jitter for z in elevs_uniform]


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
