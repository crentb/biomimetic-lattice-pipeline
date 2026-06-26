# Copyright 2026 Cameron B. Renteria
# SPDX-License-Identifier: Apache-2.0
"""
lattice_cad.py — Standalone CadQuery CAD generation for decussated enamel lattices.

Refactored from Cell 2 of helicaltwist_continous_loadingplates.ipynb.

Generates a parametric decussated enamel rod lattice with:
  - Concentric rings of helically twisted rods
  - Horizontal inter-rod bridges at configurable z-elevations
  - Top and bottom loading plates fused into a single solid
  - Optional fillets at bridge-rod junctions

Usage as a module:
    from lattice_cad import generate_lattice
    result = generate_lattice({"FILLET_RADIUS": 0.1})

Usage from the command line:
    python lattice_cad.py --fillet-radius 0.1 --export-dir my_exports
"""

import cadquery as cq
import json
import math
import numpy as np
import time
import os
import sys
import argparse
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Default parameters (matching notebook Cell 2)
# ---------------------------------------------------------------------------
DEFAULTS: Dict[str, object] = {
    # Toggles
    "ADD_HORIZONTAL_BRIDGES": True,
    "CONTINUOUS_TWIST": True,
    "CUT_FLAT": True,

    # Rod geometry (4x biological scale)
    "ROD_DIAMETER": 2.0,
    "CENTER_SPACING": 2.4,
    "ENAMEL_THICKNESS": 20.0,
    "N_RINGS": 5,

    # Bridge geometry
    "BRIDGE_DIAMETER": 1.0,
    "N_BRIDGE_LAYERS": 4,
    "BRIDGE_Z_OFFSETS": None,  # Custom z-positions; None = uniform spacing

    # Loading plates
    "PLATE_OVERHANG": 2.0,
    "PLATE_THICKNESS": 1.2,
    "PLATE_OVERLAP": 0.5,

    # Twist
    "RING_ROTATION": {0: 0.0, 1: 10.0, 2: -20.0, 3: 30.0, 4: -45.0, 5: 60.0},
    "Z_SAMPLES": 50,
    "TWIST_TYPE": "linear",

    # Junction smoothing
    "FILLET_RADIUS": 0.0,  # mm, 0 = no fillet
    "CHAMFER_SIZE": 0.0,   # mm, 0 = no chamfer (simpler than fillet, meshes better)
    "JUNCTION_SPHERE_FACTOR": 0.0,  # Sphere diameter = factor * rod_diameter at bridge-rod junctions (0 = off)
    "ROD_TAPER_FACTOR":       0.0,  # Flare rod ends at plates: 0 = none, 0.3 = 30% larger diameter at plate

    # Export
    "EXPORT_DIR": "enamel_exports",
    "STL_TOLERANCE": 0.01,
    "STL_ANGULAR_TOLERANCE": 0.1,
}


# ---------------------------------------------------------------------------
# Twist functions
# ---------------------------------------------------------------------------

def linear_twist(z: float, H: float, total_rotation_deg: float) -> float:
    return (z / H) * math.radians(total_rotation_deg)


def accelerating_twist(z: float, H: float, total_rotation_deg: float) -> float:
    return ((z / H) ** 2) * math.radians(total_rotation_deg)


def sigmoid_twist(z: float, H: float, total_rotation_deg: float) -> float:
    k = 6
    z_norm = z / H
    sigmoid_val = 1 / (1 + math.exp(-k * (z_norm - 0.5)))
    sigmoid_normalized = (sigmoid_val - 1/(1+math.exp(k*0.5))) / \
                         (1/(1+math.exp(-k*0.5)) - 1/(1+math.exp(k*0.5)))
    return sigmoid_normalized * math.radians(total_rotation_deg)


def get_twist_function(twist_type: str):
    functions = {
        "linear": linear_twist,
        "accelerating": accelerating_twist,
        "sigmoid": sigmoid_twist,
    }
    return functions.get(twist_type, linear_twist)


# ---------------------------------------------------------------------------
# Radial positions
# ---------------------------------------------------------------------------

def generate_radial_positions(
    n_rings: int, spacing: float
) -> List[Tuple[float, float, int, float]]:
    """Generate (x, y, ring_index, theta) for all rod base positions."""
    positions = [(0.0, 0.0, 0, 0.0)]
    for k in range(1, n_rings + 1):
        radius = k * spacing
        n_rods = 6 * k
        for i in range(n_rods):
            theta = 2 * math.pi * i / n_rods
            positions.append((
                radius * math.cos(theta),
                radius * math.sin(theta),
                k,
                theta,
            ))
    return positions


# ---------------------------------------------------------------------------
# Rod creation
# ---------------------------------------------------------------------------

def create_continuously_twisted_rod(
    x0: float, y0: float, ring_index: int, theta0: float,
    rotation_deg: float, z_samples: int, rod_diameter: float,
    enamel_thickness: float, twist_type: str,
) -> Tuple[cq.Workplane, Tuple[float, float]]:
    r = math.sqrt(x0**2 + y0**2)
    if r < 1e-6:
        rod = cq.Workplane("XY").moveTo(0, 0).circle(rod_diameter / 2).extrude(enamel_thickness)
        return rod, (0.0, 0.0)

    twist_func = get_twist_function(twist_type)
    path_points = []
    z_values = np.linspace(0, enamel_thickness, z_samples)
    for z in z_values:
        twist_angle = twist_func(z, enamel_thickness, rotation_deg)
        theta_z = theta0 + twist_angle
        x_z = r * math.cos(theta_z)
        y_z = r * math.sin(theta_z)
        path_points.append((x_z, y_z, z))

    x_top, y_top, _ = path_points[-1]

    if len(path_points) >= 3:
        path = cq.Workplane("XY").spline(path_points)
    else:
        path = cq.Workplane("XY").polyline(path_points)

    profile = cq.Workplane("XY").workplane(offset=0).moveTo(x0, y0).circle(rod_diameter / 2)
    try:
        rod = profile.sweep(path, multisection=False)
    except Exception as e:
        print(f"Warning: Sweep failed for ring {ring_index}, using simple extrusion. Error: {e}")
        path = cq.Workplane("XY").polyline([(x0, y0, 0), (x_top, y_top, enamel_thickness)])
        rod = profile.sweep(path, multisection=False)

    return rod, (x_top, y_top)


def create_simple_rotated_rod(
    x0: float, y0: float, ring_index: int, theta0: float,
    rotation_deg: float, rod_diameter: float, enamel_thickness: float,
) -> Tuple[cq.Workplane, Tuple[float, float]]:
    rotation_rad = math.radians(rotation_deg)
    r = math.sqrt(x0**2 + y0**2)
    theta1 = theta0 + rotation_rad
    x1 = r * math.cos(theta1)
    y1 = r * math.sin(theta1)
    path = cq.Workplane("XY").polyline([(x0, y0, 0), (x1, y1, enamel_thickness)])
    profile = cq.Workplane("XY").moveTo(x0, y0).circle(rod_diameter / 2)
    rod = profile.sweep(path, multisection=False)
    return rod, (x1, y1)


# ---------------------------------------------------------------------------
# Rod end taper (RPJ stress relief)
# ---------------------------------------------------------------------------

def _add_rod_taper(
    rod_solid: cq.Workplane,
    x_bot: float, y_bot: float,
    rod_diameter: float,
    plate_overlap: float,
    cut_top_z: float,
    enamel_thickness: float,
    taper_factor: float,
    twist_func,
    theta0: float,
    r: float,
    rotation_deg: float,
) -> cq.Workplane:
    """Union conical frustums at both rod ends to reduce RPJ stress concentration.

    The rod diameter flares from ``rod_diameter`` at the start of the plate-overlap
    zone to ``rod_diameter * (1 + taper_factor)`` at the plate face.  This mirrors
    the biological enamel-dentin junction where rods widen as they anchor into the
    dentin substrate.

    Parameters
    ----------
    rod_solid : cq.Workplane
        Existing rod solid to augment.
    x_bot, y_bot : float
        Rod centre at z = 0 (bottom plate face).
    rod_diameter : float
        Base rod diameter (mm) at mid-span.
    plate_overlap : float
        Depth the rod embeds into each plate (mm) — defines taper length.
    cut_top_z : float
        Z-height where rods are trimmed (= enamel_thickness - plate_overlap).
    enamel_thickness : float
        Full rod height before trimming (mm).
    taper_factor : float
        Fractional increase in radius at the plate face (e.g. 0.3 = 30% larger).
    twist_func : callable
        Twist function ``f(z, H, rotation_deg) -> angle_radians``.
    theta0 : float
        Initial angular position of the rod (radians).
    r : float
        Radial distance of the rod from the specimen axis (mm).
    rotation_deg : float
        Total twist rotation for this rod's ring (degrees).

    Returns
    -------
    cq.Workplane
        Rod solid with frustums unioned at both ends.
    """
    base_r   = rod_diameter / 2.0
    flared_r = base_r * (1.0 + taper_factor)

    # --- Bottom frustum: z=0 (flared) → z=plate_overlap (base) ---
    # The angular twist over plate_overlap (~0.5 mm) is negligible, so we keep
    # the frustum centred at (x_bot, y_bot) for both profiles.
    try:
        bottom_frustum = (
            cq.Workplane("XY")
            .workplane(offset=0)
            .moveTo(x_bot, y_bot)
            .circle(flared_r)
            .workplane(offset=plate_overlap)
            .moveTo(x_bot, y_bot)
            .circle(base_r)
            .loft()
        )
        rod_solid = rod_solid.union(bottom_frustum)
    except Exception as e:
        print(f"    Warning: bottom taper failed ({e}), skipping.")

    # --- Top frustum: z=(cut_top_z - plate_overlap) (base) → z=cut_top_z (flared) ---
    z_taper_start = cut_top_z - plate_overlap
    try:
        ang_start = twist_func(z_taper_start, enamel_thickness, rotation_deg)
        x_ts = r * math.cos(theta0 + ang_start)
        y_ts = r * math.sin(theta0 + ang_start)

        ang_top = twist_func(cut_top_z, enamel_thickness, rotation_deg)
        x_tc = r * math.cos(theta0 + ang_top)
        y_tc = r * math.sin(theta0 + ang_top)

        top_frustum = (
            cq.Workplane("XY")
            .workplane(offset=z_taper_start)   # z = cut_top_z - plate_overlap
            .moveTo(x_ts, y_ts)
            .circle(base_r)
            .workplane(offset=plate_overlap)   # relative offset → z = cut_top_z
            .moveTo(x_tc, y_tc)
            .circle(flared_r)
            .loft()
        )
        rod_solid = rod_solid.union(top_frustum)
    except Exception as e:
        print(f"    Warning: top taper failed ({e}), skipping.")

    return rod_solid


# ---------------------------------------------------------------------------
# Rod position at height (for bridge placement)
# ---------------------------------------------------------------------------

def get_rod_position_at_height(
    x_base: float, y_base: float, x_top: float, y_top: float,
    z: float, H: float, ring_index: int,
    continuous_twist: bool, ring_rotation: dict, twist_type: str,
) -> Tuple[float, float]:
    if not continuous_twist:
        t = z / H
        return x_base + t * (x_top - x_base), y_base + t * (y_top - y_base)

    r = math.sqrt(x_base**2 + y_base**2)
    if r < 1e-6:
        return 0.0, 0.0

    theta_base = math.atan2(y_base, x_base)
    rotation_deg = ring_rotation.get(ring_index, 0)
    twist_func = get_twist_function(twist_type)
    twist_angle = twist_func(z, H, rotation_deg)
    theta_z = theta_base + twist_angle
    return r * math.cos(theta_z), r * math.sin(theta_z)


# ---------------------------------------------------------------------------
# Bridge creation
# ---------------------------------------------------------------------------

def make_cylinder_bridge(
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    diameter: float,
) -> cq.Workplane:
    v0 = cq.Vector(*p0)
    v1 = cq.Vector(*p1)
    direction = (v1 - v0).normalized()
    if abs(direction.z) < 0.9:
        ref = cq.Vector(0, 0, 1)
    else:
        ref = cq.Vector(1, 0, 0)
    x_axis = direction.cross(ref).normalized()
    plane = cq.Plane(origin=v0, xDir=x_axis, normal=direction)
    profile = cq.Workplane(plane).circle(diameter / 2)
    edge = cq.Edge.makeLine(v0, v1)
    return profile.sweep(edge, isFrenet=True)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_lattice(params: Optional[Dict] = None) -> Dict:
    """Generate the full decussated enamel lattice with loading plates.

    Parameters
    ----------
    params : dict, optional
        Override any default parameter. See DEFAULTS for all keys.

    Returns
    -------
    dict
        {"step_path": str, "stl_path": str,
         "model_z_min": float, "model_z_max": float,
         "n_rods": int, "n_bridges": int, "n_solids": int,
         "bridge_elevations": list}
    """
    cfg = dict(DEFAULTS)
    if params:
        for key, value in params.items():
            if key in cfg:
                cfg[key] = value
            else:
                print(f"WARNING: Unknown parameter '{key}' -- ignored")

    # Unpack parameters
    rod_diameter = cfg["ROD_DIAMETER"]
    center_spacing = cfg["CENTER_SPACING"]
    enamel_thickness = cfg["ENAMEL_THICKNESS"]
    n_rings = cfg["N_RINGS"]
    bridge_diameter = cfg["BRIDGE_DIAMETER"]
    n_bridge_layers = cfg["N_BRIDGE_LAYERS"]
    bridge_z_offsets = cfg["BRIDGE_Z_OFFSETS"]
    plate_overhang = cfg["PLATE_OVERHANG"]
    plate_thickness = cfg["PLATE_THICKNESS"]
    plate_overlap = cfg["PLATE_OVERLAP"]
    ring_rotation = cfg["RING_ROTATION"]
    z_samples = cfg["Z_SAMPLES"]
    twist_type = cfg["TWIST_TYPE"]
    continuous_twist = cfg["CONTINUOUS_TWIST"]
    add_bridges = cfg["ADD_HORIZONTAL_BRIDGES"]
    fillet_radius = cfg["FILLET_RADIUS"]
    chamfer_size = cfg["CHAMFER_SIZE"]
    junction_sphere_factor = cfg["JUNCTION_SPHERE_FACTOR"]
    rod_taper_factor       = cfg["ROD_TAPER_FACTOR"]
    export_dir = cfg["EXPORT_DIR"]

    # --- Parameter validation ---
    if not (0.0 <= rod_taper_factor <= 1.0):
        raise ValueError(
            f"ROD_TAPER_FACTOR must be in [0, 1.0], got {rod_taper_factor}. "
            f"Use 0.0 for no taper, 0.3 for 30% larger diameter at the plate face."
        )
    if bridge_diameter >= rod_diameter:
        raise ValueError(
            f"BRIDGE_DIAMETER ({bridge_diameter} mm) must be strictly less than "
            f"ROD_DIAMETER ({rod_diameter} mm). The bridge cannot be wider than "
            f"the rod it connects to — this produces degenerate OCC boolean geometry."
        )
    if junction_sphere_factor > 0:
        sphere_diam = junction_sphere_factor * rod_diameter
        if bridge_diameter > sphere_diam:
            raise ValueError(
                f"BRIDGE_DIAMETER ({bridge_diameter} mm) exceeds junction sphere diameter "
                f"({sphere_diam:.3f} mm = JUNCTION_SPHERE_FACTOR {junction_sphere_factor} × "
                f"ROD_DIAMETER {rod_diameter} mm). The sphere must be at least as wide as the "
                f"bridge to cover the junction. Increase JUNCTION_SPHERE_FACTOR or reduce "
                f"BRIDGE_DIAMETER."
            )

    print("\n" + "=" * 60)
    print("GENERATING RADIAL ENAMEL LATTICE (4x SCALE)")
    print("=" * 60)
    print(f"Twist mode: {'CONTINUOUS HELICAL' if continuous_twist else 'SIMPLE ROTATION'}")
    print(f"Twist type: {twist_type}")
    print(f"Rod diameter: {rod_diameter} mm")
    print(f"Center spacing: {center_spacing} mm")
    print(f"Height: {enamel_thickness} mm")
    print(f"Fillet radius: {fillet_radius} mm")
    print("=" * 60 + "\n")

    # --- Generate rods ---
    rod_positions = generate_radial_positions(n_rings, center_spacing)
    rods = cq.Workplane("XY")
    rod_meta = []

    if rod_taper_factor > 0:
        print(f"Rod taper enabled: {rod_taper_factor:.2f}x "
              f"(flared diameter = {rod_diameter * (1 + rod_taper_factor):.2f} mm at plates)")

    print("Creating rods...")
    for idx, (x0, y0, ring, theta0) in enumerate(rod_positions):
        rotation = ring_rotation.get(ring, 0)
        if continuous_twist:
            rod, (x1, y1) = create_continuously_twisted_rod(
                x0, y0, ring, theta0, rotation, z_samples,
                rod_diameter, enamel_thickness, twist_type,
            )
        else:
            rod, (x1, y1) = create_simple_rotated_rod(
                x0, y0, ring, theta0, rotation, rod_diameter, enamel_thickness,
            )

        # Apply end taper (RPJ stress relief) if requested
        if rod_taper_factor > 0:
            r_rad = math.sqrt(x0**2 + y0**2)
            if r_rad > 1e-6:   # skip the centre rod (r=0, no angular twist)
                tf = get_twist_function(twist_type)
                # cut_top_z not computed yet — use enamel_thickness - plate_overlap
                _cut_top = enamel_thickness - plate_overlap
                rod = _add_rod_taper(
                    rod, x0, y0,
                    rod_diameter, plate_overlap, _cut_top,
                    enamel_thickness, rod_taper_factor,
                    tf, theta0, r_rad, rotation,
                )

        rods = rods.union(rod)
        rod_meta.append((ring, theta0, x0, y0, x1, y1))
        if (idx + 1) % 20 == 0:
            print(f"  Created {idx + 1}/{len(rod_positions)} rods...")
    print(f"All {len(rod_positions)} rods created\n")

    # --- Compute bridge elevations ---
    outer_radius = (n_rings + 1) * center_spacing
    # Trim rods below enamel_thickness to avoid coplanar trim/sweep faces
    # (OCC boolean fails when trim box top coincides with rod sweep endpoint)
    cut_top_z = enamel_thickness - plate_overlap

    if bridge_z_offsets is not None:
        bridge_z_list = list(bridge_z_offsets)
    else:
        # Place bridges so their edges clear the plate faces (avoid tangent coplanarity).
        bridge_half = bridge_diameter / 2
        clearance = 0.02  # prevent bridge edges being tangent to plate faces
        safe_z_min = plate_overlap + bridge_half + clearance
        safe_z_max = cut_top_z - plate_overlap - bridge_half - clearance
        if n_bridge_layers == 1:
            bridge_z_list = [(safe_z_min + safe_z_max) / 2]
        else:
            bridge_z_list = [
                safe_z_min + i * (safe_z_max - safe_z_min) / (n_bridge_layers - 1)
                for i in range(n_bridge_layers)
            ]

    # --- Generate bridges ---
    bridges = cq.Workplane("XY")
    bridge_count = 0
    junction_points = []  # Collect bridge-rod intersection points for spheres

    if add_bridges:
        print("Creating bridges...")
        for ring in range(1, n_rings + 1):
            ring_rods = []
            for (r_idx, theta0, x0, y0, x1, y1) in rod_meta:
                if r_idx == ring:
                    ring_rods.append((theta0, x0, y0, x1, y1))
            ring_rods_sorted = sorted(ring_rods, key=lambda t: t[0])

            for z in bridge_z_list:
                for i in range(len(ring_rods_sorted)):
                    theta0, x0_base, y0_base, x0_top, y0_top = ring_rods_sorted[i]
                    theta1, x1_base, y1_base, x1_top, y1_top = ring_rods_sorted[
                        (i + 1) % len(ring_rods_sorted)
                    ]
                    x0_z, y0_z = get_rod_position_at_height(
                        x0_base, y0_base, x0_top, y0_top, z, enamel_thickness,
                        ring, continuous_twist, ring_rotation, twist_type,
                    )
                    x1_z, y1_z = get_rod_position_at_height(
                        x1_base, y1_base, x1_top, y1_top, z, enamel_thickness,
                        ring, continuous_twist, ring_rotation, twist_type,
                    )
                    p0 = (x0_z, y0_z, z)
                    p1 = (x1_z, y1_z, z)
                    bridges = bridges.union(make_cylinder_bridge(p0, p1, bridge_diameter))
                    bridge_count += 1
                    junction_points.append(p0)
                    junction_points.append(p1)
        print(f"Created {bridge_count} bridges")
        print(f"  Junction points collected: {len(junction_points)}\n")

    # --- Create junction reinforcement spheres ---
    sphere_solids_list = []
    if junction_sphere_factor > 0 and len(junction_points) > 0:
        sphere_diam = junction_sphere_factor * rod_diameter
        print(f"Creating junction spheres (diameter={sphere_diam:.2f} mm)...")
        # Deduplicate junction points (many bridges share rod positions)
        unique_pts = {}
        for pt in junction_points:
            key = (round(pt[0], 3), round(pt[1], 3), round(pt[2], 3))
            unique_pts[key] = pt
        unique_junction_pts = list(unique_pts.values())
        print(f"  Unique junction locations: {len(unique_junction_pts)}")

        for pt in unique_junction_pts:
            sphere = cq.Workplane("XY").transformed(
                offset=cq.Vector(*pt)
            ).sphere(sphere_diam / 2)
            sphere_solids_list.extend(sphere.solids().vals())
        print(f"  Created {len(sphere_solids_list)} spheres\n")

    # --- Collect and trim solids ---
    rod_solids = rods.solids().vals()
    bridge_solids = bridges.solids().vals()
    print(f"Rod solids: {len(rod_solids)}")
    print(f"Bridge solids: {len(bridge_solids)}")
    print(f"Sphere solids: {len(sphere_solids_list)}")
    print(f"Total solids: {len(rod_solids) + len(bridge_solids) + len(sphere_solids_list)}\n")

    # Trim box: extend epsilon past both rod endpoints to avoid coplanar faces.
    # Rod sweeps go z=0 → z=enamel_thickness.  Trim must NOT share those planes
    # or OCC boolean intersection produces degenerate geometry.
    trim_z_min = -0.01          # slightly below rod start (z=0)
    trim_z_max = cut_top_z + 0.01  # slightly above cut line (avoids rod-end coplanarity)
    trim_height = trim_z_max - trim_z_min
    print(f"Trimming solids to z=[{trim_z_min}, {trim_z_max}] ...")
    trim_box = (cq.Workplane("XY")
                .box(2 * (outer_radius + 2), 2 * (outer_radius + 2),
                     trim_height, centered=(True, True, False))
                .translate((0, 0, trim_z_min)))

    trimmed_rod_solids = []
    for s in rod_solids:
        try:
            trimmed = cq.Workplane("XY").newObject([s]).intersect(trim_box)
            vals = trimmed.solids().vals()
            if len(vals) == 1 and vals[0].isValid():
                trimmed_rod_solids.append(vals[0])
        except Exception:
            continue
    print(f"  Trimmed rods: {len(trimmed_rod_solids)}")

    trimmed_bridge_solids = []
    for s in bridge_solids:
        try:
            trimmed = cq.Workplane("XY").newObject([s]).intersect(trim_box)
            vals = trimmed.solids().vals()
            if len(vals) == 1 and vals[0].isValid():
                trimmed_bridge_solids.append(vals[0])
        except Exception:
            continue
    print(f"  Trimmed bridges: {len(trimmed_bridge_solids)}")

    trimmed_sphere_solids = []
    for s in sphere_solids_list:
        try:
            trimmed = cq.Workplane("XY").newObject([s]).intersect(trim_box)
            vals = trimmed.solids().vals()
            if len(vals) == 1 and vals[0].isValid():
                trimmed_sphere_solids.append(vals[0])
        except Exception:
            continue
    if trimmed_sphere_solids:
        print(f"  Trimmed spheres: {len(trimmed_sphere_solids)}")

    # --- Loading plates ---
    plate_side = 2 * outer_radius + 2 * plate_overhang

    bottom_plate = (cq.Workplane("XY")
        .box(plate_side, plate_side, plate_thickness + plate_overlap,
             centered=(True, True, False))
        .translate((0, 0, -plate_thickness)))

    top_plate = (cq.Workplane("XY")
        .box(plate_side, plate_side, plate_thickness + plate_overlap,
             centered=(True, True, False))
        .translate((0, 0, cut_top_z - plate_overlap)))

    # --- Fuse all solids ---
    print("\nFusing all solids into single connected body...")
    print("This may take several minutes.\n")
    t_fuse_start = time.time()

    n_fuse_steps = 3 + (1 if trimmed_sphere_solids else 0)
    step_num = 0

    # Step 1: Bottom plate + all rods
    step_num += 1
    print(f"  Step {step_num}/{n_fuse_steps}: Fusing {len(trimmed_rod_solids)} rods with bottom plate...")
    t1 = time.time()
    try:
        rod_compound = cq.Compound.makeCompound(trimmed_rod_solids)
        unified = bottom_plate.union(cq.Workplane("XY").newObject([rod_compound]))
        print(f"    Compound union OK ({time.time() - t1:.0f}s)")
    except Exception as e:
        print(f"    Compound union failed ({e}), using sequential...")
        unified = bottom_plate
        for i, s in enumerate(trimmed_rod_solids):
            unified = unified.union(cq.Workplane("XY").newObject([s]))
            if (i + 1) % 10 == 0:
                print(f"      {i+1}/{len(trimmed_rod_solids)} rods...")
        print(f"    Sequential union done ({time.time() - t1:.0f}s)")

    # Step 2: + all bridges
    step_num += 1
    print(f"  Step {step_num}/{n_fuse_steps}: Fusing {len(trimmed_bridge_solids)} bridges...")
    t2 = time.time()
    try:
        bridge_compound = cq.Compound.makeCompound(trimmed_bridge_solids)
        unified = unified.union(cq.Workplane("XY").newObject([bridge_compound]))
        print(f"    Compound union OK ({time.time() - t2:.0f}s)")
    except Exception as e:
        print(f"    Compound union failed ({e}), using sequential...")
        for i, s in enumerate(trimmed_bridge_solids):
            unified = unified.union(cq.Workplane("XY").newObject([s]))
            if (i + 1) % 50 == 0:
                print(f"      {i+1}/{len(trimmed_bridge_solids)} bridges...")
        print(f"    Sequential union done ({time.time() - t2:.0f}s)")

    # Step 2b: + junction spheres (if any)
    if trimmed_sphere_solids:
        step_num += 1
        print(f"  Step {step_num}/{n_fuse_steps}: Fusing {len(trimmed_sphere_solids)} junction spheres...")
        t2b = time.time()
        try:
            sphere_compound = cq.Compound.makeCompound(trimmed_sphere_solids)
            unified = unified.union(cq.Workplane("XY").newObject([sphere_compound]))
            print(f"    Compound union OK ({time.time() - t2b:.0f}s)")
        except Exception as e:
            print(f"    Compound union failed ({e}), using sequential...")
            for i, s in enumerate(trimmed_sphere_solids):
                unified = unified.union(cq.Workplane("XY").newObject([s]))
                if (i + 1) % 50 == 0:
                    print(f"      {i+1}/{len(trimmed_sphere_solids)} spheres...")
            print(f"    Sequential union done ({time.time() - t2b:.0f}s)")

    # Step 3: + top plate
    step_num += 1
    print(f"  Step {step_num}/{n_fuse_steps}: Fusing top plate...")
    t3 = time.time()
    unified = unified.union(top_plate)
    print(f"    Done ({time.time() - t3:.0f}s)")

    n_result_solids = len(unified.solids().vals())
    print(f"\n  Total fuse time: {time.time() - t_fuse_start:.0f}s")
    print(f"  Result: {n_result_solids} solid(s)")
    if n_result_solids > 1:
        print("  WARNING: Multiple solids -- some components may not be connected")

    # --- Apply chamfer or fillet (junction smoothing) ---
    if chamfer_size > 0:
        print(f"\nApplying chamfers (size={chamfer_size} mm)...")
        t_smooth = time.time()
        try:
            unified = unified.edges().chamfer(chamfer_size)
            print(f"  Global chamfer OK ({time.time() - t_smooth:.0f}s)")
        except Exception as e_global:
            print(f"  Global chamfer failed: {e_global}")
            print("  Attempting selective chamfer on short edges...")
            try:
                all_edges = unified.edges().vals()
                short_edges = [e for e in all_edges
                               if chamfer_size * 2 < e.Length() < 2 * bridge_diameter]
                if short_edges:
                    print(f"  Found {len(short_edges)} short edges to chamfer...")
                    unified = unified.newObject(short_edges).chamfer(chamfer_size)
                    print(f"  Selective chamfer OK ({time.time() - t_smooth:.0f}s)")
                else:
                    print("  No suitable edges for selective chamfer")
            except Exception as e_sel:
                print(f"  Selective chamfer also failed: {e_sel}")
                print("  Continuing without chamfers.")
    elif fillet_radius > 0:
        print(f"\nApplying fillets (radius={fillet_radius} mm)...")
        t_smooth = time.time()
        try:
            unified = unified.edges().fillet(fillet_radius)
            print(f"  Global fillet OK ({time.time() - t_smooth:.0f}s)")
        except Exception as e_global:
            print(f"  Global fillet failed: {e_global}")
            print("  Attempting selective fillet on short edges...")
            try:
                all_edges = unified.edges().vals()
                short_edges = [e for e in all_edges
                               if fillet_radius * 2 < e.Length() < 2 * bridge_diameter]
                if short_edges:
                    print(f"  Found {len(short_edges)} short edges to fillet...")
                    unified = unified.newObject(short_edges).fillet(fillet_radius)
                    print(f"  Selective fillet OK ({time.time() - t_smooth:.0f}s)")
                else:
                    print("  No suitable edges for selective fillet")
            except Exception as e_sel:
                print(f"  Selective fillet also failed: {e_sel}")
                print("  Continuing without fillets.")

    # --- Export ---
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    step_path = os.path.join(export_dir, "compound_enamel_lattice.step")
    cq.exporters.export(unified, step_path)
    print(f"\nSTEP exported: {step_path}")

    stl_path = os.path.join(export_dir, "compound_enamel_lattice.stl")
    cq.exporters.export(
        unified, stl_path,
        tolerance=cfg["STL_TOLERANCE"],
        angularTolerance=cfg["STL_ANGULAR_TOLERANCE"],
    )
    print(f"STL exported:  {stl_path}")

    model_z_min = -plate_thickness
    model_z_max = cut_top_z + plate_thickness
    print(f"\nModel bounds: z = {model_z_min:.3f} to {model_z_max:.3f} mm")
    print(f"  For compression_test.py regions:")
    print(f"    Bottom: 'vertices in z < {model_z_min + 0.001:.3f}'")
    print(f"    Top:    'vertices in z > {model_z_max - 0.001:.3f}'")

    return {
        "step_path": os.path.abspath(step_path),
        "stl_path": os.path.abspath(stl_path),
        "model_z_min": model_z_min,
        "model_z_max": model_z_max,
        "n_rods": len(rod_positions),
        "n_bridges": bridge_count,
        "n_solids": n_result_solids,
        "bridge_elevations": bridge_z_list,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate decussated enamel rod lattice CAD model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rod-diameter", type=float, default=DEFAULTS["ROD_DIAMETER"])
    parser.add_argument("--center-spacing", type=float, default=DEFAULTS["CENTER_SPACING"])
    parser.add_argument("--enamel-thickness", type=float, default=DEFAULTS["ENAMEL_THICKNESS"])
    parser.add_argument("--n-rings", type=int, default=DEFAULTS["N_RINGS"])
    parser.add_argument("--bridge-diameter", type=float, default=DEFAULTS["BRIDGE_DIAMETER"])
    parser.add_argument("--n-bridge-layers", type=int, default=DEFAULTS["N_BRIDGE_LAYERS"])
    parser.add_argument("--plate-thickness", type=float, default=DEFAULTS["PLATE_THICKNESS"])
    parser.add_argument("--plate-overlap", type=float, default=DEFAULTS["PLATE_OVERLAP"])
    parser.add_argument("--twist-type", choices=["linear", "accelerating", "sigmoid"],
                        default=DEFAULTS["TWIST_TYPE"])
    parser.add_argument("--fillet-radius", type=float, default=DEFAULTS["FILLET_RADIUS"],
                        help="Fillet radius at junctions (mm). 0 = no fillet.")
    parser.add_argument("--chamfer-size", type=float, default=DEFAULTS["CHAMFER_SIZE"],
                        help="Chamfer size at junctions (mm). 0 = no chamfer. Simpler than fillet.")
    parser.add_argument("--junction-sphere-factor", type=float,
                        default=DEFAULTS["JUNCTION_SPHERE_FACTOR"],
                        help="Junction sphere diam = factor * rod_diam. 0 = no spheres.")
    parser.add_argument("--rod-taper-factor", type=float,
                        default=DEFAULTS["ROD_TAPER_FACTOR"],
                        help="Flare rod ends at plates: 0=none, 0.3=30%% larger diameter at plate face.")
    parser.add_argument("--export-dir", default=DEFAULTS["EXPORT_DIR"])
    parser.add_argument("--no-bridges", action="store_true",
                        help="Disable horizontal bridges.")
    parser.add_argument("--show", action="store_true",
                        help="Display 3D visualization (requires jupyter_cadquery).")

    args = parser.parse_args()

    params = {
        "ROD_DIAMETER": args.rod_diameter,
        "CENTER_SPACING": args.center_spacing,
        "ENAMEL_THICKNESS": args.enamel_thickness,
        "N_RINGS": args.n_rings,
        "BRIDGE_DIAMETER": args.bridge_diameter,
        "N_BRIDGE_LAYERS": args.n_bridge_layers,
        "PLATE_THICKNESS": args.plate_thickness,
        "PLATE_OVERLAP": args.plate_overlap,
        "TWIST_TYPE": args.twist_type,
        "FILLET_RADIUS": args.fillet_radius,
        "CHAMFER_SIZE": args.chamfer_size,
        "JUNCTION_SPHERE_FACTOR": args.junction_sphere_factor,
        "ROD_TAPER_FACTOR":       args.rod_taper_factor,
        "EXPORT_DIR": args.export_dir,
        "ADD_HORIZONTAL_BRIDGES": not args.no_bridges,
    }

    result = generate_lattice(params)
    print(f"\nDone. STEP: {result['step_path']}")
    print(f"       STL: {result['stl_path']}")

    # Write a JSON sidecar so downstream scripts (extract_metrics.py, etc.)
    # know the exact bridge elevations and key geometry for this design.
    export_dir = params.get("EXPORT_DIR", "enamel_exports")
    sidecar = {
        "bridge_elevations": result["bridge_elevations"],
        "model_z_min": result["model_z_min"],
        "model_z_max": result["model_z_max"],
        "specimen_height": result["model_z_max"] - result["model_z_min"],
        "cut_top_z": result["model_z_max"] - params.get("PLATE_THICKNESS", DEFAULTS["PLATE_THICKNESS"]),
        "plate_overlap": params.get("PLATE_OVERLAP", DEFAULTS["PLATE_OVERLAP"]),
        "n_rods": result["n_rods"],
        "n_bridges": result["n_bridges"],
        "params": {k: v for k, v in params.items() if k != "EXPORT_DIR"},
    }
    sidecar_path = os.path.join(export_dir, "lattice_params.json")
    with open(sidecar_path, "w") as fh:
        json.dump(sidecar, fh, indent=2)
    print(f"       JSON: {sidecar_path}")


if __name__ == "__main__":
    main()
