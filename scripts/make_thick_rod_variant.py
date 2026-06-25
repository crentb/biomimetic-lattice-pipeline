#!/usr/bin/env python
"""
make_thick_rod_variant.py
=========================

Regenerate the compound enamel lattice for an existing trial with the LARGEST
rod diameter that keeps every pair of rod surfaces strictly non-intersecting,
i.e. produces a lattice whose rods are "barely not touching" everywhere along
their twisted height.

Why this script exists
----------------------
The stock CAD generator (cad_modeling/.../lattice_cad.py) places rods on a
hex-pack of 6 concentric rings at radius k * CENTER_SPACING (k = 0..N_RINGS,
6k rods per ring) and twists each ring by RING_ROTATION[k] degrees about the
z-axis as z runs from 0 to ENAMEL_THICKNESS. Because different rings spin at
different angular rates, the centre-to-centre distance between rods in
adjacent rings VARIES with z. The minimum allowed ROD_DIAMETER such that the
rods never intersect is therefore set by the worst-case (over rod pairs and
over z) centre-to-centre distance d_min, minus a small clearance:

    new ROD_DIAMETER = d_min - clearance

Same-ring pairs see a constant inter-rod distance (they spin together);
different-ring pairs see a time-varying d(z) and are the only pairs that can
ever cross. The script samples z densely and finds the global d_min by
brute-force pairwise distance computation across all rods at every sampled z.

What this script does NOT modify
--------------------------------
- The stock CAD module (cad_modeling/...) is imported but never written.
- The original trial CAD outputs (compound_enamel_lattice.stl/.step,
  lattice_params.json, cad_driver.py, cad_run.log, cad_params_used.json)
  are left untouched.
- Nothing outside the trial is touched except this script's own home.
- BRIDGE_DIAMETER is intentionally left unchanged: the user wants the
  enamel rods enlarged, not the bridges. The stock validation
  BRIDGE_DIAMETER < ROD_DIAMETER continues to hold trivially because we
  only grow ROD_DIAMETER.

Inputs (CLI)
------------
--trial-dir          : path to the trial directory containing cad/.
--clearance          : surface-to-surface gap (mm) between adjacent rods.
                       Default 0.025 (matches user's choice).
--z-samples          : number of z samples for the inter-rod distance scan.
                       Default 2001 over [0, ENAMEL_THICKNESS].
--sandbox-name       : sub-directory under cad/ for the new CAD run.
                       Default '_thick_variant_work'.
--output-stl-name    : filename of the new STL placed alongside the original.
                       Default 'compound_enamel_lattice_thick.stl'.
--cad-env            : conda env where lattice_cad runs (optional override;
                       cad_runner's default is used otherwise).

Outputs
-------
- <trial>/cad/<sandbox-name>/                         : NEW sandbox containing
    cad_driver.py, cad_params_used.json, cad_run.log,
    compound_enamel_lattice.step,
    compound_enamel_lattice.stl,
    lattice_params.json
- <trial>/cad/<output-stl-name>                       : COPY of the new STL
- stdout report                                        : geometry summary

Geometric caveat
----------------
This analysis uses XY-plane centre-to-centre distance as the proxy for rod
clearance. Stock lattice_cad sweeps a CIRCULAR profile in the XY plane along
each rod's twisted centreline; for rods that are appreciably tilted (outer
rings), the cross-section perpendicular to the actual centreline is slightly
elliptical when projected to XY, so the true surface-to-surface gap is a
hair LARGER than (d_min - new_ROD_DIAMETER). The reported clearance value is
therefore a conservative lower bound, which is the safe direction for
"barely not touching".
"""

from __future__ import annotations

# --- Standard library --------------------------------------------------------
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

# --- Third party -------------------------------------------------------------
import numpy as np

# --- Pipeline path setup -----------------------------------------------------
# Make `generators.cad_runner` importable without installing the package.
THIS = Path(__file__).resolve()
BIOMIMETIC_ROOT = THIS.parent.parent
if str(BIOMIMETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(BIOMIMETIC_ROOT))

from biomimetic_pipeline.generators import cad_runner  # noqa: E402

# ---------------------------------------------------------------------------
# Twist arithmetic (numpy-vectorised mirror of stock lattice_cad twist funcs)
# ---------------------------------------------------------------------------
# Stock definitions live in cad_modeling/.../lattice_cad.py (linear_twist,
# accelerating_twist, sigmoid_twist). We re-implement them here so we can
# call them WITHOUT importing cadquery (lattice_cad pulls in cq at import
# time). The vector form accepts an array of per-rod rotation_deg values
# and returns the per-rod twist angle (radians) at a single z.


def _twist_vector(twist_type: str, z: float, H: float, rot_deg: np.ndarray) -> np.ndarray:
    """Return twist angle (radians) at height z for each rod's ring rotation.

    Parameters
    ----------
    twist_type : 'linear' | 'accelerating' | 'sigmoid' (else: linear)
    z          : evaluation height (mm), in [0, H]
    H          : ENAMEL_THICKNESS (mm)
    rot_deg    : per-rod array of total ring rotation in degrees

    Returns
    -------
    np.ndarray of twist angles in radians, same shape as rot_deg.
    """
    if twist_type == "linear":
        # Linear ramp: theta(z) = (z/H) * total_rotation
        return (z / H) * np.radians(rot_deg)

    if twist_type == "accelerating":
        # Quadratic ramp: stays near 0 until later in the rod height
        return ((z / H) ** 2) * np.radians(rot_deg)

    if twist_type == "sigmoid":
        # S-curve normalised so theta(0)=0 and theta(H)=total_rotation.
        # Constants mirror stock sigmoid_twist exactly (k=6).
        k = 6
        z_norm = z / H
        sigmoid_val = 1.0 / (1.0 + math.exp(-k * (z_norm - 0.5)))
        denom = 1.0 / (1.0 + math.exp(-k * 0.5)) - 1.0 / (1.0 + math.exp(k * 0.5))
        sigmoid_norm = (sigmoid_val - 1.0 / (1.0 + math.exp(k * 0.5))) / denom
        return sigmoid_norm * np.radians(rot_deg)

    # Fallback matches stock get_twist_function default.
    return (z / H) * np.radians(rot_deg)


# ---------------------------------------------------------------------------
# Hex-pack rod centres (mirrors stock lattice_cad.generate_radial_positions)
# ---------------------------------------------------------------------------


def _build_rod_centers(n_rings: int, center_spacing: float):
    """Return list of (x0, y0, ring_idx, theta0) for every rod base position.

    Layout:
      ring 0: 1 rod at the origin
      ring k>=1: 6k rods on a circle of radius k*center_spacing, uniformly
                 spaced at theta_i = 2*pi*i / (6k).
    Total: 1 + 6 + 12 + 18 + 24 + 30 = 91 rods for N_RINGS=5.
    """
    centers = [(0.0, 0.0, 0, 0.0)]
    for k in range(1, n_rings + 1):
        radius = k * center_spacing
        n_rods = 6 * k
        for i in range(n_rods):
            theta = 2.0 * math.pi * i / n_rods
            centers.append((radius * math.cos(theta), radius * math.sin(theta), k, theta))
    return centers


# ---------------------------------------------------------------------------
# Global minimum centre-to-centre distance over (z, rod-pair)
# ---------------------------------------------------------------------------


def _min_pairwise_center_distance(
    rod_centers, ring_rotation, twist_type, H, z_samples, continuous_twist
):
    """Find the worst-case (smallest) centre-to-centre distance.

    For continuous_twist=True the rod centreline at height z lies on a
    horizontal circle of radius r at angle theta0 + twist(z, H, rot_deg).
    For continuous_twist=False the centreline is the straight segment from
    (x0, y0, 0) to (x_top, y_top, H) where (x_top, y_top) is (x0, y0)
    rotated about the z-axis by the full ring rotation.
    """
    # Per-rod constants (r, theta0, ring_index)
    r0 = np.array([math.hypot(x, y) for x, y, _, _ in rod_centers])
    theta0 = np.array(
        [math.atan2(y, x) if math.hypot(x, y) > 1e-12 else 0.0 for x, y, _, _ in rod_centers]
    )
    ring = np.array([rk for _, _, rk, _ in rod_centers], dtype=int)
    rot_deg = np.array([float(ring_rotation.get(int(k), 0.0)) for k in ring])

    z_vals = np.linspace(0.0, float(H), int(z_samples))
    n = len(rod_centers)

    best = {"d_min": float("inf"), "z": None, "i": -1, "j": -1}

    for z in z_vals:
        if continuous_twist:
            # Helical centreline: each rod stays on its ring radius r0.
            twist = _twist_vector(twist_type, float(z), float(H), rot_deg)
            theta_z = theta0 + twist
            x = r0 * np.cos(theta_z)
            y = r0 * np.sin(theta_z)
        else:
            # Simple rotation: straight tilted rod from base to top.
            theta_top = theta0 + np.radians(rot_deg)
            x_top = r0 * np.cos(theta_top)
            y_top = r0 * np.sin(theta_top)
            t = float(z) / float(H)
            x_base = r0 * np.cos(theta0)
            y_base = r0 * np.sin(theta0)
            x = x_base + t * (x_top - x_base)
            y = y_base + t * (y_top - y_base)

        # Pairwise distance matrix; mask diagonal so argmin ignores self pairs.
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, np.inf)

        flat = int(np.argmin(d))
        i, j = divmod(flat, n)
        d_here = float(d[i, j])
        if d_here < best["d_min"]:
            best = {"d_min": d_here, "z": float(z), "i": i, "j": j}

    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate a trial's lattice with the largest non-touching rod diameter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trial-dir",
        required=True,
        type=Path,
        help="Path to the trial directory (must contain cad/lattice_params.json).",
    )
    parser.add_argument(
        "--clearance",
        type=float,
        default=0.025,
        help="Surface-to-surface clearance gap (mm) between adjacent rods. Default 0.025.",
    )
    parser.add_argument(
        "--z-samples",
        type=int,
        default=2001,
        help="Number of z samples for the inter-rod distance scan. Default 2001.",
    )
    parser.add_argument(
        "--sandbox-name",
        type=str,
        default="_thick_variant_work",
        help="Sub-directory under cad/ for the new CAD run. Default '_thick_variant_work'.",
    )
    parser.add_argument(
        "--output-stl-name",
        type=str,
        default="compound_enamel_lattice_thick.stl",
        help="Filename of the new STL placed alongside the original.",
    )
    parser.add_argument(
        "--cad-env",
        type=str,
        default=None,
        help="Override conda env where lattice_cad runs (default: cad_runner's default).",
    )
    args = parser.parse_args()

    # ---- 1. Locate and parse the trial sidecar -----------------------------
    trial_dir = args.trial_dir.expanduser().resolve()
    cad_dir = trial_dir / "cad"
    sidecar_path = cad_dir / "lattice_params.json"
    if not sidecar_path.is_file():
        sys.exit(f"ERROR: missing {sidecar_path}")

    sidecar = json.loads(sidecar_path.read_text())
    # `params` holds the stock CAD generator inputs; `biomimetic_unknown_keys`
    # holds extras (per_ring_diameter, rod_cross_section, provenance) that the
    # stock generator does not consume but the pipeline records for traceability.
    params = dict(sidecar["params"])
    unknown_keys = dict(sidecar.get("biomimetic_unknown_keys", {}))

    # Pull the geometric inputs we need for the distance scan.
    center_spacing = float(params["CENTER_SPACING"])
    n_rings = int(params["N_RINGS"])
    enamel_thickness = float(params["ENAMEL_THICKNESS"])
    twist_type = str(params.get("TWIST_TYPE", "linear"))
    continuous_twist = bool(params.get("CONTINUOUS_TWIST", True))
    # JSON serialises int keys as strings; coerce for consistency with stock.
    ring_rotation = {int(k): float(v) for k, v in (params.get("RING_ROTATION") or {}).items()}
    rod_diameter_old = float(params["ROD_DIAMETER"])
    bridge_diameter = float(params["BRIDGE_DIAMETER"])

    # ---- 2. Reconstruct rod centres + run the worst-case distance scan ----
    rod_centers = _build_rod_centers(n_rings, center_spacing)
    n_rods_expected = 1 + sum(6 * k for k in range(1, n_rings + 1))
    assert len(rod_centers) == n_rods_expected, "rod_centers length disagrees with hex-pack count"
    # Cross-check against the sidecar's n_rods value (warn-only).
    sidecar_n_rods = int(sidecar.get("n_rods", n_rods_expected))
    if sidecar_n_rods != n_rods_expected:
        print(
            f"WARNING: sidecar n_rods={sidecar_n_rods} != computed {n_rods_expected}; "
            "continuing with computed value (matches stock generator math)."
        )

    scan = _min_pairwise_center_distance(
        rod_centers=rod_centers,
        ring_rotation=ring_rotation,
        twist_type=twist_type,
        H=enamel_thickness,
        z_samples=args.z_samples,
        continuous_twist=continuous_twist,
    )
    d_min = scan["d_min"]
    if not math.isfinite(d_min) or d_min <= 0:
        sys.exit(f"ERROR: d_min came out non-positive ({d_min}); cannot enlarge rods.")

    # ---- 3. Compute the new ROD_DIAMETER and validate constraints ---------
    rod_diameter_new = d_min - args.clearance

    if rod_diameter_new <= rod_diameter_old:
        # The trial already had rods bigger than (d_min - clearance), which means
        # the original CAD itself sits right at or above the chosen clearance.
        # We continue but warn so the user sees the result is a no-op or shrink.
        print(
            f"WARNING: new ROD_DIAMETER ({rod_diameter_new:.4f} mm) <= old "
            f"({rod_diameter_old:.4f} mm). The original trial already met or "
            "exceeded the clearance target; the variant will not be larger."
        )
    if rod_diameter_new <= bridge_diameter:
        # Stock generator raises ValueError if BRIDGE_DIAMETER >= ROD_DIAMETER.
        sys.exit(
            f"ERROR: new ROD_DIAMETER ({rod_diameter_new:.4f} mm) <= "
            f"BRIDGE_DIAMETER ({bridge_diameter:.4f} mm); stock lattice_cad "
            "would reject. Reduce clearance, reduce BRIDGE_DIAMETER, or "
            "increase CENTER_SPACING."
        )

    # ---- 4. Report the geometry decision before generating ----------------
    i = scan["i"]
    j = scan["j"]
    ring_i = rod_centers[i][2]
    ring_j = rod_centers[j][2]
    print("=" * 72)
    print("THICK-ROD VARIANT  -  ROD DIAMETER ANALYSIS")
    print("=" * 72)
    print(f"  trial dir:                {trial_dir}")
    print(f"  n_rods (computed):        {len(rod_centers)}")
    print(f"  CENTER_SPACING:           {center_spacing:.6f} mm")
    print(f"  ENAMEL_THICKNESS (H):     {enamel_thickness:.6f} mm")
    print(f"  TWIST_TYPE:               {twist_type}")
    print(f"  CONTINUOUS_TWIST:         {continuous_twist}")
    print(f"  z samples:                {args.z_samples}")
    print(f"  Global min centre-distance d_min:  {d_min:.6f} mm")
    print(f"    constraining pair: rod #{i} (ring {ring_i}) <-> rod #{j} (ring {ring_j})")
    print(f"    at z = {scan['z']:.4f} mm")
    print(f"  Clearance request:        {args.clearance:.4f} mm")
    print(f"  ROD_DIAMETER old:         {rod_diameter_old:.6f} mm")
    print(f"  ROD_DIAMETER new:         {rod_diameter_new:.6f} mm")
    print(f"  delta ROD_DIAMETER:       {rod_diameter_new - rod_diameter_old:+.6f} mm")
    print(f"  Predicted surface gap:    {d_min - rod_diameter_new:.4f} mm  (== clearance)")
    print(f"  BRIDGE_DIAMETER kept:     {bridge_diameter:.6f} mm  (unchanged)")
    print("=" * 72)

    # ---- 5. Build the parameter dict for the stock generator --------------
    # We deep-copy the existing params, swap ROD_DIAMETER, and re-attach the
    # unknown keys so cad_driver.py stashes them under biomimetic_unknown_keys
    # in the new sidecar (same provenance behaviour as the original run).
    new_params = dict(params)
    new_params["ROD_DIAMETER"] = rod_diameter_new

    # Scale per_ring_diameter proportionally so the metadata stays internally
    # consistent. Stock lattice_cad ignores this field; downstream reporting
    # code in the biomimetic_pipeline may read it, so keeping the ratios
    # honest avoids confusion.
    per_ring = unknown_keys.get("per_ring_diameter")
    if per_ring:
        scale = rod_diameter_new / rod_diameter_old
        unknown_keys["per_ring_diameter"] = {str(k): float(v) * scale for k, v in per_ring.items()}

    # Re-attach unknown keys. cad_runner's driver will filter them back out
    # of the stock-call but record them under biomimetic_unknown_keys.
    for k, v in unknown_keys.items():
        new_params[k] = v

    # Append a scale_clamp-style provenance entry so the variant explains
    # itself in the new sidecar.
    prov = dict(new_params.get("provenance", {}))
    clamps = list(prov.get("scale_clamps", []))
    clamps.append(
        {
            "field": "ROD_DIAMETER",
            "raw_value": rod_diameter_old,
            "clamped_value": rod_diameter_new,
            "reason": (
                f"thick-rod variant: grew ROD_DIAMETER from {rod_diameter_old:.6f} "
                f"to {rod_diameter_new:.6f} mm so worst-case rod-surface clearance "
                f"= {args.clearance:.4f} mm "
                f"(d_min = {d_min:.6f} mm at z = {scan['z']:.3f} mm, "
                f"ring{ring_i} <-> ring{ring_j}). "
                f"Generated by scripts/make_thick_rod_variant.py."
            ),
        }
    )
    prov["scale_clamps"] = clamps
    new_params["provenance"] = prov

    # ---- 6. Invoke the stock CAD generator in a sandbox sub-directory -----
    sandbox_dir = cad_dir / args.sandbox_name
    if sandbox_dir.exists():
        # Refuse to clobber: forces the user to inspect / move / pick a new name.
        sys.exit(
            f"ERROR: sandbox dir already exists: {sandbox_dir}. Remove it or "
            "pass --sandbox-name <other> so we do not overwrite earlier work."
        )
    sandbox_dir.mkdir(parents=True, exist_ok=False)
    print(f"[thick_variant] Running stock lattice_cad in sandbox: {sandbox_dir}")
    runner_kwargs = {}
    if args.cad_env:
        runner_kwargs["cad_env"] = args.cad_env
    cad_result = cad_runner.run(new_params, sandbox_dir, **runner_kwargs)

    # ---- 7. Copy the new STL out next to the original (without overwrite) -
    target_stl = cad_dir / args.output_stl_name
    if target_stl.exists():
        sys.exit(
            f"ERROR: target STL already exists, refusing to overwrite: {target_stl}. "
            "Move/delete it, or pass --output-stl-name <other> to publish under a "
            "different name."
        )
    shutil.copy2(cad_result.stl_path, target_stl)

    # ---- 8. Final summary -------------------------------------------------
    print(f"[thick_variant] Copied STL -> {target_stl}")
    print(f"[thick_variant] Sandbox preserved at: {sandbox_dir}")
    print(f"  - STEP:    {cad_result.step_path}")
    print(f"  - STL:     {cad_result.stl_path}")
    print(f"  - sidecar: {cad_result.sidecar_path}")
    print(f"  - log:     {cad_result.log_path}")


if __name__ == "__main__":
    main()
