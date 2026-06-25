"""Measured-profile twist generator.

Replaces the stock three-way twist registry (linear, accelerating, sigmoid)
with a CubicSpline built from the morphometrics `measured_twist_profile`
(z_um → twist_deg). The resulting twist function has the same
(z_current, z_total, total_rotation_deg) → radians signature lattice_cad.py
expects, so `generate_lattice(params)` runs unmodified.

Invocation (identical to the embedded continuous_twist driver):
    python measured_profile_twist.py --params cad_params.json --export-dir DIR
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Make the stock CAD module importable.
CAD_STACK_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "cad_modeling"
    / "Decussated Models Continous Twist"
)
if str(CAD_STACK_DIR) not in sys.path:
    sys.path.insert(0, str(CAD_STACK_DIR))

import lattice_cad  # noqa: E402


def _build_measured_twist(profile, enamel_thickness_mm):
    """Return a callable (z_current, H, total_rotation_deg) -> radians.

    `profile` is {"z_um": [...], "twist_deg": [...]}. We rescale z_um to the
    mm-space of the generated lattice so the full measured twist fits within
    H (enamel thickness).
    """
    try:
        import numpy as np
        from scipy.interpolate import CubicSpline  # type: ignore

        have_scipy = True
    except ImportError:
        have_scipy = False

    z_um = profile["z_um"]
    t_deg = profile["twist_deg"]
    if not z_um or not t_deg or len(z_um) != len(t_deg):
        # Degenerate input; return stock linear twist.
        return lattice_cad.linear_twist

    z0, zN = float(z_um[0]), float(z_um[-1])
    z_range_um = max(zN - z0, 1e-9)
    H_mm = float(enamel_thickness_mm)

    if have_scipy:
        import numpy as np

        z_arr = np.asarray(z_um, dtype=float)
        t_arr = np.asarray(t_deg, dtype=float)
        spline = CubicSpline(z_arr, t_arr, extrapolate=True)

        def twist_fn(z_current: float, H: float, total_rotation_deg: float) -> float:
            # Map lattice z in [0, H] to profile z in [z0, zN] and evaluate.
            frac = 0.0 if H <= 0 else max(0.0, min(1.0, z_current / H))
            z_profile = z0 + frac * z_range_um
            base_deg = float(spline(z_profile))
            # Normalise so that z=0 corresponds to 0 rotation, then scale so
            # that the z=H rotation is `total_rotation_deg`.
            base0 = float(spline(z0))
            baseN = float(spline(zN))
            span = baseN - base0
            if abs(span) < 1e-9:
                return math.radians(total_rotation_deg) * frac
            scaled_deg = (base_deg - base0) / span * total_rotation_deg
            return math.radians(scaled_deg)

        return twist_fn

    # Fallback: piecewise-linear interpolation if scipy is absent.
    def twist_fn_plain(z_current: float, H: float, total_rotation_deg: float) -> float:
        frac = 0.0 if H <= 0 else max(0.0, min(1.0, z_current / H))
        z_profile = z0 + frac * z_range_um
        # linear search
        for i in range(1, len(z_um)):
            if z_profile <= z_um[i]:
                z_a, z_b = z_um[i - 1], z_um[i]
                t_a, t_b = t_deg[i - 1], t_deg[i]
                t = (z_profile - z_a) / max(z_b - z_a, 1e-12)
                base_deg = t_a + t * (t_b - t_a)
                break
        else:
            base_deg = t_deg[-1]
        base0 = t_deg[0]
        span = t_deg[-1] - t_deg[0]
        if abs(span) < 1e-9:
            return math.radians(total_rotation_deg) * frac
        scaled_deg = (base_deg - base0) / span * total_rotation_deg
        return math.radians(scaled_deg)

    return twist_fn_plain


def _patch_twist_registry(twist_fn) -> None:
    """Register our twist function as the 'measured' option.

    lattice_cad.py has a function get_twist_function(name) that maps names
    to its three built-in twists. We monkey-patch it to recognise 'measured'.
    """
    original = lattice_cad.get_twist_function

    def patched(name: str):
        if name == "measured":
            return twist_fn
        return original(name)

    lattice_cad.get_twist_function = patched


def _coerce_ring_rotation(rr):
    if rr is None:
        return None
    return {int(k): float(v) for k, v in rr.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    p.add_argument("--export-dir", required=True)
    args = p.parse_args()

    with open(args.params) as fh:
        user_params = json.load(fh)

    profile = user_params.get("measured_twist_profile")
    if not profile:
        print(
            "[measured_profile_twist] no measured_twist_profile in params; "
            "falling back to TWIST_TYPE=sigmoid"
        )
        user_params["TWIST_TYPE"] = "sigmoid"
    else:
        twist_fn = _build_measured_twist(profile, user_params.get("ENAMEL_THICKNESS", 20.0))
        _patch_twist_registry(twist_fn)
        user_params["TWIST_TYPE"] = "measured"

    if "RING_ROTATION" in user_params:
        user_params["RING_ROTATION"] = _coerce_ring_rotation(user_params["RING_ROTATION"])

    known_keys = set(lattice_cad.DEFAULTS.keys())
    stock_params = {}
    for k, v in user_params.items():
        if k in known_keys:
            stock_params[k] = v
    stock_params["EXPORT_DIR"] = args.export_dir

    result = lattice_cad.generate_lattice(stock_params)

    sidecar = {
        "bridge_elevations": result["bridge_elevations"],
        "model_z_min": result["model_z_min"],
        "model_z_max": result["model_z_max"],
        "specimen_height": result["model_z_max"] - result["model_z_min"],
        "cut_top_z": result["model_z_max"]
        - stock_params.get("PLATE_THICKNESS", lattice_cad.DEFAULTS["PLATE_THICKNESS"]),
        "plate_overlap": stock_params.get("PLATE_OVERLAP", lattice_cad.DEFAULTS["PLATE_OVERLAP"]),
        "n_rods": result["n_rods"],
        "n_bridges": result["n_bridges"],
        "params": {k: v for k, v in stock_params.items() if k != "EXPORT_DIR"},
        "model_type": "measured_profile",
        "measured_profile_points": len(profile["z_um"]) if profile else 0,
    }
    with open(os.path.join(args.export_dir, "lattice_params.json"), "w") as fh:
        json.dump(sidecar, fh, indent=2, default=str)
    print(f"Sidecar written: {os.path.join(args.export_dir, 'lattice_params.json')}")


if __name__ == "__main__":
    main()
