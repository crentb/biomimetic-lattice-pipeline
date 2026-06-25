"""Map pitch/yaw depth profiles to RING_ROTATION, TWIST_TYPE, and optionally a
measured_twist_profile for the measured_profile_twist generator.

Core idea: integrate signed pitch along z to get cumulative twist phi(z);
resample onto ring positions; classify by RMSE of canonical twist shapes
(linear, sigmoid, power-law). If none fits within tolerance, emit TWIST_TYPE
'measured' plus the full samples.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

MEASURED_RMSE_FRAC_THRESHOLD = 0.10


def map_twist(
    morphometrics: Dict[str, Any],
    n_rings: int,
    enamel_thickness_mm: float,
) -> Dict[str, Any]:
    """Return a fragment with RING_ROTATION, TWIST_TYPE, and (optionally)
    measured_twist_profile plus twist_perturbation_amplitude_deg.
    """
    import numpy as np

    dp = morphometrics.get("depth_profiles", {}) or {}
    depth_um = np.asarray(dp.get("depth_um", []), dtype=float)
    pitch_signed = np.asarray(dp.get("pitch_signed_deg", []), dtype=float)

    ring_rotation: Dict[int, float] = {i: 0.0 for i in range(n_rings + 1)}
    twist_type = "linear"
    measured_profile = None
    perturb_amp = 0.0

    if depth_um.size >= 2 and pitch_signed.size == depth_um.size:
        order = np.argsort(depth_um)
        z_um = depth_um[order]
        p_signed = pitch_signed[order]

        # pitch_deg is in degrees; treat as dφ/dz in deg/um; integrate
        phi_deg = _cumtrapz(p_signed, z_um)

        z_total_um = z_um[-1] - z_um[0]
        if z_total_um <= 0:
            z_total_um = 1.0

        # Resample onto ring z positions across ENAMEL_THICKNESS_mm (scale ratio)
        ring_z_mm = np.linspace(0, enamel_thickness_mm, n_rings + 1)
        ring_z_um = ring_z_mm / enamel_thickness_mm * z_total_um + z_um[0]
        phi_at_rings = np.interp(ring_z_um, z_um, phi_deg)
        phi_relative = phi_at_rings - phi_at_rings[0]

        for i in range(n_rings + 1):
            ring_rotation[i] = float(phi_relative[i])

        twist_type, rmse_frac = _classify_twist(z_um, phi_deg)
        if rmse_frac > MEASURED_RMSE_FRAC_THRESHOLD:
            twist_type = "measured"
            measured_profile = {"z_um": z_um.tolist(), "twist_deg": phi_deg.tolist()}

    tortuosity = _lookup_scalar(morphometrics, "rod_tracks_summary", "tortuosity_mean", default=1.0)
    perturb_amp = float(max(0.0, min(15.0, (tortuosity - 1.0) * 180.0)))

    out: Dict[str, Any] = {
        "RING_ROTATION": ring_rotation,
        "TWIST_TYPE": twist_type,
        "twist_perturbation_amplitude_deg": perturb_amp,
    }
    if measured_profile:
        out["measured_twist_profile"] = measured_profile
    return out


def _cumtrapz(y: Any, x: Any) -> Any:
    import numpy as np

    dx = np.diff(x)
    incr = 0.5 * (y[:-1] + y[1:]) * dx
    out = np.zeros_like(y)
    out[1:] = np.cumsum(incr)
    return out


def _classify_twist(z: Any, phi: Any) -> Tuple[str, float]:
    import numpy as np

    if z.size < 3:
        return "linear", 0.0
    z_norm = (z - z[0]) / max(z[-1] - z[0], 1e-9)
    phi_scale = float(max(abs(phi).max(), 1e-9))
    phi_hat = phi / phi_scale

    linear = z_norm
    accel = z_norm**2
    k = 8.0
    sigmoid = 1.0 / (1.0 + np.exp(-k * (z_norm - 0.5)))
    sigmoid = (sigmoid - sigmoid.min()) / max(sigmoid.max() - sigmoid.min(), 1e-9)

    candidates = {"linear": linear, "accelerating": accel, "sigmoid": sigmoid}

    best_name = "linear"
    best_rmse_frac = float("inf")
    for name, shape in candidates.items():
        shape_scaled = shape * float(phi_hat[-1] if phi_hat[-1] != 0 else 1.0)
        rmse = float(np.sqrt(np.mean((phi_hat - shape_scaled) ** 2)))
        rmse_frac = rmse / max(abs(phi_hat).max(), 1e-9)
        if rmse_frac < best_rmse_frac:
            best_rmse_frac = rmse_frac
            best_name = name
    return best_name, best_rmse_frac


def _lookup_scalar(morphometrics: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    cur: Any = morphometrics
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default
