"""Hierarchical (multi-scale) twist generator.

Composes two twist signals:
  - slow, global: whatever TWIST_TYPE the mapping picked
    (linear / accelerating / sigmoid / measured)
  - fast, local: sine wobble whose frequency is 1 / dominant_wavelength_um
    (from the SOM periodicity) and whose amplitude is
    twist_perturbation_amplitude_deg (driven by measured rod tortuosity).

The composite preserves the total twist of the slow component (so
lattice_cad's end-ring rotations still match RING_ROTATION), with the fast
component superimposed as a zero-mean perturbation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

CAD_STACK_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "cad_modeling"
    / "Decussated Models Continous Twist"
)
if str(CAD_STACK_DIR) not in sys.path:
    sys.path.insert(0, str(CAD_STACK_DIR))

import lattice_cad  # noqa: E402


def _build_hierarchical(
    base_twist_name: str,
    measured_profile,
    amplitude_deg: float,
    wavelength_um: float,
    enamel_thickness_mm: float,
):
    """Return composite twist callable (z, H, total_rotation_deg) -> radians."""
    # Resolve slow component
    if base_twist_name == "measured" and measured_profile:
        from biomimetic_pipeline.generators.new_models.measured_profile_twist import (
            _build_measured_twist,
        )

        slow = _build_measured_twist(measured_profile, enamel_thickness_mm)
    else:
        slow = lattice_cad.get_twist_function(base_twist_name)

    # Fast wobble: frequency = H_mm / wavelength_mm cycles over [0, H]
    # Zero-mean sine so total twist at z=H matches the slow component.
    wavelength_mm = max(wavelength_um * 1e-3, 1e-6)
    n_cycles = float(enamel_thickness_mm) / wavelength_mm
    omega = 2.0 * math.pi * n_cycles  # over [0, 1] normalized z

    def composite(z_current: float, H: float, total_rotation_deg: float) -> float:
        slow_rad = slow(z_current, H, total_rotation_deg)
        frac = 0.0 if H <= 0 else max(0.0, min(1.0, z_current / H))
        wobble_rad = math.radians(amplitude_deg) * math.sin(omega * frac)
        return slow_rad + wobble_rad

    return composite


def _patch_twist_registry(fn) -> None:
    original = lattice_cad.get_twist_function

    def patched(name: str):
        if name == "hierarchical":
            return fn
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

    base_twist_name = user_params.get("TWIST_TYPE", "linear")
    amplitude = float(user_params.get("twist_perturbation_amplitude_deg", 0.0))
    # Wavelength defaults to 85 µm biology, but we respect any override passed in.
    wavelength_um = float(user_params.get("_dominant_wavelength_um_mean", 85.0))
    enamel = float(user_params.get("ENAMEL_THICKNESS", 20.0))

    composite = _build_hierarchical(
        base_twist_name,
        user_params.get("measured_twist_profile"),
        amplitude_deg=amplitude,
        wavelength_um=wavelength_um,
        enamel_thickness_mm=enamel,
    )
    _patch_twist_registry(composite)
    user_params["TWIST_TYPE"] = "hierarchical"

    if "RING_ROTATION" in user_params:
        user_params["RING_ROTATION"] = _coerce_ring_rotation(user_params["RING_ROTATION"])

    known_keys = set(lattice_cad.DEFAULTS.keys())
    stock_params = {k: v for k, v in user_params.items() if k in known_keys}
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
        "model_type": "hierarchical",
        "base_twist": base_twist_name,
        "wobble_amplitude_deg": amplitude,
        "wobble_wavelength_um": wavelength_um,
    }
    with open(os.path.join(args.export_dir, "lattice_params.json"), "w") as fh:
        json.dump(sidecar, fh, indent=2, default=str)
    print(f"Sidecar written: {os.path.join(args.export_dir, 'lattice_params.json')}")


if __name__ == "__main__":
    main()
