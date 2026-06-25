"""Radially-graded lattice generator.

Overrides ROD_DIAMETER per ring using the `per_ring_diameter` dict populated
by `mapping/rod_mappers.py` when the measured rod-diameter profile is
strongly non-monotone.

Implementation note: lattice_cad.py uses a single `rod_diameter` for all
rings. We achieve per-ring grading by generating the lattice ring-by-ring,
each time with a scalar ROD_DIAMETER override, and fusing the resulting
solids. That requires either (a) re-running generate_lattice N times with
different N_RINGS=1 hacks (messy) or (b) a minimal shim that calls
generate_lattice once but filters rods by ring. For Phase 4 we take the
simpler approach: generate with the mean ROD_DIAMETER, and emit a clearly-
labeled warning in the sidecar. Full per-ring variation will land in Phase 5
alongside stochastic_lattice when we drop the hexagonal grid generator.
"""

from __future__ import annotations

import argparse
import json
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


def _coerce_ring_rotation(rr):
    if rr is None:
        return None
    return {int(k): float(v) for k, v in rr.items()}


def _coerce_per_ring_diameter(prd):
    if prd is None:
        return None
    return {int(k): float(v) for k, v in prd.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    p.add_argument("--export-dir", required=True)
    args = p.parse_args()

    with open(args.params) as fh:
        user_params = json.load(fh)

    per_ring = _coerce_per_ring_diameter(user_params.get("per_ring_diameter"))
    if per_ring:
        mean_diam = sum(per_ring.values()) / len(per_ring)
        effective_rod_diameter = float(mean_diam)
        user_params["ROD_DIAMETER"] = effective_rod_diameter
    else:
        effective_rod_diameter = float(user_params.get("ROD_DIAMETER", 2.0))

    if "RING_ROTATION" in user_params:
        user_params["RING_ROTATION"] = _coerce_ring_rotation(user_params["RING_ROTATION"])

    known_keys = set(lattice_cad.DEFAULTS.keys())
    stock_params = {k: v for k, v in user_params.items() if k in known_keys}
    stock_params["EXPORT_DIR"] = args.export_dir

    if stock_params.get("TWIST_TYPE") == "measured":
        stock_params["TWIST_TYPE"] = "sigmoid"

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
        "model_type": "radially_graded",
        "effective_rod_diameter_mm": effective_rod_diameter,
        "per_ring_diameter": per_ring,
        "warning": (
            "Phase 4 uses the mean of per_ring_diameter; true per-ring "
            "variation requires dropping the hexagonal grid (Phase 5)."
        ),
    }
    with open(os.path.join(args.export_dir, "lattice_params.json"), "w") as fh:
        json.dump(sidecar, fh, indent=2, default=str)
    print(f"Sidecar written: {os.path.join(args.export_dir, 'lattice_params.json')}")


if __name__ == "__main__":
    main()
