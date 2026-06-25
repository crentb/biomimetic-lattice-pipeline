#!/usr/bin/env python
"""Item B -- decussation-amplitude sweep for Fig 5C.

Decussation in the continuous-twist lattice is carried by the per-ring
rotation angles (the RING_ROTATION dict), not by any single scalar knob.
This driver sweeps a SCALE FACTOR applied to the measured RING_ROTATION:

    factor ~0.05 -> near-straight rods, effectively NO decussation -- the
                    control point. An exact 0.0 is avoided because perfectly
                    straight rods degenerate the OpenCascade pipe-shell sweep
                    in the stock lattice_cad generator.
    factor 1.0   -> the measured biological lattice, unchanged
    factor f     -> every per-ring rotation multiplied by f

Each factor runs through the full pipeline at the SLA-resin modulus
(3000 MPa, the run_pipeline default) and the 200 MPa von-Mises strain-solve
target, so Fig 5C is directly comparable to the Fig 4 / Section 2.4 lattice.
Per-trial results are appended to runs/<run_name>/sweep_log.csv.

Usage (from the biomimetic_pipeline directory):
    python -m scripts.run_decussation_sweep \
        --morphometrics runs/live_001/morphometrics.json \
        --baseline-cad-params runs/live_001_digital_twin/cad_params.json \
        --run-name sweep_decussation_resin \
        --factors 0.0 0.5 1.0 1.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make the biomimetic_pipeline package importable when run as a script.
THIS = Path(__file__).resolve()
BIOMIMETIC_ROOT = THIS.parent.parent
if str(BIOMIMETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(BIOMIMETIC_ROOT))

from biomimetic_pipeline.orchestration import pipeline, sweep_log  # noqa: E402


def _scaled_ring_rotation(baseline: dict, factor: float) -> dict:
    """Return the per-ring rotation dict with every angle multiplied by ``factor``.

    Keys are preserved as-is (the CAD driver coerces them to int downstream).
    A factor of 0.0 collapses the lattice to straight, non-decussated rods.
    """
    return {key: float(angle) * factor for key, angle in baseline.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Decussation-amplitude sweep: scales the measured RING_ROTATION dict."
    )
    ap.add_argument(
        "--morphometrics",
        required=True,
        help="morphometrics.json driving the feature-to-CAD mapper.",
    )
    ap.add_argument(
        "--baseline-cad-params",
        required=True,
        help="cad_params.json whose RING_ROTATION is the factor=1.0 baseline.",
    )
    ap.add_argument(
        "--run-name", required=True, help="Sweep run directory name, created under runs/."
    )
    ap.add_argument(
        "--factors",
        nargs="+",
        type=float,
        default=[0.05, 0.5, 1.0, 1.5],
        help="Scale factors applied to the measured RING_ROTATION dict. "
        "The control point is ~0.05, not 0.0: an exact zero "
        "degenerates the CAD pipe-shell sweep (straight rods).",
    )
    ap.add_argument(
        "--objective",
        default="crack_deflection",
        help="Objective name; must carry a stress_target so the "
        "pipeline strain-solves to a von-Mises target.",
    )
    ap.add_argument(
        "--allow-broken-cad",
        action="store_true",
        help=(
            "Continue past the post-CAD integrity check on each decussation "
            "trial even if it fails. Default: any trial with missing bridges "
            "/ wrong rod count / non-watertight STL aborts the sweep before "
            "mesh."
        ),
    )
    args = ap.parse_args()

    baseline_params = json.loads(Path(args.baseline_cad_params).read_text())
    if "RING_ROTATION" not in baseline_params:
        raise KeyError(f"'RING_ROTATION' not found in {args.baseline_cad_params}")
    baseline_ring_rotation = baseline_params["RING_ROTATION"]

    root = BIOMIMETIC_ROOT / "runs" / args.run_name
    root.mkdir(parents=True, exist_ok=False)
    csv_path = root / "sweep_log.csv"
    sweep_log.ensure_header(csv_path)

    for i, factor in enumerate(args.factors):
        scaled = _scaled_ring_rotation(baseline_ring_rotation, factor)
        factor_tag = f"{factor:.4g}".replace(".", "p").replace("-", "m")
        trial_name = f"{args.run_name}/trial_{i:03d}_decussation_{factor_tag}"

        t0 = time.time()
        # extra_overrides replaces RING_ROTATION in the mapped cad_params, so
        # the lattice is identical to live_001 except for the decussation
        # amplitude. The material modulus defaults to the 3000 MPa SLA resin.
        result = pipeline.run_pipeline(
            morphometrics_path=Path(args.morphometrics),
            run_name=trial_name,
            objective_name=args.objective,
            model_type="continuous_twist",
            extra_overrides={"RING_ROTATION": scaled},
            skip_probe=(i > 0),  # probe the conda envs only on the first trial
            allow_broken_cad=args.allow_broken_cad,
        )
        elapsed = time.time() - t0

        cad_params = json.loads(Path(result.cad_params_path).read_text())
        sweep_log.append(
            csv_path,
            params=cad_params,
            metrics=result.metrics,
            run_name=trial_name,
            objective=args.objective,
            label=f"decussation:factor={factor}",
            total_time_s=elapsed,
        )
        print(
            f"[decussation_sweep] factor={factor}: done in {elapsed:.0f}s " f"-> {result.run_dir}"
        )

    print(f"[decussation_sweep] complete. CSV: {csv_path}")


if __name__ == "__main__":
    main()
