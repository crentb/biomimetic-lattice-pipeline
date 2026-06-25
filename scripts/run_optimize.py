#!/usr/bin/env python
"""Optuna-driven Bayesian optimization over CAD parameters.

Usage:
    python -m scripts.run_optimize --morphometrics runs/ingest/morphometrics.json \
        --run-name opt_crack_deflection \
        --objective crack_deflection --n-trials 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
BIOMIMETIC_ROOT = THIS.parent.parent
if str(BIOMIMETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(BIOMIMETIC_ROOT))

from biomimetic_pipeline.orchestration import optimize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--morphometrics", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--objective", required=True)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--material-E-mpa", type=float, default=None)
    ap.add_argument("--mesh-size-mm", type=float, default=None)
    ap.add_argument("--junction-refinement", type=float, default=1.0)
    ap.add_argument("--skip-probe", action="store_true")
    args = ap.parse_args()

    kwargs = {}
    if args.material_E_mpa is not None:
        kwargs["material_E_mpa"] = args.material_E_mpa
    if args.mesh_size_mm is not None:
        kwargs["mesh_size_mm"] = args.mesh_size_mm

    result = optimize.run_optimize(
        morphometrics_path=Path(args.morphometrics),
        run_name=args.run_name,
        objective_name=args.objective,
        n_trials=args.n_trials,
        seed=args.seed,
        junction_refinement_factor=args.junction_refinement,
        skip_probe=args.skip_probe,
        **kwargs,
    )
    print(f"Optimization complete. Best value: {result.best_value}")
    print(f"Best params: {result.best_params}")
    print(f"CSV: {result.csv_path}")


if __name__ == "__main__":
    main()
