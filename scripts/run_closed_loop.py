#!/usr/bin/env python
"""Closed-loop biomimicry: find CAD configs that maximize an objective,
then reverse-map them to measured morphometric ranges to look for in specimens.

Usage:
    python -m scripts.run_closed_loop --run-name closed_cd_001 \
        --objective crack_deflection --n-trials 30 --top-k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
BIOMIMETIC_ROOT = THIS.parent.parent
if str(BIOMIMETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(BIOMIMETIC_ROOT))

from biomimetic_pipeline.orchestration import closed_loop  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--objective", default="crack_deflection")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--morphometrics",
        default=None,
        help="Optional seed morphometrics; synthetic is generated if omitted",
    )
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

    result = closed_loop.run_closed_loop(
        run_name=args.run_name,
        objective_name=args.objective,
        n_trials=args.n_trials,
        top_k=args.top_k,
        seed=args.seed,
        morphometrics_path=Path(args.morphometrics) if args.morphometrics else None,
        junction_refinement_factor=args.junction_refinement,
        skip_probe=args.skip_probe,
        **kwargs,
    )
    print("Closed-loop run complete.")
    print(f"Targets: {result.targets_path}")
    print(f"CSV:     {result.csv_path}")
    print(f"Top-{len(result.top_k_designs)} designs stored in biomimicry_targets.json")


if __name__ == "__main__":
    main()
