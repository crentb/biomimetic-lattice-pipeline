#!/usr/bin/env python
"""Parametric sweep over one CAD parameter.

Usage:
    python -m scripts.run_sweep --morphometrics runs/ingest/morphometrics.json \
        --run-name sweep_bridge_diameter \
        --param BRIDGE_DIAMETER --values 0.5 0.75 1.0 1.25 1.5 \
        --objective crack_deflection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
BIOMIMETIC_ROOT = THIS.parent.parent
if str(BIOMIMETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(BIOMIMETIC_ROOT))

from biomimetic_pipeline.orchestration import sweep  # noqa: E402


def _coerce_values(param: str, values):
    int_params = {"N_BRIDGE_LAYERS", "N_RINGS", "Z_SAMPLES"}
    str_params = {"TWIST_TYPE"}
    if param in int_params:
        return [int(v) for v in values]
    if param in str_params:
        return list(values)
    return [float(v) for v in values]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--morphometrics", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument(
        "--param", required=True, help="CAD parameter key to sweep (e.g. BRIDGE_DIAMETER)"
    )
    ap.add_argument("--values", nargs="+", required=True)
    ap.add_argument("--objective", default=None)
    ap.add_argument("--material-E-mpa", type=float, default=None)
    ap.add_argument("--mesh-size-mm", type=float, default=None)
    ap.add_argument("--junction-refinement", type=float, default=1.0)
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument(
        "--allow-broken-cad",
        action="store_true",
        help=(
            "Continue past the post-CAD integrity check on each trial even "
            "if it fails. Default: any trial with missing bridges / wrong "
            "rod count / non-watertight STL aborts the sweep before mesh."
        ),
    )
    args = ap.parse_args()

    kwargs = {}
    if args.material_E_mpa is not None:
        kwargs["material_E_mpa"] = args.material_E_mpa
    if args.mesh_size_mm is not None:
        kwargs["mesh_size_mm"] = args.mesh_size_mm

    values = _coerce_values(args.param, args.values)

    result = sweep.run_sweep(
        morphometrics_path=Path(args.morphometrics),
        run_name=args.run_name,
        param=args.param,
        values=values,
        objective_name=args.objective,
        junction_refinement_factor=args.junction_refinement,
        skip_probe=args.skip_probe,
        allow_broken_cad=args.allow_broken_cad,
        **kwargs,
    )
    print(f"Sweep complete. CSV: {result.csv_path}")
    print(f"Trials run: {len(result.trial_dirs)}")


if __name__ == "__main__":
    main()
