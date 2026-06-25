#!/usr/bin/env python
"""Run the digital-twin pipeline end-to-end for a single specimen.

Usage:
    python -m scripts.run_pipeline --morphometrics runs/ingest/morphometrics.json \
        --run-name digital_twin_001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
BIOMIMETIC_ROOT = THIS.parent.parent
if str(BIOMIMETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(BIOMIMETIC_ROOT))

from biomimetic_pipeline.logging_config import configure_logging  # noqa: E402
from biomimetic_pipeline.orchestration import pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose (DEBUG-level) logging.")
    ap.add_argument("--morphometrics", required=True, help="Canonical morphometrics.json path")
    ap.add_argument("--run-name", required=True, help="Run directory name under runs/")
    ap.add_argument(
        "--objective",
        default=None,
        help="Objective name (crack_deflection|toughness|stiffness_target|biomimicry_score|composite). "
        "If omitted, runs a single FEA evaluation at --compress-disp-mm (no strain solver).",
    )
    ap.add_argument(
        "--model-type",
        default="continuous_twist",
        choices=["continuous_twist", "measured_profile", "hierarchical", "radially_graded"],
        help="CAD generator variant. New models consume extra morphometric fields.",
    )
    ap.add_argument("--material-E-mpa", type=float, default=pipeline.DEFAULT_SLA_MATERIAL_E_MPA)
    ap.add_argument("--material-nu", type=float, default=pipeline.DEFAULT_SLA_MATERIAL_NU)
    ap.add_argument("--compress-disp-mm", type=float, default=pipeline.DEFAULT_COMPRESS_DISP_MM)
    ap.add_argument("--mesh-size-mm", type=float, default=pipeline.DEFAULT_MESH_SIZE_MM)
    ap.add_argument("--junction-refinement", type=float, default=1.0)
    ap.add_argument(
        "--skip-probe", action="store_true", help="Skip conda env probe (use with care)"
    )
    ap.add_argument(
        "--allow-broken-cad",
        action="store_true",
        help=(
            "Continue past the post-CAD integrity check even if it fails. "
            "By default the pipeline aborts before mesh + FEA if any of "
            "bridge presence / rod count / watertight fail; this flag is "
            "the diagnostic escape hatch for runs that intentionally test "
            "broken geometry."
        ),
    )
    args = ap.parse_args()
    # Configure the library's logging stream once, at the application entry point
    # (libraries log; the app decides where/how loud). --verbose -> DEBUG.
    configure_logging(verbose=args.verbose)

    result = pipeline.run_pipeline(
        morphometrics_path=Path(args.morphometrics),
        run_name=args.run_name,
        objective_name=args.objective,
        model_type=args.model_type,
        material_E_mpa=args.material_E_mpa,
        material_nu=args.material_nu,
        compress_disp_mm=args.compress_disp_mm,
        mesh_size_mm=args.mesh_size_mm,
        junction_refinement_factor=args.junction_refinement,
        skip_probe=args.skip_probe,
        allow_broken_cad=args.allow_broken_cad,
    )
    print(f"Run complete: {result.run_dir}")
    print(f"Metrics: {result.metrics_path}")


if __name__ == "__main__":
    main()
