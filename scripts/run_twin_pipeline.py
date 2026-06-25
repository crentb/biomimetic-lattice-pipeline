"""End-to-end driver for the digital_twin model-type.

Invokes orchestration.pipeline.run_pipeline with model_type="digital_twin",
routing the run through the SDF + plating + voxel-hex + hex-FEA path that
was wired in on 2026-05-23.

Usage from the biomimetic_pipeline/ root:

    conda run -n base python scripts/run_twin_pipeline.py \\
        --run-name twin_pipeline_smoke \\
        --morphometrics runs/live_001/morphometrics.json \\
        --objective crack_deflection

Why ``base``: the digital-twin path needs scikit-image (marching cubes) +
pyvista + trimesh + scipy + meshio for the SDF/plating/voxel-hex stages.
``base`` has all of these; ``sfepy_env`` and ``cad_env`` do not have skimage.
The FEA stage further down the pipeline is subprocessed to ``sfepy_env`` via
``conda run`` regardless (see ``fea/fea_runner.py``), so the invoking env
only needs the early-stage deps.

Background and gotchas:
    - This run will be SLOW: the direct sfepy solver on the plated 134k-hex
      twin takes ~2 h of LU factorization on a Mac M-series laptop. Each
      strain-solver iteration calls sfepy from scratch, so plan accordingly
      (the bisection usually converges in 2 iters for linear elastic).
    - The voxel-hex mesh coordinates are in micrometres, but sfepy reads
      them as the material's coordinate unit. All stress/strain/modulus
      metrics remain dimensionally consistent (they are dimensionless
      ratios or pressure ratios). Absolute force values are not physical;
      the strain solver and downstream metrics rely on ratio quantities
      (E_eff = avg_sigma_zz / strain, SCF = VM_P99 / VM_mean, etc.) that
      are invariant under length-unit scaling.
    - The biomimicry score is forced to 1.0 for digital_twin by the
      pipeline (the twin IS the biological reference); the feature-to-CAD
      mapping is bypassed entirely.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make biomimetic_pipeline/ importable when invoked from anywhere.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biomimetic_pipeline.orchestration.pipeline import run_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="Run the digital-twin pipeline end-to-end.")
    p.add_argument(
        "--run-name",
        required=True,
        help="Subdirectory name under runs/ for this run.",
    )
    p.add_argument(
        "--morphometrics",
        type=Path,
        required=True,
        help="Path to morphometrics.json (used for run-context provenance only "
        "-- the twin geometry comes from the canonical PIV parquet "
        "configured in digital_twin_sdf_runner.DEFAULT_PIV_PARQUET).",
    )
    p.add_argument(
        "--objective",
        default=None,
        help="Built-in objective name (e.g. crack_deflection). If omitted, "
        "a single fixed-disp FEA iteration is run instead of a strain "
        "solve -- useful only for smoke testing.",
    )
    args = p.parse_args()

    result = run_pipeline(
        morphometrics_path=args.morphometrics,
        run_name=args.run_name,
        objective_name=args.objective,
        model_type="digital_twin",
    )
    print(f"\n[twin] run_dir: {result.run_dir}")
    print(f"[twin] metrics: {result.metrics_path}")
    if result.score is not None:
        print(f"[twin] objective score: {result.score}")


if __name__ == "__main__":
    main()
