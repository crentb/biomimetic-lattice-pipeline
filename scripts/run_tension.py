#!/usr/bin/env python
"""Item A -- tension load case for the bioinspired parametric lattice (Fig 4B).

Re-solves the EXISTING live_001 lattice mesh under tension to the same
200 MPa volume-averaged von-Mises target used for the compression run in
Section 2.4 of the manuscript. No re-CAD and no re-mesh: only the load case
changes, so this is a single strain-solve (a few sfepy iterations) on the
mesh already produced for the compression analysis.

The result populates Fig 4 Panel B and answers the Section 2.4 tension-vs-
compression TODO.

Usage (run from the biomimetic_pipeline directory):
    python -m scripts.run_tension \
        --mesh runs/live_001_digital_twin/mesh/compound_enamel_lattice.msh \
        --out  runs/live_001_tension \
        --specimen-height-mm 21.9
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the biomimetic_pipeline package importable when run as a script.
THIS = Path(__file__).resolve()
BIOMIMETIC_ROOT = THIS.parent.parent
if str(BIOMIMETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(BIOMIMETIC_ROOT))

from biomimetic_pipeline.fea.fea_runner import copy_fea_outputs  # noqa: E402
from biomimetic_pipeline.fea.strain_solver import solve_to_target_stress  # noqa: E402

# SLA-resin material defaults. The single source of truth is
# orchestration.pipeline.DEFAULT_SLA_MATERIAL_E_MPA / _NU (3000 MPa, 0.40);
# hard-coded here as CLI defaults so this driver does not have to import the
# full pipeline module (and its CAD/FEA dependencies) just for two constants.
DEFAULT_MATERIAL_E_MPA = 3000.0
DEFAULT_MATERIAL_NU = 0.40


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tension FEA strain-solve on an existing lattice mesh (Fig 4B)."
    )
    ap.add_argument("--mesh", required=True, help="Path to the existing tetrahedral .msh file.")
    ap.add_argument(
        "--out", required=True, help="Run directory to create/use for the tension solve."
    )
    ap.add_argument(
        "--specimen-height-mm",
        type=float,
        required=True,
        help="Specimen z-extent in mm; critical_strain = disp / height.",
    )
    ap.add_argument(
        "--target-mpa",
        type=float,
        default=200.0,
        help="Volume-averaged von-Mises stress target in MPa (default 200).",
    )
    ap.add_argument(
        "--material-E-mpa",
        type=float,
        default=DEFAULT_MATERIAL_E_MPA,
        help="Solid-phase Young's modulus in MPa (default 3000, SLA resin).",
    )
    ap.add_argument(
        "--material-nu",
        type=float,
        default=DEFAULT_MATERIAL_NU,
        help="Solid-phase Poisson ratio (default 0.40).",
    )
    args = ap.parse_args()

    fea_dir = Path(args.out).resolve() / "fea"
    fea_dir.mkdir(parents=True, exist_ok=True)

    # The strain solver bisects the applied displacement until the volume-
    # averaged von-Mises stress reaches the target. load_mode="tension" makes
    # the FEA runner derive a tension problem-def from the stock script.
    result = solve_to_target_stress(
        mesh_source=Path(args.mesh).resolve(),
        iter_root_dir=fea_dir,
        specimen_height_mm=args.specimen_height_mm,
        material_E_mpa=args.material_E_mpa,
        material_nu=args.material_nu,
        target_mpa=args.target_mpa,
        load_mode="tension",
    )

    # Promote the accepted iteration's result files to fea/final/ so the
    # figure scripts can read them at a stable path.
    copy_fea_outputs(result.fea_result.iter_dir, fea_dir / "final", load_mode="tension")

    status = "CONVERGED" if result.accepted else "did NOT converge"
    print(f"[run_tension] tension solve {status}")
    print(f"[run_tension]   critical_disp_mm  = {result.final_iter.disp_mm:.4f}")
    print(f"[run_tension]   critical_strain   = {result.critical_strain:.4f}")
    print(f"[run_tension]   avg_von_mises_MPa = {result.final_iter.vm_value_mpa:.2f}")
    print(f"[run_tension]   outputs -> {fea_dir / 'final'}")


if __name__ == "__main__":
    main()
