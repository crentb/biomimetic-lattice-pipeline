"""Strain solver unit test using a synthetic FEA runner.

Linear-elasticity assumption: stress ∝ strain ∝ displacement at fixed
geometry. So if we stub the FEA with VM = k * disp, the solver should converge
to the target displacement in a single iteration (it's a linear inversion).
We verify (a) convergence within 1 iteration for exactly-linear response,
(b) convergence within `max_iters` for slightly-non-linear response, and
(c) graceful non-convergence on severely non-linear response.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make biomimetic_pipeline importable.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biomimetic_pipeline.fea.fea_runner import FeaRunResult
from biomimetic_pipeline.fea.strain_solver import solve_to_target_stress


def _make_runner(vm_of_disp):
    """Return a (mesh, iter_dir, E, nu, disp) -> FeaRunResult stub."""

    def _run(
        mesh_source,
        iter_dir,
        material_E_mpa,
        material_nu,
        compress_disp_mm,
        load_mode="compression",
        element_type="tet",
    ):
        iter_dir = Path(iter_dir)
        iter_dir.mkdir(parents=True, exist_ok=True)
        vm = vm_of_disp(compress_disp_mm)
        return FeaRunResult(
            iter_dir=iter_dir,
            global_results_path=iter_dir / "global_results_compression.csv",
            element_results_path=iter_dir / "element_results_compression.csv",
            avg_von_mises_mpa=vm,
            avg_sigma_zz_mpa=-vm,
            force_N=-vm * 10.0,
            log_path=iter_dir / "fea_run.log",
        )

    return _run


def test_linear_converges_one_iter():
    # VM = 50 * disp -> target 200 MPa -> disp* = 4
    with tempfile.TemporaryDirectory() as tmp:
        res = solve_to_target_stress(
            mesh_source=Path(tmp) / "fake.msh",
            iter_root_dir=Path(tmp) / "iters",
            specimen_height_mm=20.0,
            material_E_mpa=3000.0,
            material_nu=0.40,
            target_mpa=200.0,
            seed_disp_mm=1.0,
            tolerance_pct=5.0,
            max_iters=5,
            runner=_make_runner(lambda d: 50.0 * d),
        )
        assert res.accepted, f"Should have converged: {res.as_dict()}"
        assert (
            len(res.iterations) <= 2
        ), f"Linear should hit target in ≤2 iters, got {len(res.iterations)}"
        assert abs(res.final_iter.disp_mm - 4.0) < 0.3
        assert abs(res.final_iter.vm_value_mpa - 200.0) <= 10.0
        # critical_strain = 4 mm / 20 mm = 0.2
        assert abs(res.critical_strain - 0.2) < 0.02


def test_mildly_nonlinear_converges():
    # VM = 50 * disp * (1 + 0.05 * disp)   (mild hardening)
    def nonlin(d):
        return 50.0 * d * (1.0 + 0.05 * d)

    with tempfile.TemporaryDirectory() as tmp:
        res = solve_to_target_stress(
            mesh_source=Path(tmp) / "fake.msh",
            iter_root_dir=Path(tmp) / "iters",
            specimen_height_mm=20.0,
            material_E_mpa=3000.0,
            material_nu=0.40,
            target_mpa=200.0,
            seed_disp_mm=1.0,
            tolerance_pct=5.0,
            max_iters=8,
            runner=_make_runner(nonlin),
        )
        assert res.accepted, f"Mild nonlinearity should converge: {res.as_dict()}"
        assert abs(res.final_iter.vm_value_mpa - 200.0) / 200.0 <= 0.05


def test_runs_iterations_file():
    with tempfile.TemporaryDirectory() as tmp:
        res = solve_to_target_stress(
            mesh_source=Path(tmp) / "fake.msh",
            iter_root_dir=Path(tmp) / "iters",
            specimen_height_mm=20.0,
            material_E_mpa=3000.0,
            material_nu=0.40,
            target_mpa=200.0,
            max_iters=3,
            runner=_make_runner(lambda d: 50.0 * d),
        )
        summary = Path(tmp) / "iters" / "strain_solve_summary.json"
        assert summary.exists(), "Summary JSON should be written"
        import json as _json

        data = _json.loads(summary.read_text())
        assert data["accepted"] is True
        assert data["iterations"][0]["iter"] == 1


if __name__ == "__main__":
    test_linear_converges_one_iter()
    test_mildly_nonlinear_converges()
    test_runs_iterations_file()
    print("All strain_solver tests passed.")
