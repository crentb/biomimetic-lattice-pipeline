"""End-to-end pipeline: one specimen, ingest → map → CAD → mesh → FEA → metrics → report.

Phase 2 wires in:
  - strain solver (bisect COMPRESS_DISP_MM to hit a target VM stress),
  - crack-deflection streamline tortuosity,
  - biomimicry score,
  - pluggable objective via YAML.

If no objective is provided, we fall back to a single FEA run with
compress_disp_mm=DEFAULT_COMPRESS_DISP_MM (Phase-1 behavior) so the pipeline
still works for quick smoke checks.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from biomimetic_pipeline.fea import fea_runner, metrics_runner, strain_solver
from biomimetic_pipeline.generators import (
    cad_integrity,
    cad_runner,
    digital_twin_sdf_runner,
    mesh_runner,
)
from biomimetic_pipeline.mapping import feature_to_cad
from biomimetic_pipeline.metrics import biomimicry_score, crack_deflection
from biomimetic_pipeline.objectives import registry as objective_registry
from biomimetic_pipeline.orchestration.run_context import RunContext, now_iso

logger = logging.getLogger(__name__)

# Young's modulus of the FEA solid phase, in MPa. This is the *realization*
# material -- a standard 405 nm SLA rigid photopolymer (E ~ 2.5-3.5 GPa) --
# and NOT the biological material. Lion enamel measures ~85 GPa by
# nanoindentation; the pipeline claim is architectural (the lattice geometry,
# not its bulk modulus, is what transfers from measurement), so every FEA run
# uses the modulus of the material the lattice is actually printed in. The
# stock compression_test.py default is 85000 MPa (hydroxyapatite); this
# constant overrides it to the SLA-resin value for all biomimetic_pipeline runs.
DEFAULT_SLA_MATERIAL_E_MPA = 3000.0
DEFAULT_SLA_MATERIAL_NU = 0.40
DEFAULT_COMPRESS_DISP_MM = 1.0
DEFAULT_MESH_SIZE_MM = 0.5


@dataclass
class PipelineResult:
    run_dir: Path
    cad_params_path: Path
    step_path: Path
    mesh_path: Path
    fea_dir: Path
    metrics_path: Path
    metrics: Dict[str, Any]
    score: Optional[float] = None


def run_pipeline(
    morphometrics_path: Path,
    run_name: str,
    objective_name: Optional[str] = None,
    model_type: str = "continuous_twist",
    material_E_mpa: float = DEFAULT_SLA_MATERIAL_E_MPA,
    material_nu: float = DEFAULT_SLA_MATERIAL_NU,
    compress_disp_mm: float = DEFAULT_COMPRESS_DISP_MM,
    mesh_size_mm: float = DEFAULT_MESH_SIZE_MM,
    junction_refinement_factor: float = 1.0,
    skip_probe: bool = False,
    extra_overrides: Optional[Dict[str, Any]] = None,
    # NEW: forwarded to the post-CAD integrity check below. False (the
    # default) aborts the pipeline if any check fails -- saves hours of
    # FEA compute on silently-broken geometry. True is the diagnostic
    # escape hatch (also exposed as --allow-broken-cad on the CLIs).
    allow_broken_cad: bool = False,
) -> PipelineResult:
    ctx = RunContext.create(run_name)
    if not skip_probe:
        ctx.probe_envs()

    log = _open_log(ctx.run_dir / "pipeline.log")
    log(f"[{now_iso()}] pipeline start run={run_name} objective={objective_name}")

    # --- 1. Ingest morphometrics, map to CAD parameters -------------------
    morphometrics_path = Path(morphometrics_path)
    morphometrics = json.loads(morphometrics_path.read_text())

    # The digital_twin model-type bypasses the feature-to-CAD mapping
    # entirely: the twin is built directly from PIV rod trajectories, so
    # there are no parametric CAD knobs to derive. Stage 2 (CAD + mesh)
    # follows a parallel SDF + voxel-hex path -- see the branch below.
    if model_type == "digital_twin":
        cad_params = {
            "model_type": "digital_twin",
            "source": "output_piv (PIV-tracked rod trajectories)",
            "note": (
                "digital_twin bypasses feature-to-CAD; geometry inherited "
                "directly from the 2,433-track PIV parquet via SDF + plating "
                "rather than a parametric mapping"
            ),
        }
        cad_params_path = ctx.run_dir / "cad_params.json"
        cad_params_path.write_text(json.dumps(cad_params, indent=2))
        log(f"[{now_iso()}] cad_params -> {cad_params_path} (digital_twin stub)")
    else:
        cad_params = feature_to_cad.map_morphometrics(
            morphometrics,
            morphometrics_source=morphometrics_path,
            extra_overrides=extra_overrides,
        )
        feature_to_cad.validate(cad_params)
        cad_params_path = feature_to_cad.save(cad_params, ctx.run_dir / "cad_params.json")
        log(f"[{now_iso()}] cad_params -> {cad_params_path}")

    # --- 2. CAD + mesh -----------------------------------------------------
    if model_type == "digital_twin":
        # Digital-twin path: SDF capsule-union from PIV centerlines + plate
        # slabs, then voxel-hex meshing direct from the plated occupancy
        # (gmsh STL-to-tet fails on the organic merged-rod surface; see
        # generators/mesh_runner.py docstring for the long explanation).
        twin_result = digital_twin_sdf_runner.run(
            export_dir=ctx.run_dir,
            morphometrics_path=morphometrics_path,
            plate=True,
        )
        mesh_result = mesh_runner.run_for_digital_twin(
            occupancy_npy=twin_result.plated_occupancy_path,
            mesh_dir=ctx.run_dir / "mesh",
            voxel_size_um=twin_result.voxel_size_um,
            downsample_factor=3,
        )
        # Twin specimen height in mm: the plated grid is in micrometres but
        # sfepy reads coords without unit interpretation, so we keep the
        # numerical "mm" value equal to the micrometre count. All downstream
        # stress / strain / modulus metrics are dimensionally consistent
        # under that convention -- see the manuscript Methods section.
        specimen_height_mm = float(twin_result.plated_specimen_height_um or 0.0)
        bridge_elevs: list = []
        log(
            f"[{now_iso()}] digital_twin mesh {mesh_result.mesh_path} "
            f"({mesh_result.element_type}, height={specimen_height_mm:.2f})"
        )
        # Stub a cad_result-shape object so the downstream sidecar-copy step
        # has something to point at; we just hand it the SDF provenance JSON.
        from dataclasses import dataclass as _dc

        @_dc
        class _TwinCadStub:
            step_path: Path
            stl_path: Path
            sidecar_path: Path

        cad_result = _TwinCadStub(
            step_path=twin_result.stl_path,  # no STEP; STL serves as the geometry artifact
            stl_path=twin_result.stl_path,
            sidecar_path=twin_result.provenance_path,
        )
    else:
        cad_result = cad_runner.run(
            cad_params, export_dir=ctx.run_dir / "cad", model_type=model_type
        )
        log(f"[{now_iso()}] cad model_type={model_type} step={cad_result.step_path}")

        sidecar = json.loads(cad_result.sidecar_path.read_text())
        specimen_height_mm = float(
            sidecar.get("specimen_height", cad_params.get("ENAMEL_THICKNESS", 20.0))
        )
        bridge_elevs = [float(z) for z in sidecar.get("bridge_elevations", [])]

        # --- 2a. CAD integrity check (post-CAD, pre-mesh) ----------------
        # Slice the emitted STL at every requested BRIDGE_Z_OFFSETS to
        # verify bridges actually fused, count rod cross-sections at the
        # rod-only baseline to verify no silent merges/drops, and confirm
        # the STL is watertight. By default the pipeline ABORTS before
        # mesh + FEA if any of the three fail (the OCCT silent-drop bug
        # at N=8 being the canonical example -- see
        # mapping/bridge_mappers.OCCT_TANGENT_JITTER_MM). Override via
        # allow_broken_cad=True for diagnostic / forced runs.
        integrity_report = cad_integrity.verify_cad_integrity(
            stl_path=cad_result.stl_path,
            bridge_z_offsets=bridge_elevs,
            n_rings=int(cad_params.get("N_RINGS", 5)),
        )
        integrity_path = ctx.run_dir / "cad_integrity_report.json"
        integrity_path.write_text(json.dumps(integrity_report.to_dict(), indent=2))
        log(
            f"[{now_iso()}] cad_integrity passed={integrity_report.passed} "
            f"bridges={integrity_report.bridge.get('n_present', 0)}/"
            f"{integrity_report.bridge.get('n_expected', 0)} "
            f"rods={integrity_report.rods.get('n_present', 0)}/"
            f"{integrity_report.rods.get('n_expected', 0)} "
            f"watertight={integrity_report.watertight} -> {integrity_path}"
        )
        if not integrity_report.passed:
            if allow_broken_cad:
                log(
                    f"[{now_iso()}] cad_integrity FAILED but allow_broken_cad=True; "
                    f"continuing. Failures: {integrity_report.failures}"
                )
            else:
                cad_integrity.raise_if_failed(integrity_report)

        mesh_result = mesh_runner.run(
            step_path=cad_result.step_path,
            mesh_dir=ctx.run_dir / "mesh",
            mesh_size=mesh_size_mm,
            junction_refinement_factor=junction_refinement_factor,
            bridge_elevations=bridge_elevs,
        )
        log(f"[{now_iso()}] mesh {mesh_result.mesh_path}")

    # --- 3. FEA: either strain-solve to target VM or single fixed run -----
    if objective_name:
        objective = objective_registry.load_builtin(objective_name)
    else:
        objective = None

    fea_root = ctx.run_dir / "fea"
    # element_type is "hex" for the voxel-hex twin path and "tet" everywhere
    # else; the strain solver / fea_runner threads it through to the runtime
    # problem-def derivation (see fea/fea_runner.py).
    element_type = getattr(mesh_result, "element_type", "tet")
    if objective and objective.stress_target:
        st = objective.stress_target
        solve_result = strain_solver.solve_to_target_stress(
            mesh_source=mesh_result.mesh_path,
            iter_root_dir=fea_root,
            specimen_height_mm=specimen_height_mm,
            material_E_mpa=material_E_mpa,
            material_nu=material_nu,
            target_mpa=st.vm_target_mpa,
            target_field=st.field,
            seed_disp_mm=st.seed_disp_mm,
            tolerance_pct=st.solver_tolerance_pct,
            max_iters=st.max_iters,
            element_type=element_type,
        )
        fea_result = solve_result.fea_result
        log(
            f"[{now_iso()}] strain_solve accepted={solve_result.accepted} "
            f"disp={solve_result.final_iter.disp_mm:.4f}mm "
            f"vm={solve_result.final_iter.vm_value_mpa:.2f}MPa "
            f"strain={solve_result.critical_strain:.5f}"
        )
        strain_metrics = {
            f"critical_strain_at_{int(st.vm_target_mpa)}MPa": solve_result.critical_strain,
            f"critical_disp_mm_at_{int(st.vm_target_mpa)}MPa": solve_result.final_iter.disp_mm,
            "strain_solve_accepted": bool(solve_result.accepted),
            "strain_solve_iters": len(solve_result.iterations),
        }
    else:
        fea_iter_dir = fea_root / "iter_1"
        fea_result = fea_runner.run_one_iteration(
            mesh_source=mesh_result.mesh_path,
            iter_dir=fea_iter_dir,
            material_E_mpa=material_E_mpa,
            material_nu=material_nu,
            compress_disp_mm=compress_disp_mm,
            element_type=element_type,
        )
        strain_metrics = {}
        log(
            f"[{now_iso()}] fea disp={compress_disp_mm} mm vm={fea_result.avg_von_mises_mpa:.2f} MPa"
        )

    # --- 4. Copy final iteration to stable location, run extract_metrics --
    final_fea_dir = fea_root / "final"
    fea_runner.copy_fea_outputs(fea_result.iter_dir, final_fea_dir)
    shutil.copy2(cad_result.sidecar_path, final_fea_dir / "lattice_params.json")

    metrics_result = metrics_runner.run(
        results_dir=final_fea_dir, sidecar_path=cad_result.sidecar_path
    )
    log(f"[{now_iso()}] metrics -> {metrics_result.metrics_path}")

    # --- 5. Crack deflection + biomimicry ---------------------------------
    cd_result = crack_deflection.compute(final_fea_dir / "element_results_compression.csv")
    cd_path = crack_deflection.save(cd_result, final_fea_dir / "crack_deflection.json")
    log(
        f"[{now_iso()}] crack_deflection mean={cd_result.tortuosity_mean:.4f} "
        f"p90={cd_result.tortuosity_p90:.4f} n={cd_result.n_streamlines}"
    )

    # --- 6. Aggregate metrics and score the objective ---------------------
    augmented: Dict[str, Any] = dict(metrics_result.metrics)
    augmented.update(cd_result.as_dict())
    augmented.update(strain_metrics)
    augmented["avg_von_mises_MPa"] = fea_result.avg_von_mises_mpa

    # Biomimicry: the digital twin IS the biological reference (every
    # architectural feature inherited directly from PIV trajectories), so
    # the score is unity by construction with no per-feature deviations.
    # Skip the biomimicry_score.compute() call, which expects parametric
    # cad_params with feature-keyed values.
    if model_type == "digital_twin":
        augmented["biomimicry_score"] = 1.0
        augmented["biomimicry_n_features"] = 0
        augmented["biomimicry_feature_pairs"] = []
        augmented["biomimicry_note"] = "digital_twin inherits all measured features by construction"
    else:
        bm = biomimicry_score.compute(morphometrics, cad_params, fea_metrics=augmented)
        augmented.update(bm.as_dict())

    augmented["_pipeline"] = {
        "compress_disp_mm": compress_disp_mm,
        "material_E_MPa": material_E_mpa,
        "material_nu": material_nu,
        "mesh_size_mm": mesh_size_mm,
        "run_name": run_name,
        "run_timestamp_iso": ctx.created_at_iso,
        "objective_name": objective_name,
        "model_type": model_type,
    }

    score_val: Optional[float] = None
    if objective is not None:
        score_val = objective.score(augmented)
        augmented["_objective_score"] = score_val
        (ctx.run_dir / "score.json").write_text(
            json.dumps(
                {
                    "objective": objective_name,
                    "direction": objective.direction,
                    "score": score_val,
                },
                indent=2,
            )
        )
        log(
            f"[{now_iso()}] objective={objective_name} score={score_val:.6g} "
            f"direction={objective.direction}"
        )

    metrics_path = ctx.run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(augmented, indent=2))

    try:
        from biomimetic_pipeline.reporting import latex_report

        latex_report.build(ctx.run_dir)
        log(f"[{now_iso()}] report -> {ctx.run_dir / 'report'}")
    except Exception as exc:
        log(f"[{now_iso()}] report build failed (non-fatal): {exc}")

    log(f"[{now_iso()}] pipeline done")
    return PipelineResult(
        run_dir=ctx.run_dir,
        cad_params_path=cad_params_path,
        step_path=cad_result.step_path,
        mesh_path=mesh_result.mesh_path,
        fea_dir=final_fea_dir,
        metrics_path=metrics_path,
        metrics=augmented,
        score=score_val,
    )


def _open_log(path: Path):
    fh = open(path, "a")

    def _log(msg: str) -> None:
        fh.write(msg + "\n")
        fh.flush()
        logger.info(msg)

    return _log
