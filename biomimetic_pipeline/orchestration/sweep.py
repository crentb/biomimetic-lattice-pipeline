"""Parametric sweep driver: repeatedly run the pipeline with one (or more)
CAD parameters overridden.

Each trial calls `pipeline.run_pipeline(..., extra_overrides={param: value})`
and appends a row to `runs/<run_name>/sweep_log.csv`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from biomimetic_pipeline.orchestration import pipeline, sweep_log
from biomimetic_pipeline.orchestration.run_context import BIOMIMETIC_ROOT


@dataclass
class SweepResult:
    run_root: Path
    csv_path: Path
    trial_dirs: List[Path]
    trial_metrics: List[Dict[str, Any]]


def run_sweep(
    morphometrics_path: Path,
    run_name: str,
    param: str,
    values: Sequence[Any],
    objective_name: Optional[str] = None,
    material_E_mpa: float = pipeline.DEFAULT_SLA_MATERIAL_E_MPA,
    material_nu: float = pipeline.DEFAULT_SLA_MATERIAL_NU,
    mesh_size_mm: float = pipeline.DEFAULT_MESH_SIZE_MM,
    junction_refinement_factor: float = 1.0,
    skip_probe: bool = False,
    # NEW: forwarded to pipeline.run_pipeline() per trial -- when False
    # (default), a trial that fails the post-CAD integrity check (bridges
    # missing / rod count mismatch / not watertight) aborts the entire
    # sweep so we never waste FEA compute on broken geometry. Set True
    # only for diagnostic sweeps that intentionally probe broken regimes.
    allow_broken_cad: bool = False,
) -> SweepResult:
    root = BIOMIMETIC_ROOT / "runs" / run_name
    root.mkdir(parents=True, exist_ok=False)
    csv_path = root / "sweep_log.csv"
    sweep_log.ensure_header(csv_path)

    trial_dirs: List[Path] = []
    trial_metrics: List[Dict[str, Any]] = []

    for i, value in enumerate(values):
        trial_name = f"{run_name}/trial_{i:03d}_{param}_{_fmt_value(value)}"
        t0 = time.time()
        # Each pipeline run creates its own run_dir, so we pass a slash-joined
        # run_name; RunContext.create treats this as a nested path.
        result = pipeline.run_pipeline(
            morphometrics_path=Path(morphometrics_path),
            run_name=trial_name,
            objective_name=objective_name,
            material_E_mpa=material_E_mpa,
            material_nu=material_nu,
            mesh_size_mm=mesh_size_mm,
            junction_refinement_factor=junction_refinement_factor,
            skip_probe=skip_probe or i > 0,  # only probe envs on first trial
            extra_overrides={param: value},
            allow_broken_cad=allow_broken_cad,
        )
        elapsed = time.time() - t0
        cad_params = json.loads(Path(result.cad_params_path).read_text())

        sweep_log.append(
            csv_path,
            params={
                **cad_params,
                "mesh_size": mesh_size_mm,
                "junction_refinement_factor": junction_refinement_factor,
            },
            metrics=result.metrics,
            run_name=trial_name,
            objective=objective_name,
            label=f"sweep:{param}={value}",
            total_time_s=elapsed,
        )
        trial_dirs.append(result.run_dir)
        trial_metrics.append(result.metrics)

    return SweepResult(
        run_root=root,
        csv_path=csv_path,
        trial_dirs=trial_dirs,
        trial_metrics=trial_metrics,
    )


def _fmt_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}".replace(".", "p").replace("-", "m")
    return str(v).replace(".", "p")
