"""Optuna-driven Bayesian optimization over CAD parameter space.

Reuses `pipeline.run_pipeline` as the evaluator; the objective's
`direction` flag decides whether Optuna minimizes or maximizes.

The default search space mirrors the existing optimize_lattice.py ranges so
the biomimetic pipeline can be compared against pure-design baselines.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from biomimetic_pipeline.objectives import registry as objective_registry
from biomimetic_pipeline.orchestration import pipeline, sweep_log
from biomimetic_pipeline.orchestration.run_context import BIOMIMETIC_ROOT

DEFAULT_SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    "ROD_DIAMETER": {"type": "float", "low": 1.5, "high": 3.0},
    "CENTER_SPACING": {"type": "float", "low": 2.0, "high": 3.5},
    "BRIDGE_DIAMETER": {"type": "float", "low": 0.5, "high": 1.5},
    "JUNCTION_SPHERE_FACTOR": {"type": "float", "low": 0.0, "high": 1.0},
    "ROD_TAPER_FACTOR": {"type": "float", "low": 0.0, "high": 0.4},
    "N_BRIDGE_LAYERS": {"type": "int", "low": 2, "high": 8},
    "TWIST_TYPE": {"type": "categorical", "choices": ["linear", "accelerating", "sigmoid"]},
}


@dataclass
class OptimizeResult:
    run_root: Path
    csv_path: Path
    best_params: Dict[str, Any]
    best_value: float
    n_trials_completed: int


def run_optimize(
    morphometrics_path: Path,
    run_name: str,
    objective_name: str,
    n_trials: int = 30,
    search_space: Optional[Dict[str, Dict[str, Any]]] = None,
    seed: int = 42,
    material_E_mpa: float = pipeline.DEFAULT_SLA_MATERIAL_E_MPA,
    material_nu: float = pipeline.DEFAULT_SLA_MATERIAL_NU,
    mesh_size_mm: float = pipeline.DEFAULT_MESH_SIZE_MM,
    junction_refinement_factor: float = 1.0,
    skip_probe: bool = False,
) -> OptimizeResult:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "optuna is not installed. Install with `pip install optuna` or add to the conda env."
        ) from exc

    objective = objective_registry.load_builtin(objective_name)
    direction = "maximize" if objective.direction == "maximize" else "minimize"
    space = search_space or DEFAULT_SEARCH_SPACE

    run_root = BIOMIMETIC_ROOT / "runs" / run_name
    run_root.mkdir(parents=True, exist_ok=False)
    csv_path = run_root / "optimize_log.csv"
    sweep_log.ensure_header(csv_path)

    trial_counter = {"i": 0}

    def _suggest(trial: "optuna.Trial") -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for key, spec in space.items():
            t = spec["type"]
            if t == "float":
                params[key] = trial.suggest_float(key, spec["low"], spec["high"])
            elif t == "int":
                params[key] = trial.suggest_int(key, spec["low"], spec["high"])
            elif t == "categorical":
                params[key] = trial.suggest_categorical(key, spec["choices"])
            else:
                raise ValueError(f"Unknown search-space type: {t}")
        return params

    def _obj(trial: "optuna.Trial") -> float:
        i = trial_counter["i"]
        trial_counter["i"] += 1
        overrides = _suggest(trial)
        trial_name = f"{run_name}/trial_{i:03d}"
        t0 = time.time()
        try:
            result = pipeline.run_pipeline(
                morphometrics_path=Path(morphometrics_path),
                run_name=trial_name,
                objective_name=objective_name,
                material_E_mpa=material_E_mpa,
                material_nu=material_nu,
                mesh_size_mm=mesh_size_mm,
                junction_refinement_factor=junction_refinement_factor,
                skip_probe=skip_probe or i > 0,
                extra_overrides=overrides,
            )
        except Exception as exc:
            # Penalize failed trials with a very bad score in the correct direction.
            trial.set_user_attr("error", repr(exc))
            return float("inf") if direction == "minimize" else float("-inf")

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
            label=f"optuna_trial_{i:03d}",
            total_time_s=elapsed,
        )
        score = (
            result.score
            if result.score is not None
            else (result.metrics.get("_objective_score") or 0.0)
        )
        return float(score)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(_obj, n_trials=n_trials, show_progress_bar=False)

    (run_root / "best.json").write_text(
        json.dumps(
            {
                "objective": objective_name,
                "direction": direction,
                "best_value": study.best_value,
                "best_params": study.best_params,
                "n_trials": len(study.trials),
            },
            indent=2,
        )
    )

    return OptimizeResult(
        run_root=run_root,
        csv_path=csv_path,
        best_params=study.best_params,
        best_value=study.best_value,
        n_trials_completed=len(study.trials),
    )
