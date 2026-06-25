"""Closed-loop biomimicry driver (stage 3 of the three-stage pipeline).

Runs unconstrained Bayesian optimization over the full CAD parameter space
(not driven by any specific morphometrics input), identifies the top-K
designs that maximize the objective (default: crack deflection), and
reverse-maps those CAD parameters back to morphometric target ranges. The
resulting `biomimicry_targets.json` names what biology would need to look
like to produce such lattices — these are the features worth searching for
in real microCT specimens.

Reverse map mirrors the forward map in mapping/feature_to_cad.py:
  - ROD_DIAMETER_mm * 1e3 / biology_scale_factor         -> rod_diameter_um
  - CENTER_SPACING_mm * 1e3 / biology_scale_factor       -> band_width_um
  - ENAMEL_THICKNESS / N_BRIDGE_LAYERS / biology_scale_factor * 1e3
        -> dominant_wavelength_um
  - RING_ROTATION['0']                                   -> bands[0].mean_direction_deg
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from biomimetic_pipeline.mapping.scale import DEFAULT_BIOLOGY_SCALE_FACTOR
from biomimetic_pipeline.objectives import registry as objective_registry
from biomimetic_pipeline.orchestration import pipeline, sweep_log
from biomimetic_pipeline.orchestration.run_context import BIOMIMETIC_ROOT


@dataclass
class ClosedLoopResult:
    run_root: Path
    targets_path: Path
    csv_path: Path
    n_trials: int
    top_k_designs: List[Dict[str, Any]]


def run_closed_loop(
    run_name: str,
    objective_name: str = "crack_deflection",
    n_trials: int = 30,
    top_k: int = 5,
    seed: int = 42,
    morphometrics_path: Optional[Path] = None,
    material_E_mpa: float = pipeline.DEFAULT_SLA_MATERIAL_E_MPA,
    material_nu: float = pipeline.DEFAULT_SLA_MATERIAL_NU,
    mesh_size_mm: float = pipeline.DEFAULT_MESH_SIZE_MM,
    junction_refinement_factor: float = 1.0,
    skip_probe: bool = False,
) -> ClosedLoopResult:
    """Run an unconstrained Optuna study, then reverse-map top-K to biology.

    If `morphometrics_path` is provided, it's used only for the mapping step
    that seeds trial-0 baselines; subsequent trials override CAD params via
    Optuna suggestions. If omitted, a synthetic minimal morphometrics file is
    generated for that seeding.
    """
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("optuna is required for closed-loop runs") from exc

    objective = objective_registry.load_builtin(objective_name)
    direction = "maximize" if objective.direction == "maximize" else "minimize"

    run_root = BIOMIMETIC_ROOT / "runs" / run_name
    run_root.mkdir(parents=True, exist_ok=False)
    csv_path = run_root / "closed_loop_log.csv"
    sweep_log.ensure_header(csv_path)

    seed_morph_path = (
        Path(morphometrics_path) if morphometrics_path else _write_seed_morphometrics(run_root)
    )

    # Track trials so we can sort and reverse-map top-K afterwards.
    trials_record: List[Dict[str, Any]] = []

    def _suggest(trial):
        from biomimetic_pipeline.orchestration.optimize import DEFAULT_SEARCH_SPACE

        params: Dict[str, Any] = {}
        for key, spec in DEFAULT_SEARCH_SPACE.items():
            t = spec["type"]
            if t == "float":
                params[key] = trial.suggest_float(key, spec["low"], spec["high"])
            elif t == "int":
                params[key] = trial.suggest_int(key, spec["low"], spec["high"])
            elif t == "categorical":
                params[key] = trial.suggest_categorical(key, spec["choices"])
        return params

    counter = {"i": 0}

    def _obj(trial):
        import time

        i = counter["i"]
        counter["i"] += 1
        overrides = _suggest(trial)
        trial_name = f"{run_name}/trial_{i:03d}"
        t0 = time.time()
        try:
            result = pipeline.run_pipeline(
                morphometrics_path=seed_morph_path,
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
            label=f"closed_loop_trial_{i:03d}",
            total_time_s=elapsed,
        )
        score = result.score if result.score is not None else 0.0
        trials_record.append(
            {
                "trial": i,
                "score": float(score),
                "cad_params": cad_params,
                "run_dir": str(result.run_dir),
            }
        )
        return float(score)

    import optuna

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(_obj, n_trials=n_trials, show_progress_bar=False)

    # Sort records by score (direction-aware).
    reverse = direction == "maximize"
    trials_record.sort(key=lambda r: r["score"], reverse=reverse)
    top = trials_record[: max(1, top_k)]

    targets = _reverse_map_top_k(top)
    targets_out = {
        "schema_version": "1.0.0",
        "closed_loop_run": run_name,
        "objective": objective_name,
        "direction": direction,
        "n_trials": n_trials,
        "top_k_designs": top,
        "morphometric_targets": targets,
        "interpretation": (
            f"These are the measured morphometric features that would produce lattices "
            f"optimized for '{objective_name}'. Compare specimens' morphometrics to these "
            f"ranges to identify biology with high predicted objective value."
        ),
    }
    targets_path = run_root / "biomimicry_targets.json"
    targets_path.write_text(json.dumps(targets_out, indent=2, default=str))

    return ClosedLoopResult(
        run_root=run_root,
        targets_path=targets_path,
        csv_path=csv_path,
        n_trials=len(trials_record),
        top_k_designs=top,
    )


def _write_seed_morphometrics(run_root: Path) -> Path:
    """Emit a minimal synthetic morphometrics.json that validates against the
    schema. Used when the user doesn't pass a specimen."""
    from datetime import datetime, timezone

    payload = {
        "schema_version": "1.0.0",
        "specimen_id": f"synthetic_{run_root.name}",
        "units": {"length": "um", "angle": "deg", "stress": "MPa"},
        "provenance": {
            "source_paths": {},
            "source_hashes": {},
            "ingestion_timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "coordinate_frame": {
                "z_origin": "synthetic",
                "z_axis": "+z",
                "xy_units": "um",
                "z_units": "um",
            },
            "pipeline_version": "0.1.0",
            "scale_clamps": [],
        },
        "depth_profiles": {
            "depth_um": [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
            "pitch_signed_deg": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            "rod_diameter_um_mean": [4.0, 4.2, 4.3, 4.2, 4.1, 4.0],
            "rod_diameter_um_std": [0.5] * 6,
            "eccentricity_mean": [0.1] * 6,
            "eccentricity_std": [0.05] * 6,
        },
        "angle_stats": {
            "pitch": {
                "n": 100,
                "mean": 0.0,
                "median": 0.0,
                "std": 10.0,
                "p5": -20.0,
                "p95": 20.0,
                "min": -30.0,
                "max": 30.0,
            },
            "yaw": {
                "n": 100,
                "mean": 0.0,
                "median": 0.0,
                "std": 10.0,
                "p5": -20.0,
                "p95": 20.0,
                "min": -30.0,
                "max": 30.0,
            },
            "tilt": {
                "n": 100,
                "mean": 15.0,
                "median": 15.0,
                "std": 8.0,
                "p5": 3.0,
                "p95": 27.0,
                "min": 0.0,
                "max": 40.0,
            },
        },
        "bands": [
            {
                "band_id": 0,
                "area_fraction_pct": 25.0,
                "mean_direction_deg": 0.0,
                "circular_variance": 0.1,
                "mean_displacement_mag_um": 2.0,
                "band_width_um": {"mean": 20.0, "std": 5.0, "min": 10.0, "max": 40.0},
            },
            {
                "band_id": 1,
                "area_fraction_pct": 25.0,
                "mean_direction_deg": 90.0,
                "circular_variance": 0.1,
                "mean_displacement_mag_um": 2.0,
                "band_width_um": {"mean": 20.0, "std": 5.0, "min": 10.0, "max": 40.0},
            },
        ],
        "periodicity": {"dominant_wavelength_um_mean": 85.0, "dominant_wavelength_um_std": 10.0},
        "rod_slice_stats": [],
        "rod_tracks_summary": {
            "n_tracks": 0,
            "n_complete": 0,
            "mean_arc_length_um": 0.0,
            "mean_chord_length_um": 0.0,
            "tortuosity_mean": 1.0,
            "tortuosity_p90": 1.0,
        },
    }
    out_path = run_root / "seed_morphometrics.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def _reverse_map_top_k(top_designs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not top_designs:
        return {}

    fields = {
        "rod_diameter_um": [],
        "band_width_um": [],
        "dominant_wavelength_um": [],
        "band0_direction_deg": [],
        "n_bridge_layers": [],
        "twist_type": [],
    }

    for entry in top_designs:
        cad = entry["cad_params"]
        bsf = float(cad.get("biology_scale_factor", DEFAULT_BIOLOGY_SCALE_FACTOR))
        enamel_mm = float(cad.get("ENAMEL_THICKNESS", 20.0))

        # Avoid divide-by-zero on bsf.
        bsf_safe = max(bsf, 1e-6)
        fields["rod_diameter_um"].append(float(cad.get("ROD_DIAMETER", 0.0)) / bsf_safe * 1e3)
        fields["band_width_um"].append(float(cad.get("CENTER_SPACING", 0.0)) / bsf_safe * 1e3)
        n_bridges = max(float(cad.get("N_BRIDGE_LAYERS", 2)), 1.0)
        fields["dominant_wavelength_um"].append(enamel_mm / n_bridges / bsf_safe * 1e3)
        ring_rot = cad.get("RING_ROTATION", {}) or {}
        fields["band0_direction_deg"].append(float(ring_rot.get("0", ring_rot.get(0, 0.0))))
        fields["n_bridge_layers"].append(int(cad.get("N_BRIDGE_LAYERS", 0)))
        fields["twist_type"].append(str(cad.get("TWIST_TYPE", "")))

    summary: Dict[str, Any] = {}
    for k, vals in fields.items():
        if not vals:
            continue
        if isinstance(vals[0], str):
            # Mode for categoricals
            from collections import Counter

            c = Counter(vals).most_common(1)[0]
            summary[k] = {"mode": c[0], "count": c[1], "values": vals}
        elif isinstance(vals[0], (int, float)):
            nums = [float(v) for v in vals]
            summary[k] = {
                "mean": sum(nums) / len(nums),
                "min": min(nums),
                "max": max(nums),
                "values": nums,
            }
    return summary
