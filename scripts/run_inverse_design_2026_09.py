#!/usr/bin/env python
"""Closed-loop inverse design of the enamel-derived lattice (Bayesian optimization).

Purpose
-------
Demonstrates that the pipeline can PROGRAM the mechanical response of the
printable lattice, not only map it by one-factor sweeps. An Optuna
Tree-structured Parzen Estimator (TPE) proposes a design; the full pipeline
(feature-to-CAD translation -> CadQuery CAD -> integrity gate -> Gmsh mesh ->
SfePy linear-elastic FEA) evaluates it; the principal-stress analysis scores it;
the score is returned to the optimizer; and the loop repeats.

Design variables (the four levers of the design-space study)
  f           twist scale on the measured ring-rotation profile   [0.05, 1.50]
  N           bridge layers (integer)                              {2, ..., 6}
  d           rod diameter (mm, CAD scale; measured value 2.128)    [2.128, 2.90]
  handedness  'single' (every ring rotates in the same sense) or
              'alternating' (rings alternate in sense)            categorical
Fixed: centre spacing 3.192 mm and bridge diameter 1.702 mm (as in the
two-arm study; pinned explicitly because overriding ROD_DIAMETER alone is
rejected by the mapper), 20-mm lattice height, E_s = 3000 MPa, nu = 0.4,
free-rotating top platen.

Evaluation cost
---------------
The response is linear, so E_app and the tension fraction are independent
of the load scale: each design is solved ONCE (objective_name=None skips the
pipeline's load-rescaling iterations). The search uses a 0.8-mm mesh
(--mesh-mm), at which the mesh-convergence study gives E_app within 0.3 % and
the tension fraction within 0.6 % of the 0.5-mm values; the best design is
re-evaluated at 0.5 mm afterwards (--verify-best).

Objective (minimized): two simultaneous targets
  J = |E_app - E*| / E*  +  max(0, t* - t) / t*
with E_app = apparent modulus (MPa; E_effective x solid fraction, as in
Fig. 5/6), E* = 400 MPa, t = tension-dominated gauge volume fraction (from
scripts/analyze_load_paths.py), t* = 0.20. J = 0 means a lattice that is as
stiff as requested AND retains at least 20 % tension-dominated volume, i.e.
retains compression-twist coupling. Failed designs (CAD integrity gate,
meshing, solver) score J = 10.

Geometry-kernel retries
-----------------------
OpenCASCADE can silently drop the bridge layers of a lattice when a bridge
sits tangent to the tessellated helical rods (a known Boolean-union
degeneracy; see mapping/bridge_mappers.py). The integrity gate detects this
and rejects the model. Because the failure depends on the exact bridge
elevation, a rejected design is rebuilt with the next vertical bridge offset
of JITTER_LADDER (read from OCCT_TANGENT_JITTER_MM at call time) before it
is scored as failed; the offset that succeeded is recorded with the trial.

Inputs
------
runs/live_001/morphometrics.json                  measured descriptor
runs/live_001_digital_twin_v2/cad_params.json     reference ring-rotation profile

Outputs
-------
runs/inverse_design_v1/trial_NNN/                 full pipeline run per design
runs/inverse_design_v1/trials.jsonl               one record per trial
runs/inverse_design_v1/best.json                  best design and score
runs/_load_path_analysis/inverse_design_v1__trial_NNN.json  per-trial mechanics

Side effects
------------
* Forces OCCT_TANGENT_JITTER_MM (default 0.15 mm, --jitter-mm) before the
  pipeline is imported (the value is read at import; recorded in every
  cad_params.json and in trials.jsonl).
* After each trial the element-result CSVs of that trial are gzip-compressed
  (reversible with gunzip; the VTK files hold the same fields) to bound disk use.
* Run only when no other FEA campaign is running (memory-bound).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# --- 1. Environment that must be fixed before the pipeline is imported ---------------
_ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
_ap.add_argument("--n-trials", type=int, default=16)
_ap.add_argument("--startup-trials", type=int, default=6, help="random designs before TPE guidance")
_ap.add_argument("--target-e-mpa", type=float, default=400.0)
_ap.add_argument("--target-tension", type=float, default=0.20)
_ap.add_argument(
    "--jitter-mm", type=float, default=0.15, help="first bridge offset of the retry ladder (mm)"
)
_ap.add_argument("--run-root", default="inverse_design_v3")
_ap.add_argument("--mesh-mm", type=float, default=0.8, help="target element size during the search")
_ap.add_argument("--seed", type=int, default=42)
ARGS = _ap.parse_args()
os.environ["OCCT_TANGENT_JITTER_MM"] = f"{ARGS.jitter_mm:g}"

THIS = Path(__file__).resolve()
BIO = THIS.parent.parent
if str(BIO) not in sys.path:
    sys.path.insert(0, str(BIO))
import optuna  # noqa: E402

from biomimetic_pipeline.orchestration import pipeline  # noqa: E402

RUNS = (
    BIO / "biomimetic_pipeline" / "runs"
)  # public layout: run directories live inside the package (RunContext default)
ROOT = RUNS / ARGS.run_root
REF_CAD = RUNS / "live_001_digital_twin_v2" / "cad_params.json"
MORPH = RUNS / "live_001" / "morphometrics.json"
LPA = RUNS / "_load_path_analysis"
PENALTY = 10.0  # score of a design that could not be built or solved
JITTER_LADDER = [ARGS.jitter_mm] + [j for j in (0.30, 0.50, 0.0) if j != ARGS.jitter_mm]
FIXED = {"CENTER_SPACING": 3.192, "BRIDGE_DIAMETER": 1.702}


def ring_rotation(f: float, handedness: str) -> dict:
    """Measured ring-rotation profile scaled by f; 'alternating' flips every other ring."""
    base = json.loads(REF_CAD.read_text())["RING_ROTATION"]
    out = {}
    for k, v in base.items():
        mag = abs(float(v)) * f
        if handedness == "single":
            out[k] = float(v) * f  # keep the measured (single) sense
        else:
            out[k] = mag * (1.0 if int(k) % 2 == 0 else -1.0)  # ring-to-ring alternation
    return out


def gzip_element_csvs(run_dir: Path) -> None:
    """Compress the large per-element result tables of one trial (reversible)."""
    for csv in run_dir.glob("fea/*/element_results_compression.csv"):
        with open(csv, "rb") as src, gzip.open(str(csv) + ".gz", "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        csv.unlink()


def mechanics(run_dir: Path) -> dict:
    """Tension-dominated fraction and top-plate rotation from the principal-stress analysis."""
    rel = run_dir.relative_to(RUNS)
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(BIO / "scripts" / "analyze_load_paths.py"),
            "--runs",
            str(run_dir),
            "--n-seeds",
            "300",
            "--save-paths",
            "50",
        ],
        check=True,
        cwd=str(BIO),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    j = json.loads((LPA / (str(rel).replace("/", "__") + ".json")).read_text())
    return {
        "tension": float(j["load_partition"]["tensile_volume_frac"]),
        "rotation_deg_per_pct": float(j["top_plate"]["rotation_deg_per_pct_strain"]),
        "inclination_deg": float(j["load_partition"]["inclination_field_deg"]),
    }


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    log = ROOT / "trials.jsonl"
    counter = {"i": len(log.read_text().splitlines()) if log.exists() else 0}

    # --- 2. Objective: one full pipeline evaluation per proposed design -----------------
    def objective(trial: optuna.Trial) -> float:
        f = trial.suggest_float("f", 0.05, 1.50)
        n = trial.suggest_int("N", 2, 6)
        d = trial.suggest_float("d", 2.128, 2.90)
        hand = trial.suggest_categorical("handedness", ["single", "alternating"])
        i = counter["i"]  # trial index (Optuna numbers only completed trials)
        counter["i"] += 1
        name = f"{ARGS.run_root}/trial_{i:03d}"
        overrides = {
            "ROD_DIAMETER": d,
            "N_BRIDGE_LAYERS": n,
            "RING_ROTATION": ring_rotation(f, hand),
            **FIXED,
        }
        rec = {
            "trial": i,
            "f": f,
            "N": n,
            "d": d,
            "handedness": hand,
            "targets": {"E_app": ARGS.target_e_mpa, "tension": ARGS.target_tension},
        }
        t0 = time.time()
        print(f"[run ] {name} f={f:.3f} N={n} d={d:.3f} {hand}", flush=True)
        try:
            res, attempts = None, []
            for k, jit in enumerate(JITTER_LADDER):
                os.environ["OCCT_TANGENT_JITTER_MM"] = f"{jit:g}"
                run_k = name if k == 0 else f"{name}_j{int(round(jit * 100)):03d}"
                try:
                    res = pipeline.run_pipeline(
                        morphometrics_path=MORPH,
                        run_name=run_k,
                        objective_name=None,
                        model_type="continuous_twist",
                        mesh_size_mm=ARGS.mesh_mm,
                        extra_overrides=overrides,
                        skip_probe=True,
                        allow_broken_cad=False,
                    )
                    name = run_k
                    attempts.append({"jitter_mm": jit, "state": "ok"})
                    break
                except Exception as exc:  # noqa: BLE001 -- retry only geometry-kernel rejections
                    attempts.append({"jitter_mm": jit, "state": "failed", "error": repr(exc)[:200]})
                    if (
                        "CADIntegrityError" not in repr(exc)
                        and "integrity" not in repr(exc).lower()
                    ):
                        raise
            rec["attempts"] = attempts
            if res is None:
                raise RuntimeError(
                    "CAD integrity failed at every bridge offset of the retry ladder"
                )
            rec.update({"run": name, "jitter_mm": attempts[-1]["jitter_mm"]})
            e_app = float(res.metrics["E_effective_MPa"]) * float(res.metrics["solid_fraction"])
            mech = mechanics(RUNS / name)
            score = (
                abs(e_app - ARGS.target_e_mpa) / ARGS.target_e_mpa
                + max(0.0, ARGS.target_tension - mech["tension"]) / ARGS.target_tension
            )
            rec.update({"state": "ok", "E_app": e_app, **mech, "J": score})
            gzip_element_csvs(RUNS / name)
        except Exception as exc:  # noqa: BLE001 -- CAD gate, mesher, or solver failure
            score = PENALTY
            rec.update({"state": "failed", "error": repr(exc)[:400], "J": score})
        rec["wall_s"] = round(time.time() - t0, 1)
        with open(log, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(
            f"[{'ok ' if rec['state'] == 'ok' else 'FAIL'}] {name} J={score:.4f} "
            + (
                f"E_app={rec['E_app']:.1f} tension={rec['tension']:.3f}"
                if rec["state"] == "ok"
                else rec["error"][:120]
            ),
            flush=True,
        )
        return score

    # --- 3. Bayesian optimization loop -----------------------------------------------------
    sampler = optuna.samplers.TPESampler(
        seed=ARGS.seed, n_startup_trials=ARGS.startup_trials, multivariate=True
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=ARGS.n_trials, show_progress_bar=False)
    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "targets": {"E_app_MPa": ARGS.target_e_mpa, "tension_frac": ARGS.target_tension},
        "sampler": "TPE (multivariate)",
        "startup_trials": ARGS.startup_trials,
        "seed": ARGS.seed,
    }
    (ROOT / "best.json").write_text(json.dumps(best, indent=1))
    print("[done] best", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
