#!/usr/bin/env python
"""Re-evaluate the best closed-loop design at the production mesh size (verification).

Purpose
-------
run_inverse_design_2026_09.py searches the design space on a 0.8-mm mesh to keep
each evaluation short. The results reported in the manuscript use 0.5-mm meshes.
This script takes the best design of a finished study (lowest objective J among
the trials that completed), rebuilds it with the identical parameter overrides
through the full pipeline (feature-to-CAD translation -> CadQuery -> integrity
gate -> Gmsh -> SfePy) at 0.5 mm, recomputes the same mechanics, and reports how
far the verified values lie from the search values and from the targets.

It mirrors the driver exactly:
  * overrides   ROD_DIAMETER = d, N_BRIDGE_LAYERS = N, RING_ROTATION = measured
                profile x f (single sense, or alternating sense ring to ring),
                CENTER_SPACING = 3.192 mm, BRIDGE_DIAMETER = 1.702 mm;
  * E_app       E_effective x solid fraction (metrics.json), as in the paper;
  * tension     tension-dominated gauge volume fraction, platen rotation, and
                field inclination from scripts/analyze_load_paths.py (300 seeds);
  * objective   J = |E_app - E*| / E* + max(0, t* - t) / t*, with the targets
                recorded in trials.jsonl;
  * bridges     the OCCT bridge offset (OCCT_TANGENT_JITTER_MM) that succeeded
                in the search is tried first, then the rest of the retry ladder.
In addition it evaluates the gauge-window percentile stress ratio kappa_99 and
the gauge solid fraction phi_g (scripts/compute_gauge_metrics.py) for both the
search run and the verification run.

Inputs
------
runs/<run-root>/trials.jsonl                       one record per trial (driver)
runs/live_001/morphometrics.json                   measured descriptor
runs/live_001_digital_twin_v2/cad_params.json      reference ring-rotation profile

Outputs
-------
runs/<run-root>/best_verify_h050[_jNNN]/           full pipeline run at 0.5 mm
runs/<run-root>/best_verified.json                 search vs verified values
runs/_load_path_analysis/<run-root>__best_verify_h050*.json   mechanics
runs/_load_path_analysis/gauge_metrics.csv         rows added for both runs

Usage (conda base, from biomimetic_pipeline/; run after the study finished):
    python scripts/verify_inverse_design_2026_09.py [--run-root inverse_design_v3] [--mesh-mm 0.5]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# --- 1. Arguments and paths (the bridge offset must be in the environment before the call) ----
ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
ap.add_argument("--run-root", default="inverse_design_v3")
ap.add_argument(
    "--mesh-mm", type=float, default=0.5, help="target element size of the verification"
)
ap.add_argument("--trial", type=int, default=None, help="trial index to verify (default: lowest J)")
ARGS = ap.parse_args()

THIS = Path(__file__).resolve()
BIO = THIS.parent.parent  # biomimetic_pipeline/ (holds runs/ and scripts/)
if str(BIO) not in sys.path:
    sys.path.insert(0, str(BIO))
RUNS = (
    BIO / "biomimetic_pipeline" / "runs"
)  # public layout: run directories live inside the package (RunContext default)
ROOT = RUNS / ARGS.run_root
LPA = RUNS / "_load_path_analysis"
REF_CAD = RUNS / "live_001_digital_twin_v2" / "cad_params.json"
MORPH = RUNS / "live_001" / "morphometrics.json"
FIXED = {"CENTER_SPACING": 3.192, "BRIDGE_DIAMETER": 1.702}  # as in the driver


def ring_rotation(f: float, handedness: str) -> dict:
    """Measured ring-rotation profile scaled by f; 'alternating' flips every other ring (as the driver)."""
    base = json.loads(REF_CAD.read_text())["RING_ROTATION"]
    if handedness == "single":
        return {k: float(v) * f for k, v in base.items()}
    return {k: abs(float(v)) * f * (1.0 if int(k) % 2 == 0 else -1.0) for k, v in base.items()}


def mechanics(run_dir: Path) -> dict:
    """Principal-stress analysis of one run (same settings as the driver)."""
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


def gauge_metrics(run_dirs: list[Path]) -> dict:
    """kappa_99 and phi_g per run from compute_gauge_metrics.py (rows for other runs are kept)."""
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(BIO / "scripts" / "compute_gauge_metrics.py"),
            "--runs",
            *[str(r) for r in run_dirs],
        ],
        check=True,
        cwd=str(BIO),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    out = {}
    with open(LPA / "gauge_metrics.csv") as fh:
        for row in csv.DictReader(fh):
            out[row["run"]] = row
    return {str(r.relative_to(RUNS)): out.get(str(r.relative_to(RUNS)), {}) for r in run_dirs}


def main() -> None:
    # --- 2. Pick the design: lowest J among completed trials (or --trial) -----------------------
    trials = [
        json.loads(line)
        for line in (ROOT / "trials.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ok = [t for t in trials if t.get("state") == "ok"]
    best = (
        next(t for t in ok if t["trial"] == ARGS.trial)
        if ARGS.trial is not None
        else min(ok, key=lambda t: t["J"])
    )
    e_star, t_star = best["targets"]["E_app"], best["targets"]["tension"]
    overrides = {
        "ROD_DIAMETER": best["d"],
        "N_BRIDGE_LAYERS": best["N"],
        "RING_ROTATION": ring_rotation(best["f"], best["handedness"]),
        **FIXED,
    }
    print(
        f"[verify] trial {best['trial']:03d} f={best['f']:.3f} N={best['N']} d={best['d']:.3f} "
        f"{best['handedness']} (search J={best['J']:.4f}) at {ARGS.mesh_mm} mm",
        flush=True,
    )

    # --- 3. Rebuild and solve at the production mesh, bridge offset of the search first ------------
    from biomimetic_pipeline.orchestration import (
        pipeline,  # noqa: E402  (imported after the environment is set)
    )

    ladder = [best["jitter_mm"]] + [j for j in (0.15, 0.30, 0.50, 0.0) if j != best["jitter_mm"]]
    base = f"{ARGS.run_root}/best_verify_h{int(round(ARGS.mesh_mm * 100)):03d}"
    res, name, attempts, t0 = None, None, [], time.time()
    for k, jit in enumerate(ladder):
        os.environ["OCCT_TANGENT_JITTER_MM"] = f"{jit:g}"
        run_k = base if k == 0 else f"{base}_j{int(round(jit * 100)):03d}"
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
            if "integrity" not in repr(exc).lower():
                raise
    if res is None:
        raise RuntimeError("CAD integrity failed at every bridge offset of the retry ladder")

    # --- 4. Mechanics of the verification run and gauge metrics of both runs -------------------
    e_app = float(res.metrics["E_effective_MPa"]) * float(res.metrics["solid_fraction"])
    mech = mechanics(RUNS / name)
    J = abs(e_app - e_star) / e_star + max(0.0, t_star - mech["tension"]) / t_star
    gm = gauge_metrics([RUNS / best["run"], RUNS / name])

    def rel(a: float, b: float) -> float:
        return (a - b) / b if b else float("nan")

    out = {
        "trial": best["trial"],
        "design": {k: best[k] for k in ("f", "N", "d", "handedness")},
        "targets": {"E_app_MPa": e_star, "tension_frac": t_star},
        "search": {
            "mesh_mm": 0.8,
            "run": best["run"],
            "E_app": best["E_app"],
            "tension": best["tension"],
            "rotation_deg_per_pct": best["rotation_deg_per_pct"],
            "inclination_deg": best["inclination_deg"],
            "J": best["J"],
            "gauge": gm.get(best["run"], {}),
        },
        "verified": {
            "mesh_mm": ARGS.mesh_mm,
            "run": name,
            "attempts": attempts,
            "E_app": e_app,
            **mech,
            "J": J,
            "gauge": gm.get(name, {}),
            "wall_s": round(time.time() - t0, 1),
        },
        "relative_change_verified_vs_search": {
            "E_app": rel(e_app, best["E_app"]),
            "tension": rel(mech["tension"], best["tension"]),
        },
        "relative_error_vs_target": {
            "E_app": (e_app - e_star) / e_star,
            "tension_shortfall": max(0.0, t_star - mech["tension"]) / t_star,
        },
    }
    (ROOT / "best_verified.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
