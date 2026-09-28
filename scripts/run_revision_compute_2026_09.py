#!/usr/bin/env python
"""Revision compute driver (2026-09): dense decussation sweep + mesh convergence.

Purpose
-------
Runs the two new FEA campaigns requested for the Matter revision of
``manuscript/matter_v2`` and logs every trial in the same ``sweep_log.csv``
format the existing sweeps use, so the figure scripts can read old and new
trials with one code path:

  1. ``decussation`` -- a DENSER decussation-amplitude sweep around the
     "knee" reported in the v1 manuscript. The v1 sweep had only four factors
     {0.05, 0.5, 1.0, 1.5}; a threshold claim cannot rest on four points. This
     adds {0.6, 0.7, 0.8, 0.9, 1.25, 2.0} plus a factor-1.0 REPRODUCIBILITY
     re-run (run first) that anchors the new points to the May-2026 trials.
  2. ``mesh`` -- a mesh-convergence study of the canonical N=2 lattice at
     Gmsh target sizes {0.8, 0.65, 0.4} mm. The canonical run used 0.5 mm, so
     together with the factor-1.0 re-run this gives a 4-level ladder
     (0.8 / 0.65 / 0.5 / 0.4 mm) for E_eff, SCF, and path-tortuosity metrics.

Why this exists
---------------
The v1 manuscript's decussation "knee" and every stress-concentration number
were reported without a convergence check or adequate sampling. Reviewers will
ask for both. The existing ``run_decussation_sweep.py`` aborts the whole sweep
on the first failed trial and refuses an existing run directory; this driver
isolates failures per trial (a factor-2.0 lattice may be un-buildable in
OpenCASCADE) and can resume.

Geometry fidelity (critical)
----------------------------
The canonical lattice (``runs/live_001_digital_twin_v2``) and the v1
decussation trials were generated on 2026-05-27, BEFORE the OCCT bridge
Z-jitter default was introduced. Their ``BRIDGE_Z_OFFSETS`` are
[1.851, 17.649] mm (zero jitter). The current mapper default is 0.50 mm, which
would shift the bridges and silently change the geometry. This driver
therefore forces ``OCCT_TANGENT_JITTER_MM=0`` for every trial; a pre-flight
check (run 2026-09-25) confirmed that the current mapper with jitter 0
reproduces the canonical ``cad_params.json`` with zero differing keys.

Inputs
------
CLI:
  --phase {decussation,mesh,all}   which campaign(s) to run (default: all)
  --morphometrics PATH             canonical morphometrics.json
                                   (default runs/live_001/morphometrics.json)
  --baseline-cad-params PATH       cad_params.json whose RING_ROTATION is the
                                   factor-1.0 baseline
                                   (default runs/live_001_digital_twin_v2/cad_params.json)
  --factors F [F ...]              decussation factors (default: 1.0 repro first,
                                   then 0.8 0.9 0.7 0.6 1.25 2.0)
  --mesh-sizes H [H ...]           Gmsh target sizes in mm (default 0.8 0.65 0.4)
Environment: forces OCCT_TANGENT_JITTER_MM=0 (see above).

Outputs
-------
runs/sweep_decussation_dense_v3/trial_XXX_decussation_<f>/   full pipeline runs
runs/sweep_decussation_dense_v3/sweep_log.csv                one row per trial
runs/mesh_convergence_v1/trial_XXX_mesh_<h>mm/              full pipeline runs
runs/mesh_convergence_v1/sweep_log.csv
runs/<campaign>/status.json      per-trial status (ok / failed + error text,
                                  wall time) so a crashed or partial campaign
                                  is diagnosable without re-reading logs.

Side effects / non-obvious behaviour
------------------------------------
* Each trial writes ~0.2-1.5 GB (VTK + element CSV); the 0.4 mm mesh is the
  largest. Disk had ~33 GB free when this was written.
* Trials run sequentially: the sfepy direct solver (SuperLU, no UMFPACK in
  sfepy_env) is memory-hungry and the machine has 17 GB RAM.
* Already-completed trials (status "ok" in status.json) are skipped, so the
  driver can be re-invoked after an interruption.
* Existing pipeline code is imported, never modified.

Usage (from the biomimetic_pipeline directory, conda base active)
-----------------------------------------------------------------
    python -m scripts.run_revision_compute_2026_09 --phase all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# --- 0. Import path + geometry-fidelity environment ------------------------
# The biomimetic_pipeline root must be importable when run as `python -m`.
THIS = Path(__file__).resolve()
BIOMIMETIC_ROOT = THIS.parent.parent
if str(BIOMIMETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(BIOMIMETIC_ROOT))

# Force zero bridge Z-jitter BEFORE the mapper is imported/used so every trial
# reproduces the canonical May-27 bridge elevations (see module docstring).
# A cell whose CAD fails the integrity gate at zero jitter (OpenCASCADE drops
# bridges at tangent geometries) can be re-run with --jitter-mm (e.g. 0.15, the
# value used for most cells of the published two-arm sweep); the value is
# recorded in that cell's cad_params.json (BRIDGE_Z_OFFSETS) and status.json.
_JITTER = "0"
for _i, _a in enumerate(sys.argv):
    if _a == "--jitter-mm" and _i + 1 < len(sys.argv):
        _JITTER = sys.argv[_i + 1]
os.environ["OCCT_TANGENT_JITTER_MM"] = _JITTER

from biomimetic_pipeline.orchestration import pipeline, sweep_log  # noqa: E402

# --- 1. Campaign defaults ----------------------------------------------------
# Factor 1.0 is listed FIRST: it is the reproducibility anchor. If it matches
# the May-2026 trial (E_eff 638.94 MPa, SCF 7.38), the v1 points at
# 0.05/0.5/1.5 remain valid and only the new factors are needed.
DEFAULT_FACTORS = [1.0, 0.8, 0.9, 0.7, 0.6, 1.25, 2.0]
# Mesh ladder around the canonical 0.5 mm target. 0.35 mm was ruled out: the
# SuperLU fill-in at ~1.7M DOF would exceed 17 GB RAM (0.5 mm = 566k DOF).
DEFAULT_MESH_SIZES_MM = [0.8, 0.65, 0.4]
CANONICAL_MESH_SIZE_MM = 0.5
OBJECTIVE = "crack_deflection"  # carries the 200 MPa stress_target

DECUSSATION_ROOT = "sweep_decussation_dense_v3"
MESH_ROOT = "mesh_convergence_v1"


def _tag(x: float) -> str:
    """Filesystem-safe numeric tag: 0.65 -> '0p65', 1.25 -> '1p25'."""
    return f"{x:.4g}".replace(".", "p").replace("-", "m")


def _load_status(root: Path) -> dict:
    """Per-trial status ledger; lets an interrupted campaign resume."""
    p = root / "status.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_status(root: Path, status: dict) -> None:
    (root / "status.json").write_text(json.dumps(status, indent=2))


def _run_trial(
    *,
    root: Path,
    trial_name: str,
    label: str,
    overrides: dict,
    mesh_size_mm: float,
    morphometrics: Path,
    skip_probe: bool,
    status: dict,
) -> None:
    """Run one full pipeline trial with per-trial failure isolation.

    A failure (OCCT boolean degeneracy, integrity-gate abort, solver crash)
    is recorded in status.json with its traceback and does NOT stop the
    campaign -- the remaining trials still carry information.
    """
    key = trial_name.split("/", 1)[1]
    if status.get(key, {}).get("state") == "ok":
        print(f"[skip] {key} already completed", flush=True)
        return
    print(f"[run ] {key}  mesh={mesh_size_mm} mm  label={label}", flush=True)
    t0 = time.time()
    try:
        result = pipeline.run_pipeline(
            morphometrics_path=morphometrics,
            run_name=trial_name,
            objective_name=OBJECTIVE,
            model_type="continuous_twist",
            mesh_size_mm=mesh_size_mm,
            extra_overrides=overrides or None,
            skip_probe=skip_probe,
            allow_broken_cad=False,  # integrity gate stays ON
        )
        elapsed = time.time() - t0
        cad_params = json.loads(Path(result.cad_params_path).read_text())
        # Record the mesh size explicitly: the stock sweep_log leaves
        # param_mesh_size blank because the default is implicit.
        cad_params["mesh_size"] = mesh_size_mm
        sweep_log.append(
            root / "sweep_log.csv",
            params=cad_params,
            metrics=result.metrics,
            run_name=trial_name,
            objective=OBJECTIVE,
            label=label,
            total_time_s=elapsed,
        )
        status[key] = {"state": "ok", "wall_s": round(elapsed, 1), "label": label}
        print(f"[ ok ] {key}  {elapsed/60:.1f} min", flush=True)
    except Exception as exc:  # noqa: BLE001 -- isolate every failure mode
        elapsed = time.time() - t0
        status[key] = {
            "state": "failed",
            "wall_s": round(elapsed, 1),
            "label": label,
            "error": repr(exc),
            "traceback": traceback.format_exc()[-4000:],
        }
        print(f"[FAIL] {key}  {exc!r}", flush=True)
    finally:
        _save_status(root, status)


def run_decussation(args) -> None:
    """Dense decussation sweep: scale the measured RING_ROTATION by each factor."""
    # --- 2. Baseline per-ring rotation (factor 1.0 = measured lion enamel) ---
    baseline = json.loads(Path(args.baseline_cad_params).read_text())["RING_ROTATION"]
    root = BIOMIMETIC_ROOT / "biomimetic_pipeline" / "runs" / DECUSSATION_ROOT
    root.mkdir(parents=True, exist_ok=True)
    sweep_log.ensure_header(root / "sweep_log.csv")
    status = _load_status(root)
    for i, f in enumerate(args.factors):
        # Every per-ring angle (deg) is multiplied by f; f=1.0 is the measured
        # architecture, f<1 weaker decussation, f>1 exaggerated decussation.
        scaled = {k: float(v) * f for k, v in baseline.items()}
        _run_trial(
            root=root,
            trial_name=f"{DECUSSATION_ROOT}/trial_{i:03d}_decussation_{_tag(f)}{args.trial_suffix}",
            label=f"decussation:factor={f}"
            + (f";jitter={args.jitter_mm}" if args.jitter_mm != "0" else ""),
            overrides={"RING_ROTATION": scaled},
            mesh_size_mm=CANONICAL_MESH_SIZE_MM,
            morphometrics=Path(args.morphometrics),
            skip_probe=(i > 0),
            status=status,
        )


def run_mesh(args) -> None:
    """Mesh-convergence ladder on the unmodified canonical lattice (factor 1.0)."""
    root = BIOMIMETIC_ROOT / "biomimetic_pipeline" / "runs" / MESH_ROOT
    root.mkdir(parents=True, exist_ok=True)
    sweep_log.ensure_header(root / "sweep_log.csv")
    status = _load_status(root)
    # Coarse-to-fine so the memory-heaviest (0.4 mm) trial runs last; if it
    # exhausts RAM the coarser levels are already safely logged.
    for i, h in enumerate(sorted(args.mesh_sizes, reverse=True)):
        _run_trial(
            root=root,
            trial_name=f"{MESH_ROOT}/trial_{i:03d}_mesh_{_tag(h)}mm",
            label=f"mesh_size_mm={h}",
            overrides={},  # canonical geometry, only the mesh changes
            mesh_size_mm=h,
            morphometrics=Path(args.morphometrics),
            skip_probe=True,
            status=status,
        )


def main() -> None:
    # --- 3. CLI -------------------------------------------------------------
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--phase", choices=["decussation", "mesh", "all"], default="all")
    ap.add_argument(
        "--morphometrics",
        default=str(BIOMIMETIC_ROOT / "biomimetic_pipeline/runs/live_001/morphometrics.json"),
    )
    ap.add_argument(
        "--baseline-cad-params",
        default=str(
            BIOMIMETIC_ROOT / "biomimetic_pipeline/runs/live_001_digital_twin_v2/cad_params.json"
        ),
    )
    ap.add_argument("--factors", nargs="+", type=float, default=DEFAULT_FACTORS)
    ap.add_argument("--mesh-sizes", nargs="+", type=float, default=DEFAULT_MESH_SIZES_MM)
    ap.add_argument(
        "--jitter-mm",
        default="0",
        help="OCCT bridge Z-jitter (mm); read before import, see top of file",
    )
    ap.add_argument(
        "--trial-suffix",
        default="",
        help="suffix for trial directory names (e.g. _j015 for a jitter re-run)",
    )
    args = ap.parse_args()
    print(
        f"OCCT_TANGENT_JITTER_MM={os.environ['OCCT_TANGENT_JITTER_MM']} (forced for canonical fidelity)",
        flush=True,
    )

    # --- 4. Campaigns (decussation first: it carries the headline claim) ----
    if args.phase in ("decussation", "all"):
        run_decussation(args)
    if args.phase in ("mesh", "all"):
        run_mesh(args)
    print("campaigns finished", flush=True)


if __name__ == "__main__":
    main()
