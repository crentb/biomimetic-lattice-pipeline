#!/usr/bin/env python
"""
run_trial_tall.py
=================

Purpose
-------
Run ONE biomimetic-pipeline trial at an explicit ENAMEL_THICKNESS (specimen
height H), for EITHER the bio or the thick-rod variant. This is the general
single-trial driver for the H=24 sweep rebuild: identical in spirit to
scripts/run_single_trial.py but adds an ``--enamel-thickness`` flag (the height
override that spreads the bridge layers) and makes ROD/BRIDGE/CS OPTIONAL so the
same tool builds both arms.

Why this exists
---------------
run_single_trial.py exposes --N/--ROD/--BRIDGE/--CENTER-SPACING but has NO
height flag, and the repo rule forbids editing it in place. ENAMEL_THICKNESS
reaches the bridge mapper through extra_overrides: feature_to_cad applies
extra_overrides at map_morphometrics:~114, THEN reads ENAMEL_THICKNESS into the
bridge-elevation computation at ~215/~262 -- so injecting it via extra_overrides
genuinely SPREADS the bridge layers (verified empirically 2026-06-03; raising H
20->24 is what makes N=9 thick meshable at the canonical 3.167 mm rod, where the
old "geometrically unmeshable" verdict was wrong -- see runs/_n9_tall_test/).

Both arms from one tool:
  * bio   : pass NO rod/bridge/cs  -> the mapper derives the morphometric
            geometry (the zero-free-parameter path).
  * thick : pass all three (ROD 3.1669 / BRIDGE 1.7024 / CS 3.1919) -> the
            near-touching thick variant. A LONE ROD override RAISES in the
            mapper (historical bridge-inflation guard), so thick must pin all 3.

Inputs (CLI)
------------
  --morphometrics     canonical specimen (default runs/live_001/morphometrics.json)
  --run-name          REQUIRED; run dir under runs/.
  --N                 REQUIRED; N_BRIDGE_LAYERS.
  --enamel-thickness  specimen height H (mm); default 24.0 (pipeline default 20.0).
  --ROD-DIAMETER / --BRIDGE-DIAMETER / --CENTER-SPACING
                      optional thick contract (omit ALL THREE for bio).
  --objective         forwarded to run_pipeline (default crack_deflection).

Outputs
-------
  runs/<run-name>/ with cad/, mesh/, fea/, metrics.json on full success. The
  pipeline gates on CAD integrity then a non-empty mesh, so a bridge-drop or
  0-tet failure aborts before any FEA compute. Idempotent: an existing run dir
  short-circuits with "skip: already exists" (delete by hand to rebuild).

Side effects / non-obvious behavior
-----------------------------------
  * OCCT_TANGENT_JITTER_MM (env) selects the bridge-escape jitter for the build;
    the H=24 sweep orchestrator sets it per attempt during the jitter hunt.
  * skip_probe=True (trust the conda envs, exactly like run_single_trial).
"""

from __future__ import annotations

# --- stdlib + path bootstrap ------------------------------------------------
import argparse
import sys
from pathlib import Path

# Script lives at <biomimetic_root>/scripts/, so root = parents[1]. Put it on
# sys.path so `import orchestration` resolves like the other scripts do.
THIS = Path(__file__).resolve()
BIOMIMETIC_ROOT = THIS.parent.parent
if str(BIOMIMETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(BIOMIMETIC_ROOT))

from biomimetic_pipeline.orchestration import (
    pipeline,  # noqa: E402 (post-sys.path import, intentional)
)


def main() -> int:
    # --- 1. Parse CLI ------------------------------------------------------
    ap = argparse.ArgumentParser(
        description="Run one trial at an explicit ENAMEL_THICKNESS (bio or thick)."
    )
    ap.add_argument(
        "--morphometrics",
        type=Path,
        default=BIOMIMETIC_ROOT / "runs" / "live_001" / "morphometrics.json",
        help="canonical specimen (default live_001).",
    )
    ap.add_argument(
        "--run-name",
        required=True,
        help="run dir under runs/ (e.g. sweep_H24_thick/trial_003_N_BRIDGE_LAYERS_4).",
    )
    ap.add_argument("--N", type=int, required=True, help="N_BRIDGE_LAYERS.")
    ap.add_argument(
        "--enamel-thickness",
        dest="enamel_thickness",
        type=float,
        default=24.0,
        help="specimen height H in mm (default 24.0; pipeline default is 20.0).",
    )
    # ROD/BRIDGE/CS optional: present => thick variant; absent => bio (mapper-derived).
    ap.add_argument(
        "--ROD-DIAMETER",
        dest="rod_diameter",
        type=float,
        default=None,
        help="optional ROD_DIAMETER (mm); set for thick, omit for bio.",
    )
    ap.add_argument(
        "--BRIDGE-DIAMETER",
        dest="bridge_diameter",
        type=float,
        default=None,
        help="optional BRIDGE_DIAMETER (mm); pin with ROD for thick.",
    )
    ap.add_argument(
        "--CENTER-SPACING",
        dest="center_spacing",
        type=float,
        default=None,
        help="optional CENTER_SPACING (mm); pin with ROD for thick.",
    )
    ap.add_argument(
        "--objective",
        default="crack_deflection",
        help="objective forwarded to run_pipeline (default crack_deflection).",
    )
    args = ap.parse_args()

    # --- 2. Idempotency guard ---------------------------------------------
    # Existing dir => assume done/partial; bail without raising so the jitter
    # hunt and re-runs are safe. Delete the dir by hand to force a rebuild.
    target_dir = BIOMIMETIC_ROOT / "runs" / args.run_name
    if target_dir.exists():
        print(f"skip: already exists {target_dir}")
        return 0

    # --- 3. Build extra_overrides -----------------------------------------
    # Always set N + ENAMEL_THICKNESS. ROD/BRIDGE/CS are added ONLY if provided
    # (thick); for bio they stay absent so the mapper derives them from the
    # morphometrics. The mapper RAISES on a lone ROD (without BRIDGE), so the
    # thick caller must pin all three -- enforced upstream by that guard.
    extra_overrides = {
        "N_BRIDGE_LAYERS": int(args.N),
        "ENAMEL_THICKNESS": float(args.enamel_thickness),
    }
    if args.rod_diameter is not None:
        extra_overrides["ROD_DIAMETER"] = float(args.rod_diameter)
    if args.bridge_diameter is not None:
        extra_overrides["BRIDGE_DIAMETER"] = float(args.bridge_diameter)
    if args.center_spacing is not None:
        extra_overrides["CENTER_SPACING"] = float(args.center_spacing)

    # --- 4. Run the full pipeline -----------------------------------------
    # Same call shape as run_single_trial; the pipeline gates on CAD integrity
    # (watertight + bridge presence) and a non-empty mesh before FEA.
    result = pipeline.run_pipeline(
        morphometrics_path=args.morphometrics,
        run_name=args.run_name,
        objective_name=args.objective,
        extra_overrides=extra_overrides,
        skip_probe=True,
    )

    # --- 5. Report ---------------------------------------------------------
    print(f"Run complete: {result.run_dir}")
    print(f"Metrics: {result.metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
