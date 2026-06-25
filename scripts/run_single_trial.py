#!/usr/bin/env python
"""Single-trial wrapper for the biomimetic pipeline.

Purpose
-------
This helper runs `orchestration.pipeline.run_pipeline(...)` exactly once
for a single (N_BRIDGE_LAYERS, optional ROD_DIAMETER) override. It exists
so the layer-sweep-with-thick-variant orchestrator shell can launch each
trial as an independent OS process, keeping per-trial memory bounded
and giving us per-trial idempotency without having to re-run the
multi-trial `orchestration.sweep` driver.

Why a thin wrapper, not a direct pipeline call from bash:
  * `pipeline.run_pipeline` is a Python function with a non-trivial
    signature; reusing it via a tiny CLI keeps the orchestrator shell
    declarative and lets us add the idempotency guard below in one
    obvious place.
  * The orchestrator runs two pipeline variants per N (biomimetic +
    thick rod), so we want a single entry point that handles both
    cases via the optional --ROD-DIAMETER flag.

Inputs (CLI)
------------
  --morphometrics   Path to the canonical morphometrics.json (default
                    is the live_001 specimen the rest of the sweep
                    uses).
  --run-name        Relative path under <biomimetic_root>/runs/ where
                    this trial will live (e.g.
                    "sweep_layers_v2/trial_004_N_BRIDGE_LAYERS_5").
                    Required.
  --N               Integer N_BRIDGE_LAYERS override. Required.
  --ROD-DIAMETER    Optional float; if given, overrides ROD_DIAMETER
                    too (used to build the "thick" variant where rod
                    diameter is CENTER_SPACING - 0.025 mm).
  --objective       Objective name forwarded to the pipeline; defaults
                    to "crack_deflection" to match the rest of the
                    layer sweep.

Outputs
-------
  Creates <biomimetic_root>/runs/<run_name>/ on success and prints the
  result paths from the pipeline. On the idempotency short-circuit it
  prints "skip: already exists" and exits 0 (so the orchestrator can
  re-run safely after an interruption).

Side effects / non-obvious behavior
-----------------------------------
  * `skip_probe=True` is always passed: the conda env probe inside
    `pipeline.run_pipeline` is slow and we already trust the envs in
    this orchestrated context.
  * `extra_overrides` is built as {"N_BRIDGE_LAYERS": N, ...} and
    forwarded into the pipeline; this is how feature_to_cad knows
    which knobs to set instead of the auto-mapped defaults.
  * The idempotency guard checks for *directory existence only*; it
    does NOT verify the run completed successfully. If a previous
    run crashed mid-way and left a partial directory, you must
    delete that directory by hand before re-running.
"""

from __future__ import annotations

# --- stdlib imports ---------------------------------------------------------
import argparse
import sys
from pathlib import Path

# --- locate the biomimetic_pipeline package on sys.path ---------------------
# This script lives at <biomimetic_root>/scripts/run_single_trial.py, so the
# parent of the parent is the biomimetic_pipeline root. We insert that root
# on sys.path so that `import orchestration` works the same way it does for
# the existing scripts (run_pipeline.py, run_sweep.py).
THIS = Path(__file__).resolve()
BIOMIMETIC_ROOT = THIS.parent.parent
if str(BIOMIMETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(BIOMIMETIC_ROOT))

from biomimetic_pipeline.orchestration import (
    pipeline,  # noqa: E402  (deliberate post-sys.path import)
)


def main() -> int:
    """Parse args, gate on idempotency, call run_pipeline. Returns process exit code."""

    # --- 1. CLI parsing ----------------------------------------------------
    # All flags are kebab-case for shell ergonomics; argparse maps them to
    # underscored attribute names automatically.
    ap = argparse.ArgumentParser(
        description=(
            "Run a single biomimetic-pipeline trial with explicit "
            "N_BRIDGE_LAYERS (and optionally ROD_DIAMETER) overrides."
        )
    )
    ap.add_argument(
        "--morphometrics",
        type=Path,
        default=BIOMIMETIC_ROOT / "runs" / "live_001" / "morphometrics.json",
        help=(
            "Path to morphometrics.json (default: the live_001 specimen, "
            "which is the canonical input for the layer sweep)."
        ),
    )
    ap.add_argument(
        "--run-name",
        required=True,
        help=(
            "Relative path under <biomimetic_root>/runs/ for this trial "
            "directory (e.g. 'sweep_layers_v2/trial_004_N_BRIDGE_LAYERS_5')."
        ),
    )
    ap.add_argument(
        "--N",
        type=int,
        required=True,
        help="N_BRIDGE_LAYERS override (positive integer).",
    )
    ap.add_argument(
        "--ROD-DIAMETER",
        dest="rod_diameter",
        type=float,
        default=None,
        help=(
            "Optional ROD_DIAMETER override (mm). Set for the 'thick' "
            "variant; leave unset for the baseline biomimetic geometry."
        ),
    )
    # The thick-rod variant GROWS the rod while KEEPING the biomimetic bridge and
    # ring spacing. The mapper (post-2026-05-31 fix) honors explicit BRIDGE/CS
    # overrides and RAISES on a lone ROD override, so the thick arm must pass all
    # three -- this is what prevents the historical silent re-derivation that
    # inflated BRIDGE to 0.80*thick_rod (2.47 mm) and CS to 1.05*thick_rod.
    ap.add_argument(
        "--BRIDGE-DIAMETER",
        dest="bridge_diameter",
        type=float,
        default=None,
        help="Optional BRIDGE_DIAMETER override (mm); pass with --ROD-DIAMETER for the thick variant to keep the biomimetic bridge.",
    )
    ap.add_argument(
        "--CENTER-SPACING",
        dest="center_spacing",
        type=float,
        default=None,
        help="Optional CENTER_SPACING override (mm); pass with --ROD-DIAMETER for the thick variant to keep the biomimetic ring spacing.",
    )
    ap.add_argument(
        "--objective",
        default="crack_deflection",
        help=(
            "Objective name forwarded to the pipeline. Default "
            "'crack_deflection' matches the rest of the layer sweep."
        ),
    )
    args = ap.parse_args()

    # --- 2. Idempotency guard ---------------------------------------------
    # If the target run directory already exists we assume an earlier
    # invocation completed (or partially completed) this trial and bail
    # without raising. The orchestrator shell calls this script for every
    # N regardless of state, and this is the single point where "already
    # done" is decided. We deliberately do NOT inspect contents here -- if
    # a run crashed mid-way the user must clean it up by hand.
    target_dir = BIOMIMETIC_ROOT / "runs" / args.run_name
    if target_dir.exists():
        print(f"skip: already exists {target_dir}")
        return 0

    # --- 3. Build the extra_overrides dict ---------------------------------
    # feature_to_cad.map_morphometrics merges this on top of the auto-mapped
    # defaults; any key here wins. We only ever set N_BRIDGE_LAYERS plus
    # (optionally) ROD_DIAMETER -- nothing else in the sweep is varied.
    extra_overrides = {"N_BRIDGE_LAYERS": int(args.N)}
    if args.rod_diameter is not None:
        extra_overrides["ROD_DIAMETER"] = float(args.rod_diameter)
    # Pin BRIDGE/CS explicitly (thick variant). The mapper honors these instead
    # of re-deriving them from the (grown) ROD; passing ROD alone now raises.
    if args.bridge_diameter is not None:
        extra_overrides["BRIDGE_DIAMETER"] = float(args.bridge_diameter)
    if args.center_spacing is not None:
        extra_overrides["CENTER_SPACING"] = float(args.center_spacing)

    # --- 4. Run the pipeline ----------------------------------------------
    # skip_probe=True bypasses the per-run conda-env probe; we already
    # trust the envs in the orchestrated context and the probe is slow.
    # We use the run_pipeline default values for everything else (material
    # E, nu, compression displacement, mesh size) so the trial matches the
    # existing N=4/6/8 trials byte-for-byte aside from the explicit
    # overrides above.
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
