#!/usr/bin/env bash
# =============================================================================
# rerun_all_12.sh  —  clean full re-run of the canonical layer sweep
# =============================================================================
#
# PURPOSE
#   Rebuild ALL 12 trials of the canonical sweep from scratch on the corrected
#   pipeline (post override-inflation fix), overwriting the existing run dirs:
#       * normal / biomimetic arm : runs/sweep_layers_v2/        N=4..9  (×6)
#       * thick-rod arm           : runs/sweep_layers_v2_thick/  N=4..9  (×6)
#   One coherent, single-provenance result set after the two-session confusion.
#
# WHY THIS EXISTS
#   The two parallel sessions left the sweep in a mixed-provenance state (some
#   trials rebuilt at different jitters, one dir archived/rebuilt). The user
#   asked to "rerun the whole thing, no [geometry] changes, overwrite". This
#   driver does exactly that, with two safeguards baked in (below).
#
# NO GEOMETRY CHANGE
#   The 0.025 mm thick rod-rod gap is UNCHANGED (thick ROD = CENTER_SPACING -
#   0.025 = 3.1669 mm). The only per-trial knob set here is the OCCT tangent-
#   escape jitter (a sub-mm bridge-Z nudge, the *documented build recipe* — see
#   mapping/bridge_mappers.py). It is NOT a geometry change; it only slides the
#   bridge stack off OCCT's tangent-resonance positions so the boolean union
#   builds a watertight solid. Uniform 0.50 is KNOWN to fail thick N=7 (needs
#   0.15), so each trial gets its known-good jitter to avoid avoidable failures.
#
# EXPECTED OUTCOME (honest, given the gap is unchanged)
#   - normal N=4..9            : all 6 complete (meshed + FEA)
#   - thick  N=4,6,8           : complete
#   - thick  N=7               : attempt at jitter 0.15 (CAD-drop case)
#   - thick  N=5, N=9          : build valid CAD but FAIL at volume meshing
#                                (the 0.025 mm gap -> 0-tet slivers; proven
#                                deterministic, not a flake). These document
#                                the real limitation; the gap fix is a
#                                separate, user-gated decision.
#
# INPUTS
#   - runs/live_001/morphometrics.json   (the single specimen morphometrics)
#   - scripts/run_single_trial.py        (per-trial CAD->mesh->FEA->metrics)
#   - OCCT_TANGENT_JITTER_MM env var     (set per trial; read by bridge_mappers)
#
# OUTPUTS / SIDE EFFECTS
#   - Overwrites runs/sweep_layers_v2[/_thick]/trial_*  (rm -rf before each).
#     *** Caller MUST archive those dirs first *** (this script does not back
#     up; see the launch step in the chat that invoked it).
#   - Appends a structured progress log to runs/_rerun_all_12.log.
#   - Each trial writes its own cad/, mesh/, metrics.json, cad_integrity_report.
#
# CONCURRENCY
#   STRICTLY SEQUENTIAL. 16 GB box; concurrent sfepy FEA OOMs. One trial at a
#   time, normal arm first (high success expectation -> validates pipeline),
#   then thick arm.
#
# RESTARTABILITY
#   Idempotent per trial: each run rm -rf's its target dir then rebuilds, so
#   re-invoking the script re-does everything cleanly. To resume only the
#   remainder, comment out the run_one lines already completed.
# =============================================================================

# Continue past a failing trial (we WANT to attempt all 12 and tally at the
# end); do NOT use `set -e`. Still catch unset vars and pipe failures.
set -uo pipefail

# --- 1. Constants -----------------------------------------------------------
ROOT="${MICROCT_PIPELINE_ROOT:-microct_pipeline}/biomimetic_pipeline"
MORPHO="${ROOT}/runs/live_001/morphometrics.json"
LOG="${ROOT}/runs/_rerun_all_12.log"

# Thick-rod coupled geometry (pinned together so the fixed mapper keeps them;
# a lone ROD override now raises). ROD = CENTER_SPACING - 0.025 mm (0.025 mm
# worst-case rod-rod gap, ORIGINAL spec, UNCHANGED). BRIDGE + CS held at the
# biomimetic values (NOT re-derived from the thick rod).
THICK_ROD="3.1669323693369393"
BRIDGE="1.702363930313036"
CS="3.1919323693369424"

# --- 2. Per-trial runner ----------------------------------------------------
# run_one <arm> <N> <trial_dir> <jitter_mm> [extra run_single_trial flags...]
#   arm        : sweep_layers_v2 (normal) | sweep_layers_v2_thick (thick)
#   jitter_mm  : exported as OCCT_TANGENT_JITTER_MM -> inherited by the cad_env
#                subprocess where feature_to_cad computes BRIDGE_Z_OFFSETS.
#   extra      : geometry overrides (thick pins ROD/BRIDGE/CS; normal passes none)
run_one() {
  local arm="$1" N="$2" trial="$3" jit="$4"; shift 4
  local extra=("$@")
  local t0 rc
  t0="$(date -u +%H:%M:%SZ)"
  echo "==== [${arm} N=${N} ${trial}] jitter=${jit} START ${t0} ====" | tee -a "${LOG}"
  # Overwrite: remove any existing/partial dir so run_single_trial rebuilds
  # fresh (it skip-exits if the dir already exists).
  rm -rf "${ROOT}/runs/${arm}/${trial}"
  # Build. OCCT_TANGENT_JITTER_MM is the only per-trial knob; geometry is fixed.
  OCCT_TANGENT_JITTER_MM="${jit}" conda run -n base python "${ROOT}/scripts/run_single_trial.py" \
    --morphometrics "${MORPHO}" --N "${N}" "${extra[@]}" \
    --run-name "${arm}/${trial}"
  rc=$?
  echo "==== [${arm} N=${N} ${trial}] rc=${rc} DONE $(date -u +%H:%M:%SZ) ====" | tee -a "${LOG}"
}

echo "######## rerun_all_12 START $(date -u +%Y-%m-%dT%H:%M:%SZ) ########" | tee -a "${LOG}"

# --- 3. NORMAL (biomimetic) arm — pure morphometric, NO geometry overrides --
# Documented per-trial jitter: N4,6 @ 0.0 (clean pre-jitter), N5,7,8 @ 0.15,
# N9 @ 0.50 (lower jitters leave N=9 non-watertight).
run_one sweep_layers_v2 4 trial_003_N_BRIDGE_LAYERS_4 0.0
run_one sweep_layers_v2 5 trial_004_N_BRIDGE_LAYERS_5 0.15
run_one sweep_layers_v2 6 trial_001_N_BRIDGE_LAYERS_6 0.0
run_one sweep_layers_v2 7 trial_005_N_BRIDGE_LAYERS_7 0.15
run_one sweep_layers_v2 8 trial_002_N_BRIDGE_LAYERS_8 0.15
run_one sweep_layers_v2 9 trial_006_N_BRIDGE_LAYERS_9 0.50

# --- 4. THICK-rod arm — pin ROD+BRIDGE+CS together --------------------------
# Jitter rationale (each trial gets its best / a FAIR shot — geometry unchanged):
#   N4,6,8 @ 0.50  : reproduces the three thick trials that already complete.
#   N7      @ 0.15 : builds a valid CAD where 0.50 drops bridges (CAD-resonance).
#   N5      @ 0.15 : 0.50 already 0-tet'd N5 once; 0.15 is UNTESTED for N5 -> a
#                    fair fresh attempt rather than replaying the failed config.
#   N9      @ 0.30 : both 0.15 AND 0.50 already 0-tet'd N9 (4 attempts incl.
#                    finer meshes); 0.30 is the one untested mid-value -> a fair
#                    fresh shot. If N5/N9 STILL 0-tet, that is strong evidence
#                    the 0.025 mm rod-rod gap (not the jitter) is the cause, and
#                    the gap fix becomes the next, user-gated step.
run_one sweep_layers_v2_thick 4 trial_003_N_BRIDGE_LAYERS_4 0.50 --ROD-DIAMETER "${THICK_ROD}" --BRIDGE-DIAMETER "${BRIDGE}" --CENTER-SPACING "${CS}"
run_one sweep_layers_v2_thick 5 trial_004_N_BRIDGE_LAYERS_5 0.15 --ROD-DIAMETER "${THICK_ROD}" --BRIDGE-DIAMETER "${BRIDGE}" --CENTER-SPACING "${CS}"
run_one sweep_layers_v2_thick 6 trial_001_N_BRIDGE_LAYERS_6 0.50 --ROD-DIAMETER "${THICK_ROD}" --BRIDGE-DIAMETER "${BRIDGE}" --CENTER-SPACING "${CS}"
run_one sweep_layers_v2_thick 7 trial_005_N_BRIDGE_LAYERS_7 0.15 --ROD-DIAMETER "${THICK_ROD}" --BRIDGE-DIAMETER "${BRIDGE}" --CENTER-SPACING "${CS}"
run_one sweep_layers_v2_thick 8 trial_002_N_BRIDGE_LAYERS_8 0.50 --ROD-DIAMETER "${THICK_ROD}" --BRIDGE-DIAMETER "${BRIDGE}" --CENTER-SPACING "${CS}"
run_one sweep_layers_v2_thick 9 trial_006_N_BRIDGE_LAYERS_9 0.30 --ROD-DIAMETER "${THICK_ROD}" --BRIDGE-DIAMETER "${BRIDGE}" --CENTER-SPACING "${CS}"

echo "######## rerun_all_12.sh DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ########" | tee -a "${LOG}"
