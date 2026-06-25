#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_layer_sweep_with_thick.sh
# -----------------------------------------------------------------------------
# Purpose
# -------
# Orchestrate the full N_BRIDGE_LAYERS sweep AND a matched "thick-rod"
# variant for the same N values. For each N in N_VALUES we run four steps
# in order (a -> d):
#   a) BIOMIMETIC : run the baseline biomimetic pipeline at this N (CAD +
#                   mesh + FEA), writing to sweep_layers_v2/trial_<idx>_N_BRIDGE_LAYERS_<N>.
#                   Skipped (via run_single_trial.py's idempotency guard)
#                   if the trial directory already exists -- this is how
#                   we preserve the existing N=4/6/8 trials.
#   b) THICK STL  : run make_thick_rod_variant.py on that biomimetic trial.
#                   Produces compound_enamel_lattice_thick.stl alongside
#                   the original. Idempotent in the script's _thick_variant_work
#                   sandbox -- if the sandbox already exists the script
#                   exits non-zero, which we tolerate via `|| true` and
#                   then assert the output STL exists.
#   c) THICK FEA  : run the biomimetic pipeline again at the same N, but
#                   with ROD_DIAMETER overridden to THICK_ROD. Writes to
#                   sweep_layers_v2_thick/trial_<idx>_N_BRIDGE_LAYERS_<N>.
#   d) PNGS       : render thick-vs-original cross-sections for inspection.
#
# All four steps run SEQUENTIALLY per N, and the N loop itself is
# sequential -- intentional, to avoid the OOM risk we hit on parallel
# CAD + FEA processes (each pipeline run pulls a multi-GB STL into RAM).
#
# Everything is tee'd to LOG so the user can `tail -f` it from another
# terminal while this runs in the background.
# -----------------------------------------------------------------------------

set -euo pipefail

# --- 1. Conda activation ----------------------------------------------------
# Activate the project's base env so `python` resolves to the right
# interpreter. The downstream pipeline subprocesses themselves switch into
# cad_env / sfepy_env via `conda run` internally; we only need `base` here.
eval "$(conda shell.bash hook)"
conda activate base

# --- 2. Constants -----------------------------------------------------------
# BIOMIMETIC_ROOT is the only absolute path. Everything else is derived
# from it. THICK_ROD = CENTER_SPACING (3.1919323693369424 mm in the live_001
# specimen) - 0.025 mm = 3.166932 mm; we hard-code the float because
# CENTER_SPACING is constant across the entire sweep.
BIOMIMETIC_ROOT="${MICROCT_PIPELINE_ROOT:-microct_pipeline}/biomimetic_pipeline"
MORPHO="${BIOMIMETIC_ROOT}/runs/live_001/morphometrics.json"
THICK_ROD="3.166932"
N_VALUES=(4 5 6 7 8 9)

# Output log -- everything (stdout + stderr from every step) is appended.
LOG="${BIOMIMETIC_ROOT}/runs/sweep_layers_v2/orchestrator.log"
mkdir -p "$(dirname "${LOG}")"

# Header lines so the user can tell when this invocation started.
{
  echo "=============================================================="
  echo "run_layer_sweep_with_thick.sh start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "BIOMIMETIC_ROOT=${BIOMIMETIC_ROOT}"
  echo "MORPHO=${MORPHO}"
  echo "THICK_ROD=${THICK_ROD}"
  echo "N_VALUES=${N_VALUES[*]}"
  echo "=============================================================="
} | tee -a "${LOG}"

# --- 3. Helper: find_or_alloc_trial_idx -------------------------------------
# Given an N, return either:
#   * the 3-digit index of an existing trial_<idx>_N_BRIDGE_LAYERS_<N> dir,
#   * or the next free 3-digit index >= 004 if no such dir exists yet.
# The ">= 004" floor exists because trials 000-003 are already taken
# (000 was the archived N=2 trial; 001=N=6, 002=N=8, 003=N=4 are the
# existing runs we want to preserve).
find_or_alloc_trial_idx() {
  local N="$1"
  local sweep_dir="${BIOMIMETIC_ROOT}/runs/sweep_layers_v2"

  # Look for an existing trial directory with this N. There should be
  # at most one; we pick the first lex-sorted match if more.
  local existing
  existing=$(ls -1 "${sweep_dir}" 2>/dev/null \
              | grep -E "^trial_[0-9]{3}_N_BRIDGE_LAYERS_${N}$" \
              | sort | head -n 1 || true)
  if [[ -n "${existing}" ]]; then
    # Extract the 3-digit index from trial_XXX_N_BRIDGE_LAYERS_<N>.
    echo "${existing}" | sed -E 's/^trial_([0-9]{3})_.*/\1/'
    return 0
  fi

  # No existing dir for this N -- allocate the next free index >= 004.
  # Collect all existing 3-digit indices under sweep_layers_v2, then pick
  # the smallest unused integer that is also >= 4.
  local used
  used=$(ls -1 "${sweep_dir}" 2>/dev/null \
          | grep -E "^trial_[0-9]{3}_" \
          | sed -E 's/^trial_([0-9]{3})_.*/\1/' \
          | sort -u || true)
  local i=4
  while echo "${used}" | grep -qE "^$(printf '%03d' ${i})$"; do
    i=$((i + 1))
  done
  printf '%03d' "${i}"
}

# --- 4. Main loop -----------------------------------------------------------
# For each N: (a) biomimetic, (b) thick STL, (c) thick FEA, (d) PNGs.
# Each step's stdout+stderr is tee'd into the log.
for N in "${N_VALUES[@]}"; do
  echo "" | tee -a "${LOG}"
  echo "==================== N=${N} : $(date -u +%Y-%m-%dT%H:%M:%SZ) ====================" | tee -a "${LOG}"

  # Resolve (or allocate) the 3-digit trial index for this N. This is
  # what couples the biomimetic and thick variants into the same trial
  # number -- e.g. trial_004 in both sweep_layers_v2/ and sweep_layers_v2_thick/.
  IDX=$(find_or_alloc_trial_idx "${N}")
  TRIAL_NAME="trial_${IDX}_N_BRIDGE_LAYERS_${N}"
  BIOM_TRIAL_DIR="${BIOMIMETIC_ROOT}/runs/sweep_layers_v2/${TRIAL_NAME}"
  THICK_TRIAL_DIR="${BIOMIMETIC_ROOT}/runs/sweep_layers_v2_thick/${TRIAL_NAME}"

  echo "[N=${N}] trial_name=${TRIAL_NAME}" | tee -a "${LOG}"
  echo "[N=${N}] biomimetic dir=${BIOM_TRIAL_DIR}" | tee -a "${LOG}"
  echo "[N=${N}] thick      dir=${THICK_TRIAL_DIR}" | tee -a "${LOG}"

  # -- (a) BIOMIMETIC --
  # run_single_trial.py is idempotent: if BIOM_TRIAL_DIR already exists
  # it prints "skip: already exists" and exits 0 without running anything.
  # That is exactly what we want for the existing N=4/6/8 trials.
  echo "[N=${N}] (a) BIOMIMETIC pipeline" | tee -a "${LOG}"
  python "${BIOMIMETIC_ROOT}/scripts/run_single_trial.py" \
    --morphometrics "${MORPHO}" \
    --N "${N}" \
    --run-name "sweep_layers_v2/${TRIAL_NAME}" \
    2>&1 | tee -a "${LOG}"

  # -- (b) THICK STL --
  # make_thick_rod_variant.py builds compound_enamel_lattice_thick.stl
  # inside the biomimetic trial dir. It exits non-zero if its internal
  # _thick_variant_work sandbox already exists (idempotency guard inside
  # the script), so we wrap with `|| true` and then assert the output
  # STL is present -- that file is the ground truth for "this step ran".
  echo "[N=${N}] (b) THICK STL generation" | tee -a "${LOG}"
  python "${BIOMIMETIC_ROOT}/scripts/make_thick_rod_variant.py" \
    --trial-dir "${BIOM_TRIAL_DIR}" \
    2>&1 | tee -a "${LOG}" || true

  THICK_STL="${BIOM_TRIAL_DIR}/cad/compound_enamel_lattice_thick.stl"
  if [[ ! -f "${THICK_STL}" ]]; then
    echo "[N=${N}] ERROR: thick STL missing after step (b): ${THICK_STL}" | tee -a "${LOG}"
    exit 1
  fi
  echo "[N=${N}] thick STL OK: ${THICK_STL}" | tee -a "${LOG}"

  # -- (c) THICK FEA --
  # Re-run the pipeline at the same N but with ROD_DIAMETER overridden.
  # run_single_trial.py's idempotency guard means this is a no-op if
  # the thick FEA dir already exists from a prior invocation.
  echo "[N=${N}] (c) THICK FEA pipeline (ROD_DIAMETER=${THICK_ROD})" | tee -a "${LOG}"
  python "${BIOMIMETIC_ROOT}/scripts/run_single_trial.py" \
    --morphometrics "${MORPHO}" \
    --N "${N}" \
    --ROD-DIAMETER "${THICK_ROD}" \
    --run-name "sweep_layers_v2_thick/${TRIAL_NAME}" \
    2>&1 | tee -a "${LOG}"

  # -- (d) PNGs --
  # Render thick-vs-original cross-section images for visual inspection.
  # Safe to re-run -- existing PNGs are overwritten.
  echo "[N=${N}] (d) cross-section PNGs" | tee -a "${LOG}"
  python "${BIOMIMETIC_ROOT}/scripts/cross_sections_thick_vs_original.py" \
    --trial-dir "${BIOM_TRIAL_DIR}" \
    2>&1 | tee -a "${LOG}"

  echo "[N=${N}] DONE at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
done

# --- 5. End-of-run marker ---------------------------------------------------
{
  echo ""
  echo "=============================================================="
  echo "DONE: all N values complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "=============================================================="
} | tee -a "${LOG}"
