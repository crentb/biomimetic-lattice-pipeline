#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_layer_sweep_with_thick_batched.sh
# -----------------------------------------------------------------------------
# Purpose
# -------
# Same end goal as run_layer_sweep_with_thick.sh (biomimetic + thick FEA across
# N_BRIDGE_LAYERS=4..9), but with BATCHED PARALLELISM: N values are processed
# BATCH_SIZE-at-a-time. Within a batch, each N runs its own (a -> b -> c -> d)
# sequence concurrently in the background; we `wait` on the batch before
# launching the next.
#
# Why a separate script instead of editing the sequential one
# -----------------------------------------------------------
# Project convention is "new files, not in-place edits" so the sequential
# script stays available as a known-good fallback. The two scripts share the
# four-step per-N recipe; only the outer loop differs.
#
# Why batches of 3 specifically
# -----------------------------
# The user's box has 10 physical cores. A single biomimetic / thick FEA
# saturates roughly 1-2 cores (sfepy is largely single-threaded; gmsh
# meshing burns ~2 briefly). Running 3 N values concurrently means at peak
# ~6 cores are busy with FEA, which leaves ~4 cores free for the OS, the
# user's interactive work, and the orchestrator itself. Anything beyond
# 3 risked OOM (each pipeline holds a multi-GB CAD/STL in RAM).
#
# Trial-index pre-allocation
# --------------------------
# When parallel jobs all need to allocate a fresh trial_<idx>_... directory,
# `ls`-based allocation has a race (every concurrent call sees the same
# free index and tries to claim it). To avoid that we HARD-CODE the index
# map up front: existing trials keep their numbers (4->003, 6->001, 8->002),
# new trials get the next free indices in N-order (5->004, 7->005, 9->006).
# If you re-run after the sweep has progressed, the idempotency guards
# inside run_single_trial.py (skip-if-exists) make this safe to re-execute.
#
# Logging
# -------
# Top-level events go to LOG (orchestrator_batched.log).
# Per-N output goes to PER_N_LOG_DIR/N<value>.log so concurrent stdout
# streams don't interleave. After each batch we cat the per-N logs into
# the top-level log so a single `tail -f LOG` still tells the whole story
# without flipping between files.
# -----------------------------------------------------------------------------

# We intentionally do NOT use `set -e`: if one N fails in a batch, the rest
# of the batch should still run to completion. `set -u` catches typos in
# variable names; `pipefail` ensures `tee` doesn't mask upstream failures.
set -uo pipefail

# --- 1. Conda activation ----------------------------------------------------
# Activate base so `python` is the project's interpreter; the pipeline's own
# subprocesses switch into cad_env / sfepy_env internally via `conda run`.
eval "$(conda shell.bash hook)"
conda activate base

# --- 2. Constants -----------------------------------------------------------
# THICK_ROD = CENTER_SPACING (3.191932 mm in the live_001 specimen) - 0.025 mm
# clearance = 3.166932 mm. Constant across the sweep because CENTER_SPACING
# does not vary with N_BRIDGE_LAYERS.
BIOMIMETIC_ROOT="${MICROCT_PIPELINE_ROOT:-microct_pipeline}/biomimetic_pipeline"
MORPHO="${BIOMIMETIC_ROOT}/runs/live_001/morphometrics.json"
THICK_ROD="3.166932"
N_VALUES=(4 5 6 7 8 9)
BATCH_SIZE=3

# Top-level log (one combined story). Per-N logs are below.
LOG="${BIOMIMETIC_ROOT}/runs/sweep_layers_v2/orchestrator_batched.log"
PER_N_LOG_DIR="${BIOMIMETIC_ROOT}/runs/sweep_layers_v2/orchestrator_batched_per_n"
mkdir -p "$(dirname "${LOG}")" "${PER_N_LOG_DIR}"

# Header so the user can tell when this invocation started, distinct from
# the sequential orchestrator's log.
{
  echo "=============================================================="
  echo "run_layer_sweep_with_thick_batched.sh start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "BIOMIMETIC_ROOT=${BIOMIMETIC_ROOT}"
  echo "MORPHO=${MORPHO}"
  echo "THICK_ROD=${THICK_ROD}"
  echo "N_VALUES=${N_VALUES[*]}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "PER_N_LOG_DIR=${PER_N_LOG_DIR}"
  echo "=============================================================="
} | tee -a "${LOG}"

# --- 3. Pre-allocated trial index map --------------------------------------
# See "Trial-index pre-allocation" in the header for the rationale. Edit
# this if the sweep_layers_v2 directory layout changes.
declare -A IDX_FOR_N
IDX_FOR_N[4]="003"   # existing biomimetic trial (preserved)
IDX_FOR_N[6]="001"   # existing biomimetic trial (preserved)
IDX_FOR_N[8]="002"   # existing biomimetic trial (preserved)
IDX_FOR_N[5]="004"   # new biomimetic trial
IDX_FOR_N[7]="005"   # new biomimetic trial
IDX_FOR_N[9]="006"   # new biomimetic trial

# --- 4. process_N: one N's full (a -> b -> c -> d) chain --------------------
# Called in the background for each N within a batch. All stdout/stderr is
# redirected into the per-N log so concurrent runs don't interleave. Exit
# status is propagated to the caller via `wait $pid` so we can report
# per-N success/failure after the batch completes.
process_N() {
  local N="$1"
  local idx="${IDX_FOR_N[${N}]}"
  local trial_name="trial_${idx}_N_BRIDGE_LAYERS_${N}"
  local biom_trial_dir="${BIOMIMETIC_ROOT}/runs/sweep_layers_v2/${trial_name}"
  local n_log="${PER_N_LOG_DIR}/N${N}.log"

  {
    echo ""
    echo "==================== N=${N} : $(date -u +%Y-%m-%dT%H:%M:%SZ) ===================="
    echo "[N=${N}] trial_name=${trial_name}"
    echo "[N=${N}] biomimetic dir=${biom_trial_dir}"
    echo "[N=${N}] thick      dir=${BIOMIMETIC_ROOT}/runs/sweep_layers_v2_thick/${trial_name}"

    # -- (a) BIOMIMETIC --
    # run_single_trial.py is idempotent: prints "skip" and exits 0 if the
    # target dir already exists. Existing N=4/6/8 trials short-circuit here.
    echo "[N=${N}] (a) BIOMIMETIC pipeline at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    python "${BIOMIMETIC_ROOT}/scripts/run_single_trial.py" \
      --morphometrics "${MORPHO}" \
      --N "${N}" \
      --run-name "sweep_layers_v2/${trial_name}"
    local rc_a=$?
    echo "[N=${N}] (a) returned rc=${rc_a}"
    if [[ ${rc_a} -ne 0 ]]; then
      echo "[N=${N}] ABORT: biomimetic step failed; skipping (b)/(c)/(d)"
      return ${rc_a}
    fi

    # -- (b) THICK STL --
    # make_thick_rod_variant.py exits non-zero if the _thick_variant_work
    # sandbox already exists, so we tolerate that and confirm the output
    # STL is present as the ground-truth "this step ran" sentinel.
    echo "[N=${N}] (b) THICK STL generation at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    python "${BIOMIMETIC_ROOT}/scripts/make_thick_rod_variant.py" \
      --trial-dir "${biom_trial_dir}" || true

    local thick_stl="${biom_trial_dir}/cad/compound_enamel_lattice_thick.stl"
    if [[ ! -f "${thick_stl}" ]]; then
      echo "[N=${N}] ERROR: thick STL missing after step (b): ${thick_stl}"
      return 1
    fi
    echo "[N=${N}] thick STL OK: ${thick_stl}"

    # -- (c) THICK FEA --
    # Same N, ROD_DIAMETER overridden to THICK_ROD. Idempotent on existing
    # sweep_layers_v2_thick/<trial_name> dir.
    echo "[N=${N}] (c) THICK FEA pipeline (ROD_DIAMETER=${THICK_ROD}) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    python "${BIOMIMETIC_ROOT}/scripts/run_single_trial.py" \
      --morphometrics "${MORPHO}" \
      --N "${N}" \
      --ROD-DIAMETER "${THICK_ROD}" \
      --run-name "sweep_layers_v2_thick/${trial_name}"
    local rc_c=$?
    echo "[N=${N}] (c) returned rc=${rc_c}"
    if [[ ${rc_c} -ne 0 ]]; then
      echo "[N=${N}] WARNING: thick FEA failed; continuing to (d) PNGs anyway"
    fi

    # -- (d) PNGs --
    # Side-by-side cross-section PNGs vs the original. Idempotent (overwrites
    # PNGs of the same name).
    echo "[N=${N}] (d) cross-section PNGs at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    python "${BIOMIMETIC_ROOT}/scripts/cross_sections_thick_vs_original.py" \
      --trial-dir "${biom_trial_dir}"
    local rc_d=$?
    echo "[N=${N}] (d) returned rc=${rc_d}"

    echo "[N=${N}] DONE at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >> "${n_log}" 2>&1
}

# --- 5. Batched main loop --------------------------------------------------
# Launch BATCH_SIZE process_N's in parallel as background jobs, wait for the
# batch to finish, then start the next batch. After each batch we cat the
# per-N logs into the top-level log so `tail -f $LOG` shows the whole story.
TOTAL=${#N_VALUES[@]}
batch_idx=0

for ((start=0; start<TOTAL; start+=BATCH_SIZE)); do
  batch_idx=$((batch_idx + 1))
  batch_ns=()
  pids=()

  # Launch all members of this batch
  for ((offset=0; offset<BATCH_SIZE; offset++)); do
    i=$((start + offset))
    if [[ ${i} -ge ${TOTAL} ]]; then break; fi
    N=${N_VALUES[${i}]}
    batch_ns+=("${N}")
    process_N "${N}" &
    pids+=($!)
    echo "[batched] launched N=${N} as PID $!" | tee -a "${LOG}"
  done

  echo "[batched] batch ${batch_idx} launched: N=(${batch_ns[*]}); waiting..." | tee -a "${LOG}"

  # Wait for every PID in the batch; collect each one's rc into a parallel array.
  declare -a rcs=()
  for pid in "${pids[@]}"; do
    wait "${pid}"
    rcs+=($?)
  done

  # Per-N rc summary (uses parallel-index alignment of batch_ns + rcs).
  for ((k=0; k<${#batch_ns[@]}; k++)); do
    echo "[batched] batch ${batch_idx} N=${batch_ns[${k}]} rc=${rcs[${k}]}" | tee -a "${LOG}"
  done

  # Fold per-N logs into the top-level log so tail -f LOG sees them.
  for N in "${batch_ns[@]}"; do
    n_log="${PER_N_LOG_DIR}/N${N}.log"
    if [[ -f "${n_log}" ]]; then
      {
        echo ""
        echo "----- begin per-N log: ${n_log} -----"
        cat "${n_log}"
        echo "----- end per-N log: ${n_log} -----"
        echo ""
      } >> "${LOG}"
    fi
  done

  echo "[batched] batch ${batch_idx} complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
done

# --- 6. End-of-run marker --------------------------------------------------
{
  echo ""
  echo "=============================================================="
  echo "DONE: all batches complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Per-N logs are at: ${PER_N_LOG_DIR}/N<value>.log"
  echo "=============================================================="
} | tee -a "${LOG}"
