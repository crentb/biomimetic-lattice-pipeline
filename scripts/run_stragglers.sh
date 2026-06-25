#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_stragglers.sh
# -----------------------------------------------------------------------------
# Purpose
# -------
# Build the THREE high-density OCCT "straggler" trials that fail at the default
# jitter 0.15 mm, using the global OCCT_TANGENT_JITTER_MM = 0.50 (now committed
# in mapping/bridge_mappers.py). The other 9 sweep trials are already built and
# valid at 0.15 and are left UNTOUCHED ("quick mixed" jitter strategy -- see
# the bridge_mappers comment: no single jitter is clean for all 12; 0.50 fixes
# the dense three but breaks N=7 thick, so the sweep is intentionally mixed).
#
# Stragglers (all built at jitter 0.50):
#   N=9 biomimetic -> runs/sweep_layers_v2/trial_006_N_BRIDGE_LAYERS_9
#   N=8 thick      -> runs/sweep_layers_v2_thick/trial_002_N_BRIDGE_LAYERS_8
#   N=9 thick      -> runs/sweep_layers_v2_thick/trial_006_N_BRIDGE_LAYERS_9
#
# Why build directly via run_single_trial (not the orchestrator)
# --------------------------------------------------------------
# The orchestrator's thick-STL step (make_thick_rod_variant.py) INHERITS the
# bio trial's saved BRIDGE_Z_OFFSETS. N=8 bio is at 0.15, so make_thick N=8
# would re-trigger the Null-shape crash. run_single_trial instead goes through
# map_morphometrics, which recomputes offsets using the global 0.50 constant ->
# watertight. We then COPY each thick-arm STL into its bio dir as the co-located
# visual companion (consistent at 0.50, unlike make_thick's 0.15).
#
# Robustness: set -u + pipefail but NOT -e -- a failure in one trial (or the
# optional PNG step) must not abort the rest; we echo each step's rc so the log
# tells the whole story.
# -----------------------------------------------------------------------------
set -uo pipefail

# --- 1. Conda + paths -------------------------------------------------------
eval "$(conda shell.bash hook)"
conda activate base
ROOT=${MICROCT_PIPELINE_ROOT:-microct_pipeline}/biomimetic_pipeline
MORPHO="${ROOT}/runs/live_001/morphometrics.json"
THICK=3.166932                      # CENTER_SPACING - 0.025 mm (thick rod diameter)
LOG="${ROOT}/runs/sweep_layers_v2/stragglers.log"

# All output (stdout+stderr) goes to LOG so `tail -f` shows the whole run.
exec >>"${LOG}" 2>&1
echo "================================================================"
echo "run_stragglers.sh start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "jitter = $(python -c 'from mapping.bridge_mappers import OCCT_TANGENT_JITTER_MM as J; print(J)' 2>/dev/null || echo '?') mm"
echo "================================================================"

# --- 2. Build the three stragglers sequentially -----------------------------
# (sequential = one FEA pipeline at a time; safe on the 16 GB box.)

echo ""; echo "==== (1/3) N=9 biomimetic @ 0.50 : $(date -u +%FT%TZ) ===="
python "${ROOT}/scripts/run_single_trial.py" \
  --morphometrics "${MORPHO}" --N 9 \
  --run-name sweep_layers_v2/trial_006_N_BRIDGE_LAYERS_9
echo "[N9 bio] rc=$?"

echo ""; echo "==== (2/3) N=8 thick @ 0.50 : $(date -u +%FT%TZ) ===="
python "${ROOT}/scripts/run_single_trial.py" \
  --morphometrics "${MORPHO}" --N 8 --ROD-DIAMETER "${THICK}" \
  --run-name sweep_layers_v2_thick/trial_002_N_BRIDGE_LAYERS_8
echo "[N8 thick] rc=$?"

echo ""; echo "==== (3/3) N=9 thick @ 0.50 : $(date -u +%FT%TZ) ===="
python "${ROOT}/scripts/run_single_trial.py" \
  --morphometrics "${MORPHO}" --N 9 --ROD-DIAMETER "${THICK}" \
  --run-name sweep_layers_v2_thick/trial_006_N_BRIDGE_LAYERS_9
echo "[N9 thick] rc=$?"

# --- 3. Co-locate thick STLs in the bio dirs (consistent at 0.50) -----------
# Copy each thick-arm STL into the matching biomimetic trial's cad/ as the
# visual companion. (N=8 bio had no co-located thick STL -- the make_thick gap.)
echo ""; echo "==== co-locate thick STLs : $(date -u +%FT%TZ) ===="
n8_thick_stl="${ROOT}/runs/sweep_layers_v2_thick/trial_002_N_BRIDGE_LAYERS_8/cad/compound_enamel_lattice.stl"
n9_thick_stl="${ROOT}/runs/sweep_layers_v2_thick/trial_006_N_BRIDGE_LAYERS_9/cad/compound_enamel_lattice.stl"
[ -f "${n8_thick_stl}" ] && cp "${n8_thick_stl}" "${ROOT}/runs/sweep_layers_v2/trial_002_N_BRIDGE_LAYERS_8/cad/compound_enamel_lattice_thick.stl" && echo "[N8 co-located thick STL copied]" || echo "[N8 co-located copy SKIPPED (thick STL missing)]"
[ -f "${n9_thick_stl}" ] && cp "${n9_thick_stl}" "${ROOT}/runs/sweep_layers_v2/trial_006_N_BRIDGE_LAYERS_9/cad/compound_enamel_lattice_thick.stl" && echo "[N9 co-located thick STL copied]" || echo "[N9 co-located copy SKIPPED (thick STL missing)]"

# --- 4. Cross-section PNGs (optional, non-blocking) -------------------------
echo ""; echo "==== cross-section PNGs : $(date -u +%FT%TZ) ===="
python "${ROOT}/scripts/cross_sections_thick_vs_original.py" \
  --trial-dir "${ROOT}/runs/sweep_layers_v2/trial_002_N_BRIDGE_LAYERS_8"
echo "[N8 PNGs] rc=$?"
python "${ROOT}/scripts/cross_sections_thick_vs_original.py" \
  --trial-dir "${ROOT}/runs/sweep_layers_v2/trial_006_N_BRIDGE_LAYERS_9"
echo "[N9 PNGs] rc=$?"

echo ""; echo "==== run_stragglers.sh DONE: $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
