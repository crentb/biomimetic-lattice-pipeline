#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_thick_rerun.sh
# -----------------------------------------------------------------------------
# Purpose
# -------
# Re-build the ENTIRE thick arm (N=4..9) at a single, uniform, OCCT-buildable
# rod diameter so the thick comparison is a clean one-diameter series.
#
# Why the diameter changed
# ------------------------
# The original "barely-touching" thick diameter (CENTER_SPACING - 0.025 =
# 3.167 mm, 25 um gap) is OCCT-unbuildable at the dense high-N cases (N=8/N=9
# silently drop all bridges). A shrink probe (scripts/probe_thick_shrink.py,
# full watertight+bridges integrity) found D = CENTER_SPACING - 0.1 = 3.0919 mm
# (0.10 mm gap) is the LARGEST diameter that builds valid at ALL N=4..9 (the
# response is non-monotonic: 0.2 mm gap FAILS N=9, but 0.1 and 0.3 mm work).
# So the whole thick arm adopts 3.0919 mm (0.10 mm worst-case gap).
#
# Built DIRECTLY via run_single_trial (NOT the orchestrator / make_thick, which
# inherit the bio trial's offsets). Uses the current global jitter (0.50);
# the probe confirmed D=3.0919 passes at 0.50 for the hard N=8/N=9, and the
# per-trial cad_integrity check gates every build (watertight + bridges) before
# any mesh/FEA, so a bad trial aborts cheaply at the CAD stage.
#
# Pre-req: the 6 thick trial dirs must already be DELETED (run_single_trial
# skips existing dirs). Deletion is done separately before launching this.
#
# set -u + pipefail but NOT -e: one trial's failure must not abort the rest.
# -----------------------------------------------------------------------------
set -uo pipefail
eval "$(conda shell.bash hook)"
conda activate base

ROOT=${MICROCT_PIPELINE_ROOT:-microct_pipeline}/biomimetic_pipeline
MORPHO="${ROOT}/runs/live_001/morphometrics.json"
# Original "barely-touching" thick spec. It is buildable after all -- the earlier
# shrink to 3.092 was an artifact of the inflated-bridge bug, not the diameter.
# We GROW the rod but PIN the biomimetic BRIDGE and CENTER_SPACING so the fixed
# mapper keeps them (no silent 0.80*thick re-derivation).
THICK=3.1669323693369393          # CENTER_SPACING - 0.025 mm = 0.025 mm worst-case gap (original spec)
BIOMIM_BRIDGE=1.702363930313036   # keep biomimetic bridge (NOT 0.80*thick rod)
BIOMIM_CS=3.1919323693369424      # keep biomimetic ring spacing
LOG="${ROOT}/runs/sweep_layers_v2_thick/thick_rerun.log"

# Canonical N -> trial index map (same indices as the bio arm).
declare -A IDX=( [4]=003 [5]=004 [6]=001 [7]=005 [8]=002 [9]=006 )

exec >>"${LOG}" 2>&1
echo "================================================================"
echo "run_thick_rerun.sh start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "thick ROD_DIAMETER = ${THICK} mm (CENTER_SPACING - 0.025 = 0.025 mm gap, ORIGINAL spec); BRIDGE=${BIOMIM_BRIDGE} CS=${BIOMIM_CS} (biomimetic, pinned)"
echo "global jitter = $(python -c 'from mapping.bridge_mappers import OCCT_TANGENT_JITTER_MM as J; print(J)' 2>/dev/null || echo '?') mm"
echo "================================================================"

# --- 1. Build all six thick trials (sequential; safe on 16 GB) --------------
for N in 4 5 6 7 8 9; do
  trial="trial_${IDX[$N]}_N_BRIDGE_LAYERS_${N}"
  echo ""; echo "==== N=${N} thick -> ${trial} : $(date -u +%FT%TZ) ===="
  python "${ROOT}/scripts/run_single_trial.py" \
    --morphometrics "${MORPHO}" --N "${N}" \
    --ROD-DIAMETER "${THICK}" --BRIDGE-DIAMETER "${BIOMIM_BRIDGE}" --CENTER-SPACING "${BIOMIM_CS}" \
    --run-name "sweep_layers_v2_thick/${trial}"
  echo "[N=${N} thick] rc=$?"
done

# --- 2. Co-locate each new thick STL into its bio dir (overwrites old/bad) ---
echo ""; echo "==== co-locate thick STLs into bio dirs : $(date -u +%FT%TZ) ===="
for N in 4 5 6 7 8 9; do
  trial="trial_${IDX[$N]}_N_BRIDGE_LAYERS_${N}"
  src="${ROOT}/runs/sweep_layers_v2_thick/${trial}/cad/compound_enamel_lattice.stl"
  dst="${ROOT}/runs/sweep_layers_v2/${trial}/cad/compound_enamel_lattice_thick.stl"
  if [ -f "${src}" ]; then cp "${src}" "${dst}" && echo "[N=${N} co-located]"; else echo "[N=${N} co-locate SKIPPED (no thick STL)]"; fi
done

# --- 3. Regenerate cross-section comparison PNGs (non-blocking) --------------
echo ""; echo "==== cross-section PNGs : $(date -u +%FT%TZ) ===="
for N in 4 5 6 7 8 9; do
  trial="trial_${IDX[$N]}_N_BRIDGE_LAYERS_${N}"
  python "${ROOT}/scripts/cross_sections_thick_vs_original.py" \
    --trial-dir "${ROOT}/runs/sweep_layers_v2/${trial}"
  echo "[N=${N} PNGs] rc=$?"
done

echo ""; echo "==== run_thick_rerun.sh DONE: $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
