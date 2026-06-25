#!/usr/bin/env python
"""
run_thick_trial_tall.py
=======================

Purpose
-------
Build ONE thick-rod lattice trial at an OVERRIDDEN ``ENAMEL_THICKNESS`` (a
taller specimen / longer enamel rods) and run it through the full pipeline
(CAD -> mesh -> FEA). It is the experiment for the hypothesis that *lengthening
the rods* relieves the N=9 thick volume-mesh failure WITHOUT shrinking the
canonical 3.167 mm rod.

Why this exists
---------------
Canonical thick N=9 (ROD 3.167 mm, rod-rod gap 0.025 mm) will not volume-mesh:
its 9 bridge layers stack only ~0.21 mm apart vertically while its rods sit
0.025 mm apart horizontally, and gmsh cannot seat a tetrahedron in a junction
pocket pinched tight in BOTH directions at once (0 tets, all 3 algorithms).
The sliver-face hypothesis was *falsified* empirically
(``scripts/diagnose_n9_slivers.py``, 2026-06-03): the two MESHABLE controls
(N=8 @ 3.167, N=9 @ 2.75) carry the same ~800-970 near-degenerate sliver faces
as the failing case, so slivers are not the cause. The data instead show that
relieving EITHER pinch restores meshability:
  * N=8 relaxes the VERTICAL bridge gap to 0.48 mm  -> meshes.
  * the 2.75 rod relaxes the HORIZONTAL rod gap to 0.44 mm -> meshes.
This script relieves the VERTICAL pinch a third way, at the canonical rod:
``ENAMEL_THICKNESS`` grows the safe bridge band 1:1
(band = ENAMEL_THICKNESS - 3*PLATE_OVERLAP - 2*margin, see
``mapping/bridge_mappers.compute_safe_bridge_elevations``), so the 9 layers
spread out and dz rises above the meshable threshold -- keeping ROD = 3.167 mm
and the 0.025 mm near-touching gap BOTH intact. A tall N=9 @ 3.167 is then
geometrically analogous to N=8 @ 3.167 (same 0.025 mm rod gap, comparable
vertical clearance), which already meshes -- so it is predicted to mesh.

The override reaches the bridge mapper correctly (this is the make-or-break
point): ``feature_to_cad.map_morphometrics`` applies ``extra_overrides`` (incl.
``ENAMEL_THICKNESS``) at line ~114, THEN reads ``ENAMEL_THICKNESS`` into the
bridge-elevation computation at lines ~215/~262. So the bridges genuinely
spread; they are NOT left at the old H=20 positions with bare rod added on top.
``run_single_trial.py`` exposes no height flag and must not be edited in place
(project rule), hence this minimal sibling driver.

Inputs (CLI)
------------
  --enamel-thickness  Overridden specimen height H (mm). Default 24.0 (up from
                      the pipeline default 20.0). At N=9 this yields a predicted
                      bridge dz ~ 2.41 mm / surface gap ~ 0.71 mm -- comfortably
                      above the meshable threshold (which lies between the 0.21
                      that fails and the 0.48 that meshes).
  --N                 N_BRIDGE_LAYERS. Default 9 (the failing case).
  --ROD-DIAMETER / --BRIDGE-DIAMETER / --CENTER-SPACING
                      The thick contract. The mapper RAISES on a lone ROD
                      override (historical 2.47 mm bridge-inflation guard), so
                      all three are pinned together. Defaults = canonical thick:
                      3.1669.. / 1.70236.. / 3.19193.. mm.
  --run-name          Run dir under runs/ (default a scratch _n9_tall_test/...
                      path so the canonical trial_006 @ rod 2.75 is untouched).
  --morphometrics     Canonical specimen (default runs/live_001/...).
  --objective         Forwarded to run_pipeline (default crack_deflection,
                      matching the sweep).

Outputs
-------
  runs/<run-name>/ with cad/, mesh/, fea/, metrics.json (full pipeline). The
  pipeline gates on CAD integrity before meshing and on a non-empty mesh before
  FEA, so a bridge-drop or 0-tet failure aborts cheaply (no wasted FEA). The
  decisive verdict is in mesh/mesh_run.log (tet count) and, if it meshes,
  metrics.json (E_effective_MPa).

Side effects / non-obvious behavior
-----------------------------------
  * OCCT_TANGENT_JITTER_MM (env) still governs the SEPARATE OCCT bridge-drop
    bug. A new H relocates every bridge, so the tangent resonance is a fresh
    roll; start at 0.50 and retry other jitters if CAD integrity drops bridges.
  * skip_probe=True (trust the conda envs, like run_single_trial).
  * Writes only under runs/<run-name>/; does NOT touch any canonical trial.
"""

from __future__ import annotations

# --- stdlib + path bootstrap ------------------------------------------------
import argparse
import os
import sys
from pathlib import Path

# This script lives at <biomimetic_root>/scripts/, so root = parents[1]. Put it
# on sys.path so `import orchestration` resolves exactly as it does for
# run_single_trial.py / run_pipeline.py.
THIS = Path(__file__).resolve()
BIOMIMETIC_ROOT = THIS.parent.parent
if str(BIOMIMETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(BIOMIMETIC_ROOT))

from biomimetic_pipeline.orchestration import (
    pipeline,  # noqa: E402 (post-sys.path import, intentional)
)

# --- Canonical thick contract (mm) ------------------------------------------
# Pinned so a bare invocation builds the maximally-thick / near-touching
# geometry the sweep uses. BRIDGE and CENTER_SPACING are the FIXED biomimetic
# values (identical across all 12 trials); ROD = CENTER_SPACING - 0.025 is the
# thick "near-touching" rod (0.025 mm rod-rod gap). These match
# the cad_params.json on disk.
THICK_ROD_MM = 3.1669323693369393  # CENTER_SPACING - 0.025 mm
BIO_BRIDGE_MM = 1.702363930313036  # biomimetic bridge diameter (fixed)
BIO_CENTER_SPACING_MM = 3.1919323693369424  # biomimetic ring pitch (fixed)

# --- Bridge-band constants (mm) for the PREDICTED dz/gap log line -----------
# Mirrored from LATTICE_CAD_STOCK_DEFAULTS + the mapper so we can print the
# intended bridge clearance before the (slow) build. Source of truth remains
# the emitted BRIDGE_Z_OFFSETS in cad_params.json -- this is informational only.
PLATE_OVERLAP_MM = 0.5  # LATTICE_CAD_STOCK_DEFAULTS["PLATE_OVERLAP"]
PLATE_CLEARANCE_MM = 0.5  # DEFAULT_PLATE_CLEARANCE_MM (mapper's plate clearance)


def _predicted_bridge_gap(enamel_mm: float, n: int, bridge_mm: float, jitter_mm: float) -> tuple:
    """Predict (dz, surface_gap) in mm for the spread bridge layers.

    Mirrors ``compute_safe_bridge_elevations`` for junction_sphere_factor = 0
    (JUNCTION_SPHERE_FACTOR auto-clamps to 0 for this geometry -- no plate room).
    Informational only; verify against cad_params.json BRIDGE_Z_OFFSETS.
        margin     = bridge_half + plate_clearance
        band_full  = enamel - 3*plate_overlap - 2*margin   (= safe_z_max - safe_z_min)
        elev_range = band_full - jitter   (jitter headroom reserved at the top)
        dz         = elev_range / (n - 1)                  (vertical layer pitch)
        gap        = dz - bridge_diameter                  (clearance between cylinders)
    """
    margin = 0.5 * bridge_mm + PLATE_CLEARANCE_MM
    band_full = enamel_mm - 3.0 * PLATE_OVERLAP_MM - 2.0 * margin
    elev_range = band_full - jitter_mm
    dz = elev_range / max(1, (n - 1))
    return dz, dz - bridge_mm


def main() -> int:
    # --- 1. Parse CLI ------------------------------------------------------
    ap = argparse.ArgumentParser(
        description=(
            "Build one thick trial at an overridden ENAMEL_THICKNESS "
            "(taller rods) and run the full pipeline."
        )
    )
    ap.add_argument(
        "--morphometrics",
        type=Path,
        default=BIOMIMETIC_ROOT / "runs" / "live_001" / "morphometrics.json",
        help="canonical specimen (default live_001).",
    )
    # NOTE: the default run-name encodes H24; pass an explicit --run-name if you
    # change --enamel-thickness so the dir name does not lie about the height.
    ap.add_argument(
        "--run-name",
        default="_n9_tall_test/N9_thick_rod3167_H24",
        help="run dir under runs/ (default scratch; never the canonical trial).",
    )
    ap.add_argument(
        "--N", type=int, default=9, help="N_BRIDGE_LAYERS (default 9, the failing case)."
    )
    ap.add_argument(
        "--enamel-thickness",
        dest="enamel_thickness",
        type=float,
        default=24.0,
        help="overridden specimen height H in mm (default 24.0; pipeline default is 20.0).",
    )
    ap.add_argument(
        "--ROD-DIAMETER",
        dest="rod_diameter",
        type=float,
        default=THICK_ROD_MM,
        help="thick rod diameter (mm); default canonical 3.1669.",
    )
    ap.add_argument(
        "--BRIDGE-DIAMETER",
        dest="bridge_diameter",
        type=float,
        default=BIO_BRIDGE_MM,
        help="bridge diameter (mm); default biomimetic 1.70236 (pinned with ROD to avoid re-derivation).",
    )
    ap.add_argument(
        "--CENTER-SPACING",
        dest="center_spacing",
        type=float,
        default=BIO_CENTER_SPACING_MM,
        help="ring pitch (mm); default biomimetic 3.19193.",
    )
    ap.add_argument(
        "--objective",
        default="crack_deflection",
        help="objective forwarded to run_pipeline (default crack_deflection).",
    )
    args = ap.parse_args()

    # --- 2. Idempotency guard ---------------------------------------------
    # Refuse to clobber an existing dir so a re-run after interruption is safe.
    # Delete the dir by hand to force a rebuild. (Same contract as run_single_trial.)
    target_dir = BIOMIMETIC_ROOT / "runs" / args.run_name
    if target_dir.exists():
        print(f"skip: already exists {target_dir}")
        return 0

    # --- 3. Build extra_overrides (thick contract + height) ----------------
    # ENAMEL_THICKNESS is the new lever: it flows through extra_overrides into
    # the bridge mapper (feature_to_cad:114 -> :215 -> :262) and SPREADS the
    # bridge layers. ROD/BRIDGE/CS are pinned together because a lone ROD
    # override raises (the historical bridge-inflation guard).
    extra_overrides = {
        "N_BRIDGE_LAYERS": int(args.N),
        "ROD_DIAMETER": float(args.rod_diameter),
        "BRIDGE_DIAMETER": float(args.bridge_diameter),
        "CENTER_SPACING": float(args.center_spacing),
        "ENAMEL_THICKNESS": float(args.enamel_thickness),
    }

    # --- 4. Log the intent (predicted bridge geometry) ---------------------
    # Read the effective OCCT jitter the mapper will use (env override, else the
    # mapper default 0.50) purely to PREDICT dz/gap for the run record.
    jitter = float(os.environ.get("OCCT_TANGENT_JITTER_MM", "0.50") or "0.50")
    dz, gap = _predicted_bridge_gap(args.enamel_thickness, args.N, args.bridge_diameter, jitter)
    print(
        f"[tall-trial] N={args.N} ROD={args.rod_diameter:.4f} "
        f"BRIDGE={args.bridge_diameter:.4f} CS={args.center_spacing:.4f} "
        f"ENAMEL_THICKNESS={args.enamel_thickness:.3f} (pipeline default 20.0) "
        f"jitter={jitter:.2f}"
    )
    print(
        f"[tall-trial] predicted bridge dz={dz:.4f} mm  surface gap={gap:.4f} mm "
        f"(meshable threshold is between 0.21 mm [fails] and 0.48 mm [meshes]; "
        f"verify vs cad_params.json BRIDGE_Z_OFFSETS)"
    )
    print(f"[tall-trial] run dir -> {target_dir}")

    # --- 5. Run the full pipeline -----------------------------------------
    # Same call shape as run_single_trial. The pipeline gates on CAD integrity
    # (watertight + bridge presence) and a non-empty mesh, so a bridge-drop or
    # 0-tet failure aborts before any FEA compute is spent.
    result = pipeline.run_pipeline(
        morphometrics_path=args.morphometrics,
        run_name=args.run_name,
        objective_name=args.objective,
        extra_overrides=extra_overrides,
        skip_probe=True,
    )

    # --- 6. Report ---------------------------------------------------------
    print(f"Run complete: {result.run_dir}")
    print(f"Metrics: {result.metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
