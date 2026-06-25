#!/usr/bin/env python
"""
probe_n8_thick_offsets.py
=========================

Purpose
-------
The N=8 *thick-rod* lattice variant (ROD_DIAMETER = 3.166932 mm, the
"barely-not-touching" 25 um-clearance geometry) fails to build in the stock
CadQuery/OCCT generator: the final `unified.union(top_plate)` boolean returns a
`Null TopoDS_Shape`, aborting CAD. This is the long-standing N=8 OCCT tangent
gremlin reproducing in the fat-rod case -- the 0.15 mm bridge-elevation jitter
that rescued the *biomimetic* (thin-rod) N=8 build is not enough for the thick
rods.

This script empirically searches for a BRIDGE_Z_OFFSETS layout that lets the
thick N=8 geometry build cleanly, CAD-only (no mesh/FEA), so we can answer:
"does an offset/jitter tweak rescue the thick high-N build, and if so which one?"

Why this exists (vs. editing make_thick_rod_variant.py)
-------------------------------------------------------
`make_thick_rod_variant.py` inherits BRIDGE_Z_OFFSETS verbatim from the trial's
biomimetic sidecar and offers no override hook. Rather than edit that
(project rule: never edit existing scripts in place), this standalone probe
loads the *exact* thick params that failed and re-runs the stock CAD with
different offset arrays. The stock CAD module is imported via cad_runner and
never modified.

Two hypotheses tested
---------------------
  (a) intermediate-layer resonance -> a LARGER uniform jitter. NOTE: the
      production jitter pins the TOP bridge at safe_z_max regardless of jitter
      magnitude (it reserves headroom then adds it back), so larger jitter only
      moves the bottom/intermediate layers. If the culprit is an intermediate
      layer sitting on an OCCT tangent, more jitter should escape it.
  (b) top-bridge-too-close-to-plate -> pull JUST the top bridge DOWN, widening
      the bridge->top-plate gap. The fat rods crowd the plate region; a top
      bridge at z=17.649 (top surface 18.50 mm, plate underside 19.0 mm) may be
      the degenerate contact OCCT chokes on.

Inputs (CLI)
------------
  --baseline-params : path to the failed thick cad_params_used.json
                      (default: the N=8 thick sandbox that errored).
  --out-root        : directory to hold per-strategy CAD sandboxes
                      (default: runs/n8_thick_offset_probe).

Outputs
-------
  <out-root>/<strategy>/compound_enamel_lattice.stl (+ .step, sidecar, log)
  <out-root>/probe_results.json   -- machine-readable result table
  stdout                          -- human-readable PASS/FAIL table

Side effects / non-obvious behavior
-----------------------------------
  * Each strategy is an independent stock-CAD run (~1-3 min, cad_env subprocess
    spawned by cad_runner). Runs sequentially to bound memory (safe to run
    alongside one background FEA; do NOT fan these out).
  * A strategy is BUILD_OK if cad_runner returns without raising (no Null
    shape). WATERTIGHT is a secondary trimesh check on the produced STL.
  * Offsets are constructed directly (not via mapping.bridge_mappers) so we can
    probe layouts the production jitter cannot express (e.g. lowered top bridge).
"""

from __future__ import annotations

# --- Standard library --------------------------------------------------------
import argparse
import json
import sys
import traceback
from pathlib import Path

# --- Third party -------------------------------------------------------------
import numpy as np
import trimesh

# --- Pipeline path setup -----------------------------------------------------
# Make `generators.cad_runner` importable without installing the package; this
# script lives in <root>/scripts/, so the parent of the parent is the root.
THIS = Path(__file__).resolve()
BIOMIMETIC_ROOT = THIS.parent.parent
if str(BIOMIMETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(BIOMIMETIC_ROOT))

from biomimetic_pipeline.generators import (
    cad_runner,  # noqa: E402  (deliberate post-sys.path import)
)


def build_strategies(baseline_offsets: list[float]) -> list[tuple[str, list[float]]]:
    """Return a list of (name, BRIDGE_Z_OFFSETS) layouts to try for N=8 thick.

    The baseline is the production-jittered layout that FAILS, included as a
    sanity check (it should reproduce the Null-shape failure). All other
    layouts are derived from the baseline's safe band:

        safe_z_min = baseline[0] - 0.15   (undo the production jitter)
        safe_z_max = baseline[-1]         (top is pinned here in production)

    Units: all elevations in mm, measured from the lattice base (z=0).
    """
    n = len(baseline_offsets)  # 8 bridge layers for N=8
    safe_z_min = baseline_offsets[0] - 0.15  # production jitter was +0.15
    safe_z_max = baseline_offsets[-1]  # production pins top here

    def jitter(J: float) -> list[float]:
        # Production-style jitter: uniform over [safe_z_min, safe_z_max - J]
        # then shift up by J. Equivalent to linspace(safe_z_min + J, safe_z_max).
        # Top stays pinned at safe_z_max for ANY J -- only lower layers move.
        return list(np.linspace(safe_z_min + J, safe_z_max, n))

    def top_down(drop: float) -> list[float]:
        # Keep the production-jittered layout but pull ONLY the top bridge down
        # by `drop` mm, widening the bridge->top-plate gap (tests hypothesis b).
        offs = list(baseline_offsets)
        offs[-1] = baseline_offsets[-1] - drop
        return offs

    def jitter_and_top_down(J: float, drop: float) -> list[float]:
        # Combine: larger jitter on the lower layers AND a lowered top bridge.
        offs = jitter(J)
        offs[-1] = offs[-1] - drop
        return offs

    return [
        ("baseline_J0.15", list(baseline_offsets)),  # known FAIL (sanity)
        ("jitter_J0.30", jitter(0.30)),  # hypothesis (a)
        ("jitter_J0.50", jitter(0.50)),  # hypothesis (a), stronger
        ("top_down_0.5", top_down(0.5)),  # hypothesis (b)
        ("top_down_1.0", top_down(1.0)),  # hypothesis (b), stronger
        ("jitterJ0.30_top_1.0", jitter_and_top_down(0.30, 1.0)),  # combined
    ]


def main() -> int:
    # --- 1. CLI ------------------------------------------------------------
    ap = argparse.ArgumentParser(
        description="Probe BRIDGE_Z_OFFSETS layouts that let the N=8 thick CAD build."
    )
    ap.add_argument(
        "--baseline-params",
        type=Path,
        default=BIOMIMETIC_ROOT
        / "runs/sweep_layers_v2/trial_002_N_BRIDGE_LAYERS_8/cad/_thick_variant_work/cad_params_used.json",
        help="The failed N=8 thick cad_params_used.json to use as the baseline.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=BIOMIMETIC_ROOT / "runs/n8_thick_offset_probe",
        help="Directory for per-strategy CAD sandboxes + results JSON.",
    )
    args = ap.parse_args()

    # --- 2. Load the exact failed thick params ----------------------------
    # This JSON already carries ROD_DIAMETER=3.166932, the jittered offsets,
    # scaled per_ring_diameter, plates, twist -- everything. We only swap
    # BRIDGE_Z_OFFSETS per strategy, so the geometry is identical to the
    # failing build except for bridge elevations.
    base_params = json.loads(args.baseline_params.read_text())
    baseline_offsets = list(base_params["BRIDGE_Z_OFFSETS"])
    print(
        f"Loaded baseline thick params: ROD_DIAMETER={base_params['ROD_DIAMETER']:.4f} mm, "
        f"N_BRIDGE_LAYERS={base_params['N_BRIDGE_LAYERS']}, {len(baseline_offsets)} offsets"
    )
    print(f"Baseline offsets: {[round(z,3) for z in baseline_offsets]}")

    strategies = build_strategies(baseline_offsets)
    args.out_root.mkdir(parents=True, exist_ok=True)

    # --- 3. Run each strategy CAD-only ------------------------------------
    results = []
    for name, offsets in strategies:
        sandbox = args.out_root / name
        print("\n" + "=" * 72)
        print(f"STRATEGY: {name}")
        print(f"  offsets: {[round(z,3) for z in offsets]}")
        print("=" * 72)

        # Fresh params for this strategy (don't mutate the shared dict).
        params = dict(base_params)
        params["BRIDGE_Z_OFFSETS"] = list(offsets)

        rec = {
            "strategy": name,
            "offsets": [float(z) for z in offsets],
            "build_ok": False,
            "watertight": None,
            "n_faces": None,
            "error": None,
        }

        # cad_runner.run raises RuntimeError on the Null-shape failure; catch it
        # so one failing strategy doesn't abort the whole probe sweep.
        try:
            if sandbox.exists():
                # cad_runner writes into export_dir; start clean so a prior probe
                # run doesn't confuse the watertight check.
                import shutil

                shutil.rmtree(sandbox)
            sandbox.mkdir(parents=True, exist_ok=True)
            result = cad_runner.run(params, sandbox)
            stl_path = Path(result.stl_path)
            rec["build_ok"] = stl_path.is_file()
            if rec["build_ok"]:
                # Secondary quality check: is the produced solid watertight?
                mesh = trimesh.load(stl_path, process=False)
                rec["watertight"] = bool(mesh.is_watertight)
                rec["n_faces"] = int(len(mesh.faces))
                print(f"  -> BUILD OK | watertight={rec['watertight']} | faces={rec['n_faces']:,}")
            else:
                print("  -> cad_runner returned but STL missing")
        except Exception as exc:  # noqa: BLE001 -- want to record ANY failure
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  -> BUILD FAILED: {rec['error']}")
            # Keep the full traceback in the record's log for post-mortem.
            rec["traceback_tail"] = traceback.format_exc().splitlines()[-4:]

        results.append(rec)

    # --- 4. Summary table + machine-readable dump -------------------------
    print("\n" + "=" * 72)
    print("PROBE RESULTS  (N=8 thick BRIDGE_Z_OFFSETS sweep)")
    print("=" * 72)
    print(f"{'strategy':<22} {'build':<7} {'watertight':<11} {'faces':<10}")
    for r in results:
        faces = f"{r['n_faces']:,}" if r["n_faces"] else "-"
        print(f"{r['strategy']:<22} {str(r['build_ok']):<7} {str(r['watertight']):<11} {faces:<10}")

    out_json = args.out_root / "probe_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_json}")

    # A strategy "wins" if it builds AND is watertight. Report the first winner.
    winners = [r for r in results if r["build_ok"] and r["watertight"]]
    if winners:
        print(f"\nWINNER(S) (build_ok + watertight): {[w['strategy'] for w in winners]}")
    else:
        builds = [r for r in results if r["build_ok"]]
        if builds:
            print(f"\nBuilt but not watertight: {[b['strategy'] for b in builds]}")
        else:
            print("\nNO strategy built -- offset tuning does not rescue thick N=8.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
