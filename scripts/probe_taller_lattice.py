#!/usr/bin/env python
"""
probe_taller_lattice.py
=======================

Purpose
-------
Test the hypothesis that making the lattice TALLER (raising
ENAMEL_THICKNESS, i.e. longer rods) rescues the high-N OCCT build failures by
spreading the bridge layers vertically and relieving the near-tangency that
breaks the boolean fusion.

Background
----------
At the canonical ENAMEL_THICKNESS = 20 mm, the bridge band is fixed at
~[1.851, 17.649] mm = 15.8 mm. Packing N layers of 1.70 mm-diameter bridges
into that band leaves shrinking vertical gaps as N grows:
    N=8 -> dz 2.235 mm -> ~0.53 mm gap   (builds OK)
    N=9 -> dz 1.975 mm -> ~0.27 mm gap   (non-watertight: too tight)
Raising ENAMEL_THICKNESS stretches the band top (safe_z_max = cut_top_z -
plate_overlap - margin, and cut_top_z scales with ENAMEL_THICKNESS) while the
bottom (safe_z_min ~ 1.851) is fixed -> the layers spread apart. At H=25 mm the
N=9 gap grows to ~0.90 mm, healthier than N=8 at H=20.

This probe builds the three FAILING cases at a taller height, CAD-only, and
reports whether each is watertight (the criterion that failed at H=20).

What it does NOT change
-----------------------
- Stock CAD module (imported via cad_runner, never written).
- The canonical 20 mm trials already on disk. This writes only to its own
  scratch out-root.

Inputs (CLI)
------------
  --heights   : one or more ENAMEL_THICKNESS values (mm) to test. Default 25.
  --out-root  : scratch dir for per-case CAD sandboxes. Default
                runs/taller_lattice_probe.

Outputs
-------
  <out-root>/<case>_H<height>/compound_enamel_lattice.stl (+ step/sidecar/log)
  <out-root>/probe_results.json   -- machine-readable table
  stdout                          -- PASS/FAIL table

Side effects / non-obvious behavior
-----------------------------------
  * THICK cases set ROD_DIAMETER = 3.166932 mm AFTER map_morphometrics returns,
    so BRIDGE_DIAMETER (re-derived inside map_morphometrics as
    bridge_ratio * ROD_DIAMETER) stays at the biomimetic 1.70 mm -- mirrors
    make_thick_rod_variant.py, which grows only the rods, not the bridges.
  * The thick diameter is set by XY ring packing (CENTER_SPACING), which does
    NOT depend on ENAMEL_THICKNESS, so 3.166932 mm is correct at any height.
  * Each case is an independent stock-CAD run (~1-3 min). Runs sequentially.
"""

from __future__ import annotations

# --- Standard library --------------------------------------------------------
import argparse
import json
import sys
import traceback
from pathlib import Path

# --- Third party -------------------------------------------------------------
import trimesh

# --- Pipeline path setup -----------------------------------------------------
THIS = Path(__file__).resolve()
BIOMIMETIC_ROOT = THIS.parent.parent
if str(BIOMIMETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(BIOMIMETIC_ROOT))

from biomimetic_pipeline.generators import cad_runner  # noqa: E402
from biomimetic_pipeline.mapping import feature_to_cad  # noqa: E402

# Canonical thick-rod diameter (mm): CENTER_SPACING - 0.025 mm clearance.
# Thickness-independent (set by XY ring packing), so valid at any height.
THICK_ROD_DIAMETER_MM = 3.1669323693369393

# The three cases that failed to build watertight at H=20 mm.
# (label, N_BRIDGE_LAYERS, is_thick)
CASES = [
    ("n9_bio", 9, False),
    ("n8_thick", 8, True),
    ("n9_thick", 9, True),
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Probe whether a taller lattice rescues high-N OCCT builds."
    )
    ap.add_argument(
        "--heights",
        type=float,
        nargs="+",
        default=[25.0],
        help="ENAMEL_THICKNESS values (mm) to test. Default 25.",
    )
    ap.add_argument(
        "--morphometrics",
        type=Path,
        default=BIOMIMETIC_ROOT / "runs/live_001/morphometrics.json",
        help="Canonical morphometrics.json input.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=BIOMIMETIC_ROOT / "runs/taller_lattice_probe",
        help="Scratch dir for per-case CAD sandboxes + results JSON.",
    )
    args = ap.parse_args()

    morph = json.loads(args.morphometrics.read_text())
    args.out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for height in args.heights:
        for label, n_layers, is_thick in CASES:
            case = f"{label}_H{height:g}"
            sandbox = args.out_root / case
            print("\n" + "=" * 72)
            print(f"CASE: {case}  (N={n_layers}, thick={is_thick}, ENAMEL_THICKNESS={height} mm)")
            print("=" * 72)

            rec = {
                "case": case,
                "label": label,
                "n_layers": n_layers,
                "thick": is_thick,
                "height_mm": height,
                "build_ok": False,
                "watertight": None,
                "n_faces": None,
                "bridge_gap_mm": None,
                "error": None,
            }

            try:
                # 1. Map morphometrics at the taller height. enamel_thickness_mm
                #    threads in EARLY so BRIDGE_Z_OFFSETS recomputes for the
                #    stretched band; N_BRIDGE_LAYERS via extra_overrides is seen
                #    by the offset computation downstream.
                cad = feature_to_cad.map_morphometrics(
                    morph,
                    morphometrics_source=args.morphometrics,
                    enamel_thickness_mm=float(height),
                    extra_overrides={"N_BRIDGE_LAYERS": int(n_layers)},
                )

                # 2. For thick cases, grow ONLY the rods (bridges stay 1.70 mm).
                if is_thick:
                    cad["ROD_DIAMETER"] = THICK_ROD_DIAMETER_MM

                # 3. Record the resulting vertical bridge gap for context.
                offs = cad.get("BRIDGE_Z_OFFSETS") or []
                if len(offs) >= 2:
                    dz = (offs[-1] - offs[0]) / (len(offs) - 1)
                    rec["bridge_gap_mm"] = round(dz - float(cad["BRIDGE_DIAMETER"]), 4)
                print(
                    f"  offsets[0..-1]=[{offs[0]:.3f} .. {offs[-1]:.3f}] mm, "
                    f"layers={len(offs)}, vertical gap={rec['bridge_gap_mm']} mm"
                )

                # 4. Build CAD-only.
                if sandbox.exists():
                    import shutil

                    shutil.rmtree(sandbox)
                sandbox.mkdir(parents=True, exist_ok=True)
                result = cad_runner.run(cad, sandbox)
                stl_path = Path(result.stl_path)
                rec["build_ok"] = stl_path.is_file()
                if rec["build_ok"]:
                    mesh = trimesh.load(stl_path, process=False)
                    rec["watertight"] = bool(mesh.is_watertight)
                    rec["n_faces"] = int(len(mesh.faces))
                    print(
                        f"  -> BUILD OK | watertight={rec['watertight']} | faces={rec['n_faces']:,}"
                    )
                else:
                    print("  -> cad_runner returned but STL missing")
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["traceback_tail"] = traceback.format_exc().splitlines()[-4:]
                print(f"  -> BUILD FAILED: {rec['error']}")

            results.append(rec)

    # Summary
    print("\n" + "=" * 72)
    print("TALLER-LATTICE PROBE RESULTS")
    print("=" * 72)
    print(f"{'case':<16} {'gap(mm)':<9} {'build':<7} {'watertight':<11} {'faces':<10}")
    for r in results:
        faces = f"{r['n_faces']:,}" if r["n_faces"] else "-"
        print(
            f"{r['case']:<16} {str(r['bridge_gap_mm']):<9} {str(r['build_ok']):<7} "
            f"{str(r['watertight']):<11} {faces:<10}"
        )

    out_json = args.out_root / "probe_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_json}")

    wins = [r for r in results if r["build_ok"] and r["watertight"]]
    print(f"\nWatertight winners: {[w['case'] for w in wins] or 'NONE'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
