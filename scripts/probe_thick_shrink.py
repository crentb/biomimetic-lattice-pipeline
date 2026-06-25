#!/usr/bin/env python
"""
probe_thick_shrink.py
=====================

Purpose
-------
Find the LARGEST thick-rod diameter that still builds VALID N=8 and N=9 thick
lattices (watertight AND all bridges present). The "barely-touching" diameter
CENTER_SPACING - 0.025 = 3.167 mm is OCCT-unbuildable at high N (the dense
bridges silently drop). Shrinking the rods gives the bridge fusion more room.
This probe sweeps rod diameter x N x jitter and runs the FULL production
integrity check on each build, so we don't repeat the watertight-only mistake
(a watertight rods+plates solid with every bridge dropped is NOT valid).

Why the whole thick arm, not just N=8/9
---------------------------------------
For a consistent thick comparison every thick trial must share ONE rod diameter.
So we want the largest diameter that clears the hard cases (N=8, N=9); it will
then also work for the sparser N=4..7, and the whole thick arm is re-run at it.

Geometry note
-------------
Shrinking ROD_DIAMETER does NOT change the bridge band (that depends on
BRIDGE_DIAMETER=1.702 mm + plates, both unchanged), so for a given (N, jitter)
the BRIDGE_Z_OFFSETS are fixed; only ROD_DIAMETER varies across diameters.

Inputs (CLI)
------------
  --diameters : thick rod diameters (mm) to test. Default 3.092 2.992 2.892
                (CENTER_SPACING - 0.1 / -0.2 / -0.3).
  --jitters   : bridge jitters (mm) to test. Default 0.15 0.50.
  --ns        : N_BRIDGE_LAYERS to test. Default 8 9 (the hard cases).
  --out-root  : scratch dir. Default runs/thick_shrink_probe.

Outputs
-------
  <out-root>/probe_results.json + stdout table.

Side effects: CAD-only sandboxes under --out-root.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
BIOMIMETIC_ROOT = THIS.parent.parent
if str(BIOMIMETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(BIOMIMETIC_ROOT))

from biomimetic_pipeline.generators import cad_integrity, cad_runner  # noqa: E402
from biomimetic_pipeline.mapping import feature_to_cad  # noqa: E402

CENTER_SPACING_MM = 3.1919323693369424  # for reporting the surface gap


def bridge_band_offsets(cad: dict, n: int, jitter: float) -> list[float]:
    """Reproduce compute_safe_bridge_elevations for an arbitrary jitter.

    safe_z_min = plate_overlap + (bridge_half + clearance)
    safe_z_max = (enamel - plate_overlap) - plate_overlap - (bridge_half + clearance)
    offsets    = linspace(safe_z_min + jitter, safe_z_max, n)   (top pinned)
    Band depends on BRIDGE_DIAMETER + plates only (NOT rod diameter).
    """
    enamel = float(cad["ENAMEL_THICKNESS"])
    plate_overlap = float(cad["PLATE_OVERLAP"])
    bridge_half = 0.5 * float(cad["BRIDGE_DIAMETER"])
    clearance = 0.5  # DEFAULT_PLATE_CLEARANCE_MM
    margin = bridge_half + clearance
    cut_top_z = enamel - plate_overlap
    safe_z_min = plate_overlap + margin
    safe_z_max = cut_top_z - plate_overlap - margin
    return [float(z) for z in np.linspace(safe_z_min + jitter, safe_z_max, n)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Probe shrunken thick-rod diameters for valid high-N builds."
    )
    ap.add_argument("--diameters", type=float, nargs="+", default=[3.092, 2.992, 2.892])
    ap.add_argument("--jitters", type=float, nargs="+", default=[0.15, 0.50])
    ap.add_argument("--ns", type=int, nargs="+", default=[8, 9])
    ap.add_argument(
        "--morphometrics", type=Path, default=BIOMIMETIC_ROOT / "runs/live_001/morphometrics.json"
    )
    ap.add_argument("--out-root", type=Path, default=BIOMIMETIC_ROOT / "runs/thick_shrink_probe")
    args = ap.parse_args()

    morph = json.loads(args.morphometrics.read_text())
    args.out_root.mkdir(parents=True, exist_ok=True)
    results = []

    for D in args.diameters:
        gap = CENTER_SPACING_MM - D
        for n in args.ns:
            for J in args.jitters:
                tag = f"D{D:.3f}_N{n}_J{J:.2f}"
                sandbox = args.out_root / tag
                if sandbox.exists():
                    shutil.rmtree(sandbox)
                sandbox.mkdir(parents=True, exist_ok=True)

                # Map at default 20mm; override N, then set thick rod + offsets.
                cad = feature_to_cad.map_morphometrics(
                    morph,
                    morphometrics_source=args.morphometrics,
                    extra_overrides={"N_BRIDGE_LAYERS": int(n)},
                )
                offsets = bridge_band_offsets(cad, n, J)
                cad["ROD_DIAMETER"] = float(D)  # shrink rods (bridges unchanged)
                cad["BRIDGE_Z_OFFSETS"] = offsets

                rec = {
                    "tag": tag,
                    "D_mm": D,
                    "gap_mm": round(gap, 3),
                    "N": n,
                    "jitter": J,
                    "built": False,
                    "passed": False,
                    "watertight": None,
                    "bridges": None,
                    "error": None,
                }
                try:
                    result = cad_runner.run(cad, sandbox)
                    rec["built"] = Path(result.stl_path).is_file()
                    if rec["built"]:
                        # FULL production check: watertight AND bridge presence.
                        report = cad_integrity.verify_cad_integrity(
                            Path(result.stl_path),
                            bridge_z_offsets=offsets,
                            n_rings=5,
                        )
                        rec["passed"] = bool(report.passed)
                        rec["watertight"] = bool(report.watertight)
                        rec["bridges"] = (
                            f"{report.bridge['n_present']}/{report.bridge['n_expected']}"
                        )
                except Exception as exc:  # noqa: BLE001
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                results.append(rec)
                print(
                    f"  {tag:<22} gap={rec['gap_mm']}mm  built={rec['built']!s:<5} "
                    f"passed={rec['passed']!s:<5} watertight={rec['watertight']!s:<5} "
                    f"bridges={rec['bridges']}  {rec['error'] or ''}"
                )

    (args.out_root / "probe_results.json").write_text(json.dumps(results, indent=2))

    # Summary: largest diameter whose N=8 AND N=9 both pass at the same jitter.
    print("\n" + "=" * 64)
    print("SUMMARY: (diameter, jitter) combos where ALL tested N pass")
    by_DJ = {}
    for r in results:
        by_DJ.setdefault((r["D_mm"], r["jitter"]), []).append(r)
    winners = []
    for (D, J), recs in sorted(by_DJ.items(), key=lambda kv: -kv[0][0]):
        if all(r["passed"] for r in recs) and len(recs) == len(args.ns):
            winners.append((D, J))
            print(f"  D={D:.3f} mm (gap {CENTER_SPACING_MM-D:.3f}), jitter={J}: ALL PASS")
    if winners:
        D, J = winners[0]
        print(
            f"\nLARGEST valid thick diameter: {D:.3f} mm (gap {CENTER_SPACING_MM-D:.3f} mm) at jitter {J}"
        )
    else:
        print("\nNo (diameter, jitter) clears all tested N -- shrink further or reconsider.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
