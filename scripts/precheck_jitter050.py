#!/usr/bin/env python
"""
precheck_jitter050.py
=====================

Purpose
-------
Before committing to a multi-hour full re-run at the new OCCT_TANGENT_JITTER_MM
= 0.50, confirm the bigger jitter does NOT break a lattice that already built
cleanly at the old 0.15. Builds the densest previously-good cases CAD-only via
the PRODUCTION mapping path (so it uses whatever jitter the module constant is
set to right now) and checks watertightness with a MERGED-VERTEX mesh -- the
correct method (cf. the process=False triangle-soup trap documented in
mapping/bridge_mappers.py).

Why only the densest cases
--------------------------
The jitter is a position resonance; the risk that 0.50 lands a bridge on a NEW
tangency is highest where bridges are most numerous/closest -- i.e. the highest
N and the fat-rod (thick) variant. If N=8 biomimetic (8 layers) and N=7 thick
(7 layers, 3.167 mm rods) both stay watertight at 0.50, the sparser N=4..6
cases (fewer bridges, more room) are safe by margin. The three high-N stragglers
(N=9 bio, N=8/9 thick) were already confirmed watertight at 0.50 separately.

Inputs (CLI)
------------
  --out-root : scratch dir for the CAD-only builds. Default runs/precheck_j050.

Outputs
-------
  stdout : per-case watertight verdict; exit code 0 iff ALL cases watertight.

Side effects: writes CAD-only sandboxes under --out-root (no mesh/FEA).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import trimesh

THIS = Path(__file__).resolve()
BIOMIMETIC_ROOT = THIS.parent.parent
if str(BIOMIMETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(BIOMIMETIC_ROOT))

from biomimetic_pipeline.generators import cad_runner  # noqa: E402
from biomimetic_pipeline.mapping import feature_to_cad  # noqa: E402

THICK_ROD_DIAMETER_MM = 3.1669323693369393

# Densest previously-good cases: (label, N_BRIDGE_LAYERS, is_thick)
CASES = [
    ("n8_bio", 8, False),  # densest biomimetic that built at 0.15
    ("n7_thick", 7, True),  # densest thick that built at 0.15
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pre-check that jitter 0.50 keeps the densest good cases watertight."
    )
    ap.add_argument(
        "--morphometrics", type=Path, default=BIOMIMETIC_ROOT / "runs/live_001/morphometrics.json"
    )
    ap.add_argument("--out-root", type=Path, default=BIOMIMETIC_ROOT / "runs/precheck_j050")
    args = ap.parse_args()

    from biomimetic_pipeline.mapping.bridge_mappers import OCCT_TANGENT_JITTER_MM

    print(f"Production OCCT_TANGENT_JITTER_MM = {OCCT_TANGENT_JITTER_MM} mm")
    morph = json.loads(args.morphometrics.read_text())
    args.out_root.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for label, n_layers, is_thick in CASES:
        sandbox = args.out_root / label
        if sandbox.exists():
            shutil.rmtree(sandbox)
        sandbox.mkdir(parents=True, exist_ok=True)
        # Production mapping (uses the current jitter constant). Grow only the
        # rods for the thick variant (BRIDGE_DIAMETER stays as mapped).
        cad = feature_to_cad.map_morphometrics(
            morph,
            morphometrics_source=args.morphometrics,
            extra_overrides={"N_BRIDGE_LAYERS": int(n_layers)},
        )
        if is_thick:
            cad["ROD_DIAMETER"] = THICK_ROD_DIAMETER_MM
        try:
            result = cad_runner.run(cad, sandbox)
            m = trimesh.load(result.stl_path)  # merged vertices == correct check
            wt = bool(m.is_watertight)
        except Exception as exc:  # noqa: BLE001
            wt = False
            print(f"  {label}: BUILD ERROR {type(exc).__name__}: {exc}")
        print(f"  {label} (N={n_layers}, thick={is_thick}): watertight={wt}")
        all_ok = all_ok and wt

    print(
        "\nPRECHECK:",
        (
            "PASS - safe to re-run all 12 at 0.50"
            if all_ok
            else "FAIL - 0.50 broke a previously-good case; do NOT delete/re-run"
        ),
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
