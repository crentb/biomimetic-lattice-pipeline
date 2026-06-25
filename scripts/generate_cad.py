#!/usr/bin/env python
"""Morphometrics → CAD parameters → STEP. Skips mesh/FEA/report.

Useful for visual inspection of the CAD mapping without running FEA.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
BIOMIMETIC_ROOT = THIS.parent.parent
if str(BIOMIMETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(BIOMIMETIC_ROOT))

from biomimetic_pipeline.generators import cad_runner  # noqa: E402
from biomimetic_pipeline.mapping import feature_to_cad  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--morphometrics", required=True, help="Canonical morphometrics.json path")
    ap.add_argument("--export-dir", required=True, help="Output dir for STEP/STL/sidecar")
    ap.add_argument("--params-out", default=None, help="Optional path for cad_params.json")
    args = ap.parse_args()

    morphometrics_path = Path(args.morphometrics)
    morphometrics = json.loads(morphometrics_path.read_text())
    cad_params = feature_to_cad.map_morphometrics(
        morphometrics, morphometrics_source=morphometrics_path
    )
    feature_to_cad.validate(cad_params)
    params_out = (
        Path(args.params_out) if args.params_out else Path(args.export_dir) / "cad_params.json"
    )
    feature_to_cad.save(cad_params, params_out)

    result = cad_runner.run(cad_params, export_dir=Path(args.export_dir))
    print(f"STEP: {result.step_path}")
    print(f"STL : {result.stl_path}")
    print(f"JSON: {result.sidecar_path}")


if __name__ == "__main__":
    main()
