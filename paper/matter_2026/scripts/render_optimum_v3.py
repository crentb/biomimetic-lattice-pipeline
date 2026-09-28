#!/usr/bin/env python
"""Render the lattice selected by the closed-loop optimization (Figure 6, panel I).

Purpose
-------
Produces figures/panels/render_optimum.png with exactly the camera, lighting,
and colors of the other lattice renders of the paper (render_models_v2.render_lattice;
elevation 20 deg, azimuth -60 deg, zoom 0.84), so that the optimized design can be
compared visually with the design-space renders in the top strip of Figure 6.

The STL is taken from the 0.5-mm verification run when it exists
(runs/<run-root>/best_verified.json -> verified.run), otherwise from the search run
of the best trial (lowest J in trials.jsonl). Both runs build the same geometry;
only the FE mesh differs.

Inputs
------
biomimetic_pipeline/runs/<run-root>/best_verified.json   (optional)
biomimetic_pipeline/runs/<run-root>/trials.jsonl
biomimetic_pipeline/runs/<run>/cad/compound_enamel_lattice.stl

Outputs
-------
figures/panels/render_optimum.png; entry "optimum" in data/render_manifest.json

Usage (conda base, from manuscript/matter_v2/):  python scripts/render_optimum_v3.py [--run-root inverse_design_v3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render_models_v2 as R  # noqa: E402  (shared render recipe; importing does not render anything)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--run-root", default="inverse_design_v3")
    args = ap.parse_args()
    root = R.BIO / "runs" / args.run_root

    # --- 1. Which run: verified (0.5 mm) if available, else the best search trial ---------------
    ver = root / "best_verified.json"
    if ver.exists():
        run = json.loads(ver.read_text())["verified"]["run"]
    else:
        trials = [
            json.loads(x) for x in (root / "trials.jsonl").read_text().splitlines() if x.strip()
        ]
        run = min((t for t in trials if t.get("state") == "ok"), key=lambda t: t["J"])["run"]
    stl = R.BIO / "runs" / run / "cad" / "compound_enamel_lattice.stl"

    # --- 2. Render with the recipe of the design-space renders and record it in the manifest -----
    info = R.render_lattice("optimum", stl, elev=20.0, azim=-60.0, zoom=0.84)
    man = json.loads(R.MANIFEST.read_text()) if R.MANIFEST.exists() else {}
    man["optimum"] = {**(info or {}), "run": run}
    R.MANIFEST.write_text(json.dumps(man, indent=1))
    print(f"rendered optimum from {run}")


if __name__ == "__main__":
    main()
