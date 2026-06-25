#!/usr/bin/env python
"""
build_sweep_log.py
==================

Purpose
-------
Harvest a complete sweep_log.csv from the per-trial artifacts. The layer-sweep
orchestrators (run_single_trial.py et al.) do NOT write sweep_log.csv -- they
only emit per-trial cad_params.json + metrics.json. This script walks a sweep
root, reads each trial's geometry (cad_params.json) and FEA metrics
(metrics.json), and writes one CSV row per trial, sorted by N_BRIDGE_LAYERS.

Why this exists
---------------
The committed sweep_layers_v2/sweep_log.csv is stale (3 rows from an older
driver). Figure/plot scripts read sweep_log.csv, so it must reflect the actual
trials on disk. Run this after a sweep completes to regenerate the source of
truth for both arms (biomimetic + thick).

Inputs (CLI)
------------
  --sweep-root  : dir containing trial_*/ subdirs (default sweep_layers_v2).
  --out         : output CSV path (default <sweep-root>/sweep_log.csv).

Outputs
-------
  CSV: run_name, N_BRIDGE_LAYERS, geometry (ROD/BRIDGE/CENTER_SPACING, bridge
  band), then every SCALAR metric found in metrics.json (union across trials).
  Rows sorted by N. Trials lacking metrics.json (incomplete) are skipped with a
  warning so the log only contains finished trials.

Side effects: writes the CSV; reads only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent

# Geometry columns pulled from cad_params.json (the load-bearing design values).
GEOM_KEYS = ["ROD_DIAMETER", "BRIDGE_DIAMETER", "CENTER_SPACING", "N_BRIDGE_LAYERS"]


def harvest(sweep_root: Path):
    """Return (rows, metric_columns) for every COMPLETE trial under sweep_root."""
    rows = []
    metric_cols: list[str] = []
    seen_metric = set()
    trial_dirs = sorted(sweep_root.glob("trial_*"))
    for d in trial_dirs:
        cad_p = d / "cad_params.json"
        met_p = d / "metrics.json"
        if not met_p.is_file():
            print(f"  skip (no metrics.json, incomplete): {d.name}")
            continue
        cad = json.loads(cad_p.read_text()) if cad_p.is_file() else {}
        met = json.loads(met_p.read_text())

        # N from the dir name as a cross-check against cad_params.
        m = re.search(r"_LAYERS_(\d+)", d.name)
        n_from_name = int(m.group(1)) if m else None

        offs = cad.get("BRIDGE_Z_OFFSETS") or []
        row = {
            "run_name": f"{sweep_root.name}/{d.name}",
            "N_BRIDGE_LAYERS": int(cad.get("N_BRIDGE_LAYERS", n_from_name or 0)),
            "ROD_DIAMETER": cad.get("ROD_DIAMETER"),
            "BRIDGE_DIAMETER": cad.get("BRIDGE_DIAMETER"),
            "CENTER_SPACING": cad.get("CENTER_SPACING"),
            "bridge_z_bottom": (round(offs[0], 4) if offs else None),
            "bridge_z_top": (round(offs[-1], 4) if offs else None),
            "n_bridge_offsets": len(offs),
        }
        # All SCALAR metrics (skip nested dict/list metrics like _pipeline).
        for k, v in met.items():
            if isinstance(v, (int, float, bool)):
                row[k] = v
                if k not in seen_metric:
                    seen_metric.add(k)
                    metric_cols.append(k)
        rows.append(row)

    rows.sort(key=lambda r: r["N_BRIDGE_LAYERS"])
    return rows, metric_cols


def main() -> int:
    ap = argparse.ArgumentParser(description="Harvest sweep_log.csv from per-trial metrics.")
    ap.add_argument("--sweep-root", type=Path, default=ROOT / "runs" / "sweep_layers_v2")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sweep_root = args.sweep_root.resolve()
    out = args.out or (sweep_root / "sweep_log.csv")
    rows, metric_cols = harvest(sweep_root)
    if not rows:
        print(f"No complete trials under {sweep_root}")
        return 1

    base_cols = [
        "run_name",
        "N_BRIDGE_LAYERS",
        "ROD_DIAMETER",
        "BRIDGE_DIAMETER",
        "CENTER_SPACING",
        "bridge_z_bottom",
        "bridge_z_top",
        "n_bridge_offsets",
    ]
    cols = base_cols + metric_cols
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(
        f"Wrote {out}  ({len(rows)} trials, N={[r['N_BRIDGE_LAYERS'] for r in rows]}, {len(cols)} cols)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
