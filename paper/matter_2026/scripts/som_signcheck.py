#!/usr/bin/env python
"""Re-run the canonical SOM with sign-corrected PIV displacement fields (check).

Purpose
-------
The SOM decussation-band analysis (som_approach/som_fullstack_morphometrics.py)
accumulates the stored PIV fields as displacements. Those fields hold
phase_cross_correlation(w1, w2), the NEGATIVE of the displacement from slice s
to s+1 (see track_rods_piv_signcorrected.py). This script runs a verbatim copy
of the SOM script in its own sandbox with the single change that the loaded
fields are negated (U, V -> -U, -V), and compares the resulting band
partition with the canonical one:
  * adjusted Rand index (ARI) between the two pixel-label maps,
  * per-band area fraction, circular mean direction and circular variance.
A global sign flip reflects every normalised displacement feature (x -> 1 - x)
and therefore preserves Euclidean distances, so the partition is expected to
be (nearly) unchanged with directions rotated by 180 deg; the Lagrangian
accumulation follows a slightly different path, which is what this checks.

Inputs
------
data/_som_sandbox/som_approach/som_fullstack_morphometrics.py (verbatim canonical copy)
microct_pipeline/roi_imagestack_100, microct_pipeline/output_piv_roi_back (symlinked)
data/canonical_som_bands.npz (canonical, label-verified cluster map)

Outputs
-------
data/_som_sandbox_signfix/...                  sandbox (script copy, figures, JSON)
data/som_signcheck.json                        ARI and per-band statistics
data/som_signcheck_bands.npz                   sign-corrected cluster map and fields
"""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
MANU = HERE.parent
REPO = MANU.parents[2]
SRC = MANU / "data" / "_som_sandbox" / "som_approach" / "som_fullstack_morphometrics.py"
BOX = MANU / "data" / "_som_sandbox_signfix"
OLD_LINE = "    cx, cy, U, V = d['cx'], d['cy'], d['U'], d['V']"
NEW_LINE = (
    "    cx, cy, U, V = d['cx'], d['cy'], -d['U'], -d['V']   "
    "# SIGN FIX: stored fields are -(displacement s->s+1)"
)


def band_stats(labels, dx, dy):
    ang = np.arctan2(dy, dx)
    out = []
    for c in range(labels.max() + 1):
        a = ang[labels == c]
        C, S_ = np.cos(a).mean(), np.sin(a).mean()
        out.append(
            {
                "band": int(c),
                "area_pct": float(100 * np.mean(labels == c)),
                "mu_deg": float(np.degrees(np.arctan2(S_, C))),
                "circ_var": float(1 - np.hypot(C, S_)),
            }
        )
    return out


def main() -> None:
    # --- 1. Sandbox with symlinked inputs and a patched script copy ---------------------------
    (BOX / "som_approach").mkdir(parents=True, exist_ok=True)
    for name in ("roi_imagestack_100", "output_piv_roi_back"):
        link = BOX / name
        if not link.exists():
            os.symlink(REPO / name, link)
    src = SRC.read_text()
    assert src.count(OLD_LINE) == 1, "field-loading line not found; canonical script changed?"
    script = BOX / "som_approach" / "som_fullstack_morphometrics.py"
    script.write_text(src.replace(OLD_LINE, NEW_LINE))

    # --- 2. Run it (writes only inside the sandbox) -----------------------------------------------
    cwd = os.getcwd()
    os.chdir(script.parent)
    try:
        g = runpy.run_path(str(script), run_name="__main__")
    finally:
        os.chdir(cwd)
    lab_new = (
        np.asarray(g["cluster_map"]).astype(int)
        if "cluster_map" in g
        else np.asarray(g["labels_img"]).astype(int)
    )
    dx_new, dy_new = np.asarray(g["accum_dx_sm"]), np.asarray(g["accum_dy_sm"])

    # --- 3. Compare with the canonical (label-verified) map -----------------------------------------
    can = np.load(MANU / "data" / "canonical_som_bands.npz")
    lab_can = can["cluster_map"].astype(int)
    ari = float(adjusted_rand_score(lab_can.ravel(), lab_new.ravel()))
    res = {
        "adjusted_rand_index_vs_canonical": ari,
        "canonical_bands": band_stats(lab_can, can["accum_dx_sm"], can["accum_dy_sm"]),
        "signfix_bands": band_stats(lab_new, dx_new, dy_new),
        "corr_accum_dx_canonical_vs_signfix": float(
            np.corrcoef(can["accum_dx_sm"].ravel(), dx_new.ravel())[0, 1]
        ),
    }
    (MANU / "data" / "som_signcheck.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(
        MANU / "data" / "som_signcheck_bands.npz",
        cluster_map=lab_new.astype(np.int8),
        accum_dx_sm=dx_new.astype(np.float32),
        accum_dy_sm=dy_new.astype(np.float32),
    )
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
