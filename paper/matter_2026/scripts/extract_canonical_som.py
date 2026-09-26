#!/usr/bin/env python
"""Recover the canonical SOM band map behind morphometrics.json (for Fig. 2).

Purpose
-------
The manuscript's architectural descriptor (band directions {2.3, 122.4, 12.6,
-167.6} deg, circular variances {0.170, 0.186, 0.068, 0.134}, area fractions
{27.5, 21.8, 27.7, 23.0} %) comes from
``som_approach/som_fullstack_morphometrics.py``, which writes only figures and
``morphometrics.json`` -- it never saves the per-pixel cluster map. The v1
Fig. 2F/G instead plotted a DIFFERENT SOM product
(``output_som_perslice/labels_3d.npy``, slice 51, dense-ROI PIV, magnitude-
weighted means), so the figure (172/170/8/7 deg) contradicted the text. This
script re-executes the canonical script unchanged, in a sandbox, captures its
in-memory results, and saves exactly the arrays Fig. 2 needs.

Why a sandbox
-------------
The canonical script derives its input and output paths from its own
``__file__`` (../roi_imagestack_100, ../output_piv_roi_back, and
./output_som_morphometrics). Running a verbatim COPY from a sandbox folder
whose sibling input folders are symlinks to the real data makes it read the
real inputs but write its figures and JSON inside the sandbox, so the
canonical outputs in som_approach/ are never touched. The original script is
not modified.

Inputs
------
microct_pipeline/som_approach/som_fullstack_morphometrics.py   (copied verbatim)
microct_pipeline/roi_imagestack_100/axialroi*.png              (symlinked)
microct_pipeline/output_piv_roi_back/piv_fields/*.npz          (symlinked)
microct_pipeline/som_approach/output_som_morphometrics/morphometrics.json
                                                               (reference for verification)

Outputs
-------
manuscript/matter_v2/data/canonical_som.npz with
  cluster_map (H, W) int   canonical 4-band labels (0..3, sorted by mean x
                           displacement exactly as the canonical script does)
  accum_dx_sm, accum_dy_sm accumulated, smoothed PIV displacement (px)
  rod_mask_51 (H, W)       binary rod mask of slice 51 (green channel > 0.5)
  pixel_size_um            0.345 um/px (20/58, as in the canonical script)
manuscript/matter_v2/data/canonical_som_verification.json
  field-by-field comparison of the regenerated morphometrics.json with the
  canonical one (the arrays are only trusted if every band statistic matches).

Side effects / non-obvious behaviour
------------------------------------
* The canonical script is deterministic (SEED = 42 for numpy, MiniSom and
  KMeans); the verification JSON proves the regenerated bands are identical.
* Writes ~20 PNGs into the sandbox (discarded; not used by the paper).
* Runtime ~5-15 min (SOM training + best-matching-unit search).
"""
from __future__ import annotations

import json
import os
import runpy
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # the canonical script calls plt.show()-free savefig; no display
import numpy as np

# --- 1. Paths ---------------------------------------------------------------
HERE = Path(__file__).resolve().parent  # matter_v2/scripts
MANU = HERE.parent  # matter_v2
REPO = MANU.parents[2]  # microct_pipeline
CANON_SCRIPT = REPO / "som_approach" / "som_fullstack_morphometrics.py"
CANON_JSON = REPO / "som_approach" / "output_som_morphometrics" / "morphometrics.json"
SANDBOX = MANU / "data" / "_som_sandbox"
OUT_NPZ = MANU / "data" / "canonical_som.npz"
OUT_VERIFY = MANU / "data" / "canonical_som_verification.json"


def build_sandbox() -> Path:
    """sandbox/som_approach/<script copy> with sibling symlinks to real inputs."""
    sa = SANDBOX / "som_approach"
    sa.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CANON_SCRIPT, sa / CANON_SCRIPT.name)  # verbatim copy
    for name in ("roi_imagestack_100", "output_piv_roi_back"):
        link = SANDBOX / name
        if not link.exists():
            link.symlink_to(REPO / name, target_is_directory=True)
    return sa / CANON_SCRIPT.name


def main() -> None:
    # --- 2. Run the verbatim copy; runpy returns its module globals ---------
    script = build_sandbox()
    cwd = os.getcwd()
    os.chdir(script.parent)
    try:
        g = runpy.run_path(str(script), run_name="__main__")
    finally:
        os.chdir(cwd)

    # --- 3. Verify the regenerated morphometrics against the canonical JSON --
    regen = json.loads(
        (script.parent / "output_som_morphometrics" / "morphometrics.json").read_text()
    )
    canon = json.loads(CANON_JSON.read_text())
    report, all_ok = {}, True

    def walk(a, b, path=""):
        nonlocal all_ok
        if isinstance(a, dict):
            for k in a:
                walk(a[k], b.get(k) if isinstance(b, dict) else None, f"{path}.{k}")
        elif isinstance(a, list):
            for i, x in enumerate(a):
                walk(x, b[i] if isinstance(b, list) and i < len(b) else None, f"{path}[{i}]")
        elif isinstance(a, (int, float)):
            ok = b is not None and abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(a)))
            report[path] = {"canonical": a, "regenerated": b, "match": bool(ok)}
            all_ok &= ok

    walk(canon, regen)
    OUT_VERIFY.write_text(json.dumps({"all_match": all_ok, "fields": report}, indent=1))
    print(
        f"verification: all_match={all_ok}  ({sum(v['match'] for v in report.values())}/{len(report)} numeric fields)"
    )

    # --- 4. Save only the arrays Fig. 2 needs --------------------------------
    rod51 = g["load_rod_mask"](51)
    np.savez_compressed(
        OUT_NPZ,
        cluster_map=np.asarray(g["cluster_map"], dtype=np.int8),
        accum_dx_sm=np.asarray(g["accum_dx_sm"], dtype=np.float32),
        accum_dy_sm=np.asarray(g["accum_dy_sm"], dtype=np.float32),
        rod_mask_51=rod51.astype(np.uint8),
        pixel_size_um=np.float64(g["PIXEL_SIZE_UM"]),
        all_match=np.bool_(all_ok),
    )
    print(f"saved {OUT_NPZ}")
    if not all_ok:
        sys.exit(
            "regenerated band statistics differ from the canonical JSON -- do NOT use the arrays"
        )


if __name__ == "__main__":
    main()
