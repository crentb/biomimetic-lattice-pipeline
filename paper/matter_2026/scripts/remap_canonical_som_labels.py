#!/usr/bin/env python
"""Map the regenerated SOM labels onto the canonical band numbering.

Purpose
-------
``extract_canonical_som.py`` re-ran som_fullstack_morphometrics.py in a
sandbox. Its band statistics reproduce the canonical morphometrics.json
EXACTLY (area fraction, mean direction, circular variance, band width), but
the label numbering differs: the script on disk was edited after the
canonical JSON was written and now orders bands differently (it also emits a
new "fabric" key). The strict field-by-field check in the extractor therefore
reported a mismatch that is only a permutation. This script:
  1. finds the label permutation that maps regenerated -> canonical bands by
     matching (area %, mean direction, circular variance);
  2. independently recomputes area fraction and circular-mean direction for
     every band FROM THE SAVED MAP ITSELF (so the arrays, not just the JSON,
     are verified);
  3. writes the remapped map in canonical numbering.

Inputs
------
matter_v2/data/canonical_som.npz                 (from extract_canonical_som.py)
matter_v2/data/_som_sandbox/.../morphometrics.json (regenerated)
microct_pipeline/som_approach/output_som_morphometrics/morphometrics.json (canonical)

Outputs
-------
matter_v2/data/canonical_som_bands.npz   cluster_map in CANONICAL numbering +
                                         the displacement fields and slice-51
                                         rod mask carried over unchanged
matter_v2/data/canonical_som_bands_verification.json
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
REPO = HERE.parents[3]
CANON = REPO / "som_approach" / "output_som_morphometrics" / "morphometrics.json"
REGEN = DATA / "_som_sandbox" / "som_approach" / "output_som_morphometrics" / "morphometrics.json"


def band_vec(b):
    """(area %, mean direction deg, circular variance) -- the matched keys."""
    return np.array(
        [b["area_fraction_pct"], b["mean_direction_deg"], b["circular_variance"]], float
    )


def main() -> None:
    canon = json.loads(CANON.read_text())["bands"]
    regen = json.loads(REGEN.read_text())["bands"]
    ck, rk = sorted(canon), sorted(regen)  # 'band_0'..'band_3'

    # --- 1. Best permutation canonical -> regenerated ------------------------
    best = min(
        itertools.permutations(range(4)),
        key=lambda p: sum(
            np.abs(band_vec(canon[ck[i]]) - band_vec(regen[rk[p[i]]])).sum() for i in range(4)
        ),
    )
    resid = sum(
        np.abs(band_vec(canon[ck[i]]) - band_vec(regen[rk[best[i]]])).sum() for i in range(4)
    )
    # regenerated label r corresponds to canonical label c where best[c] == r
    regen_to_canon = {best[c]: c for c in range(4)}

    # --- 2. Remap the saved map and recompute band statistics from it ------
    z = np.load(DATA / "canonical_som.npz")
    raw = z["cluster_map"].astype(int)
    lut = np.array([regen_to_canon[r] for r in range(4)])
    cmap = lut[raw]
    dx, dy = z["accum_dx_sm"].astype(float), z["accum_dy_sm"].astype(float)
    ang = np.arctan2(dy, dx)
    stats = {}
    for c in range(4):
        m = cmap == c
        a = ang[m]
        mean_dir = np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))
        circ_var = 1.0 - np.hypot(np.sin(a).mean(), np.cos(a).mean())
        stats[f"band_{c}"] = {
            "area_fraction_pct_from_map": round(100.0 * m.mean(), 2),
            "mean_direction_deg_from_map": round(float(mean_dir), 2),
            "circular_variance_from_map": round(float(circ_var), 4),
            "canonical": canon[ck[c]] | {},
        }
    ok = all(
        abs(stats[f"band_{c}"]["area_fraction_pct_from_map"] - canon[ck[c]]["area_fraction_pct"])
        < 0.02
        and abs(
            stats[f"band_{c}"]["mean_direction_deg_from_map"] - canon[ck[c]]["mean_direction_deg"]
        )
        < 0.02
        for c in range(4)
    )

    # --- 3. Save -------------------------------------------------------------
    out = {k: z[k] for k in z.files if k != "cluster_map"}
    out["cluster_map"] = cmap.astype(np.int8)
    out["verified_against_canonical"] = np.bool_(ok)
    np.savez_compressed(DATA / "canonical_som_bands.npz", **out)
    (DATA / "canonical_som_bands_verification.json").write_text(
        json.dumps(
            {
                "permutation_canonical_to_regenerated": list(best),
                "json_residual_sum_abs": resid,
                "map_statistics_match_canonical": ok,
                "bands": stats,
            },
            indent=1,
            default=float,
        )
    )
    print(f"permutation canon->regen={best}  JSON residual={resid:.6f}  map-verified={ok}")
    for c in range(4):
        s = stats[f"band_{c}"]
        print(
            f"  band_{c}: area {s['area_fraction_pct_from_map']:6.2f}%  dir {s['mean_direction_deg_from_map']:8.2f} deg"
            f"  var {s['circular_variance_from_map']:.4f}   | canonical {canon[ck[c]]['area_fraction_pct']}%, "
            f"{canon[ck[c]]['mean_direction_deg']} deg, {canon[ck[c]]['circular_variance']}"
        )


if __name__ == "__main__":
    main()
