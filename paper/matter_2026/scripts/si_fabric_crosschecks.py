#!/usr/bin/env python
"""SI cross-checks of the fabric: Woodcock orientation tensor and k-means.

Purpose
-------
Recomputes, on the paper's primary trajectory set (2,433 forward-pass PIV
tracks, output_piv/track_centerlines_piv.parquet), the two cross-checks that
v1 quoted from a different track set without saved outputs:
  * Woodcock (1977) fabric: orientation tensor T = (1/N) sum v v^T of the
    per-track unit net-displacement vectors v = (dx, dy, dz)/|.|; strength
    C = ln(l1/l3) and shape K = ln(l1/l2)/ln(l2/l3) (method identical to
    microct_pipeline/analysis_decussation.py);
  * k-means on standardized (pitch, yaw) with silhouette scores for
    k = 2..8. v1 claimed the silhouette "peaked near 0.43, higher than
    k in {3, 4, 6, 7}", but analysis_decussation.py only tests k = 2..5, so
    its "best k = 5" was the edge of the search range; the wider range is
    reported here.
Also reports the per-track tilt and signed-pitch statistics quoted in the text.

Outputs
-------
matter_v2/data/si_fabric_crosschecks.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
MANU = HERE.parent
REPO = MANU.parents[2]
PARQUET = (
    REPO / "output_piv_signcorrected/track_centerlines_piv.parquet"
)  # sign-corrected PIV tracking (track_rods_piv_signcorrected.py; see SI)
OUT = MANU / "data/si_fabric_crosschecks.json"


def main() -> None:
    d = pd.read_parquet(PARQUET)
    # --- 1. Woodcock fabric of net-displacement unit vectors -------------------
    V = d[["dx_um", "dy_um", "dz_um"]].to_numpy(float)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    T = V.T @ V / len(V)
    ev = np.sort(np.linalg.eigvalsh(T))[::-1]  # l1 >= l2 >= l3
    C = float(np.log(ev[0] / ev[2]))
    K = float(np.log(ev[0] / ev[1]) / np.log(ev[1] / ev[2]))

    # --- 2. k-means on standardized (pitch, yaw), silhouette for k = 2..8 -------
    X = StandardScaler().fit_transform(d[["pitch_deg", "yaw_deg"]].to_numpy(float))
    sil = {}
    for k in range(2, 9):
        lab = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X)
        sil[k] = float(silhouette_score(X, lab))
    best = max(sil, key=sil.get)

    out = {
        "n_tracks": int(len(d)),
        "woodcock": {"eigenvalues": ev.tolist(), "C": C, "K": K},
        "kmeans_silhouette": sil,
        "kmeans_best_k": int(best),
        "pitch_deg": {"mean": float(d.pitch_deg.mean()), "sd": float(d.pitch_deg.std())},
        "yaw_deg": {"mean": float(d.yaw_deg.mean()), "sd": float(d.yaw_deg.std())},
        "tilt_deg": {
            "mean": float(d.tilt_deg.mean()),
            "median": float(d.tilt_deg.median()),
            "p90": float(d.tilt_deg.quantile(0.9)),
        },
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
