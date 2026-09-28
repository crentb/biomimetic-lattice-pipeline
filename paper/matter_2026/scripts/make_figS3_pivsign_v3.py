#!/usr/bin/env python
"""Figure S3 (v3): effect of correcting the PIV displacement sign on the rod descriptor.

Purpose
-------
The tracking stage originally advected each rod by the value returned by
skimage.registration.phase_cross_correlation(w1, w2), which is the NEGATIVE
of the pattern displacement from slice s to s+1 (the shift that registers
w2 onto w1). The corrected tracker (microct_pipeline/track_rods_piv_signcorrected.py)
advects by the true displacement. This figure documents the difference:
  A  signed pitch of the 2,433 rods at slice 51, uncorrected sign
  B  the same, corrected sign (coherent stripes = decussation bands)
  C  cumulative distributions of rod tilt, uncorrected and corrected, with
     the rod inclinations of the as-translated lattice (f = 1)
  D  fraction of rods re-anchored to a detected centroid at each slice
     transition, uncorrected and corrected
  E  distance from centerline points to the nearest detected centroid,
     uncorrected and corrected, with the random-point baseline
The sign test itself (synthetic shift of +2/+3 px returned as -2/-3 px) and
the exact reproduction of the canonical tracks with the original sign are
reported in the SI text.

Inputs
------
microct_pipeline/output_piv/track_centerlines_piv.parquet              (uncorrected)
microct_pipeline/output_piv_signcorrected/track_centerlines_piv.parquet (corrected)
microct_pipeline/output_piv_signcorrected/piv_summary.csv               (corrected snap fraction)
microct_pipeline/output_validation_100/slices/slice_NNN_rods.csv        (detected centroids)
biomimetic_pipeline/runs/live_001_digital_twin_v2/cad_params.json      (lattice inclinations)

Outputs
-------
figures/figureS3.{pdf,png,tif}; data/figS3_values.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _style_v2 as S  # noqa: E402
import make_fig3_v2 as F3  # noqa: E402  (lattice_inclination)

MANU = HERE.parent
REPO = MANU.parents[2]
BIO = MANU.parents[1]
OLD = REPO / "output_piv" / "track_centerlines_piv.parquet"
NEW = REPO / "output_piv_signcorrected" / "track_centerlines_piv.parquet"
DET = REPO / "output_validation_100" / "slices"
FIELDS = REPO / "output_piv" / "piv_fields"
OUT = MANU / "figures" / "figureS3"
VALUES = MANU / "data" / "figS3_values.json"
PX = 20.0 / 58.0
SLICE_IDX = 50


def load(path: Path):
    d = pd.read_parquet(path).sort_values("track_id").reset_index(drop=True)
    X = np.array([np.fromstring(v, sep=";") for v in d.cx_um])
    Y = np.array([np.fromstring(v, sep=";") for v in d.cy_um])
    return d, X, Y


def fidelity(X, Y):
    """Distances (um) from centerline points to the nearest detected centroid, pooled over slices."""
    out = []
    for i in range(X.shape[1]):
        det = pd.read_csv(DET / f"slice_{i + 1:03d}_rods.csv")[
            ["centroid_x", "centroid_y"]
        ].to_numpy()
        out.append(cKDTree(det).query(np.column_stack([X[:, i], Y[:, i]]) / PX)[0])
    return np.concatenate(out) * PX


def snap_fraction_uncorrected() -> np.ndarray:
    """Re-anchoring fraction per transition for the as-scripted sign (in-memory replay)."""
    det = {s: pd.read_csv(DET / f"slice_{s:03d}_rods.csv") for s in range(1, 102)}
    seeds = det[1][~det[1]["is_boundary"].astype(bool)]
    x, y = seeds.centroid_x.to_numpy(float), seeds.centroid_y.to_numpy(float)
    frac = []
    for s in range(1, 101):
        f = np.load(FIELDS / f"slice_{s:03d}_uv.npz")
        iu = RegularGridInterpolator((f["cy"], f["cx"]), f["U"], bounds_error=False, fill_value=0.0)
        iv = RegularGridInterpolator((f["cy"], f["cx"]), f["V"], bounds_error=False, fill_value=0.0)
        pts = np.column_stack([y, x])
        xn, yn = x + iu(pts), y + iv(pts)
        snap = det[s + 1][["centroid_x", "centroid_y"]].to_numpy().astype(np.float32).astype(float)
        dist, idx = cKDTree(snap).query(np.column_stack([xn, yn]))
        ok = dist <= 4
        xn[ok], yn[ok] = snap[idx[ok], 0], snap[idx[ok], 1]
        x, y = xn, yn
        frac.append(ok.mean())
    return np.array(frac)


def main() -> None:
    S.apply()
    d0, X0, Y0 = load(OLD)
    d1, X1, Y1 = load(NEW)
    cad = json.loads((BIO / "runs/live_001_digital_twin_v2/cad_params.json").read_text())
    inc = F3.lattice_inclination(cad, 1.0)
    f0, f1 = fidelity(X0, Y0), fidelity(X1, Y1)
    snap0 = snap_fraction_uncorrected()
    snap1 = pd.read_csv(
        REPO / "output_piv_signcorrected" / "piv_summary.csv"
    ).snapped_frac.to_numpy()
    vals = {
        "tilt_mean_uncorrected": float(d0.tilt_deg.mean()),
        "tilt_mean_corrected": float(d1.tilt_deg.mean()),
        "snap_mean_uncorrected": float(snap0.mean()),
        "snap_mean_corrected": float(snap1.mean()),
        "fidelity_median_um_uncorrected": float(np.median(f0)),
        "fidelity_median_um_corrected": float(np.median(f1)),
    }

    FW, FH = 174.0, 120.0
    fig = plt.figure(figsize=(FW * S.MM, FH * S.MM))

    def ax_mm(x, y, w, h, **kw):
        return fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH], **kw)

    def label_mm(letter, x, y):
        fig.text(x / FW, 1 - y / FH, letter, fontsize=S.LABEL_PT, fontweight="bold", va="top")

    # --- A, B: pitch maps -------------------------------------------------------------------
    norm = Normalize(-30, 30)
    for k, (d, X, Y, ttl) in enumerate(
        ((d0, X0, Y0, "uncorrected sign"), (d1, X1, Y1, "corrected sign"))
    ):
        ax = ax_mm(4 + k * 58, 8, 52, 48)
        ax.set_facecolor("#141414")
        ax.scatter(
            X[:, SLICE_IDX],
            Y[:, SLICE_IDX],
            c=d.pitch_deg.values,
            cmap=S.DIVERGING,
            norm=norm,
            s=1.6,
            linewidths=0,
        )
        ax.set_xlim(0, 850 * PX)
        ax.set_ylim(792 * PX, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Rod pitch at slice 51, {ttl}", fontsize=S.BASE_PT, pad=2)
        label_mm("AB"[k], k * 58, 4)
    sm = plt.cm.ScalarMappable(cmap=S.DIVERGING, norm=norm)
    cax = ax_mm(117.5, 14, 1.8, 36)
    cb = fig.colorbar(sm, cax=cax, ticks=[-30, 0, 30])
    cb.set_label("pitch (°)", fontsize=S.SMALL_PT)
    cb.ax.tick_params(labelsize=S.SMALL_PT)

    # --- C: tilt distributions ---------------------------------------------------------------
    axC = ax_mm(141, 10, 30, 44)

    def ecdf(v):
        s = np.sort(v)
        return np.r_[0, s], np.r_[0, np.arange(1, s.size + 1) / s.size]

    axC.step(*ecdf(d0.tilt_deg.values), where="post", color=S.MUTED, lw=1.0, label="uncorrected")
    axC.step(*ecdf(d1.tilt_deg.values), where="post", color=S.NAVY_D, lw=1.2, label="corrected")
    axC.step(
        *ecdf(inc), where="post", color=S.GOLD_D, lw=1.0, ls=(0, (3, 1.5)), label="lattice, f = 1"
    )
    axC.set_xlabel("tilt or inclination (°)")
    axC.set_ylabel("cumulative fraction")
    axC.set_xlim(0, 60)
    axC.legend(fontsize=5.2, loc="lower right", handlelength=1.4)
    S.recessive_grid(axC)
    label_mm("C", 128, 4)

    # --- D: re-anchoring fraction -------------------------------------------------------------
    axD = ax_mm(14, 70, 60, 40)
    z = 10 + np.arange(1, 101) * PX
    axD.plot(z, snap0, color=S.MUTED, lw=1.0, label=f"uncorrected (mean {snap0.mean():.2f})")
    axD.plot(z, snap1, color=S.NAVY_D, lw=1.1, label=f"corrected (mean {snap1.mean():.2f})")
    axD.set_xlabel("depth from DEJ (µm)")
    axD.set_ylabel("fraction re-anchored")
    # Upper right is empty for depths > 15 um (both curves sit at 0.53-0.64 there);
    # "lower right" overlapped the uncorrected curve.
    axD.legend(fontsize=5.4, loc="upper right")
    S.recessive_grid(axD)
    label_mm("D", 0, 66)

    # --- E: fidelity distributions -------------------------------------------------------------
    axE = ax_mm(100, 70, 70, 40)
    bins = np.linspace(0, 4, 41)
    axE.hist(
        f0,
        bins=bins,
        density=True,
        histtype="step",
        color=S.MUTED,
        lw=1.0,
        label=f"uncorrected (median {np.median(f0):.2f} µm)",
    )
    axE.hist(
        f1,
        bins=bins,
        density=True,
        color=S.NAVY_D,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.2,
        label=f"corrected (median {np.median(f1):.2f} µm)",
    )
    axE.set_xlabel("distance to nearest detected rod centroid (µm)")
    axE.set_ylabel("probability density")
    axE.legend(fontsize=5.4, loc="upper right")
    S.recessive_grid(axE)
    label_mm("E", 88, 66)

    S.save(fig, OUT)
    VALUES.write_text(json.dumps(vals, indent=1))
    print(json.dumps(vals, indent=1))


if __name__ == "__main__":
    main()
