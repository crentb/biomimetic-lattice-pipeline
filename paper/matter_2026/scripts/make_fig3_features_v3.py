#!/usr/bin/env python
"""Figure 3 (v3): morphometric features and decussation bands of lion enamel.

Purpose
-------
Morphometric feature extraction from the 2,433 PIV-tracked rod trajectories
and the pixel-level displacement field (the image-analysis chain that produces
them is Figure 2):
  A  Rod positions at slice 51 coloured by per-trajectory signed pitch: stripes
     of alternating sign are the decussation (Hunter-Schreger) bands.
  B  Self-organizing-map (SOM) segmentation of the ROI into four decussation
     bands (B1-B4) with streamlines of the accumulated displacement field.
  C  Per-band rose diagrams of displacement direction (circular mean mu and
     circular variance V, as in morphometrics.json).
  D  Trajectory tortuosity (arc/chord) of smoothed centrelines (1-um Gaussian)
     with the raw distribution as an outline.
  E  Band period across the bands: FFT spectra of eight vertical transects of
     the pixel-level pitch field and the period from sign-change spacing.
  F  Depth profile of the per-slice mean signed pitch: the quantity the CAD
     mapper integrates into ring rotations (negative at every depth).

Inputs
------
output_piv/track_centerlines_piv.parquet (via make_fig2_v2.load_tracks)
matter_v2/data/som_signcheck_bands.npz   (SOM map and accumulated displacement from the
                                          sign-corrected PIV fields; scripts/som_signcheck.py)
matter_v2/data/canonical_som_bands.npz   (rod mask of slice 51)
biomimetic_pipeline/runs/live_001/morphometrics.json (pitch depth profile)

Outputs
-------
figures/figure3_features.{pdf,png,tif}; data/fig3_features_values.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _style_v2 as S  # noqa: E402
import make_fig2_v2 as F2  # noqa: E402  (loaders, constants, rose())

MANU = HERE.parent
BIO = MANU.parents[1]
MORPH = BIO / "runs/live_001/morphometrics.json"
OUT = MANU / "figures" / "figure3_features"
VALUES = MANU / "data" / "fig3_features_values.json"
PX_UM = F2.PX_UM


def band_stats(cmap_lbl, dx, dy):
    """Per-band circular statistics exactly as the canonical script computes them."""
    ang = np.arctan2(dy, dx)
    out = []
    for c in range(4):
        a = ang[cmap_lbl == c]
        C, Sn = np.cos(a).mean(), np.sin(a).mean()
        out.append(
            {
                "name": F2.BAND_NAMES[c],
                "area_pct": 100.0 * np.mean(cmap_lbl == c),
                "mu_deg": float(np.degrees(np.arctan2(Sn, C))),
                "circ_var": float(1.0 - np.hypot(C, Sn)),
                "angles": a,
            }
        )
    return out


def across_band_period(pitch_map):
    """Vertical-transect FFT spectra and sign-change spacing (as make_fig2_v2 / SI)."""
    Hr, Wr = pitch_map.shape
    spectra, dom = [], []
    for c in np.linspace(50, Wr - 50, 8, dtype=int):
        prof = pitch_map[:, max(0, c - 5) : c + 6].mean(axis=1)
        f = np.abs(np.fft.rfft(prof - prof.mean()))
        fr = np.fft.rfftfreq(len(prof), d=1.0)
        wl = np.where(fr > 0, 1.0 / np.where(fr > 0, fr, 1), np.inf) * PX_UM
        m = (wl >= 20 * PX_UM) & (wl <= Hr * PX_UM / 1.5)
        spectra.append((wl[m], f[m] / f[m].max()))
        dom.append(wl[m][np.argmax(f[m])])
    per_col = []
    for c in range(60, Wr - 60, 20):
        prof = pitch_map[40 : Hr - 40, c - 5 : c + 6].mean(axis=1)
        zc = np.where(np.diff(np.sign(prof - np.median(prof))) != 0)[0]
        if len(zc) >= 2:
            per_col.append(2.0 * np.diff(zc).mean() * PX_UM)
    return spectra, np.array(dom), np.array(per_col)


def main() -> None:
    S.apply()
    vals: dict = {}
    d, xs, ys, zs, tau_raw, tau = F2.load_tracks()
    # SOM bands and accumulated displacement from the SIGN-CORRECTED PIV fields (the
    # stored fields are -(displacement); see track_rods_piv_signcorrected.py and SI)
    som = np.load(MANU / "data" / "som_signcheck_bands.npz")
    cmap_lbl = som["cluster_map"].astype(int)
    dx, dy = som["accum_dx_sm"].astype(float), som["accum_dy_sm"].astype(float)
    rod_roi = np.load(F2.SOM_NPZ)["rod_mask_51"].astype(bool)
    bands = band_stats(cmap_lbl, dx, dy)
    pitch_map = np.degrees(np.arctan2(dx, F2.N_TRANSITIONS))
    spectra, dom, per_col = across_band_period(pitch_map)
    Hr, Wr = pitch_map.shape
    full, _, _ = F2.load_masks()
    H, W = full.shape
    morph = json.loads(MORPH.read_text())
    zdep = np.asarray(morph["depth_profiles"]["depth_um"], float)
    pdep = np.asarray(morph["depth_profiles"]["pitch_signed_deg"], float)
    vals.update(
        {
            "n_tracks": int(len(d)),
            "tortuosity_smoothed_mean": float(tau.mean()),
            "tortuosity_raw_mean": float(tau_raw.mean()),
            "period_sign_change_mean_um": float(per_col.mean()),
            "period_sign_change_sd_um": float(per_col.std()),
            "pitch_depth_mean_deg": float(pdep.mean()),
            "pitch_depth_min_deg": float(pdep.min()),
            "pitch_depth_max_deg": float(pdep.max()),
            "bands": [{k: v for k, v in b.items() if k != "angles"} for b in bands],
        }
    )

    FW, FH = 174.0, 142.0
    fig = plt.figure(figsize=(FW * S.MM, FH * S.MM))

    def ax_mm(x, y, w, h, **kw):
        return fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH], **kw)

    def label_mm(letter, x, y):
        fig.text(x / FW, 1 - y / FH, letter, fontsize=S.LABEL_PT, fontweight="bold", va="top")

    # --- A: rod pitch at slice 51 ---------------------------------------------------------
    img_h = 50.0 * H / W
    div, pnorm = S.DIVERGING, Normalize(-F2.PITCH_LIM, F2.PITCH_LIM)
    axA = ax_mm(4, 7, 50, img_h)
    px = np.array([x[F2.SLICE_IDX] for x in xs]) / PX_UM
    py = np.array([y[F2.SLICE_IDX] for y in ys]) / PX_UM
    axA.set_facecolor("#141414")
    rod_pt = 4.26 / PX_UM * (50.0 / W) / 25.4 * 72.0  # marker = mean rod diameter
    sc = axA.scatter(px, py, c=d.pitch_deg.values, cmap=div, norm=pnorm, s=rod_pt**2, linewidths=0)
    axA.set_xlim(0, W)
    axA.set_ylim(H, 0)
    axA.set_xticks([])
    axA.set_yticks([])
    for sp in axA.spines.values():
        sp.set_visible(False)
    S.scalebar(axA, 50 / PX_UM, "50 µm", loc=(0.05, 0.05), color="white")
    cax = ax_mm(55.5, 7 + 0.15 * img_h, 1.8, 0.7 * img_h)
    cb = fig.colorbar(sc, cax=cax, ticks=[-30, -15, 0, 15, 30])
    cb.set_label("rod pitch (°)", fontsize=S.SMALL_PT, labelpad=1)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2, labelsize=S.SMALL_PT)
    axA.set_title(f"{len(d):,} tracked rods, slice 51", fontsize=S.BASE_PT, pad=2)
    label_mm("A", 0, 3)

    # --- B: SOM decussation bands with streamlines -------------------------------------------
    dh = 50.0 * Hr / Wr
    axB = ax_mm(68, 7, 50, dh)
    axB.imshow(
        cmap_lbl,
        cmap=ListedColormap(S.CATEGORICAL),
        vmin=-0.5,
        vmax=3.5,
        interpolation="nearest",
        alpha=0.9,
    )
    axB.imshow(
        np.where(rod_roi, np.nan, 1.0),
        cmap=ListedColormap(["black"]),
        alpha=0.33,
        interpolation="nearest",
    )
    step = 12
    yy, xx = np.mgrid[0:Hr:step, 0:Wr:step]
    axB.streamplot(
        xx[0],
        yy[:, 0],
        dx[::step, ::step],
        dy[::step, ::step],
        color="white",
        linewidth=0.35,
        density=1.2,
        arrowsize=0.35,
        arrowstyle="-|>",
    )
    axB.set_xlim(0, Wr)
    axB.set_ylim(Hr, 0)
    axB.set_axis_off()
    S.scalebar(axB, 50 / PX_UM, "50 µm", loc=(0.04, 0.05), color="white")
    handles = [
        plt.Line2D(
            [],
            [],
            marker="s",
            ls="",
            ms=5.0,
            mfc=S.CATEGORICAL[c],
            mec="none",
            label=f"{b['name']}  {b['mu_deg']:+.1f}°  {b['area_pct']:.1f} %".replace("-", "−"),
        )
        for c, b in enumerate(bands)
    ]
    axB.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        fontsize=S.SMALL_PT,
        handletextpad=0.3,
        columnspacing=0.9,
        borderaxespad=0,
        frameon=False,
    )
    axB.set_title("SOM decussation bands (ROI)", fontsize=S.BASE_PT, pad=2)
    label_mm("B", 63, 3)

    # --- C: rose diagrams, 2 x 2 ---------------------------------------------------------------
    for c, b in enumerate(bands):
        col, row = c % 2, c // 2
        ax = ax_mm(125 + col * 23.5, 11 + row * 29, 19, 19, projection="polar")
        F2.rose(ax, b["angles"], S.CATEGORICAL[c], b["name"], b["mu_deg"], b["circ_var"])
        ax.set_title(
            f"{b['name']}  μ {b['mu_deg']:+.1f}°\nV {b['circ_var']:.3f}".replace("-", "−"),
            fontsize=5.4,
            pad=3.0,
        )
    label_mm("C", 122, 3)

    # --- D: tortuosity ------------------------------------------------------------------------------
    y2 = 7 + max(img_h, dh) + 16
    h2 = FH - y2 - 12
    axD = ax_mm(13, y2, 42, h2)
    bins = np.linspace(1.0, 1.25, 51)  # corrected tracks: tortuosity < 1.25
    axD.hist(
        tau,
        bins=bins,
        color=S.NAVY_D,
        edgecolor="white",
        linewidth=0.25,
        label=f"smoothed: mean {tau.mean():.3f}",
    )
    axD.hist(
        tau_raw,
        bins=bins,
        histtype="step",
        color=S.MUTED,
        linewidth=0.7,
        label=f"raw: mean {tau_raw.mean():.3f}",
    )
    axD.legend(loc="upper right", fontsize=5.4, handlelength=1.2, borderaxespad=0.2)
    axD.set_xlabel("trajectory tortuosity (arc/chord)")
    axD.set_ylabel("rods")
    axD.set_xlim(1.0, 1.25)
    S.recessive_grid(axD)
    label_mm("D", 0, y2 - 4)

    # --- E: across-band period -------------------------------------------------------------------------
    axE = ax_mm(70, y2, 42, h2)
    for wl, p in spectra:
        axE.plot(wl, p, color=S.RULE, lw=0.5, marker="o", ms=1.2, alpha=0.9)
    pm, ps = per_col.mean(), per_col.std()
    axE.axvspan(pm - ps, pm + ps, color=S.GOLD_D, alpha=0.18, lw=0)
    axE.axvline(pm, color=S.GOLD_D, lw=0.9)
    for k in (2, 3):
        axE.axvline(Hr * PX_UM / k, color=S.RULE, lw=0.6, ls=(0, (2, 2)))
    axE.text(
        pm - 3,
        1.36,
        f"{pm:.0f} ± {ps:.0f} µm",
        fontsize=S.SMALL_PT,
        ha="right",
        va="top",
        color=S.INK,
    )
    axE.set_xlabel("wavelength across bands (µm)")
    axE.set_ylabel("normalized FFT amplitude")
    axE.set_xlim(15, 180)
    axE.set_ylim(0, 1.42)
    axE.set_yticks([0, 0.5, 1.0])
    S.recessive_grid(axE)
    label_mm("E", 58, y2 - 4)

    # --- F: pitch depth profile (mapper input) -------------------------------------------------------------
    axF = ax_mm(128, y2, 42, h2)
    axF.axhline(0, color=S.RULE, lw=0.5)
    axF.plot(zdep, pdep, color=S.WINE_D, lw=1.0)
    axF.set_xlabel("depth from DEJ (µm)")
    axF.set_ylabel("mean signed pitch (°)")
    axF.set_ylim(-3.6, 0.8)
    axF.text(
        0.03,
        0.95,
        f"mean {pdep.mean():+.2f}°; negative at every depth".replace("-", "−"),
        transform=axF.transAxes,
        fontsize=S.SMALL_PT,
        va="top",
    )
    S.recessive_grid(axF)
    label_mm("F", 116, y2 - 4)

    S.save(fig, OUT)
    VALUES.write_text(json.dumps(vals, indent=1, default=float))
    print(json.dumps({k: v for k, v in vals.items() if k != "bands"}, indent=1, default=float))


if __name__ == "__main__":
    main()
