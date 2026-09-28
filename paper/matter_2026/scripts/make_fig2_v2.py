#!/usr/bin/env python
"""Figure 2 (v2): measurement layer -- segmentation, trajectories, SOM fabric.

Purpose
-------
Rebuilds Figure 2 entirely from the canonical data products so that every
number printed in the panels is the number used in the text and passed to
the CAD mapper. Replaces the v1 composite, which (i) captioned a binary
U-Net mask as a "raw" CT slice, (ii) plotted a different SOM run than the
text reports (172/170/8/7 deg vs {2.3, 122.4, 12.6, -167.6} deg), (iii) used
illegible embedded colour bars, and (iv) mixed four visual styles.

Panels (174 mm wide, Cell Press double column)
  A  U-Net segmentation of transverse slice 51 (full field) with the true
     analysis ROI (located by cross-correlation) and a 20-um zoom inset.
  B  Rod positions at slice 51 coloured by per-track pitch (signed, deg):
     alternating-sign stripes are the Hunter-Schreger decussation bands.
  C  3-D trajectories in a 100 x 100 um column (true aspect), coloured by
     pitch -- rods of opposite pitch cross (decussate).
  D  Canonical four-band SOM segmentation (bands B1-B4) with streamlines of
     the accumulated PIV displacement field.
  E  Per-band rose diagrams of displacement direction; circular mean and
     circular variance printed (identical to morphometrics.json).
  F  Per-trajectory tortuosity (arc/chord), n = 2,433, of centrelines
     smoothed with a 1-um Gaussian (3 slices; below the rod radius), with
     the raw distribution as an outline: sub-micrometre centroid
     fluctuations at 0.345-um slice spacing inflate raw arc length
     (revision 2026-09-25; si_measurement_checks.py).
  G  Band periodicity ACROSS the bands: FFT spectra of eight vertical
     transects of the pixel-level pitch field (the bands run roughly along
     x, so the v1/v2 horizontal transects ran along them), with the
     sign-change spacing along 35 columns, which is not quantised to the
     W/k bins of a 261-um field.

Inputs
------
microct_pipeline/onehundred_image_stack/lion_axial51.png   full-field mask
microct_pipeline/roi_imagestack_100/axialroi51.png          ROI overlay (green = rod)
microct_pipeline/output_piv/track_centerlines_piv.parquet  2,433 canonical tracks
matter_v2/data/canonical_som_bands.npz                      canonical SOM labels +
                                                            accumulated displacement
                                                            (extract_canonical_som.py,
                                                            remap_canonical_som_labels.py)
Outputs
-------
matter_v2/figures/figure2.{pdf,png,tif}
matter_v2/data/fig2_values.json   every number printed on the figure

Side effects: none outside matter_v2/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style_v2 as S  # noqa: E402

# --- 1. Paths and constants ----------------------------------------------------
HERE = Path(__file__).resolve().parent
MANU = HERE.parent
REPO = MANU.parents[2]
FULL_PNG = REPO / "onehundred_image_stack" / "lion_axial51.png"
ROI_PNG = REPO / "roi_imagestack_100" / "axialroi51.png"
PARQUET = (
    REPO / "output_piv_signcorrected" / "track_centerlines_piv.parquet"
)  # sign-corrected PIV tracking (track_rods_piv_signcorrected.py; see SI)
SOM_NPZ = MANU / "data" / "canonical_som_bands.npz"
OUT = MANU / "figures" / "figure2"
VALUES = MANU / "data" / "fig2_values.json"

PX_UM = 20.0 / 58.0  # 0.3448 um per pixel (PIV/SOM scripts)
SLICE_IDX = 50  # 0-based index of slice 51 in each centreline
N_TRANSITIONS = 100  # slice transitions spanned by the accumulated field
PITCH_LIM = 30.0  # deg, symmetric colour limit for signed pitch
SMOOTH_SLICES = 3.0  # Gaussian sigma for centreline smoothing (3 slices = 1.0 um < rod radius)
BAND_NAMES = ["B1", "B2", "B3", "B4"]


def load_masks():
    """Full-field rod mask, ROI rod mask, and the ROI offset inside the field.

    The ROI stack is a crop of the full stack; its offset is found by
    maximising the cross-correlation of the two binary masks (the v1 figure
    drew an approximate box of the wrong size).
    """
    full = np.asarray(Image.open(FULL_PNG).convert("L"), float) / 255.0 > 0.5
    roi_rgb = np.asarray(Image.open(ROI_PNG), float) / 255.0
    roi = roi_rgb[..., 1] > 0.5  # green channel = rod
    a = full.astype(float) - full.mean()
    b = roi.astype(float) - roi.mean()
    cc = fftconvolve(a, b[::-1, ::-1], mode="valid")  # valid offsets only
    oy, ox = np.unravel_index(np.argmax(cc), cc.shape)
    return full, roi, (int(ox), int(oy))


def load_tracks():
    """Per-track slice-51 position (um), pitch (deg), centreline, and
    tortuosity of the raw and of the smoothed centreline (Gaussian, 3 slices)."""
    d = pd.read_parquet(PARQUET)
    xs, ys, zs, tau, tau_s = [], [], [], [], []

    def arc_over_chord(P):
        return np.linalg.norm(np.diff(P, axis=0), axis=1).sum() / np.linalg.norm(P[-1] - P[0])

    for _, r in d.iterrows():
        x = np.fromstring(r.cx_um, sep=";")
        y = np.fromstring(r.cy_um, sep=";")
        z = np.fromstring(r.cz_um, sep=";")
        xs.append(x)
        ys.append(y)
        zs.append(z)
        tau.append(arc_over_chord(np.column_stack([x, y, z])))
        tau_s.append(
            arc_over_chord(
                np.column_stack(
                    [gaussian_filter1d(x, SMOOTH_SLICES), gaussian_filter1d(y, SMOOTH_SLICES), z]
                )
            )
        )
    return d, xs, ys, zs, np.asarray(tau), np.asarray(tau_s)


def rose(ax, angles_rad, colour, name, mu_deg, var):
    """Rose diagram in the image frame (angles clockwise from +x, y down)."""
    bins = np.linspace(-np.pi, np.pi, 37)
    h, e = np.histogram(angles_rad, bins=bins)
    h = h / h.max()
    ctr = 0.5 * (e[1:] + e[:-1])
    ax.bar(ctr, h, width=np.diff(e), color=colour, edgecolor="white", linewidth=0.3, alpha=0.95)
    mu = np.radians(mu_deg)
    ax.plot([mu, mu], [0, 1.08], color=S.INK, lw=0.9)
    ax.set_theta_direction(-1)  # clockwise, matching the image frame
    ax.set_theta_zero_location("E")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([])
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["0°", "90°", "", ""], fontsize=5.2, color=S.MUTED)
    ax.tick_params(pad=-2)
    ax.spines["polar"].set_linewidth(0.4)
    ax.spines["polar"].set_color(S.RULE)
    ax.grid(color=S.GRID, linewidth=0.3)
    ax.set_title(f"{name}:  μ = {mu_deg:+.1f}°,  V = {var:.3f}", fontsize=S.SMALL_PT, pad=5.0)


def main() -> None:
    S.apply()
    vals: dict = {}

    # --- 2. Data -----------------------------------------------------------------
    full, roi, (ox, oy) = load_masks()
    H, W = full.shape
    vals["roi_offset_px"] = [ox, oy]
    vals["roi_size_px"] = [int(roi.shape[1]), int(roi.shape[0])]
    d, xs, ys, zs, tau_raw, tau = load_tracks()  # tau = smoothed (reported), tau_raw = raw
    vals["n_tracks"] = int(len(d))
    vals["tortuosity_mean"] = float(tau.mean())
    vals["tortuosity_p90"] = float(np.percentile(tau, 90))
    vals["tortuosity_raw_mean"] = float(tau_raw.mean())
    vals["tortuosity_raw_p90"] = float(np.percentile(tau_raw, 90))
    som = np.load(SOM_NPZ)
    assert bool(som["verified_against_canonical"]), "SOM map not verified against canonical JSON"
    cmap_lbl = som["cluster_map"].astype(int)
    dx, dy = som["accum_dx_sm"].astype(float), som["accum_dy_sm"].astype(float)
    rod_roi = som["rod_mask_51"].astype(bool)

    # Per-band circular statistics exactly as the canonical script computes them.
    ang = np.arctan2(dy, dx)
    bands = []
    for c in range(4):
        a = ang[cmap_lbl == c]
        C, Sn = np.cos(a).mean(), np.sin(a).mean()
        bands.append(
            {
                "name": BAND_NAMES[c],
                "area_pct": 100.0 * np.mean(cmap_lbl == c),
                "mu_deg": float(np.degrees(np.arctan2(Sn, C))),
                "circ_var": float(1.0 - np.hypot(C, Sn)),
                "angles": a,
            }
        )
    vals["bands"] = [{k: v for k, v in b.items() if k != "angles"} for b in bands]

    # Periodicity ACROSS the bands. The bands run roughly along x (2-D FFT wave
    # vector along y; si_measurement_checks.py), so transects are vertical
    # columns (each averaged over 11 columns) of the pixel-level pitch field.
    pitch_map = np.degrees(np.arctan2(dx, N_TRANSITIONS))
    Hr, Wr = pitch_map.shape
    cols = np.linspace(50, Wr - 50, 8, dtype=int)
    spectra, dom = [], []
    for c in cols:
        prof = pitch_map[:, max(0, c - 5) : c + 6].mean(axis=1)
        f = np.abs(np.fft.rfft(prof - prof.mean()))
        fr = np.fft.rfftfreq(len(prof), d=1.0)
        wl = np.where(fr > 0, 1.0 / np.where(fr > 0, fr, 1), np.inf) * PX_UM
        m = (wl >= 20 * PX_UM) & (wl <= Hr * PX_UM / 1.5)
        spectra.append((wl[m], f[m] / f[m].max()))
        dom.append(wl[m][np.argmax(f[m])])
    # Sign-change spacing along vertical profiles (rows 40..H-40 avoid the
    # zero-padded margin of the smoothed field); one period = two sign changes.
    per_col = []
    for c in range(60, Wr - 60, 20):
        prof = pitch_map[40 : Hr - 40, c - 5 : c + 6].mean(axis=1)
        zc = np.where(np.diff(np.sign(prof - np.median(prof))) != 0)[0]
        if len(zc) >= 2:
            per_col.append(2.0 * np.diff(zc).mean() * PX_UM)
    vals["period_transect_um"] = [float(x) for x in dom]
    vals["period_fft_mode_um"] = float(pd.Series(np.round(dom, 1)).mode()[0])
    vals["period_sign_change_mean_um"] = float(np.mean(per_col))
    vals["period_sign_change_sd_um"] = float(np.std(per_col))
    vals["period_sign_change_n_columns"] = len(per_col)
    vals["field_height_um"] = float(Hr * PX_UM)
    vals["field_width_um"] = float(Wr * PX_UM)

    # --- 3. Layout: explicit millimetre placement (174 x 170 mm) -------------------
    FW, FH = 174.0, 170.0
    fig = plt.figure(figsize=(FW * S.MM, FH * S.MM))

    def ax_mm(x, y, w, h, **kw):
        """Axes at (x, y) mm from the TOP-LEFT corner, size w x h mm."""
        return fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH], **kw)

    def label_mm(letter, x, y):
        fig.text(
            x / FW,
            1 - y / FH,
            letter,
            fontsize=S.LABEL_PT,
            fontweight="bold",
            va="top",
            ha="left",
            color=S.INK,
        )

    div = S.DIVERGING
    pnorm = Normalize(-PITCH_LIM, PITCH_LIM)
    img_h = 50.0 * H / W  # image panels 50 mm wide, true aspect

    # A -- segmentation, true ROI, zoom inset -------------------------------------
    axA = ax_mm(4, 7, 50, img_h)
    axA.imshow(full, cmap="gray", interpolation="nearest")
    axA.add_patch(Rectangle((ox, oy), roi.shape[1], roi.shape[0], fill=False, ec=S.GOLD_D, lw=0.9))
    axA.text(
        ox + 8,
        oy + 8,
        "analysis ROI",
        color="white",
        fontsize=S.SMALL_PT,
        va="top",
        ha="left",
        bbox=dict(fc=S.GOLD_D, ec="none", pad=0.8),
    )
    axA.set_axis_off()
    zx0, zy0, zw = 380, 360, int(round(20 / PX_UM))  # 20-um zoom window
    axA.add_patch(Rectangle((zx0, zy0), zw, zw, fill=False, ec=S.GOLD_D, lw=0.9))
    S.scalebar(axA, 50 / PX_UM, "50 µm", loc=(0.05, 0.05), color="white")
    axA.add_patch(
        Rectangle(
            (0.02, 0.02),
            0.26,
            0.11,
            transform=axA.transAxes,
            fc="black",
            ec="none",
            alpha=0.65,
            zorder=9,
        )
    )
    ins = axA.inset_axes([0.60, 0.60, 0.38, 0.38])
    ins.imshow(full[zy0 : zy0 + zw, zx0 : zx0 + zw], cmap="gray", interpolation="nearest")
    ins.set_xticks([])
    ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_color(S.GOLD_D)
        sp.set_linewidth(1.0)
    axA.set_title("U-Net segmentation (slice 51)", fontsize=S.BASE_PT, pad=2)
    label_mm("A", 0, 3)

    # B -- per-rod pitch at slice 51 (markers drawn at the mean rod diameter) ---------
    axB = ax_mm(58, 7, 50, img_h)
    px = np.array([x[SLICE_IDX] for x in xs]) / PX_UM
    py = np.array([y[SLICE_IDX] for y in ys]) / PX_UM
    axB.set_facecolor("#141414")
    # Marker area (pt^2) for a 4.4-um rod: panel 52 mm spans W px.
    rod_pt = 4.4 / PX_UM * (50.0 / W) / 25.4 * 72.0
    sc = axB.scatter(px, py, c=d.pitch_deg.values, cmap=div, norm=pnorm, s=rod_pt**2, linewidths=0)
    axB.set_xlim(0, W)
    axB.set_ylim(H, 0)
    axB.set_xticks([])
    axB.set_yticks([])
    for sp in axB.spines.values():
        sp.set_visible(False)
    S.scalebar(axB, 50 / PX_UM, "50 µm", loc=(0.05, 0.05), color="white")
    cax = ax_mm(109.5, 7 + 0.15 * img_h, 1.8, 0.7 * img_h)
    cb = fig.colorbar(sc, cax=cax, ticks=[-30, -15, 0, 15, 30])
    cb.set_label("rod pitch (°)", fontsize=S.SMALL_PT, labelpad=1)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2, labelsize=S.SMALL_PT)
    axB.set_title(f"{len(d):,} tracked rods: pitch at slice 51", fontsize=S.BASE_PT, pad=2)
    label_mm("B", 54, 3)

    # C -- 3-D trajectories in a 100 x 100 um column ---------------------------------
    axC = ax_mm(121, 3, 50, img_h + 6, projection="3d")
    cx0, cy0, side = 100.0, 100.0, 100.0
    sel = [
        i
        for i in range(len(xs))
        if cx0 <= xs[i][SLICE_IDX] < cx0 + side and cy0 <= ys[i][SLICE_IDX] < cy0 + side
    ]
    colours = div(pnorm(d.pitch_deg.values[sel]))
    for k, i in enumerate(sel):
        axC.plot(xs[i] - cx0, ys[i] - cy0, zs[i] - zs[i][0], color=colours[k], lw=0.35, alpha=0.95)
    depth = float(zs[0][-1] - zs[0][0])
    axC.set_box_aspect((side, side, depth), zoom=0.98)
    axC.view_init(elev=20, azim=-60)
    axC.set_xlabel("x (µm)", labelpad=-10, fontsize=5.5)
    axC.set_ylabel("y (µm)", labelpad=-10, fontsize=5.5)
    axC.set_zlabel("z (µm)", labelpad=-12, fontsize=5.5)
    axC.tick_params(labelsize=4.8, pad=-3.5)
    axC.set_xticks([0, 50, 100])
    axC.set_yticks([0, 50, 100])
    axC.set_zticks([0, 30])
    for a in (axC.xaxis, axC.yaxis, axC.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))
        a.pane.set_edgecolor(S.GRID)
        a._axinfo["grid"]["color"] = S.GRID
        a._axinfo["grid"]["linewidth"] = 0.3
    vals["panelC_n_tracks"] = len(sel)
    fig.text(
        (121 + 25.0) / FW,
        1 - 4.2 / FH,
        f"{len(sel)} trajectories, 100 × 100 × {depth:.1f} µm",
        fontsize=S.BASE_PT,
        ha="center",
        va="top",
    )
    label_mm("C", 118, 3)

    # D -- canonical SOM bands + displacement streamlines ------------------------------
    yD = 12 + img_h
    dh = 66.0 * Hr / Wr
    axD = ax_mm(4, yD + 4, 66, dh)
    band_cmap = ListedColormap(S.CATEGORICAL)
    axD.imshow(cmap_lbl, cmap=band_cmap, vmin=-0.5, vmax=3.5, interpolation="nearest", alpha=0.9)
    axD.imshow(
        np.where(rod_roi, np.nan, 1.0),
        cmap=ListedColormap(["black"]),
        alpha=0.33,
        interpolation="nearest",
    )
    step = 12
    yy, xx = np.mgrid[0:Hr:step, 0:Wr:step]
    axD.streamplot(
        xx[0],
        yy[:, 0],
        dx[::step, ::step],
        dy[::step, ::step],
        color="white",
        linewidth=0.35,
        density=1.3,
        arrowsize=0.35,
        arrowstyle="-|>",
    )
    axD.set_xlim(0, Wr)
    axD.set_ylim(Hr, 0)
    axD.set_axis_off()
    S.scalebar(axD, 50 / PX_UM, "50 µm", loc=(0.04, 0.05), color="white")
    handles = [
        plt.Line2D(
            [],
            [],
            marker="s",
            ls="",
            ms=5.5,
            mfc=S.CATEGORICAL[c],
            mec="none",
            label=f"{b['name']}   {b['mu_deg']:+.1f}°   {b['area_pct']:.1f} %",
        )
        for c, b in enumerate(bands)
    ]
    leg = axD.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.03, 1.0),
        fontsize=S.SMALL_PT,
        handletextpad=0.4,
        borderaxespad=0,
        labelspacing=0.9,
        title="band   mean dir.   area",
        title_fontsize=S.SMALL_PT,
        alignment="left",
    )
    axD.set_title(
        "Canonical SOM bands with displacement streamlines (ROI)",
        fontsize=S.BASE_PT,
        pad=2,
        loc="left",
    )
    label_mm("D", 0, yD)

    # E -- per-band rose diagrams (2 x 2) ------------------------------------------------
    rw, rh = 25.0, 25.0
    for c, b in enumerate(bands):
        col, row = c % 2, c // 2
        ax = ax_mm(112 + col * 31, yD + 7 + row * 35, rw, rh, projection="polar")
        rose(ax, b["angles"], S.CATEGORICAL[c], b["name"], b["mu_deg"], b["circ_var"])
    label_mm("E", 108, yD)

    # F -- tortuosity distribution --------------------------------------------------------
    yF = yD + 6 + max(dh, 72)
    axF = ax_mm(13, yF, 66, FH - yF - 12)
    bins = np.linspace(1.0, 1.45, 46)
    # Summary statistics live in the legend (no labelled reference lines to collide).
    axF.hist(
        tau,
        bins=bins,
        color=S.NAVY_D,
        edgecolor="white",
        linewidth=0.25,
        label=f"smoothed (1-µm Gaussian): mean {tau.mean():.3f}, P90 {np.percentile(tau, 90):.3f}",
    )
    axF.hist(
        tau_raw,
        bins=bins,
        histtype="step",
        color=S.MUTED,
        linewidth=0.7,
        label=f"raw centrelines: mean {tau_raw.mean():.3f}, P90 {np.percentile(tau_raw, 90):.3f}",
    )
    axF.legend(loc="upper right", fontsize=5.4, handlelength=1.2, borderaxespad=0.2)
    axF.set_xlabel("trajectory tortuosity (arc / chord)")
    axF.set_ylabel("rods")
    axF.set_xlim(1.0, 1.45)
    S.recessive_grid(axF)
    label_mm("F", 0, yF - 3)

    # G -- periodicity ----------------------------------------------------------------------
    axG = ax_mm(104, yF, 66, FH - yF - 12)
    for wl, p in spectra:
        axG.plot(wl, p, color=S.RULE, lw=0.5, marker="o", ms=1.2, alpha=0.9)
    k_bins = sorted(Hr * PX_UM / k for k in (2, 3))  # W/k bins of the field height
    for kb in k_bins:
        axG.axvline(kb, color=S.RULE, lw=0.6, ls=(0, (2, 2)))
    pm, ps = np.mean(per_col), np.std(per_col)
    axG.axvspan(pm - ps, pm + ps, color=S.GOLD_D, alpha=0.18, lw=0)
    axG.axvline(pm, color=S.GOLD_D, lw=0.9)
    axG.text(
        18,
        1.36,
        f"sign-change spacing (gold): {pm:.0f} ± {ps:.0f} µm, {len(per_col)} columns\n"
        f"FFT peak (8 transects): k = 2 bin in {sum(np.isclose(dom, k_bins[1], atol=1)):d} of 8",
        fontsize=S.SMALL_PT,
        ha="left",
        va="top",
        color=S.INK,
    )
    axG.text(
        k_bins[1] + 2, 0.06, f"k = 2\n{k_bins[1]:.0f} µm", fontsize=5.2, color=S.MUTED, va="bottom"
    )
    axG.text(
        k_bins[0] - 2,
        0.06,
        f"k = 3\n{k_bins[0]:.0f} µm",
        fontsize=5.2,
        color=S.MUTED,
        va="bottom",
        ha="right",
    )
    axG.set_xlabel("wavelength across the bands (µm)")
    axG.set_ylabel("normalised FFT amplitude")
    axG.set_xlim(15, 180)
    axG.set_ylim(0, 1.42)
    axG.set_yticks([0, 0.5, 1.0])
    S.recessive_grid(axG)
    label_mm("G", 92, yF - 3)

    S.save(fig, OUT)
    VALUES.write_text(json.dumps(vals, indent=1, default=float))
    print(json.dumps({k: v for k, v in vals.items() if k != "bands"}, indent=1, default=float))
    for b in vals["bands"]:
        print(b)


if __name__ == "__main__":
    main()
