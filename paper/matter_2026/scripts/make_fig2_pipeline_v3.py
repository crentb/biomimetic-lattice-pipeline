#!/usr/bin/env python
"""Figure 2 (v3): the image-analysis chain from segmentation to rod trajectories.

Purpose
-------
Shows, on the analysed data, each machine-learning / image-analysis stage that
turns synchrotron microCT slices of lion enamel into rod trajectories:
  A  U-Net segmentation of transverse slice 51 (full field) with the analysis
     region of interest (ROI) and the zoom of panel B.
  B  Rod instance detection: Euclidean distance-transform watershed of the
     segmented slice, with the acceptance criteria applied to every region
     (area 60-2,000 px^2, circularity >= 0.25, solidity >= 0.45). Accepted
     rods are outlined; rejected regions are filled. The detection is
     recomputed here with the exact functions of the pipeline script
     microct_pipeline/validate_stack_100.py (imported, not re-implemented).
  C  Detected rods per slice through the 101-slice stack (all accepted rods,
     interior rods) and the 2,433 rods tracked through every slice.
  D  Particle-image-velocimetry (PIV) displacement field between slices 51 and
     52 (64-px interrogation windows, 50 % overlap), drawn over the mask.
  E  Tracked rod trajectories in a 100 x 100 um column (true aspect ratio),
     coloured by signed pitch.
  F  Tracking fidelity: distance from every tracked centreline point to the
     nearest segmented rod centroid of the same slice, against the same
     distance for uniformly random points (chance baseline).

Inputs
------
microct_pipeline/onehundred_image_stack/lion_axial51.png     U-Net mask, slice 51
microct_pipeline/validate_stack_100.py                        detection functions + criteria
microct_pipeline/output_validation_100/summary.csv            per-slice detection counts
microct_pipeline/output_validation_100/slices/slice_NNN_rods.csv  per-slice centroids
microct_pipeline/output_piv/piv_fields/slice_051_uv.npz        PIV field 51 -> 52
microct_pipeline/output_piv/track_centerlines_piv.parquet      2,433 trajectories

Outputs
-------
figures/figure2_pipeline.{pdf,png,tif}; data/fig2_pipeline_values.json

Side effects: none outside matter_v2/ (the pipeline script is only imported).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch, Rectangle
from scipy.spatial import cKDTree
from skimage import measure
from skimage.segmentation import find_boundaries

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _style_v2 as S  # noqa: E402
import make_fig2_v2 as F2  # noqa: E402  (shared loaders: masks, tracks, ROI offset)

# --- 1. Paths and constants ----------------------------------------------------------
MANU = HERE.parent
REPO = MANU.parents[2]  # microct_pipeline/
DET_SCRIPT = REPO / "validate_stack_100.py"
SUMMARY = REPO / "output_validation_100" / "summary.csv"
SLICE_CSV = REPO / "output_validation_100" / "slices" / "slice_{:03d}_rods.csv"
PIV_NPZ = REPO / "output_piv" / "piv_fields" / "slice_051_uv.npz"
OUT = MANU / "figures" / "figure2_pipeline"
VALUES = MANU / "data" / "fig2_pipeline_values.json"
PX_UM = 20.0 / 58.0  # µm per pixel
SLICE = 51
ZOOM_UM = 60.0  # side of the panel-B zoom (µm)
ZOOM_ORIGIN_PX = (300, 300)  # (x, y) of the zoom's top-left corner
RNG = np.random.default_rng(20260926)


def load_detection_module():
    """Import the pipeline's detection script without running its main()."""
    spec = importlib.util.spec_from_file_location("validate_stack_100", DET_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def detect(mod, slice_no: int):
    """Watershed labels and accept/reject classification, exactly as the pipeline does."""
    img = mod.load_image(str(REPO / mod.STACK_DIR / f"lion_axial{slice_no}.png"))
    mask = mod.preprocess(img)
    labels = mod.watershed_labels(mask)
    cfg = mod.CFG
    accepted, rejected = np.zeros_like(labels), np.zeros_like(labels)
    n_acc = n_rej = 0
    for p in measure.regionprops(labels):
        a, per = p.area, p.perimeter
        circ = 4 * np.pi * a / per**2 if per > 0 else 0.0
        ok = (
            cfg["min_area"] <= a <= cfg["max_area"]
            and circ >= cfg["min_circularity"]
            and p.solidity >= cfg["min_solidity"]
        )
        tgt = accepted if ok else rejected
        tgt[labels == p.label] = p.label
        n_acc += ok
        n_rej += not ok
    return mask, labels, accepted, rejected, n_acc, n_rej, cfg


def main() -> None:
    S.apply()
    vals: dict = {}
    mod = load_detection_module()
    mask, labels, accepted, rejected, n_acc, n_rej, cfg = detect(mod, SLICE)
    vals.update(
        {
            "slice": SLICE,
            "n_regions": int(labels.max()),
            "n_accepted": int(n_acc),
            "n_rejected": int(n_rej),
            "criteria": {
                k: cfg[k]
                for k in (
                    "min_distance",
                    "min_area",
                    "max_area",
                    "min_circularity",
                    "min_solidity",
                    "gaussian_sigma",
                    "close_radius",
                    "open_radius",
                    "boundary_margin",
                )
            },
        }
    )
    full, roi, (ox, oy) = F2.load_masks()
    H, W = full.shape

    FW, FH = 174.0, 122.0
    fig = plt.figure(figsize=(FW * S.MM, FH * S.MM))

    def ax_mm(x, y, w, h, **kw):
        return fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH], **kw)

    def label_mm(letter, x, y):
        fig.text(x / FW, 1 - y / FH, letter, fontsize=S.LABEL_PT, fontweight="bold", va="top")

    # --- A: segmentation, ROI and zoom box ---------------------------------------------
    img_h = 50.0 * H / W
    axA = ax_mm(4, 7, 50, img_h)
    axA.imshow(full, cmap="gray", interpolation="nearest")
    axA.add_patch(Rectangle((ox, oy), roi.shape[1], roi.shape[0], fill=False, ec=S.GOLD_D, lw=0.9))
    zpx = ZOOM_UM / PX_UM
    zx, zy = ZOOM_ORIGIN_PX
    axA.add_patch(Rectangle((zx, zy), zpx, zpx, fill=False, ec=S.TEAL_D, lw=0.9))
    axA.text(
        ox + 8,
        oy + 22,
        "analysis ROI",
        color=S.GOLD_D,
        fontsize=S.SMALL_PT,
        va="top",
        bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85),
    )
    S.scalebar(axA, 50.0 / PX_UM, "50 µm")
    axA.set_axis_off()
    axA.set_title(f"U-Net segmentation (slice {SLICE})", fontsize=S.BASE_PT, pad=2)
    label_mm("A", 0, 3)

    # --- B: watershed detection with acceptance criteria ------------------------------------
    axB = ax_mm(61, 7, img_h, img_h)
    sl = (slice(zy, int(zy + zpx)), slice(zx, int(zx + zpx)))
    axB.imshow(mask[sl], cmap=ListedColormap(["#FFFFFF", "#D9DCE3"]), interpolation="nearest")
    rej = rejected[sl] > 0
    axB.imshow(
        np.ma.masked_where(~rej, rej),
        cmap=ListedColormap([S.WINE_D]),
        alpha=0.75,
        interpolation="nearest",
    )
    bnd = find_boundaries(accepted[sl], mode="inner")
    axB.imshow(
        np.ma.masked_where(~bnd, bnd), cmap=ListedColormap([S.NAVY_D]), interpolation="nearest"
    )
    props = measure.regionprops(accepted[sl])
    axB.plot(
        [p.centroid[1] for p in props],
        [p.centroid[0] for p in props],
        "o",
        ms=1.4,
        mfc=S.GOLD_D,
        mec="none",
    )
    axB.set_axis_off()
    axB.set_title("Distance-transform watershed", fontsize=S.BASE_PT, pad=2)
    axB.legend(
        handles=[
            Patch(fc="white", ec=S.NAVY_D, lw=0.8, label="accepted rod"),
            Patch(fc=S.WINE_D, ec="none", alpha=0.75, label="rejected region"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        fontsize=5.4,
        frameon=False,
        handlelength=1.0,
        columnspacing=0.8,
    )
    axB.plot([6, 6 + 10 / PX_UM], [zpx - 6, zpx - 6], color=S.INK, lw=1.2)
    axB.text(6 + 5 / PX_UM, zpx - 9, "10 µm", fontsize=S.SMALL_PT, ha="center", va="bottom")
    label_mm("B", 57, 3)

    # --- C: detections per slice through the stack -------------------------------------------
    summ = pd.read_csv(SUMMARY)
    z_um = 10.0 + (summ.slice.values - 1) * PX_UM
    axC = ax_mm(128, 9, 42, img_h - 6)
    axC.plot(z_um, summ.n_total, color="#8A93A3", lw=1.0, label="accepted rods")
    axC.plot(z_um, summ.n_interior, color=S.NAVY_D, lw=1.0, label="interior rods")
    axC.axhline(2433, color=S.GOLD_D, lw=0.9, ls=(0, (3, 2)), label="tracked (2,433)")
    axC.set_xlabel("depth from DEJ (µm)")
    axC.set_ylabel("rods per slice")
    axC.set_ylim(2300, 2900)
    axC.legend(loc="upper left", fontsize=5.4, handlelength=1.4, borderaxespad=0.2)
    S.recessive_grid(axC)
    vals["per_slice_counts"] = {
        "n_total_min": int(summ.n_total.min()),
        "n_total_max": int(summ.n_total.max()),
        "n_interior_min": int(summ.n_interior.min()),
        "n_interior_max": int(summ.n_interior.max()),
        "n_interior_mean": float(summ.n_interior.mean()),
        "diam_um_mean": float(summ.mean_diam_um.mean()),
    }
    label_mm("C", 118, 3)

    # --- D: PIV displacement field, slices 51 -> 52 ---------------------------------------------
    y2 = 12 + img_h + 8
    axD = ax_mm(4, y2, 50, img_h)
    piv = np.load(PIV_NPZ)
    cx, cy, U, V = piv["cx"], piv["cy"], piv["U"], piv["V"]
    GX, GY = np.meshgrid(cx, cy)
    axD.imshow(full, cmap=ListedColormap(["#FFFFFF", "#D9DCE3"]), interpolation="nearest")
    mag = np.hypot(U, V)
    q = axD.quiver(
        GX,
        GY,
        U,
        V,
        mag,
        cmap=S.SEQUENTIAL,
        norm=Normalize(0, np.percentile(mag, 98)),
        angles="xy",
        scale_units="xy",
        scale=1 / 45.0,
        width=0.006,
        headwidth=3.2,
        headlength=3.8,
    )
    # key below the image so it cannot collide with the panel title
    axD.quiverkey(
        q,
        0.30,
        -0.06,
        0.5,
        "0.5 px per slice (arrows ×45)",
        labelpos="E",
        coordinates="axes",
        fontproperties={"size": S.SMALL_PT},
        color=S.INK,
    )
    axD.set_xlim(0, W)
    axD.set_ylim(H, 0)
    axD.set_axis_off()
    axD.set_title(f"PIV field, slices {SLICE}→{SLICE + 1}", fontsize=S.BASE_PT, pad=2, loc="left")
    vals["piv"] = {"grid": [int(len(cx)), int(len(cy))], "median_disp_px": float(np.median(mag))}
    label_mm("D", 0, y2 - 4)

    # --- E: tracked trajectories ---------------------------------------------------------------
    d, xs, ys, zs, _, _ = F2.load_tracks()
    axE = ax_mm(57, y2 - 4, 52, img_h + 8, projection="3d")
    div, pnorm = S.DIVERGING, Normalize(-F2.PITCH_LIM, F2.PITCH_LIM)
    cx0, cy0, side = 100.0, 100.0, 100.0
    sel = [
        i
        for i in range(len(xs))
        if cx0 <= xs[i][F2.SLICE_IDX] < cx0 + side and cy0 <= ys[i][F2.SLICE_IDX] < cy0 + side
    ]
    colours = div(pnorm(d.pitch_deg.values[sel]))
    for k, i in enumerate(sel):
        axE.plot(xs[i] - cx0, ys[i] - cy0, zs[i] - zs[i][0], color=colours[k], lw=0.35, alpha=0.95)
    depth = float(zs[0][-1] - zs[0][0])
    axE.set_box_aspect((side, side, depth), zoom=0.98)
    axE.view_init(elev=20, azim=-60)
    axE.set_xlabel("x (µm)", labelpad=-10, fontsize=5.5)
    axE.set_ylabel("y (µm)", labelpad=-10, fontsize=5.5)
    axE.set_zlabel("z (µm)", labelpad=-12, fontsize=5.5)
    axE.tick_params(labelsize=4.8, pad=-3.5)
    axE.set_xticks([0, 50, 100])
    axE.set_yticks([0, 50, 100])
    axE.set_zticks([0, 30])
    for a in (axE.xaxis, axE.yaxis, axE.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))
        a.pane.set_edgecolor(S.GRID)
        a._axinfo["grid"]["color"] = S.GRID
        a._axinfo["grid"]["linewidth"] = 0.3
    fig.text(
        (57 + 26) / FW,
        1 - (y2 - 3.5) / FH,
        f"{len(sel)} trajectories (color: pitch, ±30°)",
        fontsize=S.BASE_PT,
        ha="center",
        va="top",
    )
    label_mm("E", 57, y2 - 4)

    # --- F: tracking fidelity ----------------------------------------------------------------------
    X = np.array(xs) / PX_UM
    Y = np.array(ys) / PX_UM
    dist_all = []
    for i in range(X.shape[1]):
        det = pd.read_csv(str(SLICE_CSV).format(i + 1))[["centroid_x", "centroid_y"]].to_numpy()
        dd, _ = cKDTree(det).query(np.column_stack([X[:, i], Y[:, i]]))
        dist_all.append(dd)
    dist_all = np.concatenate(dist_all) * PX_UM
    det51 = pd.read_csv(str(SLICE_CSV).format(SLICE))[["centroid_x", "centroid_y"]].to_numpy()
    rnd = np.column_stack(
        [
            RNG.uniform(det51[:, 0].min(), det51[:, 0].max(), 50000),
            RNG.uniform(det51[:, 1].min(), det51[:, 1].max(), 50000),
        ]
    )
    dr = cKDTree(det51).query(rnd)[0] * PX_UM
    axF = ax_mm(131, y2 + 2, 40, img_h - 6)
    bins = np.linspace(0, 4.0, 41)
    axF.hist(
        dist_all,
        bins=bins,
        density=True,
        color=S.NAVY_D,
        edgecolor="white",
        linewidth=0.25,
        label=f"tracked centerlines (median {np.median(dist_all):.2f} µm)",
    )
    axF.hist(
        dr,
        bins=bins,
        density=True,
        histtype="step",
        color=S.MUTED,
        linewidth=0.8,
        label=f"random points (median {np.median(dr):.2f} µm)",
    )
    axF.set_xlabel("distance to nearest rod centroid (µm)")
    axF.set_ylabel("probability density")
    axF.legend(loc="upper right", fontsize=5.2, handlelength=1.2, borderaxespad=0.2)
    S.recessive_grid(axF)
    vals["tracking_fidelity"] = {
        "median_um": float(np.median(dist_all)),
        "random_median_um": float(np.median(dr)),
        "frac_within_0p69um": float(np.mean(dist_all < 2 * PX_UM)),
        "frac_within_1um": float(np.mean(dist_all < 1.0)),
        "random_frac_within_0p69um": float(np.mean(dr < 2 * PX_UM)),
        "n_points": int(dist_all.size),
    }
    label_mm("F", 119, y2 - 4)

    S.save(fig, OUT)
    VALUES.write_text(json.dumps(vals, indent=1, default=float))
    print(json.dumps(vals, indent=1, default=float))


if __name__ == "__main__":
    main()
