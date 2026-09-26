#!/usr/bin/env python3
"""
validate_stack.py
=================
Runs rod detection on slices 1–20 of the stack to validate that parameters
hold up consistently across depth.

Key design decisions:
  - Boundary rods (touching image edge) are KEPT and flagged with is_boundary=True
    rather than discarded.  They will be needed for 3D tracking — a rod that exits
    the frame on one side often re-enters or links to a partial detection in the
    adjacent slice.
  - Interior rods get cyan outlines; boundary rods get orange outlines so you
    can immediately see how many partial rods exist at each edge.
  - Outputs:
      output_validation/slices/slice_NNN_labeled.png   — one per slice
      output_validation/slices/slice_NNN_rods.csv      — one per slice
      output_validation/summary.csv                    — one row per slice
      output_validation/summary_plots.png              — QC dashboard

Usage:
  python validate_stack.py
"""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as ndi_label
from skimage import color, exposure, filters, io, measure, morphology, segmentation
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
STACK_DIR = "onehundred_image_stack"
FIRST_SLICE = 1
LAST_SLICE = 101
OUTPUT_DIR = "output_validation_100"

# Calibration: scale bar = 58 px = 20 µm  →  0.3448 µm/px
PIXEL_SIZE_UM = 20.0 / 58.0

# Detection parameters (Run 2 — Finer Watershed, validated on trans_without_scalebar.png)
CFG = {
    "gaussian_sigma": 1.0,
    "clahe_kernel_size": 60,
    "clahe_clip_limit": 0.03,
    "threshold_method": "otsu",
    "close_radius": 2,
    "open_radius": 2,
    "min_distance": 8,  # px — fine watershed split
    "min_area": 60,  # px²
    "max_area": 2000,  # px²
    "min_circularity": 0.25,
    "min_solidity": 0.45,
    # Boundary: rods whose bbox is within this many px of image edge → flagged
    # but NOT removed.  Set to 1 to catch anything truly touching the border.
    "boundary_margin": 1,
    "label_font_size": 5,
    "label_every_n": 1,
}

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────


def load_image(path):
    raw = io.imread(path, as_gray=True).astype(np.float32)
    lo, hi = raw.min(), raw.max()
    return (raw - lo) / (hi - lo) if hi > lo else raw


def preprocess(img):
    denoised = filters.gaussian(img, sigma=CFG["gaussian_sigma"])
    enhanced = exposure.equalize_adapthist(
        denoised,
        kernel_size=CFG["clahe_kernel_size"],
        clip_limit=CFG["clahe_clip_limit"],
    )
    t = filters.threshold_otsu(enhanced)
    binary = enhanced > t
    closed = morphology.binary_closing(binary, morphology.disk(CFG["close_radius"]))
    cleaned = morphology.binary_opening(closed, morphology.disk(CFG["open_radius"]))
    return cleaned


def watershed_labels(mask):
    dist = distance_transform_edt(mask)
    coords = peak_local_max(
        dist, min_distance=CFG["min_distance"], labels=mask, exclude_border=False
    )
    seeds = np.zeros(dist.shape, dtype=bool)
    seeds[tuple(coords.T)] = True
    markers, _ = ndi_label(seeds)
    return watershed(-dist, markers, mask=mask)


def extract_rods(raw_labels, img, slice_num):
    """
    Return a list of rod dicts.  Every detected region is included;
    boundary rods are flagged with is_boundary=True rather than removed.
    """
    H, W = raw_labels.shape
    margin = CFG["boundary_margin"]
    props = measure.regionprops(raw_labels, intensity_image=img)
    records = []

    for p in props:
        a = p.area
        if a < CFG["min_area"] or a > CFG["max_area"]:
            continue
        perim = p.perimeter
        circ = (4 * np.pi * a / perim**2) if perim > 0 else 0.0
        if circ < CFG["min_circularity"]:
            continue
        if p.solidity < CFG["min_solidity"]:
            continue

        minr, minc, maxr, maxc = p.bbox
        # Flag — but do NOT skip — boundary rods
        is_boundary = bool(
            minr <= margin or minc <= margin or maxr >= H - margin or maxc >= W - margin
        )

        cy, cx = p.centroid
        records.append(
            {
                "slice": slice_num,
                "rod_id": None,  # filled after sorting
                "centroid_x": round(cx, 2),
                "centroid_y": round(cy, 2),
                "area_px": int(a),
                "equiv_diam_px": round(p.equivalent_diameter, 2),
                "equiv_diam_um": round(p.equivalent_diameter * PIXEL_SIZE_UM, 3),
                "major_axis_px": round(p.major_axis_length, 2),
                "minor_axis_px": round(p.minor_axis_length, 2),
                "eccentricity": round(p.eccentricity, 4),
                "orientation_deg": round(np.degrees(p.orientation), 2),
                "circularity": round(circ, 4),
                "solidity": round(p.solidity, 4),
                "mean_intensity": round(p.mean_intensity, 4),
                "is_boundary": is_boundary,
                "_label": p.label,
            }
        )

    # Stable spatial sort: rows top→bottom, then left→right
    records.sort(key=lambda r: (int(r["centroid_y"] // CFG["min_distance"]), r["centroid_x"]))
    for i, r in enumerate(records, start=1):
        r["rod_id"] = i

    # Build relabeled image (interior=rod_id, boundary rods keep their id too)
    label_map = {r["_label"]: r["rod_id"] for r in records}
    relabeled = np.zeros_like(raw_labels)
    boundary_set = set()
    for r in records:
        relabeled[raw_labels == r["_label"]] = r["rod_id"]
        if r["is_boundary"]:
            boundary_set.add(r["rod_id"])
        del r["_label"]

    return relabeled, records, boundary_set


def save_labeled_image(img_gray, relabeled, records, boundary_set, out_path):
    """
    Cyan outlines  = interior rods
    Orange outlines = boundary (partial) rods
    """
    rgb = color.gray2rgb((img_gray * 255).astype(np.uint8))

    # Build separate masks for interior vs boundary
    interior_mask = np.isin(relabeled, [r["rod_id"] for r in records if not r["is_boundary"]])
    boundary_mask = np.isin(relabeled, list(boundary_set))

    rgb[segmentation.find_boundaries(interior_mask.astype(int) * relabeled, mode="thick")] = [
        0,
        230,
        230,
    ]  # cyan
    rgb[segmentation.find_boundaries(boundary_mask.astype(int) * relabeled, mode="thick")] = [
        255,
        140,
        0,
    ]  # orange

    fig, ax = plt.subplots(figsize=(rgb.shape[1] / 100, rgb.shape[0] / 100), dpi=100)
    ax.imshow(rgb)
    ax.axis("off")
    every_n = CFG["label_every_n"]
    fs = CFG["label_font_size"]
    for r in records:
        if r["rod_id"] % every_n == 0:
            c = "orange" if r["is_boundary"] else "red"
            ax.text(
                r["centroid_x"],
                r["centroid_y"],
                str(r["rod_id"]),
                color=c,
                fontsize=fs,
                ha="center",
                va="center",
                fontweight="bold",
            )
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    io.imsave(out_path, buf)


def save_summary_plots(summary_rows, out_path):
    df = pd.DataFrame(summary_rows)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Rod count vs slice
    ax = axes[0, 0]
    ax.plot(df["slice"], df["n_total"], "o-", label="Total", color="steelblue")
    ax.plot(df["slice"], df["n_interior"], "s-", label="Interior", color="green")
    ax.plot(df["slice"], df["n_boundary"], "^-", label="Boundary", color="orange")
    ax.set_xlabel("Slice")
    ax.set_ylabel("Rod count")
    ax.set_title("Rod count vs slice")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Mean diameter vs slice
    ax = axes[0, 1]
    ax.plot(df["slice"], df["mean_diam_um"], "o-", color="purple")
    ax.fill_between(
        df["slice"],
        df["mean_diam_um"] - df["std_diam_um"],
        df["mean_diam_um"] + df["std_diam_um"],
        alpha=0.2,
        color="purple",
        label="±1 std",
    )
    ax.axhline(4, ls="--", color="red", lw=0.8, label="4 µm min")
    ax.axhline(9, ls="--", color="green", lw=0.8, label="9 µm max")
    ax.set_xlabel("Slice")
    ax.set_ylabel("Equiv. diameter (µm)")
    ax.set_title("Mean rod diameter vs slice")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Mean eccentricity vs slice
    ax = axes[1, 0]
    ax.plot(df["slice"], df["mean_ecc"], "o-", color="darkorange")
    ax.fill_between(
        df["slice"],
        df["mean_ecc"] - df["std_ecc"],
        df["mean_ecc"] + df["std_ecc"],
        alpha=0.2,
        color="darkorange",
    )
    ax.axhline(0.6, ls="--", color="gray", lw=0.8, label="ecc=0.6 (oblique threshold)")
    ax.set_xlabel("Slice")
    ax.set_ylabel("Eccentricity")
    ax.set_title("Mean eccentricity vs slice\n(high = obliquely cut rods / decussation zone)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Boundary fraction vs slice
    ax = axes[1, 1]
    frac = df["n_boundary"] / df["n_total"].clip(lower=1) * 100
    ax.bar(df["slice"], frac, color="orange", alpha=0.7)
    ax.set_xlabel("Slice")
    ax.set_ylabel("Boundary rods (%)")
    ax.set_title("Fraction of boundary (partial) rods per slice")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Validation: slices {summary_rows[0]['slice']}–{summary_rows[-1]['slice']}  "
        f"|  pixel_size={PIXEL_SIZE_UM:.4f} µm/px  "
        f"|  Run 2 params (min_d={CFG['min_distance']}, close_r={CFG['close_radius']})",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main():
    slices_out = os.path.join(OUTPUT_DIR, "slices")
    os.makedirs(slices_out, exist_ok=True)

    summary_rows = []

    for s in range(FIRST_SLICE, LAST_SLICE + 1):
        fname = os.path.join(STACK_DIR, f"lion_axial{s}.png")
        if not os.path.exists(fname):
            print(f"[skip] {fname} not found")
            continue

        print(f"[slice {s:3d}] loading...", end=" ", flush=True)
        img = load_image(fname)
        mask = preprocess(img)
        raw_labels = watershed_labels(mask)
        relabeled, records, boundary_set = extract_rods(raw_labels, img, s)

        n_total = len(records)
        n_boundary = sum(1 for r in records if r["is_boundary"])
        n_interior = n_total - n_boundary

        # Per-slice CSV
        csv_path = os.path.join(slices_out, f"slice_{s:03d}_rods.csv")
        col_order = [
            "slice",
            "rod_id",
            "centroid_x",
            "centroid_y",
            "area_px",
            "equiv_diam_px",
            "equiv_diam_um",
            "major_axis_px",
            "minor_axis_px",
            "eccentricity",
            "orientation_deg",
            "circularity",
            "solidity",
            "mean_intensity",
            "is_boundary",
        ]
        pd.DataFrame(records)[col_order].to_csv(csv_path, index=False)

        # Per-slice labeled image
        img_path = os.path.join(slices_out, f"slice_{s:03d}_labeled.png")
        save_labeled_image(img, relabeled, records, boundary_set, img_path)

        # Summary row (interior rods only for diameter/ecc stats — boundary ones are partial)
        interior = [r for r in records if not r["is_boundary"]]
        diams = np.array([r["equiv_diam_um"] for r in interior]) if interior else np.array([0])
        eccs = np.array([r["eccentricity"] for r in interior]) if interior else np.array([0])

        summary_rows.append(
            {
                "slice": s,
                "n_total": n_total,
                "n_interior": n_interior,
                "n_boundary": n_boundary,
                "mean_diam_um": round(diams.mean(), 3),
                "std_diam_um": round(diams.std(), 3),
                "mean_ecc": round(eccs.mean(), 4),
                "std_ecc": round(eccs.std(), 4),
            }
        )

        print(
            f"total={n_total:4d}  interior={n_interior:4d}  "
            f"boundary={n_boundary:3d}  "
            f"diam={diams.mean():.2f}±{diams.std():.2f} µm  "
            f"ecc={eccs.mean():.3f}"
        )

    # Summary CSV + plots
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
    save_summary_plots(summary_rows, os.path.join(OUTPUT_DIR, "summary_plots.png"))

    print("\n── Overall ─────────────────────────────────────────────")
    print(f"  Slices processed : {len(summary_rows)}")
    print(
        f"  Rod count range  : {summary_df['n_total'].min()}–{summary_df['n_total'].max()}"
        f"  (std={summary_df['n_total'].std():.1f})"
    )
    print(
        f"  Mean diam range  : {summary_df['mean_diam_um'].min():.2f}–"
        f"{summary_df['mean_diam_um'].max():.2f} µm"
    )
    print(
        f"  Boundary frac    : {(summary_df['n_boundary']/summary_df['n_total']).mean()*100:.1f}% avg"
    )
    print("────────────────────────────────────────────────────────")
    print(f"\nOutputs in: {OUTPUT_DIR}/")
    print("  slices/slice_NNN_labeled.png  — cyan=interior  orange=boundary")
    print("  slices/slice_NNN_rods.csv")
    print("  summary.csv")
    print("  summary_plots.png")


if __name__ == "__main__":
    main()
