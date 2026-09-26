#!/usr/bin/env python
"""Additional real-data thumbnails for the v3 pipeline schematic (Figure 1).

Purpose
-------
Figure 1 (figures/src/figure1_pipeline_v3.tex) shows one circular thumbnail
per pipeline stage, each a crop of this study's own data. make_thumbnails_v2.py
produces the segmentation, trajectory, SOM, CAD, FEA and print thumbnails; the
v3 schematic adds three stages that need their own:
  thumb_watershed.png  distance-transform watershed rod detection (slice 51):
                       accepted rods outlined, rejected regions filled
  thumb_piv.png        PIV displacement vectors (slices 51 -> 52) over the mask
  thumb_twin.png       crop of the rod-resolved digital twin render
Each is a square 800 x 800 px PNG (the TikZ figure clips it to a circle).

Inputs
------
microct_pipeline/validate_stack_100.py (detection functions, imported)
microct_pipeline/output_piv/piv_fields/slice_051_uv.npz
figures/panels/render_twin_crop.png

Outputs
-------
figures/panels/thumb_{watershed,piv,twin}.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image
from skimage.segmentation import find_boundaries

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _style_v2 as S  # noqa: E402
import make_fig2_pipeline_v3 as P  # noqa: E402  (detection wrapper and paths)

PANELS = HERE.parent / "figures" / "panels"
SIZE_PX = 800


def save_square(fig, path: Path) -> None:
    fig.savefig(path, dpi=SIZE_PX / 2.0, facecolor="white")
    plt.close(fig)


def main() -> None:
    S.apply()
    mod = P.load_detection_module()
    mask, labels, accepted, rejected, *_ = P.detect(mod, P.SLICE)
    # --- 1. Watershed thumbnail: 40 x 40 um window ------------------------------------
    w = int(40 / P.PX_UM)
    y0, x0 = 330, 330
    sl = (slice(y0, y0 + w), slice(x0, x0 + w))
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(mask[sl], cmap=ListedColormap(["#FFFFFF", "#D9DCE3"]), interpolation="nearest")
    rej = rejected[sl] > 0
    ax.imshow(
        np.ma.masked_where(~rej, rej),
        cmap=ListedColormap([S.WINE_D]),
        alpha=0.8,
        interpolation="nearest",
    )
    b = find_boundaries(accepted[sl], mode="inner")
    ax.imshow(np.ma.masked_where(~b, b), cmap=ListedColormap([S.NAVY_D]), interpolation="nearest")
    save_square(fig, PANELS / "thumb_watershed.png")

    # --- 2. PIV thumbnail: central 120 x 120 um of the field --------------------------------
    piv = np.load(P.PIV_NPZ)
    GX, GY = np.meshgrid(piv["cx"], piv["cy"])
    H, W = mask.shape
    c = int(120 / P.PX_UM)
    ys, xs = (H - c) // 2, (W - c) // 2
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(mask, cmap=ListedColormap(["#FFFFFF", "#D9DCE3"]), interpolation="nearest")
    ax.quiver(
        GX,
        GY,
        piv["U"],
        piv["V"],
        color=S.NAVY_D,
        angles="xy",
        scale_units="xy",
        scale=1 / 60.0,
        width=0.012,
        headwidth=3.2,
        headlength=3.8,
    )
    ax.set_xlim(xs, xs + c)
    ax.set_ylim(ys + c, ys)
    save_square(fig, PANELS / "thumb_piv.png")

    # --- 3. Digital-twin thumbnail: centre crop of the column render ---------------------------
    im = Image.open(PANELS / "render_twin_crop.png").convert("RGBA")
    bg = Image.new("RGBA", im.size, "white")
    bg.alpha_composite(im)
    s = min(im.size)
    left, top = (im.width - s) // 2, (im.height - s) // 2
    bg.crop((left, top, left + s, top + s)).convert("RGB").resize(
        (SIZE_PX, SIZE_PX), Image.LANCZOS
    ).save(PANELS / "thumb_twin.png")

    # --- 4. Design-space thumbnail: apparent modulus vs twist scale (Fig. 6 data) --------------
    import json

    dec = json.loads((HERE.parent / "data" / "fig5_values.json").read_text())["decussation"]
    f = np.array([r["f"] for r in dec])
    e = np.array([r["E_app"] for r in dec])
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes([0.2, 0.22, 0.62, 0.56])
    ax.plot(f, e, "-o", color=S.NAVY_D, lw=2.2, ms=4.5, mec="white", mew=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("twist", fontsize=11)
    ax.set_ylabel("stiffness", fontsize=11)
    save_square(fig, PANELS / "thumb_design.png")
    print("thumbnails: thumb_watershed.png, thumb_piv.png, thumb_twin.png, thumb_design.png")


if __name__ == "__main__":
    main()
