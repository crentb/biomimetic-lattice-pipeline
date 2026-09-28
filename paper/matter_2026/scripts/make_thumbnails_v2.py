#!/usr/bin/env python
"""Square real-data thumbnails for the Fig. 1 schematic and graphical abstract.

Purpose
-------
The v1 Figure 1 and graphical abstract were Gemini-generated images whose
insets depicted results that do not exist (a hand-held black lattice with
"1 mm" scale bars, an FEA field with a garbled colour-bar label) and carried
the generator's watermark. Cell Press does not permit general-purpose
generative-AI images in graphical abstracts or in images that present data.
This script cuts every vignette from the paper's own data so the redrawn
vector schematic shows only real results.

Outputs (figures/panels/thumb_*.png, 800 x 800 px, RGBA)
  thumb_ct      U-Net segmentation of slice 51, ~70 um field
  thumb_tracks  60-um column of the digital twin (render_twin_crop.png)
  thumb_som     sign-corrected SOM bands (as in Figure 3) with displacement streamlines, ~150 um field
  thumb_cad     reference lattice render (render_lattice.png)
  thumb_fea     signed dominant principal stress, mid-gauge section
  thumb_print   the FDM print photograph, square crop (no enhancement)

Inputs: the same files as Figs. 2-6 (see those scripts).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, Normalize
from PIL import Image

HERE = Path(__file__).resolve().parent
MANU = HERE.parent
BIO = MANU.parents[1]
REPO = BIO.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BIO / "scripts"))
import _style_v2 as S  # noqa: E402

PANELS = MANU / "figures" / "panels"
SIZE = 800


def square_png(arr: np.ndarray, out: Path, bg=(255, 255, 255, 0)) -> None:
    """Pad an image array to a square canvas and resize to SIZE x SIZE."""
    im = Image.fromarray(arr)
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    side = max(im.size)
    canvas = Image.new("RGBA", (side, side), bg)
    canvas.paste(im, ((side - im.size[0]) // 2, (side - im.size[1]) // 2), im)
    canvas.resize((SIZE, SIZE), Image.LANCZOS).save(out)


def trimmed(png: Path) -> np.ndarray:
    a = np.asarray(Image.open(png).convert("RGBA"))
    ys, xs = np.where(a[..., 3] > 10)
    return a[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def main() -> None:
    S.apply()
    PANELS.mkdir(parents=True, exist_ok=True)

    # --- 1. CT segmentation crop (~70 um) -----------------------------------------
    full = np.asarray(Image.open(REPO / "onehundred_image_stack/lion_axial51.png").convert("L"))
    px = int(70 / (20 / 58))
    y0, x0 = 300, 330
    crop = full[y0 : y0 + px, x0 : x0 + px]
    square_png(np.stack([crop] * 3 + [np.full_like(crop, 255)], -1), PANELS / "thumb_ct.png")

    # --- 2. trajectories: digital-twin column render ---------------------------------
    square_png(trimmed(PANELS / "render_twin_crop.png"), PANELS / "thumb_tracks.png")

    # --- 3. SOM bands + streamlines crop (~150 um) ------------------------------------
    z = np.load(
        MANU / "data/som_signcheck_bands.npz"
    )  # sign-corrected partition (som_signcheck.py), as in Fig. 3
    lab, dx, dy = z["cluster_map"], z["accum_dx_sm"], z["accum_dy_sm"]
    s0, n = 200, int(150 / (20 / 58))
    sl = (slice(s0, s0 + n), slice(s0 + 60, s0 + 60 + n))
    fig = plt.figure(figsize=(4, 4), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(
        lab[sl], cmap=ListedColormap(S.CATEGORICAL), vmin=-0.5, vmax=3.5, interpolation="nearest"
    )
    yy, xx = np.mgrid[0:n:10, 0:n:10]
    ax.streamplot(
        xx[0],
        yy[:, 0],
        dx[sl][::10, ::10],
        dy[sl][::10, ::10],
        color="white",
        linewidth=0.8,
        density=1.1,
        arrowsize=0.6,
    )
    ax.set_xlim(0, n - 1)
    ax.set_ylim(n - 1, 0)
    ax.set_axis_off()
    fig.savefig(PANELS / "thumb_som.png", dpi=200, transparent=True)
    plt.close(fig)
    square_png(
        np.asarray(Image.open(PANELS / "thumb_som.png").convert("RGBA")), PANELS / "thumb_som.png"
    )

    # --- 4. CAD lattice -------------------------------------------------------------------
    square_png(trimmed(PANELS / "render_lattice.png"), PANELS / "thumb_cad.png")

    # --- 5. FEA: signed dominant principal stress section ----------------------------------
    import make_fig4_v2 as F4  # reuse the exact section routine of Fig. 4

    # load_fields returns the gauge-window mask and mean since the 2026-09-25 revision
    tets, cent, vm, dom, vol, _win, vm_bar, _ = F4.load_fields(F4.REF)
    import analyze_load_paths as L

    zlo, zhi = L.gauge_bounds(cent)
    polys, sec = F4.section(tets, {"dom": dom / vm_bar}, 0.5 * (zlo + zhi))
    fig = plt.figure(figsize=(4, 4), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.add_collection(
        PolyCollection(
            polys,
            array=sec["dom"],
            cmap=S.DIVERGING,
            norm=Normalize(-2, 2),
            edgecolors="face",
            linewidths=0.05,
        )
    )  # same range as Fig. 4B
    ax.set_xlim(-19, 19)
    ax.set_ylim(-19, 19)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.savefig(PANELS / "thumb_fea.png", dpi=200, transparent=True)
    plt.close(fig)

    # --- 6. print photograph, square crop around the lattice (no enhancement) ---------------
    ph = np.asarray(Image.open(PANELS / "photo_fdm_print_original.png").convert("RGB"))
    # 230-px square centred on the lattice, above the burned-in scale bar
    sq = ph[18:248, 100:330]
    square_png(np.dstack([sq, np.full(sq.shape[:2], 255, np.uint8)]), PANELS / "thumb_print.png")
    print("thumbnails written:", sorted(p.name for p in PANELS.glob("thumb_*.png")))


if __name__ == "__main__":
    main()
