#!/usr/bin/env python
"""Figure 6 (v2): physical realization -- CAD geometry and the FDM print.

Purpose
-------
  A  Render of the exact STL that was printed: the N = 7 near-touching cell
     of the height-matched sweep (runs/sweep_H24_thick/trial_005_...), from a
     viewpoint close to the photograph's (side-on, 9 deg elevation).
  B  Photograph of the fused-deposition-modelling (FDM) print (Creality
     Ender-3 V2, grey PLA) exactly as supplied (JT4 deck, slide 2), without
     any enhancement, upscaling, or retouching -- Cell Press image-integrity
     rules. The burned-in "2.5 inches" scale bar is the author's.

Why this exists
---------------
v1 described the prototype as a stereolithography (SLA) print and claimed the
0.025-mm inter-rod gaps printed "without rod fusion". The print is FDM
(0.4-mm class extrusion), which cannot resolve that gap: the photograph shows
adjacent outer rods fused along their contact lines while the helical rod
columns, the seven bridge layers (junction arcs), and the plate-rod-plate
sandwich are reproduced. The caption and text now say exactly that.

Inputs
------
figures/panels/render_print.png                    (render_models_v2.py)
<scratchpad>/jt4_slide2/slide2_pic29_Picture_1.png copied to
figures/panels/photo_fdm_print_original.png        (432 x 304 px; see caveat)

Outputs
-------
figures/figure6.{pdf,png,tif}

Caveat
------
The only copy of the photograph found on disk is 432 x 304 px (every deck
that contains it embeds the same file). At 84 mm width that is ~130 dpi,
below the Cell Press 300-dpi guideline for photographs; replace
photo_fdm_print_original.png with the original camera file (>= 1,000 px
wide) and re-run this script -- no code change needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style_v2 as S  # noqa: E402

HERE = Path(__file__).resolve().parent
MANU = HERE.parent
RENDER = MANU / "figures" / "panels" / "render_print.png"
PHOTO = MANU / "figures" / "panels" / "photo_fdm_print_original.png"
OUT = MANU / "figures" / "figure6"


def trimmed(png: Path) -> np.ndarray:
    """RGBA render with transparent margins removed."""
    a = np.asarray(Image.open(png))
    if a.shape[-1] == 4:
        ys, xs = np.where(a[..., 3] > 10)
        a = a[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return a


def main() -> None:
    S.apply()
    FW, FH = 174.0, 66.0
    fig = plt.figure(figsize=(FW * S.MM, FH * S.MM))

    def ax_mm(x, y, w, h):
        return fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH])

    # --- A: CAD render of the printed STL -------------------------------------
    axA = ax_mm(4, 8, 80, 54)
    axA.imshow(trimmed(RENDER))
    axA.set_axis_off()
    axA.set_title(
        "CAD geometry of the printed lattice (densified, N = 7)",
        fontsize=S.BASE_PT,
        pad=3,
        loc="left",
    )
    fig.text(0.0, 1 - 3 / FH, "A", fontsize=S.LABEL_PT, fontweight="bold", va="top")

    # --- B: photograph, unmodified ------------------------------------------------
    photo = np.asarray(Image.open(PHOTO).convert("RGB"))
    ph_w = 80.0
    ph_h = ph_w * photo.shape[0] / photo.shape[1]
    axB = ax_mm(92, 8 + (54 - ph_h) / 2, ph_w, ph_h)
    axB.imshow(photo, interpolation="antialiased")  # display resampling only
    axB.set_axis_off()
    axB.set_title("FDM print (Ender-3 V2, grey PLA)", fontsize=S.BASE_PT, pad=3, loc="left")
    fig.text(88 / FW, 1 - 3 / FH, "B", fontsize=S.LABEL_PT, fontweight="bold", va="top")

    S.save(fig, OUT)
    print(
        f"photo {photo.shape[1]}x{photo.shape[0]} px shown at {ph_w:.0f} mm "
        f"-> {photo.shape[1] / (ph_w / 25.4):.0f} dpi effective"
    )


if __name__ == "__main__":
    main()
