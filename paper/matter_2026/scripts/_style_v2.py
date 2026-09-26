"""Shared figure style for the Matter v2 manuscript figures.

Purpose
-------
One source of truth for colour, type, sizing, and export so every data figure
(Figs. 2-6, SI) reads as one system and matches the vector schematics in the
GitHub repositories (biomimetic-lattice-pipeline, ct-segmentation-toolkit,
som-multimodal-datareduction), which use page #EEEEEE, navy #1E3252, gold
#B8894A, ink #222222, and Helvetica.

Why this exists
---------------
The v1 figures mixed four visual styles (default matplotlib, afmhot maps,
pyvista screenshots with illegible embedded colourbars, a Gemini
infographic). Cell Press asks for Arial/Helvetica at 6-8 pt at final size,
figure widths of 85 / 114 / 174 mm, and >= 300 dpi for images (>= 1000 dpi
line art, or vector).

Palette provenance (validated 2026-09-25 with the dataviz skill's
validate_palette.js, OKLab CVD model of Machado et al. 2009)
-----------------------------------------------------------------------
* The house navy/gold (#1E3252/#B8894A) FAIL the categorical lightness band
  and chroma floor as data marks (navy L=0.32, C=0.06), so they are kept for
  diagram chrome, text accents and schematics only.
* Data marks use brighter steps in the same hue families:
      NAVY_D #325C99, GOLD_D #C4852E  -> 2-series: all checks PASS
      (+ TEAL_D #2E9A7F, WINE_D #A5465C) -> 4-series, --pairs all: all PASS
      (worst CVD pair dE 8.8 deutan; normal-vision floor dE 18.5).
* Categorical slots are assigned in this fixed order and never cycled.
* Sequential fields (stress magnitude) use 'cividis' (navy -> yellow,
  perceptually uniform, CVD-safe). Signed fields (tension/compression) use a
  navy <-> grey <-> gold diverging map with a neutral midpoint.

Inputs / outputs
----------------
Import-only module; no side effects beyond ``apply()`` setting rcParams.
``save(fig, stem)`` writes <stem>.pdf (vector, TrueType-embedded), <stem>.png
(600 dpi) and <stem>.tif (600 dpi, LZW) next to each other.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- 1. Colour tokens --------------------------------------------------------
# Diagram chrome (house style, from the repository TikZ sources)
PAGE = "#EEEEEE"  # schematic page grey
NAVY = "#1E3252"  # stage boxes, headings, emphasis text
GOLD = "#B8894A"  # rings/highlights in schematics
INK = "#222222"  # primary text
MUTED = "#6E6E6A"  # secondary text / phase labels
RULE = "#8C8C8C"  # brackets, reference lines
GRID = "#E3E3E0"  # hairline gridlines (recessive)
TONE = "#C9CED6"  # neutral fills

# Data marks (validated categorical order: slot 1..4)
NAVY_D = "#325C99"  # slot 1  biomimetic arm / primary series
GOLD_D = "#C4852E"  # slot 2  near-touching arm / second series
TEAL_D = "#2E9A7F"  # slot 3
WINE_D = "#A5465C"  # slot 4
CATEGORICAL = [NAVY_D, GOLD_D, TEAL_D, WINE_D]

# Signed (diverging) map: compression navy <- neutral grey -> tension gold.
DIVERGING = LinearSegmentedColormap.from_list(
    "navy_grey_gold", ["#17325E", "#325C99", "#9DB2D3", "#EDEDEB", "#E4C28F", "#C4852E", "#7A4A12"]
)
SEQUENTIAL = mpl.colormaps["cividis"]

# --- 2. Cell Press geometry (mm -> inches) -----------------------------------
MM = 1.0 / 25.4
WIDTH_1COL = 85 * MM
WIDTH_15COL = 114 * MM
WIDTH_2COL = 174 * MM
MAX_HEIGHT = 225 * MM

# --- 3. Typography -----------------------------------------------------------
BASE_PT = 7.0  # body text in figures (Cell Press: 6-8 pt at final size)
SMALL_PT = 6.0  # ticks, annotations
LABEL_PT = 9.0  # panel letters


def apply() -> None:
    """Set rcParams for publication figures (call once per script)."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.sf": "Arial",
            "font.size": BASE_PT,
            "axes.titlesize": BASE_PT,
            "axes.labelsize": BASE_PT,
            "xtick.labelsize": SMALL_PT,
            "ytick.labelsize": SMALL_PT,
            "legend.fontsize": SMALL_PT,
            "legend.frameon": False,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.5,  # hairline axes (pt)
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.6,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "patch.linewidth": 0.5,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,  # embed TrueType (editable text)
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax, letter: str, x: float = -0.02, y: float = 1.0, **kw) -> None:
    """Bold uppercase panel letter at the axes' top-left (Cell Press style)."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=LABEL_PT,
        fontweight="bold",
        va="bottom",
        ha="right",
        color=INK,
        **kw,
    )


def fig_label(fig, letter: str, x: float, y: float) -> None:
    """Panel letter placed in figure coordinates (for image panels)."""
    fig.text(x, y, letter, fontsize=LABEL_PT, fontweight="bold", va="top", ha="left", color=INK)


def recessive_grid(ax, axis: str = "y") -> None:
    """Hairline, solid, recessive gridlines behind data (never dashed)."""
    ax.grid(True, axis=axis, color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def scalebar(
    ax,
    length_px: float,
    label: str,
    loc=(0.05, 0.06),
    color="white",
    lw: float = 2.0,
    fontsize: float = SMALL_PT,
) -> None:
    """Axis-fraction-anchored scale bar for image panels (length in data px)."""
    x0, y0 = loc
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    w = abs(xlim[1] - xlim[0])
    frac = length_px / w
    ax.plot(
        [x0, x0 + frac],
        [y0, y0],
        transform=ax.transAxes,
        color=color,
        lw=lw,
        solid_capstyle="butt",
        zorder=10,
    )
    ax.text(
        x0 + frac / 2,
        y0 + 0.025,
        label,
        transform=ax.transAxes,
        color=color,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        zorder=10,
    )


def save(fig, stem: str | Path, tiff: bool = True) -> None:
    """Write <stem>.pdf (vector) + <stem>.png and <stem>.tif at 600 dpi."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    if tiff:
        fig.savefig(stem.with_suffix(".tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
