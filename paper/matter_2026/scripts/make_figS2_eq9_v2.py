#!/usr/bin/env python
"""Figure S2 (v2): material-pair condition for crack deflection at a stiff rod.

Purpose
-------
Moved from main-text Figure 4F in the 2026-09-25 revision: the panel depends
only on the material pair, not on the lattice, so it is context for the
Discussion rather than a result of this study.

Plots the He-Hutchinson-type fit used by Jia & Wang (2019, their Eq. 9) for a
crack in the compliant phase meeting a stiffer rod: the crack deflects along
the interface when the interface-to-rod toughness ratio lies below

    G_d / G_p = 0.254 / (1 - alpha^1.2),
    alpha = (E_rod - E_m) / (E_rod + E_m)       (Dundurs alpha, equal nu),

so the shaded region below the curve is the deflection regime. Candidate
printable pairs are marked at their stiffness ratios (nominal moduli).

Inputs: none (closed form).
Outputs: figures/figureS2.{pdf,png,tif}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _style_v2 as S  # noqa: E402

OUT = HERE.parent / "figures" / "figureS2"


def main() -> None:
    S.apply()
    fig = plt.figure(figsize=(85 * S.MM, 66 * S.MM))
    ax = fig.add_axes([0.16, 0.17, 0.80, 0.74])

    # --- 1. Deflection boundary vs stiffness ratio ----------------------------------------
    ratio = np.logspace(0, 4, 400)  # E_rod / E_matrix
    alpha = (ratio - 1.0) / (ratio + 1.0)  # Dundurs alpha (crack in compliant phase)
    gdgp = 0.254 / (1.0 - alpha**1.2)
    ax.fill_between(ratio, 1e-2, gdgp, color=S.TEAL_D, alpha=0.12, lw=0)
    ax.plot(ratio, gdgp, color=S.TEAL_D, lw=1.1)

    # --- 2. Candidate material pairs (nominal Young's moduli, MPa) ---------------------------
    pairs = [
        ("resin / TPU", 3000 / 40),
        ("PLA / silicone", 3500 / 1.5),
        ("rigid resin / PDMS", 3000 / 2.0),
        ("Al$_2$O$_3$ / epoxy", 370 / 3.0),
    ]
    offsets = {
        "resin / TPU": (6, -9, "left"),
        "PLA / silicone": (-2, -13, "right"),
        "rigid resin / PDMS": (-5, 5, "right"),
        "Al$_2$O$_3$ / epoxy": (-5, 5, "right"),
    }
    for name, rr in pairs:
        a = (rr - 1) / (rr + 1)
        g = 0.254 / (1 - a**1.2)
        ax.plot(rr, g, "o", ms=3.2, mfc=S.INK, mec="white", mew=0.5, zorder=5)
        dx, dy, ha = offsets[name]
        ax.annotate(
            name,
            (rr, g),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=S.SMALL_PT,
            ha=ha,
            va="bottom",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e4)
    ax.set_ylim(0.1, 1e3)
    ax.set_xlabel(r"stiffness ratio $E_{\mathrm{rod}}/E_{\mathrm{matrix}}$")
    ax.set_ylabel(r"$G_d/G_p$")
    ax.text(
        2.2e3,
        0.2,
        "deflection if\n" r"$\Gamma_{\mathrm{int}}/\Gamma_{\mathrm{rod}} < G_d/G_p$",
        fontsize=S.SMALL_PT,
        color=S.INK,
        va="bottom",
        ha="right",
    )
    S.recessive_grid(ax, "both")
    S.save(fig, OUT)
    print("figure S2 written")


if __name__ == "__main__":
    main()
