#!/usr/bin/env python
"""Figure 3 (v2): from the measured descriptor to two synthetic systems.

Purpose
-------
  A  Digital twin: all 2,433 canonical trajectories swept at the MEASURED
     rod radius (2.19 um), coloured by signed pitch; close-up inset.
  B  Reference bioinspired lattice (continuous-twist, N = 2), plates grey.
  C  Transfer ledger: every CAD parameter with its value, its source, and
     how it was transferred (measured / rule / calibrated / clamped /
     default), followed by the measured descriptors that the continuous-
     twist lattice does NOT carry (SOM bands, alternating handedness,
     tortuosity, rod eccentricity; revision 2026-09-25, so that omissions
     are as visible as transfers). Replaces the v1 "biomimicry z-score" bars, which (i) were
     relative errors, not z-scores, (ii) displayed a lower-is-better score
     as if the twin scored 1.00, (iii) compared rod pitch against SOM band
     width, and (iv) called floor/default values "exact transfers".
  D  Geometric fidelity test: measured rod-tilt distribution (2,433 tracks)
     vs the lattice's rod-inclination distribution at the as-mapped twist
     (f = 1.0) and at the tilt-matched scale f* (minimum 1-Wasserstein
     distance). This is the angle-to-angle comparison the lattice can be
     held to.

Inputs
------
figures/panels/render_twin.png, render_twin_crop.png, render_lattice.png
                                         (render_models_v2.py)
runs/live_001_digital_twin_v2/cad_params.json      reference lattice
runs/live_001/morphometrics.json                   measured descriptor
output_piv/track_centerlines_piv.parquet           measured rod tilt

Outputs
-------
figures/figure3.{pdf,png,tif}; data/fig3_values.json (every printed number)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style_v2 as S  # noqa: E402

# --- 1. Paths -----------------------------------------------------------------
HERE = Path(__file__).resolve().parent
MANU = HERE.parent
BIO = MANU.parents[1]
REPO = BIO.parent
PANELS = MANU / "figures" / "panels"
CAD = BIO / "runs/live_001_digital_twin_v2/cad_params.json"
MORPH = BIO / "runs/live_001/morphometrics.json"
PARQUET = (
    REPO / "output_piv_signcorrected/track_centerlines_piv.parquet"
)  # sign-corrected PIV tracking (track_rods_piv_signcorrected.py; see SI)
OUT = MANU / "figures" / "figure3"
VALUES = MANU / "data" / "fig3_values.json"

GAUGE_MM = 19.0  # rod length between plate interfaces (20 mm - 2 x 0.5 mm overlap)
PX_UM = 20.0 / 58.0


def hex_radii(r: int, cs: float) -> np.ndarray:
    """Radial positions (mm) of the 6r rods of hexagonal ring r (1 for r = 0)."""
    if r == 0:
        return np.array([0.0])
    pts = []
    for side in range(6):
        a0, a1 = np.radians(60 * side), np.radians(60 * (side + 1))
        c0 = r * cs * np.array([np.cos(a0), np.sin(a0)])
        c1 = r * cs * np.array([np.cos(a1), np.sin(a1)])
        pts += [c0 + (c1 - c0) * k / r for k in range(r)]
    return np.linalg.norm(np.array(pts), axis=1)


def lattice_inclination(cad: dict, f: float) -> np.ndarray:
    """Per-rod helix inclination (deg) for a linear continuous twist scaled by f.

    A rod at radius R in a ring that rotates by phi over the gauge length L
    is a helix with tan(theta) = R * |phi| / L.
    """
    cs = cad["CENTER_SPACING"]
    out = []
    for k, phi_deg in cad["RING_ROTATION"].items():
        R = hex_radii(int(k), cs)
        out += list(np.degrees(np.arctan(R * abs(np.radians(phi_deg)) * f / GAUGE_MM)))
    return np.array(out)


def place_image(ax, png: Path, crop_frac=None):
    im = Image.open(png)
    arr = np.asarray(im)
    if arr.shape[-1] == 4:  # trim transparent margins
        ys, xs = np.where(arr[..., 3] > 10)
        arr = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    ax.imshow(arr)
    ax.set_axis_off()
    return arr.shape


def main() -> None:
    S.apply()
    cad = json.loads(CAD.read_text())
    morph = json.loads(MORPH.read_text())
    tracks = pd.read_parquet(PARQUET)
    tilt = tracks.tilt_deg.values
    vals: dict = {
        "measured_tilt_mean_deg": float(tilt.mean()),
        "measured_tilt_median_deg": float(np.median(tilt)),
        "measured_tilt_p90_deg": float(np.percentile(tilt, 90)),
    }

    # Tilt-matched decussation scale f*: minimum 1-Wasserstein distance.
    fs = np.round(np.arange(0.05, 2.0001, 0.01), 2)
    w1 = np.array([wasserstein_distance(lattice_inclination(cad, f), tilt) for f in fs])
    f_star = float(fs[np.argmin(w1)])
    inc1, incs = lattice_inclination(cad, 1.0), lattice_inclination(cad, f_star)
    vals.update(
        {
            "f_star": f_star,
            "W1_f_star_deg": float(w1.min()),
            "W1_f1_deg": float(wasserstein_distance(inc1, tilt)),
            "lattice_incl_mean_f1_deg": float(inc1.mean()),
            "lattice_incl_mean_fstar_deg": float(incs.mean()),
        }
    )

    # Measured rod packing pitch (hexagonal equivalent of the areal density).
    area_um2 = (
        (tracks.start_x_px.max() - tracks.start_x_px.min())
        * (tracks.start_y_px.max() - tracks.start_y_px.min())
        * PX_UM**2
    )
    pitch_um = float(np.sqrt(2.0 * area_um2 / len(tracks) / np.sqrt(3.0)))
    diam_um = float(np.mean(morph["depth_profiles"]["rod_diameter_um_mean"]))
    vals.update(
        {
            "packing_pitch_um": pitch_um,
            "rod_diameter_um": diam_um,
            "cad_pitch_bio_um": cad["CENTER_SPACING"] / 500.0 * 1000.0,
            "packing_area_um2": float(area_um2),
            "packing_n_rods": int(len(tracks)),
        }
    )
    # Descriptors listed as not transferred (values shown in the ledger).
    tau_s = json.loads((MANU / "data/fig2_values.json").read_text())["tortuosity_mean"]  # smoothed
    ecc = float(np.nanmean(morph["depth_profiles"]["eccentricity_mean"]))
    pitch_mean = float(np.mean(morph["depth_profiles"]["pitch_signed_deg"]))
    vals.update(
        {
            "tortuosity_smoothed_mean": tau_s,
            "eccentricity_mean": ecc,
            "pitch_signed_depth_mean_deg": pitch_mean,
        }
    )

    # --- 2. Layout (mm) ------------------------------------------------------------
    FW, FH = 174.0, 146.0
    fig = plt.figure(figsize=(FW * S.MM, FH * S.MM))

    def ax_mm(x, y, w, h, **kw):
        return fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH], **kw)

    def label_mm(letter, x, y):
        fig.text(x / FW, 1 - y / FH, letter, fontsize=S.LABEL_PT, fontweight="bold", va="top")

    # A -- digital twin -------------------------------------------------------------
    axA = ax_mm(4, 7, 84, 56)
    place_image(axA, PANELS / "render_twin.png")
    insA = axA.inset_axes([0.745, -0.02, 0.27, 0.46])
    place_image(insA, PANELS / "render_twin_crop.png")
    insA.set_axis_on()
    insA.set_xticks([])
    insA.set_yticks([])
    insA.set_facecolor("white")
    for sp in insA.spines.values():
        sp.set_color(S.RULE)
        sp.set_linewidth(0.6)
    insA.text(
        0.04,
        0.03,
        "60 × 60 µm column",
        transform=insA.transAxes,
        fontsize=5.4,
        color=S.MUTED,
        va="bottom",
        ha="left",
        bbox=dict(fc="white", ec="none", pad=0.6),
    )
    axA.set_title(
        f"Digital twin: {len(tracks):,} measured trajectories, rod Ø {diam_um:.2f} µm",
        fontsize=S.BASE_PT,
        pad=2,
        loc="left",
    )
    axA.text(
        0.0,
        -0.02,
        "281 × 262 × 34.5 µm; color = rod pitch (±30°, Fig. 3A)",
        transform=axA.transAxes,
        fontsize=S.SMALL_PT,
        color=S.MUTED,
        va="top",
    )
    label_mm("A", 0, 3)

    # B -- reference lattice -----------------------------------------------------------
    axB = ax_mm(96, 7, 74, 56)
    place_image(axB, PANELS / "render_lattice.png")
    axB.set_title(
        "Printable lattice (reference: N = 2, as translated, f = 1)",
        fontsize=S.BASE_PT,
        pad=2,
        loc="left",
    )
    axB.text(
        0.0,
        -0.02,
        "42 × 42 × 22.4 mm; 91 rods in 5 hexagonal rings; 500× scale",
        transform=axB.transAxes,
        fontsize=S.SMALL_PT,
        color=S.MUTED,
        va="top",
    )
    label_mm("B", 92, 3)

    # C -- transfer ledger ------------------------------------------------------------------
    axC = ax_mm(4, 72, 100, 66)
    axC.set_axis_off()
    axC.set_xlim(0, 100)
    axC.set_ylim(0, 100)
    kind_col = {
        "measured": S.NAVY_D,
        "verified": S.TEAL_D,
        "heuristic": S.WINE_D,
        "rule": S.MUTED,
        "clamped": S.GOLD_D,
        "default": "#B4B4B0",
        "not transferred": "white",
    }
    rr = cad["RING_ROTATION"]
    rows = [
        (
            "ROD_DIAMETER",
            f"{cad['ROD_DIAMETER']:.3f} mm",
            f"mean rod Ø {diam_um:.2f} µm × 500",
            "measured",
        ),
        (
            "CENTER_SPACING",
            f"{cad['CENTER_SPACING']:.3f} mm",
            f"1.5 d ceiling = {vals['cad_pitch_bio_um']:.2f} µm; packing pitch {pitch_um:.2f} µm",
            "rule",
        ),
        (
            "RING_ROTATION",
            f"{rr['1']:+.1f} … {rr['5']:+.1f}°".replace("-", "\u2212"),
            f"depth integral of mean signed pitch ({pitch_mean:+.1f}°, one sign)".replace(
                "-", "\u2212"
            ),
            "heuristic",
        ),
        (
            "twist scale f",
            "1",
            f"as translated; vs measured tilt $W_1$ {vals['W1_f1_deg']:.1f}° (optimum f* = {f_star:.2f})",
            "verified",
        ),
        (
            "N_BRIDGE_LAYERS",
            "2",
            "band period (≥85 µm) gives round(20/≥42.5) = 0 → floor 2",
            "clamped",
        ),
        ("BRIDGE_DIAMETER", f"{cad['BRIDGE_DIAMETER']:.3f} mm", "0.8 × ROD_DIAMETER", "rule"),
        ("ROD_TAPER_FACTOR", "0", "measured −0.041 → clamped to [0, 1]", "clamped"),
        ("JUNCTION_SPHERES", "off", "no room below plate (0.8 → 0)", "clamped"),
        ("N_RINGS", "5 (91 rods)", "generator default (sets specimen width)", "default"),
    ]
    # Measured descriptors that the continuous-twist lattice does not carry.
    omitted = [
        ("SOM bands", "B1–B4", "no band structure: one ring-wise twist profile"),
        ("handedness", "alternates", "lattice: every ring rotates in the same sense"),
        ("tortuosity", f"{tau_s:.3f}", "not used: rods are smooth helices"),
        ("eccentricity", f"{ecc:.2f}", "not used: circular rod sections"),
    ]
    y = 97.0
    axC.text(0, y, "CAD parameter", fontsize=S.SMALL_PT, fontweight="bold", va="center")
    axC.text(27, y, "value", fontsize=S.SMALL_PT, fontweight="bold", va="center")
    axC.text(43, y, "source", fontsize=S.SMALL_PT, fontweight="bold", va="center")
    axC.plot([0, 100], [y - 2.8, y - 2.8], color=S.INK, lw=0.5)
    dy = 6.3

    def ledger_row(yy, p, v, src, kind, last):
        """One ledger line: class swatch, name, value, source; hairline below."""
        axC.add_patch(
            FancyBboxPatch(
                (-3.2, yy - 1.4),
                2.0,
                2.8,
                boxstyle="round,pad=0.0,rounding_size=0.5",
                fc=kind_col[kind],
                ec=S.MUTED if kind == "not transferred" else "none",
                lw=0.5,
                clip_on=False,
            )
        )
        axC.text(0, yy, p, fontsize=5.6, va="center", family="DejaVu Sans Mono")
        axC.text(27, yy, v, fontsize=S.SMALL_PT, va="center")
        axC.text(43, yy, src, fontsize=5.6, va="center", color=S.INK)
        if not last:
            axC.plot([0, 100], [yy - dy / 2, yy - dy / 2], color=S.GRID, lw=0.4)

    for i, (p, v, src, kind) in enumerate(rows):
        ledger_row(y - 6.3 - i * dy, p, v, src, kind, i == len(rows) - 1)
    # Sub-header and the omitted descriptors
    y2 = y - 6.3 - len(rows) * dy - 1.2
    axC.text(
        0,
        y2,
        "Measured but not carried into the lattice",
        fontsize=S.SMALL_PT,
        fontweight="bold",
        va="center",
    )
    axC.plot([0, 100], [y2 - 2.8, y2 - 2.8], color=S.INK, lw=0.5)
    for i, (p, v, src) in enumerate(omitted):
        ledger_row(y2 - 6.3 - i * dy, p, v, src, "not transferred", i == len(omitted) - 1)
    # Legend of transfer kinds (two rows of four keep the labels legible)
    for j, kind in enumerate(
        ("measured", "verified", "heuristic", "rule", "clamped", "default", "not transferred")
    ):
        lx, ly = (j % 4) * 25.0, -5.0 - (j // 4) * 4.6
        axC.add_patch(
            FancyBboxPatch(
                (lx, ly - 1.4),
                2.2,
                2.8,
                boxstyle="round,pad=0,rounding_size=0.5",
                fc=kind_col[kind],
                ec=S.MUTED if kind == "not transferred" else "none",
                lw=0.5,
                clip_on=False,
            )
        )
        axC.text(lx + 3.4, ly, kind, fontsize=S.SMALL_PT, va="center", clip_on=False)
    label_mm("C", 0, 69)
    fig.text(
        4 / FW,
        1 - 69.5 / FH,
        "Traceability matrix: measurement to CAD",
        fontsize=S.BASE_PT,
        va="top",
    )

    # D -- geometric fidelity: rod tilt vs lattice inclination -----------------------------
    axD = ax_mm(117, 78, 53, 52)

    def ecdf(x):
        xs = np.sort(x)
        return np.r_[0, xs], np.r_[0, np.arange(1, xs.size + 1) / xs.size]

    xm, ym = ecdf(tilt)
    axD.step(
        xm, ym, where="post", color="#8A93A3", lw=2.2, label=f"enamel rods (n = {len(tilt):,})"
    )
    axD.step(
        *ecdf(inc1),
        where="post",
        color=S.NAVY_D,
        lw=1.2,
        label=f"lattice as translated, f = 1 ($W_1$ {vals['W1_f1_deg']:.1f}°)",
    )
    # inset: 1-Wasserstein distance between lattice inclinations and measured tilt vs twist scale
    ins = axD.inset_axes([0.60, 0.30, 0.37, 0.30])
    ins.plot(fs, w1, color=S.INK, lw=0.8)
    ins.axvline(f_star, color=S.TEAL_D, lw=0.7)
    ins.set_xlabel("twist scale f", fontsize=5.0, labelpad=0.5)
    ins.set_ylabel("$W_1$ (°)", fontsize=5.0, labelpad=0.5)
    ins.tick_params(labelsize=4.8, length=1.5, pad=1)
    ins.set_xlim(0, 2)
    ins.set_ylim(0, None)
    ins.text(
        f_star + 0.06,
        0.85,
        f"f* = {f_star:.2f}",
        transform=ins.get_xaxis_transform(),
        fontsize=4.8,
        color=S.TEAL_D,
    )
    axD.set_ylim(0, 1.02)
    axD.set_xlabel("rod inclination to loading axis (°)")
    axD.set_ylabel("cumulative fraction of rods")
    axD.set_xlim(0, 60)
    axD.legend(
        loc="lower right", fontsize=5.6, handlelength=1.4
    )  # lower right is free of both curves
    S.recessive_grid(axD)
    axD.set_title("Geometric verification: inclination", fontsize=S.BASE_PT, pad=3, loc="left")
    label_mm("D", 110, 69)

    S.save(fig, OUT)
    VALUES.write_text(json.dumps(vals, indent=1))
    print(json.dumps(vals, indent=1))


if __name__ == "__main__":
    main()
