#!/usr/bin/env python
"""Figure 4 (v2): linear-elastic mechanics of the reference lattice.

Purpose
-------
Shows HOW the continuous-twist lattice carries an axial compressive load,
using stress tensors rebuilt in sfepy's true component order (see
biomimetic_pipeline/scripts/analyze_load_paths.py for the two defects in the
stock post-processor that this avoids). Because the analysis is linear,
every stress is shown normalised by sigma_bar_g, the volume-averaged von
Mises stress over the GAUGE WINDOW (the lattice between the plates, minus
1 mm next to each plate interface). The solid plates are lightly stressed,
so a whole-model mean (the solver's 200-MPa scaling convention) would
understate the stress level in the lattice by a factor of ~1.7 (review
revision 2026-09-25). Only ratios are physical.

  A  von Mises stress, mid-gauge transverse section (sigma / sigma_VM_bar).
  B  Signed dominant principal stress on the same section: inner, near-axial
     rods carry compression; the strongly inclined outer ring carries TENSION.
  C  Principal-stress trajectories (integral curves of the dominant
     principal direction) in 3-D, coloured compression (navy) / tension
     (gold) along their length.
  D  Ring-resolved load partition: volume-weighted mean dominant stress per
     hexagonal ring for the straight-rod control (f = 0.05), the lattice
     closest to the tilt-matched twist (f = 0.55; f* = 0.53) and the
     as-mapped reference (f = 1.0).
  E  Twist-compression coupling: in-plane displacement of the free top plate
     (rigid rotation), per 1 % axial strain.
  F  Mechanistic controls at the reference ring-rotation magnitudes:
     tension-dominated gauge volume fraction (bars) with the top-plate
     rotation per 1 % strain printed beside each bar, for the straight-rod
     control, the reference, an alternating-handedness (achiral) lattice,
     and the reference under a non-rotating (in-plane clamped) top platen.
     Controls that have not been run are listed as pending.
     (The Jia-Wang Eq. 9 material-pair panel moved to Fig. S2.)

Inputs
------
runs/live_001_digital_twin_v2/fea/final/compound_enamel_lattice.vtk  (f = 1.0)
runs/sweep_decussation_v2/trial_000_decussation_0p05/...             (straight control)
runs/sweep_decussation_dense_v3/trial_011_decussation_0p55/...       (near f*)
runs/control_achiral_v1/trial_000_alternating, runs/control_platen_fixed_v1
runs/_load_path_analysis/*.json, *_paths.npz                         (trajectories)

Outputs
-------
figures/figure4.{pdf,png,tif}; data/fig4_values.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
MANU = HERE.parent
BIO = MANU.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BIO / "scripts"))
import _style_v2 as S  # noqa: E402
import analyze_load_paths as L  # noqa: E402
from compute_robust_scf import weighted_percentile  # noqa: E402

REF = BIO / "runs/live_001_digital_twin_v2"
CTRL = BIO / "runs/sweep_decussation_v2/trial_000_decussation_0p05"
FSTAR = BIO / "runs/sweep_decussation_dense_v3/trial_011_decussation_0p55"
ACHIRAL = BIO / "runs/control_achiral_v1/trial_000_alternating"
PLATEN = BIO / "runs/control_platen_fixed_v1"
WINDOW_INSET_MM = 1.0  # gauge window = gauge minus 1 mm at each plate (as the partition metrics)
LPA = BIO / "runs/_load_path_analysis"
OUT = MANU / "figures" / "figure4"
VALUES = MANU / "data" / "fig4_values.json"


def load_fields(run: Path):
    """Tets, centroids, VM (per tet), signed dominant principal stress, gauge
    window mask, gauge-window mean VM (sigma_bar_g), grid."""
    tets, cent, e_dom, s_dom, evals, grid = L.load_run(run, "dominant")
    vm = np.asarray(grid.cell_data["von_mises"])[np.asarray(grid.celltypes) == 10]
    dom = np.where(np.abs(evals[:, 0]) >= np.abs(evals[:, 2]), evals[:, 0], evals[:, 2])
    vol = np.abs(tets.compute_cell_sizes(length=False, area=False, volume=True).cell_data["Volume"])
    z_lo, z_hi = L.gauge_bounds(cent)
    win = (cent[:, 2] > z_lo + WINDOW_INSET_MM) & (cent[:, 2] < z_hi - WINDOW_INSET_MM)
    vm_bar = float((vm[win] * vol[win]).sum() / vol[win].sum())  # plates excluded
    return tets, cent, vm, dom, vol, win, vm_bar, grid


def section(tets, arrays: dict, z: float):
    """Planar section at height z: list of polygons (xy) + per-polygon values."""
    for k, v in arrays.items():
        tets.cell_data[k] = v
    sl = tets.slice(normal="z", origin=(0.0, 0.0, z))
    polys, out = [], {k: [] for k in arrays}
    faces = sl.faces
    i = 0
    cid = 0
    while i < len(faces):
        n = faces[i]
        idx = faces[i + 1 : i + 1 + n]
        polys.append(sl.points[idx, :2])
        for k in arrays:
            out[k].append(sl.cell_data[k][cid])
        i += n + 1
        cid += 1
    return polys, {k: np.asarray(v) for k, v in out.items()}


def main() -> None:
    S.apply()
    vals: dict = {}
    tets, cent, vm, dom, vol, win, vm_bar, grid = load_fields(REF)
    z_lo, z_hi = L.gauge_bounds(cent)
    z_mid = 0.5 * (z_lo + z_hi)
    vm_bar_all = float((vm * vol).sum() / vol.sum())
    # kappa_99: volume-weighted P99 / mean von Mises over the gauge window.
    vals.update(
        {
            "vm_bar_gauge_MPa": vm_bar,
            "vm_bar_all_MPa": vm_bar_all,
            "sigma_ratio_gauge_over_all": vm_bar / vm_bar_all,
            "z_mid_mm": z_mid,
            "kappa99_gauge": weighted_percentile(vm[win], vol[win], 99.0) / vm_bar,
        }
    )
    polys, sec = section(tets, {"vm": vm / vm_bar, "dom": dom / vm_bar}, z_mid)

    FW, FH = 174.0, 128.0
    fig = plt.figure(figsize=(FW * S.MM, FH * S.MM))

    def ax_mm(x, y, w, h, **kw):
        return fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH], **kw)

    def label_mm(letter, x, y):
        fig.text(x / FW, 1 - y / FH, letter, fontsize=S.LABEL_PT, fontweight="bold", va="top")

    lim = 19.0
    # --- A: von Mises section --------------------------------------------------------
    axA = ax_mm(3, 8, 48, 48)
    pcA = PolyCollection(
        polys,
        array=sec["vm"],
        cmap=S.SEQUENTIAL,
        norm=Normalize(0, 2.5),
        edgecolors="face",
        linewidths=0.05,
    )
    axA.add_collection(pcA)
    axA.set_xlim(-lim, lim)
    axA.set_ylim(-lim, lim)
    axA.set_aspect("equal")
    axA.set_axis_off()
    caxA = ax_mm(8, 57.5, 38, 1.6)
    cb = fig.colorbar(
        pcA, cax=caxA, orientation="horizontal", extend="max", ticks=[0, 0.5, 1, 1.5, 2, 2.5]
    )
    cb.set_label(r"von Mises  $\sigma_{VM}/\bar{\sigma}_{g}$", fontsize=S.SMALL_PT, labelpad=1)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2)
    axA.plot([-lim + 1, -lim + 11], [-lim + 1, -lim + 1], color=S.INK, lw=1.2)
    axA.text(-lim + 6, -lim + 1.8, "10 mm", fontsize=S.SMALL_PT, ha="center")
    axA.set_title("Mid-gauge section, compression", fontsize=S.BASE_PT, pad=2)
    label_mm("A", 0, 3)

    # --- B: signed dominant principal stress --------------------------------------------
    axB = ax_mm(60, 8, 48, 48)
    pcB = PolyCollection(
        polys,
        array=sec["dom"],
        cmap=S.DIVERGING,
        norm=Normalize(-2.0, 2.0),
        edgecolors="face",
        linewidths=0.05,
    )
    axB.add_collection(pcB)
    axB.set_xlim(-lim, lim)
    axB.set_ylim(-lim, lim)
    axB.set_aspect("equal")
    axB.set_axis_off()
    caxB = ax_mm(65, 57.5, 38, 1.6)
    cb = fig.colorbar(
        pcB, cax=caxB, orientation="horizontal", extend="both", ticks=[-2, -1, 0, 1, 2]
    )
    cb.set_label(
        r"dominant principal  $\sigma_{dom}/\bar{\sigma}_{g}$  (− compression, + tension)",
        fontsize=S.SMALL_PT,
        labelpad=1,
    )
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2)
    axB.set_title("Tension–compression partition", fontsize=S.BASE_PT, pad=2)
    label_mm("B", 57, 3)

    # --- C: principal-stress trajectories, coloured by sign of the dominant stress ---------
    axC = ax_mm(114, 2, 60, 60, projection="3d")
    z = np.load(LPA / "live_001_digital_twin_v2_paths.npz")
    lengths, xyz = z["lengths"], z["xyz"]
    off = np.r_[0, np.cumsum(lengths)]
    tree = cKDTree(cent)
    rng = np.random.default_rng(3)
    pick = rng.choice(len(lengths), size=min(170, len(lengths)), replace=False)
    segs, cols = [], []
    for i in pick:
        p = xyz[off[i] : off[i + 1]][::3]
        if len(p) < 3:
            continue
        _, j = tree.query(0.5 * (p[1:] + p[:-1]))
        sign = np.sign(dom[j])
        segs += [np.stack([p[k], p[k + 1]]) for k in range(len(p) - 1)]
        cols += [S.GOLD_D if sg > 0 else S.NAVY_D for sg in sign]
    lc = Line3DCollection(segs, colors=cols, linewidths=0.45, alpha=0.9)
    axC.add_collection3d(lc)
    b = tets.bounds
    for zz in (z_lo, z_hi):  # plate interfaces (outline)
        xs = [b[0], b[1], b[1], b[0], b[0]]
        ys = [b[2], b[2], b[3], b[3], b[2]]
        axC.plot(xs, ys, [zz] * 5, color=S.RULE, lw=0.4)
    axC.set_xlim(b[0], b[1])
    axC.set_ylim(b[2], b[3])
    axC.set_zlim(z_lo, z_hi)
    axC.set_box_aspect((b[1] - b[0], b[3] - b[2], z_hi - z_lo), zoom=0.9)
    axC.view_init(elev=16, azim=-60)
    axC.set_axis_off()
    fig.text(
        (114 + 30) / FW,
        1 - 4 / FH,
        "Principal-stress trajectories",
        fontsize=S.BASE_PT,
        ha="center",
        va="top",
    )
    fig.legend(
        handles=[
            plt.Line2D([], [], color=S.NAVY_D, lw=1.2, label="compression"),
            plt.Line2D([], [], color=S.GOLD_D, lw=1.2, label="tension"),
        ],
        loc="upper center",
        bbox_to_anchor=((114 + 30) / FW, 1 - 56 / FH),
        ncol=2,
        fontsize=S.SMALL_PT,
        handlelength=1.6,
        columnspacing=1.2,
        frameon=False,
    )
    label_mm("C", 111, 3)

    # --- D: ring-resolved partition at three twist levels ---------------------------------
    # Each run is normalised by its OWN gauge-window mean von Mises stress.
    axD = ax_mm(12, 74, 44, 44)
    cad = json.loads((REF / "cad_params.json").read_text())
    rings = np.arange(6)
    series = [
        (
            "straight rods, f = 0.05",
            CTRL,
            "sweep_decussation_v2__trial_000_decussation_0p05",
            "#B4B4B0",
        ),
        (
            "intermediate, f = 0.55",
            FSTAR,
            "sweep_decussation_dense_v3__trial_011_decussation_0p55",
            S.TEAL_D,
        ),
        ("as translated, f = 1.0", REF, "live_001_digital_twin_v2", S.NAVY_D),
    ]
    bw = 0.26
    axD.axhline(0, color=S.RULE, lw=0.5)
    for j, (lab, run, key, col) in enumerate(series):
        res = json.loads((LPA / f"{key}.json").read_text())
        bar_g = vm_bar if run == REF else load_fields(run)[6]
        v = (
            np.array(
                [
                    res["load_partition"]["by_ring"][str(k)]["dominant_stress_mean_MPa"]
                    for k in rings
                ]
            )
            / bar_g
        )
        axD.bar(rings + (j - 1) * bw, v, width=bw * 0.92, color=col, zorder=3, label=lab, lw=0)
        vals[f"ring_dom_norm_{key}"] = v.tolist()
        vals[f"tensile_volume_frac_{key}"] = res["load_partition"]["tensile_volume_frac"]
    incl = [
        np.degrees(
            np.arctan(
                k * cad["CENTER_SPACING"] * abs(np.radians(cad["RING_ROTATION"][str(k)])) / 19.0
            )
        )
        for k in rings
    ]
    vals["ring_incl_deg_f1"] = incl
    axD.set_xticks(rings)
    axD.set_xticklabels([f"{k}\n{a:.0f}°" for k, a in zip(rings, incl)])
    axD.set_xlabel("ring  /  helix inclination at f = 1")
    axD.set_ylabel(r"mean $\sigma_{dom}/\bar{\sigma}_{g}$  (+ tension)")
    axD.set_ylim(-2.6, 0.9)
    axD.legend(
        loc="lower right", fontsize=5.4, handlelength=1.0, borderaxespad=0.2, labelspacing=0.3
    )
    S.recessive_grid(axD)
    label_mm("D", 0, 70)

    # --- E: twist-compression coupling (top-plate in-plane displacement) ------------------
    axE = ax_mm(60, 74, 42, 42)
    ref = json.loads((LPA / "live_001_digital_twin_v2.json").read_text())
    P = grid.points
    U = np.asarray(grid.point_data["u"])
    top = np.where(P[:, 2] > P[:, 2].max() - 1e-6)[0]
    gx, gy = np.meshgrid(np.linspace(-19, 19, 11), np.linspace(-19, 19, 11))
    _, near = cKDTree(P[top, :2]).query(np.column_stack([gx.ravel(), gy.ravel()]))
    sub = top[np.unique(near)]
    strain_pct = 100.0 * ref["top_plate"]["nominal_axial_strain"]
    q = axE.quiver(
        P[sub, 0],
        P[sub, 1],
        U[sub, 0] / strain_pct,
        U[sub, 1] / strain_pct,
        color=S.NAVY_D,
        angles="xy",
        scale_units="xy",
        scale=0.16,
        width=0.007,
        headwidth=3.5,
        headlength=4,
    )
    axE.quiverkey(
        q,
        0.06,
        0.955,
        0.8,
        "0.8 mm per 1 % strain",
        labelpos="E",
        coordinates="axes",
        fontproperties={"size": S.SMALL_PT},
    )
    axE.set_aspect("equal")
    axE.set_xlim(-23, 23)
    axE.set_ylim(-23, 29)
    axE.set_xticks([-20, 0, 20])
    axE.set_yticks([-20, 0, 20])
    axE.set_xlabel("x (mm)")
    axE.set_ylabel("y (mm)")
    rot = ref["top_plate"]["rotation_deg_per_pct_strain"]
    # True minus sign on the number only.
    axE.set_title(
        "Platen rotation\n" + f"{rot:+.2f}".replace("-", "\u2212") + "° per 1 % strain",
        fontsize=S.BASE_PT,
        pad=2,
    )
    vals["rotation_deg_per_pct_strain"] = rot
    label_mm("E", 52, 70)

    # --- F: mechanistic controls ---------------------------------------------------------
    # One measure on the axis (tension-dominated gauge volume fraction); the
    # top-plate rotation is printed as text beside each bar (no second axis).
    axF = ax_mm(128, 74, 42, 40)
    controls = [
        (
            "straight rods\n(f = 0.05)",
            "sweep_decussation_v2__trial_000_decussation_0p05",
            "#B4B4B0",
            CTRL,
        ),
        ("reference\n(f = 1, single handedness)", "live_001_digital_twin_v2", S.NAVY_D, REF),
        ("alternating\nhandedness", "control_achiral_v1__trial_000_alternating", S.GOLD_D, ACHIRAL),
        ("rotation-\nconstrained platen", "control_platen_fixed_v1", S.WINE_D, PLATEN),
    ]
    # Apparent modulus of each control from its axial stiffness F/delta: all four
    # share the reference footprint and height, so E_app scales with F/delta.
    ref_m = json.loads((REF / "metrics.json").read_text())
    e_app_ref = float(ref_m["E_effective_MPa"]) * float(ref_m["solid_fraction"])

    def stiffness(run: Path) -> float:
        import pandas as pd

        gr = pd.read_csv(run / "fea" / "final" / "global_results_compression.csv").iloc[0]
        return abs(float(gr.force_N) / float(gr.disp_mm))

    k_ref = stiffness(REF)
    for i, (lab, key, col, run) in enumerate(controls):
        y = len(controls) - 1 - i
        f = LPA / f"{key}.json"
        if not f.exists():
            axF.text(0.012, y, "pending", fontsize=S.SMALL_PT, color=S.MUTED, va="center")
            continue
        res = json.loads(f.read_text())
        t = res["load_partition"]["tensile_volume_frac"]
        rot = res["top_plate"]["rotation_deg_per_pct_strain"]
        e_app = e_app_ref * stiffness(run) / k_ref
        rot_txt = "0° (imposed)" if run == PLATEN else f"{rot:+.2f}°/1 %".replace("-", "\u2212")
        axF.barh(y, t, height=0.56, color=col, lw=0, zorder=3)
        axF.text(
            t + 0.008,
            y,
            f"{t:.3f};  {rot_txt}\n$E_{{app}}$ {e_app:.0f} MPa",
            fontsize=5.2,
            va="center",
            color=S.INK,
        )
        vals[f"control_{key}"] = {
            "tensile_volume_frac": t,
            "rotation_deg_per_pct_strain": rot,
            "E_app_MPa": e_app,
        }
    axF.set_yticks(range(len(controls)))
    axF.set_yticklabels([c[0] for c in controls][::-1], fontsize=5.4)
    axF.set_xlim(0, 0.52)
    axF.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    axF.set_xlabel("tension-dominated volume fraction")
    S.recessive_grid(axF)
    axF.set_title("Mechanistic controls", fontsize=S.BASE_PT, pad=2)
    label_mm("F", 103, 70)

    S.save(fig, OUT)
    VALUES.write_text(json.dumps(vals, indent=1, default=float))
    print(json.dumps(vals, indent=1, default=float))


if __name__ == "__main__":
    main()
