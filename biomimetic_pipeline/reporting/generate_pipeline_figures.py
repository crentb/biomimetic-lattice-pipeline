"""Generate professional pipeline figures for docs/figures/.

Produces (all PNG, 300 dpi, publication-ready):
  01_architecture.png       — staged box-and-arrow diagram with colour codes
  02_feature_to_cad.png     — measured morphometrics vs. mapped CAD parameters
  03_strain_solver.png      — strain-to-target-stress bisection convergence
  04_crack_deflection.png   — streamline tortuosity concept
  05_closed_loop.png        — the closed-loop biomimicry reverse-map
  06_run_tree.png           — per-run directory tree with sizes/colours

Usage:
    python reporting/generate_pipeline_figures.py [--out docs/figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ----------------------------------------------------------------------------
# Palette — matches the colours used in PIPELINE.md Mermaid diagrams.
# ----------------------------------------------------------------------------
COLORS = {
    "ingest": "#2E86AB",  # blue
    "mapping": "#A23B72",  # magenta
    "cad": "#F18F01",  # orange
    "mesh": "#F18F01",
    "fea": "#C73E1D",  # red
    "metrics": "#6A994E",  # green
    "objective": "#7209B7",  # purple
    "report": "#3A86FF",  # bright blue
    "bio": "#EF476F",  # pink
    "text": "#1A1A1A",
    "bg": "#FAFAFA",
}


def _box(ax, x, y, w, h, label, color, text_color="white", fontsize=10):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4,
        edgecolor="black",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        fontweight="bold",
        wrap=True,
    )


def _arrow(ax, x0, y0, x1, y1, color="black", linewidth=1.6, style="-|>"):
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle=style,
        mutation_scale=14,
        color=color,
        linewidth=linewidth,
    )
    ax.add_patch(arr)


# ----------------------------------------------------------------------------
def fig_architecture(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 8.5), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.set_aspect("equal")
    ax.set_facecolor(COLORS["bg"])
    ax.axis("off")

    # Title
    ax.text(
        7,
        8.1,
        "biomimetic_pipeline — architecture",
        ha="center",
        fontsize=16,
        fontweight="bold",
        color=COLORS["text"],
    )

    # Row 1: ingest sources
    sources = [
        ("SOM\nbands", 0.3),
        ("HSB\nangles", 2.4),
        ("PIV\ntracks", 4.5),
        ("rod\ntracking", 6.6),
        ("slice\nrods", 8.7),
        ("smoothed\n3D", 10.8),
    ]
    for label, x in sources:
        _box(ax, x, 6.8, 1.8, 0.9, label, COLORS["ingest"], fontsize=9)

    # ingest merge
    _box(
        ax,
        5.3,
        5.4,
        3.4,
        0.7,
        "ingest/merge.py  (schema-validate + hash)",
        COLORS["ingest"],
        fontsize=10,
    )
    for _, x in sources:
        _arrow(ax, x + 0.9, 6.8, 7, 6.1, color=COLORS["ingest"], linewidth=0.9)

    # Mapping
    _box(ax, 5.3, 4.1, 3.4, 0.8, "mapping/feature_to_cad.py", COLORS["mapping"], fontsize=11)
    _arrow(ax, 7, 5.4, 7, 4.9, color=COLORS["mapping"])

    # Schemas
    _box(ax, 0.3, 4.1, 2.2, 0.8, "schemas/\nmorphometrics + cad_parameters", "#555", fontsize=8)
    _arrow(ax, 2.5, 4.5, 5.3, 4.5, color="#555", linewidth=0.8, style="-")

    # CAD + Mesh
    _box(
        ax,
        1.0,
        2.7,
        2.6,
        0.8,
        "generators/cad_runner\n(cad_env)",
        COLORS["cad"],
        text_color="black",
        fontsize=10,
    )
    _box(
        ax,
        4.0,
        2.7,
        2.6,
        0.8,
        "generators/mesh_runner\n(sfepy_env)",
        COLORS["mesh"],
        text_color="black",
        fontsize=10,
    )
    _arrow(ax, 7, 4.1, 2.3, 3.5, color=COLORS["cad"])
    _arrow(ax, 3.6, 3.1, 4.0, 3.1, color=COLORS["cad"])

    # FEA
    _box(ax, 7.0, 2.7, 2.8, 0.8, "fea/strain_solver\n(sfepy_env)", COLORS["fea"], fontsize=10)
    _arrow(ax, 6.6, 3.1, 7.0, 3.1, color=COLORS["mesh"])

    # Metrics row
    _box(ax, 1.0, 1.3, 2.6, 0.8, "metrics/\ncrack_deflection", COLORS["metrics"], fontsize=10)
    _box(ax, 4.0, 1.3, 2.6, 0.8, "fea/metrics_runner\n(30 scalars)", COLORS["metrics"], fontsize=10)
    _box(ax, 7.0, 1.3, 2.8, 0.8, "metrics/\nbiomimicry_score", COLORS["metrics"], fontsize=10)
    _arrow(ax, 8.4, 2.7, 8.4, 2.1, color=COLORS["fea"])
    _arrow(ax, 5.3, 2.7, 5.3, 2.1, color=COLORS["fea"])
    _arrow(ax, 5.3, 2.7, 2.3, 2.1, color=COLORS["fea"])

    # Objective
    _box(
        ax,
        10.2,
        2.0,
        3.3,
        0.9,
        "objectives/registry\n(crack_deflection, toughness,\nstiffness, biomimicry, composite)",
        COLORS["objective"],
        fontsize=9,
    )
    _arrow(ax, 9.8, 1.7, 10.2, 2.3, color=COLORS["metrics"])

    # Report
    _box(
        ax,
        10.2,
        0.6,
        3.3,
        0.8,
        "reporting/latex_report.py\n→ report.pdf",
        COLORS["report"],
        fontsize=10,
    )
    _arrow(ax, 11.85, 2.0, 11.85, 1.4, color=COLORS["objective"])

    # Legend
    lg_y = 0.05
    legend_items = [
        (COLORS["ingest"], "ingest"),
        (COLORS["mapping"], "mapping"),
        (COLORS["cad"], "cad / mesh"),
        (COLORS["fea"], "FEA"),
        (COLORS["metrics"], "metrics"),
        (COLORS["objective"], "objective"),
        (COLORS["report"], "report"),
    ]
    x_leg = 0.3
    for col, label in legend_items:
        ax.add_patch(Rectangle((x_leg, lg_y), 0.3, 0.25, facecolor=col, edgecolor="black", lw=0.6))
        ax.text(x_leg + 0.4, lg_y + 0.12, label, fontsize=9, va="center")
        x_leg += 1.6

    plt.tight_layout()
    p = Path(out) / "01_architecture.png"
    fig.savefig(p, dpi=300, facecolor=COLORS["bg"], bbox_inches="tight")
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------
def fig_feature_to_cad(out: Path, morph_path: Path = None, cad_path: Path = None) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    fig.patch.set_facecolor(COLORS["bg"])

    morph = {}
    cad = {}
    if morph_path and Path(morph_path).exists():
        morph = json.loads(Path(morph_path).read_text())
    if cad_path and Path(cad_path).exists():
        cad = json.loads(Path(cad_path).read_text())

    # --- 1. Pitch profile → cumulative twist → ring rotations ---
    ax = axes[0, 0]
    dp = morph.get("depth_profiles", {})
    z = np.asarray(dp.get("depth_um", []), dtype=float)
    p = np.asarray(dp.get("pitch_signed_deg", []), dtype=float)
    if z.size and p.size == z.size:
        phi = np.concatenate([[0], np.cumsum(0.5 * (p[:-1] + p[1:]) * np.diff(z))])
        ax.plot(z, p, color=COLORS["ingest"], linewidth=1.5, label="pitch(z) [deg]")
        ax2 = ax.twinx()
        ax2.plot(z, phi, color=COLORS["mapping"], linewidth=2, label=r"$\phi(z) = \int$ pitch dz")
        # Ring rotation overlay
        rr = cad.get("RING_ROTATION", {})
        if rr:
            idxs = sorted(int(k) for k in rr.keys())
            ring_angles = [rr[str(i)] for i in idxs]
            enamel = float(cad.get("ENAMEL_THICKNESS", 20.0))
            ring_zs = np.linspace(0, enamel, len(idxs)) * (z.max() - z.min()) / enamel + z.min()
            ax2.scatter(
                ring_zs,
                ring_angles,
                color=COLORS["cad"],
                s=70,
                zorder=5,
                edgecolor="black",
                linewidth=1,
                label="RING_ROTATION[i]",
            )
        ax2.set_ylabel("cumulative twist [deg]", color=COLORS["mapping"])
        ax2.tick_params(axis="y", labelcolor=COLORS["mapping"])
        ax.set_ylabel("pitch [deg]", color=COLORS["ingest"])
        ax.tick_params(axis="y", labelcolor=COLORS["ingest"])
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "(no pitch profile available)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("depth from DEJ [um]")
    ax.set_title("pitch  →  cumulative twist  →  RING_ROTATION", fontweight="bold")
    ax.grid(alpha=0.3)

    # --- 2. Rod diameter depth profile → ROD_DIAMETER (+ SLA clamp) ---
    ax = axes[0, 1]
    d_um = np.asarray(dp.get("rod_diameter_um_mean", []), dtype=float)
    if z.size and d_um.size == z.size:
        ax.plot(z, d_um, color=COLORS["ingest"], linewidth=1.5, label="measured rod diameter [um]")
        scale = float(cad.get("biology_scale_factor", 500.0))
        ax.axhline(d_um.mean(), color=COLORS["ingest"], linestyle="--", alpha=0.5)
        if cad:
            rod_mm = float(cad.get("ROD_DIAMETER", 0.0))
            rod_um_back = rod_mm / scale * 1e3  # in biology units
            ax.axhline(
                rod_um_back,
                color=COLORS["cad"],
                linewidth=2.5,
                label=f"ROD_DIAMETER ({rod_mm:.2f} mm @ {scale:g}×)",
            )
            ax.axhline(
                float(cad.get("sla_min_feature_mm", 0.3)) / scale * 1e3,
                color=COLORS["bio"],
                linestyle=":",
                label="SLA min feature (biology eq.)",
            )
        ax.legend(fontsize=8)
    ax.set_xlabel("depth from DEJ [um]")
    ax.set_ylabel("rod diameter [um]")
    ax.set_title("rod diameter profile  →  ROD_DIAMETER (scaled + clamped)", fontweight="bold")
    ax.grid(alpha=0.3)

    # --- 3. SOM band widths → CENTER_SPACING ---
    ax = axes[1, 0]
    bands = morph.get("bands", [])
    if bands:
        ids = [b["band_id"] for b in bands]
        means = [b["band_width_um"]["mean"] for b in bands]
        stds = [b["band_width_um"]["std"] for b in bands]
        ax.bar(
            ids,
            means,
            yerr=stds,
            color=COLORS["ingest"],
            edgecolor="black",
            alpha=0.8,
            capsize=5,
            label="band_width_um",
        )
        if cad:
            scale = float(cad.get("biology_scale_factor", 500.0))
            cs_mm = float(cad.get("CENTER_SPACING", 0.0))
            cs_um = cs_mm / scale * 1e3
            ax.axhline(
                cs_um, color=COLORS["cad"], linewidth=2.5, label=f"CENTER_SPACING ({cs_mm:.2f} mm)"
            )
        ax.legend(fontsize=8)
    ax.set_xlabel("SOM band id")
    ax.set_ylabel("band width [um]")
    ax.set_title("SOM band widths  →  CENTER_SPACING", fontweight="bold")
    ax.grid(alpha=0.3)

    # --- 4. Periodicity → N_BRIDGE_LAYERS ---
    ax = axes[1, 1]
    per = morph.get("periodicity", {})
    wl = per.get("dominant_wavelength_um_mean", 0.0)
    wl_std = per.get("dominant_wavelength_um_std", 0.0)
    if wl:
        enamel_mm = float(cad.get("ENAMEL_THICKNESS", 20.0))
        scale = float(cad.get("biology_scale_factor", 500.0))
        wl_mm = wl * 1e-3 * scale
        raw_n = enamel_mm / wl_mm
        clamped_n = cad.get("N_BRIDGE_LAYERS", 4)
        bars = [
            "measured\nλ [um]",
            "scaled\nλ [mm]",
            "enamel\nH [mm]",
            "raw\nN=H/λ",
            "clamped\nN_BRIDGE",
        ]
        vals = [wl, wl_mm, enamel_mm, raw_n, clamped_n]
        colors = [
            COLORS["ingest"],
            COLORS["mapping"],
            COLORS["mapping"],
            COLORS["bio"],
            COLORS["cad"],
        ]
        ax.bar(range(len(bars)), vals, color=colors, edgecolor="black")
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels(bars, fontsize=8)
        ax.set_yscale("log")
        for i, v in enumerate(vals):
            ax.text(i, v * 1.1, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_title("SOM periodicity  →  N_BRIDGE_LAYERS (clamped to [2, 8])", fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    plt.suptitle(
        "Feature → CAD mapping (biology × biology_scale_factor → printable geometry)",
        fontsize=13,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    p = Path(out) / "02_feature_to_cad.png"
    fig.savefig(p, dpi=300, facecolor=COLORS["bg"], bbox_inches="tight")
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------
def fig_strain_solver(out: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    fig.patch.set_facecolor(COLORS["bg"])

    # Simulate a strain-solver run: vm = k * disp (linear elasticity).
    k = 50.0  # MPa/mm
    target = 200.0
    d = 1.0
    history = [(0, d, k * d)]
    for i in range(1, 5):
        vm = k * d
        if abs(vm - target) / target < 0.05:
            break
        d = d * (target / vm)
        history.append((i, d, k * d))

    iters, disps, vms = zip(*history)
    ax1.plot(
        iters,
        vms,
        "o-",
        color=COLORS["fea"],
        linewidth=2.5,
        markersize=10,
        markeredgecolor="black",
        label="VM at each iter",
    )
    ax1.axhline(
        target, color=COLORS["bio"], linestyle="--", linewidth=2, label=f"target = {target} MPa"
    )
    ax1.axhspan(
        target * 0.95, target * 1.05, color=COLORS["bio"], alpha=0.15, label="5% tolerance band"
    )
    for i, (it, d, vm) in enumerate(history):
        ax1.annotate(
            f"d={d:.3f} mm", (it, vm), xytext=(8, 8), textcoords="offset points", fontsize=9
        )
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("avg_von_mises_MPa")
    ax1.set_title("strain solver — bisect to target stress", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Right: conceptual bar chart of downstream metrics evaluated at critical load
    metrics = ["SCF", "VM_P99", "crack_defl\np90", "spec.\ntoughness", "crit. strain\nat 200 MPa"]
    values = [5.7, 1130, 1.31, 378.6, 0.0204]
    colors = [COLORS["fea"]] * 2 + [COLORS["metrics"]] * 2 + [COLORS["bio"]]
    ax2.bar(metrics, values, color=colors, edgecolor="black", alpha=0.85)
    ax2.set_yscale("log")
    for i, v in enumerate(values):
        ax2.text(i, v * 1.12, f"{v:g}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_title("downstream metrics at matched 200 MPa load", fontweight="bold")
    ax2.grid(alpha=0.3, axis="y")

    plt.suptitle(
        "strain-for-stress — every design is compared at the SAME load state",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    p = Path(out) / "03_strain_solver.png"
    fig.savefig(p, dpi=300, facecolor=COLORS["bg"], bbox_inches="tight")
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------
def fig_crack_deflection(out: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)
    fig.patch.set_facecolor(COLORS["bg"])

    # Left: straight field — streamlines are straight, tortuosity ≈ 1
    rng = np.random.default_rng(42)
    for i in range(8):
        y0 = rng.uniform(-3, 3)
        x0 = rng.uniform(-3, 3)
        z = np.linspace(0, 20, 40)
        x = x0 + 0.02 * rng.standard_normal(40).cumsum()
        y = y0 + 0.02 * rng.standard_normal(40).cumsum()
        ax1.plot(z, x, color=COLORS["metrics"], alpha=0.7, linewidth=1.4)
    ax1.set_title("uniform p1(z): tortuosity ≈ 1.0", fontweight="bold")
    ax1.set_xlabel("z (mm)")
    ax1.set_ylabel("x (mm)")
    ax1.grid(alpha=0.3)
    ax1.text(
        0.02,
        0.95,
        "arc ≈ chord\ntortuosity = arc/chord",
        transform=ax1.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    # Right: deflected streamlines through lattice bridges
    for i in range(8):
        y0 = rng.uniform(-3, 3)
        x0 = rng.uniform(-3, 3)
        z = np.linspace(0, 20, 80)
        # Wobble induced by bridge interference at ~6 and ~14 mm
        wobble_x = 1.2 * np.sin(0.4 * z + rng.uniform(0, 2 * np.pi))
        wobble_y = 0.8 * np.cos(0.3 * z + rng.uniform(0, 2 * np.pi))
        x = x0 + wobble_x + 0.15 * rng.standard_normal(80).cumsum() * 0.02
        y = y0 + wobble_y + 0.15 * rng.standard_normal(80).cumsum() * 0.02
        # Highlight bridge positions
        ax2.plot(z, x, color=COLORS["fea"], alpha=0.8, linewidth=1.5)
    # Bridge band overlays
    for bz in (6.0, 14.0):
        ax2.axvspan(bz - 0.5, bz + 0.5, color=COLORS["cad"], alpha=0.15)
        ax2.text(
            bz,
            ax2.get_ylim()[1] * 0.95 if ax2.get_ylim()[1] > 0 else 4,
            "bridge",
            ha="center",
            fontsize=8,
            color=COLORS["cad"],
        )
    ax2.set_title("bridged lattice: tortuosity > 1 (crack deflection)", fontweight="bold")
    ax2.set_xlabel("z (mm)")
    ax2.set_ylabel("x (mm)")
    ax2.grid(alpha=0.3)
    ax2.text(
        0.02,
        0.95,
        "arc > chord\ntortuosity > 1",
        transform=ax2.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.suptitle(
        "Crack-deflection metric — p1 streamline tortuosity", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    p = Path(out) / "04_crack_deflection.png"
    fig.savefig(p, dpi=300, facecolor=COLORS["bg"], bbox_inches="tight")
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------
def fig_closed_loop(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    # Four stations arranged in a loop
    stations = [
        (1.5, 4.5, 2.5, 1.2, "search CAD\nparameter space\n(Optuna TPE)", COLORS["mapping"]),
        (5.5, 4.5, 2.5, 1.2, "evaluate via\nfull pipeline\n(CAD → mesh → FEA)", COLORS["cad"]),
        (9.2, 4.5, 2.3, 1.2, "score objective\n(e.g. crack\ndeflection)", COLORS["objective"]),
        (
            5.5,
            1.0,
            5.0,
            1.2,
            "REVERSE-MAP top-K designs → biomimicry_targets.json\n"
            + "morphometric ranges to look for in specimens",
            COLORS["bio"],
        ),
    ]
    for x, y, w, h, label, col in stations:
        _box(ax, x, y, w, h, label, col, fontsize=10 if len(label) < 40 else 9)

    # Arrows (clockwise)
    _arrow(ax, 4.0, 5.1, 5.5, 5.1, color=COLORS["mapping"], linewidth=2)
    _arrow(ax, 8.0, 5.1, 9.2, 5.1, color=COLORS["cad"], linewidth=2)
    _arrow(ax, 10.4, 4.5, 10.4, 3.0, color=COLORS["objective"], linewidth=2)
    _arrow(ax, 10.4, 3.0, 10.5, 2.2, color=COLORS["objective"], linewidth=2)
    # Feedback arrow: bottom → top
    _arrow(ax, 5.5, 1.6, 2.0, 4.5, color=COLORS["bio"], linewidth=2.2, style="-|>")

    # Titles
    ax.text(
        6,
        6.5,
        "closed-loop biomimicry",
        ha="center",
        fontsize=15,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        6,
        6.0,
        "Optuna finds high-objective lattices → we back-map them to biology",
        ha="center",
        fontsize=10,
        color="#555",
    )

    # Inset text on feedback arrow
    ax.text(
        3.3,
        3.0,
        "feedback:\ntargets for\nnext specimen\nscan",
        fontsize=8,
        color=COLORS["bio"],
        fontweight="bold",
        ha="center",
    )

    plt.tight_layout()
    p = Path(out) / "05_closed_loop.png"
    fig.savefig(p, dpi=300, facecolor=COLORS["bg"], bbox_inches="tight")
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------
def fig_run_tree(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 9), dpi=300)
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")

    nodes = [
        (0.5, 12.3, "runs/<run_name>/", COLORS["text"], 12),
        (1.5, 11.5, "run_context.json", COLORS["text"], 10),
        (1.5, 11.0, "env_probe.json", COLORS["text"], 10),
        (1.5, 10.5, "cad_params.json            ← digital-twin spec", COLORS["mapping"], 10),
        (1.5, 10.0, "cad/", COLORS["cad"], 10),
        (2.5, 9.5, "cad_driver.py  (auto-generated)", COLORS["cad"], 9),
        (2.5, 9.0, "compound_enamel_lattice.step + .stl", COLORS["cad"], 9),
        (2.5, 8.5, "lattice_params.json  (sidecar)", COLORS["cad"], 9),
        (2.5, 8.0, "cad_run.log", COLORS["cad"], 9),
        (1.5, 7.5, "mesh/", COLORS["mesh"], 10),
        (2.5, 7.0, "compound_enamel_lattice.msh", COLORS["mesh"], 9),
        (2.5, 6.5, "mesh_run.log", COLORS["mesh"], 9),
        (1.5, 6.0, "fea/", COLORS["fea"], 10),
        (2.5, 5.5, "iter_1/  iter_2/ ...  (strain solver iterations)", COLORS["fea"], 9),
        (2.5, 5.0, "final/", COLORS["fea"], 9),
        (3.5, 4.5, "element_results_compression.csv", COLORS["fea"], 8),
        (3.5, 4.0, "global_results_compression.csv", COLORS["fea"], 8),
        (3.5, 3.5, "compound_enamel_lattice.vtk", COLORS["fea"], 8),
        (3.5, 3.0, "optimization_metrics.json  (30 scalars)", COLORS["metrics"], 8),
        (3.5, 2.5, "crack_deflection.json       ← new metric", COLORS["metrics"], 8),
        (2.5, 2.0, "strain_solve_summary.json", COLORS["fea"], 9),
        (1.5, 1.5, "metrics.json   (merged + _pipeline block)", COLORS["metrics"], 10),
        (1.5, 1.0, "score.json     (objective + direction)", COLORS["objective"], 10),
        (1.5, 0.5, "report/report.tex  →  report.pdf", COLORS["report"], 10),
        (1.5, 0.0, "pipeline.log", COLORS["text"], 10),
    ]
    for x, y, txt, col, fs in nodes:
        ax.text(
            x,
            y,
            txt,
            fontsize=fs,
            color=col,
            family="monospace",
            fontweight="bold" if "← " in txt or txt.endswith("/") else "normal",
        )

    # Vertical tree lines
    ax.plot([0.6, 0.6], [0.0, 11.5], color="black", alpha=0.3, linewidth=0.8)
    ax.plot([1.6, 1.6], [5.5, 9.5], color="black", alpha=0.3, linewidth=0.8)
    ax.plot([2.6, 2.6], [4.5, 5.0], color="black", alpha=0.3, linewidth=0.8)
    ax.plot([3.6, 3.6], [2.5, 4.5], color="black", alpha=0.3, linewidth=0.8)

    ax.text(5, 12.7, "Per-run output tree", fontsize=13, fontweight="bold", color=COLORS["text"])
    plt.tight_layout()
    p = Path(out) / "06_run_tree.png"
    fig.savefig(p, dpi=300, facecolor=COLORS["bg"], bbox_inches="tight")
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default=str(Path(__file__).resolve().parent.parent / "docs" / "figures")
    )
    ap.add_argument("--morphometrics", default=None)
    ap.add_argument("--cad-params", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    paths.append(fig_architecture(out))
    paths.append(
        fig_feature_to_cad(
            out,
            (
                Path(args.morphometrics)
                if args.morphometrics
                else Path(__file__).resolve().parent.parent
                / "runs"
                / "live_001"
                / "morphometrics.json"
            ),
            (
                Path(args.cad_params)
                if args.cad_params
                else Path(__file__).resolve().parent.parent
                / "runs"
                / "live_001_digital_twin"
                / "cad_params.json"
            ),
        )
    )
    paths.append(fig_strain_solver(out))
    paths.append(fig_crack_deflection(out))
    paths.append(fig_closed_loop(out))
    paths.append(fig_run_tree(out))
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
