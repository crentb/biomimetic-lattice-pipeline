#!/usr/bin/env python
"""
cross_sections_thick_vs_original.py
===================================

Render side-by-side cross-section PNGs of the ORIGINAL compound enamel
lattice STL and the THICK-rod variant produced by
`make_thick_rod_variant.py`, to visually verify that the new rod diameter
keeps the rods "barely not touching."

What it produces
----------------
For a given trial, it writes four PNGs into
`<trial>/cad/cross_sections_thick_vs_original/`:

  01_horizontal_z<Z>.png
        Horizontal (XY-plane) slice at z = --z (default 10 mm), showing
        all rod cross-sections. Original on the left, thick on the right.

  02_horizontal_z<Z>_zoom_ring1pair.png
        Same horizontal slice, zoomed on the constraining rod pair (two
        neighbours on ring 1) where d_min sits. At full-model zoom the
        0.025 mm gap is invisible; this panel is the one you actually use
        to confirm "barely not touching".

  03_vertical_xz_y0.png
        Vertical XZ-plane slice at y = 0 (cuts ring-0 / ring-1 / ring-2
        rods and the plates). Original on the left, thick on the right.

  04_vertical_yz_x0.png
        Vertical YZ-plane slice at x = 0. Same layout.

How the slicing works
---------------------
For each plane we call `trimesh.Trimesh.section(plane_origin, plane_normal)`
on each STL, which returns a `Path3D` of the cut. We project that to 2D
with `.to_planar()`, then use the `polygons_full` attribute (shapely
Polygons) to draw filled rod cross-sections via `matplotlib.fill`. If the
section does not form closed loops (rare, for badly tessellated meshes), we
fall back to plotting the raw `Path2D.discrete` polyline segments so we at
least see the silhouette.

Why no comparison vs the analytic predictions
----------------------------------------------
We deliberately render straight from the STL geometry — not from the
analytic rod-centre math used in `make_thick_rod_variant.py`. That makes
the figures an INDEPENDENT visual check: if the analytic max-diameter
calculation was correct, you will see hairline gaps between every pair of
ring-1 neighbour rods in the zoomed view of the thick variant.

Usage
-----
    conda activate base
    python biomimetic_pipeline/scripts/cross_sections_thick_vs_original.py \
        --trial-dir biomimetic_pipeline/runs/sweep_layers_v2/trial_001_N_BRIDGE_LAYERS_6

Optional flags
--------------
    --original-stl   override path to the original STL
    --thick-stl      override path to the thick variant STL
    --out-dir        override output PNG directory
    --z              z-height of the horizontal slice (mm, default 10.0)
    --dpi            PNG resolution (default 200)
"""

from __future__ import annotations

# --- Standard library --------------------------------------------------------
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

# --- Third party -------------------------------------------------------------
import numpy as np

matplotlib.use("Agg")  # Headless render: no display server required.
import matplotlib.pyplot as plt
import trimesh

# ---------------------------------------------------------------------------
# Slice helper
# ---------------------------------------------------------------------------


def _section_polygons(
    mesh: trimesh.Trimesh,
    plane_origin: Tuple[float, float, float],
    plane_normal: Tuple[float, float, float],
):
    """Return (polygons, segments) for the given cutting plane.

    `polygons` is a list of shapely Polygons (filled cross-sections, ready
    for matplotlib.fill). `segments` is the raw list of (N, 2) polyline
    arrays from Path2D.discrete; we keep this as an outline-only fallback in
    case the section doesn't close into watertight polygons (badly
    tessellated mesh, intersection grazes the plate edge, etc.).

    Either or both can be empty if the plane misses the mesh entirely.
    """
    section3d = mesh.section(
        plane_origin=np.asarray(plane_origin, dtype=float),
        plane_normal=np.asarray(plane_normal, dtype=float),
    )
    if section3d is None:
        return [], []
    # to_planar() flattens the cut into local 2D coordinates; the returned
    # 4x4 transform is the plane-to-world basis if we ever need it (we
    # don't, because the cutting plane is always axis-aligned).
    section2d, _world_from_plane = section3d.to_planar()
    polys = []
    try:
        # polygons_full closes contours into shapely Polygons (with holes).
        polys = list(section2d.polygons_full)
    except Exception:
        polys = []
    segments = [np.asarray(d) for d in section2d.discrete] if section2d.discrete is not None else []
    return polys, segments


def _draw_section(
    ax, polys, segments, fill_color: str, edge_color: str = "black", edge_width: float = 0.25
):
    """Draw filled polygons (preferred) or outline polylines (fallback)."""
    if polys:
        for poly in polys:
            # Exterior ring: filled
            ext = np.asarray(poly.exterior.coords)
            ax.fill(
                ext[:, 0],
                ext[:, 1],
                facecolor=fill_color,
                edgecolor=edge_color,
                linewidth=edge_width,
                antialiased=True,
            )
            # Interior rings (holes): painted background colour so they
            # punch out of the exterior fill. The axis facecolor matches.
            for ring in poly.interiors:
                hole = np.asarray(ring.coords)
                ax.fill(
                    hole[:, 0],
                    hole[:, 1],
                    facecolor="white",
                    edgecolor=edge_color,
                    linewidth=edge_width,
                    antialiased=True,
                )
    elif segments:
        # No closed polygons — draw raw line segments so we can at least see
        # the cross-section outline.
        for seg in segments:
            if seg.shape[0] >= 2:
                ax.plot(seg[:, 0], seg[:, 1], color=edge_color, linewidth=edge_width)
    else:
        ax.text(
            0.5,
            0.5,
            "no intersection",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="red",
        )


# ---------------------------------------------------------------------------
# Geometry helpers (ring-1 rod centres for the zoom panel)
# ---------------------------------------------------------------------------


def _ring1_neighbor_midpoint(
    center_spacing: float, ring_rotation_deg: float, z: float, H: float, twist_type: str = "linear"
) -> Tuple[float, float, float]:
    """Compute the (x, y) midpoint between rods #1 and #2 on ring 1 at height z.

    Stock lattice_cad's `generate_radial_positions` puts ring-1 rods at
    angles theta_i = 2*pi*i/6, radius=CENTER_SPACING. With continuous twist
    each ring spins by `twist_func(z, H, ring_rotation_deg)`. We use this to
    pick a zoom centre that hits the ring-1 neighbour pair where d_min lives
    (regardless of z, since same-ring spacing is constant).

    Returns (mid_x, mid_y, pair_distance).
    """
    # Twist angle in radians at this z (linear case is sufficient: same-ring
    # spacing is independent of twist anyway, so the midpoint just rotates
    # around the origin without changing the pair distance).
    if twist_type == "linear":
        delta = (z / H) * math.radians(ring_rotation_deg)
    elif twist_type == "accelerating":
        delta = ((z / H) ** 2) * math.radians(ring_rotation_deg)
    else:
        delta = (z / H) * math.radians(ring_rotation_deg)
    theta1 = 0.0 + delta
    theta2 = (2.0 * math.pi / 6.0) + delta
    x1 = center_spacing * math.cos(theta1)
    y1 = center_spacing * math.sin(theta1)
    x2 = center_spacing * math.cos(theta2)
    y2 = center_spacing * math.sin(theta2)
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2), math.hypot(x2 - x1, y2 - y1))


# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------

# Distinct fill colours so it's obvious which panel is which, but both are
# muted so the eye lands on the rod-pair gap, not the colour.
ORIG_COLOR = "#9ec5e2"  # pale blue  — original
THICK_COLOR = "#f6b389"  # pale ochre — thick variant


def _make_side_by_side(
    fig_title: str,
    left_title: str,
    right_title: str,
    orig_polys,
    orig_segs,
    thick_polys,
    thick_segs,
    xlabel: str,
    ylabel: str,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    out_path: Path = None,
    dpi: int = 200,
):
    """Render a 2-panel figure (original | thick) and save to out_path."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    fig.suptitle(fig_title, fontsize=13)

    for ax, polys, segs, title, color in (
        (axes[0], orig_polys, orig_segs, left_title, ORIG_COLOR),
        (axes[1], thick_polys, thick_segs, right_title, THICK_COLOR),
    ):
        _draw_section(ax, polys, segs, fill_color=color)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="box")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25, linewidth=0.5)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render side-by-side cross-section PNGs of original vs thick lattice STLs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trial-dir",
        required=True,
        type=Path,
        help="Trial directory (must contain cad/compound_enamel_lattice.stl + thick variant).",
    )
    parser.add_argument(
        "--original-stl",
        type=Path,
        default=None,
        help="Override path to original STL (default: <trial>/cad/compound_enamel_lattice.stl).",
    )
    parser.add_argument(
        "--thick-stl",
        type=Path,
        default=None,
        help="Override path to thick STL (default: <trial>/cad/compound_enamel_lattice_thick.stl).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="Override output directory for PNGs."
    )
    parser.add_argument(
        "--z",
        type=float,
        default=10.0,
        help="z-height of the horizontal slice (mm). Default 10.0 (model centre).",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution. Default 200.")
    args = parser.parse_args()

    # ---- 1. Resolve paths --------------------------------------------------
    trial_dir = args.trial_dir.expanduser().resolve()
    cad_dir = trial_dir / "cad"
    orig_stl = (args.original_stl or (cad_dir / "compound_enamel_lattice.stl")).resolve()
    thick_stl = (args.thick_stl or (cad_dir / "compound_enamel_lattice_thick.stl")).resolve()
    out_dir = (args.out_dir or (cad_dir / "cross_sections_thick_vs_original")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for stl in (orig_stl, thick_stl):
        if not stl.is_file():
            sys.exit(f"ERROR: missing STL: {stl}")

    # ---- 2. Load both meshes -----------------------------------------------
    print(f"[xsec] Loading original STL: {orig_stl}")
    mesh_orig = trimesh.load_mesh(str(orig_stl), process=False)
    print(f"[xsec] Loading thick STL:    {thick_stl}")
    mesh_thick = trimesh.load_mesh(str(thick_stl), process=False)
    # If load returns a Scene (multi-body), concatenate into a single Trimesh.
    if isinstance(mesh_orig, trimesh.Scene):
        mesh_orig = trimesh.util.concatenate(mesh_orig.dump())
    if isinstance(mesh_thick, trimesh.Scene):
        mesh_thick = trimesh.util.concatenate(mesh_thick.dump())
    print(f"[xsec] Original bounds: {mesh_orig.bounds.tolist()}")
    print(f"[xsec] Thick    bounds: {mesh_thick.bounds.tolist()}")

    # ---- 3. Pull layout info from the sidecar for the zoom panel ----------
    # We need CENTER_SPACING + ring 1 rotation to find the binding rod pair.
    sidecar_path = cad_dir / "lattice_params.json"
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text())
        params = sidecar["params"]
        center_spacing = float(params["CENTER_SPACING"])
        enamel_thickness = float(params["ENAMEL_THICKNESS"])
        ring_rotation = {int(k): float(v) for k, v in params.get("RING_ROTATION", {}).items()}
        ring1_rotation_deg = ring_rotation.get(1, 0.0)
        twist_type = str(params.get("TWIST_TYPE", "linear"))
    else:
        # Sensible fallbacks if the sidecar is gone.
        print(f"WARNING: missing {sidecar_path}; using defaults for zoom centre.")
        center_spacing = 3.19
        enamel_thickness = 20.0
        ring1_rotation_deg = 0.0
        twist_type = "linear"

    # ---- 4. Horizontal slice (XY-plane) at z = args.z ---------------------
    print(f"[xsec] Horizontal slice at z = {args.z:.3f} mm ...")
    orig_polys_h, orig_segs_h = _section_polygons(mesh_orig, (0.0, 0.0, args.z), (0.0, 0.0, 1.0))
    thick_polys_h, thick_segs_h = _section_polygons(mesh_thick, (0.0, 0.0, args.z), (0.0, 0.0, 1.0))

    # Use the union of both X/Y extents so the axes are comparable.
    def _bbox_of_sections(polys, segs):
        xs, ys = [], []
        for p in polys:
            arr = np.asarray(p.exterior.coords)
            xs.append(arr[:, 0].min())
            xs.append(arr[:, 0].max())
            ys.append(arr[:, 1].min())
            ys.append(arr[:, 1].max())
        for s in segs:
            if s.shape[0] >= 1:
                xs.append(s[:, 0].min())
                xs.append(s[:, 0].max())
                ys.append(s[:, 1].min())
                ys.append(s[:, 1].max())
        if not xs:
            return None
        return (min(xs), max(xs), min(ys), max(ys))

    bb_o = _bbox_of_sections(orig_polys_h, orig_segs_h)
    bb_t = _bbox_of_sections(thick_polys_h, thick_segs_h)
    if bb_o and bb_t:
        xmin = min(bb_o[0], bb_t[0])
        xmax = max(bb_o[1], bb_t[1])
        ymin = min(bb_o[2], bb_t[2])
        ymax = max(bb_o[3], bb_t[3])
        pad = 0.5
        xlim = (xmin - pad, xmax + pad)
        ylim = (ymin - pad, ymax + pad)
    else:
        xlim = ylim = None

    z_tag = f"{args.z:.1f}".replace(".", "p")
    out_h = out_dir / f"01_horizontal_z{z_tag}.png"
    _make_side_by_side(
        fig_title=f"Horizontal cross-section (XY plane, z = {args.z:.2f} mm)",
        left_title="ORIGINAL  (ROD_DIAMETER = 2.128 mm)",
        right_title="THICK     (ROD_DIAMETER = 3.167 mm)",
        orig_polys=orig_polys_h,
        orig_segs=orig_segs_h,
        thick_polys=thick_polys_h,
        thick_segs=thick_segs_h,
        xlabel="x (mm)",
        ylabel="y (mm)",
        xlim=xlim,
        ylim=ylim,
        out_path=out_h,
        dpi=args.dpi,
    )

    # ---- 5. Same horizontal slice, zoomed on the ring-1 binding pair ------
    mid_x, mid_y, pair_d = _ring1_neighbor_midpoint(
        center_spacing=center_spacing,
        ring_rotation_deg=ring1_rotation_deg,
        z=args.z,
        H=enamel_thickness,
        twist_type=twist_type,
    )
    # Half-window pad: cover both rod radii plus a comfortable margin so the
    # 0.025 mm gap sits ~middle of the figure with white space around it.
    half = pair_d * 0.95
    zoom_xlim = (mid_x - half, mid_x + half)
    zoom_ylim = (mid_y - half, mid_y + half)
    out_h_zoom = out_dir / f"02_horizontal_z{z_tag}_zoom_ring1pair.png"
    _make_side_by_side(
        fig_title=(
            f"Zoom on the ring-1 neighbour pair (z = {args.z:.2f} mm)\n"
            f"centre (x,y) = ({mid_x:.3f}, {mid_y:.3f}) mm,  "
            f"pair centre-distance = {pair_d:.4f} mm"
        ),
        left_title="ORIGINAL — large gap (~1.06 mm)",
        right_title="THICK — gap ≈ 0.025 mm",
        orig_polys=orig_polys_h,
        orig_segs=orig_segs_h,
        thick_polys=thick_polys_h,
        thick_segs=thick_segs_h,
        xlabel="x (mm)",
        ylabel="y (mm)",
        xlim=zoom_xlim,
        ylim=zoom_ylim,
        out_path=out_h_zoom,
        dpi=args.dpi,
    )

    # ---- 6. Vertical XZ-plane slice at y = 0 ------------------------------
    print("[xsec] Vertical XZ slice at y = 0 ...")
    orig_polys_xz, orig_segs_xz = _section_polygons(mesh_orig, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    thick_polys_xz, thick_segs_xz = _section_polygons(mesh_thick, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    # In the XZ slice, Path2D.to_planar() lays out (u, v) where u≈x and v≈z
    # in *some* orientation chosen by trimesh. We don't try to rename axes
    # — both panels share the same convention, so the comparison is honest.
    out_xz = out_dir / "03_vertical_xz_y0.png"
    _make_side_by_side(
        fig_title="Vertical cross-section (XZ plane, y = 0)",
        left_title="ORIGINAL  (ROD_DIAMETER = 2.128 mm)",
        right_title="THICK     (ROD_DIAMETER = 3.167 mm)",
        orig_polys=orig_polys_xz,
        orig_segs=orig_segs_xz,
        thick_polys=thick_polys_xz,
        thick_segs=thick_segs_xz,
        xlabel="in-plane axis 1 (mm)",
        ylabel="in-plane axis 2 (mm)",
        out_path=out_xz,
        dpi=args.dpi,
    )

    # ---- 7. Vertical YZ-plane slice at x = 0 ------------------------------
    print("[xsec] Vertical YZ slice at x = 0 ...")
    orig_polys_yz, orig_segs_yz = _section_polygons(mesh_orig, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    thick_polys_yz, thick_segs_yz = _section_polygons(mesh_thick, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    out_yz = out_dir / "04_vertical_yz_x0.png"
    _make_side_by_side(
        fig_title="Vertical cross-section (YZ plane, x = 0)",
        left_title="ORIGINAL  (ROD_DIAMETER = 2.128 mm)",
        right_title="THICK     (ROD_DIAMETER = 3.167 mm)",
        orig_polys=orig_polys_yz,
        orig_segs=orig_segs_yz,
        thick_polys=thick_polys_yz,
        thick_segs=thick_segs_yz,
        xlabel="in-plane axis 1 (mm)",
        ylabel="in-plane axis 2 (mm)",
        out_path=out_yz,
        dpi=args.dpi,
    )

    print("\n[xsec] All cross-section PNGs written to:")
    print(f"  {out_dir}")


if __name__ == "__main__":
    main()
