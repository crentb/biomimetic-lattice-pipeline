#!/usr/bin/env python
"""Off-screen 3-D renders of the digital twin and lattice geometries (v2).

Purpose
-------
Produces the geometry images used in Figs. 1, 3, 5 and 6 and the graphical
abstract, with one consistent camera/lighting/colour recipe:
  twin      rod-by-rod digital twin of the 2,433 canonical PIV trajectories,
            swept at the MEASURED rod radius (2.13 um = half the 4.256-um
            mean equivalent diameter: the depth-profile mean in
            morphometrics.json, the same value the CAD mapper scales to
            ROD_DIAMETER = 2.128 mm; revision 2026-09-25 replaced a pooled
            4.378-um value so that twin, lattice and text agree). v1 used 5 um (a
            10-um diameter), which overlaps neighbouring rods at the measured
            ~6.1-um pitch and turned the twin into a near-solid block.
  lattice   canonical bioinspired lattice (N = 2) from
            runs/live_001_digital_twin_v2/cad/compound_enamel_lattice.stl
  print     the N = 7 near-touching cell that was FDM-printed
            (runs/sweep_H24_thick/trial_005_N_BRIDGE_LAYERS_7), camera set
            close to the photograph's side-on viewpoint
  bio_N{4,9} / thick_N{4,9}   design-space corner cells for Fig. 5 insets

Why this exists
---------------
v1 renders were cropped (twin), clipped (titles), used an all-navy palette
that hid the plate/rod distinction, and were produced by three scripts with
different lighting. This script reuses the v1 tube-building approach but is
a new file (the v1 scripts are left untouched).

Inputs
------
microct_pipeline/output_piv/track_centerlines_piv.parquet
runs/<run>/cad/compound_enamel_lattice.stl  (see TARGETS)

Outputs
-------
matter_v2/figures/panels/render_<name>.png   (transparent background, 2x
                                             supersampled, ~2000 px wide)
matter_v2/data/render_manifest.json          source path, bounds (mm or um),
                                             camera, and rod radius per render

Side effects: read-only on runs/ and the parquet.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style_v2 as S  # noqa: E402

# --- 1. Paths ---------------------------------------------------------------
HERE = Path(__file__).resolve().parent
MANU = HERE.parent
BIO = MANU.parents[1]  # biomimetic_pipeline
REPO = BIO.parent  # microct_pipeline
PARQUET = (
    REPO / "output_piv_signcorrected" / "track_centerlines_piv.parquet"
)  # sign-corrected PIV tracking (track_rods_piv_signcorrected.py; see SI)
OUT = MANU / "figures" / "panels"
MANIFEST = MANU / "data" / "render_manifest.json"

ROD_RADIUS_UM = 4.256 / 2.0  # measured mean equivalent diameter / 2 (depth-profile mean; = mapper)
TWIN_SIGMA = 3.0  # light centreline smoothing (slices) -- keeps real tortuosity
ROD_COLOUR = S.NAVY_D  # rods
PLATE_COLOUR = "#B9C1CC"  # plates: neutral steel grey, distinct from rods

TARGETS = {
    "lattice": BIO / "runs/live_001_digital_twin_v2/cad/compound_enamel_lattice.stl",
    "print": BIO
    / "runs/sweep_H24_thick/trial_005_N_BRIDGE_LAYERS_7/cad/compound_enamel_lattice.stl",
    "bio_N4": BIO
    / "runs/sweep_H24_bio/trial_003_N_BRIDGE_LAYERS_4/cad/compound_enamel_lattice.stl",
    "bio_N9": BIO
    / "runs/sweep_H24_bio/trial_006_N_BRIDGE_LAYERS_9/cad/compound_enamel_lattice.stl",
    "thick_N4": BIO
    / "runs/sweep_H24_thick/trial_003_N_BRIDGE_LAYERS_4/cad/compound_enamel_lattice.stl",
    "thick_N9": BIO
    / "runs/sweep_H24_thick/trial_006_N_BRIDGE_LAYERS_9/cad/compound_enamel_lattice.stl",
    "control": BIO
    / "runs/sweep_decussation_v2/trial_000_decussation_0p05/cad/compound_enamel_lattice.stl",
}


def add_lights(pl: pv.Plotter) -> None:
    """Three-point lighting: key (upper left), fill (right), rim (behind)."""
    pl.remove_all_lights()
    # Camera-light coordinates: +x right, +y up, +z toward the viewer.
    pl.add_light(pv.Light(position=(-0.8, 1.3, 1.0), light_type="camera light", intensity=0.80))
    pl.add_light(pv.Light(position=(1.2, 0.4, 0.8), light_type="camera light", intensity=0.35))
    pl.add_light(pv.Light(position=(0.0, 1.0, -0.8), light_type="camera light", intensity=0.20))


def split_plates(mesh: pv.PolyData, plate_frac: float = 0.97):
    """Boolean mask of faces that belong to the two loading plates.

    Plate side faces reach the full plate footprint; rod/bridge faces stay
    inside the bundle. The plate slab limits are the vertex z-extent of those
    edge faces (bottom plate top = highest edge-face vertex in the lower half;
    top plate bottom = lowest in the upper half). A face is "plate" if all its
    vertices lie inside either slab. Using vertex extents (not centroids)
    keeps the plates' horizontal faces in the plate class.
    """
    faces = mesh.faces.reshape(-1, 4)[:, 1:]  # triangles
    z = mesh.points[:, 2][faces]  # (n_faces, 3)
    xy = np.abs(mesh.points[:, :2])[faces].max(axis=(1, 2))
    zmin, zmax = mesh.bounds[4], mesh.bounds[5]
    zmid = 0.5 * (zmin + zmax)
    half = max(abs(mesh.bounds[0]), abs(mesh.bounds[1]))
    edge = xy > plate_frac * half
    lower = edge & (z.max(axis=1) < zmid)
    upper = edge & (z.min(axis=1) > zmid)
    bot_top = z[lower].max() if lower.any() else zmin
    top_bot = z[upper].min() if upper.any() else zmax
    eps = 1e-4
    return (z.max(axis=1) <= bot_top + eps) | (z.min(axis=1) >= top_bot - eps)


def render_lattice(
    name: str, stl: Path, elev: float, azim: float, zoom: float, size=(2000, 1700)
) -> dict:
    mesh = pv.read(str(stl)).clean().triangulate()
    mesh = mesh.compute_normals(cell_normals=False, split_vertices=True, feature_angle=35)
    plate = split_plates(mesh)
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("white")
    add_lights(pl)
    pl.add_mesh(
        mesh.extract_cells(np.where(~plate)[0]),
        color=ROD_COLOUR,
        smooth_shading=True,
        ambient=0.22,
        diffuse=0.78,
        specular=0.25,
        specular_power=18,
    )
    pl.add_mesh(
        mesh.extract_cells(np.where(plate)[0]),
        color=PLATE_COLOUR,
        smooth_shading=True,
        ambient=0.30,
        diffuse=0.70,
        specular=0.15,
        specular_power=10,
    )
    pl.enable_anti_aliasing("ssaa")
    b = mesh.bounds
    ctr = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
    span = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
    e, a = np.radians(elev), np.radians(azim)
    eye = ctr + 3.2 * span * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    pl.camera.position, pl.camera.focal_point, pl.camera.up = eye, ctr, (0, 0, 1)
    pl.camera.view_angle = 20.0
    pl.camera.zoom(zoom)
    out = OUT / f"render_{name}.png"
    pl.screenshot(str(out), transparent_background=True)
    pl.close()
    return {
        "source": str(stl.relative_to(BIO)),
        "bounds_mm": [round(x, 3) for x in b],
        "elev_deg": elev,
        "azim_deg": azim,
        "zoom": zoom,
        "png": str(out.relative_to(MANU)),
    }


def render_twin(
    name: str = "twin",
    crop=None,
    elev: float = 32.0,
    azim: float = -58.0,
    zoom: float = 0.95,
    size=(2000, 1600),
) -> dict:
    """Rod-by-rod twin: canonical trajectories swept at the measured radius.

    crop=(x0, y0, side) restricts to rods whose slice-51 centre lies in that
    square (um), giving a readable close-up column. Tubes are coloured by the
    track's signed pitch with the Fig. 2 diverging map (+/-30 deg), so rods of
    opposite pitch -- the decussation -- are visible in 3-D.
    """
    df = pd.read_parquet(PARQUET)
    blocks = []
    for _, r in df.iterrows():
        x = gaussian_filter1d(np.fromstring(r.cx_um, sep=";"), TWIN_SIGMA)
        y = gaussian_filter1d(np.fromstring(r.cy_um, sep=";"), TWIN_SIGMA)
        z = np.fromstring(r.cz_um, sep=";")
        if crop is not None:
            x0, y0, side = crop
            if not (x0 <= x[50] < x0 + side and y0 <= y[50] < y0 + side):
                continue
        idx = np.linspace(0, len(x) - 1, 34).astype(int)  # ~1 um axial spacing
        line = pv.lines_from_points(np.column_stack([x[idx], y[idx], z[idx]]))
        tube = line.tube(radius=ROD_RADIUS_UM, n_sides=14, capping=True)
        tube.cell_data["pitch"] = np.full(tube.n_cells, float(r.pitch_deg))
        blocks.append(tube)
    twin = pv.merge(blocks)
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("white")
    add_lights(pl)
    pl.add_mesh(
        twin,
        scalars="pitch",
        cmap=S.DIVERGING,
        clim=(-30, 30),
        show_scalar_bar=False,
        smooth_shading=True,
        ambient=0.28,
        diffuse=0.72,
        specular=0.18,
        specular_power=15,
    )
    pl.enable_anti_aliasing("ssaa")
    b = twin.bounds
    ctr = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
    span = max(b[1] - b[0], b[3] - b[2])
    e, a = np.radians(elev), np.radians(azim)
    eye = ctr + 3.0 * span * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    pl.camera.position, pl.camera.focal_point, pl.camera.up = eye, ctr, (0, 0, 1)
    pl.camera.view_angle = 20.0
    pl.camera.zoom(zoom)
    out = OUT / f"render_{name}.png"
    pl.screenshot(str(out), transparent_background=True)
    pl.close()
    return {
        "source": str(PARQUET.relative_to(REPO)),
        "n_rods": int(len(blocks)),
        "rod_radius_um": ROD_RADIUS_UM,
        "centreline_sigma_slices": TWIN_SIGMA,
        "crop_um": crop,
        "colour": "signed pitch, diverging, +/-30 deg",
        "bounds_um": [round(x, 2) for x in b],
        "elev_deg": elev,
        "azim_deg": azim,
        "png": str(out.relative_to(MANU)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["twin", *TARGETS]
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for name in which:
        if name == "twin":
            man["twin"] = render_twin()
        elif name == "twin_crop":
            man["twin_crop"] = render_twin(
                "twin_crop", crop=(100.0, 100.0, 60.0), elev=28.0, zoom=1.0, size=(1600, 1600)
            )
        elif name == "print":
            # Close to the photograph: nearly side-on, slight elevation.
            man[name] = render_lattice(name, TARGETS[name], elev=9.0, azim=-90.0, zoom=0.92)
        else:
            man[name] = render_lattice(name, TARGETS[name], elev=20.0, azim=-60.0, zoom=0.84)
        print(f"rendered {name}", flush=True)
    MANIFEST.write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
