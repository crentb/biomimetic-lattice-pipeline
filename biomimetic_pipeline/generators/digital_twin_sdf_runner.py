"""Digital-twin generator (implicit-SDF route): rods modeled as capsule
distance functions, unioned via element-wise minimum, surface extracted by
marching cubes.

Why this exists: the gaussian/B-spline tube methods produce per-rod meshes
that are individually watertight but have **non-manifold geometry where rods
touch each other** (overlapping triangle soup). For downstream meshing/FEA
this is a problem. The SDF route gives a single, fully-watertight unioned
solid: where two rods touch they auto-blend at the surface; the topology is
clean.

Pipeline:
    1. Read PIV parquet, gaussian-smooth each rod centerline.
    2. Convert each rod to a sequence of capsule (cylinder + hemisphere caps)
       segments along its smoothed centerline.
    3. Build a voxel grid covering the bounding box at VOXEL_SIZE_UM resolution.
    4. For each voxel, find the K nearest capsule midpoints via cKDTree, compute
       the capsule SDF for those K segments, and take the minimum (= union).
    5. Run marching cubes on the resulting field at iso=0.
    6. Optionally Taubin-smooth the resulting mesh, then export STL.

Output: <export_dir>/cad/digital_twin_sdf.stl + provenance.json + log.

Caveats:
    - Voxel resolution drives both quality and runtime. At 1 um voxels with
      bounds ~300x280x40 um the grid is ~3.4M voxels; this takes ~1-2 minutes.
    - The output is a SINGLE solid (the rod cloud as an implicit union); rods
      cannot be selected individually downstream. For per-rod analysis, use
      the gaussian or B-spline route instead.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

# Use scikit-image for marching cubes (more robust than vtk for this case).
from skimage.measure import marching_cubes

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import pyvista as pv

# Canonical PIV parquet for the lion digital twin. NOTE: the previous default
# pointed at output_piv_back (2,645 tracks) but as of the 2026-05-22 track-count
# provenance resolution
# the manuscript-canonical track population is the 2,433-track set in
# output_piv -- that's the parquet feeding morphometrics.json's
# rod_tracks_summary and every downstream CAD/FEA/sweep run. The default below
# was updated 2026-05-23 to match.
DEFAULT_PIV_PARQUET = (
    Path(os.environ.get("MICROCT_PIPELINE_ROOT", "microct_pipeline"))
    / "output_piv"
    / "track_centerlines_piv.parquet"
)
MIN_TRACK_LENGTH_DEFAULT = 30
ROD_RADIUS_UM_DEFAULT = 5.0
GAUSSIAN_SIGMA_DEFAULT = 12.0  # smoother centerlines feed cleaner capsules
N_AXIAL_POINTS_DEFAULT = 40  # capsule segments per rod
VOXEL_SIZE_UM_DEFAULT = 1.0  # SDF grid resolution
K_NEAREST_DEFAULT = 8  # nearest segments queried per voxel
PADDING_UM_DEFAULT = 8.0  # bbox padding (rod_radius + margin)
TAUBIN_ITERS_DEFAULT = 20
TAUBIN_PASS_BAND_DEFAULT = 0.05

# Plating defaults (added 2026-05-23 for the digital_twin FEA integration).
# Plates serve two purposes for FEA: (i) flat top/bottom z-faces become the
# clean BC regions the strain solver's z-coordinate selectors grab onto, and
# (ii) all rods are bonded into one connected load path through the plate
# slabs, eliminating the ~575 surface-component fragmentation that
# pyvista's split_bodies() reports on the raw rod-only SDF mesh.
PLATE_THICKNESS_UM_DEFAULT = 5.0
PLATE_OVERLAP_UM_DEFAULT = 2.0


@dataclass
class SdfDigitalTwinResult:
    export_dir: Path
    stl_path: Path
    provenance_path: Path
    log_path: Path
    n_rods: int
    n_segments: int
    n_voxels: int
    n_faces: int
    bounds_um: tuple
    voxel_size_um: float
    model_type: str = "digital_twin_sdf"
    # Plated FEA inputs (populated only when run(..., plate=True)).
    plated_occupancy_path: Optional[Path] = None
    plated_specimen_height_um: Optional[float] = None
    plated_solid_fraction: Optional[float] = None


def _parse(s: str) -> np.ndarray:
    return np.fromstring(s, sep=";")


def _build_capsule_segments(
    df: pd.DataFrame,
    *,
    gaussian_sigma: float,
    n_axial_points: int,
):
    """For each rod, return a list of capsule endpoints (a, b)."""
    a_list = []
    b_list = []
    n_rods = 0
    for _, row in df.iterrows():
        x = _parse(row["cx_um"])
        y = _parse(row["cy_um"])
        z = _parse(row["cz_um"])
        n = len(x)
        if n < 4:
            continue
        x = gaussian_filter1d(x, sigma=gaussian_sigma)
        y = gaussian_filter1d(y, sigma=gaussian_sigma)
        if n > n_axial_points:
            idx = np.linspace(0, n - 1, n_axial_points).astype(int)
            x, y, z = x[idx], y[idx], z[idx]
        pts = np.column_stack([x, y, z])
        for i in range(len(pts) - 1):
            a_list.append(pts[i])
            b_list.append(pts[i + 1])
        n_rods += 1
    a_arr = np.array(a_list, dtype=np.float64)
    b_arr = np.array(b_list, dtype=np.float64)
    return a_arr, b_arr, n_rods


def _capsule_sdf_batch(P: np.ndarray, a: np.ndarray, b: np.ndarray, r: float) -> np.ndarray:
    """SDF of capsules (segments a->b, radius r) at points P.

    P: (N, 3); a, b: (N, 3) — broadcast 1:1 (one capsule per point)
    Returns: (N,) signed distances.
    """
    pa = P - a
    ba = b - a
    ba2 = np.einsum("ij,ij->i", ba, ba)
    h = np.clip(np.einsum("ij,ij->i", pa, ba) / np.maximum(ba2, 1e-12), 0.0, 1.0)
    closest = a + h[:, None] * ba
    return np.linalg.norm(P - closest, axis=1) - r


def _build_sdf_grid(
    a_arr: np.ndarray,
    b_arr: np.ndarray,
    *,
    rod_radius_um: float,
    voxel_size_um: float,
    padding_um: float,
    k_nearest: int,
    log,
):
    bbox_min = np.minimum(a_arr.min(axis=0), b_arr.min(axis=0)) - padding_um
    bbox_max = np.maximum(a_arr.max(axis=0), b_arr.max(axis=0)) + padding_um
    log(f"[sdf-twin] bbox: {bbox_min.round(1)} -> {bbox_max.round(1)} um")

    xs = np.arange(bbox_min[0], bbox_max[0] + voxel_size_um, voxel_size_um)
    ys = np.arange(bbox_min[1], bbox_max[1] + voxel_size_um, voxel_size_um)
    zs = np.arange(bbox_min[2], bbox_max[2] + voxel_size_um, voxel_size_um)
    nx, ny, nz = len(xs), len(ys), len(zs)
    n_vox = nx * ny * nz
    log(f"[sdf-twin] grid: {nx} x {ny} x {nz} = {n_vox:,} voxels at {voxel_size_um} um")

    midpoints = 0.5 * (a_arr + b_arr)
    log(f"[sdf-twin] kdtree on {len(midpoints):,} capsule midpoints...")
    tree = cKDTree(midpoints)

    # Process voxels in chunks to control memory
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    voxels = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    sdf = np.full(n_vox, np.inf, dtype=np.float32)
    chunk = 250_000
    log(f"[sdf-twin] computing SDF (k_nearest={k_nearest}) in chunks of {chunk:,}...")
    for c0 in range(0, n_vox, chunk):
        c1 = min(c0 + chunk, n_vox)
        P = voxels[c0:c1]
        _, idxs = tree.query(P, k=k_nearest)
        sdf_chunk = np.full(c1 - c0, np.inf, dtype=np.float32)
        for k in range(k_nearest):
            seg_idx = idxs[:, k]
            d = _capsule_sdf_batch(P, a_arr[seg_idx], b_arr[seg_idx], rod_radius_um)
            np.minimum(sdf_chunk, d.astype(np.float32), out=sdf_chunk)
        sdf[c0:c1] = sdf_chunk
    log("[sdf-twin] SDF complete")
    return sdf.reshape(nx, ny, nz), (xs, ys, zs)


def _plate_occupancy(
    rod_occ,
    voxel_size_um: float,
    plate_thickness_um: float,
    plate_overlap_um: float,
    log,
):
    """Add solid top + bottom plate slabs to a rod-only occupancy grid.

    Returns ``(plated_occ, plate_specimen_height_um, plate_metadata)``.
    The plates extend the *physical* z-extent of the specimen outward from
    the rod bbox by ``plate_thickness_um - plate_overlap_um`` on each end,
    while the plate inner faces sit ``plate_overlap_um`` *inside* the rod
    z-extent so rods penetrate the plates and bond into a single load path
    through the plate material (no tangent-contact ambiguity).

    The grid is enlarged in z as needed to hold the plate slabs; the
    rod-only region in the middle is unchanged. This matches the plating
    pattern used for the 2026-05-22 gate twin.
    """
    import numpy as np

    plate_t_vox = max(int(round(plate_thickness_um / voxel_size_um)), 1)
    plate_o_vox = max(int(round(plate_overlap_um / voxel_size_um)), 0)
    plate_o_vox = min(plate_o_vox, plate_t_vox - 1)  # never fully inside rod

    # Rod z-extent (in voxel indices) within the existing grid.
    z_any = rod_occ.any(axis=(0, 1))
    if not z_any.any():
        raise RuntimeError("rod occupancy grid is empty -- nothing to plate")
    rod_z_min = int(np.argmax(z_any))
    rod_z_max = int(len(z_any) - 1 - np.argmax(z_any[::-1]))

    # Pad the grid to accommodate plate slabs (rod region unchanged).
    extra_below = max(plate_t_vox - plate_o_vox - rod_z_min, 0)
    extra_above = max(plate_t_vox - plate_o_vox - (rod_occ.shape[2] - 1 - rod_z_max), 0)

    plated = np.pad(rod_occ, ((0, 0), (0, 0), (extra_below, extra_above)))
    new_rod_z_min = rod_z_min + extra_below
    new_rod_z_max = rod_z_max + extra_below

    # Bottom plate: spans from (new_rod_z_min + plate_o_vox - plate_t_vox)
    # up to (new_rod_z_min + plate_o_vox). Fully filled across x,y footprint.
    b_top = new_rod_z_min + plate_o_vox + 1  # +1 for slice-exclusive upper
    b_bot = max(b_top - plate_t_vox, 0)
    plated[:, :, b_bot:b_top] = True

    # Top plate: mirror.
    t_bot = new_rod_z_max - plate_o_vox
    t_top = min(t_bot + plate_t_vox, plated.shape[2])
    plated[:, :, t_bot:t_top] = True

    specimen_height_um = plated.shape[2] * voxel_size_um
    solid_fraction = float(plated.sum()) / float(plated.size)

    metadata = {
        "plate_thickness_um": plate_thickness_um,
        "plate_overlap_um": plate_overlap_um,
        "plate_thickness_voxels": plate_t_vox,
        "plate_overlap_voxels": plate_o_vox,
        "grid_extra_below_voxels": int(extra_below),
        "grid_extra_above_voxels": int(extra_above),
        "rod_z_extent_voxels": [int(rod_z_min), int(rod_z_max)],
        "plated_shape_voxels": list(plated.shape),
        "plated_specimen_height_um": specimen_height_um,
        "plated_solid_fraction": solid_fraction,
    }
    log(
        f"[sdf-twin] plated: grid {plated.shape}  "
        f"height {specimen_height_um:.1f} um  solid_frac {solid_fraction:.4f}"
    )
    return plated, specimen_height_um, metadata


def run(
    export_dir: Path,
    *,
    piv_parquet: Optional[Path] = None,
    min_track_length: int = MIN_TRACK_LENGTH_DEFAULT,
    rod_radius_um: float = ROD_RADIUS_UM_DEFAULT,
    gaussian_sigma: float = GAUSSIAN_SIGMA_DEFAULT,
    n_axial_points: int = N_AXIAL_POINTS_DEFAULT,
    voxel_size_um: float = VOXEL_SIZE_UM_DEFAULT,
    k_nearest: int = K_NEAREST_DEFAULT,
    padding_um: float = PADDING_UM_DEFAULT,
    taubin_iters: int = TAUBIN_ITERS_DEFAULT,
    taubin_pass_band: float = TAUBIN_PASS_BAND_DEFAULT,
    morphometrics_path: Optional[Path] = None,
    plate: bool = False,
    plate_thickness_um: float = PLATE_THICKNESS_UM_DEFAULT,
    plate_overlap_um: float = PLATE_OVERLAP_UM_DEFAULT,
) -> SdfDigitalTwinResult:
    export_dir = Path(export_dir)
    cad_dir = export_dir / "cad"
    cad_dir.mkdir(parents=True, exist_ok=True)
    log_path = cad_dir / "digital_twin_sdf_run.log"
    stl_path = cad_dir / "digital_twin_sdf.stl"
    provenance_path = cad_dir / "digital_twin_sdf_provenance.json"

    parquet = Path(piv_parquet) if piv_parquet else DEFAULT_PIV_PARQUET
    if not parquet.exists():
        raise FileNotFoundError(f"PIV parquet not found: {parquet}")

    log = []

    def _log(s: str):
        log.append(s)
        print(s)

    t0 = time.time()
    _log(f"[sdf-twin] reading {parquet}")
    df = pd.read_parquet(parquet)
    df = df[df["n_slices"] >= min_track_length].reset_index(drop=True)
    _log(f"[sdf-twin] {len(df)} tracks pass n_slices >= {min_track_length}")

    a_arr, b_arr, n_rods = _build_capsule_segments(
        df,
        gaussian_sigma=gaussian_sigma,
        n_axial_points=n_axial_points,
    )
    _log(f"[sdf-twin] {n_rods} rods -> {len(a_arr):,} capsule segments")

    sdf_grid, (xs, ys, zs) = _build_sdf_grid(
        a_arr,
        b_arr,
        rod_radius_um=rod_radius_um,
        voxel_size_um=voxel_size_um,
        padding_um=padding_um,
        k_nearest=k_nearest,
        log=_log,
    )

    # Plating: produce a watertight, plated occupancy grid for the FEA path.
    # The STL marching-cubes output below is still produced from the original
    # (unplated) SDF for visualization; plating only enters the FEA mesh.
    plated_occ_path: Optional[Path] = None
    plated_specimen_height_um: Optional[float] = None
    plated_solid_fraction: Optional[float] = None
    plate_metadata: dict = {}
    if plate:
        from scipy.ndimage import binary_fill_holes

        rod_occ = sdf_grid <= 0.0
        # Fill enclosed inter-rod voids (sealed pockets that would otherwise
        # be counted as separate surface components by split_bodies). On the
        # 2026-05-22 gate twin this nudged solid fraction 0.6779 -> 0.6788.
        rod_occ_filled = binary_fill_holes(rod_occ)
        plated_occ, plated_specimen_height_um, plate_metadata = _plate_occupancy(
            rod_occ_filled,
            voxel_size_um=voxel_size_um,
            plate_thickness_um=plate_thickness_um,
            plate_overlap_um=plate_overlap_um,
            log=_log,
        )
        plated_solid_fraction = plate_metadata["plated_solid_fraction"]
        plated_occ_path = cad_dir / "twin_plated_occ.npy"
        np.save(str(plated_occ_path), plated_occ)
        _log(f"[sdf-twin] wrote plated occupancy -> {plated_occ_path}")

    _log("[sdf-twin] running marching cubes at iso=0...")
    verts, faces, _normals, _values = marching_cubes(
        sdf_grid,
        level=0.0,
        spacing=(voxel_size_um, voxel_size_um, voxel_size_um),
    )
    # Marching cubes returns verts in grid coords starting at (0,0,0). Offset to bbox.
    verts = verts + np.array([xs[0], ys[0], zs[0]])
    _log(f"[sdf-twin] marching cubes: {len(verts):,} verts, {len(faces):,} faces")

    mesh_pv = pv.PolyData(
        verts, np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces]).ravel()
    )
    if taubin_iters > 0:
        _log(f"[sdf-twin] Taubin smoothing ({taubin_iters} iters)...")
        mesh_pv = mesh_pv.smooth_taubin(
            n_iter=taubin_iters,
            pass_band=taubin_pass_band,
            boundary_smoothing=True,
            feature_smoothing=False,
            non_manifold_smoothing=True,
        )

    _log(f"[sdf-twin] writing STL -> {stl_path}")
    # Save via trimesh for cleaner manifold STL header
    tm = trimesh.Trimesh(
        vertices=mesh_pv.points, faces=mesh_pv.faces.reshape(-1, 4)[:, 1:], process=False
    )
    tm.export(str(stl_path))

    bounds = mesh_pv.bounds
    elapsed = time.time() - t0

    provenance = {
        "model_type": "digital_twin_sdf",
        "source_parquet": str(parquet),
        "morphometrics_path": str(morphometrics_path) if morphometrics_path else None,
        "inclusion_rule": {"min_track_length": min_track_length},
        "n_rods": int(n_rods),
        "n_capsule_segments": int(len(a_arr)),
        "n_voxels": int(sdf_grid.size),
        "n_faces": int(mesh_pv.n_cells),
        "n_points": int(mesh_pv.n_points),
        "bounds_um": {
            "x": [float(bounds[0]), float(bounds[1])],
            "y": [float(bounds[2]), float(bounds[3])],
            "z": [float(bounds[4]), float(bounds[5])],
        },
        "smoothing": {
            "method": "implicit SDF (capsule-min) + marching cubes",
            "centerline_gaussian_sigma_slices": gaussian_sigma,
            "n_axial_points": n_axial_points,
            "voxel_size_um": voxel_size_um,
            "k_nearest_segments_per_voxel": k_nearest,
            "marching_cubes_iso_level": 0.0,
            "padding_um": padding_um,
            "taubin_iterations": taubin_iters,
            "taubin_pass_band": taubin_pass_band,
        },
        "rod_radius_um": rod_radius_um,
        "stl_size_bytes": int(stl_path.stat().st_size),
        "elapsed_seconds": round(elapsed, 2),
        "comparison_note": (
            "SDF route: produces a SINGLE watertight unioned solid (vs per-rod "
            "tube meshes from the gaussian/B-spline routes). Rods cannot be "
            "selected individually downstream; for per-rod analysis use the "
            "tube-based runners. Best for: clean meshing input to FEA, true "
            "single-solid 3D printing."
        ),
        "plated": bool(plate),
        "plating": plate_metadata if plate else None,
        "plated_occupancy_path": str(plated_occ_path) if plated_occ_path else None,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2))
    log_path.write_text("\n".join(log))

    return SdfDigitalTwinResult(
        export_dir=export_dir,
        stl_path=stl_path,
        provenance_path=provenance_path,
        log_path=log_path,
        n_rods=int(n_rods),
        n_segments=int(len(a_arr)),
        n_voxels=int(sdf_grid.size),
        n_faces=int(mesh_pv.n_cells),
        bounds_um=(
            float(bounds[1] - bounds[0]),
            float(bounds[3] - bounds[2]),
            float(bounds[5] - bounds[4]),
        ),
        voxel_size_um=voxel_size_um,
        plated_occupancy_path=plated_occ_path,
        plated_specimen_height_um=plated_specimen_height_um,
        plated_solid_fraction=plated_solid_fraction,
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Generate SDF-based digital-twin STL.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--piv-parquet", type=Path, default=None)
    p.add_argument("--rod-radius-um", type=float, default=ROD_RADIUS_UM_DEFAULT)
    p.add_argument("--gaussian-sigma", type=float, default=GAUSSIAN_SIGMA_DEFAULT)
    p.add_argument("--n-axial-points", type=int, default=N_AXIAL_POINTS_DEFAULT)
    p.add_argument("--voxel-size-um", type=float, default=VOXEL_SIZE_UM_DEFAULT)
    p.add_argument("--k-nearest", type=int, default=K_NEAREST_DEFAULT)
    p.add_argument("--padding-um", type=float, default=PADDING_UM_DEFAULT)
    p.add_argument("--taubin-iters", type=int, default=TAUBIN_ITERS_DEFAULT)
    p.add_argument("--taubin-pass-band", type=float, default=TAUBIN_PASS_BAND_DEFAULT)
    p.add_argument("--min-track-length", type=int, default=MIN_TRACK_LENGTH_DEFAULT)
    p.add_argument(
        "--plate",
        action="store_true",
        help="Add solid plate slabs and save plated_occ.npy for FEA.",
    )
    p.add_argument("--plate-thickness-um", type=float, default=PLATE_THICKNESS_UM_DEFAULT)
    p.add_argument("--plate-overlap-um", type=float, default=PLATE_OVERLAP_UM_DEFAULT)
    args = p.parse_args()
    res = run(
        args.out_dir,
        piv_parquet=args.piv_parquet,
        min_track_length=args.min_track_length,
        rod_radius_um=args.rod_radius_um,
        gaussian_sigma=args.gaussian_sigma,
        n_axial_points=args.n_axial_points,
        voxel_size_um=args.voxel_size_um,
        k_nearest=args.k_nearest,
        padding_um=args.padding_um,
        taubin_iters=args.taubin_iters,
        taubin_pass_band=args.taubin_pass_band,
        plate=args.plate,
        plate_thickness_um=args.plate_thickness_um,
        plate_overlap_um=args.plate_overlap_um,
    )
    print(
        f"\n[sdf-twin] OK: {res.n_rods} rods, {res.n_voxels:,} voxels, "
        f"{res.n_faces:,} faces, {res.stl_path}"
    )


if __name__ == "__main__":
    _cli()
