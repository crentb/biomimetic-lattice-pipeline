#!/usr/bin/env python
"""Principal-stress load-path analysis of completed lattice FEA runs.

Purpose
-------
Traces *load paths* -- integral curves of the principal-stress direction that
carries the load -- through the solid of each finite-element result and
reports their tortuosity (arc length / chord length), their inclination to
the loading axis, and how the load is partitioned between tension and
compression. These "principal stress trajectories" are the classical way to
visualise how an architecture routes load (Kelly & Elsley 1995; Kelly & Tosh
2000), and their inclination is the geometric precondition for crack
deflection when the lattice is realised as a two-phase composite.

Which principal direction is followed (``--mode``):
  dominant (default)  the eigenvector of the principal stress with the
                      largest MAGNITUDE in each element. Required for the
                      decussated lattice: under global compression the
                      near-vertical inner rods carry axial compression
                      (sigma_3 ~ -500..-750 MPa at the 200 MPa matched
                      state) while the strongly inclined outermost ring
                      (helix ~38 deg) carries axial TENSION (sigma_1 ~
                      +130 MPa, sigma_3 only ~ -13 MPa and ~84 deg off the
                      rod axis). sigma_3 trajectories therefore stall in the
                      outer rods; dominant-direction trajectories follow the
                      rod axis in every ring (verified 2026-09-25 on
                      runs/live_001_digital_twin_v2).
  compression         sigma_3 (most compressive) eigenvector everywhere.
  tension             sigma_1 (most tensile) eigenvector everywhere.

Why this exists
---------------
It replaces two defective legacy measures used in the v1 manuscript:
  1. ``metrics/crack_deflection.py`` integrates the GRADIENT of a scalar
     stress field rasterised onto a 32^3 bounding-box grid (void cells filled
     by neighbour averaging, upward component clamped) -- not a principal
     direction, and not confined to material.
  2. The Fig 4C script fed the three principal-stress VALUES to a stream
     tracer as if they were an (x, y, z) velocity vector.
Both also inherit two naming defects in the stock FEA post-processor
(cad_modeling/.../compression_test.py, read-only upstream code):
  (a) eigenvalues are written ASCENDING, so the CSV column ``p1_MPa`` is the
      minimum (most compressive) principal stress, not sigma_1;
  (b) sfepy stores symmetric 3-D tensors as [s11, s22, s33, s12, s13, s23]
      (sfepy.mechanics.tensors.get_full_indices(3)), but the stock script
      unpacks indices 4/5 as (syz, sxz), swapping the xz and yz shears in the
      matrix it diagonalises (principal-value error: median ~2 MPa, 99th
      percentile ~50 MPa, max ~170 MPa on the canonical run).
This script reads the raw ``cauchy_stress`` cell array from the VTK and
builds every tensor in sfepy's true order, so neither defect propagates.

Method
------
1. Read ``fea/final/compound_enamel_lattice.vtk``; keep tetrahedra only
   (the file also stores boundary triangles/lines).
2. Eigen-decompose every element tensor (numpy eigh; ascending order) and
   pick the load direction e_L per --mode; |sigma_L| is used as an
   interpolation weight so low-stress, direction-noisy elements contribute
   little.
3. Detect the plate slabs from the mesh (where the solid footprint jumps to
   the full plate width) and define the gauge [z_lo, z_hi] between them.
4. Seed points on a regular xy grid (0.3 mm) at the mid-height of the widest
   bridge-free interval of the gauge (bridge bands from cad_params.json),
   keep those inside solid (exact point-in-tet test, vtkStaticCellLocator),
   randomly subsample to --n-seeds (fixed RNG seed -> reproducible).
5. Integrate each seed both upward and downward with classical RK4 (step
   --step-mm). The direction at any point is the |sigma_L|- and inverse-
   distance-weighted mean of e_L over the k nearest element centroids, each
   eigenvector sign-aligned to the current heading (eigenvectors have no
   intrinsic sign); where those neighbours disagree (coherence <
   COHERENCE_MIN, a degenerate point) the heading is kept. A step that
   leaves the solid is first re-issued tangent to the surface (the
   continuum trajectory is tangent to traction-free surfaces), then, if
   still outside, pulled halfway toward the nearest element centroid up to 3
   times before the half-path is marked "exited". A path is COMPLETE when
   both halves reach their end planes 1 mm inside the plate interfaces.
6. Tortuosity tau = arc / |end - start| for complete paths; inclination =
   angle between the local path tangent and +z, arc-length averaged.
7. Load partition: volume fraction of gauge material whose dominant
   principal stress is tensile, overall and per hexagonal ring (radial bands
   of width CENTER_SPACING around ring k * CENTER_SPACING).

Inputs
------
--runs DIR [DIR ...]    run directories (each containing fea/final/*.vtk);
                        glob patterns allowed.
--mode {dominant,compression,tension}  principal direction followed
                        (default: dominant; see above).
--n-seeds INT           max seeds per run (default 1200)
--step-mm FLOAT         RK4 step length in mm (default 0.10; mesh target 0.5)
--k-neighbours INT      elements averaged per direction query (default 10)
--save-paths INT        number of trajectories stored for plotting (default 300)
--out-dir DIR           default runs/_load_path_analysis

Outputs
-------
<out-dir>/<run_key>.json        per-run statistics (tortuosity, inclination,
                                completion fraction, corrected principal-
                                stress percentiles, gauge bounds, settings)
<out-dir>/<run_key>_paths.npz   up to --save-paths trajectories (float32) for
                                figure rendering
<out-dir>/load_path_summary.csv one row per run (appended/replaced by key)

Side effects / non-obvious behaviour
------------------------------------
* Read-only on every run directory.
* run_key = run path relative to runs/ with '/' -> '__'.
* Deterministic: the same inputs and flags give identical outputs.

References
----------
Kelly, D.W., Elsley, M. (1995) A procedure for determining load paths in
elastic continua. Eng. Comput. 12, 415-424. doi:10.1108/02644409510799721
Kelly, D.W., Tosh, M.W. (2000) Interpreting load paths and stress trajectories
in elasticity. Eng. Comput. 17, 117-135. doi:10.1108/02644400010313084
(DOIs verified against Crossref 2026-09-25.)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import cKDTree

BIOMIMETIC_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = (
    BIOMIMETIC_ROOT / "biomimetic_pipeline" / "runs"
)  # public layout: run directories live inside the package (RunContext default)
# Below this eigenvector coherence (0..1) a path keeps its heading: the local
# principal direction is degenerate (e.g., at rod-bridge junctions).
# FEA result folder; "final" normally. A linear iteration folder (e.g. iter_1)
# gives identical normalised quantities (set with --fea-subdir).
FEA_SUBDIR = "final"
COHERENCE_MIN = 0.30
# A half-path that backtracks more than this (mm) against its direction of
# vertical travel is terminated as "reversed" (see trace()).
REVERSAL_TOL_MM = 1.0


# --- 1. Tensor handling (sfepy storage order) -------------------------------
def tensors_from_sfepy(voigt6: np.ndarray) -> np.ndarray:
    """(N,6) sfepy symmetric storage [s11,s22,s33,s12,s13,s23] -> (N,3,3).

    Order verified against sfepy.mechanics.tensors.get_full_indices(3) =
    [[0,3,4],[3,1,5],[4,5,2]]: index 3 = (x,y), 4 = (x,z), 5 = (y,z).
    """
    s = np.asarray(voigt6, dtype=float)
    T = np.empty((s.shape[0], 3, 3))
    T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
    T[:, 0, 1] = T[:, 1, 0] = s[:, 3]  # sigma_xy
    T[:, 0, 2] = T[:, 2, 0] = s[:, 4]  # sigma_xz
    T[:, 1, 2] = T[:, 2, 1] = s[:, 5]  # sigma_yz
    return T


def load_run(run_dir: Path, mode: str):
    """Read tets + stress; return (tet grid, centroids, e_load, |sigma_load|, evals)."""
    vtk_path = run_dir / "fea" / FEA_SUBDIR / "compound_enamel_lattice.vtk"
    grid = pv.read(vtk_path)
    tet_ids = np.where(np.asarray(grid.celltypes) == pv.CellType.TETRA)[0]
    tets = grid.extract_cells(tet_ids)
    stress = np.asarray(grid.cell_data["cauchy_stress"])[tet_ids]
    evals, evecs = np.linalg.eigh(tensors_from_sfepy(stress))  # ascending
    # Column 0 = sigma_3 (most compressive), column 2 = sigma_1 (most tensile).
    if mode == "compression":
        col = np.zeros(len(evals), dtype=int)
    elif mode == "tension":
        col = np.full(len(evals), 2)
    else:  # dominant: whichever of sigma_1 / sigma_3 has the larger magnitude
        col = np.where(np.abs(evals[:, 0]) >= np.abs(evals[:, 2]), 0, 2)
    rows = np.arange(len(evals))
    e_load = evecs[rows, :, col]
    s_load = np.abs(evals[rows, col])
    centroids = tets.cell_centers().points
    return tets, centroids, e_load, s_load, evals, grid


def top_plate_kinematics(grid) -> dict:
    """Rigid-body fit of the top-face displacement (twist-compression coupling).

    The FEA clamps the bottom face (u = 0) and prescribes only the axial
    displacement of the top face, leaving its in-plane translation and
    rotation free. A chiral (net-twisted) lattice therefore rotates its top
    plate about z under axial load. The least-squares rigid rotation of the
    top-face nodes is  theta = sum(x*u_y - y*u_x) / sum(x^2 + y^2)  (small-
    rotation linearisation, consistent with the linear-elastic solution);
    it is reported per 1 % of nominal axial strain so runs solved at
    different critical strains are comparable (linear scaling).
    """
    P = grid.points
    U = np.asarray(grid.point_data["u"])
    top = P[:, 2] > P[:, 2].max() - 1e-6
    x, y = P[top, 0], P[top, 1]
    ux, uy, uz = U[top].T
    r2 = x * x + y * y
    m = r2 > 1.0  # skip nodes near the axis
    theta_rad = float(np.sum((x * uy - y * ux)[m]) / np.sum(r2[m]))
    height = float(P[:, 2].max() - P[:, 2].min())  # plate-to-plate, mm
    strain = float(abs(uz.mean()) / height)
    return {
        "top_uz_mm": float(uz.mean()),
        "top_rotation_deg": float(np.degrees(theta_rad)),
        "nominal_axial_strain": strain,
        "rotation_deg_per_pct_strain": (
            float(np.degrees(theta_rad) / (100.0 * strain)) if strain > 0 else float("nan")
        ),
        "top_inplane_disp_mean_mm": float(np.hypot(ux, uy).mean()),
    }


# --- 2. Gauge detection ------------------------------------------------------
def gauge_bounds(centroids: np.ndarray, dz: float = 0.1):
    """Return (z_lo, z_hi): the rod gauge between the two plate slabs.

    Plates span the full plate footprint; the rod bundle is narrower. For
    each z-bin the 99th percentile of the in-plane radius r = max(|x|,|y|)
    of element centroids is computed; plate bins sit near the plate half-
    width, rod bins well inside it. The gauge is the longest run of bins
    whose r99 is below 90 % of the plate half-width.
    """
    z = centroids[:, 2]
    r = np.maximum(np.abs(centroids[:, 0]), np.abs(centroids[:, 1]))
    edges = np.arange(z.min(), z.max() + dz, dz)
    idx = np.digitize(z, edges) - 1
    r99 = np.array(
        [np.percentile(r[idx == i], 99) if np.any(idx == i) else 0.0 for i in range(len(edges) - 1)]
    )
    half_width = r.max()
    is_rod = r99 < 0.90 * half_width
    best, cur, start = (0, 0), 0, 0
    for i, flag in enumerate(is_rod):
        if flag:
            if cur == 0:
                start = i
            cur += 1
            if cur > best[1] - best[0]:
                best = (start, i + 1)
        else:
            cur = 0
    return float(edges[best[0]]), float(edges[best[1]])


# --- 3. Direction field and point location ----------------------------------
class LoadField:
    """Sign-consistent, smoothed load-direction queries inside the solid."""

    def __init__(self, tets, centroids, e_load, s_load, k: int):
        self.tree = cKDTree(centroids)
        self.centroids = centroids
        self.e = e_load
        self.w_s = s_load / (np.median(s_load) + 1e-12)  # dimensionless stress weight
        self.k = k
        # Exact point-in-tet test; static locator is built once per run.
        self.locator = vtk.vtkStaticCellLocator()
        self.locator.SetDataSet(tets)
        self.locator.BuildLocator()
        self._gc = vtk.vtkGenericCell()
        self._pc = [0.0, 0.0, 0.0]
        self._w = [0.0] * 4

    def inside(self, pts: np.ndarray) -> np.ndarray:
        out = np.empty(len(pts), dtype=bool)
        for i, p in enumerate(pts):
            out[i] = self.locator.FindCell(p, 1e-6, self._gc, self._pc, self._w) >= 0
        return out

    def direction(self, pts: np.ndarray, heading: np.ndarray) -> np.ndarray:
        """Weighted mean load direction at pts, each vector aligned to heading.

        Coherence c = |sum w_i s_i e_i| / sum w_i (0..1) measures how well the
        neighbouring eigenvectors agree. Near rod-bridge junctions two
        principal stresses can coincide (a degenerate point) and the
        eigenvector is ill-defined; where c < COHERENCE_MIN the path keeps
        its current heading instead of following numerical noise.
        """
        d, j = self.tree.query(pts, k=self.k)
        e = self.e[j]  # (n,k,3)
        sgn = np.sign(np.einsum("nkc,nc->nk", e, heading))
        sgn[sgn == 0] = 1.0
        w = self.w_s[j] / (d + 1e-3)  # stress x inverse distance
        v = np.einsum("nk,nkc->nc", w * sgn, e)
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        coherence = norm[:, 0] / (w.sum(axis=1) + 1e-12)
        v = v / (norm + 1e-12)
        weak = coherence < COHERENCE_MIN
        v[weak] = heading[weak]
        return v

    def inward_normal(self, pts: np.ndarray) -> np.ndarray:
        """Unit vector from each point toward its nearest element centroid.

        Used as a local estimate of the inward surface normal when a step
        overshoots a traction-free surface (see trace()).
        """
        _, j = self.tree.query(pts)
        n = self.centroids[j] - pts
        return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


# --- 4. Integration ----------------------------------------------------------
def trace(
    field: LoadField,
    seeds: np.ndarray,
    z_stop: float,
    h: float,
    max_steps: int,
    upward: bool = True,
):
    """Batched RK4 integration; returns list of (n_i,3) paths and status codes.

    upward=True integrates toward +z until z >= z_stop; upward=False toward
    -z until z <= z_stop (the two halves of a bidirectional trace).
    status: 1 = complete (reached z_stop), 0 = exited solid, -1 = step limit,
    -2 = reversed (backtracked > REVERSAL_TOL_MM against its direction of
    travel). A plate-to-plate load path makes monotone vertical progress
    apart from lateral excursions through bridges; a half-path that turns
    around at a rod-bridge junction and returns through a plate is a
    trajectory of a different load path, not a continuation of this one.
    """
    n = len(seeds)
    pos = seeds.copy()
    zsgn = 1.0 if upward else -1.0
    head = np.tile([0.0, 0.0, zsgn], (n, 1))  # initial heading +/-z
    active = np.ones(n, dtype=bool)
    status = np.full(n, -1)
    best = seeds[:, 2] * zsgn  # furthest progress so far
    paths = [[p.copy()] for p in seeds]
    for _ in range(max_steps):
        idx = np.where(active)[0]
        if idx.size == 0:
            break
        p, hd = pos[idx], head[idx]
        k1 = field.direction(p, hd)
        k2 = field.direction(p + 0.5 * h * k1, k1)
        k3 = field.direction(p + 0.5 * h * k2, k2)
        k4 = field.direction(p + h * k3, k3)
        v = k1 + 2 * k2 + 2 * k3 + k4
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        new = p + h * v
        ok = field.inside(new)
        # (a) Slide: at a traction-free surface the true trajectory is
        # tangent to the surface, so an overshooting step is re-issued with
        # its outward-normal component removed (normal estimated from the
        # nearest element centroid).
        bad = np.where(~ok)[0]
        if bad.size:
            n_in = field.inward_normal(p[bad])
            vb = v[bad]
            outward = np.minimum(np.einsum("nc,nc->n", vb, n_in), 0.0)
            vb = vb - outward[:, None] * n_in
            vb /= np.linalg.norm(vb, axis=1, keepdims=True) + 1e-12
            v[bad] = vb
            new[bad] = p[bad] + h * vb
            ok[bad] = field.inside(new[bad])
        # (b) Recover: pull any remaining overshoot halfway toward the
        # nearest element centroid (inside material by construction).
        for _attempt in range(3):
            bad = np.where(~ok)[0]
            if bad.size == 0:
                break
            _, j = field.tree.query(new[bad])
            new[bad] = 0.5 * (new[bad] + field.centroids[j])
            ok[bad] = field.inside(new[bad])
        for a, i in enumerate(idx):
            if not ok[a]:
                active[i] = False
                status[i] = 0
                continue
            pos[i] = new[a]
            head[i] = v[a]
            paths[i].append(new[a].copy())
            prog = new[a, 2] * zsgn
            best[i] = max(best[i], prog)
            if (upward and new[a, 2] >= z_stop) or (not upward and new[a, 2] <= z_stop):
                active[i] = False
                status[i] = 1
            elif best[i] - prog > REVERSAL_TOL_MM:
                active[i] = False
                status[i] = -2
    return [np.asarray(pp) for pp in paths], status


def seed_height(run_dir: Path, z_lo: float, z_hi: float) -> float:
    """Mid-height of the widest bridge-free interval of the gauge (mm).

    Seeding inside a bridge band starts paths at rod-bridge junctions, where
    the principal directions are degenerate. The bridge centre elevations
    come from the run's cad_params.json (BRIDGE_Z_OFFSETS); each band is
    widened by the bridge radius. Falls back to mid-gauge if unavailable.
    """
    try:
        cad = json.loads((run_dir / "cad_params.json").read_text())
        zb = sorted(float(z) for z in cad.get("BRIDGE_Z_OFFSETS") or [])
        rb = 0.5 * float(cad.get("BRIDGE_DIAMETER", 0.0))
    except (OSError, ValueError, TypeError):
        zb, rb = [], 0.0
    edges = [z_lo] + [e for z in zb for e in (z - rb, z + rb)] + [z_hi]
    gaps = [(edges[i], edges[i + 1]) for i in range(0, len(edges) - 1, 2)]
    gaps = [g for g in gaps if g[1] > g[0]]
    if not gaps:
        return 0.5 * (z_lo + z_hi)
    a, b = max(gaps, key=lambda g: g[1] - g[0])
    return 0.5 * (a + b)


def path_stats(path: np.ndarray):
    seg = np.diff(path, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    arc = float(seg_len.sum())
    chord = float(np.linalg.norm(path[-1] - path[0]))
    # Inclination of each segment to +z (deg), arc-length weighted mean.
    cosz = np.clip(np.abs(seg[:, 2]) / (seg_len + 1e-12), 0, 1)
    incl = float(np.degrees(np.arccos(cosz)) @ seg_len / (arc + 1e-12))
    return arc, chord, incl


# --- 5. Per-run driver -------------------------------------------------------
def analyse(run_dir: Path, args) -> dict:
    t0 = time.time()
    tets, cent, e_load, s_load, evals, grid = load_run(run_dir, args.mode)
    kin = top_plate_kinematics(grid)
    z_lo, z_hi = gauge_bounds(cent)
    # Paths start mid-way through the widest bridge-free interval and are
    # traced both up and down; a path is complete only if both halves reach
    # their end planes (1 mm inside each plate interface).
    z_seed = seed_height(run_dir, z_lo, z_hi)
    z_stop, z_stop_lo = z_hi - 1.0, z_lo + 1.0
    field = LoadField(tets, cent, e_load, s_load, args.k_neighbours)

    # Seeds: regular xy grid at z_seed, kept only where the point is in solid.
    xmin, xmax = cent[:, 0].min(), cent[:, 0].max()
    ymin, ymax = cent[:, 1].min(), cent[:, 1].max()
    gx, gy = np.meshgrid(np.arange(xmin, xmax, 0.3), np.arange(ymin, ymax, 0.3))
    cand = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z_seed)])
    cand = cand[field.inside(cand)]
    rng = np.random.default_rng(args.seed)
    if len(cand) > args.n_seeds:
        cand = cand[rng.choice(len(cand), args.n_seeds, replace=False)]

    max_steps = int(4.0 * (z_stop - z_stop_lo) / args.step_mm)
    up, st_up = trace(field, cand, z_stop, args.step_mm, max_steps, upward=True)
    dn, st_dn = trace(field, cand, z_stop_lo, args.step_mm, max_steps, upward=False)
    # Join halves bottom -> top: reversed downward half + upward half (the
    # shared seed point is kept once).
    paths = [np.vstack([d[::-1], u[1:]]) for d, u in zip(dn, up)]
    # Combined status: complete only if both halves completed; otherwise the
    # worse of the two (exit 0 beats step-limit -1 for diagnostics).
    # Combined status: complete only if both halves completed; otherwise
    # exited (0) > reversed (-2) > step limit (-1), in that priority.
    both = (st_up == 1) & (st_dn == 1)
    status = np.where(
        both,
        1,
        np.where((st_up == 0) | (st_dn == 0), 0, np.where((st_up == -2) | (st_dn == -2), -2, -1)),
    )

    comp = [p for p, s in zip(paths, status) if s == 1]
    stats = np.array([path_stats(p) for p in comp]) if comp else np.zeros((0, 3))
    tau = stats[:, 0] / np.maximum(stats[:, 1], 1e-9) if len(stats) else np.array([np.nan])
    incl = stats[:, 2] if len(stats) else np.array([np.nan])

    def pct(x, q):
        return float(np.percentile(x, q)) if np.all(np.isfinite(x)) and len(x) else float("nan")

    s1, s3 = evals[:, 2], evals[:, 0]

    # --- Load partition (tension vs compression) within the gauge ----------
    # Dominant principal stress = sigma_1 if |sigma_1| > |sigma_3| else sigma_3;
    # "tensile" elements are those where the dominant stress is positive.
    vol = np.abs(tets.compute_cell_sizes(length=False, area=False, volume=True).cell_data["Volume"])
    in_gauge = (cent[:, 2] > z_lo + 1.0) & (cent[:, 2] < z_hi - 1.0)
    dom = np.where(np.abs(s3) >= np.abs(s1), s3, s1)
    tens = dom > 0
    part = {"tensile_volume_frac": float(vol[in_gauge & tens].sum() / vol[in_gauge].sum())}
    # Field-based load inclination: angle of the dominant principal direction
    # to +z, weighted by |dominant stress| x element volume (no tracing, so
    # no completion/reversal bias).
    wgt = np.abs(dom) * vol
    inc_field = np.degrees(np.arccos(np.clip(np.abs(e_load[:, 2]), 0.0, 1.0)))
    part["inclination_field_deg"] = float((inc_field * wgt)[in_gauge].sum() / wgt[in_gauge].sum())
    try:
        cad = json.loads((run_dir / "cad_params.json").read_text())
        cs = float(cad["CENTER_SPACING"])
        n_rings = int(cad.get("N_RINGS", 5))
        r = np.hypot(cent[:, 0], cent[:, 1])
        rings = {}
        for k in range(n_rings + 1):
            # Radial band around ring k (hexagonal ring k spans radii between
            # ~0.87 k*CS at the hexagon flats and k*CS at its corners).
            m = in_gauge & (r >= (k - 0.5) * cs) & (r < (k + 0.5) * cs)
            if m.sum() == 0:
                continue
            w = vol[m]
            rings[str(k)] = {
                "tensile_volume_frac": float(w[tens[m]].sum() / w.sum()),
                "dominant_stress_mean_MPa": float((dom[m] * w).sum() / w.sum()),
            }
        part["by_ring"] = rings
    except (OSError, KeyError, ValueError):
        pass
    out = {
        "run": str(run_dir.relative_to(RUNS_ROOT)),
        "mode": args.mode,
        "gauge_z_mm": [z_lo, z_hi],
        "seed_z_mm": z_seed,
        "stop_z_mm": [z_stop_lo, z_stop],
        "n_seeds": int(len(cand)),
        "n_complete": int((status == 1).sum()),
        "n_exited": int((status == 0).sum()),
        "n_step_limit": int((status == -1).sum()),
        "n_reversed": int((status == -2).sum()),
        "completion_frac": float((status == 1).mean()) if len(status) else float("nan"),
        "tortuosity": {
            "mean": float(np.nanmean(tau)),
            "median": pct(tau, 50),
            "p10": pct(tau, 10),
            "p90": pct(tau, 90),
            "p99": pct(tau, 99),
            "max": float(np.nanmax(tau)),
            "sd": float(np.nanstd(tau)),
        },
        "inclination_deg": {
            "mean": float(np.nanmean(incl)),
            "median": pct(incl, 50),
            "p90": pct(incl, 90),
        },
        # Corrected principal stresses (true sfepy order; descending naming).
        "sigma1_MPa": {"p50": pct(s1, 50), "p99": pct(s1, 99), "max": float(s1.max())},
        "sigma3_MPa": {"p1": pct(s3, 1), "p50": pct(s3, 50), "min": float(s3.min())},
        "load_partition": part,
        "top_plate": kin,
        "settings": {
            "n_seeds_max": args.n_seeds,
            "step_mm": args.step_mm,
            "k_neighbours": args.k_neighbours,
            "rng_seed": args.seed,
            "seed_grid_mm": 0.3,
        },
        "wall_s": round(time.time() - t0, 1),
    }
    # Store a reproducible subset of complete trajectories for figures.
    keep = comp[: args.save_paths]
    key = out["run"].replace("/", "__")
    np.savez_compressed(
        Path(args.out_dir) / f"{key}_paths.npz",
        lengths=np.array([len(p) for p in keep], dtype=np.int32),
        xyz=np.concatenate(keep).astype(np.float32) if keep else np.zeros((0, 3), np.float32),
        tau=np.array([path_stats(p)[0] / max(path_stats(p)[1], 1e-9) for p in keep], np.float32),
        tau_all=tau.astype(np.float32),
        incl_all=incl.astype(np.float32),
    )
    return out


def main() -> None:
    # --- 6. CLI ----------------------------------------------------------------
    ap = argparse.ArgumentParser(description="Principal-stress load-path tortuosity from FEA runs.")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--mode", choices=["dominant", "compression", "tension"], default="dominant")
    ap.add_argument("--n-seeds", type=int, default=1200)
    ap.add_argument("--step-mm", type=float, default=0.10)
    ap.add_argument("--k-neighbours", type=int, default=10)
    ap.add_argument("--save-paths", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--out-dir", default=str(RUNS_ROOT / "_load_path_analysis"))
    ap.add_argument(
        "--fea-subdir",
        default="final",
        help="fea/<subdir> holding the VTK (default final; e.g. iter_1)",
    )
    args = ap.parse_args()
    global FEA_SUBDIR
    FEA_SUBDIR = args.fea_subdir
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    run_dirs = []
    for pat in args.runs:
        hits = sorted(glob.glob(pat)) or [pat]
        run_dirs += [Path(h).resolve() for h in hits]

    summary_path = Path(args.out_dir) / "load_path_summary.csv"
    rows = {}
    if summary_path.exists():
        with open(summary_path) as fh:
            rows = {r["run"] + "|" + r["mode"]: r for r in csv.DictReader(fh)}

    for rd in run_dirs:
        if not (rd / "fea" / FEA_SUBDIR / "compound_enamel_lattice.vtk").exists():
            print(f"[skip] {rd} (no final VTK)", flush=True)
            continue
        res = analyse(rd, args)
        key = res["run"].replace("/", "__")
        (Path(args.out_dir) / f"{key}.json").write_text(json.dumps(res, indent=2))
        rows[res["run"] + "|" + res["mode"]] = {
            "run": res["run"],
            "mode": res["mode"],
            "n_seeds": res["n_seeds"],
            "completion_frac": round(res["completion_frac"], 4),
            "tau_mean": round(res["tortuosity"]["mean"], 5),
            "tau_median": round(res["tortuosity"]["median"], 5),
            "tau_p90": round(res["tortuosity"]["p90"], 5),
            "tau_max": round(res["tortuosity"]["max"], 5),
            "incl_mean_deg": round(res["inclination_deg"]["mean"], 3),
            "sigma1_p99_MPa": round(res["sigma1_MPa"]["p99"], 2),
            "sigma1_max_MPa": round(res["sigma1_MPa"]["max"], 2),
            "sigma3_p1_MPa": round(res["sigma3_MPa"]["p1"], 2),
            "tensile_volume_frac": round(res["load_partition"]["tensile_volume_frac"], 4),
            "rot_deg_per_pct_strain": round(res["top_plate"]["rotation_deg_per_pct_strain"], 4),
            "wall_s": res["wall_s"],
        }
        print(
            f"[done] {res['run']:55s} complete={res['n_complete']}/{res['n_seeds']} "
            f"tau mean={res['tortuosity']['mean']:.4f} p90={res['tortuosity']['p90']:.4f} "
            f"incl={res['inclination_deg']['mean']:.2f} deg  ({res['wall_s']} s)",
            flush=True,
        )

    with open(summary_path, "w", newline="") as fh:
        fields = list(next(iter(rows.values())).keys()) if rows else []
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows.values():
            w.writerow(r)


if __name__ == "__main__":
    main()
