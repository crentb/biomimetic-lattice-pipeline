"""Crack-deflection metric: streamline tortuosity of the max-principal-stress
field from an FEA element_results_compression.csv.

Algorithm:
  1. Read per-element (cx, cy, cz, p1_MPa) and build a RegularGrid interpolant
     of p1 and its gradient direction over the lattice bounding box.
  2. Seed streamlines on a grid across the bottom face (z = z_min).
  3. Integrate each streamline forward along grad(p1) (sign-adjusted to point
     +z) until it crosses the top face.
  4. For each streamline compute arc_length / chord_length (same arc/chord
     formula as compute_tortuosity.py and som_crack_paths_3d.py so biology and
     FEA tortuosity are numerically comparable).
  5. Aggregate mean, p90, max across streamlines.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CrackDeflectionResult:
    n_streamlines: int
    tortuosity_mean: float
    tortuosity_p90: float
    tortuosity_max: float
    mean_arc_length_mm: float
    mean_chord_length_mm: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "crack_deflection_n_streamlines": float(self.n_streamlines),
            "crack_deflection_tortuosity_mean": float(self.tortuosity_mean),
            "crack_deflection_tortuosity_p90": float(self.tortuosity_p90),
            "crack_deflection_tortuosity_max": float(self.tortuosity_max),
            "crack_deflection_mean_arc_length_mm": float(self.mean_arc_length_mm),
            "crack_deflection_mean_chord_length_mm": float(self.mean_chord_length_mm),
        }


def compute(
    element_results_path: Path,
    lattice_params_path: Optional[Path] = None,
    seed_grid_n: int = 8,
    grid_resolution: int = 32,
    max_steps: int = 500,
) -> CrackDeflectionResult:
    """Compute crack-deflection tortuosity from element_results_compression.csv."""
    import numpy as np

    pts, p1 = _load_elements(element_results_path)
    if pts.shape[0] < 100:
        return CrackDeflectionResult(0, 1.0, 1.0, 1.0, 0.0, 0.0)

    xmin, ymin, zmin = pts.min(axis=0)
    xmax, ymax, zmax = pts.max(axis=0)
    nx = ny = int(grid_resolution)
    nz = max(int(grid_resolution), 16)

    grid_x = np.linspace(xmin, xmax, nx)
    grid_y = np.linspace(ymin, ymax, ny)
    grid_z = np.linspace(zmin, zmax, nz)

    p1_grid = _rasterize_to_grid(pts, p1, grid_x, grid_y, grid_z)

    # Pre-compute gradient field and a +z-aligned direction vector at every cell.
    gz, gy, gx = np.gradient(p1_grid, grid_z, grid_y, grid_x, edge_order=1)
    gmag = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-12
    # Orient vectors to point upward (+z) so streamlines progress through the
    # specimen from bottom face to top face.
    sign = np.where(gz >= 0.0, 1.0, -1.0)
    dx = sign * gx / gmag
    dy = sign * gy / gmag
    dz = sign * gz / gmag
    # Force strictly-positive z-component after sign flip (avoid stalling).
    dz = np.where(dz < 1e-6, 1e-6, dz)
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx /= norm
    dy /= norm
    dz /= norm

    # Seed streamlines on bottom face over a square sub-grid.
    seed_x = np.linspace(xmin + 0.1 * (xmax - xmin), xmax - 0.1 * (xmax - xmin), seed_grid_n)
    seed_y = np.linspace(ymin + 0.1 * (ymax - ymin), ymax - 0.1 * (ymax - ymin), seed_grid_n)

    step = (zmax - zmin) / max_steps * 1.2

    # Integrate ALL streamlines simultaneously (vectorized on purpose).
    # The original per-seed/per-step loop made three scalar _trilinear calls
    # per step — ~83k Python-level interpolations for a 200k-element run,
    # measured at 46% of compute()'s runtime (cProfile, 2026-07-15). Batching
    # every active streamline into one array step, with the trilinear cell
    # weights computed ONCE and reused for all three direction components,
    # performs the same math in a handful of numpy ops per step:
    #   * inactive rows stop moving exactly where the scalar loop's `break`
    #     left them (stalled field, reached top, or exited x/y bounds),
    #   * arc length accumulates only over moves actually taken,
    #   * per-row arithmetic is the same expressions in the same order.
    sxg, syg = np.meshgrid(seed_x, seed_y, indexing="ij")
    pos = np.stack(
        [sxg.ravel(), syg.ravel(), np.full(sxg.size, zmin + 1e-6 * (zmax - zmin))], axis=1
    )
    start = pos.copy()
    arc = np.zeros(pos.shape[0])
    active = np.ones(pos.shape[0], dtype=bool)

    for _ in range(max_steps):
        if not active.any():
            break
        v = _trilinear_batch((dx, dy, dz), grid_x, grid_y, grid_z, pos[active])
        v_norm = np.linalg.norm(v, axis=1)
        alive = v_norm >= 1e-9  # stalled field -> stop WITHOUT moving (as before)
        idx = np.flatnonzero(active)
        active[idx[~alive]] = False
        if not alive.any():
            break
        moving = idx[alive]
        v = v[alive] / v_norm[alive][:, None]
        delta = step * v
        next_pos = pos[moving] + delta
        arc[moving] += np.linalg.norm(next_pos - pos[moving], axis=1)
        pos[moving] = next_pos
        # Termination AFTER the move, exactly like the scalar loop's breaks.
        reached_top = next_pos[:, 2] >= zmax - 1e-6 * (zmax - zmin)
        out_of_xy = ~(
            (xmin <= next_pos[:, 0])
            & (next_pos[:, 0] <= xmax)
            & (ymin <= next_pos[:, 1])
            & (next_pos[:, 1] <= ymax)
        )
        active[moving[reached_top | out_of_xy]] = False

    chord = np.linalg.norm(pos - start, axis=1)
    keep = (chord > 1e-6) & (arc > 0)
    tortuosities: List[float] = (arc[keep] / chord[keep]).tolist()
    arcs_mm: List[float] = arc[keep].tolist()
    chords_mm: List[float] = chord[keep].tolist()

    if not tortuosities:
        return CrackDeflectionResult(0, 1.0, 1.0, 1.0, 0.0, 0.0)

    t = np.asarray(tortuosities)
    return CrackDeflectionResult(
        n_streamlines=int(t.size),
        tortuosity_mean=float(t.mean()),
        tortuosity_p90=float(np.percentile(t, 90)),
        tortuosity_max=float(t.max()),
        mean_arc_length_mm=float(np.mean(arcs_mm)),
        mean_chord_length_mm=float(np.mean(chords_mm)),
    )


def _load_elements(path: Path) -> Tuple[Any, Any]:
    import numpy as np

    # Fast path: numpy's C CSV parser. csv.DictReader builds a dict per row
    # and was measured at 30% of compute()'s runtime on a 200k-element file
    # (cProfile, 2026-07-15); np.loadtxt reads the same columns ~5-10x
    # faster. Column indices are resolved from the header by NAME, so column
    # order stays flexible exactly as with DictReader.
    try:
        with open(path, newline="") as fh:
            header = fh.readline().strip().split(",")
        cols = [header.index(c) for c in ("cx_mm", "cy_mm", "cz_mm", "p1_MPa")]
        data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=cols, ndmin=2)
        return data[:, :3], data[:, 3]
    except (OSError, ValueError):
        # Tolerant fallback (original behavior): skip malformed rows / missing
        # columns instead of failing the whole load. np.loadtxt raises on any
        # dirty row, so files that need forgiveness land here.
        pass

    pts: List[Tuple[float, float, float]] = []
    p1s: List[float] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                pts.append((float(row["cx_mm"]), float(row["cy_mm"]), float(row["cz_mm"])))
                p1s.append(float(row["p1_MPa"]))
            except (KeyError, ValueError):
                continue
    return np.asarray(pts), np.asarray(p1s)


def _rasterize_to_grid(pts: Any, values: Any, gx: Any, gy: Any, gz: Any) -> Any:
    """Bin-average scattered element values onto a regular grid."""
    import numpy as np

    nx, ny, nz = gx.size, gy.size, gz.size
    grid = np.zeros((nz, ny, nx), dtype=float)
    count = np.zeros_like(grid)
    ix = np.searchsorted(gx, pts[:, 0]).clip(0, nx - 1)
    iy = np.searchsorted(gy, pts[:, 1]).clip(0, ny - 1)
    iz = np.searchsorted(gz, pts[:, 2]).clip(0, nz - 1)
    # Unbuffered scatter-add replaces the original per-element Python loop:
    # np.add.at accumulates in element order, so sums are identical to the
    # loop's (same additions, same sequence) with no per-row interpreter cost.
    np.add.at(grid, (iz, iy, ix), values)
    np.add.at(count, (iz, iy, ix), 1.0)
    mask = count > 0
    grid[mask] /= count[mask]
    # Fill empty cells by nearest-neighbor sweep (max 4 passes) so integrator
    # always has a defined value.
    for _ in range(4):
        empty = ~mask
        if not empty.any():
            break
        filled = grid.copy()
        # Pad-aware 6-neighbor mean
        for shift in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            rolled = np.roll(grid, shift, axis=(0, 1, 2))
            filled = np.where(empty & ~mask, filled + rolled, filled)
        count_add = np.zeros_like(grid)
        for shift in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            rolled_mask = np.roll(mask.astype(float), shift, axis=(0, 1, 2))
            count_add = count_add + rolled_mask
        fill_mask = empty & (count_add > 0)
        grid[fill_mask] = filled[fill_mask] / count_add[fill_mask]
        mask = mask | fill_mask
    return grid


def _trilinear(vol: Any, gx: Any, gy: Any, gz: Any, p: Any) -> float:
    """Trilinear interpolation of `vol[nz, ny, nx]` at point p=(x,y,z)."""
    import numpy as np

    x, y, z = float(p[0]), float(p[1]), float(p[2])
    ix = np.searchsorted(gx, x) - 1
    iy = np.searchsorted(gy, y) - 1
    iz = np.searchsorted(gz, z) - 1
    ix = max(0, min(ix, gx.size - 2))
    iy = max(0, min(iy, gy.size - 2))
    iz = max(0, min(iz, gz.size - 2))
    tx = (x - gx[ix]) / max(gx[ix + 1] - gx[ix], 1e-12)
    ty = (y - gy[iy]) / max(gy[iy + 1] - gy[iy], 1e-12)
    tz = (z - gz[iz]) / max(gz[iz + 1] - gz[iz], 1e-12)
    c000 = vol[iz, iy, ix]
    c100 = vol[iz, iy, ix + 1]
    c010 = vol[iz, iy + 1, ix]
    c110 = vol[iz, iy + 1, ix + 1]
    c001 = vol[iz + 1, iy, ix]
    c101 = vol[iz + 1, iy, ix + 1]
    c011 = vol[iz + 1, iy + 1, ix]
    c111 = vol[iz + 1, iy + 1, ix + 1]
    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    return float(c0 * (1 - tz) + c1 * tz)


def _trilinear_batch(vols: Tuple[Any, Any, Any], gx: Any, gy: Any, gz: Any, pts: Any) -> Any:
    """Trilinear interpolation of three co-located volumes at many points.

    Returns an (N, 3) array: column j is vols[j] sampled at pts. The cell
    indices and the eight corner weights depend only on the sample POINTS,
    not on the volume being sampled — so they are computed once and reused
    for all three direction components (the scalar `_trilinear` recomputed
    them per component, tripling the work). Same clamping and weight
    formulas as `_trilinear`, evaluated array-wise.
    """
    import numpy as np

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    ix = (np.searchsorted(gx, x) - 1).clip(0, gx.size - 2)
    iy = (np.searchsorted(gy, y) - 1).clip(0, gy.size - 2)
    iz = (np.searchsorted(gz, z) - 1).clip(0, gz.size - 2)
    tx = (x - gx[ix]) / np.maximum(gx[ix + 1] - gx[ix], 1e-12)
    ty = (y - gy[iy]) / np.maximum(gy[iy + 1] - gy[iy], 1e-12)
    tz = (z - gz[iz]) / np.maximum(gz[iz + 1] - gz[iz], 1e-12)

    out = np.empty((pts.shape[0], len(vols)))
    for j, vol in enumerate(vols):
        c000 = vol[iz, iy, ix]
        c100 = vol[iz, iy, ix + 1]
        c010 = vol[iz, iy + 1, ix]
        c110 = vol[iz, iy + 1, ix + 1]
        c001 = vol[iz + 1, iy, ix]
        c101 = vol[iz + 1, iy, ix + 1]
        c011 = vol[iz + 1, iy + 1, ix]
        c111 = vol[iz + 1, iy + 1, ix + 1]
        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        out[:, j] = c0 * (1 - tz) + c1 * tz
    return out


def save(result: CrackDeflectionResult, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.as_dict(), indent=2))
    return out_path
