"""Digital-twin generator: rod-by-rod STL from PIV trajectories.

This is the **measurement-derived** model route in the biomimetic_pipeline,
complementing the parametric continuous-twist route in `cad_runner.py`. Each
rod in the synchrotron volume is rendered as a smooth tube swept along its
PIV-tracked 3D centerline; by construction the architecture is inherited
exactly from measurement (no parameter mapping, no manufacturability clamps).

Per the biomimetic_pipeline 'wraps, never modifies' policy, this module does
NOT modify the upstream `microct_pipeline/visualize_3d_piv.py`. It re-implements
the rod-tube core in pyvista, applies HEAVIER smoothing than the upstream
canonical STL (gaussian sigma 12 vs 3, plus mesh-level Taubin smoothing for
clean cylinder surfaces), and emits an artifact pattern matching cad_runner:

    <export_dir>/cad/
        digital_twin.stl                  - rod-by-rod tube mesh
        digital_twin_provenance.json      - track count, smoothing, source paths
        digital_twin_run.log              - per-run log

Inclusion rule: tracks with n_slices >= 30 (the canonical PIV threshold).

Limitations (Path A scope, see README s24c):
  - Output STL is for visualization / printing / posterity. FEA on the
    rod-by-rod geometry is NOT yet wired into mesh_runner / fea_runner; the
    rod-by-rod boolean union is meshing-hard at this scale and is deferred to
    Path B (future work).
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

# pyvista is needed for tube + Taubin smoothing
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import pyvista as pv
from scipy.ndimage import gaussian_filter1d

# ---- canonical defaults ----------------------------------------------------

DEFAULT_PIV_PARQUET = (
    Path(os.environ.get("MICROCT_PIPELINE_ROOT", "microct_pipeline"))
    / "output_piv_back"
    / "track_centerlines_piv.parquet"
)
MIN_TRACK_LENGTH_DEFAULT = 30
ROD_RADIUS_UM_DEFAULT = 5.0
GAUSSIAN_SIGMA_DEFAULT = 24.0  # heaviest centerline smoothing
TUBE_SIDES_DEFAULT = 24
N_AXIAL_POINTS_DEFAULT = 80  # doubled: kills the "stacked cylinder" seams
TAUBIN_ITERS_DEFAULT = 50  # heavier surface smoothing
TAUBIN_PASS_BAND_DEFAULT = 0.04  # tighter pass band -> stronger per-iter


@dataclass
class DigitalTwinResult:
    export_dir: Path
    stl_path: Path
    provenance_path: Path
    log_path: Path
    n_rods: int
    n_faces: int
    bounds_um: tuple
    model_type: str = "digital_twin"


# ---- helpers ---------------------------------------------------------------


def _parse(s: str) -> np.ndarray:
    return np.fromstring(s, sep=";")


def _build_smoothed_tubes(
    df: pd.DataFrame,
    *,
    rod_radius_um: float,
    gaussian_sigma: float,
    tube_sides: int,
    n_axial_points: int,
) -> pv.PolyData:
    """Sweep one tube per rod with heavy gaussian path smoothing, then merge."""
    blocks = []
    for _, row in df.iterrows():
        x = _parse(row["cx_um"])
        y = _parse(row["cy_um"])
        z = _parse(row["cz_um"])
        if len(x) < 4:
            continue
        x = gaussian_filter1d(x, sigma=gaussian_sigma)
        y = gaussian_filter1d(y, sigma=gaussian_sigma)
        n = len(x)
        if n > n_axial_points:
            idx = np.linspace(0, n - 1, n_axial_points).astype(int)
            x, y, z = x[idx], y[idx], z[idx]
        pts = np.column_stack([x, y, z])
        line = pv.lines_from_points(pts)
        tube = line.tube(radius=rod_radius_um, n_sides=tube_sides, capping=True)
        blocks.append(tube)

    if not blocks:
        raise RuntimeError("No rod tubes built (check PIV parquet input).")

    merged = blocks[0]
    for b in blocks[1:]:
        merged = merged.merge(b, merge_points=False)
    return merged


def _taubin_smooth(mesh: pv.PolyData, *, n_iter: int, pass_band: float) -> pv.PolyData:
    return mesh.smooth_taubin(
        n_iter=n_iter,
        pass_band=pass_band,
        boundary_smoothing=True,
        feature_smoothing=False,
        non_manifold_smoothing=True,
    )


# ---- public entry point ----------------------------------------------------


def run(
    export_dir: Path,
    *,
    piv_parquet: Optional[Path] = None,
    min_track_length: int = MIN_TRACK_LENGTH_DEFAULT,
    rod_radius_um: float = ROD_RADIUS_UM_DEFAULT,
    gaussian_sigma: float = GAUSSIAN_SIGMA_DEFAULT,
    tube_sides: int = TUBE_SIDES_DEFAULT,
    n_axial_points: int = N_AXIAL_POINTS_DEFAULT,
    taubin_iters: int = TAUBIN_ITERS_DEFAULT,
    taubin_pass_band: float = TAUBIN_PASS_BAND_DEFAULT,
    morphometrics_path: Optional[Path] = None,
) -> DigitalTwinResult:
    """Generate the heavy-smoothed digital-twin STL for a single specimen.

    `morphometrics_path` is recorded in provenance only; this module does not
    consume morphometric features (the digital twin is data-driven from the
    PIV parquet directly). It is recorded so a downstream consumer can prove
    which run the twin was emitted from.
    """
    export_dir = Path(export_dir)
    cad_dir = export_dir / "cad"
    cad_dir.mkdir(parents=True, exist_ok=True)

    log_path = cad_dir / "digital_twin_run.log"
    stl_path = cad_dir / "digital_twin.stl"
    provenance_path = cad_dir / "digital_twin_provenance.json"

    parquet = Path(piv_parquet) if piv_parquet else DEFAULT_PIV_PARQUET
    if not parquet.exists():
        raise FileNotFoundError(f"PIV parquet not found: {parquet}")

    log = []

    def _log(s: str):
        log.append(s)
        print(s)

    t0 = time.time()
    _log(f"[digital_twin] reading {parquet}")
    df = pd.read_parquet(parquet)
    df = df[df["n_slices"] >= min_track_length].reset_index(drop=True)
    _log(f"[digital_twin] {len(df)} tracks pass n_slices >= {min_track_length}")

    _log(
        f"[digital_twin] sweeping tubes "
        f"(R={rod_radius_um} um, sigma={gaussian_sigma}, sides={tube_sides})..."
    )
    merged = _build_smoothed_tubes(
        df,
        rod_radius_um=rod_radius_um,
        gaussian_sigma=gaussian_sigma,
        tube_sides=tube_sides,
        n_axial_points=n_axial_points,
    )
    _log(
        f"[digital_twin] merged tube mesh: {merged.n_cells:,} faces, " f"{merged.n_points:,} points"
    )

    if taubin_iters > 0:
        _log(
            f"[digital_twin] Taubin smoothing ({taubin_iters} iter, "
            f"pass_band={taubin_pass_band})..."
        )
        merged = _taubin_smooth(merged, n_iter=taubin_iters, pass_band=taubin_pass_band)

    _log(f"[digital_twin] writing STL -> {stl_path}")
    merged.save(str(stl_path))

    bounds = merged.bounds
    elapsed = time.time() - t0

    provenance = {
        "model_type": "digital_twin",
        "source_parquet": str(parquet),
        "morphometrics_path": str(morphometrics_path) if morphometrics_path else None,
        "inclusion_rule": {"min_track_length": min_track_length},
        "n_rods": int(len(df)),
        "n_faces": int(merged.n_cells),
        "n_points": int(merged.n_points),
        "bounds_um": {
            "x": [float(bounds[0]), float(bounds[1])],
            "y": [float(bounds[2]), float(bounds[3])],
            "z": [float(bounds[4]), float(bounds[5])],
        },
        "smoothing": {
            "centerline_gaussian_sigma_slices": gaussian_sigma,
            "axial_resample_points": n_axial_points,
            "tube_sides": tube_sides,
            "taubin_iterations": taubin_iters,
            "taubin_pass_band": taubin_pass_band,
        },
        "rod_radius_um": rod_radius_um,
        "stl_size_bytes": int(stl_path.stat().st_size),
        "elapsed_seconds": round(elapsed, 2),
        "upstream_reference": (
            "microct_pipeline/visualize_3d_piv.py "
            "(canonical PIV-derived rod-tube generator; this run uses heavier "
            "smoothing parameters but the same inclusion rule and parquet)"
        ),
        "fea_status": "Path A only — FEA on rod-by-rod geometry is deferred "
        "to Path B (see biomimetic_pipeline/README.md s24c).",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2))
    log_path.write_text("\n".join(log))

    return DigitalTwinResult(
        export_dir=export_dir,
        stl_path=stl_path,
        provenance_path=provenance_path,
        log_path=log_path,
        n_rods=int(len(df)),
        n_faces=int(merged.n_cells),
        bounds_um=(
            float(bounds[1] - bounds[0]),
            float(bounds[3] - bounds[2]),
            float(bounds[5] - bounds[4]),
        ),
    )


# ---- CLI -------------------------------------------------------------------


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Generate the digital-twin rod-by-rod STL.")
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Run export directory (a 'cad/' subdir will be created).",
    )
    p.add_argument(
        "--piv-parquet",
        type=Path,
        default=None,
        help=f"Override the PIV parquet (default: {DEFAULT_PIV_PARQUET}).",
    )
    p.add_argument("--gaussian-sigma", type=float, default=GAUSSIAN_SIGMA_DEFAULT)
    p.add_argument("--rod-radius-um", type=float, default=ROD_RADIUS_UM_DEFAULT)
    p.add_argument("--tube-sides", type=int, default=TUBE_SIDES_DEFAULT)
    p.add_argument("--n-axial-points", type=int, default=N_AXIAL_POINTS_DEFAULT)
    p.add_argument("--taubin-iters", type=int, default=TAUBIN_ITERS_DEFAULT)
    p.add_argument("--taubin-pass-band", type=float, default=TAUBIN_PASS_BAND_DEFAULT)
    p.add_argument("--min-track-length", type=int, default=MIN_TRACK_LENGTH_DEFAULT)
    p.add_argument("--morphometrics", type=Path, default=None)
    args = p.parse_args()

    res = run(
        args.out_dir,
        piv_parquet=args.piv_parquet,
        min_track_length=args.min_track_length,
        rod_radius_um=args.rod_radius_um,
        gaussian_sigma=args.gaussian_sigma,
        tube_sides=args.tube_sides,
        n_axial_points=args.n_axial_points,
        taubin_iters=args.taubin_iters,
        taubin_pass_band=args.taubin_pass_band,
        morphometrics_path=args.morphometrics,
    )
    print(f"\n[digital_twin] OK: {res.n_rods} rods, " f"{res.n_faces:,} faces, {res.stl_path}")


if __name__ == "__main__":
    _cli()
