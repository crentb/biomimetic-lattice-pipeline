"""Digital-twin generator (B-spline route): cubic-spline-interpolated rod
centerlines swept as tube meshes.

Why this exists: the gaussian-smoothed-polyline approach in
`digital_twin_runner.py` discretizes each rod into N axial samples and connects
them with linear segments. With low N you see the segments as "stacks" in the
render; with high N the mesh size grows linearly. A cubic B-spline gives a
truly continuous C2 source curve, sampled at whatever density the tessellation
needs. This is structurally what CAD tools do internally for tube modelling
(NURBS sweeps), but here implemented in scipy (splprep) so we don't need an
OpenCASCADE/CadQuery subprocess.

Per-rod watertight mesh; per-rod meshes merged but not boolean-unioned (the
SDF route in `digital_twin_sdf_runner.py` is the "true single solid" option).

Output: <export_dir>/cad/digital_twin_bspline.stl + provenance.json + log
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

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import pyvista as pv
from scipy.interpolate import splev, splprep

DEFAULT_PIV_PARQUET = (
    Path(os.environ.get("MICROCT_PIPELINE_ROOT", "microct_pipeline"))
    / "output_piv_back"
    / "track_centerlines_piv.parquet"
)
MIN_TRACK_LENGTH_DEFAULT = 30
ROD_RADIUS_UM_DEFAULT = 5.0
TUBE_SIDES_DEFAULT = 24
N_SPLINE_SAMPLES_DEFAULT = 200  # samples along the C2 spline per rod
SPLINE_SMOOTHING_DEFAULT = 0.5  # splprep `s`: 0 = exact interpolate, larger = smoother
TAUBIN_ITERS_DEFAULT = 30  # lighter than tube method (spline is already smooth)
TAUBIN_PASS_BAND_DEFAULT = 0.05


@dataclass
class BsplineDigitalTwinResult:
    export_dir: Path
    stl_path: Path
    provenance_path: Path
    log_path: Path
    n_rods: int
    n_faces: int
    bounds_um: tuple
    model_type: str = "digital_twin_bspline"


def _parse(s: str) -> np.ndarray:
    return np.fromstring(s, sep=";")


def _spline_resample(x, y, z, *, n_samples: int, smoothing: float):
    n = len(x)
    if n < 4:
        return None
    k = min(3, n - 1)
    try:
        tck, _u = splprep([x, y, z], s=smoothing * n, k=k, quiet=1)
        u_new = np.linspace(0.0, 1.0, n_samples)
        x_new, y_new, z_new = splev(u_new, tck)
        return np.column_stack([x_new, y_new, z_new])
    except Exception:
        return None


def _build_tubes_and_caps(
    df: pd.DataFrame,
    *,
    rod_radius_um: float,
    tube_sides: int,
    n_spline_samples: int,
    spline_smoothing: float,
    taubin_iters: int,
    taubin_pass_band: float,
):
    blocks = []
    cap_polylines = []
    for _, row in df.iterrows():
        x = _parse(row["cx_um"])
        y = _parse(row["cy_um"])
        z = _parse(row["cz_um"])
        pts = _spline_resample(x, y, z, n_samples=n_spline_samples, smoothing=spline_smoothing)
        if pts is None:
            continue
        line = pv.lines_from_points(pts)
        tube = line.tube(radius=rod_radius_um, n_sides=tube_sides, capping=True)
        if taubin_iters > 0:
            tube = tube.smooth_taubin(
                n_iter=taubin_iters,
                pass_band=taubin_pass_band,
                boundary_smoothing=True,
                feature_smoothing=False,
                non_manifold_smoothing=True,
            )
        blocks.append(tube)

        # cap circles (for white-outline rendering, optional consumer)
        for end_idx in (0, -1):
            tan = pts[1] - pts[0] if end_idx == 0 else pts[-1] - pts[-2]
            tan = tan / (np.linalg.norm(tan) + 1e-12)
            up = np.array([0.0, 0.0, 1.0]) if abs(tan[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            n1 = np.cross(tan, up)
            n1 /= np.linalg.norm(n1) + 1e-12
            n2 = np.cross(tan, n1)
            n2 /= np.linalg.norm(n2) + 1e-12
            theta = np.linspace(0, 2 * np.pi, 24, endpoint=True)
            ring = pts[end_idx] + rod_radius_um * (
                np.outer(np.cos(theta), n1) + np.outer(np.sin(theta), n2)
            )
            cap_polylines.append(ring)

    if not blocks:
        raise RuntimeError("no rods produced (check parquet)")
    merged = blocks[0]
    for b in blocks[1:]:
        merged = merged.merge(b, merge_points=False)
    return merged, cap_polylines


def run(
    export_dir: Path,
    *,
    piv_parquet: Optional[Path] = None,
    min_track_length: int = MIN_TRACK_LENGTH_DEFAULT,
    rod_radius_um: float = ROD_RADIUS_UM_DEFAULT,
    tube_sides: int = TUBE_SIDES_DEFAULT,
    n_spline_samples: int = N_SPLINE_SAMPLES_DEFAULT,
    spline_smoothing: float = SPLINE_SMOOTHING_DEFAULT,
    taubin_iters: int = TAUBIN_ITERS_DEFAULT,
    taubin_pass_band: float = TAUBIN_PASS_BAND_DEFAULT,
    morphometrics_path: Optional[Path] = None,
) -> BsplineDigitalTwinResult:
    export_dir = Path(export_dir)
    cad_dir = export_dir / "cad"
    cad_dir.mkdir(parents=True, exist_ok=True)
    log_path = cad_dir / "digital_twin_bspline_run.log"
    stl_path = cad_dir / "digital_twin_bspline.stl"
    provenance_path = cad_dir / "digital_twin_bspline_provenance.json"

    parquet = Path(piv_parquet) if piv_parquet else DEFAULT_PIV_PARQUET
    if not parquet.exists():
        raise FileNotFoundError(f"PIV parquet not found: {parquet}")

    log = []

    def _log(s: str):
        log.append(s)
        print(s)

    t0 = time.time()
    _log(f"[bspline-twin] reading {parquet}")
    df = pd.read_parquet(parquet)
    df = df[df["n_slices"] >= min_track_length].reset_index(drop=True)
    _log(f"[bspline-twin] {len(df)} tracks pass n_slices >= {min_track_length}")

    _log(
        f"[bspline-twin] fitting cubic B-splines (s={spline_smoothing}) and "
        f"sweeping tubes (R={rod_radius_um} um, {n_spline_samples} samples per spline)..."
    )
    merged, _caps = _build_tubes_and_caps(
        df,
        rod_radius_um=rod_radius_um,
        tube_sides=tube_sides,
        n_spline_samples=n_spline_samples,
        spline_smoothing=spline_smoothing,
        taubin_iters=taubin_iters,
        taubin_pass_band=taubin_pass_band,
    )
    _log(
        f"[bspline-twin] merged tube mesh: {merged.n_cells:,} faces, " f"{merged.n_points:,} points"
    )

    _log(f"[bspline-twin] writing STL -> {stl_path}")
    merged.save(str(stl_path))

    bounds = merged.bounds
    elapsed = time.time() - t0

    provenance = {
        "model_type": "digital_twin_bspline",
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
            "method": "cubic B-spline (scipy.interpolate.splprep, k=3)",
            "smoothing_factor_s_per_n": spline_smoothing,
            "n_samples_per_spline": n_spline_samples,
            "tube_sides": tube_sides,
            "taubin_iterations": taubin_iters,
            "taubin_pass_band": taubin_pass_band,
        },
        "rod_radius_um": rod_radius_um,
        "stl_size_bytes": int(stl_path.stat().st_size),
        "elapsed_seconds": round(elapsed, 2),
        "comparison_note": (
            "B-spline route: C2 continuous source curves vs the gaussian-polyline "
            "approach in digital_twin_runner.py. STL output is still triangulated; "
            "the spline only changes WHERE the tessellation samples the curve. "
            "For STEP / NURBS export, see future work in README s24c."
        ),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2))
    log_path.write_text("\n".join(log))

    return BsplineDigitalTwinResult(
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


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Generate B-spline-based digital-twin STL.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--piv-parquet", type=Path, default=None)
    p.add_argument("--rod-radius-um", type=float, default=ROD_RADIUS_UM_DEFAULT)
    p.add_argument("--tube-sides", type=int, default=TUBE_SIDES_DEFAULT)
    p.add_argument("--n-spline-samples", type=int, default=N_SPLINE_SAMPLES_DEFAULT)
    p.add_argument("--spline-smoothing", type=float, default=SPLINE_SMOOTHING_DEFAULT)
    p.add_argument("--taubin-iters", type=int, default=TAUBIN_ITERS_DEFAULT)
    p.add_argument("--taubin-pass-band", type=float, default=TAUBIN_PASS_BAND_DEFAULT)
    p.add_argument("--min-track-length", type=int, default=MIN_TRACK_LENGTH_DEFAULT)
    args = p.parse_args()
    res = run(
        args.out_dir,
        piv_parquet=args.piv_parquet,
        min_track_length=args.min_track_length,
        rod_radius_um=args.rod_radius_um,
        tube_sides=args.tube_sides,
        n_spline_samples=args.n_spline_samples,
        spline_smoothing=args.spline_smoothing,
        taubin_iters=args.taubin_iters,
        taubin_pass_band=args.taubin_pass_band,
    )
    print(f"\n[bspline-twin] OK: {res.n_rods} rods, {res.n_faces:,} faces, {res.stl_path}")


if __name__ == "__main__":
    _cli()
