#!/usr/bin/env python3
"""PIV-guided rod tracking with the displacement sign corrected (2026-09-26).

Purpose
-------
track_rods_piv.py computes each interrogation-window displacement with
skimage.registration.phase_cross_correlation(w1, w2), which returns the
shift that REGISTERS w2 ONTO w1, i.e. the NEGATIVE of the displacement of
the image content from slice s (w1) to slice s+1 (w2). The script stores that
value as the "img2 - img1" displacement and advects every track by +U, so
whenever a track is not re-anchored to a detected centroid (43 % of steps in
the canonical run) it moves opposite to the rods. A synthetic test confirms
the convention: content displaced by (+2, +3) px returns (-2, -3).

This script re-runs the canonical FORWARD tracking (seeded at slice 1, as the
canonical output_piv/track_centerlines_piv.parquet was produced) with every
setting of track_rods_piv.py unchanged EXCEPT the sign: the stored PIV fields
are negated before advection. With the original sign (--sign +1) it
reproduces the canonical parquet (pitch/yaw/tilt within 1e-4 deg), which
validates the re-implementation.

Algorithm (identical to track_rods_piv.py, forward mode)
  seeds       interior (is_boundary = False) rods of slice 1
  advection   x <- x + U(x), y <- y + V(y); U, V from the stored 64-px /
              50 %-overlap phase-correlation fields, linearly interpolated
              (zero outside the window-centre grid), times SIGN
  re-anchor   nearest detected centroid of slice s+1 within 4 px (1.38 um)
  termination a track that leaves the image is ended
  smoothing   Savitzky-Golay (window 15, order 3) on x(z), y(z) in um
  angles      pitch = atan2(dx, dz), yaw = atan2(dy, dz),
              tilt = atan2(sqrt(dx^2 + dy^2), dz) from smoothed endpoints;
              fit_* from least-squares slopes

Inputs
------
output_piv/piv_fields/slice_NNN_uv.npz       stored PIV fields (pairs 1..100)
output_validation_100/slices/slice_NNN_rods.csv   detected rod centroids

Outputs
-------
output_piv_signcorrected/track_centerlines_piv.parquet  same schema as canonical
output_piv_signcorrected/piv_summary.csv                per-slice active tracks, snap fraction
output_piv_signcorrected/provenance.json                inputs (SHA-256), settings, sign

Usage:  python track_rods_piv_signcorrected.py [--sign -1]   (-1 = corrected; +1 = as scripted)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

# --- 1. Settings copied from track_rods_piv.py (do not change independently) -----------
ROOT = Path(__file__).resolve().parent
FIELDS = ROOT / "output_piv" / "piv_fields"
DET = ROOT / "output_validation_100" / "slices"
PIXEL_SIZE_UM = 20.0 / 58.0  # um/px (58-px = 20-um scale bar)
Z_OFFSET_UM = 10.0  # depth of slice 1 below the DEJ
Z_STEP_UM = PIXEL_SIZE_UM  # isotropic voxels
FIRST_SLICE, LAST_SLICE = 1, 101
SNAP_RADIUS_PX = 4
MIN_TRACK_SLICES = 30
SAVGOL_WINDOW, SAVGOL_ORDER = 15, 3
IMG_H, IMG_W = 792, 850  # onehundred_image_stack slice shape


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def smooth_path(xs, ys):
    """Savitzky-Golay smoothing exactly as track_rods_piv.smooth_path."""
    n = len(xs)
    wl = min(SAVGOL_WINDOW, n if n % 2 == 1 else n - 1)
    if wl < SAVGOL_ORDER + 2:
        return np.asarray(xs), np.asarray(ys)
    return savgol_filter(xs, wl, SAVGOL_ORDER), savgol_filter(ys, wl, SAVGOL_ORDER)


def angles(xs_um, ys_um, zs_um):
    """Endpoint and least-squares angles exactly as track_rods_piv.compute_angles."""
    dx, dy, dz = xs_um[-1] - xs_um[0], ys_um[-1] - ys_um[0], zs_um[-1] - zs_um[0]
    pitch = np.degrees(np.arctan2(dx, dz))
    yaw = np.degrees(np.arctan2(dy, dz))
    tilt = np.degrees(np.arctan2(np.hypot(dx, dy), dz))
    zc, xc, yc = zs_um - zs_um.mean(), xs_um - xs_um.mean(), ys_um - ys_um.mean()
    fdx = np.dot(zc, xc) / np.dot(zc, zc)
    fdy = np.dot(zc, yc) / np.dot(zc, zc)
    return (
        float(pitch),
        float(yaw),
        float(tilt),
        float(np.degrees(np.arctan2(fdx, 1.0))),
        float(np.degrees(np.arctan2(fdy, 1.0))),
        float(np.degrees(np.arctan2(np.hypot(fdx, fdy), 1.0))),
        dx,
        dy,
        dz,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--sign",
        type=float,
        default=-1.0,
        help="-1 corrected (default); +1 reproduces the canonical run",
    )
    ap.add_argument("--out", default="output_piv_signcorrected")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(exist_ok=True)

    # --- 2. Load detections and stored fields ---------------------------------------------
    det = {
        s: pd.read_csv(DET / f"slice_{s:03d}_rods.csv") for s in range(FIRST_SLICE, LAST_SLICE + 1)
    }
    fld = {s: np.load(FIELDS / f"slice_{s:03d}_uv.npz") for s in range(FIRST_SLICE, LAST_SLICE)}
    seeds = det[FIRST_SLICE][~det[FIRST_SLICE]["is_boundary"].astype(bool)]
    x = seeds["centroid_x"].to_numpy(float).copy()
    y = seeds["centroid_y"].to_numpy(float).copy()
    n = len(x)
    active = np.ones(n, bool)
    X = np.full((LAST_SLICE, n), np.nan)
    Y = np.full((LAST_SLICE, n), np.nan)
    X[0], Y[0] = x, y
    summary = []

    # --- 3. Forward advection + re-anchoring ---------------------------------------------------
    for s in range(FIRST_SLICE, LAST_SLICE):
        f = fld[s]
        iu = RegularGridInterpolator(
            (f["cy"], f["cx"]),
            f["U"] * args.sign,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        iv = RegularGridInterpolator(
            (f["cy"], f["cx"]),
            f["V"] * args.sign,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        pts = np.column_stack([y, x])
        xn, yn = x + iu(pts), y + iv(pts)
        snap = det[s + 1][["centroid_x", "centroid_y"]].to_numpy().astype(np.float32).astype(float)
        dist, idx = cKDTree(snap).query(np.column_stack([xn, yn]))
        ok = (dist <= SNAP_RADIUS_PX) & active
        xn[ok], yn[ok] = snap[idx[ok], 0], snap[idx[ok], 1]
        left = active & ((xn < 0) | (xn >= IMG_W) | (yn < 0) | (yn >= IMG_H))
        active &= ~left
        x = np.where(active, xn, x)
        y = np.where(active, yn, y)
        X[s][active], Y[s][active] = x[active], y[active]
        summary.append(
            {
                "slice": s + 1,
                "active_tracks": int(active.sum()),
                "snapped_frac": float(ok.sum() / max(active.sum(), 1)),
                "med_disp_px": float(np.median(np.hypot(f["U"], f["V"]))),
            }
        )

    # --- 4. Track records (same schema as the canonical parquet) ----------------------------------
    z_all = Z_OFFSET_UM + np.arange(LAST_SLICE) * Z_STEP_UM
    records = []
    for k in range(n):
        valid = ~np.isnan(X[:, k])
        n_sl = int(valid.sum())
        if n_sl < MIN_TRACK_SLICES:
            continue
        xs_um, ys_um, zs_um = X[valid, k] * PIXEL_SIZE_UM, Y[valid, k] * PIXEL_SIZE_UM, z_all[valid]
        xs_sm, ys_sm = smooth_path(xs_um, ys_um)
        pitch, yaw, tilt, fp, fy, ft, dx, dy, dz = angles(xs_sm, ys_sm, zs_um)
        sl = np.arange(FIRST_SLICE, LAST_SLICE + 1)[valid]
        records.append(
            {
                "track_id": k + 1,
                "n_slices": n_sl,
                "start_slice": int(sl[0]),
                "end_slice": int(sl[-1]),
                "is_complete": bool(sl[0] == FIRST_SLICE and sl[-1] == LAST_SLICE),
                "start_x_px": round(float(X[0, k]), 2),
                "start_y_px": round(float(Y[0, k]), 2),
                "dx_um": round(dx, 4),
                "dy_um": round(dy, 4),
                "dz_um": round(dz, 4),
                "pitch_deg": round(pitch, 4),
                "yaw_deg": round(yaw, 4),
                "tilt_deg": round(tilt, 4),
                "fit_pitch_deg": round(fp, 4),
                "fit_yaw_deg": round(fy, 4),
                "fit_tilt_deg": round(ft, 4),
                "cx_um": ";".join(f"{v:.3f}" for v in xs_sm),
                "cy_um": ";".join(f"{v:.3f}" for v in ys_sm),
                "cz_um": ";".join(f"{v:.3f}" for v in zs_um),
            }
        )
    df = pd.DataFrame(records)
    df.to_parquet(out / "track_centerlines_piv.parquet", index=False)
    pd.DataFrame(summary).to_csv(out / "piv_summary.csv", index=False)

    # --- 5. Provenance ---------------------------------------------------------------------------
    snapped = float(np.mean([r["snapped_frac"] for r in summary]))
    prov = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "sign_applied_to_stored_fields": args.sign,
        "note": "stored fields hold phase_cross_correlation(w1, w2) = -(displacement w1->w2); sign -1 advects "
        "by the true displacement",
        "n_seeds": int(n),
        "n_tracks": int(len(df)),
        "n_complete": int(df.is_complete.sum()),
        "mean_snapped_frac": snapped,
        "inputs_sha256": {
            "fields_manifest": hashlib.sha256(
                "".join(sha256(FIELDS / f"slice_{s:03d}_uv.npz") for s in range(1, 101)).encode()
            ).hexdigest(),
            "detections_manifest": hashlib.sha256(
                "".join(sha256(DET / f"slice_{s:03d}_rods.csv") for s in range(1, 102)).encode()
            ).hexdigest(),
        },
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=1))
    print(json.dumps({k: v for k, v in prov.items() if k != "inputs_sha256"}, indent=1))
    print(
        f"tilt {df.tilt_deg.mean():.2f} ± {df.tilt_deg.std():.2f} deg; pitch {df.pitch_deg.mean():+.2f} ± "
        f"{df.pitch_deg.std():.2f}; yaw {df.yaw_deg.mean():+.2f} ± {df.yaw_deg.std():.2f}"
    )


if __name__ == "__main__":
    main()
