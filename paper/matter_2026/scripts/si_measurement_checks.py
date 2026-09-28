#!/usr/bin/env python
"""Measurement checks requested in review: tortuosity noise floor, band period
across the bands, and the tilt of the principal fabric axis.

Purpose
-------
1. Tortuosity noise floor. With slices only 0.345 um apart, sub-micrometre
   lateral centroid fluctuations inflate arc length. Each centreline is
   split into a smooth part (Gaussian, sigma = 3 slices = 1.0 um, below the
   2.1-um rod radius) and a residual. Reported: tortuosity of the raw and
   smoothed tracks, and the tortuosity that the residual ALONE produces when
   added to a straight chord (the high-frequency contribution; it keeps the
   residual's autocorrelation, unlike i.i.d. noise of the same size, which
   overstates the floor and is kept only for reference).
2. Band period across the bands. The canonical period used horizontal
   transects, which in this field run roughly ALONG the Hunter-Schreger
   bands. Here (a) the 2-D Fourier transform of the pixel-level pitch field
   (Hann-windowed) gives the dominant wave vector and the band-normal
   direction; (b) FFT spectra of eight vertical transects (normal to the
   bands); (c) the spacing of pitch sign changes along vertical profiles
   (every 20 columns), which is not quantised to W/k bins. Two sign changes
   make one period (one band of each sense).
3. Principal fabric axis and the pitch depth profile. Angle between the
   Woodcock lambda_1 eigenvector of the unit net-displacement vectors and
   the imaging (z) axis, and the per-slice mean signed pitch that the twist
   mapper integrates into RING_ROTATION, to judge whether the mapped twist
   (and its single handedness) could reflect specimen orientation.
4. Tracking fidelity. Distance from every tracked centreline point to the
   nearest segmented rod centroid of the same slice, against uniformly
   random points (chance baseline).

Inputs
------
microct_pipeline/output_piv/track_centerlines_piv.parquet
matter_v2/data/canonical_som_bands.npz (accumulated displacement field)
microct_pipeline/output_validation_100/slices/slice_NNN_rods.csv (segmented centroids)
biomimetic_pipeline/runs/live_001/morphometrics.json (pitch depth profile)

Outputs
-------
matter_v2/data/si_measurement_checks.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
MANU = HERE.parent
REPO = MANU.parents[2]
PARQUET = (
    REPO / "output_piv_signcorrected/track_centerlines_piv.parquet"
)  # sign-corrected PIV tracking (track_rods_piv_signcorrected.py; see SI)
SOM = MANU / "data/canonical_som_bands.npz"
OUT = MANU / "data/si_measurement_checks.json"
PX_UM = 20.0 / 58.0
RNG = np.random.default_rng(20260925)


def tort(P: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum() / np.linalg.norm(P[-1] - P[0]))


def main() -> None:
    out: dict = {}
    d = pd.read_parquet(PARQUET)

    # --- 1. Tortuosity noise floor ------------------------------------------------------
    raw, smooth, resid_only, iid, jit = [], [], [], [], []
    for _, r in d.iterrows():
        x = np.fromstring(r.cx_um, sep=";")
        y = np.fromstring(r.cy_um, sep=";")
        z = np.fromstring(r.cz_um, sep=";")
        xs, ys = gaussian_filter1d(x, 3.0), gaussian_filter1d(y, 3.0)
        raw.append(tort(np.column_stack([x, y, z])))
        smooth.append(tort(np.column_stack([xs, ys, z])))
        sig = float(
            np.sqrt(0.5 * (np.var(x - xs) + np.var(y - ys)))
        )  # per-coordinate residual SD (um)
        jit.append(sig)
        # Straight chord between the smoothed end points ...
        t = (z - z[0]) / (z[-1] - z[0])
        xl, yl = xs[0] + t * (xs[-1] - xs[0]), ys[0] + t * (ys[-1] - ys[0])
        # ... plus the track's own residual series (autocorrelation preserved)
        resid_only.append(tort(np.column_stack([xl + (x - xs), yl + (y - ys), z])))
        # ... or plus i.i.d. noise of the same SD (reference only; overstates the floor)
        iid.append(
            tort(
                np.column_stack(
                    [xl + RNG.normal(0, sig, x.size), yl + RNG.normal(0, sig, y.size), z]
                )
            )
        )
    raw, smooth, resid_only, iid, jit = map(np.asarray, (raw, smooth, resid_only, iid, jit))
    out["tortuosity"] = {
        "raw_mean": float(raw.mean()),
        "raw_p90": float(np.percentile(raw, 90)),
        "smoothed_mean": float(smooth.mean()),
        "smoothed_p90": float(np.percentile(smooth, 90)),
        "residual_on_straight_chord_mean": float(resid_only.mean()),
        "iid_noise_on_straight_chord_mean": float(iid.mean()),
        "residual_sd_um_median": float(np.median(jit)),
        "slice_spacing_um": 0.345,
        "note": "smoothing: Gaussian, sigma = 3 slices (1.0 um)",
    }

    # --- 2. Band period across the bands (2-D FFT of the pixel-level pitch field) --------
    z = np.load(SOM)
    pitch = np.degrees(np.arctan2(z["accum_dx_sm"].astype(float), 100.0))
    f = pitch - pitch.mean()
    win = np.outer(np.hanning(f.shape[0]), np.hanning(f.shape[1]))
    F = np.abs(np.fft.fftshift(np.fft.fft2(f * win)))
    ny, nx = f.shape
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=PX_UM))
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=PX_UM))
    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY)
    valid = (K > 1.0 / 200.0) & (K < 1.0 / 20.0)  # wavelengths 20-200 um
    iy, ix = np.unravel_index(np.argmax(np.where(valid, F, 0)), F.shape)
    lam = 1.0 / K[iy, ix]
    ang = float(np.degrees(np.arctan2(KY[iy, ix], KX[iy, ix])) % 180.0)
    # resolution: neighbouring frequency bins along the wave vector
    dk = np.hypot(
        1.0 / (nx * PX_UM) * np.cos(np.radians(ang)), 1.0 / (ny * PX_UM) * np.sin(np.radians(ang))
    )
    out["band_period_2dfft"] = {
        "wavelength_um": float(lam),
        "wavelength_range_um": [
            float(1.0 / (K[iy, ix] + dk)),
            float(1.0 / max(K[iy, ix] - dk, 1e-9)),
        ],
        "normal_direction_deg_from_x": ang,
        "field_um": [float(nx * PX_UM), float(ny * PX_UM)],
    }
    # 1-D spectra along columns (vertical transects, ~normal to near-horizontal bands)
    cols = np.linspace(50, nx - 50, 8, dtype=int)
    dom = []
    for c in cols:
        prof = pitch[:, max(0, c - 5) : c + 6].mean(axis=1)
        A = np.abs(np.fft.rfft((prof - prof.mean()) * np.hanning(prof.size)))
        fr = np.fft.rfftfreq(prof.size, d=PX_UM)
        m = (fr > 1 / 200) & (fr < 1 / 20)
        dom.append(float(1.0 / fr[m][np.argmax(A[m])]))
    out["band_period_vertical_transects_um"] = {
        "values": dom,
        "mean": float(np.mean(dom)),
        "sd": float(np.std(dom)),
    }
    # Sign-change spacing along vertical profiles (rows 40..ny-40 avoid the
    # zero-padded margin of the smoothed field). One period = two half-periods.
    per_col, halves = [], []
    for c in range(60, nx - 60, 20):
        prof = pitch[40 : ny - 40, c - 5 : c + 6].mean(axis=1)
        zc = np.where(np.diff(np.sign(prof - np.median(prof))) != 0)[0]
        if len(zc) >= 2:
            h = np.diff(zc) * PX_UM
            halves += h.tolist()
            per_col.append(2.0 * h.mean())
    out["band_period_sign_changes_um"] = {
        "n_columns": len(per_col),
        "period_mean": float(np.mean(per_col)),
        "period_sd": float(np.std(per_col)),
        "band_width_median": float(np.median(halves)),
        "band_width_mean": float(np.mean(halves)),
    }

    # --- 3. Principal fabric axis vs imaging axis --------------------------------------
    V = d[["dx_um", "dy_um", "dz_um"]].to_numpy(float)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    w, e = np.linalg.eigh(V.T @ V / len(V))
    e1 = e[:, np.argmax(w)]
    out["fabric_axis"] = {
        "lambda1_axis": e1.tolist(),
        "angle_to_z_deg": float(np.degrees(np.arccos(abs(e1[2])))),
        "mean_pitch_deg": float(d.pitch_deg.mean()),
        "mean_yaw_deg": float(d.yaw_deg.mean()),
    }
    # --- 4. Tracking fidelity -----------------------------------------------------------
    # The stored centrelines are smoothed, so snap events cannot be read back.
    # Instead: distance from each centreline point to the nearest segmented rod
    # centroid of the SAME slice (output_validation_100/slices), against the
    # same distance for uniformly random points (the chance baseline).
    from scipy.spatial import cKDTree

    X = np.array([np.fromstring(v, sep=";") for v in d.cx_um]) / PX_UM
    Y = np.array([np.fromstring(v, sep=";") for v in d.cy_um]) / PX_UM
    med, within2 = [], []
    for i in range(X.shape[1]):
        det = pd.read_csv(REPO / f"output_validation_100/slices/slice_{i + 1:03d}_rods.csv")
        dist, _ = cKDTree(det[["centroid_x", "centroid_y"]].to_numpy()).query(
            np.column_stack([X[:, i], Y[:, i]])
        )
        med.append(np.median(dist))
        within2.append(np.mean(dist < 2.0))
    det = pd.read_csv(REPO / "output_validation_100/slices/slice_051_rods.csv")[
        ["centroid_x", "centroid_y"]
    ].to_numpy()
    R = np.column_stack(
        [
            RNG.uniform(det[:, 0].min(), det[:, 0].max(), 20000),
            RNG.uniform(det[:, 1].min(), det[:, 1].max(), 20000),
        ]
    )
    dr, _ = cKDTree(det).query(R)
    med = np.asarray(med)
    out["tracking_fidelity"] = {
        "median_dist_px_all": float(med.mean()),
        "median_dist_um_all": float(med.mean() * PX_UM),
        "median_dist_px_first10_mid10_last10": [
            float(med[:10].mean()),
            float(med[45:55].mean()),
            float(med[-11:].mean()),
        ],
        "frac_within_2px": float(np.mean(within2)),
        "random_median_dist_px": float(np.median(dr)),
        "random_median_dist_um": float(np.median(dr) * PX_UM),
        "random_frac_within_2px": float(np.mean(dr < 2.0)),
    }

    # Per-slice mean signed pitch: the quantity map_twist() integrates over depth.
    morph = json.loads((REPO / "biomimetic_pipeline/runs/live_001/morphometrics.json").read_text())
    p = np.asarray(morph["depth_profiles"]["pitch_signed_deg"], float)
    out["pitch_depth_profile"] = {
        "mean_deg": float(p.mean()),
        "min_deg": float(p.min()),
        "max_deg": float(p.max()),
        "all_same_sign": bool(np.all(p < 0) or np.all(p > 0)),
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
