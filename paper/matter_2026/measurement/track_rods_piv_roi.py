#!/usr/bin/env python3
"""
track_rods_piv_roi.py
=====================
PIV-based rod tracking on the deep-learning ROI segmentation stack
(roi_imagestack_100/axialroi{N}.png).

Uses the green channel of the ROI images for cross-correlation — the high
contrast between bright-green rod interiors and dark interprismatic matrix
gives sharper PIV signals than the original CT grayscale.

Algorithm identical to track_rods_piv.py (backward pass, seed from slice 101).

Outputs → output_piv_roi_back/
  track_centerlines_piv.parquet
  piv_fields/slice_NNN_uv.npz
  piv_summary.csv
"""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import savgol_filter
from skimage import io
from skimage.registration import phase_cross_correlation

# ─── CONFIG ───────────────────────────────────────────────────────────────────
STACK_DIR = "roi_imagestack_100"
SLICE_CSV_DIR = "output_validation_roi/slices"
FIRST_SLICE = 1
LAST_SLICE = 101
DIRECTION = "backward"

SEED_SLICE = LAST_SLICE if DIRECTION == "backward" else FIRST_SLICE
OUTPUT_DIR = "output_piv_roi_back" if DIRECTION == "backward" else "output_piv_roi"

# Always recompute PIV from the ROI green-channel images
RECOMPUTE_PIV = False  # PIV fields cached in PIV_FIELDS_DIR; set True to recompute
PIV_FIELDS_DIR = os.path.join(OUTPUT_DIR, "piv_fields")

PIXEL_SIZE_UM = 20.0 / 58.0
Z_OFFSET_UM = 10.0
Z_STEP_UM = PIXEL_SIZE_UM

# PIV parameters (same as original)
WIN_SIZE = 64
STRIDE = 32
UPSAMPLE = 10
MAX_DISP_PX = 20

SNAP_RADIUS_PX = 4
MIN_TRACK_SLICES = 30

SAVGOL_WINDOW = 15
SAVGOL_ORDER = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PIV_FIELDS_DIR, exist_ok=True)


# ─── HELPERS ──────────────────────────────────────────────────────────────────


def load_image(path):
    """Load ROI image and return the green channel as float32 in [0, 1]."""
    img = io.imread(path)  # uint8 RGB (H, W, 3)
    green = img[:, :, 1].astype(np.float32) / 255.0
    return green


def load_slice_csv(s):
    path = os.path.join(SLICE_CSV_DIR, f"slice_{s:03d}_rods.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["centroid_x", "centroid_y", "is_boundary"])


def compute_piv_field(img1, img2):
    H, W = img1.shape
    half = WIN_SIZE // 2
    cy_centers = np.arange(half, H - half + 1, STRIDE)
    cx_centers = np.arange(half, W - half + 1, STRIDE)
    ny, nx = len(cy_centers), len(cx_centers)
    U = np.zeros((ny, nx), dtype=np.float32)
    V = np.zeros((ny, nx), dtype=np.float32)
    for i, cy in enumerate(cy_centers):
        for j, cx_c in enumerate(cx_centers):
            w1 = img1[cy - half : cy + half, cx_c - half : cx_c + half]
            w2 = img2[cy - half : cy + half, cx_c - half : cx_c + half]
            try:
                shift, _, _ = phase_cross_correlation(
                    w1, w2, upsample_factor=UPSAMPLE, normalization=None
                )
                drow, dcol = shift
                if abs(drow) > MAX_DISP_PX or abs(dcol) > MAX_DISP_PX:
                    drow, dcol = 0.0, 0.0
            except Exception:
                drow, dcol = 0.0, 0.0
            V[i, j] = float(drow)
            U[i, j] = float(dcol)
    return cx_centers, cy_centers, U, V


def make_interpolators(cx_grid, cy_grid, U, V):
    interp_u = RegularGridInterpolator(
        (cy_grid, cx_grid), U, method="linear", bounds_error=False, fill_value=0.0
    )
    interp_v = RegularGridInterpolator(
        (cy_grid, cx_grid), V, method="linear", bounds_error=False, fill_value=0.0
    )
    return interp_u, interp_v


def smooth_path(xs, ys):
    n = len(xs)
    wl = min(SAVGOL_WINDOW, n if n % 2 == 1 else n - 1)
    if wl < SAVGOL_ORDER + 2:
        return np.array(xs), np.array(ys)
    return (
        savgol_filter(xs, window_length=wl, polyorder=SAVGOL_ORDER),
        savgol_filter(ys, window_length=wl, polyorder=SAVGOL_ORDER),
    )


def compute_angles(xs_um, ys_um, zs_um):
    dx = xs_um[-1] - xs_um[0]
    dy = ys_um[-1] - ys_um[0]
    dz = zs_um[-1] - zs_um[0]
    pitch = float(np.degrees(np.arctan2(dx, dz)))
    yaw = float(np.degrees(np.arctan2(dy, dz)))
    tilt = float(np.degrees(np.arctan2(np.sqrt(dx**2 + dy**2), dz)))
    zc = zs_um - zs_um.mean()
    xc = xs_um - xs_um.mean()
    yc = ys_um - ys_um.mean()
    fit_dx = np.dot(zc, xc) / np.dot(zc, zc) if np.dot(zc, zc) > 0 else 0
    fit_dy = np.dot(zc, yc) / np.dot(zc, zc) if np.dot(zc, zc) > 0 else 0
    fit_pitch = float(np.degrees(np.arctan2(fit_dx, 1.0)))
    fit_yaw = float(np.degrees(np.arctan2(fit_dy, 1.0)))
    fit_tilt = float(np.degrees(np.arctan2(np.sqrt(fit_dx**2 + fit_dy**2), 1.0)))
    return pitch, yaw, tilt, fit_pitch, fit_yaw, fit_tilt


# ─── STEP 1: SEED FROM SEED_SLICE ─────────────────────────────────────────────
print(f"Step 1: Loading slice-{SEED_SLICE} ROI segmentation …")
seed_df = load_slice_csv(SEED_SLICE)
seeds = seed_df[~seed_df["is_boundary"]].copy()
print(f"  {len(seeds)} interior rod seeds")

seed_z_um = Z_OFFSET_UM + (SEED_SLICE - 1) * Z_STEP_UM

tracks = []
for _, row in seeds.iterrows():
    tracks.append(
        {
            "x": float(row["centroid_x"]),
            "y": float(row["centroid_y"]),
            "xs": [float(row["centroid_x"])],
            "ys": [float(row["centroid_y"])],
            "zs_um": [seed_z_um],
            "slices": [SEED_SLICE],
            "active": True,
        }
    )

# ─── STEP 2: PIV ADVECTION ────────────────────────────────────────────────────
if DIRECTION == "forward":
    slice_sequence = range(FIRST_SLICE, LAST_SLICE)
else:
    slice_sequence = range(LAST_SLICE - 1, FIRST_SLICE - 1, -1)

print(f"Step 2: PIV advection ({DIRECTION}, {len(tracks)} seeds) …")
summary_rows = []

for s in slice_sequence:
    s_next = s + 1
    s_target = s_next if DIRECTION == "forward" else s
    sign = 1.0 if DIRECTION == "forward" else -1.0

    img1_path = os.path.join(STACK_DIR, f"axialroi{s}.png")
    img2_path = os.path.join(STACK_DIR, f"axialroi{s_next}.png")
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        continue

    # Load or compute PIV field
    npz_path = os.path.join(PIV_FIELDS_DIR, f"slice_{s:03d}_uv.npz")
    if not RECOMPUTE_PIV and os.path.exists(npz_path):
        d = np.load(npz_path)
        cx_grid, cy_grid, U, V = d["cx"], d["cy"], d["U"], d["V"]
    else:
        img1 = load_image(img1_path)
        img2 = load_image(img2_path)
        cx_grid, cy_grid, U, V = compute_piv_field(img1, img2)
        np.savez_compressed(npz_path, cx=cx_grid, cy=cy_grid, U=U, V=V)

    interp_u, interp_v = make_interpolators(cx_grid, cy_grid, U * sign, V * sign)

    # Snap correction centroids for this target slice
    snap_pts = None
    if SNAP_RADIUS_PX > 0:
        det = load_slice_csv(s_target)
        if len(det) > 0:
            snap_pts = det[["centroid_x", "centroid_y"]].values.astype(np.float32)

    img_ref = load_image(os.path.join(STACK_DIR, f"axialroi{s_target}.png"))
    H, W = img_ref.shape
    active_count = 0

    for t in tracks:
        if not t["active"]:
            continue
        x, y = t["x"], t["y"]
        pt = np.array([[y, x]])
        du = float(interp_u(pt)[0])
        dv = float(interp_v(pt)[0])
        x_new = x + du
        y_new = y + dv

        if snap_pts is not None and len(snap_pts) > 0:
            dists = np.hypot(snap_pts[:, 0] - x_new, snap_pts[:, 1] - y_new)
            best = int(np.argmin(dists))
            if dists[best] <= SNAP_RADIUS_PX:
                x_new = float(snap_pts[best, 0])
                y_new = float(snap_pts[best, 1])

        if x_new < 0 or x_new >= W or y_new < 0 or y_new >= H:
            t["active"] = False
            continue

        t["x"] = x_new
        t["y"] = y_new
        t["xs"].append(x_new)
        t["ys"].append(y_new)
        t["zs_um"].append(Z_OFFSET_UM + (s_target - 1) * Z_STEP_UM)
        t["slices"].append(s_target)
        active_count += 1

    med_disp = float(np.median(np.hypot(U.ravel(), V.ravel())))
    summary_rows.append(
        {
            "slice": s,
            "active_tracks": active_count,
            "piv_nx": len(cx_grid),
            "piv_ny": len(cy_grid),
            "med_disp_px": round(med_disp, 3),
        }
    )

    if s % 10 == 0:
        print(f"  Slice {s:3d}  |  active={active_count}  med_disp={med_disp:.3f} px")

pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, "piv_summary.csv"), index=False)

# ─── STEP 3: BUILD TRACK RECORDS ──────────────────────────────────────────────
print("Step 3: Building track records …")
records = []
for tid, t in enumerate(tracks, start=1):
    n = len(t["xs"])
    if n < MIN_TRACK_SLICES:
        continue
    if DIRECTION == "backward":
        xs_arr = np.array(t["xs"])[::-1]
        ys_arr = np.array(t["ys"])[::-1]
        zs_arr = np.array(t["zs_um"])[::-1]
    else:
        xs_arr = np.array(t["xs"])
        ys_arr = np.array(t["ys"])
        zs_arr = np.array(t["zs_um"])

    xs_um = xs_arr * PIXEL_SIZE_UM
    ys_um = ys_arr * PIXEL_SIZE_UM
    xs_sm, ys_sm = smooth_path(xs_um, ys_um)

    pitch, yaw, tilt, fy, fp, ft = compute_angles(xs_sm, ys_sm, zs_arr)

    records.append(
        {
            "track_id": tid,
            "n_slices": n,
            "start_slice": int(t["slices"][0]),
            "end_slice": int(t["slices"][-1]),
            "is_complete": (min(t["slices"]) == FIRST_SLICE and max(t["slices"]) == LAST_SLICE),
            "start_x_px": round(float(xs_arr[0]), 2),
            "start_y_px": round(float(ys_arr[0]), 2),
            "dx_um": round(float(xs_sm[-1] - xs_sm[0]), 4),
            "dy_um": round(float(ys_sm[-1] - ys_sm[0]), 4),
            "dz_um": round(float(zs_arr[-1] - zs_arr[0]), 4),
            "pitch_deg": round(pitch, 4),
            "yaw_deg": round(yaw, 4),
            "tilt_deg": round(tilt, 4),
            "fit_pitch_deg": round(fy, 4),
            "fit_yaw_deg": round(fp, 4),
            "fit_tilt_deg": round(ft, 4),
            "cx_um": ";".join(f"{v:.3f}" for v in xs_sm),
            "cy_um": ";".join(f"{v:.3f}" for v in ys_sm),
            "cz_um": ";".join(f"{v:.3f}" for v in zs_arr),
        }
    )

df_out = pd.DataFrame(records)
parquet_path = os.path.join(OUTPUT_DIR, "track_centerlines_piv.parquet")
df_out.to_parquet(parquet_path, index=False)
print(f"  {len(df_out)} tracks ≥ {MIN_TRACK_SLICES} slices → {parquet_path}")
print("Done.")
