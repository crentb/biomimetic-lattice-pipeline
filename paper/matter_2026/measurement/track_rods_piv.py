#!/usr/bin/env python3
"""
track_rods_piv.py
=================
High-fidelity rod tracking using Particle Image Velocimetry (PIV) between
consecutive tomographic slices, following the approach of:

  Gao, Guillen, Grimm, Renteria et al.  Acta Biomaterialia 181 (2024) 263-271
  "High throughput automated characterization of enamel microstructure using
   synchrotron X-ray tomography and optical flow imaging"

Algorithm (Fig. 8 of the paper):
  Step 1: Seed initial positions from slice-1 segmentation (our ~2600 centroids)
  Step 2: For each consecutive slice pair i → i+1:
    a) Compute PIV displacement field (u,v) via windowed FFT cross-correlation
    b) Interpolate (u,v) at each rod's current position
    c) Advect: (x,y)_{i+1} = (x,y)_i + (u,v)
    [Optional] Snap correction: if a detected centroid from segmentation is
    within SNAP_RADIUS px of the advected position, use it as the ground truth.

Advantages over centroid-matching (track_rods_100.py):
  - Sub-pixel accuracy (FFT phase cross-correlation, upsample ×10 → 0.1 px)
  - No missed assignments when rods are dense or partially occluded
  - Captures collective band motion naturally
  - No gap-tolerance hacks needed

Outputs (output_piv/):
  track_centerlines_piv.parquet   — same schema as track_rods_100 output
  piv_fields/slice_NNN_uv.npz    — (u,v) displacement fields for inspection
  piv_summary.csv                 — per-slice stats
  piv_stats.png                   — yaw/pitch/tilt summary figure
"""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import savgol_filter
from skimage import io
from skimage.registration import phase_cross_correlation

# ─── CONFIG ──────────────────────────────────────────────────────────────────
STACK_DIR = "onehundred_image_stack"
SLICE_CSV_DIR = "output_validation_100/slices"
FIRST_SLICE = 1
LAST_SLICE = 101

# "forward" : seed slice 1  → track to slice 101 (original)
# "backward": seed slice 101 → track to slice 1  (more starting rods)
DIRECTION = "backward"

# Derived from direction
SEED_SLICE = LAST_SLICE if DIRECTION == "backward" else FIRST_SLICE
OUTPUT_DIR = "output_piv_back" if DIRECTION == "backward" else "output_piv"

# Reuse already-computed PIV fields (forward pass already saved them)
PIV_FIELDS_DIR = "output_piv/piv_fields"
RECOMPUTE_PIV = False  # set True to recompute from scratch

PIXEL_SIZE_UM = 20.0 / 58.0  # µm/px
Z_OFFSET_UM = 10.0  # depth of slice 1 from DEJ
Z_STEP_UM = PIXEL_SIZE_UM  # assumed isotropic

# PIV parameters
WIN_SIZE = 64  # interrogation window size (px) — covers ~4×4 rods
STRIDE = 32  # window step (px) — 50% overlap
UPSAMPLE = 10  # FFT upsampling → 0.1 px sub-pixel precision
MAX_DISP_PX = 20  # discard spurious PIV vectors larger than this (px)

# Optional snap correction: if segmented centroid is within this radius
# of the PIV-predicted position, snap to it (set 0 to disable)
SNAP_RADIUS_PX = 4

# Minimum track length to include in final output
MIN_TRACK_SLICES = 30

# Savitzky-Golay smoothing for angle computation
SAVGOL_WINDOW = 15
SAVGOL_ORDER = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)
piv_fields_dir = os.path.join(OUTPUT_DIR, "piv_fields")
os.makedirs(piv_fields_dir, exist_ok=True)

# ─── HELPERS ─────────────────────────────────────────────────────────────────


def load_image(path):
    raw = io.imread(path, as_gray=True).astype(np.float32)
    lo, hi = raw.min(), raw.max()
    return (raw - lo) / (hi - lo) if hi > lo else raw


def load_slice_csv(s):
    """Return DataFrame of detected rods for slice s (from validate_stack output)."""
    path = os.path.join(SLICE_CSV_DIR, f"slice_{s:03d}_rods.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["centroid_x", "centroid_y", "is_boundary"])


def compute_piv_field(img1, img2):
    """
    Compute a dense displacement field between img1 and img2 using windowed
    FFT cross-correlation (PIV).

    Returns
    -------
    cx_grid : 1D array of window-center x positions (px)
    cy_grid : 1D array of window-center y positions (px)
    U       : 2D array (ny, nx) of x-displacements (img2 - img1)
    V       : 2D array (ny, nx) of y-displacements (img2 - img1)
    """
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
                # Reject implausibly large displacements
                if abs(drow) > MAX_DISP_PX or abs(dcol) > MAX_DISP_PX:
                    drow, dcol = 0.0, 0.0
            except Exception:
                drow, dcol = 0.0, 0.0
            V[i, j] = float(drow)  # row = y
            U[i, j] = float(dcol)  # col = x

    return cx_centers, cy_centers, U, V


def make_interpolators(cx_grid, cy_grid, U, V):
    """Build 2D RegularGridInterpolators for u and v displacement."""
    interp_u = RegularGridInterpolator(
        (cy_grid, cx_grid), U, method="linear", bounds_error=False, fill_value=0.0
    )
    interp_v = RegularGridInterpolator(
        (cy_grid, cx_grid), V, method="linear", bounds_error=False, fill_value=0.0
    )
    return interp_u, interp_v


def smooth_path(xs, ys):
    """Savitzky-Golay smooth a (xs, ys) centroid sequence."""
    n = len(xs)
    wl = min(SAVGOL_WINDOW, n if n % 2 == 1 else n - 1)
    if wl < SAVGOL_ORDER + 2:
        return np.array(xs), np.array(ys)
    xs_sm = savgol_filter(xs, window_length=wl, polyorder=SAVGOL_ORDER)
    ys_sm = savgol_filter(ys, window_length=wl, polyorder=SAVGOL_ORDER)
    return xs_sm, ys_sm


def compute_angles(xs_um, ys_um, zs_um):
    """
    Compute pitch, yaw, tilt from a 3D centerline (smoothed, in µm).
    Returns scalar angles (endpoint-to-endpoint), plus fit versions.
    """
    dx = xs_um[-1] - xs_um[0]
    dy = ys_um[-1] - ys_um[0]
    dz = zs_um[-1] - zs_um[0]
    pitch = float(np.degrees(np.arctan2(dx, dz)))
    yaw = float(np.degrees(np.arctan2(dy, dz)))
    tilt = float(np.degrees(np.arctan2(np.sqrt(dx**2 + dy**2), dz)))

    # Fit via least-squares line through 3D points
    zc = zs_um - zs_um.mean()
    xc = xs_um - xs_um.mean()
    yc = ys_um - ys_um.mean()
    fit_dx = np.dot(zc, xc) / np.dot(zc, zc) if np.dot(zc, zc) > 0 else 0
    fit_dy = np.dot(zc, yc) / np.dot(zc, zc) if np.dot(zc, zc) > 0 else 0
    fit_pitch = float(np.degrees(np.arctan2(fit_dx, 1.0)))
    fit_yaw = float(np.degrees(np.arctan2(fit_dy, 1.0)))
    fit_tilt = float(np.degrees(np.arctan2(np.sqrt(fit_dx**2 + fit_dy**2), 1.0)))
    return pitch, yaw, tilt, fit_pitch, fit_yaw, fit_tilt


# ─── STEP 1: SEED FROM SEED_SLICE SEGMENTATION ───────────────────────────────
print(f"Step 1: Loading slice-{SEED_SLICE} segmentation (DIRECTION={DIRECTION}) …")
seed_df = load_slice_csv(SEED_SLICE)
seeds = seed_df[~seed_df["is_boundary"]].copy()
print(f"  {len(seeds)} interior rod seeds from slice {SEED_SLICE}")

seed_z_um = Z_OFFSET_UM + (SEED_SLICE - 1) * Z_STEP_UM

# Initialise tracks: list of dicts, one per rod
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

# ─── STEP 2: PIV ADVECTION THROUGH SLICES ────────────────────────────────────
if DIRECTION == "forward":
    slice_sequence = range(FIRST_SLICE, LAST_SLICE)  # 1,2,…,100
else:
    slice_sequence = range(LAST_SLICE - 1, FIRST_SLICE - 1, -1)  # 100,99,…,1

print(f"Step 2: PIV advection ({DIRECTION}) …")
summary_rows = []

for s in slice_sequence:
    # For forward:  pair is s→s+1,  snap target is s+1
    # For backward: pair is s→s+1 (same stored field), but we NEGATE and target is s
    s_next = s + 1
    s_target = s_next if DIRECTION == "forward" else s
    sign = 1.0 if DIRECTION == "forward" else -1.0

    img1_path = os.path.join(STACK_DIR, f"lion_axial{s}.png")
    img2_path = os.path.join(STACK_DIR, f"lion_axial{s_next}.png")
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        continue

    # Load or recompute PIV field
    npz_path = os.path.join(PIV_FIELDS_DIR, f"slice_{s:03d}_uv.npz")
    if not RECOMPUTE_PIV and os.path.exists(npz_path):
        d = np.load(npz_path)
        cx_grid, cy_grid, U, V = d["cx"], d["cy"], d["U"], d["V"]
    else:
        img1 = load_image(img1_path)
        img2 = load_image(img2_path)
        cx_grid, cy_grid, U, V = compute_piv_field(img1, img2)
        out_npz = os.path.join(piv_fields_dir, f"slice_{s:03d}_uv.npz")
        np.savez_compressed(out_npz, cx=cx_grid, cy=cy_grid, U=U, V=V)

    interp_u, interp_v = make_interpolators(cx_grid, cy_grid, U * sign, V * sign)

    # Load detected centroids for snap correction (at the target slice)
    snap_pts = None
    if SNAP_RADIUS_PX > 0:
        det = load_slice_csv(s_target)
        if len(det) > 0:
            snap_pts = det[["centroid_x", "centroid_y"]].values.astype(np.float32)

    # Advect all active tracks
    img_ref = load_image(os.path.join(STACK_DIR, f"lion_axial{s_target}.png"))
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

        # Optional snap correction
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

# ─── STEP 3: BUILD TRACK RECORDS ─────────────────────────────────────────────
print("Step 3: Building track records …")

records = []
for tid, t in enumerate(tracks, start=1):
    n = len(t["xs"])
    if n < MIN_TRACK_SLICES:
        continue

    # Backward tracks are stored deepest→shallowest; reverse so z increases
    if DIRECTION == "backward":
        xs_arr = np.array(t["xs"])[::-1]
        ys_arr = np.array(t["ys"])[::-1]
        zs_arr = np.array(t["zs_um"])[::-1]
    else:
        xs_arr = np.array(t["xs"])
        ys_arr = np.array(t["ys"])
        zs_arr = np.array(t["zs_um"])

    xs_px = xs_arr
    ys_px = ys_arr
    zs_um = zs_arr

    xs_um = xs_px * PIXEL_SIZE_UM
    ys_um = ys_px * PIXEL_SIZE_UM

    # Smooth before angle computation
    xs_sm, ys_sm = smooth_path(xs_um, ys_um)

    pitch, yaw, tilt, fy, fp, ft = compute_angles(xs_sm, ys_sm, zs_um)

    dx_um = xs_sm[-1] - xs_sm[0]
    dy_um = ys_sm[-1] - ys_sm[0]
    dz_um = zs_um[-1] - zs_um[0]

    records.append(
        {
            "track_id": tid,
            "n_slices": n,
            "start_slice": int(t["slices"][0]),
            "end_slice": int(t["slices"][-1]),
            "is_complete": (min(t["slices"]) == FIRST_SLICE and max(t["slices"]) == LAST_SLICE),
            "start_x_px": round(float(xs_px[0]), 2),
            "start_y_px": round(float(ys_px[0]), 2),
            "dx_um": round(dx_um, 4),
            "dy_um": round(dy_um, 4),
            "dz_um": round(dz_um, 4),
            "pitch_deg": round(pitch, 4),
            "yaw_deg": round(yaw, 4),
            "tilt_deg": round(tilt, 4),
            "fit_pitch_deg": round(fy, 4),
            "fit_yaw_deg": round(fp, 4),
            "fit_tilt_deg": round(ft, 4),
            # Semicolon-separated centerline strings (same format as track_rods_100)
            "cx_um": ";".join(f"{v:.3f}" for v in xs_sm),
            "cy_um": ";".join(f"{v:.3f}" for v in ys_sm),
            "cz_um": ";".join(f"{v:.3f}" for v in zs_um),
        }
    )

df_out = pd.DataFrame(records)
parquet_path = os.path.join(OUTPUT_DIR, "track_centerlines_piv.parquet")
df_out.to_parquet(parquet_path, index=False)
print(f"  {len(df_out)} tracks ≥ {MIN_TRACK_SLICES} slices saved → {parquet_path}")

# ─── STEP 4: STATS FIGURE ────────────────────────────────────────────────────
print("Step 4: Generating stats figure …")

complete = df_out[df_out["is_complete"]]
print(f"  Complete tracks (slice {FIRST_SLICE}–{LAST_SLICE}): {len(complete)}")


def stat_block(vals, label):
    v = np.array(vals)
    return (
        f"{label}: mean={v.mean():.2f}°  median={np.median(v):.2f}°  "
        f"std={v.std():.2f}°  IQR={np.percentile(v,75)-np.percentile(v,25):.2f}°  "
        f"range=[{v.min():.1f}, {v.max():.1f}]"
    )


if len(complete) > 0:
    print("  " + stat_block(complete["pitch_deg"], "Pitch  "))
    print("  " + stat_block(complete["yaw_deg"], "Yaw"))
    print("  " + stat_block(complete["tilt_deg"], "Tilt "))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
pitch_cmap = "RdBu_r"
pitch_norm = mcolors.TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60)

# Row 0: histograms
for col, (col_name, label, color) in enumerate(
    [
        ("pitch_deg", "Pitch (°)", "steelblue"),
        ("yaw_deg", "Yaw (°)", "seagreen"),
        ("tilt_deg", "Tilt (°)", "darkorange"),
    ]
):
    ax = axes[0, col]
    vals = df_out[col_name].dropna().values
    ax.hist(vals, bins=60, color=color, alpha=0.75, edgecolor="none")
    ax.axvline(np.mean(vals), color="k", lw=1.5, ls="--", label=f"mean={np.mean(vals):.1f}°")
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.set_title(f"{label} distribution\n(n={len(vals)}, PIV tracked)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Row 1: spatial pitch map (scatter + smoothed)
from collections import defaultdict

from scipy.ndimage import gaussian_filter

IMG_H, IMG_W = 792, 850
GRID_STEP_PX = 10
gx = np.arange(0, IMG_W, GRID_STEP_PX)
gy = np.arange(0, IMG_H, GRID_STEP_PX)
nx, ny_g = len(gx), len(gy)
pitch_acc = defaultdict(list)
for _, row in df_out.iterrows():
    ix = int(np.clip(row["start_x_px"] / GRID_STEP_PX, 0, nx - 1))
    iy = int(np.clip(row["start_y_px"] / GRID_STEP_PX, 0, ny_g - 1))
    pitch_acc[(iy, ix)].append(row["pitch_deg"])
pitch_grid = np.full((ny_g, nx), np.nan)
for (iy, ix), vals in pitch_acc.items():
    pitch_grid[iy, ix] = np.mean(vals)
mask = ~np.isnan(pitch_grid)
filled = np.where(mask, pitch_grid, 0.0)
sm = gaussian_filter(filled, sigma=2.5) / np.where(
    gaussian_filter(mask.astype(float), sigma=2.5) > 0.05,
    gaussian_filter(mask.astype(float), sigma=2.5),
    1.0,
)
sm[~mask] = np.nan

extent = [0, IMG_W * PIXEL_SIZE_UM, IMG_H * PIXEL_SIZE_UM, 0]
ax = axes[1, 0]
sc = ax.scatter(
    df_out["start_x_px"] * PIXEL_SIZE_UM,
    df_out["start_y_px"] * PIXEL_SIZE_UM,
    c=df_out["pitch_deg"],
    cmap=pitch_cmap,
    norm=pitch_norm,
    s=6,
    linewidths=0,
)
plt.colorbar(sc, ax=ax, label="Pitch (°)")
ax.set_title("PIV Pitch — rod scatter")
ax.invert_yaxis()
ax.set_aspect("equal")
ax.set_xlabel("X (µm)")
ax.set_ylabel("Y (µm)")

ax = axes[1, 1]
im = ax.imshow(
    sm, extent=extent, cmap=pitch_cmap, norm=pitch_norm, interpolation="bilinear", aspect="equal"
)
plt.colorbar(im, ax=ax, label="Mean Pitch (°)")
ax.set_title("PIV Pitch — smoothed HSB band map")
ax.set_xlabel("X (µm)")
ax.set_ylabel("Y (µm)")

# Median displacement vs slice
ax = axes[1, 2]
sdf = pd.read_csv(os.path.join(OUTPUT_DIR, "piv_summary.csv"))
ax.plot(sdf["slice"], sdf["med_disp_px"], "o-", color="purple", ms=3)
ax.set_xlabel("Slice")
ax.set_ylabel("Median PIV displacement (px)")
ax.set_title("PIV displacement magnitude vs depth\n(quality metric)")
ax.grid(True, alpha=0.3)

fig.suptitle(
    f"PIV Hybrid Tracker — yaw/pitch/tilt summary\n"
    f"(n={len(df_out)} tracks ≥{MIN_TRACK_SLICES} slices,  "
    f"WIN={WIN_SIZE}px STRIDE={STRIDE}px UPSAMPLE={UPSAMPLE}×)",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout()
stats_path = os.path.join(OUTPUT_DIR, "piv_stats.png")
fig.savefig(stats_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {stats_path}")

# ─── SUMMARY ─────────────────────────────────────────────────────────────────
print("\n── PIV Tracking Complete ────────────────────────────────────────")
print(f"  Seeds (slice 1, interior):  {len(tracks)}")
print(f"  Tracks ≥ {MIN_TRACK_SLICES} slices:         {len(df_out)}")
print(f"  Complete tracks:            {len(complete)}")
if len(complete) > 0:
    print(
        f"  Pitch    mean ± std:  {complete['pitch_deg'].mean():.2f}° ± {complete['pitch_deg'].std():.2f}°"
    )
    print(
        f"  Yaw  mean ± std:  {complete['yaw_deg'].mean():.2f}° ± {complete['yaw_deg'].std():.2f}°"
    )
    print(
        f"  Tilt   mean ± std:  {complete['tilt_deg'].mean():.2f}° ± {complete['tilt_deg'].std():.2f}°"
    )
print("  PIV fields saved:           output_piv/piv_fields/")
print(f"  Parquet:                    {parquet_path}")
print(f"  Stats figure:               {stats_path}")
print("────────────────────────────────────────────────────────────────")
