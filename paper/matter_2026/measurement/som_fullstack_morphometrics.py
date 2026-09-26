#!/usr/bin/env python3
"""
SOM Full-Stack — Advanced Visualization & Morphometric Extraction
==================================================================
Builds on som_fullstack_displacement.py results. Adds:

Visualizations:
  1. Displacement streamlines colored by cluster (ameloblast migration paths)
  2. Band boundary map with angular mismatch (fracture mechanics)
  3. Polar rose diagrams per cluster (orientation distributions)
  4. Band transect profiles + FFT periodicity analysis
  5. Crack deflection potential map (displacement direction gradient)
  6. Publication-ready composite figure

Morphometrics (for CAD model input):
  - Band width statistics per cluster
  - Displacement angle distributions per band
  - Boundary angular mismatch (parazone/diazone transitions)
  - Band periodicity (FFT dominant wavelength)
  - Transition zone sharpness

Input:  Recomputes from same data as som_fullstack_displacement.py
Output: som_approach/output_som_morphometrics/
"""

import json
import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from minisom import MiniSom
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation, gaussian_filter, sobel, uniform_filter
from scipy.signal import fftconvolve
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from skimage.filters import gabor_kernel
from skimage.segmentation import find_boundaries
from sklearn.cluster import KMeans

# ── Publication Style ────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 200,
        "axes.grid": False,
    }
)


def _style_cb(cb, label="", fontsize=10):
    """Apply publication colorbar style."""
    if label:
        cb.set_label(label, fontsize=fontsize, fontweight="bold", rotation=270, labelpad=14)
    cb.ax.tick_params(labelsize=8, length=2, pad=2)
    for t in cb.ax.get_yticklabels():
        t.set_fontweight("bold")


def _style_spines(ax):
    """Hide top/right spines, light gray left/bottom."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")


# ── Config ───────────────────────────────────────────────────────────
STACK_DIR = os.path.join(os.path.dirname(__file__), "..", "roi_imagestack_100")
PIV_DIR = os.path.join(os.path.dirname(__file__), "..", "output_piv_roi_back", "piv_fields")
OUT_DIR = os.path.join(os.path.dirname(__file__), "output_som_morphometrics")
PIXEL_SIZE_UM = 20.0 / 58.0  # from PIV script

FIRST_SLICE = 1
LAST_SLICE = 101

# SOM config (same as fullstack)
SOM_X, SOM_Y = 15, 15
SOM_ITERATIONS = 80_000
SAMPLE_FRAC = 0.03
N_CLUSTERS = 4
SEED = 42

DENSITY_SIGMAS = [8, 16, 32]
ST_SIGMA = 8
GABOR_FREQS = [0.05, 0.1, 0.15]
GABOR_THETAS = np.linspace(0, np.pi, 6, endpoint=False)
DISP_SMOOTH_SIGMA = 8
TEXTURE_SAMPLE_SLICES = [1, 13, 26, 38, 51, 63, 76, 88, 101]

# Transect config
N_TRANSECTS = 8
TRANSECT_SMOOTH = 5  # pixels to average perpendicular to transect

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)

# Cluster colors + names
COLORS = ["#2c3e50", "#e74c3c", "#3498db", "#2ecc71"]
CMAP_BANDS = ListedColormap(COLORS[:N_CLUSTERS])
CLUSTER_NAMES = [f"Band {i}" for i in range(N_CLUSTERS)]


# ══════════════════════════════════════════════════════════════════════
# RECOMPUTE (same as fullstack script — needed for displacement fields)
# ══════════════════════════════════════════════════════════════════════


def load_rod_mask(slice_idx):
    path = os.path.join(STACK_DIR, f"axialroi{slice_idx}.png")
    img = np.array(Image.open(path))
    green = img[:, :, 1].astype(np.float64) / 255.0
    return (green > 0.5).astype(np.float64)


def extract_texture_features(rod_mask):
    features = []
    for sigma in DENSITY_SIGMAS:
        features.append(gaussian_filter(rod_mask, sigma=sigma))
    st_elements = structure_tensor(rod_mask, sigma=ST_SIGMA)
    eigvals = structure_tensor_eigenvalues(st_elements)
    lam1, lam2 = eigvals[0], eigvals[1]
    coherence = np.where((lam1 + lam2) > 1e-10, (lam1 - lam2) / (lam1 + lam2), 0.0)
    features.append(coherence)
    Axx, Axy, Ayy = st_elements
    orientation = 0.5 * np.arctan2(2 * Axy, Ayy - Axx)
    features.append(np.sin(2 * orientation))
    features.append(np.cos(2 * orientation))
    for freq in GABOR_FREQS:
        for theta in GABOR_THETAS:
            kernel = gabor_kernel(freq, theta=theta, sigma_x=3, sigma_y=3)
            resp_r = fftconvolve(rod_mask, kernel.real, mode="same")
            resp_i = fftconvolve(rod_mask, kernel.imag, mode="same")
            features.append(np.sqrt(resp_r**2 + resp_i**2))
    for win in [16, 32]:
        local_mean = uniform_filter(rod_mask, size=win)
        local_sq_mean = uniform_filter(rod_mask**2, size=win)
        features.append(np.clip(local_sq_mean - local_mean**2, 0, None))
    return np.stack(features, axis=-1)


print("=" * 60)
print("SOM Morphometrics & Advanced Visualization")
print("=" * 60)

mask0 = load_rod_mask(FIRST_SLICE)
H, W = mask0.shape
ref_mask = load_rod_mask(51)

# --- Accumulate displacement ---
print("\nAccumulating displacement fields...")
yy, xx = np.mgrid[0:H, 0:W]
coords = np.stack([yy.ravel(), xx.ravel()], axis=-1).astype(np.float64)
accum_dx = np.zeros(H * W, dtype=np.float64)
accum_dy = np.zeros(H * W, dtype=np.float64)
dx_per_slice = np.zeros((100, H * W), dtype=np.float32)
dy_per_slice = np.zeros((100, H * W), dtype=np.float32)

for i, s in enumerate(range(FIRST_SLICE, LAST_SLICE)):
    npz_path = os.path.join(PIV_DIR, f"slice_{s:03d}_uv.npz")
    d = np.load(npz_path)
    cx, cy, U, V = d["cx"], d["cy"], d["U"], d["V"]
    interp_u = RegularGridInterpolator(
        (cy.astype(np.float64), cx.astype(np.float64)),
        U.astype(np.float64),
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    interp_v = RegularGridInterpolator(
        (cy.astype(np.float64), cx.astype(np.float64)),
        V.astype(np.float64),
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    current = coords.copy()
    current[:, 1] += accum_dx
    current[:, 0] += accum_dy
    du = interp_u(current)
    dv = interp_v(current)
    dx_per_slice[i] = du.astype(np.float32)
    dy_per_slice[i] = dv.astype(np.float32)
    accum_dx += du
    accum_dy += dv

accum_dx_img = accum_dx.reshape(H, W)
accum_dy_img = accum_dy.reshape(H, W)
dx_var = np.var(dx_per_slice, axis=0).reshape(H, W)
dy_var = np.var(dy_per_slice, axis=0).reshape(H, W)
del dx_per_slice, dy_per_slice

accum_dx_sm = gaussian_filter(accum_dx_img, sigma=DISP_SMOOTH_SIGMA)
accum_dy_sm = gaussian_filter(accum_dy_img, sigma=DISP_SMOOTH_SIGMA)
accum_mag_sm = np.sqrt(accum_dx_sm**2 + accum_dy_sm**2)
accum_dir_sm = np.arctan2(accum_dy_sm, accum_dx_sm)
dx_var_sm = gaussian_filter(dx_var, sigma=DISP_SMOOTH_SIGMA)
dy_var_sm = gaussian_filter(dy_var, sigma=DISP_SMOOTH_SIGMA)
print(f"  dx range: {accum_dx_img.min():.1f} to {accum_dx_img.max():.1f} px")
print(f"  dy range: {accum_dy_img.min():.1f} to {accum_dy_img.max():.1f} px")

# ── Convert displacement → pitch / yaw / tilt angles ────────────────
N_TRANSITIONS = LAST_SLICE - FIRST_SLICE  # 100 slice transitions = dz in pixels
pitch_map = np.degrees(np.arctan2(accum_dx_sm, N_TRANSITIONS))  # atan2(dx, dz)
yaw_map = np.degrees(np.arctan2(accum_dy_sm, N_TRANSITIONS))  # atan2(dy, dz)
tilt_map = np.degrees(np.arctan2(accum_mag_sm, N_TRANSITIONS))  # total deviation
print(f"  pitch range: {pitch_map.min():.1f} to {pitch_map.max():.1f}°")
print(f"  yaw   range: {yaw_map.min():.1f} to {yaw_map.max():.1f}°")
print(f"  tilt  range: {tilt_map.min():.1f} to {tilt_map.max():.1f}°")


def orientation_tensor(V_sub):
    """3x3 orientation tensor from unit direction vectors. Returns (evals desc, evecs)."""
    T = (V_sub.T @ V_sub) / len(V_sub)
    vals, vecs = np.linalg.eigh(T)
    idx = np.argsort(vals)[::-1]
    return vals[idx], vecs[:, idx].T


def fabric_metrics(evals):
    """Woodcock strength C, shape K, and normalised eigenvalues."""
    e1, e2, e3 = evals
    e3c = max(float(e3), 1e-12)
    e2c = max(float(e2), 1e-12)
    C = np.log(float(e1) / e3c)
    K = np.log(float(e1) / e2c) / np.log(e2c / e3c) if e2c > e3c else float("inf")
    return C, K, evals / evals.sum()


# --- Texture features ---
print("\nComputing texture features...")
texture_accum = None
for s in TEXTURE_SAMPLE_SLICES:
    mask = load_rod_mask(s)
    tex = extract_texture_features(mask)
    texture_accum = tex.astype(np.float64) if texture_accum is None else texture_accum + tex
texture_avg = texture_accum / len(TEXTURE_SAMPLE_SLICES)
N_TEX = texture_avg.shape[-1]

# --- Combined features + SOM ---
disp_features = np.stack(
    [
        accum_dx_sm,
        accum_dy_sm,
        np.sin(accum_dir_sm),
        np.cos(accum_dir_sm),
        accum_mag_sm,
        np.sqrt(dx_var_sm + dy_var_sm),
    ],
    axis=-1,
)
N_DISP = disp_features.shape[-1]
combined = np.concatenate([texture_avg, disp_features], axis=-1)
N_FEAT = combined.shape[-1]

flat_features = combined.reshape(-1, N_FEAT)
feat_min = flat_features.min(axis=0)
feat_max = flat_features.max(axis=0)
feat_range = np.where(feat_max - feat_min > 1e-10, feat_max - feat_min, 1.0)
flat_norm = (flat_features - feat_min) / feat_range

print(f"\nTraining SOM ({SOM_X}×{SOM_Y}, {N_FEAT} features)...")
n_sample = int(flat_norm.shape[0] * SAMPLE_FRAC)
sample_idx = np.random.choice(flat_norm.shape[0], size=n_sample, replace=False)
train_data = flat_norm[sample_idx]

som = MiniSom(
    SOM_X,
    SOM_Y,
    N_FEAT,
    sigma=2.5,
    learning_rate=0.5,
    neighborhood_function="gaussian",
    random_seed=SEED,
)
som.pca_weights_init(train_data)
som.train(train_data, SOM_ITERATIONS, verbose=True)

weights = som.get_weights().reshape(-1, N_FEAT)
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
node_labels = kmeans.fit_predict(weights)

print("Mapping pixels to BMU...")
bmu_indices = np.array([som.winner(x) for x in flat_norm])
bmu_flat = bmu_indices[:, 0] * SOM_Y + bmu_indices[:, 1]
cluster_map_raw = node_labels[bmu_flat].reshape(H, W)

# Sort cluster labels by mean X displacement for consistent colors across scripts
mean_dx_per_label = [accum_dx_sm[cluster_map_raw == c].mean() for c in range(N_CLUSTERS)]
sorted_order = np.argsort(mean_dx_per_label)
label_remap = np.zeros(N_CLUSTERS, dtype=int)
for new_label, old_label in enumerate(sorted_order):
    label_remap[old_label] = new_label
cluster_map = label_remap[cluster_map_raw]

for c in range(N_CLUSTERS):
    pct = np.sum(cluster_map == c) / cluster_map.size * 100
    print(f"  Cluster {c}: {pct:.1f}%")


# ══════════════════════════════════════════════════════════════════════
# VISUALIZATION 1: Displacement Streamlines (Ameloblast Migration Paths)
# ══════════════════════════════════════════════════════════════════════
print("\n--- Fig 1: Streamlines ---")

fig1, ax1 = plt.subplots(1, 1, figsize=(10, 9))
ax1.imshow(ref_mask, cmap="gray", alpha=0.3)

# Build streamline seed points on a grid
seed_density = 3  # points per grid cell
ys = np.linspace(20, H - 20, 30)
xs = np.linspace(20, W - 20, 30)
seed_x, seed_y = np.meshgrid(xs, ys)

# Color streamlines by cluster at seed point
speed = accum_mag_sm.copy()
speed[speed < 0.01] = 0.01  # avoid zero speed

strm = ax1.streamplot(
    np.arange(W),
    np.arange(H),
    accum_dx_sm,
    accum_dy_sm,
    color=accum_dir_sm,
    cmap="hsv",
    density=2.5,
    linewidth=0.8,
    arrowsize=0.8,
    start_points=np.column_stack([seed_x.ravel(), seed_y.ravel()]),
)
cb_strm = plt.colorbar(strm.lines, ax=ax1, fraction=0.046)
_style_cb(cb_strm, label="Direction (rad)")

# Overlay cluster boundaries
boundaries = find_boundaries(cluster_map, mode="outer")
boundary_rgba = np.zeros((H, W, 4))
boundary_rgba[boundaries] = [1, 1, 1, 0.7]
ax1.imshow(boundary_rgba)

ax1.set_title(
    "Ameloblast Migration Streamlines\n" "Color = direction, white lines = band boundaries",
    fontsize=12,
)
ax1.axis("off")
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "01_streamlines.png"), dpi=250, bbox_inches="tight", facecolor="white"
)
plt.close()
print("  Saved: 01_streamlines.png")


# ══════════════════════════════════════════════════════════════════════
# VISUALIZATION 2: Band Boundary Map with Angular Mismatch
#   (Fracture mechanics: angle change at boundary = crack deflection)
# ══════════════════════════════════════════════════════════════════════
print("--- Fig 2: Boundary angular mismatch ---")

# Compute displacement direction gradient magnitude
dir_sin = np.sin(accum_dir_sm)
dir_cos = np.cos(accum_dir_sm)
grad_sin_y = sobel(dir_sin, axis=0)
grad_sin_x = sobel(dir_sin, axis=1)
grad_cos_y = sobel(dir_cos, axis=0)
grad_cos_x = sobel(dir_cos, axis=1)
dir_gradient_mag = np.sqrt(grad_sin_y**2 + grad_sin_x**2 + grad_cos_y**2 + grad_cos_x**2)
dir_gradient_sm = gaussian_filter(dir_gradient_mag, sigma=4)

# Also compute angular mismatch at cluster boundaries
boundary_mask = find_boundaries(cluster_map, mode="thick")
# For each boundary pixel, compute angle difference between neighboring clusters
mismatch_map = np.zeros((H, W), dtype=np.float64)

# Dilate each cluster and find where they overlap at boundaries
for c1 in range(N_CLUSTERS):
    for c2 in range(c1 + 1, N_CLUSTERS):
        mask_c1 = cluster_map == c1
        mask_c2 = cluster_map == c2
        # Interface pixels: boundary pixels adjacent to both clusters
        dilated_c1 = binary_dilation(mask_c1, iterations=2)
        dilated_c2 = binary_dilation(mask_c2, iterations=2)
        interface = dilated_c1 & dilated_c2 & boundary_mask

        if interface.sum() > 0:
            # Mean direction in each cluster
            dir_c1 = np.arctan2(accum_dy_sm[mask_c1].mean(), accum_dx_sm[mask_c1].mean())
            dir_c2 = np.arctan2(accum_dy_sm[mask_c2].mean(), accum_dx_sm[mask_c2].mean())
            angle_diff = np.abs(
                np.degrees(np.arctan2(np.sin(dir_c1 - dir_c2), np.cos(dir_c1 - dir_c2)))
            )
            mismatch_map[interface] = angle_diff

fig2, axes2 = plt.subplots(1, 3, figsize=(22, 7))

# Panel 1: Direction gradient (continuous)
im_grad = axes2[0].imshow(dir_gradient_sm, cmap="hot")
axes2[0].set_title("Displacement Direction Gradient\n(crack deflection potential)", fontsize=11)
axes2[0].axis("off")
cb_grad = plt.colorbar(im_grad, ax=axes2[0], fraction=0.046)
_style_cb(cb_grad, label="Gradient Magnitude")

# Panel 2: Band boundaries colored by angular mismatch
axes2[1].imshow(ref_mask, cmap="gray", alpha=0.4)
mismatch_vis = np.ma.masked_where(mismatch_map == 0, mismatch_map)
im_mm = axes2[1].imshow(
    mismatch_vis, cmap="plasma", vmin=0, vmax=max(mismatch_map.max(), 1), interpolation="nearest"
)
axes2[1].set_title(
    "Band Boundary Angular Mismatch\n(\u00b0 between adjacent band directions)", fontsize=11
)
axes2[1].axis("off")
cb_mm = plt.colorbar(im_mm, ax=axes2[1], fraction=0.046)
_style_cb(cb_mm, label="Mismatch (\u00b0)")

# Panel 3: Cluster map with boundaries highlighted
axes2[2].imshow(cluster_map, cmap=CMAP_BANDS, interpolation="nearest", alpha=0.7)
boundary_highlight = np.zeros((H, W, 4))
boundary_highlight[boundary_mask] = [1, 1, 0, 1]  # yellow boundaries
axes2[2].imshow(boundary_highlight)
axes2[2].set_title("Band Segmentation with Boundaries", fontsize=11)
axes2[2].axis("off")

plt.suptitle(
    "Fracture Mechanics — Crack Deflection at Band Boundaries", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "02_boundary_mismatch.png"),
    dpi=250,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("  Saved: 02_boundary_mismatch.png")


# ══════════════════════════════════════════════════════════════════════
# VISUALIZATION 3: Polar Rose Diagrams (Orientation Distribution per Band)
# ══════════════════════════════════════════════════════════════════════
print("--- Fig 3: Rose diagrams ---")

fig3, axes3 = plt.subplots(
    1, N_CLUSTERS + 1, figsize=(5 * (N_CLUSTERS + 1), 5), subplot_kw=dict(projection="polar")
)

n_bins_rose = 36
bin_edges = np.linspace(-np.pi, np.pi, n_bins_rose + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_width = 2 * np.pi / n_bins_rose

# All pixels
all_dirs = accum_dir_sm.ravel()
counts_all, _ = np.histogram(all_dirs, bins=bin_edges, density=True)
axes3[0].bar(
    bin_centers, counts_all, width=bin_width, color="gray", alpha=0.7, edgecolor="k", linewidth=0.5
)
axes3[0].set_title("All Pixels", fontsize=11, fontweight="bold", pad=15)
axes3[0].set_yticklabels([])

for c in range(N_CLUSTERS):
    mask_c = cluster_map == c
    dirs_c = accum_dir_sm[mask_c]
    counts_c, _ = np.histogram(dirs_c, bins=bin_edges, density=True)
    axes3[c + 1].bar(
        bin_centers,
        counts_c,
        width=bin_width,
        color=COLORS[c],
        alpha=0.7,
        edgecolor="k",
        linewidth=0.5,
    )

    mean_dir = np.arctan2(np.sin(dirs_c).mean(), np.cos(dirs_c).mean())
    mean_mag = accum_mag_sm[mask_c].mean()
    axes3[c + 1].annotate(
        "",
        xy=(mean_dir, counts_c.max() * 0.9),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
    )
    axes3[c + 1].set_title(
        f"Band {c}\nμ={np.degrees(mean_dir):.1f}°, |d|={mean_mag:.1f} px",
        fontsize=11,
        fontweight="bold",
        pad=15,
    )
    axes3[c + 1].set_yticklabels([])

plt.suptitle(
    "Displacement Direction Rose Diagrams \u2014 Ameloblast Migration Orientation per Band",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "03_rose_diagrams.png"), dpi=250, bbox_inches="tight", facecolor="white"
)
plt.close()
print("  Saved: 03_rose_diagrams.png")


# ══════════════════════════════════════════════════════════════════════
# VISUALIZATION 4: Band Transect Profiles + FFT Periodicity
# ══════════════════════════════════════════════════════════════════════
print("--- Fig 4: Transect profiles + periodicity ---")

# Take horizontal transects at evenly spaced rows
transect_rows = np.linspace(50, H - 50, N_TRANSECTS, dtype=int)

fig4, axes4 = plt.subplots(
    N_TRANSECTS + 1, 2, figsize=(16, 3 * (N_TRANSECTS + 1)), gridspec_kw={"width_ratios": [2, 1]}
)

# Show where transects are on the cluster map
ax_map = axes4[0, 0]
ax_map.imshow(cluster_map, cmap=CMAP_BANDS, interpolation="nearest")
for i, row in enumerate(transect_rows):
    ax_map.axhline(y=row, color="white", linewidth=1.5, alpha=0.8)
    ax_map.text(
        W + 5,
        row,
        f"T{i+1}",
        color="white",
        fontsize=9,
        va="center",
        path_effects=[pe.withStroke(linewidth=2, foreground="black")],
    )
ax_map.set_title("Transect Locations on Band Map")
ax_map.axis("off")

# FFT summary
axes4[0, 1].set_visible(False)

band_widths_all = {c: [] for c in range(N_CLUSTERS)}
dominant_periods = []

for ti, row in enumerate(transect_rows):
    # Average over a few rows for stability
    r_lo = max(0, row - TRANSECT_SMOOTH)
    r_hi = min(H, row + TRANSECT_SMOOTH + 1)

    # Cluster profile
    cluster_profile = cluster_map[row, :]
    # Pitch profile along transect
    dx_profile = np.mean(pitch_map[r_lo:r_hi, :], axis=0)

    # Plot cluster colors as background + displacement curve
    ax_prof = axes4[ti + 1, 0]
    for x in range(W):
        ax_prof.axvspan(x, x + 1, color=COLORS[cluster_profile[x]], alpha=0.3)
    ax_prof.plot(dx_profile, color="black", linewidth=1.0)
    ax_prof.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_prof.set_xlim(0, W)
    ax_prof.set_ylabel(f"T{ti+1}\n(row {row})", fontsize=9)
    if ti < N_TRANSECTS - 1:
        ax_prof.set_xticklabels([])
    else:
        ax_prof.set_xlabel("X Position (px)", fontweight="bold")
    ax_prof.set_title("Pitch Profile (\u00b0)" if ti == 0 else "", fontsize=10)

    # FFT of displacement profile
    dx_detrend = dx_profile - np.mean(dx_profile)
    fft_vals = np.abs(np.fft.rfft(dx_detrend))
    freqs = np.fft.rfftfreq(len(dx_detrend), d=1.0)  # cycles/pixel

    # Skip DC component
    fft_vals[0] = 0
    # Convert to wavelength (pixels)
    valid = freqs > 0
    wavelengths = 1.0 / freqs[valid]
    fft_power = fft_vals[valid]

    # Only show meaningful range (20-400 px wavelength)
    wl_mask = (wavelengths >= 20) & (wavelengths <= 400)
    if wl_mask.sum() > 0:
        ax_fft = axes4[ti + 1, 1]
        ax_fft.plot(wavelengths[wl_mask], fft_power[wl_mask], color=COLORS[1], linewidth=1.0)
        peak_idx = np.argmax(fft_power[wl_mask])
        dom_wl = wavelengths[wl_mask][peak_idx]
        dom_wl_um = dom_wl * PIXEL_SIZE_UM
        dominant_periods.append(dom_wl)
        ax_fft.axvline(dom_wl, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
        ax_fft.text(
            dom_wl + 5,
            fft_power[wl_mask].max() * 0.8,
            f"{dom_wl:.0f} px\n({dom_wl_um:.0f} µm)",
            fontsize=8,
        )
        ax_fft.set_xlim(20, 400)
        if ti == 0:
            ax_fft.set_title("FFT Power Spectrum", fontsize=10)
        if ti < N_TRANSECTS - 1:
            ax_fft.set_xticklabels([])
        else:
            ax_fft.set_xlabel("Wavelength (px)")

    # Extract band widths from cluster profile
    for c in range(N_CLUSTERS):
        is_c = (cluster_profile == c).astype(int)
        diff_c = np.diff(is_c)
        starts = np.where(diff_c == 1)[0] + 1
        ends = np.where(diff_c == -1)[0] + 1
        if is_c[0] == 1:
            starts = np.concatenate([[0], starts])
        if is_c[-1] == 1:
            ends = np.concatenate([ends, [len(is_c)]])
        for s, e in zip(starts, ends):
            width_px = e - s
            if width_px > 5:  # ignore tiny fragments
                band_widths_all[c].append(width_px)

plt.suptitle(
    "Band Transect Analysis \u2014 Pitch Profiles & FFT Periodicity", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "04_transect_periodicity.png"),
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("  Saved: 04_transect_periodicity.png")


# ══════════════════════════════════════════════════════════════════════
# VISUALIZATION 5: Crack Deflection Potential Map
# ══════════════════════════════════════════════════════════════════════
print("--- Fig 5: Crack deflection potential ---")

fig5, axes5 = plt.subplots(1, 3, figsize=(22, 7))

# Panel 1: Gradient magnitude as crack deflection potential
# Smooth more aggressively for cleaner visualization
cdp = gaussian_filter(dir_gradient_mag, sigma=12)
im_cdp = axes5[0].imshow(cdp, cmap="inferno")
axes5[0].set_title("Crack Deflection Potential\n(displacement direction gradient)", fontsize=11)
axes5[0].axis("off")
cb_cdp = plt.colorbar(im_cdp, ax=axes5[0], fraction=0.046)
_style_cb(cb_cdp, label="Deflection Potential")

# Panel 2: Overlay on rod mask with contours
axes5[1].imshow(ref_mask, cmap="gray", alpha=0.6)
axes5[1].contour(cdp, levels=5, colors="red", linewidths=0.8, alpha=0.8)
axes5[1].contour(cdp, levels=[np.percentile(cdp, 85)], colors="yellow", linewidths=1.5)
axes5[1].set_title("Deflection Contours on Rod Mask\n(yellow = 85th percentile)", fontsize=11)
axes5[1].axis("off")

# Panel 3: Band map with deflection potential at boundaries
axes5[2].imshow(cluster_map, cmap=CMAP_BANDS, interpolation="nearest", alpha=0.5)
boundary_cdp = cdp.copy()
boundary_cdp[~binary_dilation(boundary_mask, iterations=3)] = 0
im_bcdp = axes5[2].imshow(
    np.ma.masked_where(boundary_cdp == 0, boundary_cdp), cmap="hot", interpolation="nearest"
)
axes5[2].set_title("Deflection Potential at Band Interfaces", fontsize=11)
axes5[2].axis("off")
cb_bcdp = plt.colorbar(im_bcdp, ax=axes5[2], fraction=0.046)
_style_cb(cb_bcdp, label="Deflection Potential")

plt.suptitle(
    "Crack Deflection Analysis — Microstructure \u2192 Fracture Resistance",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "05_crack_deflection.png"),
    dpi=250,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("  Saved: 05_crack_deflection.png")


# ══════════════════════════════════════════════════════════════════════
# VISUALIZATION 6: Publication Composite Figure
# ══════════════════════════════════════════════════════════════════════
print("--- Fig 6: Publication composite ---")

fig6, axes6 = plt.subplots(2, 3, figsize=(20, 13))

# (a) Rod mask with scale bar
axes6[0, 0].imshow(ref_mask, cmap="gray")
# Scale bar: 50 µm
bar_px = 50.0 / PIXEL_SIZE_UM
axes6[0, 0].plot([W - 30 - bar_px, W - 30], [H - 30, H - 30], color="white", linewidth=3)
axes6[0, 0].text(
    W - 30 - bar_px / 2,
    H - 45,
    "50 µm",
    color="white",
    fontsize=10,
    ha="center",
    fontweight="bold",
    path_effects=[pe.withStroke(linewidth=2, foreground="black")],
)
axes6[0, 0].set_title("(a) Axial Rod Mask (slice 51)", fontsize=12)
axes6[0, 0].axis("off")

# (b) Displacement field (streamlines + cluster boundaries)
axes6[0, 1].imshow(ref_mask, cmap="gray", alpha=0.3)
strm2 = axes6[0, 1].streamplot(
    np.arange(W),
    np.arange(H),
    accum_dx_sm,
    accum_dy_sm,
    color=accum_dir_sm,
    cmap="hsv",
    density=2.0,
    linewidth=0.6,
    arrowsize=0.6,
)
boundary_rgba2 = np.zeros((H, W, 4))
boundary_rgba2[boundaries] = [1, 1, 1, 0.6]
axes6[0, 1].imshow(boundary_rgba2)
axes6[0, 1].set_title("(b) Migration Streamlines + Boundaries", fontsize=12)
axes6[0, 1].axis("off")

# (c) Band segmentation
axes6[0, 2].imshow(ref_mask, cmap="gray", alpha=0.4)
axes6[0, 2].imshow(cluster_map, cmap=CMAP_BANDS, alpha=0.55, interpolation="nearest")
for c in range(N_CLUSTERS):
    mask_c = cluster_map == c
    cy_c, cx_c = np.where(mask_c)
    axes6[0, 2].text(
        cx_c.mean(),
        cy_c.mean(),
        f"{c}",
        color="white",
        fontsize=16,
        fontweight="bold",
        ha="center",
        va="center",
        path_effects=[pe.withStroke(linewidth=3, foreground="black")],
    )
axes6[0, 2].set_title(f"(c) SOM Band Segmentation ({N_CLUSTERS} clusters)", fontsize=12)
axes6[0, 2].axis("off")

# (d) Crack deflection potential
im_d = axes6[1, 0].imshow(cdp, cmap="inferno")
axes6[1, 0].contour(cdp, levels=[np.percentile(cdp, 85)], colors="white", linewidths=1.0)
axes6[1, 0].set_title("(d) Crack Deflection Potential", fontsize=12)
axes6[1, 0].axis("off")
cb_d = plt.colorbar(im_d, ax=axes6[1, 0], fraction=0.046)
_style_cb(cb_d)

# (e) Angular mismatch at boundaries
axes6[1, 1].imshow(ref_mask, cmap="gray", alpha=0.3)
axes6[1, 1].imshow(cluster_map, cmap=CMAP_BANDS, alpha=0.3, interpolation="nearest")
mismatch_vis2 = np.ma.masked_where(mismatch_map == 0, mismatch_map)
im_e = axes6[1, 1].imshow(mismatch_vis2, cmap="plasma", interpolation="nearest")
axes6[1, 1].set_title("(e) Boundary Angular Mismatch (°)", fontsize=12)
axes6[1, 1].axis("off")
cb_e = plt.colorbar(im_e, ax=axes6[1, 1], fraction=0.046)
_style_cb(cb_e, label="\u00b0")

# (f) Displacement scatter (dx vs dy) colored by cluster
for c in range(N_CLUSTERS):
    mask_c = cluster_map == c
    pitch_c = pitch_map[mask_c]
    yaw_c = yaw_map[mask_c]
    axes6[1, 2].scatter(
        pitch_c[::30], yaw_c[::30], s=2, alpha=0.3, color=COLORS[c], label=f"Band {c}"
    )
axes6[1, 2].set_xlabel("Pitch (\u00b0)", fontweight="bold")
axes6[1, 2].set_ylabel("Yaw (\u00b0)", fontweight="bold")
axes6[1, 2].set_title("(f) Pitch vs Yaw by Band", fontsize=12)
axes6[1, 2].legend(markerscale=5, fontsize=9)
axes6[1, 2].set_aspect("equal")
axes6[1, 2].axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
axes6[1, 2].axvline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
_style_spines(axes6[1, 2])

plt.suptitle(
    "Enamel Rod Decussation — SOM Band Segmentation & Fracture Mechanics",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "06_publication_composite.png"),
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("  Saved: 06_publication_composite.png")


# ══════════════════════════════════════════════════════════════════════
# MORPHOMETRIC EXTRACTION
# ══════════════════════════════════════════════════════════════════════
print("\n--- Morphometric Extraction ---")
import csv

morphometrics = {
    "pixel_size_um": PIXEL_SIZE_UM,
    "image_size_px": [int(H), int(W)],
    "image_size_um": [round(H * PIXEL_SIZE_UM, 1), round(W * PIXEL_SIZE_UM, 1)],
    "n_slices": LAST_SLICE - FIRST_SLICE + 1,
    "n_clusters": N_CLUSTERS,
    "bands": {},
    "boundaries": {},
    "periodicity": {},
    "fabric": {},
}

# ── Per-band morphometrics ───────────────────────────────────────────
band_rows = []  # for CSV

for c in range(N_CLUSTERS):
    mask_c = cluster_map == c
    dirs_c = accum_dir_sm[mask_c]
    mags_c = accum_mag_sm[mask_c]
    pitch_c = pitch_map[mask_c]
    yaw_c = yaw_map[mask_c]
    tilt_c = tilt_map[mask_c]

    mean_dir = np.arctan2(np.sin(dirs_c).mean(), np.cos(dirs_c).mean())
    circ_var = 1.0 - np.sqrt(np.sin(dirs_c).mean() ** 2 + np.cos(dirs_c).mean() ** 2)

    # Orientation tensor per band (3D unit vectors: dx, dy, dz)
    dx_c = accum_dx_sm[mask_c]
    dy_c = accum_dy_sm[mask_c]
    dz_c = np.full_like(dx_c, N_TRANSITIONS)
    V_raw = np.column_stack([dx_c, dy_c, dz_c])
    norms = np.linalg.norm(V_raw, axis=1, keepdims=True)
    V_unit = V_raw / np.where(norms < 1e-12, 1.0, norms)
    evals_c, _ = orientation_tensor(V_unit)
    C_c, K_c, enorm_c = fabric_metrics(evals_c)

    # Band width from transects
    widths_px = band_widths_all[c]
    widths_um = [w * PIXEL_SIZE_UM for w in widths_px]

    band_data = {
        "area_fraction_pct": round(mask_c.sum() / cluster_map.size * 100, 2),
        "mean_pitch_deg": round(float(pitch_c.mean()), 2),
        "std_pitch_deg": round(float(pitch_c.std()), 2),
        "mean_yaw_deg": round(float(yaw_c.mean()), 2),
        "std_yaw_deg": round(float(yaw_c.std()), 2),
        "mean_tilt_deg": round(float(tilt_c.mean()), 2),
        "std_tilt_deg": round(float(tilt_c.std()), 2),
        "mean_direction_deg": round(float(np.degrees(mean_dir)), 2),
        "circular_variance": round(float(circ_var), 4),
        "mean_displacement_mag_px": round(float(mags_c.mean()), 2),
        "mean_displacement_mag_um": round(float(mags_c.mean() * PIXEL_SIZE_UM), 2),
        "woodcock_C": round(float(C_c), 4),
        "woodcock_K": round(float(K_c), 4) if not np.isinf(K_c) else "inf",
        "eigenvalues_norm": [round(float(e), 4) for e in enorm_c],
        "band_width_um_mean": round(float(np.mean(widths_um)), 1) if widths_um else None,
        "band_width_um_std": round(float(np.std(widths_um)), 1) if widths_um else None,
        "band_width_um_min": round(float(np.min(widths_um)), 1) if widths_um else None,
        "band_width_um_max": round(float(np.max(widths_um)), 1) if widths_um else None,
        "n_segments": len(widths_px),
    }
    morphometrics["bands"][f"band_{c}"] = band_data

    band_rows.append(
        {
            "band": c,
            "area_pct": band_data["area_fraction_pct"],
            "mean_pitch_deg": band_data["mean_pitch_deg"],
            "std_pitch_deg": band_data["std_pitch_deg"],
            "mean_yaw_deg": band_data["mean_yaw_deg"],
            "std_yaw_deg": band_data["std_yaw_deg"],
            "mean_tilt_deg": band_data["mean_tilt_deg"],
            "std_tilt_deg": band_data["std_tilt_deg"],
            "mean_direction_deg": band_data["mean_direction_deg"],
            "circular_variance": band_data["circular_variance"],
            "disp_mag_px": band_data["mean_displacement_mag_px"],
            "disp_mag_um": band_data["mean_displacement_mag_um"],
            "woodcock_C": band_data["woodcock_C"],
            "woodcock_K": band_data["woodcock_K"],
            "band_width_um_mean": band_data["band_width_um_mean"],
            "band_width_um_std": band_data["band_width_um_std"],
            "n_segments": band_data["n_segments"],
        }
    )

# ── Global orientation tensor + Woodcock ─────────────────────────────
dx_all = accum_dx_sm.ravel()
dy_all = accum_dy_sm.ravel()
dz_all = np.full_like(dx_all, N_TRANSITIONS)
V_all_raw = np.column_stack([dx_all, dy_all, dz_all])
norms_all = np.linalg.norm(V_all_raw, axis=1, keepdims=True)
V_all_unit = V_all_raw / np.where(norms_all < 1e-12, 1.0, norms_all)
evals_global, evecs_global = orientation_tensor(V_all_unit)
C_global, K_global, enorm_global = fabric_metrics(evals_global)

morphometrics["fabric"] = {
    "global_woodcock_C": round(float(C_global), 4),
    "global_woodcock_K": round(float(K_global), 4) if not np.isinf(K_global) else "inf",
    "global_eigenvalues_norm": [round(float(e), 4) for e in enorm_global],
    "global_mean_pitch_deg": round(float(pitch_map.mean()), 2),
    "global_std_pitch_deg": round(float(pitch_map.std()), 2),
    "global_mean_yaw_deg": round(float(yaw_map.mean()), 2),
    "global_std_yaw_deg": round(float(yaw_map.std()), 2),
    "global_mean_tilt_deg": round(float(tilt_map.mean()), 2),
    "global_std_tilt_deg": round(float(tilt_map.std()), 2),
}

# Add global row to CSV data
band_rows.append(
    {
        "band": "Global",
        "area_pct": 100.0,
        "mean_pitch_deg": round(float(pitch_map.mean()), 2),
        "std_pitch_deg": round(float(pitch_map.std()), 2),
        "mean_yaw_deg": round(float(yaw_map.mean()), 2),
        "std_yaw_deg": round(float(yaw_map.std()), 2),
        "mean_tilt_deg": round(float(tilt_map.mean()), 2),
        "std_tilt_deg": round(float(tilt_map.std()), 2),
        "mean_direction_deg": "",
        "circular_variance": "",
        "disp_mag_px": round(float(accum_mag_sm.mean()), 2),
        "disp_mag_um": round(float(accum_mag_sm.mean() * PIXEL_SIZE_UM), 2),
        "woodcock_C": round(float(C_global), 4),
        "woodcock_K": round(float(K_global), 4) if not np.isinf(K_global) else "inf",
        "band_width_um_mean": "",
        "band_width_um_std": "",
        "n_segments": "",
    }
)

# ── Pairwise boundary morphometrics ──────────────────────────────────
boundary_rows = []

for c1 in range(N_CLUSTERS):
    for c2 in range(c1 + 1, N_CLUSTERS):
        mask_c1 = cluster_map == c1
        mask_c2 = cluster_map == c2

        pitch_c1_mean = float(pitch_map[mask_c1].mean())
        pitch_c2_mean = float(pitch_map[mask_c2].mean())
        yaw_c1_mean = float(yaw_map[mask_c1].mean())
        yaw_c2_mean = float(yaw_map[mask_c2].mean())

        dir_c1 = np.arctan2(accum_dy_sm[mask_c1].mean(), accum_dx_sm[mask_c1].mean())
        dir_c2 = np.arctan2(accum_dy_sm[mask_c2].mean(), accum_dx_sm[mask_c2].mean())
        angle_diff = np.abs(
            np.degrees(np.arctan2(np.sin(dir_c1 - dir_c2), np.cos(dir_c1 - dir_c2)))
        )

        dilated_c1 = binary_dilation(mask_c1, iterations=2)
        dilated_c2 = binary_dilation(mask_c2, iterations=2)
        interface = dilated_c1 & dilated_c2 & boundary_mask
        interface_length_px = interface.sum()

        bnd_data = {
            "angular_mismatch_deg": round(float(angle_diff), 2),
            "pitch_diff_deg": round(abs(pitch_c1_mean - pitch_c2_mean), 2),
            "yaw_diff_deg": round(abs(yaw_c1_mean - yaw_c2_mean), 2),
            "interface_length_px": int(interface_length_px),
            "interface_length_um": round(float(interface_length_px * PIXEL_SIZE_UM), 1),
        }
        morphometrics["boundaries"][f"band_{c1}_vs_{c2}"] = bnd_data

        boundary_rows.append(
            {
                "pair": f"{c1} vs {c2}",
                "angular_mismatch_deg": bnd_data["angular_mismatch_deg"],
                "pitch_diff_deg": bnd_data["pitch_diff_deg"],
                "yaw_diff_deg": bnd_data["yaw_diff_deg"],
                "interface_length_um": bnd_data["interface_length_um"],
            }
        )

# ── Periodicity ──────────────────────────────────────────────────────
if dominant_periods:
    dominant_periods_arr = np.array(dominant_periods)
    morphometrics["periodicity"] = {
        "dominant_wavelength_px_mean": round(float(dominant_periods_arr.mean()), 1),
        "dominant_wavelength_px_std": round(float(dominant_periods_arr.std()), 1),
        "dominant_wavelength_um_mean": round(float(dominant_periods_arr.mean() * PIXEL_SIZE_UM), 1),
        "dominant_wavelength_um_std": round(float(dominant_periods_arr.std() * PIXEL_SIZE_UM), 1),
    }

# ── Save morphometrics JSON ──────────────────────────────────────────
json_path = os.path.join(OUT_DIR, "morphometrics.json")
with open(json_path, "w") as f:
    json.dump(morphometrics, f, indent=2)
print("  Saved: morphometrics.json")

# ── Save comprehensive CSV stats sheet ───────────────────────────────
csv_path = os.path.join(OUT_DIR, "stats_bands.csv")
fieldnames = list(band_rows[0].keys())
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(band_rows)
print("  Saved: stats_bands.csv")

csv_bnd_path = os.path.join(OUT_DIR, "stats_boundaries.csv")
if boundary_rows:
    bnd_fields = list(boundary_rows[0].keys())
    with open(csv_bnd_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=bnd_fields)
        writer.writeheader()
        writer.writerows(boundary_rows)
    print("  Saved: stats_boundaries.csv")

# ── Periodicity + global summary CSV ─────────────────────────────────
csv_summary_path = os.path.join(OUT_DIR, "stats_summary.csv")
with open(csv_summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["parameter", "value", "unit"])
    writer.writerow(["pixel_size", PIXEL_SIZE_UM, "um/px"])
    writer.writerow(["image_height", H, "px"])
    writer.writerow(["image_width", W, "px"])
    writer.writerow(["image_height_um", round(H * PIXEL_SIZE_UM, 1), "um"])
    writer.writerow(["image_width_um", round(W * PIXEL_SIZE_UM, 1), "um"])
    writer.writerow(["n_slices", LAST_SLICE - FIRST_SLICE + 1, ""])
    writer.writerow(["stack_depth_um", round(N_TRANSITIONS * PIXEL_SIZE_UM, 1), "um"])
    writer.writerow(["n_clusters", N_CLUSTERS, ""])
    writer.writerow(["global_woodcock_C", round(float(C_global), 4), ""])
    writer.writerow(
        ["global_woodcock_K", round(float(K_global), 4) if not np.isinf(K_global) else "inf", ""]
    )
    writer.writerow(["global_eigenvalue_1", round(float(enorm_global[0]), 4), "normalised"])
    writer.writerow(["global_eigenvalue_2", round(float(enorm_global[1]), 4), "normalised"])
    writer.writerow(["global_eigenvalue_3", round(float(enorm_global[2]), 4), "normalised"])
    writer.writerow(["global_mean_pitch", round(float(pitch_map.mean()), 2), "deg"])
    writer.writerow(["global_std_pitch", round(float(pitch_map.std()), 2), "deg"])
    writer.writerow(["global_mean_yaw", round(float(yaw_map.mean()), 2), "deg"])
    writer.writerow(["global_std_yaw", round(float(yaw_map.std()), 2), "deg"])
    writer.writerow(["global_mean_tilt", round(float(tilt_map.mean()), 2), "deg"])
    writer.writerow(["global_std_tilt", round(float(tilt_map.std()), 2), "deg"])
    if morphometrics.get("periodicity"):
        p = morphometrics["periodicity"]
        writer.writerow(["band_periodicity_mean", p["dominant_wavelength_um_mean"], "um"])
        writer.writerow(["band_periodicity_std", p["dominant_wavelength_um_std"], "um"])
print("  Saved: stats_summary.csv")

# ── Print summary ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MORPHOMETRIC SUMMARY")
print("=" * 60)
print(f"Image: {H}x{W} px ({H*PIXEL_SIZE_UM:.0f}x{W*PIXEL_SIZE_UM:.0f} um)")
print(
    f"Stack: {LAST_SLICE - FIRST_SLICE + 1} slices, "
    f"{N_TRANSITIONS * PIXEL_SIZE_UM:.1f} um depth"
)
print(f"Clusters: {N_CLUSTERS}")
print(f"\nGlobal fabric: Woodcock C = {C_global:.3f}, K = {K_global:.3f}")
print(f"  eigenvalues (normalised): {enorm_global}")
print()

for c in range(N_CLUSTERS):
    b = morphometrics["bands"][f"band_{c}"]
    print(f"Band {c}:")
    print(f"  Area: {b['area_fraction_pct']:.1f}%")
    print(f"  Pitch: {b['mean_pitch_deg']:.1f} +/- {b['std_pitch_deg']:.1f} deg")
    print(f"  Yaw:   {b['mean_yaw_deg']:.1f} +/- {b['std_yaw_deg']:.1f} deg")
    print(f"  Tilt:  {b['mean_tilt_deg']:.1f} +/- {b['std_tilt_deg']:.1f} deg")
    print(f"  Woodcock: C = {b['woodcock_C']}, K = {b['woodcock_K']}")
    if b["band_width_um_mean"]:
        print(
            f"  Band width: {b['band_width_um_mean']:.0f} +/- "
            f"{b['band_width_um_std']:.0f} um (n={b['n_segments']})"
        )
    print()

print("Pairwise boundary mismatch:")
for key, val in morphometrics["boundaries"].items():
    print(
        f"  {key}: {val['angular_mismatch_deg']:.1f} deg "
        f"(pitch diff {val['pitch_diff_deg']:.1f} deg, "
        f"yaw diff {val['yaw_diff_deg']:.1f} deg, "
        f"{val['interface_length_um']:.0f} um interface)"
    )

if morphometrics.get("periodicity"):
    p = morphometrics["periodicity"]
    print(
        f"\nBand periodicity: {p['dominant_wavelength_um_mean']:.0f} +/- "
        f"{p['dominant_wavelength_um_std']:.0f} um"
    )

print(f"\nAll outputs in: {OUT_DIR}/")
print("Done.")
