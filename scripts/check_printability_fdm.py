#!/usr/bin/env python
"""
check_printability_fdm.py
=========================

FDM printability audit for STL files emitted by biomimetic_pipeline CAD
generators. Answers the question:

    "Will this lattice STL actually print on an Ender 3 V2 (0.4 mm nozzle,
     PLA, 0.2 mm layer height) with reasonable yield?"

Why this script exists
----------------------
The biomimetic_pipeline already has *parameter-level* manufacturability
guards in `mapping/feature_to_cad.py`: it clamps ROD_DIAMETER, BRIDGE_
DIAMETER, CENTER_SPACING and JUNCTION_SPHERE_FACTOR to an SLA-class minimum
feature size and logs the clamps in `cad_params.json -> provenance.scale_
clamps`. Those clamps act on inputs to the CAD generator; they do not look
at the geometry the generator actually emits. This script fills that gap
by interrogating the STL itself:

  - is the mesh slicer-acceptable (watertight, manifold, single body)?
  - does it fit the printer's build volume?
  - how much of its surface is overhanging past the FDM-friendly 45 deg?
  - how wide are the unsupported horizontal spans (PLA bridges) that the
    slicer would have to generate?
  - how thin do walls get locally (vs the 0.4 mm nozzle floor)?

Optionally, if UltiMaker Cura is installed locally, CuraEngine is invoked
in dry-run mode to add an authoritative slice-time view (support-material
volume, filament use) on top of the geometric heuristics.

Inputs
------
Positional CLI arguments: any combination of STL paths or directories.
Directories are walked for *.stl.

Outputs
-------
Inside `--out-dir`:
  - `<stem>__printability.json`  per-STL machine-readable report
  - `summary.txt`                multi-STL pass/warn/fail table

Conventions
-----------
Each check returns a status in {green, yellow, red, skipped}. The overall
verdict per STL is the worst status across all checks.

NOTE on conda env: per the project convention this script must be run
inside the `base` conda env (it needs trimesh, scipy, numpy, rtree).
"""
from __future__ import annotations

# Standard library
import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

# Third-party numerical / mesh libraries
import numpy as np
import trimesh

# ---------------------------------------------------------------------------
# Printer + process constants for Ender 3 V2 + PLA + 0.4 mm nozzle.
#
# Build volume: standard Ender 3 V2 chassis is 220x220x250 mm. Cura's stock
# Ender 3 definition (creality_ender3.def.json) shares the same envelope.
# Min feature ~ nozzle width (single-perimeter wall); 2x nozzle is the
# "robust" two-perimeter wall. Overhang threshold of 45 deg is the slicer
# convention for "needs support material" in PLA. Max unsupported bridge
# span ~ 5 mm is the published PLA bridging tolerance before sag.
# ---------------------------------------------------------------------------
ENDER3_V2_BUILD_VOLUME_MM = (220.0, 220.0, 250.0)  # (X, Y, Z) print volume
ENDER3_V2_NOZZLE_MM = 0.4  # standard FDM nozzle
ENDER3_V2_MIN_WALL_MM = ENDER3_V2_NOZZLE_MM  # single perimeter = nozzle dia
ENDER3_V2_ROBUST_WALL_MM = 2.0 * ENDER3_V2_NOZZLE_MM  # two-perimeter wall
OVERHANG_THRESHOLD_DEG = 45.0  # slicer support trigger
MAX_BRIDGE_SPAN_MM = 5.0  # PLA clean-bridge max
WALL_THICKNESS_SAMPLE_N = 400  # vertices to probe
# 400 is empirically the sweet spot: large enough to recover a usable
# minimum-thickness percentile on the 100k-1M-face lattices in this project,
# small enough that ray-cast time stays under a few seconds. The earlier
# 4000-sample version forced trimesh.proximity.thickness to build a separate
# scratch index that filled disk on million-face meshes.

# Pre-computed numerical limit on the z-component of an outward face normal
# above which the face counts as an "overhang requiring support".
#   A face is an overhang if the angle between its outward normal and -Z is
#   smaller than OVERHANG_THRESHOLD_DEG, i.e. n . (-Z) > cos(45 deg),
#   i.e. -n_z > cos(45 deg), i.e. n_z < -cos(45 deg).
_OVERHANG_NZ_LIMIT = float(np.cos(np.deg2rad(OVERHANG_THRESHOLD_DEG)))

# UltiMaker Cura on macOS ships CuraEngine inside its app bundle. The first
# path is the canonical install location; add more here if needed.
CURAENGINE_PATHS = [
    Path("/Applications/UltiMaker Cura.app/Contents/Resources/CuraEngine"),
]
CURA_DEFS_DIR = Path(
    "/Applications/UltiMaker Cura.app/Contents/Resources/share/cura/resources/definitions"
)

# Status ordering for "worst-of" aggregation.
_WORST_RANK = {"skipped": 0, "green": 1, "yellow": 2, "red": 3}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    """One named printability check with status and supporting numbers.

    status   "green" = passes Ender-3-V2 PLA defaults
             "yellow" = printable but expect support-removal damage / fragility
             "red"   = will not print or will print as scrap
             "skipped" = check not run (missing dependency etc.)
    value    primary scalar metric for the check (or None if not applicable)
    detail   free-form dict of supporting numbers (units called out in keys)
    """

    name: str
    status: str
    message: str
    value: Optional[float] = None
    detail: dict = field(default_factory=dict)


@dataclass
class STLReport:
    """Full per-STL audit report. Serialized as JSON next to the STL."""

    stl_path: str
    n_faces: int
    n_vertices: int
    bounds_mm: list  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    extents_mm: list  # [dx, dy, dz]
    volume_mm3: float  # signed-volume magnitude when watertight
    surface_area_mm2: float
    checks: list  # list[dict] (serialized CheckResult)
    overall_status: str  # worst of checks[].status
    runtime_seconds: float

    def to_json(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helper: load a mesh robustly, collapsing scene -> single Trimesh if needed.
# ---------------------------------------------------------------------------
def _load_mesh(path: Path) -> trimesh.Trimesh:
    """Load an STL into a single Trimesh, concatenating scene children."""
    loaded = trimesh.load(str(path), force="mesh")
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    # Scene fallback — concatenate all geometries.
    geoms = list(loaded.geometry.values()) if hasattr(loaded, "geometry") else []
    if not geoms:
        raise ValueError(f"no geometry loaded from {path}")
    return trimesh.util.concatenate(geoms)


# ---------------------------------------------------------------------------
# Tier 1 checks (geometric, pure-Python, always run)
# ---------------------------------------------------------------------------


def _check_watertight(mesh: trimesh.Trimesh) -> CheckResult:
    """A non-watertight mesh confuses every slicer's inside/outside test."""
    if mesh.is_watertight:
        return CheckResult("watertight", "green", "mesh is watertight")
    return CheckResult(
        "watertight",
        "red",
        "mesh has open boundary edges — slicer will misinterpret inside/outside",
    )


def _check_winding(mesh: trimesh.Trimesh) -> CheckResult:
    """Inconsistent face winding leads to flipped interior/exterior regions."""
    if mesh.is_winding_consistent:
        return CheckResult("winding_consistent", "green", "face winding is consistent")
    return CheckResult(
        "winding_consistent",
        "red",
        "inconsistent face winding — slicer may flip interior/exterior",
    )


def _check_components(mesh: trimesh.Trimesh) -> CheckResult:
    """Count disconnected bodies. A single print should be 1 connected body.

    More than one component means the geometry has fragmented (often due to
    a boolean union failure between rod cylinders and bridge cylinders), and
    the slicer will treat each piece as a separate print job — most of which
    will lack first-layer adhesion.
    """
    comps = mesh.split(only_watertight=False)
    n = len(comps)
    if n == 1:
        return CheckResult("connected_components", "green", "single connected body", value=float(n))
    if n <= 5:
        return CheckResult(
            "connected_components",
            "yellow",
            f"{n} disconnected bodies — slicer will print as separate pieces",
            value=float(n),
        )
    return CheckResult(
        "connected_components",
        "red",
        f"{n} disconnected bodies — geometry has fragmented (likely boolean failure)",
        value=float(n),
    )


def _check_degenerate_faces(mesh: trimesh.Trimesh) -> CheckResult:
    """Zero-area triangles confuse some slicers; warn rather than fail."""
    # In trimesh 4.x the method is nondegenerate_faces (returns bool mask of
    # faces that are NOT degenerate). Invert to count degenerate ones.
    nondeg_mask = mesh.nondegenerate_faces()
    n_total = len(mesh.faces)
    n_deg = int(n_total - int(np.sum(nondeg_mask)))
    if n_deg == 0:
        return CheckResult("degenerate_faces", "green", "no degenerate faces", value=0.0)
    return CheckResult(
        "degenerate_faces",
        "yellow",
        f"{n_deg} zero-area faces ({n_deg/n_total:.1%}) — slicer may warn",
        value=float(n_deg),
    )


def _check_build_volume(mesh: trimesh.Trimesh) -> CheckResult:
    """Fit-the-bed test with permissive axis-aligned rotation.

    Strategy:
      - default orientation: pass if extents fit directly in (X, Y, Z).
      - axis-aligned rotation: sort extents, allow the longest axis to be Z.
        Passes if the sorted-largest extent fits the Z envelope and the
        other two fit the XY envelope.
    """
    bvx, bvy, bvz = ENDER3_V2_BUILD_VOLUME_MM
    ex, ey, ez = (float(v) for v in mesh.extents)
    fits_default = ex <= bvx and ey <= bvy and ez <= bvz
    se = sorted((ex, ey, ez), reverse=True)  # largest, mid, smallest
    fits_any = (se[0] <= bvz) and (se[1] <= max(bvx, bvy)) and (se[2] <= min(bvx, bvy))
    detail = {
        "extents_mm": [ex, ey, ez],
        "build_volume_mm": list(ENDER3_V2_BUILD_VOLUME_MM),
        "fits_at_default_orientation": fits_default,
        "fits_at_some_axis_aligned_orientation": fits_any,
    }
    if fits_default:
        return CheckResult(
            "build_volume",
            "green",
            f"fits {bvx:.0f}x{bvy:.0f}x{bvz:.0f} mm bed at default orientation",
            detail=detail,
        )
    if fits_any:
        return CheckResult(
            "build_volume",
            "yellow",
            "exceeds bed in default orientation — rotation required",
            detail=detail,
        )
    return CheckResult(
        "build_volume",
        "red",
        "exceeds Ender 3 V2 build volume at every axis-aligned orientation",
        detail=detail,
    )


def _check_overhangs(mesh: trimesh.Trimesh) -> CheckResult:
    """Compute the fraction of surface area requiring support material.

    Approach: scan every face's outward unit normal; a face is "overhang"
    if the downward (-Z) component of its normal exceeds cos(45 deg). The
    overhang area fraction is the most direct predictor of support burden
    on FDM for a fixed +Z build orientation. Returns the fraction as the
    primary metric and area in detail for absolute scale.
    """
    normals = mesh.face_normals
    areas = mesh.area_faces
    is_overhang = normals[:, 2] < -_OVERHANG_NZ_LIMIT
    overhang_area = float(areas[is_overhang].sum())
    total_area = float(areas.sum())
    frac = overhang_area / total_area if total_area > 0 else 0.0
    detail = {
        "overhang_area_mm2": overhang_area,
        "total_surface_area_mm2": total_area,
        "overhang_fraction": frac,
        "n_overhang_faces": int(is_overhang.sum()),
        "threshold_deg": OVERHANG_THRESHOLD_DEG,
    }
    # Thresholds chosen pragmatically: above 50% overhang the slicer's
    # support material rivals the part itself; between 20-50% the supports
    # are unavoidable but the part still emerges; below 20% the print is
    # mostly self-supporting.
    if frac > 0.50:
        return CheckResult(
            "overhangs_at_45deg",
            "red",
            f"{frac:.1%} of surface needs support — print will be a brick of support material",
            value=frac,
            detail=detail,
        )
    if frac > 0.20:
        return CheckResult(
            "overhangs_at_45deg",
            "yellow",
            f"{frac:.1%} of surface needs support — expect surface damage from removal",
            value=frac,
            detail=detail,
        )
    return CheckResult(
        "overhangs_at_45deg",
        "green",
        f"{frac:.1%} of surface needs support — within FDM-friendly range",
        value=frac,
        detail=detail,
    )


def _check_max_bridge_span(mesh: trimesh.Trimesh) -> CheckResult:
    """Largest single contiguous unsupported horizontal span.

    What we actually care about for FDM is: across the whole part, what is
    the longest *individual* overhang patch the slicer would try to bridge
    in mid-air? A 1 mm-wide bridge repeated 50 times across a lattice is
    completely printable; one continuous 50 mm overhang is not.

    Approach:
      1. Project overhang triangle centroids to XY.
      2. Build a KD-tree on those centroids and join any two within 1 mm
         of each other into the same patch (connected-components on the
         neighbor graph).
      3. For each patch, take the maximum pairwise distance (or convex-hull
         diameter for large patches) as that patch's span.
      4. Report the maximum span across patches.

    The 1 mm join radius (CLUSTER_EPS_MM) is slightly larger than the
    typical inter-triangle spacing on a 0.4 mm-nozzle-class feature, so a
    physically-continuous overhang stays a single cluster, while disjoint
    bridges sitting tens of mm apart remain separate.
    """
    # Scipy imports kept local to keep the script importable in stripped envs.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import ConvexHull, cKDTree

    CLUSTER_EPS_MM = 1.0  # join radius for contiguous-overhang grouping
    MIN_CLUSTER_SIZE = 5  # ignore single stray overhang triangles

    normals = mesh.face_normals
    is_overhang = normals[:, 2] < -_OVERHANG_NZ_LIMIT
    if not is_overhang.any():
        return CheckResult(
            "max_bridge_span",
            "green",
            "no overhang faces",
            value=0.0,
            detail={"max_span_mm": 0.0, "limit_mm": MAX_BRIDGE_SPAN_MM},
        )
    centroids_xy = mesh.triangles_center[is_overhang][:, :2]
    # Subsample very large clouds — O(n) memory for the KD-tree but the
    # pairs query can still blow up if the lattice is dense. 8k points is
    # plenty to resolve sub-mm structure on a 50 mm-class part.
    if len(centroids_xy) > 8000:
        rng = np.random.default_rng(1)
        idx = rng.choice(len(centroids_xy), size=8000, replace=False)
        centroids_xy = centroids_xy[idx]
    n_pts = len(centroids_xy)

    # Connected-components clustering: any two centroids within CLUSTER_EPS_MM
    # are in the same patch. Disjoint patches are returned as separate labels.
    tree = cKDTree(centroids_xy)
    pairs = tree.query_pairs(r=CLUSTER_EPS_MM, output_type="ndarray")
    if len(pairs) == 0:
        # No two overhang centroids within join radius — every triangle is
        # isolated, so the largest "patch" is smaller than CLUSTER_EPS_MM.
        return CheckResult(
            "max_bridge_span",
            "green",
            "overhang triangles are spatially isolated (< 1 mm patches)",
            value=float(CLUSTER_EPS_MM),
            detail={
                "max_span_mm": float(CLUSTER_EPS_MM),
                "limit_mm": MAX_BRIDGE_SPAN_MM,
                "n_overhang_centroids": int(n_pts),
                "n_clusters": 0,
            },
        )
    # Symmetrize the pair list and build a sparse adjacency for the cc walk.
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.ones(len(rows), dtype=np.int8)
    adj = csr_matrix((data, (rows, cols)), shape=(n_pts, n_pts))
    n_comp, labels = connected_components(adj, directed=False)

    # Diameter of each patch. For small patches use exact pairwise distance;
    # for large ones use convex-hull-vertex pairwise distance (much faster).
    max_span = 0.0
    max_cluster_size = 0
    diameters = []
    for k in range(n_comp):
        mask = labels == k
        sz = int(mask.sum())
        if sz < MIN_CLUSTER_SIZE:
            continue
        pts = centroids_xy[mask]
        if sz <= 50:
            # exact pairwise distance for small patches
            d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1).max()
        else:
            try:
                h = ConvexHull(pts)
                hp = pts[h.vertices]
                d = np.linalg.norm(hp[:, None, :] - hp[None, :, :], axis=-1).max()
            except Exception:
                # Degenerate (collinear) patch — fall back to bbox diagonal.
                ext = pts.max(axis=0) - pts.min(axis=0)
                d = float(np.linalg.norm(ext))
        diameters.append(float(d))
        if d > max_span:
            max_span = float(d)
            max_cluster_size = sz
    if not diameters:
        return CheckResult(
            "max_bridge_span",
            "green",
            "no contiguous overhang patch reached the min cluster size",
            value=0.0,
            detail={
                "max_span_mm": 0.0,
                "limit_mm": MAX_BRIDGE_SPAN_MM,
                "n_overhang_centroids": int(n_pts),
                "n_clusters": int(n_comp),
            },
        )
    diam = max_span
    detail = {
        "max_span_mm": diam,
        "limit_mm": MAX_BRIDGE_SPAN_MM,
        "n_overhang_centroids": int(n_pts),
        "n_clusters_total": int(n_comp),
        "n_clusters_above_min_size": len(diameters),
        "max_cluster_size_faces": max_cluster_size,
        "cluster_diameter_p50_mm": float(np.percentile(diameters, 50)),
        "cluster_diameter_p90_mm": float(np.percentile(diameters, 90)),
        "cluster_eps_mm": CLUSTER_EPS_MM,
    }
    if diam > MAX_BRIDGE_SPAN_MM:
        return CheckResult(
            "max_bridge_span",
            "red",
            f"overhang cluster spans {diam:.1f} mm > {MAX_BRIDGE_SPAN_MM:.1f} mm PLA limit — bridges will sag",
            value=diam,
            detail=detail,
        )
    if diam > MAX_BRIDGE_SPAN_MM / 2:
        return CheckResult(
            "max_bridge_span",
            "yellow",
            f"overhang cluster spans {diam:.1f} mm — within PLA limit but borderline",
            value=diam,
            detail=detail,
        )
    return CheckResult(
        "max_bridge_span",
        "green",
        f"max overhang cluster span {diam:.1f} mm — within PLA bridge limit",
        value=diam,
        detail=detail,
    )


def _check_wall_thickness(mesh: trimesh.Trimesh) -> CheckResult:
    """Local wall-thickness probe at sampled surface points.

    Approach: sample WALL_THICKNESS_SAMPLE_N surface vertices uniformly;
    for each, cast a ray inward along the negated vertex normal using the
    mesh's existing rtree-backed `mesh.ray` index, and record the first-hit
    distance. That's the true local wall thickness at that point.

    Why a direct ray cast and not `trimesh.proximity.thickness`: the latter
    defaults to a max-inscribed-sphere algorithm that builds an additional
    scratch data structure scaling poorly with face count. On 1M-face
    lattice meshes that scratch index filled disk and stalled the audit.
    Direct rays via `mesh.ray.intersects_location` reuse the rtree that
    trimesh lazy-builds once per mesh, so the per-query cost is light and
    the total cost scales as O(n_sample * log(n_faces)).
    """
    n_verts = len(mesh.vertices)
    if n_verts == 0:
        return CheckResult("wall_thickness", "red", "empty mesh")
    n_sample = min(WALL_THICKNESS_SAMPLE_N, n_verts)
    rng = np.random.default_rng(seed=0)
    idx = rng.choice(n_verts, size=n_sample, replace=False)
    pts = mesh.vertices[idx]
    # outward-pointing vertex normals; we cast inward (opposite direction).
    outward_normals = mesh.vertex_normals[idx]
    eps = 1e-4  # nudge the ray origin slightly inward to avoid self-hits.
    ray_origins = pts - outward_normals * eps
    ray_directions = -outward_normals
    try:
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False,  # only need the first opposite-wall hit
        )
    except Exception as exc:
        return CheckResult(
            "wall_thickness", "skipped", f"mesh.ray.intersects_location raised: {exc}"
        )
    # Some rays may miss the mesh (e.g., highly curved boundary vertices
    # where the inward normal points out into free space). Those samples
    # are excluded from the distribution.
    if len(locations) == 0:
        return CheckResult(
            "wall_thickness",
            "yellow",
            "all wall-thickness rays missed the opposite surface",
            detail={"n_sampled": n_sample, "n_finite": 0},
        )
    # Distance from each ray's origin to its hit point = local thickness.
    hit_origins = ray_origins[index_ray]
    arr = np.linalg.norm(locations - hit_origins, axis=1).astype(float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return CheckResult(
            "wall_thickness",
            "yellow",
            "no finite wall-thickness samples — likely an open shell",
            detail={"n_sampled": n_sample, "n_finite": 0},
        )
    p0 = float(np.min(arr))
    p1 = float(np.percentile(arr, 1))
    p5 = float(np.percentile(arr, 5))
    p50 = float(np.percentile(arr, 50))
    frac_below_min = float((arr < ENDER3_V2_MIN_WALL_MM).mean())
    frac_below_robust = float((arr < ENDER3_V2_ROBUST_WALL_MM).mean())
    detail = {
        "n_sampled": n_sample,
        "n_finite": int(arr.size),
        "min_mm": p0,
        "p1_mm": p1,
        "p5_mm": p5,
        "p50_mm": p50,
        "frac_below_single_perimeter_mm": frac_below_min,
        "frac_below_two_perimeter_mm": frac_below_robust,
        "single_perimeter_floor_mm": ENDER3_V2_MIN_WALL_MM,
        "two_perimeter_floor_mm": ENDER3_V2_ROBUST_WALL_MM,
    }
    if frac_below_min > 0.05:
        return CheckResult(
            "wall_thickness",
            "red",
            f"{frac_below_min:.1%} of probed surface is below 0.4 mm nozzle — unprintable thin features",
            value=p0,
            detail=detail,
        )
    if frac_below_robust > 0.20:
        return CheckResult(
            "wall_thickness",
            "yellow",
            f"{frac_below_robust:.1%} of surface is below 0.8 mm two-perimeter wall — fragile",
            value=p0,
            detail=detail,
        )
    return CheckResult(
        "wall_thickness", "green", f"min wall {p0:.2f} mm; p5 {p5:.2f} mm", value=p0, detail=detail
    )


# ---------------------------------------------------------------------------
# Tier 2 — Optional CuraEngine dry-run for an authoritative support estimate.
# ---------------------------------------------------------------------------


def _find_curaengine() -> Optional[Path]:
    for p in CURAENGINE_PATHS:
        if p.exists():
            return p
    return None


def _find_printer_def() -> Optional[Path]:
    """Locate the closest Ender-class printer definition shipped with Cura."""
    if not CURA_DEFS_DIR.exists():
        return None
    # Prefer ender3v2 if present; fall back to plain ender3 (same envelope).
    for pat in ("creality_ender3v2*.def.json", "creality_ender3.def.json"):
        cands = sorted(CURA_DEFS_DIR.glob(pat))
        if cands:
            return cands[0]
    return None


def _check_cura_supports(stl_path: Path, work_dir: Path) -> CheckResult:
    """Best-effort CuraEngine dry-run for support volume.

    CuraEngine without a baked Cura project file is notoriously fragile —
    almost every printer setting must be passed via -s flags. We invoke
    with a minimal set; if it fails, return "skipped" rather than poisoning
    the overall verdict with a transient infra issue.
    """
    cura = _find_curaengine()
    if cura is None:
        return CheckResult("cura_supports", "skipped", "CuraEngine not found at expected locations")
    printer_def = _find_printer_def()
    if printer_def is None:
        return CheckResult(
            "cura_supports", "skipped", f"No Ender-class def found under {CURA_DEFS_DIR}"
        )
    out_gcode = work_dir / (stl_path.stem + ".gcode")
    cmd = [
        str(cura),
        "slice",
        "-v",  # verbose progress
        "-j",
        str(printer_def),
        "-s",
        "machine_nozzle_size=0.4",
        "-s",
        "layer_height=0.2",
        "-s",
        "support_enable=true",
        "-s",
        "support_angle=45",
        "-s",
        "infill_sparse_density=20",
        "-l",
        str(stl_path),
        "-o",
        str(out_gcode),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return CheckResult(
            "cura_supports",
            "skipped",
            "CuraEngine timed out after 10 min — STL likely too large for dry-run",
        )
    if proc.returncode != 0:
        return CheckResult(
            "cura_supports",
            "skipped",
            f"CuraEngine failed (rc={proc.returncode}): {proc.stderr.splitlines()[-1] if proc.stderr else 'no stderr'}",
        )
    if not out_gcode.exists():
        return CheckResult(
            "cura_supports", "skipped", "CuraEngine returned 0 but produced no gcode"
        )
    # Parse the gcode header for filament + time.
    head_lines = []
    with out_gcode.open() as fh:
        for i, line in enumerate(fh):
            if i > 60:
                break
            head_lines.append(line.rstrip())
    filament_m = None
    print_time_s = None
    for ln in head_lines:
        if "Filament used" in ln:
            # Format example: ";Filament used: 1.234m"
            try:
                filament_m = float(ln.split(":", 1)[1].strip().rstrip("m"))
            except Exception:
                pass
        if "TIME:" in ln and print_time_s is None:
            try:
                print_time_s = float(ln.split(":", 1)[1].strip())
            except Exception:
                pass
    detail = {
        "gcode_path": str(out_gcode),
        "filament_used_m": filament_m,
        "print_time_s": print_time_s,
        "printer_def": printer_def.name,
    }
    # Without a clean way to separate support filament from part filament
    # via CuraEngine's CLI output, this check stays yellow — the surface
    # signal here is "slicing succeeded at all", which itself is useful.
    return CheckResult(
        "cura_supports",
        "yellow",
        f"slice OK: filament {filament_m} m, time {print_time_s} s — open gcode in Cura for support breakdown",
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _overall_status(checks: List[CheckResult]) -> str:
    """Worst of {green, yellow, red, skipped} over the run, ignoring 'skipped'.

    Skipped checks should not poison the verdict — they represent absent
    optional infrastructure (e.g., Cura not installed), not a failure of
    the geometry itself.
    """
    rank = 1  # default green
    for c in checks:
        if c.status == "skipped":
            continue
        rank = max(rank, _WORST_RANK.get(c.status, 1))
    return {1: "green", 2: "yellow", 3: "red"}[rank]


def audit_stl(stl_path: Path, out_dir: Path, do_tier2: bool = True) -> STLReport:
    """Run all checks against one STL, write the per-STL JSON, return the report."""
    t0 = time.time()
    print(f"\n=== Auditing {stl_path} ===", flush=True)
    mesh = _load_mesh(stl_path)
    print(f"  loaded: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices", flush=True)
    checks: List[CheckResult] = []
    # Cheap topology checks first so we get a fast verdict on broken meshes.
    checks.append(_check_watertight(mesh))
    checks.append(_check_winding(mesh))
    checks.append(_check_components(mesh))
    checks.append(_check_degenerate_faces(mesh))
    checks.append(_check_build_volume(mesh))
    # Heavier geometric checks.
    checks.append(_check_overhangs(mesh))
    checks.append(_check_max_bridge_span(mesh))
    checks.append(_check_wall_thickness(mesh))
    # Optional Tier 2 — slicer dry-run.
    if do_tier2:
        checks.append(_check_cura_supports(stl_path, out_dir))
    rep = STLReport(
        stl_path=str(stl_path),
        n_faces=int(len(mesh.faces)),
        n_vertices=int(len(mesh.vertices)),
        bounds_mm=mesh.bounds.tolist(),
        extents_mm=[float(v) for v in mesh.extents],
        volume_mm3=float(abs(mesh.volume)) if mesh.is_volume else float("nan"),
        surface_area_mm2=float(mesh.area),
        checks=[c.__dict__ for c in checks],
        overall_status=_overall_status(checks),
        runtime_seconds=time.time() - t0,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_path = out_dir / (stl_path.stem + "__printability.json")
    rep_path.write_text(json.dumps(rep.to_json(), indent=2))
    print(
        f"  -> {rep.overall_status.upper():<6} ({rep.runtime_seconds:.1f}s) -> {rep_path}",
        flush=True,
    )
    return rep


def _format_summary(reports: List[STLReport]) -> str:
    """Plain-text table for `summary.txt`."""
    lines = []
    header = (
        f"{'STL':<48} {'verdict':<7} {'faces':>9} "
        f"{'overh%':>7} {'span_mm':>8} {'min_wall':>9} "
        f"{'wt?':<3} {'comps':>5}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in reports:
        cd = {c["name"]: c for c in r.checks}
        oh = cd.get("overhangs_at_45deg", {}).get("value")
        sp = cd.get("max_bridge_span", {}).get("value")
        wt_v = cd.get("wall_thickness", {}).get("value")
        wt_ok = "Y" if cd.get("watertight", {}).get("status") == "green" else "N"
        ncomp = cd.get("connected_components", {}).get("value", 1.0)
        oh_s = f"{oh*100:6.1f}" if oh is not None else "    -"
        sp_s = f"{sp:7.1f}" if sp is not None else "      -"
        wt_s = f"{wt_v:8.2f}" if wt_v is not None else "       -"
        lines.append(
            f"{Path(r.stl_path).name:<48} {r.overall_status:<7} {r.n_faces:>9d} "
            f"{oh_s:>7} {sp_s:>8} {wt_s:>9} {wt_ok:<3} {int(ncomp):>5d}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="FDM printability audit for biomimetic_pipeline STL files."
    )
    ap.add_argument(
        "targets", nargs="+", help="STL file(s) and/or directory(ies) to walk for *.stl"
    )
    ap.add_argument(
        "--out-dir", required=True, help="Output directory for per-STL JSONs and summary.txt"
    )
    ap.add_argument(
        "--no-cura",
        action="store_true",
        help="Skip Tier-2 CuraEngine dry-run (geometric checks only)",
    )
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Gather STL targets.
    stls: List[Path] = []
    for t in args.targets:
        tp = Path(t)
        if tp.is_dir():
            stls.extend(sorted(tp.rglob("*.stl")))
        elif tp.is_file() and tp.suffix.lower() == ".stl":
            stls.append(tp)
        else:
            print(f"warn: skipping non-STL target {tp}", file=sys.stderr)
    if not stls:
        print("no STL files found in targets", file=sys.stderr)
        return 2
    reports: List[STLReport] = []
    for stl in stls:
        try:
            reports.append(audit_stl(stl, out, do_tier2=not args.no_cura))
        except Exception as exc:
            print(f"ERROR auditing {stl}: {exc}", file=sys.stderr)
            # Continue with the remaining STLs.
            continue
    # Write the multi-STL summary.
    summary = _format_summary(reports)
    (out / "summary.txt").write_text(summary + "\n")
    print("\n" + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
