#!/usr/bin/env python3
"""
generators/cad_integrity.py
============================

Runtime CAD-integrity checks invoked AFTER every parametric lattice CAD
generation (continuous_twist model-type) and BEFORE the mesh + FEA
stages, so that silent geometry failures are caught before they propagate
into hours of wasted compute.

Three checks are run on the emitted STL against the expected geometry:

(1) **Bridge presence** -- slice the STL horizontally at every
    BRIDGE_Z_OFFSETS value (the elevations the CAD was asked to place
    horizontal bridges at) and confirm the cross-section area at each
    slice is well above the rod-only baseline. The motivating bug is
    the OpenCASCADE silent-drop at N_BRIDGE_LAYERS = 8 with the
    post-fix pipeline defaults (see
    mapping/bridge_mappers.OCCT_TANGENT_JITTER_MM and memory
    project_occt_bridge_tangent_volume_loss.md): every bridge layer
    in the input is silently dropped from the OCCT boolean fuse but the
    STL is still topologically watertight (rods + plates only). The
    bridge-presence check is the only diagnostic that catches this.

(2) **Rod count** -- slice the STL at a rod-only baseline z (below the
    first bridge, above the bottom plate) and count closed polygon loops
    whose area exceeds a per-rod threshold. The expected count follows
    from the hexagonal-ring geometry of the lattice:
        N_rods = 1 (central rod) + 6 * (1 + 2 + ... + N_RINGS)
               = 1 + 3 * N_RINGS * (N_RINGS + 1)
    For N_RINGS = 5: 1 + 3 * 5 * 6 = 91 rods.
    Catches silently merged rods (paired tangent cylinders that OCCT
    fuses into one solid) or silently dropped rods.

(3) **Watertight** -- trimesh.is_watertight on the loaded STL. Catches
    boolean-fusion topological defects that escape (1) and (2). A
    not-watertight STL is split into a REAL hole (open boundary edges =
    missing surface -> ALWAYS fails) versus a benign NON-MANIFOLD-only
    artifact (edges shared by >2 faces from OCCT's STL tessellation of
    near-tangent surfaces, e.g. the 0.025 mm thick rod-rod contacts; 0
    open holes). The latter is tolerated (warning, not failure) when the
    opt-in env var CAD_TOLERATE_NONMANIFOLD_STL is set, because gmsh meshes
    the STEP (not the STL) and a non-manifold-only STL still meshes cleanly.

A failure in any of the three populates `IntegrityReport.failures` and
sets `.passed = False`. The orchestration layer
(orchestration/pipeline.py) calls `raise_if_failed()` immediately after
the CAD stage; if any check failed and `allow_broken_cad` is False
(the default), a CADIntegrityError is raised before the mesh stage
starts. Use the --allow-broken-cad flag on the run_*.py CLIs (or the
allow_broken_cad=True kwarg on run_pipeline/run_sweep) for diagnostic
runs that intentionally produce broken geometry.

The per-run report is written to <run_dir>/cad_integrity_report.json
for post-mortem inspection regardless of pass / fail.

NON-MANIFOLD-STL TOLERANCE (added 2026-06-03 -- full story in
runs/sweep_H24_thick/WHY_THICK_MESHABILITY_H24.txt, Part B):
    Check 3 splits a not-watertight STL into a REAL hole (open boundary edges
    > 0 = genuinely missing surface -> ALWAYS fails) versus a benign
    NON-MANIFOLD-only artifact (0 open holes; a few edges shared by >2
    triangles, produced by OCCT's STL tessellation of near-tangent surfaces --
    e.g. the 0.025 mm thick rod-rod contacts at ENAMEL_THICKNESS = 24 mm). The
    latter is downgraded to a WARNING when the opt-in env var
    CAD_TOLERATE_NONMANIFOLD_STL is truthy (default behaviour is unchanged).

    WHY THIS IS SAFE / CORRECT. gmsh meshes the STEP (the exact B-rep), NOT the
    STL, so a non-manifold-only STL still meshes cleanly -- verified 2026-06-03:
    thick N=4 @ H=24 had a non-watertight STL (7 non-manifold edges, 0 holes)
    yet its STEP meshed to 1,551,256 tets. The STL-watertight check is therefore
    a FALSE-NEGATIVE proxy for meshability on near-tangent geometries.

    CONSEQUENCES -- this tolerance CHANGES NO GEOMETRY. It edits a CHECK, not
    CAD generation: the STEP/STL files are byte-for-byte unchanged. The FEA mesh
    is derived from the STEP, so every result is from the exact geometry and is
    unaffected. The STL keeps its non-manifold edges (we stop discarding the
    run over them, not the edges themselves); they are sub-print-resolution
    (0.025 mm contacts << ~0.3 mm SLA, so those rods fuse on a real print and
    slicers auto-repair minor non-manifold edges). Real protection is preserved:
    dropped bridges / missing rods (Checks 1 & 2) and real holes (open edges)
    still hard-fail, and the mesh stage (0 tets) is the final backstop. The
    clean long-term fix is to gate on the STEP/mesh instead of the STL, after
    which this flag can retire.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import trimesh

# ----------------------------------------------------------------------
# Tunable thresholds. Chosen conservatively for the current geometry at
# the post-clearance-fix pipeline defaults (ROD_DIAMETER ~ 2.128 mm,
# BRIDGE_DIAMETER ~ 1.702 mm, N_RINGS = 5 -> 91 rods, ENAMEL_THICKNESS
# = 20 mm). Tighten only if false-positives appear in normal runs; do
# NOT relax to suppress true positives (the whole point is to fail
# fast on broken geometry).
# ----------------------------------------------------------------------

# Bridge-bearing slice area is ~3,000-3,500 mm² (rods + 84 inter-rod
# bridges); rod-only slice area is ~300-330 mm². The 1,000 mm² threshold
# sits comfortably between the two regimes and is robust to small
# geometric perturbations from sweeps.
BRIDGE_LAYER_MIN_AREA_MM2: float = 1000.0

# Single rod cross-section at z = rod-only baseline is pi * (rod_radius)²
# ~= 3.5 mm² at the default rod_diameter = 2.128 mm. The 1.0 mm² floor
# accepts loops down to a rod-radius of ~0.56 mm (smaller than any
# plausible rod) while filtering trimesh-sectioning numerical artifacts
# (which can leave tiny degenerate loops near plate corners at < 0.1 mm²).
ROD_MIN_AREA_MM2: float = 1.0

# z position in mm at which to slice the STL for the rod-only count.
# Sits above the bottom plate inner face (z = 0 to ~1.2 mm with
# PLATE_OVERLAP) and below the lowest possible bridge centerline at
# safe_z_min = 1.851 mm under the current geometric defaults. Empirically
# verified during the 2026-05-27 probe sequence to return rod-only area
# of ~322.9 mm² (= 91 rods x ~3.55 mm²/rod). Hardcoded for the current
# 20 mm-thick lattice; if ENAMEL_THICKNESS or PLATE_OVERLAP change
# materially, this constant should be re-derived.
ROD_BASELINE_Z_MM: float = 0.6


# ----------------------------------------------------------------------
# Public exception type
# ----------------------------------------------------------------------


class CADIntegrityError(Exception):
    """Raised when a generated CAD lattice fails its runtime integrity
    checks. Carries the structured report as ``.report`` for downstream
    logging / metrics aggregation."""

    def __init__(self, message: str, report: Dict[str, Any]):
        super().__init__(message)
        self.report: Dict[str, Any] = report


# ----------------------------------------------------------------------
# Report structure -- serialisable to JSON for the per-run report file.
# ----------------------------------------------------------------------


@dataclass
class IntegrityReport:
    """Aggregate result of all three integrity checks. ``passed`` is True
    only if every individual check passed; ``failures`` is a list of
    human-readable strings, one per check that failed (empty if passed).
    ``warnings`` holds non-fatal notes (e.g. a non-manifold-only STL that was
    tolerated -- see Check 3); ``watertight_detail`` records the open-boundary
    vs non-manifold edge counts when the STL is not watertight."""

    passed: bool = True
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    bridge: Dict[str, Any] = field(default_factory=dict)
    rods: Dict[str, Any] = field(default_factory=dict)
    watertight: bool = False
    watertight_detail: Dict[str, Any] = field(default_factory=dict)
    stl_volume_mm3: float = 0.0
    stl_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view, JSON-serialisable."""
        return asdict(self)


# ----------------------------------------------------------------------
# Geometry helpers (private)
# ----------------------------------------------------------------------


def _polygon_area_mm2(xy: np.ndarray) -> float:
    """Shoelace area in mm² of a closed 2-D polygon.

    Parameters
    ----------
    xy : np.ndarray, shape (n_pts, 2)
        Ordered vertex coordinates of a single closed planar loop. The
        first and last vertices need not be identical; the formula is
        cyclic-safe so long as the loop is closed in topological terms.

    Returns
    -------
    float
        Absolute polygon area in mm² (unsigned; orientation-independent).
    """
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1])))


def _section_polygons(mesh: trimesh.Trimesh, z: float) -> List[np.ndarray]:
    """Return a list of closed-loop (n_pts, 2) arrays at z = const.

    Each entry is one independent polygon traced out by the intersection
    of the horizontal plane with ``mesh``; degenerate loops with < 3
    vertices are dropped. Returns an empty list if the section is empty
    (e.g., the plane sits above the top plate or below the bottom).
    """
    sect = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if sect is None:
        return []
    return [p[:, :2] for p in sect.discrete if p.shape[0] >= 3]


def _open_and_nonmanifold_edge_counts(mesh: trimesh.Trimesh) -> tuple:
    """Return ``(n_open_boundary_edges, n_non_manifold_edges)`` for ``mesh``.

    An undirected edge shared by exactly ONE face is an OPEN boundary edge
    (a hole rim = genuinely missing surface). Shared by MORE than two faces
    it is NON-MANIFOLD -- almost always an OCCT STL-tessellation overlap at
    near-tangent surfaces (e.g. the 0.025 mm rod-rod contacts of the thick
    variant). ``is_watertight`` is False if either is present, but only OPEN
    edges mean the surface is actually broken; a non-manifold-only STL is a
    benign export artifact -- the STEP (which gmsh actually meshes) is closed
    and meshes cleanly (verified 2026-06-03: thick N=4 @ H=24 STL had 7
    non-manifold edges / 0 open holes, yet its STEP meshed to 1.55M tets).
    """
    # Undirected edges: 3 per face; count how many faces share each.
    es = np.sort(mesh.edges, axis=1)
    _, counts = np.unique(es, axis=0, return_counts=True)
    n_open = int((counts == 1).sum())  # hole rims (real missing surface)
    n_non_manifold = int((counts > 2).sum())  # >2 faces on an edge (tessellation overlap)
    return n_open, n_non_manifold


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def verify_cad_integrity(
    stl_path: Path,
    bridge_z_offsets: List[float],
    n_rings: int,
    *,
    bridge_min_area_mm2: float = BRIDGE_LAYER_MIN_AREA_MM2,
    rod_min_area_mm2: float = ROD_MIN_AREA_MM2,
    rod_baseline_z_mm: float = ROD_BASELINE_Z_MM,
) -> IntegrityReport:
    """Run all three CAD integrity checks against the emitted STL.

    Parameters
    ----------
    stl_path : Path
        Path to the STL produced by cad_runner.run().
    bridge_z_offsets : List[float]
        The bridge centerline elevations (mm) the CAD was asked to place;
        each is sliced to verify the bridge actually fused.
    n_rings : int
        Number of hexagonal rings of rods, used to compute the expected
        rod count. The lattice convention is N_RINGS rings AROUND a
        central rod, so the total rod count is 1 + 3 * N_RINGS * (N_RINGS + 1).
    bridge_min_area_mm2 : float, optional
        Threshold above which a slice is considered "bridge present".
        Defaults to BRIDGE_LAYER_MIN_AREA_MM2.
    rod_min_area_mm2 : float, optional
        Per-loop area floor for rod counting (filters numerical artifacts).
        Defaults to ROD_MIN_AREA_MM2.
    rod_baseline_z_mm : float, optional
        z slice height for rod-only counting. Defaults to ROD_BASELINE_Z_MM
        (0.6 mm); change only if the lattice gauge layout changes.

    Returns
    -------
    IntegrityReport
        Structured report with one dict per check plus aggregate flags.
        Does NOT raise -- caller chooses whether to abort (see
        ``raise_if_failed``).
    """
    report = IntegrityReport(stl_path=str(stl_path))

    # Loading the STL is the single most expensive step (~1-3 s for a
    # 2M-face mesh). All three checks share the same loaded mesh.
    mesh = trimesh.load(str(stl_path))
    report.stl_volume_mm3 = float(mesh.volume)
    report.watertight = bool(mesh.is_watertight)

    # ---- Check 1: bridge presence ----
    n_expected_bridges = len(bridge_z_offsets)
    bridge_areas: List[float] = []
    for z in bridge_z_offsets:
        polys = _section_polygons(mesh, float(z))
        area = sum(_polygon_area_mm2(p) for p in polys)
        bridge_areas.append(round(area, 3))
    n_present_bridges = sum(1 for a in bridge_areas if a > bridge_min_area_mm2)
    bridge_passed = n_present_bridges == n_expected_bridges
    report.bridge = {
        "n_expected": int(n_expected_bridges),
        "n_present": int(n_present_bridges),
        "per_layer_area_mm2": bridge_areas,
        "threshold_area_mm2": float(bridge_min_area_mm2),
        "passed": bool(bridge_passed),
    }
    if not bridge_passed:
        # Identify which layers dropped, by index, so the post-mortem can
        # cross-reference against the BRIDGE_Z_OFFSETS list.
        missing_idx = [i for i, a in enumerate(bridge_areas) if a <= bridge_min_area_mm2]
        report.failures.append(
            f"bridge_presence: {n_present_bridges}/{n_expected_bridges} bridges "
            f"present (missing layer indices {missing_idx}; "
            f"area threshold = {bridge_min_area_mm2:.1f} mm^2)"
        )

    # ---- Check 2: rod count ----
    # Hexagonal-ring closed form. Falls back to a single rod if n_rings <= 0
    # (an edge case the lattice generator does not actually emit).
    n_rings_int = max(0, int(n_rings))
    n_expected_rods = 1 + 3 * n_rings_int * (n_rings_int + 1)
    rod_polys = _section_polygons(mesh, float(rod_baseline_z_mm))
    rod_loops_above = [p for p in rod_polys if _polygon_area_mm2(p) > rod_min_area_mm2]
    n_present_rods = len(rod_loops_above)
    rods_passed = n_present_rods == n_expected_rods
    report.rods = {
        "n_expected": int(n_expected_rods),
        "n_present": int(n_present_rods),
        "baseline_z_mm": float(rod_baseline_z_mm),
        "n_rings": int(n_rings_int),
        "loop_areas_mm2": [round(_polygon_area_mm2(p), 3) for p in rod_polys],
        "threshold_area_mm2": float(rod_min_area_mm2),
        "passed": bool(rods_passed),
    }
    if not rods_passed:
        report.failures.append(
            f"rod_count: {n_present_rods}/{n_expected_rods} rods present at "
            f"z = {rod_baseline_z_mm:.3f} mm (loop-area threshold = "
            f"{rod_min_area_mm2:.2f} mm^2; n_rings = {n_rings_int})"
        )

    # ---- Check 3: watertight ----
    # report.watertight is is_watertight (set above). If False, distinguish a
    # REAL hole (open boundary edges -> missing surface) from a benign OCCT
    # STL-tessellation artifact (NON-MANIFOLD-only: edges shared by >2 faces at
    # near-tangent rod contacts, with 0 open holes). The pipeline meshes the
    # STEP, not the STL, so a non-manifold-only STL still meshes cleanly. When
    # the opt-in env var CAD_TOLERATE_NONMANIFOLD_STL is set (truthy), such a
    # case is downgraded to a WARNING; a real hole (open edges > 0) ALWAYS
    # fails. Bridge/rod presence are checked independently above, so a dropped
    # bridge or missing rod still fails regardless of this tolerance.
    if not report.watertight:
        n_open, n_non_manifold = _open_and_nonmanifold_edge_counts(mesh)
        report.watertight_detail = {
            "open_boundary_edges": int(n_open),
            "non_manifold_edges": int(n_non_manifold),
        }
        _flag = os.environ.get("CAD_TOLERATE_NONMANIFOLD_STL", "").strip().lower()
        tolerate_nonmanifold = _flag not in ("", "0", "false", "no")
        msg = (
            f"watertight: STL is not watertight "
            f"(open_boundary_edges={n_open}, non_manifold_edges={n_non_manifold})"
        )
        if n_open == 0 and tolerate_nonmanifold:
            # Non-manifold-only + opt-in flag: benign tessellation artifact; the
            # STEP meshes. Record as a warning, do NOT fail the integrity gate.
            report.warnings.append(
                msg + " -- TOLERATED (non-manifold-only; STEP meshes; "
                "CAD_TOLERATE_NONMANIFOLD_STL set)"
            )
        else:
            report.failures.append(msg)

    report.passed = len(report.failures) == 0
    return report


def raise_if_failed(report: IntegrityReport) -> None:
    """Raise a CADIntegrityError with the structured report attached if
    any check in ``report`` failed. Use this immediately after
    ``verify_cad_integrity()`` to abort the pipeline before the mesh
    stage; pass ``allow_broken_cad=True`` upstream to skip this call."""
    if not report.passed:
        raise CADIntegrityError(
            "CAD integrity check failed: " + " | ".join(report.failures),
            report=report.to_dict(),
        )
