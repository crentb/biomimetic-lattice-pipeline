#!/usr/bin/env python
"""
analyze_lattice_defect.py
=========================

Purpose
-------
Localize WHERE a non-watertight lattice STL is broken, to find the geometric
mechanism behind the high-N OCCT fusion failures (rather than dismissing them
as generic "OCCT instability"). Loads a compound lattice STL, finds the
topologically broken edges (open boundary edges = holes; non-manifold edges =
shared by >2 faces), and reports their distribution in Z (height) and radius
(distance from the lattice axis), overlaying the known BRIDGE_Z_OFFSETS.

If the broken edges cluster at the bridge Z-elevations -> the defect is at the
bridge<->rod (or bridge<->bridge) junctions. If they cluster at the outermost
radius -> it's the outer-ring rods/plate rim. Either way it pinpoints the
feature OCCT failed on.

Why this exists
---------------
The cad_integrity check only reports a boolean "watertight: False". To debug
the cause we need the spatial location of the defect, which this script
extracts directly from the mesh topology.

Inputs (CLI)
------------
  --stl          : path to the compound lattice STL to analyze.
  --cad-params   : optional cad_params.json to read BRIDGE_Z_OFFSETS for the
                   Z overlay (defaults next to the STL's trial dir).

Outputs
-------
  stdout : watertight status, broken-edge counts, Z-histogram (with bridge
           layers flagged), radius-histogram.

Side effects: none (read-only analysis).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from trimesh.grouping import group_rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Localize broken edges in a non-watertight lattice STL."
    )
    ap.add_argument("--stl", required=True, type=Path)
    ap.add_argument("--cad-params", type=Path, default=None)
    args = ap.parse_args()

    # --- 1. Load mesh (process=False to keep raw topology, not auto-merged) ---
    m = trimesh.load(args.stl, process=False)
    print(f"STL: {args.stl}")
    print(f"  vertices={len(m.vertices):,}  faces={len(m.faces):,}")
    print(f"  watertight={m.is_watertight}  euler={m.euler_number}  volume={m.volume:.1f} mm^3")
    print(f"  bounds Z=[{m.bounds[0][2]:.2f}, {m.bounds[1][2]:.2f}] mm")

    # --- 2. Find broken edges by face-incidence count ------------------------
    # Each undirected edge should be shared by exactly 2 faces in a closed
    # manifold. Edges in exactly 1 face = open boundary (a hole). Edges in
    # >2 faces = non-manifold (surfaces meeting along a seam). Both break
    # watertightness and both choke a volume mesher.
    edges = np.sort(m.edges, axis=1)  # undirected
    boundary_idx = group_rows(edges, require_count=1)  # appear once -> hole rim
    # Non-manifold: appear 3+ times. group_rows(require_count=None) groups all;
    # we count group sizes.
    groups = group_rows(edges)  # list of index-arrays per unique edge
    nonmanifold_groups = [g for g in groups if len(g) > 2]
    nm_idx = (
        np.array([g[0] for g in nonmanifold_groups], dtype=int)
        if nonmanifold_groups
        else np.array([], dtype=int)
    )

    print(f"\n  open boundary edges (holes):   {len(boundary_idx):,}")
    print(f"  non-manifold edges (>2 faces): {len(nm_idx):,}")

    # Collect the midpoints of all broken edges for spatial localization.
    broken = []
    if len(boundary_idx):
        broken.append(edges[boundary_idx])
    if len(nm_idx):
        broken.append(edges[nm_idx])
    if not broken:
        print("\n  No broken edges found by incidence count (defect may be a self-")
        print("  intersection rather than a hole/non-manifold edge).")
        return 0
    broken_edges = np.vstack(broken)
    mids = m.vertices[broken_edges].mean(axis=1)  # (n,3) edge midpoints
    z = mids[:, 2]
    r = np.hypot(mids[:, 0], mids[:, 1])

    # --- 3. Load bridge Z-offsets for the overlay ---------------------------
    offsets = None
    cad_params = args.cad_params
    if cad_params is None:
        # Default: trial dir is two levels up from cad/<stl>.
        guess = args.stl.parent.parent / "cad_params.json"
        if guess.is_file():
            cad_params = guess
    if cad_params and cad_params.is_file():
        p = json.loads(cad_params.read_text())
        offsets = p.get("BRIDGE_Z_OFFSETS")

    # --- 4. Z-histogram of broken edges, with bridge layers flagged ---------
    print("\n  Broken-edge Z-distribution (mm):")
    zmin, zmax = z.min(), z.max()
    nbins = 22
    counts, edges_z = np.histogram(z, bins=nbins, range=(zmin, zmax))
    for c, z0, z1 in zip(counts, edges_z[:-1], edges_z[1:]):
        bar = "#" * int(60 * c / max(counts.max(), 1))
        flag = ""
        if offsets:
            # Mark bins that contain a bridge layer elevation.
            if any(z0 <= zo < z1 for zo in offsets):
                flag = "  <-- bridge layer"
        print(f"    z[{z0:5.2f},{z1:5.2f})  {c:6d} {bar}{flag}")

    # --- 5. Radius distribution (outer ring vs interior) --------------------
    print("\n  Broken-edge radius distribution (mm from axis):")
    rc, re = np.histogram(r, bins=12, range=(0, r.max()))
    for c, r0, r1 in zip(rc, re[:-1], re[1:]):
        bar = "#" * int(60 * c / max(rc.max(), 1))
        print(f"    r[{r0:5.2f},{r1:5.2f})  {c:6d} {bar}")

    if offsets:
        print(f"\n  BRIDGE_Z_OFFSETS: {[round(o,3) for o in offsets]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
