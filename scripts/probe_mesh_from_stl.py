#!/usr/bin/env python
"""
probe_mesh_from_stl.py — volume-mesh a watertight STL directly via gmsh.

Purpose
-------
Diagnostic probe: attempt to tetrahedralize a CAD solid from its TRIANGULATED
STL surface instead of the exact STEP/BREP. The stock pipeline meshes the STEP
(exact OpenCASCADE geometry); for thick N=9 that path yields 0 tets — a
boolean/tangency degeneracy in the *exact* BREP (a self-intersection or
zero-thickness sliver that gmsh's 3D mesher can't fill, even though the surface
is closed/watertight). A discrete STL surface is an approximation that drops the
exact degeneracy, so it MAY tetrahedralize where the BREP won't. This is the
"cheap N=9 fix" the user chose before considering the heavier SDF path.

Why this exists
---------------
Stock `cad_modeling/.../lattice_mesh.py` only meshes the STEP (and is read-only).
This is a NEW, separate probe — it does not touch the stock module or the
pipeline. If it produces a real volume mesh, the next step is wiring that mesh
into the FEA stage for N=9; if not, we fall back (accept 11/12 or SDF).

Inputs (CLI)
------------
  --stl        : path to the (watertight) STL to volume-mesh.
  --out        : output .msh path (written only if tets are produced).
  --mesh-size  : target max element size (mm). Sweep default is 0.5.
  --angle      : feature-classification angle (deg); edges sharper than this
                 split the surface into patches (default 40, a common value).

Outputs / exit code
-------------------
  Prints surface/volume element counts. Writes <out> ONLY if >0 tets.
  Exit 0 iff a usable volume mesh (>0 tets) was produced; else exit 1.

Side effects: writes the .msh on success; reads only the STL. Must run in an
env that has gmsh (sfepy_env).
"""
from __future__ import annotations

import argparse
import math
import sys


def main() -> int:
    # --- 1. Parse CLI ------------------------------------------------------
    ap = argparse.ArgumentParser(description="Volume-mesh a watertight STL via gmsh (STL->volume).")
    ap.add_argument("--stl", required=True, help="Input watertight STL.")
    ap.add_argument("--out", required=True, help="Output .msh (written only on success).")
    ap.add_argument("--mesh-size", type=float, default=0.5, help="Max element size (mm).")
    ap.add_argument("--angle", type=float, default=40.0, help="Surface-classification angle (deg).")
    args = ap.parse_args()

    import gmsh  # imported here so --help works without gmsh on PATH

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)

        # --- 2. Load the triangulated surface (the STL as a discrete mesh) ---
        gmsh.merge(args.stl)

        # --- 3. Classify triangles into surface patches ----------------------
        # classifySurfaces(angle_rad, boundary, forReparametrization, curveAngle):
        #   angle_rad           : dihedral threshold separating smooth patches.
        #   boundary=True       : keep patch boundary edges as model curves.
        #   forReparametrization=True : REQUIRED so createGeometry() can build a
        #                         spline parametrization each patch can be meshed on.
        #   curveAngle=pi       : don't over-split curved patches.
        ang = args.angle * math.pi / 180.0
        gmsh.model.mesh.classifySurfaces(ang, True, True, math.pi)

        # --- 4. Build CAD-like geometry from the classified discrete surface --
        gmsh.model.mesh.createGeometry()

        # --- 5. Define a single volume bounded by ALL surface patches --------
        surfs = [e[1] for e in gmsh.model.getEntities(2)]
        n2_patches = len(surfs)
        loop = gmsh.model.geo.addSurfaceLoop(surfs)
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()

        # --- 6. Tetrahedralize ----------------------------------------------
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5 * args.mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", args.mesh_size)
        gmsh.model.mesh.generate(3)

        # gmsh element type id 4 == 4-node tetrahedron.
        tet_tags = gmsh.model.mesh.getElementsByType(4)[0]
        n_tets = len(tet_tags)
        print(f"surface patches classified: {n2_patches}")
        print(f"3D tetrahedra: {n_tets}")

        # --- 7. Report / write ----------------------------------------------
        if n_tets > 0:
            gmsh.write(args.out)
            print(f"SUCCESS — wrote {args.out}")
            return 0
        print("FAIL — 0 tets; STL volume meshing also degenerate")
        return 1
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    sys.exit(main())
