"""Subprocess wrapper around stock lattice_mesh.py + voxel-hex builder for
the digital_twin model-type.

Two entry points:
  - run(...): the lattice path. Wraps stock lattice_mesh.py CLI (step file +
    --mesh-size + --junction-refinement-factor + --bridge-elevations). No
    modifications to the stock script.
  - run_for_digital_twin(...): the voxel-hex path. Reads a 1-um plated
    occupancy grid (.npy), downsamples it to a tractable element count,
    and emits one VTK-ordered hexahedron per solid voxel as a gmsh-2.2 .msh.
    Used for the digital_twin model-type, where gmsh STL-to-tet meshing
    fails on the organic merged-rod surface (the SDF marching-cubes output's
    parametrization fragments under gmsh's classifySurfaces). The voxel-hex
    route is the standard method for FEA-ing micro-CT-derived structures
    and produces a single connected, watertight, closed solid mesh that
    feeds directly into the strain solver with element_type="hex".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from biomimetic_pipeline.orchestration.run_context import (
    CAD_ENV,
    SFEPY_ENV,
    STOCK_LATTICE_MESH,
    run_in_env,
)

logger = logging.getLogger(__name__)


@dataclass
class MeshRunResult:
    mesh_path: Path
    log_path: Path
    element_type: str = "tet"  # "tet" for stock lattice mesher, "hex" for voxel


def run(
    step_path: Path,
    mesh_dir: Path,
    mesh_size: float = 0.5,
    junction_refinement_factor: float = 1.0,
    bridge_elevations: Optional[List[float]] = None,
    primary_env: str = SFEPY_ENV,
    fallback_env: str = CAD_ENV,
    timeout: int = 1800,
) -> MeshRunResult:
    """Mesh with Gmsh. The stock pipeline runs gmsh in sfepy_env (that's
    where the gmsh Python module is installed) and falls back to cad_env if
    that fails — we mirror that behavior."""
    step_path = Path(step_path)
    mesh_dir = Path(mesh_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = mesh_dir / "compound_enamel_lattice.msh"
    log_path = mesh_dir / "mesh_run.log"

    args = [
        "python",
        str(STOCK_LATTICE_MESH),
        str(step_path),
        "-o",
        str(mesh_path),
        "--mesh-size",
        str(mesh_size),
    ]
    if junction_refinement_factor and junction_refinement_factor != 1.0:
        args += ["--junction-refinement-factor", str(junction_refinement_factor)]
    if bridge_elevations:
        args += ["--bridge-elevations"] + [str(z) for z in bridge_elevations]

    log_chunks = [f"# cmd\n{' '.join(args)}\n"]
    completed = run_in_env(primary_env, args, cwd=mesh_dir, timeout=timeout)
    log_chunks.append(
        f"# attempt primary_env={primary_env}\n# stdout\n{completed.stdout or ''}\n# stderr\n{completed.stderr or ''}\n"
    )
    if completed.returncode != 0:
        completed = run_in_env(fallback_env, args, cwd=mesh_dir, timeout=timeout)
        log_chunks.append(
            f"# attempt fallback_env={fallback_env}\n# stdout\n{completed.stdout or ''}\n# stderr\n{completed.stderr or ''}\n"
        )
    log_path.write_text("\n".join(log_chunks))
    if completed.returncode != 0:
        raise RuntimeError(f"mesh_runner failed (rc={completed.returncode}). See {log_path}")
    if not mesh_path.exists():
        raise RuntimeError(f"mesh file not produced at {mesh_path}. See {log_path}")

    return MeshRunResult(mesh_path=mesh_path, log_path=log_path)


def load_bridge_elevations(sidecar_path: Path) -> List[float]:
    data = json.loads(Path(sidecar_path).read_text())
    elevs = data.get("bridge_elevations") or []
    return [float(z) for z in elevs]


# ============================================================================
# Voxel-hex meshing for the digital_twin model-type
# ============================================================================
#
# The digital twin is built by digital_twin_sdf_runner as a SINGLE-solid
# capsule-union plus loading plates and is exported as both an STL (for
# visualization) and a 1-um plated occupancy grid (.npy, the FEA input).
# gmsh STL-to-tet meshing fails on the SDF surface (classifySurfaces
# fragments the smooth merged-rod parametrization into thousands of patches),
# so the FEA path mesh-builds directly from the occupancy grid instead.
#
# This is the canonical image-based micro-FE approach: one 8-node hexahedron
# per solid voxel. The mesh is watertight by construction (the occupancy
# grid is a regular Cartesian array), one connected component (validated by
# the plating step which fills enclosed voids and bonds all rods through
# the top/bottom plates), and feeds the strain solver through fea_runner's
# element_type="hex" path with no further surgery.


def run_for_digital_twin(
    occupancy_npy: Path,
    mesh_dir: Path,
    voxel_size_um: float = 1.0,
    downsample_factor: int = 3,
) -> MeshRunResult:
    """Build a gmsh-2.2 hex .msh from a plated occupancy grid.

    Args:
        occupancy_npy: bool ndarray on disk; True voxels are solid material.
        mesh_dir: output directory. The mesh is written as
                  `compound_enamel_lattice.msh` to match the filename the
                  stock compression_test.py expects.
        voxel_size_um: physical size of one input occupancy voxel in
                       micrometres (raw SDF grid is typically 1 um).
        downsample_factor: integer block-mean downsample factor applied to
                           the occupancy grid before meshing. Default 3
                           gives a 3-um element size, which validated as
                           tractable for sfepy's direct solver on the lion
                           digital twin (~134k hex / ~446k DOFs, ~2 h LU
                           factorization on a Mac M-series laptop).

    Returns: MeshRunResult with element_type="hex" so the downstream FEA
             knows to apply the hex transform to the stock problem-def.

    Implementation note: pyvista.ImageData.threshold emits VTK_VOXEL (type
    11) cells that meshio silently DROPS when writing gmsh22, producing
    an empty mesh. We build meshio `hexahedron` cells directly to avoid
    this trap. Node ordering follows the VTK/meshio convention (bottom
    quad CCW from origin, then top quad in the same order) so the hex
    transform in fea_runner._apply_hex_transform interprets edges
    correctly.
    """
    import meshio  # without sfepy_env active, so keep heavy deps local
    import numpy as np  # local import: pipeline modules import this even

    mesh_dir = Path(mesh_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = mesh_dir / "compound_enamel_lattice.msh"
    log_path = mesh_dir / "mesh_run.log"

    log_lines: List[str] = []

    def _log(msg: str) -> None:
        log_lines.append(msg)
        logger.info(msg)

    occ = np.load(str(occupancy_npy)).astype(bool)
    _log(
        f"[hex-mesh] loaded {occupancy_npy} shape={occ.shape} "
        f"voxel_um={voxel_size_um} solid={int(occ.sum()):,}"
    )

    # Block-mean downsample so each output voxel represents a
    # downsample_factor**3 cube of input voxels; majority-true wins.
    ds = int(downsample_factor)
    pad = [(-n) % ds for n in occ.shape]
    occ_p = np.pad(occ, [(0, p) for p in pad])
    NX, NY, NZ = (s // ds for s in occ_p.shape)
    red = occ_p.reshape(NX, ds, NY, ds, NZ, ds).mean(axis=(1, 3, 5)) >= 0.5
    _log(f"[hex-mesh] downsampled x{ds}: grid {red.shape}, " f"{int(red.sum()):,} solid hexes")

    # Corner-node grid (NX+1, NY+1, NZ+1). Flat index ordering must be
    # i + j*nNX + k*nNX*nNY so neighbour hexes share nodes correctly.
    nNX, nNY, nNZ = NX + 1, NY + 1, NZ + 1
    flat = np.arange(nNX * nNY * nNZ)
    points = np.column_stack(
        [
            flat % nNX,
            (flat // nNX) % nNY,
            flat // (nNX * nNY),
        ]
    ).astype(
        float
    ) * (ds * voxel_size_um)

    # VTK / meshio hex node order: bottom quad CCW from origin, then top.
    solid = np.argwhere(red)
    i, j, k = solid[:, 0], solid[:, 1], solid[:, 2]

    def _nid(i, j, k):
        return i + j * nNX + k * nNX * nNY

    hexes = np.column_stack(
        [
            _nid(i, j, k),
            _nid(i + 1, j, k),
            _nid(i + 1, j + 1, k),
            _nid(i, j + 1, k),
            _nid(i, j, k + 1),
            _nid(i + 1, j, k + 1),
            _nid(i + 1, j + 1, k + 1),
            _nid(i, j + 1, k + 1),
        ]
    )

    # Strip unreferenced nodes (most of the corner grid is empty interior
    # of voids); without this the mesh carries millions of unused nodes.
    used = np.unique(hexes)
    remap = np.full(points.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    hexes = remap[hexes]
    points = points[used]

    m = meshio.Mesh(points=points, cells=[("hexahedron", hexes)])
    meshio.write(str(mesh_path), m, file_format="gmsh22", binary=False)
    _log(f"[hex-mesh] wrote {mesh_path}: {len(hexes):,} hex elements, " f"{len(points):,} nodes")

    log_path.write_text("\n".join(log_lines) + "\n")

    return MeshRunResult(mesh_path=mesh_path, log_path=log_path, element_type="hex")
