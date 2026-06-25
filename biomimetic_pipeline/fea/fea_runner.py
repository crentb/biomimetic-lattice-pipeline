"""Subprocess wrapper around stock compression_test.py.

Contract enforced by stock compression_test.py:
  - Invoked via `sfepy-run`, NOT `python`.
  - Mesh filename MUST be `compound_enamel_lattice.msh` in the CWD.
  - Reads env vars MATERIAL_E, MATERIAL_NU, COMPRESS_DISP_MM.
  - Writes a fixed set of CSVs / VTK into the CWD.

We create a fresh working directory per iteration, copy the mesh in, and run
sfepy there. The mesh is held constant across iterations of the strain solver
(Phase 2) — only COMPRESS_DISP_MM changes.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from biomimetic_pipeline.orchestration.run_context import (
    SFEPY_ENV,
    STOCK_COMPRESSION_TEST,
    _find_conda,
)

FEA_MESH_FILENAME = "compound_enamel_lattice.msh"


def output_files_for(load_mode: str = "compression") -> tuple:
    """Fixed set of result files one FEA iteration emits, for a given load case.

    The stock problem-definition script suffixes its CSV/TXT outputs with the
    load case ("compression"/"tension"); the VTK field dump is named after the
    mesh and is therefore load-case-agnostic.
    """
    return (
        f"element_results_{load_mode}.csv",
        f"global_results_{load_mode}.csv",
        f"force_displacement_{load_mode}.csv",
        f"reaction_force_bottom_z_{load_mode}.txt",
        "compound_enamel_lattice.vtk",
    )


# Backwards-compatible module-level default (compression) for existing importers.
FEA_OUTPUT_FILES = output_files_for("compression")


@dataclass
class FeaRunResult:
    iter_dir: Path
    global_results_path: Path
    element_results_path: Path
    avg_von_mises_mpa: float
    avg_sigma_zz_mpa: float
    force_N: float
    log_path: Path


def _prepare_problem_def(
    iter_dir: Path,
    load_mode: str,
    element_type: str = "tet",
) -> Path:
    """Write the sfepy problem-definition script into ``iter_dir``.

    The stock ``compression_test.py`` is the single source of truth for the FEA
    physics and post-processing. For the "compression" load case it is used
    verbatim. For "tension" the problem definition is derived from that same
    stock file by flipping the one sign that sets the loading direction (the
    stock script hard-codes a downward, compressive top-face displacement) and
    re-suffixing the four CSV/TXT output filenames. Deriving tension at runtime
    -- rather than maintaining a ~500-line duplicate script -- guarantees the
    physics and post-processing can never drift between the two load cases.

    ``element_type`` (added 2026-05-23 for the digital_twin model-type) selects
    the cell-volume / centroid kernel used inside the stock post-processor:
      - "tet": stock script is used verbatim; its compute_element_volumes /
        compute_element_centroids assume 4-node tetrahedra.
      - "hex": the two tet kernels are swapped at runtime for 8-node hexahedron
        equivalents (parallelepiped-volume formula + arithmetic centroid).
        Validated by /tmp/twin_fea_gate.py against the gate output on
        2026-05-23 (134,143-hex plated digital-twin mesh, exit 0, 7046 s).
    No upstream stock file is modified -- the hex variant is derived in
    the same runtime-string-substitution way that the tension variant is.
    """
    stock_text = Path(STOCK_COMPRESSION_TEST).read_text()
    if load_mode == "compression":
        problem_text = stock_text
    elif load_mode == "tension":
        # Stock: `compress_disp = -abs(_compress_disp_mm)` (always compressive).
        # Tension applies the same magnitude as an upward (+z) pull.
        flipped = stock_text.replace(
            "compress_disp = -abs(_compress_disp_mm)",
            "compress_disp = abs(_compress_disp_mm)",
        )
        if flipped == stock_text:
            raise RuntimeError(
                "tension transform failed: expected displacement-sign line not "
                f"found in {STOCK_COMPRESSION_TEST} -- stock script may have changed."
            )
        # Re-suffix the four result files so a tension run is self-describing.
        problem_text = flipped.replace("_compression.csv", "_tension.csv").replace(
            "_compression.txt", "_tension.txt"
        )
    else:
        raise ValueError(f"load_mode must be 'compression' or 'tension', got {load_mode!r}")

    # Hex transform: swap the two tet-specific element-kernel functions for
    # their 8-node-hex equivalents. Required when the mesh is a voxel-hex
    # build (e.g. the plated digital twin) because the stock kernels
    # .reshape(n_cells, 4) the connectivity and would crash on hex (8 nodes
    # per cell). The parallelepiped-volume formula handles axis-aligned
    # voxel hexes exactly and any general hex via the triple-product of three
    # edges out of corner 0.
    if element_type == "hex":
        problem_text = _apply_hex_transform(problem_text)
    elif element_type != "tet":
        raise ValueError(f"element_type must be 'tet' or 'hex', got {element_type!r}")

    problem_path = iter_dir / f"{load_mode}_test.py"
    problem_path.write_text(problem_text)
    return problem_path


# Tet-only blocks from stock compression_test.py that must be swapped for
# 8-node-hex equivalents when feeding the FEA a voxel-hex mesh. The match
# strings are exact; if the stock script changes, this will fail loudly via
# the assert in _apply_hex_transform.
_TET_VOLUMES_BLOCK = '''def compute_element_volumes(problem):
    """
    Compute volume of each tetrahedral element (vectorized).
    Returns array of element volumes matching the cell ordering.
    """
    coors = problem.domain.get_mesh_coors()
    cmesh = problem.domain.cmesh

    conn = cmesh.get_conn(3, 0)  # 3D cells -> vertices
    n_cells = len(conn.offsets) - 1
    tet_conn = conn.indices.reshape(n_cells, 4)

    v0 = coors[tet_conn[:, 0]]
    v1 = coors[tet_conn[:, 1]]
    v2 = coors[tet_conn[:, 2]]
    v3 = coors[tet_conn[:, 3]]

    d1 = v1 - v0
    d2 = v2 - v0
    d3 = v3 - v0

    cross = nm.cross(d2, d3)
    det = nm.sum(d1 * cross, axis=1)
    volumes = nm.abs(det) / 6.0

    return volumes'''

_HEX_VOLUMES_BLOCK = '''def compute_element_volumes(problem):
    """
    Compute volume of each 8-node hexahedral element (vectorized).
    Hex transform inserted at runtime by biomimetic_pipeline.fea.fea_runner
    when element_type="hex" (e.g. for voxel-hex digital-twin meshes).

    VTK / meshio hex node order: 0,1,2,3 = bottom quad CCW from origin,
    4,5,6,7 = top quad in the same order. The three edges out of corner 0
    are (v1-v0), (v3-v0), (v4-v0); their triple product is the hex volume
    (exact for any parallelepipedal hex including axis-aligned voxel hex).
    """
    coors = problem.domain.get_mesh_coors()
    cmesh = problem.domain.cmesh

    conn = cmesh.get_conn(3, 0)  # 3D cells -> vertices
    n_cells = len(conn.offsets) - 1
    hex_conn = conn.indices.reshape(n_cells, 8)

    v0 = coors[hex_conn[:, 0]]
    v1 = coors[hex_conn[:, 1]]
    v3 = coors[hex_conn[:, 3]]
    v4 = coors[hex_conn[:, 4]]

    e_i = v1 - v0   # local x edge
    e_j = v3 - v0   # local y edge
    e_k = v4 - v0   # local z edge

    cross = nm.cross(e_j, e_k)
    det = nm.sum(e_i * cross, axis=1)
    volumes = nm.abs(det)

    return volumes'''

_TET_CENTROIDS_BLOCK = '''def compute_element_centroids(problem):
    """
    Compute centroid (x, y, z) of each tetrahedral element.
    Returns array of shape (n_elements, 3).
    """
    coors = problem.domain.get_mesh_coors()
    cmesh = problem.domain.cmesh

    conn = cmesh.get_conn(3, 0)
    n_cells = len(conn.offsets) - 1
    tet_conn = conn.indices.reshape(n_cells, 4)

    centroids = (coors[tet_conn[:, 0]] + coors[tet_conn[:, 1]]
                 + coors[tet_conn[:, 2]] + coors[tet_conn[:, 3]]) / 4.0
    return centroids'''

_HEX_CENTROIDS_BLOCK = '''def compute_element_centroids(problem):
    """
    Compute centroid (x, y, z) of each 8-node hexahedral element.
    Hex transform inserted at runtime by biomimetic_pipeline.fea.fea_runner
    when element_type="hex". Arithmetic mean of the 8 corner coordinates.
    """
    coors = problem.domain.get_mesh_coors()
    cmesh = problem.domain.cmesh

    conn = cmesh.get_conn(3, 0)
    n_cells = len(conn.offsets) - 1
    hex_conn = conn.indices.reshape(n_cells, 8)

    centroids = coors[hex_conn].mean(axis=1)
    return centroids'''


def _apply_hex_transform(text: str) -> str:
    """Swap the two tet-only kernel functions for their 8-node-hex variants.

    Fails loudly (assertion) if the stock script's blocks have drifted from
    the expected text -- there's no reasonable silent fallback.
    """
    assert _TET_VOLUMES_BLOCK in text, (
        "hex transform: stock compute_element_volumes block not found verbatim "
        f"in {STOCK_COMPRESSION_TEST}. Stock script may have been edited."
    )
    text = text.replace(_TET_VOLUMES_BLOCK, _HEX_VOLUMES_BLOCK)
    assert _TET_CENTROIDS_BLOCK in text, (
        "hex transform: stock compute_element_centroids block not found verbatim "
        f"in {STOCK_COMPRESSION_TEST}. Stock script may have been edited."
    )
    text = text.replace(_TET_CENTROIDS_BLOCK, _HEX_CENTROIDS_BLOCK)
    return text


def run_one_iteration(
    mesh_source: Path,
    iter_dir: Path,
    material_E_mpa: float,
    material_nu: float,
    compress_disp_mm: float,
    load_mode: str = "compression",
    element_type: str = "tet",
    sfepy_env: str = SFEPY_ENV,
    # Default timeout bumped 2026-05-23 from 3600 -> 14400 s (4 h) to cover
    # the LU-factorization cost of the digital_twin's plated voxel-hex mesh
    # (~134k hex, ~446k DOFs; measured 7046 s on the 2026-05-22 gate run).
    # Lattice-class meshes finish in ~5-15 min and aren't affected by the
    # larger ceiling; only OOM-killed runs change behavior.
    timeout: int = 14400,
) -> FeaRunResult:
    # Resolve to an absolute path: sfepy-run is invoked with cwd set to this
    # directory, and it imports the problem-def script by module name -- a
    # relative path here would break that import.
    iter_dir = Path(iter_dir).resolve()
    iter_dir.mkdir(parents=True, exist_ok=True)

    dest_mesh = iter_dir / FEA_MESH_FILENAME
    shutil.copy2(mesh_source, dest_mesh)

    # sfepy-run changes CWD to the problem-def script's parent dir, so
    # `filename_mesh = "compound_enamel_lattice.msh"` (relative) would resolve
    # to any stale mesh sitting next to the stock script. We defeat that by
    # writing the problem-def into our iter dir and running THAT copy. For
    # tension the problem-def is derived from the stock script (see
    # _prepare_problem_def); for compression it is the stock script verbatim.
    # The element_type argument additionally swaps the two tet-only kernel
    # functions for 8-node hex variants when meshing the digital twin.
    local_problem = _prepare_problem_def(iter_dir, load_mode, element_type=element_type)

    log_path = iter_dir / "fea_run.log"

    env_vars = {
        "MATERIAL_E": f"{float(material_E_mpa)}",
        "MATERIAL_NU": f"{float(material_nu)}",
        "COMPRESS_DISP_MM": f"{float(compress_disp_mm)}",
    }

    conda = _find_conda()
    cmd = [
        conda,
        "run",
        "--no-capture-output",
        "-n",
        sfepy_env,
        "sfepy-run",
        str(local_problem),
    ]
    import os

    env = os.environ.copy()
    env.update(env_vars)

    completed = subprocess.run(
        cmd, cwd=str(iter_dir), env=env, capture_output=True, text=True, timeout=timeout
    )

    log_path.write_text(
        "# cmd\n"
        + " ".join(cmd)
        + "\n\n"
        + "# env_overrides\n"
        + "\n".join(f"{k}={v}" for k, v in env_vars.items())
        + "\n\n"
        + "# stdout\n"
        + (completed.stdout or "")
        + "\n\n"
        + "# stderr\n"
        + (completed.stderr or "")
        + "\n"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"fea_runner failed (rc={completed.returncode}). See {log_path}")

    global_path = iter_dir / f"global_results_{load_mode}.csv"
    element_path = iter_dir / f"element_results_{load_mode}.csv"
    if not global_path.exists():
        raise RuntimeError(f"global_results_{load_mode}.csv not produced in {iter_dir}")

    globals_dict = _read_global_results(global_path)

    return FeaRunResult(
        iter_dir=iter_dir,
        global_results_path=global_path,
        element_results_path=element_path,
        avg_von_mises_mpa=globals_dict.get("avg_von_mises_MPa", float("nan")),
        avg_sigma_zz_mpa=globals_dict.get("avg_sigma_zz_MPa", float("nan")),
        force_N=globals_dict.get("force_N", float("nan")),
        log_path=log_path,
    )


def _read_global_results(path: Path) -> Dict[str, float]:
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        row = next(reader)
    return {header[i]: float(row[i]) for i in range(min(len(header), len(row)))}


def copy_fea_outputs(src_dir: Path, dst_dir: Path, load_mode: str = "compression") -> None:
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in output_files_for(load_mode):
        s = src_dir / name
        if s.exists():
            shutil.copy2(s, dst_dir / name)
