# Copyright 2026 Cameron B. Renteria
# SPDX-License-Identifier: Apache-2.0
"""
lattice_mesh.py — Standalone Gmsh meshing module for decussated enamel lattices.

Refactored from Cell 3 of helicaltwist_continous_loadingplates.ipynb.

Takes a pre-fused STEP file (single solid with rods, bridges, and loading
plates already united in CadQuery) and generates a tetrahedral volume mesh
in MSH 2.2 format suitable for SfePy finite-element analysis.

Usage as a module:
    from lattice_mesh import mesh_lattice
    msh_path = mesh_lattice("model.step", "model.msh")

Usage from the command line:
    python lattice_mesh.py model.step -o model.msh --mesh-size 0.4
"""

import gmsh
import os
import sys
import time
import argparse
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Default parameters (matching the notebook's Cell 3 values)
# ---------------------------------------------------------------------------
DEFAULTS: Dict[str, object] = {
    "mesh_size": 0.5,
    "mesh_algorithm_2d": 6,          # Frontal-Delaunay
    "mesh_algorithm_3d": 10,         # HXT — fast parallel tet mesher
    "min_element_size": None,        # derived: mesh_size * 0.05
    "max_element_size": None,        # derived: mesh_size
    "mesh_format": "msh22",
    "num_threads": 4,
    "elements_per_two_pi": 12,
    "smoothing_steps": 5,
    "junction_refinement_factor": 1.0,
    "bridge_elevations": [1.25, 7.08, 12.92, 18.75],
    "refinement_band_width": 1.0,
}


def _apply_junction_refinement(
    mesh_size: float,
    factor: float,
    elevations: List[float],
    band_width: float,
) -> None:
    """Create Gmsh distance/threshold fields to refine mesh near bridge junctions."""
    field = gmsh.model.mesh.field
    refined_size = mesh_size * factor
    field_ids: List[int] = []

    for i, z_elev in enumerate(elevations):
        dist_id = 100 + 2 * i
        field.add("MathEval", dist_id)
        field.setString(dist_id, "F", f"Abs(z - {z_elev})")

        thresh_id = 101 + 2 * i
        field.add("Threshold", thresh_id)
        field.setNumber(thresh_id, "InField", dist_id)
        field.setNumber(thresh_id, "SizeMin", refined_size)
        field.setNumber(thresh_id, "SizeMax", mesh_size)
        field.setNumber(thresh_id, "DistMin", 0.0)
        field.setNumber(thresh_id, "DistMax", band_width)
        field.setNumber(thresh_id, "StopAtDistMax", 1)

        field_ids.append(thresh_id)

    min_id = 200
    field.add("Min", min_id)
    field.setNumbers(min_id, "FieldsList", field_ids)
    field.setAsBackgroundMesh(min_id)

    print(f"  Junction refinement: factor={factor}, "
          f"refined_size={refined_size:.3f} mm, "
          f"band_width={band_width} mm, "
          f"elevations={elevations}")


def _print_mesh_statistics() -> None:
    """Query and print mesh quality statistics from the current Gmsh model."""
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    n_nodes = len(node_tags)

    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
    total_3d = sum(len(t) for t in elem_tags)

    elem_types_2d, elem_tags_2d, _ = gmsh.model.mesh.getElements(2)
    total_2d = sum(len(t) for t in elem_tags_2d)

    print(f"\n--- Mesh Statistics ---")
    print(f"  Nodes:          {n_nodes:,}")
    print(f"  2D elements:    {total_2d:,}")
    print(f"  3D elements:    {total_3d:,}")

    if total_3d > 0:
        try:
            qualities = gmsh.model.mesh.getElementQualities(
                elementTags=[], qualityName="minSICN"
            )
            if qualities:
                q_min = min(qualities)
                q_max = max(qualities)
                q_avg = sum(qualities) / len(qualities)
                print(f"  Quality (SICN): min={q_min:.4f}  avg={q_avg:.4f}  max={q_max:.4f}")
        except Exception:
            pass

    try:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        vol = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
        if total_3d > 0:
            approx_size = (vol / total_3d) ** (1.0 / 3.0)
            print(f"  Bounding box:   ({xmin:.2f},{ymin:.2f},{zmin:.2f}) "
                  f"to ({xmax:.2f},{ymax:.2f},{zmax:.2f})")
            print(f"  Approx avg element size: {approx_size:.4f} mm")
    except Exception:
        pass

    print(f"-----------------------")


def mesh_lattice(
    step_path: str,
    output_path: str,
    params: Optional[Dict] = None,
) -> str:
    """Generate a tetrahedral mesh from a pre-fused STEP file.

    Parameters
    ----------
    step_path : str
        Path to the input STEP file (single fused solid).
    output_path : str
        Path for the output .msh file (MSH 2.2 format).
    params : dict, optional
        Override any default parameter.

    Returns
    -------
    str
        Absolute path of the written .msh file.
    """
    cfg = dict(DEFAULTS)
    if params:
        for key, value in params.items():
            if key not in cfg:
                print(f"WARNING: Unknown parameter '{key}' -- ignored")
            else:
                cfg[key] = value

    mesh_size = cfg["mesh_size"]
    min_size = cfg["min_element_size"] if cfg["min_element_size"] is not None else mesh_size * 0.05
    max_size = cfg["max_element_size"] if cfg["max_element_size"] is not None else mesh_size
    algo_2d = cfg["mesh_algorithm_2d"]
    algo_3d = cfg["mesh_algorithm_3d"]
    num_threads = cfg["num_threads"]
    elements_per_2pi = cfg["elements_per_two_pi"]
    smoothing = cfg["smoothing_steps"]
    jrf = cfg["junction_refinement_factor"]
    bridge_elevations = cfg["bridge_elevations"]
    band_width = cfg["refinement_band_width"]

    step_path = os.path.abspath(step_path)
    output_path = os.path.abspath(output_path)

    if not os.path.isfile(step_path):
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    t0 = time.time()

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", 3)
        gmsh.option.setNumber("General.NumThreads", num_threads)

        gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)
        gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
        gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)

        print(f"Loading STEP: {step_path}")
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        print(f"  Import time: {time.time() - t0:.1f}s")

        vols = gmsh.model.getEntities(3)
        surfs = gmsh.model.getEntities(2)
        edges = gmsh.model.getEntities(1)
        print(f"  Volumes:  {len(vols)}")
        print(f"  Surfaces: {len(surfs)}")
        print(f"  Edges:    {len(edges)}")

        if len(vols) == 0:
            raise RuntimeError("No volumes found in STEP file.")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", elements_per_2pi)
        gmsh.option.setNumber("Mesh.Algorithm", algo_2d)
        gmsh.option.setNumber("Mesh.Algorithm3D", algo_3d)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.Smoothing", smoothing)
        gmsh.option.setNumber("Mesh.QualityType", 2)
        gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.5)
        gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", 1e-1)

        if jrf < 1.0:
            _apply_junction_refinement(mesh_size, jrf, bridge_elevations, band_width)

        print(f"\nGenerating surface mesh (2D)...")
        t1 = time.time()
        try:
            gmsh.model.mesh.generate(2)
            print(f"  Surface mesh OK ({time.time() - t1:.1f}s)")
        except Exception as e:
            print(f"  Surface mesh warning: {e}")

        print(f"Generating volume mesh (3D)...")
        t2 = time.time()
        try:
            gmsh.model.mesh.generate(3)
            print(f"  Volume mesh OK ({time.time() - t2:.1f}s)")
        except Exception as e:
            print(f"  Volume mesh FAILED: {e}")
            print("  Retrying with Delaunay 3D (algo 1)...")
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)
            try:
                gmsh.model.mesh.generate(3)
                print(f"  Delaunay fallback OK ({time.time() - t2:.1f}s)")
            except Exception as e2:
                gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
                debug_path = output_path.replace(".msh", "_surface_only.msh")
                gmsh.write(debug_path)
                print(f"  Saved surface-only mesh: {debug_path}")
                raise RuntimeError(
                    f"Volume meshing failed with both HXT and Delaunay: {e2}"
                ) from e2

        # Verify 3D elements were actually created
        elem_types_3d, elem_tags_3d, _ = gmsh.model.mesh.getElements(3)
        total_3d = sum(len(t) for t in elem_tags_3d)
        if total_3d == 0:
            print("  WARNING: 0 3D elements after meshing — retrying with relaxed tolerances...")
            gmsh.option.setNumber("Mesh.Algorithm3D", 4)  # Frontal 3D
            gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", 1e0)
            gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 1.0)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size * 0.5)
            try:
                gmsh.model.mesh.generate(3)
                elem_types_3d, elem_tags_3d, _ = gmsh.model.mesh.getElements(3)
                total_3d = sum(len(t) for t in elem_tags_3d)
                if total_3d == 0:
                    raise RuntimeError("Still 0 3D elements after Frontal fallback")
                print(f"  Frontal 3D fallback OK — {total_3d:,} elements")
            except Exception as e3:
                print(f"  Frontal 3D fallback also failed: {e3}")
                gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
                debug_path = output_path.replace(".msh", "_surface_only.msh")
                gmsh.write(debug_path)
                print(f"  Saved surface-only mesh: {debug_path}")
                raise RuntimeError(
                    f"Volume meshing produced 0 elements with all algorithms: {e3}"
                ) from e3

        _print_mesh_statistics()

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(output_path)

        file_size = os.path.getsize(output_path) / 1e6
        total_time = time.time() - t0
        print(f"\nMesh written: {output_path}")
        print(f"File size:    {file_size:.1f} MB")
        print(f"Total time:   {total_time:.1f}s")

    finally:
        gmsh.finalize()

    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mesh a pre-fused enamel-lattice STEP file with Gmsh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("step_file", help="Path to the input STEP file.")
    p.add_argument("-o", "--output", default=None,
                   help="Output .msh path. Defaults to <step_file>.msh.")
    p.add_argument("--mesh-size", type=float, default=DEFAULTS["mesh_size"],
                   help="Global characteristic mesh size (mm).")
    p.add_argument("--min-element-size", type=float, default=None)
    p.add_argument("--max-element-size", type=float, default=None)
    p.add_argument("--algo-2d", type=int, default=DEFAULTS["mesh_algorithm_2d"])
    p.add_argument("--algo-3d", type=int, default=DEFAULTS["mesh_algorithm_3d"])
    p.add_argument("--num-threads", type=int, default=DEFAULTS["num_threads"])
    p.add_argument("--smoothing", type=int, default=DEFAULTS["smoothing_steps"])
    p.add_argument("--junction-refinement-factor", type=float,
                   default=DEFAULTS["junction_refinement_factor"],
                   help="Mesh refinement factor at bridge junctions (< 1.0 to refine).")
    p.add_argument("--bridge-elevations", type=float, nargs="+",
                   default=DEFAULTS["bridge_elevations"])
    p.add_argument("--refinement-band-width", type=float,
                   default=DEFAULTS["refinement_band_width"])
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = os.path.splitext(args.step_file)[0] + ".msh"

    params = {
        "mesh_size": args.mesh_size,
        "mesh_algorithm_2d": args.algo_2d,
        "mesh_algorithm_3d": args.algo_3d,
        "min_element_size": args.min_element_size,
        "max_element_size": args.max_element_size,
        "num_threads": args.num_threads,
        "smoothing_steps": args.smoothing,
        "junction_refinement_factor": args.junction_refinement_factor,
        "bridge_elevations": args.bridge_elevations,
        "refinement_band_width": args.refinement_band_width,
    }

    try:
        mesh_lattice(args.step_file, output, params)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
