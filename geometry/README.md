# `geometry/` — CadQuery + SfePy CAD/FEA engine

This directory holds the geometry and finite-element engine that the pipeline
drives as subprocess steps. The orchestration layer in `biomimetic_pipeline/`
calls these scripts through `RunContext`
(`biomimetic_pipeline/orchestration/run_context.py` points `CAD_STACK_DIR` here
by default).

## Scripts

| Script | Stage | Role |
|---|---|---|
| `lattice_cad.py` | CAD | Parametric CadQuery generation of the decussated enamel rod lattice — helically twisted rods in concentric rings, inter-rod bridges at configurable elevations, fused top/bottom loading plates, optional junction fillets. Exposes `generate_lattice(params)` and `DEFAULTS`; emits STEP/STL. |
| `lattice_mesh.py` | Mesh | Gmsh meshing of the CAD solid into a tetrahedral `.msh`. |
| `compression_test.py` | FEA | SfePy linear-elastic compression solve; writes per-element and global result CSVs. |
| `extract_metrics.py` | Metrics | Parses the FEA CSVs into optimization-relevant scalar metrics (effective modulus, stress-concentration factor, von Mises percentiles, specific toughness, …). |

## Environment

These scripts depend on heavy, conda-managed libraries, so they are kept out of
the default `pip install` and the fast CI (which exercise the pure-Python
pipeline logic only). The pipeline expects two conda environments, matching
`RunContext.probe_envs()`:

- **`cad_env`** — CadQuery (CAD generation).
  Example: `conda create -n cad_env -c conda-forge cadquery`
- **`sfepy_env`** — SfePy + gmsh + numpy + scipy (meshing + FEA).
  Example: `conda create -n sfepy_env -c conda-forge sfepy gmsh numpy scipy`

`RunContext.probe_envs()` fails fast if either environment is missing.

## Notes

- Authored by Cameron B. Renteria; Apache-2.0 (see the repository `LICENSE` and `NOTICE`).
- Carried as research code: invoked via subprocess (or standalone through each
  script's `--help` CLI), not imported as a package, and intentionally excluded
  from the repository's ruff/black/mypy gates.
- To drive an external copy of the engine instead of this in-repo one, set the
  `MICROCT_PIPELINE_ROOT` environment variable (see `run_context.py`).
