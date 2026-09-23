# biomimetic-lattice-pipeline

From synchrotron micro-computed tomography (micro-CT) of tooth enamel to biomimetic lattices: parametric computer-aided design (CAD), finite-element analysis (FEA), and closed-loop design optimization.

[![CI](https://github.com/crentb/biomimetic-lattice-pipeline/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/crentb/biomimetic-lattice-pipeline/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/biomimetic-lattice-pipeline)](https://pypi.org/project/biomimetic-lattice-pipeline/)
[![Python](https://img.shields.io/badge/python-3.10--3.14-blue)](https://github.com/crentb/biomimetic-lattice-pipeline/blob/main/pyproject.toml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21148570.svg)](https://doi.org/10.5281/zenodo.21148570)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/crentb/biomimetic-lattice-pipeline/blob/main/LICENSE)

![Pipeline overview: measurement, design, and realization stages, with an Optuna optimization loop that maps the best designs back into measurement targets](https://raw.githubusercontent.com/crentb/biomimetic-lattice-pipeline/main/docs/figures/pipeline_overview.png)

The pipeline turns measured three-dimensional rod geometry from synchrotron micro-CT of tooth enamel into **parametric CAD lattices**, runs **linear-elastic FEA** on them, scores the results against pluggable **objectives**, and can drive a **closed-loop Optuna optimization** over the manufacturability-constrained design space. Every run emits a metrics JSON and a LaTeX/PDF report.

**Associated manuscript:** C. B. Renteria, J. R. Grimm, A. Yunker, D. Y. Parkinson, D. D. Arola, "Translating Helically Decussated Enamel into Damage-Tolerant Bioinspired Lattices," *Matter* (in preparation).

## Architecture

![Pipeline architecture](https://raw.githubusercontent.com/crentb/biomimetic-lattice-pipeline/main/docs/figures/01_architecture.png)

A single canonical `morphometrics.json` is the only coupling point between stages, so each component can be replaced independently, and a JSON Schema contract is validated at every seam. The one-page end-to-end schematic, with the module behind each stage, is in [docs/biomimetic_pipeline_schematic.pdf](https://github.com/crentb/biomimetic-lattice-pipeline/blob/main/docs/biomimetic_pipeline_schematic.pdf).

## Installation

From PyPI:

```bash
pip install biomimetic-lattice-pipeline            # design, mapping, analytics, optimization
pip install "biomimetic-lattice-pipeline[fea]"     # + PyVista, Gmsh, meshio, scikit-image
```

As a signed container image:

```bash
docker pull ghcr.io/crentb/biomimetic-lattice-pipeline:v0.2.0
```

From source, for development:

```bash
git clone https://github.com/crentb/biomimetic-lattice-pipeline.git
cd biomimetic-lattice-pipeline
python -m pip install -e ".[dev]"       # core + pytest, ruff, black, mypy, pre-commit
```

> **Full FEA** additionally requires **CadQuery** and **SfePy**, installed with conda because they are difficult to pip-install across platforms. The CAD and FEA engine ships in this repository under `geometry/` (see [geometry/README.md](https://github.com/crentb/biomimetic-lattice-pipeline/blob/main/geometry/README.md)). The default install and CI exercise the pure-Python design and analytics logic; the CAD and FEA path is gated behind the `slow` pytest marker.

## Quick start

```bash
# End-to-end single-specimen run: morphometrics -> CAD -> mesh -> FEA -> metrics -> report
python scripts/run_pipeline.py --morphometrics path/to/morphometrics.json \
    --run-name demo --objective crack_deflection --model-type continuous_twist

# Parametric sweep over one free CAD parameter
python scripts/run_sweep.py --morphometrics path/to/morphometrics.json \
    --run-name sweep_layers --param N_BRIDGE_LAYERS --values 4 6 8

# Optuna optimization of an objective
python scripts/run_optimize.py --morphometrics path/to/morphometrics.json \
    --run-name opt_cd --objective crack_deflection --n-trials 30

# Closed loop: optimize, then map the top designs back into morphometric targets
python scripts/run_closed_loop.py --run-name closed_cd \
    --objective crack_deflection --n-trials 30 --top-k 5
```

The closed loop writes `biomimicry_targets.json`: the rod diameters, band widths, wavelengths, and directions to look for in a real specimen. The remaining scripts in `scripts/` are sweep, diagnostic, and printability utilities.

## Repository layout

```text
biomimetic_pipeline/
  ingest/          measured micro-CT morphometrics -> canonical morphometrics.json
  mapping/         morphometrics -> CAD parameters (deterministic, with logged manufacturability clamps)
  generators/      parametric continuous-twist CAD family and rod-by-rod digital-twin variants
  fea/             strain solver driving SfePy to a target representative stress
  metrics/         crack-deflection streamlines, biomimicry score, stress concentration, toughness
  objectives/      registry-based scoring (crack deflection, toughness, ...)
  config/          objective definitions, one YAML file per objective
  orchestration/   single-run, sweep, Optuna optimization, and closed-loop drivers; provenance
  reporting/       LaTeX/PDF report generation
  schemas/         JSON Schema contracts between stages
geometry/          CadQuery + SfePy CAD, mesh, and FEA engine, driven by generators/ and fea/
docs/              design document, parameter report, provenance, performance, figures
scripts/           command-line entry points and analysis utilities
tests/             fast pure-Python suite; CAD/FEA integration tests are marked slow
```

## Documentation

| Document | Contents |
|---|---|
| [PIPELINE.pdf](https://github.com/crentb/biomimetic-lattice-pipeline/blob/main/docs/PIPELINE.pdf) | Design document: stages, contracts, mappings, and tests |
| [MODEL_PARAMETER_REPORT.pdf](https://github.com/crentb/biomimetic-lattice-pipeline/blob/main/docs/MODEL_PARAMETER_REPORT.pdf) | Every CAD parameter, its source, and its measured or clamped value |
| [PROVENANCE.md](https://github.com/crentb/biomimetic-lattice-pipeline/blob/main/docs/PROVENANCE.md) | Stage-boundary verification, hash-linked artifact chains, and signing |
| [PERF.md](https://github.com/crentb/biomimetic-lattice-pipeline/blob/main/docs/PERF.md) | Profiling of the crack-deflection metric, with flame graphs |
| [biomimetic_pipeline_schematic.pdf](https://github.com/crentb/biomimetic-lattice-pipeline/blob/main/docs/biomimetic_pipeline_schematic.pdf) | One-page end-to-end schematic |

## Testing and continuous integration

```bash
pytest -m "not slow"    # fast, pure-Python logic tests (what CI runs)
pytest                  # everything, including the FEA integration test (needs the conda FEA stack)
```

Every push and pull request runs one gate, defined in [ci.yml](https://github.com/crentb/biomimetic-lattice-pipeline/blob/main/.github/workflows/ci.yml):

- **Quality:** ruff, black, mypy (advisory), and pytest with coverage on Python 3.10 to 3.14.
- **Security (blocking):** gitleaks secret detection over the full history, bandit static analysis at medium severity and above, and pip-audit against known vulnerabilities.
- **Container:** image build, a trivy scan that blocks on fixable critical and high findings, the test suite run inside the image, and an SPDX software bill of materials signed keylessly with cosign.

The same gate re-runs weekly on `main` ([scheduled-scan.yml](https://github.com/crentb/biomimetic-lattice-pipeline/blob/main/.github/workflows/scheduled-scan.yml)), so a newly published vulnerability surfaces without a code change. A version tag re-runs it on the tagged commit before [release.yml](https://github.com/crentb/biomimetic-lattice-pipeline/blob/main/.github/workflows/release.yml) publishes to PyPI through Trusted Publishing (no stored tokens) and pushes a scanned, cosign-signed image with SLSA build provenance to the GitHub Container Registry. To verify a published image:

```bash
cosign verify ghcr.io/crentb/biomimetic-lattice-pipeline:v0.2.0 \
  --certificate-identity-regexp 'github.com/crentb/' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
gh attestation verify oci://ghcr.io/crentb/biomimetic-lattice-pipeline:v0.2.0 --owner crentb
```

To report a vulnerability, see [SECURITY.md](https://github.com/crentb/biomimetic-lattice-pipeline/blob/main/SECURITY.md).

## Citation

Please cite the software and the associated manuscript. GitHub's "Cite this repository" button reads [CITATION.cff](https://github.com/crentb/biomimetic-lattice-pipeline/blob/main/CITATION.cff).

```bibtex
@software{renteria_biomimetic_lattice_pipeline,
  author    = {Renteria, Cameron B.},
  title     = {biomimetic-lattice-pipeline},
  version   = {0.2.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21148570},
  url       = {https://github.com/crentb/biomimetic-lattice-pipeline}
}
```

## License

Apache-2.0. See [LICENSE](https://github.com/crentb/biomimetic-lattice-pipeline/blob/main/LICENSE) and [NOTICE](https://github.com/crentb/biomimetic-lattice-pipeline/blob/main/NOTICE).
