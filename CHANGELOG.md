# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-07-16

### Added
- **Artifact provenance across pipeline stages**
  (`biomimetic_pipeline.orchestration.provenance`, `docs/PROVENANCE.md`): `run_context`
  could always *record* SHA-256 artifact hashes, but nothing ever *checked* them. Stage
  boundaries now **verify** each consumed artifact against the hash recorded when it was
  produced (a mid-run swap or corruption fails at the seam where it happened); a
  `ProvenanceChain` **chains** one hash-linked entry per stage into
  `provenance_chain.json`, giving unbroken lineage from raw morphometrics to final
  metrics, auditable end-of-run via `verify_chain` and a CLI; and `sign()` appends an
  **HMAC-SHA256** over the chain's canonical JSON (keyed by `BLP_PROVENANCE_KEY`, never
  stored) so any edit to the chain file itself — including rewriting hashes to match
  tampered artifacts — is detectable.
- **Continuous delivery** (`.github/workflows/release.yml`): pushing a `v*` tag re-runs the
  full CI gate on the tagged commit, then publishes the sdist + wheel to PyPI via Trusted
  Publishing (OIDC — no stored API token) and a signed container image to GHCR carrying a
  SLSA build-provenance attestation. `ci.yml` is now `workflow_call`-able so the release
  re-uses the exact same gates instead of a copy that could drift.
- **DevSecOps stages in CI**, all gating: full-history secret detection (gitleaks), SAST
  (bandit, medium+), dependency-CVE audit (pip-audit), and container vulnerability scanning
  (trivy, CRITICAL/HIGH). The image additionally ships an SPDX SBOM (syft) signed keylessly
  with cosign via GitHub OIDC.

### Changed
- Crack-deflection metric: **~12× faster, with identical results.**

### Security
- Patched the 3 fixable HIGH-severity CVEs that trivy found in the container image.

### Fixed
- Corrected the trivy-action pin (`0.24.0`, which does not exist → `v0.36.0`), and scoped
  the trivy gate to *fixable* CRITICAL/HIGH so an unpatchable upstream CVE cannot
  permanently block releases.

## [0.1.0] - 2026-07-03

### Added
- Initial standalone, open-source version of the micro-CT-driven biomimetic
  lattice design + FEA optimization pipeline (software only; extracted from the
  `microct_pipeline` research tree).
- Single-namespace package layout (`biomimetic_pipeline`) for clean,
  collision-free imports.
- Type checking (mypy, lenient baseline), pre-commit hooks, test-coverage
  reporting, and a container image (Dockerfile) with a CI build/smoke-test.
- Structured logging via `logging_config.configure_logging` and a `--verbose`
  CLI flag; library modules log through `logging.getLogger(__name__)`.
- Apache-2.0 license (`LICENSE` + `NOTICE`).
- pip-installable package (`pyproject.toml`, PEP 621) with `[fea]` (heavy
  geometry stack) and `[dev]` (test/lint) optional-dependency extras.
- pytest suite — 21 tests across 6 modules — with the canonical specimen
  morphometrics committed as a fixture so the golden value-carry-through test is
  self-contained.
- ruff (lint) and black (format) configuration, with a clean pass over the
  codebase.
- GitHub Actions CI: lint + format-check + pytest on Python 3.10–3.12, with the
  conda-only FEA integration path gated behind the `slow` pytest marker.
- Coverage reported in CI (`pytest --cov --cov-report=term-missing`); ruff and black
  gate the build, and mypy runs in package mode (`-p`, advisory while annotations are
  completed). Hard coverage-gating is deferred: the default CI excludes the heavy FEA/CAD
  path (run via the `slow` marker), so it exercises only the pure-Python core.
- `CITATION.cff` (software + manuscript citation) and `CONTRIBUTING.md` (developer guide).
