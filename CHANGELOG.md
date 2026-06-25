# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-24

### Added
- Initial standalone, open-source release of the micro-CT-driven biomimetic
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
