# Contributing

Thanks for your interest in `biomimetic-lattice-pipeline`. This guide covers the
development setup and the checks CI enforces, so a contribution lands green on the
first try.

## Development setup

```bash
git clone https://github.com/crentb/biomimetic-lattice-pipeline.git
cd biomimetic-lattice-pipeline
python -m pip install -e ".[dev]"     # core runtime + dev tools (pytest, ruff, black, mypy, pre-commit)
pre-commit install                    # run ruff/black/mypy automatically on every commit
```

The default install and CI exercise the **pure-Python design and analytics logic**.
The heavy geometry/FEA path (CadQuery, SfePy, gmsh, pyvista) is optional and lives
behind the `[fea]` extra plus a conda environment; those integration tests are marked
`slow` and are skipped by default:

```bash
python -m pip install -e ".[fea]"     # heavy geometry/visualization stack (optional)
```

## Checks (what CI runs)

Run these before opening a pull request; CI runs the same on Python 3.10-3.12:

```bash
ruff check .                  # lint
black --check .               # formatting
mypy -p biomimetic_pipeline   # type-checking (package mode)
pytest -m "not slow" --cov    # fast, pure-Python tests with coverage
```

To run the full suite including the FEA integration test (needs the conda FEA stack):

```bash
pytest
```

Coverage must not drop below the configured floor, and the package must stay lint-,
format-, and type-clean.

## Architecture note

A single canonical `morphometrics.json` is the only coupling point between stages
(`ingest -> mapping -> generators -> fea -> metrics -> objectives -> orchestration ->
reporting`), so each component can be changed independently. New stages should read and
write that contract (see `biomimetic_pipeline/schemas/`) rather than coupling directly to
another stage.

## Pull requests

1. Branch from `main`.
2. Keep changes focused; add or update tests for any behavior change (the golden
   value-carry-through fixture in `tests/` guards numerical results).
3. Update `CHANGELOG.md` under `[Unreleased]`.
4. Ensure all checks above pass locally.

## License

By contributing, you agree that your contributions are licensed under the project's
**Apache-2.0** license (see `LICENSE` and `NOTICE`).
