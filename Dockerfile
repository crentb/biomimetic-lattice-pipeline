# syntax=docker/dockerfile:1
# =============================================================================
# Container image for biomimetic-lattice-pipeline (core / analytics path).
#
# Installs the package plus its pure-Python core + dev dependencies on a slim
# Python base, so the design/mapping/metrics/objectives logic, the test suite,
# and the CLI run in a reproducible container.
#
# NOTE: the heavy FEA/CAD stack (CadQuery, SfePy, gmsh) is conda-only and is
# intentionally NOT in this image. A separate micromamba-based image adding the
# full FEA path is future work; this image covers the importable, unit-tested
# logic (and is what CI builds + smoke-tests).
# =============================================================================
FROM python:3.12-slim

# Reproducible, quiet Python: unbuffered output, no .pyc files, no pip cache.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- 1. Copy install metadata first for better layer caching ----------------
# (pyproject reads README.md for the long description, so it must be present.)
COPY pyproject.toml README.md LICENSE NOTICE ./

# --- 2. Copy the package sources + the test/config/schema trees -------------
COPY biomimetic_pipeline/ biomimetic_pipeline/
COPY scripts/ scripts/
COPY tests/ tests/

# --- 3. Install the package with dev extras (core deps only; no [fea]) -------
RUN python -m pip install --upgrade pip && python -m pip install -e ".[dev]"

# --- 4. Default command: run the fast suite to prove the image works ---------
CMD ["pytest", "-m", "not slow"]
