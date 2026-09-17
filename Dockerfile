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

# --- 0. Apply Debian security updates ----------------------------------------
# The upstream python:*-slim tag is rebuilt on Docker's own cadence (the tag
# current when this was added was last pushed 2026-09-02), so a freshly pulled
# base image can still contain OS packages for which Debian has ALREADY
# published fixed versions. The CI container gate scans CRITICAL/HIGH with
# ignore-unfixed:true, so those already-fixed-upstream packages are exactly
# what trips it: the 2026-09-14 weekly re-scan failed on 12 findings
# (9 HIGH / 3 CRITICAL) in gzip, libpcre2-8-0, libsqlite3-0 and perl-base,
# every one of them fixed in the Debian archive but not yet in the base image.
# Upgrading here closes the window between Debian shipping a fix and Docker
# rebuilding the base, so the weekly gate stops going red each time that gap
# opens. `upgrade` (not `dist-upgrade`) keeps this to in-place version bumps
# within the stable release: no packages are added or removed.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
 && rm -rf /var/lib/apt/lists/*

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
RUN python -m pip install --upgrade pip setuptools wheel "jaraco.context>=6.1.0" && python -m pip install -e ".[dev]"

# --- 4. Default command: run the fast suite to prove the image works ---------
CMD ["pytest", "-m", "not slow"]
