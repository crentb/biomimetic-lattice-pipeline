"""
logging_config.py
=================

Purpose
-------
Central logging setup for biomimetic-lattice-pipeline. Library modules obtain a
logger via ``logging.getLogger(__name__)`` and never configure handlers
themselves; the *applications* (the CLI entry points in ``scripts/``) call
``configure_logging()`` once at startup to decide where the library's messages
go and at what verbosity.

Why this exists
---------------
A library must not call ``print()`` or attach its own handlers -- doing so robs
the calling application of control over output (destination, level, silencing).
This module gives the apps one place to turn the library's log stream on, with a
``--verbose`` switch mapping to DEBUG.

Inputs / Outputs
----------------
``configure_logging(verbose=False)`` configures the root logger's format and
level (INFO, or DEBUG when ``verbose``) and returns it. Pure side effect on the
Python logging system; no files are written.
"""

from __future__ import annotations

import logging

# --- Log line format: timestamp, level, logger name, message ----------------
_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(verbose: bool = False) -> logging.Logger:
    """Configure root logging for an application (CLI) run.

    Parameters
    ----------
    verbose:
        If True, set the level to DEBUG; otherwise INFO.

    Returns
    -------
    logging.Logger
        The configured root logger.
    """
    level = logging.DEBUG if verbose else logging.INFO
    # force=True so repeated calls (e.g. nested CLI entry points) re-apply the
    # config cleanly instead of stacking duplicate handlers.
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_DATE_FORMAT, force=True)
    return logging.getLogger()
