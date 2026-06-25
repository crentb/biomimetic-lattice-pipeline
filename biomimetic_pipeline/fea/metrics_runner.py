"""Subprocess wrapper around stock extract_metrics.py.

Runs extract_metrics.py inside sfepy_env against the FEA output directory,
producing optimization_metrics.json with 30 stock scalars.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from biomimetic_pipeline.orchestration.run_context import (
    SFEPY_ENV,
    STOCK_EXTRACT_METRICS,
    run_in_env,
)


@dataclass
class MetricsRunResult:
    metrics_path: Path
    metrics: Dict[str, Any]
    log_path: Path


def run(
    results_dir: Path,
    sidecar_path: Optional[Path] = None,
    output_name: str = "optimization_metrics.json",
    sfepy_env: str = SFEPY_ENV,
    timeout: int = 600,
) -> MetricsRunResult:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / output_name
    log_path = results_dir / "metrics_run.log"

    args = [
        "python",
        str(STOCK_EXTRACT_METRICS),
        "--results-dir",
        str(results_dir),
        "--output",
        str(out_path),
    ]
    if sidecar_path is not None and Path(sidecar_path).exists():
        args += ["--params-json", str(sidecar_path)]

    completed = run_in_env(sfepy_env, args, cwd=results_dir, timeout=timeout)

    log_path.write_text(
        "# cmd\n"
        + " ".join(args)
        + "\n\n"
        + "# stdout\n"
        + (completed.stdout or "")
        + "\n\n"
        + "# stderr\n"
        + (completed.stderr or "")
        + "\n"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"metrics_runner failed (rc={completed.returncode}). See {log_path}")
    if not out_path.exists():
        raise RuntimeError(f"{output_name} not produced in {results_dir}")

    metrics = json.loads(out_path.read_text())
    return MetricsRunResult(metrics_path=out_path, metrics=metrics, log_path=log_path)
