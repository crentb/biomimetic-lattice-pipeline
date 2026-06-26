"""Shared infrastructure used by every stage of the pipeline.

RunContext owns:
  - paths to the upstream cad_modeling/ stack (never mutated)
  - paths to the per-run output directory under biomimetic_pipeline/runs/
  - conda environment names (inherited from cad_modeling/)
  - environment probe (fail-fast if cad_env or sfepy_env is broken)
  - file hashing + provenance helpers
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PIPELINE_VERSION = "0.1.0"

BIOMIMETIC_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BIOMIMETIC_ROOT.parent
MICROCT_ROOT = REPO_ROOT  # retained for the provenance stub / backward compatibility

# The CadQuery (CAD + meshing) and SfePy (FEA) geometry engine ships inside this
# repository under geometry/. To drive an external copy instead (e.g. a local
# microct_pipeline checkout laid out as
# <root>/cad_modeling/Decussated Models Continous Twist/), set the
# MICROCT_PIPELINE_ROOT environment variable.
_EXTERNAL_ROOT = os.environ.get("MICROCT_PIPELINE_ROOT")
if _EXTERNAL_ROOT:
    CAD_STACK_DIR = (
        Path(_EXTERNAL_ROOT).expanduser().resolve()
        / "cad_modeling"
        / "Decussated Models Continous Twist"
    )
else:
    CAD_STACK_DIR = REPO_ROOT / "geometry"

STOCK_LATTICE_CAD = CAD_STACK_DIR / "lattice_cad.py"
STOCK_LATTICE_MESH = CAD_STACK_DIR / "lattice_mesh.py"
STOCK_COMPRESSION_TEST = CAD_STACK_DIR / "compression_test.py"
STOCK_EXTRACT_METRICS = CAD_STACK_DIR / "extract_metrics.py"

CAD_ENV = "cad_env"
SFEPY_ENV = "sfepy_env"


@dataclass
class RunContext:
    run_name: str
    run_dir: Path
    pipeline_version: str = PIPELINE_VERSION
    cad_env: str = CAD_ENV
    sfepy_env: str = SFEPY_ENV
    created_at_iso: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    probe_results: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, run_name: str, runs_root: Optional[Path] = None) -> "RunContext":
        root = runs_root or (BIOMIMETIC_ROOT / "runs")
        run_dir = root / run_name
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        for sub in ("cad", "mesh", "fea", "metrics", "report"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        ctx = cls(run_name=run_name, run_dir=run_dir)
        ctx.write_provenance_stub()
        return ctx

    def write_provenance_stub(self) -> None:
        stub = {
            "run_name": self.run_name,
            "pipeline_version": self.pipeline_version,
            "created_at_iso": self.created_at_iso,
            "biomimetic_root": str(BIOMIMETIC_ROOT),
            "cad_stack_dir": str(CAD_STACK_DIR),
            "cad_env": self.cad_env,
            "sfepy_env": self.sfepy_env,
            "microct_root": str(MICROCT_ROOT),
        }
        (self.run_dir / "run_context.json").write_text(json.dumps(stub, indent=2))

    def probe_envs(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        # cad_env owns CadQuery; gmsh lives in sfepy_env (the stock pipeline
        # runs the meshing stage there, not in cad_env).
        results["cad_env"] = _probe_conda_env(self.cad_env, ["python", "-c", "import cadquery"])
        results["sfepy_env"] = _probe_conda_env(
            self.sfepy_env, ["python", "-c", "import sfepy, numpy, scipy, gmsh"]
        )
        self.probe_results = results
        (self.run_dir / "env_probe.json").write_text(json.dumps(results, indent=2))
        failures = [env for env, r in results.items() if r["returncode"] != 0]
        if failures:
            raise RuntimeError(
                f"Conda env probe failed for: {failures}. See {self.run_dir / 'env_probe.json'}"
            )
        return results


def _probe_conda_env(env_name: str, python_args: List[str]) -> Dict[str, Any]:
    conda = _find_conda()
    cmd = [conda, "run", "--no-capture-output", "-n", env_name] + python_args
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return {
            "env": env_name,
            "cmd": cmd,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-1000:] if completed.stdout else "",
            "stderr_tail": completed.stderr[-1000:] if completed.stderr else "",
        }
    except FileNotFoundError as exc:
        return {"env": env_name, "cmd": cmd, "returncode": -1, "stderr_tail": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {"env": env_name, "cmd": cmd, "returncode": -2, "stderr_tail": f"timeout: {exc}"}


def _find_conda() -> str:
    for cand in ("mamba", "conda"):
        if shutil.which(cand):
            return cand
    raise RuntimeError("Neither conda nor mamba found in PATH.")


def run_in_env(
    env_name: str,
    python_args: List[str],
    cwd: Optional[Path] = None,
    env_vars: Optional[Dict[str, str]] = None,
    timeout: int = 1800,
) -> subprocess.CompletedProcess:
    """Invoke a command inside a conda environment using `conda run`."""
    conda = _find_conda()
    cmd = [conda, "run", "--no-capture-output", "-n", env_name] + python_args
    env = os.environ.copy()
    if env_vars:
        env.update({k: str(v) for k, v in env_vars.items()})
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_existing(paths: Dict[str, Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, path in paths.items():
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            continue
        if p.is_dir():
            # Directory source: hash the manifest of filenames + sizes as a stable proxy.
            entries = sorted(p.iterdir(), key=lambda e: e.name)
            manifest = "\n".join(f"{e.name}:{e.stat().st_size}" for e in entries).encode()
            out[key] = hashlib.sha256(manifest).hexdigest()
        else:
            out[key] = sha256_file(p)
    return out


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
