"""Subprocess wrapper around stock lattice_cad.py.

We cannot drive RING_ROTATION / CONTINUOUS_TWIST through lattice_cad's CLI, so
we write a tiny driver script into the run directory that imports the stock
lattice_cad as a library, merges our params over DEFAULTS, calls
generate_lattice(), and emits the standard `lattice_params.json` sidecar.
The stock file is untouched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from biomimetic_pipeline.orchestration.run_context import (
    BIOMIMETIC_ROOT,
    CAD_ENV,
    CAD_STACK_DIR,
    run_in_env,
)

NEW_MODELS_DIR = BIOMIMETIC_ROOT / "generators" / "new_models"


MODEL_DRIVERS: Dict[str, str] = {
    "continuous_twist": "<embedded>",
    "measured_profile": str(NEW_MODELS_DIR / "measured_profile_twist.py"),
    "hierarchical": str(NEW_MODELS_DIR / "hierarchical_twist.py"),
    "radially_graded": str(NEW_MODELS_DIR / "radially_graded.py"),
}


@dataclass
class CadRunResult:
    export_dir: Path
    step_path: Path
    stl_path: Path
    sidecar_path: Path
    log_path: Path
    model_type: str = "continuous_twist"


def run(
    cad_params: Dict[str, Any],
    export_dir: Path,
    model_type: str = "continuous_twist",
    cad_env: str = CAD_ENV,
    timeout: int = 1200,
) -> CadRunResult:
    """Generate CAD from a cad_parameters dict. Writes STEP/STL and sidecar.

    `model_type` selects the driver script:
      - "continuous_twist": stock lattice_cad.generate_lattice (embedded driver).
      - "measured_profile"/"hierarchical"/"radially_graded": new-model wrappers
        that live under generators/new_models/ and import lattice_cad as a lib.
    """
    if model_type not in MODEL_DRIVERS:
        raise ValueError(f"Unknown model_type '{model_type}'. Known: {list(MODEL_DRIVERS)}")

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    params_path = export_dir / "cad_params_used.json"
    params_path.write_text(json.dumps(cad_params, indent=2))

    driver_path = export_dir / "cad_driver.py"
    if model_type == "continuous_twist":
        driver_path.write_text(_build_driver_script())
    else:
        # New models live as standalone scripts; driver stub simply invokes them.
        driver_path.write_text(_build_delegator_script(MODEL_DRIVERS[model_type]))

    log_path = export_dir / "cad_run.log"

    completed = run_in_env(
        cad_env,
        ["python", str(driver_path), "--params", str(params_path), "--export-dir", str(export_dir)],
        cwd=export_dir,
        timeout=timeout,
    )

    log_path.write_text(
        f"# stdout\n{completed.stdout or ''}\n\n# stderr\n{completed.stderr or ''}\n"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cad_runner failed (rc={completed.returncode}). See {log_path}")

    step_path = export_dir / "compound_enamel_lattice.step"
    stl_path = export_dir / "compound_enamel_lattice.stl"
    sidecar_path = export_dir / "lattice_params.json"

    _verify_sidecar(sidecar_path)

    return CadRunResult(
        export_dir=export_dir,
        step_path=step_path,
        stl_path=stl_path,
        sidecar_path=sidecar_path,
        log_path=log_path,
        model_type=model_type,
    )


def _build_driver_script() -> str:
    """Driver script: loads cad_params.json and calls stock generate_lattice().

    Written as a string so we don't need to import cadquery at planning time.
    The script runs inside cad_env where cadquery is available.
    """
    stock_dir = str(CAD_STACK_DIR).replace("\\", "/")
    return f'''"""Auto-generated CAD driver for biomimetic_pipeline."""
import argparse
import json
import os
import sys

# Make the stock CAD module importable without copying or modifying it.
_STOCK_DIR = r"{stock_dir}"
if _STOCK_DIR not in sys.path:
    sys.path.insert(0, _STOCK_DIR)

import lattice_cad  # noqa: E402


def _coerce_ring_rotation(rr):
    if rr is None:
        return None
    coerced = {{}}
    for k, v in rr.items():
        try:
            coerced[int(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return coerced


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    p.add_argument("--export-dir", required=True)
    args = p.parse_args()

    with open(args.params) as fh:
        user_params = json.load(fh)

    # Coerce RING_ROTATION keys int. per_ring_diameter likewise (not used by stock).
    if "RING_ROTATION" in user_params:
        user_params["RING_ROTATION"] = _coerce_ring_rotation(user_params["RING_ROTATION"])

    # Strip keys the stock module doesnt know about; log them for provenance.
    known_keys = set(lattice_cad.DEFAULTS.keys())
    stock_params = {{}}
    unknown = {{}}
    for k, v in user_params.items():
        if k in known_keys:
            stock_params[k] = v
        else:
            unknown[k] = v
    stock_params["EXPORT_DIR"] = args.export_dir

    # If TWIST_TYPE is "measured" we fall back to sigmoid for stock runs; the
    # measured_profile_twist generator (Phase 4) handles the real case.
    if stock_params.get("TWIST_TYPE") == "measured":
        print("[cad_driver] TWIST_TYPE='measured' not supported by stock lattice_cad; "
              "falling back to 'sigmoid' for this run.")
        stock_params["TWIST_TYPE"] = "sigmoid"

    result = lattice_cad.generate_lattice(stock_params)

    sidecar = {{
        "bridge_elevations": result["bridge_elevations"],
        "model_z_min": result["model_z_min"],
        "model_z_max": result["model_z_max"],
        "specimen_height": result["model_z_max"] - result["model_z_min"],
        "cut_top_z": result["model_z_max"] - stock_params.get("PLATE_THICKNESS", lattice_cad.DEFAULTS["PLATE_THICKNESS"]),
        "plate_overlap": stock_params.get("PLATE_OVERLAP", lattice_cad.DEFAULTS["PLATE_OVERLAP"]),
        "n_rods": result["n_rods"],
        "n_bridges": result["n_bridges"],
        "params": {{k: v for k, v in stock_params.items() if k != "EXPORT_DIR"}},
        "biomimetic_unknown_keys": unknown,
    }}
    sidecar_path = os.path.join(args.export_dir, "lattice_params.json")
    with open(sidecar_path, "w") as fh:
        json.dump(sidecar, fh, indent=2, default=str)
    print(f"Sidecar written: {{sidecar_path}}")


if __name__ == "__main__":
    main()
'''


def _build_delegator_script(target_path: str) -> str:
    """Driver stub that re-execs a new-model script with the same args."""
    return f'''"""Auto-generated delegator. Hands off to a new-model script."""
import os
import subprocess
import sys

_TARGET = r"{target_path}"

def main():
    rc = subprocess.call([sys.executable, _TARGET] + sys.argv[1:])
    sys.exit(rc)

if __name__ == "__main__":
    main()
'''


def _verify_sidecar(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"lattice_params.json sidecar missing: {path}")
    data = json.loads(path.read_text())
    for required in ("bridge_elevations", "specimen_height", "model_z_min", "model_z_max"):
        if required not in data:
            raise RuntimeError(f"sidecar {path} missing key '{required}'")
