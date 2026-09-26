#!/usr/bin/env python
"""Two mechanistic controls for the twist-coupled load partition (2026-09).

Purpose
-------
The v2 manuscript reports that, under axial compression with a free-to-rotate
top platen, the continuous-twist lattice rotates and its outermost, most
inclined rods go into tension. A referee-style review asked (i) whether this
is caused by the lattice's NET CHIRALITY (every ring rotates in the same
sense) rather than by rods crossing per se, and (ii) whether it survives a
platen that cannot rotate. This driver runs both controls against the
reference lattice (f = 1, N = 2, 20-mm height, 0.5-mm mesh):

  platen   same mesh and material as runs/live_001_digital_twin_v2, but the top
           face is clamped in-plane (u_x = u_y = 0) as well as displaced
           axially -- a bonded, non-rotating platen. Solved directly with the
           upstream SfePy problem file, copied and edited (BCs only) into the
           new run folder; the original file is untouched.
  achiral  the reference ring-rotation magnitudes with alternating sign from
           ring to ring {+2.3, -5.8, +15.1, -31.0, +45.0, -54.2} deg, so
           neighbouring rings twist in opposite senses (closer to the
           alternating handedness of Hunter-Schreger bands) and the net twist
           largely cancels. Full pipeline, zero bridge jitter (as the
           reference).

Inputs
------
runs/live_001_digital_twin_v2/{mesh/compound_enamel_lattice.msh, cad_params.json,
    fea/iter_2/compression_test.py}
runs/live_001/morphometrics.json

Outputs
-------
runs/control_platen_fixed_v1/  cad_params.json (copy), fea/final/* (SfePy output)
runs/control_achiral_v1/trial_000_alternating/   full pipeline run
runs/control_status_2026_09.json   per-control status

Side effects / notes
--------------------
* The platen control is linear, so its displacement (1 mm) only sets the
  scale; every reported quantity is a ratio.
* Run after the other FEA campaigns (memory-bound; do not overlap).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve()
BIO = THIS.parent.parent
if str(BIO) not in sys.path:
    sys.path.insert(0, str(BIO))
os.environ["OCCT_TANGENT_JITTER_MM"] = "0"  # canonical bridge elevations
from biomimetic_pipeline.orchestration import pipeline  # noqa: E402

REF = BIO / "biomimetic_pipeline/runs/live_001_digital_twin_v2"
STATUS = BIO / "biomimetic_pipeline/runs/control_status_2026_09.json"


def platen_fixed() -> dict:
    """Reference mesh, top face clamped in-plane (non-rotating platen)."""
    out = BIO / "biomimetic_pipeline/runs/control_platen_fixed_v1"
    fea = out / "fea" / "final"
    fea.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REF / "cad_params.json", out / "cad_params.json")
    shutil.copy2(REF / "mesh/compound_enamel_lattice.msh", fea / "compound_enamel_lattice.msh")
    src = (REF / "fea/iter_2/compression_test.py").read_text()
    old = "    'compress_top': ('Top',  {'u.2': compress_disp}),"
    new = (
        "    # CONTROL (2026-09): non-rotating, bonded platen -- in-plane motion of the\n"
        "    # top face suppressed; everything else identical to the reference run.\n"
        "    'compress_top': ('Top',  {'u.0': 0.0, 'u.1': 0.0, 'u.2': compress_disp}),"
    )
    assert src.count(old) == 1, "unexpected boundary-condition block"
    (fea / "compression_test.py").write_text(src.replace(old, new))
    env = dict(os.environ, MATERIAL_E="3000.0", MATERIAL_NU="0.4", COMPRESS_DISP_MM="1.0")
    t0 = time.time()
    r = subprocess.run(
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "sfepy_env",
            "sfepy-run",
            "compression_test.py",
        ],
        cwd=str(fea),
        env=env,
        capture_output=True,
        text=True,
    )
    (fea / "fea_run.log").write_text(r.stdout + "\n" + r.stderr)
    ok = r.returncode == 0 and (fea / "compound_enamel_lattice.vtk").exists()
    return {
        "state": "ok" if ok else "failed",
        "rc": r.returncode,
        "wall_s": round(time.time() - t0, 1),
    }


def achiral() -> dict:
    """Alternating-handedness ring rotations at the reference magnitudes."""
    ref_rot = json.loads((REF / "cad_params.json").read_text())["RING_ROTATION"]
    alt = {k: (abs(float(v)) * (1.0 if int(k) % 2 == 0 else -1.0)) for k, v in ref_rot.items()}
    t0 = time.time()
    try:
        pipeline.run_pipeline(
            morphometrics_path=BIO / "biomimetic_pipeline/runs/live_001/morphometrics.json",
            run_name="control_achiral_v1/trial_000_alternating",
            objective_name="crack_deflection",
            model_type="continuous_twist",
            mesh_size_mm=0.5,
            extra_overrides={"RING_ROTATION": alt},
            skip_probe=True,
            allow_broken_cad=False,
        )
        return {"state": "ok", "ring_rotation_deg": alt, "wall_s": round(time.time() - t0, 1)}
    except Exception as exc:  # noqa: BLE001 -- record CAD/mesh failures
        return {
            "state": "failed",
            "error": repr(exc),
            "ring_rotation_deg": alt,
            "wall_s": round(time.time() - t0, 1),
        }


def main() -> None:
    status = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    for name, fn in (("platen_fixed", platen_fixed), ("achiral", achiral)):
        if status.get(name, {}).get("state") == "ok":
            continue
        print(f"[run ] control {name}", flush=True)
        status[name] = fn()
        print(
            f"[{'ok' if status[name]['state'] == 'ok' else 'FAIL'}] control {name} {status[name]}",
            flush=True,
        )
        STATUS.write_text(json.dumps(status, indent=1))


if __name__ == "__main__":
    main()
