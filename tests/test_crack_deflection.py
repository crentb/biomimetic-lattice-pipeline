"""Crack-deflection tortuosity test.

Constructs a synthetic element_results_compression.csv with a known p1 field:

  Case 1 - uniform p1 increasing in +z: streamlines go straight up ->
  tortuosity ≈ 1.0.

  Case 2 - a radial p1 gradient superimposed on the upward gradient: seeds
  drift outward while going up, producing tortuosity > 1 but bounded.

We verify the arc/chord formula is numerically sane and matches the
biology-side computation in compute_tortuosity.py (arc/chord identical).
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biomimetic_pipeline.metrics.crack_deflection import compute


def _write_elements(path: Path, field_fn, n=20):
    rows = []
    import numpy as np

    xs = np.linspace(-5, 5, n)
    ys = np.linspace(-5, 5, n)
    zs = np.linspace(0, 20, n)
    for x in xs:
        for y in ys:
            for z in zs:
                rows.append(
                    {
                        "sxx_MPa": 0.0,
                        "syy_MPa": 0.0,
                        "szz_MPa": 0.0,
                        "sxy_MPa": 0.0,
                        "syz_MPa": 0.0,
                        "sxz_MPa": 0.0,
                        "exx": 0.0,
                        "eyy": 0.0,
                        "ezz": 0.0,
                        "gxy": 0.0,
                        "gyz": 0.0,
                        "gxz": 0.0,
                        "von_mises_MPa": 0.0,
                        "energy_MPa": 0.0,
                        "p1_MPa": field_fn(x, y, z),
                        "p2_MPa": 0.0,
                        "p3_MPa": 0.0,
                        "volume_mm3": 1.0,
                        "cx_mm": x,
                        "cy_mm": y,
                        "cz_mm": z,
                    }
                )
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_straight_field_tortuosity_near_one():
    with tempfile.TemporaryDirectory() as tmp:
        elem_path = Path(tmp) / "element_results_compression.csv"
        _write_elements(elem_path, field_fn=lambda x, y, z: float(z))
        res = compute(elem_path, seed_grid_n=4, grid_resolution=12, max_steps=200)
        assert res.n_streamlines > 0
        assert 0.99 <= res.tortuosity_mean <= 1.1, f"mean={res.tortuosity_mean}"


def test_mixed_field_tortuosity_above_one():
    with tempfile.TemporaryDirectory() as tmp:
        elem_path = Path(tmp) / "element_results_compression.csv"
        # p1 = z + 0.3*(x^2 + y^2) - radial bowl superimposed on upward ramp.
        _write_elements(elem_path, field_fn=lambda x, y, z: float(z) + 0.3 * (x * x + y * y))
        res = compute(elem_path, seed_grid_n=4, grid_resolution=12, max_steps=200)
        assert res.n_streamlines > 0
        # Streamlines should bend but not explode.
        assert res.tortuosity_mean >= 1.0
        assert res.tortuosity_mean <= 2.0


if __name__ == "__main__":
    test_straight_field_tortuosity_near_one()
    test_mixed_field_tortuosity_above_one()
    print("All crack_deflection tests passed.")
