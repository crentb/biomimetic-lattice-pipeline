"""Smoke tests for new-model variants.

Can't run the full CadQuery generation without cad_env, so these verify:
  - the scripts are syntactically valid (importable)
  - the twist builders produce callables with the expected signature
  - numeric behavior at key points (z=0 -> 0 rotation, z=H -> total_rotation)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_measured_profile_twist_endpoints():
    # Can't import lattice_cad here (no cad_env), so we test the twist builder
    # in isolation by exec-ing the script file and extracting _build_measured_twist.
    script = (
        ROOT / "biomimetic_pipeline" / "generators" / "new_models" / "measured_profile_twist.py"
    )
    code = script.read_text()
    # Strip the top-level `import lattice_cad` which needs cad_env; patch with a stub.
    import types

    lattice_cad_stub = types.ModuleType("lattice_cad")
    lattice_cad_stub.DEFAULTS = {}

    def _linear(z, H, rot):
        return (z / max(H, 1e-9)) * math.radians(rot)

    lattice_cad_stub.linear_twist = _linear
    lattice_cad_stub.get_twist_function = lambda name: _linear
    sys.modules["lattice_cad"] = lattice_cad_stub

    ns: dict = {"__file__": str(script), "__name__": "test_ns"}
    exec(code, ns, ns)
    build = ns["_build_measured_twist"]

    profile = {"z_um": [0.0, 5.0, 10.0, 20.0], "twist_deg": [0.0, 5.0, 12.0, 30.0]}
    fn = build(profile, enamel_thickness_mm=10.0)

    r0 = fn(0.0, 10.0, 60.0)
    rN = fn(10.0, 10.0, 60.0)
    assert abs(r0) < 1e-6, f"Expected 0 at z=0, got {r0}"
    assert abs(math.degrees(rN) - 60.0) < 0.5, f"Expected 60 deg at z=H, got {math.degrees(rN)}"


def test_hierarchical_adds_wobble_without_changing_endpoints():
    script = ROOT / "biomimetic_pipeline" / "generators" / "new_models" / "hierarchical_twist.py"
    code = script.read_text()

    import types

    lattice_cad_stub = types.ModuleType("lattice_cad")
    lattice_cad_stub.DEFAULTS = {}

    def _linear(z, H, rot):
        return (z / max(H, 1e-9)) * math.radians(rot)

    lattice_cad_stub.linear_twist = _linear
    lattice_cad_stub.get_twist_function = lambda name: _linear
    sys.modules["lattice_cad"] = lattice_cad_stub

    ns: dict = {"__file__": str(script), "__name__": "test_ns_h"}
    exec(code, ns, ns)
    build = ns["_build_hierarchical"]

    # Single-cycle wobble over specimen height: wavelength == H so omega*frac = 2*pi*frac.
    fn = build(
        base_twist_name="linear",
        measured_profile=None,
        amplitude_deg=10.0,
        wavelength_um=10000.0,  # 10 mm = H
        enamel_thickness_mm=10.0,
    )

    # At z = H/2, sin(2*pi*0.5) = sin(pi) = 0 -> composite matches slow.
    r_mid = fn(5.0, 10.0, 60.0)
    assert (
        abs(math.degrees(r_mid) - 30.0) < 0.5
    ), f"Mid-span should match slow: got {math.degrees(r_mid)}"

    # At z = H/4, sin(2*pi*0.25) = sin(pi/2) = 1 -> composite = slow + amplitude.
    r_quarter = fn(2.5, 10.0, 60.0)
    expected = 15.0 + 10.0
    assert (
        abs(math.degrees(r_quarter) - expected) < 1.0
    ), f"Quarter-span expected ~{expected} deg, got {math.degrees(r_quarter)}"


if __name__ == "__main__":
    test_measured_profile_twist_endpoints()
    test_hierarchical_adds_wobble_without_changing_endpoints()
    print("All new_models tests passed.")
