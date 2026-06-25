#!/usr/bin/env python
"""
test_value_carrythrough.py
==========================

Purpose
-------
Golden / regression tests that lock the morphometrics -> cad_params VALUE flow,
written after a value-flow audit.
The motivating bug: overriding ROD_DIAMETER silently RE-DERIVED BRIDGE_DIAMETER
(= bridge_ratio x ROD), shipping inflated 2.47 mm thick bridges instead of the
intended 1.702 mm. These tests pin the correct contract so that regression — and
its whole class (silent re-derivation / mis-scaling of measured-or-pinned values)
— can never ship silently again.

Why this exists
---------------
The manuscript's central claim is ZERO FREE PARAMETERS: every CAD parameter must
trace to a measured morphometric or a documented constant/clamp. A silent
re-derivation breaks that traceability invisibly. Unit-level tests on
map_morphometrics() catch it without the (slow) CAD/FEA subprocess path.

Scope (this file, growing)
--------------------------
1. Override of BRIDGE_DIAMETER is HONORED (not re-derived from ROD).
2. Override of CENTER_SPACING is HONORED (not bumped to 1.05 x ROD).
3. A lone ROD_DIAMETER override RAISES (silent-coupling footgun guard).
4. Canonical no-override carry-through reproduces the golden cad_params exactly.

Inputs: the canonical specimen morphometrics, committed as tests/fixtures/morphometrics.json.
Outputs: assertions; exit 0 if all pass (run directly or via tests/run_all.py).
Side effects: none (pure mapping, no CAD/FEA subprocess).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# --- locate package root + canonical morphometrics --------------------------
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biomimetic_pipeline.mapping import feature_to_cad  # noqa: E402

# Canonical specimen morphometrics, committed as a self-contained test fixture
# (was runs/live_001/morphometrics.json in the research tree) so the regression
# test runs without the gitignored runs/ output directory present.
MORPH_PATH = THIS.parent / "fixtures" / "morphometrics.json"

# Golden canonical (biomimetic, no overrides except N) values — verified to the
# last digit in the audit. ROD = um_to_mm_scaled(mean rod um, 500x); BRIDGE =
# 0.80 x ROD; CENTER_SPACING = 1.5 x ROD (ceiling clamp).
GOLD_ROD = 2.127954912891295
GOLD_BRIDGE = 1.702363930313036
GOLD_CS = 3.1919323693369425
# The thick-rod variant's intended geometry (grow rod, KEEP bridge & spacing).
THICK_ROD = 3.1669323693369393
TOL = 1e-9


def _load_morph():
    return json.loads(MORPH_PATH.read_text())


def test_bridge_diameter_override_is_honored():
    """Explicit BRIDGE_DIAMETER must survive — NOT be re-derived to 0.80 x ROD.

    This is the exact regression that shipped the inflated 2.47 mm thick bridges.
    Pinning ROD=thick AND BRIDGE=1.702 must emit BRIDGE=1.702, not 0.80*3.167=2.53.
    """
    morph = _load_morph()
    cad = feature_to_cad.map_morphometrics(
        morph,
        morphometrics_source=MORPH_PATH,
        extra_overrides={"ROD_DIAMETER": THICK_ROD, "BRIDGE_DIAMETER": GOLD_BRIDGE},
    )
    assert abs(cad["BRIDGE_DIAMETER"] - GOLD_BRIDGE) < TOL, (
        f"BRIDGE_DIAMETER was re-derived to {cad['BRIDGE_DIAMETER']} instead of the "
        f"pinned {GOLD_BRIDGE} (0.80*ROD inflation bug)"
    )
    assert abs(cad["ROD_DIAMETER"] - THICK_ROD) < TOL


def test_center_spacing_override_is_honored():
    """Explicit CENTER_SPACING must survive — NOT be bumped to 1.05 x ROD.

    The thick rule puts CS only 0.025 mm above ROD (<5%), so the old 1.05*rod
    floor silently fired on every thick variant.
    """
    morph = _load_morph()
    cad = feature_to_cad.map_morphometrics(
        morph,
        morphometrics_source=MORPH_PATH,
        extra_overrides={
            "ROD_DIAMETER": THICK_ROD,
            "BRIDGE_DIAMETER": GOLD_BRIDGE,
            "CENTER_SPACING": GOLD_CS,
        },
    )
    assert (
        abs(cad["CENTER_SPACING"] - GOLD_CS) < TOL
    ), f"CENTER_SPACING was bumped to {cad['CENTER_SPACING']} instead of the pinned {GOLD_CS}"


def test_lone_rod_override_raises():
    """Overriding ROD without BRIDGE must FAIL LOUDLY, not silently re-scale BRIDGE.

    The coupling footgun: a lone ROD override would re-derive BRIDGE=0.80*ROD.
    Thick variants must co-specify BRIDGE (or use make_thick_rod_variant.py).
    """
    morph = _load_morph()
    raised = False
    try:
        feature_to_cad.map_morphometrics(
            morph,
            morphometrics_source=MORPH_PATH,
            extra_overrides={"ROD_DIAMETER": THICK_ROD},
        )
    except ValueError:
        raised = True
    assert (
        raised
    ), "lone ROD_DIAMETER override should raise ValueError (silent BRIDGE re-derivation)"


def test_canonical_carrythrough_golden():
    """No-override mapping reproduces the golden biomimetic cad_params exactly.

    Pins the known-good measured->CAD carry-through; any future mapper/scale/clamp
    change that perturbs a measured-derived value trips this immediately.
    """
    morph = _load_morph()
    cad = feature_to_cad.map_morphometrics(
        morph,
        morphometrics_source=MORPH_PATH,
        extra_overrides={"N_BRIDGE_LAYERS": 8},
    )
    assert abs(cad["ROD_DIAMETER"] - GOLD_ROD) < TOL, cad["ROD_DIAMETER"]
    assert abs(cad["BRIDGE_DIAMETER"] - GOLD_BRIDGE) < TOL, cad["BRIDGE_DIAMETER"]
    assert abs(cad["CENTER_SPACING"] - GOLD_CS) < TOL, cad["CENTER_SPACING"]
    # Invariants: BRIDGE = 0.80*ROD (derived), CENTER_SPACING = 1.5*ROD (ceiling).
    assert abs(cad["BRIDGE_DIAMETER"] - 0.80 * cad["ROD_DIAMETER"]) < TOL
    assert abs(cad["CENTER_SPACING"] - 1.5 * cad["ROD_DIAMETER"]) < TOL


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(
        f"\n{'ALL PASSED' if failed == 0 else str(failed) + ' FAILED'} ({len(tests)} value-carry-through tests)"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
