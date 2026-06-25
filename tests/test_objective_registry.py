"""Objective registry tests: YAML loading, scoring, direction semantics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biomimetic_pipeline.objectives import registry


def test_all_builtin_objectives_load():
    for name in (
        "crack_deflection",
        "toughness",
        "stiffness_target",
        "biomimicry_score",
        "composite",
    ):
        obj = registry.load_builtin(name)
        assert obj.name == name
        assert obj.direction in ("minimize", "maximize")
        assert len(obj.terms) >= 1


def test_crack_deflection_scoring_monotone_in_tortuosity():
    obj = registry.load_builtin("crack_deflection")
    low = obj.score(
        {
            "crack_deflection_tortuosity_p90": 1.0,
            "crack_deflection_tortuosity_mean": 1.0,
            "critical_strain_at_200MPa": 0.01,
            "avg_von_mises_MPa": 200.0,
        }
    )
    high = obj.score(
        {
            "crack_deflection_tortuosity_p90": 2.0,
            "crack_deflection_tortuosity_mean": 2.0,
            "critical_strain_at_200MPa": 0.01,
            "avg_von_mises_MPa": 200.0,
        }
    )
    assert obj.better(high, low), f"Higher tortuosity should score better: {low} vs {high}"


def test_stress_penalty_applied_when_off_target():
    obj = registry.load_builtin("crack_deflection")
    on_target = obj.score(
        {
            "crack_deflection_tortuosity_p90": 1.5,
            "crack_deflection_tortuosity_mean": 1.2,
            "critical_strain_at_200MPa": 0.02,
            "avg_von_mises_MPa": 200.0,  # exactly target
        }
    )
    off_target = obj.score(
        {
            "crack_deflection_tortuosity_p90": 1.5,
            "crack_deflection_tortuosity_mean": 1.2,
            "critical_strain_at_200MPa": 0.02,
            "avg_von_mises_MPa": 400.0,  # 2x target
        }
    )
    # direction=maximize, so on_target >= off_target (penalty subtracted for maximize)
    assert on_target >= off_target


def test_stiffness_target_minimize_direction():
    obj = registry.load_builtin("stiffness_target")
    assert obj.direction == "minimize"
    close = obj.score({"E_effective_MPa": 50000.0, "avg_von_mises_MPa": 200.0})
    far = obj.score({"E_effective_MPa": 10000.0, "avg_von_mises_MPa": 200.0})
    # Lower (closer to target) is better for minimize direction.
    assert close < far, f"Stiffness closer to target should score lower: close={close} far={far}"


if __name__ == "__main__":
    test_all_builtin_objectives_load()
    test_crack_deflection_scoring_monotone_in_tortuosity()
    test_stress_penalty_applied_when_off_target()
    test_stiffness_target_minimize_direction()
    print("All objective tests passed.")
