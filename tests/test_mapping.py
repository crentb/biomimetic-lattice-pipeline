"""Mapping layer tests using synthetic morphometrics.

Verifies:
  - twist integration produces monotone RING_ROTATION when pitch is all-positive
  - bridge layer count is clamped to [2, 8]
  - rod diameter clamps to SLA min-feature
  - eccentricity > 0.2 triggers elliptical cross-section
  - non-monotone rod diameter profile triggers per_ring_diameter
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biomimetic_pipeline.mapping import feature_to_cad


def _synthetic_morphometrics(
    pitch_signed=None,
    rod_diameter_um=None,
    eccentricity=None,
    dominant_wavelength_um=85.0,
    band_width_um=20.0,
    band0_dir=0.0,
):
    pitch = pitch_signed or [1.0] * 11
    diam = rod_diameter_um or [4.0] * 11
    ecc = eccentricity or [0.1] * 11
    depth = [i * 2.0 for i in range(len(pitch))]
    return {
        "schema_version": "1.0.0",
        "specimen_id": "synth",
        "units": {"length": "um", "angle": "deg", "stress": "MPa"},
        "provenance": {
            "source_paths": {},
            "source_hashes": {},
            "ingestion_timestamp_iso": "2026-04-13T00:00:00+00:00",
            "coordinate_frame": {"z_origin": "s", "z_axis": "s", "xy_units": "um", "z_units": "um"},
        },
        "depth_profiles": {
            "depth_um": depth,
            "pitch_signed_deg": pitch,
            "rod_diameter_um_mean": diam,
            "rod_diameter_um_std": [0.2] * len(diam),
            "eccentricity_mean": ecc,
            "eccentricity_std": [0.05] * len(ecc),
        },
        "angle_stats": {
            "pitch": {
                "n": 100,
                "mean": 0.0,
                "median": 0.0,
                "std": 10.0,
                "p5": -20.0,
                "p95": 20.0,
                "min": -30.0,
                "max": 30.0,
            },
            "yaw": {
                "n": 100,
                "mean": 0.0,
                "median": 0.0,
                "std": 10.0,
                "p5": -20.0,
                "p95": 20.0,
                "min": -30.0,
                "max": 30.0,
            },
            "tilt": {
                "n": 100,
                "mean": 15.0,
                "median": 15.0,
                "std": 8.0,
                "p5": 3.0,
                "p95": 27.0,
                "min": 0.0,
                "max": 40.0,
            },
        },
        "bands": [
            {
                "band_id": 0,
                "area_fraction_pct": 25.0,
                "mean_direction_deg": band0_dir,
                "circular_variance": 0.1,
                "mean_displacement_mag_um": 2.0,
                "band_width_um": {"mean": band_width_um, "std": 5.0, "min": 10.0, "max": 40.0},
            },
            {
                "band_id": 1,
                "area_fraction_pct": 25.0,
                "mean_direction_deg": 90.0,
                "circular_variance": 0.1,
                "mean_displacement_mag_um": 2.0,
                "band_width_um": {"mean": band_width_um, "std": 5.0, "min": 10.0, "max": 40.0},
            },
        ],
        "periodicity": {
            "dominant_wavelength_um_mean": dominant_wavelength_um,
            "dominant_wavelength_um_std": 10.0,
        },
        "rod_slice_stats": [],
        "rod_tracks_summary": {
            "n_tracks": 100,
            "n_complete": 100,
            "mean_arc_length_um": 10.0,
            "mean_chord_length_um": 10.0,
            "tortuosity_mean": 1.0,
            "tortuosity_p90": 1.0,
        },
    }


def test_positive_pitch_gives_monotone_ring_rotation():
    m = _synthetic_morphometrics(pitch_signed=[1.0] * 11)
    cad = feature_to_cad.map_morphometrics(m)
    ring = cad["RING_ROTATION"]
    vals = [ring[str(i)] for i in range(int(cad["N_RINGS"]) + 1)]
    # Strictly increasing (plus the band0 anchor).
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    assert all(d >= 0 for d in diffs), f"Ring rotation not monotone: {vals}"


def test_n_bridge_layers_clamped():
    # Very small wavelength -> huge raw N -> clamp to 8
    m = _synthetic_morphometrics(dominant_wavelength_um=10.0)
    cad = feature_to_cad.map_morphometrics(m)
    assert 2 <= cad["N_BRIDGE_LAYERS"] <= 8

    # Very large wavelength -> tiny raw N -> clamp to 2
    m2 = _synthetic_morphometrics(dominant_wavelength_um=100000.0)
    cad2 = feature_to_cad.map_morphometrics(m2)
    assert cad2["N_BRIDGE_LAYERS"] == 2


def test_rod_diameter_sla_clamp_logged():
    # Tiny biological rod (0.1 um) * 500x scale = 0.05 mm, under 0.3 mm SLA floor.
    m = _synthetic_morphometrics(rod_diameter_um=[0.1] * 11)
    cad = feature_to_cad.map_morphometrics(m)
    # ROD is bumped to 2*SLA_min + 0.05 = 0.65 mm so BRIDGE (at 0.5*ROD) stays > SLA floor.
    assert cad["ROD_DIAMETER"] >= 0.3
    assert cad["BRIDGE_DIAMETER"] < cad["ROD_DIAMETER"]
    assert cad["BRIDGE_DIAMETER"] >= 0.3  # SLA floor
    clamp_fields = [c["field"] for c in cad["provenance"]["scale_clamps"]]
    assert "ROD_DIAMETER" in clamp_fields


def test_high_eccentricity_triggers_ellipse():
    m = _synthetic_morphometrics(eccentricity=[0.5] * 11)
    cad = feature_to_cad.map_morphometrics(m)
    assert "rod_cross_section" in cad
    assert cad["rod_cross_section"]["shape"] == "ellipse"
    assert cad["rod_cross_section"]["aspect_ratio"] >= 1.5


def test_non_monotone_diameter_triggers_per_ring():
    # Zig-zag diameter profile should trigger per_ring_diameter
    m = _synthetic_morphometrics(rod_diameter_um=[4, 5, 3, 6, 2, 7, 4, 5, 3, 6, 4])
    cad = feature_to_cad.map_morphometrics(m)
    assert cad.get("per_ring_diameter") is not None
    # Key format must be strings (schema requirement)
    for k in cad["per_ring_diameter"].keys():
        assert isinstance(k, str)


def test_band_direction_anchors_ring_zero():
    m = _synthetic_morphometrics(band0_dir=45.0)
    cad = feature_to_cad.map_morphometrics(m)
    # ring 0 should be ≥ ~45.0 (the anchor, plus whatever twist starts at 0)
    assert float(cad["RING_ROTATION"]["0"]) >= 44.0


if __name__ == "__main__":
    test_positive_pitch_gives_monotone_ring_rotation()
    test_n_bridge_layers_clamped()
    test_rod_diameter_sla_clamp_logged()
    test_high_eccentricity_triggers_ellipse()
    test_non_monotone_diameter_triggers_per_ring()
    test_band_direction_anchors_ring_zero()
    print("All mapping tests passed.")
