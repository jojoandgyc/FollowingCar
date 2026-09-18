from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from car_control_modular.control_types import DepthTargetObservation, PersonTarget
from car_control_modular.distance_runtime import DistanceRuntime, DistanceRuntimeConfig


def make_runtime(**overrides):
    values = dict(
        distance_source="vision_depth",
        vision_mmwave_source_aliases=frozenset({"vision_mmwave"}),
        module_mmwave_enable=False, module_ultrasonic_enable=False,
        vision_hfov_deg=60.0, vision_mmwave_angle_margin_deg=8.0,
        vision_mmwave_angle_offset_deg=0.0, vision_mmwave_angle_sign=1.0,
        vision_mmwave_match_mode="angle", vision_mmwave_distance_bias_m=0.0,
        vision_mmwave_min_distance_m=0.5, vision_mmwave_max_distance_m=10.0,
        vision_mmwave_min_output_distance_m=0.03,
        vision_mmwave_hard_stop_ttl_sec=0.3, vision_mmwave_log_every_frames=0,
        module_astra_depth_enable=True, vision_depth_require_detector_bbox=True,
        vision_depth_max_distance_jump_m=0.6, vision_depth_jump_confirm_frames=5,
    )
    values.update(overrides)

    class Sensors:
        def __init__(self):
            self.calls = []
            self.measurement = SimpleNamespace(
                distance_m=2.0, raw_distance_m=2.0, sample_age_sec=0.02,
                valid_pixels=500, detail="depth_multiregion",
            )

        def get_astra_target_distance(self, bbox, width, height, **kwargs):
            self.calls.append((bbox, width, height, kwargs))
            return self.measurement

    sensors = Sensors()
    runtime = DistanceRuntime(object(), DistanceRuntimeConfig(**values), sensor_runtime=sensors)
    return runtime, sensors


def person(**observation_changes):
    # CAP1015: tracking expansion alone crosses the 90%-height clipped guard.
    detector = (255.9901, 78.5458, 389.8033, 455.6407)
    display = (242.5452, 40.2886, 415.2570, 479.0)
    observation = DepthTargetObservation(detector, 1, 11, 1015, 99.9)
    observation = replace(observation, **observation_changes)
    return PersonTarget(display, 1, 0.9, 75775.0, depth_observation=observation)


@pytest.fixture(autouse=True)
def stable_clock(monkeypatch):
    monkeypatch.setattr("car_control_modular.distance_runtime.time.monotonic", lambda: 100.0)


def test_detector_bbox_drives_depth_and_visual_fusion_without_mutating_yaw():
    runtime, sensors = make_runtime()
    target = person()
    state = runtime.get_vision_depth_state(640, 480, target, capture_timestamp=99.9)
    bbox, width, height, kwargs = sensors.calls[0]
    assert bbox == target.depth_observation.bbox
    assert bbox != target.bbox
    assert (width, height, kwargs["target_id"]) == (640, 480, 1)
    assert kwargs["reference_timestamp"] == 99.9
    assert kwargs["evidence_capture_frame_id"] == 1015
    assert state.used_distance_m == 2.0
    assert runtime._vision_depth_fusion._anchor_bbox_height_px == pytest.approx(377.0949)
    assert target.bbox[-1] == 479.0
    assert runtime._last_vision_depth_target is target


def test_visual_scale_uses_raw_height_not_changed_tracking_height():
    runtime, sensors = make_runtime()
    target = person(bbox=(200.0, 100.0, 400.0, 300.0))
    runtime.get_vision_depth_state(640, 480, target)
    sensors.measurement = SimpleNamespace(
        distance_m=2.0, raw_distance_m=None, sample_age_sec=0.03,
        valid_pixels=0, detail="insufficient_depth_pixels_hold",
    )
    target = replace(target, depth_observation=replace(
        target.depth_observation, bbox=(200.0, 100.0, 400.0, 400.0), capture_frame_id=1016,
    ))
    held = runtime.get_vision_depth_state(640, 480, target)
    assert held.fusion_visual_distance_m == pytest.approx(2.0 * 200.0 / 300.0)
    assert held.used_distance_m == pytest.approx(2.0 * 200.0 / 300.0)


@pytest.mark.parametrize("changes,reason", [
    ({"target_id": 2}, "target_mismatch"),
    ({"target_id": 0}, "invalid_identity"),
    ({"target_id": float("nan")}, "invalid_identity"),
    ({"raw_track_id": 0}, "invalid_identity"),
    ({"raw_track_id": -2}, "invalid_identity"),
    ({"raw_track_id": 11.5}, "invalid_identity"),
    ({"capture_frame_id": 0}, "invalid_identity"),
    ({"capture_frame_id": None}, "invalid_observation"),
    ({"source": "deepsort"}, "invalid_source"),
    ({"bbox": (0, 1, float("nan"), 400)}, "invalid_geometry"),
    ({"bbox": (0, 1, float("inf"), 400)}, "invalid_geometry"),
    ({"bbox": (0, 1, 200)}, "invalid_geometry"),
    ({"bbox": (-1, 1, 200, 400)}, "outside_frame"),
    ({"bbox": (10, 1, 641, 400)}, "outside_frame"),
    ({"bbox": (10, 1, 200, 481)}, "outside_frame"),
    ({"bbox": (10, 10, 10, 400)}, "outside_frame"),
    ({"capture_timestamp": None}, "invalid_observation"),
    ({"capture_timestamp": 0}, "invalid_timestamp"),
    ({"capture_timestamp": float("nan")}, "invalid_timestamp"),
    ({"capture_timestamp": 100.001}, "stale"),
    ({"capture_timestamp": 99.749}, "stale"),
])
def test_invalid_detector_snapshot_clears_cache_and_never_samples(changes, reason):
    runtime, sensors = make_runtime()
    runtime.get_vision_depth_state(640, 480, person())
    invalid = person(**changes)
    state = runtime.get_vision_depth_state(640, 480, invalid)
    assert state.used_distance_m is None
    assert state.raw_distance_m is None
    assert state.source_detail == "depth_detector_bbox_" + reason
    assert len(sensors.calls) == 1
    assert runtime._last_vision_depth_target is None
    assert runtime._vision_depth_fusion._anchor_distance_m is None
    assert runtime.get_recent_vision_depth_state().used_distance_m is None


def test_missing_required_detector_bbox_never_falls_back_to_track_or_cache():
    runtime, sensors = make_runtime()
    runtime.get_vision_depth_state(640, 480, person())
    missing = replace(person(), depth_observation=None)
    result = runtime.get_vision_depth_state(640, 480, missing)
    assert result.source_detail == "depth_detector_bbox_missing"
    assert result.used_distance_m is None and len(sensors.calls) == 1
    assert runtime.get_recent_vision_depth_state().used_distance_m is None
    sensors.measurement = SimpleNamespace(
        distance_m=None, raw_distance_m=None, sample_age_sec=None,
        valid_pixels=0, detail="depth_unavailable",
    )
    assert runtime.get_vision_depth_state(640, 480, person()).used_distance_m is None


def test_legacy_without_observation_still_works_only_when_not_required():
    runtime, sensors = make_runtime(vision_depth_require_detector_bbox=False)
    target = replace(person(), depth_observation=None)
    assert runtime.get_vision_depth_state(640, 480, target).used_distance_m == 2.0
    assert sensors.calls[0][0] == target.bbox
    invalid = person(source="deepsort")
    assert runtime.get_vision_depth_state(640, 480, invalid).used_distance_m is None
    assert len(sensors.calls) == 1


def test_confirmed_detector_probe_uses_stable_uid_not_reserved_raw_id():
    runtime, sensors = make_runtime()
    runtime.get_vision_depth_state(640, 480, person(raw_track_id=-1))
    assert sensors.calls[0][3]["target_id"] == 1


def test_latest_refresh_is_stricter_than_rgb_aligned_visual_request():
    runtime, sensors = make_runtime()
    target = person(capture_timestamp=99.8)
    assert runtime.get_vision_depth_state(640, 480, target).used_distance_m == 2.0
    assert sensors.calls[0][3]["reference_timestamp"] == 99.8
    latest = runtime.get_vision_depth_state(640, 480, target, use_latest_depth=True)
    assert latest.source_detail == "depth_detector_bbox_stale"
    assert len(sensors.calls) == 1


def test_latest_refresh_age_is_configurable_and_does_not_supply_reference_time():
    runtime, sensors = make_runtime(vision_depth_detector_bbox_max_age_sec=0.1)
    runtime.get_vision_depth_state(640, 480, person(capture_timestamp=99.95), use_latest_depth=True)
    assert sensors.calls[0][3]["use_latest_depth"] is True
    assert "reference_timestamp" not in sensors.calls[0][3]
    state = runtime.get_vision_depth_state(640, 480, person(capture_timestamp=99.89), use_latest_depth=True)
    assert state.source_detail == "depth_detector_bbox_stale"
    assert len(sensors.calls) == 1


@pytest.mark.parametrize("timestamp", [99.8, 0.0, float("nan"), float("inf")])
def test_external_capture_timestamp_must_match_bound_snapshot(timestamp):
    runtime, sensors = make_runtime()
    state = runtime.get_vision_depth_state(640, 480, person(), capture_timestamp=timestamp)
    assert state.source_detail == "depth_detector_bbox_capture_mismatch"
    assert not sensors.calls


def test_mmwave_fusion_never_enables_depth_confirmation():
    runtime, _ = make_runtime()
    assert runtime._vision_depth_fusion.config.allow_depth_jump_confirmation
    assert not runtime._distance_fusion.config.allow_depth_jump_confirmation
