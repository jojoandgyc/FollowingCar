"""Real sensor selection + fusion; no camera, serial port or motor runtime."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import request_0513_modular as main
from car_control_modular.astra_depth import AstraDepthConfig, AstraDepthRuntime
from car_control_modular.control_types import ControlAction, ControlDecision, SteeringFeedback
from car_control_modular.depth_roi_policy import bounded_depth_sample_allowed, bounded_roi_history_allowed
from car_control_modular.sensor_modules import SensorRuntime
from test_depth_raw_geometry_runtime import make_runtime, person
from test_lateral_zero_runtime import owner, NOW, _intent
from test_longitudinal_authority_runtime import _frame


@pytest.fixture
def chain(monkeypatch):
    clock = [100.]
    monkeypatch.setattr("car_control_modular.astra_depth.time.monotonic", lambda: clock[0])
    depth = AstraDepthRuntime(AstraDepthConfig(median_window=1))
    depth._np = np
    sensors = SensorRuntime.__new__(SensorRuntime)
    sensors.config = SimpleNamespace(astra_depth_enable=True)
    sensors.astra_depth = depth
    runtime, _ = make_runtime(vision_depth_detector_bbox_max_age_sec=.25)
    runtime.sensor_runtime = sensors
    fb = SteeringFeedback(timestamp=99.99, trustworthy=True,
                          yaw_rate_right_dps=20., raw_yaw_rate_right_dps=22.)
    return runtime, sensors, depth, fb, clock


def frames(depth, *stamps):
    # Latest is intentionally different (background) and out of association
    # bounds. A passing test must actually range the eligible history image.
    depth._depth_history.clear()
    for stamp in stamps:
        image = np.full((480, 640), 5000, dtype=np.uint16)
        x1, y1, x2, y2 = [int(v) for v in person().depth_observation.bbox]
        image[y1:y2, x1:x2] = 2000
        image.setflags(write=False)
        depth._depth_history.append((stamp, image))
    depth._latest_depth = np.full((480, 640), 5000, dtype=np.uint16)
    depth._latest_depth_ts = 99.99


def measure(chain, capture=99.78, **changes):
    runtime, _, _, fb, _ = chain
    target = person(capture_timestamp=capture, **changes)
    return runtime.get_vision_depth_state(640, 480, target, use_latest_depth=True,
                                         capture_timestamp=capture, steering_feedback=fb)


@pytest.mark.parametrize("cap,roi_age", [(213, .2099), (371, .2221), (1628, .1945)])
def test_delayed_turning_roi_uses_newest_frame_inside_original_window(chain, cap, roi_age):
    # The logged processing ages are reproduced, not the unrecorded depth pixels.
    runtime, sensors, depth, _, _ = chain
    capture = 100.-roi_age
    selected = capture+.17
    frames(depth, capture+.10, selected, capture+.19)
    result = measure(chain, capture, capture_frame_id=cap)
    assert sensors.supports_bounded_depth_roi
    assert result.raw_distance_m == 2.
    assert result.used_distance_m == 2.
    assert result.sample_timestamp == pytest.approx(selected)
    assert depth._last_accepted_ts == pytest.approx(selected)
    assert result.sample_age_sec == pytest.approx(100.-selected)
    assert runtime._vision_depth_fusion._last_accepted_depth_sample_ts == pytest.approx(selected)


def test_repeat_does_not_retime_anchor_or_become_new_measurement(chain):
    runtime, _, depth, _, clock = chain
    frames(depth, 99.94)
    first = measure(chain)
    anchor = runtime._vision_depth_fusion._last_accepted_depth_sample_ts
    clock[0] += .01
    repeated = measure(chain)
    assert first.raw_distance_m == 2.
    assert repeated.raw_distance_m is None
    assert runtime._vision_depth_fusion._last_accepted_depth_sample_ts == anchor == 99.94
    assert depth._last_accepted_ts == 99.94


@pytest.mark.parametrize("yaw", [-5, 5])
def test_bounded_sample_grants_composed_axes_with_original_sample_deadline(chain, owner, monkeypatch, yaw):
    monkeypatch.setattr(main, "ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC", .25)
    _, _, depth, fb, _ = chain
    frames(depth, 99.94)
    state = measure(chain, capture_frame_id=213)
    assert state.raw_distance_m == 2.
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: fb)
    _intent(owner, initial_correction_rpm=yaw)
    owner._lateral_intent_last_correction_rpm = yaw
    frame = replace(_frame(stamp=state.sample_timestamp, distance=state.used_distance_m),
                    distance_state=state, steering_feedback=fb)
    decision = ControlDecision(actions=[ControlAction.forward(17, "bounded_history")], reason="bounded_history")
    actions, accepted = owner._commit_depth_linear_decision(decision, frame, 1, is_fresh_depth=True)
    assert accepted and actions[0].speed_percent > 0
    assert owner._depth30_linear_timing.depth_expires_at == pytest.approx(99.94+.25)
    owner._service_lateral_intent(NOW)
    expected = main.ACTION_STEER_RIGHT if yaw > 0 else main.ACTION_STEER_LEFT
    assert owner._queued_calls[-1][0] == (expected,)
    assert owner._current_forward_percent > 0
    assert owner._current_steer_correction_rpm == abs(yaw)


def test_history_cannot_replace_newer_accepted_sensor_anchor(chain):
    _, sensors, depth, _, _ = chain
    frames(depth, 99.94)
    depth._latest_depth_ts = 99.97
    depth._latest_depth.fill(2000)
    current = sensors.get_astra_target_distance(person().depth_observation.bbox, 640, 480,
                                                target_id=1, use_latest_depth=True)
    assert current.sample_timestamp == 99.97
    older = measure(chain)
    assert older.raw_distance_m is None
    assert depth._last_accepted_ts == 99.97


@pytest.mark.parametrize("stamps", [(), (99.77,), (99.97,), (99.80,), (100.01,)])
def test_no_bounded_frame_never_falls_back_to_latest(chain, stamps):
    _, _, depth, _, _ = chain
    frames(depth, *stamps)
    result = measure(chain)
    assert result.raw_distance_m is None
    assert depth._last_accepted_ts == 0.
    assert len(depth._attempted_depth_samples) == 0


def test_no_history_can_use_latest_only_when_physically_in_window(chain):
    _, _, depth, _, _ = chain
    frames(depth)
    depth._latest_depth_ts = 99.94
    depth._latest_depth.fill(2000)
    result = measure(chain)
    assert result.raw_distance_m == 2.
    assert result.sample_timestamp == 99.94


@pytest.mark.parametrize("capture", [99.749, 100.01, float("nan"), 0.])
def test_invalid_or_expired_roi_stays_ineligible(chain, capture):
    _, _, depth, _, _ = chain
    frames(depth, 99.94)
    assert measure(chain, capture).raw_distance_m is None
    assert depth._last_accepted_ts == 0.


@pytest.mark.parametrize("change", [dict(target_id=2), dict(source="predicted"), dict(bbox=(-1, 0, 100, 200))])
def test_new_path_does_not_bypass_uid_source_or_bbox_checks(chain, change):
    _, _, depth, _, _ = chain
    frames(depth, 99.94)
    assert measure(chain, **change).raw_distance_m is None
    assert depth._last_accepted_ts == 0.


@pytest.mark.parametrize("field,value", [
    ("sample_timestamp", 99.99), ("sample_timestamp", None),
    ("sample_timestamp", float("nan")), ("observation_sample_timestamp", 99.99),
    ("observation_source", "latest"),
])
def test_backend_provenance_is_rechecked_before_fusion(chain, monkeypatch, field, value):
    runtime, sensors, depth, _, _ = chain
    frames(depth, 99.94)
    original = sensors.get_astra_target_distance
    monkeypatch.setattr(sensors, "get_astra_target_distance",
                        lambda *args, **kwargs: replace(original(*args, **kwargs), **{field: value}))
    result = measure(chain)
    assert result.raw_distance_m is None
    assert result.source_detail == "bounded_roi_provenance_rejected"
    assert runtime._vision_depth_fusion._last_accepted_depth_sample_ts is None


def test_old_backend_cannot_claim_bounded_selection(chain):
    runtime, sensors, depth, _, _ = chain
    frames(depth, 99.94)
    depth.supports_bounded_depth_roi = False
    assert not sensors.supports_bounded_depth_roi
    assert measure(chain).raw_distance_m is None
    assert depth._last_accepted_ts == 0.
    direct = sensors.get_astra_target_distance(person().bbox, 640, 480,
                                               bounded_roi_capture_timestamp=99.78)
    assert direct.detail == "bounded_roi_unsupported"


def test_sampling_expiration_does_not_update_accepted_anchor(chain, monkeypatch):
    _, _, depth, _, clock = chain
    frames(depth, 99.94)
    original = depth._torso_sampling_regions

    def delayed(*args, **kwargs):
        regions = original(*args, **kwargs)
        clock[0] = 100.13  # The selected physical frame itself crossed180ms.
        return regions

    monkeypatch.setattr(depth, "_torso_sampling_regions", delayed)
    result = measure(chain)
    assert result.raw_distance_m is None
    assert depth._last_accepted_ts == 0.


def test_visual_gate_only_enables_bounded_attempt_with_capable_sensor(owner, monkeypatch):
    monkeypatch.setattr(main, "ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC", .25)
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: None)
    owner._sensor_runtime = SimpleNamespace(supports_bounded_depth_roi=True)
    assert owner._depth_roi_age_allowed(NOW-.22, NOW)
    assert not owner._depth_roi_age_allowed(NOW-.251, NOW)
    owner._sensor_runtime.supports_bounded_depth_roi = False
    assert not owner._depth_roi_age_allowed(NOW-.22, NOW)


@pytest.mark.parametrize("cap,sample,now,valid", [
    (10., 10.18, 10.25, True), (10., 10.180001, 10.25, False),
    (10., 9.99, 10.20, False), (10., 10.01, 10.20, False),
    (10., 10.18, 10.250001, False), (10., 10.23, 10.22, False),
    (True, 1.18, 1.22, False), (10., float("nan"), 10.22, False),
])
def test_two_physical_time_bounds(cap, sample, now, valid):
    assert bounded_depth_sample_allowed(cap, sample, now) is valid


def test_short_config_does_not_enable_extra_roi_time():
    assert not bounded_roi_history_allowed(99.78, 100., .18)
