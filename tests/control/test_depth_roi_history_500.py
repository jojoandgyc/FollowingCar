"""500ms scheduling is not 500ms geometry or a 500ms motor grant."""
from dataclasses import replace

import pytest

import request_0513_modular as main
from car_control_modular.control_types import ControlAction, ControlDecision
from car_control_modular.depth_roi_policy import (
    BoundedDepthSelection, bounded_depth_sample_allowed,
    bounded_roi_history_allowed, roi_age_status,
)
from test_turn_depth_history import chain, frames, measure
from test_depth_raw_geometry_runtime import person
from test_lateral_zero_runtime import owner, NOW
from test_longitudinal_authority_runtime import _frame


@pytest.mark.parametrize("age,allowed", [(.250, True), (.319, True), (.499, True), (.501, False)])
def test_scheduling_age_ceiling_is_500_only_when_explicit(age, allowed):
    assert bounded_roi_history_allowed(100.-age, 100., .5) is allowed
    if age > .25:
        assert not bounded_roi_history_allowed(100.-age, 100., .25)
        assert not bounded_roi_history_allowed(100.-age, 100.)


@pytest.mark.parametrize("age", [.319, .499, .501])
def test_latest_frame_path_does_not_inherit_history_window(chain, age):
    _, _, _, feedback, _ = chain
    calm = replace(feedback, yaw_rate_right_dps=0., raw_yaw_rate_right_dps=0.)
    assert roi_age_status(100.-age, 100., .5, calm) == "expired"


@pytest.mark.parametrize("age", [.250, .319, .350])
def test_actual_ranging_at_delayed_roi_uses_associated_fresh_depth(chain, age):
    runtime, _, depth, _, _ = chain
    runtime.config = replace(runtime.config, vision_depth_detector_bbox_max_age_sec=.5)
    capture = 100.-age
    sample = capture+.175
    frames(depth, sample, capture+.19)
    state = measure(chain, capture, capture_frame_id=223)
    assert state.raw_distance_m == pytest.approx(2.)
    assert state.sample_timestamp == pytest.approx(sample)
    assert depth._measurement_bounded_selection.max_roi_age_sec == .5
    assert state.sample_age_sec == pytest.approx(age-.175)
    # The latest depth image contains the 5m background. It was not sampled.
    assert state.raw_distance_m != 5.


@pytest.mark.parametrize("age", [.499, .501])
def test_500_scheduling_cannot_find_a_fresh_associated_sample_at_499(chain, age):
    runtime, _, depth, _, _ = chain
    runtime.config = replace(runtime.config, vision_depth_detector_bbox_max_age_sec=.5)
    capture = 100.-age
    frames(depth, capture+.175)
    assert measure(chain, capture).raw_distance_m is None
    assert depth._last_accepted_ts == 0.
    assert not depth._attempted_depth_samples


@pytest.mark.parametrize("sample", [99.68, 99.81, 99.862, 99.99, 100.01])
def test_319_roi_rejects_before_capture_stale_future_and_turn_mismatched_depth(chain, sample):
    runtime, _, depth, _, _ = chain
    runtime.config = replace(runtime.config, vision_depth_detector_bbox_max_age_sec=.5)
    frames(depth, sample)
    # Capture99.681 + original association180ms =99.861. Latest99.99
    # is wrong during rotation even though that frame itself is fresh.
    assert measure(chain, 99.681).raw_distance_m is None
    assert depth._last_accepted_ts == 0.


@pytest.mark.parametrize("change", [dict(target_id=2), dict(source="predicted"), dict(raw_track_id=0)])
def test_319_window_never_bypasses_target_geometry_identity(chain, change):
    runtime, _, depth, _, _ = chain
    runtime.config = replace(runtime.config, vision_depth_detector_bbox_max_age_sec=.5)
    frames(depth, 99.851)
    assert measure(chain, 99.681, **change).raw_distance_m is None
    assert depth._last_accepted_ts == 0.


def test_configured_250_is_rechecked_inside_camera_lock(chain):
    _, sensors, depth, _, clock = chain
    frames(depth, 99.90)
    # Simulates acquiring the camera lock after a nominal250ms deadline.
    clock[0] = 100.001
    measurement = sensors.get_astra_target_distance(
        person().depth_observation.bbox, 640, 480, target_id=1,
        use_latest_depth=True, evidence_capture_frame_id=223,
        bounded_roi_capture_timestamp=99.75, bounded_roi_max_age_sec=.25)
    assert measurement.raw_distance_m is None
    assert depth._last_accepted_ts == 0.


@pytest.mark.parametrize("delay,accepted", [(.020, True), (.032, False)])
def test_prepared_319_task_rechecks_physical_freshness_at_commit(chain, delay, accepted):
    runtime, _, depth, feedback, clock = chain
    runtime.config = replace(runtime.config, vision_depth_detector_bbox_max_age_sec=.5)
    frames(depth, 99.851)
    target = person(capture_timestamp=99.681, capture_frame_id=223)
    prepared = runtime.prepare_depth_measurement(640, 480, target, steering_feedback=feedback)
    assert prepared is not None
    assert prepared.geometry_reason == "yolo_detector_bounded_history"
    assert prepared.transaction.kwargs["bounded_roi_max_age_sec"] == .5
    result = prepared.transaction.run()
    assert result.raw_distance_m == 2.
    assert depth._last_accepted_ts == 0.  # Uncommitted private work.
    clock[0] += delay
    assert runtime.commit_prepared_depth(prepared, target=target) is accepted
    assert depth._last_accepted_ts == (99.851 if accepted else 0.)


def test_selection_proof_cannot_override_consumers_shorter_window():
    bbox = (10., 10., 20., 20.)
    proof = BoundedDepthSelection(99.681, 99.851, 100., 1, 223, bbox, max_roi_age_sec=.5)
    params = dict(capture_timestamp=99.681, sample_timestamp=99.851, now=100.01,
                  target_id=1, capture_frame_id=223, bbox=bbox)
    assert proof.valid_for(**params, max_roi_age=.5)
    assert not proof.valid_for(**params, max_roi_age=.25)
    assert not proof.valid_for(**{**params, "target_id": 2}, max_roi_age=.5)
    assert not proof.valid_for(**{**params, "capture_frame_id": 224}, max_roi_age=.5)
    assert not replace(proof, max_roi_age_sec=float("nan")).valid_for(**params)


def test_unbounded_configuration_cannot_extend_physical_association():
    assert not bounded_roi_history_allowed(10., 10.501, 999.)
    assert not bounded_depth_sample_allowed(10., 10.180001, 10.319, max_roi_age=.5)
    assert not bounded_depth_sample_allowed(10., 10.18, 10.361, max_roi_age=.5)


def test_repeated_319_read_cannot_renew_sample_timestamp(chain):
    runtime, _, depth, _, clock = chain
    runtime.config = replace(runtime.config, vision_depth_detector_bbox_max_age_sec=.5)
    frames(depth, 99.851)
    first = measure(chain, 99.681)
    clock[0] += .01
    second = measure(chain, 99.681)
    assert first.sample_timestamp == 99.851
    assert second.raw_distance_m is None
    assert depth._last_accepted_ts == 99.851


def test_319_roi_uses_physical_sample_deadline_not_500ms_capture_deadline(chain, owner, monkeypatch):
    monkeypatch.setattr(main, "ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC", .5)
    monkeypatch.setattr(main, "ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC", .3)
    runtime, _, depth, feedback, _ = chain
    runtime.config = replace(runtime.config, vision_depth_detector_bbox_max_age_sec=.5)
    frames(depth, NOW-.149)
    state = measure(chain, NOW-.319, capture_frame_id=223)
    frame = replace(_frame(stamp=state.sample_timestamp, distance=2.),
                    distance_state=state, steering_feedback=feedback)
    decision = ControlDecision(actions=[ControlAction.forward(17, "bounded_history")],
                               reason="bounded_history")
    actions, accepted = owner._commit_depth_linear_decision(decision, frame, 1, is_fresh_depth=True)
    assert accepted and actions[0].speed_percent > 0
    assert owner._depth30_linear_timing.depth_expires_at == pytest.approx(NOW-.149+.3)
    assert owner._depth30_linear_timing.depth_expires_at < NOW-.319+.5
    assert owner._fresh_depth_linear_snapshot(1, now=NOW+.152) is None
