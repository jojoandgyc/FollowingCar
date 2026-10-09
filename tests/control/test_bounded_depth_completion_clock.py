"""Selection-time ROI qualification survives safe processing-time delay."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

import request_0513_modular as main
from car_control_modular.control_types import ControlAction, ControlDecision
from car_control_modular.depth_roi_policy import BoundedDepthSelection, bounded_depth_sample_allowed
from test_depth_raw_geometry_runtime import person
from test_lateral_zero_runtime import owner, NOW
from test_longitudinal_authority_runtime import _frame
from test_turn_depth_history import chain, frames, measure


@pytest.mark.parametrize("cap,initial_roi_age,skew,processing", [
    (294, .1891, .1595, .1008),
    (330, .2249, .1775, .0525),
    (360, .2379, .1720, .0535),
    (362, .2166, .1642, .0721),
    (370, .2000, .1750, .1180),
])
def test_real_clock_delays_keep_fixed_fresh_sample_through_sensor_and_fusion(
        chain, monkeypatch, cap, initial_roi_age, skew, processing):
    runtime, _, depth, _, clock = chain
    capture = clock[0] - initial_roi_age
    stamp = capture + skew
    frames(depth, stamp)
    depth._latest_depth_ts = 100.  # Out of association; never use this background.
    original = depth._torso_sampling_regions

    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = 100. + processing
        return result

    monkeypatch.setattr(depth, "_torso_sampling_regions", delayed)
    state = measure(chain, capture, capture_frame_id=cap)
    assert clock[0]-capture > .25
    assert clock[0]-stamp < .18
    assert state.raw_distance_m == state.used_distance_m == 2.
    assert state.sample_timestamp == pytest.approx(stamp)
    assert state.sample_age_sec == pytest.approx(clock[0]-stamp)
    assert depth._last_accepted_ts == pytest.approx(stamp)
    assert runtime._vision_depth_fusion._last_accepted_depth_sample_ts == pytest.approx(stamp)
    selection = depth._measurement_bounded_selection
    assert isinstance(selection, BoundedDepthSelection)
    assert selection.selected_timestamp == 100.
    assert selection.capture_frame_id == cap and selection.target_id == 1
    # Completing work does not grant permission to start work again with the
    # same now-expired ROI, even if its old physical sample is still young.
    assert not bounded_depth_sample_allowed(capture, stamp, clock[0])
    repeated = measure(chain, capture, capture_frame_id=cap)
    assert repeated.raw_distance_m is None
    assert depth._last_accepted_ts == pytest.approx(stamp)


@pytest.mark.parametrize("processing", [.121, .18, .5])
def test_actual_physical_sample_expiration_never_updates_anchor(chain, monkeypatch, processing):
    runtime, _, depth, _, clock = chain
    frames(depth, 99.94)
    original = depth._torso_sampling_regions

    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = 100. + processing
        return result

    monkeypatch.setattr(depth, "_torso_sampling_regions", delayed)
    state = measure(chain)
    assert state.raw_distance_m is None
    assert depth._last_accepted_ts == 0.
    assert runtime._vision_depth_fusion._last_accepted_depth_sample_ts is None


@pytest.mark.parametrize("lock_name", ["_measurement_lock", "_depth_lock"])
def test_waiting_for_lock_does_not_start_a_now_expired_roi(chain, lock_name):
    _, _, depth, _, clock = chain
    frames(depth, 99.94)

    class DelayedLock:
        def __enter__(self):
            clock[0] = 100.04  # Capture age260ms only after upstream admission.

        def __exit__(self, *args):
            return False

    setattr(depth, lock_name, DelayedLock())
    state = measure(chain)
    assert state.raw_distance_m is None
    assert depth._last_accepted_ts == 0.
    assert len(depth._attempted_depth_samples) == 0


def test_frame_arriving_during_camera_lock_wait_is_not_false_future(chain):
    _, sensors, depth, _, clock = chain

    class ArrivalLock:
        def __enter__(self):
            clock[0] = 100.02
            frames(depth, 100.01)

        def __exit__(self, *args):
            return False

    depth._depth_lock = ArrivalLock()
    target = person(capture_timestamp=99.85)
    # Exercise the actual bounded-capable sensor API. The frame timestamp is
    # after function entry, but before its admitted physical selection time.
    measurement = sensors.get_astra_target_distance(
        target.depth_observation.bbox, 640, 480, target_id=1,
        evidence_capture_frame_id=1015, bounded_roi_capture_timestamp=99.85,
        use_latest_depth=True,
    )
    assert measurement.raw_distance_m == 2.
    assert measurement.temporal_status == "new_sample"
    assert measurement.sample_timestamp == pytest.approx(100.01)
    assert measurement.sample_age_sec == pytest.approx(.01)
    assert measurement.bounded_roi_selection.selected_timestamp == 100.02
    assert depth._last_accepted_ts == pytest.approx(100.01)


@pytest.mark.parametrize("selected_at", [None, "missing", "absent", float("nan"), True])
def test_missing_or_invalid_selected_clock_fails_closed_without_type_error(chain, monkeypatch, selected_at):
    _, _, depth, _, _ = chain
    frames(depth, 99.94)
    original = depth._aligned_depth_locked

    def missing_evidence(*args, **kwargs):
        result = original(*args, **kwargs)
        if selected_at == "absent":
            del depth._measurement_bounded_selected_at
        else:
            depth._measurement_bounded_selected_at = selected_at
        return result

    monkeypatch.setattr(depth, "_aligned_depth_locked", missing_evidence)
    state = measure(chain)
    assert state.raw_distance_m is None
    assert depth._last_accepted_ts == 0.
    assert depth._measurement_temporal_status == "bounded_selection_invalid"
    assert len(depth._attempted_depth_samples) == 0


@pytest.mark.parametrize("change", [
    {"target_id": 2}, {"target_id": True}, {"capture_frame_id": 9},
    {"capture_timestamp": 99.80}, {"sample_timestamp": 99.95},
    {"selected_timestamp": 100.04}, {"selected_timestamp": 99.93},
    {"bbox": (1., 1., 10., 10.)},
])
def test_changed_selection_binding_is_rejected_before_fusion(chain, monkeypatch, change):
    runtime, sensors, depth, _, _ = chain
    frames(depth, 99.94)
    original = sensors.get_astra_target_distance

    def tampered(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, bounded_roi_selection=replace(result.bounded_roi_selection, **change))

    monkeypatch.setattr(sensors, "get_astra_target_distance", tampered)
    state = measure(chain)
    assert state.source_detail == "bounded_roi_provenance_rejected"
    assert state.raw_distance_m is None
    assert runtime._vision_depth_fusion._last_accepted_depth_sample_ts is None


def test_missing_selection_proof_cannot_claim_completion_exception(chain, monkeypatch):
    runtime, sensors, depth, _, _ = chain
    frames(depth, 99.94)
    original = sensors.get_astra_target_distance
    monkeypatch.setattr(sensors, "get_astra_target_distance", lambda *args, **kwargs:
                        replace(original(*args, **kwargs), bounded_roi_selection=None))
    state = measure(chain)
    assert state.source_detail == "bounded_roi_provenance_rejected"
    assert runtime._vision_depth_fusion._last_accepted_depth_sample_ts is None


def test_physical_expiry_after_sensor_return_is_rechecked_before_fusion(chain, monkeypatch):
    runtime, sensors, depth, _, clock = chain
    frames(depth, 99.94)
    original = sensors.get_astra_target_distance

    def delayed_delivery(*args, **kwargs):
        measurement = original(*args, **kwargs)
        assert measurement.raw_distance_m == 2.
        clock[0] = 100.121
        return measurement

    monkeypatch.setattr(sensors, "get_astra_target_distance", delayed_delivery)
    state = measure(chain)
    assert state.raw_distance_m is None
    assert state.source_detail == "bounded_roi_provenance_rejected"
    assert runtime._vision_depth_fusion._last_accepted_depth_sample_ts is None


def test_exception_cannot_reuse_prior_selection_for_a_later_empty_attempt(chain, monkeypatch):
    _, _, depth, _, _ = chain
    frames(depth, 99.94)
    original = depth._torso_sampling_regions

    def fail(*args, **kwargs):
        raise ValueError("sampling failure after frame selection")

    monkeypatch.setattr(depth, "_torso_sampling_regions", fail)
    with pytest.raises(ValueError, match="sampling failure"):
        measure(chain)
    assert depth._measurement_bounded_selection is not None
    monkeypatch.setattr(depth, "_torso_sampling_regions", original)
    frames(depth)  # Latest is out of association and history is now empty.
    result = measure(chain)
    assert result.raw_distance_m is None
    assert depth._measurement_bounded_selection is None
    assert depth._last_accepted_ts == 0.


def test_completion_exception_does_not_extend_motor_deadline(chain, owner, monkeypatch):
    _, _, depth, fb, clock = chain
    frames(depth, 99.94)
    original = depth._torso_sampling_regions

    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = 100.06
        return result

    monkeypatch.setattr(depth, "_torso_sampling_regions", delayed)
    state = measure(chain)
    assert state.raw_distance_m == 2.
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: replace(fb, timestamp=100.05))
    monkeypatch.setattr(main, "ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC", .25)
    frame = replace(_frame(stamp=state.sample_timestamp, distance=2.),
                    distance_state=state, steering_feedback=replace(fb, timestamp=100.05))
    decision = ControlDecision(actions=[ControlAction.forward(10, "bounded_completion")],
                               reason="bounded_completion")
    actions, accepted = owner._commit_depth_linear_decision(decision, frame, 1, is_fresh_depth=True)
    assert accepted and actions[0].speed_percent > 0
    assert owner._depth30_linear_timing.depth_expires_at == pytest.approx(99.94+.25)
    assert owner._depth30_linear_snapshot[3] == pytest.approx(99.94)
    # The bounded selection grants no independent identity rights. Existing
    # final motion readers still reject it after a UID switch or hard stop.
    owner._follow_controller.active_target_id = 2
    assert owner._fresh_depth_linear_snapshot(1, now=clock[0]) is None
    owner._follow_controller.active_target_id = 1
    owner._explicit_stop_requested = True
    assert owner._fresh_depth_linear_snapshot(1, now=clock[0]) is None
