"""A rejected update cannot transiently replace a qualified physical lease."""
from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction, ControlDecision
from test_distance_only_runtime_authority import distance_only, publish
from test_distance_pi_runtime import pi_owner
from test_lateral_zero_runtime import owner


def next_sample(a, *, distance=3.5):
    a.stamp += .01
    a.shared = replace(a.shared, sample_timestamp=a.stamp, distance_m=distance)
    a.frame = replace(a.frame, distance_m=distance, distance_state=replace(
        a.frame.distance_state, raw_distance_m=distance, sample_timestamp=a.stamp))


@pytest.mark.parametrize("failure", ["missing_assessment", "new_distance_budget_missing"])
def test_rejected_update_keeps_independent_old_lease_and_deadline(distance_only, failure):
    a = distance_only
    publish(a)
    previous, timing = a.owner._depth30_linear_snapshot, a.owner._depth30_linear_timing
    revision = a.owner._lateral_yaw_revision
    next_sample(a)
    if failure == "new_distance_budget_missing":
        a.controller.distance_only_forward_percent = lambda *_: None
    result = publish(a, shared=None if failure == "missing_assessment" else a.shared)
    assert result == ([], False)
    assert a.owner._depth30_linear_snapshot is previous
    assert a.owner._depth30_linear_timing is timing
    assert a.owner._lateral_yaw_revision == revision
    assert a.owner._fresh_depth_linear_snapshot(1) == previous
    assert a.owner._depth30_linear_sample_watermark == (1, a.stamp)
    a.clock.now = timing.depth_expires_at + .000001
    assert a.owner._fresh_depth_linear_snapshot(1) is None


def test_failed_candidate_never_exposed_and_never_marks_old_veto(distance_only, monkeypatch):
    a = distance_only
    publish(a)
    previous, timing = a.owner._depth30_linear_snapshot, a.owner._depth30_linear_timing
    next_sample(a)
    candidate_stamp = a.stamp
    original = runtime.PersonTracker._depth_forward_continuation_limit
    observed = []

    def fail_new(self, linear, candidate_timing, now, **kwargs):
        observed.append(self._depth30_linear_snapshot)
        if linear[3] == candidate_stamp:
            return 0, "missing_shared_braking_evidence"
        return original(self, linear, candidate_timing, now, **kwargs)

    monkeypatch.setattr(runtime.PersonTracker, "_depth_forward_continuation_limit", fail_new)
    assert publish(a) == ([], False)
    assert observed and all(value is previous for value in observed)
    assert a.owner._depth30_linear_timing is timing
    assert getattr(a.owner, "_depth30_continuation_veto", None) is None
    assert getattr(a.owner, "_depth30_read_veto", None) is None


def test_success_previews_while_old_lease_is_visible_then_publishes_once(distance_only, monkeypatch):
    a = distance_only
    publish(a)
    previous = a.owner._depth30_linear_snapshot
    next_sample(a, distance=3.6)
    evaluate = runtime.PersonTracker._depth_forward_continuation_limit
    publication = runtime.PersonTracker._publish_depth_linear_pair
    checked, published = [], []

    def evaluate_with_reader(self, linear, timing, now, **kwargs):
        if linear[3] == a.stamp:
            checked.append(self._depth30_linear_snapshot)
        return evaluate(self, linear, timing, now, **kwargs)

    def record_publication(self, linear, timing):
        published.append(linear)
        return publication(self, linear, timing)

    monkeypatch.setattr(runtime.PersonTracker, "_depth_forward_continuation_limit", evaluate_with_reader)
    monkeypatch.setattr(runtime.PersonTracker, "_publish_depth_linear_pair", record_publication)
    actions, accepted = publish(a)
    assert accepted and actions[0].speed_percent > 0
    assert checked == [previous]
    assert published == [a.owner._depth30_linear_snapshot]
    assert published[0][3] == a.stamp
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(a.stamp + .25)


@pytest.mark.parametrize("fault", ["near", "closer", "zero", "reverse", "hazard", "obstacle",
                                  "identity", "explicit_stop", "brake", "old_expired"])
def test_adverse_evidence_never_uses_update_rejection_hold(distance_only, fault):
    a = distance_only
    publish(a)
    previous = a.owner._depth30_linear_snapshot
    next_sample(a, distance=1.4 if fault == "near" else 3.4 if fault == "closer" else 3.5)
    if fault == "hazard": a.frame = replace(a.frame, hazard=replace(a.frame.hazard, active=True))
    if fault == "obstacle": a.frame = replace(a.frame, obstacles=replace(a.frame.obstacles, front=True))
    if fault == "identity": a.controller.active_target_id = 2
    if fault == "explicit_stop": a.owner._explicit_stop_requested = True
    if fault == "brake": a.frame = replace(a.frame, distance_state=replace(a.frame.distance_state, brake_latched=True))
    if fault == "old_expired":
        a.clock.now = previous[3] + .251
        a.stamp = a.clock.now - .01
        a.frame = replace(a.frame, distance_state=replace(a.frame.distance_state, sample_timestamp=a.stamp))
        a.feedback[0] = replace(a.feedback[0], timestamp=a.clock.now)
        a.owner._last_vision_control_ts = a.clock.now
    if fault == "reverse":
        decision = ControlDecision(actions=[ControlAction.backward(20, "reverse")])
        assert not a.owner._retain_depth_after_update_rejection(previous, decision, a.frame, 1, "test")
        return
    actions, _ = publish(a, shared=None, percent=0 if fault == "zero" else 40)
    assert all(action.speed_percent == 0 for action in actions)
    assert a.owner._depth30_linear_snapshot is None


def test_preview_itself_does_not_mutate_live_authority_or_diagnostics(distance_only):
    a = distance_only
    publish(a)
    previous, timing = a.owner._depth30_linear_snapshot, a.owner._depth30_linear_timing
    candidate = ("forward", 40, 1, a.stamp + .01)
    wrong = replace(timing, snapshot=candidate)  # assessment belongs to OLD sample
    before = dict(a.owner.__dict__)
    assert a.owner._fresh_depth_linear_snapshot(
        1, quiet=True, _candidate=candidate, _candidate_timing=wrong) is None
    assert a.owner.__dict__ == before
    assert a.owner._depth30_linear_snapshot is previous


@pytest.mark.parametrize("fault", ["reverse", "overspeed", "invalid", "untrustworthy", "drive_error"])
def test_early_rejected_budget_cannot_hide_new_adverse_wheel_evidence(distance_only, fault):
    a = distance_only
    publish(a)
    next_sample(a)
    # Runtime cache is still a qualified positive report. The newly prepared
    # observation independently carries a contrary report; reject-before-
    # assessment must inspect it too, not just re-read the old cache.
    changes = {
        "reverse": {"left_forward_rpm": -5.},
        "overspeed": {"left_forward_rpm": 210.},
        "invalid": {"left_forward_rpm": float("nan")},
        "untrustworthy": {"trustworthy": False},
        "drive_error": {"left_error": 1},
    }[fault]
    a.frame = replace(a.frame, steering_feedback=replace(a.frame.steering_feedback, **changes))
    a.controller.distance_only_forward_percent = lambda *_: None
    actions, _ = publish(a)
    assert all(action.speed_percent == 0 for action in actions)
    assert a.owner._depth30_linear_snapshot is None


def test_expiry_during_preview_cannot_publish_candidate(distance_only, monkeypatch):
    a = distance_only
    publish(a)
    previous = a.owner._depth30_linear_snapshot
    next_sample(a)
    reader = a.owner._action_runtime.get_steering_feedback
    seen = []

    def delayed_feedback():
        seen.append(a.owner._depth30_linear_snapshot)
        a.clock.now += .30
        return reader()

    monkeypatch.setattr(a.owner._action_runtime, "get_steering_feedback", delayed_feedback)
    actions, _ = publish(a)
    assert seen[0] is previous
    assert all(action.speed_percent == 0 for action in actions)
    assert a.owner._depth30_linear_snapshot is None


@pytest.mark.parametrize("proof_pending", [False, True])
def test_feedback_and_braking_preview_do_not_hold_motor_lock(distance_only, monkeypatch, proof_pending):
    a = distance_only
    publish(a)
    next_sample(a)
    a.owner.motor_io_lock = threading.Lock()
    if proof_pending:
        a.controller._distance_pi_execution_anchor_proof = (a.stamp,)
        a.controller.longitudinal_execution_proof_valid = lambda *_: True
    feedback = a.owner._action_runtime.get_steering_feedback
    reads = []

    def checked_feedback():
        assert not a.owner.motor_io_lock.locked()
        reads.append(True)
        return feedback()

    monkeypatch.setattr(a.owner._action_runtime, "get_steering_feedback", checked_feedback)
    actions, accepted = publish(a)
    assert accepted and actions[0].speed_percent > 0
    assert reads


@pytest.mark.parametrize("change", ["explicit_stop", "uid", "driver_stop"])
def test_stop_or_identity_change_after_preview_blocks_publication(distance_only, monkeypatch, change):
    a = distance_only
    publish(a)
    previous = a.owner._depth30_linear_snapshot
    next_sample(a)
    a.owner.motor_io_lock = threading.Lock()
    a.owner._action_runtime.backend = SimpleNamespace(stop_write_generation=0)
    original = runtime.PersonTracker._fresh_depth_linear_snapshot
    seen = []

    def change_after_preview(self, target_id, **kwargs):
        result = original(self, target_id, **kwargs)
        if kwargs.get("_candidate") is not None:
            assert result is not None
            seen.append(self._depth30_linear_snapshot)
            if change == "explicit_stop": self._explicit_stop_requested = True
            elif change == "uid": self._follow_controller.active_target_id = 2
            else: self._action_runtime.backend.stop_write_generation += 1
        return result

    monkeypatch.setattr(runtime.PersonTracker, "_fresh_depth_linear_snapshot", change_after_preview)
    actions, _ = publish(a)
    assert seen == [previous]
    assert all(action.speed_percent == 0 for action in actions)
    assert a.owner._depth30_linear_snapshot is None


def test_real_depth_process_does_not_clear_retained_lease_after_empty_commit(distance_only):
    from test_longitudinal_authority_runtime import _process_fixture
    a = distance_only
    publish(a)
    previous, timing = a.owner._depth30_linear_snapshot, a.owner._depth30_linear_timing
    action_runtime = a.owner._action_runtime
    next_sample(a)
    decision = ControlDecision(actions=[ControlAction.forward(40, "distance_pi")], reason="distance_pi")
    target = _process_fixture(a.owner, decision, lambda *_args, **_kwargs: decision)
    a.owner._action_runtime = action_runtime
    a.owner._distance_runtime.get_frame_distance_state = lambda *_args, **_kwargs: a.frame.distance_state
    a.controller._distance_pid_last_sample_timestamp = a.stamp
    a.controller.last_distance_pid_result = SimpleNamespace(
        approach_mode="distance_pi", output_rpm=80., pi_braking_assessment=None)
    actions = a.owner._process_detections_modular(
        640, 480, [(target.bbox, target.track_id, target.confidence, target.area)],
        control_source="depth30")
    assert actions == []
    assert a.owner._depth30_linear_snapshot is previous
    assert a.owner._depth30_linear_timing is timing
    assert not a.owner._queued_calls
