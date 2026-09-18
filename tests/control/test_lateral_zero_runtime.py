"""Offline regressions for real runtime yaw ownership and zero publication.

No PersonTracker constructor, camera, serial port or motor thread is started.
The queue spy replaces only dispatch, not the ownership/merging helpers.
"""
from __future__ import annotations

import sys
import queue
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction, ControlDecision, PersonTarget
from car_control_modular.lateral_intent import LateralControlIntent, LateralIntentStore


NOW = 100.0


@pytest.fixture
def owner(monkeypatch):
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW)
    for key in (
        "LATERAL_INTENT_CONTROL_ENABLE", "VISIBLE_STEERING_PID_ENABLE",
        "ASTRA_DEPTH_LONGITUDINAL_CONTROL_ENABLE", "MODULE_ASTRA_DEPTH_ENABLE",
        "VISION_DEPTH_ENABLED",
    ):
        monkeypatch.setattr(runtime, key, True)
    monkeypatch.setattr(runtime, "FOLLOW_ROTATION_ONLY", False)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC", 0.18)
    tracker = object.__new__(runtime.PersonTracker)
    tracker._control_update_lock = threading.RLock()
    tracker.command_lock = threading.Lock()
    tracker.action_queue_lock = threading.Lock()
    tracker._lateral_intent_store = LateralIntentStore()
    tracker._follow_controller = SimpleNamespace(
        active_target_id=1,
        last_steering_pid_result=SimpleNamespace(
            correction_rpm=5, correction_limit_rpm=10.0, base_rpm=0,
            target_rate_valid=True, target_image_rate_dps=20.0,
            desired_yaw_rate_dps=12.0, measured_yaw_rate_dps=0.0,
            output_floor_reason="test", feedback_used=False,
        ),
        _visible_motion_samples=[],
    )
    tracker.search_state = "none"
    tracker._vision_control_state = "target_visible_depth_valid"
    tracker._explicit_stop_requested = False
    tracker._runtime_shutdown_requested = False
    tracker.running = True
    tracker._brake_hold_active = False
    tracker._use_soft_stop_next = False
    tracker._depth30_linear_snapshot = None
    tracker.frame_index = 223
    tracker._active_capture_frame_id = 576
    tracker._last_control_decision_reason = "near_distance_rotation_only"
    tracker._last_command_capture_frame = 576
    tracker._last_command_capture_timestamp = NOW - 0.09
    tracker._last_vision_control_ts = NOW - 0.09
    tracker._last_vision_correction_rpm = 5
    tracker._last_vision_correction_at = NOW - 0.01
    tracker._last_vision_correction_target_id = 1
    tracker._current_rotate_raw_source = "lateral_intent_30hz"
    tracker._current_rotate_raw_target = 5
    tracker._current_forward_percent = 60
    tracker._current_steer_base_percent = 60
    tracker._current_steer_correction_rpm = 5
    tracker._lateral_intent_last_sequence = -1
    tracker._lateral_intent_zero_sequence = -1
    tracker._lateral_intent_last_correction_rpm = 5
    tracker._lateral_intent_last_target_id = 1
    tracker._lateral_intent_last_tick_ts = NOW - 0.03
    tracker._lateral_intent_owned_frame = -1
    tracker._lateral_intent_last_publish_ts = NOW - 1.0
    tracker._lateral_intent_last_log_direction = "hold"
    tracker._lateral_intent_last_log_ts = NOW
    tracker._last_dispatched_action = runtime.ACTION_STOP
    tracker._last_lateral_yaw_sign_mismatch_log_ts = 0.0
    tracker.current_command = runtime.ACTION_ROTATE_RIGHT
    tracker._queued_calls = []
    tracker._should_skip_redundant_action_queue = lambda actions, reason: False
    tracker._replace_action_queue = lambda actions, reason: tracker._queued_calls.append(
        (tuple(actions), reason)
    )
    return tracker


def _target(uid=1):
    return PersonTarget((390.0, 20.0, 550.0, 460.0), uid, 0.9, 70400.0)


def _intent(owner, **changes):
    intent = LateralControlIntent(
        sequence=0, target_id=1, frame_index=223, published_at=NOW - 0.01,
        valid_until=NOW + 0.14, x_ratio=0.73, motion_dx_ratio=0.03,
        target_image_rate_dps=20.0, mode="yaw_only", base_percent=0,
        base_rpm=0, initial_correction_rpm=5, correction_limit_rpm=10.0,
        confidence=0.9, bbox_quality="reliable", reason="near_distance_rotation_only",
        capture_frame_id=576, capture_timestamp=NOW - 0.09,
        decision_capture_frame_id=576, near_distance_mode=True,
    )
    return owner._lateral_intent_store.publish(replace(intent, **changes))


def _publish_stop(owner):
    return owner._publish_lateral_intent_from_decision(
        width=640, target=_target(), runtime_actions=[ControlAction.stop("center")],
        control_source="vision", target_steerable=True, low_quality_visible=False,
    )


def test_explicit_stop_keeps_zero_intent_and_immediately_queues_zero(owner):
    _intent(owner)
    assert _publish_stop(owner)
    current = owner._lateral_intent_store.snapshot()
    assert current is not None and current.hold_zero
    assert current.initial_correction_rpm == 0
    assert owner._queued_calls[-1][0] == (runtime.ACTION_STOP,)
    assert owner._current_rotate_raw_target == 0
    assert owner._current_steer_correction_rpm == 0
    assert not owner._visible_rotate_command_allowed(runtime.ACTION_ROTATE_RIGHT)


def test_low_quality_stop_needs_no_pid_result(owner):
    owner._follow_controller.last_steering_pid_result = None
    assert owner._publish_lateral_intent_from_decision(
        width=640, target=_target(), runtime_actions=[ControlAction.stop("quality_hold")],
        control_source="vision", target_steerable=False, low_quality_visible=True,
    )
    assert owner._lateral_intent_store.snapshot().hold_zero
    assert owner._queued_calls[-1][0] == (runtime.ACTION_STOP,)


@pytest.mark.parametrize("kind,expected", [
    ("forward", runtime.ACTION_STEER_RIGHT), ("backward", runtime.ACTION_BACKWARD),
])
def test_zero_preserves_fresh_depth_linear_speed_without_visual_acceleration(owner, kind, expected):
    owner._depth30_linear_snapshot = (kind, 7, 1, NOW - 0.05)
    _publish_stop(owner)
    assert owner._queued_calls[-1][0] == (expected,)
    assert owner._current_forward_percent == 7
    assert owner._current_steer_base_percent == (7 if kind == "forward" else 0)
    assert owner._current_steer_correction_rpm == 0
    assert owner._current_steer_inner_ratio_percent == 100
    assert owner._current_steer_outer_ratio_percent == 100


@pytest.mark.parametrize("snapshot,active_uid", [
    (("forward", 40, 1, NOW - 0.181), 1),
    (("forward", 40, 2, NOW - 0.05), 1),
    (("forward", 40, 1, NOW - 0.05), 2),
    (("forward", 40, 1, NOW + 0.01), 1),
])
def test_invalid_depth_authority_never_preserves_translation(owner, snapshot, active_uid):
    owner._depth30_linear_snapshot = snapshot
    owner._follow_controller.active_target_id = active_uid
    owner._publish_lateral_zero(_intent(owner), "test")
    assert owner._queued_calls[-1][0] == (runtime.ACTION_STOP,)


@pytest.mark.parametrize("field,value", [
    ("search_state", "searching"), ("search_state", "direction_unresolved"),
    ("_vision_control_state", "target_lost"), ("_explicit_stop_requested", True),
    ("_runtime_shutdown_requested", True), ("running", False),
])
def test_zero_cannot_preempt_search_or_safety_handoff(owner, field, value):
    previous = _intent(owner)
    setattr(owner, field, value)
    owner._clear_lateral_intent("handoff")
    assert not owner._publish_lateral_zero(previous, "expired")
    assert owner._queued_calls == []


def test_zero_does_not_release_existing_hard_brake_hold(owner):
    owner._brake_hold_active = True
    owner._depth30_linear_snapshot = ("forward", 40, 1, NOW - 0.05)
    _publish_stop(owner)
    assert owner._queued_calls[-1][0] == (runtime.ACTION_STOP,)
    assert owner._brake_hold_active
    assert not owner._use_soft_stop_next


def test_expiry_cannot_revoke_a_newer_sequence_published_before_control_lock(owner):
    expired = _intent(owner, valid_until=NOW - 0.001)
    replacement = []

    class PublishOnEntry:
        def __enter__(self):
            replacement.append(_intent(owner))

        def __exit__(self, *args):
            return False

    owner._control_update_lock = PublishOnEntry()
    owner._service_lateral_intent(NOW)
    assert replacement[0].sequence > expired.sequence
    assert owner._lateral_intent_store.snapshot() == replacement[0]
    assert owner._queued_calls == []


def test_expired_visible_yaw_is_revoked_without_waiting_motor_timeout(owner):
    _intent(owner, valid_until=NOW - 0.001)
    owner._service_lateral_intent(NOW)
    assert owner._lateral_intent_store.snapshot() is None
    assert owner._queued_calls[-1][0] == (runtime.ACTION_STOP,)
    assert not owner._visible_rotate_command_allowed(runtime.ACTION_ROTATE_RIGHT)


def test_intent_that_expires_while_waiting_for_control_lock_is_not_executed(owner, monkeypatch):
    _intent(owner)

    class AgeWhileWaiting:
        def __enter__(self):
            monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW + 0.30)

        def __exit__(self, *args):
            return False

    owner._control_update_lock = AgeWhileWaiting()
    # No action runtime exists: a stale PID refresh would fail this test.
    owner._service_lateral_intent(NOW)
    assert owner._lateral_intent_store.snapshot() is None
    assert owner._queued_calls[-1][0] == (runtime.ACTION_STOP,)


@pytest.mark.parametrize("explicit", [True, False])
def test_same_frame_zero_cannot_restart_pid_or_queued_turn(owner, explicit):
    current = _intent(owner, hold_zero=explicit)
    if explicit:
        owner._publish_lateral_zero(current, "explicit_stop")
    else:
        owner._publish_lateral_zero(current, "pid_zero")
    queue_before = list(owner._queued_calls)
    # Intentionally no _action_runtime. A restarted service would access it.
    owner._service_lateral_intent(NOW + 0.03)
    owner._service_lateral_intent(NOW + 0.06)
    assert owner._queued_calls == queue_before
    assert not owner._has_fresh_lateral_yaw(1)


def test_first_tick_pid_zero_bypasses_slew_and_waits_for_new_capture(owner, monkeypatch):
    # With a very slow brake slew the old +5 would otherwise survive this tick.
    monkeypatch.setattr(runtime, "LATERAL_INTENT_BRAKE_RPM_PER_SEC", 0.1)
    current = _intent(owner, initial_correction_rpm=0)
    owner._follow_controller.last_steering_pid_result.correction_rpm = 0
    feedback_calls = []
    owner._action_runtime = SimpleNamespace(
        get_steering_feedback=lambda: feedback_calls.append(True)
    )

    def forbidden_refresh(**kwargs):
        raise AssertionError("a zeroed capture must not refresh its PID")

    owner._follow_controller.refresh_parked_lateral_pid = forbidden_refresh
    owner._service_lateral_intent(NOW)
    assert owner._lateral_intent_last_correction_rpm == 0
    assert owner._lateral_intent_zero_sequence == current.sequence
    assert owner._queued_calls[-1][0] == (runtime.ACTION_STOP,)
    queued = len(owner._queued_calls)
    owner._service_lateral_intent(NOW + 0.02)
    assert len(owner._queued_calls) == queued
    assert len(feedback_calls) == 1

    next_intent = _intent(owner, capture_frame_id=578, decision_capture_frame_id=578)
    owner._follow_controller.last_steering_pid_result.correction_rpm = 5
    owner._service_lateral_intent(NOW + 0.04)
    assert next_intent.sequence > current.sequence
    assert owner._lateral_intent_last_correction_rpm > 0
    assert owner._queued_calls[-1][0] == (runtime.ACTION_ROTATE_RIGHT,)
    assert owner._visible_rotate_command_allowed(runtime.ACTION_ROTATE_RIGHT)


def test_non_depth_forward_zero_keeps_normal_translation(owner, monkeypatch):
    monkeypatch.setattr(runtime, "MODULE_ASTRA_DEPTH_ENABLE", False)
    _intent(owner, mode="forward", base_rpm=35, base_percent=35, initial_correction_rpm=0)
    result = owner._follow_controller.last_steering_pid_result
    result.base_rpm = 35
    result.correction_rpm = 0
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: None)
    owner._service_lateral_intent(NOW)
    assert owner._queued_calls[-1][0] == (runtime.ACTION_FORWARD,)
    assert owner._current_forward_percent == 35
    assert owner._lateral_intent_zero_sequence == -1


@pytest.mark.parametrize("invalid", ["expired", "hold_zero", "zero_sequence", "wrong_uid", "active_changed"])
def test_invalid_yaw_cannot_be_revived_by_depth_preserve_or_merge(owner, invalid):
    changes = {}
    if invalid == "expired":
        changes["valid_until"] = NOW - 0.01
    elif invalid == "hold_zero":
        changes["hold_zero"] = True
    elif invalid == "wrong_uid":
        changes["target_id"] = 2
    current = _intent(owner, **changes)
    if invalid == "zero_sequence":
        owner._lateral_intent_zero_sequence = current.sequence
    elif invalid == "active_changed":
        owner._follow_controller.active_target_id = 2
    zero = ControlAction.forward(0, "longitudinal_distance_untrusted_hold")
    assert not owner._has_fresh_lateral_yaw(1)
    assert not owner._depth30_should_preserve_lateral_motion(_target(), [zero], width=640)
    decision = ControlDecision(actions=[zero], reason=zero.reason)
    assert not owner._depth_zero_longitudinal_preserves_visible_rotation(decision)
    merged = owner._merge_longitudinal_action_with_visual_steering(
        ControlAction.forward(7, "depth_follow"), 1
    )
    assert merged.kind == "forward" and merged.steer_correction_rpm == 0


def test_valid_yaw_is_preserved_and_merged_for_current_uid(owner):
    _intent(owner)
    zero = ControlAction.forward(0, "longitudinal_distance_untrusted_hold")
    assert owner._has_fresh_lateral_yaw(1)
    assert owner._depth30_should_preserve_lateral_motion(_target(), [zero], width=640)
    merged = owner._merge_longitudinal_action_with_visual_steering(
        ControlAction.forward(7, "depth_follow"), 1
    )
    assert merged.kind == "steer_right"
    assert merged.speed_percent == 7 and merged.steer_correction_rpm == 5


def test_equal_wheel_depth_speed_change_is_not_deduplicated(owner, monkeypatch):
    monkeypatch.setattr(runtime, "MOTOR_STEER_RAW_TARGET", 15)
    owner._should_skip_redundant_action_queue = (
        runtime.PersonTracker._should_skip_redundant_action_queue.__get__(owner)
    )
    owner.action_queue = queue.Queue()
    owner._last_action_queue_signature = None
    owner._depth30_linear_snapshot = ("forward", 7, 1, NOW - 0.05)
    owner._publish_lateral_zero(_intent(owner), "first")
    owner._last_action_queue_signature = owner._actions_signature([runtime.ACTION_STEER_RIGHT])
    owner.current_command = runtime.ACTION_STEER_RIGHT
    owner.stop_action_execution = False
    owner.person_detected_flag = False
    owner._last_redundant_action_skip_log_ts = NOW
    owner._depth30_linear_snapshot = ("forward", 40, 1, NOW - 0.01)
    owner._publish_lateral_zero(owner._lateral_intent_store.snapshot(), "updated_depth")
    assert len(owner._queued_calls) == 2
    assert owner._queued_calls[-1][0] == (runtime.ACTION_STEER_RIGHT,)
    assert owner._current_steer_base_percent == 40


def test_search_rotation_bypasses_only_visible_owner_guard(owner):
    _intent(owner, hold_zero=True)
    owner.search_state = "searching"
    owner._current_rotate_raw_source = "search"
    assert owner._visible_rotate_command_allowed(runtime.ACTION_ROTATE_LEFT)
