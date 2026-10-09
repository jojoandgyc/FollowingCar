"""Candidate-only brake holds suspend a finite search; they do not erase it."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from car_control_modular.control_types import ControlAction, PersonTarget, SensorFrame
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController


def feedback(now, angle):
    return SimpleNamespace(timestamp=now, integrated_yaw_right_deg=angle,
                           trustworthy=True, left_error=0, right_error=0)


def searching(monkeypatch, direction="left"):
    clock = [95.]
    monkeypatch.setattr("car_control_modular.controllers.time.monotonic", lambda: clock[0])
    c = FollowSafetyController(FollowPolicyConfig(direction_history_enable=True,
        lost_confirm_frames=3, lost_confirm_sec=0., search_timeout_sec=5.,
        search_revolution_deg=360., release_target_on_lost=False))
    c.active_target_id = 1
    c._has_seen_person = True
    c.search_state, c.search_direction = "searching", direction
    c._lost_started_at = 94.
    c._lost_exit_direction = direction
    c._lost_hint_source, c._lost_hint_confidence = "latest_reliable_capture_side", .92
    c._direction_loss_capture_id = 417
    c.lost_confirm_frames = 6
    c._begin_search_rotation_measurement(SensorFrame(width=640, height=480,
        steering_feedback=feedback(clock[0], 0.)))
    c._mark_search_rotation_started(clock[0])
    clock[0] = 100.
    sign = -1 if direction == "left" else 1
    for angle in (72., 144.):
        c._update_search_rotation_progress(SensorFrame(width=640, height=480,
            steering_feedback=feedback(clock[0], sign * angle)))
    x = .1 if direction == "left" else .9
    add_visible(c, 412, 99.9, x)
    c._last_visible_steer_action = ControlAction.rotate_right("must_not_replay")
    c.last_steering_pid_result = object()
    return c, clock, sign


def add_visible(c, cap, stamp, x):
    bbox = (x * 640 - 30, 10., x * 640 + 30, 470.)
    c._record_target_direction_evidence(SensorFrame(width=640, height=480,
        capture_frame_id=cap, capture_timestamp=stamp),
        PersonTarget(bbox, 1, .95, 27600.), reliable=True)


@pytest.mark.parametrize("direction", ["left", "right"])
def test_failed_candidate_resumes_same_scan_without_restoring_old_control(monkeypatch, direction):
    c, clock, sign = searching(monkeypatch, direction)
    # The pre-hold visible call can update the loss marker; capture the actual
    # original task state before a short-lived UID contaminates its hints.
    c._direction_loss_capture_id = 417
    ctx = c.capture_search_brake_resume()
    assert ctx is not None
    with pytest.raises(FrozenInstanceError):
        ctx.direction = "left" if direction == "right" else "right"
    add_visible(c, 589, 100.05, .52)
    c._lost_hint_source, c._lost_hint_confidence = "candidate_center", .1
    c._direction_loss_capture_id = None
    c.defer_search_timeout(.3)
    clock[0] = 101.
    assert c.retire_pre_search_brake_direction(100.1, resume_context=ctx,
        steering_feedback=feedback(clock[0], sign * 144.))
    assert (c.search_state, c.search_direction) == ("searching", direction)
    assert c.search_status(clock[0]).progress_deg == 144.
    assert c.search_status(clock[0]).elapsed_sec == 6.
    assert c._search_rotation_origin_integrated_yaw_deg == 0.
    assert c._search_rotation_started_at == 95.
    assert c._lost_started_at == 94.
    assert c._direction_loss_capture_id == 417
    assert c._lost_hint_source == "latest_reliable_capture_side"
    assert c._lost_hint_confidence == .92
    assert c._target_direction_history.latest_visible_evidence() is None
    assert c.last_person_center_x is None
    assert c.last_steering_pid_result is None and c._last_visible_steer_action is None
    assert not c._search_observation_hold
    clock[0] += .05
    decision = c.decide(606, SensorFrame(width=640, height=480,
        capture_frame_id=606, capture_timestamp=clock[0],
        steering_feedback=feedback(clock[0], sign * 144.)))
    assert decision.reason == "search_" + direction
    assert [action.kind for action in decision.actions] == ["rotate_" + direction]
    assert c.search_status(clock[0]).progress_deg == 144.
    assert c._search_rotation_started_at == 95. and c._lost_started_at == 94.


@pytest.mark.parametrize("bad", ["unlocked", "never_seen", "geometry_uid", "unknown",
                                 "direction_missing", "not_started", "infinite_unmeasured"])
def test_only_an_existing_target_owned_finite_scan_can_be_saved(monkeypatch, bad):
    c, _, _ = searching(monkeypatch)
    if bad == "unlocked": c.active_target_id = None
    elif bad == "never_seen": c._has_seen_person = False
    elif bad == "geometry_uid": c.active_target_id = -2
    elif bad == "unknown": c.search_state = "direction_unresolved"
    elif bad == "direction_missing": c.search_direction = None
    elif bad == "not_started": c._search_rotation_started_at = None
    else:
        from dataclasses import replace
        c.cfg = replace(c.cfg, search_timeout_sec=0.)
        c._search_rotation_feedback_seen = False
    assert c.capture_search_brake_resume() is None


@pytest.mark.parametrize("change", ["uid", "new_scan", "reset_same_start", "ended", "direction"])
def test_old_search_context_cannot_restore_another_or_finished_task(monkeypatch, change):
    c, clock, sign = searching(monkeypatch)
    ctx = c.capture_search_brake_resume()
    if change == "uid": c.active_target_id = 2
    elif change == "new_scan":
        c._begin_search_rotation_measurement(SensorFrame(width=640, height=480,
            steering_feedback=feedback(clock[0], 0.)))
    elif change == "reset_same_start":
        c._reset_search_timeout()
        c._search_rotation_started_at = ctx.rotation_started_at
    elif change == "ended": c.release_search_on_confirmed_target()
    else: c.search_direction = "right"
    clock[0] += 1.
    status = c.search_status(clock[0])
    history = c._target_direction_history.entries
    floor = c._target_direction_history.not_before_timestamp
    pid = c.last_steering_pid_result
    action = c._last_visible_steer_action
    task_token = c._search_resume_token
    assert not c.retire_pre_search_brake_direction(100.1, resume_context=ctx,
        steering_feedback=feedback(clock[0], sign * 144.))
    # Ignoring an old completion is a no-op, not a STOP/reset of a newer task.
    assert c.search_status(clock[0]) == status
    assert c._target_direction_history.entries == history
    assert c._target_direction_history.not_before_timestamp == floor
    assert c.last_steering_pid_result is pid and c._last_visible_steer_action is action
    assert c._search_resume_token is task_token


def test_stale_pause_cannot_erase_new_right_scan_or_its_observation(monkeypatch):
    c, clock, _ = searching(monkeypatch)
    old_context = c.capture_search_brake_resume()
    c.release_search_on_confirmed_target()
    clock[0] = 100.2
    add_visible(c, 606, 100.2, .85)
    c.search_state, c.search_direction = "searching", "right"
    c._lost_exit_direction = "right"
    c._lost_hint_source, c._lost_hint_confidence = "latest_reliable_capture_side", .92
    c._lost_started_at, c._direction_loss_capture_id = 100.3, 607
    c._begin_search_rotation_measurement(SensorFrame(width=640, height=480,
        steering_feedback=feedback(clock[0], -144.)))
    c._mark_search_rotation_started(100.3)
    clock[0] = 100.5
    c._update_search_rotation_progress(SensorFrame(width=640, height=480,
        steering_feedback=feedback(clock[0], -130.)))
    new_status = c.search_status(clock[0])
    assert not c.retire_pre_search_brake_direction(100.4, resume_context=old_context,
        steering_feedback=feedback(clock[0], -130.))
    assert c.search_status(clock[0]) == new_status
    assert c.search_direction == "right" and new_status.progress_deg == 14.
    assert c._lost_started_at == 100.3 and c._direction_loss_capture_id == 607
    assert c._target_direction_history.latest_visible_evidence().capture_frame_id == 606
    assert c._target_direction_history.latest_reliable_side().direction == "right"


def test_without_resume_context_existing_retirement_behavior_is_unchanged(monkeypatch):
    c, _, _ = searching(monkeypatch)
    assert not c.retire_pre_search_brake_direction(100.1)
    assert c.search_state == "none" and c.search_direction is None
    assert c._search_rotation_started_at is None
    assert c._target_direction_history.latest_visible_evidence() is None
    assert c.last_steering_pid_result is None and c._last_visible_steer_action is None


@pytest.mark.parametrize("change,expected", [
    ("same", True), ("timed_out", True), ("uid", False), ("token", False),
    ("direction", False), ("ended", False), ("not_started", False),
    ("never_seen", False), ("none", False), ("malformed", False),
])
def test_context_current_reader_is_pure_and_shares_retirement_ownership(monkeypatch, change, expected):
    c, clock, _ = searching(monkeypatch)
    ctx = c.capture_search_brake_resume()
    if change == "timed_out": c.search_state, c.search_direction = "timed_out", None
    elif change == "uid": c.active_target_id = 2
    elif change == "token": c._search_resume_token = object()
    elif change == "direction": c.search_direction = "right"
    elif change == "ended": c.search_state, c.search_direction = "none", None
    elif change == "not_started": c._search_rotation_started_at = None
    elif change == "never_seen": c._has_seen_person = False
    elif change == "none": ctx = None
    elif change == "malformed": ctx = {"uid": 1, "direction": "left"}
    state = dict(vars(c))
    history = c._target_direction_history.entries
    floor = c._target_direction_history.not_before_timestamp
    assert c.search_brake_resume_context_current(ctx) is expected
    assert c.search_brake_resume_context_current(ctx) is expected
    assert vars(c) == state
    assert c._target_direction_history.entries == history
    assert c._target_direction_history.not_before_timestamp == floor


def test_post_stop_trusted_target_precedes_original_left_scan(monkeypatch):
    c, clock, sign = searching(monkeypatch)
    ctx = c.capture_search_brake_resume()
    add_visible(c, 606, 100.6, .85)
    clock[0] = 101.
    assert not c.retire_pre_search_brake_direction(100.1, resume_context=ctx,
        steering_feedback=feedback(clock[0], sign * 144.))
    assert c.search_state == "none" and c.search_direction is None
    latest = c._target_direction_history.latest_reliable_side()
    assert latest.direction == "right" and latest.last_visible_capture_frame_id == 606
    assert c.last_steering_pid_result is None


def test_resume_counts_encoder_motion_during_hold_toward_original_scan(monkeypatch):
    c, clock, _ = searching(monkeypatch)
    ctx = c.capture_search_brake_resume()
    clock[0] = 101.
    assert c.retire_pre_search_brake_direction(100.1, resume_context=ctx,
        steering_feedback=feedback(clock[0], -149.))
    assert c.search_status(clock[0]).progress_deg == 149.
    assert c._search_rotation_started_at == 95.


@pytest.mark.parametrize("kind", ["missing", "stale", "future", "error", "untrustworthy"])
def test_expired_timeout_without_usable_feedback_remains_terminal(monkeypatch, kind):
    c, clock, _ = searching(monkeypatch)
    ctx = c.capture_search_brake_resume()
    clock[0] = 101.
    sample = feedback(clock[0], -144.)
    if kind == "missing": sample = None
    elif kind == "stale": sample.timestamp = 100.
    elif kind == "future": sample.timestamp = 102.
    elif kind == "error": sample.left_error = 1
    else: sample.trustworthy = False
    assert not c.retire_pre_search_brake_direction(100.1, resume_context=ctx,
        steering_feedback=sample)
    assert c.search_state == "timed_out" and c.search_direction is None
    assert c._search_rotation_started_at == 95.
    assert c._lost_started_at == 94.
    assert c.search_status(clock[0]).progress_deg == 144.


def test_completed_revolution_is_not_restarted_after_brake(monkeypatch):
    c, clock, _ = searching(monkeypatch)
    ctx = c.capture_search_brake_resume()
    for angle in (-220., -290., -355.):
        c._update_search_rotation_progress(SensorFrame(width=640, height=480,
            steering_feedback=feedback(clock[0], angle)))
    clock[0] = 101.
    assert not c.retire_pre_search_brake_direction(100.1, resume_context=ctx,
        steering_feedback=feedback(clock[0], -360.))
    assert c.search_state == "timed_out"
    assert c.search_status(clock[0]).progress_deg == 360.
    # Reusing the same hold context cannot reset a terminal scan into waiting.
    assert not c.retire_pre_search_brake_direction(100.1, resume_context=ctx)
    assert c.search_state == "timed_out" and c.search_status(clock[0]).progress_deg == 360.


def test_timeout_budget_is_not_renewed_when_old_context_is_accepted(monkeypatch):
    c, clock, _ = searching(monkeypatch)
    c._lost_started_at = 99.
    ctx = c.capture_search_brake_resume()
    clock[0] = 101.
    assert c.retire_pre_search_brake_direction(100.1, resume_context=ctx)
    assert c._lost_started_at == 99. and c._search_rotation_started_at == 95.
    clock[0] = 104.01
    decision = c._search_timeout_decision(clock[0])
    assert decision.reason == "search_timeout_stop"
    assert c.search_state == "timed_out"
