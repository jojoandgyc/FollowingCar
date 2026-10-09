"""CAP589 -> CAP606: rejected reacquisition must not erase a finite search.

Connect production PersonTracker, follow controller, search-brake runtime and
fake motor driver. No detector inference, serial port or hardware thread starts.
"""
import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import request_0513_modular as application

from car_control_modular.action_command import ActionCommandSnapshot
from car_control_modular.control_types import PersonTarget, SensorFrame, SteeringFeedback
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController

# The repository's hardware-free motor helpers are plain pytest modules.
sys.path.append(str(Path(__file__).resolve().parents[1] / "motor"))
from test_follow_wheel_periodic import setup_periodic


class NormalControlEntry(Exception):
    """Stop the harness after real hold release, before unrelated perception."""


def setup_search(monkeypatch, direction):
    rt, motor_owner, driver, symbols, clock, axes = setup_periodic(monkeypatch)
    clock[0] = 100.
    axes[:] = [0., 0., 0., 0.]
    tracker = object.__new__(application.PersonTracker)
    tracker.__dict__.update(vars(motor_owner))
    tracker._action_runtime = rt
    rt.owner = tracker
    tracker.command_lock = threading.Lock()
    tracker.action_queue = queue.Queue()
    tracker.action_queue_lock = threading.Lock()
    tracker.frame_index = 250
    tracker._rknn_pipeline = SimpleNamespace(tracker=SimpleNamespace(last_identity_observations=[]))
    tracker._longitudinal_valid_until = 99.8
    tracker._current_forward_percent = 0
    tracker._follow_controller = FollowSafetyController(FollowPolicyConfig(
        direction_history_enable=True, lost_confirm_frames=3, lost_confirm_sec=0.,
        release_target_on_lost=False,  # Production locked-UID search policy.
        initial_target_confirm_frames=1, search_timeout_sec=60., search_revolution_deg=360.))
    ctl = tracker._follow_controller
    ctl.active_target_id = 1
    ctl._has_seen_person = True
    ctl.search_state, ctl.search_direction = "searching", direction
    ctl._lost_exit_direction = direction
    ctl._lost_started_at = 97.
    ctl.lost_confirm_frames = ctl.cfg.lost_confirm_frames
    ctl._lost_hint_source = "latest_reliable_capture_side"
    ctl._lost_hint_confidence = 1.
    tracker.search_state, tracker.search_direction = "searching", direction
    target = PersonTarget((15, 20, 120, 450) if direction == "left" else (520, 20, 625, 450),
                          1, .95, 45150)
    ctl._record_target_direction_evidence(SensorFrame(
        width=640, height=480, capture_frame_id=580, capture_timestamp=99.7), target, reliable=True)
    ctl._direction_loss_capture_id = 581
    ctl._begin_search_rotation_measurement(SensorFrame(
        width=640, height=480, steering_feedback=SteeringFeedback(
            timestamp=98., integrated_yaw_right_deg=0., trustworthy=True)))
    ctl._mark_search_rotation_started(98.)
    sign = -1 if direction == "left" else 1
    for degrees in (72., 144.):
        ctl._update_search_rotation_progress(SensorFrame(
            width=640, height=480, steering_feedback=SteeringFeedback(
                timestamp=99.8, integrated_yaw_right_deg=degrees * sign, trustworthy=True)))
    heading = 144. * sign
    rt.config.rotate_pulse_settle_feedback_stale_sec = .30
    rt.config.rotate_pulse_settle_max_wheel_rpm = 5
    rt.config.rotate_pulse_settle_max_yaw_rate_dps = 10.
    rt.get_steering_feedback = lambda: SteeringFeedback(
        timestamp=clock[0], trustworthy=True, integrated_yaw_right_deg=heading)

    def stop_at_normal_entry(**kwargs):
        assert kwargs.get("reason") == "visual_frame_begin"
        raise NormalControlEntry

    tracker._clear_longitudinal_context = stop_at_normal_entry
    return tracker, rt, driver, symbols, clock, heading


def observation(tracker, clock, cap, uid, x, *, stamp=None):
    stamp = clock[0] - .04 if stamp is None else stamp
    tracker._active_capture_frame_id, tracker._active_capture_timestamp = cap, stamp
    bbox = (x * 640 - 55., 20., x * 640 + 55., 450.)
    rec = SimpleNamespace(track_id=4, reid_uid=uid, class_id=0, time_since_update=0,
                          score=.95, x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
    tracker._rknn_pipeline.tracker.last_identity_observations = [dict(
        raw_track_id=4, uid=uid, detector_bbox=bbox,
        sample_metadata=dict(capture_frame_id=cap, capture_timestamp=stamp, is_fresh=True),
        assignment=dict(best_uid=1, bbox_quality_ok=True, reacquire_geometry_ok=True,
                        distance=.20, reason="skip_update_reacquire_quarantine"))]
    return rec


def request_and_stop(tracker, rt, clock, *, candidate_x):
    record = observation(tracker, clock, 589, 1, candidate_x)
    assert rt.request_search_reacquire_brake(589, tracker._active_capture_timestamp,
                                           "candidate_center")
    request = rt._search_reacquire_brake_request
    tracker._consume_track_records([record], 640, 480, "test")
    assert tracker._search_brake_latest_observation == (1, 589)
    rt._service_follow_wheels()
    assert rt._search_reacquire_brake_applied is request
    assert tracker._active_capture_timestamp < rt._search_reacquire_brake_sent_at
    return request


def settle(rt, clock):
    # Release normal parking current, then obtain two distinct post-release
    # quiet samples. Repeated cached feedback never counts as another sample.
    for now in (100. + tick * .05 for tick in range(1, 20)):
        clock[0] = now
        rt._service_follow_wheels()


def release_with_capture(tracker, rt, clock, *, uid=0, x=.5):
    clock[0] = 100.99
    record = observation(tracker, clock, 606, uid, x)
    with pytest.raises(NormalControlEntry):
        tracker._consume_track_records([record], 640, 480, "test")
    assert rt._search_reacquire_brake_request is None


def next_missing_decision(tracker, clock, heading, cap=611):
    clock[0] += .06
    return tracker._follow_controller.decide(cap, SensorFrame(
        width=640, height=480, capture_frame_id=cap, capture_timestamp=clock[0] - .01,
        steering_feedback=SteeringFeedback(timestamp=clock[0], trustworthy=True,
                                          integrated_yaw_right_deg=heading)))


@pytest.mark.parametrize("direction", ["left", "right"])
def test_failed_candidate_preserves_original_search_and_144_degree_budget(monkeypatch, direction):
    tracker, rt, driver, symbols, clock, heading = setup_search(monkeypatch, direction)
    ctl = tracker._follow_controller
    opposite_x = .8 if direction == "left" else .2
    request = request_and_stop(tracker, rt, clock, candidate_x=opposite_x)
    assert rt._search_reacquire_resume_context is not None
    assert rt._search_reacquire_resume_request is request
    # A misleading UID1 frame is observable, not permission to replace the
    # interrupted search direction or execute an old wheel packet.
    assert ctl._target_direction_history.latest_reliable_side().direction != direction
    assert ctl.search_direction == direction
    stale = ActionCommandSnapshot(action=symbols.forward, revision=1,
        enqueued_at=clock[0], capture_frame_id=589, capture_timestamp=clock[0] - .04,
        control_frame=250, soft_stop=False, reason="old_candidate_forward")
    tracker.action_queue.put(stale)
    settle(rt, clock)
    release_with_capture(tracker, rt, clock)
    assert tracker.search_state == ctl.search_state == "searching"
    assert tracker.search_direction == ctl.search_direction == direction
    assert ctl._target_direction_history.latest_visible_evidence() is None
    assert ctl.last_person_center_x is None
    assert ctl.search_status(clock[0]).progress_deg == pytest.approx(144.)
    assert ctl._lost_started_at == 97. and ctl._search_rotation_started_at == 98.
    assert tracker.action_queue.empty()
    assert rt._search_reacquire_resume_context is None
    assert rt._search_reacquire_resume_request is None
    decision = next_missing_decision(tracker, clock, heading)
    assert any(a.kind == "rotate_" + direction for a in decision.actions)
    assert not any(a.kind in ("forward", "steer_left", "steer_right", "backward")
                   for a in decision.actions)
    assert ctl.search_status(clock[0]).progress_deg == pytest.approx(144.)
    assert tracker._current_forward_percent == 0 and tracker._longitudinal_valid_until == 99.8
    assert driver.pairs == [(0, 0)] and driver.stops == [1, 0, 2]


def test_successful_post_stop_right_observation_overrides_old_left_search(monkeypatch):
    tracker, rt, driver, _, clock, heading = setup_search(monkeypatch, "left")
    request_and_stop(tracker, rt, clock, candidate_x=.3)
    settle(rt, clock)
    release_with_capture(tracker, rt, clock, uid=1, x=.8)
    ctl = tracker._follow_controller
    side = ctl._target_direction_history.latest_reliable_side()
    assert side.direction == "right" and side.last_visible_capture_frame_id == 606
    assert ctl.search_state == "none" and ctl.search_direction is None
    for cap in (611, 613, 615):
        decision = next_missing_decision(tracker, clock, heading, cap)
        assert not any(a.kind in ("rotate_left", "forward") for a in decision.actions)
    assert ctl.search_direction == "right"
    assert any(a.kind == "rotate_right" for a in decision.actions)
    assert all(pair == (0, 0) for pair in driver.pairs)


def test_no_resume_context_retains_old_pre_stop_direction_retirement(monkeypatch):
    tracker, rt, _, _, clock, heading = setup_search(monkeypatch, "left")
    tracker._follow_controller.capture_search_brake_resume = lambda: None
    request_and_stop(tracker, rt, clock, candidate_x=.3)
    settle(rt, clock)
    release_with_capture(tracker, rt, clock)
    ctl = tracker._follow_controller
    assert ctl.search_state == "none" and ctl.search_direction is None
    assert ctl._target_direction_history.latest_visible_evidence() is None
    for cap in (611, 613, 615):
        decision = next_missing_decision(tracker, clock, heading, cap)
        assert not any(a.kind in ("rotate_left", "rotate_right", "forward") for a in decision.actions)


@pytest.mark.parametrize("invalid", ["request_mismatch", "uid_changed", "explicit_stop", "safety_hold"])
def test_old_resume_context_cannot_survive_replacement_or_higher_priority_stop(monkeypatch, invalid):
    tracker, rt, driver, _, clock, _ = setup_search(monkeypatch, "left")
    request_and_stop(tracker, rt, clock, candidate_x=.3)
    settle(rt, clock)
    # Complete the actual motor release, then invalidate before PersonTracker
    # consumes the one-shot resume token. No direct private token injection.
    clock[0] = 100.99
    assert not rt.search_reacquire_brake_pending(capture_timestamp=clock[0] - .01)
    if invalid == "request_mismatch":
        rt._search_reacquire_resume_request = object()
    elif invalid == "uid_changed":
        tracker._follow_controller.active_target_id = 2
    elif invalid == "explicit_stop":
        tracker._explicit_stop_requested = True
    elif invalid == "safety_hold":
        tracker._brake_hold_label = "safety_hold_hard_stop"
    prior = (tracker._follow_controller.search_state, tracker._follow_controller.search_direction,
             tracker._follow_controller.search_status(clock[0]).progress_deg)
    tracker._finish_search_brake_observation_hold()
    # An invalid receipt or higher-priority stop cannot resume motion, nor may
    # it masquerade as "no saved context" and erase a newer controller task.
    assert (tracker._follow_controller.search_state, tracker._follow_controller.search_direction,
            tracker._follow_controller.search_status(clock[0]).progress_deg) == prior
    assert rt._search_reacquire_resume_context is None
    assert rt._search_reacquire_resume_request is None
    assert tracker._brake_hold_active
    assert all(pair == (0, 0) for pair in driver.pairs)


@pytest.mark.parametrize("receipt", ["valid", "missing"])
def test_old_left_completion_cannot_clear_new_right_scan_or_its_budget(monkeypatch, receipt):
    tracker, rt, driver, _, clock, heading = setup_search(monkeypatch, "left")
    request_and_stop(tracker, rt, clock, candidate_x=.8)
    settle(rt, clock)
    clock[0] = 100.99
    assert not rt.search_reacquire_brake_pending(capture_timestamp=clock[0] - .01)
    ctl = tracker._follow_controller
    old_context = rt._search_reacquire_resume_context
    # A different finite scan begins before the old vision-side completion
    # runs. Keep the old observed CAP589 on purpose: a fallback call to
    # resume_direction_after_brake_hold would otherwise erase this new scan.
    assert tracker._search_brake_latest_observation == (1, 589)
    assert ctl._direction_latest_visible_capture_id == 589
    ctl._begin_search_rotation_measurement(SensorFrame(
        width=640, height=480, steering_feedback=SteeringFeedback(
            timestamp=clock[0], integrated_yaw_right_deg=heading, trustworthy=True)))
    ctl.search_state, ctl.search_direction = "searching", "right"
    ctl._mark_search_rotation_started(clock[0])
    ctl._lost_started_at = 100.98
    ctl._direction_loss_capture_id = 607
    ctl._update_search_rotation_progress(SensorFrame(
        width=640, height=480, steering_feedback=SteeringFeedback(
            timestamp=clock[0], integrated_yaw_right_deg=heading + 72., trustworthy=True)))
    tracker.search_state, tracker.search_direction = "searching", "right"
    assert not ctl.search_brake_resume_context_current(old_context)
    if receipt == "missing":
        rt._search_reacquire_brake_applied = None
    prior = (ctl.search_state, ctl.search_direction, ctl._search_rotation_started_at,
             ctl._lost_started_at, ctl._direction_loss_capture_id,
             ctl.search_status(clock[0]).progress_deg,
             tuple(ctl._target_direction_history.entries))
    tracker._finish_search_brake_observation_hold()
    assert (ctl.search_state, ctl.search_direction, ctl._search_rotation_started_at,
            ctl._lost_started_at, ctl._direction_loss_capture_id,
            ctl.search_status(clock[0]).progress_deg,
            tuple(ctl._target_direction_history.entries)) == prior
    assert ctl.search_status(clock[0]).progress_deg == pytest.approx(72.)
    assert tracker.search_state == "searching" and tracker.search_direction == "right"
    assert rt._search_reacquire_resume_context is None
    assert rt._search_reacquire_resume_request is None
    assert tracker._brake_hold_active and all(pair == (0, 0) for pair in driver.pairs)


def test_hold_finish_is_one_shot_and_does_not_replace_new_search(monkeypatch):
    tracker, rt, _, _, clock, _ = setup_search(monkeypatch, "left")
    request_and_stop(tracker, rt, clock, candidate_x=.8)
    settle(rt, clock)
    release_with_capture(tracker, rt, clock)
    ctl = tracker._follow_controller
    assert ctl.search_direction == "left"
    ctl.search_direction = "right"
    tracker._finish_search_brake_observation_hold()
    assert ctl.search_direction == "right"


def test_finishing_previous_hold_cannot_consume_a_new_brake_request(monkeypatch):
    tracker, rt, _, _, clock, _ = setup_search(monkeypatch, "left")
    old = request_and_stop(tracker, rt, clock, candidate_x=.8)
    settle(rt, clock)
    clock[0] = 100.99
    assert not rt.search_reacquire_brake_pending(capture_timestamp=clock[0] - .01)
    assert rt.request_search_reacquire_brake(607, clock[0] - .01, "next_candidate")
    new = rt._search_reacquire_brake_request
    context = rt._search_reacquire_resume_context
    assert new is not old and context is not None
    tracker._finish_search_brake_observation_hold()
    assert rt._search_reacquire_brake_request is new
    assert rt._search_reacquire_resume_request is new
    assert rt._search_reacquire_resume_context is context
    assert rt.search_reacquire_brake_pending()
    assert tracker._search_brake_observation_active


def test_repeated_candidate_request_cannot_replace_original_resume_context(monkeypatch):
    tracker, rt, driver, _, clock, _ = setup_search(monkeypatch, "left")
    request = request_and_stop(tracker, rt, clock, candidate_x=.8)
    context = rt._search_reacquire_resume_context
    clock[0] += .08
    assert rt.request_search_reacquire_brake(591, clock[0] - .01, "new_candidate")
    assert rt._search_reacquire_brake_request is request
    assert rt._search_reacquire_resume_request is request
    assert rt._search_reacquire_resume_context is context
    tracker._finish_search_brake_observation_hold()
    assert tracker._search_brake_observation_active
    assert rt._search_reacquire_resume_context is context
    assert tracker._follow_controller.search_direction == "left"
    assert driver.pairs == [(0, 0)] and driver.stops == [1, 0]


@pytest.mark.parametrize("direction", ["left", "right"])
def test_resumed_search_new_decision_releases_hold_and_reaches_motor_writer(monkeypatch, direction):
    tracker, rt, driver, symbols, clock, heading = setup_search(monkeypatch, direction)
    request_and_stop(tracker, rt, clock, candidate_x=.8 if direction == "left" else .2)
    # Simulate stale work queued while NORMAL was settling. Actual runtime
    # release must discard this packet before adopting the NEW search result.
    tracker.action_queue.put(ActionCommandSnapshot(
        action=symbols.forward, revision=1, enqueued_at=clock[0], control_frame=250,
        capture_frame_id=589, capture_timestamp=clock[0] - .04,
        reason="old_forward_during_hold", soft_stop=False))
    settle(rt, clock)
    release_with_capture(tracker, rt, clock)
    assert tracker.action_queue.empty()
    decision = next_missing_decision(tracker, clock, heading)
    assert [a.kind for a in decision.actions] == ["rotate_" + direction]
    assert tracker._brake_hold_active  # Only the motor adoption may release it.

    tracker.search_state = tracker._follow_controller.search_state
    tracker.search_direction = tracker._follow_controller.search_direction
    tracker._last_control_decision_reason = decision.reason
    tracker._current_forward_percent = decision.current_forward_percent
    tracker.stop_action_execution = decision.stop_action_execution
    tracker.person_detected_flag = decision.person_detected_flag
    tracker._brake_hold_refresh_interval_sec = .5
    tracker._rotate_follows_previous_rotate = False
    tracker._last_rotate_end_ts = 0.
    tracker._last_hard_stop_check_ts = 0.
    tracker._hard_stop_check_interval_sec = .05
    class BoundedEvent(threading.Event):
        checks = 0
        def is_set(self):
            self.checks += 1
            if self.checks > 20:
                self.set()
            return super().is_set()
    tracker.action_stop_event = BoundedEvent()
    rt.config.enable_motor_rpm_feedback = False
    rt.config.action_intent_stale_sec = .45
    rt.config.rotate_chain_memory_sec = .3
    action = symbols.rotate_left if direction == "left" else symbols.rotate_right
    tracker.action_queue.put(ActionCommandSnapshot(
        action=action, revision=tracker._action_command_revision,
        enqueued_at=clock[0], control_frame=251, capture_frame_id=611,
        capture_timestamp=clock[0] - .01, reason=decision.reason, soft_stop=False))

    # Run the production command adoption/release/dispatch loop once. End on
    # the first real fake-driver motion write, not by clearing brake_hold in
    # the test. A bounded attempt counter makes regressions fail, not hang.
    writes = driver.set_left_speed
    def finish_on_motion(value):
        writes(value)
        if value:
            tracker.action_stop_event.set()
    monkeypatch.setattr(driver, "set_left_speed", finish_on_motion)
    attempts = [0]
    def bounded_safety(_action):
        attempts[0] += 1
        if attempts[0] > 25:
            tracker.action_stop_event.set()
        return False
    rt.hard_stop_check = bounded_safety
    rt.run_loop()

    assert not tracker._brake_hold_active
    nonzero = [(left, -right) for left, right in driver.pairs if (left, right) != (0, 0)]
    assert len(nonzero) == 1
    left, right = nonzero[0]
    assert left == -right and left + right == 0  # No old forward authority.
    assert (left < 0) if direction == "left" else (left > 0)
    assert tracker.current_command == action
    assert tracker._longitudinal_valid_until == 99.8
