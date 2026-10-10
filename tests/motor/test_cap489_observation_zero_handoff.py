"""CAP486 soft zero -> CAP489 confirmation: no hardware, fake serial/clock."""
from dataclasses import replace
from queue import Queue
from types import SimpleNamespace
import threading

import pytest

import request_0513_modular as main
from car_control_modular.control_types import SteeringFeedback
from car_control_modular.controllers import FollowPolicyConfig
from car_control_modular.search_reacquire_braking import (
    OBSERVATION_ZERO_HANDOFF_MAX_SEC,
    OBSERVATION_ZERO_HANDOFF_MAX_WHEEL_RPM,
    OBSERVATION_ZERO_HANDOFF_MAX_YAW_DPS,
    search_brake_reason,
)
from test_depth_drive_rpm import make_runtime


BOX = (222.31787109375, 3.2401885986328125, 332.78900146484375, 431.250732421875)


def sample(stamp, left=-2., right=-2., yaw=0., **changes):
    fb = SteeringFeedback(timestamp=stamp, trustworthy=True,
        left_forward_rpm=left, right_forward_rpm=right,
        raw_yaw_rate_right_dps=yaw, yaw_rate_right_dps=yaw,
        left_read_started=stamp-.008, left_read_finished=stamp-.006,
        right_read_started=stamp-.004, right_read_finished=stamp-.002)
    return replace(fb, **changes)


@pytest.fixture
def rig(monkeypatch):
    rt, old_owner, driver, symbols = make_runtime()
    owner = object.__new__(main.PersonTracker)
    owner.__dict__.update(vars(old_owner))
    rt.owner = owner
    clock = [10.]
    monkeypatch.setattr(main.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(main.time, "time", lambda: clock[0])
    owner._action_runtime = rt
    owner.running = True
    owner.search_state, owner.search_direction = "searching", "left"
    owner._search_epoch = 2
    owner.command_lock = threading.Lock()
    owner.action_queue = Queue()
    owner.action_queue_lock = threading.Lock()
    owner._follow_controller = SimpleNamespace(active_target_id=1,
        search_state="searching", search_direction="left",
        cfg=FollowPolicyConfig(center_left_ratio=.47, center_right_ratio=.53,
            visible_steering_pid_camera_hfov_deg=60.,
            visible_steering_pid_camera_latency_sec=.13,
            visible_steering_pid_predictive_brake_decel_dps2=100.,
            visible_steering_pid_predictive_brake_margin_deg=1.,
            visible_steering_pid_predictive_brake_response_sec=.05))
    owner._search_retry_zero_requested_at = 9.99
    owner._search_retry_zero_sent_at = None
    owner._active_capture_frame_id = 489
    owner._active_capture_timestamp = 9.99
    owner.frame_index = 156
    owner.clears = []
    owner._clear_lateral_intent = lambda reason: owner.clears.append(reason)
    owner._clear_longitudinal_context = lambda **kw: owner.clears.append(kw)
    owner._publish_observation_soft_zero = lambda *a, **kw: pytest.fail("no extra observation stop")
    rt.get_steering_feedback = lambda: sample(clock[0]-.01)
    return rt, owner, driver, symbols, clock


def zero_pair(rig):
    rt, _, _, _, clock = rig
    rt.send_rotate_pulse_zero_stop()
    first = rt._search_observation_zero_evidence
    assert first is not None
    clock[0] += .015
    rt.send_percent_drive(0)
    assert rt._search_observation_zero_evidence.completed_at == first.completed_at
    assert rt._search_observation_zero_evidence.receipt is rt.backend.last_speed_receipt
    return first


def decide(owner, confirmed=True, eligible=True):
    return owner._hold_search_reacquire_brake(
        bbox=BOX, width=640, eligible=eligible, confirmed=confirmed, raw_track_id=4)


def test_actual_main_and_writer_reuse_two_completed_observation_zeros(rig, caplog):
    rt, owner, driver, _, clock = rig
    zero_pair(rig)
    clock[0] = 10.08
    # One current post-zero dual-wheel read is enough for LOW RESIDUAL; this
    # is not a new two-feedback/static-frame gate. CAP489's RGB predates zero.
    owner._active_capture_timestamp = 9.98633
    rt.get_steering_feedback = lambda: sample(10.07, -3., 1., -6.27,
        yaw_rate_right_dps=-10.9725)
    assert search_brake_reason(bbox=BOX, width=640, direction="left", eligible=True,
        now=clock[0], capture_timestamp=owner._active_capture_timestamp, max_age=.21,
        feedback=rt.get_steering_feedback(), policy=owner._follow_controller.cfg,
        execution_delay_sec=.05) == "candidate_predictive_stop"
    owner._search_evidence_observation_active = True
    owner._release_search_observation_for_control("candidate_observation_completed")
    assert owner._search_retry_zero_requested_at == 0.
    before = list(driver.pairs)
    with caplog.at_level("INFO"):
        assert not decide(owner)
    assert "low_residual_handoff" in caplog.text and "quiet_claim=False" in caplog.text
    assert driver.pairs == before == [(0, 0), (0, 0)]
    assert driver.stops == [] and driver.register_writes == []
    assert rt._search_reacquire_brake_request is None
    assert rt._search_reacquire_settling is None and not owner._brake_hold_active
    assert owner.clears == [] and owner._follow_controller.active_target_id == 1
    assert owner.search_state == "searching"  # this check does not create a depth grant


def test_search_release_to_handoff_does_not_rearm_the_same_parking_cycle(rig):
    rt, owner, driver, _, clock = rig
    zero_pair(rig); clock[0] = 10.08
    rt.get_steering_feedback = lambda: sample(clock[0]-.01, -3., 1., -6.27)
    owner._release_search_observation_for_control("candidate_observation_completed")
    assert not decide(owner)
    assert owner._search_candidate_brake_episode == 10.08
    owner.search_state = owner._follow_controller.search_state = "none"
    owner._search_handoff_uid, owner._search_handoff_direction = 1, "left"
    owner._search_handoff_started_capture_ts = owner._active_capture_timestamp
    owner._fresh_depth_linear_snapshot = lambda *a, **kw: None
    clock[0] = 10.15
    owner._active_capture_frame_id += 1
    owner._active_capture_timestamp = 10.10
    assert not decide(owner)
    assert rt._search_reacquire_brake_request is None
    assert not driver.stops and driver.pairs == [(0, 0), (0, 0)]


def test_queued_zero_completing_after_confirmed_release_preserves_original_origin(rig):
    rt, owner, driver, _, clock = rig
    first = zero_pair(rig)
    clock[0] = 10.06
    owner._release_search_observation_for_control("candidate_observation_completed")
    rt.send_percent_drive(0)  # already queued soft-zero completion, no new observation
    evidence = rt._search_observation_zero_evidence
    assert evidence is not None and evidence.completed_at == first.completed_at
    assert evidence.episode == first.episode
    assert evidence.receipt is rt.backend.last_speed_receipt
    clock[0] = 10.08
    rt.get_steering_feedback = lambda: sample(10.07, -3., 1., -6.27)
    assert not decide(owner)
    assert not driver.stops and driver.pairs == [(0, 0)] * 3


def test_cap486_pre_zero_feedback_is_not_post_zero_progress(rig):
    rt, _, _, _, clock = rig
    zero_pair(rig)
    clock[0] = 10.08
    rt.get_steering_feedback = lambda: sample(9.995, -3., 1., -6.27)
    assert not rt.search_observation_zero_handoff(uid=1)
    # CAP489's subsequent independent feedback can take over without another
    # zero or an obligatory second sample, even at -2/-2 RPM.
    rt.get_steering_feedback = lambda: sample(10.07, -2., -2., 0.)
    assert rt.search_observation_zero_handoff(uid=1)


@pytest.mark.parametrize("left,right,yaw,expected", [
    (-3, 1, -6.27, True), (-2, -2, 0, True), (0, 0, 0, True),
    (3, -3, 10, True), (3.01, 0, 0, False), (0, -3.01, 0, False),
    (2, -2, 10.01, False), (-2, 2, -10.01, False),
])
def test_low_residual_bounds_are_not_strict_stillness(rig, left, right, yaw, expected):
    rt, _, driver, _, clock = rig
    zero_pair(rig); clock[0] = 10.08
    rt.get_steering_feedback = lambda: sample(10.07, left, right, yaw)
    before = list(driver.pairs)
    assert rt.search_observation_zero_handoff(uid=1) is expected
    assert driver.pairs == before and not driver.stops
    assert OBSERVATION_ZERO_HANDOFF_MAX_WHEEL_RPM == 3.
    assert OBSERVATION_ZERO_HANDOFF_MAX_YAW_DPS == 10.


@pytest.mark.parametrize("changes", [
    dict(trustworthy=False), dict(yaw_rate_confirmed=False), dict(left_error=1),
    dict(right_error=1), dict(timestamp=9.99), dict(timestamp=10.20),
    dict(left_read_started=None), dict(left_read_started=9.99),
    dict(right_read_started=10.09), dict(right_read_finished=10.071),
    dict(raw_yaw_rate_right_dps=None), dict(raw_yaw_rate_right_dps=float("nan")),
    dict(left_forward_rpm=float("nan")),
])
def test_invalid_or_pre_write_read_cannot_supply_handoff(rig, changes):
    rt, _, _, _, clock = rig
    zero_pair(rig); clock[0] = 10.08
    rt.get_steering_feedback = lambda: sample(10.07, **changes)
    assert not rt.search_observation_zero_handoff(uid=1)


def test_cache_polling_and_repeated_zero_do_not_extend_physical_deadlines(rig):
    rt, _, driver, _, clock = rig
    first = zero_pair(rig)
    rt.get_steering_feedback = lambda: sample(10.02)
    clock[0] = 10.08
    assert rt.search_observation_zero_handoff(uid=1)
    clock[0] = 10.121
    assert not rt.search_observation_zero_handoff(uid=1)
    rt.send_percent_drive(0)
    assert rt._search_observation_zero_evidence.completed_at == first.completed_at
    clock[0] = first.completed_at + OBSERVATION_ZERO_HANDOFF_MAX_SEC + .001
    rt.get_steering_feedback = lambda: sample(clock[0]-.01)
    assert not rt.search_observation_zero_handoff(uid=1)
    assert driver.stops == []


@pytest.mark.parametrize("change", ["nonzero", "stop", "unowned_zero", "inflight", "fault"])
def test_any_intervening_write_breaks_the_observation_chain(rig, change):
    rt, _, driver, _, clock = rig
    zero_pair(rig); clock[0] = 10.03
    if change == "nonzero": rt.backend.send_targets(7, 7, "TURN")
    elif change == "stop": rt.backend.send_stop("real_stop", mode="free")
    elif change == "unowned_zero": rt.backend.send_targets(0, 0, "other_zero")
    elif change == "inflight": rt.backend.last_speed_receipt = None
    else:
        def fail(_): raise OSError("test serial failure")
        driver.set_left_speed = fail
        with pytest.raises(OSError):
            rt.backend.send_targets(7, 7, "failed_write")
    assert not rt.search_observation_zero_handoff(uid=1)
    if change != "fault":
        # Restoring zero is a new physical transition, not continuity of the
        # original observation. It cannot silently re-arm this spent proof.
        rt.send_percent_drive(0)
        clock[0] = 10.08
        assert not rt.search_observation_zero_handoff(uid=1)


@pytest.mark.parametrize("change", ["uid", "epoch", "new_observation", "explicit", "shutdown",
    "not_running", "search_ended", "controller_search_ended", "parking", "parking_unknown", "hard_stop"])
def test_other_episode_or_stop_owner_cannot_reuse_zero(rig, change):
    rt, owner, _, _, clock = rig
    zero_pair(rig); clock[0] = 10.08
    if change == "uid": owner._follow_controller.active_target_id = 2
    elif change == "epoch": owner._search_epoch += 1
    elif change == "new_observation": owner._search_retry_zero_requested_at += .01
    elif change == "explicit": owner._explicit_stop_requested = True
    elif change == "shutdown": owner._runtime_shutdown_requested = True
    elif change == "not_running": owner.running = False
    elif change == "search_ended": owner.search_state = "none"
    elif change == "controller_search_ended": owner._follow_controller.search_state = "none"
    elif change == "parking": rt.backend.parking_current_a = 5.
    elif change == "parking_unknown": rt.backend._parking_current_uncertain = True
    else: rt.hard_stop_check = lambda action: True
    assert not rt.search_observation_zero_handoff(uid=1)
    assert rt._search_observation_zero_evidence is None


def test_unconfirmed_candidate_keeps_original_brake_decision(rig):
    rt, owner, _, _, clock = rig
    zero_pair(rig); clock[0] = 10.08
    rt.get_steering_feedback = lambda: sample(10.07, -3., 1., -6.27)
    assert decide(owner, confirmed=False)
    assert rt._search_reacquire_brake_request is not None


def test_large_residual_keeps_original_full_brake_path(rig):
    rt, owner, driver, _, clock = rig
    zero_pair(rig); clock[0] = 10.08
    rt.get_steering_feedback = lambda: sample(10.07, -18., 18., -50.)
    assert decide(owner)
    assert rt._search_reacquire_brake_request is not None
    rt._service_search_reacquire_brake()
    assert driver.stops and owner._brake_hold_active
    # A real brake still owns this case, but no fixed minimum dwell is needed.
    assert rt._search_reacquire_settling.MAX_CURRENT_HOLD_SEC == .5
    assert rt._search_reacquire_settling.current_released_at is None


def test_new_observation_can_start_new_zero_progress_after_invalid_episode(rig):
    rt, owner, _, _, clock = rig
    zero_pair(rig)
    rt.backend.send_targets(7, 7, "TURN")
    assert not rt.search_observation_zero_handoff(uid=1)
    clock[0] = 10.1
    owner._search_retry_zero_requested_at = clock[0]
    owner._search_retry_zero_sent_at = None
    rt.send_rotate_pulse_zero_stop()
    clock[0] = 10.15
    assert rt.search_observation_zero_handoff(uid=1)
