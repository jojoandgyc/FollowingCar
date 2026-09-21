"""CAP2008/2011/2077 regressions, cached encoders and fake serial only."""
import queue
import threading
from dataclasses import replace

import pytest

from car_control_modular.action_command import ActionCommandSnapshot
from car_control_modular.control_types import SteeringFeedback
from test_follow_wheel_periodic import setup_periodic


def command(s, action=None, **kw):
    values = dict(action=s.stop if action is None else action, revision=1,
                  enqueued_at=10., control_frame=797, capture_frame_id=2074,
                  capture_timestamp=9.9, reason="lateral_zero:expired", soft_stop=True)
    values.update(kw)
    return ActionCommandSnapshot(**values)


@pytest.mark.parametrize("write", ["stop", "drive", "raw", "diff", "brake", "hold"])
def test_old_popped_command_cannot_write_after_new_publication(monkeypatch, write):
    rt, owner, driver, s, _, _ = setup_periodic(monkeypatch)
    rt._dispatch_context.command = command(s)
    owner._action_command_revision = 2
    if write == "stop": rt.send_robot_command(s.stop)
    elif write == "drive": rt.send_percent_drive(0)
    elif write == "raw": rt._send_follow_wheel_targets(0, 0, "FOLLOW20")
    elif write == "diff": rt.send_percent_diff(0, 1, 0, 1, "STEER")
    elif write == "brake": rt.send_percent_brake()
    else: rt.send_stop_with_brake_hold("stop_signal")
    assert driver.stops == [] and driver.pairs == []
    assert not owner._brake_hold_active


def test_old_command_cannot_suppress_actual_safety_stop(monkeypatch):
    rt, owner, driver, s, _, _ = setup_periodic(monkeypatch)
    rt._dispatch_context.command = command(s)
    owner._action_command_revision = 2
    rt.send_stop_with_brake_hold("hard_stop")
    assert driver.stops == [1]


def test_protected_stop_survives_new_motion_revision(monkeypatch):
    rt, owner, driver, s, _, _ = setup_periodic(monkeypatch)
    rt._dispatch_context.command = command(s, protected_stop=True, soft_stop=False)
    owner._action_command_revision = 2
    rt.send_robot_command(s.stop)
    assert driver.stops == [0]


def test_stop_uses_own_soft_mode_not_new_global_flags(monkeypatch):
    rt, owner, driver, s, _, _ = setup_periodic(monkeypatch)
    owner.search_state = "searching"
    owner._action_command_revision = 1
    rt._dispatch_context.command = command(s, soft_stop=False)
    owner._use_soft_stop_next = True
    rt.send_robot_command(s.stop)
    assert driver.stops == [0]


def arm(rt, owner, clock):
    owner.command_lock = threading.Lock()
    rt.config.rotate_pulse_settle_feedback_stale_sec = .30
    rt.config.rotate_pulse_settle_max_wheel_rpm = 5
    rt.config.rotate_pulse_settle_max_yaw_rate_dps = 10.
    owner.action_queue = queue.Queue()
    owner.action_queue_lock = threading.Lock()
    rt.request_search_reacquire_brake(2008, clock[0]-.10, "candidate_center")


def quiet(stamp, **kw):
    return SteeringFeedback(timestamp=stamp, trustworthy=True, **kw)


def test_search_stop_is_normal_once_and_no_zero_or_motion_can_unlock(monkeypatch):
    rt, owner, driver, s, clock, _ = setup_periodic(monkeypatch)
    arm(rt, owner, clock)
    assert driver.stops == []  # intent producer never touches serial
    assert rt.search_reacquire_brake_pending()  # not applied yet
    rt._service_follow_wheels()
    for _ in range(3):
        rt._service_follow_wheels()
        rt.send_robot_command(s.rotate_left)
        rt.send_percent_drive(0)
        rt.send_rotate_pulse_zero_stop()
        rt.send_stop_with_brake_hold("ordinary_stop")
    assert driver.stops == [0] and driver.pairs == [(0, 0)]  # NORMAL's pre-stop zero only
    assert not rt.can_release_brake_hold(s.rotate_right)


def test_only_two_distinct_post_stop_quiet_feedbacks_release(monkeypatch):
    rt, owner, driver, _, clock, _ = setup_periodic(monkeypatch)
    arm(rt, owner, clock)
    rt._service_follow_wheels()
    rt.get_steering_feedback = lambda: quiet(10.)
    assert rt.search_reacquire_brake_pending()  # feedback predates/completes at stop
    clock[0] = 10.05
    rt.get_steering_feedback = lambda: quiet(10.05)
    assert rt.search_reacquire_brake_pending()
    assert rt.search_reacquire_brake_pending()  # repeated cache is not sample two
    clock[0] = 10.10
    rt.get_steering_feedback = lambda: quiet(10.10)
    assert not rt.search_reacquire_brake_pending()
    assert owner._brake_hold_active  # latched until a new command is adopted
    assert rt.can_release_brake_hold(rt.symbols.rotate_right)
    assert driver.stops == [0] and driver.pairs == [(0, 0)]


@pytest.mark.parametrize("kind", ["moving", "stale", "future", "invalid", "nan", "gap"])
def test_bad_feedback_never_releases_or_accumulates_quiet(kind, monkeypatch):
    rt, owner, _, _, clock, _ = setup_periodic(monkeypatch)
    arm(rt, owner, clock)
    rt._service_follow_wheels()
    clock[0] = 10.05
    rt.get_steering_feedback = lambda: quiet(10.05)
    assert rt.search_reacquire_brake_pending()
    clock[0] = 10.10 if kind != "gap" else 10.60
    fb = quiet(clock[0])
    if kind == "moving": fb = replace(fb, left_speed_rpm=8, raw_yaw_rate_right_dps=-19)
    elif kind == "stale": fb = replace(fb, timestamp=9.)
    elif kind == "future": fb = replace(fb, timestamp=12.)
    elif kind == "invalid": fb = replace(fb, trustworthy=False)
    elif kind == "nan": fb = replace(fb, raw_yaw_rate_right_dps=float("nan"))
    rt.get_steering_feedback = lambda: fb
    assert rt.search_reacquire_brake_pending()


def test_search_settle_never_releases_newer_safety_hold(monkeypatch):
    rt, owner, driver, _, clock, _ = setup_periodic(monkeypatch)
    arm(rt, owner, clock)
    rt._service_follow_wheels()
    rt.send_stop_with_brake_hold("hard_stop")
    for t in (10.05, 10.10):
        clock[0] = t
        rt.get_steering_feedback = lambda: quiet(clock[0])
        rt.search_reacquire_brake_pending()
    assert owner._brake_hold_active and owner._brake_hold_label == "safety_hold_hard_stop"
    assert driver.stops == [0, 1]


def test_revision_changes_after_dispatch_before_final_stop_write(monkeypatch):
    rt, owner, driver, s, _, _ = setup_periodic(monkeypatch)
    owner._action_command_revision = 1
    rt._dispatch_context.command = command(s, soft_stop=False)
    class PublishBeforeWrite:
        def __enter__(self):
            owner._action_command_revision = 2
        def __exit__(self, *args):
            pass
    owner.motor_io_lock = PublishBeforeWrite()
    rt.send_robot_command(s.stop)
    assert driver.stops == [] and driver.pairs == []


def test_actual_dispatch_log_uses_command_snapshot_not_new_owner_fields(monkeypatch, caplog):
    rt, owner, driver, s, _, _ = setup_periodic(monkeypatch)
    owner._action_command_revision = 1
    rt._dispatch_context.command = command(s, soft_stop=False)
    owner._last_action_queue_reason = "new_right_turn"
    owner._last_command_capture_frame = 2077
    with caplog.at_level("INFO"):
        rt.send_robot_command(s.stop)
    lines = [record.message for record in caplog.records if "电机下发时序" in record.message]
    assert len(lines) == 1
    assert "reason=lateral_zero:expired" in lines[0] and "依据采集帧=2074" in lines[0]
    assert "new_right_turn" not in lines[0]


def test_real_loop_rejects_popped_stop_when_new_motion_published(monkeypatch):
    rt, owner, driver, s, _, _ = setup_periodic(monkeypatch)
    owner.action_stop_event = threading.Event()
    owner.action_queue = queue.Queue()
    owner.action_queue.put(command(s, soft_stop=False))
    owner._action_command_revision = 1
    owner.current_command = None
    rt.config.enable_motor_rpm_feedback = False
    rt.config.action_intent_stale_sec = .45
    rt._service_follow_wheels = lambda: None
    rt._service_yaw_pulses = lambda: None
    class NewCommandAtAdoption:
        def __enter__(self):
            owner._action_command_revision = 2
            owner.action_stop_event.set()
        def __exit__(self, *args):
            pass
    owner.command_lock = NewCommandAtAdoption()
    rt.run_loop()
    assert driver.stops == [] and owner.current_command is None


def test_released_search_brake_refresh_cannot_repark_new_motion(monkeypatch):
    rt, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    arm(rt, owner, clock)
    rt._service_follow_wheels()
    for t in (10.05, 10.10):
        clock[0] = t
        rt.get_steering_feedback = lambda: quiet(clock[0])
        rt.search_reacquire_brake_pending()
    state[:] = [24., 5., 10.25, 10.25]
    assert rt.can_release_brake_hold(rt.symbols.steer_right)
    owner._brake_hold_active = False  # executor's release after a new decision
    owner._brake_hold_stop_mode = None
    owner._brake_hold_label = "brake"
    rt._service_follow_wheels()
    rt.send_percent_brake(mode="normal", label="search_reacquire_brake")
    assert driver.stops == [0]
    assert driver.pairs[-1] == (29,-19)
