"""CAP1163: bounded current-axis rebuild and cached feedback diagnostics.

Fake motor, encoder and clock only; no device or launcher is opened.
"""
from types import SimpleNamespace

import pytest

from test_follow_wheel_periodic import setup_periodic
from test_visible_wheel_continuity import feedback, visible_runtime


@pytest.mark.parametrize("old_base,new_base,new_yaw", [(0, 60, -10), (40, 60, -10), (60, 20, -5)])
def test_new_valid_grant_rebuilds_without_inserting_zero(monkeypatch, old_base, new_base, new_yaw):
    r, o, d, _, clock, state = setup_periodic(monkeypatch)
    state[:] = [old_base, -7., 11., 11.]
    reads = []
    def read():
        reads.append(1)
        if len(reads) == 1:
            state[:2] = [new_base, new_yaw]
        return feedback(clock[0], 15, 20)
    r.get_steering_feedback = read
    r.config.follow_forward_handoff_enable = True
    r._service_follow_wheels()
    assert len(reads) == 2
    assert d.pairs == [(new_base+new_yaw, -(new_base-new_yaw))]
    assert not d.stops
    assert r._follow_wheel_clock.last_axes[2:] == (new_base, new_yaw)


def test_second_axis_replacement_stops_instead_of_unbounded_retry(monkeypatch):
    r, o, d, _, clock, state = setup_periodic(monkeypatch)
    state[:] = [20., 0., 11., 11.]
    reads = []
    def read():
        reads.append(1)
        state[0] += 1
        return feedback(clock[0], 0, 0)
    r.get_steering_feedback = read
    r._service_follow_wheels()
    assert len(reads) == 2
    assert d.pairs == [(0, 0)]
    assert r._follow_wheel_clock.last_axes is None


@pytest.mark.parametrize("cause", ["expired_depth", "uid", "shutdown", "danger", "search"])
def test_rebuild_does_not_override_revocation(monkeypatch, cause):
    r, o, d, _, clock, state = setup_periodic(monkeypatch)
    state[:] = [20., -7., 11., 11.]
    reads = []
    def read():
        reads.append(1)
        state[0] = 40
        if cause == "expired_depth": state[2] = 9.
        elif cause == "uid": o._follow_controller.active_target_id = 2
        elif cause == "shutdown": o._runtime_shutdown_requested = True
        elif cause == "danger": r.hard_stop_check = lambda _: True
        elif cause == "search": o.search_state = "searching"
        return feedback(clock[0], 20, 20)
    r.get_steering_feedback = read
    r._service_follow_wheels()
    assert len(reads) <= 2
    assert all(pair == (0, 0) for pair in d.pairs)
    if cause == "danger": assert d.stops


def test_rebuild_keeps_actual_opposing_reverse_protection(monkeypatch):
    r, o, d, _, clock, state = setup_periodic(monkeypatch)
    state[:] = [0., -7., 11., 11.]
    r.config.follow_forward_handoff_enable = True
    def read():
        state[0] = 40
        return feedback(clock[0], -30, -30)
    r.get_steering_feedback = read
    r._service_follow_wheels()
    assert d.pairs == [(0, 0)]


def test_final_safety_check_applies_to_unchanged_pair(monkeypatch):
    r, o, d, _, clock = visible_runtime(monkeypatch)
    r.hard_stop_check = lambda _: True
    r._send_follow_wheel_targets(24, -24, "UNCHANGED")
    assert not d.pairs
    assert d.stops


def test_feedback_records_individual_read_intervals_without_extra_io(monkeypatch):
    r, o, d, _, clock = visible_runtime(monkeypatch)
    r.config.steering_feedback_poll_interval_sec = .05
    r.config.steering_feedback_log_interval_sec = .5
    r.config.steering_feedback_left_body_deg_per_encoder_deg = .1
    r.config.steering_feedback_right_body_deg_per_encoder_deg = .1
    class Event:
        stopped = False
        def is_set(self): return self.stopped
        def wait(self, _): self.stopped = True
    o.action_stop_event = Event()
    sides = []
    def read(side):
        sides.append(side)
        clock[0] += .02 if side == "left" else .03
        return SimpleNamespace(speed_rpm=3, position_degree=0, error_code=0)
    d.read_motor_status = read
    r._steering_feedback_loop()
    f = r._steering_feedback
    assert sides == ["left", "right"]
    assert f.left_read_started == pytest.approx(10.)
    assert f.left_read_finished == pytest.approx(10.02)
    assert f.right_read_started == pytest.approx(10.02)
    assert f.right_read_finished == f.timestamp == pytest.approx(10.05)
    assert not d.pairs and not d.stops


def test_periodic_saturation_limits_yaw_not_approved_base(monkeypatch):
    r, o, d, _, clock, state = setup_periodic(monkeypatch)
    r.config.motor_forward_max_target_rpm = 200
    state[:] = [196., -10., 11., 11.]
    r._service_follow_wheels()
    assert d.pairs == [(192, -200)]


@pytest.mark.parametrize("loss", ["depth", "yaw", "uid", "stop"])
def test_final_safety_callback_cannot_carry_old_authority_across_deadline(monkeypatch, loss):
    r, o, d, _, clock = visible_runtime(monkeypatch)
    def check(_):
        if loss == "depth": o._fresh_depth_linear_snapshot = lambda uid, now=None: None
        elif loss == "yaw": o._has_fresh_lateral_yaw = lambda uid: False
        elif loss == "uid": o._follow_controller.active_target_id = 2
        elif loss == "stop": o._explicit_stop_requested = True
        return False
    r.hard_stop_check = check
    r._send_follow_wheel_targets(20, -28, "FINAL_RACE")
    assert d.pairs == [(0, 0)]


@pytest.mark.parametrize("case", ["stale", "fault", "untrusted", "nan"])
def test_regular_quiet_release_rechecks_stop_proof_after_safety_callback(monkeypatch, case):
    r, o, d, _, clock = visible_runtime(monkeypatch)
    o._fresh_depth_linear_snapshot = lambda uid, now=None: None
    o._has_fresh_lateral_yaw = lambda uid: clock[0] < 10.30
    for now, wheels in ((10., (20, 20)), (10.05, (0, 0))):
        clock[0] = now
        r.get_steering_feedback = lambda: feedback(clock[0], *wheels)
        r._send_follow_wheel_targets(-7, -7, "QUIET_RELEASE")
    clock[0] = 10.10
    sample = feedback(clock[0], 0, 0)
    r.get_steering_feedback = lambda: sample
    def check(_):
        if case == "stale": clock[0] += .16
        elif case == "fault": sample.left_error = 1
        elif case == "untrusted": sample.trustworthy = False
        elif case == "nan": sample.left_forward_rpm = float("nan")
        return False
    r.hard_stop_check = check
    r._send_follow_wheel_targets(-7, -7, "QUIET_RELEASE")
    assert d.pairs == [(0, 0)]*3
