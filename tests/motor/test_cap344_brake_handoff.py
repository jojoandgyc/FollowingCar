"""CAP344–393 regression: fake encoder/driver only; never opens serial."""
import ast
import logging
from pathlib import Path

import pytest

from car_control_modular.wheel_zero_cross import WheelZeroCrossGuard
from test_visible_wheel_continuity import feedback, visible_runtime


@pytest.mark.parametrize("direction", [-1, 1])
def test_crossing_between_samples_can_release_current_turn(direction):
    g = WheelZeroCrossGuard()
    turn = (8 * direction, -8 * direction)
    assert g.limit(turn, feedback(10., 21, 20), 10., allow_aligned_turn=True)[0] == (0, 0)
    g.note_sent((0, 0), 10.)
    aligned = (3 * direction, -4 * direction)
    assert g.limit(turn, feedback(10.05, *aligned), 10.05, allow_aligned_turn=True)[0] == (0, 0)
    assert g.limit(turn, feedback(10.05, *aligned), 10.06, allow_aligned_turn=True)[0] == (0, 0)
    result = g.limit((6 * direction, -6 * direction), feedback(10.10, *aligned),
                     10.10, allow_aligned_turn=True)
    assert result == ((6 * direction, -6 * direction), "cross_confirmed_turn_handoff")
    assert g.pending_signs is None and g.resume_signs is None


@pytest.mark.parametrize("case", ["stale", "future", "nan", "untrusted", "duplicate",
                                  "opposing", "overspeed", "direction", "gap", "opt_out"])
def test_turn_release_cannot_ignore_bad_or_conflicting_evidence(case):
    g = WheelZeroCrossGuard()
    turn = (8, -8)
    g.limit(turn, feedback(10., 21, 20), 10., allow_aligned_turn=True)
    g.limit(turn, feedback(10.05, 3, -3), 10.05, allow_aligned_turn=True)
    fb, now = feedback(10.10, 3, -3), 10.10
    if case == "stale": fb.timestamp = 9.
    if case == "future": fb.timestamp = 11.
    if case == "nan": fb.left_forward_rpm = float("nan")
    if case == "untrusted": fb.trustworthy = False
    if case == "duplicate": fb.timestamp = 10.05
    if case == "opposing": fb.right_forward_rpm = 5
    if case == "overspeed": fb.left_forward_rpm = 12
    if case == "direction": turn = (-8, 8)
    if case == "gap": now = fb.timestamp = 10.25
    assert g.limit(turn, fb, now, allow_aligned_turn=case != "opt_out")[0] == (0, 0)


def enable_runtime(monkeypatch):
    r, owner, driver, symbols, clock = visible_runtime(monkeypatch)
    r.config.follow_cross_brake_enable = True
    r.config.follow_cross_brake_mode = "normal"  # Explicit legacy experiment, not runtime default.
    r.config.follow_forward_handoff_enable = True
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
    return r, owner, driver, symbols, clock


def test_cap349_brakes_once_then_holds_without_zero_keepalives(monkeypatch, caplog):
    r, owner, driver, _, clock = enable_runtime(monkeypatch)
    with caplog.at_level(logging.INFO):
        for t, wheels in [(10., (18, 20)), (10.05, (6, 7)), (10.30, (5, 5))]:
            clock[0] = t
            r.get_steering_feedback = lambda: feedback(clock[0], *wheels)
            r._send_follow_wheel_targets(5, 5, "TEST")
    assert driver.stops == [1, 0]  # EMERGENCY preparation, then NORMAL hold
    assert driver.pairs == [(0, 0)]  # send_stop's pre-zero only
    assert r.backend.normal_zero_hold
    assert r._visible_wheel_guard.last_sent == 10.
    assert "cross_brake=applied packet_written=True" in caplog.text
    assert "cross_brake=held packet_written=False cross_wait_ms=300.0" in caplog.text
    assert "reason=cross_timeout_zero" in caplog.text
    # No sleep/forced release after 250ms, nor duplicate confirmation.
    for t in (10.35, 10.35, 10.40):
        clock[0] = t
        r.get_steering_feedback = lambda: feedback(clock[0], 0, 0)
        r._send_follow_wheel_targets(5, 5, "TEST")
    assert driver.pairs == [(0, 0), (5, 5)]
    assert not r.backend.normal_zero_hold


@pytest.mark.parametrize("direction", [-1, 1])
def test_depth_returns_during_brake_uses_latest_forward_not_old_turn(monkeypatch, direction):
    r, owner, driver, _, clock = enable_runtime(monkeypatch)
    r.get_steering_feedback = lambda: feedback(clock[0], 21, 20)
    r._send_follow_wheel_targets(8 * direction, 8 * direction, "TEST")
    assert driver.stops == [1, 0]
    clock[0] = 10.05
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: ("forward", 62)
    r.get_steering_feedback = lambda: feedback(clock[0], 12, 11)
    pair = (64, 60) if direction == 1 else (60, 64)
    r._send_follow_wheel_targets(pair[0], -pair[1], "TEST")
    assert driver.pairs[-1] == (pair[0], -pair[1])
    assert not r.backend.normal_zero_hold
    assert r._visible_wheel_guard.pending_signs is None


def test_cap384_real_reverse_still_waits_then_releases_latest_grant(monkeypatch):
    r, owner, driver, _, clock = enable_runtime(monkeypatch)
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: ("forward", 72)
    for t, wheels in [(10., (-4, -6)), (10.05, (-1, -3)), (10.10, (0, 0)),
                      (10.10, (0, 0)), (10.15, (-2, 0))]:
        clock[0] = t
        r.get_steering_feedback = lambda: feedback(clock[0], *wheels)
        r._send_follow_wheel_targets(72, -72, "TEST")
        assert driver.pairs == [(0, 0)]
    for t in (10.20, 10.25):
        clock[0] = t
        r.get_steering_feedback = lambda: feedback(clock[0], 3, 1)
        r._send_follow_wheel_targets(58, -58, "TEST")
    assert driver.pairs == [(0, 0), (58, -58)]
    assert driver.stops == [1, 0]


@pytest.mark.parametrize("loss", ["depth", "yaw", "uid", "search", "shutdown"])
def test_qualification_lost_during_feedback_cannot_release_brake(monkeypatch, loss):
    r, owner, driver, _, clock = enable_runtime(monkeypatch)
    r.get_steering_feedback = lambda: feedback(clock[0], 21, 20)
    r._send_follow_wheel_targets(8, 8, "TEST")
    clock[0] = 10.05
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: ("forward", 62)
    def read_feedback():
        if loss == "depth": owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
        if loss == "yaw": owner._has_fresh_lateral_yaw = lambda uid: False
        if loss == "uid": owner._follow_controller.active_target_id = 2
        if loss == "search": owner.search_state = "searching"
        if loss == "shutdown": owner._runtime_shutdown_requested = True
        return feedback(clock[0], 12, 11)
    r.get_steering_feedback = read_feedback
    r._send_follow_wheel_targets(64, -60, "TEST")
    assert driver.pairs == [(0, 0)]
    assert r.backend.normal_zero_hold


def test_explicit_zero_and_other_zero_writers_do_not_unlock_brake(monkeypatch):
    r, owner, driver, _, clock = enable_runtime(monkeypatch)
    r.get_steering_feedback = lambda: feedback(clock[0], 21, 20)
    r._send_follow_wheel_targets(8, 8, "TEST")
    r._send_follow_wheel_targets(0, 0, "TEST")
    for label in ("FOLLOW20_REVOKED", "TURN_ZERO", "FORWARD_ZERO"):
        r.backend.send_targets(0, 0, label)
    assert driver.pairs == [(0, 0)] and driver.stops == [1, 0]
    assert r._visible_wheel_guard.pending_signs is None


@pytest.mark.parametrize("mode,expected", [("emergency", 1), ("free", 2), ("normal", 0)])
def test_other_explicit_stop_always_supersedes_transition_hold(monkeypatch, mode, expected):
    r, owner, driver, _, clock = enable_runtime(monkeypatch)
    r.get_steering_feedback = lambda: feedback(clock[0], 21, 20)
    r._send_follow_wheel_targets(8, 8, "TEST")
    r.backend.send_stop("safety_or_explicit", mode=mode)
    assert driver.stops == [1, 0] + ([1, 0] if expected == 0 else [expected])
    assert not r.backend.normal_zero_hold


@pytest.mark.parametrize("failed_operation", ["motion", "stop"])
def test_partial_serial_failure_cannot_leave_false_brake_latch(monkeypatch, failed_operation):
    r, _, driver, _, _ = enable_runtime(monkeypatch)
    r.backend.send_stop("test", mode="normal", preserve_zero=True)
    def fail(*args, **kwargs):
        raise OSError("fake serial failure")
    if failed_operation == "motion":
        monkeypatch.setattr(driver, "set_left_speed", fail)
        with pytest.raises(OSError):
            r.backend.send_targets(8, 8, "TEST")
    else:
        monkeypatch.setattr(driver, "stop_all", fail)
        with pytest.raises(OSError):
            r.backend.send_stop("test", mode="normal", preserve_zero=True)
    assert not r.backend.normal_zero_hold


def test_periodic_base_expiry_brakes_without_authorizing_forward(monkeypatch):
    from test_follow_wheel_periodic import setup_periodic
    r, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    r.config.follow_cross_brake_enable = True
    r.config.follow_cross_brake_mode = "normal"
    r.config.follow_forward_handoff_enable = True
    state[:] = [24., 8., 10.01, 11.]
    r._service_follow_wheels()
    assert driver.pairs == [(32, -16)]
    clock[0] = 10.011
    r._service_follow_wheels()
    assert driver.stops == [1, 0] and driver.pairs[-1] == (0, 0)
    for t in (10.06, 10.11):
        clock[0] = t
        r._service_follow_wheels()
    assert driver.pairs == [(32, -16), (0, 0)]
    # A new grant is the ONLY reason to restore forward, with current pair.
    state[:] = [62., 2., 11., 11.]
    clock[0] = 10.16
    r.get_steering_feedback = lambda: feedback(clock[0], 12, 11)
    r._service_follow_wheels()
    assert driver.pairs[-1] == (64, -60)
    assert driver.stops == [1, 0]


def test_periodic_hard_stop_preempts_transition_hold_between_ticks(monkeypatch):
    from test_follow_wheel_periodic import setup_periodic
    r, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    r.config.follow_cross_brake_enable = True
    r.config.follow_cross_brake_mode = "normal"
    state[:] = [0., 8., 11., 11.]
    r._service_follow_wheels()
    assert driver.stops == [1, 0]
    clock[0] = 10.005
    r.hard_stop_check = lambda action: True
    stops = []
    r.send_stop_with_brake_hold = stops.append
    r._service_follow_wheels()
    assert stops == ["follow20_hard_stop"]
    assert driver.pairs == [(0, 0)]


@pytest.mark.parametrize("state", ["search", "low_quality", "rotation_only"])
def test_opt_in_does_not_expand_visible_control_eligibility(monkeypatch, state):
    r, owner, driver, _, clock = enable_runtime(monkeypatch)
    if state == "search": owner.search_state = "searching"
    if state == "low_quality": owner._vision_control_state = "target_visible_low_quality"
    if state == "rotation_only": r.config.rotation_only = True
    r.get_steering_feedback = lambda: feedback(clock[0], 21, 20)
    assert not r._visible_wheel_control_active()
    r._send_follow_wheel_targets(8, 8, "TEST", visible_required=True)
    assert not driver.pairs and not driver.stops


def test_normal_config_wires_opt_in_without_changing_rotation_only(monkeypatch):
    from car_control_modular.config_loader import load_config_to_env
    import os
    root = Path(__file__).resolve().parents[2]
    with monkeypatch.context() as m:
        # Loader writes many keys: isolate the complete environment.
        m.setattr(os, "environ", {})
        load_config_to_env(str(root / "car_control_modular/config/reid_runtime.ini"))
        assert os.environ["FOLLOW_CROSS_BRAKE_ENABLE"] == "1"
        assert os.environ["FOLLOW_CROSS_BRAKE_MODE"] == "zero"
    tree = ast.parse((root / "request_0513_modular.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and any(k.arg == "follow_cross_brake_enable" for k in n.keywords)]
    assert len(calls) == 1
    expression = next(k.value for k in calls[0].keywords if k.arg == "follow_cross_brake_enable")
    for enabled in ("0", "1"):
        m = type("Env", (), {"environ": {"FOLLOW_CROSS_BRAKE_ENABLE": enabled}})
        assert eval(compile(ast.Expression(expression), "binding", "eval"), {"os": m}) == (enabled == "1")
    assert "follow_cross_brake_enable = true" not in (root / "car_control_modular/config/reid_runtime_rotation_only.ini").read_text()
