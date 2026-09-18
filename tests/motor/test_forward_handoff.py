"""Residual turn -> forward handoff; never opens a motor or camera."""
import pytest

from car_control_modular.wheel_zero_cross import WheelZeroCrossGuard
from test_visible_wheel_continuity import feedback, visible_runtime


@pytest.mark.parametrize("measured", [(0, -3), (-3, 0), (8, -8), (-8, 8), (2, -4)])
@pytest.mark.parametrize("requested", [(60, 60), (65, 59), (60, 0)])
def test_forward_grant_does_not_wait_for_residual_rotation(measured, requested):
    g = WheelZeroCrossGuard()
    applied, reason = g.limit(requested, feedback(10, *measured), 10,
                              allow_forward_handoff=True)
    assert applied == requested
    assert reason == ("forward_controller_handoff" if any(
        command > 0 and speed < -1 for command, speed in zip(requested, measured)
    ) else "continuous")
    assert g.pending_signs is None and g.resume_signs is None


def test_handoff_clears_old_single_wheel_wait_and_no_relaunch_ramp():
    g = WheelZeroCrossGuard()
    assert g.limit((60, 60), feedback(10, 0, -3), 10)[1] == "cross_wait_zero"
    g.note_sent((0, 0), 10)
    assert g.limit((60, 60), feedback(10.05, 0, -2), 10.05,
                   allow_forward_handoff=True) == ((60, 60), "forward_controller_handoff")
    g.note_sent((60, 60), 10.05)
    assert g.limit((80, 80), feedback(10.1, 3, 2), 10.1,
                   allow_forward_handoff=True) == ((80, 80), "continuous")


def test_old_opposite_wheel_command_does_not_create_a_second_wait():
    g = WheelZeroCrossGuard()
    g.note_sent((8, -8), 10)
    assert g.limit((60, 60), feedback(9.99), 10.01,
                   allow_forward_handoff=True)[0] == (60, 60)


@pytest.mark.parametrize("bad", [None, feedback(9), feedback(11),
    feedback(10, trustworthy=False), feedback(float("nan")), feedback(10, right=float("inf"))])
def test_handoff_cannot_bypass_feedback_quality(bad):
    assert WheelZeroCrossGuard().limit((60, 60), bad, 10,
        allow_forward_handoff=True) == ((0, 0), "feedback_unavailable")


def test_explicit_zero_always_cancels_handoff():
    g = WheelZeroCrossGuard()
    assert g.limit((0, 0), None, 10, allow_forward_handoff=True) == ((0, 0), "explicit_zero")


@pytest.mark.parametrize("pair", [(8, -8), (-8, 8), (-60, -60)])
def test_reverse_or_rotation_requests_retain_guard(pair):
    assert WheelZeroCrossGuard().limit(pair, feedback(10, 20, 20), 10,
        allow_forward_handoff=True)[1] == "cross_wait_zero"


def test_whole_car_reverse_retains_two_quiet_samples():
    g = WheelZeroCrossGuard()
    assert g.limit((60, 60), feedback(10, -10, -10), 10,
                   allow_forward_handoff=True)[0] == (0, 0)
    g.note_sent((0, 0), 10)
    assert g.limit((60, 60), feedback(10.05), 10.05,
                   allow_forward_handoff=True)[1] == "cross_wait_zero"
    assert g.limit((60, 60), feedback(10.1), 10.1,
                   allow_forward_handoff=True) == ((60, 60), "continuous")


def test_whole_reverse_provenance_survives_changed_forward_curve():
    g = WheelZeroCrossGuard()
    g.limit((44, 44), feedback(10, -8, -8), 10, allow_forward_handoff=True)
    g.note_sent((0, 0), 10)
    assert g.limit((44, 0), feedback(10.05, -3, 0), 10.05,
                   allow_forward_handoff=True)[1] == "cross_wait_zero"
    assert g.pending_full_reverse
    assert g.limit((44, 0), feedback(10.1), 10.1,
                   allow_forward_handoff=True)[1] == "cross_wait_zero"
    assert g.limit((44, 0), feedback(10.15), 10.15,
                   allow_forward_handoff=True)[0] == (44, 0)


def test_whole_reverse_aligned_release_continues_original_ramp_without_rearming():
    g = WheelZeroCrossGuard()
    for stamp, speed in ((10, -10), (10.05, 8), (10.1, 8)):
        pair, reason = g.limit((44, 44), feedback(stamp, speed, speed), stamp,
                               allow_forward_handoff=True)
        g.note_sent(pair, stamp)
    assert reason == "cross_aligned_resume" and pair == (5, 5)
    pair, reason = g.limit((44, 44), feedback(10.15, 8, 8), 10.15,
                           allow_forward_handoff=True)
    assert reason == "cross_resume_ramp" and 5 < pair[0] <= 9


def test_single_wheel_wait_escalates_to_both_when_whole_car_reverses():
    g = WheelZeroCrossGuard()
    # Strict initial path could have been entered before the forward option
    # was available; discovering full reverse must widen the quiet check.
    g.limit((60, 0), feedback(10, -3, 2), 10)
    g.note_sent((0, 0), 10)
    for stamp, left, right in ((10.05, -2, -3), (10.1, 0, -3), (10.15, 0, -3)):
        pair, reason = g.limit((60, 0), feedback(stamp, left, right), stamp,
                               allow_forward_handoff=True)
        g.note_sent(pair, stamp)
        assert pair == (0, 0) and reason == "cross_wait_zero"
    assert g.pending_full_reverse and g.pending_wheels == (0, 1)


@pytest.mark.parametrize("enabled", [False, True])
def test_real_writer_scopes_handoff_to_configured_forward(monkeypatch, enabled):
    r, owner, driver, _, clock = visible_runtime(monkeypatch)
    r.config.follow_forward_handoff_enable = enabled
    r.get_steering_feedback = lambda: feedback(clock[0], 0, -3)
    with owner.motor_io_lock:
        r._send_follow_wheel_targets(24, -24, "FORWARD_TEST")
    assert driver.pairs == ([(24, -24)] if enabled else [(0, 0)])
    assert r._visible_wheel_waiting is not enabled


def test_missing_depth_cannot_use_forward_handoff(monkeypatch):
    r, owner, driver, _, clock = visible_runtime(monkeypatch)
    r.config.follow_forward_handoff_enable = True
    r.get_steering_feedback = lambda: feedback(clock[0], 20, 20)
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
    with owner.motor_io_lock:
        r._send_follow_wheel_targets(32, -16, "EXPIRED_TEST")
    assert driver.pairs == [(0, 0)]  # Only opposite-wheel yaw remains; still guarded.


def test_full_reverse_guard_survives_forward_handoff_option(monkeypatch):
    r, owner, driver, s, clock = visible_runtime(monkeypatch)
    r.config.follow_forward_handoff_enable = True
    r.get_steering_feedback = lambda: feedback(clock[0], -10, -10)
    assert r.needs_transition_stop(s.backward, s.forward)
    with owner.motor_io_lock:
        r._send_follow_wheel_targets(24, -24, "REVERSE_TEST")
    assert driver.pairs == [(0, 0)]
