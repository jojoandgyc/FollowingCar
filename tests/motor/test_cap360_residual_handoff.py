"""CAP360-363 replay: fake wheel feedback/driver, never motor hardware."""
import pytest

from car_control_modular.wheel_zero_cross import WheelZeroCrossGuard
from test_visible_wheel_continuity import feedback, visible_runtime


def seed():
    guard = WheelZeroCrossGuard()
    assert guard.limit((-6, 6), feedback(10, -9, -2), 10)[0] == (0, 0)
    guard.note_sent((0, 0), 10)
    return guard


def step(g, t, pair, measured, **kwargs):
    result = g.limit(pair, feedback(t, *measured), t,
                     allow_forward_handoff=True, residual_reverse_max_rpm=8, **kwargs)
    g.note_sent(result[0], t)
    return result


def test_cap360_current_forward_replaces_old_reverse_wait_without_launch_ramp():
    g = seed()
    assert step(g, 10.05, (24, 44), (-6, -5))[0] == (0, 0)
    assert step(g, 10.10, (73, 91), (-3, -1)) == (
        (73, 91), "cross_bounded_residual_forward_handoff")
    assert step(g, 10.15, (52, 72), (0, 2))[0] == (52, 72)
    assert g.pending_signs is None and g.resume_signs is None


def test_duplicate_feedback_does_not_confirm_twice():
    g = seed()
    f = feedback(10.05, -6, -5)
    for t in [10.05, 10.06, 10.07]:
        pair, _ = g.limit((24, 44), f, t, allow_forward_handoff=True, residual_reverse_max_rpm=8)
        assert pair == (0, 0)


@pytest.mark.parametrize("measured", [(-9, -2), (-6, -7), (-8.01, -1)])
def test_large_or_worsening_reverse_still_waits(measured):
    g = seed()
    step(g, 10.05, (24, 44), (-6, -5))
    assert step(g, 10.10, (73, 91), measured)[0] == (0, 0)


def test_commanded_backward_survives_zero_and_does_not_use_residual_exception():
    g = WheelZeroCrossGuard()
    g.note_sent((-30, -30), 9.9)
    g.limit((0, 0), None, 10)
    g.note_sent((0, 0), 10)
    for t, speeds in [(10.05, (-6, -5)), (10.10, (-3, -2))]:
        assert step(g, t, (24, 44), speeds)[0] == (0, 0)


@pytest.mark.parametrize("pair", [(-6, 6), (6, -6), (-20, -20), (0, 44)])
def test_non_forward_requests_cannot_release(pair):
    g = seed()
    for t in [10.05, 10.10]:
        assert step(g, t, pair, (-6, -5))[1] != "cross_bounded_residual_forward_handoff"


def test_real_writer_and_depth_expiry(monkeypatch):
    r, owner, driver, _, clock = visible_runtime(monkeypatch)
    r.config.follow_forward_handoff_enable = True
    r.config.follow_residual_reverse_max_rpm = 8
    r._visible_wheel_uid = 1
    r._visible_wheel_guard = seed()
    for t, speeds in [(10.05, (-6, -5)), (10.10, (-3, -1))]:
        clock[0] = t
        r.get_steering_feedback = lambda: feedback(clock[0], *speeds)
        with owner.motor_io_lock:
            r._send_follow_wheel_targets(14, -34, "CAP360")
    assert driver.pairs == [(0, 0), (14, -34)]
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
    clock[0] = 10.15
    with owner.motor_io_lock:
        r._send_follow_wheel_targets(14, -34, "EXPIRED")
    assert sum((driver.pairs[-1][0], -driver.pairs[-1][1])) == 0


def test_released_residual_does_not_reenter_wait_on_next_feedback():
    g = seed()
    step(g, 10.05, (24, 44), (-6, -5))
    assert step(g, 10.10, (73, 91), (-4, -3))[0] == (73, 91)
    deadline = g.residual_forward_until
    for t in [10.15, 10.20]:
        assert step(g, t, (52, 72), (-4, -3))[0] == (52, 72)
        assert g.residual_forward_until == deadline
    assert step(g, 10.26, (52, 72), (-4, -3))[0] == (0, 0)


def test_residual_continuation_stops_when_reverse_worsens():
    g = seed()
    step(g, 10.05, (24, 44), (-6, -5))
    step(g, 10.10, (73, 91), (-4, -3))
    assert step(g, 10.15, (52, 72), (-5, -4))[0] == (0, 0)
