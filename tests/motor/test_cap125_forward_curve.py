"""CAP125-226: obsolete in-place turn must not stop a safe forward curve."""
import logging

import pytest

from car_control_modular.wheel_zero_cross import WheelZeroCrossGuard
from test_visible_wheel_continuity import feedback, visible_runtime


@pytest.mark.parametrize("right", [False, True])
@pytest.mark.parametrize("pair", [(14, 22), (24, 40), (57, 71)])
def test_current_forward_pair_supersedes_old_turn(right, pair):
    turn = (6, -6) if right else (-6, 6)
    if right:
        pair = pair[::-1]
    guard = WheelZeroCrossGuard()
    guard.note_sent((20, 20), 9.98)
    assert guard.limit(turn, feedback(10., 12, 17), 10.)[0] == (0, 0)
    guard.note_sent((0, 0), 10.)
    applied, reason = guard.limit(pair, feedback(10.05, 11, 12), 10.05,
                                  allow_forward_handoff=True)
    assert applied == pair
    assert reason == "cross_obsolete_turn_forward_handoff"
    assert guard.pending_signs is None and guard.resume_signs is None
    assert not guard.pending_full_reverse
    guard.note_sent(applied, 10.05)
    assert guard.limit((10, 12), feedback(10.10, 12, 17), 10.10,
                       allow_forward_handoff=True)[0] == (10, 12)


@pytest.mark.parametrize("invalid", ["reverse_inner", "stale", "future", "duplicate", "untrusted", "nan", "opt_out"])
def test_obsolete_turn_release_requires_current_safe_feedback(invalid):
    guard = WheelZeroCrossGuard()
    guard.limit((-6, 6), feedback(10., 12, 17), 10.)
    fb = feedback(10.05, 11, 12)
    if invalid == "reverse_inner": fb.left_forward_rpm = -4
    if invalid == "stale": fb.timestamp = 9.
    if invalid == "future": fb.timestamp = 11.
    if invalid == "duplicate": fb.timestamp = 10.
    if invalid == "untrusted": fb.trustworthy = False
    if invalid == "nan": fb.left_forward_rpm = float("nan")
    assert guard.limit((14, 22), fb, 10.05,
                       allow_forward_handoff=invalid != "opt_out")[0] == (0, 0)


def test_new_true_reversal_still_waits_after_forward_release():
    guard = WheelZeroCrossGuard()
    guard.limit((-6, 6), feedback(10., 12, 17), 10.)
    pair, _ = guard.limit((14, 22), feedback(10.05, 11, 12), 10.05,
                          allow_forward_handoff=True)
    guard.note_sent(pair, 10.05)
    assert guard.limit((-6, 6), feedback(10.10, 14, 22), 10.10)[0] == (0, 0)
    assert guard.limit((-6, 6), feedback(10.15, 0, 0), 10.15)[0] == (0, 0)
    assert guard.limit((-6, 6), feedback(10.20, 0, 0), 10.20)[0] == (-6, 6)


@pytest.mark.parametrize("withdrawal", ["none", "expired", "at_write", "uid_change"])
def test_real_writer_validates_forward_handoff_and_logs_differential(monkeypatch, caplog, withdrawal):
    r, owner, driver, _, clock = visible_runtime(monkeypatch)
    r.config.follow_forward_handoff_enable = True
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
    r.get_steering_feedback = lambda: feedback(clock[0], 12, 17)
    r._send_follow_wheel_targets(-6, -6, "TEST")
    assert driver.pairs[-1] == (0, 0)
    clock[0] = 10.05
    if withdrawal != "expired":
        owner._fresh_depth_linear_snapshot = lambda uid, now=None: ("forward", 50)
    def read_feedback():
        if withdrawal == "at_write":
            owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
        if withdrawal == "uid_change":
            owner._follow_controller.active_target_id = 2
        return feedback(clock[0], 11, 12)
    r.get_steering_feedback = read_feedback
    with caplog.at_level(logging.INFO):
        r._send_follow_wheel_targets(14, -22, "TEST")
    assert driver.pairs[-1] == ((14, -22) if withdrawal == "none" else (0, 0))
    if withdrawal == "none":
        assert "obsolete_turn_handoff=True" in caplog.text
        assert "requested_diff_rpm=-8.0 applied_diff_rpm=-8.0 feedback_diff_rpm=-1" in caplog.text
        assert "feedback_after_previous_send_ms=50.0" in caplog.text
