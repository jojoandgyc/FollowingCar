"""CAP437 reversal then fresh forward release; fake driver only."""
import pytest

from car_control_modular.wheel_zero_cross import WheelZeroCrossGuard
from test_visible_wheel_continuity import feedback, visible_runtime


@pytest.mark.parametrize('requested', [(98, 98), (101, 107), (180, 180)])
def test_two_aligned_samples_release_current_not_old_request(requested):
    g = WheelZeroCrossGuard()
    for t, wheels in [(10., (-13, -5)), (10.05, (-19, -11)),
                      (10.10, (0, -29)), (10.15, (-1, 0)),
                      (10.20, (-1, 8)), (10.25, (1, 9))]:
        pair, _ = g.limit((26, 26), feedback(t, *wheels), t, allow_forward_handoff=True)
        assert pair == (0, 0)
        g.note_sent(pair, t)
    pair, reason = g.limit(requested, feedback(10.30, 5, 0), 10.30,
                           allow_forward_handoff=True)
    assert pair == requested and reason == 'cross_confirmed_forward_handoff'
    assert not g.pending_full_reverse and g.resume_signs is None
    g.note_sent(pair, 10.30)
    # A newer SMALLER cap is applied immediately; never replay98/180.
    assert g.limit((16, 16), feedback(10.35, 8, 8), 10.35,
                   allow_forward_handoff=True)[0] == (16, 16)


@pytest.mark.parametrize('case', ['duplicate', 'stale', 'future', 'untrusted', 'reverse'])
def test_bad_second_confirmation_cannot_release(case):
    g = WheelZeroCrossGuard()
    g.limit((98, 98), feedback(10., -10, -10), 10., allow_forward_handoff=True)
    g.limit((98, 98), feedback(10.05, 5, 5), 10.05, allow_forward_handoff=True)
    fb = feedback(10.10, 5, 5)
    if case == 'duplicate': fb.timestamp = 10.05
    if case == 'stale': fb.timestamp = 9.
    if case == 'future': fb.timestamp = 11.
    if case == 'untrusted': fb.trustworthy = False
    if case == 'reverse': fb.left_forward_rpm = -4.
    assert g.limit((98, 98), fb, 10.10, allow_forward_handoff=True)[0] == (0, 0)


def test_opt_out_keeps_existing_ramp():
    g = WheelZeroCrossGuard()
    for t, speed in [(10., -10), (10.05, 5), (10.10, 5)]:
        pair, reason = g.limit((98, 98), feedback(t, speed, speed), t)
        g.note_sent(pair, t)
    assert pair == (5, 5) and reason == 'cross_aligned_resume'
    assert g.limit((98, 98), feedback(10.15, 5, 5), 10.15)[1] == 'cross_resume_ramp'


@pytest.mark.parametrize('withdrawal', ['none', 'expired_before_confirm', 'expires_during_write'])
def test_actual_writer_rechecks_depth_after_confirm(monkeypatch, withdrawal):
    r, owner, driver, _, clock = visible_runtime(monkeypatch)
    r.config.follow_forward_handoff_enable = True
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: ('forward', 98)
    for t, wheels in [(10., (-13, -5)), (10.05, (1, 9))]:
        clock[0] = t
        r.get_steering_feedback = lambda: feedback(clock[0], *wheels)
        r._send_follow_wheel_targets(98, -98, 'TEST')
        assert driver.pairs[-1] == (0, 0)
    clock[0] = 10.10
    if withdrawal == 'expired_before_confirm':
        owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
    def final_feedback():
        if withdrawal == 'expires_during_write':
            owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
        return feedback(clock[0], 5, 0)
    r.get_steering_feedback = final_feedback
    r._send_follow_wheel_targets(98, -98, 'TEST')
    assert driver.pairs[-1] == ((98, -98) if withdrawal == 'none' else (0, 0))
