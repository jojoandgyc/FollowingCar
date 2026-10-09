"""Real wheel dispatch, fake clock/driver: no serial or hardware access."""
import pytest

from car_control_modular.forward_loss_handoff import ForwardLossHandoff
from car_control_modular.wheel_zero_cross import WheelZeroCrossGuard
from test_visible_wheel_continuity import visible_runtime, feedback


def runtime(monkeypatch):
    r, o, d, s, clock = visible_runtime(monkeypatch)
    r.config.follow_turn_residual_max_rpm = 4
    r.config.motor_forward_max_target_rpm = 200
    r.config.follow_forward_loss_handoff_enable = True
    r.config.follow_cross_brake_enable = True
    r.config.follow_forward_handoff_enable = True
    return r, o, d, s, clock


@pytest.mark.parametrize('sign', [-1, 1])
def test_base_falls_below_yaw_without_creating_reverse(monkeypatch, sign):
    r, o, d, _, clock = runtime(monkeypatch)
    r.get_steering_feedback = lambda: feedback(clock[0], 13, 10)
    o._fresh_depth_linear_snapshot = lambda uid, now=None: ('forward', 4)
    # base=8, yaw=+/-15; motor raw right sign is reversed.
    r._send_follow_wheel_targets(8+sign*15, -(8-sign*15), 'FOLLOW20')
    expected = (0, -16) if sign < 0 else (16, 0)
    assert d.pairs[-1] == expected
    assert r._visible_wheel_guard.pending_signs is None
    assert not d.stops


def test_depth_loss_does_not_make_unapproved_arc(monkeypatch):
    r, o, d, _, clock = runtime(monkeypatch)
    r.get_steering_feedback = lambda: feedback(clock[0], 13, 10)
    r._send_follow_wheel_targets(3, -33, 'FOLLOW20')
    o._fresh_depth_linear_snapshot = lambda uid, now=None: None
    r._send_follow_wheel_targets(-7, -23, 'FOLLOW20')
    assert d.pairs[-1] == (0, 0)


def test_small_residual_releases_then_does_not_rewait_during_response(monkeypatch):
    r, o, d, _, clock = runtime(monkeypatch)
    r.get_steering_feedback = lambda: feedback(clock[0], 13, 10)
    r._send_follow_wheel_targets(3, -33, 'FOLLOW20')
    o._fresh_depth_linear_snapshot = lambda uid, now=None: None
    for t in (10.05, 10.10, 10.15):
        clock[0] = t
        r.get_steering_feedback = lambda: feedback(clock[0], 3, 2)
        r._send_follow_wheel_targets(-15, -15, 'FOLLOW20')
    assert d.pairs[-1] == (-15, -15)
    deadline = r._visible_wheel_guard.residual_turn_until
    for t in (10.20, 10.30, 10.45, 10.60):
        clock[0] = t
        r._send_follow_wheel_targets(-15, -15, 'FOLLOW20')
        assert d.pairs[-1] == (-15, -15)
        assert r._visible_wheel_guard.residual_turn_until == deadline
    # 0.5s is a response window, never permission to ignore expired yaw.
    o._has_fresh_lateral_yaw = lambda uid: False
    clock[0] = 10.61
    r._send_follow_wheel_targets(-15, -15, 'FOLLOW20')
    assert d.pairs[-1] == (0, 0)


@pytest.mark.parametrize('bad', ['duplicate', 'stale', 'opposing', 'direction', 'explicit_zero'])
def test_residual_turn_cannot_bypass_evidence(bad):
    g = WheelZeroCrossGuard()
    g.note_sent((0, 0), 9.99)
    for t in (10, 10.05):
        pair, _ = g.limit((-15, 15), feedback(t, 3, 2), t,
            allow_aligned_turn=True, residual_turn_max_rpm=4)
        assert pair == (0, 0)
        g.note_sent(pair, t)
    now, stamp, left, req = 10.1, 10.1, 3, (-15, 15)
    if bad == 'duplicate': stamp = 10.05
    if bad == 'stale': stamp = 9
    if bad == 'opposing': left = 12
    if bad == 'direction': req = (15, -15)
    if bad == 'explicit_zero': req = (0, 0)
    pair, _ = g.limit(req, feedback(stamp, left, 2), now,
        allow_aligned_turn=True, residual_turn_max_rpm=4)
    assert pair == (0, 0)


def test_unsent_forward_request_does_not_reset_handoff():
    h = ForwardLossHandoff()
    h.note_sent((3, 33))
    h.limit((-15, 15), feedback(10, 13, 10), 10)
    h.limit((5, 31), feedback(10.05, -7, -4), 10.05)
    h.note_sent((0, 0))  # actual wheel guard rejected the forward request
    assert h.started == 10
    h.note_sent((5, 31))
    assert h.started is None


@pytest.mark.parametrize('reason', ['stop_signal', 'queued_action_stop_signal'])
def test_visible_turn_switch_does_not_insert_normal_or_zero(monkeypatch, reason):
    r, o, d, _, _ = runtime(monkeypatch)
    o._vision_control_state = 'target_visible_low_quality'
    o._last_action_queue_reason = 'lateral_intent_30hz:target_visible_low_quality_yaw'
    r.send_stop_with_brake_hold(reason)
    assert not d.stops and not d.pairs
    assert not getattr(o, '_brake_hold_active', False)


@pytest.mark.parametrize('reason', ['front_ir', 'explicit_stop', 'hard_stop'])
def test_danger_and_explicit_stop_still_interrupt(monkeypatch, reason):
    r, o, d, _, _ = runtime(monkeypatch)
    o._last_action_queue_reason = 'lateral_intent_30hz:target_visible_low_quality_yaw'
    r.send_stop_with_brake_hold(reason)
    assert d.stops


def test_plain_stop_signal_without_motion_provenance_not_ignored(monkeypatch):
    r, o, d, _, _ = runtime(monkeypatch)
    o._last_action_queue_reason = 'explicit_stop'
    r.send_stop_with_brake_hold('stop_signal')
    assert d.stops


@pytest.mark.parametrize('check_raises', [False, True])
def test_generic_interrupt_rechecks_new_danger(monkeypatch, check_raises):
    r, o, d, _, _ = runtime(monkeypatch)
    o._last_action_queue_reason = 'lateral_intent_30hz:target_visible_low_quality_yaw'
    def check(_):
        if check_raises:
            raise RuntimeError('feedback unavailable')
        return True
    r.hard_stop_check = check
    r.send_stop_with_brake_hold('stop_signal')
    assert d.stops
    assert o._last_stop_command_reason == 'hard_stop'


@pytest.mark.parametrize('cause', ['feedback_stale', 'strong_reverse', 'expired_response'])
def test_response_window_does_not_mask_new_invalid_motion(cause):
    g = WheelZeroCrossGuard()
    g.note_sent((0, 0), 9.99)
    for t in (10, 10.05, 10.10):
        pair, _ = g.limit((-15, 15), feedback(t, 3, 2), t,
            allow_aligned_turn=True, residual_turn_max_rpm=4)
        g.note_sent(pair, t)
    assert pair == (-15, 15)
    now = 10.7 if cause == 'expired_response' else 10.15
    stamp = 9 if cause == 'feedback_stale' else now
    left = 20 if cause == 'strong_reverse' else 3
    pair, _ = g.limit((-15, 15), feedback(stamp, left, 2), now,
        allow_aligned_turn=True, residual_turn_max_rpm=4)
    assert pair == (0, 0)


def test_periodic_repeated_live_depth_updates_keep_turn_for_half_second(monkeypatch):
    from test_follow_wheel_periodic import setup_periodic
    r, o, d, _, clock, state = setup_periodic(monkeypatch)
    r.config.follow_turn_residual_max_rpm = 4
    r.config.follow_forward_loss_handoff_enable = True
    r.config.follow_forward_handoff_enable = True
    r.config.follow_cross_brake_enable = True
    state[:] = [8., -15., 10.25, 10.22]
    for i in range(12):
        clock[0] = 10 + i*.06
        # These represent NEW approved observations, not renewed old leases.
        state[2:] = [clock[0]+.25, clock[0]+.22]
        r.get_steering_feedback = lambda: feedback(clock[0], 13, 10)
        r._service_follow_wheels()
        assert d.pairs[-1] == (0, -16)
    assert len(d.pairs) >= 10
    assert not d.stops
    state[2] = state[3] = clock[0]+.01
    clock[0] += .011
    r._service_follow_wheels()
    assert d.pairs[-1] == (0, 0)
