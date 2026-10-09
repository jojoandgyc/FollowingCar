"""CAP422-456: no hardware. Classify zeros and trim only common acceleration."""
import ast
from pathlib import Path

import pytest

from car_control_modular.turn_acceleration_priority import limit_common_acceleration
from car_control_modular.wheel_zero_cross import WheelZeroCrossGuard
from test_cap676_turn_continuity import runtime
from test_visible_wheel_continuity import feedback


@pytest.mark.parametrize('mirror', [False, True])
@pytest.mark.parametrize('requested,measured,expected', [
    ((56,40), (16,15), (33,17)),
    ((68,56), (21,18), (32,20)),
    ((83,73), (49,51), (63,53)),
    ((70,46), (88,85), (70,46)),  # requested braking stays untouched
    ((41,11), (89,82), (41,11)),
    ((60,30), (48,22), (60,30)),  # differential established
    ((20,20), (0,0), (20,20)),
    ((15,-15), (87,68), (15,-15)),  # never authorize reversal
    ((0,0), (87,68), (0,0)),
])
def test_capture_examples_and_mirrored_left_turn(mirror, requested, measured, expected):
    if mirror:
        requested, measured, expected = requested[::-1], measured[::-1], expected[::-1]
    result, _ = limit_common_acceleration(requested, feedback(10,*measured), 10)
    assert result == expected
    if result != requested:
        assert result[0]-result[1] == requested[0]-requested[1]
        assert all(measured[i] <= result[i] <= requested[i] for i in range(2))
        assert min(result) > 0


@pytest.mark.parametrize('bad', ['missing','stale','future','nan','untrusted','reverse'])
def test_invalid_feedback_cannot_trim_or_authorize(bad):
    f = feedback(10,16,15)
    if bad == 'missing': f = None
    elif bad == 'stale': f.timestamp = 9
    elif bad == 'future': f.timestamp = 11
    elif bad == 'nan': f.left_forward_rpm = float('nan')
    elif bad == 'untrusted': f.trustworthy = False
    else: f.left_forward_rpm = -5
    assert limit_common_acceleration((56,40), f, 10)[0] == (56,40)


def test_real_writer_preserves_base_even_with_legacy_flag_and_safety_axes(monkeypatch, caplog):
    r,o,d,_,clock = runtime(monkeypatch)
    r.config.follow_turn_acceleration_priority_enable = True
    o._fresh_depth_linear_snapshot = lambda uid, now=None: ('forward',24)
    r.get_steering_feedback = lambda: feedback(clock[0],16,15)
    with caplog.at_level('INFO'):
        r._send_follow_wheel_targets(56,-40,'FOLLOW20')
    assert d.pairs[-1] == (56,-40)
    assert 'turn_build_common_acceleration_limited' not in caplog.text
    assert 'common_accel_removed_rpm=0.0' in caplog.text
    assert 'zero_owner=none' in caplog.text
    clock[0] += .05
    r.get_steering_feedback = lambda: feedback(clock[0],31,17)
    r._send_follow_wheel_targets(56,-40,'FOLLOW20')
    assert d.pairs[-1] == (56,-40)
    o._has_fresh_lateral_yaw = lambda uid: False
    r._send_follow_wheel_targets(56,-40,'FOLLOW20')
    assert d.pairs[-1] == (48,-48)  # loss of yaw does not manufacture a turn
    o._fresh_depth_linear_snapshot = lambda uid, now=None: None
    r._send_follow_wheel_targets(56,-40,'FOLLOW20')
    assert d.pairs[-1] == (0,0)
    assert not d.stops


def test_real_zero_owner_and_no_momentum_override(monkeypatch, caplog):
    r,o,d,_,clock = runtime(monkeypatch)
    r.config.follow_turn_acceleration_priority_enable = True
    r.get_steering_feedback = lambda: feedback(clock[0],87,68)
    r._send_follow_wheel_targets(41,-11,'FOLLOW20')
    o._fresh_depth_linear_snapshot = lambda uid, now=None: None
    clock[0] += .05
    with caplog.at_level('INFO'):
        r._send_follow_wheel_targets(15,15,'FOLLOW20')
    assert d.pairs[-1] == (0,0)
    assert 'zero_owner=forward_loss_handoff' in caplog.text
    assert 'forward_loss_input_rpm=(15, -15)' in caplog.text
    assert 'wheel_guard_input_rpm=(0, 0)' in caplog.text


@pytest.mark.parametrize('forward', [False, True])
def test_repeated_zero_writes_do_not_move_feedback_reference(forward):
    g = WheelZeroCrossGuard()
    req = (24,44) if forward else (10,-10)
    measured = (-3,-2) if forward else (3,2)
    kwargs = dict(allow_forward_handoff=True, residual_reverse_max_rpm=8) if forward else dict(
        allow_aligned_turn=True, residual_turn_max_rpm=4)
    # Feedback is new at every tick but 40ms behind a 25ms writer. Comparing
    # it with LAST zero write can never succeed; first zero remains 10.002.
    first_nonzero = None
    for i in range(12):
        now = 10+i*.025
        pair,reason = g.limit(req, feedback(now-.04,*measured), now, **kwargs)
        g.note_sent(pair, now+.002)
        if any(pair):
            first_nonzero = now
            break
        assert g.zero_since == pytest.approx(10.002)
    assert first_nonzero is not None and first_nonzero <= 10.10
    assert pair == req


def test_no_feedback_from_before_actual_zero_can_release():
    g = WheelZeroCrossGuard()
    g.note_sent((0,0),10.)
    for now in (10.01,10.04,10.08):
        pair,_ = g.limit((10,-10), feedback(9.99,3,2), now,
            allow_aligned_turn=True, residual_turn_max_rpm=4)
        g.note_sent(pair, now)
        assert pair == (0,0)


def test_nonzero_write_retires_zero_reference():
    g = WheelZeroCrossGuard()
    g.note_sent((0,0),10.)
    g.note_sent((20,30),10.1)
    assert g.zero_since is None
    g.note_sent((0,0),10.2)
    assert g.zero_since == 10.2


def test_actual_default_config_and_entrypoint(monkeypatch):
    from car_control_modular.config_loader import load_config_to_env
    import os
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(os, 'environ', {})
    load_config_to_env(str(root/'car_control_modular/config/reid_runtime.ini'))
    assert os.environ['FOLLOW_TURN_ACCELERATION_PRIORITY_ENABLE'] == '0'
    tree = ast.parse((root/'request_0513_modular.py').read_text())
    values = [k.value for n in ast.walk(tree) if isinstance(n,ast.Call)
              for k in n.keywords if k.arg == 'follow_turn_acceleration_priority_enable']
    assert len(values) == 1
    assert eval(compile(ast.Expression(values[0]),'binding','eval'), {'os':os}) is False
