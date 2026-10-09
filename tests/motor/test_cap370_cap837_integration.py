"""Both completed changes together: real executor, fake clock/serial only."""
import pytest

from test_cap837_turn_buildup import setup, image_intent
from test_near_yaw_park_execution import request_park
from test_visible_wheel_continuity import feedback


def resume_and_build(monkeypatch, sign=1):
    r, o, d, _, clock = setup(monkeypatch)
    request = request_park(o, clock)
    r._service_follow_wheels()
    clock[0] = 10.10
    assert r.request_near_yaw_forward_resume(request, 10.08, 10.09, clock[0])
    r._service_follow_wheels()
    assert r._near_yaw_park_settling.current_released_at == 10.10
    clock[0] = 10.12
    r.get_steering_feedback = lambda: feedback(clock[0], 19, 19)
    assert r.near_yaw_park_motion_ready(request, 10.08, clock[0])
    o._near_yaw_park_request = None
    o._brake_hold_active = False
    o._brake_hold_label = 'brake'
    for n, t in enumerate((10.12, 10.23, 10.28)):
        clock[0] = t
        o._lateral_intent_store.publish(image_intent(t, 374+n, sign,
                                                    capture_timestamp=t-.02))
        r._send_follow_wheel_targets(82+sign*10, -(82-sign*10), 'FOLLOW20')
    assert d.pairs[-3:] == [(82+sign*10, -(82-sign*10))]*3
    assert r._turn_buildup.started == 10.28
    return r, o, d, clock


@pytest.mark.parametrize('sign', [-1, 1])
def test_early_resume_keeps_current_forward_grant_through_bounded_turn(monkeypatch, sign):
    r, o, d, clock = resume_and_build(monkeypatch, sign)
    stops = list(d.stops)
    for n in range(1, 10):
        # Cross the boundary explicitly (binary 10.63-10.28 is < .35).
        clock[0] = 10.28+n*.05+1e-6
        o._lateral_intent_store.publish(image_intent(clock[0], 377+n, sign,
                                                    capture_timestamp=clock[0]-.02))
        r._send_follow_wheel_targets(82+sign*10, -(82-sign*10), 'FOLLOW20')
        mean = (d.pairs[-1][0]-d.pairs[-1][1])/2
        assert mean == 82
        assert r._turn_buildup.started == 10.28
        assert r._turn_buildup.closed == (n >= 7)
    assert d.stops == stops  # no second parking episode or current bounce


@pytest.mark.parametrize('change', ['lower_forward', 'expired_depth', 'identity', 'danger'])
def test_resumed_turn_never_restores_old_authority(monkeypatch, change):
    r, o, d, clock = resume_and_build(monkeypatch)
    clock[0] = 10.34
    o._lateral_intent_store.publish(image_intent(clock[0], 380, capture_timestamp=clock[0]-.02))
    if change == 'lower_forward': o._fresh_depth_linear_snapshot = lambda uid, now=None: ('forward', 15)
    if change == 'expired_depth': o._fresh_depth_linear_snapshot = lambda uid, now=None: None
    if change == 'identity': o._vision_control_state = 'lost_confirming'
    if change == 'danger': r.hard_stop_check = lambda _: True
    before = len(d.pairs)
    r._send_follow_wheel_targets(92, -72, 'FOLLOW20', visible_required=True)
    if change == 'lower_forward':
        assert len(d.pairs) > before
        assert (d.pairs[-1][0]-d.pairs[-1][1])/2 <= 30
    else:
        assert all((left-right)/2 <= 0 for left, right in d.pairs[before:])


def test_guard_time_danger_preempts_unchanged_additive_pair(monkeypatch):
    r, o, d, clock = resume_and_build(monkeypatch)
    clock[0] = 10.34
    o._lateral_intent_store.publish(image_intent(clock[0], 380,
                                                capture_timestamp=clock[0]-.02))
    original = r._visible_wheel_guard.limit

    def danger_during_guard(*args, **kwargs):
        result = original(*args, **kwargs)
        r.hard_stop_check = lambda _: True
        return result

    r._visible_wheel_guard.limit = danger_during_guard
    before = len(d.pairs)
    stops_before = len(d.stops)
    assert r._send_follow_wheel_targets(92, -72, 'FOLLOW20') is False
    assert len(d.pairs) == before
    assert len(d.stops) > stops_before
