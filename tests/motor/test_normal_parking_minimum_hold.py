"""500ms NORMAL dwell: fake clock, cached feedback and fake motor only."""
import pytest

from car_control_modular.near_yaw_parking import ParkSettlingEvidence
from test_follow_wheel_periodic import setup_periodic
from test_near_yaw_park_execution import request_park
from test_search_handoff_execution import arm
from test_visible_wheel_continuity import feedback


@pytest.mark.parametrize('elapsed', [.0, .04, .08, .099, .1, .2, .299, .3, .4, .499])
@pytest.mark.parametrize('motion', [False, True])
def test_quiet_or_motion_cannot_release_before_500ms(elapsed, motion):
    e = ParkSettlingEvidence(object(), 10.)
    e.observe(feedback(10.01, 0, 0), 10.01)
    e.observe(feedback(10.06, 0, 0), 10.06)
    method = e.motion_handoff_ready if motion else e.release_ready
    assert not method(10.+elapsed, feedback(10.+elapsed, 0, 0), 10.+elapsed)
    assert e.reason == 'minimum_normal_hold_500ms'


def test_boundary_is_not_automatic_permission():
    e = ParkSettlingEvidence(object(), 10.)
    assert not e.release_ready(10.50, feedback(10.50, 4, -4), 10.50)
    assert not e.motion_handoff_ready(9.99, feedback(10.50, 0, 0), 10.50)
    assert not e.motion_handoff_ready(10.50, None, 10.50)
    assert e.motion_handoff_ready(10.50, feedback(10.50, 0, 0), 10.50)


@pytest.mark.parametrize('search', [False, True])
def test_both_park_paths_hold_from_completed_write(monkeypatch, search):
    rt, owner, driver, _, clock, _ = setup_periodic(monkeypatch)
    request = arm(rt, owner, clock) if search else request_park(owner, clock)
    original = rt.backend.send_stop
    def delayed_stop(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] += .03
        return result
    rt.backend.send_stop = delayed_stop
    rt._service_follow_wheels()
    evidence = rt._search_reacquire_settling if search else rt._near_yaw_park_settling
    assert evidence.sent_at == pytest.approx(10.03)
    before = list(driver.pairs)
    for t in (10.05, 10.10, 10.12, 10.40, 10.50, 10.529):
        clock[0] = t
        rt.get_steering_feedback = lambda: feedback(clock[0], 0, 0)
        rt._service_follow_wheels()
        if search:
            assert rt.search_reacquire_brake_pending(capture_timestamp=t)
        else:
            assert not rt.near_yaw_park_motion_ready(request, t, t)
            assert not rt.near_yaw_park_release_ready(request, t, t)
    assert driver.pairs == before and driver.stops == [1, 0]
    for t in (10.54, 10.58, 10.63):
        clock[0] = t
        rt._service_follow_wheels()
    clock[0] = 10.64
    if search:
        assert not rt.search_reacquire_brake_pending(capture_timestamp=10.64)
    else:
        assert rt.near_yaw_park_release_ready(request, 10.64, 10.64)
    assert driver.pairs == before  # independent 0A release must not write speed


def test_emergency_not_delayed_by_500ms(monkeypatch):
    rt, owner, driver, _, clock, _ = setup_periodic(monkeypatch)
    request_park(owner, clock)
    rt._service_follow_wheels()
    clock[0] += .01
    assert rt.send_percent_brake('emergency', 'safety_hold_front_ir')
    assert driver.stops[-1] == 1
