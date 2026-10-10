"""Feedback-driven NORMAL exit: fake clock, cached feedback and fake motor only."""
import pytest

from car_control_modular.near_yaw_parking import ParkSettlingEvidence
from test_follow_wheel_periodic import setup_periodic
from test_near_yaw_park_execution import request_park
from test_search_handoff_execution import arm
from test_visible_wheel_continuity import feedback


@pytest.mark.parametrize('elapsed', [.06, .08, .099, .1, .2, .299, .3, .4, .499])
def test_completed_quiet_evidence_releases_current_without_fixed_dwell(elapsed):
    e = ParkSettlingEvidence(object(), 10., require_current_release=True)
    assert not e.current_release_ready(feedback(10.+elapsed-.05, 0, 0), 10.+elapsed-.05)
    assert e.current_release_ready(feedback(10.+elapsed, 0, 0), 10.+elapsed)
    # Readback/FREE is still mandatory: the quiet samples alone authorize
    # neither current-release completion nor a new motion command.
    assert not e.recovery_gate(10.+elapsed)
    assert e.reason == 'await_current_release'


def test_boundary_is_not_automatic_permission():
    e = ParkSettlingEvidence(object(), 10., require_current_release=True)
    assert e.current_release_ready(None, 10.50)
    assert not e.release_ready(10.50, feedback(10.50, 4, -4), 10.50)
    assert not e.motion_handoff_ready(10.50, feedback(10.50, 0, 0), 10.50)
    e.mark_current_released(10.50)
    assert not e.motion_handoff_ready(9.99, feedback(10.50, 0, 0), 10.50)
    assert not e.motion_handoff_ready(10.50, None, 10.50)
    assert e.motion_handoff_ready(10.52, feedback(10.52, 0, 0), 10.52)


@pytest.mark.parametrize('search', [False, True])
def test_both_park_paths_release_on_quiet_measured_after_completed_write(monkeypatch, search):
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
    for t in (10.02, 10.03, 10.05):
        clock[0] = t
        rt.get_steering_feedback = lambda: feedback(clock[0], 0, 0)
        rt._service_follow_wheels()
        if search:
            assert rt.search_reacquire_brake_pending(capture_timestamp=t)
        else:
            assert not rt.near_yaw_park_motion_ready(request, t, t)
            assert not rt.near_yaw_park_release_ready(request, t, t)
    assert driver.pairs == before and driver.stops == [1, 0]
    clock[0] = 10.10
    rt._service_follow_wheels()
    assert evidence.current_released_at == pytest.approx(10.13)
    assert driver.stops == [1, 0, 2]
    assert owner._brake_hold_active
    # Clearing 5A/FREE does not reuse the pre-release quiet samples. New
    # encoder samples and the existing image gate finish normal settling.
    for t in (10.14, 10.19):
        clock[0] = t
        rt._service_follow_wheels()
    clock[0] = 10.20
    if search:
        assert not rt.search_reacquire_brake_pending(capture_timestamp=10.20)
    else:
        assert rt.near_yaw_park_release_ready(request, 10.20, 10.20)
    assert driver.pairs == before  # independent 0A release must not write speed


def test_emergency_not_delayed_by_feedback_settling(monkeypatch):
    rt, owner, driver, _, clock, _ = setup_periodic(monkeypatch)
    request_park(owner, clock)
    rt._service_follow_wheels()
    clock[0] += .01
    assert rt.send_percent_brake('emergency', 'safety_hold_front_ir')
    assert driver.stops[-1] == 1
