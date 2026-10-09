"""Actual motor runtime with fake feedback/serial: no real hardware."""
from dataclasses import replace
import pytest
from test_search_handoff_execution import arm, quiet
from test_follow_wheel_periodic import setup_periodic


@pytest.mark.parametrize("rpm", [-5,-2,2,5])
def test_cap159_residual_motion_cannot_release_search_brake(monkeypatch, rpm):
    rt, owner, driver, _, clock, _ = setup_periodic(monkeypatch)
    arm(rt, owner, clock)
    rt._service_follow_wheels()
    for t in (10.05,10.10,10.15):
        clock[0] = t
        rt.get_steering_feedback = lambda: replace(quiet(clock[0]), left_forward_rpm=rpm)
        assert rt.search_reacquire_brake_pending(capture_timestamp=t)
    assert not rt.can_release_brake_hold(rt.symbols.rotate_left)
    assert driver.stops == [1, 0]


@pytest.mark.parametrize("kind", ["no_image", "pre_stop", "pre_quiet", "future", "stale", "center"])
def test_quiet_wheels_are_not_sufficient_for_motion(monkeypatch, kind):
    rt, owner, _, _, clock, _ = setup_periodic(monkeypatch)
    arm(rt, owner, clock); rt._service_follow_wheels()
    for t in (10.45,10.50):
        clock[0] = t
        rt.get_steering_feedback = lambda: quiet(clock[0])
        assert rt.search_reacquire_brake_pending()
    stamp = dict(no_image=None,pre_stop=9.99,pre_quiet=10.46,future=10.60,stale=10.50,center=10.50)[kind]
    if kind == "stale":
        # Continue fresh quiet feedback; only the image expires.
        for t in (10.60,10.70,10.80):
            clock[0] = t
            rt.search_reacquire_brake_pending()
    assert rt.search_reacquire_brake_pending(capture_timestamp=stamp,keep_observing=kind=="center")
    assert not rt.can_release_brake_hold(rt.symbols.rotate_left)


def test_minimum_quiet_span_not_just_two_rapid_polls(monkeypatch):
    rt, owner, _, _, clock, _ = setup_periodic(monkeypatch)
    arm(rt,owner,clock);rt._service_follow_wheels()
    clock[0] = 10.50
    rt._service_follow_wheels()  # 0A first, then collect independent quiet samples
    for t in (10.51,10.52):
        clock[0]=t
        rt.get_steering_feedback=lambda:quiet(clock[0])
        assert rt.search_reacquire_brake_pending(capture_timestamp=t)
    clock[0]=10.56
    assert not rt.search_reacquire_brake_pending(capture_timestamp=clock[0])


def test_motor_tick_collects_quiet_without_waiting_two_vision_results(monkeypatch):
    rt,owner,driver,_,clock,_=setup_periodic(monkeypatch)
    arm(rt,owner,clock);rt._service_follow_wheels()
    for t in (10.50,10.51,10.56):
        clock[0]=t
        rt.get_steering_feedback=lambda:quiet(clock[0])
        rt._service_follow_wheels()
    assert rt._search_reacquire_settling.ready_at == 10.56
    assert rt.search_reacquire_brake_pending()  # encoder-only cannot unlock
    clock[0]=10.58
    assert not rt.search_reacquire_brake_pending(capture_timestamp=10.57)
    assert driver.stops == [1, 0, 2] and driver.pairs == [(0,0)]  # only NORMAL pre-zero
