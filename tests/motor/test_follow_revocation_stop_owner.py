"""A completed STOP owns the motor mode after periodic follow exits."""

from test_follow_wheel_periodic import setup_periodic


def test_hazard_stop_is_not_replaced_by_stale_follow_speed_zero(monkeypatch):
    runtime, owner, driver, _, clock, _ = setup_periodic(monkeypatch)
    runtime._service_follow_wheels()
    assert driver.pairs == [(24, -24)]

    clock[0] += .01
    owner._explicit_stop_requested = True
    owner.stop_action_execution = True
    owner._brake_hold_active = True
    runtime.send_percent_brake(mode="emergency", label="safety_hazard")
    assert driver.stops == [1]
    assert not runtime.backend.normal_zero_hold  # The backend cannot mask a stale zero.

    runtime._service_follow_wheels()
    assert driver.pairs == [(24, -24)]
    assert driver.stops == [1]
    assert runtime._follow_wheel_clock.last_axes is None


def test_unclaimed_old_follow_pair_still_gets_revocation_zero(monkeypatch):
    runtime, owner, driver, _, clock, _ = setup_periodic(monkeypatch)
    runtime._service_follow_wheels()
    clock[0] += .01
    owner.search_state = "searching"

    runtime._service_follow_wheels()
    assert driver.pairs == [(24, -24), (0, 0)]
    assert not driver.stops

