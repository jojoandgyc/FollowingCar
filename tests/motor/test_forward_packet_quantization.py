"""A fractional forward cap is not a revocation after integer wheel mixing."""
import math
from dataclasses import replace

import pytest

from test_unified_forward_snapshot import feedback, inject, writer


@pytest.mark.parametrize("yaw,expected", [(5., (15, -5)), (-5., (5, -15))])
def test_ten_rpm_forward_and_ten_rpm_wheel_difference_are_one_packet(
        monkeypatch, yaw, expected):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch)
    publish(10., yaw)
    # This low-speed case has independently measured slow wheels, rather
    # than the generic fixture's 24 RPM travel/feedback budget.
    timing = owner._depth30_linear_timing
    timing = replace(timing, braking_assessment=replace(timing.braking_assessment,
        travel_bound_rpm=4., outer_rpm=4.))
    owner._depth30_linear_timing = owner._depth30_prepared_timing = timing
    rt._steering_feedback = feedback(clock[0], 4., 4.)

    rt._service_follow_wheels()

    assert len(driver.pairs) == 2 and driver.pairs[-1] == expected
    assert not driver.stops


@pytest.mark.parametrize("base", [16.1, 24.49, 24.5, 24.6, 60.8, 94.6, 99.9])
@pytest.mark.parametrize("yaw", [-9.8, -6., 0., 5., 9.8])
def test_fractional_forward_cap_never_becomes_zero_from_rounding(monkeypatch, base, yaw):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch)
    publish(base, yaw)
    grant = owner._depth30_linear_timing

    rt._service_follow_wheels()

    left, raw_right = driver.pairs[-1]
    right = -raw_right
    assert len(driver.pairs) == 2 and left >= 0 and right >= 0
    assert 0 < .5*(left+right) <= base
    assert max(left, right) <= 100
    assert abs(left-right) <= 20  # Existing maximum single-wheel correction=10.
    assert (left-right)*yaw >= 0
    assert owner._depth30_linear_timing is grant
    assert rt._forward_execution_anchor.sample_timestamp == grant.snapshot[3]
    assert not driver.stops and not owner.motor_io_lock.locked()


@pytest.mark.parametrize("stage", ["feedback", "guard", "commit", "terminal"])
@pytest.mark.parametrize("base,yaw,expected", [
    (94.6, 0., (94, -94)), (60.8, -6., (54, -66)), (60.8, 6., (66, -54))])
def test_new_fractional_grant_is_mixed_before_final_admission(
        monkeypatch, stage, base, yaw, expected):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch)
    calls = inject(monkeypatch, rt, stage, lambda: publish(base, yaw))

    rt._service_follow_wheels()

    assert calls == [True] and len(driver.pairs) == 2
    assert driver.pairs[-1] == expected
    assert not driver.stops


@pytest.mark.parametrize("bad_base", [math.nan, math.inf, -math.inf, 0., -1.])
def test_quantization_does_not_turn_invalid_or_nonforward_grant_into_motion(
        monkeypatch, bad_base):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch)
    # Publication arrives inside the valid initial plan, so this exercises
    # admission as well as the real terminal packet path.
    inject(monkeypatch, rt, "commit", lambda: publish(bad_base, 0.))

    rt._service_follow_wheels()

    assert all(pair == (0, 0) for pair in driver.pairs[1:])
    assert not owner.motor_io_lock.locked()
