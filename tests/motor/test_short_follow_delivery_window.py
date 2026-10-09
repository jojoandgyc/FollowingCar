"""Finite delivery-window regression through the real writer, with fake I/O.

CAP501/602 supply physical sample intervals and publication-delay shapes, not
predictions of wheel motion. No sleep, sensor or serial device is involved.
"""
from dataclasses import replace

import pytest

from car_control_modular.detector_identity_lease import (
    ValidatedVisualObservation, publish_visual_identity_evidence,
)
from car_control_modular.short_follow import (
    ShortFollowConfig, ShortFollowController, ShortFollowObservation,
)
from test_short_follow_executor import short_runtime


def delivery_runtime(monkeypatch, *, ttl=.35, capture_lag=0., distance=2.0):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    owner._short_follow = ShortFollowController(
        replace(owner._short_follow.config, depth_ttl_sec=ttl))
    owner._short_follow.activate(1, clock[0])
    clock[0] += .3
    sample = clock[0]
    capture = sample - capture_lag
    proof = ValidatedVisualObservation(1, 1, 100, capture, sample, capture + .5, "full")
    publish_visual_identity_evidence(owner, observation=proof, lease=None)
    old = owner._short_follow.update(ShortFollowObservation(
        1, 100, capture, sample, distance, .65), sample)
    assert old is not None
    rt._service_short_follow()
    assert driver.pairs and not driver.stops
    return rt, owner, driver, clock, old


@pytest.mark.parametrize("ttl", [.30, .35])
@pytest.mark.parametrize("capture_lag,sample_gap,new_capture,expiry_tick,delivery", [
    # Old CAP497 -> new CAP501; STOP tick at old sample age 312.958 ms.
    (.153416701, .240708064, .074484301, .312957962, .326),
    # Old CAP598 -> new CAP602; STOP tick at old sample age 328.104 ms.
    (.152348644, .215267770, .049370375, .328103841, .346),
], ids=["cap501", "cap602"])
def test_short_publication_tail_has_no_stop_only_with_finite_trial_window(
        monkeypatch, ttl, capture_lag, sample_gap, new_capture, expiry_tick, delivery):
    rt, owner, driver, clock, old = delivery_runtime(
        monkeypatch, ttl=ttl, capture_lag=capture_lag)
    stamp = old.depth_timestamp
    # Identity has arrived, but the new physical measurement is still being
    # processed. This identity publication must NOT replace the old deadline.
    clock[0] = stamp + .26
    capture_ts = stamp + new_capture
    proof = ValidatedVisualObservation(1, 1, 101, capture_ts, clock[0], capture_ts + .5, "full")
    publish_visual_identity_evidence(owner, observation=proof, lease=None)
    clock[0] = stamp + expiry_tick
    rt._service_short_follow()
    assert old.expires_at == pytest.approx(min(stamp + ttl, old.capture_timestamp + .5))
    if ttl == .30:
        assert driver.stops, "The original watchdog must still expire"
        return
    assert not driver.stops
    assert owner._short_follow.snapshot().plan is old
    clock[0] = stamp + delivery
    new = owner._short_follow.update(ShortFollowObservation(
        1, 101, capture_ts, stamp + sample_gap, 2.1, .60), clock[0])
    assert new is not None and new is not old
    assert new.expires_at == pytest.approx(min(stamp + sample_gap + ttl, capture_ts + .5))
    rt._service_short_follow()
    assert not driver.stops
    assert all(pair != (0, 0) for pair in driver.pairs)


@pytest.mark.parametrize("ttl", [.30, .35])
def test_duplicate_depth_and_encoder_ticks_do_not_extend_original_deadline(monkeypatch, ttl):
    rt, owner, driver, clock, old = delivery_runtime(monkeypatch, ttl=ttl)
    for offset in (.1, .2, ttl - .001):
        clock[0] = old.depth_timestamp + offset
        # A later image cannot turn the SAME physical depth into a new sample.
        result = owner._short_follow.update(ShortFollowObservation(
            1, 101, clock[0], old.depth_timestamp, 2., .5), clock[0])
        assert result is None
        rt._service_short_follow()
        assert owner._short_follow.snapshot().plan is old
        assert not driver.stops
    clock[0] = old.expires_at
    rt._service_short_follow()
    assert driver.stops == [1]


@pytest.mark.parametrize("offset", [.350, .351, .5])
def test_trial_stops_at_or_after_350ms_without_new_measurement(monkeypatch, offset):
    rt, owner, driver, clock, old = delivery_runtime(monkeypatch)
    clock[0] = old.depth_timestamp + offset
    rt._service_short_follow()
    assert driver.stops == [1]


@pytest.mark.parametrize("identity_case", ["expired", "rejected", "wrong_uid"])
def test_trial_window_cannot_outlive_identity_or_override_rejection(monkeypatch, identity_case):
    rt, owner, driver, clock, old = delivery_runtime(monkeypatch)
    stamp = old.depth_timestamp
    clock[0] = stamp + .31
    proof = False if identity_case == "rejected" else ValidatedVisualObservation(
        2 if identity_case == "wrong_uid" else 1, 1, 101, stamp, stamp,
        stamp + (.31 if identity_case == "expired" else .5), "full")
    publish_visual_identity_evidence(owner, observation=proof, lease=None)
    rt._service_short_follow()
    assert clock[0] < old.expires_at
    assert driver.stops == [1]


def test_old_capture_visibility_can_end_trial_before_depth_watchdog(monkeypatch):
    rt, owner, driver, clock, old = delivery_runtime(monkeypatch, capture_lag=.18)
    # New visual evidence does not retimestamp a plan based on an older ROI.
    clock[0] = old.depth_timestamp + .30
    proof = ValidatedVisualObservation(1, 1, 101, clock[0], clock[0], clock[0] + .5, "full")
    publish_visual_identity_evidence(owner, observation=proof, lease=None)
    clock[0] = old.expires_at
    assert clock[0] == pytest.approx(old.depth_timestamp + .32)
    rt._service_short_follow()
    assert driver.stops == [1]


@pytest.mark.parametrize("distance", [1.8, 2., 2.348, 2.81853])
def test_entire_extended_watchdog_is_accounted_for_by_same_braking_cap(distance):
    baseline = ShortFollowController(ShortFollowConfig(depth_ttl_sec=.30))
    trial = ShortFollowController(ShortFollowConfig(depth_ttl_sec=.35))
    speed_rpm = trial._speed_cap(distance)
    assert 0 < speed_rpm < baseline._speed_cap(distance)
    speed = speed_rpm * trial.config.wheel_circumference_m / 60.
    space = distance - trial.config.braking_stop_distance_m - trial.config.braking_margin_m
    required = (speed * (trial.config.depth_ttl_sec + trial.config.response_delay_sec)
                + speed * speed / (2. * trial.config.deceleration_m_s2))
    assert required == pytest.approx(space)


def test_extended_delivery_window_does_not_integrate_missing_320ms(monkeypatch):
    _, owner, _, clock, old = delivery_runtime(monkeypatch, distance=1.60)
    clock[0] = old.depth_timestamp + .32
    new = owner._short_follow.update(ShortFollowObservation(
        1, 101, clock[0], clock[0], 1.60, .5), clock[0])
    assert new is not None and new.integral_dt_sec == 0.
    assert owner._short_follow.config.max_integral_gap_sec == .30
