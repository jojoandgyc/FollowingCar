"""Ordinary current release uses measured settling; fake clock/driver only."""
from types import SimpleNamespace

import pytest

from car_control_modular.near_yaw_parking import ParkSettlingEvidence
from test_follow_wheel_periodic import setup_periodic
from test_near_yaw_park_execution import request_park
from test_search_handoff_execution import arm
from test_visible_wheel_continuity import feedback


def parked(monkeypatch, search):
    rt, owner, driver, _, clock, _ = setup_periodic(monkeypatch)
    if search:
        arm(rt, owner, clock)
        request = rt._search_reacquire_brake_request
    else:
        request = request_park(owner, clock)
    rt._service_follow_wheels()
    evidence = rt._search_reacquire_settling if search else rt._near_yaw_park_settling
    assert evidence.sent_at == 10.
    assert rt.backend.parking_current_a == 5.
    return rt, owner, driver, clock, evidence, request


def tick(rt, clock, stamp, sample):
    clock[0] = stamp
    rt.get_steering_feedback = lambda: sample
    rt._service_follow_wheels()


@pytest.mark.parametrize('search', [False, True])
def test_quiet_releases_current_before_500ms_but_does_not_release_motion(monkeypatch, search):
    rt, owner, driver, clock, evidence, request = parked(monkeypatch, search)
    initial_pairs = list(driver.pairs)
    tick(rt, clock, 10.02, feedback(10.02))
    assert evidence.current_released_at is None
    tick(rt, clock, 10.07, feedback(10.07))
    assert evidence.current_released_at == 10.07
    assert rt.backend.parking_current_a == 0.
    assert driver.stops == [1, 0, 2]
    assert driver.pairs == initial_pairs
    assert owner._brake_hold_active
    assert evidence.ready_at is None and evidence.quiet_count == 0
    assert evidence.sent_at == 10.

    # Reusing the sample from before FREE cannot finish the search/quiet gate.
    if search:
        assert rt.search_reacquire_brake_pending(capture_timestamp=10.07)
    else:
        assert not rt.near_yaw_park_release_ready(request, 10.07, 10.07)
    for stamp in (10.08, 10.13):
        tick(rt, clock, stamp, feedback(stamp))
    if search:
        assert not rt.search_reacquire_brake_pending(capture_timestamp=10.13)
    else:
        assert rt.near_yaw_park_release_ready(request, 10.13, 10.13)
    assert driver.pairs == initial_pairs and driver.stops == [1, 0, 2]
    assert not any(value for pair in driver.pairs for value in pair)


@pytest.mark.parametrize('search', [False, True])
def test_duplicate_feedback_does_not_accumulate_quiet_or_extend_timeout(monkeypatch, search):
    rt, owner, driver, clock, evidence, request = parked(monkeypatch, search)
    sample = feedback(10.02)
    for stamp in (10.02, 10.07, 10.10, 10.15, 10.3, 10.49):
        tick(rt, clock, stamp, sample)
        assert evidence.current_released_at is None
        assert evidence.sent_at == 10.
        assert rt.backend.parking_current_a == 5.
        assert evidence.quiet_count <= 1
    tick(rt, clock, 10.5, None)
    assert evidence.current_released_at == 10.5
    assert rt.backend.parking_current_a == 0.
    assert owner._brake_hold_active
    for stamp in (10.55, 10.8, 11.):
        tick(rt, clock, stamp, None)
        assert evidence.current_released_at == 10.5
        if search:
            assert rt.search_reacquire_brake_pending(capture_timestamp=stamp)
        else:
            assert not rt.near_yaw_park_release_ready(request, stamp, stamp)
            assert not rt.near_yaw_park_motion_ready(request, stamp, stamp)
    assert driver.stops == [1, 0, 2]
    assert not any(value for pair in driver.pairs for value in pair)


@pytest.mark.parametrize('bad', ['pre_stop', 'moving', 'untrusted', 'future', 'nan', 'gap'])
def test_invalid_second_sample_cannot_release_current_early(bad):
    evidence = ParkSettlingEvidence(object(), 10., require_current_release=True)
    assert not evidence.current_release_ready(feedback(10.02), 10.02)
    sample = feedback(10.07)
    now = 10.07
    if bad == 'pre_stop': sample.timestamp = 10.
    elif bad == 'moving': sample.left_forward_rpm = 1.01
    elif bad == 'untrusted': sample.trustworthy = False
    elif bad == 'future': sample.timestamp = 10.08
    elif bad == 'nan': sample.right_forward_rpm = float('nan')
    elif bad == 'gap': sample.timestamp = now = 10.18
    assert not evidence.current_release_ready(sample, now)
    assert evidence.current_released_at is None
    assert not evidence.recovery_gate(now)


@pytest.mark.parametrize('search', [False, True])
@pytest.mark.parametrize('failure', ['current', 'free', 'safety_during_current'])
def test_early_release_failure_remains_latched_and_never_writes_motion(monkeypatch, search, failure):
    rt, owner, driver, clock, evidence, request = parked(monkeypatch, search)
    tick(rt, clock, 10.02, feedback(10.02))
    release = rt.backend.release_parking_current_only
    stop = rt.backend.send_stop
    dangerous = [False]
    rt.hard_stop_check = lambda action: dangerous[0]
    if failure == 'current':
        def fail_current():
            raise OSError('fake current readback failure')
        rt.backend.release_parking_current_only = fail_current
    elif failure == 'free':
        def fail_free(*args, **kwargs):
            if kwargs.get('mode') == 'free':
                raise OSError('fake FREE write failure')
            return stop(*args, **kwargs)
        rt.backend.send_stop = fail_free
    else:
        def current_then_danger():
            release()
            dangerous[0] = True
        rt.backend.release_parking_current_only = current_then_danger
    tick(rt, clock, 10.07, feedback(10.07))
    assert evidence.fault
    assert evidence.current_released_at is None
    assert owner._brake_hold_active and owner._brake_hold_stop_mode == 'emergency'
    assert driver.stops[-1] == 1 and 2 not in driver.stops
    before = (list(driver.pairs), list(driver.stops), list(driver.register_writes))
    rt.backend.release_parking_current_only = release
    rt.backend.send_stop = stop
    dangerous[0] = False
    for stamp in (10.10, 10.6, 11.):
        tick(rt, clock, stamp, feedback(stamp))
    assert (driver.pairs, driver.stops, driver.register_writes) == before
    assert not any(value for pair in driver.pairs for value in pair)


@pytest.mark.parametrize('search', [False, True])
def test_emergency_preempts_quiet_exit_without_clearing_parking_current(monkeypatch, search):
    rt, owner, driver, clock, evidence, _ = parked(monkeypatch, search)
    tick(rt, clock, 10.02, feedback(10.02))
    rt.hard_stop_check = lambda action: True
    tick(rt, clock, 10.07, feedback(10.07))
    assert owner._brake_hold_active and owner._brake_hold_stop_mode == 'emergency'
    assert driver.stops[-1] == 1 and 2 not in driver.stops
    assert evidence.current_released_at is None
    assert rt.backend.parking_current_a == 5.
    assert not any(value for pair in driver.pairs for value in pair)


def test_clock_or_fault_cannot_turn_fallback_into_a_release_permission():
    evidence = ParkSettlingEvidence(SimpleNamespace(uid=1), 10., require_current_release=True)
    for stamp in (float('nan'), float('inf'), 9.99):
        assert not evidence.current_release_ready(None, stamp)
        assert not evidence.recovery_gate(stamp)
    evidence.fault = 'current_release_failed'
    assert not evidence.current_release_ready(feedback(11.), 11.)
    assert not evidence.recovery_gate(11.)
    assert evidence.reason == 'current_release_failed'
