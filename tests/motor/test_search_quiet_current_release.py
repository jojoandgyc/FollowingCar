"""CAP242/260 search-pause dwell regressions; fake motor and clock only.

Early current release never replays a wheel command and does not reuse quiet
feedback across the dual-0A/FREE transaction.
"""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from car_control_modular.control_types import SteeringFeedback
from test_follow_wheel_periodic import setup_periodic
from test_near_yaw_park_execution import request_park
from test_search_handoff_execution import arm


def sample(stamp, left=0., right=0., **changes):
    return SteeringFeedback(timestamp=stamp, trustworthy=True,
                            left_forward_rpm=left, right_forward_rpm=right,
                            **changes)


_DEFAULT_FEEDBACK = object()


def start_search_stop(monkeypatch, *, before=_DEFAULT_FEEDBACK, search=True):
    rt, owner, driver, symbols, clock, axes = setup_periodic(monkeypatch)
    rt.config.motor_rs485_stop_mode = "emergency"  # current field configuration
    rt.backend.config = replace(rt.backend.config, parking_current_a=10., stop_mode="emergency")
    axes[:2] = [0., 0.]
    if search:
        arm(rt, owner, clock)
    else:
        request_park(owner, clock)
    initial = sample(9.99, -7., 7.) if before is _DEFAULT_FEEDBACK else before
    rt.get_steering_feedback = lambda: initial
    rt._service_follow_wheels()
    evidence = rt._search_reacquire_settling if search else rt._near_yaw_park_settling
    assert evidence.sent_at == 10.
    assert driver.stops == [1]
    assert driver.pairs == []
    return rt, owner, driver, symbols, clock, evidence


def tick(rt, clock, now, feedback=None):
    clock[0] = now
    fb = sample(now) if feedback is None else feedback
    rt.get_steering_feedback = lambda: fb
    rt._service_follow_wheels()
    return fb


@pytest.mark.parametrize("first,second", [
    pytest.param(.226, .2765, id="cap242-quiet-at-276ms"),
    pytest.param(.0234, .0778, id="cap260-low-speed-quiet-at-78ms"),
])
def test_low_speed_search_releases_at_measured_quiet_not_fixed_500ms(monkeypatch, first, second):
    rt, owner, driver, _, clock, evidence = start_search_stop(monkeypatch)
    assert evidence.allow_early_quiet_release
    tick(rt, clock, 10. + first)
    assert evidence.current_released_at is None
    assert rt.search_reacquire_brake_pending(capture_timestamp=clock[0])
    tick(rt, clock, 10. + second)
    assert evidence.current_released_at == pytest.approx(10. + second)
    assert evidence.early_quiet_release_completed
    assert evidence.pre_release_quiet_at == pytest.approx(10. + second)
    assert evidence.ready_at is None and evidence.quiet_count == 0
    assert rt.backend.parking_current_a == 0.
    assert set(driver.registers.values()) == {0.}
    assert driver.stops == [1, 2] and driver.pairs == []
    assert rt.search_reacquire_brake_pending(capture_timestamp=clock[0])

    # Current release is a new physical boundary. Before-release feedback and
    # images cannot stand in for the encoder/image check after FREE ACK.
    first_post = 10. + second + .020
    second_post = 10. + second + .075
    tick(rt, clock, first_post)
    assert rt.search_reacquire_brake_pending(capture_timestamp=first_post)
    assert rt.search_reacquire_brake_pending(capture_timestamp=first_post)
    assert evidence.quiet_count == 1
    tick(rt, clock, second_post)
    assert evidence.ready_at == second_post
    assert rt.search_reacquire_brake_pending(capture_timestamp=first_post)
    assert not rt.search_reacquire_brake_pending(capture_timestamp=second_post)
    assert second_post < 10.5
    assert owner._brake_hold_active  # new validated search/motion owns resumption
    assert driver.stops == [1, 2] and driver.pairs == []


@pytest.mark.parametrize("bad", ["missing", "moving", "stale", "future", "untrusted", "nan",
                                 "left_error", "right_error"])
def test_pre_stop_feedback_must_qualify_low_speed_episode(monkeypatch, bad):
    initial = sample(9.99, -7., 7.)
    if bad == "missing": initial = None
    elif bad == "moving": initial = replace(initial, left_forward_rpm=11.)
    elif bad == "stale": initial = replace(initial, timestamp=9.8)
    elif bad == "future": initial = replace(initial, timestamp=10.01)
    elif bad == "untrusted": initial = replace(initial, trustworthy=False)
    elif bad == "nan": initial = replace(initial, right_forward_rpm=float("nan"))
    else: initial = replace(initial, **{bad: 1})
    rt, _, driver, _, clock, evidence = start_search_stop(monkeypatch, before=initial)
    assert not evidence.allow_early_quiet_release
    for now in (10.02, 10.08, 10.13, 10.499):
        tick(rt, clock, now)
    assert evidence.current_released_at is None
    assert rt.backend.parking_current_a == 10.
    tick(rt, clock, 10.50)
    assert evidence.current_released_at == 10.50
    assert not evidence.early_quiet_release_completed
    assert driver.stops == [1, 2] and not driver.pairs


@pytest.mark.parametrize("bad", ["repeat", "too_short", "moving", "stale", "future",
                                 "untrusted", "nan", "left_error", "right_error", "out_of_order"])
def test_early_release_requires_two_distinct_valid_post_stop_quiet_samples(monkeypatch, bad):
    rt, _, driver, _, clock, evidence = start_search_stop(monkeypatch)
    tick(rt, clock, 10.02)
    second = sample(10.08)
    if bad == "repeat": second = replace(second, timestamp=10.02)
    elif bad == "too_short": second = replace(second, timestamp=10.04)
    elif bad == "moving": second = replace(second, left_forward_rpm=2.)
    elif bad == "stale": second = replace(second, timestamp=9.8)
    elif bad == "future": second = replace(second, timestamp=10.09)
    elif bad == "untrusted": second = replace(second, trustworthy=False)
    elif bad == "nan": second = replace(second, left_forward_rpm=float("nan"))
    elif bad == "out_of_order": second = replace(second, timestamp=10.01)
    else: second = replace(second, **{bad: 1})
    tick(rt, clock, 10.08, second)
    assert evidence.current_released_at is None
    assert driver.stops == [1] and not driver.pairs
    assert rt.backend.parking_current_a == 10.


def test_stop_and_free_completion_each_establish_their_own_sample_boundary(monkeypatch):
    rt, _, driver, _, clock, evidence = start_search_stop(monkeypatch)
    read = driver.read_register
    def delayed_read(name):
        clock[0] += .02
        return read(name)
    driver.read_register = delayed_read
    stop = rt.backend.send_stop
    def delayed_free(label, **kw):
        result = stop(label, **kw)
        if label == "ordinary_park_release_free":
            clock[0] += .03
        return result
    rt.backend.send_stop = delayed_free
    tick(rt, clock, 10.02)
    tick(rt, clock, 10.08)
    assert evidence.current_released_at == pytest.approx(10.15)
    tick(rt, clock, 10.17, sample(10.12))
    assert evidence.quiet_count == 0  # read while releasing current isn't post-FREE
    assert rt.search_reacquire_brake_pending(capture_timestamp=10.17)
    tick(rt, clock, 10.19)
    tick(rt, clock, 10.25)
    assert rt.search_reacquire_brake_pending(capture_timestamp=10.20)
    assert not rt.search_reacquire_brake_pending(capture_timestamp=10.25)
    assert driver.stops == [1, 2] and not driver.pairs


@pytest.mark.parametrize("bad", ["repeat", "stale", "moving", "left_error", "right_error"])
def test_post_release_feedback_fault_or_replay_does_not_release_search(monkeypatch, bad):
    rt, _, driver, _, clock, evidence = start_search_stop(monkeypatch)
    for now in (10.02, 10.08, 10.10):
        tick(rt, clock, now)
    assert evidence.early_quiet_release_completed and evidence.quiet_count == 1
    fb = sample(10.16)
    if bad == "repeat": fb = replace(fb, timestamp=10.10)
    elif bad == "stale": fb = replace(fb, timestamp=10.08)
    elif bad == "moving": fb = replace(fb, right_forward_rpm=2.)
    else: fb = replace(fb, **{bad: 1})
    tick(rt, clock, 10.16, fb)
    assert rt.search_reacquire_brake_pending(capture_timestamp=10.16)
    assert evidence.ready_at is None
    assert driver.stops == [1, 2] and not driver.pairs


@pytest.mark.parametrize("failure", ["write", "readback", "free", "hazard", "ownership"])
def test_early_current_or_free_failure_keeps_search_blocked(monkeypatch, failure):
    rt, owner, driver, _, clock, evidence = start_search_stop(monkeypatch)
    if failure == "write":
        write = driver.write_register
        def fail_write(name, value, **kw):
            if name == "left_parking_current" and value == 0:
                raise OSError("left clear failed")
            return write(name, value, **kw)
        driver.write_register = fail_write
    elif failure == "readback":
        driver.read_register = lambda name: 10.
    elif failure == "free":
        stop = rt.backend.send_stop
        def fail_free(label, **kw):
            if label == "ordinary_park_release_free":
                raise OSError("FREE ACK failed")
            return stop(label, **kw)
        rt.backend.send_stop = fail_free
    else:
        clear = rt.backend.release_parking_current_only
        def interrupt_clear():
            clear()
            if failure == "hazard":
                rt.hard_stop_check = lambda action: True
            else:
                owner._explicit_stop_requested = True
        rt.backend.release_parking_current_only = interrupt_clear
    tick(rt, clock, 10.02)
    tick(rt, clock, 10.08)
    assert evidence.fault
    assert evidence.current_released_at is None
    assert not evidence.early_quiet_release_completed
    assert owner._brake_hold_active and owner._brake_hold_label.startswith("safety")
    assert driver.stops[-1] == 1 and not driver.pairs


@pytest.mark.parametrize("protected", ["explicit", "shutdown", "safety", "hard_stop"])
def test_new_protected_stop_never_early_releases_search_current(monkeypatch, protected):
    rt, owner, driver, _, clock, evidence = start_search_stop(monkeypatch)
    tick(rt, clock, 10.02)
    if protected == "explicit": owner._explicit_stop_requested = True
    elif protected == "shutdown": owner._runtime_shutdown_requested = True
    elif protected == "safety": owner._brake_hold_label = "safety_hold_front"
    else: rt.hard_stop_check = lambda action: True
    tick(rt, clock, 10.08)
    assert evidence.current_released_at is None
    assert rt.backend.parking_current_a == 10.
    assert 2 not in driver.stops and not driver.pairs


def test_near_distance_park_keeps_original_500ms_default(monkeypatch):
    rt, _, driver, _, clock, evidence = start_search_stop(monkeypatch, search=False)
    assert not evidence.allow_early_quiet_release
    for now in (10.02, 10.08, 10.499):
        tick(rt, clock, now)
        assert evidence.current_released_at is None
    tick(rt, clock, 10.50)
    assert evidence.current_released_at == 10.50
    assert not evidence.early_quiet_release_completed
    assert driver.stops == [1, 2] and not driver.pairs


def test_second_pause_does_not_inherit_first_pause_quiet_or_early_release(monkeypatch):
    rt, owner, driver, _, clock, first = start_search_stop(monkeypatch)
    for now in (10.02, 10.08, 10.10, 10.16):
        tick(rt, clock, now)
    assert not rt.search_reacquire_brake_pending(capture_timestamp=10.16)
    clock[0] = 10.20
    rt.get_steering_feedback = lambda: sample(10.19, 7., -7.)
    assert rt.request_search_reacquire_brake(260, 10.19, "candidate_predictive_stop")
    rt._service_follow_wheels()
    second = rt._search_reacquire_settling
    assert second is not first and second.sent_at == 10.20
    assert second.current_released_at is None and second.quiet_count == 0
    assert not second.early_quiet_release_completed
    tick(rt, clock, 10.22)
    assert second.current_released_at is None
    tick(rt, clock, 10.28)
    assert second.current_released_at == 10.28
    assert second.ready_at is None
    assert rt.search_reacquire_brake_pending(capture_timestamp=10.28)
    assert driver.stops == [1, 2, 1, 2] and not driver.pairs


def test_low_speed_qualification_uses_clock_after_feedback_cache_read(monkeypatch):
    rt, owner, driver, _, clock, axes = setup_periodic(monkeypatch)
    rt.backend.config = replace(rt.backend.config, stop_mode="emergency")
    axes[:2] = [0., 0.]
    arm(rt, owner, clock)

    def newly_published_feedback():
        clock[0] += .001
        return sample(clock[0], -7., 7.)

    rt.get_steering_feedback = newly_published_feedback
    rt._service_follow_wheels()
    evidence = rt._search_reacquire_settling
    assert evidence.allow_early_quiet_release
    assert evidence.sent_at > 10.
    assert driver.stops == [1] and not driver.pairs
    stop_at = evidence.sent_at
    tick(rt, clock, stop_at + .02)
    tick(rt, clock, stop_at + .08)
    assert evidence.early_quiet_release_completed
    assert evidence.current_released_at == pytest.approx(stop_at + .08)
    assert driver.stops == [1, 2] and not driver.pairs


def test_quiet_release_qualification_uses_clock_after_feedback_cache_read(monkeypatch):
    rt, _, driver, _, clock, evidence = start_search_stop(monkeypatch)

    def newly_published_quiet_feedback():
        clock[0] += .001
        return sample(clock[0])

    rt.get_steering_feedback = newly_published_quiet_feedback
    clock[0] = 10.02
    rt._service_follow_wheels()
    assert evidence.current_released_at is None
    assert evidence.quiet_count >= 1
    clock[0] = 10.08
    rt._service_follow_wheels()
    assert evidence.early_quiet_release_completed
    assert evidence.current_released_at == pytest.approx(10.081)
    assert evidence.sent_at == 10.  # checking later never renews the STOP clock
    assert evidence.ready_at is None  # fresh evidence still needed after FREE
    assert driver.stops == [1, 2] and not driver.pairs


def test_post_release_observation_uses_clock_after_second_cache_read(monkeypatch):
    rt, owner, driver, _, clock, evidence = start_search_stop(monkeypatch)
    tick(rt, clock, 10.02)
    tick(rt, clock, 10.08)
    assert evidence.current_released_at == 10.08

    def newly_published_quiet_feedback():
        clock[0] += .001
        return sample(clock[0])

    rt.get_steering_feedback = newly_published_quiet_feedback
    for now in (10.10, 10.16):
        clock[0] = now
        # Exercise this helper directly: the outer search service observes
        # again and could otherwise hide this erroneous clearing of quiet.
        with owner.motor_io_lock:
            rt._service_ordinary_park_exit(evidence, "search_reacquire_brake")
    assert evidence.quiet_count == 2
    assert evidence.ready_at == pytest.approx(10.162)
    assert evidence.current_released_at == 10.08
    assert evidence.sent_at == 10.
    assert driver.stops == [1, 2] and not driver.pairs
