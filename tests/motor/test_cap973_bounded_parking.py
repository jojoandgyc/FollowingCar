"""CAP944..1100: bounded 5A phase even if the motor never settles in NORMAL.

Fake clock, fake registers and real motor executor only; no hardware.
"""
from dataclasses import replace
import pytest

from test_follow_wheel_periodic import setup_periodic
from test_near_yaw_park_execution import request_park
from test_search_handoff_execution import arm
from test_visible_wheel_continuity import feedback


def park(monkeypatch, search=False, current_a=5.):
    rt, owner, driver, symbols, clock, axes = setup_periodic(monkeypatch)
    rt.backend.config = replace(rt.backend.config, parking_current_a=current_a)
    axes[:2] = [0, 0]
    if search:
        arm(rt, owner, clock)
        request = rt._search_reacquire_brake_request
    else:
        request = request_park(owner, clock)
    rt.get_steering_feedback = lambda: feedback(clock[0], -27, -19)
    rt._service_follow_wheels()
    evidence = rt._search_reacquire_settling if search else rt._near_yaw_park_settling
    label = "search_reacquire_brake" if search else "near_yaw_park"
    return rt, owner, driver, symbols, clock, request, evidence, label


@pytest.mark.parametrize("search", [False, True])
def test_500ms_clears_current_even_with_oscillation_and_blocks_old_commands(monkeypatch, search):
    rt, o, d, s, clock, req, e, label = park(monkeypatch, search)
    before = list(d.pairs)
    for now in [10.1, 10.3, 10.40, 10.499]:
        clock[0] = now
        rt._service_follow_wheels()
        assert rt.backend.parking_current_a == 5
        assert d.pairs == before
    clock[0] = 10.50
    rt._service_follow_wheels()
    assert rt.backend.parking_current_a == 0
    assert all(v == 0 for v in d.registers.values())
    assert e.current_released_at == 10.50
    assert d.pairs == before  # clearing 0A never supersedes STOP with a speed write
    assert not e.release_ready(10.51, feedback(10.51, -27, -19), 10.51)
    assert o._brake_hold_active
    packets, writes = list(d.pairs), list(d.register_writes)
    for now in [10.55, 10.70, 11.0]:
        clock[0] = now
        rt._service_follow_wheels()
        rt.send_robot_command(s.rotate_right)
        rt.send_percent_drive(0)
        rt.send_percent_brake("normal", label)
    assert d.pairs == packets and d.register_writes == writes
    assert d.stops == [1, 0, 2]  # exactly one FREE; no refresh reinstates NORMAL


@pytest.mark.parametrize("search", [False, True])
def test_release_needs_post_0a_quiet_samples_and_new_image(monkeypatch, search):
    rt, o, d, s, clock, req, e, label = park(monkeypatch, search)
    rt.get_steering_feedback = lambda: feedback(clock[0], 0, 0)
    for now in [10.40, 10.45, 10.50]:
        clock[0] = now
        rt._service_follow_wheels()
    assert e.ready_at is None  # discard pre-clear quiet evidence
    for now in [10.51, 10.56]:
        clock[0] = now
        rt._service_follow_wheels()
    assert e.ready_at == 10.56
    assert not e.release_ready(10.49, feedback(10.56), 10.57)
    assert not e.release_ready(10.53, feedback(10.56), 10.57)
    clock[0] = 10.58
    if search:
        assert not rt.search_reacquire_brake_pending(capture_timestamp=10.57)
    else:
        assert rt.near_yaw_park_release_ready(req, 10.57, 10.58)
    assert all(pair == (0, 0) for pair in d.pairs)  # eligibility does not drive


@pytest.mark.parametrize("search", [False, True])
@pytest.mark.parametrize("current_a", [0., 5.])
@pytest.mark.parametrize("wait_until", [12.51, 40.])
def test_long_settling_recovers_with_new_evidence_without_restart(monkeypatch, search, current_a, wait_until):
    rt, o, d, s, clock, req, e, label = park(monkeypatch, search, current_a)
    clock[0] = 10.50
    rt._service_follow_wheels()
    before = list(d.pairs)
    clock[0] = wait_until
    rt._service_follow_wheels()
    assert e.fault is None and rt.backend.parking_release_fault is None
    assert not e.release_ready(clock[0], feedback(clock[0], -27, -19), clock[0])
    assert rt.backend.parking_current_a == 0
    assert d.stops == [1, 0, 2]
    rt.send_robot_command(s.rotate_right)  # old motion cannot bypass the hold
    assert d.pairs == before and o._brake_hold_active
    rt.get_steering_feedback = lambda: feedback(clock[0], 0, 0)
    clock[0] = wait_until + .05
    rt._service_follow_wheels()
    assert not e.release_ready(clock[0], rt.get_steering_feedback(), clock[0])
    clock[0] += .06
    rt._service_follow_wheels()
    assert not e.release_ready(wait_until, rt.get_steering_feedback(), clock[0])
    if search:
        assert not rt.search_reacquire_brake_pending(capture_timestamp=clock[0])
    else:
        assert rt.near_yaw_park_release_ready(req, clock[0], clock[0])
    assert e.fault is None and rt.backend.parking_release_fault is None
    assert d.pairs == before and d.stops == [1, 0, 2]  # eligibility does not drive


@pytest.mark.parametrize("failure", ["write", "readback", "nan"])
def test_partial_io_failure_latches_emergency_and_never_retries_motion(monkeypatch, failure):
    rt, o, d, s, clock, req, e, label = park(monkeypatch)
    if failure == "write":
        original = d.write_register
        def write(name, value, **kw):
            if name == "left_parking_current" and value == 0:
                raise OSError("partial clear")
            original(name, value, **kw)
        d.write_register = write
    elif failure == "readback":
        d.read_register = lambda name: 5
    else:
        d.read_register = lambda name: float("nan")
    clock[0] = 10.51
    rt._service_follow_wheels()
    assert e.fault == "current_release_failed"
    assert e.current_released_at is None
    assert d.stops[-1] == 1
    clock[0] = 18
    rt._service_follow_wheels()
    assert not rt.near_yaw_park_motion_ready(req, 17.99, 18)
    assert all(pair == (0, 0) for pair in d.pairs)


@pytest.mark.parametrize("protected", ["explicit", "shutdown", "safety"])
def test_timer_never_clears_protected_stop(monkeypatch, protected):
    rt, o, d, s, clock, req, e, label = park(monkeypatch)
    if protected == "explicit": o._explicit_stop_requested = True
    elif protected == "shutdown": o._runtime_shutdown_requested = True
    else: o._brake_hold_label = "safety_hold_front"
    clock[0] = 11
    rt._service_follow_wheels()
    assert rt.backend.parking_current_a == 5
    assert e.current_released_at is None


def test_hazard_during_current_io_never_releases_motion(monkeypatch):
    rt, o, d, s, clock, req, e, label = park(monkeypatch)
    original = rt.backend.release_parking_current_only
    def hazard():
        original()
        rt.hard_stop_check = lambda action: True
    rt.backend.release_parking_current_only = hazard
    clock[0] = 10.51
    rt._service_follow_wheels()
    assert e.fault == "authority_changed_during_current_release"
    assert d.stops[-1] == 1
    assert e.current_released_at is None


def test_refresh_rechecks_hazard_before_clearing_current(monkeypatch):
    rt, o, d, s, clock, req, e, label = park(monkeypatch)
    clock[0] = 10.51
    rt.hard_stop_check = lambda action: True
    rt.send_percent_brake("normal", label)
    assert rt.backend.parking_current_a == 5
    assert e.current_released_at is None
    assert d.stops[-1] == 1
    assert o._brake_hold_label == "safety_hold_hard_stop"


def test_explicit_stop_during_current_readback_takes_emergency_ownership(monkeypatch):
    rt, o, d, s, clock, req, e, label = park(monkeypatch)
    original = d.read_register
    def interrupt(*a, **kw):
        result = original(*a, **kw)
        o._explicit_stop_requested = True
        return result
    d.read_register = interrupt
    clock[0] = 10.51
    rt._service_follow_wheels()
    assert e.fault == "authority_changed_during_current_release"
    assert e.current_released_at is None
    assert d.stops[-1] == 1


def test_next_genuine_park_restores_5a_not_old_episode(monkeypatch):
    rt, o, d, s, clock, req, e, label = park(monkeypatch)
    clock[0] = 10.51
    rt._service_follow_wheels()
    assert rt.backend.parking_current_a == 0
    clock[0] = 10.70
    o._near_yaw_park_request = replace(req, capture_frame_id=99, requested_at=10.70)
    rt._service_follow_wheels()
    assert rt.backend.parking_current_a == 5
    assert d.stops == [1, 0, 2, 1, 0]
    assert rt._near_yaw_park_settling.current_released_at is None


def test_deadline_starts_at_stop_completion_and_clear_ack_starts_new_evidence(monkeypatch):
    rt, o, d, s, clock, axes = setup_periodic(monkeypatch)
    request_park(o, clock)
    original_stop = rt.backend.send_stop
    def delayed_stop(*a, **kw):
        original_stop(*a, **kw)
        clock[0] += .07
    rt.backend.send_stop = delayed_stop
    rt._service_follow_wheels()
    e = rt._near_yaw_park_settling
    assert e.sent_at == pytest.approx(10.07)
    clock[0] = 10.56
    rt._service_follow_wheels()
    assert rt.backend.parking_current_a == 5
    read = d.read_register
    def slow_read(name):
        clock[0] += .02
        return read(name)
    d.read_register = slow_read
    clock[0] = 10.58
    rt._service_follow_wheels()
    assert e.current_released_at == pytest.approx(10.69)  # includes FREE STOP I/O
    # CAP370: qualified translation may use still-fresh post-STOP evidence
    # collected before FREE completion; settled search/recentering may not.
    assert e.motion_handoff_ready(10.63, feedback(10.64), 10.70)
    assert not e.release_ready(10.63, feedback(10.64), 10.70)
    assert e.motion_handoff_ready(10.70, feedback(10.71), 10.71)


@pytest.mark.parametrize("bad", ["missing", "stale", "nan", "untrusted"])
def test_post_clear_bad_feedback_keeps_waiting_without_time_unlock(monkeypatch, bad):
    rt, o, d, s, clock, req, e, label = park(monkeypatch)
    clock[0] = 10.50
    rt._service_follow_wheels()
    clock[0] = 12.51
    fb = feedback(clock[0])
    if bad == "missing": fb = None
    elif bad == "stale": fb.timestamp = 10.51
    elif bad == "nan": fb.left_forward_rpm = float("nan")
    else: fb.trustworthy = False
    rt.get_steering_feedback = lambda: fb
    rt._service_follow_wheels()
    assert e.fault is None and rt.backend.parking_release_fault is None
    assert not rt.near_yaw_park_release_ready(req, 12.50, 12.51)
    assert d.stops == [1, 0, 2] and d.pairs == [(0, 0)]
