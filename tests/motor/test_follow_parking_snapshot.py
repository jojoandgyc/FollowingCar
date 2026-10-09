"""Released ordinary parking must not trap new follow grants at terminal I/O.

Real executor and MssdMotorBackend, fake registers/clock only. These tests do
not claim physical braking performance or manufacture measured movement.
"""
from dataclasses import replace

import pytest

from car_control_modular.detector_identity_lease import ValidatedVisualObservation
from test_unified_forward_snapshot import feedback, writer
from test_follow_wheel_periodic import setup_periodic


def parked_writer(monkeypatch, *, base=16., yaw=2., current=10.):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=base, yaw=yaw)
    rt.backend.config = replace(rt.backend.config, parking_current_a=current, stop_mode="emergency")
    rt.backend.send_stop("parking_exit_authority_changed", mode="emergency",
                         preserve_zero=True, prepare_parking_current=True)
    assert rt.backend.normal_zero_hold and rt.backend.parking_current_a == current
    publish(base, yaw)
    rt._steering_feedback = feedback(clock[0], 0., 0.)
    return rt, owner, driver, clock, publish


@pytest.mark.parametrize("yaw", [-2., 0., 2.])
@pytest.mark.parametrize("current", [0., 10.])
def test_fresh_snapshot_exits_released_parking_without_zero_or_new_stop(
        monkeypatch, yaw, current):
    rt, owner, driver, clock, publish = parked_writer(monkeypatch, yaw=yaw, current=current)
    old = owner._depth30_linear_timing
    rt._service_follow_wheels()
    assert driver.pairs == [(int(16+yaw), -int(16-yaw))] * 2
    assert driver.stops == [1]
    assert not rt.backend.normal_zero_hold and rt.backend.parking_current_a == 0.
    assert owner._depth30_linear_timing is old  # No renewed measurement deadline.
    if current:
        assert driver.register_writes[-2:] == [
            ("right_parking_current", 0., False), ("left_parking_current", 0., False)]
    writes = list(driver.register_writes)
    clock[0] += .05
    publish(20., yaw)
    rt._steering_feedback = feedback(clock[0], 0., 0.)
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (int(20+yaw), -int(20-yaw))
    assert driver.register_writes == writes  # Current I/O is transition-only.


def change_on_release_read(monkeypatch, driver, change):
    original = driver.read_register
    seen = []

    def read(name):
        value = original(name)
        if not seen:
            seen.append(name)
            change()
        return value

    monkeypatch.setattr(driver, "read_register", read)
    return seen


@pytest.mark.parametrize("change", ["depth", "visual", "feedback", "uid", "identity",
                                    "explicit_stop", "stop", "hazard", "park"])
def test_release_io_never_bypasses_late_revocation(monkeypatch, change):
    rt, owner, driver, clock, _ = parked_writer(monkeypatch)
    before = list(driver.pairs)
    owner._validated_visual_observation = ValidatedVisualObservation(
        1, 3, 33, clock[0], clock[0], clock[0]+.10, "full")

    def revoke():
        if change == "depth":
            clock[0] += .251
            rt._steering_feedback = feedback(clock[0])
        elif change == "visual":
            clock[0] += .101
            rt._steering_feedback = feedback(clock[0])
        elif change == "feedback":
            clock[0] += .151
            owner._validated_visual_observation = None
        elif change == "uid":
            owner._follow_controller.active_target_id = 2
        elif change == "identity":
            owner._detector_identity_lease = False
        elif change == "explicit_stop":
            owner._explicit_stop_requested = True
        elif change == "stop":
            rt.backend.send_stop("newer_stop_during_release", mode="emergency", preserve_zero=True)
        elif change == "hazard":
            rt.hard_stop_check = lambda _: True
        else:
            owner._near_yaw_park_request = object()

    seen = change_on_release_read(monkeypatch, driver, revoke)
    rt._service_follow_wheels()
    assert seen
    assert all(pair == (0, 0) for pair in driver.pairs[len(before):])
    if change in {"explicit_stop", "stop", "park"}:
        assert driver.pairs == before  # A newer STOP owner is not speed-zero.
    if change in {"stop", "hazard"}:
        assert driver.stops == [1, 1]
    assert not owner.motor_io_lock.locked()


@pytest.mark.parametrize("fault", ["read", "readback", "partial_write"])
def test_parking_release_failure_cannot_send_speed(monkeypatch, fault):
    rt, owner, driver, _, _ = parked_writer(monkeypatch)
    before = list(driver.pairs)
    if fault == "partial_write":
        original = driver.write_register

        def write(name, value, **kw):
            if name == "left_parking_current" and value == 0:
                raise OSError("offline left release write failure")
            return original(name, value, **kw)

        monkeypatch.setattr(driver, "write_register", write)
    else:
        def read(_name):
            if fault == "read":
                raise OSError("offline release read failure")
            return 10.  # Both zero writes acknowledged, physical readback disagrees.

        monkeypatch.setattr(driver, "read_register", read)
    with pytest.raises((OSError, RuntimeError)):
        rt._service_follow_wheels()
    assert driver.pairs == before
    assert driver.stops[-1] == 1
    assert rt.backend._parking_current_uncertain
    assert not owner.motor_io_lock.locked()


@pytest.mark.parametrize("new_base,new_yaw", [(16., 0.), (30., -2.), (30., 2.)])
def test_new_axes_during_release_are_rebuilt_without_reparking(monkeypatch, new_base, new_yaw):
    rt, owner, driver, clock, publish = parked_writer(monkeypatch, base=24.)

    def update():
        clock[0] += .02
        publish(new_base, new_yaw)
        rt._steering_feedback = feedback(clock[0])

    assert change_on_release_read(monkeypatch, driver, update) == []
    rt._service_follow_wheels()
    assert driver.pairs == [(26, -22), (int(new_base+new_yaw), -int(new_base-new_yaw))]
    assert driver.stops == [1]
    assert rt.backend.parking_current_a == 0.
    assert owner._depth30_linear_snapshot[3] == clock[0]


@pytest.mark.parametrize("before", ["stop", "hazard", "expiry"])
def test_released_park_is_not_cleared_after_precommit_revocation(monkeypatch, before):
    rt, owner, driver, clock, _ = parked_writer(monkeypatch)
    original = rt._begin_follow_commit
    calls = []

    def commit():
        result = original()
        if not calls:
            calls.append(True)
            if before == "stop":
                owner._explicit_stop_requested = True
            elif before == "hazard":
                rt.hard_stop_check = lambda _: True
            else:
                clock[0] += .251
        return result

    monkeypatch.setattr(rt, "_begin_follow_commit", commit)
    rt._service_follow_wheels()
    assert calls and len(driver.pairs) == 1
    assert rt.backend.parking_current_a == 10.
    assert all(value == 10. for _, value, _ in driver.register_writes)


@pytest.mark.parametrize("yaw", [-4., 4.])
@pytest.mark.parametrize("current", [0., 10.])
def test_yaw_only_publication_during_parking_exit_does_not_reapply_current(monkeypatch, yaw, current):
    rt, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    state[:] = [0., yaw, 10.25, 10.25]
    rt.get_steering_feedback = lambda: feedback(clock[0])
    rt.backend.config = replace(rt.backend.config, parking_current_a=current)
    rt.backend.send_stop("released_park", mode="emergency",
                         preserve_zero=True, prepare_parking_current=True)

    def update():
        owner._lateral_yaw_revision += 1
        state[1] = yaw * .5

    # 0A/FREE can preserve the ordinary STOP latch without requiring current
    # register I/O. An update during prepare must still rebuild that case.
    original = rt.backend.prepare_speed_mode
    seen = []

    def prepare():
        original()
        if not seen:
            seen.append(True)
            update()

    monkeypatch.setattr(rt.backend, "prepare_speed_mode", prepare)
    rt._service_follow_wheels()
    assert seen and driver.pairs == [(int(yaw*.5), int(yaw*.5))]
    assert driver.stops == [1]
    assert rt.backend.parking_current_a == 0.
    assert driver.register_writes == ([
        ("right_parking_current", 10., False), ("left_parking_current", 10., False),
        ("right_parking_current", 0., False), ("left_parking_current", 0., False)] if current else [])


@pytest.mark.parametrize("fault", ["stop", "hazard", "yaw_expired", "uid", "explicit_stop"])
def test_yaw_update_during_release_cannot_bypass_real_revocation(monkeypatch, fault):
    rt, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    rt.backend.config = replace(rt.backend.config, stop_mode="emergency")
    state[:] = [0., 4., 10.25, 10.25]
    rt.get_steering_feedback = lambda: feedback(clock[0])
    rt.backend.send_stop("released_park", mode="emergency",
                         preserve_zero=True, prepare_parking_current=True)

    def update():
        owner._lateral_yaw_revision += 1
        state[1] = 2.
        if fault == "stop":
            rt.backend.send_stop("newer_stop", mode="emergency", preserve_zero=True)
        elif fault == "hazard":
            rt.hard_stop_check = lambda _: True
        elif fault == "yaw_expired":
            clock[0] += .251
        elif fault == "uid":
            owner._follow_controller.active_target_id = 2
        else:
            owner._explicit_stop_requested = True

    seen = change_on_release_read(monkeypatch, driver, update)
    rt._service_follow_wheels()
    assert seen and driver.pairs == []
    assert driver.stops == [1, 1]
    assert not owner.motor_io_lock.locked()
