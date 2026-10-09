"""NORMAL entry ordering, using fake transport only (no serial access)."""
from enum import IntEnum

import pytest

from test_depth_drive_rpm import make_runtime
from test_follow_wheel_periodic import setup_periodic
from test_near_yaw_park_execution import request_park
from test_search_handoff_execution import arm


class Modes(IntEnum):
    NORMAL = 0
    EMERGENCY = 1
    FREE = 2


def trace(monkeypatch, driver):
    events = []
    for name in ('set_right_speed', 'set_left_speed', 'stop_all',
                 'write_register', 'read_register'):
        original = getattr(driver, name)
        def wrapped(*args, _name=name, _original=original, **kwargs):
            events.append((_name, *args))
            return _original(*args, **kwargs)
        monkeypatch.setattr(driver, name, wrapped)
    return events


@pytest.mark.parametrize('enum_modes', [False, True])
@pytest.mark.parametrize('cached', [False, True])
def test_emergency_precedes_current_preparation_and_normal(monkeypatch, enum_modes, cached):
    rt, _, driver, _ = make_runtime()
    backend = rt.backend
    if enum_modes:
        backend.classes = (object, Modes)
    if cached:
        backend.set_parking_current(5., persist=False)
    events = trace(monkeypatch, driver)
    backend.send_stop('trial', mode='normal', preserve_zero=True)
    expected = [('set_right_speed', 0), ('set_left_speed', 0), ('stop_all', 1)]
    if not cached:
        expected += [('write_register', 'right_parking_current', 5.),
                     ('write_register', 'left_parking_current', 5.),
                     ('read_register', 'right_parking_current'),
                     ('read_register', 'left_parking_current')]
    assert events == expected + [('stop_all', 0)]
    assert backend.normal_zero_hold and not backend.motion_armed
    events.clear()
    backend.refresh_normal_stop('refresh')
    assert events == [('stop_all', 0)]  # no repeated emergency in held NORMAL


@pytest.mark.parametrize('failed_mode', [1, 0])
def test_failed_stop_cannot_claim_normal_hold(monkeypatch, caplog, failed_mode):
    rt, _, driver, _ = make_runtime()
    backend = rt.backend
    backend.motion_armed = True
    calls = []
    def fail(mode):
        calls.append(int(mode))
        if mode == failed_mode:
            raise OSError('simulated partial dual-wheel STOP acknowledgement')
    monkeypatch.setattr(driver, 'stop_all', fail)
    with caplog.at_level('INFO'), pytest.raises(OSError):
        backend.send_stop('trial', mode='normal', preserve_zero=True)
    # A failed NORMAL must end in best-effort EMERGENCY, never another speed
    # transaction; a failed EMERGENCY has already tried both available sides.
    assert calls == ([1] if failed_mode == 1 else [1, 0, 1])
    assert backend.motion_write_fault
    assert not backend.motion_armed and not backend.normal_zero_hold
    assert 'LZ30EMA 停车命令:' not in caplog.text
    if failed_mode == 1:
        assert not driver.register_writes


@pytest.mark.parametrize('search', [False, True])
def test_hold_clock_starts_after_normal_not_pre_emergency(monkeypatch, search):
    rt, owner, driver, _, clock, _ = setup_periodic(monkeypatch)
    if search:
        arm(rt, owner, clock)
    else:
        request_park(owner, clock)
    original = driver.stop_all
    def delayed(mode):
        clock[0] += .04 if mode == 1 else .02
        original(mode)
    monkeypatch.setattr(driver, 'stop_all', delayed)
    rt._service_follow_wheels()
    evidence = rt._search_reacquire_settling if search else rt._near_yaw_park_settling
    assert driver.stops == [1, 0]
    assert evidence.sent_at == pytest.approx(10.06)
    clock[0] = 10.55
    rt._service_follow_wheels()
    assert rt.backend.parking_current_a == 5.
    assert driver.stops == [1, 0]
