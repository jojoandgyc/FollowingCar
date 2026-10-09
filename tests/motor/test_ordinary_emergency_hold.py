"""0A ordinary EMERGENCY hold: fake serial/clock, never real hardware."""
from dataclasses import replace
import os
from pathlib import Path

import pytest

from test_follow_wheel_periodic import setup_periodic
from test_near_yaw_park_execution import request_park
from test_search_handoff_execution import arm
from test_visible_wheel_continuity import feedback
from test_cap82_predictive_turn_brake import setup as countersteer_setup


def test_active_profile_selects_ten_amp_emergency(monkeypatch):
    from car_control_modular.config_loader import load_config_to_env
    env = {}
    monkeypatch.setattr(os, 'environ', env)
    load_config_to_env(str(Path(__file__).resolve().parents[2] /
                           'car_control_modular/config/reid_runtime.ini'))
    assert env['MOTOR_RS485_STOP_MODE'] == 'emergency'
    assert float(env['MOTOR_PARKING_CURRENT_A']) == 10.


def setup(monkeypatch, search=False):
    rt, owner, driver, symbols, clock, axes = setup_periodic(monkeypatch)
    rt.backend.config = replace(rt.backend.config, stop_mode='emergency', parking_current_a=0.)
    axes[:2] = [0, 0]
    if search:
        arm(rt, owner, clock)
    else:
        request_park(owner, clock)
    rt._service_follow_wheels()
    return rt, owner, driver, symbols, clock, axes


@pytest.mark.parametrize('search', [False, True])
def test_single_emergency_no_normal_and_refresh_cannot_reenter_after_free(monkeypatch, search):
    rt, owner, driver, symbols, clock, _ = setup(monkeypatch, search)
    label = 'search_reacquire_brake' if search else 'near_yaw_park'
    evidence = rt._search_reacquire_settling if search else rt._near_yaw_park_settling
    assert owner._brake_hold_stop_mode == 'emergency'
    assert driver.stops == [1] and not driver.pairs and not driver.register_writes
    for t in (10.1, 10.499):
        clock[0] = t
        rt.send_percent_brake('emergency', label)
        rt._service_follow_wheels()
        assert driver.stops == [1]
    clock[0] = 10.50
    rt.send_percent_brake('emergency', label)
    assert driver.stops == [1, 2] and evidence.current_released_at == 10.50
    for mode in ('normal', 'emergency'):
        rt.send_percent_brake(mode, label)
    rt.send_percent_drive(0)
    rt.send_robot_command(symbols.rotate_left)
    assert driver.stops == [1, 2] and not driver.pairs
    rt.get_steering_feedback = lambda: feedback(clock[0], 0, 0)
    for t in (12.51, 12.57):
        clock[0] = t
        rt._service_follow_wheels()
    if search:
        assert not rt.search_reacquire_brake_pending(capture_timestamp=clock[0])
    else:
        assert rt.near_yaw_park_release_ready(evidence.request, clock[0], clock[0])
    assert rt.backend.parking_release_fault is None and driver.stops == [1, 2]


@pytest.mark.parametrize('search', [False, True])
@pytest.mark.parametrize('protected', ['safety', 'explicit', 'shutdown'])
def test_protected_stop_never_uses_ordinary_timed_release(monkeypatch, search, protected):
    rt, owner, driver, _, clock, _ = setup(monkeypatch, search)
    if protected == 'safety':
        rt.send_stop_with_brake_hold('hard_stop')
    elif protected == 'explicit':
        owner._explicit_stop_requested = True
    else:
        owner._runtime_shutdown_requested = True
    before = list(driver.stops)
    clock[0] = 12.51
    rt._service_follow_wheels()
    assert 2 not in driver.stops and 0 not in driver.stops
    assert driver.stops == before + ([1] if protected == 'shutdown' else [])
    assert not driver.pairs
    # A newly latched shutdown explicitly confirms STOP, but cannot start an
    # ordinary FREE release or burst-repeat the shutdown transaction.
    after = list(driver.stops)
    rt._service_follow_wheels()
    assert driver.stops == after and not driver.pairs


def test_near_park_can_replace_search_owned_emergency(monkeypatch):
    rt, owner, driver, _, clock, _ = setup(monkeypatch, True)
    clock[0] += .1
    req = request_park(owner, clock)
    rt._service_follow_wheels()
    assert rt._near_yaw_park_settling.request is req
    assert owner._brake_hold_label == 'near_yaw_park'
    assert driver.stops == [1, 1] and not driver.pairs


def test_countersteer_ends_at_emergency_not_normal(monkeypatch):
    rt, _, driver, clock, _, sample = countersteer_setup(monkeypatch)
    rt.backend.config = replace(rt.backend.config, stop_mode='emergency', parking_current_a=0.)
    rt._service_follow_wheels()
    assert driver.pairs == [(-2, -2)] and not driver.stops
    clock[0] = 10.081
    sample[0] = replace(sample[0], timestamp=clock[0])
    rt._service_follow_wheels()
    assert driver.stops == [1] and driver.pairs == [(-2, -2)]
    assert not driver.register_writes


def test_explicitly_preserved_emergency_rejects_zero_keepalive(monkeypatch):
    rt, _, driver, _, _, _ = setup_periodic(monkeypatch)
    rt.backend.send_stop('ordinary_cross', mode='emergency', preserve_zero=True)
    rt.backend.send_targets(0, 0, 'stale_zero')
    assert driver.stops == [1] and not driver.pairs


@pytest.mark.parametrize('search', [False, True])
@pytest.mark.parametrize('current_a', [5., 10.])
def test_configured_current_emergency_entry_then_zero_amp_free_and_reentry(monkeypatch, search, current_a):
    rt, owner, driver, _, clock, axes = setup_periodic(monkeypatch)
    rt.backend.config = replace(rt.backend.config, stop_mode='emergency', parking_current_a=current_a)
    axes[:2] = [0, 0]
    writes_at_stop = []
    original = driver.stop_all
    def stopped(mode):
        writes_at_stop.append((int(mode), dict(driver.registers)))
        original(mode)
    monkeypatch.setattr(driver, 'stop_all', stopped)
    if search:
        arm(rt, owner, clock)
    else:
        request_park(owner, clock)
    rt._service_follow_wheels()
    assert driver.stops == [1] and not driver.pairs
    assert all(v == current_a for v in writes_at_stop[0][1].values())
    assert rt.backend.parking_current_a == current_a
    for t in (10.1, 10.499):
        clock[0] = t
        rt._service_follow_wheels()
        assert rt.backend.parking_current_a == current_a
    clock[0] = 10.50
    rt._service_follow_wheels()
    assert driver.stops == [1, 2]
    assert all(v == 0. for v in writes_at_stop[-1][1].values())
    clock[0] = 10.70
    request_park(owner, clock)
    rt._service_follow_wheels()
    assert driver.stops == [1, 2, 1]
    assert all(v == current_a for v in writes_at_stop[-1][1].values())
    assert not driver.pairs and not any(persist for _, _, persist in driver.register_writes)


@pytest.mark.parametrize('current_a', [5., 10.])
def test_startup_configured_current_emergency_never_sends_normal(monkeypatch, current_a):
    rt, _, driver, _, _, _ = setup_periodic(monkeypatch)
    rt.backend.config = replace(rt.backend.config, stop_mode='emergency', parking_current_a=current_a)
    monkeypatch.setattr('car_control_modular.mssd_motor.time.sleep', lambda _: None)
    rt.backend.enable_startup_parking()
    assert driver.stops == [1, 1]  # initial safe clear, then configured-current STOP
    assert rt.backend.parking_current_a == current_a
    assert all(v == current_a for v in driver.registers.values())
    rt.backend.send_targets(8, 8, 'new_motion')
    assert rt.backend.parking_current_a == 0.


@pytest.mark.parametrize('fault', [False, True])
def test_safety_emergency_never_waits_for_five_amp_setup(monkeypatch, fault):
    rt, _, driver, _, _, _ = setup_periodic(monkeypatch)
    rt.backend.config = replace(rt.backend.config, stop_mode='emergency', parking_current_a=5.)
    if fault:
        rt.backend.parking_release_fault = 'free_stop_failed'
    rt.backend.send_stop('safety', mode='emergency', prepare_parking_current=fault)
    assert driver.stops == [1] and not driver.register_writes and not driver.pairs


def test_five_amp_readback_failure_stops_without_successful_hold(monkeypatch):
    rt, _, driver, _, _, _ = setup_periodic(monkeypatch)
    rt.backend.config = replace(rt.backend.config, stop_mode='emergency', parking_current_a=5.)
    monkeypatch.setattr(driver, 'read_register', lambda _: 0.)
    with pytest.raises(RuntimeError):
        rt.backend.send_stop('ordinary', mode='emergency', prepare_parking_current=True,
                             preserve_zero=True)
    assert driver.stops == [1] and not driver.pairs
    assert not rt.backend.motion_armed and not rt.backend.normal_zero_hold
