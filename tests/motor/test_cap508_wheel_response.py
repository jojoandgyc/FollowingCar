"""Align measured differential with past writes, never the just-issued one."""
import logging
from types import SimpleNamespace

import pytest

from car_control_modular.wheel_response import WheelDifferentialResponse
from test_visible_wheel_continuity import visible_runtime, feedback


def test_does_not_attribute_old_feedback_to_new_command():
    probe = WheelDifferentialResponse()
    first = probe.observe_and_note(1, (98, 82), 10, 0, feedback(9.98, 0, 0))
    assert first['reference_diff'] is None
    second = probe.observe_and_note(1, (80, 80), 10.05, 10, feedback(10.03, 30, 28))
    assert second['reference_diff'] == 16
    assert second['measured_diff'] == 2
    assert second['reference_sent'] == 10
    assert not second['lagging']


@pytest.mark.parametrize('side', [-1, 1])
def test_flags_sustained_small_response_once_per_feedback(side):
    probe = WheelDifferentialResponse()
    pair = (90 + side*8, 90 - side*8)
    previous = 0
    for t in [10, 10.08, 10.16, 10.24, 10.32, 10.40, 10.48, 10.56]:
        result = probe.observe_and_note(1, pair, t, previous, feedback(t-.01, 20+side, 20))
        previous = t
        if t < 10.5: assert not result['lagging']
    assert result['lagging'] and result['sustained_ms'] == pytest.approx(550)
    duplicate = probe.observe_and_note(1, pair, 10.57, 10.56, feedback(10.55, 20+side, 20))
    assert not duplicate['new_feedback'] and not duplicate['lagging']


@pytest.mark.parametrize('kind', ['uid', 'writer_reset', 'gap', 'zero', 'reverse'])
def test_discontinuity_does_not_inherit_sustained_response(kind):
    probe = WheelDifferentialResponse()
    probe.observe_and_note(1, (98, 82), 10, 0, None)
    pair = {'zero': (0, 0), 'reverse': (82, 98)}.get(kind, (98, 82))
    uid = 2 if kind == 'uid' else 1
    previous = 0 if kind == 'writer_reset' else 10
    t = 10.3 if kind == 'gap' else 10.1
    probe.observe_and_note(uid, pair, t, previous, None)
    r = probe.observe_and_note(uid, pair, t+.03, t, feedback(t+.02, 0, 0))
    assert r['sustained_ms'] == pytest.approx(20)
    assert not r['lagging']


@pytest.mark.parametrize('bad', [None, feedback(9), feedback(10.3),
                               feedback(10, trustworthy=False), feedback(10, float('nan'), 0)])
def test_bad_feedback_has_no_response_claim(bad):
    probe = WheelDifferentialResponse()
    probe.observe_and_note(1, (98, 82), 10, 0, None)
    r = probe.observe_and_note(1, (98, 82), 10.1, 10, bad)
    assert not r['new_feedback'] and r['reference_diff'] is None


def test_actual_writer_logs_reference_without_extra_io(monkeypatch, caplog):
    motor, owner, driver, _, clock = visible_runtime(monkeypatch)
    reads = []
    def read():
        reads.append(clock[0])
        return feedback(clock[0]-.01, 24, 24)
    motor.get_steering_feedback = read
    caplog.set_level(logging.INFO)
    for t in (10., 10.05):
        clock[0] = t
        with owner.motor_io_lock:
            motor._send_follow_wheel_targets(32, -16, 'STEER')
    assert driver.pairs == [(32, -16), (32, -16)]
    assert len(reads) == 2
    assert 'response_reference_diff_rpm=16' in caplog.text
    assert not driver.stops
