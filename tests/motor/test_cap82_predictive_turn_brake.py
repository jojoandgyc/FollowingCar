"""Real parking executor, fake clock/encoder/serial; never operates hardware."""
from dataclasses import replace

import pytest

from car_control_modular.control_types import SteeringFeedback
from car_control_modular.near_yaw_parking import NearYawParkRequest
from test_follow_wheel_periodic import setup_periodic


def setup(monkeypatch, sign=1, rpm=2):
    r, o, d, s, clock, axes = setup_periodic(monkeypatch)
    axes[:2] = [0, 0]
    req = NearYawParkRequest(1, 82, 9.89, 10., "predictive_countersteer_then_normal",
                            -rpm*sign, 13.67*sign, -28.01*sign, 10.14)
    o._near_yaw_park_request = req
    sample = [SteeringFeedback(timestamp=9.99, trustworthy=True,
        left_forward_rpm=8*sign, right_forward_rpm=-5*sign,
        yaw_rate_right_dps=17.9*sign, raw_yaw_rate_right_dps=21.15*sign)]
    r.get_steering_feedback = lambda: sample[0]
    return r, o, d, clock, req, sample


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("rpm", [2, 3, 6])
def test_cap82_opposite_braking_packet_then_normal_not_cross_wait(monkeypatch, sign, rpm):
    r, o, d, clock, req, sample = setup(monkeypatch, sign, rpm)
    r._service_follow_wheels()
    assert d.pairs[-1] == (-rpm*sign, -rpm*sign)  # raw right polarity
    assert not d.stops
    assert r._near_yaw_park_settling is None
    assert not r.near_yaw_park_release_ready(req, 10.01, 10.01)
    before = list(d.pairs)
    r._send_follow_wheel_targets(15, 15, "OLD_TURN")
    assert d.pairs == before
    for t in [10.02, 10.04, 10.07]:
        clock[0] = t
        sample[0] = replace(sample[0], timestamp=t)
        r._service_follow_wheels()
        assert d.pairs == before and not d.stops
        assert r._predictive_turn_brake_started == 10
    clock[0] = 10.081
    sample[0] = replace(sample[0], timestamp=clock[0])
    r._service_follow_wheels()
    assert d.stops == [1, 0]  # NORMAL, 5A setup handled by the existing backend
    assert r.backend.parking_current_a == 5
    assert r._near_yaw_park_settling.sent_at == pytest.approx(10.081)
    assert not r.near_yaw_park_release_ready(req, 10.12, 10.12)
    r._service_follow_wheels()
    assert d.stops == [1, 0]  # no pulse replay after parking


@pytest.mark.parametrize("bad", ["stale", "untrusted", "nan", "high", "translation",
                                  "wrong_sign", "center", "old_capture", "lease", "uid"])
@pytest.mark.parametrize("rpm", [2, 6])
def test_bad_pulse_evidence_falls_back_to_normal_without_opposite_packet(monkeypatch, bad, rpm):
    r, o, d, clock, req, sample = setup(monkeypatch, rpm=rpm)
    if bad == "stale": sample[0] = replace(sample[0], timestamp=9.8)
    if bad == "untrusted": sample[0] = replace(sample[0], trustworthy=False)
    if bad == "nan": sample[0] = replace(sample[0], raw_yaw_rate_right_dps=float("nan"))
    if bad == "high": sample[0] = replace(sample[0], left_forward_rpm=38)
    if bad == "translation": sample[0] = replace(sample[0], left_forward_rpm=15, right_forward_rpm=5)
    if bad == "wrong_sign": sample[0] = replace(sample[0], raw_yaw_rate_right_dps=-20)
    if bad == "center": o._near_yaw_park_request = replace(req, visual_error_deg=5)
    if bad == "old_capture": o._near_yaw_park_request = replace(req, capture_timestamp=9.7)
    if bad == "lease": o._near_yaw_park_request = replace(req, countersteer_until=9.99)
    if bad == "uid": o._follow_controller.active_target_id = 2
    r._service_follow_wheels()
    assert all(pair == (0, 0) for pair in d.pairs) and d.stops == [1, 0]


@pytest.mark.parametrize("change", ["stopped", "reversed", "expired", "lost", "danger", "shutdown", "explicit"])
@pytest.mark.parametrize("rpm", [2, 6])
def test_pulse_ends_early_without_minimum_on_time(monkeypatch, change, rpm):
    r, o, d, clock, req, sample = setup(monkeypatch, rpm=rpm)
    r._service_follow_wheels()
    clock[0] = 10.01
    if change == "stopped": sample[0] = replace(sample[0], raw_yaw_rate_right_dps=2)
    elif change == "reversed": sample[0] = replace(sample[0], raw_yaw_rate_right_dps=-6)
    elif change == "expired": sample[0] = replace(sample[0], timestamp=9.8)
    elif change == "lost": o._vision_control_state = "target_lost"
    elif change == "shutdown": o._runtime_shutdown_requested = True
    elif change == "explicit": o._explicit_stop_requested = True
    else: r.hard_stop_check = lambda action: True
    r._service_follow_wheels()
    assert d.stops and [p for p in d.pairs if p != (0, 0)] == [(-rpm, -rpm)]


def test_mode_io_expiry_does_not_emit_old_braking_pulse(monkeypatch):
    r, o, d, clock, req, sample = setup(monkeypatch)
    original = r.backend.prepare_speed_mode
    def slow():
        original()
        clock[0] += .20
    r.backend.prepare_speed_mode = slow
    r._service_follow_wheels()
    assert all(pair == (0, 0) for pair in d.pairs) and d.stops == [1, 0]


def test_partial_write_failure_stops_and_cannot_retry_pulse(monkeypatch):
    r, o, d, clock, req, sample = setup(monkeypatch)
    def fail(*args, **kwargs):
        raise OSError("fake partial serial write")
    r.backend.send_targets = fail
    with pytest.raises(OSError):
        r._service_follow_wheels()
    assert d.stops  # emergency attempted immediately
    r._service_follow_wheels()  # no send_targets retry
    assert r._near_yaw_park_applied is req


def test_danger_during_mode_io_uses_emergency_without_relocking(monkeypatch):
    r, o, d, clock, req, sample = setup(monkeypatch)
    original = r.backend.prepare_speed_mode
    def hazard():
        original()
        r.hard_stop_check = lambda action: True
    r.backend.prepare_speed_mode = hazard
    r._service_follow_wheels()
    assert d.stops == [1]
    assert all(pair == (0, 0) for pair in d.pairs)
    assert o._brake_hold_label.startswith("safety_hold_")


def test_out_of_order_feedback_ends_pulse_even_if_still_fresh(monkeypatch):
    r, o, d, clock, req, sample = setup(monkeypatch)
    r._service_follow_wheels()
    clock[0] = 10.01
    sample[0] = replace(sample[0], timestamp=9.98)
    r._service_follow_wheels()
    assert d.stops == [1, 0]


def test_slow_pulse_write_does_not_wait_another_tick_to_stop(monkeypatch):
    r, o, d, clock, req, sample = setup(monkeypatch)
    original = r.backend.send_targets
    def slow(*args, **kwargs):
        original(*args, **kwargs)
        clock[0] += .085
    r.backend.send_targets = slow
    r._service_follow_wheels()
    assert d.stops == [1, 0]
    assert r._near_yaw_park_settling is not None
