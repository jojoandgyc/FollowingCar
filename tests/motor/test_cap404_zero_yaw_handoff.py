"""CAP404: current deliberate yaw zero must retain a fresh forward grant.

Production periodic/direct writers and fake serial only. Physical sample and
capture timestamps below come from run_20260928_223853, not refreshed leases.
"""
from dataclasses import replace

import pytest

import request_0513_modular as main
from car_control_modular.control_types import SteeringFeedback
from car_control_modular.lateral_intent import LateralIntentStore
from car_control_modular.search_reacquire_braking import MovingHandoffEvidence
from test_cap296_moving_handoff_execution import setup
from test_cap837_turn_buildup import image_intent


def cap404(monkeypatch, *, second_tick=False):
    rt, owner, driver, _, clock, state = setup(monkeypatch)
    clock[0] = 5559.831103741 if second_tick else 5559.769
    base = 66. if second_tick else 52.
    capture = 5559.638091262
    depth_stamp = 5559.752292174 if second_tick else 5559.71994046
    state[:] = [base, 0., depth_stamp+.25, capture+.21]
    owner._search_handoff_started_capture_ts = 5558.905694
    owner._follow_controller.cfg = replace(owner._follow_controller.cfg,
        center_left_ratio=.45, center_right_ratio=.55,
        visible_steering_pid_predictive_brake_margin_deg=1.25,
        visible_steering_pid_camera_latency_sec=.10,
        visible_steering_pid_image_error_only=True)
    owner._search_handoff_moving_evidence = MovingHandoffEvidence(
        1, 2, 404, capture, .283, "left", .21)
    owner._explicit_stop_requested = owner._runtime_shutdown_requested = False
    owner._lateral_intent_store = LateralIntentStore()
    owner._lateral_intent_store.publish(image_intent(
        5559.760, 404, capture_timestamp=capture, initial_correction_rpm=0,
        hold_zero=True, park_requested=True, x_ratio=.283,
        reason="visual_pid_direction_guard_hold"))
    owner._has_fresh_lateral_yaw = lambda uid: main.PersonTracker._has_fresh_lateral_yaw(owner, uid)
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: (
        ("forward", base/2., 1, depth_stamp)
        if uid == 1 and clock[0] <= state[2] else None)
    rt.config.motor_forward_max_target_rpm = 200
    rt.backend.config = replace(rt.backend.config, max_target=200)
    # The first physical left wheel is still finishing the search pivot.
    # Existing bounded forward-handoff checks continue to own its reversal.
    rt.config.follow_forward_handoff_enable = True
    rt.config.follow_residual_reverse_max_rpm = 8.
    rt.get_steering_feedback = lambda: SteeringFeedback(
        timestamp=5559.831103741 if second_tick else 5559.732054233,
        trustworthy=True, left_forward_rpm=0. if second_tick else -3.,
        right_forward_rpm=4. if second_tick else 6.,
        raw_yaw_rate_right_dps=-6.5088 if second_tick else -10.971,
        yaw_rate_right_dps=-14.1075)
    return rt, owner, driver, clock, state


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("second_tick", [True, False])
def test_actual_cap404_zero_yaw_preserves_approved_forward(monkeypatch, periodic, second_tick):
    rt, owner, driver, _, state = cap404(monkeypatch, second_tick=second_tick)
    assert not owner._has_fresh_lateral_yaw(1)
    if periodic:
        rt._service_follow_wheels()
    else:
        rt.config.follow_wheel_period_sec = 0.
        with owner.motor_io_lock:
            rt._send_follow_wheel_targets(state[0], -state[0], "CAP404")
    assert driver.pairs == [(state[0], -state[0])]
    assert not driver.stops


@pytest.mark.parametrize("veto", ["image", "intent", "cap", "stamp", "uid", "quality",
                                  "near", "missing", "no_zero", "depth", "residual",
                                  "initial_zero_only", "continuation"])
def test_current_zero_requires_full_evidence_and_safe_residual(monkeypatch, veto):
    rt, owner, driver, clock, state = cap404(monkeypatch)
    intent = owner._lateral_intent_store.snapshot()
    updates = {
        "intent": dict(valid_until=clock[0]-.001), "cap": dict(capture_frame_id=401),
        "stamp": dict(capture_timestamp=intent.capture_timestamp-.001),
        "uid": dict(target_id=2), "quality": dict(bbox_quality="limited"),
        "near": dict(near_distance_mode=True),
        "no_zero": dict(hold_zero=False, initial_correction_rpm=7),
        "initial_zero_only": dict(hold_zero=False),
        "continuation": dict(nominal_valid_until=clock[0]-.001),
    }
    if veto in updates:
        owner._lateral_intent_store.publish(replace(intent, **updates[veto]))
    if veto == "image": clock[0] = intent.capture_timestamp+.211
    if veto == "missing": owner._lateral_intent_store.clear()
    if veto == "depth": state[2] = clock[0]-.001
    if veto == "no_zero": owner._has_fresh_lateral_yaw = lambda uid: False
    if veto == "residual":
        # Existing braking envelope needs counter-differential; a zero intent
        # may never acquire that yaw authority merely by retaining Depth.
        fb = rt.get_steering_feedback()
        rt.get_steering_feedback = lambda: replace(
            fb, raw_yaw_rate_right_dps=-35., yaw_rate_right_dps=-35.)
    rt._service_follow_wheels()
    assert driver.pairs and all(pair == (0, 0) for pair in driver.pairs)


@pytest.mark.parametrize("veto", ["image", "intent", "replacement", "depth", "residual", "hard_stop"])
def test_final_safety_callback_rechecks_zero_contract(monkeypatch, veto):
    rt, owner, driver, clock, state = cap404(monkeypatch)
    intent = owner._lateral_intent_store.snapshot()
    def check(command):
        if veto == "image": clock[0] = intent.capture_timestamp+.211
        if veto == "intent":
            owner._lateral_intent_store.publish(replace(intent, valid_until=clock[0]-.001))
        if veto == "replacement":
            owner._lateral_intent_store.publish(replace(intent, capture_frame_id=405))
        if veto == "depth": state[2] = clock[0]-.001
        if veto == "residual":
            fb = rt.get_steering_feedback()
            rt.get_steering_feedback = lambda: replace(
                fb, raw_yaw_rate_right_dps=-35., yaw_rate_right_dps=-35.)
        return veto == "hard_stop"
    rt.hard_stop_check = check
    rt.config.follow_wheel_period_sec = 0.
    with owner.motor_io_lock:
        rt._send_follow_wheel_targets(52., -52., "CAP404")
    assert driver.pairs or driver.stops
    assert all(pair == (0, 0) for pair in driver.pairs)
