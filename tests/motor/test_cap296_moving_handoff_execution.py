"""Real periodic/direct writer with fake encoder and serial. No hardware."""
from dataclasses import replace

import pytest

from car_control_modular.controllers import FollowPolicyConfig
from car_control_modular.control_types import SteeringFeedback
from car_control_modular.search_reacquire_braking import MovingHandoffEvidence
from car_control_modular.lateral_intent import LateralIntentStore
from test_cap837_turn_buildup import image_intent
from test_follow_wheel_periodic import setup_periodic
import request_0513_modular as main


def setup(monkeypatch):
    runtime, owner, driver, symbols, clock, state = setup_periodic(monkeypatch)
    owner._follow_controller.cfg = FollowPolicyConfig(
        visible_steering_pid_camera_hfov_deg=66.,
        visible_steering_pid_predictive_brake_decel_dps2=60.,
        visible_steering_pid_predictive_brake_response_sec=.05,
        visible_steering_pid_predictive_countersteer_max_correction_rpm=6.,
        visible_steering_pid_predictive_countersteer_gain_rpm_per_dps=.25)
    owner._search_handoff_uid = 1
    owner._search_handoff_moving_active = True
    owner._search_handoff_cap_rpm = 7.
    owner._search_handoff_started_capture_ts = 9.7
    owner._search_handoff_moving_evidence = MovingHandoffEvidence(1, 1, 296, 9.9, .55, "right", .21)
    state[:] = [42., 7., 10.3, 10.3]
    runtime.get_steering_feedback = lambda: SteeringFeedback(timestamp=clock[0], trustworthy=True,
        left_forward_rpm=42., right_forward_rpm=24., raw_yaw_rate_right_dps=29., yaw_rate_right_dps=26.)
    return runtime, owner, driver, symbols, clock, state


@pytest.mark.parametrize("periodic", [True, False])
def test_forward_stays_positive_while_residual_right_yaw_gets_counter_differential(monkeypatch, periodic):
    runtime, owner, driver, _, _, _ = setup(monkeypatch)
    if periodic:
        runtime._service_follow_wheels()
    else:
        runtime.config.follow_wheel_period_sec = 0.
        with owner.motor_io_lock:
            runtime._send_follow_wheel_targets(49, -35, "DIRECT")
    assert len(driver.pairs) == 1 and not driver.stops
    left, raw_right = driver.pairs[0]
    assert 0 < left < -raw_right
    assert left-raw_right == 84  # original 42 RPM base preserved
    assert -raw_right-left <= 12


def test_next_tick_uses_new_encoder_not_cached_reverse_yaw(monkeypatch):
    runtime, _, driver, _, clock, _ = setup(monkeypatch)
    runtime._service_follow_wheels()
    clock[0] += .05
    runtime.get_steering_feedback = lambda: SteeringFeedback(timestamp=clock[0], trustworthy=True,
        left_forward_rpm=24., right_forward_rpm=24., raw_yaw_rate_right_dps=0., yaw_rate_right_dps=1.)
    runtime._service_follow_wheels()
    assert driver.pairs[-1] == (42, -42)


@pytest.mark.parametrize("veto", ["image_expired", "feedback_expired", "depth_expired", "yaw_expired", "reverse"])
def test_expired_proof_and_actual_wheel_reversal_remain_fail_closed(monkeypatch, veto):
    runtime, owner, driver, _, clock, state = setup(monkeypatch)
    if veto == "image_expired": clock[0] = 10.15
    if veto == "feedback_expired":
        old = runtime.get_steering_feedback()
        runtime.get_steering_feedback = lambda: replace(old, timestamp=9.8)
    if veto == "depth_expired": state[2] = 9.
    if veto == "yaw_expired": state[3] = 9.
    if veto == "reverse":
        old = runtime.get_steering_feedback()
        runtime.get_steering_feedback = lambda: replace(old, left_forward_rpm=-39., right_forward_rpm=-2.)
    runtime._service_follow_wheels()
    assert driver.pairs and all(pair == (0, 0) for pair in driver.pairs)


@pytest.mark.parametrize("veto", ["image", "feedback", "depth", "replace", "uid", "danger"])
def test_final_safety_callback_cannot_leave_obsolete_moving_packet(monkeypatch, veto):
    runtime, owner, driver, _, clock, state = setup(monkeypatch)
    def check(command):
        if veto == "image":
            clock[0] = 10.115
        if veto == "feedback":
            clock[0] = 10.105
            stale = runtime.get_steering_feedback()
            runtime.get_steering_feedback = lambda: replace(stale, timestamp=10.)
        if veto == "depth": state[2] = 9.
        if veto == "replace": owner._search_handoff_moving_evidence = replace(owner._search_handoff_moving_evidence, cap=299)
        if veto == "uid": owner._follow_controller.active_target_id = 2
        return veto == "danger"
    # Direct execution avoids the separate pre-entry callback: the mutation
    # happens after yaw calculation, inside the last hard-stop check.
    runtime.hard_stop_check = check
    runtime.config.follow_wheel_period_sec = 0.
    with owner.motor_io_lock:
        runtime._send_follow_wheel_targets(49, -35, "DIRECT")
    assert all(pair == (0, 0) for pair in driver.pairs)
    assert driver.pairs or driver.stops


@pytest.mark.parametrize("periodic", [True, False])
def test_final_new_feedback_recomputes_and_rebuilds_once(monkeypatch, periodic):
    runtime, owner, driver, _, clock, _ = setup(monkeypatch)
    owner._search_handoff_moving_evidence = replace(owner._search_handoff_moving_evidence, x=.65)
    old = runtime.get_steering_feedback()
    reads = []
    def read():
        reads.append(True)
        return replace(old, raw_yaw_rate_right_dps=15., yaw_rate_right_dps=15.) if len(reads) == 1 else old
    runtime.get_steering_feedback = read
    if periodic:
        runtime._service_follow_wheels()
        assert len(reads) == 4
        left, raw_right = driver.pairs[-1]
        assert 0 < left < -raw_right and left-raw_right == 84
    else:
        runtime.config.follow_wheel_period_sec = 0.
        with owner.motor_io_lock:
            runtime._send_follow_wheel_targets(49, -35, "DIRECT")
        assert driver.pairs == [(0, 0)]


def test_new_negative_wheel_feedback_cannot_escape_guard_when_yaw_pair_is_unchanged(monkeypatch):
    runtime, _, driver, _, _, _ = setup(monkeypatch)
    old = runtime.get_steering_feedback()
    reads = []
    def read():
        reads.append(True)
        return old if len(reads) == 1 else replace(old, left_forward_rpm=-39., right_forward_rpm=-2.)
    runtime.get_steering_feedback = read
    runtime._service_follow_wheels()
    assert driver.pairs == [(0, 0)]


@pytest.mark.parametrize("hold_zero", [False, True])
@pytest.mark.parametrize("residual", [0., 29.])
def test_production_zero_yaw_contract_keeps_legal_straight_base_when_settled(monkeypatch, hold_zero, residual):
    runtime, owner, driver, _, clock, state = setup(monkeypatch)
    state[1] = 0.
    owner._explicit_stop_requested = owner._runtime_shutdown_requested = False
    owner._lateral_intent_store = LateralIntentStore()
    owner._lateral_intent_store.publish(image_intent(10., 296,
        initial_correction_rpm=0, hold_zero=hold_zero, x_ratio=.5))
    owner._has_fresh_lateral_yaw = lambda uid: main.PersonTracker._has_fresh_lateral_yaw(owner, uid)
    owner._search_handoff_moving_evidence = replace(owner._search_handoff_moving_evidence, x=.5)
    runtime.get_steering_feedback = lambda: SteeringFeedback(timestamp=clock[0], trustworthy=True,
        left_forward_rpm=24., right_forward_rpm=24., raw_yaw_rate_right_dps=residual, yaw_rate_right_dps=residual)
    assert not owner._has_fresh_lateral_yaw(1)  # valid zero, NOT an expired UID
    runtime._service_follow_wheels()
    assert driver.pairs == ([(42, -42)] if residual == 0. else [(0, 0)])
    assert not driver.stops


def test_initial_depth_pending_release_does_not_take_ownership_of_yaw_only(monkeypatch):
    runtime, owner, driver, _, clock, state = setup(monkeypatch)
    owner._search_handoff_moving_active = False
    state[0] = 0.
    runtime.get_steering_feedback = lambda: SteeringFeedback(timestamp=clock[0], trustworthy=True,
        left_forward_rpm=0., right_forward_rpm=0., raw_yaw_rate_right_dps=0., yaw_rate_right_dps=0.)
    runtime._service_follow_wheels()
    assert driver.pairs == [(7, 7)]  # physical +7/-7 pivot; no forward permission
