"""CAP217-366: bounded ordinary correction + forward-only integral capacity."""
from dataclasses import replace

import pytest
import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction, ControlDecision, ObstacleState
from car_control_modular.steering_pid import LongitudinalDistancePid, DistancePidConfig
from test_longitudinal_catchup import limits
from test_distance_tracking_response import setup, decide
from test_longitudinal_authority_runtime import owner, _frame, NOW


def pid(forward_limit=7.5):
    return LongitudinalDistancePid(DistancePidConfig(
        kp_rpm_per_m=24, ki_rpm_per_m_s=.8, kd_rpm_s_per_m=0,
        integral_limit_m_s=1.5, forward_integral_limit_m_s=forward_limit,
        deadband_m=.03,
    ))


def test_positive_offset_can_accumulate_six_rpm_but_reverse_stays_legacy():
    controller = pid()
    for i in range(400):
        result = controller.update(1.85, 1.5, now=100+i*.1)
    assert result.i_rpm == pytest.approx(6)
    assert result.integral_cap_rpm == 6
    for i in range(400):
        result = controller.update(1.1, 1.5, now=150+i*.1)
    assert result.i_rpm == pytest.approx(-1.2)
    assert result.integral_cap_rpm == pytest.approx(1.2)


@pytest.mark.parametrize("base", [None, 25.0])
def test_target_band_and_reset_do_not_carry_large_positive_integral(base):
    controller = pid()
    for i in range(250):
        controller.update(1.9, 1.5, now=100+i*.1, tracking_base_rpm=base)
    result = controller.update(1.5, 1.5, now=126, tracking_base_rpm=base)
    assert result.i_rpm <= 1.2 + 1e-6
    if base is None:
        assert result.output_rpm == 0
    controller.reset()
    assert controller.update(1.9, 1.5, now=127).i_rpm < .1


def test_duplicate_and_external_limit_cannot_accumulate_extra_integral():
    controller = pid()
    first = controller.update(1.85, 1.5, now=100)
    assert controller.update(1.85, 1.5, now=100) is first
    for i in range(1, 300):
        result = controller.update(1.85, 1.5, now=100+i*.1)
        controller.accept_output_limit(100+i*.1, 20)
    assert result.i_rpm < .1


def cap(distance, requested, budget=None, kind="forward"):
    return runtime.PersonTracker._cap_depth_longitudinal_actions(
        [getattr(ControlAction, kind)(requested, "test")], distance,
        distance_control_percent=budget,
    )[0].speed_percent


def test_current_ordinary_pid_is_not_cut_to_twenty_at_18m(limits, monkeypatch):
    monkeypatch.setattr(runtime, "FOLLOW_FORWARD_START_DISTANCE_M", 1.58)
    assert cap(1.8, 29) == 20
    assert cap(1.8, 29, 29) == 29
    assert cap(1.8, 99, 99) == 40  # launch20 + at most20 correction
    assert cap(1.8, 12, 29) == 12  # no forced minimum or acceleration
    assert cap(1.8, 0, 29) == 0
    assert cap(1.5, 80, 80) == 20  # no new budget in setpoint band
    assert cap(1.8, 99, 99, "backward") == 20
    assert cap(3, 99, 99) == 60
    monkeypatch.setattr(runtime, "FORWARD_MAX_RPM", 200)
    assert cap(1.8, 29, 29) == 20  # 40 physical RPM, not 40 percent


@pytest.mark.parametrize("budget", [None, float("nan"), float("inf"), -1, 0])
def test_missing_or_invalid_pid_budget_retains_original_cap(limits, budget):
    assert cap(1.8, 80, budget) == 20


def test_real_controller_only_exposes_current_trusted_distance_pid(setup):
    clock, controller, frame = setup
    observation = frame(1.85, yaw=20)  # no FF; still a reliable range
    decide(controller, observation)
    assert controller.distance_only_forward_percent(observation, clock.now) > 20
    assert controller.distance_only_forward_percent(observation, clock.now-.03) == 0
    assert controller.distance_only_forward_percent(
        replace(observation, obstacles=ObstacleState(front=True)), clock.now) == 0
    assert controller.distance_only_forward_percent(
        replace(observation, distance_state=replace(observation.distance_state,
                                                  source_detail="distance_jump_pending_1_of_3")),
        clock.now) == 0


def test_current_budget_commit_preserves_physical_deadline(owner, limits):
    owner._follow_controller.distance_only_forward_percent = lambda frame, stamp: 29
    actions, accepted = owner._commit_depth_linear_decision(
        ControlDecision(actions=[ControlAction.forward(29, "longitudinal_distance_pid")],
                        reason="longitudinal_distance_pid"),
        _frame(distance=1.85), 1, is_fresh_depth=True,
    )
    assert accepted and actions[0].speed_percent == 29
    assert owner._depth30_linear_snapshot[3] == NOW-.04
    assert owner._depth30_linear_timing.depth_expires_at == pytest.approx(NOW-.04+.18)


def test_small_idealized_plant_reduces_bias_without_raising_global_speed(limits, monkeypatch):
    """Synthetic 30RPM person, 0.6m/rev wheel, 150ms wheel response; not hardware."""
    monkeypatch.setattr(runtime, "FOLLOW_FORWARD_START_DISTANCE_M", 1.58)
    def simulate(new):
        controller = pid(7.5 if new else 0)
        distance, speed = 1.85, 30.0
        errors = []
        for i in range(600):
            result = controller.update(distance, 1.5, now=100+i*.05)
            command = cap(distance, result.output_rpm, result.output_rpm if new else None)
            controller.accept_output_limit(100+i*.05, command)
            speed += (command-speed)*(.05/.15)
            distance += (30-speed)*.6/60*.05
            if i >= 500:
                errors.append(distance-1.5)
        return sum(errors)/len(errors)
    assert simulate(True) < simulate(False)*.65
