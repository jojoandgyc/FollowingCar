"""350ms is a prediction model, never an extra execution delay."""
from dataclasses import replace
import ast
import os
from pathlib import Path
import pytest
from car_control_modular.control_types import SteeringFeedback
from car_control_modular.steering_pid import VisualSteeringPid, VisualSteeringPidConfig
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from test_cap443_capture_braking import sample
from test_lateral_zero_runtime import owner, _intent, NOW
from types import SimpleNamespace


def config(trial=.35):
    return VisualSteeringPidConfig(enabled=True, image_error_only=True, image_brake_assist=True,
        image_capture_motion=True, execution_response_trial_sec=trial,
        camera_hfov_deg=66, deadband_deg=3, dynamic_small_error_deg=3,
        dynamic_large_error_deg=10, max_correction_rpm=10,
        predictive_brake_response_sec=.05, predictive_countersteer_response_sec=.08,
        predictive_brake_decel_dps2=60, predictive_countersteer_max_correction_rpm=6,
        predictive_countersteer_min_correction_rpm=3,
        predictive_countersteer_gain_rpm_per_dps=.25,
        predictive_countersteer_min_yaw_rate_dps=4)


def fb(now=10., sign=-1):
    return SteeringFeedback(timestamp=now-.02, trustworthy=True,
        left_forward_rpm=32+sign*9, right_forward_rpm=32-sign*9,
        yaw_rate_right_dps=sign*25, raw_yaw_rate_right_dps=sign*28)


@pytest.mark.parametrize("sign", [-1,1])
def test_trial_replaces_old_motor_budget_without_double_counting(sign):
    def run(c):
        return VisualSteeringPid(c).update(.5+sign*.15,40,fb(sign=sign),now=10,
            visual_age_sec=.12,target_image_rate_dps=-sign*20)
    r=run(config())
    assert r.prediction_latency_sec == pytest.approx(.12+.05+.35)
    assert r.stopping_distance_deg == pytest.approx(28*(.12+.05+.35))
    changed=run(replace(config(),image_motion_response_sec=.29,
        predictive_countersteer_response_sec=.01,predictive_brake_decel_dps2=120))
    assert changed.stopping_distance_deg == r.stopping_distance_deg
    assert r.correction_rpm == -sign*6
    assert r.base_rpm == 40


@pytest.mark.parametrize("sign", [-1,1])
def test_main_and_fast_forward_brake_trial_and_rollback(sign,monkeypatch):
    for trial in [0.,.35]:
        c=FollowSafetyController(FollowPolicyConfig(visible_steering_pid_enable=True,
            visible_steering_pid_image_error_only=True,visible_steering_pid_image_brake_assist=True,
            visible_steering_pid_image_capture_motion=True,
            visible_steering_pid_execution_response_trial_sec=trial,
            visible_steering_pid_camera_hfov_deg=66,
            visible_steering_pid_predictive_countersteer_max_correction_rpm=6,
            visible_steering_pid_predictive_countersteer_min_correction_rpm=3,
            visible_steering_pid_predictive_countersteer_gain_rpm_per_dps=.25,
            visible_steering_pid_predictive_countersteer_min_yaw_rate_dps=4))
        c.active_target_id=1
        c._visual_steering_pid.config=config(trial)
        monkeypatch.setattr(c,"_visible_base_forward_percent",lambda *a,**kw:20)
        for cap,t,x in [(377,9.62,.30),(379,9.75,.33),(382,9.88,.37)]:
            target,frame=sample(cap,t,.5+sign*(.5-x))
            frame=replace(frame,steering_feedback=fb(t+.12,sign))
            c._pid_action_for_visible_target(target,frame,t+.12)
        main=c.last_steering_pid_result
        fast=c.refresh_visible_lateral_pid(x_ratio=.5+sign*.13,base_rpm=40,
            feedback=fb(sign=sign),now=10,visual_age_sec=.12,target_image_rate_dps=-sign*20)
        assert main.correction_rpm == fast.correction_rpm == (-sign*6 if trial else 0)
        assert c._pid_direction_guard_reason(.5+sign*.13,0,-sign*6) is not None


def test_config_single_switch_binding_and_rollback(monkeypatch,tmp_path):
    from car_control_modular.config_loader import load_config_to_env
    root=Path(__file__).resolve().parents[2]
    monkeypatch.setattr(os,"environ",{})
    load_config_to_env(str(root/"car_control_modular/config/reid_runtime.ini"))
    assert float(os.environ["VISIBLE_STEERING_PID_EXECUTION_RESPONSE_TRIAL_SEC"])==.35
    tree=ast.parse((root/"request_0513_modular.py").read_text())
    bindings=[k.value for n in ast.walk(tree) if isinstance(n,ast.Call)
        for k in n.keywords if k.arg=="visible_steering_pid_execution_response_trial_sec"]
    assert len(bindings)==1
    for value in [0.,.35]:
        assert eval(compile(ast.Expression(bindings[0]),"binding","eval"),
            {"VISIBLE_STEERING_PID_EXECUTION_RESPONSE_TRIAL_SEC":value})==value


def test_fast_forward_brake_bypasses_slew_zero_without_parking_or_pivot(owner):
    c=FollowSafetyController(FollowPolicyConfig(visible_steering_pid_enable=True,
        visible_steering_pid_image_error_only=True,visible_steering_pid_image_brake_assist=True,
        visible_steering_pid_execution_response_trial_sec=.35,
        visible_steering_pid_predictive_countersteer_max_correction_rpm=6))
    c.active_target_id=1
    c._visual_steering_pid.config=config()
    f=fb(NOW)
    result=c.refresh_visible_lateral_pid(x_ratio=.35,base_rpm=40,feedback=f,
        now=NOW,visual_age_sec=.09,target_image_rate_dps=20)
    assert result.correction_rpm==6
    owner._follow_controller=c
    owner._depth30_linear_snapshot=("forward",20,1,NOW-.05)
    owner._action_runtime=SimpleNamespace(get_steering_feedback=lambda:f)
    i=_intent(owner,x_ratio=.35,mode="forward",base_rpm=40,initial_correction_rpm=6,
        target_image_rate_dps=20,forward_countersteer=True,near_distance_mode=False)
    owner._lateral_intent_last_correction_rpm=-7
    owner._service_lateral_intent(NOW)
    assert owner._current_steer_correction_rpm==6
    assert owner._lateral_intent_zero_sequence != i.sequence
    assert getattr(owner,"_near_yaw_park_request",None) is None
    owner._depth30_linear_snapshot=None
    assert not owner._request_predictive_turn_brake(i,result)
    assert getattr(owner,"_near_yaw_park_request",None) is None
