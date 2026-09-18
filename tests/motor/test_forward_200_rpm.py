"""RPM-to-percent-to-driver regression at the requested 200RPM ceiling."""
import pytest
from dataclasses import replace
import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from test_depth_drive_rpm import make_runtime
from test_follow_wheel_periodic import setup_periodic
from test_visible_wheel_continuity import feedback


@pytest.mark.parametrize('rpm,expected',[(20,20),(84,84),(160,160),(200,200),(240,200)])
def test_pid_units_survive_200rpm_scale(rpm,expected):
    controller=FollowSafetyController(FollowPolicyConfig(forward_max_rpm=200,max_forward_percent=100))
    percent=controller._forward_percent_for_rpm(rpm,allow_below_min=True)
    rt,owner,driver,symbols=make_runtime(percent=percent,max_rpm=200)
    rt.send_robot_command(symbols.forward)
    assert driver.pairs[-1]==(expected,-expected)


def test_depth_limits_keep_near_slow_and_far_ceiling(monkeypatch):
    monkeypatch.setattr(runtime,'FORWARD_MAX_RPM',200)
    monkeypatch.setattr(runtime,'ASTRA_DEPTH_LONGITUDINAL_MAX_FORWARD_PERCENT',10)
    monkeypatch.setattr(runtime,'ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT',100)
    monkeypatch.setattr(runtime,'FOLLOW_REVERSE_RUNTIME_CAP_RPM',60)
    cap=runtime.PersonTracker._cap_depth_longitudinal_actions
    assert cap([ControlAction.forward(100,'test')],distance_m=1.6)[0].speed_percent==10
    assert cap([ControlAction.forward(42,'test')],distance_m=3.4)[0].speed_percent==42
    assert cap([ControlAction.forward(100,'test')],distance_m=3.4)[0].speed_percent==100
    assert cap([ControlAction.backward(100,'test')],distance_m=3.4)[0].speed_percent==30


def test_periodic_wheel_ceiling_keeps_differential_and_zero_expiry(monkeypatch):
    rt,owner,driver,s,clock,state=setup_periodic(monkeypatch)
    rt.config.motor_forward_max_target_rpm=200
    rt.backend.config=replace(rt.backend.config,max_target=200)
    state[:2]=[200,5]
    owner._fresh_depth_linear_snapshot=lambda uid,now=None:('forward',100) if clock[0]<state[2] else None
    rt.get_steering_feedback=lambda:feedback(clock[0],190,190)
    rt._service_follow_wheels()
    assert driver.pairs[-1]==(200,-190)
    clock[0]=10.19
    rt._service_follow_wheels()
    assert driver.pairs[-1]==(0,0)


def test_reverse_and_search_do_not_double_with_forward_scale():
    ctl=FollowSafetyController(FollowPolicyConfig(
        forward_max_rpm=200, reverse_max_rpm=100, reverse_runtime_cap_rpm=60,
        distance_pid_enable=False))
    percent=ctl._reverse_percent_for_distance(.8,approach_speed_m_s=2.,now=10.)
    assert percent==30
    rt,owner,driver,s=make_runtime(percent=percent,max_rpm=200)
    rt.send_robot_command(s.backward)
    assert driver.pairs[-1]==(-60,60)
    rt.send_robot_command(s.rotate_right)
    assert max(abs(v) for v in driver.pairs[-1])==8
