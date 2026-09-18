"""Typed parking is distinct from emergency, unknown and rotation holds."""
import pytest
from types import SimpleNamespace
from test_depth_drive_rpm import make_runtime
from car_control_modular.follow_distance_hold import FollowDistanceHold


@pytest.mark.parametrize('case', ['ordinary','hard_stop','other_reason','unknown_queue','safety_mode',
                                 'explicit','rotation_only','search','old_safety','depth_disabled'])
def test_only_known_distance_park_creates_recovery_token(case):
    rt, owner, driver, symbols = make_runtime()
    owner._follow_controller = SimpleNamespace(active_target_id=1)
    owner._depth_longitudinal_authority_enabled = lambda: case != 'depth_disabled'
    owner._vision_control_state = 'target_visible_depth_valid'
    owner.search_state = 'none'
    owner._last_control_decision_reason = 'near_distance_rotation_only'
    owner._last_action_queue_reason = 'lateral_zero:near_distance_rotation_only'
    reason='queued_action_stop_signal'
    if case=='hard_stop': reason='hard_stop'
    elif case=='other_reason': reason='direct_stop'
    elif case=='unknown_queue': owner._last_action_queue_reason='other'
    elif case=='safety_mode': owner._brake_hold_stop_mode='brake'
    elif case=='explicit': owner._explicit_stop_requested=True
    elif case=='rotation_only': rt.config.rotation_only=True
    elif case=='search': owner.search_state='searching'
    elif case=='old_safety': owner._brake_hold_label='safety_hold_hazard'
    # Entry state change is real; motor and pulse methods are inert spies.
    rt.cancel_yaw_pulses=lambda *a,**kw:None
    rt.send_robot_command=lambda *a,**kw:None
    rt.send_percent_brake=lambda *a,**kw:None
    rt.send_stop_with_brake_hold(reason)
    assert owner._brake_hold_active
    assert isinstance(owner._follow_distance_hold,FollowDistanceHold) == (case=='ordinary')
    assert (owner._brake_hold_label=='follow_distance_hold') == (case=='ordinary')
