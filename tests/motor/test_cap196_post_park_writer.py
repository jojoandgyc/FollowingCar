"""Final motor clamp is independent of an old intent's too-large ceiling."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from test_cap1663_turn_response import prepare
from test_visible_wheel_continuity import feedback


@pytest.mark.parametrize("sign", [-1,1])
@pytest.mark.parametrize("base,policy,expected", [(0,7,4),(0,2,2),(42,10,10)])
def test_post_park_final_writer_and_forward_scope(monkeypatch,sign,base,policy,expected):
    rt,o,d,_,clock=prepare(monkeypatch)
    o._follow_controller.cfg=SimpleNamespace(visible_steering_pid_image_error_only=True,
        visible_steering_pid_max_correction_rpm=10,near_distance_rotation_only_max_rpm=7)
    o._follow_controller.post_park_recenter_limit=lambda uid:4 if uid==1 else None
    original=o._lateral_intent_store.snapshot()
    o._lateral_intent_store.publish(replace(original,correction_limit_rpm=policy,
        initial_correction_rpm=sign*15,near_distance_mode=False,
        x_ratio=.5+sign*.18,response_boost_allowed=False))
    o._fresh_depth_linear_snapshot=lambda uid,now=None:("forward",21) if base else None
    rt.get_steering_feedback=lambda:feedback(clock[0],0,0)
    for i in range(3):
        clock[0]=10+i*.05
        rt._send_follow_wheel_targets(base+sign*15,-(base-sign*15),"FOLLOW20")
        left,raw_right=d.pairs[-1]
        assert left+raw_right==sign*2*expected
        assert (left-raw_right)/2==base
    assert not d.stops
    o._has_fresh_lateral_yaw=lambda uid:False
    o._fresh_depth_linear_snapshot=lambda uid,now=None:None
    rt._send_follow_wheel_targets(sign*15,sign*15,"FOLLOW20")
    assert d.pairs[-1]==(0,0)
