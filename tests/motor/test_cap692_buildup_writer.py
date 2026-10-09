"""Numerically eligible taper and CAP711 sign anomaly, fake serial only."""
import pytest

from car_control_modular.steering_pid import encoder_yaw_rate_right_dps, VisualSteeringPid, VisualSteeringPidConfig
from car_control_modular.control_types import SteeringFeedback
from car_control_modular.turn_buildup import steering_buildup_allowed
from test_cap837_turn_buildup import setup, image_intent
from test_visible_wheel_continuity import feedback


def result(yaw=6.27, sign=1):
    # Keep this test runnable without collecting the control test directory.
    cfg=VisualSteeringPidConfig(enabled=True,image_error_only=True,image_brake_assist=True,
        image_capture_motion=True,camera_hfov_deg=66,deadband_deg=3,
        dynamic_large_error_deg=10,max_correction_rpm=10,
        predictive_brake_decel_dps2=60,predictive_brake_margin_deg=1.25,
        predictive_brake_response_sec=.1,execution_response_trial_sec=.35)
    f=SteeringFeedback(timestamp=10.,trustworthy=True,
        yaw_rate_right_dps=sign*yaw,raw_yaw_rate_right_dps=sign*yaw)
    return VisualSteeringPid(cfg).update(.5+sign*.2889,68,f,now=10.,
        visual_age_sec=.2153,target_image_rate_dps=sign*21.7)


@pytest.mark.parametrize('sign',[-1,1])
def test_numeric_taper_permission_reaches_real_writer(monkeypatch,caplog,sign):
    r,o,d,_,clock=setup(monkeypatch)
    proof=result(sign=sign)
    assert steering_buildup_allowed(proof)
    for t in [10.,10.11,10.16]:
        clock[0]=t
        i=o._lateral_intent_store.publish(image_intent(t,int(832+(t-10)*20),sign,
            response_boost_allowed=steering_buildup_allowed(proof)))
        o._lateral_turn_response_policy=(i.sequence,steering_buildup_allowed(proof))
        with caplog.at_level('INFO'):
            r._send_follow_wheel_targets(82+sign*10,-(82-sign*10),'FOLLOW20')
    assert d.pairs[-1] == (82+sign*10,-(82-sign*10))
    assert 'response_phase=buildup_yaw_only' in caplog.text
    assert 'execution_base_loss_rpm=0.0' in caplog.text
    # Raw cached feedback diagnostics add no reads or changes to wheel signs.
    assert 'feedback_raw_rpm=' in caplog.text and 'feedback_position_deg=' in caplog.text
    assert not d.stops


def test_actual_taper_in_fast_loop_blocks_boost_before_write(monkeypatch):
    r,o,d,_,clock=setup(monkeypatch)
    for t in [10.,10.11,10.16]:
        clock[0]=t
        i=o._lateral_intent_store.publish(image_intent(t,int(832+(t-10)*20)))
        o._lateral_turn_response_policy=(i.sequence,steering_buildup_allowed(result(yaw=17.5)))
        r._send_follow_wheel_targets(92,-72,'FOLLOW20')
    assert d.pairs == [(92,-72)]*3


def test_cap711_abnormal_feedback_is_not_hidden_or_used_to_force_reverse(monkeypatch,caplog):
    r,o,d,_,clock=setup(monkeypatch)
    o._fresh_depth_linear_snapshot=lambda uid,now=None:None
    i=image_intent(10.,711,-1,mode='yaw_only',bbox_quality='limited',response_boost_allowed=False)
    o._lateral_intent_store.publish(i)
    # Serial right-positive means physical reverse with this installation.
    normalized=encoder_yaw_rate_right_dps(40,7,1,-1,.5225,.5225)
    assert normalized[:2] == (40.,-7.) and normalized[2] > 0
    f=feedback(10.,40,-7)
    f.left_speed_rpm,f.right_speed_rpm=40,7
    f.left_position_deg,f.right_position_deg=123,456
    r.get_steering_feedback=lambda:f
    with caplog.at_level('INFO'):
        r._send_follow_wheel_targets(-7,-7,'FOLLOW20')
    assert d.pairs[-1] == (0,0)
    assert 'reason=cross_wait_zero' in caplog.text
    assert 'feedback_raw_rpm=(40, 7)' in caplog.text
    assert 'feedback_position_deg=(123, 456)' in caplog.text
    assert f.left_forward_rpm == 40 and f.right_forward_rpm == -7
