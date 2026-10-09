from dataclasses import replace
from types import SimpleNamespace
import pytest
from test_depth_target_geometry import observation, DISPLAY, CAPTURE, TIMESTAMP
from test_cap443_capture_braking import sample, pid_config
from car_control_modular.depth_target_geometry import resolve_depth_target_observation, resolve_braking_target_observation
from car_control_modular.visual_steering_evidence import CaptureSteeringEvidence
from car_control_modular.controllers import FollowSafetyController, FollowPolicyConfig
from car_control_modular.steering_pid import VisualSteeringPid


def weak():
    r=observation();r['uid']=0
    r['assignment'].update(uid=0,mapped_uid=1,reason='mapped_low_quality',
                           bbox_quality_ok=False,bbox_quality_tier='weak')
    return r


def args(records):
    return dict(target_id=1,display_bbox=DISPLAY,capture_frame_id=CAPTURE,
        capture_timestamp=TIMESTAMP,width=640,height=480,observations=records)


@pytest.mark.parametrize('reason', ['mapped_low_quality', 'mapped_weak_observed'])
def test_mapped_weak_geometry_is_braking_only_not_depth(reason):
    record=weak();record['assignment']['reason']=reason
    a=args([record])
    assert resolve_depth_target_observation(**a) is None
    out=resolve_braking_target_observation(**a)
    assert out.source=='yolo_braking_only' and out.target_id==1


@pytest.mark.parametrize('bad',['identity','competition','excluded','geometry','uid','duplicate','stale','pending'])
def test_braking_does_not_invent_association(bad):
    r=weak();records=[r]
    if bad=='identity':r['assignment']['identity_control_rejected']=True
    if bad=='competition':r['sample_metadata']['identity_competition']={'passed':False}
    if bad=='excluded':r['assignment']['search_excluded']=True
    if bad=='geometry':r['assignment']['reacquire_geometry_ok']=False
    if bad=='uid':r['assignment']['mapped_uid']=2
    if bad=='duplicate':records.append(weak())
    if bad=='stale':r['sample_metadata']['is_fresh']=False
    if bad=='pending':r['assignment']['reason']='preferred_search_reacquire_wait'
    assert resolve_braking_target_observation(**args(records)) is None


@pytest.mark.parametrize('xs,inward', [([.8,.75,.7],True),([.7,.75,.8],False),([.2,.25,.3],True)])
def test_only_inward_velocity_and_no_position_rewrite(xs,inward):
    e=CaptureSteeringEvidence()
    for i,x in enumerate(xs):
        t,f=sample(315+i,10+i*.1,x,tracker_x=x+.02)
        t=replace(t,depth_observation=None,
                  braking_observation=replace(t.depth_observation,source='yolo_braking_only'))
        out=e.observe(t,f,10+i*.1+.1,66)
        assert out.control_x==pytest.approx(x+.02) and t.depth_observation is None
    assert (out.rate_dps is not None)==inward


def test_real_person_builder_keeps_two_permissions_separate():
    from request_0513_modular import PersonTracker
    owner=SimpleNamespace(_active_capture_frame_id=CAPTURE,_active_capture_timestamp=TIMESTAMP,
        _follow_controller=SimpleNamespace(active_target_id=1),
        _rknn_pipeline=SimpleNamespace(tracker=SimpleNamespace(last_identity_observations=[weak()])))
    t=PersonTracker._persons_to_targets(owner,[(DISPLAY,1,.9,70000)],width=640,height=480)[0]
    assert t.depth_observation is None and t.braking_observation.source=='yolo_braking_only'
    owner._follow_controller.active_target_id=2
    assert PersonTracker._persons_to_targets(owner,[(DISPLAY,1,.9,70000)],width=640,height=480)[0].braking_observation is None


def test_pivot_response_budget_not_applied_to_normal_forward():
    c=FollowSafetyController(FollowPolicyConfig(visible_steering_pid_image_brake_assist=True))
    normal,park=c._visual_steering_pid.config,c._parked_recenter_pid.config
    assert normal.image_brake_latency_max_sec==.25
    assert park.image_brake_latency_max_sec==.5
    assert park.predictive_brake_response_sec>=.10
    assert park.image_motion_response_sec>=.25


@pytest.mark.parametrize('x,rate',[(.8,-25),(.2,25)])
def test_longer_pivot_response_only_reduces_same_direction_demand(x,rate):
    old=replace(pid_config(),image_motion_response_sec=.18)
    new=replace(old,image_motion_response_sec=.25)
    before=VisualSteeringPid(old).update(x,0,None,now=10,
        visual_age_sec=.15,target_image_rate_dps=rate)
    after=VisualSteeringPid(new).update(x,0,None,now=10,
        visual_age_sec=.15,target_image_rate_dps=rate)
    assert abs(after.correction_rpm)<abs(before.correction_rpm)
    assert after.correction_rpm*(x-.5)>=0


def test_deployed_pivot_cap_separate_from_forward_cap():
    import configparser
    from pathlib import Path
    cfg=configparser.ConfigParser()
    cfg.read(Path(__file__).resolve().parents[2]/'car_control_modular/config/reid_runtime.ini')
    caps=[s.getfloat('near_distance_rotation_only_max_rpm') for s in cfg.values()
          if 'near_distance_rotation_only_max_rpm' in s]
    assert caps==[7]
    assert cfg['steering_pid'].getfloat('max_correction_rpm')==10


def test_unconfirmed_single_high_yaw_sample_cannot_trigger_predictive_park():
    from car_control_modular.control_types import SteeringFeedback
    c=replace(pid_config(), predictive_brake_decel_dps2=60,
              predictive_brake_response_sec=.1,
              predictive_countersteer_max_correction_rpm=2.5)
    feedback=SteeringFeedback(timestamp=9.95, yaw_rate_right_dps=-56,
        raw_yaw_rate_right_dps=-56, trustworthy=True, yaw_rate_confirmed=False)
    result=VisualSteeringPid(c).update(.33, 0, feedback, now=10,
        visual_age_sec=.15, target_image_rate_dps=None,
        max_correction_override_rpm=7)
    assert not result.predictive_braking
    assert result.correction_rpm < 0


def test_confirmed_inward_motion_uses_bounded_early_countersteer():
    from car_control_modular.control_types import SteeringFeedback
    c=replace(pid_config(), predictive_brake_decel_dps2=60,
              predictive_brake_response_sec=.1,
              predictive_countersteer_max_correction_rpm=2.5,
              predictive_countersteer_min_yaw_rate_dps=8)
    feedback=SteeringFeedback(timestamp=9.95, yaw_rate_right_dps=-23.5,
        raw_yaw_rate_right_dps=-25, trustworthy=True, yaw_rate_confirmed=True)
    result=VisualSteeringPid(c).update(.266, 0, feedback, now=10,
        visual_age_sec=.106, target_image_rate_dps=42,
        max_correction_override_rpm=7)
    assert result.predictive_braking
    assert result.correction_rpm == 2
    assert result.output_floor_reason == "predictive_countersteer"


def test_confirmed_residual_yaw_brakes_when_target_has_stopped_near_center():
    """CAP166-style stationary target must not wait for an inward derivative."""
    from car_control_modular.control_types import SteeringFeedback
    c=replace(pid_config(), predictive_brake_decel_dps2=60,
              predictive_brake_response_sec=.1,
              predictive_countersteer_max_correction_rpm=2.5,
              predictive_countersteer_min_yaw_rate_dps=4)
    feedback=SteeringFeedback(timestamp=9.95, yaw_rate_right_dps=-5.5,
        raw_yaw_rate_right_dps=-5.0, trustworthy=True, yaw_rate_confirmed=True)
    result=VisualSteeringPid(c).update(.44, 0, feedback, now=10,
        visual_age_sec=.10, target_image_rate_dps=1.0,
        max_correction_override_rpm=7)
    assert result.predictive_braking
    assert result.correction_rpm == 1
    assert result.output_floor_reason == "predictive_countersteer"
