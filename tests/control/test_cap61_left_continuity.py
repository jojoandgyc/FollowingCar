"""CAP61..97 regressions: real control/intent functions, no hardware."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.control_types import DepthTargetObservation, PersonTarget, SensorFrame, SteeringFeedback
from car_control_modular.controllers import FollowSafetyController
from car_control_modular.steering_pid import VisualSteeringPid, VisualSteeringPidConfig
from car_control_modular.visual_steering_evidence import CaptureSteeringEvidence
from car_control_modular.outward_trajectory import make_outward_lead
from car_control_modular.turn_buildup import TurnBuildup
from test_cap58_outward_trajectory import config
from test_cap443_capture_braking import sample
from test_lateral_zero_runtime import owner, NOW


def pid():
    return VisualSteeringPid(VisualSteeringPidConfig(enabled=True, image_error_only=True,
        image_brake_assist=True, image_capture_motion=True, camera_hfov_deg=66,
        deadband_deg=3, dynamic_large_error_deg=10, max_correction_rpm=10,
        predictive_brake_decel_dps2=60, predictive_brake_margin_deg=1.25,
        predictive_brake_response_sec=.1,
        execution_response_trial_sec=.35, image_slow_brake_continuity_sec=.5))


def feedback(now, sign=-1):
    return SteeringFeedback(timestamp=now-.02, trustworthy=True,
        left_forward_rpm=12, right_forward_rpm=34,
        yaw_rate_right_dps=sign*28, raw_yaw_rate_right_dps=sign*30)


def forward_episode(monkeypatch, sign=-1, start=10.):
    c = FollowSafetyController(config(forward_max_rpm=200,
        center_left_ratio=.45, center_right_ratio=.55,
        steer_release_left_ratio=.48, steer_release_right_ratio=.52))
    c.active_target_id = 1
    monkeypatch.setattr(c, "_visible_base_forward_percent", lambda *a, **kw: 30)
    # CAP61,63,65 detector centers and capture spacing, mirrored for right.
    for i, (dt, x) in enumerate([(0., .4621), (.099441, .4453), (.200659, .4369)]):
        t, f = sample(61+i*2, start+dt, .5+sign*(.5-x))
        action = c._pid_action_for_visible_target(t, f, start+dt+.10)
    return c, t, f, action


@pytest.mark.parametrize("sign", [-1, 1])
def test_forward_enters_before_4_8_degree_hold_and_fast_loop_keeps_lead(monkeypatch, sign):
    c,t,f,action = forward_episode(monkeypatch, sign)
    result = c.last_steering_pid_result
    assert result.correction_rpm == sign*4
    assert result.base_rpm == 60
    assert result.outward_lead is not None
    for dt in [.11, .15, .20]:
        fast = c.refresh_visible_lateral_pid(x_ratio=result.outward_lead.x,
            base_rpm=60, feedback=None, now=f.capture_timestamp+dt,
            target_image_rate_dps=result.target_image_rate_dps,
            visual_age_sec=dt, outward_lead=result.outward_lead)
        assert fast.correction_rpm == sign*4 and fast.base_rpm == 60
    for kwargs in [dict(hold_zero=True), dict(allow_forward_tracking=False)]:
        fast = c.refresh_visible_lateral_pid(x_ratio=result.outward_lead.x,
            base_rpm=60, feedback=None, now=f.capture_timestamp+.15,
            outward_lead=result.outward_lead, **kwargs)
        assert fast.outward_lead is None
    expired = c.refresh_visible_lateral_pid(x_ratio=result.outward_lead.x,
        base_rpm=60, feedback=None, now=f.capture_timestamp+.251,
        outward_lead=result.outward_lead)
    assert expired.outward_lead is None


def test_forward_lead_survives_actual_publisher_and_refresh(owner, monkeypatch):
    c,t,f,action = forward_episode(monkeypatch, start=NOW-.300659)
    owner._follow_controller = c
    owner._last_command_capture_frame = f.capture_frame_id
    owner._last_command_capture_timestamp = f.capture_timestamp
    owner._last_control_decision_reason = "visible_target"
    assert owner._publish_lateral_intent_from_decision(width=640, target=t,
        runtime_actions=[action], control_source="vision",
        target_steerable=True, low_quality_visible=False)
    intent = owner._lateral_intent_store.snapshot()
    assert intent.mode == "forward" and intent.outward_lead is not None
    assert not intent.response_boost_allowed
    # Exercise real runtime refresh, but intercept result at the PID boundary
    # since the fake owner has no longitudinal grant (it must not invent one).
    fast = c.refresh_visible_lateral_pid(x_ratio=intent.x_ratio, base_rpm=60,
        feedback=None, now=NOW+.01, visual_age_sec=.11,
        target_image_rate_dps=intent.target_image_rate_dps, outward_lead=intent.outward_lead)
    assert fast.correction_rpm == -4
    assert owner._depth30_linear_snapshot is None


def test_qualified_forward_turn_still_has_bounded_buildup_window(owner,monkeypatch):
    c=FollowSafetyController(config(forward_max_rpm=200,
        visible_steering_pid_max_correction_rpm=10))
    c.active_target_id=1
    monkeypatch.setattr(c,"_visible_base_forward_percent",lambda *a,**kw:30)
    for cap,ts,x in [(74,NOW-.3,.4),(76,NOW-.2,.35),(78,NOW-.1,.3)]:
        t,f=sample(cap,ts,x)
        action=c._pid_action_for_visible_target(t,f,ts+.1)
    owner._follow_controller=c
    owner._last_command_capture_frame=78
    owner._last_command_capture_timestamp=NOW-.1
    owner._last_control_decision_reason="visible_target"
    assert owner._publish_lateral_intent_from_decision(width=640,target=t,
        runtime_actions=[action],control_source="vision",target_steerable=True,low_quality_visible=False)
    i=owner._lateral_intent_store.snapshot()
    assert i.response_boost_allowed and i.mode == "forward"
    b=TurnBuildup(response_delay_sec=.1)  # Historical short-latency policy fixture.
    b.note_sent(1,(50,70),NOW-.18,0.)
    fb=SteeringFeedback(timestamp=NOW-.05,trustworthy=True,left_forward_rpm=10,right_forward_rpm=10)
    assert b.adjust(60,-10,i,fb,NOW,1,limit=10)[2] == "buildup_wait_two_feedback"
    b.note_sent(1,(50,70),NOW,NOW-.18)
    r=b.adjust(60,-10,i,replace(fb,timestamp=NOW+.03),NOW+.04,1,limit=10)
    assert r == (60,-10,"buildup_yaw_only")
    # The crop/early-lead route is deliberately not granted this boost scope.
    assert b.adjust(60,-10,replace(i,bbox_quality="limited"),fb,NOW+.05,1,limit=10)[:2] == (60,-10)


# Real detector/display geometry and capture timestamps, translated to a
# compact relative clock. Only CAP78 is reliable; remaining crops are mapped.
CROPS = [
    (78, 0., (94.418,16.767,326.502,476.899), (71.383,0,371.134,479)),
    (83, .241825, (.023,2.356,256.781,472.446), (0,0,304.304,479)),
    (85, .335814, (0,2.335,237.647,473.543), (0,0,276.162,479)),
    (87, .438669, (.759,1.780,225.526,472.937), (0,0,263.785,479)),
    (88, .499769, (.015,2.734,224.086,471.372), (0,0,258.480,479)),
    (89, .567748, (.757,2.004,221.099,472.957), (0,0,255.727,479)),
]


def crop_episode(sign=-1, bad=None, start=10.):
    e = CaptureSteeringEvidence()
    outputs=[]
    for cap,dt,box,display in CROPS:
        if sign > 0:
            box = (640-box[2],box[1],640-box[0],box[3])
            display = (640-display[2],display[1],640-display[0],display[3])
        ts = start+dt
        obs = DepthTargetObservation(box,1,1,cap,ts,
            "yolo_detector" if cap == 78 else "yolo_braking_only")
        target = PersonTarget(display,1,.95,100000,
            depth_observation=obs if cap == 78 else None,
            braking_observation=obs if cap != 78 else None)
        f = SensorFrame(width=640,height=480,persons=[target],capture_frame_id=cap,capture_timestamp=ts)
        if bad == "no_anchor" and cap == 78:
            continue
        if bad == "raw" and cap != 78:
            target=replace(target,braking_observation=replace(obs,raw_track_id=2))
        if bad == "missing" and cap == 87:
            target=replace(target,braking_observation=None)
        if bad == "old_anchor" and cap != 78:
            e.reliable_anchor = ((1,1),start-1)
        result=e.observe(target,f,ts+.1,66)
        outputs.append((target,f,result))
    return e,outputs


@pytest.mark.parametrize("sign", [-1,1])
def test_real_edge_crop_motion_prevents_false_predictive_zero(sign):
    e, outputs = crop_episode(sign)
    for target,frame,obs in outputs[-2:]:
        assert obs.rate_dps is None  # no ordinary velocity/lead permission
        assert obs.outward_continuity_rate_dps*sign > 1
        assert make_outward_lead(obs,frame.capture_timestamp+.1,hfov=66,deadband=3,
            release_margin=1.8,enabled=True) is None
        now=frame.capture_timestamp+.1
        base=pid().update(obs.control_x,0,feedback(now,sign),now=now,visual_age_sec=.1)
        r=pid().update(obs.control_x,0,feedback(now,sign),now=now,visual_age_sec=.1,
            outward_continuity_rate_dps=obs.outward_continuity_rate_dps)
        assert base.correction_rpm == 0
        assert r.correction_rpm == sign*2 and not r.predictive_braking
        assert r.forward_phase == "image_crop_outward_continuity"
        assert not r.target_rate_valid and r.outward_lead is None and r.base_rpm == 0


@pytest.mark.parametrize("bad", ["no_anchor","raw","missing","old_anchor"])
def test_crop_cannot_self_certify_or_cross_invalid_identity_chain(bad):
    _,outputs = crop_episode(bad=bad)
    assert outputs[-1][-1].outward_continuity_rate_dps is None


@pytest.mark.parametrize("bad", ["expired","future","nan","inward","tiny","near_center"])
def test_crop_relief_never_overrides_invalid_or_inward_evidence(bad):
    rate,age,x=-2.,.1,.2
    if bad == "expired": age=.251
    if bad == "future": age=-.1
    if bad == "nan": rate=float("nan")
    if bad == "inward": rate=10
    if bad == "tiny": rate=-.1
    if bad == "near_center": x=.46
    r=pid().update(x,0,feedback(10),now=10,visual_age_sec=age,
        outward_continuity_rate_dps=rate)
    baseline=pid().update(x,0,feedback(10),now=10,visual_age_sec=age)
    assert r.correction_rpm == baseline.correction_rpm
    assert r.forward_phase != "image_crop_outward_continuity"


def test_full_quality_outward_is_bounded_and_inward_still_brakes():
    out=pid().update(.2,60,feedback(10),now=10,visual_age_sec=.1,target_image_rate_dps=-10)
    inward=pid().update(.2,60,feedback(10),now=10,visual_age_sec=.1,target_image_rate_dps=40)
    assert out.correction_rpm == -4 and out.base_rpm == 60
    assert inward.correction_rpm == 0 and inward.predictive_braking
    assert inward.base_rpm == 60


def test_high_yaw_and_zero_override_still_dominate_outward_cue():
    fb=replace(feedback(10),yaw_rate_right_dps=-50,raw_yaw_rate_right_dps=-50)
    r=pid().update(.2,0,fb,now=10,visual_age_sec=.1,outward_continuity_rate_dps=-2)
    assert r.correction_rpm == 0 and r.predictive_braking
    r=pid().update(.2,0,feedback(10),now=10,visual_age_sec=.1,
        outward_continuity_rate_dps=-2,max_correction_override_rpm=0)
    assert r.correction_rpm == 0


@pytest.mark.parametrize("match", [True,False])
def test_crop_producer_and_fast_pid_keep_distinct_scope(owner,match):
    e,outputs=crop_episode(start=NOW-.667748)
    t,f,obs=outputs[-1]
    c=FollowSafetyController(config(visible_steering_pid_predictive_brake_decel_dps2=60,
        visible_steering_pid_predictive_brake_margin_deg=1.25,
        visible_steering_pid_predictive_brake_response_sec=.1,
        visible_steering_pid_execution_response_trial_sec=.35))
    c.active_target_id=1
    c._capture_steering_evidence=e
    f=replace(f,steering_feedback=feedback(NOW))
    action=c._pid_action_for_parked_target(t,f,NOW,"left",max_correction_rpm=7)
    owner._follow_controller=c
    owner._last_command_capture_frame=f.capture_frame_id if match else f.capture_frame_id+1
    owner._last_command_capture_timestamp=f.capture_timestamp
    owner._last_control_decision_reason="target_visible_low_quality_yaw"
    assert owner._publish_lateral_intent_from_decision(width=640,target=t,
        runtime_actions=[action],control_source="vision",target_steerable=False,low_quality_visible=True)
    intent=owner._lateral_intent_store.snapshot()
    assert intent.mode == "yaw_only" and intent.base_percent == 0
    assert intent.target_image_rate_dps is None and intent.outward_lead is None
    assert not intent.response_boost_allowed
    assert (intent.outward_continuity_rate_dps is not None) == match
    r=c.refresh_parked_lateral_pid(x_ratio=intent.x_ratio,base_rpm=0,
        feedback=feedback(NOW+.01),now=NOW+.01,visual_age_sec=.11,
        outward_continuity_rate_dps=intent.outward_continuity_rate_dps,max_correction_rpm=7)
    assert r.correction_rpm == (-2 if match else 0)
    assert owner._depth30_linear_snapshot is None
    if match:
        owner._action_runtime=SimpleNamespace(get_steering_feedback=lambda:feedback(NOW))
        owner._lateral_intent_last_correction_rpm=-2
        owner._service_lateral_intent(NOW)
        owner._service_lateral_intent(NOW+.01)
        assert abs(owner._current_rotate_raw_target) == 2
        assert owner._depth30_linear_snapshot is None
