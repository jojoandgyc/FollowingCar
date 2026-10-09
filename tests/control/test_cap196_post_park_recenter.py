"""CAP192 restart / CAP199 turnaround: real controller, no hardware."""
from dataclasses import replace
from types import SimpleNamespace
import pytest

import request_0513_modular as runtime
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.control_types import SteeringFeedback, ControlAction
from car_control_modular.outward_trajectory import make_outward_lead
from car_control_modular.visual_steering_evidence import CaptureSteeringEvidence
from test_cap443_capture_braking import sample
from test_lateral_zero_runtime import owner, NOW, _intent


def controller():
    c = FollowSafetyController(FollowPolicyConfig(visible_steering_pid_enable=True,
        visible_steering_pid_image_error_only=True, visible_steering_pid_image_brake_assist=True,
        visible_steering_pid_image_capture_motion=True, visible_steering_pid_camera_hfov_deg=66,
        visible_steering_pid_deadband_deg=3, visible_steering_pid_dynamic_small_error_deg=3,
        visible_steering_pid_dynamic_large_error_deg=10,
        visible_steering_pid_predictive_brake_decel_dps2=60,
        near_distance_rotation_only_max_rpm=7, parked_recenter_min_rpm=5,
        parked_recenter_max_rpm=14, visible_steering_pid_max_correction_rpm=10))
    c.active_target_id = 1
    return c


def refresh(c, x=.3193, now=10., **kw):
    args = dict(x_ratio=x, base_rpm=0, feedback=None, now=now,
        visual_age_sec=.1, near_distance_mode=True, max_correction_rpm=7)
    args.update(kw)
    return c.refresh_parked_lateral_pid(**args)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("x,expected", [(.3193,4),(.3373,4),(.37,3),(.4,2),(.4241,1),(.5,0)])
def test_main_and_fast_post_park_curve_has_ceiling_not_floor(monkeypatch, sign, x, expected):
    c = controller()
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 10.)
    c.set_normal_parking(True, 1)
    c.set_normal_parking(False, 1)
    x = .5 + sign*(.5-x)
    target, frame = sample(196, 10.01, x)
    c._pid_action_for_parked_target(target, frame, 10.11,
        "right" if sign > 0 else "left", near_distance_mode=True)
    main = c.last_steering_pid_result
    fast = refresh(c, x, now=10.11)
    assert main.correction_rpm == fast.correction_rpm == sign*expected
    assert main.post_park_recenter and fast.post_park_recenter
    assert main.correction_policy_limit_rpm == fast.correction_policy_limit_rpm == 4
    assert abs(refresh(c, x, now=50.).correction_rpm) <= 4  # no timer restores 7


def test_scope_first_turn_forward_other_uid_search_and_lower_cap():
    c = controller()
    assert refresh(c).correction_rpm == -7
    c.set_normal_parking(True, 2)
    assert refresh(c).correction_rpm == -7
    c.set_normal_parking(True, 1)
    assert refresh(c, max_correction_rpm=2).correction_rpm == -2
    forward = c.refresh_visible_lateral_pid(x_ratio=.8, base_rpm=80,
        feedback=None, now=10, visual_age_sec=.1, max_correction_rpm=10)
    assert forward.correction_rpm == 10 and not forward.post_park_recenter
    c.search_state = "searching"
    assert c.post_park_recenter_limit(1) is None
    c.search_state = "none"
    c.clear_active_target()
    assert c.post_park_recenter_limit(1) is None


@pytest.mark.parametrize("bad", [None,"duplicate","braking_only","old","small_error","inward"])
def test_only_new_sustained_clear_outward_trend_exits_cap(monkeypatch, bad):
    c = controller()
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 10.)
    c.set_normal_parking(True, 1); c.set_normal_parking(False, 1)
    for i, x in enumerate([.73,.76,.79]):
        if bad == "small_error": x -= .15
        if bad == "inward": x = 1.52-x
        stamp = 10.1+i*.1 if bad != "old" else 9.8+i*.1
        cap = 200+i
        if bad == "duplicate" and i == 2: cap, stamp = 201, 10.2
        t, f = sample(cap, stamp, x)
        if bad == "braking_only":
            t = replace(t, depth_observation=None,
                braking_observation=replace(t.depth_observation, source="yolo_braking_only"))
        c._capture_steering_observation(t, f, stamp+.09)
    assert c.post_park_recenter_limit(1) == (None if bad is None else 4)


@pytest.mark.parametrize("sign", [-1,1])
def test_first_inward_interval_is_separate_taper_evidence(sign):
    e = CaptureSteeringEvidence()
    for cap, stamp, x in [(194,10.,.323),(196,10.132381,.3193),(199,10.268608,.3255)]:
        t, f = sample(cap,stamp,.5+sign*(x-.5))
        obs = e.observe(t,f,stamp+.1,66)
    assert obs.rate_dps is None and obs.reason == "rate_inconsistent"
    assert obs.inward_turnaround_rate_dps == pytest.approx(sign*3.003809, abs=.0001)
    assert make_outward_lead(obs,stamp+.1,hfov=66,deadband=3,release_margin=1.8,enabled=True) is None
    assert e.observe(t,f,stamp+.11,66) is obs
    c = controller(); c.set_normal_parking(True,1)
    fb = SteeringFeedback(timestamp=stamp+.08,trustworthy=True,
        left_forward_rpm=-3*sign,right_forward_rpm=2*sign,
        yaw_rate_right_dps=-6*sign,raw_yaw_rate_right_dps=-6*sign)
    plain = refresh(c,obs.control_x,now=stamp+.1,feedback=fb)
    tapered = refresh(c,obs.control_x,now=stamp+.1,feedback=fb,
                      braking_image_rate_dps=obs.inward_turnaround_rate_dps)
    assert abs(tapered.correction_rpm) < abs(plain.correction_rpm)
    assert tapered.correction_rpm*plain.correction_rpm >= 0
    assert not tapered.target_rate_valid
    assert tapered.output_floor_reason != "predictive_countersteer"
    assert tapered.braking_image_rate_dps is not None


@pytest.mark.parametrize("bad", ["missing","stale","wrong_yaw","disagree","stale_image","outward","nan"])
def test_turnaround_cue_needs_current_agreeing_motion_and_image(bad):
    c = controller(); c.set_normal_parking(True,1)
    fb = SteeringFeedback(timestamp=9.98,trustworthy=True,left_forward_rpm=-3,right_forward_rpm=2,
        yaw_rate_right_dps=-6,raw_yaw_rate_right_dps=-6)
    rate, age = 3.1, .1
    if bad == "missing": fb = None
    if bad == "stale": fb=replace(fb,timestamp=9.8)
    if bad == "wrong_yaw": fb=replace(fb,yaw_rate_right_dps=6,raw_yaw_rate_right_dps=6)
    if bad == "disagree": fb=replace(fb,raw_yaw_rate_right_dps=6)
    if bad == "stale_image": age=.3
    if bad == "outward": rate=-3.1
    if bad == "nan": rate=float("nan")
    a=refresh(c,feedback=fb,visual_age_sec=age)
    b=refresh(c,feedback=fb,visual_age_sec=age,braking_image_rate_dps=rate)
    assert a.correction_rpm == b.correction_rpm
    assert b.braking_image_rate_dps is None


def test_old_large_intent_first_tick_and_slew_cannot_override_recenter(owner,monkeypatch):
    c=controller(); c.set_normal_parking(True,1); c.set_normal_parking(False,1)
    owner._follow_controller=c
    c.last_steering_pid_result=refresh(c,now=NOW)
    _intent(owner,x_ratio=.3193,initial_correction_rpm=-7,correction_limit_rpm=7,
            target_image_rate_dps=None)
    owner._action_runtime=SimpleNamespace(get_steering_feedback=lambda:None)
    owner._lateral_intent_last_correction_rpm=-7
    owner._service_lateral_intent(NOW)
    assert owner._current_rotate_raw_target == 4
    owner._service_lateral_intent(NOW+.04)
    assert owner._current_rotate_raw_target == 4
    deadline=owner._lateral_intent_store.snapshot().valid_until
    owner._service_lateral_intent(deadline+.01)
    assert owner._current_rotate_raw_target == 0


@pytest.mark.parametrize("bad", [None, "uid", "cap", "stamp"])
def test_turnaround_cue_publisher_and_fast_loop_require_same_capture(owner, bad):
    c = controller(); c.set_normal_parking(True, 1); c.set_normal_parking(False, 1)
    for cap, stamp, x in [(574,NOW-.33,.323),(575,NOW-.197619,.3193),
                          (576,NOW-.061392,.3255)]:
        t, f = sample(cap, stamp, x)
        sample_now = stamp+.061392
        fb = SteeringFeedback(timestamp=sample_now-.02, trustworthy=True,
            left_forward_rpm=-3, right_forward_rpm=2,
            yaw_rate_right_dps=-6, raw_yaw_rate_right_dps=-6)
        f = replace(f, steering_feedback=fb)
        c._pid_action_for_parked_target(t,f,sample_now,"left",near_distance_mode=True)
    assert c.last_capture_steering_observation.inward_turnaround_rate_dps is not None
    assert c.last_steering_pid_result.correction_rpm == -3
    owner._follow_controller = c
    owner._last_command_capture_timestamp = stamp
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: fb)
    owner._lateral_intent_last_correction_rpm = -4
    obs = c.last_capture_steering_observation
    if bad == "uid": obs = replace(obs, target_id=2)
    if bad == "cap": obs = replace(obs, capture_frame_id=575)
    if bad == "stamp": obs = replace(obs, capture_timestamp=stamp-.01)
    c.last_capture_steering_observation = obs
    assert owner._publish_lateral_intent_from_decision(width=640, target=t,
        runtime_actions=[ControlAction.rotate_left("test")], control_source="vision",
        target_steerable=True, low_quality_visible=False)
    intent = owner._lateral_intent_store.snapshot()
    assert (intent.braking_image_rate_dps is not None) == (bad is None)
    assert intent.target_image_rate_dps is None and intent.outward_lead is None
    assert intent.mode == "yaw_only" and owner._depth30_linear_snapshot is None
    owner._service_lateral_intent(NOW)
    owner._service_lateral_intent(NOW+.03)
    if bad is None:
        assert owner._current_rotate_raw_target == 3


@pytest.mark.parametrize("bad", ["uid","raw","gap","missing","jitter","confidence"])
def test_turnaround_cue_rejects_broken_chain_and_small_jitter(bad):
    e = CaptureSteeringEvidence()
    for cap, stamp, x in [(194,10.,.323),(196,10.132381,.3193)]:
        t,f=sample(cap,stamp,x)
        e.observe(t,f,stamp+.1,66)
    t,f=sample(199,10.268608,.3255)
    if bad == "uid": t,f=sample(199,10.268608,.3255,uid=2)
    if bad == "raw": t,f=sample(199,10.268608,.3255,raw=7)
    if bad == "gap": t,f=sample(199,10.5,.3255)
    if bad == "missing": t=replace(t,depth_observation=None)
    if bad == "jitter": t,f=sample(199,10.268608,.322)
    if bad == "confidence": t=replace(t,confidence=.2)
    assert e.observe(t,f,f.capture_timestamp+.1,66).inward_turnaround_rate_dps is None


@pytest.mark.parametrize("fresh", [True,False])
def test_only_fresh_translation_ends_recenter_not_old_forward_intent(owner, fresh):
    c=controller(); c.set_normal_parking(True,1); c.set_normal_parking(False,1)
    owner._follow_controller=c
    c.last_steering_pid_result=refresh(c,now=NOW)
    _intent(owner,x_ratio=.3193,mode="forward",base_rpm=50,
            initial_correction_rpm=-7,correction_limit_rpm=7,target_image_rate_dps=None)
    owner._depth30_linear_snapshot=("forward",25,1,NOW-(.05 if fresh else .3))
    owner._action_runtime=SimpleNamespace(get_steering_feedback=lambda:None)
    owner._service_lateral_intent(NOW)
    assert c.post_park_recenter_limit(1) == (None if fresh else 4)
