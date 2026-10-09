"""CAP58: collect while parked, then small capture-bound outward correction."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

import car_control_modular.controllers as control
import request_0513_modular as runtime
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.control_types import ControlAction, SteeringFeedback
from car_control_modular.near_yaw_parking import NearYawParkRequest, ParkSettlingEvidence
from car_control_modular.visual_steering_evidence import CaptureSteeringEvidence
from car_control_modular.outward_trajectory import make_outward_lead, apply_outward_lead
from test_cap443_capture_braking import sample
from test_lateral_zero_runtime import owner, NOW


def config(**changes):
    return replace(FollowPolicyConfig(visible_steering_pid_enable=True,
        visible_steering_pid_image_error_only=True, visible_steering_pid_image_brake_assist=True,
        visible_steering_pid_image_capture_motion=True, visible_steering_pid_camera_hfov_deg=66,
        visible_steering_pid_deadband_deg=3, visible_steering_pid_dynamic_large_error_deg=10,
        visible_steering_pid_image_center_release_margin_deg=1.8,
        visible_steering_pid_outward_lead_enable=True,
        parked_recenter_min_rpm=5, parked_recenter_max_rpm=14,
        near_distance_rotation_only_max_rpm=7), **changes)


def episode(monkeypatch, sign=1, start=10., cfg=None, positions=(.495, .51, .54)):
    c = FollowSafetyController(cfg or config())
    c.active_target_id = c._near_settle_target_id = 1
    c._near_settle_until = start+5
    outputs = []
    for i, pos in enumerate(positions):
        stamp = start+i*.1
        monkeypatch.setattr(control.time, "monotonic", lambda t=stamp: t+.09)
        t, f = sample(574+i, stamp, .5+sign*(pos-.5))
        decision = c._near_distance_rotation_only_decision(i, f, t, 1.28, 1.43)
        outputs.append(decision)
    return c, t, f, outputs


@pytest.mark.parametrize("sign", [-1, 1])
def test_collects_during_hold_and_starts_small_before_center_release(monkeypatch, sign):
    c, t, f, decisions = episode(monkeypatch, sign)
    assert [d.actions[0].kind for d in decisions[:2]] == ["stop", "stop"]
    result = c.last_steering_pid_result
    assert result.output_floor_reason == "outward_trajectory_lead"
    assert sign*result.correction_rpm == 4
    assert decisions[-1].actions[0].kind == ("rotate_right" if sign > 0 else "rotate_left")
    assert not decisions[-1].near_yaw_park_requested
    assert decisions[-1].current_forward_percent == 0
    assert c.last_capture_steering_observation.span_sec == pytest.approx(.2)
    lead = result.outward_lead
    # Fast loop consumes exactly this certificate; it neither resamples nor
    # amplifies the per-wheel 4 RPM into the normal startup floor of 5.
    for now in [10.30, 10.33, 10.36]:
        fast = c.refresh_parked_lateral_pid(x_ratio=lead.x, base_rpm=0,
            feedback=None, now=now, target_image_rate_dps=lead.rate_dps,
            max_correction_rpm=7, visual_age_sec=now-lead.capture_timestamp,
            near_distance_mode=True, outward_lead=lead)
        assert fast.correction_rpm == result.correction_rpm
        assert fast.outward_lead is lead
        assert lead.capture_timestamp == 10.2


@pytest.mark.parametrize("positions", [(.51,.51,.51), (.51,.54,.52), (.53,.52,.51),
                                      (.501,.504,.507), (.52,.52,.54)])
def test_jitter_static_inward_and_single_jump_do_not_qualify(monkeypatch, positions):
    c, t, f, decisions = episode(monkeypatch, positions=positions)
    assert getattr(c.last_steering_pid_result, "outward_lead", None) is None


def test_disabled_preserves_hold(monkeypatch):
    c, t, f, ds = episode(monkeypatch, cfg=config(visible_steering_pid_outward_lead_enable=False))
    assert ds[-1].actions[0].kind == "stop"


@pytest.mark.parametrize("change", ["duplicate", "uid", "raw", "quality", "braking_only", "gap"])
def test_bad_capture_chain_cannot_grant_lead(change):
    evidence = CaptureSteeringEvidence()
    for cap, stamp, x in [(1,10.,.495),(2,10.1,.51),(3,10.2,.54)]:
        t,f=sample(cap,stamp,x)
        if cap == 3:
            if change == "duplicate": t,f=sample(2,10.1,.54)
            if change == "uid": t,f=sample(cap,stamp,x,uid=2)
            if change == "raw": t,f=sample(cap,stamp,x,raw=99)
            if change == "quality": t=replace(t,confidence=.2)
            if change == "gap": t,f=sample(cap,10.5,x); stamp=10.5
            if change == "braking_only":
                t=replace(t,depth_observation=None,
                    braking_observation=replace(t.depth_observation,source="yolo_braking_only"))
        obs=evidence.observe(t,f,stamp+.09,66)
    assert make_outward_lead(obs, stamp+.09, hfov=66,deadband=3,
                            release_margin=1.8,enabled=True) is None


def test_predictive_braking_quality_cap_and_expiry_dominate_lead(monkeypatch):
    c,t,f,ds=episode(monkeypatch)
    result=c.last_steering_pid_result
    lead=result.outward_lead
    base=replace(result, outward_lead=None, correction_rpm=0, output_floor_reason="center_hold")
    for guarded in [replace(base,predictive_braking=True),
                    replace(base,correction_limit_rpm=1),
                    replace(base,correction_limit_rpm=3),
                    replace(base,output_floor_reason="predictive_countersteer")]:
        assert apply_outward_lead(guarded,lead,lead.x,10.29) is guarded
    assert apply_outward_lead(base,lead,lead.x,10.451) is base
    assert apply_outward_lead(base,lead,.49,10.29) is base


def parked_owner(owner, monkeypatch):
    c,t,f,ds=episode(monkeypatch,start=NOW-.29)
    monkeypatch.setattr(runtime.time,"monotonic",lambda:NOW)
    owner._follow_controller=c
    owner._last_command_capture_frame=f.capture_frame_id
    owner._last_command_capture_timestamp=f.capture_timestamp
    req=NearYawParkRequest(1,570,NOW-.80,NOW-.70,"center_hold")
    owner._near_yaw_park_request=req
    owner._near_yaw_park_evidence=(573,NOW-.40)
    owner._brake_hold_active=True
    owner._brake_hold_label="near_yaw_park"
    evidence=ParkSettlingEvidence(req,NOW-.6)
    feedback=lambda ts: SteeringFeedback(timestamp=ts,trustworthy=True,
        left_forward_rpm=0,right_forward_rpm=0)
    evidence.observe(feedback(NOW-.3),NOW-.3)
    evidence.observe(feedback(NOW-.24),NOW-.24)
    evidence.observe(feedback(NOW-.12),NOW-.12)
    owner._action_runtime=SimpleNamespace(_near_yaw_park_settling=evidence,
        near_yaw_park_release_ready=lambda req,stamp,now:
            evidence.request is req and evidence.release_ready(stamp,feedback(NOW-.02),now),
        near_yaw_park_motion_ready=lambda *a: pytest.fail("early lead used loose motion gate"),
        get_steering_feedback=lambda:feedback(NOW-.02))
    return c,t,f,ds,evidence


def publish(owner,t,ds):
    return owner._publish_lateral_intent_from_decision(width=640,target=t,
        runtime_actions=ds[-1].actions,control_source="vision",
        target_steerable=True,low_quality_visible=False)


def test_end_to_end_park_release_publish_and_fast_tick(owner,monkeypatch):
    c,t,f,ds,e=parked_owner(owner,monkeypatch)
    assert publish(owner,t,ds)
    assert owner._near_yaw_park_request is None
    assert not owner._brake_hold_active
    intent=owner._lateral_intent_store.snapshot()
    assert intent.outward_lead is c.last_steering_pid_result.outward_lead
    assert intent.base_percent == 0 and intent.mode == "yaw_only"
    owner._service_lateral_intent(NOW)
    owner._service_lateral_intent(NOW+.01)
    assert 2 <= owner._current_rotate_raw_target <= 4
    assert owner._depth30_linear_snapshot is None


@pytest.mark.parametrize("bad",["minimum_hold","pre_stop_samples","old_capture","unquiet", "safety", "wrong_uid"])
def test_early_release_keeps_all_parking_safety_gates(owner,monkeypatch,bad):
    c,t,f,ds,e=parked_owner(owner,monkeypatch)
    if bad == "minimum_hold": e.sent_at=NOW-.05
    if bad == "pre_stop_samples": e.sent_at=NOW-.15
    if bad == "old_capture": owner._last_command_capture_frame-=1
    if bad == "unquiet": owner._action_runtime.near_yaw_park_release_ready=lambda *a:False
    if bad == "safety": owner._brake_hold_label="safety_hold"
    if bad == "wrong_uid": t=replace(t,track_id=2)
    assert not publish(owner,t,ds)
    assert owner._near_yaw_park_request is not None
    assert owner._depth30_linear_snapshot is None


def test_static_new_frame_removes_old_lead(owner,monkeypatch):
    c,t,f,ds,e=parked_owner(owner,monkeypatch)
    assert publish(owner,t,ds)
    t,f=sample(577,NOW-.01,.54)
    c._near_distance_rotation_only_decision(4,f,t,1.28,1.43)
    assert getattr(c.last_steering_pid_result,"outward_lead",None) is None


def test_ini_wires_explicit_feature_switch(monkeypatch):
    from pathlib import Path
    import os
    from car_control_modular.config_loader import load_config_to_env
    monkeypatch.delenv("VISIBLE_STEERING_PID_OUTWARD_LEAD_ENABLE",raising=False)
    load_config_to_env(str(Path(__file__).resolve().parents[2]/"car_control_modular/config/reid_runtime.ini"))
    assert os.environ["VISIBLE_STEERING_PID_OUTWARD_LEAD_ENABLE"] == "1"


def test_braking_only_samples_cannot_seed_recovered_outward_authority():
    e=CaptureSteeringEvidence()
    for cap,stamp,x in [(1,10.,.495),(2,10.1,.51),(3,10.2,.54)]:
        t,f=sample(cap,stamp,x)
        if cap < 3:
            t=replace(t,depth_observation=None,
                braking_observation=replace(t.depth_observation,source="yolo_braking_only"))
        obs=e.observe(t,f,stamp+.09,66)
    assert obs.reason == "warming" and obs.rate_dps is None


def test_slew_cannot_leave_previous_large_yaw_above_lead_cap(owner,monkeypatch):
    c,t,f,ds,e=parked_owner(owner,monkeypatch)
    assert publish(owner,t,ds)
    owner._lateral_intent_last_correction_rpm=7
    owner._lateral_intent_last_tick_ts=NOW-.001
    owner._service_lateral_intent(NOW)
    assert 0 < owner._current_rotate_raw_target <= 4
