"""No hardware: capture clock, bounded geometry, main/fast brake agreement."""
from dataclasses import replace
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest

from car_control_modular.control_types import PersonTarget, DepthTargetObservation, SensorFrame
from car_control_modular.controllers import FollowSafetyController, FollowPolicyConfig
from car_control_modular.steering_pid import VisualSteeringPid, VisualSteeringPidConfig
from car_control_modular.visual_steering_evidence import CaptureSteeringEvidence, matching_control_x


def sample(cap, stamp, x, tracker_x=None, uid=1, raw=6):
    tracker_x = x if tracker_x is None else tracker_x
    box = (x*640-60, 50, x*640+60, 450)
    track = (tracker_x*640-65, 45, tracker_x*640+65, 455)
    target = PersonTarget(track, uid, .95, 130*410,
        DepthTargetObservation(box, uid, raw, cap, stamp))
    frame = SensorFrame(width=640, height=480, persons=[target], capture_frame_id=cap,
        capture_timestamp=stamp, distance_m=1.28)
    return target, frame


def observe(e, cap, stamp, x, **kw):
    t, f = sample(cap, stamp, x, **kw)
    return e.observe(t, f, stamp+.1, 60)


def pid_config():
    return VisualSteeringPidConfig(enabled=True, image_error_only=True, image_brake_assist=True,
        image_capture_motion=True, camera_hfov_deg=60, deadband_deg=3,
        dynamic_small_error_deg=3, dynamic_large_error_deg=10,
        max_correction_rpm=10, predictive_brake_margin_deg=1.25)


def test_rate_uses_capture_time_not_variable_processing_delay():
    e = CaptureSteeringEvidence()
    for cap, stamp, x, delay in [(441,10.,.82,.05),(447,10.1,.75,.19),(451,10.2,.68,.1)]:
        t,f = sample(cap,stamp,x,tracker_x=x+.03)
        result = e.observe(t,f,stamp+delay,60)
    assert result.rate_dps == pytest.approx(-42)
    assert result.control_x == pytest.approx(.68)
    assert result.span_sec == pytest.approx(.2)


def test_duplicate_and_old_frames_cannot_extend_or_update_velocity():
    e = CaptureSteeringEvidence()
    observe(e,1,10.,.82)
    t,f = sample(2,10.1,.75)
    r = e.observe(t,f,10.2,60)
    assert r.rate_dps is None
    assert e.observe(t,f,10.3,60) is r
    assert len(e.samples) == 2
    assert e.observe(t,f,10.4,60).rate_dps is None
    assert observe(e,1,10.,.85).rate_dps is None
    assert len(e.samples) == 2
    assert observe(e,3,10.2,.68).rate_dps == pytest.approx(-42)


@pytest.mark.parametrize("change", ["uid", "raw", "gap", "jump", "area", "missing", "confidence", "stamp", "source", "future"])
def test_bad_association_breaks_rate_chain(change):
    e = CaptureSteeringEvidence()
    observe(e,1,10.,.8); observe(e,2,10.1,.75)
    t,f = sample(3,10.2,.7)
    now = 10.3
    if change == "uid": t,f = sample(3,10.2,.7,uid=2)
    if change == "raw": t,f = sample(3,10.2,.7,raw=7)
    if change == "gap": t,f = sample(3,10.5,.7); now=10.6
    if change == "jump": t,f = sample(3,10.2,.4)
    if change == "area": t=replace(t,depth_observation=replace(t.depth_observation,bbox=(410,50,485,200)))
    if change == "missing": t=replace(t,depth_observation=None)
    if change == "confidence": t=replace(t,confidence=.2)
    if change == "stamp": t=replace(t,depth_observation=replace(t.depth_observation,capture_timestamp=10.15))
    if change == "source": t=replace(t,depth_observation=replace(t.depth_observation,source="predicted"))
    if change == "future": now=10.19
    assert e.observe(t,f,now,60).rate_dps is None


def test_jitter_does_not_become_closing_rate():
    e=CaptureSteeringEvidence()
    observe(e,1,10.,.7); observe(e,2,10.1,.75)
    assert observe(e,3,10.2,.7).reason == "rate_inconsistent"


def test_position_bounded_and_cannot_reverse_on_detector_crossing():
    e=CaptureSteeringEvidence()
    observe(e,1,10.,.61,tracker_x=.69)
    r=observe(e,2,10.1,.598099,tracker_x=.6563376)
    assert r.control_x == pytest.approx(.598099)
    assert abs(r.control_x-r.tracker_x) <= .06
    e=CaptureSteeringEvidence()
    observe(e,1,10.,.50,tracker_x=.54)
    r=observe(e,2,10.1,.49,tracker_x=.53)
    assert r.control_x == .5


@pytest.mark.parametrize("x", [.2,.35,.44,.5,.56,.65,.8])
@pytest.mark.parametrize("rate", [-80,0,80,None,float("nan")])
def test_visual_motion_only_reduces_same_side_demand(x,rate):
    base=VisualSteeringPid(pid_config()).update(x,58,None,now=10,visual_age_sec=.1)
    result=VisualSteeringPid(pid_config()).update(x,58,None,now=10,
        visual_age_sec=.1,target_image_rate_dps=rate)
    assert abs(result.correction_rpm) <= abs(base.correction_rpm)
    assert result.correction_rpm*(x-.5) >= 0
    assert result.base_rpm == base.base_rpm == 58


def test_cap447_visual_approach_changes_braking_even_without_qualified_yaw():
    def run(rate):
        return VisualSteeringPid(pid_config()).update(.7944,0,None,now=10,
            visual_age_sec=.1,target_image_rate_dps=rate)
    assert run(0).correction_rpm == run(80).correction_rpm == 10
    assert run(-80).correction_rpm == 0
    assert run(-80).predictive_braking
    assert run(-80).forward_phase == "image_visual_brake:predictive_stop"


@pytest.mark.parametrize("age", [None,-.1,.251,float("nan")])
def test_invalid_visual_age_cannot_brake(age):
    r=VisualSteeringPid(pid_config()).update(.7944,0,None,now=10,
        visual_age_sec=age,target_image_rate_dps=-80)
    assert not r.target_rate_valid and r.correction_rpm == 10


def controller():
    return FollowSafetyController(FollowPolicyConfig(visible_steering_pid_enable=True,
        visible_steering_pid_image_error_only=True, visible_steering_pid_image_brake_assist=True,
        visible_steering_pid_image_capture_motion=True, visible_steering_pid_camera_hfov_deg=60,
        visible_steering_pid_deadband_deg=3,visible_steering_pid_dynamic_large_error_deg=10,
        visible_steering_pid_predictive_brake_margin_deg=1.25,
        center_left_ratio=.45,center_right_ratio=.55,forward_max_rpm=200))


@pytest.mark.parametrize("parked", [False,True])
def test_real_controller_paths_and_fast_loop_share_capture_evidence(monkeypatch, parked):
    c=controller()
    monkeypatch.setattr(c,"_visible_base_forward_percent",lambda *a,**kw:29)
    for cap,stamp,x in [(443,10.,.85),(447,10.1,.77),(451,10.2,.69)]:
        t,f=sample(cap,stamp,x,tracker_x=x+.04)
        if parked:
            c._pid_action_for_parked_target(t,f,stamp+.1,"right",near_distance_mode=True)
        else:
            c._pid_action_for_visible_target(t,f,stamp+.1)
    r=c.last_steering_pid_result
    o=c.last_capture_steering_observation
    assert r.target_rate_valid and r.target_image_rate_dps == pytest.approx(-48)
    assert o.control_x == pytest.approx(.69)
    assert r.correction_rpm == 0 and r.predictive_braking
    refresh = c.refresh_parked_lateral_pid if parked else c.refresh_visible_lateral_pid
    fast=refresh(x_ratio=o.control_x,base_rpm=r.base_rpm,feedback=None,now=10.31,
        target_image_rate_dps=r.target_image_rate_dps,visual_age_sec=.11)
    assert fast.correction_rpm == 0 and fast.predictive_braking
    assert matching_control_x(o,1,451,10.2,.73) == pytest.approx(.69)
    assert matching_control_x(o,2,451,10.2,.73) == .73
    assert matching_control_x(o,1,452,10.3,.73) == .73


def test_ini_through_real_constructor_binding(monkeypatch):
    import os
    from car_control_modular.config_loader import load_config_to_env
    root=Path(__file__).resolve().parents[2]
    monkeypatch.setattr(os,"environ",{})
    load_config_to_env(str(root/"car_control_modular/config/reid_runtime.ini"))
    assert os.environ["VISIBLE_STEERING_PID_IMAGE_CAPTURE_MOTION"] == "1"
    assert float(os.environ["VISIBLE_STEERING_PID_IMAGE_MOTION_RESPONSE_SEC"]) == .18
    tree=ast.parse((root/"request_0513_modular.py").read_text())
    names={"visible_steering_pid_image_capture_motion","visible_steering_pid_image_motion_response_sec"}
    values={k.arg:eval(compile(ast.Expression(k.value),"binding","eval"),{
        "VISIBLE_STEERING_PID_IMAGE_CAPTURE_MOTION":True,"VISIBLE_STEERING_PID_IMAGE_MOTION_RESPONSE_SEC":.18})
        for n in ast.walk(tree) if isinstance(n,ast.Call) for k in n.keywords if k.arg in names}
    c=FollowSafetyController(FollowPolicyConfig(**values))
    assert c._visual_steering_pid.config.image_capture_motion
    assert c._parked_recenter_pid.config.image_motion_response_sec == .18
