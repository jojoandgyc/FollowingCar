"""CAP606 cache and CAP664 live-grant gap regressions; no hardware."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from test_depth_raw_geometry_runtime import make_runtime, person
from test_distance_tracking_response import setup
from test_distance_pi_controller import configured, accumulated, step, integral


def depth(stamp, *, replay=False):
    return SimpleNamespace(
        distance_m=1.5346, raw_distance_m=None if replay else 1.5346,
        sample_timestamp=None if replay else stamp,
        observation_sample_timestamp=stamp,
        temporal_status="older_than_anchor" if replay else "new_sample",
        sample_age_sec=.167 if replay else .019,
        valid_pixels=0 if replay else 16000,
        detail="depth_observation_reused_hold" if replay else "depth_multiregion",
    )


@pytest.mark.parametrize("clear_reason", ["stale", "identity_changed", "invalid_bbox", "no_target"])
def test_stale_roi_keeps_only_same_uid_unexpired_anchor(monkeypatch, clear_reason):
    clock = SimpleNamespace(now=100.019)
    monkeypatch.setattr("car_control_modular.distance_runtime.time.monotonic", lambda: clock.now)
    r, sensor = make_runtime(vision_depth_detector_bbox_max_age_sec=.25)
    sensor.measurement = depth(100.)
    target = person(capture_timestamp=99.9)
    assert r.get_vision_depth_state(640,480,target).used_distance_m == pytest.approx(1.5346)
    clock.now = 100.05
    rejected = replace(target, depth_observation=replace(target.depth_observation,capture_timestamp=99.7))
    if clear_reason == "identity_changed":
        rejected = replace(rejected,track_id=2,depth_observation=replace(rejected.depth_observation,target_id=2))
    elif clear_reason == "invalid_bbox":
        rejected = replace(rejected,depth_observation=replace(rejected.depth_observation,bbox=(0.,0.,-1.,480.)))
    elif clear_reason == "no_target":
        rejected = None
    failed = r.get_vision_depth_state(640,480,rejected,use_latest_depth=True)
    assert failed.raw_distance_m is None and failed.sample_timestamp is None
    assert len(sensor.calls)==1  # stale box must never sample new depth
    clock.now=100.167
    sensor.measurement=depth(99.969,replay=True)
    replay=r.get_vision_depth_state(640,480,person(capture_timestamp=100.))
    assert replay.raw_distance_m is replay.sample_timestamp is None
    if clear_reason == "stale":
        assert replay.used_distance_m==pytest.approx(1.5346)
        assert replay.sample_age_sec==pytest.approx(.167)
        assert r._vision_depth_fusion._last_anchor_ts==100.
    else:
        assert replay.used_distance_m is None


def pi_sample(c, stamp, now, *, ego=27.5, distance=2.4, rate=.5):
    return c.update(distance,1.5,sample_timestamp=stamp,execution_now=now,
                    deadband_m=.03,max_output_rpm=200,rise_rpm_per_sec=240,
                    ego_forward_rpm=ego,range_rate_m_s=rate,raw_closure_valid=True)


@pytest.mark.parametrize("gap", [.181,.186,.220,.249])
def test_live_grant_sampling_gap_freezes_integral_without_restarting_ramp(gap):
    c=DistancePiController(DistancePiConfig(physical_ttl_sec=.25))
    first=pi_sample(c,100.,100.,ego=66.)
    result=pi_sample(c,100.+gap,100.+gap)
    assert result.sample_dt_sec==0 and result.integral_m_s==first.integral_m_s
    assert result.output_rpm>=first.output_rpm>27.5
    assert result.status=="tracking_gap_no_integral"


@pytest.mark.parametrize("event", ["expired","revoked","braking"])
def test_gap_fix_cannot_override_real_stop_or_braking(event):
    c=DistancePiController(DistancePiConfig(physical_ttl_sec=.25))
    pi_sample(c,100.,100.,ego=66.)
    if event=="revoked": c.suspend(100.15,"actual_revoke",reset_execution=True)
    stamp=100.26 if event=="expired" else 100.186
    result=pi_sample(c,stamp,stamp,distance=1.5 if event=="braking" else 2.4,
                     rate=-27.5*.816814/60 if event=="braking" else .5)
    assert result.output_rpm<=27.5
    if event=="braking": assert result.output_rpm==0
    else: assert result.status=="recovering"


@pytest.mark.parametrize("distance", [1.6,1.8])
def test_p_trial_doubles_small_error_response_without_changing_safety(distance):
    outputs=[]
    for kp in (1.,2.):
        c=DistancePiController(DistancePiConfig(kp_per_sec=kp))
        pi_sample(c,100.,100.,ego=0,distance=distance)
        r=pi_sample(c,100.1,100.1,ego=10,distance=distance)
        outputs.append(r.p_m_s)
    assert outputs[1]==pytest.approx(2*outputs[0])


@pytest.mark.parametrize("elapsed,box_stamp", [(.201,99.7),(.05,100.06),(5.,99.7)])
def test_expired_anchor_or_future_box_cannot_be_retained(monkeypatch, elapsed, box_stamp):
    clock=SimpleNamespace(now=100.019)
    monkeypatch.setattr("car_control_modular.distance_runtime.time.monotonic",lambda:clock.now)
    r,sensor=make_runtime(vision_depth_detector_bbox_max_age_sec=.25)
    sensor.measurement=depth(100.)
    r.get_vision_depth_state(640,480,person())
    clock.now=100.+elapsed
    r.get_vision_depth_state(640,480,person(capture_timestamp=box_stamp),use_latest_depth=True)
    assert r._vision_depth_fusion._last_anchor_ts is None
    assert len(sensor.calls)==1


@pytest.mark.parametrize("veto", [None,"stopped","uid_changed","backward","different_sample"])
def test_runtime_authority_not_just_pi_clock_controls_gap_resume(setup,veto):
    clock,c,frame=configured(setup,depth_longitudinal_sample_max_age_sec=.25)
    step(c,frame(2.4,rpm=66))
    stamp=c._distance_pid_last_sample_timestamp
    previous=c.last_distance_pid_result.output_rpm
    live=("forward",int(previous/2),1,stamp)
    if veto=="stopped": live=None
    elif veto=="uid_changed": live=("forward",30,2,stamp)
    elif veto=="backward": live=("backward",30,1,stamp)
    elif veto=="different_sample": live=("forward",30,1,stamp-.01)
    c._live_longitudinal_authority_reader=lambda uid:live
    clock.now+=.186
    step(c,frame(2.4,rpm=27.5))
    r=c.last_distance_pid_result
    assert r.pi_sample_dt_sec==0
    if veto is None:
        assert r.pi_status=="tracking_gap_no_integral"
    else:
        assert r.pi_status=="recovering" and r.output_rpm<=27.5


def test_same_uid_old_observation_preserves_pi_memory_without_new_action(setup):
    clock,c,frame=accumulated(setup,depth_longitudinal_sample_max_age_sec=.25)
    stamp=c._distance_pid_last_sample_timestamp
    before=integral(c)
    current=frame(1.8)
    current=replace(current,distance_state=replace(current.distance_state,
        raw_distance_m=None,sample_timestamp=None,observation_timestamp=stamp-.03,
        temporal_status="older_than_anchor",fusion_confidence=.45,fusion_mode="depth_radar_hold",
        source_detail="depth_sample_observation_discarded_fused_radar_hold_hold"))
    decision=step(c,current)
    assert not decision.actions
    assert integral(c)==before
    assert c._distance_pid_last_sample_timestamp==stamp
