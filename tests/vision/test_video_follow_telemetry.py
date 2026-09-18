"""Per-capture diagnostics are not future measurements or motor commands."""
import csv
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from car_control_modular.control_types import (
    ControlDecision, DepthLinearTiming, DistanceState, PersonTarget, SensorFrame,
)
from car_control_modular.longitudinal_feedforward import LongitudinalFeedforwardEvidence
from car_control_modular.video_follow_telemetry import (
    FollowRecordingSnapshot, FollowRecordingView, build_follow_snapshot, recording_authority,
)
from car_control_modular.video_recorder import AsyncVideoRecorder, VideoRecorderConfig, VideoFrameOverlay


def snapshot(**kw):
    return replace(FollowRecordingSnapshot(
        published_at=100.05, capture_frame_id=123, uid=1, source='depth30',
        reason='longitudinal_distance_pid', set_distance_m=1.5, target_x=.61,
        raw_distance_m=2.0, used_distance_m=1.98, depth_timestamp=100., depth_detail='depth_multiregion',
        speed_timestamp=100., target_speed_m_s=.45, relative_speed_m_s=.10, speed_status='ready',
        pid_timestamp=100., pid_rpm=55., matching_base_rpm=40.,
        forward_scale_rpm=200.), **kw)


def timing(**kw):
    return replace(DepthLinearTiming(snapshot=('forward', 24, 1, 100.),
        accepted_depth_timestamp=100., depth_expires_at=100.18), **kw)


def test_current_speed_units_distance_error_pid_and_authority_are_distinct():
    v = FollowRecordingView.from_snapshot(snapshot(), timing(), 100.10)
    assert v.target_speed_m_s == .45 and v.relative_speed_m_s == .10
    assert v.distance_error_m == pytest.approx(.48)
    assert v.pid_rpm == 55 and v.matching_base_rpm == 40 and v.authorized_rpm == 48
    assert v.depth_age_ms == pytest.approx(100)
    assert v.snapshot_age_ms == pytest.approx(50)
    assert v.observation_cap == 123
    assert 'Vt +0.45 Vrel +0.10 m/s FRESH' in v.labels()[0]


@pytest.mark.parametrize('now,status', [(99., 'future'), (100.04,'future'), (100.25,'stale')])
def test_future_or_expired_snapshot_never_claims_current_target_speed(now,status):
    v = FollowRecordingView.from_snapshot(snapshot(), timing(), now)
    assert v.status == status
    assert v.target_speed_m_s is None
    assert 'Vt n/a' in v.labels()[0]


def test_fresh_publish_does_not_renew_old_speed_or_depth():
    v = FollowRecordingView.from_snapshot(snapshot(published_at=100.22),timing(),100.23)
    assert v.status == 'current' and v.speed_status == 'stale'
    assert v.target_speed_m_s is None and v.pid_rpm is None
    assert v.depth_status == 'stale' and v.authority_status == 'expired'
    assert v.authorized_rpm == 0


def test_missing_not_zero_but_true_stationary_zero_is_displayed():
    assert FollowRecordingView.from_snapshot(None,None,100).target_speed_m_s is None
    v = FollowRecordingView.from_snapshot(snapshot(target_speed_m_s=0.,
        speed_status='no_forward_motion', relative_speed_m_s=-.3),None,100.1)
    assert v.target_speed_m_s == 0 and v.relative_speed_m_s == -.3
    assert 'Vt +0.00' in v.labels()[0]


def test_bridge_uses_original_speed_time_not_new_depth_time():
    c, f, d = builder_inputs()
    c._longitudinal_motion_evidence = replace(c._longitudinal_motion_evidence,
        status='transient_bridge', sample_timestamp=100.2)
    c._longitudinal_bridge = SimpleNamespace(origin=SimpleNamespace(sample_timestamp=100.))
    s = build_follow_snapshot(c,f,d,'depth30',100.21)
    assert s.speed_timestamp == 100.
    v = FollowRecordingView.from_snapshot(s,None,100.22)
    assert v.target_speed_m_s is None and v.speed_status == 'stale'


def builder_inputs():
    evidence = LongitudinalFeedforwardEvidence('ready', target_id=1, sample_timestamp=100.,
        target_speed_m_s=.45, range_rate_m_s=.10)
    c = SimpleNamespace(active_target_id=1,search_state='none',
        cfg=SimpleNamespace(target_distance_m=1.5,forward_max_rpm=200,reverse_max_rpm=60),
        _longitudinal_motion_evidence=evidence,_distance_pid_last_sample_timestamp=100.,
        last_distance_pid_result=SimpleNamespace(output_rpm=55,tracking_base_rpm=40))
    f = SensorFrame(width=640,height=480,persons=[PersonTarget((200,20,400,460),1,.9,88000)],
        distance_m=1.98,distance_state=DistanceState(raw_distance_m=2.,sample_timestamp=100.),
        capture_frame_id=123)
    return c,f,ControlDecision(reason='test')


@pytest.mark.parametrize('case',['uid','lost','search'])
def test_unconfirmed_other_uid_or_missing_person_cannot_inherit_speed(case):
    c,f,d = builder_inputs()
    if case == 'uid': c._longitudinal_motion_evidence=replace(c._longitudinal_motion_evidence,target_id=2)
    elif case == 'lost': f=replace(f,persons=[])
    else: c.search_state='searching'
    s = build_follow_snapshot(c,f,d,'vision',100.05)
    assert s.target_speed_m_s is None


def test_snapshot_does_not_mutate_controller_and_remains_immutable():
    c,f,d = builder_inputs()
    before = vars(c).copy()
    s = build_follow_snapshot(c,f,d,'depth30',100.05)
    assert vars(c) == before
    c._longitudinal_motion_evidence=replace(c._longitudinal_motion_evidence,target_speed_m_s=1.2)
    assert s.target_speed_m_s == .45
    with pytest.raises(FrozenInstanceError): s.uid=2


@pytest.mark.parametrize('case',['revoke','stop','brake','search','mismatch'])
def test_readonly_authority_accessor_does_not_resurrect_cleared_authority(case):
    t=timing()
    owner=SimpleNamespace(_depth30_linear_timing=t,_depth30_linear_snapshot=t.snapshot)
    assert recording_authority(owner) is t
    if case=='revoke': owner._depth30_linear_snapshot=None
    elif case=='stop': owner._explicit_stop_requested=True
    elif case=='brake': owner._brake_hold_active=True
    elif case=='search': owner.search_state='searching'
    else: owner._depth30_linear_snapshot=('forward',30,2,100.)
    assert recording_authority(owner) is None


def test_feedforward_expiry_is_not_depth_expiry_and_not_actual_motor_rpm():
    t=timing(feedforward_expires_at=100.07,feedforward_timestamp=99.89,distance_only_percent=10)
    v=FollowRecordingView.from_snapshot(snapshot(),t,100.1)
    assert v.authority_status=='distance_only' and v.authorized_rpm==20
    assert t.snapshot[1]==24  # Rendering cannot mutate authority.


def test_authority_uid_mismatch_and_reverse_units():
    assert FollowRecordingView.from_snapshot(snapshot(uid=2),timing(),100.1).authorized_rpm is None
    t=timing(snapshot=('backward',20,1,100.))
    # Reverse percent uses the same forward RPM scale, NOT its independent cap.
    assert FollowRecordingView.from_snapshot(snapshot(),t,100.1).authorized_rpm==-40


def test_nan_and_warming_are_not_reported_as_valid_speed():
    v=FollowRecordingView.from_snapshot(snapshot(target_speed_m_s=float('nan'),
        relative_speed_m_s=None,speed_status='warming_up'),None,100.1)
    assert v.target_speed_m_s is None and v.speed_status=='unavailable'


def test_each_capture_has_its_own_snapshot_even_without_yolo_metadata(tmp_path):
    r=AsyncVideoRecorder(VideoRecorderConfig(str(tmp_path/'v.avi'),30,overlay_wait_sec=0),cv2_module=cv2)
    img=np.zeros((480,640,3),np.uint8)
    assert r.submit(img,capture_frame_id=124,monotonic_sec=100.1,
        follow_snapshot=snapshot(),linear_timing=timing())
    assert r.submit(img,capture_frame_id=125,monotonic_sec=100.12,
        follow_snapshot=snapshot(published_at=100.11,target_speed_m_s=.6))
    assert r.submit(img,capture_frame_id=126,monotonic_sec=100.3,follow_snapshot=snapshot())
    assert r.submit(img,capture_frame_id=127,monotonic_sec=100.4)
    assert r.close(timeout_sec=3) and r.error is None
    rows=list(csv.DictReader(Path(r.index_path).open()))
    assert [x['follow_target_speed_m_s'] for x in rows]==['0.4500','0.6000','','']
    assert rows[0]['follow_authorized_rpm']=='48.0000'
    assert rows[0]['follow_observation_cap']=='123'
    assert all(x['control_frame_id']=='' for x in rows)
    assert not img.any()


def test_all_five_labels_render_on_frames_without_control_metadata():
    r=AsyncVideoRecorder(VideoRecorderConfig('unused.avi',30),cv2_module=cv2)
    lines=[]
    r._draw_text_box=lambda image,text,**kw: lines.append(text)
    v=FollowRecordingView.from_snapshot(snapshot(),timing(),100.1)
    r._annotate_frame(np.zeros((480,640,3),np.uint8),1,124,VideoFrameOverlay(),follow=v)
    assert all(line in lines for line in v.labels())


def test_runtime_diagnostic_failure_cannot_interrupt_control(monkeypatch):
    import request_0513_modular as runtime
    owner=SimpleNamespace(_camera_video_recorder=object(),_follow_controller=object())
    def fail(*args): raise ValueError('broken diagnostic only')
    monkeypatch.setattr(runtime,'build_follow_snapshot',fail)
    runtime.PersonTracker._publish_follow_recording(owner,None,None,'depth30')
    assert owner._video_follow_snapshot is None
    assert owner._video_follow_error_logged
