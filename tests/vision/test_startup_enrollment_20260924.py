"""Actual startup crop geometry/times, synthetic vectors; no camera/motors."""
import numpy as np
import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from rk_vision.template_memory import TemplateMemory
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection
from car_control_modular.video_recorder import (
    VideoFrameOverlay, VideoControlOverlay, VideoTrackOverlay, VideoDetectionOverlay, startup_status,
)
from car_control_modular.controllers import FollowSafetyController, FollowPolicyConfig
from car_control_modular.control_types import PersonTarget, SensorFrame

CAP17=(207.1622,1.2317,462.8204,474.9603)
CAP23=(216.9424,2.0483,477.1373,474.5447)
OLD27=(545.4487,107.8929,639.9957,437.1239)
V=np.array([1.,0.,0.],dtype=np.float32)


def meta(cap=17,t=1411.19838795,frame=2,box=CAP17,**kw):
    m=dict(capture_frame_id=cap,capture_timestamp=t,frame_index=frame,control_frame_id=frame,
           image_width=640,image_height=480,detector_bbox=box,bbox=box,quality_bbox_ok=True,
           bbox_quality_tier='strong',is_fresh=True,candidate_count=1,
           detector_edge_touch_count=2,partial_feature_source='osnet_torso',
           detector_center_x_ratio=(box[0]+box[2])/1280,
           detector_area_ratio=(box[2]-box[0])*(box[3]-box[1])/(640*480))
    m.update(kw);return m


def bank():
    return IdentityBank(IdentityBankConfig(template_memory_enable=True,template_crosscheck_enable=True,
        appearance_region_safety_enable=True,new_identity_confirm_frames=2,update_interval=1))


def assign(b,m,track=1,vector=V):
    x1,y1,x2,y2=m['detector_bbox']
    return b.assign(track_id=track,feature=vector,partial_feature=V,confidence=.94,
        area=(x2-x1)*(y2-y1),frame_index=m['frame_index'],sample_metadata=m,
        candidate_count=m['candidate_count'],bbox_quality_ok=m['quality_bbox_ok'],
        bbox_quality_tier=m['bbox_quality_tier'])


def test_actual_startup_crops_create_identity_on_second_new_capture():
    b=bank()
    assert assign(b,meta())==0
    assert b.last_assignments[1]['reason']=='pending_initial_identity'
    assert not b.identities and not b.track_to_uid
    assert assign(b,meta(23,1411.497862257,3,CAP23))==1
    assert b.last_assignments[1]['reason']=='created_confirmed'
    assert b.identities[1].feature_metadata[0]['capture_frame_id']==23
    assert assign(b,meta(24,1411.55,4,CAP23))==1


def test_real_tracker_passes_startup_metadata_and_retains_confirmed_uid():
    tracker = DeepSortTracker(DeepSortTrackerConfig(
        n_init=1, identity_template_memory_enable=True,
        identity_template_crosscheck_enable=True,
        identity_appearance_region_safety_enable=True))
    observations = []
    reasons = []
    for i in range(5):
        records = tracker.update(
            [Detection((207., 1., 463., 475.), .94, 0)], [V],
            partial_features=[V], partial_feature_sources=['osnet_torso'],
            image_width=640, image_height=480,
            frame_context=dict(control_frame_id=i+1, capture_frame_id=i+17,
                               capture_timestamp=1411.+.1*i))
        observations.append([r.reid_uid for r in records])
        reasons.append(tracker.identity_bank.last_assignments.get(1, {}).get('reason'))
    assert observations == [[], [0], [1], [1], [1]]
    assert reasons[1:3] == ['pending_initial_identity', 'created_confirmed']


@pytest.mark.parametrize('box',[CAP17,CAP23,(200.,0.,400.,479.),(200.,40.,400.,460.)])
def test_vertical_cut_and_complete_body_both_eligible(box):
    assert TemplateMemory.initial_crop_usable(meta(box=box))


@pytest.mark.parametrize('box',[OLD27,(0.,1.,80.,479.),(210.,10.,250.,475.),
    (200.,40.,400.,100.),(0.,0.,639.,479.)])
def test_sliver_small_or_laterally_clipped_cannot_bootstrap(box):
    b=bank()
    for cap in range(17,22):
        assert assign(b,meta(cap,1411.+cap*.05,cap,box))==0
    assert not b.identities


@pytest.mark.parametrize('case',['duplicate','older','old_time','gap','jump','area','feature','track','multi','weak','missing'])
def test_second_observation_must_be_new_unique_and_consistent(case):
    b=bank();assign(b,meta());m=meta(23,1411.497862257,3,CAP23);v=V;track=1
    if case=='duplicate':m=meta()
    if case=='older':m['capture_frame_id']=16
    if case=='old_time':m['capture_timestamp']=1411.1
    if case=='gap':m['capture_timestamp']=1412.
    if case=='jump':m['detector_bbox']=(390.,2.,625.,475.)
    if case=='area':m['detector_bbox']=(260.,60.,350.,280.)
    if case=='feature':v=np.array([0.,1.,0.])
    if case=='track':track=2
    if case=='multi':m['candidate_count']=2
    if case=='weak':m['quality_bbox_ok']=False
    if case=='missing':m.pop('capture_timestamp')
    assert assign(b,m,track,v)==0
    assert not b.identities


def test_multi_person_does_not_choose_first_box_or_largest_for_bootstrap():
    b=bank()
    for cap in (17,18,19):
        m=meta(cap,1411.+cap*.05,cap,CAP17,candidate_count=2)
        for track in (1,2):
            assert assign(b,m,track)==0
            assert b.last_assignments[track]['reason']=='initial_candidate_ambiguous'
    assert assign(b,meta(20,1412.0,20))==0
    assert assign(b,meta(21,1412.1,21))==1


def test_reset_removes_bootstrap_proof():
    b=bank();assign(b,meta());b.reset()
    assert assign(b,meta(23,1411.497862257,3,CAP23))==0


def test_startup_helper_is_not_used_for_existing_identity_reacquisition():
    b=bank();assign(b,meta());assign(b,meta(23,1411.497862257,3,CAP23))
    def forbidden(*args):raise AssertionError('startup must not replace ReID')
    b._initial_enrollment.observe=forbidden
    m=meta(40,1412.,20,CAP23,search_reacquire_context_active=True)
    # Orthogonal wrong person cannot pass by repeating a startup candidate.
    for i in range(3):
        m.update(capture_frame_id=40+i,capture_timestamp=1412.+i*.1,frame_index=20+i)
        assert b.assign(track_id=2,feature=np.array([0.,1.,0.]),partial_feature=np.array([0.,1.,0.]),
            confidence=.94,area=120000,frame_index=20+i,sample_metadata=m,
            bbox_quality_ok=True,bbox_quality_tier='strong',preferred_uid=1,preferred_candidate_ok=True)==0


def test_existing_controller_lock_follows_enrollment_without_fixed_delay(monkeypatch):
    b=bank();c=FollowSafetyController(FollowPolicyConfig(initial_target_confirm_frames=2,
        search_before_first_seen=False))
    assert assign(b,meta())==0
    d=c.decide(2,SensorFrame(width=640,height=480,capture_frame_id=17,capture_timestamp=1411.1984))
    assert all(a.kind == 'stop' for a in d.actions) and c.active_target_id is None
    for cap,frame,t in [(23,3,1411.497862257),(24,4,1411.55)]:
        monkeypatch.setattr('car_control_modular.controllers.time.monotonic', lambda: t+.05)
        uid=assign(b,meta(cap,t,frame,CAP23));assert uid==1
        person=PersonTarget(CAP23,uid,.94,120000)
        c.decide(frame,SensorFrame(width=640,height=480,persons=[person],capture_frame_id=cap,capture_timestamp=t))
    assert c.active_target_id==1 and c._has_seen_person


@pytest.mark.parametrize('reason,label',[
    ('pending_initial_identity','ENROLL 1/2'),('initial_crop_incomplete','WAIT BODY'),
    ('initial_candidate_ambiguous','ONE PERSON'),('initial_capture_not_new','WAIT NEW FRAME'),
    ('weak_bbox_unassigned','WAIT QUALITY'),
])
def test_video_explains_bootstrap_stage(reason,label):
    o=VideoFrameOverlay(tracks=(VideoTrackOverlay(CAP17,1,assignment_reason=reason),))
    assert label in startup_status(o)


def test_video_distinguishes_detection_enrollment_lock_and_search():
    assert startup_status(VideoFrameOverlay())=='START WAIT PERSON'
    assert 'PERSON DETECTED' in startup_status(VideoFrameOverlay(detections=(VideoDetectionOverlay(CAP17,.94,0),)))
    assert 'WAIT LOCK' in startup_status(VideoFrameOverlay(tracks=(VideoTrackOverlay(CAP17,1,reid_uid=1),)))
    assert startup_status(VideoFrameOverlay(control=VideoControlOverlay(active_target_id=1,search_state='searching'))).startswith('TARGET U1')
