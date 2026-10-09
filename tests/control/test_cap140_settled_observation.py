"""No hardware. STOP proof must precede a fresh image and never grant UID."""
from dataclasses import replace
from types import SimpleNamespace
import pytest

import request_0513_modular as runtime
from car_control_modular.search_candidate_gate import (
    SearchCandidateGateDecision, SearchCandidateGate, SearchCandidateGateConfig,
    CandidateObservation,
)
from car_control_modular.search_observation_retry import (
    SettledObservation, DetectorObservationSettlement, SearchObservationRetry,
    retry_evidence_source,
)
from test_search_observation_arbitration import owner, _record

BOX=(200.,20.,400.,460.)


def feedback(t,left=0.,right=0.,**kw):
    return SimpleNamespace(timestamp=t,left_forward_rpm=left,right_forward_rpm=right,
                           trustworthy=True,**kw)


def unknown(cap=142):
    m=dict(is_fresh=True,quality_bbox_ok=True,bbox_quality_tier='strong',
           control_frame_id=cap,source_detection_index=0,capture_frame_id=cap,
           capture_timestamp=10.+(cap-142)*.05)
    a=dict(mapped_uid=1,reason='secondary_evidence_unavailable',identity_control_rejected=True,
           reacquire_partial_comparable=False,template_recent_evidence=dict(distance=.48),
           identity_competition=dict(uid=1,frame_index=cap,source_detection_index=0,
                                     passed=True,candidate_count=1))
    return a,m


@pytest.mark.parametrize('mutation',[
    'conflict','excluded','wrong_frame','multi','failed','wrong_uid','wrong_box',
    'stale','weak','geometry','nan','far','comparable',
])
def test_unverified_stop_never_bypasses_current_competition_or_conflicts(mutation):
    a,m=unknown(); assert retry_evidence_source(a,m,1)=='unverified_stop_only'
    if mutation=='conflict': a['reason']='recent_partial_conflict'
    if mutation=='excluded':a['search_excluded']=True
    if mutation=='wrong_frame':a['identity_competition']['frame_index']=1
    if mutation=='multi':a['identity_competition']['candidate_count']=2
    if mutation=='failed':a['identity_competition']['passed']=False
    if mutation=='wrong_uid':a['identity_competition']['uid']=2
    if mutation=='wrong_box':a['identity_competition']['source_detection_index']=2
    if mutation=='stale':m['is_fresh']=False
    if mutation=='weak':m['quality_bbox_ok']=False
    if mutation=='geometry':a['reacquire_geometry']=dict(search_contradiction_retained=True)
    if mutation=='nan':a['template_recent_evidence']['distance']=float('nan')
    if mutation=='far':a['template_recent_evidence']['distance']=.7
    if mutation=='comparable':a['reacquire_partial_comparable']=True
    assert retry_evidence_source(a,m,1) is None


def test_settling_needs_two_distinct_quiet_samples_and_later_image():
    s=SettledObservation()
    assert not s.update(10.1,10.09,10.01,10.,feedback(10.08,7.,-6.))
    assert not s.update(10.15,10.14,10.01,10.,feedback(10.13))
    assert not s.update(10.16,10.15,10.01,10.,feedback(10.13))
    assert not s.update(10.2,10.17,10.01,10.,feedback(10.18))
    assert s.update(10.21,10.20,10.01,10.,feedback(10.18))
    assert not s.update(10.25,10.24,10.01,10.,feedback(10.23,2.,0.))


@pytest.mark.parametrize('bad',[
    None,feedback(9.),feedback(11.),feedback(10.1,float('nan'),0.),
    feedback(10.1,0.,2.),feedback(10.1,left_error=1),
])
def test_invalid_moving_or_fault_feedback_never_proves_settled(bad):
    s=SettledObservation()
    assert not s.update(10.2,10.19,10.01,10.,bad)
    assert not s.update(10.21,10.20,10.01,10.,bad)


def test_normal_gate_does_not_release_on_second_image_in_motion_then_times_out():
    s=DetectorObservationSettlement()
    seed=SearchCandidateGateDecision(source='formal',entered=True,pause_rotation=True,bbox=BOX)
    def call(d,t,cap,left=7.):
        return s.update(d,now=t,capture_timestamp=t-.05,capture_id=cap,search_active=True,
                        zero_sent_at=10.01,feedback=feedback(t-.01,left,-left),max_hold_sec=.3)
    assert not call(seed,10.,131).completed
    second=replace(seed,entered=False,completed=True)
    assert call(second,10.077,133).pause_rotation
    assert call(second,10.2,140).pause_rotation
    done=call(second,10.31,144)
    assert done.completed and done.reason=='observation_timeout_unsettled'
    assert not done.preferred_target_match


def test_normal_gate_success_only_on_settled_capture():
    s=DetectorObservationSettlement()
    seed=SearchCandidateGateDecision(source='formal',entered=True,pause_rotation=True,bbox=BOX)
    args=dict(search_active=True,zero_sent_at=10.01,max_hold_sec=.3)
    s.update(seed,now=10.,capture_timestamp=9.98,capture_id=1,feedback=None,**args)
    next_frame=replace(seed,entered=False,completed=True)
    assert s.update(next_frame,now=10.1,capture_timestamp=10.09,capture_id=2,
                    feedback=feedback(10.08),**args).pause_rotation
    assert s.update(next_frame,now=10.2,capture_timestamp=10.17,capture_id=3,
                    feedback=feedback(10.18),**args).pause_rotation
    d=s.update(next_frame,now=10.22,capture_timestamp=10.20,capture_id=4,
               feedback=feedback(10.18),**args)
    assert d.completed and d.reason=='observation_settled_capture'


def test_retry_zero_ack_alone_no_longer_counts_as_stop_confirmation():
    s=SearchObservationRetry()
    def tick(t,cap,fb=None):
        return s.update(now=t+.02,session=(1,1),capture_id=cap,capture_timestamp=t,
                        eligible=True,bbox=BOX,score=.8,blocked=True,zero_sent_at=10.13,feedback=fb)
    assert tick(10.,1) is None
    assert tick(10.1,2).entered
    assert tick(10.25,3,feedback(10.24,7.,-6.)).pause_rotation
    assert tick(10.3,4).pause_rotation
    assert tick(10.45,5).reason=='search_retry_timeout'
    assert s.spent


def test_real_runtime_unknown_candidate_requests_stop_without_identity(owner,monkeypatch):
    monkeypatch.setattr(runtime,'SEARCH_EVIDENCE_GATE_ENABLE',True)
    monkeypatch.setattr(runtime,'SEARCH_EVIDENCE_RETRY_ENABLE',True)
    owner._follow_controller.active_target_id=1
    owner._search_candidate_gate=SearchCandidateGate(SearchCandidateGateConfig())
    for cap in (142,144):
        a,m=unknown(cap);owner._assignments={3:a}
        owner._rknn_pipeline=SimpleNamespace(tracker=SimpleNamespace(last_identity_observations=[
            dict(raw_track_id=3,detector_bbox=BOX,sample_metadata=m)]))
        monkeypatch.setattr(runtime.time,'monotonic',lambda:m['capture_timestamp']+.05)
        d=owner._retry_search_candidate_observation(
            SearchCandidateGateDecision(reason='candidate_already_observed'),
            search_active=True,width=640,height=480,
            formal_candidates=(CandidateObservation(BOX,.9),),capture_id=cap,
            capture_timestamp=m['capture_timestamp'])
    assert d.entered and d.source=='credible_retry'
    owner._apply_search_candidate_gate_decision(d,prepare_only=True)
    owner._search_evidence_pause_current_frame=True
    owner._consume_track_records([_record(track=3,uid=0,bbox=BOX,score=.9)],640,480,'test')
    assert owner._events==[('queue',[runtime.ACTION_STOP],'search_candidate_retry_observe')]
    assert owner._follow_controller.active_target_id==1
    assert owner._current_forward_percent==owner._current_rotate_raw_target==0


def test_normal_runtime_entry_arms_actual_zero_ack_and_waits_for_wheels(owner,monkeypatch):
    monkeypatch.setattr(runtime,'SEARCH_EVIDENCE_GATE_ENABLE',True)
    owner._search_candidate_gate=SearchCandidateGate(SearchCandidateGateConfig())
    fb=feedback(10.,7.,-6.)
    owner._action_runtime=SimpleNamespace(get_steering_feedback=lambda:fb)
    now=[10.]
    monkeypatch.setattr(runtime.time,'monotonic',lambda:now[0])
    seed=SearchCandidateGateDecision(source='formal',entered=True,pause_rotation=True,bbox=BOX,score=.9)
    def tick(d,cap,stamp):
        return owner._retry_search_candidate_observation(d,search_active=True,width=640,height=480,
            formal_candidates=(CandidateObservation(BOX,.9),),capture_id=cap,capture_timestamp=stamp)
    assert tick(seed,131,9.95).pause_rotation
    assert owner._search_retry_zero_requested_at==10.
    assert owner._search_retry_zero_sent_at is None
    owner._search_retry_zero_sent_at=10.01
    now[0]=10.077;fb.timestamp=10.06
    second=replace(seed,entered=False,completed=True)
    assert tick(second,133,10.03).pause_rotation
    now[0]=10.15;fb.timestamp=10.13;fb.left_forward_rpm=fb.right_forward_rpm=0.
    assert tick(second,140,10.1).pause_rotation
    now[0]=10.22;fb.timestamp=10.18
    done=tick(second,142,10.20)
    assert done.completed and done.reason=='observation_settled_capture'
    assert not done.preferred_target_match and owner._events==[]


@pytest.mark.parametrize('replacement',[
    SearchCandidateGateDecision(),
    SearchCandidateGateDecision(bbox=(450.,20.,630.,460.),score=.9),
])
def test_normal_pause_never_relabels_old_box_as_current_candidate(replacement):
    s=DetectorObservationSettlement()
    args=dict(search_active=True,zero_sent_at=10.01,feedback=None,max_hold_sec=.3)
    s.update(SearchCandidateGateDecision(entered=True,pause_rotation=True,bbox=BOX),
             now=10.,capture_timestamp=9.98,capture_id=1,**args)
    d=s.update(replacement,now=10.1,capture_timestamp=10.08,capture_id=2,**args)
    assert d.completed and d.bbox is None and not d.preferred_target_match
    assert d.reason=='observation_candidate_lost_or_changed'
