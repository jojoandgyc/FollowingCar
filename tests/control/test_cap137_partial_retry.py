"""Actual CAP137/139 metadata; synthetic embeddings; no NPU or motors."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import request_0513_modular as runtime
from car_control_modular.search_candidate_gate import (
    CandidateObservation, SearchCandidateGate, SearchCandidateGateConfig, SearchCandidateGateDecision,
)
from car_control_modular.search_observation_retry import retry_evidence_source
from rk_vision.template_memory import TemplateMemory
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection
from test_search_observation_arbitration import owner, _record

DATA = json.loads((Path(__file__).parents[1] / 'vision/fixtures/cap137_search_retry.json').read_text())
ROWS = {r['cap']: r for r in DATA['rows']}


def feature(distance):
    x = 1. - distance
    return np.array([x, np.sqrt(1.-x*x), 0.], dtype=np.float32)


def tracker():
    t = DeepSortTracker(DeepSortTrackerConfig(identity_template_memory_enable=True,
        identity_template_crosscheck_enable=True, identity_partial_appearance_enable=True,
        identity_min_confidence=.6, identity_partial_match_threshold=.45))
    m = deepcopy(ROWS[137]['metadata'])
    box = [200.,20.,400.,475.]
    m.update(capture_frame_id=54, capture_timestamp=34071.19688, detector_bbox=box,
             quality_bbox=box, bbox=box, detector_edge_touch_count=1,
             detector_center_x_ratio=.46875, detector_area_ratio=200*455/(640*480),
             search_reacquire_context_active=False)
    t.identity_bank._create_identity(feature(0),20,m,feature(0))
    t.identity_bank.identities[1].last_strong_observation = deepcopy(
        ROWS[137]['assignment']['reacquire_geometry']['reference'])
    t.set_search_reacquire_context(active_uid=1,searching=True,direction='right')
    return t


def observe(t, cap, *, box=None):
    r = deepcopy(ROWS[cap]);m=r['metadata'];a=r['assignment'];b=t.identity_bank
    full = feature(a['template_recent_evidence']['distance'])
    part = feature(a['reacquire_recent_partial_evidence']['distance'])
    if cap==137:
        uid=b.assign(track_id=1,feature=full,partial_feature=part,confidence=m['detector_confidence'],
            area=100000,frame_index=r['frame'],bbox_quality_ok=True,bbox_quality_tier='strong',
            sample_metadata=m,preferred_uid=1,preferred_candidate_ok=True)
        t.last_identity_observations=[dict(raw_track_id=1,detector_bbox=m['detector_bbox'],sample_metadata=m)]
    else:
        t._frame_index=r['frame']
        t._frame_context={k:m[k] for k in ('capture_frame_id','capture_timestamp','control_frame_id','yaw_rate_dps','integrated_yaw_deg')}
        record=t._search_probe_record(
            [Detection(tuple(box or m['detector_bbox']),m['detector_confidence'],0)],[full],
            partial_features=[part],partial_feature_sources=['osnet_torso'],image_width=640,image_height=480)
        assert record is not None
        uid=record.reid_uid
    observation=t.last_identity_observations[-1]
    return uid,observation,deepcopy(b.last_assignments[observation['raw_track_id']])


@pytest.mark.parametrize('cap',[139,147])
def test_real_probe_populates_quality_contract_and_reaches_partial_verifier(cap):
    t=tracker();uid,o,a=observe(t,cap)
    m=o['sample_metadata']
    assert m['quality_bbox_ok'] is True
    assert m['quality_bbox_source']=='detector'
    assert TemplateMemory.partial_usable(m)
    assert a['reacquire_partial_state']=='match'
    assert a['reason']!='secondary_evidence_unavailable'
    assert not a['bank_updated']
    assert not TemplateMemory.partial_usable({k:v for k,v in m.items() if k!='quality_bbox_ok'})


def test_probe_does_not_promote_three_edge_crop_quality():
    t=tracker();_,o,a=observe(t,139,box=(0.,0.,350.,479.))
    assert o['sample_metadata']['quality_bbox_ok'] is False
    assert not TemplateMemory.partial_usable(o['sample_metadata'])
    assert a['reason']=='secondary_evidence_unavailable'


def feed(owner, monkeypatch, t, cap):
    uid,o,a=observe(t,cap)
    assert uid==0  # Neither raw-ID change nor stop observation confirms identity.
    m=o['sample_metadata'];stamp=m['capture_timestamp'];box=tuple(o['detector_bbox'])
    monkeypatch.setattr(runtime.time,'monotonic',lambda:stamp+.08)
    owner._assignments={o['raw_track_id']:a}
    owner._rknn_pipeline=SimpleNamespace(tracker=t)
    owner._active_capture_frame_id=cap;owner._active_capture_timestamp=stamp
    owner._search_candidate_gate=SearchCandidateGate(SearchCandidateGateConfig())
    return owner._retry_search_candidate_observation(
        SearchCandidateGateDecision(source='blocked',reason='candidate_already_observed',bbox=box,score=m['detector_confidence']),
        search_active=True,width=640,height=480,formal_candidates=(CandidateObservation(box,m['detector_confidence']),),
        capture_id=cap,capture_timestamp=stamp)


def test_formal_to_probe_continuity_stops_without_inheriting_uid(owner,monkeypatch):
    monkeypatch.setattr(runtime,'SEARCH_EVIDENCE_GATE_ENABLE',True)
    monkeypatch.setattr(runtime,'SEARCH_EVIDENCE_RETRY_ENABLE',True)
    owner._follow_controller.active_target_id=1
    t=tracker()
    assert not feed(owner,monkeypatch,t,137).pause_rotation
    decision=feed(owner,monkeypatch,t,139)
    assert decision.entered and decision.pause_rotation
    assert decision.source=='credible_retry' and not decision.preferred_target_match
    assert owner._events==[]
    assert t.identity_bank.identities[1].last_strong_observation['capture_frame_id']==114
    assert len(t.identity_bank.identities[1].features)==1
    owner._apply_search_candidate_gate_decision(decision,prepare_only=True)
    owner._search_evidence_pause_current_frame=True
    owner._consume_track_records([_record(track=-1,uid=0,bbox=ROWS[139]['metadata']['detector_bbox'],score=.392)],640,480,'test')
    assert owner._events==[('queue',[runtime.ACTION_STOP],'search_candidate_retry_observe')]
    assert owner._current_forward_percent==owner._current_rotate_raw_target==0
    assert owner._follow_controller.active_target_id==1
    assert owner.search_direction=='left'  # Fixture state must not be rewritten by candidate.


def partial_evidence():
    t=tracker();_,o,a=observe(t,137)
    return a,o['sample_metadata']


@pytest.mark.parametrize('case',[
    'gray','missing_recent','old_partial_only','wrong_uid','excluded','contradiction',
    'geometry_jump','competition_failed','competition_old','competition_wrong_uid',
    'competition_other_box','missing_quality','weak','stale','wrong_source',
])
def test_partial_retry_requires_reliable_current_identity_evidence(case):
    a,m=partial_evidence()
    if case=='gray':a['reacquire_partial_state']='tentative'
    if case=='missing_recent':a.pop('reacquire_recent_partial_evidence')
    if case=='old_partial_only':a['reacquire_recent_partial_evidence']['distance']=.40
    if case=='wrong_uid':a['best_uid']=9
    if case=='excluded':a['search_excluded']=True
    if case=='contradiction':a['reacquire_geometry']['search_contradiction_retained']=True
    if case=='geometry_jump':a['reacquire_geometry'].update(ok=False,reason='center_jump')
    if case=='competition_failed':a['identity_competition']['passed']=False
    if case=='competition_old':a['identity_competition']['frame_index']=1
    if case=='competition_wrong_uid':a['identity_competition']['uid']=9
    if case=='competition_other_box':a['identity_competition']['source_detection_index']=9
    if case=='missing_quality':m.pop('quality_bbox_ok')
    if case=='weak':m['bbox_quality_tier']='weak'
    if case=='stale':m['is_fresh']=False
    if case=='wrong_source':m['partial_feature_source']='histogram'
    assert retry_evidence_source(a,m,1) is None


def test_bare_partial_distance_never_suffices():
    a,m=partial_evidence();assert retry_evidence_source(a,m,1)=='partial'
    assert retry_evidence_source(dict(best_uid=1,match_source='partial',distance=.01),m,1) is None
