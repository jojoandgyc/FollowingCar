"""Logged CAP27/140..151 boxes and scalar distances; synthetic embeddings.

Tests policy and real metadata interfaces, not OSNet accuracy or car dynamics.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from rk_vision.template_memory import TemplateMemory
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from car_control_modular.search_observation_retry import retry_evidence_source
from test_cap874_identity_reacquire import metadata, feature

ANCHOR = (545.4487,107.8929,639.9957,437.1239)
COMPLETE = (200.,40.,400.,460.)
ROWS = [
    (140,(358.9874,4.0955,639.156,473.0212),.495843,.473552,3),
    (142,(315.6666,5.3206,611.6345,473.7641),.468628,.483640,2),
    (144,(269.0461,6.8877,545.0482,475.6369),.359240,.415416,2),
    (145,(236.4645,9.0802,496.4055,371.0345),.423592,.493683,1),
    (147,(180.6879,13.9592,414.4774,364.7511),.420178,.521146,0),
    (149,(121.6875,10.9149,362.1478,469.6494),.360214,.470000,0),
    (151,(39.0293,3.8888,279.9105,344.9426),.367697,.409405,1),
]


def meta(cap, box, *, edges=0, search=True):
    m = metadata(cap,1+cap*.05,box,search=search)
    m.update(image_width=640,image_height=480,partial_feature_source='osnet_torso',
             control_frame_id=cap,frame_index=cap,source_detection_index=0,
             detector_edge_touch_count=edges,quality_bbox_ok=edges<=2,
             bbox_quality_tier='strong' if edges<=2 else 'weak')
    m['identity_competition'] = dict(uid=1,frame_index=cap,source_detection_index=0,
                                    candidate_count=1,passed=True)
    return m


def bank():
    return IdentityBank(IdentityBankConfig(template_memory_enable=True,
        template_crosscheck_enable=True,appearance_region_safety_enable=True,
        partial_match_threshold=.45,new_identity_confirm_frames=1,
        controlled_handoff_enable=True))


def send(b,cap,box,full=0.,part=0.,edges=0,track=3,search=True):
    m=meta(cap,box,edges=edges,search=search)
    result=b.assign(track_id=track,feature=feature(full),partial_feature=feature(part),
        confidence=.95,area=(box[2]-box[0])*(box[3]-box[1]),frame_index=cap,
        bbox_quality_ok=edges<=2,bbox_quality_tier=m['bbox_quality_tier'],
        sample_metadata=m,preferred_uid=1 if search else None,preferred_candidate_ok=search)
    return result,m


def test_real_bootstrap_rejects_cap27_without_creating_uid_then_accepts_complete_crop():
    b=bank()
    for cap in [27,29,31]:
        assert send(b,cap,ANCHOR,search=False)[0]==0
        assert b.last_assignments[3]['reason']=='initial_crop_incomplete'
        assert not b.identities and not b.track_to_uid and not b.pending_new
    assert send(b,77,COMPLETE,search=False)[0]==0
    assert send(b,78,COMPLETE,search=False)[0]==1
    assert len(b.identities[1].features)==1


@pytest.mark.parametrize('bad',[
    {'is_fresh':False},{'quality_bbox_ok':False},{'image_width':None},
    {'detector_bbox':(0.,20.,200.,420.)},{'detector_bbox':(200.,-5.,400.,490.)},
    {'detector_bbox':(200.,40.,230.,460.)},{'detector_bbox':(200.,40.,float('nan'),460.)},
])
def test_bootstrap_missing_weak_or_cut_crop_cannot_create_identity(bad):
    m=meta(1,COMPLETE);m.update(bad)
    assert not TemplateMemory.initial_crop_usable(m)


@pytest.mark.parametrize('row',ROWS)
def test_legacy_cap27_gallery_now_yields_unknown_not_conflict_or_uid(row):
    b=bank()
    # Simulate the already existing poor gallery, NOT new bootstrap permission.
    b._create_identity(feature(0),27,meta(27,ANCHOR,edges=1,search=False),feature(0))
    old=deepcopy(b.identities[1].last_strong_observation)
    cap,box,full,part,edges=row
    uid,m=send(b,cap,box,full,part,edges)
    a=b.last_assignments[3]
    assert uid==0 and not a['bank_updated']
    assert a['reason']=='secondary_evidence_unavailable'
    assert a['reacquire_partial_state']=='unknown'
    assert a['reacquire_partial_comparable'] is False
    assert a['template_recent_partial_evidence']['distance']==pytest.approx(part,abs=1e-5)
    assert a['reacquire_recent_partial_evidence']['count']==0
    assert b.identities[1].last_strong_observation==old
    assert len(b.identities[1].features)==1
    source=retry_evidence_source(a,m,1)
    assert source==('unverified_stop_only' if edges<=2 else None)


def test_comparable_wrong_person_still_rejected_despite_strong_full_match():
    b=bank();assert send(b,1,COMPLETE,search=False)[0]==0
    assert send(b,2,COMPLETE,search=False)[0]==1
    for cap in [140,142,144]:
        assert send(b,cap,COMPLETE,full=.14,part=.49,track=5)[0]==0
        assert b.last_assignments[5]['reason']=='recent_partial_conflict'
        assert not b.last_assignments[5]['bank_updated']


def test_unknown_evidence_does_not_clear_established_conflict():
    b=bank();assert send(b,1,COMPLETE,search=False)[0]==0
    assert send(b,2,COMPLETE,search=False)[0]==1
    assert send(b,140,COMPLETE,full=.14,part=.49,track=5)[0]==0
    suspect=deepcopy(b._reacquire_control_suspects[1])
    assert send(b,142,ANCHOR,full=.1,part=.1,edges=1,track=5)[0]==0
    assert b._reacquire_control_suspects[1]==suspect


def test_comparable_correct_candidate_can_reacquire_without_learning():
    b=bank();assert send(b,1,COMPLETE,search=False)[0]==0
    assert send(b,2,COMPLETE,search=False)[0]==1
    assert send(b,140,COMPLETE,full=.1,part=.12,track=5)[0]==0
    assert send(b,141,COMPLETE,full=.1,part=.12,track=5)[0]==1
    assert not b.last_assignments[5]['bank_updated']


def test_comparable_selector_uses_matching_region_not_best_unrelated_template():
    memory=TemplateMemory()
    memory.remember(feature(0),meta(1,ANCHOR,edges=1),'partial')
    memory.remember(feature(.49),meta(2,COMPLETE),'partial')
    raw=memory.evidence(feature(0),meta(3,COMPLETE),'partial',reliable_only=True)
    aligned=memory.evidence(feature(0),meta(3,COMPLETE),'partial',reliable_only=True,comparable_only=True)
    assert raw['winner_cap']==1
    assert aligned['winner_cap']==2 and aligned['distance']==pytest.approx(.49)


def test_runtime_configuration_reaches_real_tracker_bank(monkeypatch):
    from car_control_modular.config_loader import load_config_to_env
    from rk_vision.pipeline import RKNNVisionConfig
    monkeypatch.delenv('Y8_IDENTITY_APPEARANCE_REGION_SAFETY_ENABLE',raising=False)
    load_config_to_env(str(Path(__file__).parents[2]/'car_control_modular/config/reid_runtime.ini'))
    cfg=RKNNVisionConfig.from_env()
    assert cfg.identity_appearance_region_safety_enable
    t=DeepSortTracker(DeepSortTrackerConfig(identity_template_memory_enable=True,
        identity_appearance_region_safety_enable=cfg.identity_appearance_region_safety_enable))
    assert t.identity_bank.config.appearance_region_safety_enable
