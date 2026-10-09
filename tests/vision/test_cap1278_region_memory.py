"""Real crop/timestamp fixture, synthetic descriptors; no model/motor replay."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from rk_vision.template_memory import TemplateMemory
from rk_vision.identity_bank import IdentityBank, IdentityBankConfig, IdentityEntry, _geometry_observation
from rk_vision.reacquire_quarantine import ReacquireQuarantine
from test_cap874_identity_reacquire import feature, metadata

DATA=json.loads((Path(__file__).parent/'fixtures/cap1278_region_memory.json').read_text())
ROWS={r['cap']:r for r in DATA['rows']}


def meta(cap, **changes):
    r=ROWS[cap]
    m=dict(metadata(cap,r['stamp'],r['bbox'],r['yaw'],search=False),
        image_width=640,image_height=480,frame_index=r['frame'],track_id=r['track'],
        partial_feature_source='osnet_torso',partial_observation=True,
        detector_edge_touch_count=r['edges'])
    m.update(changes)
    return m


def test_cap1278_expiry_preserves_only_observation_role_not_recent_permission():
    mem=TemplateMemory(); mem.remember(feature(0),meta(702),'partial')
    mem.advance(meta(1277))
    assert mem.evidence(feature(.267),meta(1277),'partial',comparable_only=True)['count']==1
    mem.advance(meta(1278))
    assert mem.evidence(feature(.267),meta(1278),'partial',comparable_only=True)['count']==0
    old=mem.representative_evidence(feature(.267),meta(1278))
    assert old['winner_cap']==702 and old['permission']=='observation_only'
    assert old['winner_age_sec']==pytest.approx(30.013890126)
    assert mem.last_learning['partial']==(ROWS[702]['stamp'],702)


def test_cap1336_uses_recent_broad_crop_without_identical_vertical_border_tag():
    mem=TemplateMemory();mem.remember(feature(0),meta(989),'partial')
    assert TemplateMemory.coverage_key(meta(989))!=TemplateMemory.coverage_key(meta(1336))
    out=mem.evidence(feature(ROWS[1336]['part']),meta(1336),'partial',reliable_only=True,comparable_only=True)
    assert out['count']==1 and out['winner_cap']==989
    assert out['distance']==pytest.approx(.2155544758)
    assert out['comparison_mode']=='vertical_border_bridge'


@pytest.mark.parametrize('change',['side','small','narrow','scale','two_boundaries','unknown','stale','weak'])
def test_new_comparability_is_not_a_blanket_crop_bypass(change):
    q=meta(1336);t=meta(989)
    if change=='side':q['detector_bbox']=[0,84,174,475]
    if change=='small':q['detector_bbox']=[70,84,120,190]
    if change=='narrow':q['detector_bbox']=[70,84,140,475]
    if change=='scale':q['detector_bbox']=[200,100,370,380]
    if change=='two_boundaries':q['detector_bbox']=[70,84,244,450]
    if change=='unknown':q.pop('image_width')
    if change=='stale':q['is_fresh']=False
    if change=='weak':q['bbox_quality_tier']='weak'
    assert not TemplateMemory.vertical_border_comparable(q,t)


def test_opposite_side_crops_do_not_create_false_torso_conflict():
    mem=TemplateMemory();mem.remember(feature(0),meta(866),'partial')
    q=meta(1364,detector_bbox=[0,100,144,476])
    out=mem.evidence(feature(.46),q,'partial',comparable_only=True)
    assert out['count']==0


def test_exact_region_conflict_is_not_overridden_by_looser_border_match():
    mem=TemplateMemory();mem.remember(feature(0),meta(989),'partial')
    mem.remember(feature(.49),meta(1336,capture_frame_id=1335,capture_timestamp=ROWS[1336]['stamp']-.1),'partial')
    out=mem.evidence(feature(0),meta(1336),'partial',comparable_only=True)
    assert out['winner_cap']==1335 and out['distance']==pytest.approx(.49)
    assert out['comparison_mode']=='exact_coverage'


def test_archive_queries_cannot_renew_or_resurrect_expired_representative():
    mem=TemplateMemory();mem.remember(feature(0),meta(702),'partial')
    for age in [31,60,119]:
        q=meta(1277,capture_timestamp=ROWS[702]['stamp']+age)
        mem.advance(q); assert mem.representative_evidence(feature(0),q)['winner_cap']==702
    q=meta(1277,capture_timestamp=ROWS[702]['stamp']+121)
    mem.advance(q)
    assert mem.representative_evidence(feature(0),q)['count']==0
    assert mem.representative_evidence(feature(0),meta(1277))['count']==0


@pytest.mark.parametrize('bad',[{'identity_control_rejected':True},{'is_fresh':False},
    {'bbox_quality_tier':'weak'},{'search_observation_only':True}])
def test_unapproved_candidate_cannot_create_regional_representatives(bad):
    mem=TemplateMemory();mem.remember(feature(0),meta(989,**bad),'partial')
    assert not mem.representatives['partial']


def test_source_isolation_removes_representatives_as_well_as_recent_gallery():
    e=IdentityEntry(1,template_memory=TemplateMemory())
    e.template_memory.remember(feature(0),meta(989),'partial')
    assert e.isolate_source(2,504)==2
    assert not e.template_memory.representatives['partial']
    assert not e.template_memory.recent['partial']


def test_pair_requires_same_cap_timestamp_track_and_crop():
    mem=TemplateMemory()
    mem.remember(feature(0),meta(989),'strong')
    mem.remember(feature(0),meta(989),'partial')
    full,part=feature(.25)*2,feature(.22)*3
    a,b=full.copy(),part.copy()
    assert mem.paired_recent_evidence(full,part,meta(1336))['qualified']
    np.testing.assert_array_equal(full,a);np.testing.assert_array_equal(part,b)
    for key,val in [('capture_frame_id',988),('capture_timestamp',ROWS[989]['stamp']-.01),
                    ('track_id',99),('detector_bbox',[0,0,500,479])]:
        old=deepcopy(mem.recent['partial'][0][1]);mem.recent['partial'][0][1][key]=val
        assert not mem.paired_recent_evidence(full,part,meta(1336))['qualified']
        mem.recent['partial'][0]=(mem.recent['partial'][0][0],old)


@pytest.mark.parametrize('full,part',[(.301,.1),(.1,.261),(.22,.373),(.225,.283)])
def test_weak_or_wrong_person_pair_cannot_release_gallery(full,part):
    mem=TemplateMemory()
    for tier in ('strong','partial'):mem.remember(feature(0),meta(989),tier)
    assert not mem.paired_recent_evidence(feature(full),feature(part),meta(1336))['qualified']


def bank():
    b=IdentityBank(IdentityBankConfig(template_memory_enable=True,template_crosscheck_enable=True,
        appearance_region_safety_enable=True,partial_match_threshold=.45,mapped_verify_threshold=.45,
        update_interval=1,controlled_handoff_enable=True,camera_hfov_deg=60.))
    b._create_identity(feature(0),505,meta(989),feature(0))
    m=meta(1336,capture_frame_id=1335,capture_timestamp=ROWS[1336]['stamp']-.1,frame_index=719)
    b._bind_reacquired_identity(1,9,719,m)
    b.identities[1].last_strong_observation=_geometry_observation(m,719)
    return b


def send(b,i,full=.25,part=.22,search=False,**changes):
    r=ROWS[1336]
    m=meta(1336,capture_frame_id=1336+i,capture_timestamp=r['stamp']+.25*i,
        frame_index=720+i,search_reacquire_context_active=search,**changes)
    return b.assign(track_id=9,feature=feature(full),partial_feature=feature(part),confidence=.93,
        area=70000,frame_index=720+i,bbox_quality_ok=True,bbox_quality_tier='strong',sample_metadata=m,
        preferred_uid=1 if search else None,preferred_candidate_ok=search)


def test_bound_correct_view_can_complete_frozen_pair_proof_before_any_learning():
    b=bank();old=deepcopy(b.identities[1].template_memory.last_learning)
    for i in range(4):
        assert send(b,i)==1
        a=b.last_assignments[9]
        assert a['quarantine_region_pair']['verified']
        assert a['quarantine_region_pair']['winner_cap']==989
        assert a['template_quarantine_reason']=='region_pair_confirming'
        assert b.identities[1].template_memory.last_learning==old
    assert send(b,4)==1
    assert b.last_assignments[9]['template_quarantine_reason']=='released_region_pair'
    assert b.last_assignments[9]['bank_updated']
    assert b.identities[1].template_memory.last_learning['partial'][1]==1340


@pytest.mark.parametrize('kind',['competition','suspect','conflict','search','weak_partial','gap','duplicate'])
def test_isolation_does_not_end_from_candidate_continuity_alone(kind):
    b=bank();old=deepcopy(b.identities[1].template_memory.last_learning)
    if kind=='suspect':b._reacquire_control_suspects[1]=dict(track_id=9,streak=0,capture=None,timestamp=None,reason='partial_conflict')
    if kind=='conflict':
        b._mapped_geometry_conflicts[9]=dict(uid=1,search_contradiction=True,
            reference=deepcopy(b.identities[1].last_strong_observation))
    for i in range(8):
        k=0 if kind=='duplicate' else i*2 if kind=='gap' else i
        changes={}
        if kind=='competition':changes['identity_competition']=dict(uid=1,frame_index=720+k,passed=False)
        send(b,k,part=.29 if kind=='weak_partial' else .22,search=kind=='search',**changes)
    assert b._reacquire_quarantine.is_held(1)
    assert b.identities[1].template_memory.last_learning==old


def test_expired_torso_representative_alone_still_cannot_confirm_identity():
    b=bank();e=b.identities[1]
    # Keep a fresh full sample but only an old torso representative.
    e.template_memory.recent['partial']=[]
    e.template_memory.representatives['partial']=[(feature(0),meta(702))]
    assert send(b,0,full=.1,part=.1)==0
    a=b.last_assignments[9]
    assert a['reason']=='secondary_evidence_unavailable'
    assert a['regional_representative_evidence']['permission']=='observation_only'
    assert not a['bank_updated']


def test_search_reacquisition_still_requires_fresh_confirmation_not_one_region_hit():
    b=bank();b.track_to_uid.clear()
    b.identities[1].last_strong_observation=_geometry_observation(meta(989),505)
    assert send(b,0,search=True)==0
    assert send(b,1,search=True)==1
    assert b._reacquire_quarantine.is_held(1)
    assert not b.last_assignments[9]['bank_updated']


def test_partial_handoff_can_leave_early_return_loop_and_earn_regular_learning():
    b=bank();b.track_to_uid.clear()
    b.identities[1].last_strong_observation=_geometry_observation(meta(989),505)
    assert send(b,0,search=True)==0
    assert send(b,1,search=True)==1
    for i in range(2,6):
        assert send(b,i)==1
        a=b.last_assignments[9]
        if i==2:
            assert a.get('partial_continuation_regular_verification')
        assert a['reason']=='skip_update_reacquire_quarantine'
        assert a['template_quarantine_reason']=='region_pair_confirming'
        assert not a['bank_updated']
    assert send(b,6)==1
    assert b.last_assignments[9]['template_quarantine_reason']=='released_region_pair'
    assert b.last_assignments[9]['bank_updated']


def gate_sample(g,cap,stamp,**kw):
    args=dict(uid=1,track_id=9,capture_frame_id=cap,capture_timestamp=stamp,frame_index=cap,
        is_fresh=True,quality_ok=True,quality_tier='strong',match_source='strong',
        feature_available=True,strong_distance=.25,center_x_ratio=.4,area_ratio=.2,
        region_pair_verified=True)
    args.update(kw)
    return g.observe(**args)


def test_long_arm_age_does_not_replace_one_second_of_paired_proof():
    g=ReacquireQuarantine();g.arm(1,9,1,1.,1)
    for i in range(5):
        out=gate_sample(g,20+i,10.+i*.1)
        assert out.hold
    assert out.streak==5
    assert gate_sample(g,25,10.6).hold
    assert gate_sample(g,26,10.8).hold
    assert not gate_sample(g,27,11.).hold


@pytest.mark.parametrize('changes',[dict(is_fresh=False),dict(quality_ok=False),
    dict(region_pair_verified=False),dict(strong_distance=.31),dict(strong_distance=float('nan')),
    dict(match_source='partial'),dict(quality_tier='weak'),dict(center_x_ratio=.9),
    dict(capture_timestamp=9.9),dict(capture_frame_id=1)])
def test_bad_regional_proof_resets_progress_and_does_not_unfreeze(changes):
    g=ReacquireQuarantine();g.arm(1,9,1,1.,1)
    gate_sample(g,20,10.);gate_sample(g,21,10.2)
    args=dict(cap=22,stamp=10.4)
    out=gate_sample(g,**args,**changes)
    assert out.hold and out.streak==0
    assert g.is_held(1)


def test_duplicate_cannot_count_or_end_regional_isolation():
    g=ReacquireQuarantine();g.arm(1,9,1,1.,1)
    assert gate_sample(g,20,10.).streak==1
    for _ in range(20):
        out=gate_sample(g,20,10.)
        assert out.hold and out.streak==1


def test_representative_bank_has_bounded_capacity():
    mem=TemplateMemory(archive_capacity=2)
    for cap in (702,866,989):mem.remember(feature(0),meta(cap),'partial')
    assert len(mem.representatives['partial'])==2


def test_paired_proof_does_not_combine_best_distances_from_different_templates():
    mem=TemplateMemory()
    for tier,value in [('strong',feature(0)),('partial',feature(.45))]:
        mem.remember(value,meta(989),tier)
    m=meta(989,capture_frame_id=990,capture_timestamp=ROWS[989]['stamp']+.1)
    for tier,value in [('strong',feature(.45)),('partial',feature(0))]:mem.remember(value,m,tier)
    out=mem.paired_recent_evidence(feature(0),feature(0),meta(1336))
    assert out['count']==2 and not out['qualified']


def test_rejected_metadata_cannot_forge_quarantine_proof():
    b=bank()
    for i in range(8):
        send(b,i,part=.4,_region_pair_quarantine_verified=True)
    assert b._reacquire_quarantine.is_held(1)
    assert not b.last_assignments[9]['bank_updated']
