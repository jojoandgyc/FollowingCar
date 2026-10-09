"""Recorded CAP933--956 geometry/time; synthetic features, not model replay."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig, _geometry_observation
from rk_vision.template_memory import TemplateMemory
from test_cap874_identity_reacquire import feature, metadata

DATA=json.loads((Path(__file__).parent/'fixtures/cap935_scale_continuity.json').read_text())
ROWS={r['cap']:r for r in DATA['rows']}


def meta(cap, **kw):
    r=ROWS[cap]
    m=dict(metadata(cap,r['stamp'],r['bbox'],r['yaw'],search=False),
        frame_index=r['frame'],track_id=4,image_width=640,image_height=480,
        partial_feature_source='osnet_torso',partial_observation=True,
        detector_edge_touch_count=0,detector_confidence=r['confidence'])
    m.update(kw)
    return m


def bank():
    b=IdentityBank(IdentityBankConfig(template_memory_enable=True,template_crosscheck_enable=True,
        appearance_region_safety_enable=True,partial_match_threshold=.45,partial_confirm_threshold=.40,
        mapped_verify_threshold=.45,controlled_handoff_enable=True,camera_hfov_deg=60.,update_interval=1))
    b._create_identity(feature(0),ROWS[854]['frame'],meta(854),feature(0))
    b._bind_reacquired_identity(1,4,ROWS[931]['frame'],meta(931))
    b.identities[1].last_strong_observation=_geometry_observation(meta(931),ROWS[931]['frame'])
    return b


def send(b,cap,full=.25,part=.27,**kw):
    m=meta(cap,**kw)
    x1,y1,x2,y2=m['detector_bbox']
    m.update(detector_center_x_ratio=(x1+x2)/1280,center_x_ratio=(x1+x2)/1280,
             detector_area_ratio=(x2-x1)*(y2-y1)/(640*480),bbox=m['detector_bbox'])
    return b.assign(track_id=m['track_id'],feature=feature(full),partial_feature=feature(part),
        confidence=m['detector_confidence'],area=30000,frame_index=m['frame_index'],
        bbox_quality_ok=True,bbox_quality_tier='strong',sample_metadata=m)


def test_cap933_to935_to956_continuous_scale_no_longer_drops_identity():
    b=bank();before=deepcopy(b.identities[1].template_memory.last_learning)
    assert send(b,933)==1
    start=b._appearance_verified[1]['scale_started']
    for cap in (935,939,940,941,943,945,946,947,949,951,953,955,956):
        assert send(b,cap)==1
        a=b.last_assignments[4]
        assert a['reacquire_recent_partial_evidence']['comparison_mode']=='verified_scale_continuation'
        assert a['reacquire_recent_partial_evidence']['scale_bridge_caps']==[854]
        assert b._appearance_verified[1]['scale_started']==start
        assert not a['bank_updated']
    assert b.identities[1].template_memory.last_learning==before


def test_same_capture_pair_can_finish_quarantine_before_learning_new_scale():
    b=bank();before=deepcopy(b.identities[1].template_memory.last_learning)
    released=False
    for cap in (933,935,939,940,941,943,945,946,947,949,951,953,955,956):
        assert send(b,cap,part=.22)==1
        a=b.last_assignments[4]
        if not released and a.get('template_quarantine_reason')=='released_region_pair':
            released=True
            assert a['quarantine_region_pair']['winner_cap']==854
            assert a['bank_updated']
        elif not released:
            assert b.identities[1].template_memory.last_learning==before
    assert released
    mem=b.identities[1].template_memory
    assert any(m['capture_frame_id']>=933 for _,m in mem.recent['partial'])


@pytest.mark.parametrize('case',['new_track','no_proof','gap','duplicate','older','stale','jump',
    'side_crop','low_confidence','tiny','torso_conflict','competition','strong_mismatch','suspect','search','conflict'])
def test_scale_bridge_never_grants_identity_without_current_checks(case):
    b=bank();assert send(b,933)==1
    old=deepcopy(b.identities[1].template_memory.last_learning);kw={}
    if case=='new_track':kw['track_id']=8
    if case=='no_proof':b._appearance_verified.clear()
    if case=='gap':kw['capture_timestamp']=ROWS[933]['stamp']+.30
    if case=='duplicate':kw.update(capture_frame_id=933,capture_timestamp=ROWS[933]['stamp'])
    if case=='older':kw['capture_timestamp']=ROWS[933]['stamp']-.01
    if case=='stale':kw['is_fresh']=False
    if case=='jump':kw['detector_bbox']=[30.,150.,151.,424.]
    if case=='side_crop':kw['detector_bbox']=[0.,150.,121.,424.]
    if case=='low_confidence':kw['detector_confidence']=.37
    if case=='tiny':kw['detector_bbox']=[458.,150.,508.,235.]
    if case=='torso_conflict':kw['part']=.46
    if case=='strong_mismatch':kw['full']=.42
    if case=='competition':kw['identity_competition']=dict(uid=1,frame_index=ROWS[935]['frame'],passed=False)
    if case=='suspect':b._reacquire_control_suspects[1]=dict(track_id=4,streak=0,reason='partial_conflict')
    if case=='search':kw['search_reacquire_context_active']=True
    if case=='conflict':b._mapped_geometry_conflicts[4]=dict(uid=1,search_contradiction=True,
        reference=deepcopy(b.identities[1].last_strong_observation))
    assert send(b,935,**kw)==0
    assert b.identities[1].template_memory.last_learning==old


def test_fixed_scale_deadline_is_not_renewed_by_successful_frames():
    b=bank();send(b,933);start=ROWS[933]['stamp']
    for i in range(1,11):
        assert send(b,935,capture_frame_id=935+i,frame_index=495+i,capture_timestamp=start+.19*i)==1
        assert b._appearance_verified[1]['scale_started']==start
    assert send(b,935,capture_frame_id=960,frame_index=520,capture_timestamp=start+2.09)==0
    assert b.last_assignments[4]['reason']=='secondary_evidence_unavailable'


def test_global_bridge_unchanged_for_unconfirmed_candidates():
    for cap in (935,940,949,956):
        assert not TemplateMemory.vertical_border_comparable(meta(cap),meta(854))
        m=TemplateMemory();m.remember(feature(0),meta(854),'partial')
        assert m.evidence(feature(.01),meta(cap),'partial',comparable_only=True)['count']==0


def test_recent_template_expiry_still_wins_over_scale_continuity():
    b=bank();send(b,933)
    mem=b.identities[1].template_memory
    mem.recent['partial'][0][1]['capture_timestamp']=ROWS[935]['stamp']-30.1
    assert send(b,935)==0
    assert b.last_assignments[4]['reason']=='secondary_evidence_unavailable'


def test_independently_approved_mid_view_survives_repeated_near_view_updates():
    m=TemplateMemory(capacity=3)
    # Same coverage and same embedding: size-diverse approved representatives
    # must not be discarded as duplicate features from near views.
    for cap,box in [(1,[200,30,350,330]),(2,[200,30,390,460]),(3,[200,30,390,460]),
                    (4,[200,30,390,460]),(5,[200,30,390,460])]:
        m.remember(feature(0),meta(933,capture_frame_id=cap,capture_timestamp=float(cap),detector_bbox=box),'partial')
    assert {r[1]['capture_frame_id'] for r in m.recent['partial']}=={1,5}
    assert len(m.recent['partial'])<=3
    before=deepcopy(m.last_learning)
    for _ in range(3):m.evidence(feature(0),meta(956),'partial',comparable_only=True)
    assert m.last_learning==before


def test_exact_region_negative_evidence_cannot_be_hidden_by_scale_bridge():
    b=bank();send(b,933)
    mem=b.identities[1].template_memory
    mem.remember(feature(.99),meta(933,capture_frame_id=934,capture_timestamp=ROWS[933]['stamp']+.01),'partial')
    assert send(b,935,part=0.)==0
    assert b.last_assignments[4]['reason']=='recent_partial_conflict'


def test_scale_comparison_does_not_relax_same_capture_pair_requirement():
    b=bank();send(b,933)
    mem=b.identities[1].template_memory
    mem.recent['strong'][0][1]['capture_frame_id']=853
    assert send(b,935,part=.22)==1
    assert not b.last_assignments[4]['quarantine_region_pair']['qualified']
    assert not b.last_assignments[4]['bank_updated']
