"""Recorded CAP2382–2394 geometry, synthetic embeddings at logged distances."""
from copy import deepcopy
import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from test_cap874_identity_reacquire import feature, metadata

BOX=(16.700172,2.704285,221.151138,476.432983)
STAMP=12674.532581651
YAW=-240.1738201585
ROWS=[
    (1163,2389,12674.899811785,(339.849823,222.507263,522.550903,424.785522),-236.327217,.1974,.17288,False),
    (1164,2390,12674.964831078,(337.168579,221.983978,523.566345,427.248718),-236.327217,.172823,.14445,False),
    (1165,2392,12675.064549356,(339.768982,221.500122,524.325195,426.778137),-236.327217,.151669,.15130,True),
    (1166,2394,12675.163953518,(344.029266,221.090790,527.982910,423.251892),-236.476562,.147534,.17255,True),
]

def meta(cap,ts,box,yaw,search):
    return dict(metadata(cap,ts,box,yaw,search,opposite=search),partial_feature_source="osnet_torso")

def bank():
    b=IdentityBank(IdentityBankConfig(new_identity_confirm_frames=1,
        controlled_handoff_enable=True,template_memory_enable=True,template_crosscheck_enable=True,
        camera_hfov_deg=60.,handoff_geometry_max_gap_frames=15,
        handoff_geometry_max_center_jump_ratio=.30,handoff_geometry_min_area_similarity=.35,
        preferred_search_reacquire_max_age_sec=.35,preferred_search_reacquire_threshold=.20))
    assert b.assign(track_id=13,feature=feature(0),partial_feature=feature(0),confidence=.95,
        area=96854,frame_index=1160,sample_metadata=meta(2382,STAMP,BOX,YAW,False))==1
    return b

def send(b,row,track=16):
    frame,cap,ts,box,yaw,d,p,search=row
    return b.assign(track_id=track,feature=feature(d),partial_feature=feature(p),confidence=.95,
        area=(box[2]-box[0])*(box[3]-box[1]),frame_index=frame,
        sample_metadata=meta(cap,ts,box,yaw,search),preferred_uid=1 if search else None,
        preferred_candidate_ok=False)

def test_normal_handoff_rejection_survives_search_and_very_low_distances():
    b=bank();old=deepcopy(b.identities[1].last_strong_observation)
    for r in ROWS:
        assert send(b,r)==0
        assert b.last_assignments[16]["identity_control_rejected"]
        assert not b.last_assignments[16]["bank_updated"]
    assert b._mapped_geometry_conflicts[16]["rejected_capture"]==2389
    assert b.last_assignments[16]["reacquire_geometry"]["conflict_origin_capture"]==2389
    for i in range(20):
        r=list(ROWS[-1]);r[0]+=i+1;r[1]+=i+1;r[2]+=.1*(i+1);r[5]=r[6]=.01
        assert send(b,r)==0
    assert not b.pending_late_handoffs and 16 not in b.track_to_uid
    assert b.identities[1].last_strong_observation==old
    assert len(b.identities[1].features)==1

def test_new_raw_id_inherits_negative_evidence():
    b=bank();assert send(b,ROWS[0])==0
    for r,track in zip(ROWS[1:],(16,17,18)):
        assert send(b,r,track)==0
    assert b._mapped_geometry_conflicts[18]["rejected_capture"]==2389

@pytest.mark.parametrize("change",["long_gap","large_turn","missing_yaw","same_size","small_jump","stale"])
def test_new_rule_does_not_exclude_every_geometry_failure(change):
    b=bank();r=ROWS[0];m=meta(r[1],r[2],r[3],r[4],False)
    if change=="long_gap":m['capture_timestamp']=STAMP+1.
    if change=="large_turn":m['integrated_yaw_deg']=YAW+30.
    if change=="missing_yaw":m.pop('integrated_yaw_deg')
    if change=="stale":m['is_fresh']=False
    g=b._handoff_geometry(1,m,r[0])
    if change=="same_size":g['area_similarity']=.9
    if change=="small_jump":g['yaw_compensated_center_jump_ratio']=.2
    assert not b._search_geometry_contradiction(g,m)

def test_correct_independent_candidate_can_recover():
    b=bank();assert send(b,ROWS[0])==0
    row=(1165,2392,STAMP+.53,BOX,YAW,.08,.1,True)
    assert send(b,row,track=19)==0
    row=(1166,2394,STAMP+.63,BOX,YAW,.08,.1,True)
    assert send(b,row,track=19)==1
    assert 16 not in b.track_to_uid

def test_unresolved_origin_cannot_release_template_quarantine_even_via_direct_call():
    b=bank();assert send(b,ROWS[0])==0
    m=meta(2394,ROWS[-1][2],ROWS[-1][3],ROWS[-1][4],True)
    b._bind_reacquired_identity(1,16,1166,m)
    assert b._mapped_geometry_conflicts[16]['search_contradiction']
    for i in range(20):
        m=dict(m,capture_frame_id=2395+i,capture_timestamp=ROWS[-1][2]+.1*(i+1))
        b._observe_template_quarantine(uid=1,track_id=16,feature=feature(.01),confidence=.95,
            area=40000,frame_index=1167+i,bbox_quality_ok=True,bbox_quality_tier='strong',metadata=m)
    assert b._reacquire_quarantine.is_held(1)

def test_isolation_removes_only_attributable_post_anchor_samples():
    b=bank();e=b.identities[1]
    good=deepcopy(e.feature_metadata[0]);wrong=dict(good,track_id=16,frame_index=1167,
        capture_frame_id=2395,capture_timestamp=STAMP+.8)
    old=dict(wrong,frame_index=1000,capture_frame_id=2000,capture_timestamp=STAMP-10)
    for values,infos in ((e.features,e.feature_metadata),(e.partial_features,e.partial_feature_metadata),
                         (e.weak_features,e.weak_feature_metadata)):
        values.extend([feature(.2),feature(.3)]);infos.extend([wrong,old])
    for tier in ('strong','partial'):
        e.template_memory.recent[tier].append((feature(.2),wrong))
    watermark=e.template_memory.watermark
    assert e.isolate_source(16,1160)==5
    assert len(e.isolated_samples)==5
    assert all(x['metadata']['capture_frame_id']==2395 for x in e.isolated_samples)
    assert all(m['capture_frame_id']!=2395 for m in e.feature_metadata)
    assert any(m['capture_frame_id']==2382 for m in e.feature_metadata)
    assert any(m['capture_frame_id']==2000 for m in e.feature_metadata)
    assert e.template_memory.watermark==watermark
    assert e.isolate_source(16,1160)==0


def test_establishing_conflict_executes_source_isolation():
    b=bank();e=b.identities[1]
    info=dict(e.feature_metadata[0],track_id=16,frame_index=1161,
        capture_frame_id=2386,capture_timestamp=STAMP+.1)
    e.features.append(feature(.25));e.feature_metadata.append(info)
    e.template_memory.recent['strong'].append((feature(.25),info))
    assert send(b,ROWS[0])==0
    assert e.isolated_sample_total==2
    assert len(e.features)==1
    assert e.feature_metadata[0]['capture_frame_id']==2382


def test_gallery_helper_rejects_unresolved_origin_even_without_quarantine():
    b=bank();assert send(b,ROWS[0])==0
    m=dict(meta(2394,ROWS[-1][2],ROWS[-1][3],ROWS[-1][4],True),track_id=16)
    assert not b._reacquire_quarantine.is_held(1)
    assert not b._maybe_add_to_identity(1,feature(.01),feature(.01),1200,.01,m)
    assert len(b.identities[1].features)==1
