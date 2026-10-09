"""CAP1340/1346/1348 geometry and distance-pattern policy regression.

Synthetic embeddings and intermediate timestamps: tests identity policy, not
the model or motor behaviour. The fixture starts in an existing quarantine.
"""
from copy import deepcopy

import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig, _geometry_observation
from rk_vision.template_memory import TemplateMemory
from test_cap874_identity_reacquire import feature, metadata

OLD = (0., 2.14366, 156.51569, 380.07693)
VISIBLE = (112.39108, 71.17984, 290.44986, 476.05176)
FIRST = (206.5015, 88.8111, 371.23376, 475.01868)
SECOND = (246.65697, 90.95778, 406.76172, 425.84546)


def meta(cap, ts, box, yaw, search=False):
    return dict(metadata(cap, ts, box, yaw, search), partial_feature_source="osnet_torso")


def send(b, cap, frame, ts, box, yaw, full, part, search=False, **changes):
    m = meta(cap, ts, box, yaw, search)
    m.update(changes)
    return b.assign(track_id=4, feature=feature(full), partial_feature=feature(part),
        confidence=.95, area=(box[2]-box[0])*(box[3]-box[1]), frame_index=frame,
        candidate_count=m.get("candidate_count",1),
        bbox_quality_ok=True, bbox_quality_tier="strong", sample_metadata=m,
        preferred_uid=1 if search else None, preferred_candidate_ok=search)


def setup():
    b = IdentityBank(IdentityBankConfig(new_identity_confirm_frames=1,
        template_memory_enable=True, template_crosscheck_enable=True,
        controlled_handoff_enable=True, handoff_geometry_max_gap_frames=15,
        preferred_search_reacquire_threshold=.20, preferred_search_reacquire_max_age_sec=.35))
    assert send(b,1300,670,36485.11786,OLD,-117.9467,0,0)==1
    old = deepcopy(b.identities[1].last_strong_observation)
    b._reacquire_search_anchors[1] = old
    b._reacquire_quarantine.arm(1,4,1332,36486.9,691)
    b.identities[1].last_strong_observation = _geometry_observation(
        meta(1340,36487.21181,VISIBLE,-138.3741),695)
    assert send(b,1341,696,36487.30,VISIBLE,-139,.170,.352)==0
    assert b.last_assignments[4]["reason"] == "recent_partial_conflict"
    return b, old


def first(b):
    assert send(b,1346,698,36487.49,FIRST,-143,.175564,.324268)==0
    assert b.last_assignments[4]["reacquire_control_recovery_streak"]==1


def test_search_switch_keeps_first_proof_and_full_assign_recovers():
    b, old = setup(); first(b)
    assert send(b,1348,699,36487.61190,SECOND,-145.7552,.146548,.203707,True)==1
    a=b.last_assignments[4]
    assert a["reacquire_control_recovered"]
    assert a["reacquire_control_geometry_source"]=="local_recovery"
    assert a["reacquire_control_protected_reference_cap"]==1300
    assert a["reacquire_control_local_reference_cap"]==1346
    assert not a["bank_updated"]
    assert b._reacquire_search_anchors[1]==old
    # Continued search must not immediately reject the recovered UID again.
    assert send(b,1349,700,36487.71,SECOND,-146,.14,.20,True)==1
    assert not b.last_assignments[4]["bank_updated"]
    assert b._reacquire_search_anchors[1]==old


@pytest.mark.parametrize("changes",[
    {"is_fresh":False}, {"identity_competition":{"passed":False}},
    {"candidate_score_gap":0., "candidate_count":2},
])
def test_unqualified_second_observation_does_not_recover(changes):
    b,_=setup(); first(b)
    assert send(b,1348,699,36487.6119,SECOND,-145.7552,.146,.204,True,**changes)==0
    assert not b.last_assignments[4]["bank_updated"]


@pytest.mark.parametrize("delay",[.351,1.0])
def test_local_proof_expires_without_extending_identity_or_motion(delay):
    b,_=setup(); first(b)
    assert send(b,1348,699,36487.49+delay,SECOND,-145.7552,.146,.204,True)==0
    # Expired proof is replaced with a NEW observation-only seed, not a
    # confirmation and not a renewal of the old observation's clock.
    assert b.last_assignments[4]["reacquire_control_recovery_streak"]==1
    assert b.last_assignments[4]["reacquire_control_seeded"]
    assert b._reacquire_control_suspects[1]["local_observation"]["capture_frame_id"]==1348


def test_duplicate_cannot_be_second_proof():
    b,_=setup(); first(b)
    assert send(b,1346,699,36487.49,FIRST,-143,.175,.324,True)==0
    assert b.last_assignments[4]["reacquire_control_recovery_streak"]==1


def test_other_raw_track_cannot_inherit_first_proof():
    b,_=setup(); first(b)
    b._reacquire_control_suspects[1]["track_id"]=99
    assert send(b,1348,699,36487.6119,SECOND,-145.7552,.146,.204,True)==0
    assert b.last_assignments[4]["reacquire_control_recovery_streak"]==1
    assert b.last_assignments[4]["reacquire_control_seeded"]
    assert b._reacquire_control_suspects[1]["local_observation"]["capture_frame_id"]==1348


def test_existing_identity_contradiction_is_not_replaced_by_local_proof():
    b,_=setup(); first(b)
    # A current, explicit contradictory protected reference is never overridden.
    b._reacquire_search_anchors[1]=_geometry_observation(
        meta(1347,36487.55,(0.,100.,60.,200.),-145.7552),698)
    assert send(b,1348,699,36487.6119,SECOND,-145.7552,.10,.10,True)==0
    assert b.last_assignments[4].get("reacquire_control_geometry_source")!="local_recovery"
    assert not b.last_assignments[4]["bank_updated"]


def test_reliable_torso_conflict_still_resets_local_proof():
    b,_=setup(); first(b)
    assert send(b,1348,699,36487.6119,SECOND,-145.7552,.146,.36,True)==0
    assert b.last_assignments[4]["reason"]=="recent_partial_conflict"
    assert not b._reacquire_control_suspects[1].get("local_observation")


def test_wrong_candidate_jump_cannot_use_local_proof():
    b,old=setup(); first(b)
    wrong=(520.,200.,570.,330.)
    assert send(b,1348,699,36487.6119,wrong,-143,.10,.10,True)==0
    assert b._reacquire_search_anchors[1]==old
    assert not b.last_assignments[4]["bank_updated"]


def test_crop_shape_diagnostic_does_not_override_identity_evidence():
    a=meta(1,1,(0.,0.,200.,400.),0)
    scaled=meta(2,2,(0.,0.,100.,200.),0)
    changed=meta(2,2,(0.,0.,200.,100.),0)
    assert TemplateMemory.partial_shape_consistent(a,scaled)
    assert not TemplateMemory.partial_shape_consistent(a,changed)
    memory=TemplateMemory(); memory.remember(feature(0),a,"partial")
    evidence=memory.evidence(feature(.38),changed,"partial",reliable_only=True)
    assert evidence["distance"]==pytest.approx(.38)
    assert evidence["winner_crop_shape_consistent"] is False


def test_changed_crop_shape_cannot_excuse_torso_conflict():
    b,_=setup(); first(b)
    assert send(b,1348,699,36487.6119,(100.,100.,400.,200.),-143,.10,.38,True)==0
    assert b.last_assignments[4]["reacquire_partial_state"]=="mismatch"
    assert not b.last_assignments[4]["bank_updated"]
