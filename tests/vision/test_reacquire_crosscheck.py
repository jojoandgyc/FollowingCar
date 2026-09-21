"""Policy tests use synthetic vectors, including the CAP1161 distance pattern."""
from dataclasses import replace
import math
import numpy as np
import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from test_cap874_identity_reacquire import metadata, feature

BOX=(200.,40.,400.,460.)

def meta(cap,t,search=False,**kw):
    return dict(metadata(cap,t,BOX,search=search), partial_feature_source="osnet_torso",**kw)

def bank():
    b=IdentityBank(IdentityBankConfig(template_memory_enable=True,
        template_crosscheck_enable=True,new_identity_confirm_frames=1,
        controlled_handoff_enable=True,update_interval=1,mapped_verify_threshold=.45))
    assert send(b,1,1)==1
    return b

def send(b,cap,t,full=0.,part=0.,track=1,search=False,extra=None):
    return b.assign(track_id=track,feature=feature(full),
        partial_feature=None if part is None else part if isinstance(part,np.ndarray) else feature(part),confidence=.95,
        area=84000,frame_index=cap,bbox_quality_ok=True,bbox_quality_tier="strong",
        sample_metadata={**meta(cap,t,search),**(extra or {})},
        preferred_uid=1 if search else None,preferred_candidate_ok=search)

def test_stable_full_update_also_refreshes_recent_torso():
    b=bank()
    assert send(b,2,2,full=.01,part=.1)==1
    assert b.identities[1].template_memory.last_learning["partial"]==(2,2)
    assert b.last_assignments[1]["partial_template_update_reason"]=="trusted_update"

@pytest.mark.parametrize("part",[.14,.19,.29])
def test_trusted_torso_duplicates_and_diverse_views_remain_fresh(part):
    b=bank()
    for i in range(2,40):
        assert send(b,i,i,part=part)==1
    assert b.identities[1].template_memory.last_learning["partial"]==(39,39)

def test_mismatching_torso_cannot_poison_stable_gallery():
    b=bank();send(b,2,2,part=.7)
    assert b.identities[1].template_memory.last_learning["partial"]==(1,1)
    assert b.last_assignments[1]["partial_template_update_reason"]=="appearance_update_rejected"

@pytest.mark.parametrize("part",[.373,.389,.401,.370])
def test_cap1161_low_full_distance_does_not_override_recent_torso_conflict(part):
    b=bank()
    for i,t in [(1160,8.38),(1161,8.45),(1164,8.59),(1166,8.69)]:
        assert send(b,i,t,full=.144,part=part,track=3,search=True)==0
        a=b.last_assignments[3]
        assert a["reacquire_partial_state"]=="mismatch"
        assert a["reason"]=="recent_partial_conflict"
        assert a["identity_control_rejected"] and not a["bank_updated"]
    assert b.identities[1].last_strong_observation["capture_frame_id"]==1
    assert not b.pending_late_handoffs

@pytest.mark.parametrize("extra",[{}, {"partial_feature_source":"histogram"},
    {"detector_edge_touch_count":3}, {"quality_bbox_ok":False}])
def test_unknown_secondary_is_not_labeled_mismatch_or_allowed_late(extra):
    b=bank()
    part=None if not extra else .01
    assert send(b,20,8,full=.01,part=part,track=3,search=True,extra=extra)==0
    assert b.last_assignments[3]["reacquire_partial_state"]=="unknown"
    assert b.last_assignments[3]["reason"]=="secondary_evidence_unavailable"

def test_expired_torso_cannot_be_used_as_counterevidence():
    b=bank()
    # Full reference was independently updated, torso was not.
    b.identities[1].add(feature(0),40,20,metadata=meta(40,40))
    assert send(b,41,40.1,part=.7,track=3,search=True)==0
    assert b.last_assignments[3]["reacquire_partial_state"]=="unknown"

def test_fresh_geometric_handoff_does_not_require_extra_secondary_wait():
    b=bank()
    assert send(b,2,1.1,part=None,track=3,search=True)==1
    assert not b.last_assignments[3]["reacquire_secondary_required"]

def test_late_correct_candidate_can_still_confirm_in_two_frames():
    b=bank()
    assert send(b,20,8,full=.08,part=.12,track=3,search=True)==0
    assert send(b,21,8.1,full=.08,part=.12,track=3,search=True)==1
    assert b.identities[1].template_memory.last_learning["partial"]==(1,1)

def test_local_partial_can_support_correct_clipped_person():
    b=bank()
    extra={"partial_observation":True,"detector_edge_touch_count":2}
    assert send(b,20,8,full=.40,part=.12,track=3,search=True,extra=extra)==0
    assert send(b,21,8.1,full=.40,part=.12,track=3,search=True,extra=extra)==1

def test_recent_weak_plus_archive_strong_never_becomes_instant_match():
    b=bank();e=b.identities[1]
    # Query .144 to archive, .22 to recent. Same pattern as CAP1161.
    theta=math.acos(1-.144)-math.acos(1-.22)
    e.template_memory.recent["strong"]=[]
    e.template_memory.remember(np.array([math.cos(theta),math.sin(theta),0]),meta(2,1.05),"strong")
    assert send(b,3,1.1,full=.144,part=.01,track=3,search=True)==0
    a=b.last_assignments[3]
    assert a["authorization_full_distance_floor"]==pytest.approx(.22,abs=1e-6)
    assert not a["instant_reacquire_allowed"]
    assert a.get("soft_search_observation")

def test_post_reacquire_conflict_revokes_control_without_learning_or_new_anchor():
    b=bank();send(b,20,8,track=3,search=True);send(b,21,8.1,track=3,search=True)
    reference=dict(b.identities[1].last_strong_observation)
    assert send(b,22,8.2,full=.19,part=.39,track=3)==0
    assert b.last_assignments[3]["identity_control_rejected"]
    assert b.identities[1].last_strong_observation==reference
    assert send(b,23,8.3,full=.1,part=None,track=3)==0
    assert send(b,24,8.4,full=.1,part=.1,track=3)==0
    assert send(b,25,8.5,full=.1,part=.1,track=3)==1

def test_quarantine_and_rejected_samples_cannot_refresh_torso():
    b=bank();send(b,20,8,track=3,search=True);send(b,21,8.1,track=3,search=True)
    send(b,22,8.2,track=3)
    assert b.identities[1].template_memory.last_learning["partial"]==(1,1)

def test_stable_target_with_missing_torso_is_not_revoked():
    b=bank()
    assert send(b,2,1.1,part=None)==1
    assert b.last_assignments[1]["partial_template_update_reason"]=="feature_unavailable"

def test_normal_nonpreferred_handoff_also_requires_secondary_when_late():
    b=bank()
    assert send(b,20,8,full=.1,part=.4,track=9)==0
    assert b.last_assignments[9]["reason"]=="recent_partial_conflict"


def test_strong_full_cannot_override_recent_aggregate_conflict():
    b=bank();e=b.identities[1];angle=math.radians(80)
    p=np.array([math.cos(angle),math.sin(angle),0.]);n=p*np.array([1.,-1.,1.])
    e.partial_features=[p,n];e.partial_feature_metadata=[meta(1,1),meta(2,1.1)]
    e.template_memory.recent["partial"]=[(p,meta(1,1)),(n,meta(2,1.1))]
    assert send(b,20,8,part=p,track=3,search=True)==0
    assert send(b,21,8.1,part=n,track=3,search=True)==0
    g=b.last_assignments[3]["reacquire_geometry"]
    assert g["partial_aggregate_distance"]>.34
    assert g["partial_aggregate_override"] is False
    assert g["late_candidate_rejection"]=="partial_aggregate_threshold"


def test_runtime_config_enables_crosscheck(monkeypatch):
    import os
    from pathlib import Path
    from car_control_modular.config_loader import load_config_to_env
    from rk_vision.pipeline import RKNNVisionConfig
    from rk_vision.tracker import DeepSortTracker,DeepSortTrackerConfig
    monkeypatch.setattr(os,"environ",os.environ.copy())
    load_config_to_env(str(Path(__file__).resolve().parents[2]/"car_control_modular/config/reid_runtime.ini"))
    cfg=RKNNVisionConfig.from_env()
    assert cfg.identity_template_crosscheck_enable
    tracker=DeepSortTracker(DeepSortTrackerConfig(identity_template_memory_enable=True,
        identity_template_crosscheck_enable=cfg.identity_template_crosscheck_enable))
    assert tracker.identity_bank.config.template_crosscheck_enable
