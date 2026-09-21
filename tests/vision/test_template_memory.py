from dataclasses import replace
from pathlib import Path
import os

import numpy as np
import pytest

from rk_vision.template_memory import TemplateMemory
from rk_vision.identity_bank import IdentityBank, IdentityBankConfig, IdentityEntry
from test_cap874_identity_reacquire import metadata, feature, ROWS, ANCHOR, assign


BOX = (200., 50., 400., 450.)


def meta(cap, stamp, search=False):
    return metadata(cap, stamp, BOX, search=search)


def make_bank():
    return IdentityBank(IdentityBankConfig(template_memory_enable=True,
        new_identity_confirm_frames=1, update_interval=1, mapped_verify_threshold=.45,
        controlled_handoff_enable=True))


def push(bank, cap, stamp, value=None, track=1, search=False, **extra):
    return bank.assign(track_id=track, feature=feature(0) if value is None else value,
        confidence=.95, area=80000, frame_index=cap, bbox_quality_ok=True,
        bbox_quality_tier="strong", sample_metadata=dict(meta(cap, stamp, search), **extra),
        preferred_uid=1 if search else None, preferred_candidate_ok=search)


def test_approved_duplicate_refreshes_recent_but_not_original_anchor():
    entry=IdentityEntry(1, template_memory=TemplateMemory())
    assert entry.add(feature(0),1,20,metadata=meta(1,1))
    assert not entry.add(feature(0),2,20,metadata=meta(2,2))
    assert entry.feature_metadata[0]["capture_frame_id"]==1
    assert entry.template_memory.evidence(feature(0),meta(3,3))["winner_cap"]==2


@pytest.mark.parametrize("bad", [dict(is_fresh=False),dict(capture_timestamp=None),
    dict(capture_timestamp=float("nan")),dict(capture_frame_id=None),
    dict(identity_control_rejected=True),dict(search_observation_only=True),
    dict(quality_bbox_ok=False),dict(bbox_quality_tier="weak"),dict(bbox_quality_tier="reject"),
    dict(preferred_search_low_confidence=True)])
def test_untrusted_or_untimed_samples_cannot_renew_recent_memory(bad):
    memory=TemplateMemory()
    memory.remember(feature(0),meta(1,1),"strong")
    memory.remember(feature(.01),dict(meta(2,2),**bad),"strong")
    assert memory.last_learning["strong"]==(1,1)


def test_duplicate_out_of_order_and_queries_never_renew_memory():
    memory=TemplateMemory()
    memory.remember(feature(0),meta(10,10),"strong")
    for m in (meta(10,11),meta(11,9),meta(11,10)):
        memory.remember(feature(.1),m,"strong")
    for t in range(11,41):
        memory.advance(meta(t,t))
        memory.evidence(feature(0),meta(t,t))
    assert memory.last_learning["strong"]==(10,10)
    memory.advance(meta(41,41))
    assert not memory.recent["strong"]
    memory.advance(meta(2,2))
    assert not memory.recent["strong"]


def test_recent_capacity_and_descriptor_tiers_are_independent():
    memory=TemplateMemory()
    for i in range(12):
        v=np.eye(12)[i]
        memory.remember(v,meta(i,i),"strong")
        memory.remember(v,meta(i,i),"partial")
    assert len(memory.recent["strong"])==len(memory.recent["partial"])==8
    assert memory.evidence(np.eye(12)[-1],meta(12,12),"partial")["distance"]==0


def test_archive_aging_keeps_initial_anchor_but_removes_expired_non_anchor():
    memory=TemplateMemory()
    values=[feature(0),feature(.1),feature(.2)]
    info=[meta(1,1),meta(2,2),meta(3,100)]
    memory.advance(meta(4,123))
    assert memory.prune_archive(values,info,keep_anchor=True)==1
    assert [m["capture_frame_id"] for m in info]==[1,3]
    assert memory.prune_archive(values,info,keep_anchor=False)==1


def test_archive_capacity_recent_matching_and_diagnostics_agree():
    entry=IdentityEntry(1,template_memory=TemplateMemory())
    for i in range(12):
        entry.add(np.eye(12)[i],i,20,metadata=meta(i,i))
    assert len(entry.features)<=6
    assert len(entry.template_memory.recent["strong"])==8
    assert entry.distance(np.eye(12)[-1])==0
    evidence=entry.match_evidence(np.eye(12)[-1],.1)
    assert evidence["distance"]==0
    assert evidence["winner"]["metadata"]["template_role"]=="recent"


def test_old_match_cannot_create_its_own_recent_proof_or_uid():
    bank=make_bank()
    assert push(bank,1,1)==1
    entry=bank.identities[1]
    # Independently approved recent view differs from the archived anchor.
    entry.add(np.array([0.,1.,0.]),100,20,metadata=meta(100,35))
    for cap,t in [(101,35.1),(102,35.2),(103,35.3)]:
        assert push(bank,cap,t,track=9,search=True)==0
        a=bank.last_assignments[9]
        assert a["reason"]=="archive_only_reacquire_observe"
        assert a["template_memory_reject_reason"]=="recent_template_mismatch"
        assert a["identity_control_rejected"] and not a["bank_updated"]
    assert entry.template_memory.last_learning["strong"]==(35,100)
    assert 9 not in bank.track_to_uid
    assert not bank.pending_late_handoffs


def test_recent_expiry_is_observation_only_and_explicit():
    bank=make_bank();assert push(bank,1,1)==1
    assert push(bank,40,40,track=9,search=True)==0
    assert bank.last_assignments[9]["template_memory_reject_reason"]=="recent_template_unavailable"
    assert bank.identities[1].feature_metadata[0]["capture_frame_id"]==1


def test_non_preferred_handoff_cannot_bypass_recent_requirement():
    bank=make_bank();push(bank,1,1)
    assert push(bank,40,40,track=9,search=False)==0
    assert bank.last_assignments[9]["identity_control_rejected"]


def test_stable_mapped_target_can_refresh_recent_after_pause_without_new_uid():
    bank=make_bank();assert push(bank,1,1)==1
    assert push(bank,40,40)==1
    assert bank.identities[1].template_memory.last_learning["strong"]==(40,40)


def test_recent_support_is_only_necessary_not_identity_confirmation():
    bank=make_bank();push(bank,1,1)
    assert push(bank,20,2,track=9,search=True)==0
    assert bank.last_assignments[9]["template_recent_supported"]
    assert push(bank,21,2.1,track=9,search=True)==1
    # Reacquired crop cannot immediately update either gallery.
    assert bank.identities[1].template_memory.last_learning["strong"]==(1,1)


def test_recent_support_does_not_cancel_cap874_geometry_conflict():
    bank=make_bank()
    bank.config=replace(bank.config,handoff_geometry_min_area_similarity=.4)
    assert assign(bank,(359,857,10388.785306,ANCHOR,-66.329599,0),search=False)==1
    for row in ROWS[:4]:
        assert assign(bank,row)==0
        assert bank.last_assignments[5]["identity_control_rejected"]
    assert bank._mapped_geometry_conflicts


def test_partial_recent_support_does_not_require_full_body_similarity():
    bank=make_bank();push(bank,1,1)
    entry=bank.identities[1]
    entry.add_partial(np.array([0.,1.,0.]),2,8,metadata=meta(2,2))
    diagnostics={}
    blocked=bank._reject_archive_only_reacquire(9,0,1,np.array([0.,0.,1.]),
        np.array([0.,1.,0.]),dict(meta(3,3,True),partial_observation=True),diagnostics)
    assert not blocked and diagnostics["template_recent_supported"]


def test_disabled_policy_keeps_legacy_behavior():
    bank=IdentityBank(IdentityBankConfig(new_identity_confirm_frames=1))
    assert push(bank,1,1)==1
    assert bank.identities[1].template_memory is None


def test_runtime_config_wires_memory_policy(monkeypatch):
    from car_control_modular.config_loader import load_config_to_env
    from rk_vision.pipeline import RKNNVisionConfig
    monkeypatch.setattr(os,"environ",os.environ.copy())
    config=Path(__file__).resolve().parents[2]/"car_control_modular/config/reid_runtime.ini"
    load_config_to_env(str(config))
    cfg=RKNNVisionConfig.from_env()
    assert cfg.identity_template_memory_enable
    assert cfg.identity_template_recent_sec==30
    assert cfg.identity_template_archive_sec==120
    from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
    tracker=DeepSortTracker(DeepSortTrackerConfig(
        identity_template_memory_enable=cfg.identity_template_memory_enable,
        identity_template_recent_sec=cfg.identity_template_recent_sec,
        identity_template_archive_sec=cfg.identity_template_archive_sec))
    assert tracker.identity_bank.config.template_memory_enable
    assert tracker.identity_bank.config.template_recent_sec==30


def test_stale_sample_cannot_update_archive_or_recent_gallery():
    entry=IdentityEntry(1,template_memory=TemplateMemory())
    entry.add(feature(0),10,20,metadata=meta(10,10))
    assert not entry.add(feature(.3),9,20,metadata=meta(9,9))
    assert len(entry.features)==1
    assert entry.template_memory.last_learning["strong"]==(10,10)


def test_missing_capture_cannot_create_empty_identity():
    bank=make_bank()
    assert push(bank,1,1,capture_timestamp=None)==0
    assert not bank.identities
    assert bank.last_assignments[1]["reason"]=="template_observation_unavailable"


def test_template_quarantine_cannot_renew_recent_data():
    bank=make_bank();push(bank,1,1)
    bank._bind_reacquired_identity(1,9,2,meta(2,2))
    assert bank._reacquire_quarantine.is_held(1)
    push(bank,3,2.1,track=9)
    assert bank.identities[1].template_memory.last_learning["strong"]==(1,1)


def test_reset_removes_both_memories_and_conflicts():
    bank=make_bank();push(bank,1,1)
    bank.reset()
    assert not bank.identities and not bank._mapped_geometry_conflicts
