"""Gallery admission must not become a new identity/control rejection path."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from rk_vision.template_memory import TemplateMemory
from test_reacquire_crosscheck import meta, send, feature


def bank(enabled=True):
    b = IdentityBank(IdentityBankConfig(template_memory_enable=True,
        template_crosscheck_enable=True, template_learning_guard_enable=enabled,
        new_identity_confirm_frames=1, controlled_handoff_enable=True,
        update_interval=1, mapped_verify_threshold=.45))
    assert send(b, 1, 1.) == 1
    return b


def caps(b):
    e = b.identities[1]
    return {m.get("capture_frame_id") for m in b._learning_source_metadata(e)}


def assert_ordinary(b):
    a = b.last_assignments[1]
    assert a["uid"] == 1
    assert not a.get("identity_control_rejected")
    assert a["reason"] in {"mapped", "updated_diverse", "skip_update_redundant", "skip_update_distance"}


@pytest.mark.parametrize("part", [None, .31, .35, .7])
def test_torso_veto_precedes_any_full_or_recent_write_without_new_uid_reject(part):
    b = bank()
    before = deepcopy(b.identities[1].template_memory.last_learning)
    assert send(b, 2, 1.5, full=.1, part=part) == 1
    assert send(b, 3, 2., full=.1, part=part) == 1
    assert caps(b) == {1}
    assert b.identities[1].template_memory.last_learning == before
    assert_ordinary(b)


def test_good_current_pair_commits_both_and_pending_never_enters_matching():
    b = bank()
    assert send(b, 2, 1.5, full=.08, part=.1) == 1
    assert caps(b) == {1}
    assert b.last_assignments[1]["template_learning"]["status"] == "pending"
    assert send(b, 3, 2., full=.08, part=.1) == 1
    assert caps(b) == {1, 3}
    e = b.identities[1]
    assert e.template_memory.last_learning == {"strong": (2., 3), "partial": (2., 3)}
    assert b.last_assignments[1]["learning_written_tiers"] == ["recent_strong", "recent_partial"]
    for tier in ("strong", "partial"):
        assert e.template_memory.recent[tier][-1][1]["template_parent_caps"] == [1]
    assert_ordinary(b)


def test_every_trusted_frame_observed_but_pair_writes_stay_throttled():
    b = bank()
    b.config = replace(b.config, update_interval=5)
    for cap in (2, 3, 4):
        assert send(b, cap, 1. + cap*.1, full=.08, part=.1) == 1
        decision = b.last_assignments[1]["template_learning"]
        assert decision["status"] == "pending"
        assert decision["confirmations"] == cap-1
        assert decision["reason"] == ("pending_confirmation" if cap == 2 else "commit_throttled")
        assert caps(b) == {1}
        assert_ordinary(b)
    assert send(b, 5, 1.5, full=.08, part=.1) == 1
    assert caps(b) == {1, 5}
    assert b.identities[1].template_memory.last_learning == {"strong": (1.5, 5), "partial": (1.5, 5)}
    assert b.last_assignments[1]["template_learning"]["confirmations"] == 4


def test_nonpreferred_writer_observes_between_write_slots_too():
    b = bank()
    b.config = replace(b.config, update_interval=5)
    for cap in (2, 3, 4, 5):
        diagnostic = {}
        stored = b._maybe_add_to_identity(1, feature(.08), feature(.1), cap, .08,
            {**meta(cap, 1.+cap*.1), "track_id": 1}, diagnostic)
        assert stored is (cap == 5)
        assert diagnostic["template_learning"]["confirmations"] == cap-1
    assert caps(b) == {1, 5}


@pytest.mark.parametrize("failure", ["return_false", "value_error"])
def test_second_half_failure_rolls_back_both_tiers_and_counters(monkeypatch, failure):
    b = bank()
    send(b, 2, 1.5, full=.08, part=.1)
    original = TemplateMemory.remember
    def fail_partial(memory, value, metadata, tier):
        if metadata.get("capture_frame_id") == 3 and tier == "partial":
            if failure == "value_error":
                raise ValueError("simulated second tier failure")
            return False
        return original(memory, value, metadata, tier)
    monkeypatch.setattr(TemplateMemory, "remember", fail_partial)
    count = b.identities[1].update_count
    assert send(b, 3, 2., full=.08, part=.1) == 1
    assert caps(b) == {1}
    assert b.identities[1].update_count == count
    assert b.identities[1].template_memory.last_learning == {"strong": (1., 1), "partial": (1., 1)}
    assert b.last_assignments[1]["template_learning"]["reason"] == "pair_commit_rejected"
    assert_ordinary(b)


def test_crossing_risk_breaks_pending_on_non_write_frame_without_identity_stop():
    b = bank()
    b.config = replace(b.config, update_interval=2)
    send(b, 2, 1.5, full=.05, part=.06)
    assert send(b, 3, 1.7, full=.05, part=.06,
                extra={"template_learning_risk": {"observed": True, "risky": True, "reason": "overlapping_people"}}) == 1
    assert_ordinary(b)
    assert send(b, 4, 2., full=.05, part=.06) == 1
    assert caps(b) == {1}
    assert send(b, 6, 2.5, full=.05, part=.06) == 1
    assert 6 in caps(b)
    assert_ordinary(b)


@pytest.mark.parametrize("risk", [
    {"observed": True, "risky": True, "reason": "overlapping_people"},
    {"observed": False, "risky": False, "reason": "source_unavailable"},
])
def test_overlap_or_explicit_unknown_geometry_only_prevents_learning(risk):
    b = bank()
    for cap in (2, 3, 4):
        assert send(b, cap, 1. + cap*.25, full=.1, part=.1,
                    extra={"template_learning_risk": risk}) == 1
        assert_ordinary(b)
        assert not b.last_assignments[1]["bank_updated"]
    assert caps(b) == {1}


def test_nonpreferred_writer_cannot_bypass_paired_admission():
    b = bank()
    assert not b._maybe_add_to_identity(1, feature(.1), feature(.35), 2, .1,
                                        {**meta(2, 1.5), "track_id": 1})
    assert not b._maybe_add_to_identity(1, feature(.1), feature(.35), 3, .1,
                                        {**meta(3, 2.), "track_id": 1})
    assert caps(b) == {1}


@pytest.mark.parametrize("mutation", ["track", "crop", "time"])
def test_same_capture_different_sources_cannot_form_parent_pair(mutation):
    b = bank()
    e = b.identities[1]
    infos = e.partial_feature_metadata + [m for group in
        (e.template_memory.recent, e.template_memory.representatives) for _, m in group["partial"]]
    for m in infos:
        if mutation == "track":
            m["track_id"] = 99
        elif mutation == "crop":
            m["detector_bbox"] = [100., 50., 300., 450.]
        else:
            m["capture_timestamp"] = 1.1
    assert b._learning_pairs(e) == []


def test_source_isolation_follows_descendants_without_touching_uid_or_other_sources():
    b = bank()
    send(b, 2, 1.5, full=.08, part=.1)
    send(b, 3, 2., full=.08, part=.1)
    e = b.identities[1]
    other = {**meta(99, 2.1), "track_id": 9}
    e.template_memory.remember(feature(.25), other, "strong")
    e.template_memory.remember(feature(.25), other, "partial")
    mapping = dict(b.track_to_uid)
    observation = dict(e.last_strong_observation)
    result = b.isolate_template_sources(1, [1])
    assert result["capture_ids"] == [1, 3]
    assert caps(b) == {99}
    assert b.track_to_uid == mapping
    assert e.last_strong_observation == observation


def test_duplicate_capture_cannot_complete_pending_or_renew_evidence():
    b = bank()
    send(b, 2, 1.5, full=.08, part=.1)
    for _ in range(3):
        assert send(b, 2, 1.5, full=.08, part=.1) == 1
        assert caps(b) == {1}
    send(b, 3, 2., full=.08, part=.1)
    assert 3 in caps(b)


def test_startup_and_current_frame_uid_unchanged_when_learning_is_frozen():
    baseline = bank(False)
    trial = deepcopy(baseline)
    trial.config = replace(trial.config, template_learning_guard_enable=True)
    for source in (baseline, trial):
        assert send(source, 2, 1.5, full=.1, part=.35) == 1
        assert_ordinary(source)
    assert 2 in caps(baseline)
    assert 2 not in caps(trial)
    assert baseline.identities[1].last_strong_observation == trial.identities[1].last_strong_observation


def test_reset_discards_pending_and_risk_only_with_existing_bank_reset():
    b = bank()
    send(b, 2, 1.5, full=.1, part=.1)
    b.reset()
    assert b._template_learning.diagnostics(1) == {}
    assert send(b, 1, 1.) == 1


def test_existing_geometry_isolation_also_revokes_cross_track_learning_descendants(monkeypatch):
    b = bank()
    send(b, 2, 1.5, full=.08, part=.1)
    send(b, 3, 2., full=.08, part=.1)
    e = b.identities[1]
    # Simulate ordinary archive retirement, not rejection of this clean root.
    e.isolate_captures([1])
    for cap, ts in ((4, 4.5), (5, 5.), (6, 5.5)):
        b._learn_existing_pair(e, feature(.08), feature(.1), cap,
            {**meta(cap, ts), "track_id": 9}, {})
    assert 6 in caps(b)
    assert e.template_memory.recent["strong"][-1][1]["template_parent_caps"] == [3]
    monkeypatch.setattr(b, "_handoff_geometry", lambda *a, **kw: {
        "ok": False, "reason": "center_jump,area_change",
        "reference": {"frame_index": 1, "capture_frame_id": 1}, "current": {}})
    monkeypatch.setattr(b, "_search_geometry_contradiction", lambda *a: False)
    b.review_mapped_geometry(1, {**meta(7, 6.), "track_id": 1}, 7, commit=True)
    assert not {3, 6}.intersection(caps(b))
    assert {3, 6}.issubset(b._template_learning.diagnostics(1)["revoked_caps"])
