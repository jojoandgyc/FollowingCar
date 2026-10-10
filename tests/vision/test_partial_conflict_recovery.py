"""Torso-only suspicion can recover without loosening spatial identity gates."""
from copy import deepcopy

import pytest

from test_cap844_recovery_bootstrap import setup, send, ROWS


@pytest.mark.parametrize("full", [.201, .24, .29, .299])
def test_resolved_torso_conflict_uses_two_new_local_observations(full):
    bank = setup()
    frozen = deepcopy(bank.identities[1].template_memory.last_learning)
    anchor = deepcopy(bank._reacquire_search_anchors[1])
    assert send(bank, 866, row_changes={"full": full}) == 0
    assert bank.last_assignments[3]["reacquire_control_recovery_source"] == "partial_conflict_recheck"
    assert bank.last_assignments[3]["reacquire_control_seeded"]
    assert send(bank, 867, row_changes={"full": full}) == 1
    assert send(bank, 869, row_changes={"full": full}) == 1
    assert bank.identities[1].template_memory.last_learning == frozen
    assert bank._reacquire_search_anchors[1] == anchor
    assert bank._reacquire_quarantine.is_held(1)


@pytest.mark.parametrize("reason", ["appearance_conflict", "legacy_strict", "unknown"])
def test_generic_or_unknown_conflict_does_not_gain_soft_recovery(reason):
    bank = setup()
    bank._reacquire_control_suspects[1]["reason"] = reason
    for cap in (866, 867, 869):
        assert send(bank, cap, row_changes={"full": .25}) == 0
        assert not bank.last_assignments[3].get("reacquire_control_seeded")


@pytest.mark.parametrize("changes", [
    {"row_changes": {"full": .301}},
    {"row_changes": {"recent_partial": .38}},
    {"row_changes": {"quality": "weak", "edges": 3}},
    {"integrated_yaw_deg": None},
    {"is_fresh": False},
    {"candidate_count": 2, "candidate_score_gap": .9},
    {"identity_competition": {"passed": False}},
])
def test_new_route_requires_current_positive_evidence(changes):
    bank = setup()
    changes = deepcopy(changes)
    changes["row_changes"] = {"full": .25, **changes.get("row_changes", {})}
    assert send(bank, 866, **changes) == 0
    assert not bank.last_assignments[3].get("reacquire_control_seeded")
    assert not bank.last_assignments[3]["bank_updated"]


def test_duplicate_and_out_of_order_observations_never_complete_recovery():
    bank = setup()
    assert send(bank, 866, row_changes={"full": .25}) == 0
    for cap, stamp in [(866, ROWS[866]["ts"]), (865, ROWS[866]["ts"]-.01)]:
        assert send(bank, 867, row_changes={"cap": cap, "ts": stamp, "full": .25}) == 0
        assert not bank.last_assignments[3].get("reacquire_control_recovered")
    assert send(bank, 867, row_changes={"full": .25}) == 1


def test_explicit_spatial_conflict_is_not_erased_by_two_similar_frames():
    bank = setup()
    bank._mapped_geometry_conflicts[3] = dict(uid=1, search_contradiction=True,
        rejected_frame=410, rejected_capture=833,
        reference=deepcopy(bank._reacquire_search_anchors[1]))
    for cap in (866, 867, 869):
        assert send(bank, cap, row_changes={"full": .25}) == 0
        assert bank.last_assignments[3]["reason"] == "mapped_geometry_reject"
        assert not bank.last_assignments[3]["bank_updated"]


def test_new_torso_conflict_resets_observation_but_can_recover_again():
    bank = setup()
    assert send(bank, 866, row_changes={"full": .25}) == 0
    assert send(bank, 867, row_changes={"full": .25, "recent_partial": .5}) == 0
    assert bank._reacquire_control_suspects[1]["streak"] == 0
    assert send(bank, 869, row_changes={"full": .25}) == 0
    assert send(bank, 871, row_changes={"full": .25}) == 1


def test_later_independent_strong_evidence_can_resume_normal_quarantine_review():
    bank = setup()
    assert send(bank, 866, row_changes={"full": .25}) == 0
    assert send(bank, 867, row_changes={"full": .25}) == 1
    assert send(bank, 869, row_changes={"full": .10}) == 1
    assert bank.last_assignments[3]["reacquire_control_recovery_source"] == "strong"
    assert bank.last_assignments[3]["match_source"] == "strong"
    assert bank.last_assignments[3]["template_quarantine_streak"] > 0
    assert not bank.last_assignments[3]["bank_updated"]
