import math

import numpy as np
import pytest

from rk_vision.template_learning import TemplateLearningGuard, TemplatePair


def vector(distance=0.):
    dot = 1. - distance
    return np.array([dot, math.sqrt(max(0., 1. - dot * dot))], dtype=np.float32)


def pair(cap=10, stamp=1., full=None, partial=None):
    return TemplatePair(cap, stamp, vector() if full is None else full,
                        vector() if partial is None else partial)


def observe(gate, cap=100, stamp=10., **kwargs):
    values = dict(uid=1, track_id=3, capture_frame_id=cap, capture_timestamp=stamp,
                  full_feature=vector(), partial_feature=vector(), mature_pairs=[pair()])
    values.update(kwargs)
    return gate.observe(**values)


def risk(gate, cap=99, stamp=9.8, **kwargs):
    values = dict(uid=1, track_id=3, capture_frame_id=cap, capture_timestamp=stamp,
                  mature_pairs=[pair()], risk_reasons=("person_overlap",))
    values.update(kwargs)
    return gate.note_risk(**values)


def test_clean_pair_requires_two_distinct_captures():
    gate = TemplateLearningGuard()
    first = observe(gate)
    assert not first.allow and first.reason == "pending_confirmation"
    second = observe(gate, 105, 10.6)
    assert second.allow and second.parent_caps == (10,)
    assert second.confirmations == 2
    assert gate.diagnostics(1)["pending_cap"] is None


def test_confirmed_observations_wait_for_write_slot_without_becoming_parents():
    gate = TemplateLearningGuard()
    assert not observe(gate, commit_allowed=False).allow
    waiting = observe(gate, 101, 10.1, commit_allowed=False)
    assert not waiting.allow and waiting.reason == "commit_throttled"
    assert waiting.confirmations == 2 and waiting.parent_caps == (10,)
    assert set(gate._states[1].approved) == {10}
    assert gate.prune_sources(1, []) == set()  # Pin independent evidence while waiting.
    result = observe(gate, 105, 10.5, mature_pairs=[])
    assert result.allow and result.confirmations == 3 and result.parent_caps == (10,)
    assert set(gate._states[1].approved) == {10, 105}


@pytest.mark.parametrize("commit_allowed", [False, None, 1, "yes"])
def test_only_explicit_commit_permission_can_write(commit_allowed):
    gate = TemplateLearningGuard()
    observe(gate, commit_allowed=commit_allowed)
    result = observe(gate, 101, 10.1, commit_allowed=commit_allowed)
    assert not result.allow and result.reason == "commit_throttled"
    assert set(gate._states[1].approved) == {10}


def test_adjacent_verified_poses_can_confirm_across_a_throttled_write_interval():
    gate = TemplateLearningGuard()
    for offset in range(5):
        angle = offset * .18
        pose = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        result = observe(gate, 100 + offset, 10. + .1 * offset,
                         full_feature=pose, partial_feature=pose, commit_allowed=offset == 4)
        assert result.allow is (offset == 4)
        if offset:
            assert result.pending_full_distance == pytest.approx(1. - math.cos(.18), abs=1e-6)
    # The first and last poses differ too much to directly confirm each other,
    # but every adjacent step also independently passed the original parent.
    assert 1. - math.cos(.72) > gate.max_pending_distance
    assert result.parent_caps == (10,)
    assert result.confirmations == 5
    assert result.pending_reset_count == 0
    assert set(gate._states[1].approved) == {10, 104}


def test_gradual_pose_change_cannot_outgrow_the_independent_parent_limit():
    gate = TemplateLearningGuard()
    for offset in range(5):
        pose = vector(1. - math.cos(offset * .18))
        assert not observe(gate, 100 + offset, 10. + .1 * offset,
                           full_feature=pose, partial_feature=pose, commit_allowed=False).allow
    unsupported = vector(1. - math.cos(.9))
    result = observe(gate, 105, 10.5, full_feature=unsupported, partial_feature=unsupported,
                     mature_pairs=[pair(104, 10.4, vector(1. - math.cos(.72)),
                                        vector(1. - math.cos(.72)))])
    assert not result.allow and result.reason == "frozen_pair_mismatch"
    assert result.pending_reset_reason == "frozen_pair_mismatch"
    assert result.pending_reset_cap == 105 and result.pending_reset_count == 1
    assert set(gate._states[1].approved) == {10}
    # Elapsed time cannot make a throttled observation an independent parent.
    result = observe(gate, 110, 13., full_feature=unsupported, partial_feature=unsupported,
                     mature_pairs=[pair(104, 10.4, unsupported, unsupported)])
    assert not result.allow and result.reason == "mature_pair_unavailable"


def test_adjacent_confirmation_cannot_switch_independent_parent_during_wait():
    gate = TemplateLearningGuard()
    roots = [pair(), pair(11, 2., vector(1. - math.cos(1.)), vector(1. - math.cos(1.)))]
    for offset in range(3):
        pose = vector(1. - math.cos(offset * .4))
        result = observe(gate, 100 + offset, 10. + .1 * offset,
                         full_feature=pose, partial_feature=pose, mature_pairs=roots,
                         commit_allowed=offset == 2)
        assert not result.allow
    assert result.reason == "pending_confirmation" and result.confirmations == 1
    assert result.pending_reset_reason == "parent_changed"
    assert result.pending_reset_cap == 102 and result.pending_reset_count == 1


def test_risk_is_not_cleared_by_throttled_confirmations():
    gate = TemplateLearningGuard()
    risk(gate)
    observe(gate, commit_allowed=False)
    result = observe(gate, 101, 10.21, commit_allowed=False)
    assert result.reason == "commit_throttled" and result.risk_active
    assert set(gate._states[1].approved) == {10}
    result = observe(gate, 105, 10.5)
    assert result.allow and not result.risk_active


def test_throttled_confirmation_cannot_commit_after_current_appearance_jump():
    gate = TemplateLearningGuard()
    observe(gate, commit_allowed=False)
    assert observe(gate, 101, 10.1, commit_allowed=False).reason == "commit_throttled"
    result = observe(gate, 105, 10.5, full_feature=vector(.174))
    assert not result.allow and result.confirmations == 1
    assert result.pending_reset_reason == "pending_full_distance"
    assert result.pending_full_distance == pytest.approx(.174, abs=1e-6)


def test_same_capture_cannot_confirm_or_renew_pending():
    gate = TemplateLearningGuard()
    observe(gate)
    assert not observe(gate, 100, 10.8).allow
    assert gate.diagnostics(1)["pending_cap"] == 100
    assert gate.diagnostics(1)["confirmations"] == 1
    assert not observe(gate, 101, 11.6).allow  # Gap measured from the original.


@pytest.mark.parametrize("cap,stamp", [(99, 10.3), (101, 9.9), (100, 10.), (99, 9.9)])
def test_out_of_order_observation_does_not_change_pending(cap, stamp):
    gate = TemplateLearningGuard()
    observe(gate)
    before = gate.diagnostics(1)
    assert observe(gate, cap, stamp).reason == "duplicate_or_out_of_order"
    assert gate.diagnostics(1) == before


@pytest.mark.parametrize("value", [None, [], [float("nan"), 0], [0, 0], [float("inf"), 1]])
def test_missing_or_invalid_torso_only_blocks_learning(value):
    gate = TemplateLearningGuard()
    result = observe(gate, partial_feature=value)
    assert not result.allow and result.reason == "paired_features_unavailable"
    assert not hasattr(result, "uid") and not hasattr(result, "stop")


@pytest.mark.parametrize("changes", [{"eligible": False}, {"is_fresh": False}])
def test_rejected_learning_sample_resets_pending_only(changes):
    gate = TemplateLearningGuard()
    observe(gate)
    result = observe(gate, 101, 10.2, **changes)
    assert result.reason == "ineligible_learning_observation"
    assert not observe(gate, 102, 10.4).allow
    assert observe(gate, 103, 10.7).allow


def test_full_and_torso_must_match_one_same_frozen_parent():
    gate = TemplateLearningGuard()
    roots = [pair(full=vector(), partial=vector(1.)),
             pair(11, 2., full=vector(1.), partial=vector())]
    assert observe(gate, mature_pairs=roots).reason == "frozen_pair_mismatch"


def test_parent_snapshot_cannot_be_replaced_by_second_observation():
    gate = TemplateLearningGuard()
    observe(gate)
    wrong = vector(.7)
    result = observe(gate, 105, 10.7, full_feature=wrong, partial_feature=wrong,
                     mature_pairs=[pair(20, 3., wrong, wrong)])
    assert not result.allow and result.reason == "frozen_pair_mismatch"
    assert gate.diagnostics(1)["frozen_parent_caps"] == (10,)


def test_two_samples_cannot_switch_between_disjoint_mature_parents():
    gate = TemplateLearningGuard()
    roots = [pair(), pair(11, 2., vector(1.), vector(1.))]
    observe(gate, mature_pairs=roots)
    result = observe(gate, 105, 10.6, full_feature=vector(1.), partial_feature=vector(1.),
                     mature_pairs=roots)
    assert not result.allow and result.confirmations == 1


def test_mixed_crop_to_wrong_person_cannot_mutually_confirm_at_point174():
    gate = TemplateLearningGuard()
    # Both pass the same original parent, but not the independent short-window
    # appearance consistency requirement (.174 > .12).
    assert not observe(gate).allow
    second = observe(gate, 105, 10.7, full_feature=vector(.174))
    assert not second.allow and second.confirmations == 1


def test_partial_switch_also_resets_pending():
    gate = TemplateLearningGuard()
    observe(gate)
    result = observe(gate, 105, 10.7, partial_feature=vector(.174))
    assert not result.allow and result.confirmations == 1


@pytest.mark.parametrize("changes,reason", [
    ({"eligible": False}, "ineligible_learning_observation"),
    ({"is_fresh": False}, "ineligible_learning_observation"),
    ({"partial_feature": None}, "paired_features_unavailable"),
    ({"full_feature": vector(.174)}, "pending_full_distance"),
    ({"partial_feature": vector(.174)}, "pending_partial_distance"),
    ({"full_feature": vector(.174), "partial_feature": vector(.174)}, "pending_pair_distance"),
    ({"full_feature": vector(.7)}, "frozen_pair_mismatch"),
    ({"risk_reasons": ("overlapping_people",)}, "risk_observation"),
])
def test_pending_reset_diagnostics_identify_capture_and_cause(changes, reason):
    gate = TemplateLearningGuard()
    observe(gate)
    result = observe(gate, 101, 10.2, **changes)
    assert not result.allow
    assert result.pending_reset_reason == reason
    assert result.pending_reset_cap == 101 and result.pending_reset_count == 1
    info = gate.diagnostics(1)
    assert info["pending_reset_reason"] == reason
    assert info["pending_reset_cap"] == 101 and info["pending_reset_count"] == 1
    assert info["last_capture"] == 101


def test_pending_gap_diagnostic_uses_capture_time_and_new_sequence_span():
    gate = TemplateLearningGuard()
    observe(gate)
    result = observe(gate, 105, 11.6, commit_allowed=False)
    assert result.pending_reset_reason == "observation_gap"
    assert result.pending_reset_cap == 105 and result.confirmations == 1
    observe(gate, 106, 11.7, commit_allowed=False)
    info = gate.diagnostics(1)
    assert info["pending_started_timestamp"] == 11.6
    assert info["pending_last_timestamp"] == 11.7
    assert info["pending_span_sec"] == pytest.approx(.1)
    assert info["pending_reset_count"] == 1


def test_risk_note_freezes_before_periodic_bank_update():
    gate = TemplateLearningGuard()
    assert risk(gate).reason == "risk_frozen"
    assert not observe(gate, full_feature=vector(.25)).allow
    assert not observe(gate, 105, 10.6, full_feature=vector(.25)).allow
    assert gate.diagnostics(1)["risk_active"]


def test_risk_clear_requires_two_new_strong_pairs_and_capture_time_span():
    gate = TemplateLearningGuard()
    risk(gate)
    assert not observe(gate, 100, 10.).allow
    assert not observe(gate, 101, 10.1).allow
    assert observe(gate, 102, 10.21).allow
    assert not gate.diagnostics(1)["risk_active"]


def test_risk_frame_cannot_count_as_clear_confirmation():
    gate = TemplateLearningGuard()
    risk(gate, cap=100, stamp=10.)
    assert observe(gate, 100, 10.).reason == "risk_observation"
    assert not observe(gate, 105, 10.6).allow
    assert observe(gate, 110, 11.2).allow


def test_empty_risk_observation_does_not_release_or_renew_risk():
    gate = TemplateLearningGuard()
    risk(gate)
    observe(gate)
    snapshot = gate.diagnostics(1)
    assert risk(gate, 101, 10.1, risk_reasons=()).reason == "no_new_risk"
    assert gate.diagnostics(1) == snapshot
    assert observe(gate, 105, 10.6).allow


def test_repeated_risk_discards_pending_but_preserves_original_parent_snapshot():
    gate = TemplateLearningGuard()
    risk(gate)
    observe(gate)
    risk(gate, 102, 10.3, mature_pairs=[pair(20, 3., vector(.7), vector(.7))])
    assert gate.diagnostics(1)["pending_cap"] is None
    assert gate.diagnostics(1)["frozen_parent_caps"] == (10,)
    assert not observe(gate, 103, 10.4).allow
    assert observe(gate, 105, 10.8).allow


def test_old_risk_cannot_erase_current_confirmation_or_move_clock():
    gate = TemplateLearningGuard()
    risk(gate)
    observe(gate)
    before = gate.diagnostics(1)
    assert risk(gate).reason == "risk_duplicate_or_out_of_order"
    assert gate.diagnostics(1) == before


def test_track_change_is_learning_risk_even_if_caller_omits_reason():
    gate = TemplateLearningGuard()
    observe(gate)
    assert not observe(gate, 105, 10.6, track_id=4).allow
    assert gate.diagnostics(1)["risk_active"]
    assert not observe(gate, 110, 11.2, track_id=4, full_feature=vector(.25)).allow
    assert not observe(gate, 115, 11.8, track_id=4).allow
    assert observe(gate, 120, 12.4, track_id=4).allow


def test_recently_admitted_sample_cannot_be_immediate_learning_parent():
    gate = TemplateLearningGuard()
    observe(gate)
    assert observe(gate, 105, 10.6).allow
    newborn = pair(105, 10.6)
    result = observe(gate, 110, 11.2, mature_pairs=[newborn])
    assert result.reason == "mature_pair_unavailable"
    assert not observe(gate, 115, 13., mature_pairs=[newborn]).allow
    approved = observe(gate, 120, 13.6, mature_pairs=[newborn])
    assert approved.allow and approved.parent_caps == (105,)


def test_already_approved_bootstrap_source_is_not_mistaken_for_pending_candidate():
    gate = TemplateLearningGuard()
    source = pair(10, 9.9)
    observe(gate, mature_pairs=[source])
    assert observe(gate, 105, 10.6, mature_pairs=[source]).allow
    assert observe(gate, 110, 11.2, mature_pairs=[source]).reason == "pending_confirmation"


def test_pending_query_is_never_promoted_by_time_or_appearance_hits():
    gate = TemplateLearningGuard()
    observe(gate)
    query = pair(100, 10.)
    assert not observe(gate, 110, 13., mature_pairs=[query]).allow
    assert not observe(gate, 115, 13.6, mature_pairs=[query]).allow
    assert 100 not in gate._states[1].approved


def test_source_timestamp_cannot_be_renewed_by_caller():
    gate = TemplateLearningGuard()
    observe(gate)
    assert observe(gate, 105, 10.6).allow
    assert observe(gate, 110, 11.2, mature_pairs=[pair(10, 8.)]).reason == "mature_pair_unavailable"


def test_source_arrays_are_copied_and_not_modified_or_aliased():
    gate = TemplateLearningGuard()
    source = vector()
    query = vector()
    risk(gate, mature_pairs=[pair(full=source, partial=source)])
    assert source.flags.writeable
    source[:] = vector(1.)
    assert not observe(gate, full_feature=query, partial_feature=query).allow
    assert observe(gate, 105, 10.6, full_feature=query, partial_feature=query).allow
    np.testing.assert_array_equal(query, vector())


def test_revoke_source_cascades_only_to_its_descendants():
    gate = TemplateLearningGuard()
    roots = [pair(), pair(11, 2., vector(1.), vector(1.))]
    observe(gate, mature_pairs=roots)
    assert observe(gate, 105, 10.6, mature_pairs=roots).allow
    child = pair(105, 10.6)
    observe(gate, 110, 13., mature_pairs=[child])
    assert observe(gate, 115, 13.6, mature_pairs=[child]).allow
    assert gate.revoke_sources(1, [10]) == {10, 105, 115}
    assert 11 not in gate.diagnostics(1)["revoked_caps"]
    assert not observe(gate, 120, 16., mature_pairs=[child]).allow
    observe(gate, 125, 16.6, mature_pairs=roots, full_feature=vector(1.), partial_feature=vector(1.))
    assert observe(gate, 130, 17.2, mature_pairs=roots, full_feature=vector(1.), partial_feature=vector(1.)).allow


def test_revoke_sources_clears_pending_before_it_can_commit():
    gate = TemplateLearningGuard()
    observe(gate)
    assert gate.revoke_sources(1, [10]) == {10}
    result = observe(gate, 105, 10.6)
    assert not result.allow and result.reason == "mature_pair_unavailable"


def test_no_global_library_clear_when_revoking_unknown_source():
    gate = TemplateLearningGuard()
    observe(gate)
    assert gate.revoke_sources(1, [999]) == {999}
    assert not observe(gate, 105, 10.6).allow
    assert observe(gate, 110, 11.2).allow


def test_parent_and_uid_state_storage_is_bounded():
    gate = TemplateLearningGuard(max_parents=2, max_uids=2)
    roots = [pair(cap, float(cap)) for cap in range(1, 8)]
    observe(gate, mature_pairs=roots)
    observe(gate, uid=2, mature_pairs=roots)
    assert len(gate.diagnostics(1)["frozen_parent_caps"]) == 2
    assert not observe(gate, uid=3).allow
    assert len(gate._states) == 2
    gate.prune([1])
    assert observe(gate, uid=3).reason == "pending_confirmation"


def test_more_than_256_commits_roll_without_losing_exact_live_ancestry():
    gate = TemplateLearningGuard(max_lineage=32)
    roots = [pair()]
    last = None
    for i in range(300):
        cap, stamp = 100 + i * 10, 10. + i * 3.
        live = [10] + ([last.capture_frame_id] if last is not None else [])
        gate.prune_sources(1, live)
        sources = ([last] if last is not None else []) + roots  # newest-first
        assert not observe(gate, cap, stamp, mature_pairs=sources).allow
        result = observe(gate, cap + 5, stamp + .6, mature_pairs=sources)
        assert result.allow and result.parent_caps == (10,)
        last = pair(cap + 5, stamp + .6)
        assert gate.diagnostics(1)["lineage_count"] <= 3
    assert gate.revoke_sources(1, [10]) == {10, last.capture_frame_id,
                                          last.capture_frame_id - 10}


def test_prune_keeps_retired_gallery_parent_if_live_descendant_needs_it():
    gate = TemplateLearningGuard()
    observe(gate)
    assert observe(gate, 105, 10.6).allow
    child = pair(105, 10.6)
    observe(gate, 110, 13., mature_pairs=[child])
    assert observe(gate, 115, 13.6, mature_pairs=[child]).allow
    gate.prune_sources(1, [115])
    assert set(gate._states[1].approved) == {10, 105, 115}
    assert gate.revoke_sources(1, [10]) == {10, 105, 115}


def test_prune_does_not_remove_pending_or_risk_frozen_parent_proof():
    gate = TemplateLearningGuard()
    observe(gate)
    assert gate.prune_sources(1, []) == set()
    assert observe(gate, 105, 10.6).allow
    risk(gate, 110, 11., mature_pairs=[pair()])
    assert gate.prune_sources(1, []) == {105}
    assert not observe(gate, 115, 11.6, mature_pairs=[]).allow
    assert observe(gate, 120, 12.2, mature_pairs=[]).allow


def test_pruned_uncommitted_or_retired_child_cannot_reenter_as_parent():
    gate = TemplateLearningGuard()
    observe(gate)
    assert observe(gate, 105, 10.6).allow
    assert gate.prune_sources(1, [10]) == {105}
    result = observe(gate, 110, 13., mature_pairs=[pair(105, 10.6)])
    assert result.reason == "mature_pair_unavailable"
    assert not observe(gate, 115, 13.6, mature_pairs=[pair()]).allow
    assert observe(gate, 120, 14.2, mature_pairs=[pair()]).allow


def test_exhausted_deep_branch_does_not_freeze_unrelated_shallow_branch():
    gate = TemplateLearningGuard(max_ancestry=2)
    other = pair(11, 2., vector(1.), vector(1.))
    roots = [pair(), other]
    observe(gate, mature_pairs=roots)
    assert observe(gate, 105, 10.6, mature_pairs=roots).allow
    first = pair(105, 10.6)
    observe(gate, 110, 13., mature_pairs=[first, other])
    assert observe(gate, 115, 13.6, mature_pairs=[first, other]).allow
    deepest = pair(115, 13.6)
    gate.prune_sources(1, [115, 11])
    assert not observe(gate, 120, 16., mature_pairs=[deepest, other]).allow
    assert not observe(gate, 125, 16.6, mature_pairs=[deepest, other],
                       full_feature=vector(1.), partial_feature=vector(1.)).allow
    result = observe(gate, 130, 17.2, mature_pairs=[deepest, other],
                     full_feature=vector(1.), partial_feature=vector(1.))
    assert result.allow and result.parent_caps == (11,)


def test_registry_cannot_bootstrap_again_after_all_sources_are_retired():
    gate = TemplateLearningGuard()
    observe(gate)
    assert observe(gate, 105, 10.6).allow
    gate.prune_sources(1, [])
    assert not gate._states[1].approved
    result = observe(gate, 110, 13., mature_pairs=[pair(20, 2.)])
    assert result.reason == "mature_pair_unavailable"


@pytest.mark.parametrize("field,value", [
    ("normal_full_limit", float("nan")), ("risk_partial_limit", 0),
    ("max_gap_sec", -1), ("maturity_sec", False), ("risk_full_limit", .31),
])
def test_invalid_config_is_rejected(field, value):
    with pytest.raises(ValueError):
        TemplateLearningGuard(**{field: value})


@pytest.mark.parametrize("changes", [
    {"capture_timestamp": float("nan")}, {"capture_timestamp": -1},
    {"capture_frame_id": None}, {"capture_frame_id": True}, {"track_id": 0},
    {"uid": float("inf")},
])
def test_invalid_observation_cannot_authorize_learning(changes):
    assert observe(TemplateLearningGuard(), **changes).reason == "invalid_observation"
