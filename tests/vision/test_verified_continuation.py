"""Source-bound continuation policy only; no model, camera or motor access."""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from rk_vision.verified_continuation import evaluate_continuation


def inputs():
    previous = dict(track_id=3, capture_frame_id=830, capture_timestamp=10.,
                    is_fresh=True, quality_bbox_ok=True, bbox_quality_tier="strong",
                    integrated_yaw_deg=100.)
    return dict(uid=1, track_id=3, source="partial", previous=previous,
                current=dict(previous, capture_frame_id=832, capture_timestamp=10.1,
                             integrated_yaw_deg=101., partial_observation=False),
                geometry=dict(ok=True, yaw_compensated_center_jump_ratio=.02,
                              area_similarity=.96),
                competition_ok=True, blocked=False,
                pair=dict(full_distance=.3257, partial_distance=.3619,
                          winner_cap=414, qualified_comparison=True),
                full_limit=.40, confirm_limit=.40, observe_limit=.45)


@pytest.mark.parametrize("source", ["strong", "partial"])
@pytest.mark.parametrize("mode", [None, "exact_coverage", "vertical_border_bridge"])
def test_current_comparable_pair_continues_original_source(source, mode):
    kw = inputs()
    kw["source"] = source
    if mode is not None:
        kw["pair"]["comparison_mode"] = mode
    result = evaluate_continuation(**kw)
    assert result.status == "accept" and result.reason == "verified_pair"
    assert result.source == source and result.reference_cap == 830 and result.pair_cap == 414
    assert result.deadline == pytest.approx(10.35)


@pytest.mark.parametrize("partial,status", [(0., "accept"), (.40, "accept"),
                                           (.400001, "hold"), (.45, "hold"), (.450001, "reject")])
def test_confirm_observe_and_conflict_are_distinct(partial, status):
    kw = inputs()
    kw["pair"]["partial_distance"] = partial
    assert evaluate_continuation(**kw).status == status


def test_tentative_samples_never_renew_last_accepted_reference_or_deadline():
    kw = inputs()
    kw["pair"]["partial_distance"] = .43
    first = evaluate_continuation(**kw)
    assert first.status == "hold"
    kw["pending_deadline"] = first.deadline
    kw["current"].update(capture_frame_id=835, capture_timestamp=10.2)
    second = evaluate_continuation(**kw)
    assert second.status == "hold" and second.deadline == first.deadline
    kw["current"].update(capture_frame_id=837, capture_timestamp=10.3)
    kw["pair"]["partial_distance"] = .33
    third = evaluate_continuation(**kw)
    assert third.status == "accept" and third.reference_cap == 830
    assert third.deadline == first.deadline


def test_held_capture_cannot_be_reprocessed_with_better_descriptors_for_acceptance():
    kw = inputs()
    kw["pair"]["partial_distance"] = .43
    held = evaluate_continuation(**kw)
    assert held.status == "hold"
    kw["previous"].update(continuation_watermark_cap=832,
                          continuation_watermark_timestamp=10.1)
    kw["pending_deadline"] = held.deadline
    kw["pair"]["partial_distance"] = .30
    replay = evaluate_continuation(**kw)
    assert replay.status == "reject" and replay.reason == "nonnew_capture"
    assert replay.deadline == held.deadline
    # Consuming the tentative frame does not replace the accepted reference or
    # restart its window; a genuinely new frame can still complete the recheck.
    kw["current"].update(capture_frame_id=834, capture_timestamp=10.2)
    accepted = evaluate_continuation(**kw)
    assert accepted.status == "accept" and accepted.reference_cap == 830
    assert accepted.deadline == held.deadline


@pytest.mark.parametrize("cap,stamp", [(832, 10.1), (831, 10.2), (834, 10.1), (834, 10.05)])
def test_both_sequences_must_advance_beyond_consumed_held_capture(cap, stamp):
    kw = inputs()
    kw["previous"].update(continuation_watermark_cap=832,
                          continuation_watermark_timestamp=10.1)
    kw["current"].update(capture_frame_id=cap, capture_timestamp=stamp)
    result = evaluate_continuation(**kw)
    assert result.status == "reject" and result.reason == "nonnew_capture"


@pytest.mark.parametrize("changes", [
    dict(continuation_watermark_cap=832),
    dict(continuation_watermark_timestamp=10.1),
    dict(continuation_watermark_cap=None, continuation_watermark_timestamp=None),
    dict(continuation_watermark_cap=829, continuation_watermark_timestamp=10.1),
    dict(continuation_watermark_cap=832, continuation_watermark_timestamp=9.9),
    dict(continuation_watermark_cap=True, continuation_watermark_timestamp=10.1),
    dict(continuation_watermark_cap=832, continuation_watermark_timestamp=float("nan")),
    dict(continuation_watermark_cap=832, continuation_watermark_timestamp="10.1"),
    dict(continuation_watermark_cap=832.5, continuation_watermark_timestamp=10.1),
])
def test_watermark_pair_must_be_valid_and_not_precede_accepted_reference(changes):
    kw = inputs()
    kw["previous"].update(changes)
    result = evaluate_continuation(**kw)
    assert result.status == "reject" and result.reason == "invalid_continuation_watermark"


def test_watermark_equal_to_accepted_reference_is_backward_compatible():
    kw = inputs()
    baseline = evaluate_continuation(**kw)
    kw["previous"].update(continuation_watermark_cap=830,
                          continuation_watermark_timestamp=10.)
    assert evaluate_continuation(**kw) == baseline


def test_new_capture_after_watermark_still_cannot_renew_the_pending_deadline():
    kw = inputs()
    kw["previous"].update(continuation_watermark_cap=833,
                          continuation_watermark_timestamp=10.2)
    kw["pending_deadline"] = 10.25
    kw["current"].update(capture_frame_id=835, capture_timestamp=10.25)
    result = evaluate_continuation(**kw)
    assert result.status == "reject" and result.reason == "continuation_expired"
    assert result.deadline == 10.25


@pytest.mark.parametrize("partial", [.30, .43])
@pytest.mark.parametrize("now", [10.25, 10.3, 10.35, 12.])
def test_good_and_tentative_samples_cannot_revive_expired_pending_source(partial, now):
    kw = inputs()
    kw.update(pending_deadline=10.25)
    kw["current"]["capture_timestamp"] = now
    kw["pair"]["partial_distance"] = partial
    result = evaluate_continuation(**kw)
    assert result.status == "reject" and result.reason == "continuation_expired"
    assert result.deadline == 10.25


def test_attempt_to_extend_pending_deadline_remains_capped_at_original_reference():
    kw = inputs()
    kw.update(pending_deadline=100., max_gap=100.)
    kw["pair"]["partial_distance"] = .43
    assert evaluate_continuation(**kw).deadline == pytest.approx(10.35)


@pytest.mark.parametrize("gap", [.10, .20, .35, 1.])
def test_smaller_configured_gap_and_hard_gap_are_respected(gap):
    kw = inputs()
    kw["max_gap"] = gap
    kw["current"]["capture_timestamp"] = 10. + min(gap, .35)
    result = evaluate_continuation(**kw)
    assert result.status == "reject" and result.reason == "continuation_expired"


@pytest.mark.parametrize("key,value", [
    ("uid", 0), ("uid", True), ("uid", 1.5), ("uid", 10 ** 400), ("track_id", "3"),
    ("source", "soft_partial"), ("source", None), ("source", "strong_unverified"),
    ("source", ["strong"]), ("source", {"strong": True}),
    ("competition_ok", False), ("competition_ok", 1), ("competition_ok", None),
    ("blocked", True), ("blocked", None), ("blocked", 0),
    ("full_limit", -1.), ("full_limit", float("nan")),
    ("confirm_limit", -1.), ("confirm_limit", .5),
    ("observe_limit", float("inf")), ("observe_limit", .39),
    ("max_gap", 0.), ("max_gap", -1.), ("max_gap", True),
    ("pending_deadline", float("nan")), ("pending_deadline", 10.),
    ("pending_deadline", "10.3"),
])
def test_bad_binding_flags_or_limits_fail_closed(key, value):
    kw = inputs()
    kw[key] = value
    assert evaluate_continuation(**kw).status == "reject"


@pytest.mark.parametrize("which", ["previous", "current"])
@pytest.mark.parametrize("key,value", [
    ("track_id", 4), ("track_id", None), ("track_id", True),
    ("is_fresh", False), ("is_fresh", 1), ("is_fresh", None),
    ("quality_bbox_ok", False), ("bbox_quality_tier", "weak"),
    ("search_observation_only", True), ("observation_only", True),
    ("preferred_search_low_confidence", True),
    ("capture_frame_id", None), ("capture_frame_id", True), ("capture_frame_id", 1.5),
    ("capture_timestamp", None), ("capture_timestamp", "10.1"),
    ("capture_timestamp", float("nan")), ("capture_timestamp", 0.),
    ("integrated_yaw_deg", None), ("integrated_yaw_deg", float("inf")),
    ("integrated_yaw_deg", True),
])
def test_unverified_metadata_is_not_continuation_evidence(which, key, value):
    kw = inputs()
    kw[which][key] = value
    assert evaluate_continuation(**kw).status == "reject"


@pytest.mark.parametrize("key,value", [("capture_frame_id", 830), ("capture_frame_id", 829),
                                      ("capture_timestamp", 10.), ("capture_timestamp", 9.9)])
def test_capture_id_and_time_must_both_be_strictly_new(key, value):
    kw = inputs()
    kw["current"][key] = value
    result = evaluate_continuation(**kw)
    assert result.status == "reject" and result.reason == "nonnew_capture"


@pytest.mark.parametrize("key,value", [
    ("ok", False), ("ok", None), ("ok", 1),
    ("yaw_compensated_center_jump_ratio", .150001),
    ("yaw_compensated_center_jump_ratio", -.01),
    ("yaw_compensated_center_jump_ratio", None),
    ("yaw_compensated_center_jump_ratio", float("nan")),
    ("area_similarity", .649999), ("area_similarity", 1.1),
    ("area_similarity", True), ("area_similarity", float("inf")),
])
def test_unexplained_position_or_scale_change_rejects(key, value):
    kw = inputs()
    kw["geometry"][key] = value
    assert evaluate_continuation(**kw).status == "reject"


def test_geometry_and_score_boundaries_allow_equal_values():
    kw = inputs()
    kw["geometry"].update(yaw_compensated_center_jump_ratio=.15, area_similarity=.65)
    kw["pair"].update(full_distance=.40, partial_distance=.40)
    assert evaluate_continuation(**kw).status == "accept"


@pytest.mark.parametrize("key,value", [
    ("qualified_comparison", False), ("qualified_comparison", 1),
    ("comparison_mode", "verified_pose_continuation"),
    ("comparison_mode", "verified_scale_continuation"),
    ("comparison_mode", "unavailable"), ("comparison_mode", None),
    ("winner_cap", None), ("winner_cap", True), ("winner_cap", 832), ("winner_cap", 900),
    ("full_distance", .400001), ("full_distance", -1.), ("full_distance", None),
    ("full_distance", float("nan")), ("partial_distance", None),
    ("partial_distance", True), ("partial_distance", -.1), ("partial_distance", float("inf")),
])
def test_missing_mixed_or_unapproved_pair_cannot_authorize(key, value):
    kw = inputs()
    kw["pair"][key] = value
    assert evaluate_continuation(**kw).status == "reject"


@pytest.mark.parametrize("which", ["previous", "current", "geometry", "pair"])
@pytest.mark.parametrize("value", [None, [], "invalid", {}])
def test_missing_evidence_never_raises_or_passes(which, value):
    kw = inputs()
    kw[which] = value
    assert evaluate_continuation(**kw).status == "reject"


def test_partial_crop_marker_does_not_reclassify_confirmed_evidence_source():
    kw = inputs()
    for marker in (False, True, None):
        kw["current"]["partial_observation"] = marker
        result = evaluate_continuation(**kw)
        assert result.status == "accept" and result.source == "partial"


def test_result_is_frozen_and_no_input_or_external_state_is_changed():
    kw = inputs()
    before = deepcopy(kw)
    result = evaluate_continuation(**kw)
    assert kw == before
    with pytest.raises(FrozenInstanceError):
        result.status = "hold"
