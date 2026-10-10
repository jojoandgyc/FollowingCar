"""Bounded turn/crop tracking policy; no model or hardware execution."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict

import pytest

from rk_vision.verified_continuation import evaluate_pose_continuation


def inputs():
    # Recorded CAP773→777 crop narrowing. Timestamps and yaw are synthetic.
    origin = dict(track_id=3, capture_frame_id=773, capture_timestamp=10.,
        is_fresh=True, quality_bbox_ok=True, bbox_quality_tier="strong",
        integrated_yaw_deg=100., image_width=640, image_height=480,
        detector_bbox=[145.83255, 3.18704, 326.88574, 447.20398])
    return dict(uid=1, track_id=3, source="partial", origin=origin,
        previous=dict(origin), current=dict(origin, capture_frame_id=777,
            capture_timestamp=10.2,
            detector_bbox=[206.68596, 4.18248, 364.56116, 476.28760]),
        geometry=dict(ok=True, yaw_compensated_center_jump_ratio=.02,
                      area_similarity=.93),
        competition_ok=True, blocked=False, full_distance=.1797,
        full_limit=.30, partial_distance=.424, confirm_limit=.40,
        observe_limit=.45, pair_comparable=True)


def advance(kw, result, *, cap=None, stamp=None, **changes):
    kw["state"] = asdict(result)
    kw["previous"] = dict(kw["current"])
    kw["current"] = dict(kw["current"],
        capture_frame_id=cap or kw["current"]["capture_frame_id"]+2,
        capture_timestamp=stamp or kw["current"]["capture_timestamp"]+.1)
    kw.update(changes)


def test_cap777_grey_onset_preserves_fixed_origin_through_later_torso_disagreement():
    kw = inputs()
    first = evaluate_pose_continuation(**kw)
    assert first.status == "continue"
    assert first.origin_cap == 773 and first.deadline == 12.
    advance(kw, first, partial_distance=.458, full_distance=.262)
    second = evaluate_pose_continuation(**kw)
    assert second.status == "continue"
    advance(kw, second, cap=784, partial_distance=.565, full_distance=.207)
    third = evaluate_pose_continuation(**kw)
    assert third.status == "continue"
    assert third.origin_cap == first.origin_cap and third.deadline == first.deadline
    assert third.last_cap == 784
    assert third.recovery_streak == 0


def test_recovery_requires_two_new_independently_comparable_samples():
    kw = inputs()
    result = evaluate_pose_continuation(**kw)
    advance(kw, result, partial_distance=.223, pair_comparable=False)
    # A good but retained/noncomparable score cannot close the pose episode.
    result = evaluate_pose_continuation(**kw)
    assert result.status == "continue" and result.recovery_streak == 0
    advance(kw, result, pair_comparable=True)
    kw["current"]["detector_bbox"] = list(kw["origin"]["detector_bbox"])
    result = evaluate_pose_continuation(**kw)
    assert result.status == "continue" and result.recovery_streak == 1
    assert result.reason == "pose_recovery_pending"
    advance(kw, result)
    recovered = evaluate_pose_continuation(**kw)
    assert recovered.status == "recover" and recovered.recovery_streak == 2
    assert recovered.deadline == 12.


def test_grey_sample_resets_recovery_hysteresis_without_extending_deadline():
    kw = inputs()
    result = evaluate_pose_continuation(**kw)
    advance(kw, result, partial_distance=.223)
    kw["current"]["detector_bbox"] = list(kw["origin"]["detector_bbox"])
    result = evaluate_pose_continuation(**kw)
    assert result.recovery_streak == 1
    advance(kw, result, partial_distance=.424)
    result = evaluate_pose_continuation(**kw)
    assert result.status == "continue" and result.recovery_streak == 0
    assert result.deadline == 12.


def test_temporary_crop_restoration_does_not_erase_bounded_turn_lineage():
    kw = inputs()
    result = evaluate_pose_continuation(**kw)
    advance(kw, result, partial_distance=.458)
    kw["current"]["detector_bbox"] = list(kw["origin"]["detector_bbox"])
    result = evaluate_pose_continuation(**kw)
    assert result.status == "continue" and result.reason == "pose_change_torso_uncertain"
    assert result.deadline == 12. and result.recovery_streak == 0


@pytest.mark.parametrize("partial", [.399, .40, .450001, .565])
def test_cannot_start_episode_from_ordinary_confirmation_or_isolated_conflict(partial):
    kw = inputs()
    kw["partial_distance"] = partial
    result = evaluate_pose_continuation(**kw)
    assert result.status == "reject" and result.reason == "entry_torso_unverified"


def test_narrow_shape_without_a_change_cannot_seed_an_episode():
    kw = inputs()
    kw["origin"]["detector_bbox"] = list(kw["current"]["detector_bbox"])
    kw["previous"] = dict(kw["origin"])
    result = evaluate_pose_continuation(**kw)
    assert result.status == "reject" and result.reason == "pose_change_unverified"


def test_small_vertical_crop_boundary_change_can_seed_an_episode():
    kw = inputs()
    kw["origin"]["detector_bbox"] = [220., 120., 340., 468.]
    kw["previous"] = dict(kw["origin"])
    kw["current"]["detector_bbox"] = [220., 120., 340., 474.]
    assert evaluate_pose_continuation(**kw).status == "continue"


def test_lateral_clip_is_not_a_turn_crop_comparability_excuse():
    kw = inputs()
    kw["current"]["detector_bbox"] = [0., 4., 158., 476.]
    result = evaluate_pose_continuation(**kw)
    assert result.status == "reject" and result.reason == "crop_unverified"


@pytest.mark.parametrize("phase", ["entry", "active"])
@pytest.mark.parametrize("changes", [
    dict(blocked=True), dict(blocked=0),
    dict(competition_ok=False), dict(competition_ok=1),
    dict(full_distance=.300001), dict(full_limit=.17),
    dict(full_distance=.324, full_limit=.45),
    dict(full_distance=None), dict(full_distance=float("nan")),
    dict(partial_distance=float("inf")), dict(partial_distance=-.1),
    dict(confirm_limit=.46), dict(max_gap=0.), dict(max_duration=True),
    dict(uid=True), dict(track_id=4), dict(source="soft_partial"),
    dict(pair_comparable=None),
])
def test_negative_evidence_and_strict_configuration_apply_throughout(phase, changes):
    kw = inputs()
    if phase == "active":
        advance(kw, evaluate_pose_continuation(**kw))
    kw.update(changes)
    assert evaluate_pose_continuation(**kw).status == "reject"


@pytest.mark.parametrize("full,retention,expected", [
    (.30, None, "continue"), (.300001, None, "reject"),
    (.324, .38, "continue"), (.38, .38, "continue"),
    (.380001, .45, "reject"), (.324, .32, "reject"),
    (.324, True, "reject"), (.324, float("nan"), "reject"),
])
def test_explicit_full_retention_hysteresis_never_changes_default_or_stricter_limit(full, retention, expected):
    kw = inputs()
    advance(kw, evaluate_pose_continuation(**kw), full_distance=full,
            retention_full_limit=retention)
    assert evaluate_pose_continuation(**kw).status == expected


@pytest.mark.parametrize("retention", [.38, .45, 1.])
def test_retention_permission_cannot_raise_entry_full_limit(retention):
    kw = inputs()
    kw.update(full_distance=.300001, retention_full_limit=retention)
    result = evaluate_pose_continuation(**kw)
    assert result.status == "reject" and result.reason == "full_distance_conflict"


def test_cap781_and_784_rebound_shape_does_not_break_same_episode():
    kw = inputs()
    result = evaluate_pose_continuation(**kw)
    rows = [
        (779, [218.543, 5.145, 386.843, 477.444], .262, .458),
        (781, [217.842, 6.172, 403.100, 477.195], .281, .554),
        (784, [217.583, 2.045, 402.369, 477.032], .207, .565),
    ]
    for cap, box, full, part in rows:
        advance(kw, result, cap=cap, full_distance=full, partial_distance=part)
        kw["current"]["detector_bbox"] = box
        result = evaluate_pose_continuation(**kw)
        assert result.status == "continue" and result.origin_cap == 773
        assert result.deadline == 12.


@pytest.mark.parametrize("phase", ["entry", "active"])
@pytest.mark.parametrize("key,value", [
    ("ok", False), ("ok", 1), ("yaw_compensated_center_jump_ratio", .150001),
    ("yaw_compensated_center_jump_ratio", float("nan")),
    ("area_similarity", .64999), ("area_similarity", None),
])
def test_geometry_veto_cannot_be_overruled_by_pose(phase, key, value):
    kw = inputs()
    if phase == "active":
        advance(kw, evaluate_pose_continuation(**kw))
    kw["geometry"][key] = value
    result = evaluate_pose_continuation(**kw)
    assert result.status == "reject" and result.reason == "geometry_unverified"


@pytest.mark.parametrize("which", ["origin", "previous", "current"])
@pytest.mark.parametrize("key,value", [
    ("is_fresh", False), ("quality_bbox_ok", False), ("bbox_quality_tier", "weak"),
    ("track_id", 4), ("integrated_yaw_deg", None), ("capture_frame_id", True),
    ("capture_timestamp", "10.0"), ("detector_bbox", [0., 2., 180., 470.]),
    ("detector_bbox", [100., 2., 180., float("inf")]), ("image_width", True),
])
def test_source_and_candidate_metadata_must_remain_valid(which, key, value):
    kw = inputs()
    kw[which][key] = value
    assert evaluate_pose_continuation(**kw).status == "reject"


@pytest.mark.parametrize("cap,stamp", [(777, 10.2), (776, 10.3), (779, 10.1), (779, 10.2)])
def test_consumed_capture_cannot_refresh_or_complete_pose_lineage(cap, stamp):
    kw = inputs()
    first = evaluate_pose_continuation(**kw)
    advance(kw, first, cap=cap, stamp=stamp)
    result = evaluate_pose_continuation(**kw)
    assert result.status == "reject" and result.reason == "nonnew_capture"


@pytest.mark.parametrize("key,value", [
    ("uid", 2), ("track_id", 4), ("source", "strong"), ("status", "recover"),
    ("origin_cap", 777), ("origin_timestamp", 10.2), ("deadline", 20.),
    ("last_cap", 773), ("last_timestamp", 10.), ("recovery_streak", 2),
    ("recovery_streak", True),
])
def test_state_cannot_roll_origin_or_change_binding_or_skip_consumed_capture(key, value):
    kw = inputs()
    advance(kw, evaluate_pose_continuation(**kw))
    kw["state"][key] = value
    result = evaluate_pose_continuation(**kw)
    assert result.status == "reject" and result.reason == "invalid_pose_state"


def test_fixed_epoch_does_not_renew_with_each_valid_narrow_crop():
    kw = inputs()
    for index in range(9):
        result = evaluate_pose_continuation(**kw)
        assert result.status == "continue" and result.deadline == 12.
        advance(kw, result, stamp=10.4+.2*index, partial_distance=.565)
    kw["current"]["capture_timestamp"] = 12.
    result = evaluate_pose_continuation(**kw)
    assert result.status == "reject" and result.reason == "pose_continuation_expired"


@pytest.mark.parametrize("gap", [.15, .35, 5.])
def test_configured_gap_is_honored_and_cannot_exceed_hard_cap(gap):
    kw = inputs()
    first = evaluate_pose_continuation(**kw)
    advance(kw, first, stamp=10.2+min(gap, .35), max_gap=gap)
    result = evaluate_pose_continuation(**kw)
    assert result.status == "reject" and result.reason == "continuation_gap"


def test_deadline_cannot_be_extended_by_later_configuration_change():
    kw = inputs()
    kw["max_duration"] = .5
    first = evaluate_pose_continuation(**kw)
    advance(kw, first, stamp=10.5, max_duration=20.)
    result = evaluate_pose_continuation(**kw)
    assert result.status == "reject" and result.reason == "pose_continuation_expired"
    assert result.deadline == 10.5


def test_hysteresis_retains_slight_narrowing_until_comparability_recovers():
    kw = inputs()
    first = evaluate_pose_continuation(**kw)
    advance(kw, first, partial_distance=.458)
    x1, y1, x2, y2 = kw["origin"]["detector_bbox"]
    kw["current"]["detector_bbox"] = [x1, y1, x1+.92*(x2-x1), y2]
    assert evaluate_pose_continuation(**kw).status == "continue"


def test_no_mutation_or_learning_permission_is_returned():
    kw = inputs()
    before = deepcopy(kw)
    result = evaluate_pose_continuation(**kw)
    assert kw == before
    assert "learning_allowed" not in asdict(result)
    with pytest.raises(FrozenInstanceError):
        result.status = "recover"
