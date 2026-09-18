import pytest

from rk_vision.reacquire_quarantine import ReacquireQuarantine


def sample(gate, capture_id, timestamp, **changes):
    arguments = dict(
        uid=1, track_id=35, capture_frame_id=capture_id,
        capture_timestamp=timestamp, frame_index=capture_id,
        is_fresh=True, quality_ok=True, quality_tier="strong",
        match_source="strong", feature_available=True, strong_distance=0.19,
        center_x_ratio=0.408, area_ratio=0.033,
    )
    arguments.update(changes)
    return gate.observe(**arguments)


def armed():
    gate = ReacquireQuarantine()
    gate.arm(1, 35, 1571, 10.0, 589)
    return gate


def test_cap1571_reacquire_cap1573_cannot_update_gallery():
    gate = ReacquireQuarantine()
    gate.arm(1, 35, 1571, 20546.1780784, 589)
    result = sample(gate, 1573, 20546.278135895, strong_distance=0.20739662647247314)
    assert result.hold and result.reason == "frozen_strong_distance"
    assert gate.is_held(1)
    assert sample(gate, 1575, 20546.38).hold


def test_normal_original_track_and_first_identity_are_not_quarantined():
    gate = ReacquireQuarantine()
    assert not gate.is_held(1)
    assert sample(gate, 1, 10.0).reason == "not_armed"
    assert not sample(gate, 2, 10.1).hold


def test_three_stable_frames_do_not_release_before_one_second():
    gate = armed()
    for capture, timestamp in ((1573, 10.1), (1575, 10.3), (1577, 10.5)):
        result = sample(gate, capture, timestamp)
        assert result.hold
    assert result.streak == 3 and result.reason == "minimum_duration"
    assert sample(gate, 1579, 10.75).hold
    result = sample(gate, 1581, 11.0)
    assert not result.hold and result.reason == "released"
    assert result.streak == 5 and not gate.is_held(1)


def test_elapsed_time_alone_never_releases():
    gate = armed()
    assert sample(gate, 1700, 100.0).hold
    assert sample(gate, 1701, 100.1).hold
    released = sample(gate, 1702, 100.2)
    assert not released.hold and released.streak == 3


@pytest.mark.parametrize("changes", [
    {"is_fresh": False}, {"quality_ok": False}, {"quality_tier": "weak"},
    {"match_source": "weak"}, {"match_source": "partial"},
    {"feature_available": False}, {"strong_distance": None},
    {"strong_distance": float("nan")}, {"strong_distance": 0.201},
    {"capture_timestamp": None}, {"capture_timestamp": float("nan")},
    {"capture_frame_id": None}, {"center_x_ratio": None}, {"area_ratio": 0.0},
])
def test_invalid_or_weak_observation_resets_streak_without_unfreezing(changes):
    gate = armed()
    sample(gate, 1573, 11.0)
    sample(gate, 1575, 11.1)
    rejected = sample(gate, 1577, 11.2, **changes)
    assert rejected.hold and rejected.streak == 0
    assert sample(gate, 1579, 11.3).streak == 1
    assert gate.is_held(1)


def test_duplicate_capture_does_not_count_or_reset_valid_streak():
    gate = armed()
    assert sample(gate, 1571, 10.0).reason == "duplicate_capture"
    sample(gate, 1573, 11.0)
    repeated = sample(gate, 1573, 11.0, frame_index=9999)
    assert repeated.hold and repeated.reason == "duplicate_capture" and repeated.streak == 1
    assert sample(gate, 1575, 11.1).streak == 2
    assert not sample(gate, 1577, 11.2).hold


def test_rejected_capture_cannot_be_reused_as_new_strong_evidence():
    gate = armed()
    sample(gate, 1573, 11.0)
    assert sample(gate, 1575, 11.1, quality_ok=False).streak == 0
    repeated = sample(gate, 1575, 11.1)
    assert repeated.hold and repeated.reason == "duplicate_capture" and repeated.streak == 0
    assert sample(gate, 1577, 11.2).streak == 1


@pytest.mark.parametrize("changes", [
    {"center_x_ratio": 0.7}, {"area_ratio": 0.01},
])
def test_geometry_jump_cannot_release(changes):
    gate = armed()
    sample(gate, 1573, 11.0)
    sample(gate, 1575, 11.1)
    rejected = sample(gate, 1577, 11.2, **changes)
    assert rejected.hold and rejected.reason == "geometry_discontinuity"
    assert sample(gate, 1579, 11.3).streak == 1


def test_capture_gap_and_time_reversal_reset_stable_proof():
    gate = armed()
    sample(gate, 1573, 11.0)
    sample(gate, 1575, 11.1)
    assert sample(gate, 1577, 11.5).reason == "capture_gap"
    assert sample(gate, 1579, 11.6).streak == 1
    assert sample(gate, 1578, 11.55).reason == "out_of_order_capture"
    assert sample(gate, 1581, 11.7).streak == 1


def test_track_change_restarts_duration_and_confirmation():
    gate = armed()
    sample(gate, 1573, 11.0)
    sample(gate, 1575, 11.1)
    changed = sample(gate, 1577, 11.2, track_id=36)
    assert changed.hold and changed.reason == "track_changed" and changed.streak == 0
    for capture, timestamp in ((1579, 11.4), (1581, 11.6), (1583, 11.8)):
        assert sample(gate, capture, timestamp, track_id=36).hold
    assert sample(gate, 1585, 12.0, track_id=36).hold
    assert not sample(gate, 1587, 12.2, track_id=36).hold


def test_reset_clears_proof_but_only_explicit_pruning_removes_isolation():
    gate = armed()
    sample(gate, 1573, 11.0)
    sample(gate, 1575, 11.1)
    gate.reset(1)
    assert gate.is_held(1) and sample(gate, 1577, 11.2).streak == 1
    gate.reset()
    assert gate.is_held(1) and sample(gate, 1579, 11.3).streak == 1
    gate.prune([1, 2])
    assert gate.is_held(1)
    gate.prune([])
    assert not gate.is_held(1)


def test_missing_arm_timestamp_uses_first_valid_rgb_time_as_timer_origin():
    gate = ReacquireQuarantine()
    gate.arm(1, 35, 1571, None, 589)
    assert sample(gate, 1573, None).hold
    for index in range(5):
        result = sample(gate, 1575 + index, 100.0 + index * 0.2)
        assert result.hold
    assert not sample(gate, 1580, 101.0).hold


@pytest.mark.parametrize("uid", [None, 0, float("nan"), "bad"])
def test_invalid_uid_fails_closed_and_resets_known_track(uid):
    gate = armed()
    sample(gate, 1573, 11.0)
    assert sample(gate, 1575, 11.1, uid=uid).hold
    assert sample(gate, 1577, 11.2).streak == 1


def test_rearming_new_uid_on_same_track_does_not_carry_old_proof():
    gate = armed()
    sample(gate, 1573, 11.0)
    sample(gate, 1575, 11.1)
    gate.arm(2, 35, 1577, 11.2, 590)
    assert gate.is_held(1) and gate.is_held(2)
    assert sample(gate, 1579, 11.3).streak == 1
    assert sample(gate, 1579, 11.3, uid=2).streak == 1


def test_observed_uid_change_resets_previous_identity_proof():
    gate = armed()
    sample(gate, 1573, 11.0)
    sample(gate, 1575, 11.1)
    sample(gate, 1577, 11.2, uid=2)
    assert gate.is_held(1)
    assert sample(gate, 1579, 11.3).streak == 1
