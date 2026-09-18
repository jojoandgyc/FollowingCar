"""Physical-time median window checks without camera or motor hardware."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from car_control_modular.depth_temporal_filter import append_depth_sample, reset_depth_window


def append(values, stamps, distance, stamp, *, now=None, age=0.6):
    return append_depth_sample(
        values, stamps, distance_m=distance, sample_timestamp=stamp,
        now=stamp if now is None else now, max_age_sec=age,
    )


def test_recent_three_samples_keep_original_median_and_float_deque():
    values, stamps = deque(maxlen=3), deque(maxlen=3)
    results = [append(values, stamps, distance, stamp) for distance, stamp in (
        (2.0, 100.0), (2.2, 100.033), (2.1, 100.066),
    )]
    assert [result.distance_m for result in results] == [2.0, 2.1, 2.1]
    assert tuple(values) == (2.0, 2.2, 2.1)
    assert all(isinstance(value, float) for value in values)
    assert tuple(stamps) == (100.0, 100.033, 100.066)
    assert all(result.accepted and result.expired_count == 0 for result in results)


def test_three_second_old_values_do_not_bias_recovered_distance():
    values, stamps = deque(maxlen=3), deque(maxlen=3)
    append(values, stamps, 2.137, 100.0)
    append(values, stamps, 2.204, 100.1)
    recovered = append(values, stamps, 2.858, 103.1)
    assert recovered.accepted
    assert recovered.distance_m == 2.858
    assert recovered.expired_count == 2
    assert recovered.window_count == 1
    assert recovered.oldest_sample_timestamp == recovered.newest_sample_timestamp == 103.1
    assert tuple(values) == (2.858,) and tuple(stamps) == (103.1,)


def test_partial_expiry_preserves_only_physically_recent_samples():
    values, stamps = deque([1.0, 2.0, 3.0], maxlen=3), deque([100.0, 100.3, 100.5], maxlen=3)
    result = append(values, stamps, 4.0, 100.7)
    assert result.expired_count == 1
    assert result.capacity_evicted_count == 0
    assert tuple(values) == (2.0, 3.0, 4.0)
    assert tuple(stamps) == (100.3, 100.5, 100.7)
    assert result.distance_m == 3.0


def test_age_boundary_is_inclusive_despite_float_roundoff():
    values, stamps = deque([1.0], maxlen=3), deque([100.1], maxlen=3)
    result = append(values, stamps, 2.0, 100.7)
    assert result.expired_count == 0
    assert result.distance_m == 1.5
    beyond = append(values, stamps, 3.0, 100.70001)
    assert beyond.expired_count == 1
    assert tuple(values) == (2.0, 3.0)


def test_window_age_uses_depth_timestamp_not_delayed_call_timestamp():
    values, stamps = deque([1.0], maxlen=3), deque([100.0], maxlen=3)
    result = append(values, stamps, 2.0, 100.5, now=100.7)
    assert result.expired_count == 0
    assert result.distance_m == 1.5
    assert result.newest_sample_timestamp == 100.5


def test_count_capacity_evicts_values_and_timestamps_together():
    values, stamps = deque(maxlen=3), deque(maxlen=3)
    for distance, stamp in [(1.0, 100.0), (2.0, 100.1), (3.0, 100.2)]:
        append(values, stamps, distance, stamp)
    result = append(values, stamps, 4.0, 100.3)
    assert result.capacity_evicted_count == 1
    assert result.expired_count == 0
    assert result.distance_m == 3.0
    assert tuple(values) == (2.0, 3.0, 4.0)
    assert tuple(stamps) == (100.1, 100.2, 100.3)


@pytest.mark.parametrize("distance,stamp,now,age,reason", [
    (None, 100.2, 100.2, 0.6, "invalid_distance"),
    (float("nan"), 100.2, 100.2, 0.6, "invalid_distance"),
    (float("inf"), 100.2, 100.2, 0.6, "invalid_distance"),
    (0.0, 100.2, 100.2, 0.6, "invalid_distance"),
    (-1.0, 100.2, 100.2, 0.6, "invalid_distance"),
    (True, 100.2, 100.2, 0.6, "invalid_distance"),
    ("invalid", 100.2, 100.2, 0.6, "invalid_distance"),
    (3.0, None, 100.2, 0.6, "invalid_sample_timestamp"),
    (3.0, 0.0, 100.2, 0.6, "invalid_sample_timestamp"),
    (3.0, -1.0, 100.2, 0.6, "invalid_sample_timestamp"),
    (3.0, float("nan"), 100.2, 0.6, "invalid_sample_timestamp"),
    (3.0, float("inf"), 100.2, 0.6, "invalid_sample_timestamp"),
    (3.0, True, 100.2, 0.6, "invalid_sample_timestamp"),
    (3.0, 100.3, 100.2, 0.6, "future_sample"),
    (3.0, 100.2, 101.0, 0.6, "stale_sample"),
    (3.0, 100.2, float("nan"), 0.6, "invalid_now"),
    (3.0, 100.2, -1.0, 0.6, "invalid_now"),
    (3.0, 100.2, 100.2, -1.0, "invalid_max_age"),
    (3.0, 100.2, 100.2, float("inf"), "invalid_max_age"),
    (3.0, 100.1, 100.2, 0.6, "duplicate_sample"),
    (3.0, 100.1 - 0.5e-9, 100.2, 0.6, "duplicate_sample"),
    (3.0, 100.05, 100.2, 0.6, "out_of_order_sample"),
])
def test_rejected_samples_do_not_pollute_healthy_window(distance, stamp, now, age, reason):
    values, stamps = deque([1.0, 2.0], maxlen=3), deque([100.0, 100.1], maxlen=3)
    result = append(values, stamps, distance, stamp, now=now, age=age)
    assert not result.accepted and result.reason == reason
    assert result.distance_m is None
    assert result.window_count == 2 and result.expired_count == 0
    assert tuple(values) == (1.0, 2.0) and tuple(stamps) == (100.0, 100.1)


def test_clock_rollback_does_not_roll_history_back():
    values, stamps = deque([2.0], maxlen=3), deque([100.0], maxlen=3)
    result = append(values, stamps, 9.0, 99.9, now=99.95)
    assert not result.accepted and result.reason == "out_of_order_sample"
    assert tuple(values) == (2.0,) and tuple(stamps) == (100.0,)


def test_untimestamped_legacy_values_are_discarded_not_assumed_recent():
    values, stamps = deque([2.137, 2.204], maxlen=3), deque(maxlen=3)
    result = append(values, stamps, 2.858, 103.1)
    assert result.distance_m == 2.858
    assert result.unknown_time_count == 2
    assert tuple(values) == (2.858,) and tuple(stamps) == (103.1,)


@pytest.mark.parametrize("old_values,old_stamps", [
    ([1.0, float("nan")], [100.0, 100.1]),
    ([1.0, 2.0], [100.0, None]),
    ([1.0, 2.0], [100.1, 100.0]),
    ([1.0, 2.0], [100.0, 100.0]),
    ([1.0, 2.0], [0.0, 100.0]),
    ([1.0, -2.0], [100.0, 100.1]),
])
def test_corrupted_history_is_cleared_as_one_untrusted_window(old_values, old_stamps):
    values, stamps = deque(old_values, maxlen=3), deque(old_stamps, maxlen=3)
    result = append(values, stamps, 2.858, 100.2)
    assert result.accepted and result.distance_m == 2.858
    assert result.invalid_history_count == 2
    assert tuple(values) == (2.858,) and tuple(stamps) == (100.2,)


def test_uid_or_confirmed_anchor_reset_clears_both_queues():
    values, stamps = deque([2.137, 2.204], maxlen=3), deque([100.0, 100.1], maxlen=3)
    assert reset_depth_window(values, stamps) == 2
    assert tuple(values) == tuple(stamps) == ()
    # A new UID starts independently, even if its aligned sample predates
    # the previous UID's final depth frame.
    result = append(values, stamps, 2.858, 99.9)
    assert result.distance_m == 2.858
    assert result.window_count == 1


def test_manually_clearing_legacy_values_drops_unpaired_timestamps():
    values, stamps = deque([2.0], maxlen=3), deque([100.0], maxlen=3)
    values.clear()
    result = append(values, stamps, 3.0, 100.1)
    assert result.distance_m == 3.0
    assert tuple(values) == (3.0,) and tuple(stamps) == (100.1,)


@pytest.mark.parametrize("value_capacity,stamp_capacity", [(3, 2), (3, None), (0, 0)])
def test_mismatched_or_zero_capacities_fail_closed(value_capacity, stamp_capacity):
    values, stamps = deque(maxlen=value_capacity), deque(maxlen=stamp_capacity)
    result = append(values, stamps, 2.0, 100.0)
    assert not result.accepted and result.reason == "window_capacity_mismatch"
    assert tuple(values) == tuple(stamps) == ()


def test_unbounded_deques_remain_time_bounded():
    values, stamps = deque(), deque()
    for index in range(20):
        result = append(values, stamps, 2.0, 100.0 + index * 0.1)
        assert result.accepted
        assert len(values) == len(stamps) <= 7


def test_integer_and_numeric_history_remain_float_compatible():
    values, stamps = deque([1, "2.0"], maxlen=3), deque([100, "100.1"], maxlen=3)
    result = append(values, stamps, 3, 100.2)
    assert result.distance_m == 2.0
    assert tuple(values) == (1.0, 2.0, 3.0)
    assert all(isinstance(value, float) for value in values)
