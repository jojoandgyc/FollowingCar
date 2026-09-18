"""Physical-depth ordering across latest and RGB-aligned reads; no hardware."""
from pathlib import Path
import logging
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from car_control_modular import astra_depth
from car_control_modular.astra_depth import AstraDepthConfig, AstraDepthRuntime


BBOX = (220.0, 80.0, 420.0, 400.0)
CLIPPED = (240.0, 20.0, 420.0, 479.0)


@pytest.mark.parametrize("stamp,status", [(100.10, "duplicate"), (100.08, "older_than_anchor")])
def test_old_or_duplicate_samples_skip_roi_work(ranging, monkeypatch, stamp, status):
    sensor, _, read = ranging
    read(1500, 100.10, 100.12)
    def forbidden(*args, **kwargs):
        pytest.fail("old sample must not execute ROI extraction or clustering")
    monkeypatch.setattr(sensor, "_scaled_target_roi", forbidden)
    monkeypatch.setattr(sensor, "_select_multiregion_distance", forbidden)
    result = read(5000, stamp, 100.15, latest=False)
    assert result.temporal_status == status
    assert result.distance_m == pytest.approx(1.5)
    assert result.raw_distance_m is result.sample_timestamp is result.jump_confirmation is None
    assert sensor._last_accepted_ts == 100.10


def test_duplicate_cannot_extend_expired_hold(ranging, monkeypatch):
    sensor, _, read = ranging
    read(1500, 100.0, 100.01)
    monkeypatch.setattr(sensor, "_select_multiregion_distance",
                        lambda *a: pytest.fail("duplicate must not resample"))
    result = read(1500, 100.0, 100.22)
    assert result.distance_m is None  # hold=200ms, even though frame TTL=250ms
    assert result.temporal_status == "duplicate"
    assert sensor._last_accepted_ts == 100.0


def test_pending_duplicate_skips_roi_and_preserves_confirmation(ranging, monkeypatch):
    sensor, _, read = ranging
    first = read(3000, 99.98, 100.0, bbox=CLIPPED)
    monkeypatch.setattr(sensor, "_scaled_target_roi",
                        lambda *a: pytest.fail("pending duplicate must not resample"))
    again = read(3000, 99.98, 100.03, latest=False, bbox=CLIPPED)
    assert first.confirm_count == again.confirm_count == 1
    assert again.rejection_reason == "reused_depth_frame"
    assert again.jump_confirmation is None


@pytest.fixture
def ranging(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(astra_depth.time, "monotonic", lambda: clock[0])
    sensor = AstraDepthRuntime(AstraDepthConfig())
    sensor._np = np

    def read(value, stamp, now, *, latest=True, bbox=BBOX, uid=1):
        clock[0] = now
        depth = np.full((480, 640), value, dtype=np.uint16)
        if latest:
            sensor._latest_depth, sensor._latest_depth_ts = depth, stamp
        else:
            sensor._depth_history.append((stamp, depth))
        return sensor.measure_target(
            bbox, 640, 480, target_id=uid, use_latest_depth=latest,
            reference_timestamp=None if latest else stamp,
        )

    return sensor, clock, read


def test_newer_failed_latest_does_not_block_unseen_continuous_history(ranging, caplog):
    sensor, _clock, read = ranging
    read(1960, 99.98, 100.0)
    bad = read(4890, 100.13, 100.15)
    assert bad.raw_distance_m is None
    assert bad.rejection_reason == "physically_impossible_far_jump"
    with caplog.at_level(logging.INFO):
        historical = read(1925, 100.08, 100.2, latest=False)
    assert historical.raw_distance_m == 1.925
    assert historical.sample_timestamp == sensor._last_accepted_ts == 100.08
    assert historical.sample_age_sec == pytest.approx(0.12)
    assert historical.temporal_status == "historical_after_later_attempt"
    assert sensor._last_processed_depth_ts == 100.13
    assert "sample_age_ms=120.0" in caplog.text
    assert "source=rgb_aligned" in caplog.text
    assert "candidate=1.925" in caplog.text


def test_old_history_cannot_overwrite_newer_accepted_distance(ranging):
    sensor, _clock, read = ranging
    read(1960, 99.98, 100.0)
    read(1940, 100.13, 100.15)
    before = (sensor._last_accepted_ts, sensor._last_accepted_distance_m,
              tuple(sensor._distance_history), sensor._last_accepted_bbox_area_ratio)
    old = read(1900, 100.08, 100.2, latest=False)
    assert old.raw_distance_m is old.sample_timestamp is None
    assert old.temporal_status == "older_than_anchor"
    assert before == (sensor._last_accepted_ts, sensor._last_accepted_distance_m,
                      tuple(sensor._distance_history), sensor._last_accepted_bbox_area_ratio)


@pytest.mark.parametrize("first_latest", [True, False])
def test_same_physical_sample_across_callers_counts_once(ranging, first_latest):
    sensor, _clock, read = ranging
    first = read(3000, 99.98, 100.0, latest=first_latest, bbox=CLIPPED)
    duplicate = read(3000, 99.98, 100.03, latest=not first_latest, bbox=CLIPPED)
    assert first.confirm_count == duplicate.confirm_count == 1
    assert duplicate.temporal_status == "duplicate"
    assert duplicate.jump_confirmation is None
    assert read(3000, 100.04, 100.06, bbox=CLIPPED).confirm_count == 2
    accepted = read(3000, 100.07, 100.09, bbox=CLIPPED)
    assert accepted.confirm_count == 3 and accepted.jump_confirmation is not None
    again = read(3000, 100.07, 100.1, latest=False, bbox=CLIPPED)
    assert again.raw_distance_m is again.jump_confirmation is None
    assert sensor._last_accepted_ts == 100.07


@pytest.mark.parametrize("old_stamp,value", [(100.00, 0), (99.0, 3000)])
def test_old_invalid_or_stale_history_does_not_clear_new_confirmation(ranging, old_stamp, value):
    sensor, _clock, read = ranging
    read(3000, 99.98, 100.0, bbox=CLIPPED)
    read(3000, 100.04, 100.06, bbox=CLIPPED)
    before = (sensor._pending_jump_count, sensor._pending_jump_timestamp)
    old = read(value, old_stamp, 100.1, latest=False, bbox=CLIPPED)
    assert old.raw_distance_m is old.jump_confirmation is None
    assert (sensor._pending_jump_count, sensor._pending_jump_timestamp) == before
    accepted = read(3000, 100.11, 100.13, bbox=CLIPPED)
    assert accepted.confirm_count == 3 and accepted.jump_confirmation is not None


def test_old_continuous_history_cannot_rewrite_newer_pending_state(ranging):
    sensor, _clock, read = ranging
    read(1800, 99.98, 100.0, bbox=CLIPPED)
    pending = read(2700, 100.36, 100.38, bbox=CLIPPED)
    assert pending.confirm_count == 1
    old = read(1880, 100.30, 100.4, latest=False, bbox=CLIPPED)
    assert old.temporal_status == "older_than_pending"
    assert old.raw_distance_m is old.jump_confirmation is None
    assert sensor._pending_jump_count == 1 and sensor._pending_jump_timestamp == 100.36
    assert sensor._last_accepted_ts == 99.98


def test_late_far_history_does_not_backfill_confirmation_across_newer_failure(ranging):
    sensor, _clock, read = ranging
    read(1900, 99.98, 100.0)
    read(0, 100.13, 100.15)
    late_far = read(2900, 100.08, 100.2, latest=False)
    assert late_far.raw_distance_m is late_far.jump_confirmation is None
    assert late_far.temporal_status == "out_of_order_jump_observation"
    assert sensor._pending_jump_count == 0
    assert sensor._last_accepted_ts == 99.98


def test_failed_sample_cannot_be_retried_as_new_evidence(ranging):
    sensor, _clock, read = ranging
    read(1900, 99.98, 100.0)
    read(0, 100.13, 100.15)
    duplicate = read(1900, 100.13, 100.17, latest=False)
    assert duplicate.temporal_status == "duplicate"
    assert duplicate.raw_distance_m is None
    assert sensor._last_accepted_ts == 99.98


def test_future_sample_does_not_poison_attempt_watermark(ranging):
    sensor, _clock, read = ranging
    read(1900, 99.98, 100.0)
    future = read(1900, 101.0, 100.1)
    assert future.temporal_status == "stale_or_future"
    assert sensor._last_processed_depth_ts == 99.98
    assert read(1910, 100.13, 100.15).raw_distance_m == pytest.approx(1.91)


def test_expired_during_sampling_cannot_update_sensor_anchor(ranging, monkeypatch):
    sensor, clock, read = ranging
    read(1900, 99.98, 100.0)
    select = sensor._select_multiregion_distance

    def slow_select(*args):
        result = select(*args)
        clock[0] += 0.26
        return result

    monkeypatch.setattr(sensor, "_select_multiregion_distance", slow_select)
    expired = read(1920, 100.08, 100.1)
    assert expired.temporal_status == "expired_during_sampling"
    assert expired.raw_distance_m is expired.jump_confirmation is None
    assert sensor._last_accepted_ts == 99.98


def test_switching_uid_clears_old_anchor_and_confirmation(ranging):
    sensor, _clock, read = ranging
    read(3000, 99.98, 100.0, bbox=CLIPPED)
    read(3000, 100.01, 100.03, bbox=CLIPPED)
    other = read(3000, 100.04, 100.06, bbox=CLIPPED, uid=2)
    assert other.confirm_count == 1 and other.jump_confirmation is None
    assert sensor._last_accepted_distance_m is None
    assert tuple(sensor._attempted_depth_samples) == (100.04,)
