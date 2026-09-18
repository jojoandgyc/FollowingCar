"""Real-array ranging checks for the accepted-sample median's time horizon."""
import logging
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from car_control_modular import astra_depth
from car_control_modular.astra_depth import AstraDepthConfig, AstraDepthRuntime


BBOX = (220.0, 80.0, 420.0, 400.0)


@pytest.fixture
def ranging(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(astra_depth.time, "monotonic", lambda: clock[0])
    sensor = AstraDepthRuntime(AstraDepthConfig())
    sensor._np = np

    def sample(mm, *, advance=0.08, uid=1):
        clock[0] += advance
        sensor._latest_depth = np.full((480, 640), mm, dtype=np.uint16)
        sensor._latest_depth_ts = clock[0] - 0.02
        return sensor.measure_target(BBOX, 640, 480, target_id=uid, use_latest_depth=True)

    return sensor, clock, sample


def test_three_second_old_samples_do_not_bias_recovered_distance(ranging, caplog):
    sensor, _clock, sample = ranging
    for mm in (2077, 2137, 2204):
        previous = sample(mm)
    assert previous.distance_m == pytest.approx(2.137)
    with caplog.at_level(logging.INFO):
        recovered = sample(2858, advance=3.15)
    assert recovered.raw_distance_m == recovered.distance_m == pytest.approx(2.858)
    assert recovered.filter_expired_count == 3
    assert recovered.filter_window_count == 1
    assert tuple(sensor._distance_history) == pytest.approx((2.858,))
    assert tuple(sensor._distance_history_timestamps) == (recovered.sample_timestamp,)
    assert "filter_expired=3" in caplog.text
    assert "filter_window=1" in caplog.text


def test_recent_samples_keep_existing_three_point_median(ranging):
    sensor, _clock, sample = ranging
    for mm in (2077, 2137, 2204):
        measured = sample(mm)
    assert measured.distance_m == pytest.approx(2.137)
    assert measured.raw_distance_m == pytest.approx(2.204)
    assert measured.filter_expired_count == 0
    assert measured.filter_window_count == 3
    assert len(sensor._distance_history_timestamps) == 3
    assert sample(2250).distance_m == pytest.approx(2.204)


def test_failed_and_replayed_reads_cannot_refresh_history_times(ranging):
    sensor, _clock, sample = ranging
    sample(2000)
    before = (tuple(sensor._distance_history), tuple(sensor._distance_history_timestamps))
    assert sample(0, advance=0.5).raw_distance_m is None
    repeated = sensor.measure_target(BBOX, 640, 480, target_id=1, use_latest_depth=True)
    assert repeated.raw_distance_m is None
    assert before == (tuple(sensor._distance_history), tuple(sensor._distance_history_timestamps))
    result = sample(2200, advance=0.2)
    assert result.raw_distance_m == result.distance_m == 2.2
    assert result.filter_expired_count == 1


def test_uid_switch_resets_values_and_timestamps_together(ranging):
    sensor, _clock, sample = ranging
    for mm in (2000, 2100, 2200):
        sample(mm)
    other = sample(1400, uid=2)
    assert other.distance_m == 1.4
    assert other.filter_reset_count == 3
    assert tuple(sensor._distance_history) == (1.4,)
    assert tuple(sensor._distance_history_timestamps) == (other.sample_timestamp,)


@pytest.mark.parametrize("next_mm", [900, 3100])
def test_confirmed_surface_change_resets_both_queues(ranging, next_mm):
    sensor, _clock, sample = ranging
    sample(2000)
    result = sample(next_mm, advance=0.5)
    if next_mm == 3100:
        assert result.raw_distance_m is None
        result = sample(next_mm)
    assert result.distance_m == next_mm / 1000.0
    assert tuple(sensor._distance_history) == (result.distance_m,)
    assert tuple(sensor._distance_history_timestamps) == (result.sample_timestamp,)


def test_zero_strict_anchor_age_does_not_reject_fresh_depth(monkeypatch):
    monkeypatch.setattr(astra_depth.time, "monotonic", lambda: 100.0)
    sensor = AstraDepthRuntime(AstraDepthConfig(anchor_strict_age_sec=0.0))
    sensor._np = np
    sensor._latest_depth = np.full((480, 640), 2000, dtype=np.uint16)
    sensor._latest_depth_ts = 99.85
    measured = sensor.measure_target(BBOX, 640, 480, target_id=1, use_latest_depth=True)
    assert measured.raw_distance_m == measured.distance_m == 2.0
    assert measured.sample_timestamp == 99.85
