"""Raw, physically timed continuity must not turn filter lag into a new surface."""

import pytest

from test_astra_depth_consensus import CLIPPED, sensor


def test_cap232_raw_boundary_crossing_ignores_median_lag(sensor, caplog):
    runtime, clock, sample = sensor
    caplog.set_level("INFO")
    for mm in (2230, 2345, 2455):
        before = sample(mm, advance=.12, age=.02)
    assert before.distance_m == pytest.approx(2.345)
    assert before.raw_distance_m == pytest.approx(2.455)
    anchor = before.sample_timestamp

    # CAP232: actual raw delta 80.55mm, physical gap 105.22ms.
    # The former code compared 2.536m to median 2.345m and entered 1/3.
    result = sample(2536, advance=.190218, age=.105)
    assert result.raw_distance_m == pytest.approx(2.536)
    assert result.distance_m == pytest.approx(2.455)
    assert result.sample_timestamp - anchor == pytest.approx(.105218)
    assert result.sample_age_sec == pytest.approx(.105)
    assert result.sample_timestamp == runtime._latest_depth_ts
    assert result.jump_confirmation is None
    assert runtime._pending_jump_count == 0
    assert "anchor_raw_m=2.455 anchor_filtered_m=2.345" in caplog.text
    assert "sample_gap_ms=105.2" in caplog.text


def test_cap297_raw_far_anchor_continues_while_median_is_still_near(sensor):
    runtime, clock, sample = sensor
    sample(2440)
    sample(2498, advance=.10)
    anchor = sample(2535, advance=.05)
    assert anchor.distance_m == pytest.approx(2.498)
    assert anchor.raw_distance_m == pytest.approx(2.535)
    result = sample(2553, advance=.185484, age=.0631)
    assert result.raw_distance_m == pytest.approx(2.553)
    assert result.sample_timestamp - anchor.sample_timestamp == pytest.approx(.142384)
    assert clock[0] - anchor.sample_timestamp == pytest.approx(.205484)
    assert result.jump_confirmation is None
    assert runtime._pending_jump_count == 0


@pytest.mark.parametrize("age", [.02, .10, .18, .20, .249])
def test_processing_age_is_not_added_to_physical_sample_gap(sensor, age):
    runtime, clock, sample = sensor
    old = sample(2484, age=.02)
    result = sample(2524, advance=.15 + age - .02, age=age)
    assert result.raw_distance_m == pytest.approx(2.524)
    assert result.sample_timestamp - old.sample_timestamp == pytest.approx(.15)
    assert result.sample_timestamp == runtime._latest_depth_ts
    assert result.sample_age_sec == pytest.approx(age)
    assert result.jump_confirmation is None


@pytest.mark.parametrize("interruption", ["gap", "stale", "future", "sampling_overrun"])
def test_raw_boundary_never_extends_physical_deadlines(sensor, monkeypatch, interruption):
    runtime, clock, sample = sensor
    old = sample(2484)
    arguments = dict(advance=.15, age=.02)
    if interruption == "gap":
        arguments["advance"] = .251
    elif interruption == "stale":
        arguments.update(advance=.4, age=.251)
    elif interruption == "future":
        arguments["age"] = -.001
    else:
        select = runtime._select_multiregion_distance

        def overrun(*args):
            selected = select(*args)
            clock[0] += .02
            return selected

        monkeypatch.setattr(runtime, "_select_multiregion_distance", overrun)
        arguments.update(advance=.37, age=.24)
    result = sample(2524, **arguments)
    assert result.raw_distance_m is result.sample_timestamp is None
    assert runtime._last_accepted_ts == old.sample_timestamp
    assert runtime._distance_history[-1] == pytest.approx(2.484)
    assert result.jump_confirmation is None


@pytest.mark.parametrize("change", [
    "uid", "missing_history", "unpaired_history", "different_history_timestamp",
    "weak_anchor", "shift", "scale", "real_jump", "too_fast", "weak_current",
])
def test_raw_boundary_continuity_requires_both_endpoints(sensor, monkeypatch, change):
    runtime, clock, sample = sensor
    old = sample(2455)
    args = dict(advance=.12)
    mm = 2536
    if change == "uid":
        args["uid"] = 2
    elif change == "missing_history":
        runtime._distance_history.clear()
        runtime._distance_history_timestamps.clear()
    elif change == "unpaired_history":
        runtime._distance_history_timestamps.clear()
    elif change == "different_history_timestamp":
        runtime._distance_history_timestamps[-1] -= .01
    elif change == "weak_anchor":
        runtime._last_accepted_region_count = 2
    elif change == "shift":
        args["bbox"] = (290., 20., 470., 479.)
    elif change == "scale":
        args["bbox"] = (220., 20., 450., 479.)
    elif change == "real_jump":
        mm = 2670
    elif change == "too_fast":
        args["advance"] = .033
    else:
        monkeypatch.setattr(runtime, "_select_multiregion_distance", lambda *args:
                            (2.536, 2000, 3000, 80, "chest+abdomen", True))
    result = sample(mm, **args)
    assert result.raw_distance_m is result.sample_timestamp is None
    assert result.jump_confirmation is None
    assert runtime._last_accepted_ts == (0 if change == "uid" else old.sample_timestamp)


def test_duplicate_and_out_of_order_cannot_renew_raw_boundary_anchor(sensor):
    runtime, clock, sample = sensor
    sample(2455)
    accepted = sample(2536, advance=.12)
    raw_history = tuple(runtime._distance_history)
    duplicate = runtime.measure_target(CLIPPED, 640, 480, target_id=1, use_latest_depth=True)
    assert duplicate.raw_distance_m is duplicate.sample_timestamp is None
    assert duplicate.temporal_status == "duplicate"
    assert tuple(runtime._distance_history) == raw_history
    # A later FAILED attempt does not allow backfilling boundary proof from
    # an older sample between that attempt and the trusted anchor.
    sample(0, advance=.10)
    historical = sample(2553, advance=.03, age=.07)
    assert historical.raw_distance_m is historical.sample_timestamp is None
    assert historical.temporal_status == "out_of_order_jump_observation"
    assert runtime._last_accepted_ts == accepted.sample_timestamp
    assert tuple(runtime._distance_history) == raw_history


def test_optimistic_depth_commit_keeps_raw_anchor_with_filter_history(sensor):
    runtime, clock, sample = sensor
    sample(2230)
    sample(2345, advance=.12)
    sample(2455, advance=.12)
    old_stamp = runtime._last_accepted_ts
    # Publish the camera image only; all ranging changes must stay private
    # until a validated transaction commits.
    clock[0] += .19
    runtime._latest_depth = runtime._np.full((480, 640), 2536, dtype=runtime._np.uint16)
    runtime._latest_depth_ts = clock[0] - .105
    transaction = runtime.prepare_target_measurement(
        CLIPPED, 640, 480, target_id=1, use_latest_depth=True,
    )
    measured = transaction.run()
    assert measured.raw_distance_m == pytest.approx(2.536)
    assert runtime._last_accepted_ts == old_stamp
    assert runtime._distance_history[-1] == pytest.approx(2.455)
    assert transaction.commit(now=clock[0], max_sample_age_sec=.18) == measured
    assert runtime._last_accepted_ts == measured.sample_timestamp
    assert runtime._distance_history[-1] == pytest.approx(2.536)
    assert runtime._distance_history_timestamps[-1] == measured.sample_timestamp
    continued = sample(2553, advance=.10, age=.03)
    assert continued.raw_distance_m == pytest.approx(2.553)
    assert continued.jump_confirmation is None
