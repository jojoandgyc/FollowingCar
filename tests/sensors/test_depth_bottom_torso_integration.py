"""Real-array checks for the bounded bottom-edge torso recovery path.

No camera, serial device, or motor is opened. CAP280/283 geometry comes from
run_20260915_195712; depths are synthetic evidence, not claimed ground truth.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from car_control_modular import astra_depth
from car_control_modular.astra_depth import AstraDepthConfig, AstraDepthRuntime


CAP280 = (294.313720703125, 74.28895568847656, 441.64556884765625, 474.82720947265625)
CAP283 = (287.98126220703125, 71.84228515625, 433.79193115234375, 473.53375244140625)


@pytest.fixture
def sensor(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(astra_depth.time, "monotonic", lambda: clock[0])
    runtime = AstraDepthRuntime(AstraDepthConfig())
    runtime._np = np

    def sample(depth, *, bbox=CAP280, uid=1, advance=0.033, stamp=None, latest=True):
        clock[0] += advance
        physical_stamp = clock[0] - 0.01 if stamp is None else stamp
        array = (np.full((480, 640), depth, dtype=np.uint16)
                 if np.isscalar(depth) else depth.copy())
        runtime._depth_history.append((physical_stamp, array))
        if latest:
            runtime._latest_depth, runtime._latest_depth_ts = array, physical_stamp
        return runtime.measure_target(
            bbox, 640, 480, target_id=uid, use_latest_depth=latest,
            reference_timestamp=None if latest else physical_stamp,
        )

    return runtime, clock, sample


def patches(runtime, *, bbox=CAP280, names=("left_torso",), millimeters=2721):
    """Leave other sampler regions empty, including overlapping portions.

    Filling a complete side rectangle also fills part of the chest/abdomen,
    accidentally turning a one-region regression into three-region evidence.
    """
    regions, _ = runtime._torso_sampling_regions(bbox, 640, 480, 640, 480)
    masks = {}
    for name, left, top, right, bottom in regions:
        mask = np.zeros((480, 640), dtype=bool)
        mask[top:bottom, left:right] = True
        masks[name] = mask
    result = np.zeros((480, 640), dtype=np.uint16)
    for name in names:
        exclusive = masks[name].copy()
        for other, mask in masks.items():
            if other != name:
                exclusive &= ~mask
        result[exclusive] = millimeters
    return result


def seed(sample, *, distance=2137, bbox=CAP280):
    first = sample(distance, bbox=bbox)
    assert first.raw_distance_m == pytest.approx(distance / 1000.0)
    assert first.jump_confirmation is None


def recover(runtime, sample, *, bbox=CAP280, names=("left_torso",)):
    seed(sample, bbox=bbox)
    depth = patches(runtime, bbox=bbox, names=names)
    first = sample(depth, bbox=bbox, advance=1.0)
    assert first.torso_recovery_status == "pending_1_of_2"
    second = sample(depth, bbox=bbox)
    assert second.torso_recovery_status == "confirmed_2_of_2"
    return depth, second


@pytest.mark.parametrize("bbox,names", [
    (CAP280, ("left_torso",)),
    (CAP283, ("abdomen_center", "left_torso")),
])
def test_cap_bottom_core_regions_recover_in_two_physical_frames(sensor, bbox, names):
    runtime, _clock, sample = sensor
    seed(sample, bbox=bbox)
    depth = patches(runtime, bbox=bbox, names=names)
    first = sample(depth, bbox=bbox, advance=1.0)
    assert first.region_count == len(names)
    assert first.valid_pixels >= first.required_valid_pixels
    assert first.bbox_clipped
    assert first.raw_distance_m is first.distance_m is None
    assert first.confirm_count == 1 and first.required_confirm_frames == 2
    assert runtime._pending_jump_kind == "torso_recovery"
    assert first.jump_confirmation is None
    second = sample(depth, bbox=bbox)
    assert second.torso_recovery_status == "confirmed_2_of_2"
    assert second.distance_m == second.raw_distance_m == pytest.approx(2.721)
    assert second.confirm_count == second.required_confirm_frames == 2
    assert second.region_count == len(names)
    assert second.jump_confirmation is None  # never forge a three-region proof
    assert second.sample_timestamp == runtime._latest_depth_ts
    assert tuple(runtime._distance_history) == pytest.approx((2.721,))


def test_locally_recovered_surface_updates_every_new_depth_not_every_other_frame(sensor):
    runtime, _clock, sample = sensor
    _depth, _accepted = recover(runtime, sample)
    timestamps = []
    for mm in (2730, 2740, 2750, 2760):
        result = sample(patches(runtime, millimeters=mm))
        assert result.torso_recovery_status == "continued"
        assert result.raw_distance_m == pytest.approx(mm / 1000.0)
        assert result.sample_timestamp == runtime._latest_depth_ts
        assert result.jump_confirmation is None
        assert runtime._pending_jump_count == 0
        timestamps.append(result.sample_timestamp)
    assert len(set(timestamps)) == 4


def test_one_then_two_regions_share_continuous_core_evidence(sensor):
    runtime, _clock, sample = sensor
    seed(sample)
    first = sample(patches(runtime), advance=1.0)
    second = sample(patches(runtime, bbox=CAP283, names=("abdomen_center", "left_torso")), bbox=CAP283)
    assert first.confirm_count == 1
    assert second.torso_recovery_status == "confirmed_2_of_2"
    assert second.raw_distance_m == pytest.approx(2.721)
    assert second.jump_confirmation is None


@pytest.mark.parametrize("names,mm", [
    (("lower_abdomen",), 4400),
    (("lower_abdomen",), 2721),
    (("left_torso",), 4400),
    (("left_torso",), 3001),
])
def test_lower_body_background_and_out_of_bounds_core_never_gain_recovery(sensor, names, mm):
    runtime, _clock, sample = sensor
    seed(sample)
    depth = patches(runtime, names=names, millimeters=mm)
    for index in range(3):
        result = sample(depth, advance=1.0 if index == 0 else 0.033)
        assert result.raw_distance_m is None
        assert result.jump_confirmation is None
        assert runtime._pending_jump_kind != "torso_recovery"
        assert runtime._last_torso_recovery_evidence is None


@pytest.mark.parametrize("bbox", [
    (80.0, 40.0, 560.0, 479.0),  # large area and height
    (294.0, 20.0, 441.0, 479.0),  # height alone is large
    (0.0, 74.0, 147.0, 475.0),  # side + bottom
    (493.0, 74.0, 639.0, 475.0),  # other side + bottom
    (294.0, 0.0, 441.0, 475.0),  # head + bottom
    (1.0, 2.0, 639.0, 478.0),  # almost full image
])
def test_other_clipping_and_large_boxes_do_not_use_bottom_only_exception(sensor, bbox):
    runtime, _clock, sample = sensor
    seed(sample, bbox=bbox)
    depth = patches(runtime, bbox=bbox)
    for index in range(3):
        result = sample(depth, bbox=bbox, advance=1.0 if index == 0 else 0.033)
        assert result.raw_distance_m is None
        assert result.jump_confirmation is None
        assert runtime._last_torso_recovery_evidence is None


@pytest.mark.parametrize("scenario", ["no_anchor", "old_anchor", "wrong_uid", "far_anchor"])
def test_recovery_requires_same_recent_nearby_anchor(sensor, scenario):
    runtime, _clock, sample = sensor
    if scenario != "no_anchor":
        seed(sample, distance=2000 if scenario == "far_anchor" else 2137)
    depth = patches(runtime)
    for index in range(2):
        result = sample(depth, uid=2 if scenario == "wrong_uid" else 1,
                        advance=(3.01 if scenario == "old_anchor" else 1.0) if index == 0 else 0.033)
        assert result.raw_distance_m is None
        assert result.jump_confirmation is None
        assert runtime._pending_jump_kind != "torso_recovery"


def test_physically_impossible_core_jump_cannot_start_two_frame_shortcut(sensor):
    runtime, _clock, sample = sensor
    seed(sample)
    depth = patches(runtime)
    for _ in range(2):
        result = sample(depth, advance=0.033)
        assert result.rejection_reason == "torso_recovery_rate_guard"
        assert result.raw_distance_m is None
        assert runtime._pending_jump_count == 0
        assert result.jump_confirmation is None


@pytest.mark.parametrize("change", ["region", "bbox", "time_gap", "distance"])
def test_changed_local_evidence_restarts_confirmation(sensor, change):
    runtime, _clock, sample = sensor
    seed(sample)
    first = sample(patches(runtime), advance=1.0)
    assert first.confirm_count == 1
    bbox = CAP280 if change != "bbox" else tuple(
        value - 90.0 if index in (0, 2) else value for index, value in enumerate(CAP280)
    )
    names = ("right_torso",) if change == "region" else ("left_torso",)
    mm = 2540 if change == "distance" else 2721
    changed = patches(runtime, bbox=bbox, names=names, millimeters=mm)
    second = sample(changed, bbox=bbox, advance=0.27 if change == "time_gap" else 0.033)
    assert second.raw_distance_m is None
    assert second.torso_recovery_status == "pending_1_of_2"
    assert second.confirm_count == 1
    assert second.jump_confirmation is None
    third = sample(changed, bbox=bbox)
    assert third.torso_recovery_status == "confirmed_2_of_2"
    assert third.raw_distance_m == pytest.approx(mm / 1000.0)


@pytest.mark.parametrize("had_latch", [False, True])
def test_new_depth_hole_breaks_chain_and_requires_two_frames_again(sensor, had_latch):
    runtime, _clock, sample = sensor
    if had_latch:
        depth, _accepted = recover(runtime, sample)
    else:
        seed(sample)
        depth = patches(runtime)
        assert sample(depth, advance=1.0).confirm_count == 1
    failed = sample(0)
    assert failed.temporal_status == "new_sample"
    assert runtime._last_torso_recovery_evidence is None
    assert runtime._pending_jump_count == 0
    first = sample(depth)
    assert first.raw_distance_m is None and first.confirm_count == 1
    second = sample(depth)
    assert second.torso_recovery_status == "confirmed_2_of_2"
    assert second.raw_distance_m == pytest.approx(2.721)


@pytest.mark.parametrize("had_latch", [False, True])
@pytest.mark.parametrize("kind", ["duplicate", "old_valid", "old_invalid", "old_expired", "future"])
def test_rejected_time_observation_cannot_advance_or_revoke_current_evidence(sensor, had_latch, kind):
    runtime, clock, sample = sensor
    if had_latch:
        depth, _accepted = recover(runtime, sample)
    else:
        seed(sample)
        depth = patches(runtime)
        assert sample(depth, advance=1.0).confirm_count == 1
    accepted_stamp = runtime._last_accepted_ts
    pending = (runtime._pending_jump_kind, runtime._pending_jump_count,
               runtime._pending_jump_timestamp, runtime._pending_torso_recovery_evidence)
    latch = runtime._last_torso_recovery_evidence
    previous_stamp = runtime._latest_depth_ts
    stamp = (previous_stamp if kind == "duplicate" else clock[0] + 0.10 if kind == "future"
             else previous_stamp - 0.30 if kind == "old_expired" else previous_stamp - 0.015)
    rejected = sample(0 if kind == "old_invalid" else depth, advance=0.002, stamp=stamp, latest=False)
    assert rejected.raw_distance_m is None
    assert rejected.jump_confirmation is None
    assert runtime._last_accepted_ts == accepted_stamp
    assert (runtime._pending_jump_kind, runtime._pending_jump_count,
            runtime._pending_jump_timestamp, runtime._pending_torso_recovery_evidence) == pending
    assert runtime._last_torso_recovery_evidence is latch
    resumed = sample(depth)
    assert resumed.torso_recovery_status == ("continued" if had_latch else "confirmed_2_of_2")
    assert resumed.raw_distance_m == pytest.approx(2.721)


def test_new_failure_followed_by_late_valid_history_cannot_rebuild_pending(sensor):
    runtime, _clock, sample = sensor
    depth, _accepted = recover(runtime, sample)
    accepted_stamp = runtime._last_accepted_ts
    sample(0, advance=0.08)
    assert runtime._last_torso_recovery_evidence is None
    late = sample(depth, stamp=accepted_stamp + 0.04, latest=False, advance=0.002)
    assert late.temporal_status == "out_of_order_jump_observation"
    assert late.raw_distance_m is None
    assert runtime._pending_jump_count == 0
    assert runtime._last_torso_recovery_evidence is None
    first = sample(depth)
    assert first.torso_recovery_status == "pending_1_of_2"
    assert sample(depth).torso_recovery_status == "confirmed_2_of_2"


def test_disconnected_or_below_threshold_pixels_cannot_use_recovery(sensor):
    runtime, _clock, sample = sensor
    seed(sample)
    connected = patches(runtime)
    disconnected = np.zeros((480, 640), dtype=np.uint16)
    disconnected[::3, ::3] = connected[::3, ::3]
    result = sample(disconnected, advance=1.0)
    assert result.raw_distance_m is None
    assert result.jump_confirmation is None
    assert runtime._pending_jump_kind != "torso_recovery"


@pytest.mark.parametrize("start_with_core", [False, True])
def test_core_recovery_and_three_region_proof_do_not_share_confirmation_counts(sensor, start_with_core):
    runtime, _clock, sample = sensor
    seed(sample)
    core = patches(runtime)
    first = sample(core if start_with_core else 2721, advance=1.0)
    assert first.confirm_count == 1
    assert first.jump_confirmation is None
    first_other = sample(2721 if start_with_core else core)
    assert first_other.confirm_count == 1
    assert first_other.raw_distance_m is None
    assert first_other.jump_confirmation is None
    if start_with_core:
        assert first_other.region_count >= 3
        assert sample(2721).confirm_count == 2
        confirmed = sample(2721)
        assert confirmed.jump_confirmation is not None
        assert confirmed.jump_confirmation.region_count >= 3
        assert confirmed.jump_confirmation.confirm_count == 3
    else:
        confirmed = sample(core)
        assert confirmed.torso_recovery_status == "confirmed_2_of_2"
        assert confirmed.region_count == 1
        assert confirmed.jump_confirmation is None
