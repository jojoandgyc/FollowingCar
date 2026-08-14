#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.distance_runtime import DistanceRuntime, DistanceRuntimeConfig
from car_control_modular.control_types import PersonTarget


class FakeSensors:
    def __init__(self, distances_cm):
        self.distances_cm = list(distances_cm)

    def get_ultrasonic_distance_cm(self):
        if not self.distances_cm:
            return None
        return self.distances_cm.pop(0)

    def get_mmwave_distance_cm(self):
        return None


class FakeMmWaveCache:
    def __init__(self, targets=None):
        self.targets = targets or [
            {"index": 1, "angle": 18.0, "distance": 1.1},
            {"index": 2, "angle": 1.0, "distance": 2.2},
        ]
        self.target_ts = None
        self.max_age_sec = None

    def get_mmwave_targets_at(self, target_ts, max_age_sec=None):
        self.target_ts = target_ts
        self.max_age_sec = max_age_sec
        return list(self.targets), target_ts

    def get_mmwave_targets(self):
        raise AssertionError("vision_mmwave should use the async cache helper")


class FakeOwner:
    frame_index = 7


def make_runtime(distances_cm) -> DistanceRuntime:
    return DistanceRuntime(
        owner=object(),
        config=DistanceRuntimeConfig(
            distance_source="ultrasonic",
            vision_mmwave_source_aliases=frozenset({"vision_mmwave"}),
            module_mmwave_enable=False,
            module_ultrasonic_enable=True,
            vision_hfov_deg=90.0,
            vision_mmwave_angle_margin_deg=12.0,
            vision_mmwave_angle_offset_deg=0.0,
            vision_mmwave_angle_sign=1.0,
            vision_mmwave_match_mode="angle",
            vision_mmwave_distance_bias_m=0.0,
            vision_mmwave_min_distance_m=0.5,
            vision_mmwave_max_distance_m=10.0,
            vision_mmwave_min_output_distance_m=0.03,
            vision_mmwave_hard_stop_ttl_sec=0.3,
            vision_mmwave_log_every_frames=10,
            ultrasonic_filter_window=3,
            ultrasonic_target_confirm_frames=2,
            ultrasonic_brake_confirm_frames=2,
            ultrasonic_hysteresis_m=0.25,
            ultrasonic_immediate_brake_m=0.35,
        ),
        sensor_runtime=FakeSensors(distances_cm),
    )


def make_vision_mmwave_runtime(sensor_runtime) -> DistanceRuntime:
    return DistanceRuntime(
        owner=FakeOwner(),
        config=DistanceRuntimeConfig(
            distance_source="vision_mmwave",
            vision_mmwave_source_aliases=frozenset({"vision_mmwave"}),
            module_mmwave_enable=True,
            module_ultrasonic_enable=False,
            vision_hfov_deg=90.0,
            vision_mmwave_angle_margin_deg=8.0,
            vision_mmwave_angle_offset_deg=0.0,
            vision_mmwave_angle_sign=1.0,
            vision_mmwave_match_mode="angle",
            vision_mmwave_distance_bias_m=0.1,
            vision_mmwave_min_distance_m=0.5,
            vision_mmwave_max_distance_m=10.0,
            vision_mmwave_min_output_distance_m=0.03,
            vision_mmwave_hard_stop_ttl_sec=0.3,
            vision_mmwave_log_every_frames=0,
            vision_mmwave_latency_sec=0.10,
            vision_mmwave_cache_max_age_sec=0.50,
            vision_mmwave_use_async_cache=True,
        ),
        sensor_runtime=sensor_runtime,
    )


def main() -> int:
    one_bad_sample = make_runtime([567, 64, 567])
    s1 = one_bad_sample.get_sensor_distance_state(target_distance_m=1.5, brake_distance_m=0.8)
    s2 = one_bad_sample.get_sensor_distance_state(target_distance_m=1.5, brake_distance_m=0.8)
    s3 = one_bad_sample.get_sensor_distance_state(target_distance_m=1.5, brake_distance_m=0.8)
    print("one_bad_sample:", s1.trigger, s2.trigger, s3.trigger, s2.used_distance_m)
    if s2.trigger != "clear" or s2.used_distance_m <= 1.5:
        raise AssertionError(f"single ultrasonic near spike should not stop: {s2}")

    target_confirm = make_runtime([140, 135])
    t1 = target_confirm.get_sensor_distance_state(target_distance_m=1.5, brake_distance_m=0.8)
    t2 = target_confirm.get_sensor_distance_state(target_distance_m=1.5, brake_distance_m=0.8)
    print("target_confirm:", t1.trigger, t2.trigger, t1.used_distance_m, t2.used_distance_m)
    if t1.trigger != "target_unconfirmed" or t1.used_distance_m <= 1.5:
        raise AssertionError(f"first target-close sample should be softened: {t1}")
    if t2.trigger != "target_confirmed" or t2.used_distance_m > 1.5:
        raise AssertionError(f"second target-close sample should stop: {t2}")

    brake_confirm = make_runtime([70, 68])
    b1 = brake_confirm.get_sensor_distance_state(target_distance_m=1.5, brake_distance_m=0.8)
    b2 = brake_confirm.get_sensor_distance_state(target_distance_m=1.5, brake_distance_m=0.8)
    print("brake_confirm:", b1.trigger, b2.trigger, b1.used_distance_m, b2.used_distance_m)
    if b1.trigger != "brake_unconfirmed" or b1.used_distance_m <= 1.5:
        raise AssertionError(f"first brake-close sample should be softened: {b1}")
    if b2.trigger != "brake_confirmed" or b2.used_distance_m >= 0.8:
        raise AssertionError(f"second brake-close sample should brake: {b2}")

    immediate = make_runtime([30])
    i1 = immediate.get_sensor_distance_state(target_distance_m=1.5, brake_distance_m=0.8)
    print("immediate:", i1.trigger, i1.used_distance_m)
    if i1.trigger != "brake_immediate" or i1.used_distance_m >= 0.8:
        raise AssertionError(f"immediate close distance should brake: {i1}")

    mmwave_sensor = FakeMmWaveCache()
    mmwave_runtime = make_vision_mmwave_runtime(mmwave_sensor)
    target = PersonTarget((450, 100, 550, 400), track_id=1, confidence=0.9, area=30000)
    ms = mmwave_runtime.get_frame_distance_state(1000, target, target_distance_m=1.5, brake_distance_m=0.8)
    print("vision_mmwave:", ms.trigger, ms.used_distance_m, ms.source_detail, ms.matched_angle_deg, ms.sample_age_sec)
    if abs((ms.used_distance_m or 0.0) - 2.1) > 1e-6:
        raise AssertionError(f"vision_mmwave should choose the angle-matched target and apply bias: {ms}")
    if ms.source_detail != "matched" or ms.matched_angle_deg != 1.0:
        raise AssertionError(f"vision_mmwave match debug fields missing: {ms}")
    if mmwave_sensor.target_ts is None or mmwave_sensor.max_age_sec is None:
        raise AssertionError("vision_mmwave should request a timestamp-aligned cached sample")

    tie_sensor = FakeMmWaveCache(
        targets=[
            {"index": 1, "angle": 3.0, "distance": 2.19},
            {"index": 2, "angle": -32.0, "distance": 4.11},
        ]
    )
    tie_runtime = make_vision_mmwave_runtime(tie_sensor)
    tie_runtime.config = DistanceRuntimeConfig(
        **{
            **tie_runtime.config.__dict__,
            "vision_mmwave_distance_bias_m": 0.0,
            "vision_mmwave_angle_tie_margin_deg": 6.0,
        }
    )
    wide_target = PersonTarget((420, 100, 860, 400), track_id=1, confidence=0.9, area=88000)
    tie_state = tie_runtime.get_frame_distance_state(1920, wide_target, target_distance_m=1.5, brake_distance_m=0.8)
    print("vision_mmwave_tie:", tie_state.used_distance_m, tie_state.matched_angle_deg)
    if abs((tie_state.used_distance_m or 0.0) - 2.19) > 1e-6 or tie_state.matched_angle_deg != 3.0:
        raise AssertionError(f"near angle tie should prefer the closer radar target: {tie_state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
