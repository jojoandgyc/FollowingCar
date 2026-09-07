#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.distance_runtime import DistanceRuntime, DistanceRuntimeConfig
from car_control_modular.control_types import PersonTarget, SteeringFeedback


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
        self.sample_ts = None

    def get_mmwave_targets_at(self, target_ts, max_age_sec=None):
        self.target_ts = target_ts
        self.max_age_sec = max_age_sec
        return list(self.targets), target_ts if self.sample_ts is None else self.sample_ts

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


def make_vision_mmwave_runtime(sensor_runtime, **overrides) -> DistanceRuntime:
    values = dict(
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
    )
    values.update(overrides)
    return DistanceRuntime(
        owner=FakeOwner(),
        config=DistanceRuntimeConfig(**values),
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

    # 复现实际日志：近处 -6度/0.69m 消失后，远处 -6度/3.29m 不能一帧换绑。
    jump_sensor = FakeMmWaveCache(targets=[{"index": 1, "angle": -6.0, "distance": 0.69}])
    jump_runtime = make_vision_mmwave_runtime(
        jump_sensor,
        vision_mmwave_distance_bias_m=0.0,
        vision_mmwave_switch_confirm_frames=3,
    )
    centered_target = PersonTarget((350, 100, 650, 500), track_id=11, confidence=0.9, area=120000)
    accepted_near = jump_runtime.get_frame_distance_state(1000, centered_target)
    jump_sensor.targets = [{"index": 2, "angle": -6.0, "distance": 3.29}]
    jump_1 = jump_runtime.get_frame_distance_state(1000, centered_target)
    jump_2 = jump_runtime.get_frame_distance_state(1000, centered_target)
    jump_3 = jump_runtime.get_frame_distance_state(1000, centered_target)
    print(
        "distance_jump_confirm:",
        accepted_near.used_distance_m,
        jump_1.source_detail,
        jump_2.source_detail,
        jump_3.source_detail,
        jump_3.used_distance_m,
    )
    if abs((accepted_near.used_distance_m or 0.0) - 0.69) > 1e-6:
        raise AssertionError(f"initial near radar point should be accepted: {accepted_near}")
    if jump_1.used_distance_m is not None or jump_1.source_detail != "distance_jump_pending_1_of_3":
        raise AssertionError(f"first far jump must stop and wait: {jump_1}")
    if jump_2.used_distance_m is not None or jump_2.source_detail != "distance_jump_pending_2_of_3":
        raise AssertionError(f"second far jump must still wait: {jump_2}")
    if abs((jump_3.used_distance_m or 0.0) - 3.29) > 1e-6:
        raise AssertionError(f"third consistent far point should be accepted: {jump_3}")
    if jump_3.source_detail != "matched_after_continuity_confirm":
        raise AssertionError(f"confirmed jump should expose a diagnostic reason: {jump_3}")

    interrupted_sensor = FakeMmWaveCache(targets=[{"index": 1, "angle": -6.0, "distance": 0.69}])
    interrupted_runtime = make_vision_mmwave_runtime(
        interrupted_sensor,
        vision_mmwave_distance_bias_m=0.0,
    )
    interrupted_runtime.get_frame_distance_state(1000, centered_target)
    interrupted_sensor.targets = [{"index": 2, "angle": -6.0, "distance": 3.29}]
    interrupted_runtime.get_frame_distance_state(1000, centered_target)
    interrupted_sensor.targets = []
    interrupted_runtime.get_frame_distance_state(1000, centered_target)
    interrupted_sensor.targets = [{"index": 2, "angle": -6.0, "distance": 3.29}]
    interrupted_state = interrupted_runtime.get_frame_distance_state(1000, centered_target)
    if interrupted_state.source_detail != "distance_jump_pending_1_of_3":
        raise AssertionError(f"radar gap must reset consecutive jump confirmation: {interrupted_state}")

    # 远点更靠近视觉中心时，也要优先保留角度和距离都连续的旧目标点。
    continuity_sensor = FakeMmWaveCache(targets=[{"index": 1, "angle": 8.0, "distance": 1.00}])
    continuity_runtime = make_vision_mmwave_runtime(
        continuity_sensor,
        vision_mmwave_distance_bias_m=0.0,
    )
    continuity_runtime.get_frame_distance_state(1000, centered_target)
    continuity_sensor.targets = [
        {"index": 2, "angle": 0.0, "distance": 3.00},
        {"index": 1, "angle": 8.5, "distance": 1.10},
    ]
    continuity_state = continuity_runtime.get_frame_distance_state(1000, centered_target)
    print("continuity_preferred:", continuity_state.used_distance_m, continuity_state.matched_angle_deg)
    if abs((continuity_state.used_distance_m or 0.0) - 1.10) > 1e-6:
        raise AssertionError(f"continuous radar point should beat a center-closer far point: {continuity_state}")

    # 宽人体框不能放过与人物中心相差超过 20 度的旁边雷达点。
    rejected_sensor = FakeMmWaveCache(targets=[{"index": 3, "angle": 21.0, "distance": 2.0}])
    rejected_runtime = make_vision_mmwave_runtime(
        rejected_sensor,
        vision_mmwave_distance_bias_m=0.0,
        vision_mmwave_angle_margin_deg=12.0,
        vision_mmwave_max_center_angle_diff_deg=20.0,
    )
    rejected_state = rejected_runtime.get_frame_distance_state(1000, centered_target)
    print("center_angle_rejected:", rejected_state.source_detail, rejected_state.used_distance_m)
    if rejected_state.used_distance_m is not None or rejected_state.source_detail != "center_angle_rejected":
        raise AssertionError(f">20 degree center mismatch must be rejected: {rejected_state}")

    # 突然变近可能是真实碰撞风险，必须立即采用，不能等待三帧确认。
    closer_sensor = FakeMmWaveCache(targets=[{"index": 1, "angle": 0.0, "distance": 3.0}])
    closer_runtime = make_vision_mmwave_runtime(closer_sensor, vision_mmwave_distance_bias_m=0.0)
    closer_runtime.get_frame_distance_state(1000, centered_target)
    closer_sensor.targets = [{"index": 1, "angle": 1.0, "distance": 0.70}]
    closer_state = closer_runtime.get_frame_distance_state(1000, centered_target)
    print("closer_immediate:", closer_state.source_detail, closer_state.used_distance_m)
    if abs((closer_state.used_distance_m or 0.0) - 0.70) > 1e-6:
        raise AssertionError(f"sudden closer point must be accepted immediately: {closer_state}")

    # 同一已确认雷达点的角度趋势只在视觉方向未知时提供左右提示。
    motion_sensor = FakeMmWaveCache(targets=[{"index": 1, "angle": -4.0, "distance": 1.5}])
    motion_runtime = make_vision_mmwave_runtime(motion_sensor, vision_mmwave_distance_bias_m=0.0)
    motion_runtime.get_frame_distance_state(1000, centered_target)
    motion_sensor.targets = [{"index": 1, "angle": 0.0, "distance": 1.5}]
    motion_runtime.get_frame_distance_state(1000, centered_target)
    motion_direction, motion_age, motion_debug = motion_runtime.get_recent_vision_mmwave_motion_hint()
    print("mmwave_motion_hint:", motion_direction, motion_age, motion_debug)
    if motion_direction != "right":
        raise AssertionError(f"positive camera-mapped radar trend should hint right: {motion_debug}")

    # 第二帧视觉已移动到右边，但雷达返回的是第一帧时刻的左侧样本，必须按历史视觉角度匹配。
    aligned_sensor = FakeMmWaveCache(targets=[{"index": 1, "angle": -18.0, "distance": 1.5}])
    aligned_runtime = make_vision_mmwave_runtime(
        aligned_sensor,
        vision_mmwave_distance_bias_m=0.0,
        vision_mmwave_angle_margin_deg=4.0,
    )
    left_target = PersonTarget((250, 100, 350, 500), track_id=12, confidence=0.9, area=40000)
    right_target = PersonTarget((650, 100, 750, 500), track_id=12, confidence=0.9, area=40000)
    aligned_runtime.get_frame_distance_state(1000, left_target)
    aligned_sensor.sample_ts = aligned_sensor.target_ts
    aligned_state = aligned_runtime.get_frame_distance_state(1000, right_target)
    print(
        "timestamp_aligned_visual:",
        aligned_state.used_distance_m,
        aligned_runtime.last_vision_mmwave_aligned_target_angle_deg,
    )
    if abs((aligned_state.used_distance_m or 0.0) - 1.5) > 1e-6:
        raise AssertionError(f"delayed radar sample should use historical visual angle: {aligned_state}")

    # 同一 ReID 且上次距离较远时，短暂丢点沿用旧距离，但必须明确标记为非新鲜样本。
    hold_sensor = FakeMmWaveCache(targets=[{"index": 1, "angle": 0.0, "distance": 2.95}])
    hold_runtime = make_vision_mmwave_runtime(
        hold_sensor,
        vision_mmwave_distance_bias_m=0.0,
        vision_mmwave_unmatched_hold_sec=2.50,
        vision_mmwave_unmatched_hold_min_distance_m=1.40,
    )
    hold_runtime.get_frame_distance_state(1000, centered_target)
    hold_sensor.targets = []
    held_state = hold_runtime.get_frame_distance_state(1000, centered_target)
    print("temporary_mmwave_hold:", held_state.source_detail, held_state.used_distance_m, held_state.sample_count)
    if held_state.source_detail != "no_radar_targets_hold":
        raise AssertionError(f"temporary loss should expose a hold reason: {held_state}")
    if abs((held_state.used_distance_m or 0.0) - 2.95) > 1e-6:
        raise AssertionError(f"temporary loss should retain the last trusted distance: {held_state}")
    if held_state.raw_distance_m is not None or held_state.sample_count != 0:
        raise AssertionError(f"held distance must not masquerade as a fresh radar sample: {held_state}")

    # 开启融合后，距离运行时应把同一目标的人体框变化和编码器反馈写入
    # DistanceState，而不是继续冻结旧毫米波距离。
    fusion_sensor = FakeMmWaveCache(targets=[{"index": 1, "angle": 0.0, "distance": 3.0}])
    fusion_runtime = make_vision_mmwave_runtime(
        fusion_sensor,
        vision_mmwave_fusion_enable=True,
        vision_mmwave_fusion_visual_weight=0.80,
        vision_mmwave_fusion_encoder_wheel_circumference_m=0.60,
        vision_mmwave_fusion_encoder_max_step_m=0.12,
    )
    fusion_target = PersonTarget((350, 100, 650, 400), track_id=1, confidence=0.9, area=90000)
    fusion_runtime.get_frame_distance_state(
        1000,
        fusion_target,
        frame_height=1000,
        steering_feedback=SteeringFeedback(timestamp=1.0, trustworthy=True),
    )
    fusion_sensor.targets = []
    fusion_state = fusion_runtime.get_frame_distance_state(
        1000,
        PersonTarget((350, 100, 650, 475), track_id=1, confidence=0.9, area=112500),
        frame_height=1000,
        steering_feedback=SteeringFeedback(
            timestamp=1.1,
            left_forward_rpm=30,
            right_forward_rpm=30,
            trustworthy=True,
        ),
    )
    print(
        "distance_fusion_state:",
        fusion_state.fusion_mode,
        fusion_state.used_distance_m,
        fusion_state.fusion_confidence,
    )
    if fusion_state.fusion_mode != "visual_encoder" or not 2.35 < float(fusion_state.used_distance_m or 0.0) < 2.50:
        raise AssertionError(f"distance runtime must expose fused visual/encoder distance: {fusion_state}")
    if fusion_state.fusion_encoder_delta_m <= 0.0:
        raise AssertionError(f"distance runtime must carry encoder compensation: {fusion_state}")

    hold_runtime._mmwave_accepted_ts = time.monotonic() - 2.51
    expired_state = hold_runtime.get_frame_distance_state(1000, centered_target)
    if expired_state.used_distance_m is not None or expired_state.source_detail != "no_radar_targets":
        raise AssertionError(f"hold must expire after 2.50 seconds: {expired_state}")

    # 1.40m 是停车释放边界，不允许在该距离开始沿用旧雷达距离继续前进。
    near_hold_sensor = FakeMmWaveCache(targets=[{"index": 1, "angle": 0.0, "distance": 1.40}])
    near_hold_runtime = make_vision_mmwave_runtime(
        near_hold_sensor,
        vision_mmwave_distance_bias_m=0.0,
        vision_mmwave_unmatched_hold_sec=2.50,
        vision_mmwave_unmatched_hold_min_distance_m=1.40,
    )
    near_hold_runtime.get_frame_distance_state(1000, centered_target)
    near_hold_sensor.targets = []
    near_expired_state = near_hold_runtime.get_frame_distance_state(1000, centered_target)
    if near_expired_state.used_distance_m is not None:
        raise AssertionError(f"1.40m release boundary must not enable mmwave hold: {near_expired_state}")

    # 远距离只为同一条已确认雷达轨迹增加 4 度人体框余量，中心角上限仍保持不变。
    far_margin_sensor = FakeMmWaveCache(targets=[{"index": 1, "angle": 16.0, "distance": 3.0}])
    far_margin_runtime = make_vision_mmwave_runtime(
        far_margin_sensor,
        vision_mmwave_distance_bias_m=0.0,
        vision_mmwave_angle_margin_deg=8.0,
        vision_mmwave_far_margin_start_m=2.50,
        vision_mmwave_far_angle_margin_extra_deg=4.0,
        vision_mmwave_max_center_angle_diff_deg=26.0,
    )
    narrow_target = PersonTarget((400, 100, 600, 500), track_id=21, confidence=0.9, area=80000)
    far_margin_runtime.get_frame_distance_state(1000, narrow_target)
    far_margin_sensor.targets = [{"index": 1, "angle": 20.0, "distance": 3.0}]
    far_margin_state = far_margin_runtime.get_frame_distance_state(1000, narrow_target)
    if abs((far_margin_state.used_distance_m or 0.0) - 3.0) > 1e-6:
        raise AssertionError(f"far same-track point should use the extra four-degree margin: {far_margin_state}")

    # 角度不连续的远点不能趁短暂失配换绑；不连续近点仍立即接管以保证防撞。
    far_margin_sensor.targets = [{"index": 2, "angle": -20.0, "distance": 4.0}]
    protected_state = far_margin_runtime.get_frame_distance_state(1000, narrow_target)
    if protected_state.source_detail != "continuity_rejected_hold" or protected_state.used_distance_m != 3.0:
        raise AssertionError(f"discontinuous far point must not replace the held association: {protected_state}")
    if far_margin_runtime._mmwave_associated_track_id != 21:
        raise AssertionError("temporary mismatch must preserve the existing radar association")
    far_margin_sensor.targets = [{"index": 2, "angle": -20.0, "distance": 1.2}]
    emergency_state = far_margin_runtime.get_frame_distance_state(1000, narrow_target)
    if abs((emergency_state.used_distance_m or 0.0) - 1.2) > 1e-6:
        raise AssertionError(f"discontinuous emergency-close point must be accepted immediately: {emergency_state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
