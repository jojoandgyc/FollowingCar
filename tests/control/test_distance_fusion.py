#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.control_types import PersonTarget, SteeringFeedback
from car_control_modular.distance_fusion import (
    DistanceFusionConfig,
    VisionRadarEncoderDistanceFusion,
)


def feedback(timestamp: float, rpm: float) -> SteeringFeedback:
    return SteeringFeedback(
        timestamp=timestamp,
        left_forward_rpm=rpm,
        right_forward_rpm=rpm,
        trustworthy=True,
    )


def main() -> int:
    cfg = DistanceFusionConfig(
        enabled=True,
        radar_median_window=3,
        visual_weight=0.80,
        encoder_wheel_circumference_m=0.60,
        encoder_max_step_m=0.12,
        hold_max_sec=2.50,
    )
    fusion = VisionRadarEncoderDistanceFusion(cfg)
    target_far = PersonTarget((300, 100, 400, 400), track_id=1, confidence=0.9, area=30000)
    fresh = fusion.update(
        target=target_far,
        frame_height=1000,
        radar_distance_m=3.0,
        radar_fresh=True,
        sample_age_sec=0.1,
        steering_feedback=feedback(1.0, 0.0),
        now=1.0,
    )
    if fresh.distance_m != 3.0 or fresh.mode != "radar" or fresh.confidence != 1.0:
        raise AssertionError(f"fresh radar anchor failed: {fresh}")

    # 框高度从 300 增到 375，视觉估算约 2.4m；编码器前进只做 20% 小补偿。
    target_closer = PersonTarget((300, 100, 400, 475), track_id=1, confidence=0.9, area=37500)
    held = fusion.update(
        target=target_closer,
        frame_height=1000,
        radar_distance_m=3.0,
        radar_fresh=False,
        sample_age_sec=0.2,
        steering_feedback=feedback(1.1, 30.0),
        now=1.1,
    )
    if held.mode != "visual_encoder" or not 2.45 < float(held.distance_m or 0.0) < 2.60:
        raise AssertionError(f"visual/encoder hold fusion failed: {held}")
    if held.encoder_delta_m <= 0.0 or held.visual_distance_m is None:
        raise AssertionError(f"hold fusion diagnostics missing: {held}")

    # 雷达当前点完全缺失且框被裁剪时，仍使用编码器消耗的位移，
    # 不能把短时雷达空窗直接变成 unavailable 或冻结旧距离。
    clipped_hold = fusion.update(
        target=PersonTarget((0, 0, 1000, 1000), track_id=1, confidence=0.9, area=1000000),
        frame_height=1000,
        radar_distance_m=None,
        radar_fresh=False,
        sample_age_sec=0.4,
        steering_feedback=feedback(1.15, 30.0),
        now=1.15,
    )
    if clipped_hold.mode != "encoder_hold" or clipped_hold.encoder_delta_m <= 0.0:
        raise AssertionError(f"encoder-only radar gap fallback failed: {clipped_hold}")

    # 目标继续靠近时，距离缩短不能被变化率限幅延迟。
    closer_again = fusion.update(
        target=PersonTarget((300, 100, 400, 600), track_id=1, confidence=0.9, area=50000),
        frame_height=1000,
        radar_distance_m=3.0,
        radar_fresh=False,
        sample_age_sec=0.3,
        steering_feedback=feedback(1.2, 30.0),
        now=1.2,
    )
    if float(closer_again.distance_m or 9.0) >= 2.10:
        raise AssertionError(f"approaching visual target must reduce distance promptly: {closer_again}")

    # 雷达恢复到更远值时，不能一帧把控制距离跳远。
    recovered = fusion.update(
        target=target_far,
        frame_height=1000,
        radar_distance_m=4.0,
        radar_fresh=True,
        sample_age_sec=0.1,
        steering_feedback=feedback(1.3, 0.0),
        now=1.3,
    )
    if recovered.mode != "radar_recover" or float(recovered.distance_m or 0.0) >= 4.0:
        raise AssertionError(f"radar recovery must be smoothed: {recovered}")

    # 突然近点是安全事件，不能被中值窗口吞掉。
    close = fusion.update(
        target=target_far,
        frame_height=1000,
        radar_distance_m=0.70,
        radar_fresh=True,
        sample_age_sec=0.1,
        steering_feedback=feedback(1.4, 0.0),
        now=1.4,
    )
    if abs(float(close.distance_m or 9.0) - 0.70) > 1e-9:
        raise AssertionError(f"sudden close radar return must be immediate: {close}")

    # 目标框严重贴满画面时，视觉尺度不可信，退回旧雷达 hold，不盲目改距。
    clipped = fusion.update(
        target=PersonTarget((0, 0, 1000, 1000), track_id=1, confidence=0.9, area=1000000),
        frame_height=1000,
        radar_distance_m=0.70,
        radar_fresh=False,
        sample_age_sec=0.2,
        steering_feedback=feedback(1.5, 0.0),
        now=1.5,
    )
    if clipped.mode != "radar_hold" or clipped.visual_distance_m is not None:
        raise AssertionError(f"clipped bbox must not drive visual scale: {clipped}")

    # A fresh far return must not release a near-distance hold in one frame.
    jump_fusion = VisionRadarEncoderDistanceFusion(
        DistanceFusionConfig(
            enabled=True,
            radar_median_window=1,
            fresh_far_jump_m=0.60,
            fresh_far_jump_confirm_frames=3,
        )
    )
    jump_fusion.update(
        target=target_far,
        frame_height=1000,
        radar_distance_m=0.80,
        radar_fresh=True,
        sample_age_sec=0.01,
        steering_feedback=feedback(2.0, 0.0),
        now=2.0,
    )
    pending_1 = jump_fusion.update(
        target=target_far,
        frame_height=1000,
        radar_distance_m=2.46,
        radar_fresh=True,
        sample_age_sec=0.01,
        steering_feedback=feedback(2.1, 0.0),
        now=2.1,
    )
    pending_2 = jump_fusion.update(
        target=target_far,
        frame_height=1000,
        radar_distance_m=2.45,
        radar_fresh=True,
        sample_age_sec=0.01,
        steering_feedback=feedback(2.2, 0.0),
        now=2.2,
    )
    accepted_jump = jump_fusion.update(
        target=target_far,
        frame_height=1000,
        radar_distance_m=2.46,
        radar_fresh=True,
        sample_age_sec=0.01,
        steering_feedback=feedback(2.3, 0.0),
        now=2.3,
    )
    if pending_1.mode != "radar_jump_pending" or float(pending_1.distance_m or 0.0) != 0.80:
        raise AssertionError(f"first fresh far jump must hold old distance: {pending_1}")
    if pending_2.mode != "radar_jump_pending" or float(pending_2.distance_m or 0.0) != 0.80:
        raise AssertionError(f"second fresh far jump must still hold old distance: {pending_2}")
    if abs(float(accepted_jump.distance_m or 0.0) - 2.46) > 1e-6:
        raise AssertionError(f"third consistent fresh far sample should be accepted: {accepted_jump}")

    short_fusion = VisionRadarEncoderDistanceFusion(
        DistanceFusionConfig(enabled=True, radar_median_window=1, hold_max_sec=0.20)
    )
    short_fusion.update(
        target=target_far,
        frame_height=1000,
        radar_distance_m=2.0,
        radar_fresh=True,
        sample_age_sec=0.01,
        steering_feedback=feedback(10.0, 0.0),
        now=10.0,
    )
    short_hold = short_fusion.update(
        target=target_closer,
        frame_height=1000,
        radar_distance_m=None,
        radar_fresh=False,
        sample_age_sec=0.10,
        steering_feedback=feedback(10.1, 20.0),
        now=10.1,
    )
    expired = short_fusion.update(
        target=target_closer,
        frame_height=1000,
        radar_distance_m=None,
        radar_fresh=False,
        sample_age_sec=0.25,
        steering_feedback=feedback(10.25, 20.0),
        now=10.25,
    )
    if short_hold.distance_m is None or short_hold.anchor_age_sec is None:
        raise AssertionError(f"200ms fused hold should remain available briefly: {short_hold}")
    if expired.distance_m is not None or expired.mode != "expired":
        raise AssertionError(f"fused hold must expire after 200ms: {expired}")

    print("distance_fusion: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
