#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.astra_depth import AstraDepthConfig, AstraDepthRuntime
from car_control_modular.control_types import SteeringFeedback


def put_depth(runtime: AstraDepthRuntime, depth: np.ndarray) -> None:
    runtime._np = np
    with runtime._depth_lock:
        runtime._latest_depth = depth.copy()
        runtime._latest_depth_ts = time.monotonic()


def main() -> int:
    config = AstraDepthConfig(
        width=640,
        height=480,
        min_valid_pixels=20,
        median_window=3,
        hold_sec=0.20,
        max_distance_jump_m=0.80,
        jump_confirm_frames=2,
    )
    runtime = AstraDepthRuntime(config)
    bbox = (160.0, 40.0, 480.0, 460.0)

    # RGB inference finishes about 130ms after capture. The Depth ROI must use
    # the buffered frame from the RGB capture time, not the newest wall frame.
    aligned_runtime = AstraDepthRuntime(
        AstraDepthConfig(
            width=640,
            height=480,
            min_valid_pixels=20,
            median_window=1,
            rgb_processing_delay_sec=0.13,
        )
    )
    aligned_runtime._np = np
    aleft, atop, aright, abottom = aligned_runtime._scaled_target_roi(
        bbox, 640, 480, 640, 480, aligned_runtime.config
    )
    captured_depth = np.full((480, 640), 5000, dtype=np.uint16)
    captured_depth[atop:abottom, aleft:aright] = 1800
    latest_wall = np.full((480, 640), 5000, dtype=np.uint16)
    now = time.monotonic()
    with aligned_runtime._depth_lock:
        aligned_runtime._latest_depth = latest_wall
        aligned_runtime._latest_depth_ts = now - 0.01
        aligned_runtime._depth_history.append((now - 0.14, captured_depth))
        aligned_runtime._depth_history.append((now - 0.01, latest_wall))
    aligned = aligned_runtime.measure_target(bbox, 640, 480, target_id=6)
    if aligned.detail != "depth_multiregion" or abs((aligned.distance_m or 0.0) - 1.8) > 1e-6:
        raise AssertionError(f"RGB-delayed ROI must select the time-aligned Depth frame: {aligned}")

    # Background is far, while only the configured torso ROI contains the person.
    depth = np.full((480, 640), 5000, dtype=np.uint16)
    left, top, right, bottom = runtime._scaled_target_roi(
        bbox, 640, 480, 640, 480, config
    )
    depth[top:bottom, left:right] = 1800
    put_depth(runtime, depth)
    first = runtime.measure_target(bbox, 640, 480, target_id=7)
    if first.detail != "depth_multiregion" or abs((first.distance_m or 0.0) - 1.8) > 1e-6:
        raise AssertionError(f"registered torso ROI should select the person depth: {first}")

    # Chest, abdomen, left/right torso and lower abdomen are measured
    # independently. A hole in one region must not discard the other coherent
    # body surfaces or fall back to the wall median.
    multiregion = np.full((480, 640), 5000, dtype=np.uint16)
    torso_regions, clipped = runtime._torso_sampling_regions(bbox, 640, 480, 640, 480)
    if clipped or len(torso_regions) != 5:
        raise AssertionError(f"centered person should expose five torso regions: {torso_regions}")
    for index, (_name, rleft, rtop, rright, rbottom) in enumerate(torso_regions):
        multiregion[rtop:rbottom, rleft:rright] = 0 if index == 1 else 2200
    put_depth(runtime, multiregion)
    multi_measurement = runtime.measure_target(bbox, 640, 480, target_id=8)
    if multi_measurement.detail != "depth_multiregion" or abs(
        (multi_measurement.distance_m or 0.0) - 2.2
    ) > 1e-6:
        raise AssertionError(f"four coherent torso regions must beat the wall: {multi_measurement}")

    # A clipped, very small person ROI needs only max(20, visible_area*3%)
    # spatially connected samples. This case was rejected by the old fixed-80 gate.
    clipped_bbox_small = (0.0, 180.0, 40.0, 260.0)
    sparse_depth = np.zeros((480, 640), dtype=np.uint16)
    sparse_regions, sparse_clipped = runtime._torso_sampling_regions(
        clipped_bbox_small, 640, 480, 640, 480
    )
    _name, sleft, stop, _sright, _sbottom = sparse_regions[0]
    dleft, dtop, _dright, _dbottom = runtime._scaled_target_roi(
        clipped_bbox_small, 640, 480, 640, 480, config
    )
    sample_left = max(sleft, dleft)
    sample_top = max(stop, dtop)
    sparse_depth[sample_top : sample_top + 4, sample_left : sample_left + 5] = 1350
    put_depth(runtime, sparse_depth)
    sparse_measurement = runtime.measure_target(
        clipped_bbox_small, 640, 480, target_id=12
    )
    if not sparse_clipped or not sparse_measurement.bbox_clipped:
        raise AssertionError(f"edge-clipped sampling must be reported: {sparse_measurement}")
    if sparse_measurement.distance_m is None or abs(sparse_measurement.distance_m - 1.35) > 1e-6:
        raise AssertionError(f"20 connected pixels should satisfy the dynamic gate: {sparse_measurement}")

    # Once a close person is established, stable wall depth cannot unlock the
    # car while the box stays large or becomes horizontally clipped.
    near_bbox = (80.0, 0.0, 560.0, 479.0)
    near_depth = np.full((480, 640), 5000, dtype=np.uint16)
    nleft, ntop, nright, nbottom = runtime._scaled_target_roi(
        near_bbox, 640, 480, 640, 480, config
    )
    near_depth[ntop:nbottom, nleft:nright] = 700
    far_depth = np.full((480, 640), 4400, dtype=np.uint16)
    put_depth(runtime, far_depth)
    initial_far = runtime.measure_target(near_bbox, 640, 480, target_id=9)
    if initial_far.distance_m is not None or not initial_far.detail.startswith(
        "far_background_guard_large_bbox"
    ):
        raise AssertionError(f"large box must reject an initial far wall: {initial_far}")

    put_depth(runtime, near_depth)
    near = runtime.measure_target(near_bbox, 640, 480, target_id=9)
    if near.detail != "depth_multiregion" or abs((near.distance_m or 0.0) - 0.7) > 1e-6:
        raise AssertionError(f"close person depth should be accepted: {near}")

    guarded = None
    for _ in range(6):
        put_depth(runtime, far_depth)
        guarded = runtime.measure_target(near_bbox, 640, 480, target_id=9)
    if guarded is None or not guarded.detail.startswith("far_background_guard_large_bbox"):
        raise AssertionError(f"large close-person box must reject stable wall depth: {guarded}")
    if guarded.distance_m is not None and abs(guarded.distance_m - 0.7) > 1e-6:
        raise AssertionError(f"background guard must never publish the far wall: {guarded}")

    # 大框中心4x4即使落到远处背景，只要躯干ROI里仍有足量且与上一距离
    # 连续的人体深度簇，就应接回人体，而不是把整帧判成无效。
    anchored_depth = far_depth.copy()
    anchored_depth[ntop : ntop + 50, nleft : nleft + 50] = 720
    put_depth(runtime, anchored_depth)
    anchored = runtime.measure_target(near_bbox, 640, 480, target_id=9)
    if anchored.detail != "depth_foreground_fallback_large_bbox_anchor":
        raise AssertionError(f"large bbox must prefer previous-distance torso cluster: {anchored}")
    if anchored.raw_distance_m is None or abs(anchored.raw_distance_m - 0.72) > 1e-6:
        raise AssertionError(f"anchored torso cluster distance is wrong: {anchored}")

    clipped_bbox = (0.0, 80.0, 250.0, 400.0)
    put_depth(runtime, far_depth)
    clipped = runtime.measure_target(clipped_bbox, 640, 480, target_id=9)
    if not clipped.detail.startswith("far_background_guard_edge"):
        raise AssertionError(f"edge-clipped shrink must not release near lock: {clipped}")

    # A genuinely smaller, centered box may accept the farther surface, but it
    # needs the longer near-to-far confirmation window.
    far_bbox = (220.0, 80.0, 420.0, 400.0)
    pending = []
    for _ in range(config.near_far_jump_confirm_frames):
        put_depth(runtime, far_depth)
        pending.append(runtime.measure_target(far_bbox, 640, 480, target_id=9))
    if not all(
        item.detail.startswith("distance_jump_pending_")
        for item in pending[:-1]
    ):
        raise AssertionError(f"near-to-far jump must wait for all confirmations: {pending}")
    confirmed = pending[-1]
    if confirmed.detail != "depth_multiregion_after_jump_confirm" or abs((confirmed.distance_m or 0.0) - 4.4) > 1e-6:
        raise AssertionError(f"visually consistent far jump should eventually recover: {confirmed}")

    # Reusing the same Depth frame must report the threshold selected for this
    # jump (1_of_5), not fall back to the generic 1_of_2 configuration.
    reuse_runtime = AstraDepthRuntime(config)
    put_depth(reuse_runtime, near_depth)
    reuse_runtime.measure_target(near_bbox, 640, 480, target_id=13)
    put_depth(reuse_runtime, far_depth)
    first_pending = reuse_runtime.measure_target(far_bbox, 640, 480, target_id=13)
    reused_pending = reuse_runtime.measure_target(far_bbox, 640, 480, target_id=13)
    if first_pending.detail != "distance_jump_pending_1_of_5_hold":
        raise AssertionError(f"fresh near-to-far jump should select five confirmations: {first_pending}")
    if reused_pending.detail != "distance_jump_pending_1_of_5_hold":
        raise AssertionError(f"reused frame must keep the real five-frame threshold: {reused_pending}")

    # Once the old near anchor is older than 1.5s, three stable far samples
    # establish a new anchor instead of being rejected by the old 0.9m value.
    stale_runtime = AstraDepthRuntime(config)
    put_depth(stale_runtime, near_depth)
    stale_runtime.measure_target(near_bbox, 640, 480, target_id=14)
    stale_runtime._last_accepted_ts = time.monotonic() - 2.0
    stale_results = []
    for _ in range(config.reanchor_confirm_frames):
        put_depth(stale_runtime, far_depth)
        stale_results.append(stale_runtime.measure_target(far_bbox, 640, 480, target_id=14))
    if [item.confirm_count for item in stale_results[:-1]] != [1, 2]:
        raise AssertionError(f"expired anchor must count three stable samples: {stale_results}")
    if stale_results[-1].detail != "depth_reanchored_after_timeout":
        raise AssertionError(f"third stable far sample must rebuild the anchor: {stale_results[-1]}")

    # During the strict 0.6s window, a clearly smaller box plus encoder-confirmed
    # reverse displacement may shorten five confirmations to three.
    motion_runtime = AstraDepthRuntime(config)
    put_depth(motion_runtime, near_depth)
    base_ts = time.monotonic()
    motion_runtime.measure_target(
        near_bbox,
        640,
        480,
        target_id=15,
        steering_feedback=SteeringFeedback(
            timestamp=base_ts,
            left_forward_rpm=-60.0,
            right_forward_rpm=-60.0,
            trustworthy=True,
        ),
    )
    motion_results = []
    for index in range(config.motion_confirm_frames):
        put_depth(motion_runtime, far_depth)
        motion_results.append(
            motion_runtime.measure_target(
                far_bbox,
                640,
                480,
                target_id=15,
                steering_feedback=SteeringFeedback(
                    timestamp=base_ts + 0.20 + index * 0.03,
                    left_forward_rpm=-60.0,
                    right_forward_rpm=-60.0,
                    trustworthy=True,
                ),
            )
        )
    if motion_results[0].required_confirm_frames != 3:
        raise AssertionError(f"reverse and bbox evidence should use three confirmations: {motion_results}")
    if motion_results[-1].distance_m is None:
        raise AssertionError(f"motion-supported third sample should be accepted: {motion_results[-1]}")

    # A coherent closer surface is asymmetric safety evidence and is accepted
    # immediately even when it changes by more than max_distance_jump_m.
    closer_depth = np.full((480, 640), 900, dtype=np.uint16)
    put_depth(stale_runtime, closer_depth)
    closer = stale_runtime.measure_target(far_bbox, 640, 480, target_id=14)
    if closer.distance_m is None or abs(closer.distance_m - 0.9) > 1e-6:
        raise AssertionError(f"sudden closer surface must be accepted immediately: {closer}")

    # Invalid pixels may briefly use the last accepted value, never indefinitely.
    runtime._last_accepted_ts = time.monotonic()
    put_depth(runtime, np.zeros((480, 640), dtype=np.uint16))
    held = runtime.measure_target(far_bbox, 640, 480, target_id=9)
    if held.distance_m is None or not held.detail.endswith("_hold"):
        raise AssertionError(f"short invalid depth gap should hold: {held}")
    runtime._last_accepted_ts = time.monotonic() - 1.0
    expired = runtime.measure_target(far_bbox, 640, 480, target_id=9)
    if expired.distance_m is not None:
        raise AssertionError(f"expired depth hold must not drive the car: {expired}")

    print(
        "astra_depth_runtime: alignment, multi-region clusters, dynamic gate, "
        "aged anchors, motion evidence, asymmetric jumps and hold TTL passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
