#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.hazard_runtime import BunkerHazardRuntime, BunkerHazardRuntimeConfig


def _cfg() -> BunkerHazardRuntimeConfig:
    return BunkerHazardRuntimeConfig(
        enabled=True,
        mode="merged",
        workdir=str(ROOT),
        model_path="unused.rknn",
        engine="rknn",
        class_ids=(0, 1),
        class_names={0: "bunker", 1: "pond"},
        score_threshold=0.30,
        area_ratio_stop=0.05,
        num_classes=2,
        rknn_input_size=448,
        rknn_nms_threshold=0.45,
        rknn_backend="auto",
        rknn_core_mask="auto",
        rknn_input_format="RGB",
        rknn_box_format="xywh",
        sample_det_conf=0.30,
        get_frame_timeout_ms=200,
        loop_period_ms=100,
        split_restart_delay=0.8,
        split_active_hold_sec=0.35,
        stop_consec_frames=2,
        split_echo_raw=False,
        sample_binary="./sample_personv8_track",
        frame_width=100,
        frame_height=100,
        runtime_base="/tmp",
        rknn_target="rk3588",
    )


def main() -> int:
    runtime = BunkerHazardRuntime(_cfg())
    runtime.start()
    safe = runtime.check_merged_dets(
        [{"class_id": 0, "score": 0.90, "bbox": (0, 0, 10, 10)}],
        frame_area=10_000,
    )
    if safe is not None:
        raise AssertionError("small hazard area should not trigger")

    active = runtime.check_merged_dets(
        [{"class_id": 1, "score": 0.90, "bbox": (0, 0, 50, 50)}],
        frame_area=10_000,
    )
    if active is None or not active.active:
        raise AssertionError("large pond area should trigger")
    if active.class_name != "pond":
        raise AssertionError(f"unexpected class name: {active.class_name}")
    print("hazard_runtime merged ok", active.reason())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
