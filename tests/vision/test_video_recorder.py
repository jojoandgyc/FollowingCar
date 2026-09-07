#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.video_recorder import (
    AsyncVideoRecorder,
    VideoControlOverlay,
    VideoFrameOverlay,
    VideoRecorderConfig,
    VideoTrackOverlay,
)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="follow-video-recorder-") as temp_dir:
        video_path = Path(temp_dir) / "camera_raw.avi"
        recorder = AsyncVideoRecorder(
            VideoRecorderConfig(
                output_path=str(video_path),
                fps=30.0,
                fourcc="MJPG",
                queue_capacity=8,
            ),
            cv2_module=cv2,
        )

        expected_frames = 6
        for frame_index in range(1, expected_frames + 1):
            frame = np.full((120, 160, 3), frame_index * 20, dtype=np.uint8)
            if not recorder.submit(
                frame,
                capture_frame_id=frame_index,
                monotonic_sec=100.0 + frame_index / 30.0,
                unix_sec=200.0 + frame_index / 30.0,
            ):
                raise AssertionError(f"frame {frame_index} was not queued")
            recorder.update_overlay(
                frame_index,
                [SimpleNamespace(bbox=(20, 10, 80, 100), score=0.9, class_id=0)],
                control=VideoControlOverlay(
                    control_frame_id=100 + frame_index,
                    active_target_id=1,
                    selected_target_id=1,
                    action_name="rotate_left",
                    requested_rpm=-6,
                    decision_reason="test_target_follow",
                ),
            )
        recorder.close(timeout_sec=3.0)

        if recorder.error is not None:
            raise AssertionError(f"recorder failed: {recorder.error}")
        if recorder.written_frames != expected_frames:
            raise AssertionError(
                f"expected {expected_frames} written frames, got {recorder.written_frames}"
            )

        capture = cv2.VideoCapture(str(video_path))
        decoded_frames = 0
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            decoded_frames += 1
        capture.release()
        if decoded_frames != expected_frames:
            raise AssertionError(
                f"expected {expected_frames} decoded frames, got {decoded_frames}"
            )

        index_path = video_path.with_suffix(".frames.csv")
        with index_path.open("r", encoding="utf-8", newline="") as index_file:
            rows = list(csv.DictReader(index_file))
        capture_indexes = [int(row["capture_frame_id"]) for row in rows]
        if capture_indexes != list(range(1, expected_frames + 1)):
            raise AssertionError(f"unexpected capture frame id: {capture_indexes}")
        if rows[0]["control_frame_id"] != "101":
            raise AssertionError(f"control frame id was not indexed: {rows[0]}")
        if rows[0]["action"] != "rotate_left" or rows[0]["requested_rpm"] != "-6":
            raise AssertionError(f"control diagnostics were not indexed: {rows[0]}")

        diagnostic = recorder._annotate_frame(
            np.zeros((480, 640, 3), dtype=np.uint8),
            7,
            85,
            VideoFrameOverlay(
                tracks=(
                    VideoTrackOverlay(
                        bbox=(50.0, 60.0, 250.0, 350.0),
                        track_id=13,
                        reid_uid=0,
                        mapped_uid=1,
                        distance=0.269,
                        assignment_reason="mapped_low_quality",
                        quality_reason="edge_touch>2",
                        active_target=True,
                    ),
                ),
                control=VideoControlOverlay(
                    control_frame_id=33,
                    active_target_id=1,
                    selected_target_id=1,
                    candidate_bbox=(350.0, 80.0, 580.0, 320.0),
                    candidate_score=0.95,
                    candidate_source="formal",
                    action_name="rotate_left",
                    requested_rpm=-6,
                    yaw_rate_dps=-12.5,
                    result_age_ms=84.0,
                    decision_reason="candidate_identity_test",
                ),
            ),
        )
        blue_border = diagnostic[200, 50]
        if not (int(blue_border[0]) > 200 and int(blue_border[1]) < 140):
            raise AssertionError(f"active target does not have a blue border: {blue_border}")
        magenta_border = diagnostic[200, 350]
        if not (
            int(magenta_border[0]) > 200
            and int(magenta_border[2]) > 200
            and int(magenta_border[1]) < 80
        ):
            raise AssertionError(f"control candidate does not have a magenta border: {magenta_border}")
        ruler_height = max(34, int(round(480 * 0.065)))
        ruler_y = 480 - ruler_height + max(8, ruler_height // 3)
        center_tick = diagnostic[ruler_y, int(round(0.5 * 639))]
        if not (int(center_tick[0]) > 200 and int(center_tick[1]) < 140):
            raise AssertionError(f"0.5 ruler tick is not highlighted: {center_tick}")

    print("video_recorder_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
