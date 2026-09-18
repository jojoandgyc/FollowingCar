#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
import tempfile
import threading
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


def check_translucent_text() -> None:
    background = np.full((64, 320, 3), 96, dtype=np.uint8)
    rendered = {}
    for alpha in (0.0, 0.62, 1.0):
        recorder = AsyncVideoRecorder(
            VideoRecorderConfig(output_path="unused.avi", fps=30.0, overlay_text_alpha=alpha),
            cv2_module=cv2,
        )
        image = background.copy()
        recorder._draw_text_box(
            image, "SHARP 123.4", x=10, y=32,
            font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.5,
            foreground=(160, 255, 255), background=(0, 0, 0),
        )
        rendered[alpha] = image
    if not np.array_equal(rendered[0.0], background):
        raise AssertionError("zero-alpha text changed the camera image")
    if np.array_equal(rendered[1.0], background):
        raise AssertionError("opaque text was not rendered")
    expected = cv2.addWeighted(rendered[1.0], 0.62, background, 0.38, 0.0)
    if not np.array_equal(rendered[0.62], expected):
        raise AssertionError("text or its outline did not use the configured opacity")


def check_slow_recorder_does_not_block_capture(temp_dir: str) -> None:
    metric_started = threading.Event()
    metric_release = threading.Event()
    producer_done = threading.Event()
    worker_threads = []
    producer_results = []

    class GatedRecorder(AsyncVideoRecorder):
        def _measure_sharpness(self, image):
            worker_threads.append(threading.current_thread().name)
            metric_started.set()
            if not metric_release.wait(timeout=5.0):
                raise AssertionError("test did not release the slow recorder")
            return super()._measure_sharpness(image)

    recorder = GatedRecorder(
        VideoRecorderConfig(
            output_path=str(Path(temp_dir) / "slow_recorder.avi"),
            fps=30.0, queue_capacity=1, overlay_wait_sec=0.0,
        ),
        cv2_module=cv2,
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    def produce() -> None:
        try:
            producer_results.append(recorder.submit(frame, capture_frame_id=2))
            producer_results.append(recorder.update_overlay(2, ()))
            producer_results.append(recorder.submit(frame, capture_frame_id=3))
        finally:
            producer_done.set()

    producer = threading.Thread(target=produce, name="test-camera-producer")
    try:
        if not recorder.submit(frame, capture_frame_id=1):
            raise AssertionError("first frame was not queued")
        if not metric_started.wait(timeout=3.0):
            raise AssertionError("background sharpness calculation did not start")
        producer.start()
        if not producer_done.wait(timeout=1.0):
            raise AssertionError("capture/metadata submission blocked on the recorder")
        if producer_results != [True, True, False]:
            raise AssertionError(f"unexpected full-queue behavior: {producer_results}")
    finally:
        metric_release.set()
        if producer.ident is not None:
            producer.join(timeout=3.0)
        recorder.close(timeout_sec=3.0)
    if recorder.error is not None or recorder.written_frames != 2 or recorder.dropped_frames != 1:
        raise AssertionError("slow recording did not drain the queue safely")
    if worker_threads != ["camera-video-recorder", "camera-video-recorder"]:
        raise AssertionError(f"sharpness was calculated on the wrong thread: {worker_threads}")


def main() -> int:
    check_translucent_text()
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

        expected_frames = 7
        for frame_index in range(1, expected_frames + 1):
            frame = np.full((120, 160, 3), frame_index * 20, dtype=np.uint8)
            if not recorder.submit(
                frame,
                capture_frame_id=frame_index,
                monotonic_sec=100.0 + frame_index / 30.0,
                unix_sec=200.0 + frame_index / 30.0,
            ):
                raise AssertionError(f"frame {frame_index} was not queued")
            if frame_index == expected_frames:
                continue
            recorder.update_overlay(
                frame_index,
                [SimpleNamespace(bbox=(20, 10, 80, 100), score=0.9, class_id=0)],
                control=VideoControlOverlay(
                    control_frame_id=100 + frame_index,
                    active_target_id=1,
                    selected_target_id=1,
                    action_name="rotate_left",
                    requested_rpm=-6,
                    target_distance_m=2.35,
                    distance_source="vision_depth",
                    distance_detail="filtered",
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
        if rows[0]["target_distance_m"] != "2.350":
            raise AssertionError(f"target distance was not indexed: {rows[0]}")
        if rows[0]["distance_source"] != "vision_depth":
            raise AssertionError(f"distance source was not indexed: {rows[0]}")
        if any(row.get("sharpness") != "0.000" for row in rows):
            raise AssertionError("clarity must be measured on raw frames, including frames without control metadata")
        if rows[-1]["control_frame_id"] != "":
            raise AssertionError("a camera-only frame reused an older control frame")

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
                    target_distance_m=1.85,
                    distance_source="vision_depth",
                    distance_detail="filtered",
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

        # A high-contrast edge image must produce a visible sharpness metric
        # while keeping the direct annotation API source-compatible.
        edge_frame = np.zeros((120, 160, 3), dtype=np.uint8)
        edge_frame[:, 80:] = 255
        measured = recorder._measure_sharpness(edge_frame)
        if measured is None or measured <= 0.0:
            raise AssertionError(f"sharpness measurement failed: {measured}")
        blurred = recorder._measure_sharpness(cv2.GaussianBlur(edge_frame, (11, 11), 3.0))
        if blurred is None or not (0.0 <= blurred < measured):
            raise AssertionError(f"blur should reduce edge sharpness: {measured} -> {blurred}")
        annotated = recorder._annotate_frame(
            edge_frame,
            8,
            86,
            VideoFrameOverlay(),
            sharpness=measured,
        )
        if annotated.shape != edge_frame.shape:
            raise AssertionError("clarity annotation changed frame dimensions")
        check_slow_recorder_does_not_block_capture(temp_dir)

    print("video_recorder_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
