#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.search_diagnostics import (
    DirectionEvidenceObservation,
    MotionObservation,
    RecognitionObservation,
    SearchControlObservation,
    SearchDiagnosticSample,
    SearchDiagnosticsConfig,
    SearchDiagnosticsObserver,
    TransportObservation,
)


def main() -> int:
    source = (ROOT / "car_control_modular/search_diagnostics.py").read_text(encoding="utf-8")
    forbidden = ("ControlAction", "action_runtime", "mssd_motor", "steering_pid")
    present = [name for name in forbidden if name in source]
    if present:
        raise AssertionError(f"diagnostic module crossed control/motor boundary: {present}")

    cell = 16
    checker = np.indices((240, 320)).sum(axis=0) // cell
    crisp_gray = ((checker % 2) * 255).astype(np.uint8)
    crisp = cv2.cvtColor(crisp_gray, cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(crisp, (31, 31), 0)

    with tempfile.TemporaryDirectory() as temp_dir:
        observer = SearchDiagnosticsObserver(
            SearchDiagnosticsConfig(output_dir=temp_dir),
            mock.Mock(),
        )
        crisp_quality = observer.measure_frame_quality(
            crisp,
            "BGR",
            update_low_yaw_baseline=True,
        )
        blurred_quality = observer.measure_frame_quality(
            blurred,
            "BGR",
            update_low_yaw_baseline=False,
        )
        if crisp_quality is None or blurred_quality is None:
            raise AssertionError("frame quality metrics unexpectedly unavailable")
        if crisp_quality.sharpness <= blurred_quality.sharpness * 5.0:
            raise AssertionError(
                "sharpness metric did not separate frames: "
                f"crisp={crisp_quality} blurred={blurred_quality}"
            )
        if blurred_quality.state != "blur_suspected":
            raise AssertionError(f"blur was not classified: {blurred_quality}")

        logger = mock.Mock()
        observer = SearchDiagnosticsObserver(
            SearchDiagnosticsConfig(output_dir=temp_dir),
            logger,
        )
        observer.measure_frame_quality(crisp, "BGR", update_low_yaw_baseline=True)
        quality = observer.measure_frame_quality(blurred, "BGR", update_low_yaw_baseline=False)
        observer.remember_direction_frame(38, crisp, "BGR")
        base = dict(
            frame_index=42,
            timestamp=100.0,
            width=320,
            height=240,
            recognition=RecognitionObservation(
                yolo_total_ms=20.0,
                yolo_inference_ms=15.0,
                yolo_nms_ms=1.0,
                tracker_ms=0.5,
            ),
            motion=MotionObservation(command_name="rotate_left", requested_rotate_raw=10),
            transport=TransportObservation(
                frame_gap_ms=50.0,
                camera_read_ms=2.0,
                result_age_ms=30.0,
            ),
            quality=quality,
            direction_evidence=(
                DirectionEvidenceObservation(
                    frame_index=38,
                    target_id=7,
                    bbox=(20.0, 30.0, 120.0, 220.0),
                    x_ratio=0.219,
                    area=19000.0,
                    distance_m=1.8,
                ),
            ),
            image_frame=blurred,
            frame_format="BGR",
        )
        first = SearchDiagnosticSample(
            **base,
            control=SearchControlObservation(
                state_before="searching",
                direction_before="left",
                state_after="searching",
                direction_after="left",
                active_target_id=7,
                progress_deg=45.0,
            ),
        )
        if observer.observe(first) is not None:
            raise AssertionError("passive observer returned a control value")
        snapshots = list(Path(temp_dir).rglob("*.jpg"))
        if len(snapshots) != 3:
            raise AssertionError(
                "expected current decision, evidence, and checkpoint snapshots, "
                f"got {snapshots}"
            )
        snapshot_names = {item.name for item in snapshots}
        if not any(name.startswith("decision_current_") for name in snapshot_names):
            raise AssertionError(f"missing current decision snapshot: {snapshot_names}")
        if not any(name.startswith("decision_evidence_") for name in snapshot_names):
            raise AssertionError(f"missing direction evidence snapshot: {snapshot_names}")

        second = SearchDiagnosticSample(
            **{**base, "timestamp": 101.0, "image_frame": None},
            control=SearchControlObservation(
                state_before="searching",
                direction_before="left",
                state_after="timed_out",
                decision_reason="search_revolution_complete",
                active_target_id=7,
                # The controller clears progress when it leaves search. The
                # observer must preserve the last non-zero value by itself.
                progress_deg=0.0,
            ),
        )
        observer.observe(second)
        rendered = [
            call.args[0] % tuple(call.args[1:])
            for call in logger.info.call_args_list
            if call.args
        ]
        for prefix in ("search_vision_diag", "search_control_diag", "search_motion_diag"):
            count = sum(line.startswith(prefix) for line in rendered)
            if count != 2:
                raise AssertionError(f"expected two {prefix} logs, got {count}: {rendered}")
        if not any(line.startswith("search_decision_frame") for line in rendered):
            raise AssertionError(f"missing search decision frame log: {rendered}")
        evidence_logs = [line for line in rendered if line.startswith("search_decision_evidence")]
        if len(evidence_logs) != 1 or "frame=38" not in evidence_logs[0]:
            raise AssertionError(f"unexpected decision evidence logs: {evidence_logs}")
        summaries = [line for line in rendered if line.startswith("search_session_summary")]
        if len(summaries) != 1 or "outcome=search_revolution_complete" not in summaries[0]:
            raise AssertionError(f"unexpected search summary: {summaries}")
        if "scan=45.0deg" not in summaries[0]:
            raise AssertionError(f"search summary lost pre-reset scan progress: {summaries}")
        if observer.active:
            raise AssertionError("diagnostic observer remained active after completion")

    print(
        "search_frame_quality_ok",
        {
            "crisp_sharpness": round(crisp_quality.sharpness, 2),
            "blurred_sharpness": round(blurred_quality.sharpness, 2),
            "layers": "vision/control/motion",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
