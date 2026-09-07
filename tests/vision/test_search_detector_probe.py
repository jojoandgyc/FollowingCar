#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rk_vision import yolo11
from rk_vision.pipeline import RKNNVisionPipeline, SearchCandidateEvidence


class FakeSession:
    def inference(self, _inputs, data_format=None):
        return [object()]


def main() -> int:
    config = yolo11.YOLO11Config(
        model_path="unused.rknn",
        conf_threshold=0.25,
        search_diagnostic_conf_threshold=0.10,
    )
    detector = yolo11.YOLO11RKNNDetector.__new__(yolo11.YOLO11RKNNDetector)
    detector.config = config
    detector.session = FakeSession()
    detector.search_diagnostic_active = True
    detector.last_search_diagnostic_detections = []
    detector.last_timing_ms = {}

    low_person = yolo11.Detection((10.0, 20.0, 80.0, 200.0), 0.15, 0)
    formal_person = yolo11.Detection((100.0, 20.0, 180.0, 220.0), 0.80, 0)
    other_class = yolo11.Detection((200.0, 20.0, 280.0, 220.0), 0.20, 1)
    observed_thresholds = []

    def fake_decode(_outputs, decode_config, _letterbox):
        observed_thresholds.append(float(decode_config.conf_threshold))
        return [low_person, formal_person, other_class]

    with mock.patch.object(
        yolo11,
        "prepare_yolo_input",
        return_value=(object(), SimpleNamespace()),
    ), mock.patch.object(yolo11, "_decode_yolo11_outputs", side_effect=fake_decode), mock.patch.object(
        yolo11,
        "_nms",
        side_effect=lambda detections, _threshold: list(detections),
    ):
        formal_output = detector.detect(object(), "BGR")

    if observed_thresholds != [0.10]:
        raise AssertionError(f"search probe did not use low threshold: {observed_thresholds}")
    if formal_output != [formal_person]:
        raise AssertionError(f"low-confidence probe leaked into formal output: {formal_output}")
    if detector.last_search_diagnostic_detections != [low_person, formal_person]:
        raise AssertionError(
            "search probe did not retain diagnostic candidates: "
            f"{detector.last_search_diagnostic_detections}"
        )

    detector.set_search_diagnostic_active(False)
    if detector.last_search_diagnostic_detections:
        raise AssertionError("probe candidates were not cleared after search")

    context_calls = []
    pipeline = RKNNVisionPipeline.__new__(RKNNVisionPipeline)
    pipeline.last_search_candidate_evidence = SearchCandidateEvidence(
        formal_persons=(formal_person,),
        probe_persons=(low_person,),
    )
    pipeline.detector = SimpleNamespace(
        set_search_diagnostic_active=lambda active: context_calls.append(("detector", active))
    )
    pipeline.tracker = SimpleNamespace(
        set_search_reacquire_context=lambda **kwargs: context_calls.append(("tracker", kwargs))
    )
    pipeline.set_identity_reacquire_context(active_uid=7, searching=True, direction="left")
    if context_calls != [
        ("tracker", {"active_uid": 7, "searching": True, "direction": "left"}),
    ]:
        raise AssertionError(f"identity context leaked into detector: {context_calls}")
    context_calls.clear()
    pipeline.set_search_diagnostic_mode(True)
    if context_calls != [("detector", True)]:
        raise AssertionError(f"diagnostic mode leaked into tracker: {context_calls}")
    evidence = pipeline.get_search_candidate_evidence()
    if evidence.formal_persons != (formal_person,) or evidence.probe_persons != (low_person,):
        raise AssertionError(f"pipeline evidence boundary changed candidates: {evidence}")
    context_calls.clear()
    pipeline.set_search_reacquire_context(active_uid=7, searching=False, direction="right")
    if context_calls != [
        ("tracker", {"active_uid": 7, "searching": False, "direction": "right"}),
    ]:
        raise AssertionError(f"compatibility identity context crossed layers: {context_calls}")
    pipeline.set_search_diagnostic_mode(False)
    if pipeline.get_search_candidate_evidence() != SearchCandidateEvidence():
        raise AssertionError("candidate evidence was not cleared when search ended")

    detector_only_calls = []
    probe_pipeline = RKNNVisionPipeline.__new__(RKNNVisionPipeline)
    probe_pipeline.config = SimpleNamespace(person_class_id=0, conf_threshold=0.25)
    probe_pipeline.logger = None
    probe_pipeline.detector = SimpleNamespace(
        detect=lambda _packet, _fmt: detector_only_calls.append("detector")
        or [formal_person],
        last_search_diagnostic_detections=[low_person, formal_person],
        last_timing_ms={
            "preprocess": 1.0,
            "inference": 2.0,
            "decode": 3.0,
            "nms": 4.0,
            "postprocess": 7.0,
            "total": 10.0,
        },
    )
    probe_pipeline.reid = SimpleNamespace(
        extract=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("detector-only recovery called ReID")
        )
    )
    probe_pipeline.tracker = SimpleNamespace(
        update=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("detector-only recovery advanced tracker")
        )
    )
    probe_records = probe_pipeline.process_search_probe_frame(
        np.zeros((240, 320, 3), dtype=np.uint8),
        "BGR",
    )
    if probe_records:
        raise AssertionError(f"detector-only recovery emitted tracks: {probe_records}")
    if detector_only_calls != ["detector"]:
        raise AssertionError(f"detector-only recovery call boundary changed: {detector_only_calls}")
    probe_evidence = probe_pipeline.get_search_candidate_evidence()
    if probe_evidence.formal_persons != (formal_person,):
        raise AssertionError(f"formal probe evidence was lost: {probe_evidence}")
    if probe_evidence.probe_persons != (low_person,):
        raise AssertionError(f"low-confidence probe evidence was lost: {probe_evidence}")
    if probe_pipeline.last_timing_ms["reid_total"] != 0.0:
        raise AssertionError(f"detector-only recovery reported ReID time: {probe_pipeline.last_timing_ms}")
    if probe_pipeline.last_timing_ms["tracker"] != 0.0:
        raise AssertionError(f"detector-only recovery reported tracker time: {probe_pipeline.last_timing_ms}")
    print("search_detector_probe_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
