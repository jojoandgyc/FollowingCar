"""Read-only crop mixing metadata; no device or identity-policy shortcuts."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import os

import cv2
import numpy as np
import pytest

from car_control_modular.config_loader import load_config_to_env
from rk_vision.pipeline import RKNNVisionConfig, RKNNVisionPipeline
from rk_vision.reid_diagnostics import ReIDDiagnosticsWriter, template_learning_labels
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig, _template_learning_risk
from rk_vision.yolo11 import Detection


TARGET = Detection((100., 20., 300., 420.), .95, 0)
OCCLUDER = Detection((250., 80., 380., 400.), .40, 0)


def test_small_excluded_person_still_marks_mixed_crop_without_reid():
    small = Detection((240., 220., 280., 260.), .31, 0)
    result = _template_learning_risk([TARGET], 0, is_fresh=True,
                                     all_detections=[small, TARGET])
    assert result == dict(observed=True, risky=True, reason="overlapping_person_crop",
                          overlap=1., other_indices=[0])


@pytest.mark.parametrize("overlap,risky", [(0., False), (14., False), (15., True), (80., True)])
def test_learning_overlap_threshold_does_not_change_detector_eligibility(overlap, risky):
    target = Detection((0., 0., 100., 100.), .95, 0)
    other = Detection((100.-overlap, 0., 200.-overlap, 100.), .25, 0)
    result = _template_learning_risk([target, other], 0, is_fresh=True)
    assert result["observed"] and result["risky"] is risky
    assert result["overlap"] == pytest.approx(overlap/100.)


@pytest.mark.parametrize("index,fresh", [(None, True), (True, True), (-1, True), (2, True), (0., True), (0, False)])
def test_non_current_sources_cannot_claim_clear_crop(index, fresh):
    result = _template_learning_risk([TARGET, OCCLUDER], index, is_fresh=fresh)
    assert not result["observed"] and not result["risky"]


@pytest.mark.parametrize("bbox", [(0., 0., 0., 10.), (float("nan"), 0., 1., 1.), (10., 0., 1., 1.)])
def test_invalid_geometry_cannot_be_learning_evidence(bbox):
    result = _template_learning_risk([Detection(bbox, .9, 0)], 0, is_fresh=True)
    assert not result["observed"]


def test_nonpersons_are_not_occlusion_evidence_and_stale_snapshots_do_not_count():
    other = Detection(TARGET.bbox, .99, 2)
    assert not _template_learning_risk([TARGET, other], 0, is_fresh=True)["risky"]
    assert not _template_learning_risk([TARGET], 0, is_fresh=True,
                                       all_detections=[OCCLUDER])["observed"]


def test_formal_track_forwards_risk_without_modifying_uid_or_competition(monkeypatch):
    tracker = DeepSortTracker(DeepSortTrackerConfig(identity_template_learning_guard_enable=True))
    tracker._current_detections = (TARGET,)
    tracker._learning_detections = (TARGET, OCCLUDER)
    tracker._frame_context = {"capture_frame_id": 1044, "capture_timestamp": 5.,
                              "template_learning_risk": {"observed": True, "risky": False}}
    tracker._identity_competition = {0: {"ok": True, "reason": "excluded_background"}}
    received = []
    def assign(**kwargs):
        received.append(kwargs)
        return 1
    monkeypatch.setattr(tracker.identity_bank, "assign", assign)
    output = SimpleNamespace(track_id=3, source_detection_index=0, x1=100., y1=20.,
        x2=300., y2=420., confidence=.95, feature=np.array([1., 0.]),
        time_since_update=0, class_id=0, state=2)
    result = tracker._to_record(output, 640, 480, 1, partial_features=[None])
    assert result.reid_uid == 1
    metadata = received[0]["sample_metadata"]
    assert metadata["template_learning_risk"]["risky"]
    assert metadata["template_learning_risk"]["other_indices"] == [1]
    assert metadata["identity_competition"] == {"ok": True, "reason": "excluded_background"}
    assert received[0]["bbox_quality_ok"]


def test_probe_uses_current_detector_risk_not_previous_tracker_snapshot(monkeypatch):
    tracker = DeepSortTracker(DeepSortTrackerConfig())
    tracker.set_search_reacquire_context(active_uid=1, searching=True, direction="left")
    tracker._current_detections = (OCCLUDER,)
    tracker._learning_detections = (OCCLUDER, TARGET)
    received = []
    monkeypatch.setattr(tracker, "_frame_identity_competition", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(tracker.identity_bank, "observe_frame_evidence", lambda **_kwargs: None)
    monkeypatch.setattr(tracker.identity_bank, "assign", lambda **kw: received.append(kw) or 0)
    tracker._search_probe_record([TARGET], [np.array([1., 0.])],
        partial_features=[None], image_width=640, image_height=480)
    assert received[0]["sample_metadata"]["template_learning_risk"] == dict(
        observed=True, risky=False, reason="clear", overlap=0., other_indices=[])


@pytest.mark.parametrize("status,expected", [("pending", "learning_pending"),
    ("frozen", "learning_frozen"), ("rejected", "learning_rejected"),
    ("not_requested", "observation_only")])
def test_diagnostics_distinguish_observation_from_learning(status, expected):
    labels = template_learning_labels({"uid": 1, "assignment": {"bank_updated": False,
        "template_learning": {"status": status, "reason": "test_reason"}}})
    assert labels["learning_status"] == expected
    assert labels["learning_reason"] == "test_reason"
    assert labels["learning_written_tiers"] == []


@pytest.mark.parametrize("field", ["bank_updated", "recent_bank_updated"])
def test_actual_commit_is_labelled_stored_even_if_followup_learning_frozen(field):
    labels = template_learning_labels({"uid": 1, "assignment": {field: True,
        "learning_written_tiers": ["full_recent", "partial_recent"],
        "template_learning": {"status": "frozen", "reason": "later_stage"}}})
    assert labels["learning_status"] == "stored"
    assert labels["learning_written_tiers"] == ["full_recent", "partial_recent"]


def test_uid_rejection_is_not_reported_as_template_commit():
    assert template_learning_labels({"uid": 0, "assignment": {
        "reason": "secondary_evidence_unavailable", "bank_updated": False}})["learning_status"] == "identity_rejected"


def test_diagnostics_keep_raw_png_and_write_labels_in_existing_jsonl(tmp_path):
    frame = np.arange(6*8*3, dtype=np.uint8).reshape((6, 8, 3))
    writer = ReIDDiagnosticsWriter(tmp_path)
    path = writer.submit(frame, (0, 0, 8, 6), {"uid": 1,
        "assignment": {"template_learning": {"status": "frozen", "reason": "overlap"}}})
    assert path and writer.close()
    assert path.endswith("_FROZEN.png")
    records = [json.loads(x) for x in writer.index_path.read_text().splitlines()]
    assert len(records) == 1 and records[0]["learning_status"] == "learning_frozen"
    assert np.array_equal(cv2.imread(str(tmp_path/path)), frame)
    assert len(list(tmp_path.iterdir())) == 2


def test_pending_reset_and_write_throttle_evidence_survive_diagnostic_export(tmp_path):
    learning = dict(status="pending", reason="commit_throttled", confirmations=3,
                    pending_cap=827, pending_reset_reason="pending_full_distance",
                    pending_reset_cap=825, pending_reset_count=2,
                    pending_full_distance=.04, pending_partial_distance=.03,
                    parent_caps=[376])
    writer = ReIDDiagnosticsWriter(tmp_path)
    writer.submit(np.zeros((6, 8, 3), dtype=np.uint8), (0, 0, 8, 6),
                  {"uid": 1, "assignment": {"bank_updated": False, "template_learning": learning}})
    assert writer.close()
    record = json.loads(writer.index_path.read_text().strip())
    assert record["assignment"]["template_learning"] == learning
    assert record["learning_status"] == "learning_pending"
    assert record["learning_reason"] == "commit_throttled"
    assert record["learning_written_tiers"] == []


@pytest.mark.parametrize("assignment,label", [({"bank_updated": True}, "STORED"),
    ({"template_learning": {"status": "pending"}}, "PENDING"),
    ({"template_learning": {"status": "frozen"}}, "FROZEN"),
    ({"template_learning": {"status": "rejected"}}, "REJECTED"), ({}, "OBSERVE")])
def test_filename_status_suffix_preserves_replay_glob_and_exact_crop(tmp_path, assignment, label):
    pixels = np.arange(6*8*3, dtype=np.uint8).reshape((6, 8, 3))
    writer = ReIDDiagnosticsWriter(tmp_path)
    path = writer.submit(pixels, (0, 0, 8, 6), {"uid": 1, "capture_frame_id": 1044,
        "control_frame_id": 445, "raw_track_id": 3, "assignment": assignment})
    assert writer.close() and path.endswith("_" + label + ".png")
    # tools/audit_reid_crops.py discovers files by this stable CAP/track glob.
    assert list(tmp_path.glob("*_capture_00001044_track_*.png")) == [tmp_path/path]
    assert writer.sample_path(445, 3) == path
    assert np.array_equal(cv2.imread(str(tmp_path/path)), pixels)


def test_runtime_config_flag_propagates_to_real_pipeline_tracker_bank(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    load_config_to_env(str(Path(__file__).resolve().parents[2] / "car_control_modular/config/reid_runtime.ini"))
    config = RKNNVisionConfig.from_env()
    assert config.identity_template_learning_guard_enable
    config = replace(config, backend="mock", reid_diagnostics_enable=False)
    pipeline = RKNNVisionPipeline(config)
    try:
        assert pipeline.tracker.config.identity_template_learning_guard_enable
        assert pipeline.tracker.identity_bank.config.template_learning_guard_enable
    finally:
        pipeline.close()


def test_learning_guard_remains_opt_in_for_legacy_config(monkeypatch):
    monkeypatch.delenv("Y8_IDENTITY_TEMPLATE_LEARNING_GUARD_ENABLE", raising=False)
    assert not RKNNVisionConfig.from_env().identity_template_learning_guard_enable
    assert not DeepSortTrackerConfig().identity_template_learning_guard_enable


def test_pipeline_learning_snapshot_keeps_size_rejected_person_without_extracting_it(monkeypatch):
    pipeline = RKNNVisionPipeline(RKNNVisionConfig(yolo_model_path="unused", reid_model_path="unused",
        backend="mock", identity_template_learning_guard_enable=True,
        identity_min_width_px=64., identity_min_height_px=100.))
    small = Detection((240., 220., 280., 260.), .70, 0)
    batches, received = [], []
    monkeypatch.setattr(pipeline.detector, "detect", lambda *_: [small, TARGET])
    def extract(_packet, persons, _fmt):
        batches.append(persons)
        return [np.array([1., 0.]) for _ in persons]
    monkeypatch.setattr(pipeline.reid, "extract", extract)
    monkeypatch.setattr(pipeline.tracker, "update", lambda *args, **kwargs: received.append((args, kwargs)) or [])
    try:
        pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        assert batches == [[TARGET]]
        assert received[0][0][0] == [TARGET]
        assert received[0][1]["learning_detections"] == [small, TARGET]
    finally:
        pipeline.close()


@pytest.mark.parametrize("update", [{"recent_bank_updated": True},
    {"learning_written_tiers": ["full_recent"]}, {"bank_updated": True}, {}])
def test_real_template_commits_are_not_lost_to_diagnostic_observation_throttle(update):
    pipeline = RKNNVisionPipeline.__new__(RKNNVisionPipeline)
    pipeline.config = RKNNVisionConfig(yolo_model_path="unused", reid_diagnostics_mapped_interval=30)
    pipeline._frame_context = {}
    written = []
    pipeline._reid_diagnostics = SimpleNamespace(sample_path=lambda *_: None,
        submit=lambda *args, **kwargs: written.append((args, kwargs)))
    pipeline.tracker = SimpleNamespace(last_identity_observations=[{
        "frame_index": 31, "detector_bbox": TARGET.bbox, "uid": 1,
        "assignment": {"reason": "mapped", "bank_updated": False, **update}}])
    pipeline._record_reid_diagnostics(None, "BGR")
    assert len(written) == bool(update)
