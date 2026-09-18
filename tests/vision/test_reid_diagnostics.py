import json
import logging
import threading

import cv2
import numpy as np
import pytest

from rk_vision import reid_diagnostics
from rk_vision.frames import FramePacket
from rk_vision.reid import OSNetConfig, OSNetRKNNExtractor
from rk_vision.reid_diagnostics import ReIDDiagnosticsWriter, stage_crop


def _metadata(frame_id=11, track_id=3):
    return {
        "control_frame_id": frame_id,
        "capture_frame_id": frame_id + 20,
        "raw_track_id": track_id,
        "uid": 7,
        "assignment": {"reason": "created", "gallery_updated": True},
        "display_bbox": [0.0, 1.0, 5.0, 6.0],
    }


def _read_records(directory):
    return [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]


@pytest.mark.parametrize(
    "bbox",
    [(-9.5, -2.5, 7.5, 9.5), (1.5, 0.5, 5.5, 4.5), (20, 20, 30, 30), (4, 3, 2, 1)],
)
def test_stage_crop_matches_osnet_rounding_and_clamping(bbox):
    frame = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    extractor = OSNetRKNNExtractor(OSNetConfig(model_path="", enabled=False))
    expected = extractor._crop(frame, bbox)
    staged = stage_crop(frame, bbox)
    if expected is None:
        assert staged is None
        return
    assert staged is not None
    assert np.array_equal(staged.pixels, expected)
    assert not np.shares_memory(staged.pixels, frame)
    expected = expected.copy()
    frame[:] = 0
    assert np.array_equal(staged.pixels, expected)


def test_saved_crop_keeps_evidence_and_source_template_path(tmp_path):
    writer = ReIDDiagnosticsWriter(tmp_path)
    frame = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    bbox = (-0.4, 1.4, 4.6, 10.0)
    metadata = _metadata()
    first_path = writer.submit(frame, bbox, metadata)
    assert first_path is not None
    assert "frame_00000011_capture_00000031_track_00000003" in first_path
    assert writer.sample_path(11, 3) == first_path
    metadata["assignment"]["reason"] = "changed_after_submit"
    second_metadata = _metadata(12, 4)
    second_metadata["assignment"] = {"reason": "handoff", "match_evidence": {
        "distance": 0.037,
        "source_frame_index": 11,
        "source_track_id": 3,
        "sample_path": writer.sample_path(11, 3),
    }}
    second_path = writer.submit(frame, (2, 2, 6, 5), second_metadata)
    assert second_path is not None
    frame[:] = 0
    assert writer.close(timeout_sec=2.0)
    records = _read_records(tmp_path)
    assert len(records) == writer.written_samples == writer.accepted_samples == 2
    assert writer.failed_samples == writer.dropped_samples == 0
    assert records[0]["raw_detector_bbox"] == list(bbox)
    assert records[0]["crop_bbox"] == [0, 1, 5, 6]
    assert records[0]["display_bbox"] == metadata["display_bbox"]
    assert records[0]["assignment"]["reason"] == "created"
    assert records[1]["assignment"]["match_evidence"]["sample_path"] == first_path
    original = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    assert np.array_equal(cv2.imread(str(tmp_path / first_path)), original[1:6, 0:5])
    assert writer._thread is not None and not writer._thread.is_alive()


def test_metadata_uses_strict_json_and_snapshots_numpy_values(tmp_path):
    writer = ReIDDiagnosticsWriter(tmp_path)
    metadata = _metadata(np.int64(9), np.int32(2))
    metadata["match_evidence"] = {
        "nan": float("nan"), "inf": np.float32("inf"), "negative_inf": -float("inf"),
        "scores": np.array([0.037, float("nan")]), "valid": np.bool_(True),
        "unsupported": object(),
    }
    assert writer.submit(np.zeros((6, 8, 3), dtype=np.uint8), (0, 0, 4, 5), metadata)
    metadata["match_evidence"]["scores"][:] = 99
    assert writer.close()
    raw = (tmp_path / "events.jsonl").read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    record = _read_records(tmp_path)[0]
    assert record["control_frame_id"] == 9
    assert record["raw_track_id"] == 2
    assert record["match_evidence"] == {
        "nan": None, "inf": None, "negative_inf": None,
        "scores": [0.037, None], "valid": True, "unsupported": None,
    }


def test_rgb_frame_packet_is_saved_with_correct_colors(tmp_path):
    writer = ReIDDiagnosticsWriter(tmp_path)
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    frame[:] = [220, 50, 10]
    path = writer.submit(FramePacket(frame, format="RGB"), (0, 0, 4, 5), _metadata())
    assert path is not None
    assert writer.close()
    assert np.all(cv2.imread(str(tmp_path / path)) == [10, 50, 220])
    assert _read_records(tmp_path)[0]["source_format"] == "RGB"


def test_disabled_writer_and_zero_cap_do_not_create_output(tmp_path):
    for name, options in [("disabled", {"enabled": False}), ("zero", {"max_samples": 0})]:
        directory = tmp_path / name
        writer = ReIDDiagnosticsWriter(directory, **options)
        assert writer.submit(object(), (), _metadata()) is None
        assert writer.close()
        assert not directory.exists()
        assert writer.accepted_samples == 0


def test_full_queue_drops_without_waiting_and_close_flushes(tmp_path, monkeypatch, caplog):
    started = threading.Event()
    release = threading.Event()
    original_write = reid_diagnostics._write_png

    def gated_write(path, crop):
        started.set()
        assert release.wait(timeout=3.0)
        original_write(path, crop)

    monkeypatch.setattr(reid_diagnostics, "_write_png", gated_write)
    writer = ReIDDiagnosticsWriter(tmp_path, queue_capacity=1)
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    try:
        assert writer.submit(frame, (0, 0, 4, 5), _metadata())
        assert started.wait(timeout=2.0)
        assert writer.submit(frame, (0, 0, 4, 5), _metadata(12))
        assert writer.submit(frame, (0, 0, 4, 5), _metadata(13)) is None
        assert writer.accepted_samples == 2
        assert writer.dropped_samples == 1
        with caplog.at_level(logging.WARNING):
            assert not writer.close(timeout_sec=0.0)
        assert "close timed out" in caplog.text
        assert writer.submit(frame, (0, 0, 4, 5), _metadata(14)) is None
    finally:
        release.set()
        assert writer.close(timeout_sec=2.0)
    assert writer.written_samples == 2
    assert len(_read_records(tmp_path)) == 2
    assert writer.close(timeout_sec=0.0)


def test_total_sample_cap_bounds_files(tmp_path):
    writer = ReIDDiagnosticsWriter(tmp_path, max_samples=2, queue_capacity=8)
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    paths = [writer.submit(frame, (0, 0, 4, 5), _metadata(index)) for index in range(7)]
    assert writer.close()
    assert sum(path is not None for path in paths) == 2
    assert len(list(tmp_path.glob("*.png"))) == 2
    assert writer.accepted_samples == writer.written_samples == 2
    assert writer.dropped_samples == 5
    assert writer.sample_path(2, 3) is None


def test_save_failure_is_logged_and_next_sample_still_saves(tmp_path, monkeypatch, caplog):
    original_write = reid_diagnostics._write_png

    def fail_one(path, crop):
        if "frame_00000011_" in path.name:
            raise OSError("test disk failure")
        original_write(path, crop)

    monkeypatch.setattr(reid_diagnostics, "_write_png", fail_one)
    writer = ReIDDiagnosticsWriter(tmp_path)
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    with caplog.at_level(logging.WARNING):
        assert writer.submit(frame, (0, 0, 4, 5), _metadata())
        good_path = writer.submit(frame, (0, 0, 4, 5), _metadata(12))
        assert writer.close()
    assert writer.failed_samples == 1
    assert writer.written_samples == 1
    assert writer.error == "test disk failure"
    assert writer.sample_path(11, 3) is None
    assert writer.sample_path(12, 3) == good_path
    assert "test disk failure" in caplog.text
    assert len(_read_records(tmp_path)) == 1


def test_invalid_crop_or_metadata_never_escapes_to_control(tmp_path, caplog):
    writer = ReIDDiagnosticsWriter(tmp_path)
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    with caplog.at_level(logging.WARNING):
        assert writer.submit(frame, (float("nan"), 0, 4, 5), _metadata()) is None
        assert writer.submit(frame, (0, 0, 4, 5), object()) is None
        assert writer.submit_crop(None, _metadata()) is None
    assert writer.close()
    assert writer.dropped_samples == 3
    assert writer.accepted_samples == 0
    assert "diagnostic crop failed" in caplog.text
    assert "diagnostic submission failed" in caplog.text
    assert not (tmp_path / "events.jsonl").exists()


def test_encoder_failure_does_not_write_invalid_files(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(cv2, "imencode", lambda *_args: (False, None))
    writer = ReIDDiagnosticsWriter(tmp_path)
    with caplog.at_level(logging.WARNING):
        assert writer.submit(np.zeros((6, 8, 3), dtype=np.uint8), (0, 0, 4, 5), _metadata())
        assert writer.close()
    assert writer.failed_samples == 1
    assert writer.written_samples == 0
    assert not list(tmp_path.glob("*.png"))
    assert not (tmp_path / "events.jsonl").exists()
    assert "PNG encoder" in caplog.text


def test_existing_sample_is_not_overwritten(tmp_path):
    original = np.full((6, 8, 3), 20, dtype=np.uint8)
    first_writer = ReIDDiagnosticsWriter(tmp_path)
    path = first_writer.submit(original, (0, 0, 4, 5), _metadata())
    assert first_writer.close()
    second_writer = ReIDDiagnosticsWriter(tmp_path)
    assert second_writer.submit(np.zeros_like(original), (0, 0, 4, 5), _metadata()) == path
    assert second_writer.close()
    assert second_writer.failed_samples == 1
    assert second_writer.sample_path(11, 3) is None
    assert np.array_equal(cv2.imread(str(tmp_path / path)), original[:5, :4])
    assert len(_read_records(tmp_path)) == 1


def test_worker_start_failure_disables_diagnostics(tmp_path, monkeypatch, caplog):
    def fail_start(_thread):
        raise RuntimeError("test thread limit")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with caplog.at_level(logging.WARNING):
        writer = ReIDDiagnosticsWriter(tmp_path)
    assert not writer.enabled
    assert writer.error == "test thread limit"
    assert writer.submit(object(), (), _metadata()) is None
    assert writer.close()
    assert not (tmp_path / "events.jsonl").exists()
    assert "could not start" in caplog.text
