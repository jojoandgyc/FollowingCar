from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rk_vision.identity_exclusion import IdentityExclusionMemory
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection


TARGET = (361.644867, 2.539612, 637.908325, 474.887024)
CANDIDATE = (211.983749, 163.882309, 310.605469, 265.126831)
FEATURE = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


def output(track_id, source_index, bbox, *, age=0):
    x1, y1, x2, y2 = bbox
    return SimpleNamespace(
        track_id=track_id, source_detection_index=source_index,
        x1=x1, y1=y1, x2=x2, y2=y2, class_id=0, confidence=0.9,
        feature=FEATURE if age == 0 else None, time_since_update=age, state=2,
    )


def make_tracker():
    tracker = DeepSortTracker(DeepSortTrackerConfig(identity_min_confidence=0.60))
    tracker.identity_bank.track_to_uid[26] = 1
    tracker._frame_index = 581
    return tracker


@pytest.mark.parametrize("reverse", [False, True])
def test_entire_capture_is_registered_before_any_assignment_regardless_of_output_order(monkeypatch, reverse):
    tracker = make_tracker()
    memory = IdentityExclusionMemory()
    events = []
    observations = []

    def register(**kwargs):
        events.append("observe")
        observations.extend(kwargs["observations"])
        # Source appearance validation belongs to IdentityBank. This test
        # checks the tracker's pre-assignment old-mapping snapshot contract.
        kwargs["observations"] = [
            {**item, "trusted_uid": item["mapped_uid"]} for item in kwargs["observations"]
        ]
        memory.observe_frame(**kwargs)

    def assign_record(out, *args, **kwargs):
        events.append(out.track_id)
        if out.track_id == 35:
            assert memory.exclusion_for(35, 1, frame_index=582) is not None
        # Even a prior assignment mutating the mapping cannot retroactively
        # change the frame's trusted-source snapshot.
        tracker.identity_bank.track_to_uid.clear()
        return SimpleNamespace(track_id=out.track_id)

    monkeypatch.setattr(tracker.identity_bank, "observe_frame_evidence", register, raising=False)
    monkeypatch.setattr(tracker, "_to_record", assign_record)
    outputs = [
        output(26, 0, (328.375, 0, 639, 479)),
        output(35, 1, (202.353, 154.056, 322.065, 276.437)),
    ]
    monkeypatch.setattr(tracker.deepsort, "update", lambda *args, **kwargs: outputs[::-1] if reverse else outputs)
    records = tracker.update(
        [Detection(TARGET, 0.89666, 0), Detection(CANDIDATE, 0.70834, 0)],
        [FEATURE, FEATURE], image_width=640, image_height=480,
        frame_context={"capture_frame_id": 1550, "capture_timestamp": 10.0, "integrated_yaw_deg": 1.6272},
    )
    assert len(records) == 2
    assert events[0] == "observe" and len(events) == 3
    source = next(item for item in observations if item["raw_track_id"] == 26)
    assert source["mapped_uid"] == 1
    assert source["detector_bbox"] == TARGET
    assert source["capture_frame_id"] == 1550
    assert source["confidence"] == 0.89666
    assert source["feature"] is FEATURE
    assert source["is_fresh"] is True
    assert source["identity_swap"] is False


def test_predicted_or_provenance_free_outputs_cannot_be_covisibility_sources(monkeypatch):
    tracker = make_tracker()
    seen = []
    monkeypatch.setattr(tracker.identity_bank, "observe_frame_evidence", lambda **kwargs: seen.extend(kwargs["observations"]), raising=False)
    tracker._current_detections = (Detection(TARGET, 0.9, 0),)
    tracker._observe_identity_frame_evidence(
        [output(26, 0, TARGET, age=1), output(35, None, CANDIDATE), output(36, 4, CANDIDATE)],
        640, 480,
    )
    assert seen == []


def test_duplicate_and_identity_jump_flags_are_registered_without_mutating_center_history(monkeypatch):
    tracker = make_tracker()
    tracker._frame_index = 582
    tracker._current_detections = (Detection(TARGET, 0.9, 0), Detection(CANDIDATE, 0.9, 0))
    tracker._last_identity_center_by_track_id[26] = 0.1
    tracker._last_identity_center_frame_by_track_id[26] = 581
    seen = []
    monkeypatch.setattr(tracker.identity_bank, "observe_frame_evidence", lambda **kwargs: seen.extend(kwargs["observations"]), raising=False)
    tracker._observe_identity_frame_evidence(
        [output(26, 0, TARGET), output(35, 1, CANDIDATE)], 640, 480,
        duplicate_track_ids={35},
    )
    assert seen[0]["identity_swap"] is True
    assert seen[1]["duplicate"] is True
    assert tracker._last_identity_center_by_track_id[26] == 0.1


def test_empty_capture_registers_an_empty_snapshot_without_prediction_evidence(monkeypatch):
    tracker = make_tracker()
    seen = []
    monkeypatch.setattr(tracker.identity_bank, "observe_frame_evidence", lambda **kwargs: seen.append(kwargs), raising=False)
    monkeypatch.setattr(tracker.deepsort, "update", lambda *args, **kwargs: [])
    assert tracker.update([], [], image_width=640, image_height=480) == []
    assert len(seen) == 1 and seen[0]["observations"] == []


def test_probe_registers_real_detector_observation_and_inherits_before_assign(monkeypatch):
    tracker = make_tracker()
    memory = IdentityExclusionMemory()
    memory.observe_frame(frame_index=582, width=640, height=480, observations=[
        {"raw_track_id": 26, "detector_bbox": TARGET, "trusted_uid": 1,
         "capture_frame_id": 1550, "capture_timestamp": 10.0, "is_fresh": True},
        {"raw_track_id": 35, "detector_bbox": CANDIDATE, "trusted_uid": 0,
         "capture_frame_id": 1550, "capture_timestamp": 10.0, "is_fresh": True},
    ])
    tracker._frame_index = 583
    tracker._frame_context = {"capture_frame_id": 1554, "capture_timestamp": 10.1}
    tracker.set_search_reacquire_context(active_uid=1, searching=True, direction="right")
    events = []

    def register(**kwargs):
        events.append("observe")
        assert kwargs["observations"][0]["raw_track_id"] == -1
        assert kwargs["observations"][0]["detector_bbox"] == CANDIDATE
        memory.observe_frame(**kwargs)

    def assign(**kwargs):
        events.append("assign")
        assert memory.exclusion_for(-1, 1, frame_index=583) is not None
        tracker.identity_bank.last_assignments[-1] = {"uid": 0, "reason": "co_visible_distinct_person"}
        return 0

    monkeypatch.setattr(tracker.identity_bank, "observe_frame_evidence", register, raising=False)
    monkeypatch.setattr(tracker.identity_bank, "assign", assign)
    record = tracker._search_probe_record(
        [Detection(CANDIDATE, 0.70834, 0)], [FEATURE], partial_features=[None],
        image_width=640, image_height=480,
    )
    assert events == ["observe", "assign"]
    assert record is not None and record.reid_uid == 0


def test_legacy_bank_without_observer_remains_supported():
    tracker = make_tracker()
    tracker.identity_bank = SimpleNamespace()
    tracker._observe_identity_frame_evidence([], 640, 480)
    tracker.identity_bank.observe_frame_evidence = False
    tracker._observe_identity_frame_evidence([], 640, 480)
