from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rk_vision.candidate_competition import competition_evidence
from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection


@pytest.mark.parametrize("distances,passed", [
    ({0: .100519, 1: .149289}, []),  # CAP1024: even d<.15 isn't a unique winner
    ({0: .177677, 1: .189031}, []),  # CAP1085: margin only .011354
    ({0: .1, 1: .3}, [0]),
    ({0: .1, 1: .15}, [0]),  # exact configured boundary, not float roundoff
    ({0: .1, 1: .1}, []),
    ({0: .1, 1: None}, []),
    ({0: .1, 1: float("nan")}, []),
    ({0: None}, [0]),  # single partial-person path is not disabled
])
def test_same_uid_person_margin(distances, passed):
    evidence = competition_evidence(distances, uid=1, frame_index=2)
    assert [i for i, item in evidence.items() if item["passed"]] == passed


def test_competition_is_order_independent():
    pairs = [(0, .1), (1, .14), (2, .3)]
    assert competition_evidence(dict(pairs), uid=1, frame_index=2) == competition_evidence(
        dict(reversed(pairs)), uid=1, frame_index=2)


def test_ambiguous_instant_match_cannot_bind_update_or_accumulate(caplog):
    bank = IdentityBank(IdentityBankConfig(new_identity_confirm_frames=1))
    feature = np.array([1., 0., 0.], dtype=np.float32)
    assert bank.assign(track_id=1, feature=feature, confidence=.9, area=30000, frame_index=1) == 1
    original_count = len(bank.identities[1].features)
    for frame in (2, 3):
        for pending in (bank.pending_new, bank.pending_handoffs,
                        bank.pending_late_handoffs, bank.pending_weak_handoffs):
            pending[2] = object()
        evidence = competition_evidence({0: .1, 1: .11}, uid=1, frame_index=frame)[0]
        with caplog.at_level("INFO", logger="PersonTracker"):
            uid = bank.assign(
                track_id=2, feature=feature, confidence=.99, area=30000,
                frame_index=frame, preferred_uid=1, preferred_candidate_ok=True,
                sample_metadata={"is_fresh": True, "search_reacquire_context_active": True,
                                 "identity_competition": evidence},
            )
        assert uid == 0
        assert bank.last_assignments[2]["reason"] == "search_candidate_identity_ambiguous"
        assert not bank.last_assignments[2]["bank_updated"]
        assert all(2 not in pending for pending in (bank.pending_new, bank.pending_handoffs,
                   bank.pending_late_handoffs, bank.pending_weak_handoffs))
    assert len(bank.identities[1].features) == original_count
    assert '"distance_gap"' in caplog.text
    assert '"identity_competition"' in caplog.text


def test_tracker_snapshots_all_detector_features_before_any_assignment(monkeypatch):
    tracker = DeepSortTracker(DeepSortTrackerConfig())
    tracker.set_search_reacquire_context(active_uid=1, searching=True, direction="left")
    detections = [Detection((60, 100, 160, 400), .9, 0),
                  Detection((400, 100, 500, 400), .8, 0)]
    features = [object(), object()]
    events = []
    def distance(uid, feature):
        events.append("distance")
        return .1 if feature is features[0] else .11
    monkeypatch.setattr(tracker, "reid_distance_to_uid", distance)
    outputs = [SimpleNamespace(track_id=i + 1, source_detection_index=i,
               time_since_update=0) for i in range(2)]
    monkeypatch.setattr(tracker.deepsort, "update", lambda *a, **k: outputs)
    monkeypatch.setattr(tracker, "_duplicate_identity_track_ids", lambda *a: set())
    monkeypatch.setattr(tracker, "_identity_swap_track_ids", lambda *a: set())
    monkeypatch.setattr(tracker, "_observe_identity_frame_evidence", lambda *a, **k: None)
    def record(out, *args, **kwargs):
        events.append("assign")
        assert not tracker._identity_competition[out.source_detection_index]["passed"]
        return out
    monkeypatch.setattr(tracker, "_to_record", record)
    tracker.update(detections, features, image_width=640, image_height=480)
    assert events == ["distance", "distance", "assign", "assign"]
    tracker.reset()
    assert tracker._identity_competition == {}


def test_normal_tracking_does_not_add_competition(monkeypatch):
    tracker = DeepSortTracker(DeepSortTrackerConfig())
    monkeypatch.setattr(tracker, "reid_distance_to_uid", lambda *a: pytest.fail("unexpected ReID"))
    assert tracker._frame_identity_competition([Detection((0, 0, 100, 300), .9, 0)], [None]) == {}


def setup_search_tracker():
    tracker = DeepSortTracker(DeepSortTrackerConfig(identity_new_confirm_frames=1))
    anchor = np.array([1., 0., 0.], dtype=np.float32)
    assert tracker.identity_bank.assign(track_id=26, feature=anchor, confidence=.9,
                                        area=30000, frame_index=0) == 1
    tracker.set_search_reacquire_context(active_uid=1, searching=True, direction="left")
    return tracker, anchor


@pytest.mark.parametrize("tentative_competitor", [False, True])
def test_real_update_vetoes_even_when_other_person_has_no_formal_track(monkeypatch, tentative_competitor):
    tracker, anchor = setup_search_tracker()
    detections = [Detection((60, 100, 160, 400), .95, 0),
                  Detection((400, 100, 500, 400), .8, 0)]
    outputs = [SimpleNamespace(track_id=i+6, source_detection_index=i,
               x1=d.bbox[0], y1=d.bbox[1], x2=d.bbox[2], y2=d.bbox[3],
               class_id=0, confidence=d.score, feature=anchor,
               time_since_update=0, state=2) for i, d in enumerate(detections)]
    monkeypatch.setattr(tracker.deepsort, "update", lambda *a, **k:
                        outputs[:1] if tentative_competitor else outputs)
    records = tracker.update(detections, [anchor, anchor], image_width=640, image_height=480,
                             frame_context={"capture_frame_id": 1085, "capture_timestamp": 10.})
    assert all(record.reid_uid == 0 for record in records)
    assert tracker.identity_bank.last_assignments[6]["reason"] == "search_candidate_identity_ambiguous"
    assert tracker.last_identity_observations[0]["sample_metadata"]["identity_competition"]["candidate_count"] == 2


def test_detector_probe_cannot_ignore_person_without_feature():
    tracker, anchor = setup_search_tracker()
    record = tracker._search_probe_record(
        [Detection((60, 100, 160, 400), .95, 0), Detection((400, 100, 500, 400), .1, 0)],
        [anchor, None], partial_features=[None, None], image_width=640, image_height=480,
    )
    assert record is not None and record.reid_uid == 0
    assert tracker.identity_bank.last_assignments[record.track_id]["reason"] == "search_candidate_identity_ambiguous"


@pytest.mark.parametrize("search_active,evidence_frame", [(False, 1), (True, 0)])
def test_nonsearch_or_stale_competition_does_not_override_existing_assignment(search_active, evidence_frame):
    results = []
    for include_competition in (False, True):
        bank = IdentityBank(IdentityBankConfig(new_identity_confirm_frames=1))
        assert bank.assign(track_id=1, feature=np.array([1., 0., 0.]), confidence=.9,
                           area=30000, frame_index=0) == 1
        metadata = {"is_fresh": True, "search_reacquire_context_active": search_active}
        if include_competition:
            metadata["identity_competition"] = competition_evidence(
                {0: .1, 1: .1}, uid=1, frame_index=evidence_frame)[0]
        uid = bank.assign(track_id=1, feature=np.array([1., 0., 0.]), confidence=.9,
                          area=30000, frame_index=1, preferred_uid=1, sample_metadata=metadata)
        results.append((uid, bank.last_assignments[1]["reason"], dict(bank.track_to_uid)))
    assert results[0] == results[1]  # unrelated/stale evidence leaves old gates unchanged


def test_unique_reid_winner_is_not_permission_to_bypass_quality():
    tracker, anchor = setup_search_tracker()
    evidence = competition_evidence({0: .1, 1: .4}, uid=1, frame_index=1)[0]
    uid = tracker.identity_bank.assign(track_id=6, feature=anchor, confidence=.9, area=30000,
        frame_index=1, preferred_uid=1, preferred_candidate_ok=True,
        bbox_quality_ok=False, bbox_quality_tier="reject", bbox_quality_reason="empty_bbox",
        sample_metadata={"is_fresh": True, "search_reacquire_context_active": True,
                         "identity_competition": evidence})
    assert uid == 0
    assert tracker.identity_bank.last_assignments[6]["reason"] == "bbox_quality_reject"
