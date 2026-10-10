"""Follow-only UID evidence cannot silently train the raw-track gallery."""
from types import SimpleNamespace
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.pipeline import RKNNVisionConfig, RKNNVisionPipeline
from rk_vision.tracker import TrackRecord
from rk_vision.yolo11 import Detection


def feat(angle):
    return np.array([np.cos(angle), np.sin(angle), 0.], dtype=np.float32)


def snapshot(tracker):
    return {key: [value.copy() for value in rows]
            for key, rows in tracker.deepsort.tracker.metric.samples.items()}


def test_actual_deepsort_update_keeps_fresh_feature_but_not_soft_gallery(monkeypatch):
    tracker = DeepSortTracker(DeepSortTrackerConfig(n_init=1, feature_update_interval=1,
        identity_similar_follow_enable=True))
    detections = [Detection((204., 40., 407., 450.), .946, 0)]
    for cap in range(1, 5):
        tracker.update(detections, [feat(0)], image_width=640, image_height=480,
            frame_context=dict(capture_frame_id=cap, capture_timestamp=10.+cap*.1))
    before = snapshot(tracker)
    assert before[1]

    def follow_only(**kwargs):
        tracker.identity_bank.last_assignments[kwargs["track_id"]] = dict(
            uid=1, match_source="similar_follow", reason="mapped_similar_follow",
            similar_follow=dict(status="follow"), bank_updated=False,
            bbox_quality_ok=True, bbox_quality_tier="strong")
        return 1
    monkeypatch.setattr(tracker.identity_bank, "assign", follow_only)
    current = feat(.12)
    records = tracker.update(detections, [current], image_width=640, image_height=480,
        frame_context=dict(capture_frame_id=5, capture_timestamp=10.5))
    assert records[0].reid_uid == 1
    after = snapshot(tracker)
    assert len(after[1]) == len(before[1])
    for actual, old in zip(after[1], before[1]):
        np.testing.assert_array_equal(actual, old)
    track = tracker.deepsort.tracker.tracks[0]
    np.testing.assert_array_equal(track.last_feature, current)
    assert not track.features
    assert tracker.last_identity_observations[0]["assignment"]["association_gallery_frozen"]


@pytest.mark.parametrize("status", ["observe", "follow"])
def test_pending_new_raw_track_cannot_seed_association_gallery(status):
    tracker = DeepSortTracker(DeepSortTrackerConfig(identity_similar_follow_enable=True))
    current = feat(.15)
    track = SimpleNamespace(track_id=3, features=[current], last_feature=current)
    tracker.deepsort.tracker.tracks = [track]
    tracker.deepsort.tracker.metric.samples = {3: [current], 4: [feat(.3)]}
    tracker.identity_bank.last_assignments[3] = dict(similar_follow=dict(status=status))
    tracker._restore_follow_only_association_gallery({4: [feat(0)]})
    assert 3 not in tracker.deepsort.tracker.metric.samples
    assert not track.features
    np.testing.assert_array_equal(track.last_feature, current)
    # Other people's legitimate association updates are not rolled back.
    np.testing.assert_array_equal(tracker.deepsort.tracker.metric.samples[4][0], feat(.3))


def test_ordinary_assignment_is_not_blocked_by_new_learning_rule():
    tracker = DeepSortTracker(DeepSortTrackerConfig(identity_similar_follow_enable=True))
    current = feat(.15)
    tracker.deepsort.tracker.tracks = [SimpleNamespace(track_id=3, features=[current])]
    tracker.deepsort.tracker.metric.samples = {3: [current]}
    tracker.identity_bank.last_assignments[3] = dict(uid=1, match_source="strong", reason="mapped")
    tracker._restore_follow_only_association_gallery({3: [feat(0)]})
    np.testing.assert_array_equal(tracker.deepsort.tracker.metric.samples[3][0], current)


def test_legacy_fallback_cannot_drop_follow_only_learning_fence():
    tracker = DeepSortTracker(DeepSortTrackerConfig(identity_similar_follow_enable=True))
    current = feat(.15)
    tracker.deepsort.tracker.tracks = [SimpleNamespace(track_id=3, features=[current])]
    tracker.deepsort.tracker.metric.samples = {3: [current]}
    tracker.identity_bank.last_assignments[3] = dict(uid=1, match_source="strong", reason="mapped")
    tracker.identity_bank._similar_learning_fences.add(1)
    tracker._restore_follow_only_association_gallery({3: [feat(0)]})
    np.testing.assert_array_equal(tracker.deepsort.tracker.metric.samples[3][0], feat(0))
    assert not tracker.deepsort.tracker.tracks[0].features


def handoff_pipeline():
    pipeline = RKNNVisionPipeline(RKNNVisionConfig(yolo_model_path="unused",
        identity_similar_follow_enable=True))
    pipeline._frame_context = dict(capture_frame_id=336, capture_timestamp=100.)
    assignment = dict(uid=1, mapped_uid=1, match_source="similar_follow",
        reason="similar_follow_reacquire", bbox_quality_ok=True,
        reacquire_geometry_ok=True, similar_follow=dict(status="follow", capture_frame_id=336))
    pipeline.tracker.identity_bank.track_to_uid = {3: 1}
    pipeline.tracker.identity_bank.last_assignments[3] = assignment
    pipeline.tracker.last_identity_observations = [dict(raw_track_id=3, uid=1,
        sample_metadata=dict(capture_frame_id=336, capture_timestamp=100., is_fresh=True),
        assignment=deepcopy(assignment))]
    current = TrackRecord(track_id=3, reid_uid=1, x1=100., y1=40., x2=200., y2=440.,
        class_id=0, score=.94, cx=150., cy=240., area=40000., angle_deg=0.,
        tracker_state="stable", time_since_update=0)
    predicted = replace(current, track_id=1, time_since_update=1)
    return pipeline, current, predicted


@pytest.mark.parametrize("reverse", [False, True])
def test_fresh_similar_handoff_not_erased_by_revoked_old_prediction(reverse):
    pipeline, current, predicted = handoff_pipeline()
    records = [current, predicted]
    if reverse:
        records.reverse()
    result = pipeline._suppress_duplicate_reid_uids(records)
    assert {row.track_id: row.reid_uid for row in result} == {3: 1, 1: 0}
    assert pipeline.tracker.identity_bank.last_assignments[3]['superseded_predicted_track_ids'] == [1]


@pytest.mark.parametrize("failure", ["second_fresh", "old_claim_not_revoked", "current_claim_revoked",
    "old_capture", "old_timestamp", "duplicate_provenance", "no_provenance", "old_live_proof",
    "observe_only", "not_fresh", "rejected", "geometry", "competition", "unrelated_source", "disabled"])
def test_duplicate_claim_exception_requires_current_bank_proof(failure):
    pipeline, current, predicted = handoff_pipeline()
    bank = pipeline.tracker.identity_bank
    observation = pipeline.tracker.last_identity_observations[0]
    assignment = observation['assignment']
    if failure == 'second_fresh': predicted = replace(predicted, time_since_update=0)
    elif failure == 'old_claim_not_revoked': bank.track_to_uid[1] = 1
    elif failure == 'current_claim_revoked': bank.track_to_uid.pop(3)
    elif failure == 'old_capture': observation['sample_metadata']['capture_frame_id'] = 334
    elif failure == 'old_timestamp': observation['sample_metadata']['capture_timestamp'] = 99.
    elif failure == 'duplicate_provenance': pipeline.tracker.last_identity_observations.append(deepcopy(observation))
    elif failure == 'no_provenance': pipeline.tracker.last_identity_observations.clear()
    elif failure == 'old_live_proof': bank.last_assignments[3]['similar_follow']['capture_frame_id'] = 334
    elif failure == 'observe_only': assignment['similar_follow']['status'] = 'observe'
    elif failure == 'not_fresh': observation['sample_metadata']['is_fresh'] = False
    elif failure == 'rejected': assignment['identity_control_rejected'] = True
    elif failure == 'geometry': assignment['reacquire_geometry_ok'] = False
    elif failure == 'competition': assignment['identity_competition'] = dict(passed=False)
    elif failure == 'unrelated_source': assignment['match_source'] = 'strong'
    elif failure == 'disabled': pipeline.config = replace(pipeline.config, identity_similar_follow_enable=False)
    result = pipeline._suppress_duplicate_reid_uids([current, predicted])
    assert [row.reid_uid for row in result] == [0, 0]


@pytest.mark.parametrize("cap,stamp", [(0, 100.), (True, 100.), (336, float('inf')),
                                      (336, float('nan')), (336, -1.)])
def test_matching_invalid_capture_values_cannot_authorize_duplicate_exception(cap, stamp):
    pipeline, current, predicted = handoff_pipeline()
    pipeline._frame_context.update(capture_frame_id=cap, capture_timestamp=stamp)
    observation = pipeline.tracker.last_identity_observations[0]
    observation['sample_metadata'].update(capture_frame_id=cap, capture_timestamp=stamp)
    for assignment in (observation['assignment'], pipeline.tracker.identity_bank.last_assignments[3]):
        assignment['similar_follow']['capture_frame_id'] = cap
    assert [r.reid_uid for r in pipeline._suppress_duplicate_reid_uids([current, predicted])] == [0, 0]
