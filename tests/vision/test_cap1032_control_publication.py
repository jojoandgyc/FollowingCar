"""Measured crop acceptance must survive the real identity/depth interfaces.

CAP1040 box and appearance distance, synthetic descriptor; no camera, NPU,
serial port or PersonTracker constructor is used.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import request_0513_modular as runtime
from car_control_modular.depth_target_geometry import resolve_depth_target_observation
from car_control_modular.detector_identity_lease import read_visual_identity_evidence
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection


CROP = (0., 3.480804443359375, 195.2913360595703, 475.8856201171875)


def test_verified_crop_reaches_control_and_depth_without_reenrollment(monkeypatch):
    tracker = DeepSortTracker(DeepSortTrackerConfig(
        n_init=1, identity_similar_follow_enable=True,
        identity_template_memory_enable=True, identity_template_crosscheck_enable=True,
        identity_template_learning_guard_enable=True,
        identity_appearance_region_safety_enable=True))
    reference = np.array([1., 0., 0.], dtype=np.float32)
    for i in range(5):
        records = tracker.update([Detection((20., 4., 220., 476.), .94, 0)],
            [reference], partial_features=[reference], partial_feature_sources=['osnet_torso'],
            image_width=640, image_height=480,
            frame_context=dict(capture_frame_id=1031+i, capture_timestamp=100.+i*.05))
    assert len(records) == 1 and records[0].reid_uid == 1
    entry = tracker.identity_bank.identities[1]
    anchor = deepcopy(entry.last_strong_observation)
    templates = deepcopy(entry.feature_metadata)
    distance = .15411663055419922
    query = np.array([1.-distance, (1.-(1.-distance)**2)**.5, 0.], dtype=np.float32)
    records = tracker.update([Detection(CROP, .885, 0)], [query],
        partial_features=[reference], partial_feature_sources=['osnet_torso'],
        image_width=640, image_height=480,
        frame_context=dict(capture_frame_id=1040, capture_timestamp=100.3))
    assert len(records) == 1 and records[0].reid_uid == 1
    assignment = tracker.identity_bank.last_assignments[records[0].track_id]
    assert assignment['bbox_quality_ok'] and not assignment['bank_updated']
    assert entry.last_strong_observation == anchor
    assert entry.feature_metadata == templates
    row = tracker.last_identity_observations[0]
    observation = resolve_depth_target_observation(
        target_id=1, display_bbox=row['display_bbox'], capture_frame_id=1040,
        capture_timestamp=100.3, observations=tracker.last_identity_observations,
        width=640, height=480)
    assert observation is not None and observation.bbox == pytest.approx(CROP)
    owner = object.__new__(runtime.PersonTracker)
    owner._follow_controller = SimpleNamespace(active_target_id=1)
    owner._rknn_pipeline = SimpleNamespace(tracker=tracker, last_identity_processing=dict(
        mode='full', full_features_current=True, capture_frame_id=1040, capture_timestamp=100.3))
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: 100.4)
    assert owner._update_detector_identity_lease(records, 1040, 100.3, now=100.4, stale=False)
    evidence, now = read_visual_identity_evidence(owner)
    assert evidence.live(1, now)
    assert evidence.permits_depth(1, 100.35, now)
    assert not evidence.live(1, 101.)
