"""Real tracker -> runtime adapter -> startup lock, without opening hardware."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import request_0513_modular as runtime
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.control_types import SensorFrame
from car_control_modular.startup_identity import resolve_initial_identity_confirmation
from car_control_modular.video_recorder import (
    VideoControlOverlay, VideoFrameOverlay, VideoTrackOverlay, startup_status,
)
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection


BOX = (207., 1., 463., 475.)
FEATURE = np.array([1., 0., 0.], dtype=np.float32)


def tracker():
    return DeepSortTracker(DeepSortTrackerConfig(
        n_init=1, identity_template_memory_enable=True,
        identity_template_crosscheck_enable=True,
        identity_appearance_region_safety_enable=True))


def update(t, cap):
    return t.update([Detection(BOX, .94, 0)], [FEATURE],
        partial_features=[FEATURE], partial_feature_sources=['osnet_torso'],
        image_width=640, image_height=480,
        frame_context=dict(control_frame_id=cap, capture_frame_id=cap,
                           capture_timestamp=100. + .1 * cap))


def prepared():
    t = tracker()
    for cap in (1, 2, 3):
        update(t, cap)
    assert t.last_identity_observations[0]['uid'] == 1
    return t


def arguments(t):
    r = t.last_identity_observations[0]
    m = r['sample_metadata']
    return dict(target_id=r['uid'], display_bbox=r['display_bbox'],
        capture_frame_id=m['capture_frame_id'], capture_timestamp=m['capture_timestamp'],
        observations=t.last_identity_observations, width=640, height=480)


def test_real_enrollment_locks_on_same_accepted_capture_not_two_more(monkeypatch):
    t = tracker()
    c = FollowSafetyController(FollowPolicyConfig(initial_target_confirm_frames=2))
    owner = object.__new__(runtime.PersonTracker)
    owner._rknn_pipeline = SimpleNamespace(tracker=t)
    owner._follow_controller = c
    for cap in (1, 2, 3):
        monkeypatch.setattr(runtime.time, 'monotonic', lambda: 100. + .1 * cap + .05)
        update(t, cap)
        owner._active_capture_frame_id = cap
        owner._active_capture_timestamp = 100. + .1 * cap
        persons = [(r['display_bbox'], r['uid'], .94, 121344.)
                   for r in t.last_identity_observations if r['uid'] > 0]
        targets = owner._persons_to_targets(persons, width=640, height=480)
        decision = c.decide(cap, SensorFrame(width=640, height=480, persons=targets,
            capture_frame_id=cap, capture_timestamp=100. + .1 * cap))
        if cap < 3:
            assert c.active_target_id is None
            assert all(a.kind == 'stop' for a in decision.actions)
        else:
            assert targets[0].initial_identity_confirmed
            assert c.active_target_id == 1 and c._has_seen_person


def test_provenance_available_on_later_accepted_frame_if_creation_was_late():
    t = prepared()
    update(t, 4)
    assert resolve_initial_identity_confirmation(**arguments(t))


def test_original_enrolled_uid_cannot_be_replaced_before_controller_lock():
    t = prepared()
    owner = object.__new__(runtime.PersonTracker)
    owner._rknn_pipeline = SimpleNamespace(tracker=t)
    owner._follow_controller = FollowSafetyController(FollowPolicyConfig())
    owner._active_capture_frame_id = 4
    owner._active_capture_timestamp = 100.4
    other = [(BOX, 2, .99, 130000.)]
    # UID1 was selected by vision but its creation frame was late. Merely
    # returning UID2 must not invoke a second startup selection in control.
    assert owner._persons_to_targets(other, width=640, height=480) == []
    owner._follow_controller.active_target_id = 1
    owner._follow_controller._has_seen_person = True
    targets = owner._persons_to_targets(other, width=640, height=480)
    assert len(targets) == 1 and targets[0].track_id == 2
    assert owner._follow_controller._select_person(SensorFrame(
        width=640, height=480, persons=targets)) is None


@pytest.mark.parametrize('change', [
    'no_marker', 'other_uid', 'capture', 'timestamp', 'bbox', 'ambiguous',
    'weak', 'rejected', 'excluded', 'not_fresh', 'no_detector', 'pending',
    'uid_zero', 'metadata_missing', 'competition_failed',
])
def test_marker_cannot_replace_exact_current_accepted_association(change):
    t = prepared()
    a = deepcopy(arguments(t))
    r = a['observations'][0]
    if change == 'no_marker': r['assignment'].pop('initial_identity_confirmed')
    if change == 'other_uid': a['target_id'] = 2
    if change == 'capture': a['capture_frame_id'] += 1
    if change == 'timestamp': a['capture_timestamp'] += .01
    if change == 'bbox': a['display_bbox'] = BOX
    if change == 'ambiguous': a['observations'].append(deepcopy(r))
    if change == 'weak': r['assignment']['bbox_quality_ok'] = False
    if change == 'rejected': r['assignment']['identity_control_rejected'] = True
    if change == 'excluded': r['assignment']['search_excluded'] = True
    if change == 'not_fresh': r['sample_metadata']['is_fresh'] = False
    if change == 'no_detector': r.pop('detector_bbox')
    if change == 'pending': r['assignment']['reason'] = 'pending_initial_identity'
    if change == 'uid_zero': r['uid'] = 0
    if change == 'metadata_missing': r.pop('sample_metadata')
    if change == 'competition_failed': r['sample_metadata']['identity_competition'] = {'passed': False}
    assert not resolve_initial_identity_confirmation(**a)


def test_rejected_assignment_does_not_export_old_enrollment_marker():
    t = prepared()
    r = deepcopy(t.last_identity_observations[0])
    m = r['sample_metadata']
    m.update(capture_frame_id=4, capture_timestamp=100.4, frame_index=4)
    uid = t.identity_bank.assign(track_id=1, feature=FEATURE, partial_feature=FEATURE,
        confidence=.94, area=121344., frame_index=4, sample_metadata=m,
        bbox_quality_ok=False, bbox_quality_tier='weak', bbox_quality_reason='edge_touch>2')
    assert uid == 0
    assert not t.identity_bank.last_assignments[1]['initial_identity_confirmed']


def test_missing_current_features_cannot_turn_cached_uid_into_first_lock():
    t = prepared()
    t.update([Detection(BOX, .94, 0)], [None], partial_features=[None],
        partial_feature_sources=[None], image_width=640, image_height=480,
        frame_context=dict(control_frame_id=4, capture_frame_id=4, capture_timestamp=100.4))
    r = t.last_identity_observations[0]
    assert r['uid'] == 1  # Existing post-lock mapped-UID semantics are untouched.
    assert not r['assignment']['initial_identity_confirmed']
    assert not resolve_initial_identity_confirmation(**arguments(t))
    owner = object.__new__(runtime.PersonTracker)
    owner._rknn_pipeline = SimpleNamespace(tracker=t)
    owner._follow_controller = FollowSafetyController(FollowPolicyConfig())
    owner._active_capture_frame_id = 4
    owner._active_capture_timestamp = 100.4
    assert owner._persons_to_targets([(r['display_bbox'], 1, .94, 121344.)], width=640, height=480) == []


def test_current_identity_competition_failure_cannot_export_enrollment():
    t = prepared()
    r = deepcopy(t.last_identity_observations[0])
    m = r['sample_metadata']
    m.update(capture_frame_id=4, capture_timestamp=100.4, frame_index=4,
        identity_competition=dict(uid=1, frame_index=4, source_detection_index=0,
                                  candidate_count=2, passed=False))
    t.identity_bank.assign(track_id=1, feature=FEATURE, partial_feature=FEATURE,
        confidence=.94, area=121344., frame_index=4, sample_metadata=m,
        candidate_count=2, bbox_quality_ok=True, bbox_quality_tier='strong')
    assert not t.identity_bank.last_assignments[1]['initial_identity_confirmed']


def test_reset_clears_initial_provenance_and_requires_new_enrollment():
    t = prepared()
    t.reset()
    assert t.identity_bank._initial_enrolled_uid is None
    update(t, 4)
    update(t, 5)
    assert all(r['uid'] == 0 for r in t.last_identity_observations)
    assert all(not r['assignment']['initial_identity_confirmed']
               for r in t.last_identity_observations)


def test_overlay_locked_status_takes_priority_over_last_startup_reason():
    overlay = VideoFrameOverlay(control=VideoControlOverlay(
        active_target_id=1, decision_reason='initial_candidate_confirmation_hold'))
    assert startup_status(overlay).startswith('TARGET U1')


@pytest.mark.parametrize('reason,label', [
    ('initial_candidate_track_changed', 'WAIT FIRST CANDIDATE'),
    ('initial_candidate_appearance_mismatch', 'VERIFY FIRST CANDIDATE'),
])
def test_overlay_explains_reserved_candidate_wait(reason, label):
    overlay = VideoFrameOverlay(tracks=(VideoTrackOverlay(BOX, 2, assignment_reason=reason),))
    assert label in startup_status(overlay)
