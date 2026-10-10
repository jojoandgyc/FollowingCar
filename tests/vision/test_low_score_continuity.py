"""Fresh low-score detections bridge association, never identity or learning."""
from copy import deepcopy

import numpy as np
import pytest

from rk_vision.deepsort.deep_sort import DeepSort, DeepSortConfig


FEATURE = np.array([1., 0., 0.], dtype="float32")
BOX = [300., 240., 160., 380.]


def frame(ds, cap, stamp, *, boxes=None, scores=None, features=None,
          validator=lambda *_: True, yaw=0., context=True):
    boxes = [BOX] if boxes is None else boxes
    return ds.update(boxes, scores if scores is not None else [.9] * len(boxes),
        [0] * len(boxes), features if features is not None else [FEATURE] * len(boxes),
        image_shape=(480, 640), match_validator=validator,
        low_score_validator=validator,
        capture_context=dict(capture_frame_id=cap, capture_timestamp=stamp,
            integrated_yaw_deg=yaw) if context else None)


def confirmed():
    ds = DeepSort(DeepSortConfig(n_init=2, nms_max_overlap=1.))
    frame(ds, 1, .1)
    frame(ds, 2, .2)
    assert ds.tracker.tracks[0].is_confirmed()
    return ds


def test_current_low_score_match_preserves_source_capture_and_freezes_gallery():
    ds = confirmed()
    before = deepcopy(ds.tracker.metric.samples)
    out = frame(ds, 3, .3, boxes=[[30., 20., 20., 20.], BOX], scores=[.1, .328])[0]
    assert out.track_id == 1 and out.source_detection_index == 1
    assert out.time_since_update == 0 and out.low_score_continuation
    assert out.association_previous_capture_frame_id == 2
    assert out.association_previous_capture_timestamp == .2
    np.testing.assert_array_equal(ds.tracker.metric.samples[1], before[1])
    assert ds.tracker.tracks[0].features == []
    assert ds.tracker.last_timing_ms['low_score_matched_count'] == 1


def test_low_scores_never_create_or_confirm_new_tracks():
    ds = DeepSort(DeepSortConfig(n_init=2))
    for cap in range(1, 8):
        assert frame(ds, cap, .1 * cap, scores=[.328]) == []
    assert ds.tracker.tracks == [] and not ds.tracker.metric.samples
    frame(ds, 8, .8)
    assert not ds.tracker.tracks[0].is_confirmed()
    assert frame(ds, 9, .9, scores=[.328]) == []
    assert not ds.tracker.tracks


def test_high_priority_matching_cannot_be_stolen_by_weak_candidate():
    ds = confirmed()
    out = frame(ds, 3, .3, boxes=[BOX, [305., 240., 160., 380.]],
                scores=[.95, .328])
    assert len(out) == 1 and not out[0].low_score_continuation
    assert out[0].source_detection_index == 0
    assert ds.tracker._next_id == 2


@pytest.mark.parametrize('failure', ['expired', 'duplicate_cap', 'duplicate_time',
    'reverse_cap', 'reverse_time', 'no_context', 'no_validator', 'denied',
    'no_feature', 'nan_feature', 'zero_feature', 'different_feature', 'tiny_box',
    'far_box', 'vertical_jump', 'nan_yaw', 'bad_yaw', 'nan_score', 'below_floor'])
def test_invalid_or_unsafe_weak_evidence_never_changes_current_track(failure):
    ds = confirmed()
    cap, stamp, options = 3, .3, dict(scores=[.328])
    if failure == 'expired': stamp = .701
    elif failure == 'duplicate_cap': cap = 2
    elif failure == 'duplicate_time': stamp = .2
    elif failure == 'reverse_cap': cap = 1
    elif failure == 'reverse_time': stamp = .1
    elif failure == 'no_context': options['context'] = False
    elif failure == 'no_validator': options['validator'] = None
    elif failure == 'denied': options['validator'] = lambda *_: False
    elif failure == 'no_feature': options['features'] = [None]
    elif failure == 'nan_feature': options['features'] = [np.array([np.nan, 0., 0.])]
    elif failure == 'zero_feature': options['features'] = [np.zeros(3)]
    elif failure == 'different_feature': options['features'] = [np.array([0., 1., 0.])]
    elif failure == 'tiny_box': options['boxes'] = [[300., 240., 30., 80.]]
    elif failure == 'far_box': options['boxes'] = [[550., 240., 160., 380.]]
    elif failure == 'vertical_jump': options['boxes'] = [[300., 400., 160., 380.]]
    elif failure == 'nan_yaw': options['yaw'] = float('nan')
    elif failure == 'bad_yaw': options['yaw'] = 'invalid'
    elif failure == 'nan_score': options['scores'] = [float('nan')]
    elif failure == 'below_floor': options['scores'] = [.249]
    before = deepcopy(ds.tracker.metric.samples)
    track = ds.tracker.tracks[0]
    out = frame(ds, cap, stamp, **options)
    assert track.hits == 2 and track.time_since_update == 1
    assert ds.tracker._next_id == 2
    assert not any(row.time_since_update == 0 for row in out)
    np.testing.assert_array_equal(ds.tracker.metric.samples[1], before[1])


def test_low_frames_do_not_renew_original_capture_deadline():
    ds = confirmed()
    for cap, stamp in ((3, .3), (4, .45), (5, .69)):
        assert frame(ds, cap, stamp, scores=[.328])[0].low_score_continuation
    frame(ds, 6, .701, scores=[.328])
    assert ds.tracker.tracks[0].time_since_update == 1
    assert ds.tracker.tracks[0].low_score_anchor['capture_frame_id'] == 2


def test_low_match_may_fill_three_processed_steps_but_not_long_lost_track():
    ds = confirmed()
    frame(ds, 3, .3, boxes=[])
    frame(ds, 4, .4, boxes=[])
    assert frame(ds, 5, .5, scores=[.328])[0].low_score_continuation
    ds = confirmed()
    for cap in range(3, 7):
        frame(ds, cap, .2 + (cap-2) * .05, boxes=[])
    assert frame(ds, 7, .5, scores=[.328]) == []
    assert ds.tracker.tracks[0].hits == 2


def test_two_viable_weak_boxes_are_ambiguous_without_initiation():
    ds = confirmed()
    frame(ds, 3, .3, boxes=[BOX, [320., 240., 160., 380.]], scores=[.32, .33])
    assert ds.tracker.tracks[0].hits == 2 and ds.tracker._next_id == 2


def test_one_weak_box_cannot_choose_between_two_recent_tracks():
    ds = DeepSort(DeepSortConfig(n_init=2, nms_max_overlap=1.))
    boxes = [BOX, [340., 240., 160., 380.]]
    frame(ds, 1, .1, boxes=boxes)
    frame(ds, 2, .2, boxes=boxes)
    frame(ds, 3, .3, boxes=[[320., 240., 160., 380.]], scores=[.328])
    assert [track.hits for track in ds.tracker.tracks] == [2, 2]


def test_camera_compensation_uses_physical_signed_yaw():
    ds = confirmed()
    out = frame(ds, 3, .3, boxes=[[470., 240., 160., 380.]], scores=[.328], yaw=-12.)
    assert out[0].time_since_update == 0 and out[0].low_score_continuation
    ds = confirmed()
    frame(ds, 3, .3, boxes=[[470., 240., 160., 380.]], scores=[.328], yaw=12.)
    assert ds.tracker.tracks[0].hits == 2


def test_high_score_returns_to_normal_and_replaces_bridge_anchor():
    ds = confirmed()
    frame(ds, 3, .3, scores=[.328])
    out = frame(ds, 4, .4)[0]
    assert not out.low_score_continuation
    assert out.association_previous_capture_frame_id is None
    assert ds.tracker.tracks[0].low_score_anchor['capture_frame_id'] == 4


def test_adapter_delivers_current_association_metadata_not_full_evidence(monkeypatch):
    from test_provisional_association import activated, update

    tracker, _ = activated()
    submitted = []
    frame_evidence = []
    monkeypatch.setattr(tracker.identity_bank, 'assign',
                        lambda **kwargs: submitted.append(kwargs) or 0)
    monkeypatch.setattr(tracker.identity_bank, 'observe_frame_evidence',
                        lambda **kwargs: frame_evidence.extend(kwargs['observations']))
    result = update(tracker, 22, 3.2, score=.328)
    assert result == []  # association-only UID0 must not invalidate live motor proof
    assert tracker.deepsort.tracker.tracks[0].time_since_update == 0
    metadata = submitted[0]['sample_metadata']
    assert metadata['low_score_continuation'] is True
    assert metadata['association_reason'] == 'low_score_existing_track'
    assert metadata['association_previous_capture_frame_id'] == 21
    assert metadata['association_previous_capture_timestamp'] == 3.1
    assert metadata['association_confidence_limit'] == tracker.config.min_confidence
    assert metadata['capture_frame_id'] == 22
    assert metadata['source_detection_index'] == 0
    assert metadata['is_fresh'] is True
    assert frame_evidence == []  # cannot exclude other people as a UID witness
    assert not tracker.deepsort.tracker.metric.samples


@pytest.mark.parametrize('reason', ['geometry_conflict', 'revoked_uid', 'excluded', 'poor_crop'])
def test_adapter_rejects_low_score_pair_with_current_identity_negative_evidence(monkeypatch, reason):
    from test_provisional_association import activated, update

    tracker, _ = activated()
    if reason == 'geometry_conflict':
        tracker.identity_bank._mapped_geometry_conflicts[3] = dict(uid=1)
    elif reason == 'revoked_uid':
        tracker.identity_bank._geometry_revoked_uids[1] = dict(reason='test_conflict')
    elif reason == 'excluded':
        monkeypatch.setattr(tracker.identity_bank, 'search_exclusion_for',
                            lambda *args, **kwargs: dict(reason='known_other_person'))
    else:
        monkeypatch.setattr(tracker, '_bbox_quality', lambda *args: (False, 'aspect<0.18'))
    track = tracker.deepsort.tracker.tracks[0]
    before = track.hits
    update(tracker, 22, 3.2, score=.328)
    assert track.hits == before
    assert track.time_since_update == 1
    assert tracker.deepsort.tracker._next_id == 4


def test_real_adapter_keeps_pending_candidate_through_low_score_gap_without_control_or_learning():
    from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
    from rk_vision.yolo11 import Detection
    from test_cap1040_identity_continuity import (
        pending_bank, BOX1074, BOX1078, BOX1084, TIME1074, TIME1078, TIME1084, feature)
    from tools.replay_cap334_recovery import gallery_snapshot

    tracker = DeepSortTracker(DeepSortTrackerConfig(min_confidence=.60, n_init=2,
        bbox_expand_scale=1., identity_similar_follow_enable=True,
        identity_template_memory_enable=True, identity_template_crosscheck_enable=True,
        identity_template_learning_guard_enable=True))
    tracker.identity_bank = pending_bank()
    tracker.deepsort.tracker._next_id = 16
    tracker._frame_index = 474
    tracker.set_search_reacquire_context(active_uid=1, searching=True, direction='left')
    x1, y1, x2, y2 = BOX1074
    xywh = [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]
    for cap, stamp in ((1072, TIME1074-.10), (1074, TIME1074)):
        tracker.deepsort.update([xywh], [.6254], [0], [feature(.2152673)],
            image_shape=(480,640), capture_context=dict(capture_frame_id=cap,
                capture_timestamp=stamp, integrated_yaw_deg=-180.96460148135935))
    original = gallery_snapshot(tracker.identity_bank)
    gallery = deepcopy(tracker.deepsort.tracker.metric.samples)

    def step(cap, stamp, box=None, score=.9, yaw=0., full=.23):
        return tracker.update([] if box is None else [Detection(box, score, 0)],
            [] if box is None else [feature(full)], image_width=640, image_height=480,
            frame_context=dict(capture_frame_id=cap, capture_timestamp=stamp, integrated_yaw_deg=yaw))

    step(1075, TIME1074+.07)
    step(1076, TIME1074+.15)
    assert step(1078, TIME1078, BOX1078, score=.3278, yaw=-185.949) == []
    state = tracker.identity_bank._similar_follow_states[(1,16)]
    assert state['last_cap'] == 1078 and state['last_qualified_cap'] == 1074
    assert state['count'] == 1 and not state['active']
    assert len(tracker.last_identity_observations) == 1
    assert tracker.last_identity_observations[0]['uid'] == 0
    assert tracker.last_identity_observations[0]['raw_track_id'] == 16
    np.testing.assert_array_equal(tracker.deepsort.tracker.metric.samples[16], gallery[16])
    assert gallery_snapshot(tracker.identity_bank) == original
    # Three missed processed frames still cannot turn prediction into UID or
    # refresh the last qualified capture. All times are real-new observations.
    for cap, stamp in ((1080, TIME1078+.09), (1081, TIME1078+.16), (1082, TIME1078+.23)):
        assert all(record.reid_uid == 0 for record in step(cap, stamp))
    result = step(1084, TIME1084, BOX1084, score=.70075, yaw=-189.85429946853276, full=.2326126)
    assert [(record.track_id, record.reid_uid) for record in result] == [(16,1)]
    assert tracker.deepsort.tracker._next_id == 17
    assert gallery_snapshot(tracker.identity_bank) == original
