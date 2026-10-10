"""Real DeepSORT/IdentityBank continuity; no camera, inference, or motor I/O."""
from copy import deepcopy

import pytest

from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection
from test_reacquire_crosscheck import feature
from test_similar_follow_bank import make_bank
from tools.replay_cap334_recovery import gallery_snapshot


BOX = (200., 40., 400., 450.)


def update(tracker, cap, stamp, *, distance=.44, empty=False, score=.95, box=BOX):
    return tracker.update([] if empty else [Detection(box, score, 0)],
        [] if empty else [None if distance is None else feature(distance)],
        image_width=640, image_height=480,
        frame_context=dict(capture_frame_id=cap, capture_timestamp=stamp, integrated_yaw_deg=0.))


def activated(*, max_bbox_age=2, box=BOX):
    tracker = DeepSortTracker(DeepSortTrackerConfig(n_init=2, bbox_expand_scale=1.,
        feature_update_interval=1, identity_similar_follow_enable=True,
        max_bbox_age=max_bbox_age, identity_template_memory_enable=True,
        identity_template_crosscheck_enable=True, identity_template_learning_guard_enable=True))
    tracker.identity_bank = make_bank()
    tracker.deepsort.tracker._next_id = 3
    tracker._frame_index = 18
    direction = "left" if (box[0]+box[2]) / 2. < 320. else "right"
    tracker.set_search_reacquire_context(active_uid=1, searching=True, direction=direction)
    frozen = gallery_snapshot(tracker.identity_bank)
    update(tracker, 19, 2.9, box=box)
    update(tracker, 20, 3., box=box)
    rows = update(tracker, 21, 3.1, box=box)
    assert [(row.track_id, row.reid_uid) for row in rows] == [(3, 1)]
    assert tracker.identity_bank.last_assignments[3]['match_source'] == 'similar_follow'
    assert tracker._provisional_association.diagnostics(3)['samples'] == 1
    assert not tracker.deepsort.tracker.metric.samples
    assert gallery_snapshot(tracker.identity_bank) == frozen
    return tracker, frozen


@pytest.mark.parametrize("missed", [1, 2])
def test_real_tracker_retains_raw_and_uid_after_missed_frames_without_learning(missed):
    tracker, frozen = activated()
    for cap in range(22, 22+missed):
        update(tracker, cap, 3.1+.1*(cap-21), empty=True)
    result = update(tracker, 22+missed, 3.2+.1*missed)
    assert [(row.track_id, row.reid_uid) for row in result] == [(3, 1)]
    assert tracker.deepsort.tracker._next_id == 4
    assert not tracker.deepsort.tracker.metric.samples
    assert gallery_snapshot(tracker.identity_bank) == frozen
    assert tracker.last_identity_observations[0]['assignment']['provisional_association']['samples'] == 2


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_real_association_ab_isolates_provisional_cache_from_iou_rescue(cache_enabled):
    # Both arms use the pre-fix effective IoU age. This isolates the cache.
    tracker, frozen = activated(max_bbox_age=1)
    if not cache_enabled:
        tracker._provisional_association.entries.clear()
    update(tracker, 22, 3.2, empty=True)
    result = update(tracker, 23, 3.3)
    assert (tracker.deepsort.tracker._next_id == 4) is cache_enabled
    assert any(row.track_id == 3 and row.reid_uid == 1 and row.time_since_update == 0
               for row in result) is cache_enabled
    assert gallery_snapshot(tracker.identity_bank) == frozen


@pytest.mark.parametrize("failure", ["expired", "different_person", "geometry", "validator",
    "cross_uid", "reused_identity", "conflict", "no_feature", "nonnew_cap", "nonnew_stamp"])
def test_provisional_evidence_cannot_override_rejection(monkeypatch, failure):
    tracker, frozen = activated()
    track = tracker.deepsort.tracker.tracks[0]
    options = {}
    cap, stamp = 23, 3.3
    if failure == 'expired': stamp = 3.61
    elif failure == 'different_person': options['distance'] = 1.99
    elif failure == 'geometry': options['box'] = (450., 40., 630., 450.)
    elif failure == 'validator': monkeypatch.setattr(tracker, '_identity_match_allowed', lambda *args: False)
    elif failure == 'cross_uid': tracker.identity_bank.track_to_uid[3] = 2
    elif failure == 'reused_identity':
        tracker.identity_bank.identities[1] = deepcopy(tracker.identity_bank.identities[1])
    elif failure == 'conflict': tracker.identity_bank._mapped_geometry_conflicts[3] = dict(uid=1)
    elif failure == 'no_feature': options['distance'] = None
    elif failure == 'nonnew_cap': cap = 21
    elif failure == 'nonnew_stamp': stamp = 3.1
    update(tracker, 22, 3.2, empty=True)
    old_hits = track.hits
    update(tracker, cap, stamp, **options)
    assert track.hits == old_hits
    assert track.time_since_update == 2
    assert tracker.deepsort.tracker._next_id == 5
    assert not tracker.deepsort.tracker.metric.samples
    assert gallery_snapshot(tracker.identity_bank) == frozen


def test_quality_rejected_frame_never_refreshes_cache_or_extends_expiry():
    tracker, frozen = activated()
    update(tracker, 22, 3.3, score=.51)
    detail = tracker._provisional_association.diagnostics(3)
    assert detail['capture_timestamp'] == 3.1
    assert detail['samples'] == 1
    update(tracker, 23, 3.5, empty=True)
    update(tracker, 24, 3.61)
    assert tracker.deepsort.tracker.tracks[0].time_since_update == 2
    assert tracker._provisional_association.diagnostics(3)['samples'] == 0
    assert gallery_snapshot(tracker.identity_bank) == frozen


def test_current_full_conflict_invalidates_existing_cache():
    tracker, frozen = activated()
    # Close to the candidate descriptor, but beyond the UID's follow limit.
    update(tracker, 22, 3.2, distance=.8)
    assert tracker._provisional_association.diagnostics(3)['samples'] == 0
    assert tracker._provisional_association.diagnostics(3)['reason'] == 'identity_conflict'
    assert not tracker.deepsort.tracker.metric.samples
    assert gallery_snapshot(tracker.identity_bank) == frozen


def test_current_qualified_crops_refresh_only_association_across_original_cache_expiry():
    tracker, frozen = activated(box=(450., 1., 620., 478.))
    box = (470., 0., 639., 479.)
    for cap, stamp in ((22, 3.3), (23, 3.5), (24, 3.7), (25, 3.8)):
        records = update(tracker, cap, stamp, box=box)
        assert [(row.track_id, row.reid_uid) for row in records] == [(3, 1)]
        assert tracker.deepsort.tracker._next_id == 4
        assignment = tracker.identity_bank.last_assignments[3]
        assert assignment['similar_follow']['crop_continuation'] is True
        diagnostic = assignment['provisional_association']
        assert diagnostic['capture_timestamp'] == stamp
        assert diagnostic['reason'] == 'accepted_crop_follow'
        assert tracker.identity_bank._similar_follow_states[(1, 3)]['last_strong_timestamp'] == 3.1
        assert not tracker.deepsort.tracker.metric.samples
        assert gallery_snapshot(tracker.identity_bank) == frozen
    # Association has a current descriptor, but it must not extend the UID's
    # .75 s crop budget from its independently qualified CAP21 observation.
    records = update(tracker, 26, 3.86, box=box)
    assert [(row.track_id, row.reid_uid) for row in records] == [(3, 0)]
    assert tracker._provisional_association.diagnostics(3)['capture_timestamp'] == 3.8
    assert not tracker.deepsort.tracker.metric.samples
    assert gallery_snapshot(tracker.identity_bank) == frozen


def test_compound_crop_failure_does_not_refresh_provisional_evidence(monkeypatch):
    tracker, frozen = activated(box=(450., 1., 620., 478.))
    previous = tracker._provisional_association.diagnostics(3)
    monkeypatch.setattr(tracker, '_bbox_quality', lambda *args:
                        (False, 'edge_touch>2,aspect<0.18'))
    records = update(tracker, 22, 3.3, box=(470., 0., 639., 479.))
    assert [(row.track_id, row.reid_uid) for row in records] == [(3, 0)]
    assert tracker._provisional_association.diagnostics(3) == previous
    assert not tracker.deepsort.tracker.metric.samples
    assert gallery_snapshot(tracker.identity_bank) == frozen


def test_cache_capacity_capture_expiry_reset_and_raw_object_reuse():
    tracker, frozen = activated()
    for cap in range(22, 29):
        update(tracker, cap, 3.1+.05*(cap-21))
    entry = tracker._provisional_association.entries[3]
    assert len(entry.samples) == 3
    assert [row[0] for row in entry.samples] == [26, 27, 28]
    assert not tracker.deepsort.tracker.metric.samples
    assert gallery_snapshot(tracker.identity_bank) == frozen
    old_track = tracker.deepsort.tracker.tracks[0]
    tracker.deepsort.tracker.tracks[0] = deepcopy(old_track)
    prepared = tracker._provisional_association.prepare(tracker.deepsort.tracker.tracks,
        tracker.identity_bank.track_to_uid, tracker.identity_bank.identities,
        dict(capture_frame_id=29, capture_timestamp=3.5), max_gap=.5, conflicts={}, revoked={})
    assert prepared == {} and not tracker._provisional_association.entries
    tracker.reset()
    assert not tracker._provisional_association.entries


def test_nonnew_clock_cannot_consume_or_refresh_otherwise_live_evidence():
    tracker, _ = activated()
    cache = tracker._provisional_association
    entry = cache.entries[3]
    before = cache.diagnostics(3)
    for context in (dict(capture_frame_id=20, capture_timestamp=3.),
                    dict(capture_frame_id=21, capture_timestamp=3.1),
                    dict(capture_frame_id=True, capture_timestamp=float('nan'))):
        assert cache.prepare(tracker.deepsort.tracker.tracks,
            tracker.identity_bank.track_to_uid, tracker.identity_bank.identities,
            context, max_gap=.5, conflicts={}, revoked={}) == {3: []}
        assert cache.diagnostics(3) == before
    prepared = cache.prepare(tracker.deepsort.tracker.tracks,
        tracker.identity_bank.track_to_uid, tracker.identity_bank.identities,
        dict(capture_frame_id=22, capture_timestamp=3.2), max_gap=.5, conflicts={}, revoked={})
    assert prepared[3][0] is entry.samples[0][2]


@pytest.mark.parametrize("max_bbox_age, expected_id", [(1, 2), (2, 1)])
def test_configured_iou_age_is_honored_by_real_tracker(max_bbox_age, expected_id):
    from rk_vision.deepsort.deep_sort import DeepSort, DeepSortConfig
    tracker = DeepSort(DeepSortConfig(n_init=2, max_bbox_age=max_bbox_age))
    box = [[200., 200., 100., 300.]]
    for _ in range(2):
        tracker.update(box, [.95], [0], [None])
    tracker.update([], [], [], [])
    tracker.update(box, [.95], [0], [None])
    fresh = [track for track in tracker.tracker.tracks if track.time_since_update == 0]
    assert [track.track_id for track in fresh] == [expected_id]
