"""Real identity publication + paired adapter + fake motor: no weak-frame STOP.

The vision tests validate generation from a real tracker. These tests exercise
the consumer boundary and execute complete wheel pairs without hardware.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import request_0513_modular as app
from car_control_modular.associated_position import current_associated_position
from car_control_modular.control_types import SensorFrame
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.detector_identity_lease import ValidatedVisualObservation, publish_visual_identity_evidence
from car_control_modular.short_follow import ShortFollowObservation
from car_control_modular.short_follow_adapter import ShortFollowAdapter
from test_short_follow_executor import short_runtime


BOX = (114.333404541, 2.66744995, 433.2265625, 476.32910)


def position_row():
    proof = dict(uid=1, track_id=22, capture_frame_id=2087, capture_timestamp=10.215,
        bbox=BOX, reference_capture_frame_id=2083, reference_capture_timestamp=10.01,
        expires_at=10.51, appearance_distance=.093, center_jump_ratio=.1734,
        area_similarity=.98, source='low_score_existing_track',
        identity_authorized=False, learning_allowed=False)
    return dict(raw_track_id=22, uid=0, detector_bbox=BOX,
        assignment=dict(uid=0, mapped_uid=1, reason='low_score_observation_only',
            low_score_position_evidence=proof),
        sample_metadata=dict(low_score_continuation=True, is_fresh=True,
            capture_frame_id=2087, capture_timestamp=10.215, detector_confidence=.4483))


def parse(rows, **kwargs):
    args = dict(uid=1, capture=2087, timestamp=10.215, width=640, height=480, now=10.24)
    args.update(kwargs)
    return current_associated_position(rows, **args)


@pytest.fixture
def scene(monkeypatch):
    rt, owner, driver, symbols, clock = short_runtime(monkeypatch)
    ctl = FollowSafetyController(FollowPolicyConfig(direction_history_enable=True, lost_confirm_frames=3))
    ctl.active_target_id, ctl._has_seen_person = 1, True
    owner._follow_controller = ctl
    owner._action_runtime = rt
    owner._short_follow_adapter = ShortFollowAdapter(owner, owner._short_follow, rt.logger)
    clock[0] = 10.01
    proof = ValidatedVisualObservation(1, 22, 2083, 10.01, 10.02, 10.51, 'full')
    publication = publish_visual_identity_evidence(owner, observation=proof, lease=None)
    ctl._target_direction_history.record_visible(2083, 10.01, target_id=1,
        frame_width=640, confidence=.885, bbox=(256.6964, 0., 564.3388, 479.))
    original = owner._short_follow.update(ShortFollowObservation(1, 2083, 10.01,
        10.01, 2.2, .6192886), 10.02)
    clock[0] = 10.03
    rt._service_short_follow()
    assert not driver.stops
    owner._rknn_pipeline = SimpleNamespace(last_frame_width=640, last_frame_height=480,
        last_identity_processing=dict(mode='full', full_features_current=True,
            capture_frame_id=2087, capture_timestamp=10.215),
        tracker=SimpleNamespace(last_identity_observations=[position_row()]))
    clock[0] = 10.24
    return SimpleNamespace(rt=rt, owner=owner, driver=driver, clock=clock,
        ctl=ctl, publication=publication, original=original)


def deliver(s):
    return app.PersonTracker._update_detector_identity_lease(s.owner, [], 2087,
        10.215, now=s.clock[0], stale=False)


def test_current_low_score_centers_existing_forward_without_identity_or_depth_renewal(scene):
    s = scene
    assert deliver(s)
    current = s.owner._short_follow.snapshot().plan
    assert current.yaw_capture_id == 2087
    assert current.left_rpm == current.right_rpm == s.original.base_rpm > 0
    assert s.owner._visual_identity_evidence is s.publication
    for field in ('depth_timestamp', 'capture_timestamp', 'expires_at', 'p_rpm', 'i_rpm', 'base_rpm'):
        assert getattr(current, field) == getattr(s.original, field)
    s.rt._service_short_follow()
    assert not s.driver.stops
    assert s.driver.pairs[-1] == (current.base_rpm, -current.base_rpm)


def test_cap2087_real_delivery_changes_later_search_to_left_without_trusted_history_rewrite(scene):
    s = scene
    assert deliver(s)
    assert s.ctl._target_direction_history.latest_reliable_side().direction == 'right'
    s.clock[0] = 10.48
    s.ctl.lost_confirm_frames = 3
    s.ctl._capture_lost_exit_direction(SensorFrame(width=640, height=480,
        capture_frame_id=2091, capture_timestamp=10.45), search_entry=True)
    assert s.ctl._lost_exit_direction == 'left'
    assert s.ctl._lost_hint_source == 'associated_low_score_position'
    assert not s.driver.stops  # Direction recording itself sends nothing.


def test_weak_positions_do_not_keep_forward_running_after_its_original_deadline(scene):
    s = scene
    assert deliver(s)
    s.clock[0] = s.original.expires_at + .001
    s.rt._service_short_follow()
    assert s.driver.stops
    assert s.owner._visual_identity_evidence is s.publication


def test_duplicate_diagnostic_cannot_update_again_or_extend_deadline(scene):
    s = scene
    assert deliver(s)
    current = s.owner._short_follow.snapshot().plan
    assert not deliver(s)
    assert s.owner._short_follow.snapshot().plan is current
    assert s.ctl._limited_yaw_direction.expires_at == 10.51


@pytest.mark.parametrize('fault', ['missing', 'ordinary_diagnostic', 'expired', 'wrong_track', 'stop', 'future'])
def test_failed_position_update_preserves_original_pair_without_new_zero(scene, fault):
    s = scene
    row = s.owner._rknn_pipeline.tracker.last_identity_observations[0]
    if fault == 'missing': s.owner._rknn_pipeline.tracker.last_identity_observations = []
    elif fault == 'ordinary_diagnostic': row['assignment'].pop('low_score_position_evidence')
    elif fault == 'expired': row['assignment']['low_score_position_evidence']['expires_at'] = 10.23
    elif fault == 'wrong_track': row['assignment']['low_score_position_evidence']['track_id'] = 23
    elif fault == 'stop': s.owner._explicit_stop_requested = True
    elif fault == 'future': row['assignment']['low_score_position_evidence']['capture_timestamp'] = 10.3
    assert deliver(s)
    assert s.owner._short_follow.snapshot().plan is s.original
    assert s.ctl._limited_yaw_direction is None
    assert not s.driver.stops


@pytest.mark.parametrize('field,value', [
    ('source', 'kalman'), ('identity_authorized', True), ('learning_allowed', True),
    ('uid', 2), ('track_id', -1), ('capture_frame_id', 2086),
    ('capture_timestamp', 10.1), ('reference_capture_frame_id', 2087),
    ('reference_capture_timestamp', 10.215), ('expires_at', 10.52),
    ('expires_at', float('nan')), ('bbox', (-1., 0., 300., 479.)),
    ('bbox', (0., 0., 650., 479.)),
])
def test_invalid_position_contract_does_not_authorize_anything(field, value):
    row = position_row()
    row['assignment']['low_score_position_evidence'][field] = value
    assert parse([row]) is None


def test_duplicate_candidate_evidence_is_ambiguous():
    row = position_row()
    assert parse([row]) is not None
    assert parse([row, deepcopy(row)]) is None


@pytest.mark.parametrize('reason', ['low_score_observation_only', 'similar_follow_observe'])
def test_observation_reason_does_not_hide_independently_validated_position(reason):
    row = position_row()
    row['assignment']['reason'] = reason
    assert parse([row]) is not None
    row['assignment'].pop('low_score_position_evidence')
    assert parse([row]) is None  # A label without the association proof is not permission.


@pytest.mark.parametrize('field', ['assignment', 'sample_metadata'])
def test_malformed_diagnostic_metadata_is_rejected(field):
    row = position_row()
    row[field] = 'invalid'
    assert parse([row]) is None


def next_identity_frame(s, records):
    s.clock[0] = 10.29
    s.owner._rknn_pipeline.last_identity_processing.update(
        capture_frame_id=2088, capture_timestamp=10.26)
    return app.PersonTracker._update_detector_identity_lease(s.owner, records,
        2088, 10.26, now=s.clock[0], stale=False)


def test_background_formal_record_does_not_erase_newer_target_side(scene):
    s = scene
    assert deliver(s)
    hint = s.ctl._limited_yaw_direction
    assert next_identity_frame(s, [SimpleNamespace(track_id=90, reid_uid=2,
        time_since_update=0, class_id=0)])
    assert s.ctl._limited_yaw_direction is hint
    assert not s.driver.stops


def test_new_formal_rejection_of_same_target_clears_weak_history(scene):
    s = scene
    assert deliver(s)
    assert next_identity_frame(s, [SimpleNamespace(track_id=22, reid_uid=0,
        time_since_update=0, class_id=0)])
    assert s.ctl._limited_yaw_direction is None


@pytest.mark.parametrize('contradiction', ['competition_failed', 'geometry_revoked', 'search_excluded'])
def test_reliable_tracker_veto_clears_hint_even_when_formal_records_are_empty(scene, contradiction):
    s = scene
    assert deliver(s)
    pair = s.owner._short_follow.snapshot().plan
    calls = []
    def review(uid, raw, cap, stamp):
        calls.append((uid, raw, cap, stamp))
        return contradiction
    s.owner._rknn_pipeline.tracker.associated_position_contradiction = review
    assert next_identity_frame(s, [])
    assert calls == [(1, 22, 2088, 10.26)]
    assert s.ctl._limited_yaw_direction is None
    # This notification only retires direction history. Existing identity
    # and motor-safety owners still handle actual revocation, not a new STOP.
    assert s.owner._short_follow.snapshot().plan is pair
    assert not s.driver.stops


@pytest.mark.parametrize('flag', ['_explicit_stop_requested', '_runtime_shutdown_requested'])
def test_hard_stop_clears_direction_even_on_duplicate_callback(scene, flag):
    s = scene
    assert deliver(s)
    setattr(s.owner, flag, True)
    assert not deliver(s)  # Duplicate capture still cancels the historical hint.
    assert s.ctl._limited_yaw_direction is None


def test_new_identity_publication_rejects_prepared_position_without_zero(scene):
    s = scene
    position = parse([position_row()])
    publish_visual_identity_evidence(s.owner, observation=False, lease=False)
    assert s.owner._short_follow_adapter.publish_associated_lateral(position, 640,
        identity_publication=s.publication, now=s.clock[0]) is None
    assert s.owner._short_follow.snapshot().plan is s.original
    assert not s.driver.stops


@pytest.mark.parametrize('distance', [1.031, 2.2])
def test_real_tracker_proof_reaches_main_adapter_motor_and_search(monkeypatch, distance):
    # Recorded detector boxes/scores and turn increment; embeddings are
    # constructed from the logged distance because original vectors were not
    # persisted. This tests control wiring, not full model replay accuracy.
    import numpy as np
    from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
    from rk_vision.yolo11 import Detection
    from tools.replay_cap334_recovery import gallery_snapshot
    box2083 = (232.2148132324, 2.288696289, 560.474609375, 471.561767578)
    tracker = DeepSortTracker(DeepSortTrackerConfig(n_init=1, bbox_expand_scale=1.,
        hfov_deg=60., identity_new_confirm_frames=1, identity_update_interval=1))
    tracker.set_detector_continuation_context(active_uid=1, allowed=True)
    def tracked(cap, stamp, box, score, distance, yaw):
        feature = np.array([1.-distance, np.sqrt(1.-(1.-distance)**2)], dtype='float32')
        return tracker.update([Detection(box, score, 0)], [feature], image_width=640,
            image_height=480, frame_context=dict(capture_frame_id=cap,
                capture_timestamp=stamp, integrated_yaw_deg=yaw))
    for cap, stamp in ((2077, 10.), (2079, 10.05), (2081, 10.10), (2083, 10.15)):
        tracked(cap, stamp, box2083, .8853573, 0., 142.251944922)
    gallery = gallery_snapshot(tracker.identity_bank)
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    ctl = FollowSafetyController(FollowPolicyConfig(direction_history_enable=True, lost_confirm_frames=3))
    ctl.active_target_id, ctl._has_seen_person = 1, True
    owner._follow_controller = ctl
    owner._action_runtime = rt
    owner._short_follow_adapter = ShortFollowAdapter(owner, owner._short_follow, rt.logger)
    ctl._target_direction_history.record_visible(2083, 10.15, target_id=1,
        frame_width=640, confidence=.885, bbox=box2083)
    clock[0] = 10.18
    proof = ValidatedVisualObservation(1, 1, 2083, 10.15, 10.18, 10.65, 'full')
    identity = publish_visual_identity_evidence(owner, observation=proof, lease=None)
    original = owner._short_follow.update(ShortFollowObservation(1, 2083, 10.15,
        10.15, distance, .6192886), clock[0])
    rt._service_short_follow()
    stamp = 10.355166410
    assert tracked(2087, stamp, BOX, .4483298957, .0926066637, 143.337199323) == []
    owner._rknn_pipeline = SimpleNamespace(last_frame_width=640, last_frame_height=480,
        tracker=tracker, last_identity_processing=dict(mode='full', full_features_current=True,
            capture_frame_id=2087, capture_timestamp=stamp))
    clock[0] = 10.37
    assert app.PersonTracker._update_detector_identity_lease(owner, [], 2087, stamp,
        now=clock[0], stale=False)
    plan = owner._short_follow.snapshot().plan
    assert plan.yaw_capture_id == 2087
    assert plan.left_rpm == plan.right_rpm == original.base_rpm
    assert bool(plan.base_rpm > 0) == (distance > 1.5)
    assert plan.expires_at == original.expires_at
    assert owner._visual_identity_evidence is identity
    rt._service_short_follow()
    if distance > 1.5:
        assert not driver.stops
    else:
        assert driver.stops  # Real near-distance STOP is independent of direction history.
    assert gallery_snapshot(tracker.identity_bank) == gallery
    clock[0] = 10.75  # Search may begin after the weak-position admission lease.
    ctl.lost_confirm_frames = 3
    ctl._capture_lost_exit_direction(SensorFrame(width=640, height=480,
        capture_frame_id=2091, capture_timestamp=10.62), search_entry=True)
    assert ctl._lost_exit_direction == 'left'
    assert ctl._lost_hint_source == 'associated_low_score_position'
    tracker.identity_bank._geometry_revoked_uids[1] = tracker._frame_index
    owner._rknn_pipeline.last_identity_processing.update(
        capture_frame_id=2092, capture_timestamp=10.72)
    clock[0] = 10.76
    assert app.PersonTracker._update_detector_identity_lease(owner, [], 2092,
        10.72, now=clock[0], stale=False)
    assert ctl._limited_yaw_direction is None
