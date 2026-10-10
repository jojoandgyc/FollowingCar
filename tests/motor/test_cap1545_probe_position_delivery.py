"""Unbound detector position consumes an old owner, never fabricates a UID.

This is consumer/actuator contract coverage with in-memory serial output.
The vision tests separately exercise independent appearance and association.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

from test_cap2087_position_delivery import scene, deliver, parse, position_row


def probe_row():
    row = position_row()
    row['raw_track_id'] = -1
    proof = row['assignment'].pop('low_score_position_evidence')
    proof.update(source='follow_only_detector_probe', track_id=-1,
                 reference_track_id=22, gallery_distance=.277,
                 center_jump_ratio=.084, area_similarity=.908)
    row['assignment'].update(uid=0, best_uid=1, reason='preferred_search_reacquire_rejected',
                             follow_only_position_evidence=proof)
    row['sample_metadata'].update(low_score_continuation=False, search_observation_only=False,
        quality_bbox_ok=True, source_detection_index=0, detector_confidence=.791)
    return row


def test_probe_keeps_actual_raw_separate_from_permission_owner():
    row = probe_row()
    before = deepcopy(row)
    position = parse([row])
    assert position.track_id == 22
    assert position.observed_track_id == -1
    assert position.source == 'follow_only_detector_probe'
    assert position.reference_capture == 2083 and position.capture == 2087
    assert row == before


def test_probe_updates_existing_pair_without_uid_depth_or_stop(scene):
    s = scene
    row = probe_row()
    s.owner._rknn_pipeline.tracker.last_identity_observations = [row]
    assert deliver(s)
    current = s.owner._short_follow.snapshot().plan
    assert current.yaw_capture_id == 2087
    assert current.left_rpm == current.right_rpm == s.original.base_rpm > 0
    assert s.owner._visual_identity_evidence is s.publication
    for field in ('depth_timestamp', 'capture_timestamp', 'expires_at', 'base_rpm', 'p_rpm', 'i_rpm'):
        assert getattr(current, field) == getattr(s.original, field)
    s.rt._service_short_follow()
    assert not s.driver.stops
    assert row['raw_track_id'] == -1 and row['uid'] == row['assignment']['uid'] == 0
    s.clock[0] = s.original.expires_at + .001
    s.rt._service_short_follow()
    assert s.driver.stops  # Position did not extend longitudinal authorization.


@pytest.mark.parametrize('field,value', [
    ('track_id', 22), ('reference_track_id', -1), ('reference_track_id', True),
    ('gallery_distance', .301), ('gallery_distance', float('nan')),
    ('center_jump_ratio', .201), ('area_similarity', .54),
    ('identity_authorized', True), ('learning_allowed', True),
    ('expires_at', 10.52), ('source', 'kalman_prediction'),
])
def test_unbound_proof_cannot_escape_permission_or_geometry_contract(field, value):
    row = probe_row()
    row['assignment']['follow_only_position_evidence'][field] = value
    assert parse([row]) is None


@pytest.mark.parametrize('field,value', [
    ('quality_bbox_ok', False),
    ('source_detection_index', None), ('detector_confidence', .49),
    ('capture_frame_id', 2086), ('capture_timestamp', 10.1),
])
def test_probe_requires_current_formal_detection_metadata(field, value):
    row = probe_row()
    row['sample_metadata'][field] = value
    assert parse([row]) is None


@pytest.mark.parametrize('failure', ['uid', 'owner', 'dual_proof', 'ambiguity', 'excluded', 'contradiction'])
def test_incompatible_probe_never_becomes_associated_position(failure):
    row = probe_row()
    rows = [row]
    if failure == 'uid': row['uid'] = 1
    elif failure == 'owner': row['assignment'].update(best_uid=2, mapped_uid=2)
    elif failure == 'dual_proof':
        row['assignment']['low_score_position_evidence'] = position_row()['assignment']['low_score_position_evidence']
    elif failure == 'ambiguity': rows.append(deepcopy(row))
    elif failure == 'excluded': row['assignment']['search_excluded'] = True
    elif failure == 'contradiction': row['assignment']['search_contradiction_retained'] = True
    assert parse(rows) is None


@pytest.mark.parametrize('search_released', [False, True])
def test_real_tracker_probe_then_low_score_reaches_paired_writer(monkeypatch, search_released):
    # Synthetic descriptors/geometry isolate the actual producer/consumer
    # interface. Raw camera feature vectors are not persisted in old logs.
    sys.path.append(str(Path(__file__).resolve().parents[1] / 'vision'))
    from test_cap1546_follow_position import ready, probe
    from test_provisional_association import update
    from test_short_follow_executor import short_runtime
    from car_control_modular.short_follow import ShortFollowConfig, ShortFollowController, ShortFollowObservation
    from car_control_modular.short_follow_adapter import ShortFollowAdapter
    from car_control_modular.detector_identity_lease import ValidatedVisualObservation, publish_visual_identity_evidence
    from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
    from tools.replay_cap334_recovery import gallery_snapshot
    import request_0513_modular as app

    tracker = ready()
    gallery = gallery_snapshot(tracker.identity_bank)
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    owner._follow_controller = FollowSafetyController(FollowPolicyConfig(direction_history_enable=True))
    owner._follow_controller.active_target_id = 1
    owner._action_runtime = rt
    owner._short_follow = ShortFollowController(ShortFollowConfig(enabled=True))
    owner._short_follow.activate(1, 3.19)
    owner._short_follow_adapter = ShortFollowAdapter(owner, owner._short_follow, rt.logger)
    clock[0] = 3.22
    proof = ValidatedVisualObservation(1, 3, 1542, 3.2, 3.22, 3.7, 'full')
    identity = publish_visual_identity_evidence(owner, observation=proof, lease=None)
    owner._active_capture_frame_id, owner._active_capture_timestamp = 1542, 3.2
    owner.frame_index = tracker._frame_index
    # Consume the real bank's completed proof, not a hand-built reason/flag.
    assert app.PersonTracker._completed_similar_follow_confirmation(owner,
        tracker.identity_bank.last_assignments[3], uid=1, track_id=3) is proof
    original = owner._short_follow.update(ShortFollowObservation(1, 1542, 3.2, 3.2, 2.2, .46875), clock[0])
    rt._service_short_follow()
    if search_released:
        tracker.set_search_reacquire_context(active_uid=1, searching=False, direction=None)
        tracker.set_detector_continuation_context(active_uid=1, allowed=True)
        tracker.deepsort.config = replace(tracker.deepsort.config, max_output_age=0)
        row, = update(tracker, 1545, 3.361, score=.791, distance=.277,
                      box=(300., 40., 500., 450.))
    else:
        row = probe(tracker, box=(300., 40., 500., 450.))
    assert row.track_id == -1 and row.reid_uid == 0
    owner._rknn_pipeline = SimpleNamespace(tracker=tracker, last_frame_width=640, last_frame_height=480,
        last_identity_processing=dict(mode='full', full_features_current=True,
            capture_frame_id=1545, capture_timestamp=3.361))
    clock[0] = 3.39
    assert app.PersonTracker._update_detector_identity_lease(owner, [row], 1545, 3.361,
        now=clock[0], stale=False)
    first = owner._short_follow.snapshot().plan
    assert owner._visual_identity_evidence is identity
    assert first.yaw_capture_id == 1545 and first.left_rpm > first.right_rpm > 0
    rt._service_short_follow()
    assert not driver.stops
    records = update(tracker, 1546, 3.397, score=.422, distance=.287, box=(300., 40., 500., 450.))
    assert not records
    owner._rknn_pipeline.last_identity_processing.update(capture_frame_id=1546, capture_timestamp=3.397)
    clock[0] = 3.43
    assert app.PersonTracker._update_detector_identity_lease(owner, records, 1546, 3.397,
        now=clock[0], stale=False)
    second = owner._short_follow.snapshot().plan
    assert second.yaw_capture_id == 1546 and second.left_rpm > second.right_rpm > 0
    assert second.expires_at == original.expires_at
    assert owner._visual_identity_evidence is identity
    rt._service_short_follow()
    assert not driver.stops and gallery_snapshot(tracker.identity_bank) == gallery
