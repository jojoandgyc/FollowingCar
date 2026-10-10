"""CAP1522 old-left -> 1523..28 right association; no motor/vision hardware."""
from dataclasses import replace

import pytest

from car_control_modular.control_types import SensorFrame
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.historical_direction_backfill import HistoricalDirectionCandidate


@pytest.fixture
def scene(monkeypatch):
    clock = [10.50]
    monkeypatch.setattr('car_control_modular.controllers.time.monotonic', lambda: clock[0])
    controller = FollowSafetyController(FollowPolicyConfig(
        direction_history_enable=True, lost_confirm_frames=3, search_cooldown=0))
    controller.active_target_id = 1
    controller._has_seen_person = True
    controller._direction_loss_capture_id = 1529
    controller._direction_latest_visible_capture_id = 1522
    controller._target_direction_history.record_visible(
        1522, 10., target_id=1, frame_width=640, confidence=.84,
        bbox=(133.7764892578125,4.241455078125,409.14056396484375,474.23162841796875),
        vehicle_yaw_deg=-33.4)
    # Only the CAP1522 anchor box is logged. The intermediate worker boxes
    # were not persisted: use a synthetic continuous path, NOT a six-frame
    # recording replay. Actual persisted timestamps/boxes are covered below.
    samples = tuple(HistoricalDirectionCandidate(
        capture_frame_id=cap, timestamp=stamp, state='visible',
        bbox=(center*640-138,4.,center*640+138,474.), score=.90,
        frame_width=640, candidate_count=1, source='detector_formal_person_side',
        vehicle_yaw_deg=-33.4-(cap-1522)*.6)
        for cap, stamp, center in ((1523,10.068,.47),(1524,10.119,.53),
            (1525,10.175,.59),(1526,10.203,.625),(1527,10.258,.66),(1528,10.310,.685)))
    return controller, clock, samples


def hint(controller, samples, *, associate=True, **changes):
    args = dict(active_target_id=1, first_capture_frame_id=1523,
        last_capture_frame_id=1528, selected_capture_frame_ids=tuple(range(1523,1529)),
        confidence=.70, loss_capture_frame_id=1529, evidence_timestamp=10.068)
    args.update(changes)
    if associate:
        args['association_candidates'] = samples
    return controller.note_historical_direction_hint('right', **args)


def frame(cap=1532, stamp=10.5):
    return SensorFrame(width=640, height=480, capture_frame_id=cap, capture_timestamp=stamp)


def test_new_associated_position_beats_old_left_at_search_entry(scene):
    c, _, samples = scene
    assert hint(c, samples)
    assert c._historical_direction_hint['association'].anchor_capture_frame_id == 1522
    c.lost_confirm_frames = 3
    c._capture_lost_exit_direction(frame(), search_entry=True)
    assert c._lost_exit_direction == 'right'
    assert c._lost_hint_source == 'associated_historical_position'
    c._ensure_search_state(frame())
    assert c.search_state == 'searching' and c.search_direction == 'right'
    assert c._fallback_search_action(frame(), 'search').kind == 'rotate_right'
    # A direction hint did not turn any detector-only slot into target history.
    assert c._target_direction_history.latest_visible_evidence().capture_frame_id == 1522
    assert c._direction_latest_visible_capture_id == 1522
    assert c.active_target_id == 1


def test_same_selector_drives_loss_confirmation_without_stop_or_forward(scene):
    c, clock, samples = scene
    assert hint(c, samples)
    results = []
    for cap, stamp in ((1529,10.50),(1530,10.55),(1532,10.60)):
        clock[0] = stamp
        result = c.decide(cap, frame(cap, stamp))
        results.append(result)
        assert [action.kind for action in result.actions] == ['rotate_right']
        assert not result.explicit_stop_requested
        assert not result.stop_action_execution
    assert results[0].evidence_capture_frame_id == 1528
    assert results[-1].reason == 'search_right'
    assert c._target_direction_history.latest_visible_evidence().capture_frame_id == 1522


def test_late_chain_corrects_active_search_once_without_resetting_time_or_coverage(scene):
    c, clock, samples = scene
    c.search_state, c.search_direction = 'searching', 'left'
    c._lost_exit_direction, c._lost_hint_source = 'left', 'latest_reliable_capture_side'
    c._search_rotation_started_at = 10.1
    c._search_heading_from_loss_deg = -14.
    c._search_heading_min_deg, c._search_heading_max_deg = -14., 0.
    c._lost_started_at = 10.1
    assert hint(c, samples)
    c._ensure_search_state(frame())
    assert c.search_direction == c._lost_exit_direction == 'right'
    assert c._historical_direction_applied_capture_id == 1528
    for now in (10.55, 10.79, 11.):
        clock[0] = now
        c._ensure_search_state(frame(1535, now))
        assert c.search_direction == 'right'
        assert c._search_rotation_started_at == c._lost_started_at == 10.1
        assert c._search_heading_from_loss_deg == -14.
        assert (c._search_heading_min_deg,c._search_heading_max_deg) == (-14.,0.)


@pytest.mark.parametrize('condition', ['new_visible','other_uid','other_loss','hard_conflict','expired','search_braked'])
def test_new_evidence_or_episode_revocation_invalidates_association(scene, condition):
    c, clock, samples = scene
    assert hint(c, samples)
    if condition == 'new_visible':
        c._target_direction_history.record_visible(1530,10.45,target_id=1,
            bbox=(60.,4.,220.,474.),frame_width=640,confidence=.90)
        c._direction_latest_visible_capture_id = 1530
    elif condition == 'other_uid': c.active_target_id = 2
    elif condition == 'other_loss': c._direction_loss_capture_id = 1550
    elif condition == 'hard_conflict': c.clear_historical_direction_hint('identity_conflict')
    elif condition == 'expired': clock[0] = 10.769
    else: c._target_direction_history.discard_through(10.32)
    decision = c._latest_lateral_direction_side()
    assert decision.reason != 'associated_historical_position'
    c.lost_confirm_frames = 3
    c._capture_lost_exit_direction(frame(), search_entry=True)
    assert c._lost_hint_source != 'associated_historical_position'


def test_generic_hint_is_still_not_allowed_to_overwrite_formal_anchor(scene):
    c, _, samples = scene
    assert hint(c, samples, associate=False)
    c.lost_confirm_frames = 3
    c._capture_lost_exit_direction(frame(), search_entry=True)
    assert c._lost_exit_direction == 'left'
    assert c._lost_hint_source == 'latest_reliable_capture_side'


@pytest.mark.parametrize('condition', ['diagnostic','source_missing','weak_score','competition','zero_count','bool_count',
    'unknown','repeated_capture','repeated_timestamp','future','not_finite','large_jump',
    'small_area','different_resolution','large_capture_gap','large_time_gap','unselected_ambiguous',
    'invalid_yaw','large_yaw','wrong_direction','wrong_timestamp_origin'])
def test_unverified_or_discontinuous_chain_cannot_override_anchor(scene, condition):
    c, _, samples = scene
    data = list(samples)
    params = {}
    if condition == 'diagnostic': data[2] = replace(data[2],source='detector_diagnostic_person_side')
    elif condition == 'source_missing': data[2] = replace(data[2],source='')
    elif condition == 'weak_score': data[2] = replace(data[2],score=.49)
    elif condition == 'competition': data[2] = replace(data[2],candidate_count=2)
    elif condition == 'zero_count': data[2] = replace(data[2],candidate_count=0)
    elif condition == 'bool_count': data[2] = replace(data[2],candidate_count=True)
    elif condition == 'unknown': data[2] = replace(data[2],state='unknown')
    elif condition == 'repeated_capture': data[2] = replace(data[2],capture_frame_id=1524)
    elif condition == 'repeated_timestamp': data[2] = replace(data[2],timestamp=data[1].timestamp)
    elif condition == 'future': data[-1] = replace(data[-1],timestamp=10.6)
    elif condition == 'not_finite': data[2] = replace(data[2],timestamp=float('nan'))
    elif condition == 'large_jump': data[2] = replace(data[2],bbox=(0.,4.,150.,474.))
    elif condition == 'small_area': data[2] = replace(data[2],bbox=(300.,200.,325.,250.))
    elif condition == 'different_resolution': data[2] = replace(data[2],frame_width=1280)
    elif condition == 'large_capture_gap':
        c._target_direction_history._visible_entries[-1] = replace(
            c._target_direction_history.latest_visible_evidence(),capture_frame_id=1510)
    elif condition == 'large_time_gap': data[0] = replace(data[0],timestamp=10.251)
    elif condition == 'unselected_ambiguous':
        params['selected_capture_frame_ids'] = (1523,1524,1526,1527,1528)
        data[2] = replace(data[2],candidate_count=2)
    elif condition == 'invalid_yaw': data[2] = replace(data[2],vehicle_yaw_deg=float('nan'))
    elif condition == 'large_yaw': data[2] = replace(data[2],vehicle_yaw_deg=30.)
    elif condition == 'wrong_direction':
        data = [replace(item,bbox=(100.,4.,375.,474.)) for item in data]
    else: params['evidence_timestamp'] = 10.07
    assert hint(c, tuple(data), **params)
    assert c._historical_direction_hint.get('association') is None
    assert c._latest_lateral_direction_side().direction == 'left'


def test_missing_yaw_does_not_invent_compensation_or_discard_close_geometry(scene):
    c, _, samples = scene
    assert hint(c, tuple(replace(item,vehicle_yaw_deg=None) for item in samples))
    assert c._latest_lateral_direction_side().direction == 'right'


def test_confirmed_candidate_direction_is_not_overwritten_by_late_chain(scene):
    c, _, samples = scene
    c.search_state, c.search_direction = 'searching','left'
    c._lost_exit_direction,c._lost_hint_source = 'left','search_candidate_last_left'
    assert hint(c,samples)
    c._ensure_search_state(frame())
    assert c.search_direction == 'left'


def test_repeated_hint_consumption_cannot_renew_capture_age(scene):
    c, clock, samples = scene
    assert hint(c,samples)
    chain = c._historical_direction_hint['association']
    for now in (10.51,10.6,10.70):
        clock[0] = now
        assert c._latest_lateral_direction_side().direction == 'right'
        assert c._historical_direction_hint['association'] is chain
        assert c._historical_direction_hint['evidence_timestamp'] == 10.068
    clock[0] = 10.77
    assert c._latest_lateral_direction_side().direction == 'left'
    assert c._historical_direction_hint is None


def test_actual_logged_anchor_and_newer_boxes_do_not_fabricate_missing_chain(scene):
    """174853 log L20140/20190/20226: no 1523..28 raw boxes were saved.

    The recorded right boxes agree, but a missing 364ms / 7-CAP first link
    must not be promoted merely by pretending those unrecorded slots exist.
    Source below is a simulated independent-worker provenance; geometry,
    score, yaw and sampling times are exact main-detector log values.
    """
    c, clock, _ = scene
    c._target_direction_history.record_visible(
        1522,26148.372864892,target_id=1,frame_width=640,
        confidence=.8439151048660278,
        bbox=(133.7764892578125,4.241455078125,409.14056396484375,474.23162841796875),
        vehicle_yaw_deg=-33.55528926327805)
    c._direction_loss_capture_id = 1532
    clock[0] = 26149.009
    samples = (
        HistoricalDirectionCandidate(1529,26148.73699185,'visible',
            (339.3238525390625,6.5639495849609375,559.8502197265625,476.5303955078125),
            .9079622030258179,640,1,'detector_formal_person_side',-37.2191347598427),
        HistoricalDirectionCandidate(1530,26148.771941962,'visible',
            (356.29791259765625,5.7322540283203125,573.0664672851562,477.4693603515625),
            .8928923010826111,640,1,'detector_formal_person_side',-37.28469789577643),
    )
    assert hint(c,samples,first_capture_frame_id=1529,last_capture_frame_id=1530,
        selected_capture_frame_ids=(1529,1530),loss_capture_frame_id=1532,
        evidence_timestamp=26148.73699185)
    assert c._historical_direction_hint.get('association') is None
    assert c._historical_direction_hint['association_reason'] == 'incomplete_or_unqualified_anchor_chain'
    assert c._latest_lateral_direction_side().direction == 'left'


def test_association_permission_is_explained_in_existing_ready_log(scene,caplog):
    c, _, samples = scene
    with caplog.at_level('INFO'):
        assert hint(c,samples,associate=False)
        assert hint(c,samples)
    assert 'association_reason=metadata_not_supplied' in caplog.text
    assert 'association_reason=anchor_associated_position' in caplog.text
