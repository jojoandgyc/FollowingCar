"""Offline replay validation; synthetic fixtures are not recorded embeddings."""
import copy
import json
import logging

import numpy as np
import pytest

from rk_vision.identity_bank import _geometry_observation
from tools import replay_cap606_continuity as replay


def event(cap, stamp, *, raw=4, weak=False, full=.4, search=True):
    seed = cap in replay.REFERENCES
    box = [470., 0., 639., 479.] if weak else [450., 1., 620., 478.]
    m = dict(track_id=raw, capture_frame_id=cap, capture_timestamp=stamp,
        frame_index=cap, detector_bbox=box, bbox=box, quality_bbox=box,
        image_width=640, image_height=480, is_fresh=True,
        detector_confidence=.94, quality_bbox_ok=not weak,
        quality_bbox_reason='edge_touch>2' if weak else '',
        bbox_quality_reason='edge_touch>2' if weak else '',
        bbox_quality_tier='weak' if weak else 'strong',
        detector_center_x_ratio=(box[0]+box[2])/1280.,
        detector_area_ratio=(box[2]-box[0])*(box[3]-box[1])/(640*480),
        detector_edge_touch_count=3 if weak else 2, edge_touch_count=3 if weak else 2,
        integrated_yaw_deg=0., yaw_rate_dps=0., candidate_count=1,
        source_detection_index=0, partial_observation=True,
        partial_feature_source='osnet_torso',
        template_learning_risk=dict(observed=True, risky=False, reason='clear'),
        search_reacquire_context_active=search, search_direction='left' if search else None,
        search_direction_compatible=True if raw == 4 else False,
        identity_competition=dict(uid=1, frame_index=cap, candidate_count=1,
            source_detection_index=0, passed=True, reason='single_candidate', distance=full))
    a = dict(bank_updated=seed, reason='created_confirmed' if seed else 'not_replayed',
             match_evidence=dict(distance=full), partial_distance=.196 if seed else .3,
             protected_search_anchor_cap=424)
    return dict(capture_frame_id=cap, capture_timestamp=stamp, frame_index=cap,
        raw_track_id=raw, uid=1 if seed else 0, detector_bbox=box, sample_metadata=m, assignment=a)


@pytest.fixture
def fixture_events():
    events = {cap: event(cap, stamp, **extra) for cap, stamp, extra in [
        (21, 1., dict(raw=1, full=0.)), (27, 1.1, dict(raw=1, full=.024)),
        (605, 40., {}), (606, 40.1, {}), (614, 40.2, dict(weak=True)),
        (675, 40.4, dict(raw=6, search=False)), (691, 40.5, dict(raw=6))]}
    anchor = _geometry_observation(event(424, 20., raw=1)['sample_metadata'], 424)
    return events, anchor


def config():
    return replay.bank_config(replay.ROOT/'car_control_modular/config/reid_runtime.ini')


def test_synthetic_distance_construction_has_only_claimed_reference_distances():
    query = replay.vectors_for_distances(.024, .4, .38)
    ref = replay.vectors_for_distances(.024, .024, 0.)
    assert np.linalg.norm(query) == pytest.approx(1.)
    assert 1.-query[0] == pytest.approx(.4)
    assert 1.-query.dot(ref) == pytest.approx(.38, abs=1e-6)


@pytest.mark.parametrize('values', [(float('nan'), .4, .4), (.024, None, .4),
                                   (0., .4, .4), (.024, 0., 1.)])
def test_impossible_or_missing_appearance_is_never_invented(values):
    with pytest.raises(ValueError):
        replay.vectors_for_distances(*values)


def test_recorded_competition_supplies_unmapped_raw_full_score_but_not_missing_torso():
    e = event(675, 40.4, raw=6)
    e['assignment'].update(match_evidence=None, partial_distance=None)
    full, torso, individuals = replay.appearance(e, .024, .196)
    assert replay.recorded_full(e) == .4
    assert torso is None and individuals == []
    assert 1.-full[0] == pytest.approx(.4)


def test_real_assign_replay_changes_crop_and_raw_handoff_without_gallery_writes(fixture_events):
    events, anchor = fixture_events
    original = copy.deepcopy(events)
    with replay.patch('rk_vision.identity_bank.cropped_follow_continuous', return_value=False):
        before = replay.replay(config(), events, anchor, previous=True)
    after = replay.replay(config(), events, anchor)
    assert before['accepted_caps'] == [606]
    assert after['accepted_caps'] == [606, 614, 675, 691]
    assert before['gallery_write_caps'] == after['gallery_write_caps'] == []
    assert after['gallery_before'] == after['gallery_after']
    assert events == original
    assert after['rows'][3]['similar_follow']['handoff_from_track_id'] == 4


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'unapproved', 'clock', 'raw', 'score'])
def test_loader_rejects_ambiguous_or_incomplete_inputs(tmp_path, fixture_events, mutation):
    events, anchor = fixture_events
    events[605]['assignment']['reacquire_geometry'] = dict(reference=anchor)
    if mutation == 'missing': events.pop(27)
    elif mutation == 'unapproved': events[21]['assignment']['bank_updated'] = False
    elif mutation == 'clock': events[691]['capture_timestamp'] = 40.
    elif mutation == 'raw': events[675]['raw_track_id'] = 9
    elif mutation == 'score':
        events[675]['assignment']['match_evidence'] = None
        events[675]['sample_metadata']['identity_competition'].pop('distance')
    rows = list(events.values())
    if mutation == 'duplicate': rows.append(copy.deepcopy(rows[-1]))
    directory = tmp_path/'reid_diagnostics'
    directory.mkdir()
    (directory/'events.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError):
        replay.load_events(tmp_path)


def test_saved_recording_scores_match_historical_decisions_and_bound_improvement():
    run_dir = replay.ROOT/'run_request_0428_modular_logs/run_20261010_141437_30672_90f0536c'
    if not (run_dir/'reid_diagnostics/events.jsonl').is_file():
        pytest.skip('local saved recording is not part of the repository')
    previous_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        report = replay.report(run_dir, replay.ROOT/'car_control_modular/config/reid_runtime.ini')
    finally:
        logging.disable(previous_disable)
    assert report['recorded_accepted_count'] == report['baseline']['accepted_count'] == 12
    assert report['baseline_recorded_uid_agreement'] == 32
    assert report['current']['accepted_count'] == 28
    assert report['missing_torso_caps'] == [675, 677, 679]
    assert report['current']['per_raw']['5'] == dict(observed=12, accepted=9)
    for mode in ('baseline', 'current'):
        assert report[mode]['gallery_write_caps'] == []
        assert report[mode]['gallery_before'] == report[mode]['gallery_after']
