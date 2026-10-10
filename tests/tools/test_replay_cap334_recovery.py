"""Synthetic-distance, real-assign replay; no camera, inference, or motor I/O."""
import copy
import json

import pytest

from rk_vision.identity_bank import IdentityBank
from tools import replay_cap334_recovery as replay


def event(cap, *, x=None, full=.43):
    frame, stamp = cap // 2, 100. + cap * .05
    seed = cap == replay.REFERENCE
    if x is None:
        x = 70. if cap <= 355 else 340.
    box = [61., 2., 493., 473.] if seed else [x, 20., x+120., 430.]
    metadata = dict(track_id=1 if cap < 334 else 3,
        frame_index=frame, capture_frame_id=cap, capture_timestamp=stamp,
        detector_bbox=box, quality_bbox=box, bbox=box, image_width=640,
        image_height=480, detector_center_x_ratio=(box[0]+box[2])/1280.,
        detector_area_ratio=(box[2]-box[0])*(box[3]-box[1])/(640*480),
        detector_edge_touch_count=2 if seed else 0,
        detector_confidence=.94, quality_bbox_ok=True, bbox_quality_tier='strong',
        is_fresh=True, partial_observation=True, partial_feature_source='osnet_torso',
        integrated_yaw_deg=0., yaw_rate_dps=0., candidate_count=1,
        source_detection_index=0, search_reacquire_context_active=cap >= 334,
        search_direction='left' if cap >= 334 else None,
        search_direction_compatible=x < 320. if cap >= 334 else None,
        template_learning_risk=dict(observed=True, risky=False, reason='clear'),
        identity_competition=dict(uid=1, frame_index=frame, candidate_count=1,
            source_detection_index=0, passed=True, reason='single_candidate'))
    assignment = dict(bank_updated=seed, reason='created_confirmed' if seed else
        'skip_update_distance' if cap == replay.CHECKPOINT else 'secondary_evidence_unavailable',
        learning_written_tiers=['recent_strong', 'recent_partial'] if seed else [],
        reacquire_reference_age_sec=stamp-(100. + 271*.05),
        template_recent_evidence=dict(distance=full), partial_distance=.36)
    return dict(capture_frame_id=cap, frame_index=frame, capture_timestamp=stamp,
        raw_track_id=1 if cap < 334 else 3, uid=1 if cap < 334 else 0,
        detector_bbox=box, sample_metadata=metadata, assignment=assignment)


@pytest.fixture
def episode():
    caps = (19, 271, 334, 336, 338, 374, 376, 388)
    return {cap: event(cap) for cap in caps}


@pytest.fixture
def config():
    return replay.bank_config(replay.ROOT/'car_control_modular/config/reid_runtime.ini')


def write_events(tmp_path, events):
    folder = tmp_path/'reid_diagnostics'
    folder.mkdir()
    (folder/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    return tmp_path


def test_loads_only_approved_initial_pair_and_accepted_checkpoint(tmp_path, episode):
    loaded = replay.load_events(write_events(tmp_path, episode.values()))
    assert loaded == episode


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'unapproved', 'torso', 'checkpoint', 'capture', 'reference_age'])
def test_loader_refuses_wrong_or_ambiguous_recording(tmp_path, episode, mutation):
    if mutation == 'missing':
        episode.pop(388)
    elif mutation == 'unapproved':
        episode[19]['assignment']['bank_updated'] = False
    elif mutation == 'torso':
        episode[19]['assignment']['learning_written_tiers'] = ['recent_strong']
    elif mutation == 'checkpoint':
        episode[271]['uid'] = 0
    elif mutation == 'capture':
        episode[334]['sample_metadata']['capture_frame_id'] = 333
    elif mutation == 'reference_age':
        episode[334]['assignment']['reacquire_reference_age_sec'] += 1.
    rows = list(episode.values())
    if mutation == 'duplicate':
        rows.append(episode[334])
    with pytest.raises(ValueError):
        replay.load_events(write_events(tmp_path, rows))


def test_checkpoint_and_counts_include_all_retained_gallery_tiers(config, episode):
    bank = replay.checkpoint_bank(config, episode)
    counts = replay.gallery_snapshot(bank)
    assert bank.track_to_uid == {1: 1}
    assert bank.identities[1].last_strong_observation['capture_frame_id'] == 271
    assert not bank._reacquire_quarantine.is_held(1)
    assert set(counts['1']) == {'archive_strong', 'archive_weak', 'archive_partial',
        'recent_strong', 'recent_partial', 'representative_strong', 'representative_partial'}
    assert all(row['captures'] in ([19], []) for row in counts['1'].values())
    assert counts['1']['archive_strong']['count'] == counts['1']['archive_partial']['count'] == 1


def test_baseline_disables_only_new_entry_point():
    assert replay.DisabledSimilarFollowBank.assign is IdentityBank.assign
    assert replay.DisabledSimilarFollowBank()._evaluate_similar_follow(future=True) is None


def test_two_new_similar_frames_recover_without_writing(config, episode):
    original = copy.deepcopy(episode)
    baseline = replay.replay(config, episode, bank_class=replay.DisabledSimilarFollowBank,
                             caps=[334, 336, 338])
    current = replay.replay(config, episode, caps=[334, 336, 338])
    assert baseline['accepted_caps'] == []
    assert current['accepted_caps'] == [336, 338]
    assert current['gallery_write_caps'] == baseline['gallery_write_caps'] == []
    assert current['gallery_after'] == current['gallery_before']
    assert [r['evaluated_search'] for r in current['rows']] == [True, True, False]
    assert episode == original


@pytest.mark.parametrize('variant', ['explicit_conflict', 'retained_contradiction',
    'competition_failed', 'nonunique', 'new_track_each_frame', 'full_bad', 'stale'])
def test_conflict_ambiguity_and_changed_person_do_not_gain_follow(config, episode, variant):
    result = replay.replay(config, episode, caps=[334, 336, 338], variant=variant)
    assert result['accepted_caps'] == result['gallery_write_caps'] == []


@pytest.mark.parametrize('caps', [[334, 334], [336, 334], [374, 376]])
def test_duplicate_reordering_or_new_opposite_candidate_cannot_recover(config, episode, caps):
    result = replay.replay(config, episode, caps=caps, logged_context=True)
    assert result['accepted_caps'] == result['gallery_write_caps'] == []


def test_sequential_crossing_does_not_become_a_new_opposite_person(config, episode):
    sequence = [334, 336, 338, 340, 342, 344, 346]
    for index, cap in enumerate(sequence):
        episode[cap] = event(cap, x=180. + 30. * index)
    result = replay.replay(config, episode, caps=sequence, logged_context=True)
    assert result['accepted_caps'] == sequence[1:]
    assert result['gallery_write_caps'] == []
    assert all(row['evaluated_search'] for row in result['rows'])
    assert episode[346]['sample_metadata']['search_direction_compatible'] is False


@pytest.mark.parametrize('logged_context', [False, True])
def test_saved_recording_reproduces_baseline_and_follows_without_gallery_mutation(config, logged_context):
    run = replay.ROOT/'run_request_0428_modular_logs/run_20261010_134118_24171_f4a0d66f'
    if not (run/'reid_diagnostics/events.jsonl').is_file():
        pytest.skip('historical recording is optional and not committed')
    events = replay.load_events(run)
    baseline = replay.replay(config, events, bank_class=replay.DisabledSimilarFollowBank,
                             logged_context=logged_context)
    current = replay.replay(config, events, logged_context=logged_context)
    assert len(baseline['rows']) == 28
    assert all((r['uid'], r['reason']) == (r['recorded_uid'], r['recorded_reason'])
               for r in baseline['rows'])
    assert baseline['accepted_caps'] == []
    # Three recorded weak crops still fail current quality. The next strong
    # observation must seed a new two-frame proof; do not invent strong flags.
    assert current['accepted_caps'] == [r['cap'] for r in current['rows']
        if r['cap'] not in (334, 343, 345, 346, 348)]
    assert current['gallery_write_caps'] == baseline['gallery_write_caps'] == []
    assert current['gallery_after'] == current['gallery_before']
    assert all(r['evaluated_full'] == pytest.approx(r['recorded_full'], abs=1e-6)
               and r['evaluated_partial'] == pytest.approx(r['recorded_partial'], abs=1e-6)
               for r in baseline['rows'])


def test_cli_explicitly_reports_scope_and_hashes(tmp_path, episode, capsys):
    run = write_events(tmp_path, episode.values())
    report = replay.main(['--run-dir', str(run)])
    assert json.loads(capsys.readouterr().out) == report
    assert 'synthetic' in report['scope'] and 'not crop/NPU inference' in report['scope']
    assert 'query-to-query' in report['scope_notes']['features']
    assert all(len(report[key]) == 64 for key in ('events_sha256', 'config_sha256', 'policy_sha256'))
    assert all(not r['accepted_caps'] and not r['gallery_write_caps']
               for r in report['negative_variants'].values())
