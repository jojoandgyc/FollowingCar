"""No NPU: real assign calls, bounded distance inputs, saved metadata replay."""
import copy
from dataclasses import replace
import json
import pytest

from rk_vision.identity_bank import IdentityBank, _geometry_observation
from tools import replay_cap759_recovery as replay


def event(cap):
    frame, stamp = cap // 2, 100. + cap * .05
    track = 1 if cap < 613 else 2 if cap == 613 else 5
    box = [200., 1., 400., 477.]
    m = dict(track_id=track, capture_frame_id=cap, capture_timestamp=stamp,
        frame_index=frame, detector_bbox=box, quality_bbox=box, bbox=box,
        image_width=640, image_height=480, detector_center_x_ratio=300/640,
        detector_area_ratio=200*476/(640*480), detector_edge_touch_count=2,
        detector_confidence=.95, quality_bbox_ok=True, bbox_quality_tier='strong',
        is_fresh=True, partial_observation=True, partial_feature_source='osnet_torso',
        integrated_yaw_deg=0., yaw_rate_dps=0., candidate_count=1,
        source_detection_index=0, search_reacquire_context_active=cap >= 784,
        search_direction='left' if cap >= 784 else None,
        search_direction_compatible=True if cap >= 784 else None,
        identity_competition=dict(uid=1, frame_index=frame, candidate_count=1,
            source_detection_index=0, passed=True, reason='single_candidate'))
    approved = cap in (258, 335)
    a = dict(bank_updated=approved, partial_template_update_reason='trusted_paired_update'
        if approved else None, template_quarantine_reason='armed' if cap == 769 else None,
        authorization_match_source='partial' if cap == 769 else None,
        reason='skip_update_reacquire_quarantine' if cap == 773 else 'recent_partial_conflict',
        template_recent_evidence=dict(distance=.26),
        reacquire_recent_partial_evidence=dict(distance=.48 if cap in (779, 792) else .35,
            comparable_caps=[258], comparison_mode='exact_coverage'))
    if cap == 798:
        a['reacquire_recent_partial_evidence']['distance'] = .42
    return dict(capture_frame_id=cap, frame_index=frame, capture_timestamp=stamp,
        raw_track_id=track, uid=1 if cap < 779 else 0, detector_bbox=box,
        sample_metadata=m, assignment=a)


@pytest.fixture
def episode():
    events = {cap: event(cap) for cap in replay.REQUIRED_CAPS}
    events[803]['assignment']['reacquire_geometry'] = dict(reference=
        _geometry_observation(events[613]['sample_metadata'], events[613]['frame_index']))
    return events


@pytest.fixture
def config():
    return replay.bank_config(replay.ROOT/'car_control_modular/config/reid_runtime.ini')


def write_events(tmp_path, events):
    folder = tmp_path/'reid_diagnostics'
    folder.mkdir()
    (folder/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    return tmp_path


def test_loader_requires_original_approved_sources_and_checkpoints(tmp_path, episode):
    loaded = replay.load_events(write_events(tmp_path, episode.values()))
    assert set(loaded) == set(replay.REQUIRED_CAPS)
    assert loaded[803]['assignment']['reacquire_geometry']['reference']['capture_frame_id'] == 613


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'unapproved', 'torso', 'anchor'])
def test_loader_refuses_ambiguous_or_wrong_recording(tmp_path, episode, mutation):
    if mutation == 'missing':
        episode.pop(798)
    elif mutation == 'unapproved':
        episode[335]['assignment']['bank_updated'] = False
    elif mutation == 'torso':
        episode[258]['assignment']['partial_template_update_reason'] = 'appearance_update_rejected'
    elif mutation == 'anchor':
        episode[803]['assignment']['reacquire_geometry']['reference']['capture_frame_id'] = 612
    rows = list(episode.values())
    if mutation == 'duplicate':
        rows.append(episode[801])
    with pytest.raises(ValueError):
        replay.load_events(write_events(tmp_path, rows))


def test_recovery_requires_two_new_good_observations_and_never_learns(episode, config):
    original = copy.deepcopy(episode)
    baseline = replay.replay(config, episode, bank_class=replay.DisabledPartialConflictRecoveryBank)
    current = replay.replay(config, episode)
    assert baseline['accepted_caps'] == []
    assert current['accepted_caps'] == [803, 822, 826]
    rows = {r['cap']: r for r in current['rows']}
    assert rows[796]['recovery_streak'] == 1
    assert rows[798]['reason'] == 'partial_evidence_tentative'
    assert rows[801]['recovery_streak'] == 1  # Tentative CAP798 broke the chain.
    assert rows[803]['recovery_streak'] == 2
    assert rows[803]['recovery_source'] == 'partial_conflict_recheck'
    assert rows[803]['geometry_source'] == 'local_recovery'
    assert rows[818]['uid'] == 0  # The old successful pair is too old now.
    assert current['gallery_write_caps'] == baseline['gallery_write_caps'] == []
    assert all(r['quarantined'] and r['protected_cap'] == 613 for r in current['rows'])
    assert episode == original


@pytest.mark.parametrize('caps,variant', [
    ([779, 801, 803], 'nonunique'),
    ([779, 801, 803], 'retained_contradiction'),
    ([779, 801, 801], None),
    ([779, 803, 801], None),
    ([779, 801, 803], 'full_bad'),
])
def test_negative_sequences_cannot_recover_or_learn(episode, config, caps, variant):
    result = replay.replay(config, episode, caps=caps, variant=variant)
    assert result['accepted_caps'] == result['gallery_write_caps'] == []


def test_full_box_can_supply_reliable_torso_evidence(episode, config):
    result = replay.replay(config, episode, caps=[779, 801, 803], variant='full_box')
    assert result['accepted_caps'] == [803]
    assert result['gallery_write_caps'] == []


def test_checkpoint_uses_only_approved_sources_and_retains_original_quarantine(episode, config):
    bank = replay.checkpoint_bank(config, episode)
    entry = bank.identities[1]
    assert bank.track_to_uid[5] == 1
    assert bank._reacquire_quarantine.is_held(1)
    assert bank._reacquire_search_anchors[1]['capture_frame_id'] == 613
    assert entry.last_strong_observation['capture_frame_id'] == 773
    assert [m['capture_frame_id'] for _, m in entry.template_memory.recent['strong']] == [258, 335]
    assert [m['capture_frame_id'] for _, m in entry.template_memory.recent['partial']] == [258]
    assert not bank._reacquire_control_suspects  # Real rejection creates the conflict.
    episode[258]['assignment']['bank_updated'] = False
    with pytest.raises(ValueError, match='unapproved'):
        replay.checkpoint_bank(config, episode)


def test_baseline_only_disables_new_eligibility(config):
    assert replay.DisabledPartialConflictRecoveryBank._reject_reacquire_control is IdentityBank._reject_reacquire_control
    assert replay.DisabledPartialConflictRecoveryBank(config)._can_recover_partial_conflict(future=True) is False


def test_pose_checkpoint_restores_only_recorded_confirmed_lineage(episode, config):
    bank = replay.checkpoint_bank(config, episode, pose_checkpoint=True)
    prior = bank._appearance_verified[1]
    assert prior['metadata']['capture_frame_id'] == 773
    assert prior['continuation_origin_cap'] == 769
    assert prior['continuation_source'] == 'partial'
    assert prior['comparable_caps'] == [258]
    assert bank._reacquire_search_anchors[1]['capture_frame_id'] == 613
    episode[769]['assignment']['authorization_match_source'] = None
    with pytest.raises(ValueError, match='lineage'):
        replay.checkpoint_bank(config, episode, pose_checkpoint=True)


def test_actual_saved_metadata_reproduces_baseline_and_bounds_fix(config):
    run = replay.ROOT/'run_request_0428_modular_logs/run_20261010_121546_14497_53c2712d'
    if not (run/'reid_diagnostics/events.jsonl').is_file():
        pytest.skip('optional historical run is not included in the repository')
    events = replay.load_events(run)
    baseline = replay.replay(config, events, bank_class=replay.DisabledPartialConflictRecoveryBank)
    current = replay.replay(config, events)
    assert len(baseline['rows']) == 22
    assert all((r['uid'], r['reason']) == (r['recorded_uid'], r['recorded_reason'])
               for r in baseline['rows'])
    assert baseline['accepted_caps'] == []
    assert current['accepted_caps'] == [803, 822, 823, 826]
    assert current['gallery_write_caps'] == baseline['gallery_write_caps'] == []
    for row in baseline['rows'] + current['rows']:
        assert row['evaluated_full'] == pytest.approx(row['recorded_full'], abs=1e-6)
        assert row['evaluated_partial'] == pytest.approx(row['recorded_partial'], abs=1e-6)
    assert next(r for r in current['rows'] if r['cap'] == 805)['uid'] == 0


def test_actual_pose_checkpoint_reproduces_original_cap777_then_follows_without_learning(config):
    run = replay.ROOT/'run_request_0428_modular_logs/run_20261010_121546_14497_53c2712d'
    if not (run/'reid_diagnostics/events.jsonl').is_file():
        pytest.skip('optional historical run is not included in the repository')
    events = replay.load_events(run)
    baseline = replay.replay(config, events, bank_class=replay.DisabledPoseAndRecoveryBank,
                             pose_checkpoint=True)
    current = replay.replay(config, events, pose_checkpoint=True)
    assert len(baseline['rows']) == 23
    assert all((r['uid'], r['reason']) == (r['recorded_uid'], r['recorded_reason'])
               for r in baseline['rows'])
    assert baseline['rows'][0]['reason'] == 'verified_continuation_recheck'
    assert current['rows'][0]['cap'] == 777
    assert current['rows'][0]['uid'] == 1
    assert current['rows'][0]['quarantined']
    assert current['rows'][0]['pose_continuation']['origin_cap'] == 773
    assert all(r['uid'] == 1 for r in current['rows'] if r['cap'] <= 805)
    assert all(r['uid'] == 0 for r in current['rows'] if 807 <= r['cap'] <= 818)
    assert current['gallery_write_caps'] == baseline['gallery_write_caps'] == []


def test_cli_report_declares_synthetic_scope_and_source_hashes(tmp_path, episode, capsys):
    run = write_events(tmp_path, episode.values())
    result = replay.main(['--run-dir', str(run)])
    assert json.loads(capsys.readouterr().out) == result
    assert 'synthetic' in result['scope']
    assert all(len(result[key]) == 64 for key in ('events_sha256', 'config_sha256', 'policy_sha256'))
    assert set(result['negative_variants']) == {
        'nonunique', 'retained_contradiction', 'duplicate', 'out_of_order', 'full_bad'}
    assert all(not r['accepted_caps'] and not r['gallery_write_caps']
               for r in result['negative_variants'].values())


def test_historical_ab_isolated_from_runtime_similar_follow_setting(episode, config):
    enabled = replace(config, similar_follow_enable=True)
    disabled = replace(config, similar_follow_enable=False)
    for bank_class in (IdentityBank, replay.DisabledPartialConflictRecoveryBank):
        result = replay.replay(enabled, episode, bank_class=bank_class)
        assert result == replay.replay(disabled, episode, bank_class=bank_class)
        assert result['similar_follow_enabled'] is False
    assert enabled.similar_follow_enable is True
    assert replay.bank_config(replay.ROOT/'car_control_modular/config/reid_runtime.ini').similar_follow_enable


def test_cli_can_separately_report_current_runtime_policy(tmp_path, episode, capsys):
    run = write_events(tmp_path, episode.values())
    result = replay.main(['--run-dir', str(run), '--pose-checkpoint', '--current-runtime-policy'])
    assert json.loads(capsys.readouterr().out) == result
    assert result['baseline']['similar_follow_enabled'] is False
    assert result['current']['similar_follow_enabled'] is False
    assert result['pose_checkpoint']['current']['similar_follow_enabled'] is False
    assert result['current_runtime_policy']['similar_follow_enabled'] is True
    assert 'disables similar_follow' in result['scope_notes']['policy']
    assert result['current_runtime_policy']['gallery_write_caps'] == []


def test_actual_recording_also_exercises_new_runtime_policy_without_learning(config):
    run = replay.ROOT/'run_request_0428_modular_logs/run_20261010_121546_14497_53c2712d'
    if not (run/'reid_diagnostics/events.jsonl').is_file():
        pytest.skip('optional historical run is not included in the repository')
    result = replay.replay(replace(config, similar_follow_enable=True),
        replay.load_events(run), pose_checkpoint=True, current_runtime_policy=True)
    assert result['similar_follow_enabled'] is True
    assert len(result['rows']) == 23
    assert result['gallery_write_caps'] == []
    # Keep this a current-policy regression, not the pinned historical 17-frame
    # acceptance claim. The original metadata does not simulate changed motion.
    assert len(result['accepted_caps']) >= 17
