"""No model/hardware dependency: verify the saved-evidence replay contract."""
import copy
import hashlib
import json
import logging

import numpy as np
import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from tools import replay_cap828_continuation as replay


def event(cap, *, approved=False):
    stamp = 10. + cap * .052
    box = [220., 100., 340., 430.]
    frame = cap // 2
    metadata = dict(
        track_id=1 if approved else 3, frame_index=frame,
        capture_frame_id=cap, capture_timestamp=stamp, is_fresh=True,
        detector_bbox=box, quality_bbox=box, bbox=box, image_width=640, image_height=480,
        detector_center_x_ratio=280/640, detector_area_ratio=120*330/(640*480),
        detector_edge_touch_count=0, detector_confidence=.95, integrated_yaw_deg=0., yaw_rate_dps=0.,
        partial_feature_source='osnet_torso', partial_observation=False,
        quality_bbox_ok=True, bbox_quality_tier='strong', candidate_count=1,
        candidate_score_gap=.95, search_reacquire_context_active=False)
    assignment = dict(bank_updated=approved, reason='updated_diverse' if approved else 'mapped_late_continuation',
        partial_template_update_reason='trusted_update' if approved else None,
        authorization_match_source='partial', template_quarantine_reason='not_high_quality_strong',
        protected_search_anchor_cap=713, template_quarantine_streak=0,
        reacquire_partial_state='match',
        reacquire_recent_partial_evidence=dict(comparable_caps=[414], comparison_mode='exact_coverage'),
        candidate_observation=dict(start_cap=816, count=8, crossed=False))
    return dict(capture_frame_id=cap, capture_timestamp=stamp, frame_index=frame,
                uid=1, raw_track_id=metadata['track_id'], detector_bbox=box,
                sample_metadata=metadata, assignment=assignment)


@pytest.fixture
def episode():
    rows = {cap: event(cap, approved=True) for cap in replay.APPROVED_CAPS}
    for cap in (814, 816, 830, 832, 875, 878, 879, 881, 901):
        rows[cap] = event(cap)
    rows[814]['assignment'].update(template_quarantine_reason='armed',
        reacquire_geometry=dict(reference=dict(
            frame_index=241, track_id=1, capture_frame_id=713, capture_timestamp=47.1,
            center_x_ratio=.89, area=.20, area_units='ratio', geometry_source='detector',
            bbox=[505., 6., 639., 460.], integrated_yaw_deg=2.38, yaw_rate_dps=0.)))
    rows[875]['assignment']['template_quarantine_reason'] = 'armed'
    rows[879]['assignment'].update(authorization_match_source='strong',
                                  candidate_observation=dict(start_cap=878, count=2, crossed=False))
    vectors = {cap: dict(fused=np.array([1., .1, .2], dtype=np.float32),
                        torso=np.array([1., .2, .1], dtype=np.float32)) for cap in rows}
    return rows, vectors


def write_events(tmp_path, rows):
    diagnostics = tmp_path/'reid_diagnostics'
    diagnostics.mkdir()
    (diagnostics/'events.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return tmp_path


def test_load_recorded_episode_validates_both_checkpoints(tmp_path, episode):
    rows, _ = episode
    loaded = replay.load_events(write_events(tmp_path, rows.values()))
    assert loaded[830]['assignment']['authorization_match_source'] == 'partial'
    assert loaded[879]['assignment']['authorization_match_source'] == 'strong'
    assert 713 not in loaded  # Protected geometry is recorded, not a saved crop.


@pytest.mark.parametrize('mutation', ['duplicate', 'gallery', 'uid', 'source', 'anchor', 'missing_query'])
def test_rejects_wrong_or_ambiguous_recording(tmp_path, episode, mutation):
    rows, _ = episode
    if mutation == 'duplicate':
        values = list(rows.values()) + [rows[830]]
    else:
        if mutation == 'gallery':
            rows[832]['assignment']['bank_updated'] = True
        elif mutation == 'uid':
            rows[830]['uid'] = 0
        elif mutation == 'source':
            rows[879]['assignment']['authorization_match_source'] = 'partial'
        elif mutation == 'anchor':
            rows[814]['assignment']['reacquire_geometry']['reference']['capture_frame_id'] = 712
        else:
            rows.pop(881)
        values = rows.values()
    with pytest.raises(ValueError):
        replay.load_events(write_events(tmp_path, values))


@pytest.mark.parametrize('checkpoint', [830, 879])
def test_checkpoint_preserves_quarantine_and_never_learns_candidate(episode, checkpoint):
    rows, vectors = episode
    config = IdentityBankConfig(template_memory_enable=True, template_crosscheck_enable=True)
    bank = replay.checkpoint_bank(config, rows, vectors, checkpoint)
    assert bank.track_to_uid[3] == 1
    assert bank._reacquire_quarantine.is_held(1)
    assert bank._reacquire_search_anchors[1]['capture_frame_id'] == 713
    assert bank.identities[1].last_strong_observation['capture_frame_id'] == checkpoint
    assert bank._appearance_verified[1]['comparable_caps'] == [414]
    candidate = bank._candidate_observations.rows[(1, 3)]
    assert candidate['confirmed']['capture_frame_id'] == checkpoint
    assert candidate['late_confirmed_source'] == replay.CHECKPOINTS[checkpoint]['source']
    memory = bank.identities[1].template_memory
    for tier in ('strong', 'partial'):
        assert all(info['capture_frame_id'] in replay.APPROVED_CAPS
                   for _, info in memory.recent[tier])
        assert all(info['capture_frame_id'] != checkpoint
                   for _, info in memory.recent[tier])


def test_partial_gallery_accepts_only_explicit_trusted_updates(episode):
    rows, vectors = episode
    for cap in replay.APPROVED_CAPS[1:]:
        rows[cap]['assignment']['partial_template_update_reason'] = 'appearance_update_rejected'
    config = IdentityBankConfig(template_memory_enable=True, template_crosscheck_enable=True)
    bank = replay.checkpoint_bank(config, rows, vectors, 830)
    entry = bank.identities[1]
    assert len(entry.partial_features) == 1  # Initial approved enrollment only.
    assert all(info['capture_frame_id'] == 20 for _, info in entry.template_memory.recent['partial'])


def test_refuses_unapproved_gallery_input_even_without_loader(episode):
    rows, vectors = episode
    rows[414]['assignment']['bank_updated'] = False
    with pytest.raises(ValueError, match='unapproved'):
        replay.checkpoint_bank(IdentityBankConfig(), rows, vectors, 830)


@pytest.mark.parametrize('logged_search', [False, True])
def test_calls_real_assign_contract_and_does_not_mutate_source(episode, logged_search):
    rows, vectors = episode
    rows[832]['sample_metadata'].update(search_reacquire_context_active=True,
                                        search_direction='left', search_direction_compatible=False)
    before = copy.deepcopy(rows)
    calls = []

    class RecordingBank(IdentityBank):
        def assign(self, **kwargs):
            calls.append(copy.deepcopy(kwargs))
            return super().assign(**kwargs)

    config = IdentityBankConfig(template_memory_enable=True, template_crosscheck_enable=True)
    result = replay.replay(config, rows, vectors, 830,
                           logged_search=logged_search, bank_class=RecordingBank)
    assert rows == before
    assert result['total_frames'] == len(calls) == 6
    first = calls[0]
    assert first['track_id'] == 3
    assert first['sample_metadata']['capture_frame_id'] == 832
    assert first['sample_metadata']['capture_timestamp'] == rows[832]['capture_timestamp']
    assert first['area'] == 120*330
    assert first['preferred_uid'] == (1 if logged_search else None)
    assert first['sample_metadata']['search_reacquire_context_active'] is logged_search
    assert first['preferred_candidate_ok'] is False


def test_disabled_baseline_only_removes_new_evaluator():
    bank = replay.DisabledVerifiedContinuationBank(IdentityBankConfig())
    assert bank._evaluate_verified_continuation(1, future_argument=True) is None
    assert bank._reject_reacquire_control.__func__ is IdentityBank._reject_reacquire_control


def test_replay_queries_include_second_failure_and_end(episode):
    rows, _ = episode
    caps = replay.required_caps(rows)
    assert {830, 832, 879, 881, 901}.issubset(caps)
    assert 713 not in caps


@pytest.mark.parametrize('failure', [None, 'distance', 'digest', 'missing_provenance', 'nonfinite'])
def test_cache_is_checked_beyond_crop_and_model_hashes(tmp_path, monkeypatch, failure):
    archive = tmp_path/'features.npz'
    values = dict(cap_20_fused=np.array([1., 0.]), cap_20_torso=np.array([0., 1.]),
                  cap_830_fused=np.array([1., 0.]), cap_830_torso=np.array([0., 1.]))
    if failure == 'nonfinite':
        values['cap_830_fused'][0] = np.nan
    np.savez_compressed(archive, **values)
    report = dict(pairs=[dict(query=830, template=20, fused=0., torso=0.)])
    if failure == 'distance':
        report['pairs'][0]['fused'] = .25
    elif failure == 'digest':
        report['feature_archive_sha256'] = 'wrong'
    elif failure == 'missing_provenance':
        report['pairs'] = []
    monkeypatch.setattr(replay, 'validate_feature_report', lambda *args: ([], report))
    if failure is None:
        vectors, _ = replay.load_cached_vectors(archive, tmp_path/'report.json', tmp_path, [20, 830], {})
        np.testing.assert_array_equal(vectors[830]['fused'], [1., 0.])
    else:
        with pytest.raises(ValueError):
            replay.load_cached_vectors(archive, tmp_path/'report.json', tmp_path, [20, 830], {})


def test_digest_allows_own_replay_report_without_pair_matrix(tmp_path, monkeypatch):
    archive = tmp_path/'features.npz'
    np.savez_compressed(archive, cap_830_fused=np.array([1., 0.]), cap_830_torso=np.array([0., 1.]))
    report = dict(feature_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    monkeypatch.setattr(replay, 'validate_feature_report', lambda *args: ([], report))
    vectors, _ = replay.load_cached_vectors(archive, tmp_path/'report.json', tmp_path, [830], {})
    assert 830 in vectors


def test_output_cannot_modify_historical_run(tmp_path):
    with pytest.raises(SystemExit):
        replay.main(['--run-dir', str(tmp_path), '--infer', '--output', str(tmp_path/'new.json')])
    assert not (tmp_path/'new.json').exists()


def test_cached_mode_requires_provenance(tmp_path):
    with pytest.raises(SystemExit):
        replay.main(['--run-dir', str(tmp_path), '--features', str(tmp_path/'cache.npz')])


def test_inference_refuses_active_car(tmp_path, monkeypatch, episode):
    rows, _ = episode
    monkeypatch.setattr(replay, 'load_events', lambda _: rows)
    monkeypatch.setattr(replay, 'model_config', lambda _: None)
    monkeypatch.setattr(replay, 'model_fingerprint', lambda _: {})
    monkeypatch.setattr(replay, 'car_processes', lambda: [123])
    monkeypatch.setattr(replay, 'extract_saved', lambda *args: pytest.fail('must not run NPU'))
    previous_disable = logging.root.manager.disable
    try:
        logging.disable(logging.INFO)
        with pytest.raises(SystemExit):
            replay.main(['--run-dir', str(tmp_path), '--infer'])
        assert logging.root.manager.disable == logging.INFO
    finally:
        logging.disable(previous_disable)


def test_report_hashes_new_proof_and_declares_counterfactual_scope(tmp_path, monkeypatch, episode):
    rows, vectors = episode
    archive = tmp_path/'feature-cache.npz'
    archive.write_bytes(b'cached-vectors-are-injected-without-inference')
    output = tmp_path/'report.json'
    monkeypatch.setattr(replay, 'load_events', lambda _: rows)
    monkeypatch.setattr(replay, 'model_config', lambda _: None)
    monkeypatch.setattr(replay, 'model_fingerprint', lambda _: {})
    monkeypatch.setattr(replay, 'load_cached_vectors', lambda *args: (vectors, []))
    monkeypatch.setattr(replay, 'bank_config', lambda _: IdentityBankConfig(
        template_memory_enable=True, template_crosscheck_enable=True))
    monkeypatch.setattr(replay, 'extract_saved', lambda *args: pytest.fail('must not run NPU'))
    previous_disable = logging.root.manager.disable
    try:
        logging.disable(logging.WARNING)
        assert replay.main(['--run-dir', str(tmp_path/'historical-run'),
                            '--features', str(archive), '--feature-report', str(tmp_path/'audit.json'),
                            '--output', str(output)]) == 0
        assert logging.root.manager.disable == logging.WARNING
    finally:
        logging.disable(previous_disable)
    result = json.loads(output.read_text())
    name = 'rk_vision/verified_continuation.py'
    assert result['policy_sha256'][name] == hashlib.sha256((replay.ROOT/name).read_bytes()).hexdigest()
    assert 'not full-run replay' in result['scope']
    note = result['scope_notes']['disabled_verified_continuation']
    assert 'disable only _evaluate_verified_continuation' in note
    assert 'not a historical code snapshot' in note
    assert 'not a prediction of images/motion' in result['scope_notes']['continuous_visible_counterfactual']


def test_exception_restores_log_level(tmp_path, monkeypatch, episode):
    rows, _ = episode
    monkeypatch.setattr(replay, 'load_events', lambda _: rows)
    monkeypatch.setattr(replay, 'model_config', lambda _: None)
    monkeypatch.setattr(replay, 'model_fingerprint', lambda _: {})

    def invalid_cache(*args):
        assert logging.root.manager.disable == logging.CRITICAL
        raise ValueError('deliberately invalid cache')

    monkeypatch.setattr(replay, 'load_cached_vectors', invalid_cache)
    previous_disable = logging.root.manager.disable
    try:
        logging.disable(logging.DEBUG)
        with pytest.raises(ValueError, match='invalid cache'):
            replay.main(['--run-dir', str(tmp_path), '--features', str(tmp_path/'features.npz'),
                         '--feature-report', str(tmp_path/'audit.json')])
        assert logging.root.manager.disable == logging.DEBUG
    finally:
        logging.disable(previous_disable)
