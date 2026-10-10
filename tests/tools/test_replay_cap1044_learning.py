"""Hardware-free provenance and scope tests for the CAP1044 audit."""
import copy
import hashlib
import json

import numpy as np
import pytest

from rk_vision.identity_bank import IdentityBankConfig
from tools import replay_cap1044_learning as replay


def event(cap):
    track = 1 if cap < 914 else 3
    frame, timestamp = cap // 2, 10. + cap * .05
    box = [200., 20., 400., 470.]
    metadata = dict(track_id=track, capture_frame_id=cap, capture_timestamp=timestamp,
        detector_bbox=box, bbox=box, quality_bbox=box, image_width=640,
        image_height=480, detector_center_x_ratio=300/640, detector_area_ratio=200*450/(640*480),
        detector_edge_touch_count=0, detector_confidence=.95, quality_bbox_ok=True,
        bbox_quality_tier='strong', is_fresh=True, partial_feature_source='osnet_torso',
        candidate_count=1, candidate_score_gap=.95, search_reacquire_context_active=False)
    return dict(capture_frame_id=cap, frame_index=frame, capture_timestamp=timestamp,
        raw_track_id=track, uid=1, detector_bbox=box, sample_metadata=metadata,
        sample_path=f'cap_{cap}_track_{track}.png', assignment=dict(bank_updated=True,
            partial_template_update_reason='trusted_update', reacquire_geometry=dict(reference=dict(
                capture_frame_id=cap-1, frame_index=frame-1, track_id=track,
                capture_timestamp=timestamp-.05, bbox=box, center_x_ratio=300/640,
                area=200*450/(640*480), area_units='ratio', geometry_source='detector'))))


@pytest.fixture
def episode(tmp_path):
    folder = tmp_path/'run'/'reid_diagnostics'
    folder.mkdir(parents=True)
    events = {cap:event(cap) for cap in replay.APPROVED_BEFORE_941 + replay.WRITE_OPPORTUNITIES}
    for cap, row in events.items():
        (folder/row['sample_path']).write_bytes(f'saved crop {cap}'.encode())
    def save(rows):
        (folder/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    save(events.values())
    return folder.parent, events, save


def test_load_selects_approved_track_without_discarding_duplicate_cap_other_person(episode):
    run, events, save = episode
    background = copy.deepcopy(events[1044])
    background.update(raw_track_id=5, uid=0)
    background['assignment']['bank_updated'] = False
    save(list(events.values())+[background])
    loaded = replay.load_events(run)
    assert loaded[1044]['raw_track_id'] == 3


@pytest.mark.parametrize('change', ['duplicate', 'missing', 'unapproved', 'search', 'future_reference'])
def test_wrong_recording_rejected(episode, change):
    run, events, save = episode
    rows = list(events.values())
    if change == 'duplicate':
        rows.append(events[1044])
    elif change == 'missing':
        rows.remove(events[1044])
    elif change == 'unapproved':
        events[21]['assignment']['bank_updated'] = False
    elif change == 'search':
        events[1044]['sample_metadata']['search_reacquire_context_active'] = True
    else:
        events[1044]['assignment']['reacquire_geometry']['reference']['capture_frame_id'] = 1044
    save(rows)
    with pytest.raises(ValueError):
        replay.load_events(run)


def test_only_recorded_approved_history_seeds_gallery(episode):
    _, events, _ = episode
    vectors = {c:dict(fused=np.array([1., c/10000., .1]),
                      torso=np.array([1., .1, c/10000.])) for c in events}
    config = IdentityBankConfig(template_memory_enable=True, template_crosscheck_enable=True)
    bank = replay.seed_bank(config, events, vectors)
    for caps in replay.gallery_caps(bank).values():
        assert set(caps).issubset(set(replay.APPROVED_BEFORE_941))
        assert not set(caps).intersection(replay.WRITE_OPPORTUNITIES)
    events[21]['assignment']['bank_updated'] = False
    with pytest.raises(ValueError, match='unapproved'):
        replay.seed_bank(config, events, vectors)


def test_same_frame_comparison_preserves_uid_and_control_reason(episode):
    _, events, _ = episode
    vectors = {c:dict(fused=np.array([1., .1, .2], dtype=np.float32),
                      torso=np.array([1., .2, .1], dtype=np.float32)) for c in events}
    config = IdentityBankConfig(template_memory_enable=True, template_crosscheck_enable=True)
    result = replay.same_frame_uid_probe(config, events, vectors)
    assert result['changed_uid_caps'] == []
    assert result['lost_ordinary_reason_caps'] == []
    assert all(row['enabled_uid'] == 1 for row in result['rows'])


def test_learning_wait_does_not_freeze_clean_candidate_forever(episode):
    _, events, _ = episode
    vectors = {c:dict(fused=np.array([1., .1, .2], dtype=np.float32),
                      torso=np.array([1., .2, .1], dtype=np.float32)) for c in events}
    config = IdentityBankConfig(template_memory_enable=True, template_crosscheck_enable=True, update_interval=1,
                                template_learning_guard_enable=True)
    result = replay.replay(config, events, vectors)
    first, second = result['rows'][:2]
    assert first['uid'] == second['uid'] == 1
    assert first['stored_current'] is False
    assert second['stored_current'] is True


def test_path_traversal_and_missing_crop_rejected(episode):
    run, events, _ = episode
    row = events[1044]
    assert replay.crop_path(run, row).is_file()
    row['sample_path'] = '../outside.png'
    with pytest.raises(ValueError):
        replay.crop_path(run, row)


@pytest.mark.parametrize('change', ['none', 'archive', 'crop', 'model', 'nan'])
def test_cache_verifies_archive_crop_model_and_vectors(tmp_path, episode, change):
    run, events, _ = episode
    events = {1044:events[1044]}
    archive = tmp_path/'cache.npz'
    report_path = tmp_path/'report.json'
    model = dict(path='/model.rknn', sha256='model_digest', format='RGB')
    np.savez_compressed(archive, cap_1044_fused=np.array([np.nan if change=='nan' else 1., .1]),
                        cap_1044_torso=np.array([1., .2]))
    crop = replay.crop_path(run, events[1044])
    report = dict(feature_archive=str(archive), model=model.copy(),
        feature_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        crops=[dict(cap=1044, track=3, sha256=hashlib.sha256(crop.read_bytes()).hexdigest())])
    if change == 'archive':
        report['feature_archive_sha256'] = 'wrong'
    elif change == 'crop':
        report['crops'][0]['sha256'] = 'wrong'
    elif change == 'model':
        report['model']['format'] = 'BGR'
    report_path.write_text(json.dumps(report))
    if change != 'none':
        with pytest.raises(ValueError):
            replay.load_vectors(archive, report_path, run, events, model)
    else:
        vectors, _ = replay.load_vectors(archive, report_path, run, events, model)
        assert list(vectors) == [1044]


def test_cli_requires_explicit_source_and_report(tmp_path):
    with pytest.raises(SystemExit):
        replay.main(['--run-dir',str(tmp_path)])
    with pytest.raises(SystemExit):
        replay.main(['--run-dir',str(tmp_path),'--features',str(tmp_path/'f.npz')])


def test_inference_refuses_running_car_before_importing_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(replay, 'car_processes', lambda: [123])
    with pytest.raises(RuntimeError, match='car runtime active'):
        replay.infer(tmp_path, {}, (None, 0.))
