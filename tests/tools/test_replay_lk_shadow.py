import csv
import json

import numpy as np
import pytest

from tools import replay_lk_shadow as replay


def saved_run(tmp_path, monkeypatch, *, wrong_timestamp=False):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'reid_diagnostics').mkdir()
    (run / 'camera_raw.avi').touch()
    with (run / 'camera_raw.frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            'video_frame_index', 'capture_frame_id', 'capture_monotonic_sec'))
        writer.writeheader()
        for i in range(6):
            writer.writerow(dict(video_frame_index=i, capture_frame_id=i+1,
                                 capture_monotonic_sec=10+i*.05))
    with (run / 'reid_diagnostics/events.jsonl').open('w') as stream:
        for cap in (1, 3, 5):
            ts = 10 + (cap - 1) * .05 + (.01 if wrong_timestamp else 0.)
            stream.write(json.dumps(dict(capture_frame_id=cap, capture_timestamp=ts,
                raw_track_id=4, uid=1, detector_bbox=[60, 50, 180, 190],
                sample_metadata=dict(capture_timestamp=ts))) + '\n')
    rng = np.random.default_rng(31)
    gray = np.zeros((240, 320), np.uint8)
    gray[50:190, 60:180] = rng.integers(0, 256, (140, 120), dtype=np.uint8)
    frame = np.repeat(gray[:, :, None], 3, axis=2)

    class SavedReader:
        def __init__(self, path):
            assert path == str(run / 'camera_raw.avi')
        def isOpened(self):
            return True
        def set(self, key, value):
            return True
        def read(self):
            return True, frame.copy()
        def release(self):
            pass
    monkeypatch.setattr(replay.cv2, 'VideoCapture', SavedReader)
    return run


def test_saved_capture_replay_measures_flow_and_keeps_heldout_boxes_separate(tmp_path, monkeypatch):
    run = saved_run(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
    summary, rows = replay.replay(run, first=1, last=6, correction_every=2)
    assert summary['status_counts'] == dict(seeded=2, tracked=4)
    assert [r['capture_id'] for r in rows if r['correction_applied']] == [1, 5]
    assert summary['heldout_logged_detector_comparisons'] == 1
    assert rows[2]['detector_comparison']['iou'] == pytest.approx(1.)
    assert summary['tracked_interval_seconds'] == pytest.approx(.2)
    assert summary['tracked_interval_time_coverage'] == pytest.approx(.8)
    assert summary['process_wall_ms']['p95'] > 0
    assert summary['process_cpu_ms']['p95'] > 0
    assert before == {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}


def test_saved_seed_must_match_its_actual_raw_image_clock(tmp_path, monkeypatch):
    run = saved_run(tmp_path, monkeypatch, wrong_timestamp=True)
    with pytest.raises(ValueError, match='timestamps differ'):
        replay.replay(run, first=1, last=6)


def test_no_device_fallback_when_saved_video_is_missing(tmp_path):
    with pytest.raises(ValueError, match='not a regular file'):
        replay.load_recording(tmp_path, 1, 6)


def test_report_cannot_overwrite_source_recording(tmp_path, monkeypatch):
    run = saved_run(tmp_path, monkeypatch)
    monkeypatch.setattr('sys.argv', ['replay_lk_shadow.py', str(run), '--output',
                                    str(run / 'new_report.json')])
    with pytest.raises(SystemExit):
        replay.main()
    assert not (run / 'new_report.json').exists()


def test_diagnostic_input_subset_is_not_reported_as_camera_cadence_or_lost_video(tmp_path, monkeypatch):
    run = saved_run(tmp_path, monkeypatch)
    summary, rows = replay.replay(run, first=1, last=6,
                                frame_scope='diagnostic-captures')
    assert [row['capture_id'] for row in rows] == [1, 3, 5]
    assert summary['frame_scope'] == 'diagnostic-captures'
    assert summary['saved_frames'] == 6 and summary['processed_frames'] == 3
    assert summary['skipped_by_frame_scope'] == [2, 4, 6]
    assert summary['missing_capture_ids'] == []
    assert any('throttled diagnostics' in caveat for caveat in summary['caveats'])
