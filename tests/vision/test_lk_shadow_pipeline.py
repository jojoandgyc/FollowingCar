"""Pipeline-only opt-in diagnostics; no devices, inference or control writes."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rk_vision import lk_shadow
from rk_vision.frames import FramePacket
from rk_vision.pipeline import RKNNVisionConfig, RKNNVisionPipeline
from rk_vision.tracker import TrackRecord
from rk_vision.yolo11 import Detection


BOX = (60., 40., 180., 220.)


class WorkerSpy:
    def __init__(self, config):
        self.tasks = []
        self.result = None
        self.close_calls = 0
    def submit(self, frame, cap, ts, **kwargs):
        self.tasks.append(dict(frame=frame.copy(), cap=cap, ts=ts, **kwargs))
        return True
    def latest_result(self):
        return self.result
    def stats(self):
        return dict(submitted=len(self.tasks))
    def close(self):
        self.close_calls += 1
        return True


@pytest.fixture
def pipeline_case(monkeypatch):
    monkeypatch.setattr(lk_shadow, 'LKShadowWorker', WorkerSpy)
    pipeline = RKNNVisionPipeline.__new__(RKNNVisionPipeline)
    pipeline.config = RKNNVisionConfig(yolo_model_path='unused', lk_shadow_enable=True,
        predicted_reid_verify_enable=False, identity_suppress_duplicate_uids=False)
    pipeline._frame_context = {}
    pipeline._reid_diagnostics = None
    pipeline.logger = None
    calls = []
    pipeline.detector = SimpleNamespace(
        detect=lambda *args: [Detection(BOX, .95, 0)],
        last_search_diagnostic_detections=[], last_timing_ms=dict(inference=3., total=4.),
        release=lambda: calls.append('detector_close'))
    pipeline.reid = SimpleNamespace(
        extract=lambda *args: [np.array([1., 0.])],
        last_partial_features=[], last_partial_feature_sources=[],
        last_timing_ms=dict(inference=7., total=9.),
        release=lambda: calls.append('reid_close'))
    record = TrackRecord(4, 1, 50, 30, 190, 230, 0, .95, 120, 130, 28000, 0., 1, 0)
    case = SimpleNamespace(pipeline=pipeline, record=record, observations=None, calls=calls)
    def update(*args, **kwargs):
        if case.observations is None:
            pipeline.tracker.last_identity_observations = [dict(raw_track_id=4, uid=1,
                detector_bbox=BOX, sample_metadata=dict(kwargs['frame_context'], is_fresh=True),
                assignment=dict(uid=1, bbox_quality_ok=True))]
        else:
            pipeline.tracker.last_identity_observations = case.observations
        return [case.record]
    pipeline.tracker = SimpleNamespace(update=update, last_identity_observations=[])
    case.frame = np.random.default_rng(9).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    def step(cap, ts=None):
        pipeline.set_frame_context(control_frame_id=cap, capture_frame_id=cap,
                                   capture_timestamp=10.+cap*.05 if ts is None else ts)
        return pipeline.process_frame(FramePacket(case.frame, format='RGB'), 'RGB')
    case.step = step
    return case


def test_disabled_config_does_not_construct_worker_or_add_timing(pipeline_case, monkeypatch):
    case = pipeline_case
    case.pipeline.config = replace(case.pipeline.config, lk_shadow_enable=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('disabled shadow must not start a worker')
    monkeypatch.setattr(lk_shadow, 'LKShadowWorker', forbidden)
    assert case.step(1) == [case.record]
    assert not hasattr(case.pipeline, '_lk_shadow_worker')
    assert 'lk_shadow_submit' not in case.pipeline.last_timing_ms


def test_actual_processed_image_and_detector_crop_are_capture_aligned(pipeline_case):
    case = pipeline_case
    assert not hasattr(case.pipeline, '_lk_shadow_worker')
    result = case.step(1)
    task = case.pipeline._lk_shadow_worker.tasks[0]
    np.testing.assert_array_equal(task['frame'], case.frame)
    assert task['frame_format'] == 'RGB'
    assert (task['cap'], task['ts']) == (1, 10.05)
    assert task['seed'] == lk_shadow.LKShadowSeed(1, 4, 1, 10.05, BOX)
    assert task['seed'].bbox != (case.record.x1, case.record.y1, case.record.x2, case.record.y2)
    assert result[0] is case.record
    assert case.pipeline.last_timing_ms['reid_inference'] == 7.
    assert case.pipeline.last_timing_ms['yolo_inference'] == 3.
    assert case.pipeline.last_timing_ms['lk_shadow_submit'] >= 0.


def test_correction_spacing_allows_flow_between_processed_frames(pipeline_case):
    case = pipeline_case
    case.step(1, 10.)
    case.step(2, 10.1)
    case.step(3, 10.31)
    tasks = case.pipeline._lk_shadow_worker.tasks
    assert [task['seed'] is not None for task in tasks] == [True, False, True]
    assert [task['cap'] for task in tasks] == [1, 2, 3]


@pytest.mark.parametrize('variant', ['uid0', 'predicted', 'stale_cap', 'stale_time',
    'wrong_raw', 'not_fresh', 'rejected', 'missing_box', 'ambiguous'])
def test_no_uid0_prediction_or_mismatched_seed(pipeline_case, variant):
    case = pipeline_case
    obs = dict(raw_track_id=4, uid=1, detector_bbox=BOX,
        sample_metadata=dict(capture_frame_id=1, capture_timestamp=10.05, is_fresh=True),
        assignment=dict(uid=1, bbox_quality_ok=True))
    case.observations = [obs]
    if variant == 'uid0':
        case.record = replace(case.record, reid_uid=0)
    elif variant == 'predicted':
        case.record = replace(case.record, time_since_update=1)
    elif variant == 'stale_cap':
        obs['sample_metadata']['capture_frame_id'] = 99
    elif variant == 'stale_time':
        obs['sample_metadata']['capture_timestamp'] = 9.
    elif variant == 'wrong_raw':
        obs['raw_track_id'] = 99
    elif variant == 'not_fresh':
        obs['sample_metadata']['is_fresh'] = False
    elif variant == 'rejected':
        obs['assignment']['identity_control_rejected'] = True
    elif variant == 'missing_box':
        obs.pop('detector_bbox')
    elif variant == 'ambiguous':
        case.observations.append(copy.deepcopy(obs))
    before = copy.deepcopy(case.observations)
    assert case.step(1)[0] is case.record
    assert case.pipeline._lk_shadow_worker.tasks[0]['seed'] is None
    assert case.observations == before


def test_probe_submits_actual_frame_without_reusing_old_observation_as_seed(pipeline_case):
    case = pipeline_case
    case.step(1)
    case.pipeline.set_frame_context(control_frame_id=2, capture_frame_id=2, capture_timestamp=10.1)
    assert case.pipeline.process_search_probe_frame(case.frame, 'RGB') == []
    task = case.pipeline._lk_shadow_worker.tasks[-1]
    assert task['cap'] == 2 and task['seed'] is None
    assert case.pipeline.last_timing_ms['reid_inference'] == 0.


@pytest.mark.parametrize('failure', ['create', 'submit', 'poll'])
def test_optional_worker_failure_preserves_normal_records(pipeline_case, monkeypatch, failure):
    case = pipeline_case
    def fail(*args, **kwargs):
        raise RuntimeError('optional diagnostic failed')
    if failure == 'create':
        monkeypatch.setattr(lk_shadow, 'LKShadowWorker', fail)
    else:
        worker = WorkerSpy(None)
        setattr(worker, 'submit' if failure == 'submit' else 'latest_result', fail)
        case.pipeline._lk_shadow_worker = worker
    assert case.step(1)[0] is case.record
    assert case.pipeline.last_lk_shadow_diagnostic['status'] == 'disabled_after_error'
    assert case.step(2)[0] is case.record


def test_previous_result_logging_keeps_its_actual_capture_and_cpu_timing(pipeline_case):
    case = pipeline_case
    case.step(1)
    logs = []
    case.pipeline.logger = SimpleNamespace(info=lambda *args: logs.append(args), debug=lambda *args: None)
    result = lk_shadow.LKShadowResult(1, 10.05, 'seeded', 'aligned', wall_ms=9999.)
    case.pipeline._lk_shadow_worker.result = result
    case.step(2)
    assert case.pipeline.last_lk_shadow_result is result
    assert logs[0][0] == 'lk_shadow_result %s'
    import json
    payload = json.loads(logs[0][1])
    assert payload['capture_id'] == 1 and payload['observed_at_capture_id'] == 2
    assert payload['observed_capture_age_ms'] == pytest.approx(50.)
    assert payload['cadence'] == 'processed_frames_only'
    assert payload['identity_authority'] is False and payload['motion_authority'] is False
    assert payload['worker']['submitted'] == 1
    assert case.pipeline.last_timing_ms['reid_inference'] == 7.
    assert case.pipeline.last_timing_ms['total'] < 9999.
    case.step(3)
    assert len(logs) == 1


def test_close_joins_worker_and_prevents_restart(pipeline_case):
    case = pipeline_case
    case.step(1)
    worker = case.pipeline._lk_shadow_worker
    case.pipeline.close()
    assert worker.close_calls == 1
    assert case.calls == ['detector_close', 'reid_close']
    case.pipeline._submit_lk_shadow(case.frame, 'RGB', [case.record])
    assert len(worker.tasks) == 1


def test_close_failure_does_not_skip_model_release(pipeline_case):
    case = pipeline_case
    case.step(1)
    def fail():
        raise RuntimeError('optional close failed')
    case.pipeline._lk_shadow_worker.close = fail
    case.pipeline.close()
    assert case.calls == ['detector_close', 'reid_close']


def test_shutdown_logs_aggregate_without_issuing_control_or_waiting_for_new_frames(pipeline_case):
    import json
    case = pipeline_case
    case.step(1)
    logs = []
    case.pipeline.logger = SimpleNamespace(info=lambda *args: logs.append(args))
    case.pipeline.close()
    assert logs[0][0] == 'lk_shadow_summary %s'
    payload = json.loads(logs[0][1])
    assert payload['complete'] is True
    assert payload['cadence'] == 'processed_frames_only'
    assert payload['identity_authority'] is False and payload['motion_authority'] is False
    assert payload['worker']['submitted'] == 1
    assert case.calls == ['detector_close', 'reid_close']
