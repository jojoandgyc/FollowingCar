"""CAP812/816/819 clock replay and real publication path; no device/model IO."""
import ast
from dataclasses import replace
import inspect
import logging
import textwrap

import pytest

import request_0513_modular as runtime
from car_control_modular.short_follow import ShortFollowConfig, ShortFollowController, ShortFollowObservation
from car_control_modular.short_follow_adapter import ShortFollowAdapter
from rk_vision.detector_continuation import DetectorObservation, DetectorProof, continuation_reason
from test_processing_late_identity import (
    current_full, late, authority300, setup, owner, main_identity_publication,
)


def test_cap816_rechecks_before_recorded_cap819_completion_without_extending_proof():
    # Actual capture clocks, detector-completion age and CAP819 processing
    # components from run_20261010_202921, main log10766/10786/10852.
    verified = DetectorObservation(812, 35739.367821647, 0.,
        (167.9169006, 0., 348.5984497, 479.0950317), .95, 640, 480)
    current = replace(verified, capture=816, timestamp=35739.594371324)
    proof = DetectorProof(1, 9, verified, verified, 2, permission='similar_follow')
    detected_at = current.timestamp + .10671
    assert current.timestamp - verified.timestamp < .25  # old capture-only gate skipped
    assert continuation_reason(proof, current, detected_at, 60.) == 'full_recheck_due'
    # Replay, not a benchmark: use the measured next full frame's post-YOLO
    # tail as a conservative additional workload on CAP816's detector time.
    replay_full_finished = detected_at + (.24802 - .12593)
    assert replay_full_finished < verified.timestamp + .50
    assert 35739.758493129 + .28074 > verified.timestamp + .50
    assert proof.deadline == pytest.approx(verified.timestamp + .60)
    # Healthy, earlier frames retain the fast lane rather than forcing full
    # on every detection. Neither check can slide the original deadline.
    earlier = replace(current, capture=814, timestamp=verified.timestamp+.10)
    assert continuation_reason(proof, earlier, earlier.timestamp+.08, 60.) is None


@pytest.fixture
def paired_full(current_full):
    a, o = current_full, current_full.owner
    # The imported scene was published at capture+281 ms. Shift only this
    # newer source 1 ms so replaying 280.74 ms never reverses its fake clock.
    a.capture += .001
    o._active_capture_timestamp = a.capture
    o._rknn_pipeline.last_identity_processing['capture_timestamp'] = a.capture
    a.observation['sample_metadata']['capture_timestamp'] = a.capture
    ctl = ShortFollowController(ShortFollowConfig(enabled=True))
    ctl.activate(1, a.previous.timestamp-.01)
    ctl.update(ShortFollowObservation(1, a.previous.capture, a.previous.timestamp,
        a.clock.now-.03, 2.2, .2), a.clock.now)
    o._short_follow = ctl
    o._short_follow_adapter = ShortFollowAdapter(o, ctl, logging.getLogger(__name__))
    return a


def consume_main_branch(a, local):
    """Run the actual final stale-vs-control branch after real publication."""
    method = ast.parse(textwrap.dedent(inspect.getsource(runtime.PersonTracker.process_external_frame)))
    branch = next(node for node in method.body[0].body if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name) and node.test.id == 'stale_result_discarded'
        and any(isinstance(n, ast.Attribute) and n.attr == '_consume_track_records'
                for n in ast.walk(node)))
    block = ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[]))
    calls = []
    a.owner._consume_track_records = lambda records, *args, **kw: calls.append(records)
    a.owner._handle_stale_vision_result = lambda **kw: pytest.fail('fresh full result discarded')
    local.update(control_records=a.records, lateral_candidate=None)
    exec(compile(block, inspect.getsourcefile(runtime.PersonTracker), 'exec'), vars(runtime), local)
    assert calls == [a.records]


@pytest.mark.parametrize('age', [.28074, .349])
def test_current_full_uses_capture_age_and_reaches_normal_paired_consumption(paired_full, age):
    a, o = paired_full, paired_full.owner
    a.clock.now = a.capture + age
    o._depth_async_scheduler.worker_tick(now=a.clock.now)
    before = o._short_follow.snapshot().plan
    depth = o._depth30_linear_snapshot
    local = main_identity_publication(a, processing=.24802)
    assert not local['stale_result_discarded']
    assert o._late_visual_current_follow and not o._late_visual_identity_only
    proof = o._validated_visual_observation
    assert proof.capture == a.cap and proof.timestamp == a.capture
    assert proof.expires_at == pytest.approx(a.capture+.5)
    assert o._depth_async_scheduler.publication_snapshot()[1].capture_frame_id == a.cap
    assert o._depth30_linear_snapshot is depth
    after = o._short_follow.snapshot().plan
    # A current yaw may replace an unexpired pair, but never resurrect a pair
    # whose physical depth/previous visual deadline has already elapsed.
    assert after.depth_timestamp == before.depth_timestamp
    assert after.expires_at == before.expires_at
    if a.clock.now < before.expires_at:
        assert after.yaw_capture_id == a.cap
    else:
        assert after is before
    consume_main_branch(a, local)


@pytest.mark.parametrize('fault', ['uid0', 'wrong_uid', 'raw', 'rejected', 'pending',
    'features', 'detector_only', 'duplicate', 'full_admission_age', 'expired', 'future', 'old_proof',
    'epoch', 'stop', 'search', 'unsafe', 'geometry', 'worker', 'configured_window',
    'stop_during_submit', 'epoch_during_submit'])
def test_late_full_exception_cannot_launder_identity_clock_or_safety(paired_full, monkeypatch, fault):
    a, o = paired_full, paired_full.owner
    if fault == 'uid0': a.records[0].reid_uid = 0
    elif fault == 'wrong_uid': o._follow_controller.active_target_id = 2
    elif fault == 'raw': a.records[0].track_id += 1
    elif fault == 'rejected': a.data['identity_control_rejected'] = True
    elif fault == 'pending': a.data['identity_recheck_pending'] = True
    elif fault == 'features': o._rknn_pipeline.last_identity_processing['full_features_current'] = False
    elif fault == 'detector_only': o._rknn_pipeline.last_identity_processing['mode'] = 'detector_continuation'
    elif fault == 'duplicate': o._identity_processing_watermark = (a.cap, a.capture)
    elif fault == 'full_admission_age': a.clock.now = a.capture+.350
    elif fault == 'expired': a.clock.now = a.capture+.5
    elif fault == 'future': a.clock.now = a.capture-.001
    elif fault == 'old_proof': o._validated_visual_observation = replace(a.previous, expires_at=a.capture)
    elif fault == 'epoch': o._depth_async_scheduler.revoke('test_stop')
    elif fault == 'stop': o._explicit_stop_requested = True
    elif fault == 'search': o._follow_controller.search_state = 'searching'
    elif fault == 'unsafe': o._depth_roi_safety_clear = False
    elif fault == 'geometry': o._rknn_pipeline.tracker.last_identity_observations = []
    elif fault == 'worker': o._longitudinal_thread.is_alive = lambda: False
    elif fault == 'configured_window': monkeypatch.setattr(runtime, 'ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC', .25)
    else:
        original = o._depth_async_scheduler.submit
        def revoked(*args, **kwargs):
            if fault == 'stop_during_submit': o._explicit_stop_requested = True
            else: o._depth_async_scheduler.revoke('test_stop')
            return original(*args, **kwargs)
        monkeypatch.setattr(o._depth_async_scheduler, 'submit', revoked)
    before = o._short_follow.snapshot().plan
    local = main_identity_publication(a, processing=.24802)
    assert local['stale_result_discarded']
    assert not o._late_visual_current_follow
    assert o._short_follow.snapshot().plan is before


def test_repeated_new_full_does_not_slide_the_capture_deadline(paired_full):
    a, o = paired_full, paired_full.owner
    first = main_identity_publication(a, processing=.24802)
    assert not first['stale_result_discarded']
    deadline = o._validated_visual_observation.expires_at
    a.clock.now += .02
    second = main_identity_publication(a, processing=.24802)
    assert second['stale_result_discarded']
    assert not o._late_visual_current_follow
    assert o._validated_visual_observation.expires_at == deadline
