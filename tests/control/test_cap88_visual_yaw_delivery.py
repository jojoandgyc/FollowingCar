"""Real perception adapter: new yaw must not wait for another distance sample."""
from dataclasses import replace
import logging
from types import SimpleNamespace

import pytest

from car_control_modular.control_types import DepthTargetObservation, PersonTarget, SensorFrame, DistanceState
from car_control_modular.detector_identity_lease import ValidatedVisualObservation
from car_control_modular.short_follow import ShortFollowConfig, ShortFollowController, ShortFollowObservation
from car_control_modular.short_follow_adapter import ShortFollowAdapter
from test_early_depth_roi_publication import early, validate, publish
from test_depth_async_runtime import async_scene
from test_depth_optimistic_transaction import scene
from test_turn_depth_scheduling import owner
from test_visual_depth_optimistic_transaction import visual


@pytest.fixture
def yaw_scene(monkeypatch):
    clock = SimpleNamespace(now=100.15)
    monkeypatch.setattr('car_control_modular.detector_identity_lease.time.monotonic', lambda: clock.now)
    controller = ShortFollowController(ShortFollowConfig(enabled=True, depth_ttl_sec=.35))
    controller.activate(1, 99.8)
    controller.update(ShortFollowObservation(1, 83, 100., 100.12, 1.3, .2756), clock.now)
    owner = SimpleNamespace(running=True, search_state='none',
        _follow_controller=SimpleNamespace(active_target_id=1, search_state='none'),
        _validated_visual_observation=None, _detector_identity_lease=None,
        _action_runtime=SimpleNamespace(get_steering_heading_at=lambda _ts: None,
                                       get_steering_feedback=lambda: None))
    adapter = ShortFollowAdapter(owner, controller, logging.getLogger(__name__))
    clock.now = 100.36
    target = PersonTarget((138.88, 0., 338.88, 479.), 1, .93, 95800.,
        DepthTargetObservation((174.88, 2., 374.88, 477.), 1, 2, 88, 100.2))
    owner._validated_visual_observation = ValidatedVisualObservation(1, 2, 88, 100.2, 100.32, 100.7, 'full')
    return SimpleNamespace(clock=clock, owner=owner, controller=controller, adapter=adapter, target=target)


def send(a):
    return a.adapter.publish_visual_lateral(a.target, 640, 480, 88, 100.2, now=a.clock.now)


def test_cap88_real_detector_center_replaces_lagged_display_without_new_depth(yaw_scene):
    a = yaw_scene
    before = a.controller.snapshot().plan
    assert before.left_rpm < 0 < before.right_rpm
    plan = send(a)
    assert (plan.left_rpm, plan.right_rpm) == (0, 0)
    assert plan.yaw_capture_id == 88 and plan.capture_id == 83
    assert plan.yaw_center_x_ratio == pytest.approx(.4295)
    for key in ('depth_timestamp', 'capture_timestamp', 'expires_at', 'base_rpm', 'i_rpm', 'integral_dt_sec'):
        assert getattr(plan, key) == getattr(before, key)


def test_normal_visual_lateral_only_frame_updates_before_ranging(yaw_scene):
    a = yaw_scene
    frame = SensorFrame(width=640, height=480, persons=[a.target], capture_frame_id=88,
        capture_timestamp=100.2, distance_state=DistanceState(source='vision_depth', source_detail='depth_async_pending'))
    assert a.adapter.handle(frame, a.target, is_fresh_depth=False, control_source='vision',
        target_steerable=True, low_quality_visible=False, now=a.clock.now)
    plan = a.controller.snapshot().plan
    assert plan.yaw_capture_id == 88 and plan.left_rpm == plan.right_rpm == 0
    assert plan.depth_timestamp == 100.12


def test_forward_base_survives_new_center_without_new_depth(yaw_scene):
    a = yaw_scene
    a.controller.update(ShortFollowObservation(1, 83, 100., 100.3, 2.2, .2756), a.clock.now)
    before = a.controller.snapshot().plan
    assert before.forwarding and before.left_rpm < before.right_rpm
    plan = send(a)
    assert plan.left_rpm == plan.right_rpm == before.base_rpm > 0
    assert plan.expires_at == before.expires_at


@pytest.mark.parametrize('fault', ['wrong_uid', 'wrong_raw', 'old_cap', 'old_time', 'predicted',
                                 'invalid_box', 'missing_box', 'identity', 'stop', 'search', 'reacquire'])
def test_unqualified_visual_attempt_keeps_old_pair_not_new_stop(yaw_scene, fault):
    a = yaw_scene
    obs = a.target.depth_observation
    if fault == 'wrong_uid': obs = replace(obs, target_id=2)
    if fault == 'wrong_raw': obs = replace(obs, raw_track_id=3)
    if fault == 'old_cap': obs = replace(obs, capture_frame_id=87)
    if fault == 'old_time': obs = replace(obs, capture_timestamp=100.1)
    if fault == 'predicted': obs = replace(obs, source='kalman')
    if fault == 'invalid_box': obs = replace(obs, bbox=(float('nan'), 0., 300., 480.))
    if fault == 'missing_box': obs = None
    if fault == 'identity': a.owner._validated_visual_observation = False
    if fault == 'stop': a.owner._explicit_stop_requested = True
    if fault == 'search': a.owner._follow_controller.search_state = 'searching'
    if fault == 'reacquire': a.owner._reacquire_depth_pending = True
    a.target = replace(a.target, depth_observation=obs)
    before = a.controller.snapshot()
    assert send(a) is None
    assert a.controller.snapshot() == before


def test_repeated_and_late_visual_do_not_refresh_deadline_or_reverse(yaw_scene):
    a = yaw_scene
    plan = send(a)
    assert send(a) is None and a.controller.snapshot().plan is plan
    a.clock.now = plan.expires_at + .001
    assert send(a) is None and a.controller.snapshot().plan is plan


def test_real_early_roi_publication_updates_yaw_without_a_depth_task(early):
    a, o = early, early.o
    ctl = ShortFollowController(ShortFollowConfig(enabled=True))
    ctl.activate(1, a.stamp-.2)
    ctl.update(ShortFollowObservation(1, a.cap-1, a.stamp-.02,
        a.clock[0]-.03, 2.2, .2), a.clock[0])
    o._short_follow = ctl
    o._short_follow_adapter = ShortFollowAdapter(o, ctl, logging.getLogger(__name__))
    before = ctl.snapshot().plan
    assert validate(a) and publish(a)
    after = ctl.snapshot().plan
    assert after is not before and after.yaw_capture_id == a.cap
    assert after.base_rpm == before.base_rpm and after.depth_timestamp == before.depth_timestamp
    assert after.expires_at == before.expires_at
    assert not o.calls and a.camera._last_accepted_ts == 0
