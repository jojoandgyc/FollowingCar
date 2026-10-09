"""CAP358--405 queue cancellation is not a new motor-stop authorization.

Real writer, immutable lateral intent and extracted production finalizer with
fake clock/serial only. No camera, encoder device or motor thread is opened.
"""
import ast
import logging
import threading
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from car_control_modular.lateral_intent import LateralIntentStore
from test_cap837_turn_buildup import image_intent
from test_follow_wheel_periodic import setup_periodic
from test_visible_wheel_continuity import feedback


@lru_cache(maxsize=1)
def main_class():
    path = Path(__file__).resolve().parents[2]/'request_0513_modular.py'
    tree = ast.parse(path.read_text())
    return path, next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PersonTracker')


def limited_intent(now=10., cap=379):
    return replace(image_intent(now, cap, -1,
        mode='yaw_only', bbox_quality='limited', response_boost_allowed=False,
        base_percent=0, base_rpm=0, initial_correction_rpm=-7,
        reason='target_visible_low_quality_yaw'), correction_limit_rpm=7.)


def setup_handoff(monkeypatch):
    r, o, d, s, clock, state = setup_periodic(monkeypatch)
    r.config.follow_forward_loss_handoff_enable = True
    r.config.follow_cross_brake_enable = True
    r.config.follow_cross_brake_mode = 'zero'
    o._action_runtime = r
    o._action_command_revision = 10
    o.stop_action_execution = False
    o.person_detected_flag = False
    o._lateral_intent_store = LateralIntentStore()
    state[:] = [0., -7., 11., 11.]
    o._vision_control_state = 'target_visible_low_quality'
    o._lateral_intent_store.publish(limited_intent())
    r.get_steering_feedback = lambda: feedback(clock[0], -7, 7)

    # Bind the actual small main-program adapter without constructing the
    # PersonTracker or replacing its decision with a test-only implementation.
    path, cls = main_class()
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_finish_limited_yaw_handoff')
    namespace = {'logger': logging.getLogger('cap358-test'),
                 'time': SimpleNamespace(monotonic=lambda: clock[0])}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    o._finish_limited_yaw_handoff = MethodType(namespace[method.name], o)
    return r, o, d, s, clock, state


def test_queue_only_finish_preserves_live_left_yaw_without_zero(monkeypatch):
    r, o, d, _, clock, _ = setup_handoff(monkeypatch)
    r._service_follow_wheels()
    assert d.pairs == [(-7, -7)]
    clock[0] = 10.05
    assert o._finish_limited_yaw_handoff(1, 379, True)
    assert not o.stop_action_execution
    assert o._action_command_revision == 10  # Publishing already made the new revision.
    r._service_follow_wheels()
    assert d.pairs == [(-7, -7), (-7, -7)]
    assert not d.stops


@pytest.mark.parametrize('failure', [
    'not_published', 'wrong_uid', 'wrong_capture', 'expired', 'forward_intent',
    'reliable_not_limited', 'hold_zero', 'zero_axes', 'positive_depth',
    'periodic_disabled', 'stop_pending', 'person_pending', 'explicit_stop',
    'shutdown', 'search', 'identity_lost', 'park_pending',
])
def test_unqualified_handoff_keeps_stop_semantics(monkeypatch, failure):
    r, o, d, _, clock, state = setup_handoff(monkeypatch)
    intent = o._lateral_intent_store.snapshot()
    published = failure != 'not_published'
    if failure == 'wrong_uid': o._follow_controller.active_target_id = 2
    if failure == 'wrong_capture':
        o._lateral_intent_store.publish(replace(intent, capture_frame_id=378))
    if failure == 'expired': clock[0] = intent.valid_until+.01
    if failure == 'forward_intent':
        o._lateral_intent_store.publish(replace(intent, mode='forward'))
    if failure == 'reliable_not_limited':
        o._lateral_intent_store.publish(replace(intent, bbox_quality='reliable'))
    if failure == 'hold_zero':
        o._lateral_intent_store.publish(replace(intent, hold_zero=True))
    if failure == 'zero_axes': state[1] = 0
    if failure == 'positive_depth': state[0] = 24
    if failure == 'periodic_disabled': r.config.follow_wheel_period_sec = 0
    if failure == 'stop_pending': o.stop_action_execution = True
    if failure == 'person_pending': o.person_detected_flag = True
    if failure == 'explicit_stop': o._explicit_stop_requested = True
    if failure == 'shutdown': o._runtime_shutdown_requested = True
    if failure == 'search': o.search_state = 'searching'
    if failure == 'identity_lost': o._vision_control_state = 'lost_confirming'
    if failure == 'park_pending': o._near_yaw_park_request = object()
    assert not o._finish_limited_yaw_handoff(1, 379, published)
    assert o.stop_action_execution
    assert not d.pairs and not d.stops


def test_real_stop_arriving_after_queue_handoff_is_not_swallowed(monkeypatch):
    r, o, d, _, clock, _ = setup_handoff(monkeypatch)
    r._service_follow_wheels()
    assert o._finish_limited_yaw_handoff(1, 379, True)
    o.stop_action_execution = True
    clock[0] = 10.05
    r._service_follow_wheels()
    assert d.pairs == [(-7, -7), (0, 0)]
    assert o.stop_action_execution


def test_limited_queue_handoff_still_brakes_actual_opposing_inner_wheel(monkeypatch):
    r, o, d, _, clock, _ = setup_handoff(monkeypatch)
    r._service_follow_wheels()
    clock[0] = 10.05
    assert o._finish_limited_yaw_handoff(1, 379, True)
    r.get_steering_feedback = lambda: feedback(clock[0], 25, 30)
    r._service_follow_wheels()
    assert d.pairs[-1] == (0, 0)
    assert r._visible_wheel_guard.pending_signs == (-1, 1)


def test_danger_still_owns_output_after_queue_handoff(monkeypatch):
    r, o, d, _, clock, _ = setup_handoff(monkeypatch)
    r._service_follow_wheels()
    assert o._finish_limited_yaw_handoff(1, 379, True)
    r.hard_stop_check = lambda _: True
    clock[0] = 10.05
    before = len(d.pairs)
    r._service_follow_wheels()
    assert all(pair == (0, 0) for pair in d.pairs[before:])
    assert d.stops


def test_fresh_normal_forward_can_take_over_after_limited_queue_handoff(monkeypatch):
    r, o, d, _, clock, state = setup_handoff(monkeypatch)
    r._service_follow_wheels()
    assert o._finish_limited_yaw_handoff(1, 379, True)
    clock[0] = 10.05
    o._vision_control_state = 'target_visible_depth_valid'
    state[:2] = [40., -10.]
    o._lateral_yaw_revision += 1
    o._lateral_intent_store.publish(image_intent(clock[0], 384, -1))
    r.get_steering_feedback = lambda: feedback(clock[0], 5, 5)
    r._service_follow_wheels()
    assert d.pairs[-1] == (30, -50)


@pytest.mark.parametrize('new_yaw', [-6., -7.])
def test_new_limited_yaw_revision_rebuilds_once_without_unnecessary_zero(monkeypatch, new_yaw):
    r, o, d, _, clock, state = setup_handoff(monkeypatch)
    r._service_follow_wheels()
    clock[0] = 10.05
    reads = []
    def new_observation():
        reads.append(1)
        if len(reads) == 1:
            state[1] = new_yaw
            o._lateral_yaw_revision += 1
            o._lateral_intent_store.publish(limited_intent(clock[0], 382))
        return feedback(clock[0], -7, 7)
    r.get_steering_feedback = new_observation
    r._service_follow_wheels()
    assert len(reads) == 2
    assert d.pairs == [(-7, -7), (new_yaw, new_yaw)]


def test_repeated_limited_yaw_replacements_cannot_create_unbounded_retry(monkeypatch):
    r, o, d, _, clock, state = setup_handoff(monkeypatch)
    reads = []
    def changing():
        reads.append(1)
        o._lateral_yaw_revision += 1
        return feedback(clock[0], -7, 7)
    r.get_steering_feedback = changing
    r._service_follow_wheels()
    assert len(reads) == 2
    assert d.pairs == [(0, 0)]


def test_rebuilt_opposite_yaw_still_stops_before_reversing_wheels(monkeypatch):
    r, o, d, _, clock, state = setup_handoff(monkeypatch)
    r._service_follow_wheels()
    clock[0] = 10.05
    reads = []
    def changed_direction():
        reads.append(1)
        if len(reads) == 1:
            state[1] = 7.
            o._lateral_yaw_revision += 1
            o._lateral_intent_store.publish(replace(limited_intent(clock[0], 382),
                initial_correction_rpm=7, x_ratio=.8))
        return feedback(clock[0], -7, 7)
    r.get_steering_feedback = changed_direction
    r._service_follow_wheels()
    assert len(reads) == 2
    assert d.pairs == [(-7, -7), (0, 0)]
    assert r._visible_wheel_guard.pending_signs == (1, -1)


def test_limited_outer_deceleration_uses_existing_two_post_zero_samples(monkeypatch):
    r, o, d, _, clock, state = setup_handoff(monkeypatch)
    # Establish real forward provenance before the quality-only transition.
    o._vision_control_state = 'target_visible_depth_valid'
    state[:2] = [44., -10.]
    r.get_steering_feedback = lambda: feedback(clock[0], 30, 40)
    r._service_follow_wheels()
    assert d.pairs[-1] == (34, -54)
    o._vision_control_state = 'target_visible_low_quality'
    state[:2] = [0., -7.]
    for timestamp, wheel_pair in [(10.05, (30, 40)), (10.10, (0, 20)), (10.15, (0, 18))]:
        clock[0] = timestamp
        o._lateral_intent_store.publish(limited_intent(timestamp, 379))
        r.get_steering_feedback = lambda p=wheel_pair: feedback(clock[0], *p)
        r._service_follow_wheels()
    assert d.pairs[-3:] == [(0, 0), (0, 0), (-7, -7)]


@pytest.mark.parametrize('case', ['qualified', 'other_reason', 'depth_thread', 'different_uid',
    'no_target', 'explicit_stop', 'shutdown', 'near_park', 'soft_stop', 'legacy_writer'])
def test_real_main_deferral_uses_existing_target_before_publisher(case):
    path, cls = main_class()
    process = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                   and n.name == '_process_detections_modular')
    assignment = next(n for n in ast.walk(process) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'defer_limited_yaw_interrupt'
                              for t in n.targets))
    invalidate = next(n for n in ast.walk(process) if isinstance(n, ast.If)
                      and isinstance(n.test, ast.Name) and n.test.id == 'defer_limited_yaw_interrupt'
                      and any(isinstance(x, ast.Attribute) and x.attr == '_action_command_revision'
                              for x in ast.walk(n)))
    publish = next(n for n in ast.walk(process) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute)
                   and n.func.attr == '_publish_lateral_intent_from_decision')
    finish = next(n for n in ast.walk(process) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute)
                  and n.func.attr == '_finish_limited_yaw_handoff')
    assert assignment.lineno < invalidate.lineno < publish.lineno < finish.lineno
    assert any(isinstance(n, ast.Attribute) and n.attr == 'clear_action_queue'
               for n in ast.walk(assignment))

    owner = SimpleNamespace(action_queue_lock=threading.Lock(), _action_command_revision=10,
        _action_runtime=SimpleNamespace(config=SimpleNamespace(follow_wheel_period_sec=.05)))
    decision = SimpleNamespace(reason='target_visible_low_quality_yaw', stop_action_execution=True,
        clear_action_queue=True, explicit_stop_requested=False, shutdown_requested=False,
        near_yaw_park_requested=False, soft_stop_requested=False)
    namespace = dict(self=owner, decision=decision, decision_target=SimpleNamespace(track_id=1),
        active_before=1, control_source='vision', longitudinal_only=False,
        LATERAL_INTENT_CONTROL_ENABLE=True, depth_authority=True)
    # selected_target_id is intentionally NOT defined: main defines it later.
    if case == 'other_reason': decision.reason = 'search_left'
    if case == 'depth_thread': namespace['control_source'] = 'depth30'
    if case == 'different_uid': namespace['decision_target'].track_id = 2
    if case == 'no_target': namespace['decision_target'] = None
    if case == 'explicit_stop': decision.explicit_stop_requested = True
    if case == 'shutdown': decision.shutdown_requested = True
    if case == 'near_park': decision.near_yaw_park_requested = True
    if case == 'soft_stop': decision.soft_stop_requested = True
    if case == 'legacy_writer': owner._action_runtime.config.follow_wheel_period_sec = 0
    exec(compile(ast.Module(body=[assignment, invalidate], type_ignores=[]), str(path), 'exec'), namespace)
    assert namespace['defer_limited_yaw_interrupt'] == (case == 'qualified')
    assert owner._action_command_revision == (11 if case == 'qualified' else 10)
