"""Real visual decision -> live depth grant -> fake wheel writer, no hardware."""
from dataclasses import replace
import pickle
import queue
import threading
from types import SimpleNamespace

import pytest
import request_0513_modular as runtime
from car_control_modular.control_types import DepthTargetObservation, DistanceState, HazardState
from car_control_modular.depth_async_scheduler import DepthAsyncScheduler
from test_depth_authority_250 import authority, advance, seed, writer
from test_distance_pi_controller import configured
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


@pytest.fixture
def visual_authority(authority, setup, monkeypatch):
    a = authority
    _, a.controller, _ = configured(setup, visible_steering_pid_enable=True,
        distance_target_motion_control_enable=False,
        distance_pi_braking_stop_distance_m=1.1,
        depth_longitudinal_sample_max_age_sec=.25)
    obj = a.owner
    obj._follow_controller = a.controller
    a.controller._live_longitudinal_authority_reader = obj._fresh_depth_linear_snapshot
    a.controller._braking_execution_bound_reader = lambda uid, now: 40.
    stamp, original = seed(a, distance=3., rpm=40.)
    target = a.controller.last_selected_target
    a.target = target
    obj._active_capture_timestamp = stamp
    obj._last_frame_distance_state = a.frame(3.).distance_state
    obj._last_control_decision_log_key = None
    obj._last_pid_trace_log_ts = stamp
    obj._last_target_loss_trace_frame = -1
    obj._last_person_reid_debug_by_stable_id = {}
    obj._get_obstacle_status = lambda: {}
    obj._current_hazard_state_for_controller = lambda: HazardState()
    obj._maybe_schedule_historical_direction_backfill = lambda *args: None
    obj._clear_action_queue = lambda *args: pytest.fail("ordinary deferred visual cleared queue")
    obj._persons_to_targets = lambda *args, **kwargs: [a.target]
    obj._distance_runtime = SimpleNamespace(
        select_target=lambda targets: targets[0] if targets else None,
        get_frame_distance_state=lambda *args, **kwargs: pytest.fail("visual started a depth scan"),
        last_distance_state=obj._last_frame_distance_state,
        _last_vision_depth_target=target)
    obj._refresh_visual_depth_linear_authority = lambda *args, **kwargs: pytest.fail("visual renewed depth")
    obj._last_dispatched_action = runtime.ACTION_FORWARD
    obj.search_direction = None
    obj.action_queue = queue.Queue()
    obj._action_runtime_started = True
    obj._longitudinal_context_lock = threading.Lock()
    obj._longitudinal_context = None
    obj._longitudinal_wake_event = threading.Event()
    obj._depth_async_scheduler = DepthAsyncScheduler()
    obj._depth_async_scheduler.worker_tick(now=stamp)
    obj._longitudinal_thread = SimpleNamespace(is_alive=lambda: True)
    action, backend = writer(a)
    action.get_steering_feedback = lambda: a.feedback
    a.action, a.backend, a.stamp, a.original = action, backend, stamp, original
    return a


def new_visual(a, cap, x=440.):
    box = (x, 20., x+160., 460.)
    obs = DepthTargetObservation(bbox=box, target_id=1, raw_track_id=1, capture_frame_id=cap,
                                capture_timestamp=a.clock.now, source="yolo_detector")
    a.target = replace(a.target, bbox=box, depth_observation=obs)
    a.owner.frame_index += 1
    a.owner._active_capture_frame_id = cap
    a.owner._active_capture_timestamp = a.clock.now
    return [(a.target.bbox, 1, a.target.confidence, a.target.area)]


@pytest.mark.parametrize("x", [70., 240., 440.])
@pytest.mark.parametrize("display", [None, 1.1, 3.])
def test_visual_defer_never_clears_pi_or_grant_and_writer_still_moves(visual_authority, x, display):
    a = visual_authority
    obj = a.owner
    advance(a, a.stamp+.04)
    persons = new_visual(a, 800, x)
    obj._distance_runtime.last_distance_state = replace(obj._distance_runtime.last_distance_state,
                                                       used_distance_m=display)
    state = pickle.dumps(vars(a.controller._distance_pid._distance_pi))
    deadline = obj._depth30_linear_timing.depth_expires_at
    watermark = obj._depth30_linear_sample_watermark
    assert obj._queue_actions_for_persons(640, 480, persons)
    assert pickle.dumps(vars(a.controller._distance_pid._distance_pi)) == state
    assert obj._depth30_linear_snapshot[3] == a.stamp
    assert obj._depth30_linear_sample_watermark == watermark
    assert obj._depth30_linear_timing.depth_expires_at == deadline
    assert not obj._explicit_stop_requested and not obj.stop_action_execution
    assert obj._lateral_intent_store.snapshot().target_id == 1
    a.action._service_follow_wheels()
    assert a.backend.pairs[-1][0] > 0 and a.backend.pairs[-1][1] < 0


def test_continuous_visual_does_not_renew_depth_and_real_expiry_still_stops(visual_authority):
    a = visual_authority
    for i, age in enumerate((.04, .1, .17, .21, .251)):
        advance(a, a.stamp+age)
        a.owner._depth_async_scheduler.worker_tick(now=a.clock.now)
        persons = new_visual(a, 800+i, 240.)
        assert a.owner._queue_actions_for_persons(640, 480, persons)
        a.action._service_follow_wheels()
        if age < .25:
            assert a.backend.pairs[-1][0] > 0 and a.backend.pairs[-1][1] < 0
        else:
            assert a.backend.pairs[-1][:2] == (0, 0)
        assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(a.stamp+.25)


def test_display_hold_keeps_original_clock_and_cannot_be_fresh(visual_authority):
    a = visual_authority
    advance(a, a.stamp+.04)
    held = a.owner._async_visual_distance_display(a.target)
    assert held.raw_distance_m is None and held.sample_timestamp is None
    assert held.observation_timestamp == a.stamp
    assert held.temporal_status == "deferred"
    assert not a.controller._is_fresh_depth_state(replace(a.frame(3.), distance_state=held))
    assert a.owner._async_visual_distance_display(replace(a.target, track_id=2)).used_distance_m is None
