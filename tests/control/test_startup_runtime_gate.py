"""Startup runtime ownership checks; no hardware constructors or threads."""
import threading

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction, ControlDecision
from test_lateral_zero_runtime import NOW, owner, _intent, _target
from test_longitudinal_authority_runtime import _process_fixture


def _startup_fixture(owner, decision):
    target = _process_fixture(owner, decision, lambda *args, **kwargs: pytest.fail("startup PI"))
    ctl = owner._follow_controller
    ctl.active_target_id = None
    ctl.last_steering_pid_result = None
    owner._longitudinal_context_lock = threading.Lock()
    owner._longitudinal_context = {"old_roi": True}
    owner._depth30_linear_snapshot = ("forward", 35, 1, NOW - .04)
    owner._commit_depth_linear_decision = lambda *args, **kwargs: pytest.fail("startup depth grant")
    owner._refresh_visual_depth_linear_authority = lambda *args, **kwargs: pytest.fail("startup visual grant")
    owner._recorded_decisions = []
    owner._publish_follow_recording = lambda frame, result, source: owner._recorded_decisions.append(result)
    owner._target_quality_debug = lambda target: "test"
    owner._target_region_debug = lambda *args: "test"
    owner._log_distance_stop_trigger = lambda *args: None
    return target


@pytest.mark.parametrize("source", ["vision", "depth30"])
@pytest.mark.parametrize("action", [
    ControlAction.forward(50, "test"), ControlAction.backward(20, "test"),
    ControlAction.rotate_left("test"), ControlAction.rotate_right("test"),
    ControlAction.steer_left(40, 100, 100, "test"), ControlAction.steer_right(40, 100, 100, "test"),
])
def test_unlocked_controller_motion_cannot_escape_runtime(owner, source, action):
    target = _startup_fixture(owner, ControlDecision(actions=[action], reason="legacy_motion"))
    _intent(owner)
    actions = owner._process_detections_modular(
        640, 480, [(target.bbox, 1, .9, target.area)], control_source=source,
    )
    assert actions == [runtime.ACTION_STOP]
    assert owner._last_control_decision_reason == "initial_lock_required"
    assert owner._depth30_linear_snapshot is None
    assert owner._longitudinal_context is None
    assert owner._lateral_intent_store.snapshot() is None
    assert owner._lateral_intent_owned_frame == -1
    assert owner._current_forward_percent == owner._current_rotate_raw_target == 0
    assert not owner._queued_calls  # No lateral writer can steal startup STOP.


def test_initial_stop_survives_depth_owner_filter_and_actual_queue(owner):
    target = _startup_fixture(owner, ControlDecision(
        actions=[ControlAction.stop("initial_candidate_confirmation_hold", brake_hold=False)],
        soft_stop_requested=True, reason="initial_candidate_confirmation_hold",
    ))
    owner._last_target_loss_trace_frame = -1
    owner._queue_actions_for_persons_locked(
        640, 480, [(target.bbox, 1, .9, target.area)],
        depth_use_latest=False, control_source="vision", target_steerable=True,
    )
    assert owner._queued_calls == [((runtime.ACTION_STOP,), "initial_candidate_confirmation_hold")]
    assert owner._recorded_decisions[-1].soft_stop_requested


def test_initial_runtime_gate_preserves_explicit_hazard_stop(owner):
    target = _startup_fixture(owner, ControlDecision(
        actions=[], explicit_stop_requested=True, reason="front_ir_emergency",
    ))
    assert owner._process_detections_modular(640, 480, [(target.bbox, 1, .9, target.area)]) == [runtime.ACTION_STOP]
    assert owner._explicit_stop_requested
    assert owner._last_control_decision_reason == "front_ir_emergency"
    assert not owner._recorded_decisions[-1].soft_stop_requested


@pytest.mark.parametrize("uid", [None, 2])
@pytest.mark.parametrize("action", [ControlAction.rotate_right("test"), ControlAction.stop("test")])
def test_lateral_publisher_requires_matching_locked_uid(owner, uid, action):
    owner._follow_controller.active_target_id = uid
    assert not owner._publish_lateral_intent_from_decision(
        width=640, target=_target(), runtime_actions=[action], control_source="vision",
        target_steerable=True, low_quality_visible=False,
    )
    assert owner._lateral_intent_store.snapshot() is None
    assert not owner._queued_calls


@pytest.mark.parametrize("source", ["default", "search", "lateral_intent_30hz"])
def test_final_rotate_gate_rejects_unlocked_legacy_packets(owner, source):
    owner._follow_controller.active_target_id = None
    owner._current_rotate_raw_source = source
    assert not owner._visible_rotate_command_allowed(runtime.ACTION_ROTATE_RIGHT)
    assert not owner._visible_rotate_command_allowed(runtime.ACTION_ROTATE_LEFT)


def test_locked_search_still_uses_legacy_rotate_gate(owner):
    owner._current_rotate_raw_source = "search"
    owner.search_state = "searching"
    assert owner._visible_rotate_command_allowed(runtime.ACTION_ROTATE_RIGHT)


def test_manual_lock_release_revokes_motion_caches_and_popped_version(owner):
    ctl = owner._follow_controller
    def release(reason):
        ctl.active_target_id = None
        ctl.search_state, ctl.search_direction, ctl.lost_confirm_frames = "none", None, 0
    ctl.clear_active_target = release
    owner.motor_io_lock = threading.Lock()
    owner._longitudinal_context_lock = threading.Lock()
    owner._longitudinal_context = {"old_roi": True}
    owner._depth30_linear_snapshot = ("forward", 35, 1, NOW - .04)
    owner._action_command_revision = 7
    owner._reset_search_geometry_reacquire = lambda: None
    owner._clear_visual_reacquire_hold = lambda reason: None
    _intent(owner)
    owner.clear_active_target("manual", stop_current=False)
    assert ctl.active_target_id is None
    assert owner._vision_control_state == "initial_wait"
    assert owner._depth30_linear_snapshot is None
    assert owner._longitudinal_context is None
    assert owner._lateral_intent_store.snapshot() is None
    assert owner._action_command_revision > 7
    assert owner._current_forward_percent == owner._current_rotate_raw_target == 0
    assert not owner._visible_rotate_command_allowed(runtime.ACTION_ROTATE_RIGHT)
