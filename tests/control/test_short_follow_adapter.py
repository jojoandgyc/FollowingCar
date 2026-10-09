"""Paired owner enters before all legacy normal-follow motor publications."""
from dataclasses import replace
import threading
import queue
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction, ControlDecision, DepthTargetObservation, DistanceState, SensorFrame, SteeringFeedback
from car_control_modular.detector_identity_lease import ValidatedVisualObservation
from car_control_modular.short_follow import ShortFollowConfig, ShortFollowController
from car_control_modular.short_follow_adapter import ShortFollowAdapter
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.action_command import ActionCommandSnapshot
from test_lateral_zero_runtime import NOW, owner
from test_longitudinal_authority_runtime import _process_fixture


@pytest.fixture
def paired(owner):
    target = _process_fixture(owner, ControlDecision(), lambda *_: pytest.fail("legacy PI"))
    owner._follow_controller.decide = lambda *_a, **_kw: pytest.fail("legacy normal decide")
    owner._commit_depth_linear_decision = lambda *_a, **_kw: pytest.fail("legacy grant")
    owner._publish_follow_recording = lambda *_a: None
    owner._refresh_visual_depth_linear_authority = lambda *_a, **_kw: pytest.fail("legacy visual grant")
    owner._validated_visual_observation = ValidatedVisualObservation(
        1, 1, 576, NOW-.09, NOW-.01, NOW+.4, "full")
    owner._short_follow = ShortFollowController(ShortFollowConfig(enabled=True))
    owner._short_follow_adapter = ShortFollowAdapter(owner, owner._short_follow, runtime.logger)
    owner._depth30_linear_snapshot = ("forward", 90, 1, NOW-.04)
    owner._last_target_loss_trace_frame = -1
    owner._distance_runtime.get_frame_distance_state = lambda *_a, **_kw: DistanceState(
        source="vision_depth", source_detail="depth_multi_region", raw_distance_m=2.,
        used_distance_m=2., sample_timestamp=NOW-.04)
    return SimpleNamespace(owner=owner, target=target)


def process(a, source="vision"):
    target = a.target
    return a.owner._process_detections_modular(640, 480,
        [(target.bbox, target.track_id, target.confidence, target.area)], control_source=source)


@pytest.mark.parametrize("source", ["vision", "depth30"])
def test_verified_frame_publishes_one_pair_before_legacy_decide(paired, source):
    a = paired
    assert process(a, source) == []
    state = a.owner._short_follow.snapshot()
    assert state.active and state.plan.moving
    assert state.plan.left_rpm > state.plan.right_rpm > 0  # target right
    assert 40 < max(state.plan.left_rpm, state.plan.right_rpm) <= state.plan.speed_cap_rpm
    assert state.plan.base_request_rpm == pytest.approx(state.plan.p_rpm + state.plan.i_rpm)
    assert state.plan.depth_timestamp == NOW-.04
    assert state.plan.capture_timestamp == NOW-.09
    assert a.owner._depth30_linear_snapshot is None
    assert not a.owner._queued_calls


@pytest.mark.parametrize("distance", [None, "duplicate", float("nan")])
def test_missing_or_duplicate_attempt_retains_complete_plan_without_renewal(paired, distance):
    a = paired
    process(a)
    prior = a.owner._short_follow.snapshot()
    state = a.owner._distance_runtime.get_frame_distance_state()
    if distance == "duplicate":
        state = replace(state, temporal_status="duplicate")
        a.owner._follow_controller._is_fresh_depth_state = lambda *_: False
    else:
        state = replace(state, used_distance_m=distance, raw_distance_m=distance)
    a.owner._distance_runtime.get_frame_distance_state = lambda *_a, **_kw: state
    assert process(a, "depth30") == []
    current = a.owner._short_follow.snapshot()
    assert current == prior
    assert current.plan is prior.plan
    assert not a.owner._queued_calls


def test_first_lock_suppresses_legacy_high_rpm_and_waits_for_independent_identity(paired):
    a = paired
    a.owner._validated_visual_observation = None
    a.owner._follow_controller.active_target_id = None
    def initial_decide(*_a, **_kw):
        a.owner._follow_controller.active_target_id = 1
        return ControlDecision(actions=[ControlAction.forward(90, "legacy_launch")], reason="legacy_launch")
    a.owner._follow_controller.decide = initial_decide
    assert process(a) == []
    assert a.owner._short_follow.snapshot().active
    assert a.owner._short_follow.snapshot().plan is None
    assert a.owner._depth30_linear_snapshot is None
    assert not a.owner._queued_calls


def test_search_to_follow_empty_actions_cannot_append_legacy_stop(paired):
    a = paired
    a.owner.search_state = "searching"
    a.owner._follow_controller.search_state = "searching"
    def reacquire(*_a, **_kw):
        a.owner._follow_controller.search_state = "none"
        return ControlDecision(actions=[ControlAction.forward(90, "legacy_reacquire")])
    a.owner._follow_controller.decide = reacquire
    a.owner._action_runtime.send_stop_with_brake_hold = lambda *_: pytest.fail("spurious handoff STOP")
    a.owner._queue_actions_for_persons_locked(640, 480,
        [(a.target.bbox, 1, .9, a.target.area)], depth_use_latest=False,
        control_source="vision", target_steerable=True)
    assert a.owner.search_state == "none"
    assert a.owner._short_follow.snapshot().plan.moving
    assert not a.owner._queued_calls


@pytest.mark.parametrize("change", ["identity", "search", "hazard", "front_ir", "low_quality", "uid"])
def test_explicit_adverse_or_search_handoff_cannot_keep_paired_motion(paired, change):
    a = paired
    process(a)
    frame = SensorFrame(width=640, height=480, persons=[a.target], distance_m=2.,
        distance_state=a.owner._distance_runtime.get_frame_distance_state(),
        capture_frame_id=576, capture_timestamp=NOW-.09)
    kw = dict(is_fresh_depth=True, control_source="vision", target_steerable=True,
              low_quality_visible=False, now=NOW)
    if change == "identity": a.owner._validated_visual_observation = False
    if change == "search": a.owner._follow_controller.search_state = "searching"
    if change == "hazard": frame = replace(frame, hazard=replace(frame.hazard, active=True))
    if change == "front_ir": frame = replace(frame, obstacles=replace(frame.obstacles, front=True))
    if change == "low_quality": kw["low_quality_visible"] = True
    if change == "uid": a.owner._follow_controller.active_target_id = 2
    assert not a.owner._short_follow_adapter.handle(frame, a.target, **kw)
    state = a.owner._short_follow.snapshot()
    assert not state.active and state.plan is None


def test_delayed_legacy_yaw_zero_and_periodic_yaw_do_not_mutate_pair(paired):
    a = paired
    process(a)
    prior = a.owner._short_follow.snapshot()
    assert not a.owner._publish_lateral_zero(None, "expired_old_yaw")
    a.owner._service_lateral_intent(NOW)
    assert a.owner._short_follow.snapshot() == prior
    assert not a.owner._queued_calls


@pytest.mark.parametrize("initial_distance", [1.3, 2.0])
def test_real_startup_controller_enters_pair_then_next_fresh_sample_moves(paired, monkeypatch, initial_distance):
    a = paired
    clock = SimpleNamespace(now=NOW)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock.now)
    a.owner._follow_controller = FollowSafetyController(FollowPolicyConfig(
        target_distance_m=1.4, target_distance_release_m=1.5,
        initial_target_confirm_frames=1, visible_steering_pid_enable=False,
        direction_history_enable=True))
    a.owner._validated_visual_observation = None
    a.target = replace(a.target, initial_identity_confirmed=True)
    a.owner._persons_to_targets = lambda *_a, **_kw: [a.target]
    range_state = [DistanceState(source="vision_depth", source_detail="depth_multi_region",
        raw_distance_m=initial_distance, used_distance_m=initial_distance, sample_timestamp=NOW-.04)]
    a.owner._distance_runtime.get_frame_distance_state = lambda *_a, **_kw: range_state[0]
    assert process(a) == []
    assert a.owner._follow_controller.active_target_id == 1
    assert a.owner._short_follow.snapshot().active
    assert a.owner._short_follow.snapshot().plan is None
    # The second image has a real bound identity but range is still unavailable.
    clock.now += .1
    a.owner.frame_index += 1
    a.owner._active_capture_frame_id += 1
    a.owner._active_capture_timestamp = clock.now-.02
    a.owner._validated_visual_observation = ValidatedVisualObservation(
        1, 1, a.owner._active_capture_frame_id, clock.now-.02, clock.now-.01, clock.now+.48, "full")
    range_state[0] = DistanceState(source="vision_depth", source_detail="depth_pending")
    assert process(a) == []
    assert a.owner._short_follow.snapshot().plan is None
    # A third independent Depth result publishes a complete bounded PI pair.
    clock.now += .03
    range_state[0] = DistanceState(source="vision_depth", source_detail="depth_multi_region",
        raw_distance_m=2., used_distance_m=2., sample_timestamp=clock.now-.01)
    assert process(a, "depth30") == []
    assert a.owner._short_follow.snapshot().plan.moving
    plan = a.owner._short_follow.snapshot().plan
    assert 40 < max(plan.left_rpm, plan.right_rpm) <= plan.speed_cap_rpm
    assert not a.owner._queued_calls


def test_explicit_manual_stop_is_not_cleared_by_new_frame_or_even_stop_ack(paired):
    a = paired
    process(a)
    a.owner._explicit_stop_requested = True
    a.owner._last_explicit_stop_reason = "manual_emergency"
    assert process(a) == [runtime.ACTION_STOP]
    state = a.owner._short_follow.snapshot()
    assert state.plan is None
    a.owner._short_follow_completed_stop_epoch = state.epoch
    a.owner._active_capture_timestamp = NOW+.01
    assert process(a) == [runtime.ACTION_STOP]
    assert a.owner._explicit_stop_requested
    assert a.owner._last_explicit_stop_reason == "manual_emergency"


def test_missing_target_without_contradiction_holds_only_original_deadline(paired):
    a = paired
    process(a)
    previous = a.owner._short_follow.snapshot()
    frame = SensorFrame(width=640, height=480, capture_frame_id=577, capture_timestamp=NOW-.01)
    assert a.owner._short_follow_adapter.handle(frame, None, is_fresh_depth=False,
        control_source="vision", target_steerable=True, low_quality_visible=False, now=NOW)
    assert a.owner._short_follow.snapshot() is previous
    assert not a.owner._short_follow_adapter.handle(frame, None, is_fresh_depth=False,
        control_source="vision", target_steerable=True, low_quality_visible=False,
        now=previous.plan.expires_at+.001)
    assert not a.owner._short_follow.snapshot().active


def test_real_startup_wait_without_person_then_first_verified_target_not_latched(paired, monkeypatch):
    a = paired
    clock = SimpleNamespace(now=NOW)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock.now)
    a.owner._follow_controller = FollowSafetyController(FollowPolicyConfig(
        target_distance_m=1.4, initial_target_confirm_frames=1))
    a.owner._longitudinal_context_lock = threading.Lock()
    a.owner._longitudinal_context = None
    a.owner._target_quality_debug = lambda *_: "test"
    a.owner._target_region_debug = lambda *_: "test"
    a.owner._log_distance_stop_trigger = lambda *_: None
    visible = [False]
    a.owner._persons_to_targets = lambda *_a, **_kw: [a.target] if visible[0] else []
    a.owner._distance_runtime.select_target = lambda targets: targets[0] if targets else None
    a.owner._validated_visual_observation = None
    for _ in range(2):
        assert a.owner._process_detections_modular(640, 480, []) == [runtime.ACTION_STOP]
        assert a.owner._follow_controller.active_target_id is None
        assert not a.owner._short_follow.snapshot().active
        clock.now += .05
        a.owner._active_capture_timestamp = clock.now-.01
        a.owner._active_capture_frame_id += 1
    visible[0] = True
    a.target = replace(a.target, initial_identity_confirmed=True)
    assert process(a) == []
    assert a.owner._follow_controller.active_target_id == 1
    assert a.owner._short_follow.snapshot().active
    assert not a.owner._explicit_stop_requested


@pytest.mark.parametrize("reason", ["search_cooldown_stop_rotate", "search_candidate_observe"])
def test_legacy_search_control_stop_provenance_does_not_latch_manual_emergency(paired, reason):
    a = paired
    a.owner._explicit_stop_requested = True
    a.owner._last_explicit_stop_reason = reason
    a.owner._explicit_stop_provenance = ("controller_state", reason)
    assert process(a) == []
    assert a.owner._short_follow.snapshot().plan.moving
    assert not a.owner._explicit_stop_requested


def test_external_stop_cannot_inherit_old_controller_state_provenance(paired):
    a = paired
    process(a)
    a.owner._explicit_stop_requested = True
    a.owner._explicit_stop_provenance = ("controller_state", "search_cooldown_stop_rotate")
    a.owner._last_explicit_stop_reason = "manual_emergency"
    assert process(a) == [runtime.ACTION_STOP]
    assert a.owner._short_follow.snapshot().plan is None
    assert a.owner._explicit_stop_requested


def test_sensor_stop_requires_successful_stop_and_post_stop_new_safe_image(paired):
    a = paired
    process(a)
    a.owner._explicit_stop_requested = True
    a.owner._last_explicit_stop_reason = "front_ir"
    a.owner._explicit_stop_provenance = ("sensor_safety", "front_ir")
    adapter = a.owner._short_follow_adapter
    assert adapter.hold_explicit_stop(NOW-.01, NOW)
    epoch = a.owner._short_follow.snapshot().epoch
    assert adapter.hold_explicit_stop(NOW+.1, NOW+.1)  # no completed STOP
    a.owner._short_follow_completed_stop_epoch = epoch
    a.owner._short_follow_completed_stop_at = NOW+.12
    assert adapter.hold_explicit_stop(NOW+.1, NOW+.2)  # captured before STOP
    a.owner._get_obstacle_status = lambda: {"front": True}
    assert adapter.hold_explicit_stop(NOW+.15, NOW+.2)  # danger still present
    a.owner._get_obstacle_status = lambda: {}
    assert not adapter.hold_explicit_stop(NOW+.15, NOW+.2)
    assert not a.owner._explicit_stop_requested
    assert a.owner._short_follow.snapshot().plan is None  # release is not a new plan


def test_verified_takeover_retires_old_stale_recovery_without_requiring_legacy_decide(paired):
    a = paired
    a.owner._follow_controller._stale_direction_recovery_active = True
    calls = []
    def reset(reason):
        calls.append(reason)
        a.owner._follow_controller._stale_direction_recovery_active = False
    a.owner._follow_controller._reset_stale_direction_recovery = reset
    process(a)
    assert calls == ["short_follow_verified_target"]
    assert not a.owner._follow_controller._stale_direction_recovery_active


@pytest.mark.parametrize("released", [False, True])
@pytest.mark.parametrize("distance", [2., 1.45])
def test_search_brake_hold_can_only_exit_existing_settled_contract_not_legacy_pi(paired, released, distance):
    a = paired
    state = a.owner._distance_runtime.get_frame_distance_state()
    a.owner._distance_runtime.get_frame_distance_state = lambda *_a, **_kw: replace(
        state, used_distance_m=distance, raw_distance_m=distance)
    a.owner._brake_hold_active = True
    a.owner._brake_hold_label = "search_reacquire_brake"
    a.target = replace(a.target, depth_observation=DepthTargetObservation(
        a.target.bbox, 1, 1, 576, NOW-.09))
    a.owner._persons_to_targets = lambda *_a, **_kw: [a.target]
    seen = []
    def release(**kwargs):
        seen.append(kwargs)
        if released:
            a.owner._brake_hold_active = False
        return released
    a.owner._action_runtime.release_settled_search_brake_for_depth = release
    assert process(a) == []
    assert len(seen) == 1 and seen[0]["sample_timestamp"] == NOW-.04
    state = a.owner._short_follow.snapshot()
    assert state.active
    if released:
        assert state.plan.moving
        if distance > 1.5:
            assert state.plan.forwarding and 40 < state.plan.left_rpm <= state.plan.speed_cap_rpm
        else:
            assert state.plan.pivot and not state.plan.forwarding
            assert not a.owner.is_forwarding
            assert a.owner._current_forward_percent == 0
            assert a.owner._current_steer_correction_rpm != 0
    else:
        assert state.plan is None and a.owner._brake_hold_active
    assert not a.owner._queued_calls


@pytest.mark.parametrize("distance,raw_distance,center", [
    (1.45, 1.45, .5), (1.10, 1.10, .8), (1.05, 1.05, .8),
    (1.65, 1.09, .8), (1.45, float("nan"), .8),
])
def test_near_search_hold_cannot_release_without_safe_pivot_geometry(
        paired, distance, raw_distance, center):
    a = paired
    a.owner._brake_hold_active = True
    a.owner._brake_hold_label = "search_reacquire_brake"
    bbox = (640*center-80, 0, 640*center+80, 480)
    a.target = replace(a.target, bbox=bbox, depth_observation=DepthTargetObservation(
        bbox, 1, 1, 576, NOW-.09))
    a.owner._persons_to_targets = lambda *_a, **_kw: [a.target]
    state = a.owner._distance_runtime.get_frame_distance_state()
    a.owner._distance_runtime.get_frame_distance_state = lambda *_a, **_kw: replace(
        state, used_distance_m=distance, raw_distance_m=raw_distance)
    a.owner._action_runtime.release_settled_search_brake_for_depth = (
        lambda **_: pytest.fail("unsafe/centered near hold release"))
    assert process(a) == []
    assert a.owner._brake_hold_active
    assert a.owner._short_follow.snapshot().plan is None


def test_unrelated_safety_hold_is_not_cleared_or_given_to_legacy_pi(paired):
    a = paired
    a.owner._brake_hold_active = True
    a.owner._brake_hold_label = "safety_hold_hazard"
    a.owner._action_runtime.release_settled_search_brake_for_depth = lambda **_: pytest.fail("unrelated hold release")
    assert process(a) == []
    assert a.owner._short_follow.snapshot().active
    assert a.owner._short_follow.snapshot().plan is None
    assert a.owner._brake_hold_active


def test_takeover_preserves_protected_stop_but_retires_old_normal_motion(paired):
    a = paired
    a.owner.action_queue = queue.Queue()
    stop = ActionCommandSnapshot(runtime.ACTION_STOP, 1, NOW-.01, 1, 575, NOW-.1,
        "manual_emergency", False, True)
    a.owner.action_queue.put(stop)
    a.owner.action_queue.put(runtime.ACTION_FORWARD)
    assert process(a) == []
    assert list(a.owner.action_queue.queue) == [stop]


def test_queue_provenance_does_not_mark_search_state_as_protected_emergency():
    from test_search_handoff_braking import queue_owner
    a = queue_owner()
    a._explicit_stop_requested = True
    a._explicit_stop_provenance = ("controller_state", "search_cooldown_stop_rotate")
    a._replace_action_queue([runtime.ACTION_STOP], "search_cooldown_stop_rotate")
    command = a.action_queue.get_nowait()
    assert command.stop_origin == "controller_state"
    assert not command.protected_stop
    a._last_explicit_stop_reason = "manual_emergency"
    a._replace_action_queue([runtime.ACTION_STOP], "manual_emergency")
    command = a.action_queue.get_nowait()
    assert command.protected_stop and command.stop_origin == "unknown"


@pytest.mark.parametrize("distance,center", [(2., .5), (1.45, .75)])
def test_real_pending_search_stop_settles_and_new_paired_measurement_releases(monkeypatch, distance, center):
    from test_short_follow_acceptance import _writer_chain, _adapter_frame, _deliver
    a = _writer_chain(monkeypatch)
    owner, executor = a.owner, a.runtime
    owner._action_runtime = executor
    owner.command_lock = threading.Lock()
    owner.action_queue_lock = threading.Lock()
    owner.action_queue = queue.Queue()
    owner.search_state = "none"
    owner._near_yaw_park_request = None
    # Search has requested a physical stop but its writer has not consumed it.
    assert executor.request_search_reacquire_brake(1, a.now-.04, "candidate_center")
    frame, target = _adapter_frame(a, capture_id=2)
    assert _deliver(a, frame, target)
    assert executor._service_short_follow()
    request = executor._search_reacquire_brake_applied
    assert request is not None and executor._search_reacquire_settling is not None
    assert owner._brake_hold_active and a.driver.stops
    assert all(pair == (0, 0) for pair in a.driver.pairs)  # NORMAL parking preparation only
    # A real post-STOP cache supplies distinct quiet samples. The paired
    # watchdog must preserve/progress this inherited contract, not clear it.
    for tick in range(1, 15):
        a.now = 100.+tick*.05
        a.feedback = SteeringFeedback(timestamp=a.now, trustworthy=True,
            left_forward_rpm=0., right_forward_rpm=0., raw_yaw_rate_right_dps=0.)
        assert executor._service_short_follow()
        assert executor._search_reacquire_brake_applied is request
    a.now = 100.75
    a.feedback = replace(a.feedback, timestamp=a.now)
    assert executor._search_reacquire_brake_request is request
    assert owner._brake_hold_active  # quiet alone has not authorized motion
    frame, target = _adapter_frame(a, capture_id=3, distance=distance, center=center)
    target.depth_observation = DepthTargetObservation((240, 0, 400, 480), 1, 1,
        frame.capture_frame_id, frame.capture_timestamp)
    frame.persons = [target]
    assert _deliver(a, frame, target)
    assert not owner._brake_hold_active
    assert executor._search_reacquire_brake_applied is None
    assert a.controller.snapshot().plan.moving
    assert executor._service_short_follow()
    plan = a.owner._short_follow.snapshot().plan
    assert a.driver.pairs[-1] == (plan.left_rpm, -plan.right_rpm)
    if distance > 1.5:
        assert plan.base_rpm > 40 and owner.is_forwarding
    else:
        assert plan.pivot and plan.base_rpm == 0 and not owner.is_forwarding


@pytest.mark.parametrize("already_moving", [False, True])
def test_real_queued_protected_stop_is_executed_without_an_explicit_flag(monkeypatch, already_moving):
    from test_short_follow_acceptance import _writer_chain, _adapter_frame, _deliver
    a = _writer_chain(monkeypatch)
    a.owner.action_queue = queue.Queue()
    a.owner.action_queue_lock = threading.Lock()
    if already_moving:
        frame, target = _adapter_frame(a)
        assert _deliver(a, frame, target)
        assert a.runtime._service_short_follow()
        assert a.driver.pairs
        a.now += .05
    old_writes = len(a.driver.pairs)
    stop = ActionCommandSnapshot(a.symbols.stop, 2, a.now, 2, 2, a.now-.01,
        "manual_emergency", False, True, stop_origin="unknown")
    a.owner.action_queue.put(stop)
    assert not getattr(a.owner, "_explicit_stop_requested", False)
    frame, target = _adapter_frame(a, capture_id=2)
    assert _deliver(a, frame, target)
    assert a.owner.action_queue.queue[0] is stop  # producer did not discard it
    assert a.runtime._service_short_follow()
    assert len(a.driver.pairs) == old_writes
    assert a.driver.stops
    assert a.controller.snapshot().plan is None
    assert a.owner._explicit_stop_requested
    assert a.owner._last_explicit_stop_reason == "manual_emergency"


@pytest.mark.parametrize("hold_label", ["front_ir", "safety_hold_front_ir"])
def test_startup_sensor_stop_ack_and_settled_hold_on_original_owner_can_recover(paired, monkeypatch, hold_label):
    from test_short_follow_acceptance import make_runtime
    from car_control_modular.action_runtime import MotionActionRuntime
    a = paired
    clock = SimpleNamespace(now=NOW)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock.now)
    a.owner._follow_controller = FollowSafetyController(FollowPolicyConfig(target_distance_m=1.4))
    a.owner._longitudinal_context_lock = threading.Lock()
    a.owner._longitudinal_context = None
    a.owner._target_quality_debug = lambda *_: "test"
    a.owner._target_region_debug = lambda *_: "test"
    a.owner._log_distance_stop_trigger = lambda *_: None
    a.owner._get_obstacle_status = lambda: {"front": True}
    a.owner._validated_visual_observation = None
    assert process(a) == [runtime.ACTION_STOP]
    assert a.owner._explicit_stop_provenance == ("sensor_safety", "front_ir")
    assert not a.owner._short_follow.snapshot().active
    template, _, driver, symbols = make_runtime()
    motor = MotionActionRuntime(a.owner, template.backend, template.config, symbols,
        hard_stop_check=lambda *_: bool(a.owner._get_obstacle_status().get("front")),
        logger=runtime.logger)
    a.owner._action_runtime = motor
    a.owner.motor_io_lock = motor.backend.io_lock
    motor.get_steering_feedback = lambda: SteeringFeedback(timestamp=clock.now,
        trustworthy=True, left_forward_rpm=0., right_forward_rpm=0.)
    # A real inherited legacy sensor hold exists on THE SAME owner, not on an
    # unrelated test backend. Merely seeing a safe frame must not bypass it.
    a.owner._brake_hold_active = True
    a.owner._brake_hold_label = hold_label
    a.owner._brake_hold_stop_mode = "emergency"
    assert process(a) == [runtime.ACTION_STOP]
    motor.send_stop_with_brake_hold("front_ir")
    assert driver.stops and not driver.pairs
    assert a.owner._short_follow_completed_stop_epoch == a.owner._short_follow.snapshot().epoch
    clock.now += .10
    a.owner._active_capture_frame_id += 1
    a.owner._active_capture_timestamp = clock.now-.01
    a.owner._get_obstacle_status = lambda: {}
    assert process(a) == [runtime.ACTION_STOP]  # first quiet post-STOP sample
    assert a.owner._brake_hold_active
    clock.now += .05
    a.owner._active_capture_frame_id += 1
    a.owner._active_capture_timestamp = clock.now-.01
    assert process(a) == []
    assert a.owner._follow_controller.active_target_id == 1
    assert a.owner._short_follow.snapshot().active
    assert not a.owner._explicit_stop_requested
    assert not a.owner._brake_hold_active
    assert not driver.pairs  # Clearing a hold alone does not command motion.
    clock.now += .05
    a.owner._active_capture_frame_id += 1
    a.owner._active_capture_timestamp = clock.now-.01
    a.owner._validated_visual_observation = ValidatedVisualObservation(1, 1,
        a.owner._active_capture_frame_id, clock.now-.01, clock.now, clock.now+.4, "full")
    a.owner._distance_runtime.get_frame_distance_state = lambda *_a, **_kw: DistanceState(
        source="vision_depth", source_detail="depth_multi_region", raw_distance_m=2.,
        used_distance_m=2., sample_timestamp=clock.now-.005)
    assert process(a) == []
    assert a.owner._short_follow.snapshot().plan.moving
    assert motor._service_short_follow()
    assert driver.pairs[-1][0] > 0 > driver.pairs[-1][1]


@pytest.mark.parametrize("block", ["unrelated_hold", "moving", "stale", "motor_fault", "manual"])
def test_sensor_hold_cannot_bypass_provenance_quiet_feedback_or_backend_fault(monkeypatch, block):
    from test_short_follow_acceptance import _writer_chain
    a = _writer_chain(monkeypatch)
    owner = a.owner
    owner._action_runtime = a.runtime
    owner._current_hazard_state_for_controller = lambda: SimpleNamespace(active=False)
    owner._get_obstacle_status = lambda: {}
    owner._explicit_stop_requested = True
    owner._last_explicit_stop_reason = "front_ir"
    owner._explicit_stop_provenance = ("sensor_safety", "front_ir")
    owner._brake_hold_active = True
    owner._brake_hold_label = "safety_hold_front_ir"
    owner._brake_hold_stop_mode = "emergency"
    assert a.adapter.hold_explicit_stop(a.now-.01, a.now)
    assert a.runtime._service_short_follow()
    assert a.driver.stops
    if block == "unrelated_hold": owner._brake_hold_label = "safety_hold_unrelated"
    if block == "manual": owner._explicit_stop_provenance = ("unknown", "front_ir")
    if block == "motor_fault": a.runtime.backend.parking_release_fault = "test"
    for _ in range(3):
        a.now += .05
        a.feedback = SteeringFeedback(timestamp=a.now-(.2 if block == "stale" else 0),
            trustworthy=True, left_forward_rpm=10. if block == "moving" else 0., right_forward_rpm=0.)
        assert a.adapter.hold_explicit_stop(a.now-.01, a.now)
        assert owner._brake_hold_active and owner._explicit_stop_requested
    assert not a.driver.pairs
