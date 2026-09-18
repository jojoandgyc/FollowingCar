"""Physical Depth authority survives visual zero/old yaw-only intents.

Real owner methods, an in-memory command spy, no camera or motor construction.
"""
from dataclasses import replace
from types import SimpleNamespace
import threading

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import (
    ControlAction, ControlDecision, DistanceState, HazardState, ObstacleState, SensorFrame,
)
from test_lateral_zero_runtime import NOW, owner, _intent, _target, _publish_stop


def _frame(stamp=NOW - .04, distance=2.3, **changes):
    return replace(SensorFrame(
        width=640, height=480, persons=[_target()], distance_m=distance,
        capture_frame_id=576, capture_timestamp=NOW - .09,
        distance_state=DistanceState(
            source="vision_depth", raw_distance_m=distance, used_distance_m=distance,
            source_detail="depth_torso", sample_age_sec=NOW - stamp,
            sample_timestamp=stamp, fusion_confidence=1.0,
        ),
    ), **changes)


def _commit(owner, stamp=NOW-.04, percent=17, kind="forward", fresh=True):
    action = getattr(ControlAction, kind)(percent, "canonical")
    return owner._commit_depth_linear_decision(
        ControlDecision(actions=[action], reason="canonical"), _frame(stamp), 1,
        is_fresh_depth=fresh,
    )


@pytest.mark.parametrize("kind", ["forward", "backward"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_new_depth_replaces_old_yaw_only_with_composed_command(owner, kind, sign):
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: None)
    _intent(owner, initial_correction_rpm=sign*5)
    owner._lateral_intent_last_correction_rpm = sign*5
    _commit(owner, kind=kind)
    assert owner._current_rotate_raw_target == 0
    assert not owner._visible_rotate_command_allowed(runtime.ACTION_ROTATE_RIGHT)
    owner._service_lateral_intent(NOW)
    expected = (runtime.ACTION_BACKWARD if kind == "backward" else
                runtime.ACTION_STEER_RIGHT if sign > 0 else runtime.ACTION_STEER_LEFT)
    assert [call[0] for call in owner._queued_calls] == [(expected,)]
    assert owner._current_forward_percent == 17
    assert owner._current_steer_correction_rpm == (sign*5 if kind == "backward" else 5)
    assert owner._current_forward_allow_below_min
    assert owner._current_rotate_raw_target == 0
    assert not owner._visible_rotate_command_allowed(runtime.ACTION_ROTATE_RIGHT)


@pytest.mark.parametrize("stamp", [NOW-.20, NOW+.01, float("nan")])
def test_invalid_physical_time_never_grants_translation(owner, stamp):
    actions, accepted = _commit(owner, stamp=stamp)
    assert not accepted and actions == []
    assert owner._depth30_linear_snapshot is None


def test_same_or_older_sample_cannot_renew_or_reverse_authority(owner):
    _commit(owner)
    original = owner._depth30_linear_snapshot
    for stamp in [NOW-.04, NOW-.08]:
        actions, accepted = _commit(owner, stamp=stamp, kind="backward")
        assert not accepted and actions == []
        assert owner._depth30_linear_snapshot == original
    assert owner._last_depth30_translation_ts == NOW-.04


@pytest.mark.parametrize("yaw", [-5, 0, 5])
def test_visual_begin_old_aligned_depth_preserves_newer_speed_and_deadline(owner, yaw, caplog):
    """CAP1786/1788 ordering: latest -> RGB begins -> old depth -> yaw tick."""
    owner._longitudinal_context_lock = threading.Lock()
    owner._longitudinal_context = {"old_roi": True}
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: None)
    _intent(owner, initial_correction_rpm=yaw)
    owner._lateral_intent_last_correction_rpm = yaw
    _commit(owner, percent=40)
    before = owner._depth30_linear_snapshot
    revision = owner._lateral_yaw_revision
    owner._clear_longitudinal_context(revoke_translation=False, reason="visual_frame_begin")
    assert owner._longitudinal_context is None
    assert owner._depth30_linear_snapshot == before
    assert owner._lateral_yaw_revision == revision
    assert _commit(owner, stamp=NOW-.08) == ([], False)
    owner._service_lateral_intent(NOW)
    expected = runtime.ACTION_STEER_LEFT if yaw < 0 else runtime.ACTION_STEER_RIGHT
    assert owner._queued_calls[-1][0] == (expected,)
    assert owner._current_forward_percent == 40
    assert owner._current_steer_correction_rpm == abs(yaw)
    assert owner._depth30_linear_snapshot == before
    assert "deadline_renewed=False" in caplog.text


def test_visual_begin_does_not_extend_expired_depth_or_allow_old_resurrection(owner, monkeypatch):
    owner._longitudinal_context_lock = threading.Lock()
    _commit(owner)
    _intent(owner)
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: None)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.145)
    owner._clear_longitudinal_context(revoke_translation=False, reason="visual_frame_begin")
    assert owner._fresh_depth_linear_snapshot(1) is None
    assert _commit(owner, stamp=NOW-.08) == ([], False)
    # A newly arrived yaw intent does not grant a new range deadline either.
    _intent(owner, published_at=NOW+.14, valid_until=NOW+.28)
    owner._service_lateral_intent(NOW+.145)
    assert owner._queued_calls[-1][0] == (runtime.ACTION_ROTATE_RIGHT,)
    assert owner._current_forward_percent == 0


def test_actual_context_rejection_revokes_prepared_translation_not_yaw(owner, caplog):
    owner._longitudinal_context_lock = threading.Lock()
    intent = _intent(owner)
    _commit(owner)
    before = owner._lateral_yaw_revision
    owner._clear_longitudinal_context(reason="visual_target_missing_or_ambiguous")
    assert owner._depth30_linear_snapshot is None
    assert owner._current_forward_percent == owner._current_steer_base_percent == 0
    assert owner._lateral_yaw_revision == before+1
    assert owner._lateral_intent_store.snapshot() is intent
    assert not owner._current_forward_allow_below_min
    assert _commit(owner, stamp=NOW-.08) == ([], False)
    assert "depth_linear_revoked" in caplog.text
    assert "visual_target_missing_or_ambiguous" in caplog.text


@pytest.mark.parametrize("reject", [False, True])
def test_visual_boundary_while_drive_waits_for_motor_lock(owner, reject):
    from test_yaw_zero_regression import make_runtime
    motor, _, backend, _ = make_runtime()
    motor.owner = owner
    owner._longitudinal_context_lock = threading.Lock()
    _commit(owner)
    revision = owner._lateral_yaw_revision
    prepared, release = threading.Event(), threading.Event()
    class IoGate:
        def __enter__(self):
            prepared.set()
            assert release.wait(2)
        def __exit__(self, *_args):
            pass
    owner.motor_io_lock = IoGate()
    results, errors = [], []
    def send():
        try:
            results.append(motor.send_percent_drive(17, allow_below_min=True, yaw_revision=revision))
        except BaseException as exc:
            errors.append(exc)
    worker = threading.Thread(target=send, daemon=True)
    worker.start()
    try:
        assert prepared.wait(1)
        owner._clear_longitudinal_context(revoke_translation=False, reason="visual_frame_begin")
        assert _commit(owner, stamp=NOW-.08) == ([], False)
        if reject:
            owner._clear_longitudinal_context(reason="visual_target_missing_or_ambiguous")
    finally:
        release.set()
        worker.join(1)
    assert not worker.is_alive() and not errors
    assert results == [not reject]
    assert bool(backend.targets) is not reject


def test_zero_advances_watermark_and_prevents_older_positive_revival(owner):
    _commit(owner, stamp=NOW-.09)
    actions, accepted = _commit(owner, stamp=NOW-.02, percent=0)
    assert accepted and actions[0].speed_percent == 0
    assert owner._depth30_linear_snapshot is None
    assert not owner._current_forward_allow_below_min
    assert _commit(owner, stamp=NOW-.05)[1] is False
    assert owner._depth30_linear_snapshot is None


def test_zero_commit_immediately_revokes_prepared_drive_but_preserves_yaw(owner):
    from test_yaw_zero_regression import make_runtime

    motor, _unused_owner, backend, _symbols = make_runtime()
    motor.owner = owner
    intent = _intent(owner)
    _commit(owner, stamp=NOW-.09)
    owner._current_forward_percent = owner._current_steer_base_percent = 17
    owner._forward_speed_latched_percent = 17
    prepared_revision = owner._lateral_yaw_revision
    prepared = threading.Event()
    io_lock = threading.Lock()

    class ContendedIoLock:
        def __enter__(self):
            prepared.set()
            io_lock.acquire()

        def __exit__(self, *_args):
            io_lock.release()

    owner.motor_io_lock = ContendedIoLock()
    results, errors = [], []

    def prepared_drive():
        try:
            results.append(motor.send_percent_drive(
                17, allow_below_min=True, yaw_revision=prepared_revision,
            ))
        except BaseException as exc:
            errors.append(exc)

    io_lock.acquire()
    worker = threading.Thread(target=prepared_drive, daemon=True)
    try:
        worker.start()
        assert prepared.wait(1.0), "prepared DRIVE did not reach the motor lock"
        actions, committed = _commit(owner, stamp=NOW-.02, percent=0)
        assert committed and actions[0].speed_percent == 0
        assert owner._lateral_yaw_revision == prepared_revision + 1
        assert owner._current_forward_percent == owner._current_steer_base_percent == 0
        assert owner._forward_speed_latched_percent is None
        assert not owner._current_forward_allow_below_min
        assert not owner.is_forwarding
        assert owner._lateral_intent_store.snapshot() is intent
        assert owner._current_steer_correction_rpm == owner._last_vision_correction_rpm == 5
        assert owner._queued_calls == []  # revoked before any next publication
        revision_after_zero = owner._lateral_yaw_revision
        assert _commit(owner, stamp=NOW-.02, percent=0) == ([], False)
        assert owner._lateral_yaw_revision == revision_after_zero
    finally:
        io_lock.release()
        worker.join(1.0)
    assert not worker.is_alive()
    assert not errors
    assert results == [False]
    assert backend.targets == []


def test_hold_can_reduce_only_and_does_not_renew_deadline(owner, monkeypatch):
    _commit(owner, percent=19)
    actions, accepted = _commit(owner, percent=40, fresh=False)
    assert accepted and actions[0].speed_percent == 19
    assert owner._depth30_linear_snapshot == ("forward", 19, 1, NOW-.04)
    _commit(owner, percent=7, fresh=False)
    assert owner._depth30_linear_snapshot == ("forward", 7, 1, NOW-.04)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.15)
    actions, _ = _commit(owner, percent=7, fresh=False)
    assert actions[0].speed_percent == 0
    assert owner._depth30_linear_snapshot is None


@pytest.mark.parametrize("change", ["uid", "search", "weak", "brake", "stopped"])
def test_snapshot_never_authorizes_another_target_or_unsafe_state(owner, change):
    _commit(owner)
    if change == "uid":
        owner._follow_controller.active_target_id = 2
    elif change == "search":
        owner.search_state = "searching"
    elif change == "weak":
        owner._vision_control_state = "target_visible_low_quality"
    elif change == "brake":
        owner._brake_hold_active = True
    else:
        owner._explicit_stop_requested = True
    assert owner._fresh_depth_linear_snapshot(1) is None


def test_visual_fresh_range_is_committed_before_center_zero(owner):
    calls = []
    owner._follow_controller.last_action_frame = 13
    def canonical(*args, **kwargs):
        calls.append((args, kwargs))
        owner._follow_controller.last_action_frame = 99
        return ControlDecision(actions=[ControlAction.forward(17, "canonical")])
    owner._follow_controller._longitudinal_only_decision = canonical
    owner._depth30_linear_snapshot = ("forward", 35, 1, NOW-.21)
    assert owner._refresh_visual_depth_linear_authority(
        _frame(), _target(), ControlDecision(), is_fresh_depth=True,
        target_steerable=True, low_quality_visible=False,
    )
    assert owner._follow_controller.last_action_frame == 13
    assert _publish_stop(owner)
    assert len(calls) == 1
    assert owner._queued_calls[-1][0] == (runtime.ACTION_STEER_RIGHT,)
    assert owner._current_forward_percent == 17
    assert owner._current_steer_correction_rpm == 0
    assert owner._depth30_linear_snapshot[3] == NOW-.04


@pytest.mark.parametrize("case", ["same", "older", "held", "hazard", "ir", "weak", "stop"])
def test_visual_canonical_not_called_for_unqualified_observation(owner, case):
    _commit(owner)
    def forbidden(*args, **kwargs):
        pytest.fail("unqualified sample ran the canonical longitudinal PID")
    owner._follow_controller._longitudinal_only_decision = forbidden
    frame = _frame(NOW-.02)
    if case in {"same", "older"}:
        frame = _frame(NOW-.04 if case == "same" else NOW-.08)
    elif case == "hazard":
        frame = replace(frame, hazard=HazardState(active=True))
    elif case == "ir":
        frame = replace(frame, obstacles=ObstacleState(left=True))
    assert not owner._refresh_visual_depth_linear_authority(
        frame, _target(), ControlDecision(explicit_stop_requested=case == "stop"),
        is_fresh_depth=case != "held", target_steerable=case != "weak",
        low_quality_visible=case == "weak",
    )


def test_tracking_base_cap_is_bounded_and_only_explicitly_requested(owner, monkeypatch):
    monkeypatch.setattr(runtime, "FORWARD_MAX_RPM", 100)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_MAX_FORWARD_PERCENT", 20)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT", 60)
    actions = [ControlAction.forward(80, "pid")]
    assert owner._cap_depth_longitudinal_actions(actions, 1.5)[0].speed_percent == 20
    assert owner._cap_depth_longitudinal_actions(actions, 1.5, 30)[0].speed_percent == 30
    assert owner._cap_depth_longitudinal_actions(actions, 1.5, 99)[0].speed_percent == 40
    assert owner._cap_depth_longitudinal_actions(actions, 1.5, float("nan"))[0].speed_percent == 20
    owner._follow_controller._tracking_base_rpm = lambda distance, now: 30
    owner._commit_depth_linear_decision(
        ControlDecision(actions=actions), _frame(distance=1.5), 1, is_fresh_depth=True,
    )
    assert owner._depth30_linear_snapshot[1] == 30
    held, _ = owner._commit_depth_linear_decision(
        ControlDecision(actions=actions), _frame(distance=1.5), 1, is_fresh_depth=False,
    )
    assert held[0].speed_percent == 20


def _process_fixture(owner, decision, canonical):
    target = _target()
    frame = _frame()
    owner._active_capture_timestamp = NOW-.09
    owner._last_frame_distance_state = None
    owner._last_control_decision_log_key = None
    owner._get_obstacle_status = lambda: {}
    owner._persons_to_targets = lambda *args, **kwargs: [target]
    owner._current_hazard_state_for_controller = lambda: HazardState()
    owner._maybe_schedule_historical_direction_backfill = lambda *args: None
    owner._clear_action_queue = lambda *args: None
    owner._distance_runtime = SimpleNamespace(
        select_target=lambda targets: targets[0],
        get_frame_distance_state=lambda *args, **kwargs: frame.distance_state,
    )
    owner._action_runtime = SimpleNamespace(
        get_steering_feedback=lambda: None,
        cancel_rotate_pulse_observation=lambda *args: None,
    )
    ctl = owner._follow_controller
    ctl.set_last_dispatched = lambda value: None
    ctl._is_fresh_depth_state = lambda frame: True
    ctl.decide = lambda *args, **kwargs: decision
    ctl._longitudinal_only_decision = canonical
    ctl.last_selected_target = target
    ctl.search_state = "none"
    ctl.search_direction = None
    ctl.lost_confirm_frames = 0
    ctl.last_action_frame = 0
    ctl.last_person_center_x = target.center[0]
    owner.search_direction = None
    return target


def test_real_visual_process_center_uses_new_depth_in_one_publish(owner):
    canonical_calls = []
    def canonical(*args, **kwargs):
        canonical_calls.append(args)
        return ControlDecision(actions=[ControlAction.forward(17, "pid")])
    target = _process_fixture(
        owner, ControlDecision(actions=[ControlAction.stop("center", brake_hold=False)],
                               soft_stop_requested=True), canonical,
    )
    owner._follow_controller.last_steering_pid_result = None
    owner._depth30_linear_snapshot = ("forward", 35, 1, NOW-.21)
    actions = owner._process_detections_modular(
        640, 480, [(target.bbox, target.track_id, target.confidence, target.area)],
    )
    assert actions == []  # the unified intent publisher owns the one final command
    assert len(canonical_calls) == 1
    assert owner._queued_calls == [((runtime.ACTION_STEER_RIGHT,), "lateral_zero:center")]
    assert owner._current_forward_percent == 17
    assert owner._current_steer_correction_rpm == 0
    assert owner._current_forward_allow_below_min
    assert owner._last_depth30_translation_ts == NOW-.04


def test_real_visual_process_new_zero_depth_with_nonzero_yaw_only_turns(owner):
    target = _process_fixture(
        owner, ControlDecision(actions=[ControlAction.rotate_right("yaw")]),
        lambda *args, **kwargs: ControlDecision(actions=[ControlAction.forward(0, "range_zero")]),
    )
    owner._depth30_linear_snapshot = ("forward", 35, 1, NOW-.09)
    owner._process_detections_modular(
        640, 480, [(target.bbox, target.track_id, target.confidence, target.area)],
    )
    assert len(owner._queued_calls) == 1
    assert owner._queued_calls[0][0] == (runtime.ACTION_ROTATE_RIGHT,)
    assert owner._current_forward_percent == 0
    assert owner._depth30_linear_snapshot is None
    assert not owner._current_forward_allow_below_min


def test_real_depth_process_merges_yaw_and_never_replays_duplicate_sample(owner):
    decision = ControlDecision(actions=[ControlAction.forward(17, "pid")])
    target = _process_fixture(owner, decision, lambda *args, **kwargs: decision)
    owner._follow_controller.last_steering_pid_result.correction_limit_reason = "test"
    _intent(owner)
    persons = [(target.bbox, target.track_id, target.confidence, target.area)]
    actions = owner._process_detections_modular(640, 480, persons, control_source="depth30")
    assert actions == [runtime.ACTION_STEER_RIGHT]
    assert owner._current_forward_percent == owner._current_steer_base_percent == 17
    assert owner._current_forward_allow_below_min
    assert owner._current_rotate_raw_target == 0
    assert not owner._visible_rotate_command_allowed(runtime.ACTION_ROTATE_RIGHT)
    before = owner._depth30_linear_snapshot
    assert owner._process_detections_modular(640, 480, persons, control_source="depth30") == []
    assert owner._depth30_linear_snapshot == before
    assert owner._last_depth30_translation_ts == NOW-.04
