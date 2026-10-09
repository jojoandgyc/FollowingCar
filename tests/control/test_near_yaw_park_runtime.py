"""Real producer methods with fake state/queue; no hardware construction."""
from types import SimpleNamespace
import queue
import threading

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction, ControlDecision, SensorFrame, DistanceState
from car_control_modular.near_yaw_parking import ParkSettlingEvidence
from test_lateral_zero_runtime import owner, _intent, _target, NOW


def park(owner, **changes):
    intent = _intent(owner, park_requested=True, hold_zero=True, **changes)
    owner._publish_lateral_zero(intent, "near_distance_center_settle")
    return intent


def settled_park(owner, monkeypatch):
    """A successful actual write, two physical quiet samples, then a new image."""
    evidence = ParkSettlingEvidence(owner._near_yaw_park_request, NOW+.01)
    feedback = lambda t: SimpleNamespace(timestamp=t, trustworthy=True,
        left_forward_rpm=0., right_forward_rpm=0.)
    evidence.observe(feedback(NOW+.45), NOW+.45)
    evidence.observe(feedback(NOW+.51), NOW+.51)
    owner._action_runtime = SimpleNamespace(
        near_yaw_park_release_ready=lambda req, stamp, now:
            req is evidence.request and evidence.release_ready(stamp, feedback(NOW+.51), now),
        near_yaw_park_motion_ready=lambda req, stamp, now:
            req is evidence.request and evidence.motion_handoff_ready(stamp, feedback(NOW+.51), now))
    # These release tests use a minimal fake runtime; the new same-frame yaw
    # service is covered separately rather than invoking an incomplete motor.
    owner._service_lateral_intent = lambda now: None
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.55)
    return evidence


def publish_motion(owner, *, capture=577, stamp=None, qualified=True):
    if stamp is None:
        stamp = runtime.time.monotonic()-.01
    owner._last_command_capture_frame = capture
    owner._last_command_capture_timestamp = stamp
    return owner._publish_lateral_intent_from_decision(
        width=640, target=_target(), runtime_actions=[ControlAction.rotate_right(5)],
        control_source="vision", target_steerable=qualified,
        low_quality_visible=not qualified,
    )


def test_explicit_center_park_is_persistent_intent_not_soft_zero(owner):
    park(owner)
    assert owner._near_yaw_park_request.uid == 1
    assert not owner._use_soft_stop_next
    assert owner._queued_calls[-1][0] == (runtime.ACTION_STOP,)
    assert owner._current_rotate_raw_target == owner._current_forward_percent == 0


def test_new_center_frame_advances_release_barrier_not_stop_episode(owner):
    park(owner)
    original = owner._near_yaw_park_request
    park(owner, capture_frame_id=580, capture_timestamp=NOW-.02)
    assert owner._near_yaw_park_request is original
    assert owner._near_yaw_park_evidence == (580, NOW-.02)
    assert not publish_motion(owner, capture=579, stamp=NOW-.03)
    assert owner._near_yaw_park_request is original


@pytest.mark.parametrize("capture,stamp", [
    (576, NOW-.02), (577, NOW-.09), (575, NOW-.01),
    (577, NOW-.3), (577, NOW+.01), (577, float("nan")),
])
def test_old_republished_out_of_order_stale_future_frames_cannot_release(owner, capture, stamp):
    park(owner)
    assert not publish_motion(owner, capture=capture, stamp=stamp)
    assert owner._near_yaw_park_request is not None


def test_new_reliable_correction_releases_only_typed_hold(owner, monkeypatch):
    park(owner)
    settled_park(owner, monkeypatch)
    owner._brake_hold_active = True
    owner._brake_hold_label = "near_yaw_park"
    owner._brake_hold_stop_mode = "normal"
    assert publish_motion(owner)
    assert owner._near_yaw_park_request is None
    assert not owner._brake_hold_active
    assert owner._brake_hold_stop_mode is None
    assert owner._depth30_linear_snapshot is None  # no old forward lease restored


def test_new_confirmed_near_yaw_is_serviced_without_waiting_for_next_tick(owner):
    served = []
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: None)
    owner._service_lateral_intent = lambda now: served.append(now)
    assert publish_motion(owner, capture=577, stamp=NOW-.03)
    assert len(served) == 1
    assert owner._lateral_intent_store.snapshot().capture_frame_id == 577
    assert owner._lateral_intent_last_publish_ts == 0.0


def test_recenter_release_passes_same_episode_quiet_boundary_without_authority(owner, monkeypatch):
    park(owner)
    request = owner._near_yaw_park_request
    evidence = ParkSettlingEvidence(request, NOW+.01, require_current_release=True)
    evidence.mark_current_released(NOW+.40)
    def feedback(stamp):
        return SimpleNamespace(timestamp=stamp, trustworthy=True,
            left_forward_rpm=0., right_forward_rpm=0.,
            integrated_yaw_right_deg=31.28)
    evidence.observe(feedback(NOW+.46), NOW+.46)
    evidence.observe(feedback(NOW+.52), NOW+.52)
    assert evidence.ready_at == NOW+.52
    assert evidence.ready_yaw_right_deg == 31.28
    noted = []
    owner._follow_controller.note_post_park_settled = lambda *args: noted.append(args)
    owner._action_runtime = SimpleNamespace(
        _near_yaw_park_settling=evidence,
        near_yaw_park_release_ready=lambda req, stamp, now:
            req is request and evidence.release_ready(stamp, feedback(NOW+.52), now))
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.56)
    assert owner._release_near_yaw_park(
        capture_id=577, capture_timestamp=NOW+.54,
        reason="new_visible_motion", target_id=1, qualified=True,
        x_ratio=.25)
    assert noted == [(1, NOW+.52, 31.28)]
    assert owner._depth30_linear_snapshot is None


def test_unqualified_correction_cannot_release(owner):
    park(owner)
    assert not publish_motion(owner, qualified=False)
    assert owner._near_yaw_park_request is not None


def test_tick_old_sequence_cannot_restart_parked_yaw(owner):
    park(owner)
    owner._queued_calls.clear()
    owner._service_lateral_intent(NOW+.45)
    assert not owner._queued_calls
    assert owner._near_yaw_park_request is not None


def test_fresh_depth_forward_plus_zero_yaw_does_not_park(owner):
    owner._depth30_linear_snapshot = ("forward", 7, 1, NOW-.05)
    park(owner)
    assert getattr(owner, "_near_yaw_park_request", None) is None
    assert owner._current_forward_percent == 7
    assert owner._queued_calls[-1][0] == (runtime.ACTION_STEER_RIGHT,)


def test_ordinary_zero_without_predictive_or_center_reason_remains_soft(owner):
    owner._publish_lateral_zero(_intent(owner), "pid_zero:near_distance_rotation_only")
    assert getattr(owner, "_near_yaw_park_request", None) is None
    assert owner._use_soft_stop_next


def test_pending_park_survives_duplicate_depth_zero(owner):
    park(owner)
    request = owner._near_yaw_park_request
    owner._publish_lateral_zero(_intent(owner), "revoke:duplicate_depth")
    assert owner._near_yaw_park_request is request
    assert not owner._use_soft_stop_next


def test_new_search_handoff_releases_without_restoring_authority(owner, monkeypatch):
    park(owner)
    settled_park(owner, monkeypatch)
    assert owner._release_near_yaw_park(
        capture_id=577, capture_timestamp=NOW+.54,
        reason="visual_state_handoff:searching", handoff=True,
    )
    assert owner._near_yaw_park_request is None
    assert owner._depth30_linear_snapshot is None


def test_search_handoff_same_frame_cannot_release(owner):
    park(owner)
    assert not owner._release_near_yaw_park(
        capture_id=576, capture_timestamp=NOW-.09,
        reason="visual_state_handoff:searching", handoff=True,
    )


def test_new_motion_does_not_clear_intervening_safety_hold(owner):
    park(owner)
    owner._brake_hold_active = True
    owner._brake_hold_label = "safety_hold_front_ir"
    owner._brake_hold_stop_mode = "emergency"
    publish_motion(owner)
    assert owner._brake_hold_active
    assert owner._brake_hold_label == "safety_hold_front_ir"
    assert owner._brake_hold_stop_mode == "emergency"


def test_identity_mismatch_cannot_use_visible_release(owner):
    park(owner)
    assert not owner._release_near_yaw_park(
        capture_id=577, capture_timestamp=NOW-.02, target_id=2,
        qualified=True, reason="new_visible_motion",
    )


def test_publisher_carries_controller_parking_intent(owner):
    owner._publish_lateral_intent_from_decision(
        width=640, target=_target(), runtime_actions=[ControlAction.stop("near_distance_rotation_only")],
        control_source="vision", target_steerable=True, low_quality_visible=False,
        near_yaw_park_requested=True,
    )
    assert owner._lateral_intent_store.snapshot().park_requested
    assert owner._near_yaw_park_request is not None


def test_fast_pid_prediction_requests_parking(owner):
    intent = _intent(owner)
    owner._lateral_intent_last_sequence = intent.sequence
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: None)
    owner._follow_controller.refresh_parked_lateral_pid = lambda **kw: SimpleNamespace(
        correction_rpm=0, desired_yaw_rate_dps=0, measured_yaw_rate_dps=19.53,
        predictive_braking=True, output_floor_reason="predictive_brake_coast",
    )
    owner._service_lateral_intent(NOW)
    assert owner._near_yaw_park_request is not None
    assert not owner._use_soft_stop_next


def test_retire_queued_motion_and_stale_stop_on_entry_and_release(owner, monkeypatch):
    owner.motor_io_lock = threading.Lock()
    owner.action_queue = queue.Queue()
    owner.action_queue.put(runtime.ACTION_ROTATE_RIGHT)
    park(owner)
    settled_park(owner, monkeypatch)
    assert owner.current_command is None
    assert owner.action_queue.empty()
    assert owner._near_yaw_park_generation == 1
    owner.action_queue.put(runtime.ACTION_STOP)
    owner.current_command = runtime.ACTION_ROTATE_LEFT
    assert publish_motion(owner)
    assert owner.action_queue.empty()
    assert owner.current_command is None
    assert owner._near_yaw_park_generation == 2


def test_stationary_center_then_straight_departure_releases_before_fresh_depth_commit(owner, monkeypatch):
    park(owner)
    settled_park(owner, monkeypatch)
    owner._brake_hold_active = True
    owner._brake_hold_label = "near_yaw_park"
    owner._brake_hold_stop_mode = "normal"
    owner._follow_controller.last_steering_pid_result = None  # zero yaw / legacy fallback
    owner._last_command_capture_frame = 577
    frame = SensorFrame(
        width=640, height=480, persons=[_target()], distance_m=1.8,
        distance_state=DistanceState(source="vision_depth", raw_distance_m=1.8,
            used_distance_m=1.8, sample_timestamp=NOW+.54),
        capture_frame_id=577, capture_timestamp=NOW+.54,
    )
    decision = ControlDecision(actions=[ControlAction.forward(30, "fresh_distance_forward")], reason="fresh_distance_forward")
    assert owner._release_near_yaw_park_for_decision(
        frame, _target(), decision, control_source="vision",
        target_steerable=True, low_quality_visible=False,
    )
    # A new measurement can now reach the canonical longitudinal commit in
    # this same visual cycle; releasing did not restore any previous grant.
    assert owner._depth30_linear_snapshot is None
    committed = []
    owner._follow_controller.search_state = "none"
    owner._follow_controller.last_action_frame = 223
    owner._follow_controller._longitudinal_only_decision = lambda *a, **kw: decision
    owner._commit_depth_linear_decision = lambda *a, **kw: (committed.append(kw) or [], True)
    assert owner._refresh_visual_depth_linear_authority(
        frame, _target(), decision, is_fresh_depth=True,
        target_steerable=True, low_quality_visible=False,
    )
    assert len(committed) == 1


@pytest.mark.parametrize("source,weak", [("depth30", False), ("vision", True)])
def test_depth_alone_or_weak_visual_cannot_release_center_parking(owner, source, weak):
    park(owner)
    frame = SensorFrame(width=640, height=480, capture_frame_id=577, capture_timestamp=NOW-.02)
    owner._last_command_capture_frame = 577
    assert not owner._release_near_yaw_park_for_decision(
        frame, _target(), ControlDecision(actions=[ControlAction.forward(30, "fresh_distance_forward")]),
        control_source=source, target_steerable=not weak, low_quality_visible=weak,
    )
    assert owner._near_yaw_park_request is not None


@pytest.mark.parametrize("stamp", [NOW-.02, NOW+.005])
def test_newer_cap_from_before_stop_cannot_release(owner, monkeypatch, stamp):
    park(owner)
    settled_park(owner, monkeypatch)
    assert not publish_motion(owner, stamp=stamp)
    assert owner._near_yaw_park_request is not None


def test_pivot_image_must_follow_quiet_confirmation(owner, monkeypatch):
    park(owner)
    settled_park(owner, monkeypatch)
    assert not publish_motion(owner, stamp=NOW+.46)
    assert owner._near_yaw_park_request is not None
    assert publish_motion(owner, capture=578, stamp=NOW+.54)
    assert owner._near_yaw_park_request is None


def test_translation_keeps_independent_motion_handoff(owner, monkeypatch):
    park(owner)
    settled_park(owner, monkeypatch)
    assert owner._release_near_yaw_park(capture_id=577, capture_timestamp=NOW+.46,
        target_id=1, qualified=True, x_ratio=.5, translation_requested=True, reason="new_forward")


def test_residual_motion_blocks_pivot_not_translation_gate(owner, monkeypatch):
    park(owner)
    settled_park(owner, monkeypatch)
    owner._action_runtime.near_yaw_park_release_ready = lambda *a: False
    owner._action_runtime.near_yaw_park_motion_ready = lambda *a: True
    assert not publish_motion(owner, stamp=NOW+.54)


def test_new_frame_without_actual_stop_ack_cannot_release(owner):
    park(owner)
    assert not publish_motion(owner)


@pytest.mark.parametrize("x,expected", [(.5, False), (.56, False), (.44, False), (.61, True), (.39, True)])
def test_park_release_band_after_quiet(owner, monkeypatch, x, expected):
    park(owner)
    settled_park(owner, monkeypatch)
    assert owner._release_near_yaw_park(capture_id=577, capture_timestamp=NOW+.54,
        target_id=1, qualified=True, x_ratio=x, reason="test") is expected


def test_search_handoff_also_rejects_pre_stop_image(owner, monkeypatch):
    park(owner)
    settled_park(owner, monkeypatch)
    assert not owner._release_near_yaw_park(capture_id=577, capture_timestamp=NOW-.02,
        reason="visual_state_handoff:searching", handoff=True)
