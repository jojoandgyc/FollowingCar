"""Real lateral publication -> periodic writer -> fake serial, no hardware."""
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "control"))
from test_lateral_zero_runtime import owner, runtime as main, NOW, _target
from test_depth_drive_rpm import make_runtime
from test_visible_wheel_continuity import feedback
from car_control_modular.control_types import ControlAction
from car_control_modular.near_yaw_parking import NearYawParkRequest


def published_center_writer(monkeypatch, owner):
    executor, _, driver, symbols = make_runtime(max_rpm=200)
    executor.owner = owner
    executor.config.follow_wheel_period_sec = .05
    owner.motor_io_lock = executor.backend.io_lock
    owner._action_runtime = executor
    owner.current_command = symbols.forward
    owner._follow_controller.search_state = "none"
    owner._follow_controller.cfg = SimpleNamespace(
        visible_steering_pid_image_error_only=True,
        visible_steering_pid_execution_response_trial_sec=.35,
        visible_steering_pid_max_correction_rpm=10,
        near_distance_rotation_only_max_rpm=7,
    )
    owner._last_control_decision_reason = "visual_pid_center_hold"
    owner._follow_controller.last_steering_pid_result = SimpleNamespace(
        correction_rpm=0, correction_limit_rpm=10., base_rpm=60,
        target_rate_valid=False, target_image_rate_dps=0.,
        output_floor_reason="center_hold", feedback_used=False,
        predictive_braking=False, visual_error_deg=0.,
    )
    owner._depth30_linear_snapshot = ("forward", 30., 1, NOW-.01)
    owner._depth_linear_max_age_sec = lambda kind: .25
    clock = [NOW]
    monkeypatch.setattr(main.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(main, "FORWARD_MAX_RPM", 200)
    executor.get_steering_feedback = lambda: feedback(clock[0], 20, 20)

    # The real producer sets park_requested despite retaining positive Depth.
    assert owner._publish_lateral_intent_from_decision(
        width=640, target=_target(),
        runtime_actions=[ControlAction.forward(30, "visual_pid_center_hold")],
        control_source="vision", target_steerable=True, low_quality_visible=False)
    intent = owner._lateral_intent_store.snapshot()
    assert intent.park_requested and intent.initial_correction_rpm == 0
    assert intent.mode == "forward" and not intent.hold_zero
    assert getattr(owner, "_near_yaw_park_request", None) is None
    assert owner._current_steer_base_percent == 30
    assert not driver.pairs and not driver.stops

    def cap(now):
        stamp = owner._depth30_linear_snapshot[3]
        if not 0 <= now-stamp < .25:
            return 0.
        if stamp < NOW:
            return 30.
        return 31. if now < NOW+.065 else 30. if now < NOW+.085 else 29.

    owner._fresh_depth_linear_snapshot = lambda uid, now=None: (
        ("forward", cap(clock[0] if now is None else now), uid, owner._depth30_linear_snapshot[3])
        if uid == 1 and cap(clock[0] if now is None else now) else None)
    owner._depth_forward_continuation_required = lambda *args: False
    owner._follow_wheel_axes = lambda now: (1, owner._lateral_yaw_revision, 2*cap(now), 0.)
    owner._has_fresh_lateral_yaw = lambda uid: False
    executor._service_follow_wheels()
    assert driver.pairs == [(60, -60)]
    driver.pairs.clear()
    clock[0] = NOW+.05
    owner._depth30_linear_snapshot = ("forward", 31., 1, clock[0])
    reads = [0]

    def read_feedback():
        reads[0] += 1
        if reads[0] <= 2:
            clock[0] += .02
        return feedback(clock[0], 20, 20)

    executor.get_steering_feedback = read_feedback
    return executor, driver, clock, reads


def test_center_hold_candidate_keeps_same_grant_monotone_base(owner, monkeypatch, caplog):
    executor, driver, clock, reads = published_center_writer(monkeypatch, owner)
    source = owner._depth30_linear_snapshot
    with caplog.at_level("INFO"):
        executor._service_follow_wheels()
    assert reads[0] == 3
    assert owner._depth30_linear_snapshot is source
    assert clock[0] < source[3]+.25
    assert driver.pairs == [(58, -58)]
    assert not driver.stops
    assert "final_base_coalesced" in caplog.text
    assert "FOLLOW_FINAL_AUTHORITY_CHANGED" not in caplog.text
    assert executor.backend.last_speed_receipt.left_rpm == 58


@pytest.mark.parametrize("veto", ["near_mode", "yaw", "countersteer"])
def test_park_candidate_exception_is_only_straight_forward(owner, monkeypatch, caplog, veto):
    executor, driver, _, _ = published_center_writer(monkeypatch, owner)
    intent = owner._lateral_intent_store.snapshot()
    changes = {"near_distance_mode": True} if veto == "near_mode" else (
        {"initial_correction_rpm": 4} if veto == "yaw" else {"countersteer_rpm": 4})
    owner._lateral_intent_store.publish(replace(intent, **changes))
    with caplog.at_level("INFO"):
        executor._service_follow_wheels()
    assert driver.pairs == [(0, 0)]
    assert "contraction_first_reject=park_candidate_not_straight_contraction" in caplog.text
    assert "park_candidate=True straight_park_candidate=False stop_owner=none" in caplog.text


@pytest.mark.parametrize("veto", ["depth_expired", "new_grant", "feedback_stale", "stop", "park"])
def test_candidate_contraction_rechecks_final_authority(owner, monkeypatch, caplog, veto):
    executor, driver, clock, reads = published_center_writer(monkeypatch, owner)
    original = executor.hard_stop_check
    checks = [0]

    def safety(action):
        checks[0] += 1
        # This is the final safety check after a smaller pair was prepared.
        if checks[0] == 4:
            if veto == "depth_expired":
                clock[0] = owner._depth30_linear_snapshot[3]+.251
            elif veto == "new_grant":
                owner._depth30_linear_snapshot = ("forward", 31., 1, clock[0])
            elif veto == "feedback_stale":
                executor.get_steering_feedback = lambda: feedback(clock[0]-.2, 20, 20)
            elif veto == "park":
                # Inject the established owner at the final boundary. The
                # production publisher takes motor_io_lock and is exercised
                # outside that lock by the integration test below.
                owner._near_yaw_park_request = NearYawParkRequest(
                    1, 576, NOW-.09, clock[0], "low_speed_turn_stop:visual_pid_center_hold")
            else:
                executor.backend.send_stop("concurrent_stop", mode="emergency", preserve_zero=True)
                owner._brake_hold_active = True
        return original(action)

    executor.hard_stop_check = safety
    with caplog.at_level("INFO"):
        executor._service_follow_wheels()
    assert checks[0] >= 4
    if veto == "new_grant":
        # The newer same-UID physical sample still authorizes a smaller
        # straight pair; its last write check binds that exact new sample.
        assert driver.pairs == [(58, -58)]
        assert "follow_wheel_straight_handoff" in caplog.text
        assert not driver.stops
        return
    assert all(pair == (0, 0) for pair in driver.pairs)
    if veto == "stop":
        assert driver.stops == [1]
        assert not driver.pairs
    elif veto == "park":
        assert not driver.pairs
        executor._service_follow_wheels()
        assert driver.stops == [1, 0]
    elif veto == "feedback_stale":
        assert "contraction_first_reject=wheel_feedback" in caplog.text
        assert "park_candidate=True straight_park_candidate=True stop_owner=none" in caplog.text


def test_producer_qualified_pivot_park_still_owns_stop(owner, monkeypatch):
    executor, driver, clock, _ = published_center_writer(monkeypatch, owner)
    # An actual near-distance pivot stop retains whole-chassis ownership.
    # Ordinary forward center hints no longer promote residual pivot motion
    # into parking; their reversal is handled by the wheel executor instead.
    owner._depth30_linear_snapshot = ("forward", 3., 1, clock[0]-.01)
    executor.get_steering_feedback = lambda: feedback(clock[0]-.01, -8, 8)
    intent = owner._lateral_intent_store.publish(replace(
        owner._lateral_intent_store.snapshot(), mode="yaw_only", near_distance_mode=True))
    owner._publish_lateral_zero(intent, "visual_pid_center_hold")
    assert owner._near_yaw_park_request is not None
    executor._service_follow_wheels()
    assert driver.stops == [1, 0]
    assert all(pair == (0, 0) for pair in driver.pairs)
    # The configured STOP transition may pre-zero moving wheels. Once STOP
    # owns them, a periodic speed packet must not release that parking mode.
    count = len(driver.pairs)
    executor._service_follow_wheels()
    assert len(driver.pairs) == count
    assert driver.stops == [1, 0]
