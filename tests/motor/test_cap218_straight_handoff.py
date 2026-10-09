"""CAP218/229/233: fake-clock positive wheel handoffs at the final writer."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.final_yaw_coalescing import contract_straight_forward_handoff
from car_control_modular.control_types import SteeringFeedback
from car_control_modular.lateral_intent import LateralIntentStore
from test_cap331_intent_handoff import ordinary_intent
from test_follow_wheel_periodic import setup_periodic
from test_visible_wheel_continuity import feedback


def _straight_writer(monkeypatch, change):
    runtime, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    state[:] = [50., 0., 11., 11.]
    raw = [("forward", 50., 1, 10.)]
    owner._depth30_linear_snapshot = raw[0]
    owner._depth_linear_max_age_sec = lambda kind: .25
    owner._last_vision_control_ts = 10.
    owner._follow_controller.cfg = SimpleNamespace(
        visible_steering_pid_image_error_only=True,
        visible_steering_pid_execution_response_trial_sec=.35,
        visible_steering_pid_max_correction_rpm=10,
        near_distance_rotation_only_max_rpm=7)
    store = owner._lateral_intent_store = LateralIntentStore()
    old = store.publish(replace(
        ordinary_intent(sign=-1, cap=218, capture=9.9, published=10.,
                        initial_correction_rpm=0), valid_until=10.15))

    def depth(uid, now=None):
        at = clock[0] if now is None else now
        grant = raw[0]
        if uid != 1 or not grant[3] <= at < grant[3]+.25:
            return None
        return ("forward", min(grant[1], state[0]), 1, grant[3])

    owner._fresh_depth_linear_snapshot = depth
    runtime.get_steering_feedback = lambda: feedback(clock[0], 20, 20)
    runtime._service_follow_wheels()
    assert driver.pairs == [(50, -50)]
    driver.pairs.clear()
    clock[0] = 10.05
    owner._last_vision_control_ts = clock[0]
    calls = {"feedback": 0, "safety": 0}

    def read_feedback():
        calls["feedback"] += 1
        if calls["feedback"] == 1:
            state[0] = 48.
            owner._lateral_yaw_revision += 1
        return feedback(clock[0], 20, 20)

    def safety(_):
        calls["safety"] += 1
        if calls["safety"] == 3:
            change(runtime, owner, driver, clock, state, raw, store, old)
        return False

    runtime.get_steering_feedback = read_feedback
    runtime.hard_stop_check = safety
    return runtime, owner, driver, clock, state, raw, store, calls


@pytest.mark.parametrize("case,expected", [
    ("expired_park_candidate", 45),
    ("new_lower_grant", 46),
    ("new_higher_grant", 48),
])
def test_positive_straight_handoff_writes_no_zero(monkeypatch, caplog, case, expected):
    def change(runtime, owner, _driver, clock, state, raw, store, old):
        clock[0] = 10.06
        owner._last_vision_control_ts = clock[0]
        if case == "expired_park_candidate":
            # Expired visual yaw can clear while physical Depth still grants
            # translation. Its parking hint did not arm a real parking owner.
            store.clear()
            state[0] = 45.
        else:
            raw[0] = owner._depth30_linear_snapshot = (
                "forward", 46. if case == "new_lower_grant" else 58., 1, clock[0])
            state[0] = 46. if case == "new_lower_grant" else 58.
            store.publish(replace(ordinary_intent(
                sign=-1, cap=229, capture=10.02, published=clock[0],
                initial_correction_rpm=0), park_requested=True))
        owner._lateral_yaw_revision += 1

    runtime, owner, driver, clock, state, raw, store, calls = _straight_writer(monkeypatch, change)
    if case == "expired_park_candidate":
        # The captured, subsequently cleared intent is a candidate only.
        old = store.snapshot()
        store.publish(replace(old, valid_until=10.055, park_requested=True))
    with caplog.at_level("INFO"):
        runtime._service_follow_wheels()
    assert calls["feedback"] >= 2 and calls["safety"] >= 3
    assert driver.pairs == [(expected, -expected)]
    assert not driver.stops
    assert "follow_wheel_straight_handoff" in caplog.text
    assert "FOLLOW_FINAL_AUTHORITY_CHANGED" not in caplog.text


def test_straight_handoff_arithmetic_never_adds_wheel_speed():
    intent = replace(ordinary_intent(cap=229, capture=10.02, published=10.05,
                                     initial_correction_rpm=0), park_requested=True)
    assert contract_straight_forward_handoff(
        1, intent, (48, 48), (1, 2, 48., 0.), (1, 3, 58., 0.),
        58., 10.06) == (48, 48)
    assert contract_straight_forward_handoff(
        1, intent, (48, 48), (1, 2, 48., 0.), (1, 3, 58., 0.),
        0., 10.06) is None


def test_cleared_visual_intent_cannot_bind_a_new_depth_grant(monkeypatch):
    def change(_runtime, owner, _driver, clock, state, raw, store, _old):
        clock[0] = 10.06
        owner._last_vision_control_ts = clock[0]
        store.clear()
        raw[0] = owner._depth30_linear_snapshot = ("forward", 46., 1, clock[0])
        state[0] = 46.
        owner._lateral_yaw_revision += 1

    runtime, _owner, driver, _, _, _, store, _ = _straight_writer(monkeypatch, change)
    old = store.snapshot()
    store.publish(replace(old, valid_until=10.055, park_requested=True))
    runtime._service_follow_wheels()
    assert all(not (left > 0 and right < 0) for left, right in driver.pairs)


@pytest.mark.parametrize("veto", ["stop", "park", "zero", "reverse", "expired_grant",
                                    "identity", "turn", "bad_feedback"])
def test_straight_handoff_keeps_real_vetoes(monkeypatch, veto):
    def change(runtime, owner, _driver, clock, state, raw, store, _old):
        clock[0] = 10.06
        owner._last_vision_control_ts = clock[0]
        raw[0] = owner._depth30_linear_snapshot = ("forward", 46., 1, clock[0])
        state[0] = 46.
        if veto == "stop":
            owner._explicit_stop_requested = True
        elif veto == "park":
            owner._near_yaw_park_request = SimpleNamespace(
                uid=1, capture_frame_id=229, reason="real_park")
        elif veto == "zero":
            state[0] = 0.
        elif veto == "reverse":
            raw[0] = owner._depth30_linear_snapshot = ("backward", 46., 1, clock[0])
        elif veto == "expired_grant":
            clock[0] = 10.32
        elif veto == "identity":
            owner._follow_controller.active_target_id = 2
        elif veto == "turn":
            state[1] = 4.
        else:
            runtime.get_steering_feedback = lambda: feedback(clock[0], -4, 20)
        store.publish(replace(ordinary_intent(
            sign=-1, cap=229, capture=10.02, published=10.06,
            initial_correction_rpm=0), park_requested=True))
        owner._lateral_yaw_revision += 1

    runtime, _owner, driver, _clock, _, _, _, _ = _straight_writer(monkeypatch, change)
    runtime._service_follow_wheels()
    assert all(not (left > 0 and right < 0) for left, right in driver.pairs)


@pytest.mark.parametrize("late_change", ["grant", "stop", "park", "intent", "policy",
                                       "intent_turn", "policy_turn"])
def test_straight_handoff_rechecks_adopted_evidence_at_physical_write(
        monkeypatch, late_change):
    def change(_runtime, owner, _driver, clock, state, raw, store, _old):
        clock[0] = 10.06
        owner._last_vision_control_ts = clock[0]
        raw[0] = owner._depth30_linear_snapshot = ("forward", 46., 1, clock[0])
        state[0] = 46.
        store.publish(replace(ordinary_intent(
            sign=-1, cap=229, capture=10.02, published=clock[0],
            initial_correction_rpm=0), park_requested=True))
        owner._lateral_yaw_revision += 1

    runtime, owner, driver, clock, state, raw, store, _ = _straight_writer(monkeypatch, change)
    if late_change.endswith("_turn"):
        read_feedback = runtime.get_steering_feedback

        def with_yaw_feedback():
            current = read_feedback()
            return SteeringFeedback(timestamp=current.timestamp,
                left_forward_rpm=current.left_forward_rpm,
                right_forward_rpm=current.right_forward_rpm,
                trustworthy=current.trustworthy, yaw_rate_right_dps=0.,
                raw_yaw_rate_right_dps=0.)

        runtime.get_steering_feedback = with_yaw_feedback
    terminal = runtime._linear_packet_write_limit
    calls = []

    def changed_terminal(*args, **kwargs):
        calls.append(kwargs)
        if late_change == "grant":
            raw[0] = owner._depth30_linear_snapshot = ("forward", 46., 1, 10.061)
        elif late_change == "stop":
            runtime.backend.send_stop("concurrent_stop", mode="emergency", preserve_zero=True)
            owner._brake_hold_active = True
        elif late_change == "park":
            owner._near_yaw_park_request = SimpleNamespace(
                uid=1, capture_frame_id=229, reason="real_park")
        elif late_change in {"intent", "intent_turn"}:
            store.publish(replace(ordinary_intent(
                sign=-1, cap=233, capture=10.03, published=clock[0],
                initial_correction_rpm=4 if late_change == "intent_turn" else 0),
                park_requested=late_change == "intent"))
            if late_change == "intent_turn":
                state[1] = 4.
                owner._lateral_yaw_revision += 1
        else:
            owner._lateral_turn_response_policy = (999, False)
            if late_change == "policy_turn":
                # Same immutable intent, but the live PID correction changes.
                # A stale policy with real nonzero yaw must still reject the
                # old straight handoff, not inherit neutral-yaw exemption.
                owner._lateral_intent_last_sequence = store.snapshot().sequence
                owner._lateral_intent_last_correction_rpm = 4
                state[1] = 4.
                owner._lateral_yaw_revision += 1
        return terminal(*args, **kwargs)

    runtime._linear_packet_write_limit = changed_terminal
    runtime._service_follow_wheels()
    assert calls and calls[0]["source_grant"] is not None
    if late_change in {"intent", "policy"}:
        # Both publications still request ZERO yaw. Their object/sequence
        # identity cannot revoke an independently checked equal-wheel grant.
        assert state[1] == store.snapshot().initial_correction_rpm == 0
        assert driver.pairs == [(46, -46)] and not driver.stops
        assert runtime._periodic_follow_axes[2:] == (46., 0.)
        assert owner._depth30_linear_snapshot == ("forward", 46., 1, 10.06)
    else:
        assert all(not (left > 0 and right < 0) for left, right in driver.pairs)
    if late_change == "stop":
        assert driver.stops == [1]
        assert not driver.pairs
