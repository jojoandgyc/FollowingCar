"""Expired yaw metadata must not zero an unchanged live forward grant.

Fake clock and fake serial driver only; the second producer update occurs
after the one permitted wheel-plan rebuild, matching CAP577/CAP622.
"""
from types import SimpleNamespace

import pytest

from car_control_modular.lateral_intent import LateralIntentStore
from test_cap1663_turn_response import intent
from test_follow_wheel_periodic import setup_periodic
from test_visible_wheel_continuity import feedback


def expired_straight_writer(monkeypatch, *, lower_base=False, assist=False,
                            change=None, last_check=None):
    runtime, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    state[:] = [12., 0., 10.25, 10.25]
    owner._depth30_linear_snapshot = ("forward", 12., 1, 10.)
    owner._depth_linear_max_age_sec = lambda kind: .25
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: (
        ("forward", state[0], uid, owner._depth30_linear_snapshot[3])
        if clock[0] < state[2] else None)
    owner._follow_controller.cfg = SimpleNamespace(
        visible_steering_pid_image_error_only=True,
        visible_steering_pid_execution_response_trial_sec=.35,
        visible_steering_pid_max_correction_rpm=10,
        near_distance_rotation_only_max_rpm=7)
    runtime.config.follow_turn_response_assist_enable = assist
    owner._lateral_intent_store = LateralIntentStore()
    owner._lateral_intent_store.publish(intent(9.9, x_ratio=.5, valid_until=10.06))
    runtime.get_steering_feedback = lambda: feedback(clock[0], 6, 5)
    runtime._service_follow_wheels()
    assert driver.pairs == [(12, -12)]
    driver.pairs.clear()
    clock[0] = 10.05
    calls = {"feedback": 0, "safety": 0}

    def read_feedback():
        calls["feedback"] += 1
        if calls["feedback"] == 1:
            owner._lateral_yaw_revision += 1
        return feedback(clock[0], 6, 5)

    def check_safety(_):
        calls["safety"] += 1
        if calls["safety"] == 3:
            clock[0] = 10.07
            owner._lateral_intent_store.clear()
            owner._lateral_yaw_revision += 1
            if lower_base:
                state[0] = 10.
            if change:
                change(runtime, owner, clock, state)
        if calls["safety"] == 4 and last_check:
            return bool(last_check(runtime, owner, clock, state))
        return False

    runtime.get_steering_feedback = read_feedback
    runtime.hard_stop_check = check_safety
    return runtime, owner, driver, clock, state, calls


@pytest.mark.parametrize("lower_base", [False, True])
@pytest.mark.parametrize("assist", [False, True])
def test_expired_yaw_zero_keeps_same_grant_forward_pair(monkeypatch, lower_base, assist):
    runtime, owner, driver, clock, state, calls = expired_straight_writer(
        monkeypatch, lower_base=lower_base, assist=assist)
    runtime._service_follow_wheels()
    assert driver.pairs == [(state[0], -state[0])]
    assert not driver.stops
    assert calls == {"feedback": 3, "safety": 5}  # Final post-lock safety check.
    assert runtime._follow_wheel_clock.last_axes == owner._follow_wheel_axes(clock[0])


@pytest.mark.parametrize("veto", ["early_clear", "new_intent", "new_grant", "receipt",
                                  "expired_depth", "yaw", "base_increase"])
def test_expired_metadata_exception_needs_unchanged_forward_authority(monkeypatch, veto):
    def change(runtime, owner, clock, state):
        if veto == "early_clear": clock[0] = 10.055
        elif veto == "new_intent": owner._lateral_intent_store.publish(intent(clock[0]))
        elif veto == "new_grant": owner._depth30_linear_snapshot = ("forward", 12., 1, 10.01)
        elif veto == "receipt": runtime.backend.last_speed_receipt = None
        elif veto == "expired_depth": state[2] = clock[0] - .001
        elif veto == "yaw": state[1] = 1.
        elif veto == "base_increase": state[0] = 14.
    runtime, _, driver, _, _, _ = expired_straight_writer(monkeypatch, change=change)
    runtime._service_follow_wheels()
    assert all(pair == (0, 0) for pair in driver.pairs)


@pytest.mark.parametrize("veto", ["third_revision", "new_grant", "expired_depth",
                                  "reverse_feedback", "stop", "hazard"])
def test_final_checks_still_own_expired_intent_coalescing(monkeypatch, veto):
    def change(runtime, owner, clock, state):
        if veto == "third_revision": owner._lateral_yaw_revision += 1
        elif veto == "new_grant": owner._depth30_linear_snapshot = ("forward", 12., 1, 10.01)
        elif veto == "expired_depth": state[2] = clock[0] - .001
        elif veto == "reverse_feedback": runtime.get_steering_feedback = lambda: feedback(clock[0], -2, 5)
        elif veto == "stop":
            runtime.backend.send_stop("concurrent_stop", mode="emergency", preserve_zero=True)
            owner._brake_hold_active = True
        elif veto == "hazard": return True
    runtime, _, driver, _, _, calls = expired_straight_writer(monkeypatch, last_check=change)
    runtime._service_follow_wheels()
    assert calls["safety"] == 4
    assert all(pair == (0, 0) for pair in driver.pairs)
    if veto in {"stop", "hazard"}:
        assert driver.stops == [1]
        assert not driver.pairs
