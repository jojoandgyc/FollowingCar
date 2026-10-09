"""Late same-UID straight handoffs use the current grant and full wheel path.

Only fake time, cached feedback, and a fake motor driver are used.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.lateral_intent import LateralIntentStore
from test_cap331_intent_handoff import ordinary_intent
from test_cap218_straight_handoff import _straight_writer
from test_follow_wheel_periodic import setup_periodic
from test_visible_wheel_continuity import feedback


def _two_updates(monkeypatch, *, second_base=52., second_change=None):
    runtime, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    state[:] = [60., 0., 11., 11.]
    raw = [("forward", 60., 1, 10.)]
    owner._depth30_linear_snapshot = raw[0]
    owner._depth_linear_max_age_sec = lambda kind: .25
    owner._last_vision_control_ts = 10.
    owner._follow_controller.cfg = SimpleNamespace(
        visible_steering_pid_image_error_only=True,
        visible_steering_pid_execution_response_trial_sec=.35,
        visible_steering_pid_max_correction_rpm=10.,
        near_distance_rotation_only_max_rpm=7.)
    store = owner._lateral_intent_store = LateralIntentStore()
    store.publish(ordinary_intent(
        sign=1, cap=494, capture=9.95, published=10.,
        initial_correction_rpm=0))

    def depth(uid, now=None):
        at = clock[0] if now is None else now
        grant = raw[0]
        if uid != 1 or not grant[3] <= at < grant[3]+.25:
            return None
        return ("forward", min(grant[1], state[0]), uid, grant[3])

    owner._fresh_depth_linear_snapshot = depth
    runtime.get_steering_feedback = lambda: feedback(clock[0], 20, 20)
    runtime._service_follow_wheels()
    assert driver.pairs == [(60, -60)]
    driver.pairs.clear()
    clock[0] = 10.05
    owner._last_vision_control_ts = clock[0]
    reads = [0]

    def read_feedback():
        reads[0] += 1
        if reads[0] == 1:
            state[:2] = [52., 6.]
            store.publish(ordinary_intent(
                sign=1, cap=495, capture=10.01, published=clock[0],
                initial_correction_rpm=6))
            owner._lateral_yaw_revision += 1
        elif reads[0] == 2:
            state[:2] = [second_base, 0.]
            zeroed = store.publish(replace(ordinary_intent(
                sign=1, cap=496, capture=10.02, published=clock[0],
                initial_correction_rpm=0), forward_countersteer=True,
                park_requested=True))
            owner._lateral_intent_zero_sequence = zeroed.sequence
            owner._lateral_yaw_revision += 1
            if second_change is not None:
                second_change(runtime, owner, driver, clock, state, raw)
        return feedback(clock[0], 20, 20)

    runtime.get_steering_feedback = read_feedback
    return runtime, owner, driver, clock, state, raw, reads


@pytest.mark.parametrize("second_base", [52., 50.])
def test_second_yaw_zero_uses_live_same_grant_and_full_wheel_guard(
        monkeypatch, second_base):
    runtime, owner, driver, clock, _, _, reads = _two_updates(
        monkeypatch, second_base=second_base)
    guarded = []
    original = runtime._visible_wheel_guard.limit

    def observe(requested, *args, **kwargs):
        guarded.append(requested)
        return original(requested, *args, **kwargs)

    runtime._visible_wheel_guard.limit = observe
    runtime._service_follow_wheels()
    assert reads[0] >= 2
    assert guarded == [(int(second_base), int(second_base))]
    assert driver.pairs == [(int(second_base), -int(second_base))]
    assert not driver.stops
    assert runtime._follow_wheel_clock.last_axes == owner._follow_wheel_axes(clock[0])


@pytest.mark.parametrize("veto", ["new_grant", "uid", "stop", "ttl", "park", "third_change"])
def test_second_yaw_zero_does_not_borrow_invalid_authority(monkeypatch, veto):
    def second(runtime, owner, _driver, clock, _state, raw):
        if veto == "new_grant":
            raw[0] = owner._depth30_linear_snapshot = ("forward", 52., 1, 10.04)
        elif veto == "uid":
            owner._follow_controller.active_target_id = 2
        elif veto == "stop":
            runtime.backend.send_stop("concurrent_stop", mode="emergency",
                                      preserve_zero=True)
        elif veto == "park":
            owner._near_yaw_park_request = SimpleNamespace(
                uid=1, capture_frame_id=496, reason="near_distance_stop")
        elif veto == "ttl":
            clock[0] = 10.26

    runtime, owner, driver, _, state, _, _ = _two_updates(
        monkeypatch, second_change=second)
    if veto == "third_change":
        checks = [0]

        def third_during_final_check(_action):
            checks[0] += 1
            if checks[0] == 3:
                state[:2] = [48., -6.]
                owner._lateral_yaw_revision += 1
            return False

        runtime.hard_stop_check = third_during_final_check
    runtime._service_follow_wheels()
    if veto == "third_change":
        assert checks[0] >= 3
    assert all(not (left > 0 and right < 0) for left, right in driver.pairs)
    if veto == "stop":
        assert driver.stops == [1]
        assert not driver.pairs


def test_second_yaw_zero_cannot_override_stop_after_candidate(monkeypatch):
    runtime, _, driver, _, _, _, _ = _two_updates(monkeypatch)
    original = runtime._second_same_grant_straight_axes
    candidates = []

    def stop_after_candidate(*args, **kwargs):
        candidate = original(*args, **kwargs)
        if candidate is not None:
            candidates.append(candidate)
            runtime.backend.send_stop("late_candidate_stop", mode="emergency",
                                      preserve_zero=True)
        return candidate

    runtime._second_same_grant_straight_axes = stop_after_candidate
    runtime._service_follow_wheels()
    assert candidates
    assert driver.stops == [1]
    assert not driver.pairs


def _new_grant_with_final_republication(monkeypatch):
    def new_grant(_runtime, owner, _driver, clock, state, raw, store, _old):
        clock[0] = 10.06
        owner._last_vision_control_ts = clock[0]
        raw[0] = owner._depth30_linear_snapshot = ("forward", 46., 1, clock[0])
        state[:2] = [46., 0.]
        store.publish(ordinary_intent(
            cap=696, capture=10.02, published=clock[0],
            initial_correction_rpm=0))
        owner._lateral_yaw_revision += 1

    runtime, owner, driver, clock, state, raw, store, calls = _straight_writer(
        monkeypatch, new_grant)
    original = runtime._can_defer_follow_base_contraction
    published = []

    def publish_during_final_review(*args, **kwargs):
        if calls["safety"] >= 3 and not published:
            published.append(store.publish(ordinary_intent(
                cap=697, capture=10.03, published=clock[0],
                initial_correction_rpm=0)))
            owner._lateral_yaw_revision += 1
        return original(*args, **kwargs)

    runtime._can_defer_follow_base_contraction = publish_during_final_review
    return runtime, owner, driver, clock, raw, store, published


def test_new_lower_grant_rechecked_after_another_visual_publication(monkeypatch):
    """A newer visual snapshot during final review cannot force a zero pulse."""
    runtime, owner, driver, clock, _, _, published = (
        _new_grant_with_final_republication(monkeypatch))
    original_lower = runtime._final_lower_straight_grant_handoff
    lower_results = []

    def observe_lower(*args, **kwargs):
        result = original_lower(*args, **kwargs)
        lower_results.append(result)
        return result

    runtime._final_lower_straight_grant_handoff = observe_lower
    runtime._service_follow_wheels()
    assert published
    assert any(result is not None for result in lower_results)
    assert driver.pairs == [(46, -46)]
    assert not driver.stops
    assert runtime._follow_wheel_clock.last_axes == owner._follow_wheel_axes(clock[0])


@pytest.mark.parametrize("late_change", [
    "third_grant", "uid", "stop", "stop_after_terminal", "lower_cap"])
def test_new_lower_grant_rechecks_terminal_authority(monkeypatch, late_change):
    runtime, owner, driver, clock, raw, _, published = (
        _new_grant_with_final_republication(monkeypatch))
    if late_change == "lower_cap":
        original_lower = runtime._final_lower_straight_grant_handoff

        def lower_then_cap(*args, **kwargs):
            candidate = original_lower(*args, **kwargs)
            if candidate is not None:
                # Tighten the live, age-dependent cap after the candidate,
                # before its final mandatory fresh-reader pass.
                owner._fresh_depth_linear_snapshot = lambda uid, now=None: (
                    "forward", 30., uid, raw[0][3])
            return candidate

        runtime._final_lower_straight_grant_handoff = lower_then_cap
    original_terminal = runtime._linear_packet_write_limit
    terminal_calls = []

    def change_before_terminal(*args, **kwargs):
        terminal_calls.append(kwargs)
        if late_change == "third_grant":
            raw[0] = owner._depth30_linear_snapshot = (
                "forward", 40., 1, clock[0]+.001)
        elif late_change == "uid":
            owner._follow_controller.active_target_id = 2
        elif late_change == "stop":
            runtime.backend.send_stop("concurrent_stop", mode="emergency",
                                      preserve_zero=True)
        decision = original_terminal(*args, **kwargs)
        if late_change == "stop_after_terminal":
            runtime.backend.send_stop("post_terminal_stop", mode="emergency",
                                      preserve_zero=True)
        return decision

    runtime._linear_packet_write_limit = change_before_terminal
    runtime._service_follow_wheels()
    assert published and terminal_calls
    assert all(not (left > 30 and right < -30) for left, right in driver.pairs)
    if late_change in {"stop", "stop_after_terminal"}:
        assert driver.stops == [1]
        assert not driver.pairs
    elif late_change == "lower_cap":
        assert driver.pairs == [(30, -30)]
    else:
        assert all(not (left > 0 and right < 0) for left, right in driver.pairs)


@pytest.mark.parametrize("veto", ["depth_none", "ttl", "stop"])
def test_new_grant_handoff_does_not_preserve_old_speed_without_final_proof(
        monkeypatch, veto):
    def change(runtime, owner, _driver, clock, state, raw, store, _old):
        clock[0] = 10.06
        owner._last_vision_control_ts = clock[0]
        raw[0] = owner._depth30_linear_snapshot = ("forward", 46., 1, clock[0])
        state[:2] = [46., 0.]
        store.publish(ordinary_intent(
            cap=698, capture=10.02, published=clock[0],
            initial_correction_rpm=0))
        owner._lateral_yaw_revision += 1
        if veto == "depth_none":
            owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
        elif veto == "ttl":
            clock[0] = 10.32
        else:
            runtime.backend.send_stop("concurrent_stop", mode="emergency",
                                      preserve_zero=True)

    runtime, _owner, driver, *_ = _straight_writer(monkeypatch, change)
    runtime._service_follow_wheels()
    assert all(not (left > 0 and right < 0) for left, right in driver.pairs)
    if veto == "stop":
        assert driver.stops == [1]
        assert not driver.pairs
