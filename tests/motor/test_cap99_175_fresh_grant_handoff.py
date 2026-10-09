"""New same-UID Depth grants must not turn ordinary wheel updates into zero.

Only fake clock, cached feedback and fake serial driver are used here.
"""
from dataclasses import replace
from itertools import product

import pytest

from car_control_modular.final_yaw_coalescing import (
    contract_fresh_forward_handoff, contract_fresh_forward_neutral_handoff)
from test_cap218_straight_handoff import _straight_writer
from test_cap331_intent_handoff import ordinary_intent
from test_visible_wheel_continuity import feedback


@pytest.mark.parametrize("yaw,expected", [(-4., (40, 48)), (0., (48, 48)),
                                          (4., (48, 40))])
def test_fresh_grant_pair_never_increases_either_guarded_wheel(yaw, expected):
    intent = ordinary_intent(sign=-1 if yaw < 0 else 1,
                             cap=175, capture=10.02, published=10.05,
                             initial_correction_rpm=int(yaw))
    assert contract_fresh_forward_handoff(
        1, intent, (48, 48), (1, 10, 48., 0.), (1, 11, 70., yaw),
        70., 10.06) == expected


@pytest.mark.parametrize("veto", ["opposite_yaw", "zeroed_yaw", "park_yaw",
                                   "reverse_wheel", "expired_intent", "zero_cap"])
def test_fresh_grant_pair_arithmetic_rejects_unqualified_change(veto):
    intent = ordinary_intent(cap=175, capture=10.02, published=10.05,
                             initial_correction_rpm=-4)
    old = (46, 50)
    planned = (1, 10, 48., -2.)
    current = (1, 11, 70., -4.)
    now = 10.06
    cap = 70.
    if veto == "opposite_yaw":
        current = (1, 11, 70., 4.)
    elif veto == "zeroed_yaw":
        pass
    elif veto == "park_yaw":
        intent = replace(intent, park_requested=True)
    elif veto == "reverse_wheel":
        old = (0, 96)
    elif veto == "expired_intent":
        now = 10.21
    else:
        cap = 0.
    assert contract_fresh_forward_handoff(
        1, intent, old, planned, current, cap, now,
        yaw_zeroed=veto == "zeroed_yaw") is None


def test_fresh_grant_pair_matrix_never_raises_either_wheel_or_cap():
    """Cover base, yaw and brake-cap changes without assuming one CAP trace."""
    intent = ordinary_intent(cap=175, capture=10.02, published=10.05)
    for old_base, old_yaw, new_base, new_yaw, cap in product(
            (8, 32, 80), (-6, -2, 0, 2, 6), (4, 20, 70),
            (-6, -2, 0, 2, 6), (4, 18, 60)):
        old = (old_base+old_yaw, old_base-old_yaw)
        if min(old) < 0:
            continue
        pair = contract_fresh_forward_handoff(
            1, intent, old, (1, 10, float(old_base), float(old_yaw)),
            (1, 11, float(new_base), float(new_yaw)),
            float(cap), 10.06)
        if pair is None:
            continue
        assert min(pair) >= 0 and sum(pair) > 0
        assert all(new <= previous for new, previous in zip(pair, old))
        assert .5*sum(pair) <= min(old_base, new_base, cap)
        assert .5*(pair[0]-pair[1]) == new_yaw


def test_opposite_yaw_new_grant_uses_one_bounded_straight_bridge():
    intent = ordinary_intent(cap=175, capture=10.02, published=10.05,
                             initial_correction_rpm=4)
    planned, current = (1, 10, 48., -4.), (1, 11, 46., 4.)
    old = (44, 52)
    assert contract_fresh_forward_handoff(
        1, intent, old, planned, current, 46., 10.06) is None
    assert contract_fresh_forward_neutral_handoff(
        1, intent, old, planned, current, 46., 10.06) == (44, 44)


def _final_writer(monkeypatch, *, yaw=0., veto=None):
    def change(runtime, owner, _driver, clock, state, raw, store, _old):
        clock[0] = 10.06
        owner._last_vision_control_ts = clock[0]
        raw[0] = owner._depth30_linear_snapshot = ("forward", 46., 1, clock[0])
        state[0], state[1] = 46., yaw
        store.publish(ordinary_intent(
            sign=-1 if yaw < 0 else 1, cap=175, capture=10.02,
            published=clock[0], initial_correction_rpm=int(yaw)))
        owner._lateral_yaw_revision += 1
        if veto == "stop":
            owner._explicit_stop_requested = True
        elif veto == "uid":
            owner._follow_controller.active_target_id = 2
        elif veto == "expired_grant":
            clock[0] = 10.32
        elif veto == "bad_feedback":
            runtime.get_steering_feedback = lambda: feedback(clock[0]-.2, 20, 20)

    return _straight_writer(monkeypatch, change)


@pytest.mark.parametrize("yaw", [-4., 0., 4.])
def test_final_fresh_grant_handoff_writes_bounded_pair_without_zero(monkeypatch, caplog, yaw):
    runtime, _owner, driver, _, _, _, _, _ = _final_writer(monkeypatch, yaw=yaw)
    with caplog.at_level("INFO"):
        runtime._service_follow_wheels()
    assert len(driver.pairs) == 1
    pair = (driver.pairs[0][0], -driver.pairs[0][1])
    assert min(pair) >= 0 and 0 < sum(pair)
    assert all(new <= 48 for new in pair)
    assert .5*(pair[0]-pair[1]) == yaw
    assert "FOLLOW_FINAL_AUTHORITY_CHANGED" not in caplog.text
    assert ("follow_wheel_fresh_grant_handoff" in caplog.text
            if yaw else "follow_wheel_straight_handoff" in caplog.text)


@pytest.mark.parametrize("veto", ["stop", "uid", "expired_grant", "bad_feedback"])
def test_final_fresh_grant_handoff_preserves_real_vetoes(monkeypatch, veto):
    runtime, _owner, driver, _, _, _, _, _ = _final_writer(
        monkeypatch, yaw=-4., veto=veto)
    runtime._service_follow_wheels()
    assert all(not (left > 0 and right < 0) for left, right in driver.pairs)


@pytest.mark.parametrize("late_change", ["grant", "stop", "intent", "ttl"])
def test_fresh_yaw_handoff_rechecks_physical_write_boundary(monkeypatch, late_change):
    runtime, owner, driver, clock, _state, raw, store, _ = _final_writer(
        monkeypatch, yaw=-4.)
    terminal = runtime._linear_packet_write_limit
    calls = []

    def changed_terminal(*args, **kwargs):
        calls.append(kwargs)
        if late_change == "grant":
            raw[0] = owner._depth30_linear_snapshot = ("forward", 46., 1, 10.061)
        elif late_change == "stop":
            runtime.backend.send_stop("concurrent_stop", mode="emergency",
                                      preserve_zero=True)
            owner._brake_hold_active = True
        elif late_change == "intent":
            store.publish(ordinary_intent(
                cap=179, capture=10.03, published=clock[0],
                initial_correction_rpm=-4))
        else:
            clock[0] = 10.32
        return terminal(*args, **kwargs)

    runtime._linear_packet_write_limit = changed_terminal
    runtime._service_follow_wheels()
    assert calls and calls[0]["source_grant"] is not None
    assert all(not (left > 0 and right < 0) for left, right in driver.pairs)
    if late_change == "stop":
        assert driver.stops == [1]
        assert not driver.pairs


def test_fresh_handoff_second_quiet_brake_cap_can_veto_positive_pair(monkeypatch):
    runtime, owner, driver, _clock, _state, _raw, _store, _ = _final_writer(
        monkeypatch, yaw=-4.)
    owner._depth_forward_continuation_required = lambda *_: True
    caps = []

    def falling_cap(_linear, _timing, _now, *, feedback=None, quiet=False):
        assert quiet and feedback is not None
        caps.append(50. if not caps else 0.)
        return (caps[-1], "braking_margin")

    owner._depth_forward_continuation_limit = falling_cap
    runtime._service_follow_wheels()
    assert caps == [50., 0.]
    assert driver.pairs == [(0, 0)]


def test_fresh_handoff_visual_intent_expiring_before_write_cannot_turn(monkeypatch):
    runtime, owner, driver, clock, _state, _raw, store, _ = _final_writer(
        monkeypatch, yaw=-4.)
    terminal = runtime._linear_packet_write_limit

    def expire_intent(*args, **kwargs):
        # New Depth is still younger than 250 ms; only the visual yaw proof
        # has expired. No old opposite-turn packet may be written.
        clock[0] = 10.22
        assert not store.snapshot().valid(clock[0])
        assert owner._depth30_linear_snapshot[3]+.25 > clock[0]
        return terminal(*args, **kwargs)

    runtime._linear_packet_write_limit = expire_intent
    runtime._service_follow_wheels()
    assert all(not (left > 0 and right < 0) for left, right in driver.pairs)


def test_fresh_handoff_stop_generation_changed_inside_terminal_cap(monkeypatch):
    runtime, owner, driver, _clock, _state, _raw, _store, _ = _final_writer(
        monkeypatch, yaw=-4.)
    owner._depth_forward_continuation_required = lambda *_: True
    stopped = []

    def stop_during_cap(_linear, _timing, _now, *, feedback=None, quiet=False):
        assert quiet and feedback is not None
        if not stopped:
            before = runtime.backend.stop_write_generation
            runtime.backend.send_stop("concurrent_stop", mode="emergency",
                                      preserve_zero=True)
            stopped.append((before, runtime.backend.stop_write_generation))
        return 50., "same_grant_braking_margin"

    owner._depth_forward_continuation_limit = stop_during_cap
    runtime._service_follow_wheels()
    assert stopped and stopped[0][1] > stopped[0][0]
    assert driver.stops == [1]
    assert driver.pairs == []  # Not even a speed-mode zero after STOP.


def test_concurrent_stop_before_final_handoff_is_not_overwritten_by_zero(monkeypatch):
    def stop_at_final(runtime, owner, _driver, clock, state, raw, store, _old):
        clock[0] = 10.06
        owner._last_vision_control_ts = clock[0]
        raw[0] = owner._depth30_linear_snapshot = ("forward", 46., 1, clock[0])
        state[0], state[1] = 46., -4.
        store.publish(ordinary_intent(
            sign=-1, cap=175, capture=10.02, published=clock[0],
            initial_correction_rpm=-4))
        owner._lateral_yaw_revision += 1
        runtime.backend.send_stop("concurrent_stop", mode="emergency",
                                  preserve_zero=True)

    runtime, _owner, driver, *_ = _straight_writer(monkeypatch, stop_at_final)
    runtime._service_follow_wheels()
    assert driver.stops == [1]
    assert not driver.pairs  # In particular, no FOLLOW_FINAL_AUTHORITY_CHANGED.


def test_second_new_grant_requires_new_normal_admission(monkeypatch, caplog):
    runtime, owner, driver, clock, state, raw, _store, _ = _final_writer(
        monkeypatch, yaw=-4.)
    terminal = runtime._linear_packet_write_limit
    first_source = []
    changed = []

    def third_grant(*args, **kwargs):
        first_source.append(kwargs["source_grant"][0][3])
        if not changed:
            # The first new grant has already replaced the old plan. A third
            # sample now lowers the legal base; neither earlier grant may be
            # borrowed to authorize a motor packet at this boundary.
            raw[0] = owner._depth30_linear_snapshot = (
                "forward", 20., 1, 10.061)
            state[0] = 20.
            changed.append(True)
        return terminal(*args, **kwargs)

    runtime._linear_packet_write_limit = third_grant
    with caplog.at_level("INFO"):
        runtime._service_follow_wheels()
    assert first_source == [10.06]
    assert driver.pairs == [(0, 0)]
    assert "follow_wheel_fresh_grant_handoff" not in caplog.text
    assert "physical_deadline_or_grant_changed_at_write" in caplog.text

    # A subsequent tick may use the third grant only by making a fresh plan
    # through the ordinary guard and terminal checks.
    driver.pairs.clear()
    clock[0] = 10.07
    runtime._service_follow_wheels()
    assert len(driver.pairs) == 1
    left, right = driver.pairs[0]
    assert left > 0 and right < 0
    assert .5*(left-right) <= 20.


@pytest.mark.parametrize("veto", [None, "old_sample", "wrong_uid", "stop",
                                   "bad_feedback", "expired_grant"])
def test_before_guard_new_grant_rebuild_uses_normal_wheel_guard(
        monkeypatch, caplog, veto):
    runtime, owner, driver, clock, state, raw, store, calls = _straight_writer(
        monkeypatch, lambda *_: None)
    guard_limit = runtime._visible_wheel_guard.limit
    guarded_requests = []

    def observed_guard(requested, *args, **kwargs):
        guarded_requests.append(requested)
        return guard_limit(requested, *args, **kwargs)

    runtime._visible_wheel_guard.limit = observed_guard
    original = runtime.get_steering_feedback

    def advancing_feedback():
        result = original()
        if calls["feedback"] == 2:
            stamp = (9.99 if veto == "old_sample" else clock[0])
            raw[0] = owner._depth30_linear_snapshot = (
                "forward", 38., 2 if veto == "wrong_uid" else 1, stamp)
            state[0], state[1] = 38., -4.
            store.publish(ordinary_intent(
                cap=99, capture=10.02, published=clock[0],
                initial_correction_rpm=-4))
            owner._lateral_yaw_revision += 1
            if veto == "stop":
                owner._explicit_stop_requested = True
            elif veto == "bad_feedback":
                result = feedback(clock[0]-.2, 20, 20)
            elif veto == "expired_grant":
                clock[0] = 10.32
        return result

    runtime.get_steering_feedback = advancing_feedback
    with caplog.at_level("INFO"):
        runtime._service_follow_wheels()
    if veto is None:
        assert driver.pairs == [(34, -42)]
        # The first attempt was discarded before its guard; the one executed
        # guard call belongs to the new grant's rebuilt pair.
        assert guarded_requests == [(34, 42)]
        assert "follow_wheel_early_grant_adopted" in caplog.text
        assert "FOLLOW20_AUTHORITY_CHANGED" not in caplog.text
    else:
        assert all(not (left > 0 and right < 0) for left, right in driver.pairs)


def test_new_opposite_yaw_before_guard_gets_full_new_plan(monkeypatch, caplog):
    runtime, owner, driver, clock, state, raw, store, calls = _straight_writer(
        monkeypatch, lambda *_: None)
    state[1] = -4.
    owner._lateral_yaw_revision += 1
    store.publish(ordinary_intent(
        sign=-1, cap=174, capture=10.01, published=10.04,
        initial_correction_rpm=-4))
    guard_limit = runtime._visible_wheel_guard.limit
    guarded_requests = []
    def observed_guard(requested, *args, **kwargs):
        guarded_requests.append(requested)
        return guard_limit(requested, *args, **kwargs)
    runtime._visible_wheel_guard.limit = observed_guard
    original = runtime.get_steering_feedback
    def new_sample_on_feedback():
        result = original()
        if calls["feedback"] == 2:
            clock[0] = 10.06
            owner._last_vision_control_ts = clock[0]
            raw[0] = owner._depth30_linear_snapshot = (
                "forward", 38., 1, clock[0])
            state[0], state[1] = 38., 4.
            store.publish(ordinary_intent(
                sign=1, cap=175, capture=10.02, published=clock[0],
                initial_correction_rpm=4))
            owner._lateral_yaw_revision += 1
        return result
    runtime.get_steering_feedback = new_sample_on_feedback
    with caplog.at_level("INFO"):
        runtime._service_follow_wheels()
    assert guarded_requests == [(42, 34)]
    assert driver.pairs == [(42, -34)]
    assert "follow_wheel_early_grant_adopted" in caplog.text
    assert "FOLLOW20_AUTHORITY_CHANGED" not in caplog.text


@pytest.mark.parametrize("yaw", [-4., 0., 4.])
def test_new_grant_after_wheel_guard_reaches_final_handoff(monkeypatch, caplog, yaw):
    """A second-attempt Depth publication must not zero before final review."""
    runtime, owner, driver, clock, state, raw, store, _ = _straight_writer(
        monkeypatch, lambda *_: None)
    guard_limit = runtime._visible_wheel_guard.limit
    calls = []

    def publish_after_guard(*args, **kwargs):
        result = guard_limit(*args, **kwargs)
        calls.append(result)
        if len(calls) == 1:
            clock[0] = 10.06
            owner._last_vision_control_ts = clock[0]
            raw[0] = owner._depth30_linear_snapshot = (
                "forward", 46., 1, clock[0])
            state[0], state[1] = 46., yaw
            store.publish(ordinary_intent(
                sign=-1 if yaw < 0 else 1, cap=175,
                capture=10.02, published=clock[0],
                initial_correction_rpm=int(yaw)))
            owner._lateral_yaw_revision += 1
        return result

    runtime._visible_wheel_guard.limit = publish_after_guard
    with caplog.at_level("INFO"):
        runtime._service_follow_wheels()
    assert len(calls) == 1
    assert len(driver.pairs) == 1
    left, right_raw = driver.pairs[0]
    right = -right_raw
    assert 0 < left <= 48 and 0 < right <= 48
    assert .5*(left-right) == yaw
    assert "FOLLOW20_AUTHORITY_CHANGED" not in caplog.text
    assert "FOLLOW_FINAL_AUTHORITY_CHANGED" not in caplog.text
    assert (("follow_wheel_fresh_grant_handoff" if yaw else
             "follow_wheel_straight_handoff") in caplog.text)


@pytest.mark.parametrize("publication_point", ["after_guard", "final_safety"])
def test_new_grant_opposite_yaw_preserves_bounded_forward_not_old_turn(
        monkeypatch, caplog, publication_point):
    def publish(_runtime, owner, _driver, clock, state, raw, store, _old):
        clock[0] = 10.06
        owner._last_vision_control_ts = clock[0]
        raw[0] = owner._depth30_linear_snapshot = ("forward", 46., 1, clock[0])
        state[0], state[1] = 46., 4.
        store.publish(ordinary_intent(
            sign=1, cap=175, capture=10.02, published=clock[0],
            initial_correction_rpm=4))
        owner._lateral_yaw_revision += 1

    runtime, owner, driver, clock, state, raw, store, _ = _straight_writer(
        monkeypatch, publish if publication_point == "final_safety" else lambda *_: None)
    state[1] = -4.
    owner._lateral_yaw_revision += 1
    store.publish(ordinary_intent(
        sign=-1, cap=174, capture=10.01, published=10.04,
        initial_correction_rpm=-4))
    if publication_point == "after_guard":
        guard_limit = runtime._visible_wheel_guard.limit
        def publish_after_guard(*args, **kwargs):
            result = guard_limit(*args, **kwargs)
            publish(runtime, owner, driver, clock, state, raw, store, None)
            return result
        runtime._visible_wheel_guard.limit = publish_after_guard
    with caplog.at_level("INFO"):
        runtime._service_follow_wheels()
    assert len(driver.pairs) == 1
    left, right_raw = driver.pairs[0]
    assert left == -right_raw and 0 < left <= 44
    assert "FOLLOW20_AUTHORITY_CHANGED" not in caplog.text
    assert "FOLLOW_FINAL_AUTHORITY_CHANGED" not in caplog.text
    assert "follow_wheel_fresh_grant_handoff" in caplog.text
    assert "yaw_neutralized=True" in caplog.text


@pytest.mark.parametrize("veto", ["old_sample", "wrong_uid", "reverse",
                                   "expired", "cap_too_low", "hold_zero",
                                   "park_turn", "identity", "feedback",
                                   "explicit_stop", "motor_stop"])
def test_after_guard_handoff_never_borrows_bad_authority(monkeypatch, veto):
    runtime, owner, driver, clock, state, raw, store, _ = _straight_writer(
        monkeypatch, lambda *_: None)
    guard_limit = runtime._visible_wheel_guard.limit
    calls = []

    def change_after_guard(*args, **kwargs):
        result = guard_limit(*args, **kwargs)
        calls.append(result)
        if len(calls) == 1:
            clock[0] = 10.06
            owner._last_vision_control_ts = clock[0]
            stamp = 9.99 if veto == "old_sample" else clock[0]
            kind = "backward" if veto == "reverse" else "forward"
            grant_uid = 2 if veto == "wrong_uid" else 1
            raw[0] = owner._depth30_linear_snapshot = (
                kind, 20. if veto == "cap_too_low" else 46., grant_uid, stamp)
            state[0], state[1] = 46., -4.
            intent = ordinary_intent(
                sign=-1, cap=175, capture=10.02,
                published=clock[0], initial_correction_rpm=-4)
            if veto == "hold_zero":
                intent = replace(intent, hold_zero=True)
            elif veto == "park_turn":
                intent = replace(intent, park_requested=True)
            store.publish(intent)
            owner._lateral_yaw_revision += 1
            if veto == "expired":
                clock[0] = 10.32
            elif veto == "identity":
                owner._follow_controller.active_target_id = 2
            elif veto == "feedback":
                runtime.get_steering_feedback = lambda: feedback(clock[0]-.2, 20, 20)
            elif veto == "explicit_stop":
                owner._explicit_stop_requested = True
            elif veto == "motor_stop":
                runtime.backend.send_stop("concurrent_stop", mode="emergency",
                                          preserve_zero=True)
        return result

    runtime._visible_wheel_guard.limit = change_after_guard
    runtime._service_follow_wheels()
    assert len(calls) == 1
    assert all(not (left > 0 and right < 0) for left, right in driver.pairs)
    if veto == "motor_stop":
        assert driver.stops == [1]
        assert not driver.pairs  # A speed-zero packet must not undo STOP mode.
