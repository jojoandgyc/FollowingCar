"""CAP331: a fresh stronger yaw must not erase the live forward axis.

Fake clock, feedback cache and serial driver only. The permitted replacement
lowers the common base so neither guarded wheel target is increased.
"""
from types import SimpleNamespace

import pytest

from test_follow_wheel_periodic import setup_periodic
from test_visible_wheel_continuity import feedback


def cap331_writer(monkeypatch, *, sign=-1, new_yaw=7, second_update=None,
                  final_update=None, second_stage="safety"):
    runtime, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    state[:] = [26., sign * 3., 10.25, 10.25]
    owner._depth30_linear_snapshot = ("forward", 26., 1, 10.)
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: (
        ("forward", state[0], uid, owner._depth30_linear_snapshot[3])
        if uid == 1 and clock[0] < state[2] and state[0] > 0 else None)
    runtime.get_steering_feedback = lambda: feedback(clock[0], 8, 8)
    runtime._service_follow_wheels()
    assert driver.pairs == [(26 + sign * 3, -(26 - sign * 3))]
    driver.pairs.clear()
    clock[0] = 10.05
    calls = {"feedback": 0, "safety": 0}

    def apply_second_update():
        state[1] = sign * float(new_yaw)
        owner._lateral_yaw_revision += 1
        if second_update:
            second_update(runtime, owner, clock, state)

    def read_feedback():
        calls["feedback"] += 1
        if calls["feedback"] == 1:
            state[1] = sign * 2.
            owner._lateral_yaw_revision += 1
        elif calls["feedback"] == 2 and second_stage == "feedback":
            apply_second_update()
        return feedback(clock[0], 8, 8)

    def safety(_):
        calls["safety"] += 1
        if calls["safety"] == 3 and second_stage == "safety":
            apply_second_update()
        if calls["safety"] == 4 and final_update:
            return bool(final_update(runtime, owner, clock, state))
        return False

    runtime.get_steering_feedback = read_feedback
    runtime.hard_stop_check = safety
    return runtime, owner, driver, clock, state, calls


@pytest.mark.parametrize("sign", [-1, 1])
def test_cap331_stronger_yaw_contracts_base_instead_of_zero(monkeypatch, sign):
    runtime, owner, driver, _, _, calls = cap331_writer(monkeypatch, sign=sign)
    runtime._service_follow_wheels()
    expected = (21 + sign * 7, 21 - sign * 7)
    assert driver.pairs == [(expected[0], -expected[1])]
    assert not driver.stops
    assert all(0 < new <= old for new, old in zip(expected, (26 + sign * 2, 26 - sign * 2)))
    assert runtime._follow_wheel_clock.last_axes == (1, owner._lateral_yaw_revision, 21., sign * 7.)
    assert runtime._visible_wheel_guard.last_output == expected
    assert runtime._forward_execution_anchor.rpm == 21.
    assert calls["feedback"] >= 2


@pytest.mark.parametrize("sign", [-1, 1])
def test_cap331_smaller_base_and_stronger_yaw_still_never_increase_wheel(monkeypatch, sign):
    def reduce_base(_runtime, _owner, _clock, state):
        state[0] = 18.
    runtime, owner, driver, _, _, _ = cap331_writer(
        monkeypatch, sign=sign, second_update=reduce_base)
    runtime._service_follow_wheels()
    assert driver.pairs == [(18 + sign * 7, -(18 - sign * 7))]
    assert runtime._follow_wheel_clock.last_axes == (1, owner._lateral_yaw_revision, 18., sign * 7.)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("smaller_base", [False, True])
def test_cap331_early_snapshot_change_reaches_the_same_guarded_merge(monkeypatch, sign, smaller_base):
    def reduce_base(_runtime, _owner, _clock, state):
        if smaller_base:
            state[0] = 18.
    runtime, owner, driver, _, _, _ = cap331_writer(
        monkeypatch, sign=sign, second_stage="feedback", second_update=reduce_base)
    runtime._service_follow_wheels()
    base = 18 if smaller_base else 21
    assert driver.pairs == [(base + sign * 7, -(base - sign * 7))]
    assert not driver.stops
    assert runtime._follow_wheel_clock.last_axes == (1, owner._lateral_yaw_revision, base, sign * 7.)


@pytest.mark.parametrize("veto", ["new_grant", "yaw_reverse", "no_forward_room",
                                  "park", "countersteer", "pending_reverse"])
def test_cap331_snapshot_merge_cannot_bypass_identity_or_wheel_policy(monkeypatch, veto):
    def change(runtime, owner, _clock, state):
        if veto == "new_grant":
            owner._depth30_linear_snapshot = ("forward", 26., 1, 10.01)
        elif veto == "yaw_reverse":
            state[1] = 7.
        elif veto == "no_forward_room":
            state[1] = -29.
        elif veto == "pending_reverse":
            runtime._visible_wheel_guard.pending_signs = (-1, 1)
        else:
            intent = SimpleNamespace(**{"park_requested" if veto == "park" else "forward_countersteer": True})
            owner._lateral_intent_store = SimpleNamespace(snapshot=lambda: intent)
    runtime, _, driver, _, _, _ = cap331_writer(monkeypatch, second_update=change)
    runtime._service_follow_wheels()
    assert all(pair == (0, 0) for pair in driver.pairs)


@pytest.mark.parametrize("veto", ["depth", "yaw", "uid", "feedback", "hazard", "stop"])
def test_cap331_new_pair_rechecks_stop_lease_and_latest_feedback(monkeypatch, veto):
    def change(runtime, owner, clock, state):
        if veto == "depth": state[2] = clock[0] - .001
        elif veto == "yaw": state[3] = clock[0] - .001
        elif veto == "uid": owner._follow_controller.active_target_id = 2
        elif veto == "feedback":
            runtime.get_steering_feedback = lambda: feedback(clock[0], -20, 8)
        elif veto == "hazard": return True
        else:
            runtime.backend.send_stop("concurrent_stop", mode="emergency", preserve_zero=True)
            owner._brake_hold_active = True
    runtime, _, driver, _, _, calls = cap331_writer(monkeypatch, final_update=change)
    runtime._service_follow_wheels()
    assert calls["safety"] >= 4
    assert all(pair == (0, 0) for pair in driver.pairs)
    if veto in {"hazard", "stop"}:
        assert driver.stops == [1]
        assert not driver.pairs


@pytest.mark.parametrize("yaw", [-3., 0., 3.])
@pytest.mark.parametrize("terminal_cap", [60., 0.])
def test_terminal_same_grant_cap_step_lowers_both_wheels(monkeypatch, yaw, terminal_cap):
    from test_follow_same_grant_speed_contraction import _live_depth_grant
    runtime, owner, driver, clock, grant = _live_depth_grant(monkeypatch)
    grant.update(initial_percent=62., first_cap=62., middle_cap=62.,
                 late_cap=62., terminal_cap=62., yaw=yaw)
    owner._depth30_linear_snapshot = ("forward", 62., 1, grant["stamp"])
    owner._has_fresh_lateral_yaw = lambda uid: uid == 1 and bool(grant["yaw"])
    owner._depth_forward_continuation_required = lambda *args: True
    reads = []

    def final_cap(linear, timing, now, *, feedback=None, quiet=False):
        reads.append((now, quiet))
        return terminal_cap, "same_grant_relative_braking_cap" if terminal_cap else "braking_margin"

    owner._depth_forward_continuation_limit = final_cap
    runtime._service_follow_wheels()
    assert reads and all(quiet for _, quiet in reads)
    if terminal_cap:
        assert driver.pairs == [(60 + yaw, -(60 - yaw))]
        assert runtime._follow_wheel_clock.last_axes == (1, owner._lateral_yaw_revision, 60., yaw)
        assert runtime._forward_execution_anchor.rpm == 60.
    else:
        assert driver.pairs == [(0, 0)]
        assert not driver.stops


def test_terminal_cap_contraction_cannot_overwrite_a_new_stop(monkeypatch):
    from test_follow_same_grant_speed_contraction import _live_depth_grant
    runtime, owner, driver, _, grant = _live_depth_grant(monkeypatch)
    owner._depth_forward_continuation_required = lambda *args: True
    calls = [0]

    def final_cap(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            runtime.backend.send_stop("terminal_cap_stop", mode="emergency", preserve_zero=True)
            owner._brake_hold_active = True
        return 58., "same_grant_relative_braking_cap"

    owner._depth_forward_continuation_limit = final_cap
    runtime._service_follow_wheels()
    assert calls[0] >= 1
    assert driver.stops == [1]
    assert not driver.pairs


@pytest.mark.parametrize("change", ["yaw", "revision", "intent", "policy", "new_grant"])
def test_terminal_cap_compute_cannot_write_a_superseded_snapshot(monkeypatch, change):
    from test_follow_same_grant_speed_contraction import _live_depth_grant
    runtime, owner, driver, _, grant = _live_depth_grant(monkeypatch)
    grant.update(first_cap=62., middle_cap=62., late_cap=62., terminal_cap=62., yaw=-2.)
    owner._depth30_linear_snapshot = ("forward", 62., 1, grant["stamp"])
    owner._has_fresh_lateral_yaw = lambda uid: uid == 1 and bool(grant["yaw"])
    owner._depth_forward_continuation_required = lambda *args: True
    intent = [SimpleNamespace(park_requested=False)]
    owner._lateral_intent_store = SimpleNamespace(snapshot=lambda: intent[0])
    calls = [0]

    def final_cap(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            if change == "yaw":
                grant["yaw"] = -7.
                owner._lateral_yaw_revision += 1
            elif change == "revision":
                owner._lateral_yaw_revision += 1
            elif change == "intent":
                intent[0] = SimpleNamespace(park_requested=True)
            elif change == "policy":
                owner._lateral_turn_response_policy = (999, False)
            else:
                owner._depth30_linear_snapshot = ("forward", 62., 1, grant["stamp"] + .001)
        return 60., "same_grant_relative_braking_cap"

    owner._depth_forward_continuation_limit = final_cap
    runtime._service_follow_wheels()
    assert calls[0] >= 1
    assert all(pair == (0, 0) for pair in driver.pairs)
