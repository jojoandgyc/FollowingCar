"""The final writer cannot rebaseline an already adopted visual publication."""
from dataclasses import replace

import pytest

from car_control_modular.final_yaw_coalescing import ordinary_intent_handoff
from car_control_modular.lateral_intent import LateralIntentStore
from test_cap331_intent_handoff import ordinary_intent, intent_handoff_writer
from test_cap527_terminal_pair_clock import writer
from test_visible_wheel_continuity import feedback


@pytest.mark.parametrize("change", ["intent", "policy", "neither"])
def test_adopted_owner_is_not_recaptured_after_intervening_publication(monkeypatch, change):
    runtime, owner, driver, clock, _ = writer(monkeypatch, yaw=0.)
    store = owner._lateral_intent_store = LateralIntentStore()
    adopted = store.publish(ordinary_intent())
    policy = owner._lateral_turn_response_policy = (adopted.sequence, False)
    if change == "intent":
        store.publish(ordinary_intent(cap=331, capture=10., published=10.04))
    elif change == "policy":
        owner._lateral_turn_response_policy = (999, False)
    result = runtime._linear_packet_write_limit(
        owner._depth30_linear_snapshot, 1, 62., forward_pair=(62, 62),
        feedback=feedback(clock[0], 20, 20), adopted_intent=adopted,
        adopted_policy=policy)
    assert not driver.pairs and not driver.stops  # Pure final decision only.
    if change == "neither":
        assert result is not None and result.forward_pair == (60, 60)
    else:
        assert result is None
        assert runtime._follow_write_veto_reason == "adopted_intent_or_policy_changed"


@pytest.mark.parametrize("change", ["intent", "policy", "neither"])
def test_complete_new_intent_handoff_retains_adoption_at_physical_write(monkeypatch, change):
    runtime, owner, driver, clock, _, _, _, _ = intent_handoff_writer(monkeypatch)
    # Unlike legacy fake owners, production always exposes the physical TTL.
    owner._depth_linear_max_age_sec = lambda kind: .25
    terminal = runtime._linear_packet_write_limit
    calls = []

    def enter_terminal(*args, **kwargs):
        calls.append(kwargs)
        if change == "intent":
            owner._lateral_intent_store.publish(
                ordinary_intent(cap=332, capture=10.02, published=clock[0]))
        elif change == "policy":
            owner._lateral_turn_response_policy = (999, False)
        return terminal(*args, **kwargs)

    runtime._linear_packet_write_limit = enter_terminal
    runtime._service_follow_wheels()
    assert calls and calls[0]["adopted_intent"].capture_frame_id == 331
    assert not driver.stops
    if change == "neither":
        assert driver.pairs == [(14, -28)]
    else:
        assert driver.pairs == [(0, 0)]
        assert runtime._follow_write_veto_reason == "adopted_intent_or_policy_changed"


@pytest.mark.parametrize("case", ["normal", "same", "none", "old_future", "new_expired",
                                 "new_capture_old", "old_mode", "old_hold", "new_reverse"])
def test_ordinary_handoff_requires_new_capture_and_no_braking_owner(case):
    old = replace(ordinary_intent(), sequence=1)
    new = replace(ordinary_intent(cap=331, capture=10., published=10.02), sequence=2)
    now = 10.05
    if case == "same": new = old
    if case == "none": old = None
    if case == "old_future": old = replace(old, published_at=10.03)
    if case == "new_expired": now = 10.26
    if case == "new_capture_old": new = replace(new, capture_timestamp=9.92)
    if case == "old_mode": old = replace(old, mode="yaw_only")
    if case == "old_hold": old = replace(old, hold_zero=True)
    if case == "new_reverse": new = replace(new, forward_countersteer=True)
    assert ordinary_intent_handoff(1, new, (old, old), now) is (case == "normal")
