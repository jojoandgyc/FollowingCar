"""Normal new visual intents may replace an unsent, same-grant wheel plan.

Every driver/clock here is fake. Identity, brake, grant and deadline changes
remain vetoes; only already guarded non-reversing wheel speeds may contract.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.lateral_intent import LateralControlIntent, LateralIntentStore
from test_cap331_command_snapshot import cap331_writer


def ordinary_intent(*, sign=-1, cap=329, capture=9.92, published=10., **changes):
    value = LateralControlIntent(
        sequence=0, target_id=1, frame_index=100, published_at=published,
        valid_until=published+.15, x_ratio=.5+sign*.15, motion_dx_ratio=0.,
        target_image_rate_dps=None, mode="forward", base_percent=26,
        base_rpm=26, initial_correction_rpm=sign*2, correction_limit_rpm=10.,
        confidence=.94, bbox_quality="reliable", reason="target_visible",
        capture_frame_id=cap, capture_timestamp=capture,
        decision_capture_frame_id=cap, image_error_only=True,
        response_boost_allowed=False, visual_error_deg=sign*9.)
    return replace(value, **changes)


def intent_handoff_writer(monkeypatch, *, sign=-1, new_yaw=7, new_base=26.,
                          stage="safety", new_changes=None, old_changes=None,
                          after_publish=None, final_update=None):
    store = LateralIntentStore()
    old = store.publish(ordinary_intent(sign=sign, **(old_changes or {})))
    published = []

    def update(runtime, owner, clock, state):
        state[0] = new_base
        changes = dict(new_changes or {})
        changed = ordinary_intent(sign=sign, cap=331, capture=10., published=clock[0])
        changed = replace(changed, initial_correction_rpm=sign*new_yaw, **changes)
        published.append(store.publish(changed))
        if after_publish:
            after_publish(runtime, owner, clock, state, store)

    runtime, owner, driver, clock, state, calls = cap331_writer(
        monkeypatch, sign=sign, new_yaw=new_yaw, second_update=update,
        final_update=final_update, second_stage=stage)
    owner._lateral_intent_store = store
    owner._follow_controller.cfg = SimpleNamespace(
        visible_steering_pid_image_error_only=True,
        visible_steering_pid_execution_response_trial_sec=.35,
        visible_steering_pid_max_correction_rpm=10.,
        near_distance_rotation_only_max_rpm=7.)
    read_feedback = runtime.get_steering_feedback
    def complete_feedback():
        result = read_feedback()
        result.yaw_rate_right_dps = 0.
        result.raw_yaw_rate_right_dps = 0.
        return result
    runtime.get_steering_feedback = complete_feedback
    return runtime, owner, driver, clock, state, calls, old, published


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("stage", ["safety", "feedback"])
def test_new_ordinary_intent_same_grant_keeps_a_guarded_forward_arc(monkeypatch, sign, stage):
    runtime, owner, driver, _, _, _, old, published = intent_handoff_writer(
        monkeypatch, sign=sign, stage=stage)
    runtime._service_follow_wheels()
    expected = (21+sign*7, 21-sign*7)
    assert len(published) == 1 and published[0] is not old
    assert driver.pairs == [(expected[0], -expected[1])]
    assert not driver.stops
    assert all(0 < new <= previous for new, previous in zip(expected, (26+sign*2, 26-sign*2)))
    assert runtime._follow_wheel_clock.last_axes == (1, owner._lateral_yaw_revision, 21., sign*7.)
    assert runtime._forward_execution_anchor.rpm == 21.


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("new_yaw,new_base,actual_base", [(2, 18., 18.), (2, 26., 26.), (1, 26., 25.)])
def test_new_intent_and_natural_contraction_never_raise_either_wheel(
        monkeypatch, sign, new_yaw, new_base, actual_base):
    runtime, owner, driver, _, _, _, _, _ = intent_handoff_writer(
        monkeypatch, sign=sign, new_yaw=new_yaw, new_base=new_base)
    runtime._service_follow_wheels()
    expected = (actual_base+sign*new_yaw, actual_base-sign*new_yaw)
    assert driver.pairs == [(expected[0], -expected[1])]
    assert all(0 < new <= previous for new, previous in zip(expected, (26+sign*2, 26-sign*2)))
    assert runtime._follow_wheel_clock.last_axes == (
        1, owner._lateral_yaw_revision, actual_base, sign*new_yaw)


@pytest.mark.parametrize("changes", [
    {"park_requested": True}, {"hold_zero": True}, {"near_distance_mode": True},
    {"forward_countersteer": True}, {"countersteer_rpm": 2},
    {"mode": "yaw_only"}, {"mode": "reverse"}, {"bbox_quality": "limited"},
    {"target_id": 2}, {"capture_frame_id": 329}, {"capture_timestamp": 9.92},
    {"capture_timestamp": 9.7}, {"capture_timestamp": 10.06},
    {"published_at": 10.06}, {"valid_until": 10.04},
    {"capture_timestamp": float("nan")}, {"published_at": float("nan")},
])
def test_new_intent_cannot_launder_braking_identity_or_capture_failure(monkeypatch, changes):
    runtime, _, driver, _, _, _, _, _ = intent_handoff_writer(monkeypatch, new_changes=changes)
    runtime._service_follow_wheels()
    assert all(pair == (0, 0) for pair in driver.pairs)


@pytest.mark.parametrize("old_changes", [
    {"park_requested": True}, {"hold_zero": True}, {"near_distance_mode": True},
    {"forward_countersteer": True}, {"countersteer_rpm": 2},
    {"mode": "yaw_only"}, {"bbox_quality": "limited"},
])
def test_braking_or_unknown_old_intent_requires_full_rebuild(monkeypatch, old_changes):
    runtime, _, driver, _, _, _, _, _ = intent_handoff_writer(monkeypatch, old_changes=old_changes)
    runtime._service_follow_wheels()
    assert all(pair == (0, 0) for pair in driver.pairs)


@pytest.mark.parametrize("change", ["grant", "uid", "identity", "base_increase"])
def test_new_intent_does_not_expand_forward_authority(monkeypatch, change):
    def update(_runtime, owner, _clock, state, _store):
        if change == "grant":
            owner._depth30_linear_snapshot = ("forward", 26., 1, 10.01)
        elif change == "uid":
            owner._follow_controller.active_target_id = 2
        elif change == "identity":
            owner._vision_control_state = "identity_recheck"
        else:
            state[0] = 30.
    runtime, _, driver, _, _, _, _, _ = intent_handoff_writer(monkeypatch, after_publish=update)
    runtime._service_follow_wheels()
    assert all(pair == (0, 0) for pair in driver.pairs)


@pytest.mark.parametrize("change", ["third_intent", "stop", "policy", "expiry"])
def test_adopted_new_intent_must_stay_stable_through_final_checks(monkeypatch, change):
    def final(runtime, owner, clock, state):
        if change == "third_intent":
            owner._lateral_intent_store.publish(ordinary_intent(cap=332, capture=10.02, published=clock[0]))
        elif change == "stop":
            runtime.backend.send_stop("new_intent_concurrent_stop", mode="emergency", preserve_zero=True)
            owner._brake_hold_active = True
        elif change == "policy":
            owner._lateral_turn_response_policy = (999, False)
        else:
            clock[0] = 10.26
        return False
    runtime, _, driver, _, _, calls, _, _ = intent_handoff_writer(monkeypatch, final_update=final)
    runtime._service_follow_wheels()
    assert calls["safety"] >= 4
    assert all(pair == (0, 0) for pair in driver.pairs)
    if change == "stop":
        assert driver.stops == [1]
        assert not driver.pairs
