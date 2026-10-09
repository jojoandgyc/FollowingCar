"""Distance-only reversing requires distinct physical samples, not ticks."""
from dataclasses import replace

import pytest

from car_control_modular.control_types import ControlAction, ControlDecision, DepthLinearTiming
from test_distance_only_controller_policy import distance_only
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


def prepared(setup):
    clock, controller, frame = distance_only(
        setup, near_distance_rotate_only_enable=False, reverse_enable=True,
        reverse_start_distance_m=1.2, reverse_immediate_distance_m=1.2,
        reverse_stop_distance_m=1.45,
        reverse_confirm_frames=2)
    current = frame(1.1)
    controller._observe_longitudinal_motion(current, current.persons[0])
    assert controller._reverse_control_decision(1, current, current.persons[0]) is None
    assert controller._reverse_approach_confirm_frames == 1
    return clock, controller, frame, current


@pytest.mark.parametrize("stamp_kind", ["same", "older", "future", "missing", "nan", "expired"])
def test_duplicate_or_invalid_depth_cannot_supply_second_reverse_confirmation(setup, stamp_kind):
    clock, controller, frame, first = prepared(setup)
    original = first.distance_state.sample_timestamp
    clock.now += .03
    stamp = {"same": original, "older": original-.01, "future": clock.now+.01,
             "missing": None, "nan": float("nan"), "expired": clock.now-.181}[stamp_kind]
    current = replace(first, distance_state=replace(first.distance_state, sample_timestamp=stamp))
    result = controller._reverse_control_decision(2, current, current.persons[0])
    assert result is None
    assert not controller._reverse_active
    assert controller._reverse_approach_confirm_frames == 1
    assert controller._reverse_approach_sample_watermark == original
    assert controller._reverse_last_approach_at == original
    # A genuinely new distance sample, even at the identical distance, works.
    clock.now += .03
    current = frame(1.1)
    result = controller._reverse_control_decision(3, current, current.persons[0])
    assert result is not None and result.reason == "target_approaching_reverse"
    assert controller._reverse_active


def test_reset_clears_sample_confirmation_without_issuing_a_command(setup):
    clock, controller, frame, first = prepared(setup)
    controller._reset_reverse_control("identity_changed")
    assert controller._reverse_approach_confirm_frames == 0
    assert controller._reverse_approach_sample_watermark is None
    clock.now += .03
    current = frame(1.1)
    assert controller._reverse_control_decision(2, current, current.persons[0]) is None
    assert controller._reverse_approach_confirm_frames == 1


def releasing(setup):
    clock, controller, frame, _ = prepared(setup)
    clock.now += .03
    current = frame(1.1)
    controller._reverse_control_decision(2, current, current.persons[0])
    assert controller._reverse_active
    clock.now += .03
    current = frame(1.46)
    controller._reverse_control_decision(3, current, current.persons[0])
    assert controller._reverse_active
    assert controller._reverse_release_confirm_frames == 1
    return clock, controller, frame, current


@pytest.mark.parametrize("stamp_kind", [
    "same", "older", "future", "missing", "nan", "expired", "bool", "held",
])
def test_reverse_release_requires_two_independent_fresh_samples(setup, stamp_kind):
    clock, controller, frame, first = releasing(setup)
    original = first.distance_state.sample_timestamp
    clock.now += .03
    stamp = {
        "same": original, "older": original-.01, "future": clock.now+.01,
        "missing": None, "nan": float("nan"), "expired": clock.now-.181,
        "bool": True, "held": clock.now,
    }[stamp_kind]
    state = replace(first.distance_state, sample_timestamp=stamp)
    if stamp_kind == "held":
        state = replace(state, source_detail="depth_multiregion_reused_hold")
    current = replace(first, distance_state=state)
    controller._reverse_control_decision(4, current, current.persons[0])
    assert controller._reverse_active
    assert controller._reverse_release_confirm_frames == 1
    assert controller._reverse_release_sample_watermark == original

    clock.now += .03
    current = frame(1.46)
    controller._reverse_control_decision(5, current, current.persons[0])
    assert not controller._reverse_active
    assert controller._reverse_release_confirm_frames == 0
    assert controller._reverse_release_sample_watermark is None


def test_reverse_release_near_sample_breaks_streak_without_allowing_old_far_replay(setup):
    clock, controller, frame, first = releasing(setup)
    clock.now += .03
    current = frame(1.1)
    controller._reverse_control_decision(4, current, current.persons[0])
    stamp = current.distance_state.sample_timestamp
    assert controller._reverse_release_confirm_frames == 0
    assert controller._reverse_release_sample_watermark == stamp
    clock.now += .03
    controller._reverse_control_decision(5, first, first.persons[0])
    assert controller._reverse_active
    assert controller._reverse_release_confirm_frames == 0
    assert controller._reverse_release_sample_watermark == stamp
    for number in (6, 7):
        clock.now += .03
        current = frame(1.46)
        controller._reverse_control_decision(number, current, current.persons[0])
    assert not controller._reverse_active


def test_old_near_sample_does_not_clear_new_release_evidence(setup):
    clock, controller, frame, first = releasing(setup)
    stamp = first.distance_state.sample_timestamp
    clock.now += .03
    old = frame(1.1, stamp=stamp-.01)
    controller._reverse_control_decision(4, old, old.persons[0])
    assert controller._reverse_release_confirm_frames == 1
    assert controller._reverse_release_sample_watermark == stamp


def test_reverse_release_identity_reset_clears_physical_watermark(setup):
    clock, controller, frame, _ = releasing(setup)
    clock.now += .03
    other = frame(1.46, uid=2)
    controller._reverse_control_decision(4, other, other.persons[0])
    assert not controller._reverse_active
    assert controller._reverse_release_confirm_frames == 0
    assert controller._reverse_release_sample_watermark is None


@pytest.mark.parametrize("offset", [0., -.01])
def test_runtime_duplicate_reverse_decision_never_renews_existing_depth_grant(setup, owner, offset):
    clock, controller, frame, _ = prepared(setup)
    owner._follow_controller = controller
    stamp = clock.now
    original = ("backward", 14, 1, stamp)
    timing = DepthLinearTiming(snapshot=original, accepted_depth_timestamp=stamp,
                               depth_expires_at=stamp+.18)
    owner._depth30_linear_snapshot = original
    owner._depth30_linear_timing = timing
    owner._depth30_linear_sample_watermark = (1, stamp)
    clock.now += .03
    current = frame(1.1, stamp=stamp+offset)
    actions, accepted = owner._commit_depth_linear_decision(
        ControlDecision(actions=[ControlAction.backward(20, "duplicate")]),
        current, 1, is_fresh_depth=True)
    assert actions == [] and not accepted
    assert owner._depth30_linear_snapshot is original
    assert owner._depth30_linear_timing is timing
    assert timing.depth_expires_at == stamp+.18
