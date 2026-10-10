"""CAP1040: qualified crop yaw must not compete with FOLLOW20 zeros.

Real axes reader, executor and MSSD packing; fake serial only.
"""
from dataclasses import replace
from types import MethodType

import pytest
import request_0513_modular as app
from car_control_modular.detector_identity_lease import publish_visual_identity_evidence
from car_control_modular.lateral_intent import LateralControlIntent, LateralIntentStore
from car_control_modular.low_quality_lateral import (
    LimitedYawSource, LimitedYawEvidence, limited_yaw_identity_live,
)
from car_control_modular.short_follow import ShortFollowConfig, ShortFollowController
from test_visible_wheel_continuity import visible_runtime, feedback


def setup(monkeypatch):
    rt, owner, driver, symbols, clock = visible_runtime(monkeypatch)
    rt.config.follow_wheel_period_sec = .05
    rt.config.follow_forward_loss_handoff_enable = True
    rt.config.rotate_prep_coast_enable = False
    owner._vision_control_state = "target_visible_low_quality"
    owner._explicit_stop_requested = owner._runtime_shutdown_requested = False
    owner._last_command_capture_frame = 1040
    owner._fresh_depth_linear_snapshot = lambda *a, **kw: None
    owner._lateral_intent_store = LateralIntentStore()
    owner._follow_wheel_axes = MethodType(app.PersonTracker._follow_wheel_axes, owner)
    owner._has_fresh_lateral_yaw = MethodType(app.PersonTracker._has_fresh_lateral_yaw, owner)
    owner._lateral_intent_last_sequence = -1
    owner._lateral_intent_last_correction_rpm = -7
    owner._action_runtime = rt
    publication = publish_visual_identity_evidence(owner, observation=False, lease=False)
    source = LimitedYawSource(1, 16, 1040, clock[0]-.1, (0., 0., 198., 479.), publication)
    intent = owner._lateral_intent_store.publish(LateralControlIntent(
        sequence=0, target_id=1, frame_index=454, published_at=clock[0],
        valid_until=clock[0]+.15, x_ratio=.155, motion_dx_ratio=0.,
        target_image_rate_dps=None, mode="yaw_only", base_percent=0, base_rpm=0,
        initial_correction_rpm=-7, correction_limit_rpm=7., confidence=.885,
        bbox_quality="limited", reason="target_visible_low_quality_yaw",
        capture_frame_id=1040, capture_timestamp=source.timestamp,
        decision_capture_frame_id=1040))
    owner._limited_yaw_evidence = LimitedYawEvidence(source, intent, intent.valid_until)
    rt.get_steering_feedback = lambda: feedback(clock[0], -4, 4)
    return rt, owner, driver, symbols, clock, intent


def test_crop_periodic_and_stale_queue_share_one_yaw_pair(monkeypatch):
    rt, owner, driver, symbols, clock, _ = setup(monkeypatch)
    for index in range(3):
        clock[0] = 10.+.05*index
        # An obsolete straight/turn queue may not replace the new axes.
        rt.send_robot_command(symbols.forward if index % 2 else symbols.rotate_right)
        rt._service_follow_wheels()
    assert driver.pairs == [(-7, -7)] * 3
    assert not driver.stops
    assert owner._follow_wheel_axes(clock[0])[2] == 0


def test_old_soft_zero_cannot_replace_published_crop_yaw(monkeypatch):
    rt, owner, driver, symbols, clock, _ = setup(monkeypatch)
    rt._service_follow_wheels()
    clock[0] += .05
    owner._use_soft_stop_next = True
    rt.send_robot_command(symbols.stop)
    rt._service_follow_wheels()
    assert driver.pairs == [(-7, -7), (-7, -7)]
    assert not driver.stops


def test_crop_does_not_inherit_previous_forward_base(monkeypatch):
    rt, owner, driver, _, clock, _ = setup(monkeypatch)
    owner._fresh_depth_linear_snapshot = lambda *a, **kw: ("forward", 80, 1, 9.95)
    assert owner._follow_wheel_axes(clock[0])[2:] == (0., -7.)
    rt._service_follow_wheels()
    assert driver.pairs == [(-7, -7)]


@pytest.mark.parametrize("change", ["new_reject", "expiry", "uid", "search", "park", "stop"])
def test_limited_yaw_does_not_survive_new_rejection_or_expiry(monkeypatch, change):
    rt, owner, driver, _, clock, _ = setup(monkeypatch)
    rt._service_follow_wheels()
    clock[0] += .06
    if change == "new_reject":
        publish_visual_identity_evidence(owner, observation=False, lease=False)
    elif change == "expiry":
        clock[0] = 10.151
    elif change == "uid":
        owner._follow_controller.active_target_id = 2
    elif change == "search":
        owner.search_state = "searching"
    elif change == "park":
        owner._brake_hold_active = True
    else:
        owner._explicit_stop_requested = True
    assert not limited_yaw_identity_live(owner, 1, clock[0])
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (0, 0)


def test_expiry_during_feedback_read_cannot_write_old_yaw(monkeypatch):
    rt, _, driver, _, clock, _ = setup(monkeypatch)
    def delayed():
        clock[0] = 10.151
        return feedback(clock[0], -4, 4)
    rt.get_steering_feedback = delayed
    rt._service_follow_wheels()
    assert not any(pair != (0, 0) for pair in driver.pairs)


def test_paired_owner_transfers_only_after_successful_yaw_write(monkeypatch):
    rt, owner, driver, _, _, _ = setup(monkeypatch)
    owner._short_follow = ShortFollowController(ShortFollowConfig(enabled=True))
    owner._short_follow.activate(1, 9.9)
    owner._short_follow.deactivate("identity_or_search_handoff", 10.)
    executor = rt._short_follow_executor_instance()
    executor._owned = True
    executor._controller = owner._short_follow
    assert rt._service_short_follow()
    assert driver.pairs == [(-7, -7)]
    assert not driver.stops
    assert not executor._owned
    rt._service_follow_wheels()
    assert driver.pairs == [(-7, -7)]  # not a second periodic zero


@pytest.mark.parametrize("failure", ["expired", "feedback", "identity", "hazard"])
def test_failed_yaw_successor_keeps_required_stop(monkeypatch, failure):
    rt, owner, driver, _, clock, _ = setup(monkeypatch)
    owner._short_follow = ShortFollowController(ShortFollowConfig(enabled=True))
    owner._short_follow.activate(1, 9.9)
    owner._short_follow.deactivate("identity_or_search_handoff", 10.)
    executor = rt._short_follow_executor_instance()
    executor._owned = True
    executor._controller = owner._short_follow
    if failure == "expired":
        clock[0] = 10.151
    elif failure == "feedback":
        rt.get_steering_feedback = lambda: feedback(9., -4, 4)
    elif failure == "identity":
        publish_visual_identity_evidence(owner, observation=False, lease=False)
    else:
        rt.hard_stop_check = lambda _: True
    rt._service_short_follow()
    assert driver.stops
    assert not any(pair != (0, 0) for pair in driver.pairs)


@pytest.mark.parametrize("intent_available", [False, True])
def test_legacy_owner_without_axes_interface_exits_with_required_stop(monkeypatch, intent_available):
    rt, owner, driver, _, _, _ = setup(monkeypatch)
    owner._short_follow = ShortFollowController(ShortFollowConfig(enabled=True))
    owner._short_follow.activate(1, 9.9)
    owner._short_follow.deactivate("identity_or_search_handoff", 10.)
    executor = rt._short_follow_executor_instance()
    executor._owned = True
    executor._controller = owner._short_follow
    del owner._follow_wheel_axes
    if not intent_available:
        owner._lateral_intent_store = LateralIntentStore()
    assert rt._service_short_follow()
    assert driver.stops
    assert not driver.pairs
    assert not executor._owned


def test_no_successor_does_not_compute_unrelated_axes(monkeypatch):
    rt, owner, _, _, _, _ = setup(monkeypatch)
    owner._lateral_intent_store = LateralIntentStore()
    owner._follow_wheel_axes = lambda _now: pytest.fail("no yaw successor may read axes")
    assert not rt._write_limited_yaw_successor()


@pytest.mark.parametrize("field,value", [("mode", "forward"), ("bbox_quality", "reliable"),
    ("capture_frame_id", 1047), ("capture_timestamp", 9.8), ("target_id", 2)])
def test_replaced_or_mismatched_intent_cannot_borrow_crop_permission(monkeypatch, field, value):
    _, owner, _, _, clock, intent = setup(monkeypatch)
    owner._lateral_intent_store.publish(replace(intent, **{field: value}))
    assert not owner._has_fresh_lateral_yaw(1)
    assert owner._follow_wheel_axes(clock[0])[2:] == (0., 0.)
