"""Confirmed low-speed search handoff, real writer with a fake serial driver."""
from dataclasses import replace

import pytest
import request_0513_modular as main

from car_control_modular.controllers import FollowPolicyConfig
from car_control_modular.search_reacquire_braking import paired_search_takeover_available
from test_short_follow_executor import short_runtime, publish, set_feedback


def scene(monkeypatch, direction=-1):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    owner._action_runtime = rt
    owner._short_follow_adapter = object()
    owner._short_follow.deactivate("search", clock[0])
    owner.search_state = owner._follow_controller.search_state = "searching"
    owner.search_direction = owner._follow_controller.search_direction = "left" if direction < 0 else "right"
    owner._follow_controller.cfg = FollowPolicyConfig()
    owner._active_capture_frame_id = 1457
    owner._active_capture_timestamp = 10.
    owner.requests, owner.clears = [], []
    owner._clear_lateral_intent = lambda reason: owner.clears.append(reason)
    owner._clear_longitudinal_context = lambda **kw: owner.clears.append(kw)
    rt.search_reacquire_brake_pending = lambda: False
    rt.request_search_reacquire_brake = lambda *a: owner.requests.append(a)
    rt.backend.send_targets(7*direction, 7*direction, "TURN", history_uid=1)
    clock[0] += .05
    fb = set_feedback(rt, clock, 4*direction, -6*direction)
    fb.left_error = fb.right_error = 0
    fb.raw_yaw_rate_right_dps = fb.yaw_rate_right_dps = 15*direction
    # CAP1457: target has crossed to the opposite side of the old search.
    bbox = (350, 4, 487, 473) if direction < 0 else (153, 4, 290, 473)
    return rt, owner, driver, clock, fb, bbox


def gate(owner, bbox, **kw):
    return main.PersonTracker._hold_search_reacquire_brake(owner,
        bbox=bbox, width=640, eligible=kw.get("eligible", True),
        confirmed=kw.get("confirmed", True), raw_track_id=33)


@pytest.mark.parametrize("direction", [-1, 1])
def test_confirmed_crossing_uses_paired_writer_without_new_stop(monkeypatch, direction):
    rt, owner, driver, clock, fb, bbox = scene(monkeypatch, direction)
    assert not gate(owner, bbox)
    assert not owner.requests and not owner.clears
    assert len(driver.pairs) == 1 and not driver.stops  # Gate never sends motion.
    assert owner._short_follow.snapshot().plan is None
    owner.search_state = owner._follow_controller.search_state = "none"
    owner._short_follow.activate(1, clock[0])
    clock[0] += .01
    set_feedback(rt, clock, fb.left_forward_rpm, fb.right_forward_rpm)
    plan = publish(rt, clock, 1459, distance=2.60, x=.8 if direction < 0 else .2)
    assert rt._service_short_follow()
    assert driver.pairs[-1] == (plan.left_rpm, -plan.right_rpm)
    assert min(plan.left_rpm, plan.right_rpm) > 0
    assert not driver.stops and (0, 0) not in driver.pairs
    # Source deadlines remain finite despite a successful no-park handoff.
    clock[0] = plan.expires_at + .001
    set_feedback(rt, clock, 10, 10)
    rt._service_short_follow()
    assert driver.stops


@pytest.mark.parametrize("change", ["unconfirmed", "ineligible", "disabled", "no_adapter",
    "old_receipt", "no_receipt", "in_flight", "stop_generation", "motor_fault",
    "parking_fault", "parking_current", "explicit", "shutdown", "existing_brake",
    "stale_feedback", "untrusted", "error", "both_reverse", "wrong_reverse", "fast_feedback",
    "fast_pivot", "forward_receipt", "no_uid"])
def test_unknown_or_protected_motion_cannot_skip_parking(monkeypatch, change):
    rt, owner, driver, clock, fb, bbox = scene(monkeypatch)
    kw = {}
    if change == "unconfirmed": kw["confirmed"] = False
    elif change == "ineligible": kw["eligible"] = False
    elif change == "disabled": owner._short_follow.config = replace(owner._short_follow.config, enabled=False)
    elif change == "no_adapter": owner._short_follow_adapter = None
    elif change == "old_receipt": clock[0] += .11
    elif change == "no_receipt": rt.backend.last_speed_receipt = None
    elif change == "in_flight": rt.backend.last_speed_write = None
    elif change == "stop_generation": rt.backend.stop_write_generation += 1
    elif change == "motor_fault": rt.backend.motion_write_fault = "partial"
    elif change == "parking_fault": rt.backend.parking_release_fault = "uncertain"
    elif change == "parking_current": rt.backend.parking_current_a = 1
    elif change == "explicit": owner._explicit_stop_requested = True
    elif change == "shutdown": owner._runtime_shutdown_requested = True
    elif change == "existing_brake": owner._brake_hold_active = True
    elif change == "stale_feedback": fb.timestamp -= .11
    elif change == "untrusted": fb.trustworthy = False
    elif change == "error": fb.left_error = 1
    elif change == "both_reverse": fb.left_forward_rpm, fb.right_forward_rpm = -4, -4
    elif change == "wrong_reverse": fb.left_forward_rpm, fb.right_forward_rpm = 4, -6
    elif change == "fast_feedback": fb.left_forward_rpm, fb.right_forward_rpm = -11, 11
    elif change == "fast_pivot": rt.backend.send_targets(-20, -20, "TURN", history_uid=1)
    elif change == "forward_receipt": rt.backend.send_targets(10, -10, "DRIVE", history_uid=1)
    elif change == "no_uid": owner._follow_controller.active_target_id = None
    if change not in {"unconfirmed", "ineligible"}:
        assert not paired_search_takeover_available(owner, fb, clock[0])
    gate(owner, bbox, **kw)
    assert getattr(owner, "_search_candidate_brake_episode", None) is None or owner.requests
    assert not driver.stops  # Gate enqueues; it never writes the motor itself.


def test_pending_stop_not_replaced_by_new_takeover(monkeypatch):
    rt, owner, driver, clock, fb, bbox = scene(monkeypatch)
    rt.search_reacquire_brake_pending = lambda: True
    assert gate(owner, bbox)
    assert not owner.requests
    assert len(driver.pairs) == 1 and not driver.stops
