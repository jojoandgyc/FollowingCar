"""One physical STOP receipt fulfils one entry/exit barrier, not each label.

Uses the real paired writer and fake motor backend; never opens hardware.
"""
from dataclasses import replace

import pytest

from test_short_follow_executor import publish, set_feedback, short_runtime


def awaiting_entry(monkeypatch):
    rt, owner, driver, symbols, clock = short_runtime(monkeypatch)
    owner._short_follow.revoke("awaiting_observation", clock[0])
    set_feedback(rt, clock, 4, 0)  # Unknown moving takeover still needs a STOP.
    return rt, owner, driver, symbols, clock


def test_awaiting_depth_stop_ack_counts_entry_quiet_before_first_plan(monkeypatch):
    rt, owner, driver, _, clock = awaiting_entry(monkeypatch)
    rt._service_short_follow()
    writer = rt._short_follow_executor
    assert driver.stops == [1] and not driver.pairs
    assert writer._entry_stop_acknowledged
    first_ack = writer._entry_stop_at
    for now in (10.05, 10.10):
        clock[0] = now
        set_feedback(rt, clock, 0, 0)
        rt._service_short_follow()
        assert not driver.pairs  # Quiet alone never creates authority.
    assert writer._entry_stop_at is None
    clock[0] = 10.12
    plan = publish(rt, clock, 2)
    rt._service_short_follow()
    assert driver.pairs == [(plan.left_rpm, -plan.right_rpm)]
    assert driver.stops == [1]  # No new entry STOP after completed quiet proof.
    assert first_ack == 10.


def test_failed_awaiting_stop_cannot_create_receipt_or_quiet_credit(monkeypatch):
    rt, owner, driver, _, clock = awaiting_entry(monkeypatch)
    send_stop = rt.backend.send_stop

    def fail(*_args, **_kwargs):
        raise OSError("both STOP acknowledgements unavailable")

    monkeypatch.setattr(rt.backend, "send_stop", fail)
    with pytest.raises(OSError, match="STOP acknowledgements"):
        rt._service_short_follow()
    writer = rt._short_follow_executor
    assert not writer._entry_stop_acknowledged
    assert writer._entry_quiet_count == 0
    assert not driver.stops and not driver.pairs
    clock[0] = 10.1
    set_feedback(rt, clock, 0, 0)
    publish(rt, clock, 2)
    monkeypatch.setattr(rt.backend, "send_stop", send_stop)
    rt._service_short_follow()
    assert driver.stops == [1] and not driver.pairs
    assert writer._entry_stop_at == 10.1
    for now in (10.15, 10.2):
        clock[0] = now
        set_feedback(rt, clock, 0, 0)
        rt._service_short_follow()
    assert len(driver.pairs) == 1


def test_waiting_entry_repeated_or_older_quiet_sample_is_not_confirmation(monkeypatch):
    rt, owner, driver, _, clock = awaiting_entry(monkeypatch)
    rt._service_short_follow()
    writer = rt._short_follow_executor
    clock[0] = 10.05
    set_feedback(rt, clock, 0, 0)
    rt._service_short_follow()
    rt._service_short_follow()
    assert writer._entry_quiet_count == 1
    set_feedback(rt, clock, 0, 0, stamp=10.02)
    rt._service_short_follow()
    assert writer._entry_quiet_count == 1
    clock[0] = 10.08
    publish(rt, clock, 2)
    rt._service_short_follow()
    assert not driver.pairs
    assert writer._entry_stop_at == 10.
    clock[0] = 10.1
    set_feedback(rt, clock, 0, 0)
    rt._service_short_follow()
    assert len(driver.pairs) == 1


@pytest.mark.parametrize("adverse", ["hazard", "explicit", "motor_fault",
                                    "feedback_error", "large_reverse", "external_stop"])
def test_completed_waiting_entry_cannot_bypass_new_danger(monkeypatch, adverse):
    rt, owner, driver, _, clock = awaiting_entry(monkeypatch)
    rt._service_short_follow()
    for now in (10.05, 10.10):
        clock[0] = now
        set_feedback(rt, clock, 0, 0)
        rt._service_short_follow()
    assert rt._short_follow_executor._entry_stop_at is None
    clock[0] = 10.12
    publish(rt, clock, 2)
    if adverse == "hazard":
        rt.hard_stop_check = lambda _action: True
    elif adverse == "explicit":
        owner._explicit_stop_requested = True
        owner._last_explicit_stop_reason = "manual_emergency"
        owner._explicit_stop_provenance = ("operator", "manual_emergency")
    elif adverse == "motor_fault":
        rt.backend.motion_write_fault = "partial_write_unknown"
    elif adverse == "feedback_error":
        set_feedback(rt, clock, 0, 0).left_error = 1
    elif adverse == "large_reverse":
        set_feedback(rt, clock, -20, -20)
    else:
        rt.backend.send_stop("external_emergency", mode="emergency")
    stops_before = len(driver.stops)
    rt._service_short_follow()
    assert not driver.pairs
    assert owner._short_follow.snapshot().plan is None
    assert len(driver.stops) > stops_before


def identity_stopped_owner(monkeypatch):
    rt, owner, driver, symbols, clock = short_runtime(monkeypatch)
    rt._service_short_follow()
    clock[0] += .01
    owner._validated_visual_observation = False
    rt._service_short_follow()
    assert driver.stops == [1]
    assert rt.backend.last_speed_receipt is None
    assert rt._short_follow_executor._owned
    owner._short_follow.deactivate("identity_or_search_handoff", clock[0])
    return rt, owner, driver, symbols, clock


def test_identity_stop_receipt_already_fulfils_immediate_ownership_exit(monkeypatch):
    rt, owner, driver, _, clock = identity_stopped_owner(monkeypatch)
    generation = rt.backend.stop_write_generation
    clock[0] += .01
    assert rt._service_short_follow()
    assert driver.stops == [1]
    assert rt.backend.stop_write_generation == generation
    assert not rt._short_follow_executor._owned
    assert not rt._short_follow_blocks_legacy()
    assert not owner.is_forwarding


@pytest.mark.parametrize("interposed", ["zero_speed", "nonzero_speed", "external_stop", "fault", "partial"])
def test_exit_cannot_reuse_stop_across_intervening_io_or_fault(monkeypatch, interposed):
    rt, owner, driver, _, clock = identity_stopped_owner(monkeypatch)
    clock[0] += .01
    if interposed == "zero_speed":
        # Zero RPM switches to speed mode: it is NOT the prior STOP receipt.
        rt.backend.send_targets(0, 0, "interposed_speed_zero", history_uid=1)
    elif interposed == "nonzero_speed":
        rt.backend.send_targets(4, 4, "interposed_turn", history_uid=1)
    elif interposed == "external_stop":
        rt.backend.send_stop("external_emergency", mode="emergency")
    elif interposed == "fault":
        rt.backend.motion_write_fault = "unknown_external_write"
    else:
        original = driver.set_left_speed
        calls = []

        def fail_first(value):
            calls.append(value)
            if len(calls) == 1:
                raise OSError("left ACK failed after right wheel ACK")
            return original(value)

        monkeypatch.setattr(driver, "set_left_speed", fail_first)
        with pytest.raises(Exception):
            rt.backend.send_targets(4, 4, "partial_turn", history_uid=1)
        assert rt.backend.motion_write_fault
    stops_before = len(driver.stops)
    assert rt._service_short_follow()
    assert len(driver.stops) > stops_before
    assert not rt._short_follow_executor._owned


def test_first_exit_from_moving_owner_still_writes_stop(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt._service_short_follow()
    assert not driver.stops and driver.pairs
    clock[0] += .01
    owner._short_follow.deactivate("identity_or_search_handoff", clock[0])
    assert rt._service_short_follow()
    assert driver.stops == [1]
    assert rt.backend.last_speed_receipt is None
    assert not rt._short_follow_executor._owned


@pytest.mark.parametrize("change", ["mode_disabled", "controller_removed"])
def test_runtime_mode_removal_stops_and_retires_old_active_mailbox(monkeypatch, change):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    controller = owner._short_follow
    original_plan = controller.snapshot().plan
    assert original_plan is not None
    rt._service_short_follow()
    assert len(driver.pairs) == 1 and not driver.stops

    clock[0] += .01
    if change == "mode_disabled":
        controller.config = replace(controller.config, enabled=False)
    else:
        owner._short_follow = None
    assert rt._service_short_follow()
    assert driver.stops == [1]
    assert controller.snapshot().plan is None
    assert not controller.snapshot().active
    assert not rt._short_follow_executor._owned
    assert not rt._short_follow_blocks_legacy()
    assert rt.backend.last_speed_receipt is None
    assert not owner.is_forwarding

    # Restore the same object while the old plan would still have been fresh.
    # Re-enabling a mode is not a new observation or a new motion authority.
    clock[0] += .01
    assert original_plan.valid(clock[0])
    controller.config = replace(controller.config, enabled=True)
    owner._short_follow = controller
    assert not rt._service_short_follow()
    assert len(driver.pairs) == 1 and driver.stops == [1]
    assert controller.snapshot().plan is None
    assert not rt._short_follow_blocks_legacy()
