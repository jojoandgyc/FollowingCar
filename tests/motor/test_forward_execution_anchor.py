"""Completed forward-write evidence: fake clock/driver only, never hardware."""

from types import SimpleNamespace

import pytest

from test_follow_wheel_periodic import setup_periodic
from test_visible_wheel_continuity import visible_runtime


def execution_runtime(monkeypatch, *, periodic=False, sample_timestamp=9.84):
    if periodic:
        runtime, owner, driver, symbols, clock, state = setup_periodic(monkeypatch)
        state[:] = [40., 10., 11., 11.]
    else:
        runtime, owner, driver, symbols, clock = visible_runtime(monkeypatch)
        state = None
    runtime.config.motor_forward_max_target_rpm = 200
    owner._depth30_linear_snapshot = ("forward", 20, 1, sample_timestamp)
    owner._depth30_linear_timing = SimpleNamespace(
        accepted_depth_timestamp=sample_timestamp,
        depth_expires_at=sample_timestamp + .25,
    )
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: (
        owner._depth30_linear_snapshot
        if uid == 1 and 0 <= (clock[0] if now is None else now) - sample_timestamp <= .25
        else None
    )
    return runtime, owner, driver, symbols, clock, state


def write_forward(runtime, owner, *, periodic=False):
    if periodic:
        runtime._service_follow_wheels()
    else:
        with owner.motor_io_lock:
            runtime._send_follow_wheel_targets(50, -30, "STEER", visible_required=True)


@pytest.mark.parametrize("periodic", [False, True])
def test_completed_pair_binds_exact_uid_physical_sample_and_forward_rpm(monkeypatch, periodic):
    runtime, owner, driver, _, clock, _ = execution_runtime(monkeypatch, periodic=periodic)
    write_forward(runtime, owner, periodic=periodic)

    assert driver.pairs == [(50, -30)]
    anchor = runtime.forward_execution_anchor(1, 9.84, clock[0])
    assert anchor is runtime._forward_execution_anchor
    assert (anchor.uid, anchor.sample_timestamp, anchor.rpm) == (1, 9.84, 40)
    assert anchor.sent_at == clock[0]
    assert anchor.receipt is runtime.backend.last_speed_receipt
    assert (anchor.receipt.left_rpm, anchor.receipt.right_rpm) == (50, -30)
    assert runtime.forward_execution_anchor(2, 9.84, clock[0]) is None
    assert runtime.forward_execution_anchor(1, 9.85, clock[0]) is None


def test_anchor_clock_is_dual_write_completion_not_packet_preparation(monkeypatch):
    runtime, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    original = driver.set_left_speed

    def finish_left(value):
        clock[0] += .005
        original(value)

    monkeypatch.setattr(driver, "set_left_speed", finish_left)
    write_forward(runtime, owner)
    anchor = runtime.forward_execution_anchor(1, 9.84, clock[0])
    assert anchor.sent_at == pytest.approx(10.005)
    assert anchor.sent_at == anchor.receipt.completed_at


def test_continuation_veto_does_not_erase_completed_io_or_renew_old_grant(monkeypatch):
    runtime, owner, driver, _, clock, state = execution_runtime(monkeypatch, periodic=True)
    write_forward(runtime, owner, periodic=True)
    anchor = runtime._forward_execution_anchor
    snapshot, timing = owner._depth30_linear_snapshot, owner._depth30_linear_timing
    receipt = runtime.backend.last_speed_receipt
    writes = (list(driver.pairs), list(driver.stops), list(driver.register_writes))
    clock[0] = 10.03  # Physical sample is now in the 180--250ms continuation band.

    # A read-time continuation veto returns no usable motor grant, but does
    # not claim a zero packet has reached the driver or clear the source tuple.
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
    state[0] = 0.
    assert owner._fresh_depth_linear_snapshot(1, now=clock[0]) is None
    for now in (10.03, 10.04, 10.05):
        clock[0] = now
        assert runtime.forward_execution_anchor(1, 9.84, now) is anchor
        assert owner._fresh_depth_linear_snapshot(1, now=now) is None
    assert owner._depth30_linear_snapshot is snapshot
    assert owner._depth30_linear_timing is timing
    assert timing.depth_expires_at == pytest.approx(10.09)
    assert runtime.backend.last_speed_receipt is receipt
    assert anchor.sent_at == 10.
    assert (driver.pairs, driver.stops, driver.register_writes) == writes

    # The executor still consumes the revoked axes and sends zero for the
    # requested wheel reversal; reading evidence never restarts old forward.
    runtime._service_follow_wheels()
    assert driver.pairs[-1] == (0, 0)
    assert runtime.forward_execution_anchor(1, 9.84, clock[0]) is None


@pytest.mark.parametrize("pair", [(0, 0), (50, -30), (30, -10)])
def test_any_other_speed_writer_invalidates_receipt_even_for_identical_pair(monkeypatch, pair):
    runtime, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    write_forward(runtime, owner)
    anchor = runtime._forward_execution_anchor
    with owner.motor_io_lock:
        runtime.backend.send_targets(*pair, "OTHER_WRITER")
    assert driver.pairs == [(50, -30), pair]
    assert runtime.backend.last_speed_receipt is not anchor.receipt
    assert runtime.forward_execution_anchor(1, 9.84, clock[0]) is None


def test_follow_zero_clears_cached_forward_anchor(monkeypatch):
    runtime, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    write_forward(runtime, owner)
    with owner.motor_io_lock:
        runtime._send_follow_wheel_targets(0, 0, "FOLLOW_ZERO", visible_required=True)
    assert driver.pairs == [(50, -30), (0, 0)]
    assert runtime._forward_execution_anchor is None
    assert runtime.forward_execution_anchor(1, 9.84, clock[0]) is None


@pytest.mark.parametrize("mode", ["normal", "emergency", "free"])
def test_stop_invalidates_anchor(monkeypatch, mode):
    runtime, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    write_forward(runtime, owner)
    with owner.motor_io_lock:
        runtime.backend.send_stop("TEST_STOP", mode=mode)
    assert driver.stops
    assert runtime.backend.last_speed_receipt is None
    assert runtime.forward_execution_anchor(1, 9.84, clock[0]) is None


def test_partial_pair_failure_cannot_leave_previous_execution_anchor_usable(monkeypatch):
    runtime, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    write_forward(runtime, owner)
    attempts = []

    def fail_left(value):
        attempts.append((value, driver.right))
        raise RuntimeError("fake left acknowledgement failure")

    monkeypatch.setattr(driver, "set_left_speed", fail_left)
    with owner.motor_io_lock, pytest.raises(RuntimeError, match="fake left acknowledgement"):
        runtime._send_follow_wheel_targets(45, -35, "STEER", visible_required=True)
    assert attempts[0] == (45, -35)
    assert all(pair == (0, 0) for pair in attempts[1:])  # Fault zero attempt.
    assert driver.pairs == [(50, -30)]
    assert driver.stops == [1]
    assert runtime.backend.motion_write_fault
    assert runtime.backend.last_speed_receipt is None
    assert runtime.forward_execution_anchor(1, 9.84, clock[0]) is None


@pytest.mark.parametrize("transition", ["parking_current", "release_current"])
def test_current_mode_transition_invalidates_anchor_without_a_new_speed_pair(monkeypatch, transition):
    runtime, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    write_forward(runtime, owner)
    with owner.motor_io_lock:
        if transition == "parking_current":
            runtime.backend.set_parking_current(1., persist=False)
        else:
            runtime.backend.release_parking_current_only()
    assert driver.pairs == [(50, -30)]
    assert runtime.backend.last_speed_receipt is None
    assert runtime.forward_execution_anchor(1, 9.84, clock[0]) is None


@pytest.mark.parametrize("change", [
    "uid", "snapshot_cleared", "snapshot_zero", "snapshot_replaced", "snapshot_uid",
    "search", "controller_search", "low_quality", "missing_depth", "explicit_stop",
    "shutdown", "brake_hold", "near_yaw_park", "execution_stopped", "person_stop",
    "parking_fault", "not_running",
])
def test_identity_explicit_revoke_and_nonforward_states_reject_cached_anchor(monkeypatch, change):
    runtime, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    write_forward(runtime, owner)
    if change == "uid":
        owner._follow_controller.active_target_id = 2
    elif change == "snapshot_cleared":
        owner._depth30_linear_snapshot = None
    elif change == "snapshot_zero":
        owner._depth30_linear_snapshot = ("forward", 0, 1, 9.84)
    elif change == "snapshot_replaced":
        owner._depth30_linear_snapshot = ("forward", 20, 1, 9.85)
    elif change == "snapshot_uid":
        owner._depth30_linear_snapshot = ("forward", 20, 2, 9.84)
    elif change == "search":
        owner.search_state = "searching"
    elif change == "controller_search":
        owner._follow_controller.search_state = "searching"
    elif change == "low_quality":
        runtime.config.follow_forward_loss_handoff_enable = True
        owner._vision_control_state = "target_visible_low_quality"
    elif change == "missing_depth":
        owner._vision_control_state = "target_visible_depth_missing"
    elif change == "parking_fault":
        runtime.backend.parking_release_fault = "fake current readback mismatch"
    else:
        attribute = {
            "explicit_stop": "_explicit_stop_requested",
            "shutdown": "_runtime_shutdown_requested",
            "brake_hold": "_brake_hold_active",
            "near_yaw_park": "_near_yaw_park_request",
            "execution_stopped": "stop_action_execution",
            "person_stop": "person_detected_flag",
            "not_running": "running",
        }[change]
        setattr(owner, attribute, change != "not_running")
    assert runtime.forward_execution_anchor(1, 9.84, clock[0]) is None
    assert driver.pairs == [(50, -30)]
    assert not driver.stops


@pytest.mark.parametrize("sample_timestamp,now", [
    (9.95, 10.100001),  # Receipt age over 100ms; physical sample remains fresh.
    (9.84, 10.090001),  # Physical sample age over 250ms; receipt remains fresh.
    (9.95, 9.999),     # Receipt cannot come from the future.
    (9.95, float("nan")),
    (9.95, float("inf")),
])
def test_receipt_and_physical_sample_have_independent_expiry(monkeypatch, sample_timestamp, now):
    runtime, owner, driver, _, _, _ = execution_runtime(monkeypatch, sample_timestamp=sample_timestamp)
    write_forward(runtime, owner)
    assert runtime.forward_execution_anchor(1, sample_timestamp, 10.) is not None
    assert runtime.forward_execution_anchor(1, sample_timestamp, now) is None
    assert driver.pairs == [(50, -30)]


def test_two_field_depth_reader_cannot_fabricate_sample_bound_execution_anchor(monkeypatch):
    runtime, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: ("forward", 20)
    write_forward(runtime, owner)
    assert driver.pairs == [(50, -30)]
    assert runtime.backend.last_speed_receipt is not None
    assert runtime.forward_execution_anchor(1, 9.84, clock[0]) is None


def test_recent_completed_pair_survives_old_sample_ttl_for_new_depth_only(monkeypatch):
    """CAP675: the write is recent even when its source depth just expired."""
    runtime, owner, driver, _, clock, _ = execution_runtime(
        monkeypatch, sample_timestamp=9.75)
    write_forward(runtime, owner)
    anchor = runtime._forward_execution_anchor
    clock[0] = 10.025  # Last write 25ms old; old depth 275ms old.

    assert owner._fresh_depth_linear_snapshot(1, now=clock[0]) is None
    assert runtime.forward_execution_anchor(1, 9.75, clock[0]) is None
    assert runtime.recent_forward_execution_anchor(1, clock[0]) is anchor
    assert (driver.pairs, driver.stops) == ([(50, -30)], [])
    assert owner._depth30_linear_snapshot == ("forward", 20, 1, 9.75)


@pytest.mark.parametrize("change", [
    "zero", "stop", "source_revoked", "uid_changed", "identity_lost",
    "soft_stop", "parking", "search_handoff", "receipt_old",
])
def test_recent_completed_pair_rejects_real_interruption(monkeypatch, change):
    runtime, owner, _, _, clock, _ = execution_runtime(
        monkeypatch, sample_timestamp=9.75)
    write_forward(runtime, owner)
    clock[0] = 10.025
    assert runtime.recent_forward_execution_anchor(1, clock[0]) is not None
    if change == "zero":
        with owner.motor_io_lock:
            runtime.backend.send_targets(0, 0, "TEST_ZERO")
    elif change == "stop":
        with owner.motor_io_lock:
            runtime.backend.send_stop("TEST_STOP", mode="emergency")
    elif change == "source_revoked":
        owner._depth30_linear_snapshot = None
    elif change == "uid_changed":
        owner._follow_controller.active_target_id = 2
    elif change == "identity_lost":
        owner._vision_control_state = "lost_confirming"
    elif change == "soft_stop":
        owner._soft_stop_active = True
    elif change == "parking":
        runtime.backend.normal_zero_hold = True
    elif change == "search_handoff":
        owner._search_handoff_uid = 1
    else:
        clock[0] = 10.101
    assert runtime.recent_forward_execution_anchor(1, clock[0]) is None
