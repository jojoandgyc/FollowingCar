"""Recovery memory is completed I/O evidence, never an expired motor grant."""
from dataclasses import replace

import pytest

from car_control_modular.longitudinal_execution import ForwardRecoveryAnchor
from test_forward_execution_anchor import execution_runtime, write_forward


def memory(monkeypatch):
    runtime, owner, driver, _, clock, _ = execution_runtime(
        monkeypatch, sample_timestamp=9.84)
    write_forward(runtime, owner)
    clock[0] = 10.11  # Source expired 20ms ago; last completed packet 110ms ago.
    return runtime, owner, driver, clock


def test_memory_survives_short_expiry_but_does_not_renew_or_write(monkeypatch):
    runtime, owner, driver, clock = memory(monkeypatch)
    source, timing = owner._depth30_linear_snapshot, owner._depth30_linear_timing
    receipt = runtime.backend.last_speed_receipt
    assert owner._fresh_depth_linear_snapshot(1, now=clock[0]) is None
    assert runtime.recent_forward_execution_anchor(1, clock[0]) is None
    proof = runtime.recovery_forward_execution_anchor(1, clock[0])
    assert isinstance(proof, ForwardRecoveryAnchor)
    assert proof.executed is runtime._forward_execution_anchor
    assert proof.current_receipt is receipt
    assert proof.executed.rpm == 40
    assert runtime.forward_recovery_anchor_valid(1, proof, clock[0] + .01)
    assert runtime.backend.last_speed_receipt is receipt
    assert owner._depth30_linear_snapshot is source
    assert owner._depth30_linear_timing is timing
    assert timing.depth_expires_at == pytest.approx(10.09)
    assert driver.pairs == [(50, -30)] and not driver.stops


@pytest.mark.parametrize("change", [
    "zero", "stop", "reverse", "other_positive_packet", "source_revoked",
    "source_uid", "source_sample", "uid", "lost", "soft_stop", "explicit_stop",
    "shutdown", "brake", "park_request", "search", "search_handoff", "parking",
    "write_fault", "expired_memory", "future_clock", "nan", "infinite", "bool_clock",
])
def test_memory_cannot_cross_actual_interruption_or_expiry(monkeypatch, change):
    runtime, owner, driver, clock = memory(monkeypatch)
    proof = runtime.recovery_forward_execution_anchor(1, clock[0])
    assert proof is not None
    if change in {"zero", "reverse", "other_positive_packet"}:
        pair = {"zero": (0, 0), "reverse": (-30, 30),
                "other_positive_packet": (50, -30)}[change]
        runtime.backend.send_targets(*pair, "OTHER_TRANSACTION")
    elif change == "stop":
        runtime.backend.send_stop("REAL_STOP", mode="emergency")
    elif change == "source_revoked":
        owner._depth30_linear_snapshot = None
    elif change == "source_uid":
        owner._depth30_linear_snapshot = ("forward", 20, 2, 9.84)
    elif change == "source_sample":
        owner._depth30_linear_snapshot = ("forward", 20, 1, 9.85)
    elif change == "uid":
        owner._follow_controller.active_target_id = 2
    elif change == "lost":
        owner._vision_control_state = "lost_confirming"
    elif change == "soft_stop":
        owner._soft_stop_active = True
    elif change == "explicit_stop":
        owner._explicit_stop_requested = True
    elif change == "shutdown":
        owner._runtime_shutdown_requested = True
    elif change == "brake":
        owner._brake_hold_active = True
    elif change == "park_request":
        owner._near_yaw_park_request = object()
    elif change == "search":
        owner.search_state = "searching"
    elif change == "search_handoff":
        owner._search_handoff_uid = 1
    elif change == "parking":
        runtime.backend.normal_zero_hold = True
    elif change == "write_fault":
        runtime.backend.motion_write_fault = "fault"
    elif change == "expired_memory":
        clock[0] = 10.191
    elif change == "future_clock":
        clock[0] = 9.99
    elif change == "nan":
        clock[0] = float("nan")
    elif change == "infinite":
        clock[0] = float("inf")
    elif change == "bool_clock":
        clock[0] = True
    assert runtime.recovery_forward_execution_anchor(1, clock[0]) is None
    assert not runtime.forward_recovery_anchor_valid(1, proof, clock[0])


def test_proof_rechecks_stop_epoch_even_if_receipt_was_not_cleared(monkeypatch):
    runtime, _, _, clock = memory(monkeypatch)
    proof = runtime.recovery_forward_execution_anchor(1, clock[0])
    runtime.backend.stop_write_generation += 1
    assert not runtime.forward_recovery_anchor_valid(1, proof, clock[0])


@pytest.mark.parametrize("field,value", [
    ("checked_at", float("nan")), ("checked_at", float("inf")),
    ("checked_at", True), ("checked_at", 10.12),
    ("current_receipt", object()), ("stop_generation", -1),
])
def test_malformed_or_superseded_proof_never_validates(monkeypatch, field, value):
    runtime, _, _, clock = memory(monkeypatch)
    proof = runtime.recovery_forward_execution_anchor(1, clock[0])
    assert not runtime.forward_recovery_anchor_valid(
        1, replace(proof, **{field: value}), clock[0])


def test_control_state_changes_during_reader_are_rechecked(monkeypatch):
    runtime, owner, _, clock = memory(monkeypatch)
    def revoke_source():
        owner._depth30_linear_snapshot = None
        return True
    runtime._visible_wheel_control_active = revoke_source
    assert runtime.recovery_forward_execution_anchor(1, clock[0]) is None
