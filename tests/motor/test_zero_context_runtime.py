"""The zero ledger must describe its own cause, not unrelated cached work."""

import json
import logging
import threading
from types import SimpleNamespace

import pytest

from car_control_modular.action_command import ActionCommandSnapshot
from test_depth_drive_rpm import make_runtime


def stale_stop(runtime):
    return ActionCommandSnapshot(
        action=runtime.symbols.stop, revision=7, enqueued_at=9.,
        control_frame=3, capture_frame_id=205, capture_timestamp=8.9,
        reason="old_identity_lost", soft_stop=False,
        source_module="vision", uid=1,
    )


@pytest.mark.parametrize("label,stage", [
    ("safety_motion_write_fault", "command_or_direct_stop"),
    ("runtime_shutdown", "command_or_direct_stop"),
    ("startup_zero", "command_or_direct_stop"),
    ("FOLLOW20_REVOKED", "executor"),
])
def test_stale_queued_stop_never_overrides_actual_zero_cause(label, stage):
    runtime, owner, driver, _ = make_runtime()
    runtime._current_action_snapshot = stale_stop(runtime)
    context = runtime._zero_audit_context(label, "stop")
    assert context["zero_reason"] == label
    assert context["decision_stage"] == stage
    assert context["command_reason"] == "old_identity_lost"
    assert context["command_cap"] == 205
    assert context["command_source"] == "vision"
    assert not driver.pairs and not driver.stops


@pytest.mark.parametrize("prepared,committed,deadline", [
    ("new", "old", 10.25),
    ("old", "new", 10.25),
    (None, "new", 10.25),
    ("old", "old", None),
    (None, "old", None),
    (None, None, None),
])
def test_only_matching_grant_can_supply_depth_deadline(prepared, committed, deadline):
    runtime, owner, _driver, _ = make_runtime()
    raw = ("forward", 60, 1, 10.)
    old = ("forward", 20, 1, 9.)

    def timing(which):
        return None if which is None else SimpleNamespace(
            snapshot=raw if which == "new" else old,
            depth_expires_at=10.25 if which == "new" else 9.25,
        )

    owner._depth30_linear_snapshot = raw
    owner._depth30_prepared_timing = timing(prepared)
    owner._depth30_linear_timing = timing(committed)
    context = runtime._zero_audit_context("FOLLOW20", "zero_speed")
    assert context["grant"] == raw
    assert context["depth_deadline"] == deadline


def test_backend_zero_freezes_qualified_read_veto_not_old_quiet_veto(caplog):
    runtime, owner, driver, _ = make_runtime()
    owner._depth30_read_veto = (1, 10., "visibility_expired")
    owner._last_quiet_depth_veto = (2, 9., "old_feedback_veto")
    owner._depth30_linear_snapshot = ("forward", 60, 1, 10.)
    with caplog.at_level(logging.INFO):
        runtime.backend.send_targets(0, 0, "FOLLOW20_REVOKED")
    event, = [record.args[0] for record in caplog.records
              if record.msg == "motor_zero_audit %s"]
    owner._depth30_read_veto = (3, 11., "different_veto")
    owner._depth30_linear_snapshot = ("forward", 20, 3, 11.)
    context = json.loads(str(event))["context"]
    assert context["depth_read_veto"] == [1, 10., "visibility_expired"]
    assert context["grant"] == ["forward", 60, 1, 10.]
    assert event.acknowledged_sides == ("right", "left")
    assert driver.pairs == [(0, 0)]


def test_explicit_zero_reason_scope_nests_restores_and_survives_exception(monkeypatch):
    runtime, _owner, _driver, _ = make_runtime()
    monkeypatch.setattr("car_control_modular.action_runtime.time.monotonic", lambda: 10.)
    runtime._current_action_snapshot = stale_stop(runtime)
    with runtime._zero_packet_decision("snapshot_admission", "feedback_expired", uid=5):
        outer = runtime._zero_audit_context("FOLLOW20", "zero_speed")
        assert outer["decision_stage"] == "snapshot_admission"
        assert outer["zero_reason"] == "feedback_expired" and outer["uid"] == 5
        with pytest.raises(RuntimeError, match="inner failed"):
            with runtime._zero_packet_decision("queued_stop", "actual_stop_reason"):
                inner = runtime._zero_audit_context("stop", "stop")
                assert inner["zero_reason"] == "actual_stop_reason"
                assert inner["decision_stage"] == "queued_stop"
                raise RuntimeError("inner failed")
        assert runtime._zero_audit_context("FOLLOW20", "zero_speed") == outer
    restored = runtime._zero_audit_context("runtime_shutdown", "stop")
    assert restored["zero_reason"] == "runtime_shutdown"
    assert restored["decision_stage"] == "command_or_direct_stop"
    assert getattr(runtime._dispatch_context, "motor_zero_decision", None) is None


def test_zero_reason_scope_is_thread_local():
    runtime, _owner, _driver, _ = make_runtime()
    received = []
    with runtime._zero_packet_decision("snapshot_admission", "only_this_thread"):
        worker = threading.Thread(target=lambda: received.append(
            runtime._zero_audit_context("safety_motion_write_fault", "stop")))
        worker.start()
        worker.join(timeout=2.)
        assert not worker.is_alive()
        assert received[0]["zero_reason"] == "safety_motion_write_fault"
        assert received[0]["decision_stage"] == "command_or_direct_stop"
        assert runtime._zero_audit_context("FOLLOW20", "zero_speed")["zero_reason"] == "only_this_thread"


def test_zero_reason_details_are_frozen_before_scope_changes(caplog):
    runtime, _owner, driver, _ = make_runtime()
    requested = [24, 0]
    with caplog.at_level(logging.INFO):
        with runtime._zero_packet_decision(
            "wheel_composition", "wheel_crossing_wait", requested_forward_rpm=requested,
        ):
            runtime.backend.send_targets(0, 0, "FOLLOW20")
            requested[0] = 99
    event, = [record.args[0] for record in caplog.records
              if record.msg == "motor_zero_audit %s"]
    context = json.loads(str(event))["context"]
    assert context["requested_forward_rpm"] == [24, 0]
    assert context["zero_reason"] == "wheel_crossing_wait"
    assert context["decision_stage"] == "wheel_composition"
    assert runtime._zero_audit_context("FOLLOW20", "zero_speed")["zero_reason"] == "FOLLOW20"
    assert driver.pairs == [(0, 0)]
