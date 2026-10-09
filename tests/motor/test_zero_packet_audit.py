"""Every real zero/STOP is traceable; fake driver only, no serial access."""
import json
import logging

import pytest

from car_control_modular.deferred_diagnostics import deferred_diagnostics
from test_motor_speed_receipt import setup_receipt
from test_motion_write_fault import FaultDriver


def events(caplog):
    return [json.loads(record.getMessage().split("motor_zero_audit ", 1)[1])
            for record in caplog.records if record.msg == "motor_zero_audit %s"]


def test_zero_record_carries_origin_and_previous_actual_write(monkeypatch, caplog):
    backend, driver, clock = setup_receipt(monkeypatch)
    backend.zero_audit_context_provider = lambda label, kind: {
        "uid": 1, "control_cap": 205, "zero_reason": "incomplete_lateral_publication",
        "grant_age_ms": 50., "planned_axes": (24., 0.),
    }
    with caplog.at_level(logging.INFO):
        backend.send_targets(30, -20, "FOLLOW20")
        previous = backend.last_speed_receipt
        clock[0] += .05
        backend.send_targets(0, 0, "FOLLOW20_SNAPSHOT_REVOKED")
    records = events(caplog)
    assert len(records) == 1  # No extra audit for ordinary nonzero packets.
    event = records[0]
    assert event["source"] == "FOLLOW20_SNAPSHOT_REVOKED"
    assert event["previous_acknowledged_raw"] == [30, -20]
    assert event["previous_forward_yaw"] == [25., 5.]
    assert event["previous_receipt_sequence"] == previous.sequence
    assert event["receipt_after"] == backend.last_speed_receipt.sequence
    assert event["normalized_forward_yaw"] == [0., 0.]
    assert event["context"]["control_cap"] == 205
    assert event["context"]["zero_reason"] == "incomplete_lateral_publication"
    assert event["writes_complete"] and event["outcome"] == "acknowledged"
    assert event["physical_stillness"] == "unverified"
    assert driver.pairs[-1] == (0, 0)


@pytest.mark.parametrize("entry", ["raw", "percent_quantized", "clipped"])
def test_all_zero_speed_public_entries_reach_one_audit(monkeypatch, caplog, entry):
    backend, driver, _ = setup_receipt(monkeypatch)
    with caplog.at_level(logging.INFO):
        if entry == "raw":
            backend.send_targets(0, 0, "zero")
        elif entry == "percent_quantized":
            backend.send_diff(.9, 1, .9, 1, "zero")
        else:
            backend.send_targets(60, -60, "zero", max_target_override=0)
    records = events(caplog)
    assert len(records) == 1
    assert records[0]["phase"] == "speed_pair"
    assert records[0]["acknowledged_sides"] == ["right", "left"]
    assert backend.last_speed_receipt.left_rpm == backend.last_speed_receipt.right_rpm == 0


@pytest.mark.parametrize("mode,phases,values", [
    ("normal", ["normal_pre_zero", "stop_registers", "stop_registers"], [None, 1, 0]),
    ("emergency", ["stop_registers"], [1]),
    ("free", ["stop_registers"], [2]),
])
def test_stop_audit_distinguishes_pre_zero_and_actual_mode(monkeypatch, caplog, mode, phases, values):
    backend, driver, _ = setup_receipt(monkeypatch)
    with caplog.at_level(logging.INFO):
        backend.send_stop("stop_reason", mode=mode)
    records = events(caplog)
    assert [item["phase"] for item in records] == phases
    assert [item["stop_mode"] for item in records] == values
    assert all(item["writes_complete"] for item in records)
    assert all(item["stop_generation"] == 1 for item in records)
    assert backend.last_speed_receipt is None  # pre-zero remains non-authoritative.


def test_preserved_zero_hold_is_skipped_not_fabricated_write(monkeypatch, caplog):
    backend, driver, _ = setup_receipt(monkeypatch)
    backend.send_stop("park", mode="emergency", preserve_zero=True)
    before = list(driver.pairs)
    with caplog.at_level(logging.INFO):
        backend.send_targets(0, 0, "hold")
    event, = events(caplog)
    assert event["phase"] == "preserved_stop_hold"
    assert event["outcome"] == "skipped" and not event["writes_complete"]
    assert event["attempted_sides"] == event["acknowledged_sides"] == []
    assert driver.pairs == before and backend.last_speed_receipt is None


@pytest.mark.parametrize("failed_side", ["right", "left"])
def test_partial_zero_failure_and_rollback_are_separate_events(monkeypatch, caplog, failed_side):
    backend, _, _ = setup_receipt(monkeypatch)
    backend.driver = FaultDriver(fail_write=1 if failed_side == "right" else 2)
    with caplog.at_level(logging.INFO), pytest.raises(OSError, match="speed ACK"):
        backend.send_targets(0, 0, "initial_zero")
    initial, rollback, stop = events(caplog)
    assert initial["phase"] == "speed_pair" and initial["outcome"] == "failed"
    assert initial["failed_sides"] == [failed_side]
    assert initial["acknowledged_sides"] == ([] if failed_side == "right" else ["right"])
    assert not initial["writes_complete"]
    assert rollback["phase"] == "fault_rollback_zero" and rollback["writes_complete"]
    assert stop["event_kind"] == "stop" and stop["stop_mode"] == 1
    assert backend.last_speed_receipt is None
    assert initial["event_id"] < rollback["event_id"] < stop["event_id"]


def test_partial_fault_cleanup_never_claims_complete_or_new_receipt(monkeypatch, caplog):
    backend, _, _ = setup_receipt(monkeypatch)
    backend.driver = FaultDriver(zero_fail={"left"}, stop_fail={"right"})
    with caplog.at_level(logging.INFO), pytest.raises(OSError, match="speed ACK"):
        backend.send_targets(60, -60, "forward_fault")
    rollback, stop = events(caplog)
    assert rollback["failed_sides"] == ["left"] and not rollback["writes_complete"]
    assert stop["failed_sides"] == ["right"] and not stop["writes_complete"]
    assert stop["acknowledged_sides"] == ["left"]
    assert backend.last_speed_receipt is None


def test_stop_pre_zero_partial_failure_still_audits_real_stop(monkeypatch, caplog):
    backend, _, _ = setup_receipt(monkeypatch)
    backend.driver = FaultDriver(fail_write=2)
    with caplog.at_level(logging.INFO):
        backend.send_stop("normal", mode="normal")
    pre_zero, pre_stop, final_stop = events(caplog)
    assert pre_zero["outcome"] == "failed" and pre_zero["acknowledged_sides"] == ["right"]
    assert pre_stop["writes_complete"] and final_stop["writes_complete"]
    assert backend.last_speed_receipt is None


def test_frozen_context_and_formatting_are_deferred_outside_motor_lock(monkeypatch):
    backend, driver, _ = setup_receipt(monkeypatch)
    logger = logging.getLogger("zero_audit_deferred_test")
    logger.setLevel(logging.INFO)
    backend.logger = logger
    context = {"uid": 1, "planned_axes": [24., 0.], "control_cap": 205}
    backend.zero_audit_context_provider = lambda *_: context
    received = []

    class CheckedHandler(logging.Handler):
        def emit(self, record):
            assert not backend.io_lock.locked()
            if record.msg == "motor_zero_audit %s":
                received.append(json.loads(str(record.args[0])))

    handler = CheckedHandler()
    logger.addHandler(handler)
    try:
        with deferred_diagnostics(logger):
            with backend.io_lock:
                backend.send_targets(0, 0, "snapshot_zero")
                context["planned_axes"][0] = 99.
                context["control_cap"] = 999
                assert not received
        assert received[0]["context"]["planned_axes"] == [24., 0.]
        assert received[0]["context"]["control_cap"] == 205
    finally:
        logger.removeHandler(handler)


@pytest.mark.parametrize("provider", [lambda *_: None, lambda *_: 1 / 0])
def test_broken_diagnostic_provider_cannot_block_stop_or_zero(monkeypatch, caplog, provider):
    backend, driver, _ = setup_receipt(monkeypatch)
    backend.zero_audit_context_provider = provider
    with caplog.at_level(logging.INFO):
        backend.send_targets(0, 0, "zero")
        backend.send_stop("stop", mode="emergency")
    assert driver.pairs == [(0, 0)]
    assert len(events(caplog)) == 2
    assert all(item["context"]["context_error"] for item in events(caplog))


def test_failed_audit_sink_cannot_prevent_physical_zero_or_stop(monkeypatch):
    backend, driver, _ = setup_receipt(monkeypatch)
    original_info = backend.logger.info

    def broken_audit(message, *args, **kwargs):
        if message == "motor_zero_audit %s":
            raise OSError("diagnostic sink failed")
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(backend.logger, "info", broken_audit)
    backend.send_targets(0, 0, "zero")
    backend.send_stop("stop", mode="emergency")
    assert driver.pairs == [(0, 0)] and driver.stops == [1]


@pytest.mark.parametrize("entry", ["startup", "refresh", "close"])
def test_indirect_startup_refresh_and_shutdown_stop_are_audited(monkeypatch, caplog, entry):
    backend, driver, _ = setup_receipt(monkeypatch)
    with caplog.at_level(logging.INFO):
        if entry == "startup":
            backend.enable_startup_parking()
        elif entry == "refresh":
            backend.refresh_normal_stop("park_refresh")
        else:
            backend.close()
    recorded = events(caplog)
    assert recorded
    assert len([event for event in recorded if event["event_kind"] == "stop"]) == len(driver.stops)
    assert all(event["writes_complete"] for event in recorded)
    if entry == "startup":
        assert recorded[0]["source"] == "startup_zero"
