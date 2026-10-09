"""Completed write evidence only; fake serial and clocks, no hardware."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from car_control_modular.mssd_motor import MotorSpeedReceipt
from test_depth_drive_rpm import make_runtime
from test_motion_write_fault import FaultDriver


def setup_receipt(monkeypatch):
    runtime, _, driver, _ = make_runtime()
    backend = runtime.backend
    clock = [10.0]
    monkeypatch.setattr("car_control_modular.mssd_motor.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("car_control_modular.mssd_motor.time.sleep", lambda _: None)
    return backend, driver, clock


def test_receipt_is_immutable_raw_clipped_pair_completed_after_both_acks(monkeypatch):
    backend, driver, clock = setup_receipt(monkeypatch)
    backend.send_targets(10, -10, "previous")
    previous = backend.last_speed_receipt
    right, left = driver.set_right_speed, driver.set_left_speed
    events = []

    def write_right(value):
        assert backend.last_speed_receipt is None
        events.append(("right", value))
        clock[0] += .01
        right(value)

    def write_left(value):
        assert backend.last_speed_receipt is None
        events.append(("left", value))
        clock[0] += .02
        left(value)

    driver.set_right_speed, driver.set_left_speed = write_right, write_left
    backend.send_targets(150, -30, "clipped")
    receipt = backend.last_speed_receipt
    assert events == [("right", -30), ("left", 100)]
    assert receipt == MotorSpeedReceipt(previous.sequence + 1, 100, -30, clock[0])
    assert receipt.completed_at == pytest.approx(10.03)
    with pytest.raises(FrozenInstanceError):
        receipt.left_rpm = 200
    assert previous == MotorSpeedReceipt(1, 10, -10, 10.)


def test_log_cap791_pair_is_replaced_by_actual_zero_at_cap792(monkeypatch, caplog):
    backend, driver, clock = setup_receipt(monkeypatch)
    clock[0] = 12301.478955433
    with caplog.at_level("INFO"):
        backend.send_targets(50, -30, "FOLLOW20")
        sent_forward = backend.last_speed_receipt
        clock[0] = 12301.581926649
        backend.send_targets(0, 0, "FOLLOW20")
    assert sent_forward == MotorSpeedReceipt(1, 50, -30, 12301.478955433)
    assert backend.last_speed_receipt == MotorSpeedReceipt(2, 0, 0, clock[0])
    assert backend.last_speed_receipt is not sent_forward
    assert driver.pairs == [(50, -30), (0, 0)]
    assert "左轮=50转/分 右轮=-30转/分" in caplog.text
    assert "左轮=0转/分 右轮=0转/分" in caplog.text


@pytest.mark.parametrize("failed_side", ["right", "left"])
def test_partial_speed_write_cannot_leave_or_publish_receipt(monkeypatch, failed_side):
    backend, _, _ = setup_receipt(monkeypatch)
    driver = backend.driver = FaultDriver(fail_write=999)
    backend.send_targets(50, -30, "previous")
    driver.fail_write = driver.speed_calls + (1 if failed_side == "right" else 2)
    with pytest.raises(OSError, match="speed ACK"):
        backend.send_targets(40, -20, "failed")
    assert backend.last_speed_receipt is None
    assert backend.motion_write_fault
    assert driver.events[-4:] == [("speed", "right", 0), ("speed", "left", 0),
                                  ("stop", "right", 1), ("stop", "left", 1)]


@pytest.mark.parametrize("mode", ["normal", "emergency", "free"])
def test_each_stop_clears_receipt_before_any_io(monkeypatch, mode):
    backend, driver, _ = setup_receipt(monkeypatch)
    backend.send_targets(50, -30, "previous")
    for name in ("set_right_speed", "set_left_speed", "stop_all", "write_register"):
        original = getattr(driver, name)

        def check(*args, _original=original, **kwargs):
            assert backend.last_speed_receipt is None
            return _original(*args, **kwargs)

        monkeypatch.setattr(driver, name, check)
    backend.send_stop("stop", mode=mode)
    assert backend.last_speed_receipt is None
    assert driver.stops == ({"normal": [1, 0], "emergency": [1], "free": [2]}[mode])


@pytest.mark.parametrize("entry", ["refresh", "direct_stop", "startup", "close"])
def test_stop_entry_points_never_retain_old_receipt(monkeypatch, entry):
    backend, driver, _ = setup_receipt(monkeypatch)
    backend.send_targets(50, -30, "previous")
    sequence = backend.last_speed_receipt.sequence
    if entry == "refresh":
        backend.refresh_normal_stop("refresh")
    elif entry == "direct_stop":
        backend._try_stop_wheels("direct", 1)
    elif entry == "startup":
        backend.enable_startup_parking()
    else:
        backend.close()
    assert backend.last_speed_receipt is None
    if entry != "close":
        backend.send_targets(40, -20, "resumed")
        assert backend.last_speed_receipt.sequence > sequence


@pytest.mark.parametrize("mode", ["normal", "emergency", "free"])
def test_failed_stop_ack_cannot_restore_old_receipt(monkeypatch, mode):
    backend, _, _ = setup_receipt(monkeypatch)
    driver = backend.driver = FaultDriver(fail_write=999)
    backend.send_targets(50, -30, "previous")
    driver.stop_fail = {"left"}
    with pytest.raises(OSError, match="STOP ACK"):
        backend.send_stop("failed_stop", mode=mode)
    assert backend.last_speed_receipt is None
    assert backend.motion_write_fault


@pytest.mark.parametrize("entry", ["set", "prepare", "release"])
def test_current_transition_invalidates_before_register_io(monkeypatch, entry):
    backend, driver, _ = setup_receipt(monkeypatch)
    backend.send_targets(50, -30, "previous")
    backend.parking_current_a = 5.
    driver.registers.update(left_parking_current=5., right_parking_current=5.)
    write = driver.write_register

    def check(*args, **kwargs):
        assert backend.last_speed_receipt is None
        return write(*args, **kwargs)

    driver.write_register = check
    if entry == "set":
        backend.set_parking_current(0., persist=False)
    elif entry == "prepare":
        backend.prepare_speed_mode()
    else:
        backend.release_parking_current_only()
    assert backend.last_speed_receipt is None
    assert backend.parking_current_a == 0.
    assert driver.pairs == [(50, -30)]


def test_current_readback_failure_has_no_receipt(monkeypatch):
    backend, driver, _ = setup_receipt(monkeypatch)
    backend.send_targets(50, -30, "previous")
    driver.read_register = lambda _: 5.
    with pytest.raises(RuntimeError, match="readback mismatch"):
        backend.set_parking_current(0., persist=False)
    assert backend.last_speed_receipt is None


def test_noop_speed_prepare_preserves_receipt_without_io(monkeypatch):
    backend, driver, _ = setup_receipt(monkeypatch)
    backend.send_targets(50, -30, "previous")
    receipt = backend.last_speed_receipt
    backend.prepare_speed_mode()
    assert backend.last_speed_receipt is receipt
    assert not driver.register_writes


def test_preserved_zero_does_not_claim_a_new_speed_ack(monkeypatch):
    backend, driver, _ = setup_receipt(monkeypatch)
    backend.send_stop("park", mode="emergency", preserve_zero=True)
    before = list(driver.pairs)
    backend.send_targets(0, 0, "held_zero")
    assert driver.pairs == before
    assert backend.last_speed_receipt is None


@pytest.mark.parametrize("entry", ["record", "rtu_sync", "during_ack"])
def test_fault_invalidates_receipt_even_without_a_followup_stop(monkeypatch, entry):
    backend, driver, _ = setup_receipt(monkeypatch)
    backend.send_targets(50, -30, "previous")
    if entry == "record":
        backend._record_motion_write_fault("test", OSError("failure"))
    elif entry == "rtu_sync":
        backend._rtu_guard = SimpleNamespace(rx_uncertain=True, fault_reason="read timeout")
        backend.sync_transaction_fault()
    else:
        left = driver.set_left_speed

        def fault_after_write(value):
            left(value)
            backend._record_motion_write_fault("concurrent_fault", OSError("failure"))

        driver.set_left_speed = fault_after_write
        backend.send_targets(40, -20, "ack_during_fault")
    assert backend.last_speed_receipt is None
    assert backend.motion_write_fault
