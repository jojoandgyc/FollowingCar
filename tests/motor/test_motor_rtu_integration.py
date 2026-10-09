"""Vendor client + production adapter/executor, with an in-memory wire only."""
from dataclasses import replace
import importlib

import pytest

from test_depth_drive_rpm import make_runtime


@pytest.fixture
def vendor(monkeypatch):
    runtime, _, _, _ = make_runtime()
    runtime.backend.config = replace(runtime.backend.config, lib_dir="")
    try:
        library = runtime.backend._resolve_lib_dir()
    except RuntimeError:
        pytest.skip("optional vendor library is not present; pure RTU tests still run")
    monkeypatch.syspath_prepend(library)
    client = importlib.import_module("lz30ema_rs485")
    protocol = importlib.import_module("lz30ema_rs485.protocol")
    return client, protocol


class MemoryWire:
    """Replies to the real client's packets; never constructs SerialTransport."""
    timeout = .15
    write_timeout = None

    def __init__(self, protocol):
        self.protocol = protocol
        self.requests = []
        self.rx = b""
        self.drop_address = None
        self.registers = {}
        self.closed = False
        self.resets = 0

    def write(self, packet):
        packet = bytes(packet)
        self.requests.append(packet)
        fn, address = packet[1], int.from_bytes(packet[2:4], "big")
        if address == self.drop_address:
            self.rx = b""
        elif fn == 6:
            self.registers[address] = int.from_bytes(packet[4:6], "big")
            self.rx = packet
        elif fn == 16:
            self.rx = self.protocol.append_crc(packet[:6])
        elif fn == 3:
            count = int.from_bytes(packet[4:6], "big")
            data = b"".join(self.registers.get(address+i, 0).to_bytes(2, "big")
                            for i in range(count))
            self.rx = self.protocol.append_crc(bytes((packet[0], 3, len(data))) + data)
        else:
            raise AssertionError("unexpected vendor operation")
        return len(packet)

    def read(self, count):
        data, self.rx = self.rx[:count], self.rx[count:]
        return data

    @property
    def in_waiting(self):
        return len(self.rx)

    def reset_input_buffer(self):
        self.resets += 1
        self.rx = b""

    def close(self):
        self.closed = True


def create_backend(monkeypatch, vendor, *, startup=False):
    client, protocol = vendor
    runtime, owner, _, symbols = make_runtime()
    backend = runtime.backend
    wire = MemoryWire(protocol)
    backend.driver = None
    backend.config = replace(backend.config, lib_dir="", startup_parking_enabled=startup,
                             parking_current_a=0., stop_mode="emergency", timeout=.15)
    monkeypatch.setattr(client.LZ30EMAClient, "from_serial", classmethod(
        lambda cls, *args, **kw: cls(wire, slave=kw["slave"])))
    driver = backend.ensure_driver()
    return runtime, owner, driver, wire, symbols


def addresses(wire, fn):
    return [int.from_bytes(row[2:4], "big") for row in wire.requests if row[1] == fn]


@pytest.mark.parametrize("startup", [False, True])
def test_real_factory_installs_guard_before_startup_io(monkeypatch, vendor, startup):
    runtime, _, driver, wire, _ = create_backend(monkeypatch, vendor, startup=startup)
    assert runtime.backend._rtu_guard is not None
    assert not runtime.backend._rtu_guard.rx_uncertain
    assert driver.transport is wire and wire.requests
    runtime.backend.send_targets(15, -1, "FOLLOW20")
    assert addresses(wire, 16)[-2:] == [0x41, 0x45]
    assert not runtime.backend.motion_write_fault
    assert wire.timeout == .15 and wire.write_timeout is None


@pytest.mark.parametrize("side,address,count", [("right", 0x41, 1), ("left", 0x45, 2)])
def test_missing_ack_records_wheel_and_stops_without_replaying_speed(
        monkeypatch, vendor, caplog, side, address, count):
    runtime, owner, driver, wire, _ = create_backend(monkeypatch, vendor)
    wire.requests.clear()
    wire.drop_address = address
    with pytest.raises((TimeoutError, RuntimeError, OSError)):
        runtime.backend.send_targets(15, -1, "FOLLOW20")
    assert runtime.backend.motion_write_fault
    assert len(addresses(wire, 16)) == count  # No zero-speed or old-motion retry.
    assert addresses(wire, 6)[-2:] == [0x40, 0x44]
    assert "failed_side=" + side in caplog.text
    assert "requested_left_rpm=15 requested_right_rpm=-1" in caplog.text
    assert "right_acknowledged=" + str(side == "left") in caplog.text
    wire.drop_address = None
    before = list(wire.requests)
    with pytest.raises(RuntimeError):
        runtime.backend.send_targets(15, -1, "do_not_resume")
    assert wire.requests == before
    assert runtime._service_motion_write_fault()
    assert owner._runtime_shutdown_requested and not owner.running


def test_feedback_transaction_failure_is_escalated_without_waiting_for_new_motion(monkeypatch, vendor):
    runtime, owner, driver, wire, _ = create_backend(monkeypatch, vendor)
    wire.drop_address = 0x12
    with pytest.raises((TimeoutError, RuntimeError, OSError)):
        driver.read_motor_status("left")
    assert runtime.backend._rtu_guard.rx_uncertain
    # This runs independently of new vision/depth/command production.
    wire.drop_address = None
    assert runtime._service_motion_write_fault()
    assert runtime.backend.motion_write_fault
    assert owner._runtime_shutdown_requested and not owner.running
    assert addresses(wire, 6)[-2:] == [0x40, 0x44]


def test_late_multiple_write_ack_cannot_certify_zero_speed_or_clear_fault(monkeypatch, vendor):
    runtime, _, driver, wire, _ = create_backend(monkeypatch, vendor)
    wire.drop_address = 0x41
    with pytest.raises((TimeoutError, RuntimeError, OSError)):
        runtime.backend.send_targets(20, -20, "fault")
    old_request = next(row for row in wire.requests if row[1] == 16)
    wire.rx = vendor[1].append_crc(old_request[:6])
    wire.drop_address = None
    before = len(addresses(wire, 16))
    with pytest.raises((RuntimeError, OSError)):
        driver.set_right_speed(0)
    assert len(addresses(wire, 16)) == before
    runtime.backend.send_stop("still_faulted", mode="emergency")
    assert runtime.backend.motion_write_fault and runtime.backend._rtu_guard.rx_uncertain
    assert wire.resets > 0
    runtime.backend.close()
    assert wire.closed and runtime.backend.driver is None
    assert runtime.backend.motion_write_fault  # No reconnection on this object.


@pytest.mark.parametrize("startup", [False, True])
def test_failed_initialization_cannot_reconnect_to_discard_poisoned_link(monkeypatch, vendor, startup):
    client, protocol = vendor
    runtime, _, _, _ = make_runtime()
    backend = runtime.backend
    backend.driver = None
    backend.config = replace(backend.config, lib_dir="", startup_parking_enabled=startup,
                             parking_current_a=0., stop_mode="emergency", timeout=.15)
    wire = MemoryWire(protocol)
    wire.drop_address = 0x6D
    opens = []
    def factory(cls, *args, **kw):
        opens.append(True)
        return cls(wire, slave=kw["slave"])
    monkeypatch.setattr(client.LZ30EMAClient, "from_serial", classmethod(factory))
    with pytest.raises((TimeoutError, RuntimeError, OSError)):
        backend.ensure_driver()
    assert backend.driver is None and wire.closed
    assert backend._rtu_guard.rx_uncertain and backend.motion_write_fault
    wire.drop_address = None
    with pytest.raises(RuntimeError, match="restart required"):
        backend.ensure_driver()
    assert len(opens) == 1
