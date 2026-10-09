"""Startup uses the real LZ SDK and RTU guard, with an in-memory wire only."""
from dataclasses import replace

import pytest

from test_depth_drive_rpm import make_runtime
from test_motor_rtu_integration import MemoryWire, vendor


def ramp_backend(monkeypatch, vendor, requested=None, *, diagnostics=True, wire_type=MemoryWire):
    client, protocol = vendor
    action, owner, _, _ = make_runtime()
    backend = action.backend
    backend.driver = None
    backend.config = replace(
        backend.config, startup_parking_enabled=True, parking_current_a=0.,
        lib_dir="", stop_mode="emergency", ramp_diagnostics_enable=diagnostics,
        closed_loop_acceleration_rpm_s=requested,
    )
    wire = wire_type(protocol)
    wire.registers.update({0x005F: 350, 0x0060: 450, 0x001D: 0, 0x001E: 0x38, 0x0087: 0})
    opened = []
    def factory(cls, *args, **kwargs):
        opened.append(True)
        return cls(wire, slave=kwargs["slave"])
    monkeypatch.setattr(client.LZ30EMAClient, "from_serial", classmethod(factory))
    return backend, wire, opened


def operations(wire, address):
    return [p for p in wire.requests if int.from_bytes(p[2:4], "big") == address]


def test_default_runtime_diagnostic_is_one_burst_read_and_never_tunes(monkeypatch, vendor, caplog):
    backend, wire, _ = ramp_backend(monkeypatch, vendor)
    with caplog.at_level("INFO"):
        driver = backend.ensure_driver()
    assert backend.ramp_diagnostics["acceleration_rpm_s"] == 350
    assert backend.ramp_diagnostics["deceleration_rpm_s"] == 450
    assert backend.ramp_diagnostics["requested_acceleration_rpm_s"] is None
    packets = operations(wire, 0x005F)
    assert len(packets) == 1 and packets[0][1] == 3
    assert int.from_bytes(packets[0][4:6], "big") == 2
    assert "measured_acceleration=False" in caplog.text
    before = list(wire.requests)
    assert backend.ensure_driver() is driver
    assert wire.requests == before  # No per-frame/keepalive configuration I/O.


@pytest.mark.parametrize("requested", [350, 600])
def test_only_explicit_change_writes_005f_once_after_startup_stop(monkeypatch, vendor, requested):
    backend, wire, _ = ramp_backend(monkeypatch, vendor, requested, diagnostics=False)
    backend.ensure_driver()
    writes = [p for p in operations(wire, 0x005F) if p[1] != 3]
    assert len(writes) == int(requested != 350)
    if writes:
        assert writes[0][1] == 6  # Never function 0x10 / persistent configuration.
        assert int.from_bytes(writes[0][4:6], "big") == requested
        preceding = wire.requests[:wire.requests.index(writes[0])]
        assert all(any(p[1] == 6 and int.from_bytes(p[2:4], "big") == addr
                       and int.from_bytes(p[4:6], "big") == 1 for p in preceding)
                   for addr in (0x0040, 0x0044))
    assert wire.registers[0x0060] == 450
    assert not [p for p in operations(wire, 0x0060) if p[1] != 3]
    assert backend.ramp_diagnostics["acceleration_rpm_s"] == requested
    assert not backend.motion_armed
    for packet in wire.requests:
        if packet[1] == 16 and int.from_bytes(packet[2:4], "big") in (0x41, 0x45):
            assert not any(packet[7:-2])  # Init does not command nonzero wheels.


def test_read_timeout_cannot_be_ignored_to_continue_or_reconnect(monkeypatch, vendor):
    backend, wire, opened = ramp_backend(monkeypatch, vendor)
    wire.drop_address = 0x005F
    with pytest.raises((TimeoutError, RuntimeError, OSError)):
        backend.ensure_driver()
    assert backend.motion_write_fault and backend._rtu_guard.rx_uncertain
    assert backend.driver is None and wire.closed
    assert len(operations(wire, 0x005F)) == 1  # No retry or later tuning write.
    wire.drop_address = None
    with pytest.raises(RuntimeError, match="restart required"):
        backend.ensure_driver()
    assert len(opened) == 1


@pytest.mark.parametrize("fault", ["mismatch", "timeout"])
def test_ambiguous_ramp_write_latches_fault_without_retry(monkeypatch, vendor, fault):
    class BadRampWire(MemoryWire):
        def write(self, packet):
            result = super().write(packet)
            if packet[1] == 6 and int.from_bytes(packet[2:4], "big") == 0x005F:
                if fault == "mismatch": self.registers[0x005F] = 599
                else: self.rx = b""
            return result
    backend, wire, opened = ramp_backend(monkeypatch, vendor, 600, wire_type=BadRampWire)
    with pytest.raises(RuntimeError):
        backend.ensure_driver()
    assert backend.motion_write_fault and backend.driver is None and wire.closed
    writes = [p for p in operations(wire, 0x005F) if p[1] != 3]
    assert len(writes) == 1
    with pytest.raises(RuntimeError, match="restart required"):
        backend.ensure_driver()
    assert len(opened) == 1


@pytest.mark.parametrize("value", [0, 9, 65536, True, False, 350., "600", float("nan"), float("inf")])
def test_bad_override_fails_before_any_serial_open(value):
    r, _, _, _ = make_runtime()
    with pytest.raises(ValueError):
        replace(r.backend.config, closed_loop_acceleration_rpm_s=value)


def test_override_requires_startup_stop_not_merely_arming_speed_mode():
    r, _, _, _ = make_runtime()
    with pytest.raises(ValueError, match="startup"):
        replace(r.backend.config, startup_parking_enabled=False, closed_loop_acceleration_rpm_s=600)
