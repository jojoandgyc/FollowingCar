"""Ramp configuration contract with fake registers and the real SDK on memory I/O."""
import pytest

from car_control_modular.motor_ramp import (
    RampConfigurationError, configure_closed_loop_ramp,
    validate_acceleration_rpm_s, validate_closed_loop_mode,
)
from car_control_modular.motor_rtu import install_motor_rtu_guard
from test_motor_rtu_integration import MemoryWire, vendor


class Registers:
    def __init__(self):
        self.values = [350, 450]
        self.bus = {"runtime_system_mode": 1, "control_mode": 0x38}
        self.system_mode = 1
        self.operations = []
        self.read_error = None
        self.write_error = None
        self.after_write = None

    def read_holding_registers(self, address, count):
        self.operations.append(("read_pair", address, count))
        if self.read_error:
            raise self.read_error
        return list(self.values)

    def read_bus_status(self):
        self.operations.append(("read_bus",))
        return dict(self.bus)

    def read_register(self, name):
        assert name == "system_mode"
        self.operations.append(("read_register", name))
        return self.system_mode

    def write_register(self, name, value, *, persist):
        self.operations.append(("write_register", name, value, persist))
        assert name == "closed_loop_acceleration" and persist is False
        if self.write_error:
            raise self.write_error
        self.values[0] = value
        if self.after_write:
            self.after_write(self)


@pytest.mark.parametrize("value", [None, 10, 350, 65535])
def test_valid_optional_integer_acceleration(value):
    assert validate_acceleration_rpm_s(value) == value


@pytest.mark.parametrize("value", [0, 9, 65536, -1, True, False, 350., 350.5,
                                    float("nan"), float("inf"), "350", [], object()])
def test_invalid_requested_acceleration_is_rejected_before_any_io(value):
    driver = Registers()
    with pytest.raises(ValueError, match="integer from 10 to 65535"):
        configure_closed_loop_ramp(driver, value)
    assert driver.operations == []


def test_default_is_one_read_burst_and_never_reads_or_changes_mode():
    driver = Registers()
    result = configure_closed_loop_ramp(driver)
    assert driver.operations == [("read_pair", 0x005F, 2)]
    assert result == dict(before_acceleration_rpm_s=350, acceleration_rpm_s=350,
                          deceleration_rpm_s=450, requested_acceleration_rpm_s=None,
                          changed=False, persist=False, mode_policy=None)


@pytest.mark.parametrize("value", [10, 65535])
def test_ramp_readback_accepts_both_protocol_boundaries(value):
    driver = Registers()
    driver.values = [value, value]
    result = configure_closed_loop_ramp(driver)
    assert result["acceleration_rpm_s"] == result["deceleration_rpm_s"] == value


@pytest.mark.parametrize("values", [[0, 450], [350, 9], [65536, 450], [-1, 450],
                                     [True, 450], [350, False], [350., 450],
                                     [350, None], [], [350], [350, 450, 550]])
def test_invalid_initial_readback_cannot_be_used_as_current_value_or_written(values):
    driver = Registers()
    driver.values = values
    with pytest.raises(ValueError, match="readback"):
        configure_closed_loop_ramp(driver, 600)
    assert driver.operations == [("read_pair", 0x005F, 2)]


def test_initial_read_failure_propagates_without_write_or_retry():
    driver = Registers()
    error = driver.read_error = TimeoutError("first read")
    with pytest.raises(TimeoutError) as caught:
        configure_closed_loop_ramp(driver, 600)
    assert caught.value is error
    assert driver.operations == [("read_pair", 0x005F, 2)]


@pytest.mark.parametrize("mode,control", [(1, 0x08), (1, 0x18), (1, 0x28), (1, 0x38), (0, 0x38)])
def test_explicit_override_checks_compatible_mode_and_only_writes_volatile_acceleration(mode, control):
    driver = Registers()
    driver.bus.update(runtime_system_mode=mode, control_mode=control)
    driver.system_mode = mode
    result = configure_closed_loop_ramp(driver, 600)
    assert driver.operations == [("read_pair", 0x005F, 2), ("read_bus",),
                                 ("read_register", "system_mode"),
                                 ("write_register", "closed_loop_acceleration", 600, False),
                                 ("read_pair", 0x005F, 2)]
    assert result == dict(before_acceleration_rpm_s=350, acceleration_rpm_s=600,
                          deceleration_rpm_s=450, requested_acceleration_rpm_s=600,
                          changed=True, persist=False,
                          mode_policy="independent_closed_loop_1" if mode == 1 else "compat_mode_0_control_0x38")


def test_equal_acceleration_checks_mode_but_never_writes_or_repeats_ramp_read():
    driver = Registers()
    result = configure_closed_loop_ramp(driver, 350)
    assert not result["changed"] and result["requested_acceleration_rpm_s"] == 350
    assert driver.operations == [("read_pair", 0x005F, 2), ("read_bus",), ("read_register", "system_mode")]


@pytest.mark.parametrize("mode,configured,control", [
    (0, 0, 0), (0, 0, 0x30), (0, 0, 0x18), (0, 0, 0x39),
    (1, 1, 0x39), (1, 1, 0x3A), (1, 1, 0x3F), (1, 1, 0x78),
    (2, 2, 0x38), (3, 3, 0x38), (4, 4, 0x38), (9, 9, 0x38),
    (0, 1, 0x38), (1, 0, 0x38), (None, 1, 0x38), (1, None, 0x38),
    (True, 1, 0x38), (1, True, 0x38), (1, 1, True), (1, 1, None),
])
def test_invalid_mode_rejects_override_without_any_write(mode, configured, control):
    driver = Registers()
    driver.bus.update(runtime_system_mode=mode, control_mode=control)
    driver.system_mode = configured
    with pytest.raises(RuntimeError, match="未改变模式|拒绝自动改变") as caught:
        configure_closed_loop_ramp(driver, 600)
    assert not isinstance(caught.value, RampConfigurationError)
    assert not any(op[0] == "write_register" for op in driver.operations)


@pytest.mark.parametrize("bus", [None, {}, {"runtime_system_mode": 1}])
def test_shared_mode_helper_refuses_missing_status(bus):
    with pytest.raises(RuntimeError, match="数据缺失/非法"):
        validate_closed_loop_mode(bus, 1)


@pytest.mark.parametrize("method", ["read_bus_status", "read_register"])
def test_mode_read_failure_is_not_classified_as_attempted_parameter_write(method):
    driver = Registers()
    error = OSError("mode read failed")

    def fail(*args):
        raise error

    setattr(driver, method, fail)
    with pytest.raises(OSError) as caught:
        configure_closed_loop_ramp(driver, 600)
    assert caught.value is error
    assert not any(op[0] == "write_register" for op in driver.operations)


def test_write_failure_is_specific_latchable_error_without_retry_or_restore():
    driver = Registers()
    error = driver.write_error = TimeoutError("write ACK missing")
    with pytest.raises(RampConfigurationError, match="write not verified") as caught:
        configure_closed_loop_ramp(driver, 600)
    assert caught.value.__cause__ is error
    assert caught.value.write_attempted is True
    assert caught.value.diagnostics["requested_acceleration_rpm_s"] == 600
    assert caught.value.diagnostics["acceleration_rpm_s"] is None
    assert caught.value.diagnostics["deceleration_rpm_s"] is None
    assert caught.value.diagnostics["changed"] is None
    assert caught.value.diagnostics["verified"] is False
    assert driver.operations[-1] == ("write_register", "closed_loop_acceleration", 600, False)
    assert sum(op[0] == "write_register" for op in driver.operations) == 1
    assert sum(op[0] == "read_pair" for op in driver.operations) == 1


@pytest.mark.parametrize("readback", [[350, 450], [600, 451], [600, 0], [600., 450]])
def test_post_write_mismatch_or_invalid_readback_never_restores_any_setting(readback):
    driver = Registers()
    driver.after_write = lambda d: setattr(d, "values", readback)
    with pytest.raises(RampConfigurationError, match="readback") as caught:
        configure_closed_loop_ramp(driver, 600)
    assert caught.value.diagnostics["before_deceleration_rpm_s"] == 450
    assert caught.value.diagnostics["changed"] is None
    assert caught.value.diagnostics["verified"] is False
    assert [op for op in driver.operations if op[0] == "write_register"] == [
        ("write_register", "closed_loop_acceleration", 600, False)]
    assert sum(op[0] == "read_pair" for op in driver.operations) == 2


def test_post_write_read_failure_is_specific_latchable_error_without_retry():
    driver = Registers()
    error = TimeoutError("readback ACK missing")
    driver.after_write = lambda d: setattr(d, "read_error", error)
    with pytest.raises(RampConfigurationError) as caught:
        configure_closed_loop_ramp(driver, 600)
    assert caught.value.__cause__ is error
    assert sum(op[0] == "write_register" for op in driver.operations) == 1
    assert sum(op[0] == "read_pair" for op in driver.operations) == 2


def memory_driver(vendor):
    client, protocol = vendor
    wire = MemoryWire(protocol)
    wire.registers.update({0x005F: 350, 0x0060: 450, 0x001D: 1, 0x001E: 0x38, 0x0087: 1})
    driver = client.LZ30EMAClient(wire, slave=1)
    install_motor_rtu_guard(driver, baudrate=115200, timeout=.15)
    return driver, wire


def test_real_sdk_default_emits_one_03_burst_at_lz_addresses(vendor):
    driver, wire = memory_driver(vendor)
    assert configure_closed_loop_ramp(driver)["deceleration_rpm_s"] == 450
    assert len(wire.requests) == 1
    assert wire.requests[0][:6] == bytes.fromhex("01 03 00 5F 00 02")


@pytest.mark.parametrize("value,hex_value", [(600, "02 58"), (900, "03 84")])
def test_real_sdk_override_uses_only_06_at_005f_and_preserves_mode_and_deceleration(vendor, value, hex_value):
    driver, wire = memory_driver(vendor)
    result = configure_closed_loop_ramp(driver, value)
    assert result["changed"] and result["persist"] is False
    writes = [packet for packet in wire.requests if packet[1] != 3]
    assert len(writes) == 1
    assert writes[0][:6] == bytes.fromhex("01 06 00 5F " + hex_value)
    assert wire.registers[0x005F] == value
    assert wire.registers[0x0060] == 450 and wire.registers[0x0087] == 1
    ramp_reads = [packet for packet in wire.requests if packet[1:6] == bytes.fromhex("03 00 5F 00 02")]
    assert len(ramp_reads) == 2
