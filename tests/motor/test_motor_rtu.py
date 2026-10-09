"""In-memory RTU regression tests: no serial library or device is opened."""
import logging
import threading
from types import SimpleNamespace

import pytest

from car_control_modular.motor_rtu import (
    MotorRtuGuard, RtuLinkUncertain, RtuProtocolError, install_motor_rtu_guard,
)


def frame(payload):
    payload = bytes.fromhex(payload) if isinstance(payload, str) else bytes(payload)
    crc = 0xFFFF
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return payload + crc.to_bytes(2, "little")


RIGHT_SPEED = frame("01 10 00 41 00 02 04 00 00 00 64")
RIGHT_ZERO = frame("01 10 00 41 00 02 04 00 00 00 00")
LEFT_SPEED = frame("01 10 00 45 00 02 04 00 00 00 64")
LEFT_ZERO = frame("01 10 00 45 00 02 04 00 00 00 00")
RIGHT_ACK = frame("01 10 00 41 00 02")
LEFT_ACK = frame("01 10 00 45 00 02")
RIGHT_STOP = frame("01 06 00 40 00 01")
LEFT_STOP = frame("01 06 00 44 00 01")
READ_LEFT = frame("01 03 00 12 00 0A")
READ_DATA = frame(bytes.fromhex("01 03 14") + bytes(20))


class FakeClock:
    def __init__(self):
        self.now = 10.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.now += duration


class FakeSerial:
    def __init__(self, clock, responses=(), *, fragment_size=260):
        self.clock = clock
        self.responses = list(responses)
        self.pending = b""
        self.fragment_size = fragment_size
        self.timeout = 0.15
        self.write_timeout = None
        self.writes = []
        self.read_timeouts = []
        self.write_timeouts = []
        self.write_result = None
        self.read_delay = 0.0
        self.write_delay = 0.0
        self.resets = []
        self.gaps = []
        self.last_response_at = None

    @property
    def in_waiting(self):
        return len(self.pending)

    def reset_input_buffer(self):
        self.resets.append(self.pending)
        self.pending = b""

    def write(self, request):
        self.gaps.append(None if self.last_response_at is None else self.clock() - self.last_response_at)
        self.writes.append(request)
        self.write_timeouts.append(self.write_timeout)
        self.clock.now += self.write_delay
        self.pending += self.responses.pop(0) if self.responses else b""
        return len(request) if self.write_result is None else self.write_result

    def read(self, size):
        self.read_timeouts.append(self.timeout)
        if not self.pending:
            self.clock.now += self.timeout
            return b""
        self.clock.now += self.read_delay
        count = min(size, self.fragment_size)
        result, self.pending = self.pending[:count], self.pending[count:]
        self.last_response_at = self.clock()
        return result


def setup(*responses, baudrate=115200, timeout=.15, fragment_size=260):
    clock = FakeClock()
    serial = FakeSerial(clock, responses, fragment_size=fragment_size)
    guard = MotorRtuGuard(serial, baudrate=baudrate, timeout=timeout,
                          clock=clock, sleep=clock.sleep)
    return guard, serial, clock


@pytest.mark.parametrize("baudrate,gap", [(115200, .00175), (19200, 3.5 * 11 / 19200),
                                          (9600, 3.5 * 11 / 9600)])
def test_back_to_back_wheel_requests_observe_rtu_quiet_time(baudrate, gap):
    guard, serial, clock = setup(RIGHT_ACK, LEFT_ACK, baudrate=baudrate)
    assert guard.transact(RIGHT_SPEED, 8) == RIGHT_ACK
    assert guard.transact(LEFT_SPEED, 8) == LEFT_ACK
    assert serial.gaps[1] == pytest.approx(gap)
    assert clock.sleeps == pytest.approx([gap])
    assert not guard.rx_uncertain
    assert serial.timeout == .15 and serial.write_timeout is None


def test_elapsed_idle_time_does_not_add_an_extra_sleep():
    guard, serial, clock = setup(RIGHT_ACK, LEFT_ACK)
    guard.transact(RIGHT_SPEED, 8)
    clock.now += .03
    guard.transact(LEFT_SPEED, 8)
    assert not clock.sleeps
    assert serial.gaps[1] == pytest.approx(.03)


def test_fragments_share_one_deadline_and_restore_original_timeouts():
    guard, serial, _ = setup(RIGHT_ACK, fragment_size=1)
    serial.timeout = .9
    serial.write_timeout = .8
    serial.read_delay = .002
    assert guard.transact(RIGHT_SPEED, 8) == RIGHT_ACK
    assert len(serial.read_timeouts) == 8
    assert serial.read_timeouts == sorted(serial.read_timeouts, reverse=True)
    assert serial.read_timeouts[-1] == pytest.approx(.136)
    assert serial.write_timeouts == pytest.approx([.15])
    assert serial.timeout == .9 and serial.write_timeout == .8
    assert len(serial.writes) == 1


def test_partial_frame_consumes_only_remaining_budget_and_never_retries():
    guard, serial, clock = setup(RIGHT_ACK[:4], fragment_size=1)
    serial.read_delay = .02
    started = clock()
    with pytest.raises(TimeoutError, match="incomplete"):
        guard.transact(RIGHT_SPEED, 8)
    assert clock() - started == pytest.approx(.15)
    assert serial.read_timeouts[-1] == pytest.approx(.07)
    assert len(serial.writes) == 1
    assert serial.timeout == .15 and serial.write_timeout is None
    assert guard.last_transaction["rx_hex"] == RIGHT_ACK[:4].hex()
    assert guard.rx_uncertain and "stage=read" in guard.fault_reason


def test_quiet_time_is_inside_budget_and_can_expire_before_transmission():
    guard, serial, clock = setup(RIGHT_ACK, LEFT_ACK, timeout=.001)
    guard.transact(RIGHT_SPEED, 8)
    started = clock()
    with pytest.raises(TimeoutError, match="deadline"):
        guard.transact(LEFT_SPEED, 8)
    assert clock() - started == pytest.approx(.001)
    assert serial.writes == [RIGHT_SPEED]
    assert guard.last_transaction["stage"] == "quiet"


def test_short_write_is_reported_before_any_read_or_retry():
    guard, serial, _ = setup(RIGHT_ACK)
    serial.write_result = 5
    with pytest.raises(RtuProtocolError, match="short serial write: 5 of 13"):
        guard.transact(RIGHT_SPEED, 8)
    assert not serial.read_timeouts
    assert serial.writes == [RIGHT_SPEED]
    assert guard.last_transaction["tx_written"] == 5
    assert "stage=write" in guard.fault_reason


def test_write_overrun_detected_without_a_new_full_read_timeout():
    guard, serial, _ = setup(RIGHT_ACK)
    serial.write_delay = .16
    with pytest.raises(TimeoutError, match="deadline"):
        guard.transact(RIGHT_SPEED, 8)
    assert not serial.read_timeouts
    assert serial.write_timeouts == pytest.approx([.15])


@pytest.mark.parametrize("response,message", [
    (frame("02 10 00 41 00 02"), "slave"),
    (frame("01 06 00 41 00 02"), "function"),
    (LEFT_ACK, "register/value/count"),
    (frame("01 10 00 41 00 01"), "register/value/count"),
    (RIGHT_ACK[:-1] + bytes([RIGHT_ACK[-1] ^ 1]), "CRC"),
    (frame("01 90 06"), "Modbus exception 0x06"),
])
def test_wrong_frames_latch_uncertainty(response, message):
    guard, serial, _ = setup(response)
    with pytest.raises(RtuProtocolError, match=message):
        guard.transact(RIGHT_SPEED, 8)
    assert guard.rx_uncertain
    assert serial.writes == [RIGHT_SPEED]
    assert serial.timeout == .15


def test_exception_frame_is_recognized_at_five_bytes_without_waiting_for_eight():
    guard, serial, clock = setup(frame("01 90 02"))
    started = clock()
    with pytest.raises(RtuProtocolError, match="Modbus exception"):
        guard.transact(RIGHT_SPEED, 8)
    assert clock() == started
    assert len(serial.read_timeouts) == 2


def test_stop_must_echo_exact_mode():
    guard, _, _ = setup(frame("01 06 00 40 00 00"))
    with pytest.raises(RtuProtocolError, match="register/value/count"):
        guard.transact(RIGHT_STOP, 8)


def test_read_validates_byte_count_against_current_request():
    guard, _, _ = setup(frame("01 03 02 00 00"))
    with pytest.raises(RtuProtocolError, match="length"):
        guard.transact(READ_LEFT, 25)


@pytest.mark.parametrize("blocked", [RIGHT_SPEED, RIGHT_ZERO, LEFT_SPEED, LEFT_ZERO,
                                      frame("01 06 00 40 00 00"),
                                      frame("01 06 00 44 00 02"),
                                      frame("01 06 00 6D 01 F4"),
                                      frame("01 10 00 76 00 01 02 01 F4")])
def test_late_ack_cannot_be_accepted_as_zero_or_new_motion_ack(blocked):
    guard, serial, _ = setup(b"")
    with pytest.raises(TimeoutError):
        guard.transact(RIGHT_SPEED, 8)
    first_fault = guard.fault_reason
    serial.pending = RIGHT_ACK  # Late ACK for the earlier 100 RPM request.
    with pytest.raises(RtuLinkUncertain, match="blocked"):
        guard.transact(blocked, 8)
    assert serial.writes == [RIGHT_SPEED]
    assert serial.pending == RIGHT_ACK
    assert guard.fault_reason == first_fault


@pytest.mark.parametrize("safe", [RIGHT_STOP, LEFT_STOP,
                                   frame("01 06 00 6D 00 00"),
                                   frame("01 06 00 76 00 00"),
                                   frame("01 10 00 6D 00 01 02 00 00"),
                                   frame("01 10 00 76 00 01 02 00 00")])
def test_safe_fault_writes_discard_available_old_input_without_clearing_fault(safe):
    ack = safe if safe[1] == 6 else frame(safe[:6])
    guard, serial, _ = setup(b"", ack)
    with pytest.raises(TimeoutError):
        guard.transact(RIGHT_SPEED, 8)
    first_fault = guard.fault_reason
    serial.pending = RIGHT_ACK
    assert guard.transact(safe, 8) == ack
    assert serial.resets == [RIGHT_ACK]
    assert serial.writes == [RIGHT_SPEED, safe]
    assert guard.rx_uncertain and guard.fault_reason == first_fault
    assert guard.last_transaction["rx_reset"] == "performed"
    assert guard.last_transaction["rx_discard_available"] == 8
    assert guard.last_transaction["quiet_wait_ms"] >= 1.75 - 1e-8


def test_read_timeout_also_blocks_motion_and_new_reads_stay_diagnostic():
    guard, serial, _ = setup(b"", READ_DATA)
    with pytest.raises(TimeoutError):
        guard.transact(READ_LEFT, 25)
    with pytest.raises(RtuLinkUncertain, match="blocked"):
        guard.transact(RIGHT_SPEED, 8)
    with pytest.raises(RtuLinkUncertain, match="diagnostic response"):
        guard.transact(READ_LEFT, 25)
    assert serial.writes == [READ_LEFT, READ_LEFT]
    assert guard.last_transaction["rx_hex"] == READ_DATA.hex()


def test_reset_unsupported_is_explicit_and_old_read_cannot_ack_stop():
    guard, serial, _ = setup(b"", READ_DATA, LEFT_STOP)
    serial.reset_input_buffer = None
    guard._reset_owner = None
    with pytest.raises(TimeoutError):
        guard.transact(READ_LEFT, 25)
    with pytest.raises(RtuProtocolError, match="function"):
        guard.transact(RIGHT_STOP, 8)
    assert guard.last_transaction["rx_reset"] == "unsupported"
    assert guard.transact(LEFT_STOP, 8) == LEFT_STOP
    assert serial.writes == [READ_LEFT, RIGHT_STOP, LEFT_STOP]
    assert guard.rx_uncertain


def test_failure_logs_recent_bounded_success_window_and_transaction_fields(caplog):
    guard, serial, _ = setup(*([RIGHT_ACK] * 15), b"")
    with caplog.at_level(logging.INFO):
        for _ in range(15):
            guard.transact(RIGHT_SPEED, 8)
    assert not caplog.records
    with caplog.at_level(logging.ERROR), pytest.raises(TimeoutError):
        guard.transact(LEFT_SPEED, 8)
    assert len(guard.recent) == 12
    message = caplog.text
    for field in ("motor_rtu_transaction_failed", "0x0045", "tx_hex", "rx_hex",
                  "tx_written", "quiet_wait_ms", "read_calls", "recent="):
        assert field in message
    assert len(serial.writes) == 16


def test_install_is_idempotent_and_leaves_nontransport_test_adapters_untouched():
    fake = SimpleNamespace(set_right_speed=lambda _: None)
    assert install_motor_rtu_guard(fake, baudrate=115200, timeout=.15) is None
    clock = FakeClock()
    serial = FakeSerial(clock)
    driver = SimpleNamespace(transport=serial, _transact=lambda *_: b"legacy")
    guard = install_motor_rtu_guard(driver, baudrate=115200, timeout=.15)
    assert driver._transact == guard.transact
    assert install_motor_rtu_guard(driver, baudrate=115200, timeout=.15) is guard


def test_wrapped_transport_uses_serial_write_timeout_and_reset_capabilities():
    clock = FakeClock()
    serial = FakeSerial(clock, [b"", RIGHT_STOP])
    wrapper = SimpleNamespace(_serial=serial, timeout=.15, write=serial.write,
                              read=lambda size: setattr(serial, "timeout", wrapper.timeout) or serial.read(size))
    guard = MotorRtuGuard(wrapper, baudrate=115200, timeout=.15, clock=clock, sleep=clock.sleep)
    with pytest.raises(TimeoutError):
        guard.transact(RIGHT_SPEED, 8)
    serial.pending = RIGHT_ACK
    assert guard.transact(RIGHT_STOP, 8) == RIGHT_STOP
    assert serial.resets == [RIGHT_ACK]
    assert serial.write_timeout is None and wrapper.timeout == .15


def test_broken_log_sink_preserves_original_error_and_does_not_leak_lock():
    guard, serial, _ = setup(b"", RIGHT_STOP, LEFT_STOP)

    def broken(*args, **kwargs):
        raise OSError("log sink failed")

    guard.logger = SimpleNamespace(error=broken, warning=broken)
    with pytest.raises(TimeoutError, match="no response"):
        guard.transact(RIGHT_SPEED, 8)
    assert guard._lock.acquire(blocking=False)
    guard._lock.release()
    assert guard.transact(RIGHT_STOP, 8) == RIGHT_STOP
    assert guard.transact(LEFT_STOP, 8) == LEFT_STOP
    assert serial.writes == [RIGHT_SPEED, RIGHT_STOP, LEFT_STOP]
    assert guard.rx_uncertain


def test_whole_write_read_transactions_share_one_lock():
    clock = FakeClock()
    serial = FakeSerial(clock, [RIGHT_ACK, LEFT_ACK])
    first_read_entered = threading.Event()
    release_first_read = threading.Event()
    second_started = threading.Event()
    original_read = serial.read

    def controlled_read(size):
        if not first_read_entered.is_set():
            first_read_entered.set()
            assert release_first_read.wait(.5)
        return original_read(size)

    serial.read = controlled_read
    guard = MotorRtuGuard(serial, baudrate=115200, timeout=.15, clock=clock, sleep=clock.sleep)
    results = {}

    def run(request, signal=None):
        if signal is not None:
            signal.set()
        try:
            results[request] = guard.transact(request, 8)
        except Exception as exc:
            results[request] = exc

    first = threading.Thread(target=run, args=(RIGHT_SPEED,))
    second = threading.Thread(target=run, args=(LEFT_SPEED, second_started))
    first.start()
    try:
        assert first_read_entered.wait(.5)
        second.start()
        assert second_started.wait(.5)
        assert serial.writes == [RIGHT_SPEED]
    finally:
        release_first_read.set()
        first.join(.5)
        if second.ident is not None:
            second.join(.5)
    assert not first.is_alive() and not second.is_alive()
    assert results == {RIGHT_SPEED: RIGHT_ACK, LEFT_SPEED: LEFT_ACK}
    assert serial.writes == [RIGHT_SPEED, LEFT_SPEED]


def test_serial_write_exception_still_leaves_quiet_before_emergency_stop():
    guard, serial, clock = setup(RIGHT_STOP)
    original_write = serial.write
    attempts = []
    failed_at = []

    def failing_write(request):
        attempts.append(request)
        if request == RIGHT_SPEED:
            clock.now += .01
            failed_at.append(clock())
            raise OSError("partial TX then driver failure")
        assert clock() - failed_at[0] >= .00175 - 1e-8
        return original_write(request)

    serial.write = failing_write
    guard._reset_owner = None
    with pytest.raises(OSError, match="partial TX"):
        guard.transact(RIGHT_SPEED, 8)
    assert guard.transact(RIGHT_STOP, 8) == RIGHT_STOP
    assert attempts == [RIGHT_SPEED, RIGHT_STOP]
