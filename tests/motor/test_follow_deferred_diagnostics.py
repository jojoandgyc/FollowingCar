"""Real periodic writer, fake clocks/driver: diagnostics never own serial time."""
import logging
import threading

from car_control_modular.deferred_diagnostics import deferred_diagnostics
from test_follow_wheel_periodic import setup_periodic


def live_writer(monkeypatch):
    runtime, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    state[:] = [24., 0., 10.25, 10.25]
    owner._depth30_linear_snapshot = ("forward", 24., 1, 10.)
    owner._depth_linear_max_age_sec = lambda kind: .25
    logger = logging.Logger("follow-deferred-offline", logging.INFO)
    runtime.logger = runtime.backend.logger = logger

    def depth(uid, now=None):
        logger.info("depth_reader_diagnostic sample_ts=10.0")
        return (("forward", 24., uid, 10.) if clock[0] <= 10.25 else None)

    owner._fresh_depth_linear_snapshot = depth
    return runtime, owner, driver, clock, logger


def test_depth_backend_and_runtime_records_dispatch_only_after_valid_motor_write(monkeypatch):
    runtime, owner, driver, clock, logger = live_writer(monkeypatch)
    messages = []
    timestamps = []

    class DelayedHandler(logging.Handler):
        def emit(self, record):
            assert not owner.motor_io_lock.locked()
            assert driver.pairs == [(24, -24)]
            messages.append(record.getMessage())
            timestamps.append(record.created)
            # Console handling would have consumed the complete original
            # lease if performed between planning and the physical write.
            clock[0] = 10.40

    logger.addHandler(DelayedHandler())
    runtime._service_follow_wheels()
    assert driver.pairs == [(24, -24)] and not driver.stops
    assert any(message.startswith("depth_reader_diagnostic") for message in messages)
    assert any("LZ30EMA 电机命令" in message for message in messages)
    assert any(message.startswith("visible_wheel_dispatch") for message in messages)
    assert any(message.startswith("follow_wheel_tick") for message in messages)
    assert set(timestamps) == {10.0}


def test_deferred_logging_does_not_hide_real_expiry_before_physical_write(monkeypatch):
    runtime, owner, driver, clock, logger = live_writer(monkeypatch)
    observations = []

    class Handler(logging.Handler):
        def emit(self, record):
            assert not owner.motor_io_lock.locked()
            observations.append(record.getMessage())

    logger.addHandler(Handler())
    checks = []

    def hard_stop(_action):
        checks.append(clock[0])
        if len(checks) == 2:
            clock[0] = 10.251
        return False

    runtime.hard_stop_check = hard_stop
    runtime._service_follow_wheels()
    assert checks and observations
    assert driver.pairs and all(pair == (0, 0) for pair in driver.pairs)


def test_blocked_periodic_diagnostic_sink_cannot_exclude_encoder_or_stop(monkeypatch):
    runtime, owner, driver, _clock, logger = live_writer(monkeypatch)
    entered, release, encoder_done, stop_written = (
        threading.Event(), threading.Event(), threading.Event(), threading.Event())

    class BlockedHandler(logging.Handler):
        def emit(self, record):
            assert not owner.motor_io_lock.locked()
            entered.set()
            assert release.wait(2.), "fake diagnostic sink was not released"

    logger.addHandler(BlockedHandler())
    errors = []

    def run_follow():
        try:
            runtime._service_follow_wheels()
        except BaseException as exc:
            errors.append(exc)

    def encoder_and_stop():
        try:
            # Like FOLLOW20, this diagnostic scope surrounds the I/O lock;
            # the fake STOP is observable before its own output can block.
            with deferred_diagnostics(logger):
                with owner.motor_io_lock:
                    encoder_done.set()
                    runtime.backend.send_stop("offline-concurrent-stop", mode="emergency",
                                              preserve_zero=True)
                stop_written.set()
        except BaseException as exc:
            errors.append(exc)

    follow = threading.Thread(target=run_follow, daemon=True)
    other = threading.Thread(target=encoder_and_stop, daemon=True)
    try:
        follow.start()
        assert entered.wait(1.)
        assert driver.pairs == [(24, -24)]
        other.start()
        assert encoder_done.wait(.5), "encoder excluded by periodic diagnostic output"
        assert stop_written.wait(.5), "STOP excluded by periodic diagnostic output"
        assert driver.stops == [1]
    finally:
        release.set()
        follow.join(1.)
        if other.ident is not None:
            other.join(1.)
    assert not errors
    assert not follow.is_alive() and not other.is_alive()
    assert driver.pairs == [(24, -24)]  # Never replay motion after the STOP.
