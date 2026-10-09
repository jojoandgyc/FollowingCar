"""Diagnostic handler work is deferred; no motor or serial device is opened."""
import logging
import threading

import pytest

from car_control_modular.deferred_diagnostics import deferred_diagnostics


class RecordingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def logger_with_handler(handler=None):
    logger = logging.Logger("isolated-deferred-test", logging.DEBUG)
    handler = handler or RecordingHandler()
    logger.addHandler(handler)
    return logger, handler


def test_records_keep_original_timestamp_and_format_only_after_scope(monkeypatch):
    logger, handler = logger_with_handler()
    clock = [100.]
    monkeypatch.setattr(logging.time, "time", lambda: clock[0])
    formatted = []

    class Argument:
        def __str__(self):
            formatted.append(clock[0])
            return "original"

    with deferred_diagnostics(logger) as stats:
        logger.info("value=%s", Argument())
        assert not handler.records and not formatted
        assert stats.buffered == 1
        clock[0] = 102.
    assert stats.captured == stats.dispatched == 1
    assert stats.buffered == 0 and stats.dropped == stats.dispatch_errors == 0
    assert handler.records[0].created == 100.
    assert handler.records[0].getMessage() == "value=original"
    assert formatted == [102.]


def test_exception_trace_is_preserved_and_control_exception_is_not_swallowed():
    logger, handler = logger_with_handler()
    with pytest.raises(ValueError, match="control failed"):
        with deferred_diagnostics(logger):
            try:
                raise ValueError("control failed")
            except ValueError:
                logger.exception("control diagnostic")
                raise
    record, = handler.records
    assert record.exc_info[0] is ValueError
    assert record.exc_info[2] is not None
    assert "ValueError: control failed" in logging.Formatter().format(record)


def test_nested_scopes_share_capacity_and_do_not_flush_inside_outer_lock():
    logger, handler = logger_with_handler()
    lock = threading.Lock()
    with deferred_diagnostics(logger, capacity=3) as outer:
        with lock:
            logger.info("before")
            with deferred_diagnostics(logger, capacity=1) as inner:
                assert inner is outer
                logger.info("inside")
            logger.info("after")
            assert handler.records == []
    assert [r.getMessage() for r in handler.records] == ["before", "inside", "after"]
    assert outer.captured == 3 and not outer.dropped


def test_other_thread_using_same_logger_is_not_deferred():
    logger, handler = logger_with_handler()
    with deferred_diagnostics(logger):
        logger.info("deferred-main")
        other = threading.Thread(target=logger.info, args=("immediate-other",))
        other.start()
        other.join(1.)
        assert not other.is_alive()
        assert [r.getMessage() for r in handler.records] == ["immediate-other"]
    assert [r.getMessage() for r in handler.records] == ["immediate-other", "deferred-main"]


def test_other_logger_is_unchanged_and_nested_loggers_dispatch_in_capture_order():
    logger, handler = logger_with_handler()
    other, other_handler = logger_with_handler()
    with deferred_diagnostics(logger):
        logger.info("one")
        other.info("unscoped")
        assert len(other_handler.records) == 1
        with deferred_diagnostics(other):
            other.info("two")
            assert len(other_handler.records) == 1
        other.info("unscoped-again")
        assert len(other_handler.records) == 2
        logger.info("three")
    assert [r.getMessage() for r in handler.records] == ["one", "three"]
    assert [r.getMessage() for r in other_handler.records] == ["unscoped", "unscoped-again", "two"]


def test_bounded_buffer_overflow_is_counted_and_reported_after_scope():
    logger, handler = logger_with_handler()
    with deferred_diagnostics(logger, capacity=2) as stats:
        for index in range(10):
            logger.info("item=%d", index)
        assert stats.buffered == stats.captured == 2
        assert stats.dropped == 8
        assert not handler.records
    assert [r.getMessage() for r in handler.records[:2]] == ["item=0", "item=1"]
    assert "deferred_diagnostics_dropped count=8 capacity=2" in handler.records[2].getMessage()
    assert stats.dispatched == 3 and stats.buffered == 0


def test_blocked_handler_after_scope_does_not_hold_motor_lock_or_delay_stop_owner():
    entered, release, stop_completed = threading.Event(), threading.Event(), threading.Event()
    motor_lock = threading.Lock()

    class BlockingHandler(RecordingHandler):
        def emit(self, record):
            entered.set()
            assert release.wait(2.), "fake handler was not released"
            super().emit(record)

    logger, handler = logger_with_handler(BlockingHandler())
    write_events = []

    def writer():
        with deferred_diagnostics(logger):
            with motor_lock:
                logger.info("inside calculation")
                assert not entered.is_set()
                write_events.append("valid-write")
                logger.info("write completed")

    def stop_owner():
        with motor_lock:
            write_events.append("STOP")
        stop_completed.set()

    writer_thread = threading.Thread(target=writer, daemon=True)
    stopper = threading.Thread(target=stop_owner, daemon=True)
    try:
        writer_thread.start()
        assert entered.wait(1.)
        stopper.start()
        assert stop_completed.wait(.5), "diagnostic flush retained the motor lock"
        assert write_events == ["valid-write", "STOP"]
    finally:
        release.set()
        writer_thread.join(1.)
        if stopper.ident is not None:
            stopper.join(1.)
    assert not writer_thread.is_alive()
    assert [r.getMessage() for r in handler.records] == ["inside calculation", "write completed"]


def test_other_thread_holding_handler_lock_does_not_block_capture_inside_motor_lock():
    sink_entered, sink_release, motor_body_done = (
        threading.Event(), threading.Event(), threading.Event())
    motor_lock = threading.Lock()

    class BusyHandler(logging.Handler):
        def emit(self, record):
            if record.getMessage() == "other thread holds handler":
                sink_entered.set()
                assert sink_release.wait(2.)

    logger, _ = logger_with_handler(BusyHandler())
    other = threading.Thread(target=logger.info, args=("other thread holds handler",), daemon=True)

    def producer():
        with deferred_diagnostics(logger):
            with motor_lock:
                logger.info("must not acquire the busy handler lock")
            motor_body_done.set()

    producer_thread = threading.Thread(target=producer, daemon=True)
    try:
        other.start()
        assert sink_entered.wait(1.)
        producer_thread.start()
        assert motor_body_done.wait(.5), "capture tried to take a held logging handler lock"
        assert motor_lock.acquire(timeout=.1)
        motor_lock.release()
    finally:
        sink_release.set()
        other.join(1.)
        if producer_thread.ident is not None:
            producer_thread.join(1.)
    assert not producer_thread.is_alive()


def test_failed_dispatch_does_not_replace_control_exception_or_skip_cleanup():
    class FailedHandler(logging.Handler):
        def emit(self, record):
            raise OSError("diagnostic sink unavailable")

    logger, _ = logger_with_handler(FailedHandler())
    cleanup = []
    with pytest.raises(RuntimeError, match="control failure"):
        try:
            with deferred_diagnostics(logger, capacity=1) as stats:
                logger.info("first")
                logger.info("overflow")
                raise RuntimeError("control failure")
        finally:
            cleanup.append("STOP")
    assert cleanup == ["STOP"]
    assert stats.dispatch_errors == 2 and stats.buffered == 0


def test_scope_is_disabled_after_failure_and_installs_only_one_filter():
    logger, handler = logger_with_handler()
    for _ in range(3):
        with pytest.raises(ValueError):
            with deferred_diagnostics(logger):
                logger.info("deferred")
                raise ValueError()
        logger.info("immediate")
    assert len(logger.filters) == 1
    assert len(handler.records) == 6


def test_adapter_without_filter_protocol_is_compatible_noop():
    logger, handler = logger_with_handler()
    adapter = logging.LoggerAdapter(logger, {})
    with deferred_diagnostics(adapter) as stats:
        adapter.info("ordinary-adapter")
        assert len(handler.records) == 1
    assert stats.captured == stats.dispatched == stats.dropped == 0


def test_custom_filter_installation_failure_does_not_skip_control_body():
    class FailingLogger(logging.Logger):
        def addFilter(self, capture):
            raise OSError("custom diagnostic setup failed")

    logger = FailingLogger("custom-logger", logging.INFO)
    ran = []
    with deferred_diagnostics(logger):
        ran.append("control body and STOP remain reachable")
    assert ran == ["control body and STOP remain reachable"]


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True, None, 257])
def test_invalid_capacity_is_rejected(capacity):
    logger, _ = logger_with_handler()
    with pytest.raises(ValueError):
        with deferred_diagnostics(logger, capacity=capacity):
            pytest.fail("invalid capacity entered scope")
