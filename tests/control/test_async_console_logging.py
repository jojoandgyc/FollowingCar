"""Slow diagnostic output cannot hold the controller or shutdown hostage."""
import io
import logging
import threading
import time

from car_control_modular.async_console_logging import BoundedAsyncConsoleHandler


class BlockedStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def write(self, value):
        self.entered.set()
        assert self.release.wait(2.0), "test did not release fake stream"
        return super().write(value)


def record(message, args=()):
    return logging.LogRecord("test", logging.INFO, __file__, 1, message, args, None)


def test_blocked_sink_does_not_hold_producer_or_control_lock_and_overflow_is_audited():
    stream = BlockedStream()
    handler = BoundedAsyncConsoleHandler(stream=stream, capacity=2)
    control_lock = threading.Lock()
    done = threading.Event()
    try:
        handler.handle(record("first"))
        assert stream.entered.wait(1.0)

        def produce():
            with control_lock:
                for index in range(5):
                    handler.handle(record("queued %d", (index,)))
            done.set()

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        assert done.wait(0.3), "producer waited on blocked console stream"
        assert control_lock.acquire(timeout=0.1)
        control_lock.release()
        assert handler.stats()["queued"] == 2
        assert handler.stats()["dropped"] == 3
        stream.release.set()
        assert handler.stop(1.0)
        output = stream.getvalue()
        assert output.index("first") < output.index("queued 0") < output.index("queued 1")
        assert "async_console_health dropped=3" in output
        assert handler.stats()["pending"] == 0
    finally:
        stream.release.set()
        handler.close()


def test_timestamp_message_arguments_and_exception_are_frozen_before_writer():
    stream = BlockedStream()
    handler = BoundedAsyncConsoleHandler(
        stream=stream, formatter=logging.Formatter("%(created).3f %(levelname)s %(message)s"),
    )
    try:
        handler.handle(record("block"))
        assert stream.entered.wait(1.0)
        mutable = {"base": 30}
        original = record("command=%s", (mutable,))
        original_args = original.args
        original.created = 123.456
        handler.handle(original)
        mutable["base"] = 99
        try:
            raise ValueError("sample fault")
        except ValueError:
            import sys
            handler.handle(logging.LogRecord("test", logging.ERROR, __file__, 1,
                                             "fault %s", ("kept",), sys.exc_info()))
        stream.release.set()
        assert handler.stop(1.0)
        output = stream.getvalue()
        assert "123.456 INFO command={'base': 30}" in output
        assert "base': 99" not in output
        assert "fault kept\nTraceback" in output
        assert "ValueError: sample fault" in output
        assert original.args is original_args, "preparation mutated caller record"
    finally:
        stream.release.set()
        handler.close()


def test_flush_stop_and_logging_shutdown_are_bounded_with_blocked_sink():
    stream = BlockedStream()
    handler = BoundedAsyncConsoleHandler(stream=stream, shutdown_timeout_sec=0.02)
    try:
        handler.handle(record("pending"))
        assert stream.entered.wait(1.0)
        start = time.monotonic()
        assert not handler.flush()
        assert not handler.stop()
        # logging.shutdown holds the queue handler lock while flushing/closing.
        # The daemon writer must not need that lock to finish its own output.
        import weakref
        logging.shutdown([weakref.ref(handler)])
        assert time.monotonic() - start < 0.3
        assert handler._worker.daemon
        assert handler.stats()["pending"] == 1
        assert handler.stats()["stop_timeouts"] >= 2
        handler.handle(record("rejected"))
        assert handler.stats()["rejected_after_stop"] == 1
        stream.release.set()
        assert handler.stop(1.0)
        assert "rejected" not in stream.getvalue()
        assert "stop_timeouts=" in stream.getvalue()
    finally:
        stream.release.set()
        handler.close()


def test_broken_sink_is_counted_and_writer_recovers_without_stderr_fallback(capsys):
    class RecoveringStream(io.StringIO):
        calls = 0

        def write(self, value):
            self.calls += 1
            if self.calls <= 2:
                raise OSError("broken test pipe")
            return super().write(value)

    stream = RecoveringStream()
    handler = BoundedAsyncConsoleHandler(stream=stream, failure_backoff_sec=0.01)
    try:
        handler.handle(record("lost"))
        handler.handle(record("recovered"))
        assert handler.stop(1.0)
        assert handler.stats()["write_errors"] == 2
        assert "recovered" in stream.getvalue()
        assert "write_errors=2" in stream.getvalue()
        assert capsys.readouterr().err == ""
    finally:
        handler.close()


def test_bad_format_is_counted_without_synchronous_handle_error(capsys):
    stream = io.StringIO()
    handler = BoundedAsyncConsoleHandler(stream=stream)
    try:
        handler.handle(record("%d", ("invalid",)))
        handler.handle(record("valid"))
        assert handler.stop(1.0)
        assert handler.stats()["prepare_errors"] == 1
        assert "valid" in stream.getvalue()
        assert "prepare_errors=1" in stream.getvalue()
        assert capsys.readouterr().err == ""
    finally:
        handler.close()
