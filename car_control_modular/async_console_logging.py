"""Bounded console output: runtime threads never write to the log stream.

Install explicitly at application startup, not when importing the runtime.
The queue handler freezes messages with the standard QueueHandler.prepare;
the daemon writer retains the original LogRecord timestamp when formatting.
Slow/broken output can lose diagnostics, never extend a motor/depth deadline.
"""
import logging
from logging.handlers import QueueHandler
import queue
import sys
import threading
import time


class BoundedAsyncConsoleHandler(QueueHandler):
    """Nonblocking bounded enqueue with observable loss and bounded shutdown.

``formatter`` belongs to the writer. Leave the inherited handler formatter
unset so QueueHandler.prepare freezes just the message/exception text, rather
than inserting a second timestamp. ``stop`` should run after motor shutdown;
it returns False if the writer cannot drain within its budget. ``stats`` stays
available even when the stream is permanently blocked.
"""

    def __init__(self, *, stream=None, formatter=None, capacity=4096,
                 shutdown_timeout_sec=0.5, failure_backoff_sec=0.05):
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("async logging capacity must be positive")
        super().__init__(queue.Queue(maxsize=capacity))
        self.stream = sys.stderr if stream is None else stream
        self.output_formatter = formatter or logging.Formatter("%(message)s")
        self.shutdown_timeout_sec = max(0.0, float(shutdown_timeout_sec))
        self.failure_backoff_sec = max(0.01, float(failure_backoff_sec))
        self._state = threading.Condition()
        self._stop_requested = threading.Event()
        self._accepting = True
        self._accepted = self._completed = self._written = 0
        self._dropped = self._write_errors = self._prepare_errors = 0
        self._rejected_after_stop = self._stop_timeouts = 0
        self._last_error = None
        self._reported_health = (0, 0, 0, 0)
        self._last_health_attempt = 0.0
        # No registered StreamHandler sink: logging.shutdown must never wait
        # for a sink handler lock held by a blocked stream.write/flush call.
        self._worker = threading.Thread(
            target=self._write_loop, name="runtime-console-log", daemon=True,
        )
        self._worker.start()

    def emit(self, record):
        try:
            prepared = self.prepare(record)
        except Exception as exc:
            # QueueHandler's default handleError writes synchronously to
            # stderr. Even malformed diagnostics must not enter that path.
            with self._state:
                self._prepare_errors += 1
                self._last_error = type(exc).__name__
            return
        with self._state:
            if not self._accepting:
                self._rejected_after_stop += 1
                return
            try:
                self.queue.put_nowait(prepared)
            except queue.Full:
                self._dropped += 1
                return
            self._accepted += 1

    def stats(self):
        with self._state:
            return dict(
                capacity=self.queue.maxsize, queued=self.queue.qsize(),
                accepted=self._accepted, written=self._written,
                pending=self._accepted - self._completed,
                dropped=self._dropped, write_errors=self._write_errors,
                prepare_errors=self._prepare_errors,
                rejected_after_stop=self._rejected_after_stop,
                stop_timeouts=self._stop_timeouts,
                last_error=self._last_error, accepting=self._accepting,
                worker_alive=self._worker.is_alive(),
            )

    def _write(self, record):
        try:
            self.stream.write(self.output_formatter.format(record) + "\n")
            self.stream.flush()
            return True
        except Exception as exc:
            with self._state:
                self._write_errors += 1
                self._last_error = type(exc).__name__
            return False

    def _report_health(self, *, final=False):
        now = time.monotonic()
        with self._state:
            health = (self._dropped, self._write_errors, self._prepare_errors, self._stop_timeouts)
            last_error = self._last_error
        if health == self._reported_health or (not final and now-self._last_health_attempt < 1.0):
            return
        self._last_health_attempt = now
        record = logging.LogRecord(
            "PersonTracker", logging.WARNING, __file__, 0,
            "async_console_health dropped=%d write_errors=%d prepare_errors=%d "
            "stop_timeouts=%d last_error=%s capacity=%d producer_io=False",
            (*health, last_error, self.queue.maxsize), None,
        )
        if self._write(record):
            self._reported_health = health

    def _write_loop(self):
        while True:
            try:
                record = self.queue.get(timeout=0.02)
            except queue.Empty:
                if self._stop_requested.is_set():
                    self._report_health(final=True)
                    return
                self._report_health()
                continue
            success = self._write(record)
            self.queue.task_done()
            with self._state:
                self._completed += 1
                self._written += int(success)
                self._state.notify_all()
            if not success:
                # A failed sink is not a reason to spin or print a traceback
                # from a controller thread. Retry future records at <=20Hz.
                time.sleep(self.failure_backoff_sec)
            self._report_health()

    def flush(self, timeout_sec=None):
        """Wait only for records already accepted; never flush the sink here."""
        budget = self.shutdown_timeout_sec if timeout_sec is None else max(0.0, float(timeout_sec))
        deadline = time.monotonic() + budget
        with self._state:
            target = self._accepted
            while self._completed < target:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._state.wait(remaining)
        return True

    def stop(self, timeout_sec=None):
        """Stop accepting and drain within the budget; blocked writer is daemon."""
        budget = self.shutdown_timeout_sec if timeout_sec is None else max(0.0, float(timeout_sec))
        with self._state:
            self._accepting = False
            self._stop_requested.set()
        if threading.current_thread() is not self._worker:
            self._worker.join(budget)
        stopped = not self._worker.is_alive()
        if not stopped:
            with self._state:
                self._stop_timeouts += 1
        return stopped

    def close(self):
        self.stop()
        super().close()
