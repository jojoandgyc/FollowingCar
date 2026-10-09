"""Bounded, thread-local diagnostic deferral around a motor critical section.

The scope must surround (not be inside) the motor lock::

    with deferred_diagnostics(logger):
        with motor_io_lock:
            ...

No handler, formatter or output stream is called by this filter while the
scope is active. Original LogRecords retain their event timestamps and
exception information. The outermost scope dispatches after its body exits;
it does not introduce a thread, queue, retry, or synchronous error fallback.
It cannot make an arbitrary blocking handler nonblocking: dispatch occurs on
the caller after the critical section. Install the normal async sink as well.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
import threading
import weakref


_local = threading.local()
_installation_lock = threading.Lock()
_filters = weakref.WeakKeyDictionary()
_MAX_RECORDS = 256


@dataclass
class DeferredDiagnosticStats:
    """Per-outermost-scope counts; no records are retained after dispatch."""

    capacity: int
    captured: int = 0
    dropped: int = 0
    dispatched: int = 0
    dispatch_errors: int = 0
    _records: list = field(default_factory=list, repr=False)
    _active_filters: dict = field(default_factory=dict, repr=False)
    _drop_logger: object = field(default=None, repr=False)

    @property
    def buffered(self):
        return len(self._records)


class _ThreadLocalCapture(logging.Filter):
    def __init__(self, logger):
        super().__init__()
        self._logger_ref = weakref.ref(logger)

    def filter(self, record):
        batch = getattr(_local, "batch", None)
        if batch is None or self not in batch._active_filters:
            return True
        logger = self._logger_ref()
        if logger is None:
            return True
        if len(batch._records) < batch.capacity:
            batch._records.append((logger, record))
            batch.captured += 1
        else:
            batch.dropped += 1
            if batch._drop_logger is None:
                batch._drop_logger = logger
        return False


def _capture_filter(logger):
    # Simple logging adapters/test doubles do not necessarily implement the
    # Logger filter protocol. Leave them unchanged instead of replacing any
    # global logging methods or intercepting other threads' diagnostics.
    try:
        if (not callable(getattr(logger, "addFilter", None))
                or not callable(getattr(logger, "handle", None))):
            return None
        with _installation_lock:
            capture = _filters.get(logger)
            if capture is None:
                capture = _ThreadLocalCapture(logger)
                logger.addFilter(capture)
                _filters[logger] = capture
            return capture
    except Exception:
        # Includes non-weak-referenceable/non-hashable adapters and custom
        # logger setup failures. Diagnostics must not abort motor cleanup.
        return None


def _dispatch(batch):
    records, batch._records = batch._records, []
    for logger, record in records:
        try:
            logger.handle(record)
            batch.dispatched += 1
        except Exception:
            # Diagnostic output failure may not replace a control exception
            # or skip the caller's STOP/finally cleanup. No fallback logging:
            # that would re-enter the same failed sink.
            batch.dispatch_errors += 1
    if batch.dropped and batch._drop_logger is not None:
        try:
            record = logging.LogRecord(
                getattr(batch._drop_logger, "name", "deferred_diagnostics"),
                logging.WARNING, __file__, 0,
                "deferred_diagnostics_dropped count=%d capacity=%d "
                "captured=%d scope_thread=%s control_unchanged=True",
                (batch.dropped, batch.capacity, batch.captured, threading.current_thread().name),
                None,
            )
            batch._drop_logger.handle(record)
            batch.dispatched += 1
        except Exception:
            batch.dispatch_errors += 1
    batch._drop_logger = None


@contextmanager
def deferred_diagnostics(logger, capacity=256):
    """Defer this thread's records for ``logger`` until outside the scope.

    Capacity is bounded to 1..256 records. Nested scopes share the outer
    batch/capacity and never dispatch early.
    Other loggers and other threads retain their existing behavior. Filters
    are installed once and remain inert outside a scope. Arguments are not
    formatted or deeply copied in the critical section; callers should log
    scalar/frozen snapshots, not subsequently mutated application containers.

    Unsupported logging adapters are a compatible no-op. The returned stats
    count successful dispatches including an optional overflow health record.
    """
    if (isinstance(capacity, bool) or not isinstance(capacity, int)
            or not 1 <= capacity <= _MAX_RECORDS):
        raise ValueError("deferred diagnostic capacity must be an integer from 1 to 256")
    capture = _capture_filter(logger)
    if capture is None:
        yield DeferredDiagnosticStats(capacity)
        return
    batch = getattr(_local, "batch", None)
    outermost = batch is None
    if outermost:
        batch = DeferredDiagnosticStats(capacity)
        _local.batch = batch
    batch._active_filters[capture] = batch._active_filters.get(capture, 0) + 1
    try:
        yield batch
    finally:
        count = batch._active_filters[capture] - 1
        if count:
            batch._active_filters[capture] = count
        else:
            del batch._active_filters[capture]
        if outermost:
            # Disable capture before dispatch, including when the body raised.
            # Never dispatch a nested scope while its caller may hold a lock.
            del _local.batch
            _dispatch(batch)
