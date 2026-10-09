"""Optimistic Depth30 ranging: private filters until a validated commit.

The camera's published arrays are immutable. Only their references are shared;
all ranging history and diagnostic collections are copied. No worker ever holds
the live measurement lock while scanning pixels, so a visual/safety decision
cannot end up waiting for the speculative worker under the control mutex.
"""
from collections import deque
from copy import copy, deepcopy
from dataclasses import dataclass
import math
import threading
import time


@dataclass(frozen=True)
class DepthCommitReceipt:
    """Proof that this exact immutable result passed the physical commit gate.

    It is not a new sample timestamp or a motor lease. Consumers still age the
    original sample; they need not re-admit already completed pixel work merely
    because normal delivery took another few milliseconds.
    """
    measurement: object
    validated_at: float


# Deliberate allowlist: never copy hardware handles, frame history, locks,
# configuration, or shutdown flags back into the live runtime.
RANGING_STATE_FIELDS = (
    "_last_target_id", "_distance_history", "_distance_history_timestamps",
    "_last_accepted_distance_m", "_last_accepted_ts", "_pending_jump_distance_m",
    "_pending_jump_count", "_pending_jump_required_confirms", "_pending_jump_timestamp",
    "_pending_jump_kind", "_pending_jump_region_count", "_pending_torso_recovery_evidence",
    "_last_torso_recovery_evidence", "_last_processed_depth_ts", "_attempted_depth_samples",
    "_measurement_sample_ts", "_measurement_temporal_status", "_last_torso_selection",
    "_measurement_roi_valid", "_measurement_roi_required", "_measurement_sparse",
    "_measurement_torso_recovery_status", "_measurement_filter_expired_count",
    "_measurement_filter_reset_count", "_near_reference_bbox_area_ratio",
    "_last_accepted_bbox_area_ratio", "_last_accepted_bbox", "_last_accepted_region_count",
    "_last_feedback_ts", "_encoder_distance_change_since_accept_m", "_last_log_ts",
    "_last_region_log_ts", "_last_diagnostic_log_ts", "_last_diagnostic_log_key",
    "_measurement_regions", "_measurement_depth_size", "_measurement_bounded_selection",
    "_measurement_bounded_selected_at", "_shadow_range_evidence",
)


class _BufferedOutput:
    def __init__(self, destination, records):
        self.destination, self.records = destination, records

    def __getattr__(self, name):
        callback = getattr(self.destination, name)
        return lambda *args, **kwargs: self.records.append((callback, args, kwargs))


class DepthMeasurementTransaction:
    """Single-use computation, validated by runtime revision on commit."""

    @classmethod
    def begin(cls, runtime, args, kwargs):
        if not runtime._measurement_lock.acquire(blocking=False):
            return None
        try:
            if runtime._released or runtime._stop_event.is_set():
                return None
            if not runtime._depth_lock.acquire(blocking=False):
                return None
            try:
                private = copy(runtime)
                private._depth_history = deque(runtime._depth_history,
                                               maxlen=runtime._depth_history.maxlen)
                private._depth_lock = threading.Lock()
                selected_at = time.monotonic()
            finally:
                runtime._depth_lock.release()
            state = deepcopy({name: getattr(runtime, name) for name in RANGING_STATE_FIELDS
                              if hasattr(runtime, name)})
            private.__dict__.update(state)
            private._measurement_lock = threading.RLock()
            # Private work may abort BETWEEN pixel operations once the actual
            # sample expires. Never alter the live fallback API or its clock.
            private._transaction_sample_max_age_sec = .18
            transaction = cls()
            transaction.runtime, transaction.private = runtime, private
            transaction.revision = runtime._measurement_revision
            transaction.selected_at = selected_at
            transaction.args, transaction.kwargs = args, dict(kwargs)
            transaction.result = None
            transaction.commit_receipt = None
            transaction.started = transaction.consumed = transaction.committed = False
            transaction.reject_reason = None
            transaction.preflight_sample_timestamp = None
            transaction.preflight_remaining_sec = None
            # Private diagnostics only: never copied into live sensor state or
            # used to predict expiry/authorize a measurement.
            transaction.pixel_scan_started = False
            transaction.compute_stage = "not_started"
            transaction.failure_stage = None
            transaction.selected_sample_timestamp = None
            transaction.compute_wall_ms = transaction.compute_cpu_ms = 0.0
            transaction.stage_timings_ms = {}
            transaction._stage_started_wall = transaction._stage_started_cpu = None
            transaction.records = []
            private.logger = _BufferedOutput(runtime.logger, transaction.records)
            private.diagnostics = (None if runtime.diagnostics is None else
                                   _BufferedOutput(runtime.diagnostics, transaction.records))
            return transaction
        finally:
            runtime._measurement_lock.release()

    def mark_stage(self, stage, *, pixel_scan_started=False):
        """Attribute worker wall/thread CPU time without touching live state."""
        wall, cpu = time.perf_counter(), time.thread_time()
        self._record_stage_until(wall, cpu)
        self.compute_stage = stage
        self._stage_started_wall, self._stage_started_cpu = wall, cpu
        self.pixel_scan_started = self.pixel_scan_started or pixel_scan_started

    def _record_stage_until(self, wall, cpu):
        if self._stage_started_wall is not None:
            timing = self.stage_timings_ms.setdefault(self.compute_stage, {"wall_ms": 0.0, "cpu_ms": 0.0})
            timing["wall_ms"] += max(0.0, wall-self._stage_started_wall) * 1000.0
            timing["cpu_ms"] += max(0.0, cpu-self._stage_started_cpu) * 1000.0

    def run(self):
        if self.started or self.consumed:
            raise RuntimeError("Depth transaction is single use")
        self.started = True
        started_wall, started_cpu = time.perf_counter(), time.thread_time()
        measurement_started = False
        from .astra_depth import _DepthComputeExpired
        self.private._transaction_trace = self
        try:
            self.mark_stage("history_preflight" if self.kwargs.get("bounded_roi_capture_timestamp")
                            is not None else "sample_selection")
            if self.kwargs.get("bounded_roi_capture_timestamp") is not None:
                # A 500 ms ROI only permits looking into history. It is not a
                # 500 ms physical sample lease. Inspect the SAME immutable depth
                # snapshot which run() would scan, before spending pixel CPU time.
                now = time.monotonic()
                with self.private._depth_lock:
                    depth, stamp, _ = self.private._aligned_depth_locked(
                        now, use_latest_depth=True,
                        bounded_roi_capture_timestamp=self.kwargs["bounded_roi_capture_timestamp"],
                        bounded_roi_max_age_sec=self.kwargs.get("bounded_roi_max_age_sec", .25))
                self.preflight_sample_timestamp = float(stamp) if depth is not None else None
                if depth is None:
                    self.reject_reason = "history_no_eligible_sample"
                    return None
                remaining = min(.18, float(self.runtime.config.max_frame_age_sec)) - (time.monotonic()-stamp)
                self.preflight_remaining_sec = remaining
                if not math.isfinite(remaining) or remaining <= 0:
                    self.reject_reason = "history_sample_expired"
                    return None
            measurement_started = True
            self.result = self.private.measure_target(*self.args, **self.kwargs)
        except _DepthComputeExpired:
            self.reject_reason = "physical_sample_expired_during_compute"
            return None
        finally:
            # The private snapshot owns depth-history arrays. Do not leave a
            # transaction -> private -> transaction cycle waiting for GC.
            del self.private._transaction_trace
            finished_wall, finished_cpu = time.perf_counter(), time.thread_time()
            self._record_stage_until(finished_wall, finished_cpu)
            self._stage_started_wall = self._stage_started_cpu = None
            self.compute_wall_ms = max(0.0, finished_wall-started_wall) * 1000.0
            self.compute_cpu_ms = max(0.0, finished_cpu-started_cpu) * 1000.0
            self.selected_sample_timestamp = (self.private._measurement_sample_ts
                                              if measurement_started else self.preflight_sample_timestamp)
            if self.result is None:
                self.failure_stage = self.compute_stage
            else:
                self.compute_stage = "complete"
        return self.result

    def commit(self, *, now, max_sample_age_sec=.18, validate=None):
        """Caller has revalidated the ROI/identity under the control mutex."""
        if self.consumed or self.result is None:
            self.reject_reason = self.reject_reason or "consumed_or_no_result"
            return None
        self.consumed = True
        runtime = self.runtime
        if not runtime._measurement_lock.acquire(blocking=False):
            self.reject_reason = "measurement_lock_busy"
            return None
        try:
            if runtime._released or runtime._stop_event.is_set():
                self.reject_reason = "runtime_stopped"
                return None
            if runtime._measurement_revision != self.revision:
                self.reject_reason = "measurement_revision_changed"
                return None
            # Both clocks and configured ceilings are checked at the actual
            # commit, not the earlier pre-geometry/logging timestamp.
            if validate is not None and not validate(time.monotonic()):
                self.reject_reason = "geometry_or_sample_provenance"
                return None
            try:
                current_now = max(float(now), time.monotonic())
                age = current_now - float(self.result.observation_sample_timestamp)
                limits = (float(max_sample_age_sec), float(runtime.config.max_frame_age_sec))
            except (TypeError, ValueError, OverflowError):
                self.reject_reason = "invalid_sample_clock"
                return None
            if (not math.isfinite(age) or not all(math.isfinite(v) and v > 0 for v in limits)
                    or not 0 <= age <= min(limits)):
                self.reject_reason = "physical_sample_expired_or_future"
                return None
            for name in RANGING_STATE_FIELDS:
                if hasattr(self.private, name):
                    setattr(runtime, name, getattr(self.private, name))
            runtime._measurement_revision += 1
            self.committed = True
            self.commit_receipt = DepthCommitReceipt(self.result, current_now)
            return self.result
        finally:
            runtime._measurement_lock.release()

    def flush(self):
        """Only committed observations reach normal diagnostics, after unlock."""
        records, self.records = self.records, []
        if self.committed:
            for callback, args, kwargs in records:
                try:
                    callback(*args, **kwargs)
                except Exception:
                    # Diagnostics cannot invalidate a completed state commit.
                    pass
