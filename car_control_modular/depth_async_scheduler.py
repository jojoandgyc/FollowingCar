"""One physical ranging operation, with coalesced identity-bound ROI input.

This owns work, not motion permission. Heartbeats, successful completion, and
republishing a ROI never renew a measurement or motor deadline. A task keeps
its original immutable geometry/capture proof; the caller must still validate
the chosen physical Depth sample and all control gates at commit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import threading
from typing import Mapping, Optional, Tuple

from .control_types import PersonTarget


def _finite_time(value, name):
    if isinstance(value, bool):
        raise ValueError(name + " must be a finite nonnegative timestamp")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(name + " must be a finite nonnegative timestamp")
    return value


def _limit(value, name):
    value = _finite_time(value, name)
    if value <= 0:
        raise ValueError(name + " must be positive")
    return value


def _freeze_target(target):
    if not isinstance(target, PersonTarget):
        raise ValueError("person_targets must contain PersonTarget snapshots")
    fields = {"bbox": tuple(target.bbox)}
    for name in ("depth_observation", "braking_observation"):
        observation = getattr(target, name)
        if observation is not None:
            fields[name] = replace(observation, bbox=tuple(observation.bbox))
    return replace(target, **fields)


@dataclass(frozen=True)
class DepthAsyncContext:
    epoch: int
    sequence: int
    target_id: int
    capture_frame_id: int
    capture_timestamp: float
    published_ts: float
    frame_index: int
    width: int
    height: int
    person_targets: Tuple[PersonTarget, ...]
    target_steerable: bool

    def as_dict(self):
        """A detached compatibility mapping; geometry inside remains frozen."""
        return {
            "target_id": self.target_id,
            "capture_frame_id": self.capture_frame_id,
            "capture_timestamp": self.capture_timestamp,
            "published_ts": self.published_ts,
            "frame_index": self.frame_index,
            "width": self.width,
            "height": self.height,
            "person_targets": self.person_targets,
            "persons": tuple((t.bbox, t.track_id, t.confidence, t.area)
                             for t in self.person_targets),
            "target_steerable": self.target_steerable,
        }


@dataclass(frozen=True)
class DepthWorkTicket:
    epoch: int
    sequence: int
    owner: str
    started_at: float
    context: Optional[DepthAsyncContext]


@dataclass(frozen=True)
class DepthWorkerHealth:
    healthy: bool
    reason: str
    busy: bool
    busy_age_sec: Optional[float]
    heartbeat_age_sec: Optional[float]
    progress_age_sec: Optional[float]
    consecutive_errors: int
    epoch: int


class DepthAsyncScheduler:
    """Small lock-protected mailbox and scan lease, never a worker pool.

    Same-UID publication replaces only ``latest``. Revocation invalidates the
    current task's commit permission but deliberately retains its scan slot
    until ``finish``: abandoning a running calculation must not cause another
    thread to start a second physical scan concurrently.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._epoch = 0
        self._context_sequence = 0
        self._task_sequence = 0
        self._latest = None
        self._inflight = None
        self._last_worker_tick = None
        self._last_progress = None
        self._waiting_since = None
        self._consecutive_errors = 0
        self._last_revoke_reason = ""

    def _freeze_context(self, context: Mapping, now, epoch):
        targets = tuple(_freeze_target(t) for t in context["person_targets"])
        uid = int(context["target_id"])
        capture_id = int(context["capture_frame_id"])
        capture_ts = _finite_time(context["capture_timestamp"], "capture_timestamp")
        published_ts = _finite_time(context.get("published_ts", now), "published_ts")
        width, height = int(context["width"]), int(context["height"])
        if (uid < 0 or capture_id <= 0 or capture_ts <= 0 or capture_ts > now
                or published_ts > now or width <= 0 or height <= 0
                or len(targets) != 1 or targets[0].track_id != uid):
            raise ValueError("invalid identity-bound Depth context")
        observation = targets[0].depth_observation
        if (observation is None or observation.target_id != uid
                or observation.capture_frame_id != capture_id
                or observation.capture_timestamp != capture_ts
                or observation.source != "yolo_detector"):
            raise ValueError("Depth context does not match original detector proof")
        return DepthAsyncContext(
            epoch=epoch, sequence=self._context_sequence + 1, target_id=uid,
            capture_frame_id=capture_id, capture_timestamp=capture_ts,
            published_ts=published_ts, frame_index=int(context["frame_index"]),
            width=width, height=height, person_targets=targets,
            target_steerable=bool(context.get("target_steerable", True)),
        )

    def publication_snapshot(self):
        """Read an epoch plus immutable mailbox value without the control lock.

        An inference may retain the epoch, not permission to move. Revocation
        while it runs makes a conditional publication fail, including after a
        same-UID reacquisition. No physical scan slot or deadline is touched.
        """
        with self._lock:
            return self._epoch, self._latest

    def submit(self, context: Mapping, *, now: float, expected_epoch=None,
               publication_guard=None) -> Optional[DepthAsyncContext]:
        """Coalesce a new ROI; ordinary same-UID publication keeps flight valid.

        Duplicate/older same-UID captures do not replace geometry, publication
        time, or the waiting-progress clock. UID changes revoke the old epoch.

        Off-control-lock producers must supply their pre-inference epoch and
        a pure, nonblocking guard over the current identity/STOP snapshot. The
        guard MUST NOT acquire another lock or call into controller/motor code.
        Rejection changes neither the mailbox nor the in-flight scan.
        """
        now = _finite_time(now, "now")
        with self._lock:
            if ((expected_epoch is not None and expected_epoch != self._epoch)
                    or (publication_guard is not None and not publication_guard())):
                return None
            frozen = self._freeze_context(context, now, self._epoch)
            latest = self._latest
            if latest is not None and latest.target_id == frozen.target_id:
                if (frozen.capture_timestamp <= latest.capture_timestamp
                        or frozen.capture_frame_id <= latest.capture_frame_id):
                    return latest
            elif ((latest is not None and latest.target_id != frozen.target_id)
                  or (self._inflight is not None and self._inflight.context is not None
                      and self._inflight.context.target_id != frozen.target_id
                      and self._inflight.epoch == self._epoch)):
                self._revoke_locked("target_changed")
                frozen = replace(frozen, epoch=self._epoch)
            self._context_sequence += 1
            self._latest = frozen
            if self._inflight is None and self._waiting_since is None:
                self._waiting_since = now
            return frozen

    def _start_locked(self, context, now, owner):
        self._task_sequence += 1
        ticket = DepthWorkTicket(self._epoch, self._task_sequence, owner, now, context)
        self._inflight = ticket
        self._waiting_since = None
        if owner == "depth30":
            self._last_worker_tick = now
        return ticket

    def begin(self, *, now: float, max_capture_age_sec: float) -> Optional[DepthWorkTicket]:
        """Take the latest ROI, rejecting stale input before selecting Depth."""
        now = _finite_time(now, "now")
        maximum = _limit(max_capture_age_sec, "max_capture_age_sec")
        with self._lock:
            context = self._latest
            if (self._inflight is not None or context is None
                    or context.epoch != self._epoch
                    or not 0 <= now - context.capture_timestamp <= maximum):
                return None
            return self._start_locked(context, now, "depth30")

    def valid(self, ticket: DepthWorkTicket, *, now: Optional[float] = None,
              max_capture_age_sec: Optional[float] = None) -> bool:
        """Check work ownership, optionally ROI age; not sample authorization.

        Default commit checks deliberately omit ROI age. An operation may
        finish after the ROI selection window if its fixed physical sample
        and selection proof still satisfy the caller's commit-time checks.
        """
        maximum = (None if max_capture_age_sec is None else
                   _limit(max_capture_age_sec, "max_capture_age_sec"))
        if now is not None:
            now = _finite_time(now, "now")
        if maximum is not None and now is None:
            raise ValueError("now is required for a capture age check")
        with self._lock:
            if (ticket is not self._inflight or ticket is None
                    or ticket.epoch != self._epoch
                    or (now is not None and now < ticket.started_at)):
                return False
            return bool(maximum is None or (ticket.context is not None
                        and 0 <= now - ticket.context.capture_timestamp <= maximum))

    def finish(self, ticket: DepthWorkTicket, *, now: float, error: bool = False) -> bool:
        """Release only the matching scan, including a revoked running scan."""
        now = _finite_time(now, "now")
        with self._lock:
            if ticket is None or ticket is not self._inflight or now < ticket.started_at:
                return False
            self._inflight = None
            if ticket.owner == "depth30":
                self._last_worker_tick = now
                self._last_progress = now
                self._consecutive_errors = self._consecutive_errors + 1 if error else 0
            if self._latest is not None:
                self._waiting_since = now
            return True

    def _revoke_locked(self, reason):
        self._epoch += 1
        self._latest = None
        self._waiting_since = None
        self._last_revoke_reason = str(reason)

    def revoke(self, reason: str) -> None:
        with self._lock:
            self._revoke_locked(reason)

    def begin_fallback(self, context: Optional[Mapping] = None, *, now: float,
                       max_capture_age_sec: Optional[float] = None) -> Optional[DepthWorkTicket]:
        """Revoke async commit, then reserve the SAME physical scan slot.

        ``context=None`` supports bootstrap before an UID exists. It supplies
        only mutual exclusion: no identity or motion proof. If an old task is
        still calculating, fallback must defer until that task calls finish.
        """
        now = _finite_time(now, "now")
        maximum = (None if max_capture_age_sec is None else
                   _limit(max_capture_age_sec, "max_capture_age_sec"))
        with self._lock:
            frozen = None if context is None else self._freeze_context(context, now, self._epoch + 1)
            self._revoke_locked("synchronous_fallback")
            if self._inflight is not None:
                return None
            if (maximum is not None and (frozen is None
                    or not 0 <= now - frozen.capture_timestamp <= maximum)):
                return None
            if frozen is not None:
                self._context_sequence += 1
            return self._start_locked(frozen, now, "vision_fallback")

    def worker_tick(self, *, now: float) -> None:
        now = _finite_time(now, "now")
        with self._lock:
            if self._last_worker_tick is None or now >= self._last_worker_tick:
                self._last_worker_tick = now

    def health(self, *, now: float, worker_alive: bool, max_silence_sec: float = .25,
               max_busy_sec: float = .25, max_errors: int = 2) -> DepthWorkerHealth:
        """Bounded readiness, using progress/busy/error state, not liveness alone."""
        now = _finite_time(now, "now")
        silence = _limit(max_silence_sec, "max_silence_sec")
        busy_limit = _limit(max_busy_sec, "max_busy_sec")
        if isinstance(max_errors, bool) or int(max_errors) != max_errors or max_errors < 1:
            raise ValueError("max_errors must be a positive integer")
        with self._lock:
            flight = self._inflight
            busy_age = None if flight is None else now - flight.started_at
            heartbeat_age = None if self._last_worker_tick is None else now - self._last_worker_tick
            progress_age = None if self._last_progress is None else now - self._last_progress
            if not worker_alive:
                reason = "worker_not_alive"
            elif heartbeat_age is None:
                reason = "worker_not_started"
            elif heartbeat_age < 0 or (busy_age is not None and busy_age < 0):
                reason = "clock_invalid"
            elif self._consecutive_errors >= max_errors:
                reason = "worker_errors"
            elif flight is not None and flight.owner != "depth30":
                reason = "fallback_busy"
            elif flight is not None and flight.epoch != self._epoch:
                reason = "revoked_busy"
            elif busy_age is not None and busy_age > busy_limit:
                reason = "worker_busy_timeout"
            elif flight is None and heartbeat_age > silence:
                reason = "worker_silent"
            elif (flight is None and self._waiting_since is not None
                  and now - self._waiting_since > silence):
                reason = "worker_no_progress"
            else:
                reason = "ready" if flight is None else "working"
            return DepthWorkerHealth(
                healthy=reason in {"ready", "working"}, reason=reason,
                busy=flight is not None, busy_age_sec=busy_age,
                heartbeat_age_sec=heartbeat_age, progress_age_sec=progress_age,
                consecutive_errors=self._consecutive_errors, epoch=self._epoch,
            )
