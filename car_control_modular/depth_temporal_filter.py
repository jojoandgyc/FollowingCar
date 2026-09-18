"""Time-bounded median of accepted physical Depth samples.

The two caller-owned deques preserve the existing float-history interface.
There is no clock, target identity, camera access or hidden state here. Call
``reset_depth_window`` when switching UID or explicitly replacing an anchor.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from statistics import median
from typing import Any, Optional


@dataclass(frozen=True)
class DepthTemporalFilterResult:
    accepted: bool
    reason: str
    distance_m: Optional[float]
    expired_count: int = 0
    unknown_time_count: int = 0
    invalid_history_count: int = 0
    capacity_evicted_count: int = 0
    window_count: int = 0
    oldest_sample_timestamp: Optional[float] = None
    newest_sample_timestamp: Optional[float] = None


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def reset_depth_window(values: deque, timestamps: deque) -> int:
    """Reset both queues together; return the number of removed values."""
    removed = len(values)
    values.clear()
    timestamps.clear()
    return removed


def append_depth_sample(
    values: deque,
    timestamps: deque,
    *,
    distance_m: float,
    sample_timestamp: float,
    now: float,
    max_age_sec: float,
) -> DepthTemporalFilterResult:
    """Append one valid ordered sample, evict old samples, return its median.

    ``sample_timestamp`` is the physical depth timestamp, not the RGB/control
    invocation time. Accepted history older than ``max_age_sec`` relative to
    that sample is evicted. ``now`` additionally rejects future/stale incoming
    samples. Rejected new samples never enter or age a healthy window.

    Histories with missing/misaligned timestamps cannot safely be dated; both
    queues are cleared. Corrupted/non-monotonic history is likewise discarded.
    Queue capacities must match, preserving equal lengths after every append.
    """
    unknown_count = 0
    invalid_count = 0
    if len(values) != len(timestamps):
        unknown_count = reset_depth_window(values, timestamps)
    else:
        history_values = tuple(_finite_number(value) for value in values)
        history_stamps = tuple(_finite_number(stamp) for stamp in timestamps)
        if (
            any(value is None or value <= 0.0 for value in history_values)
            or any(stamp is None or stamp <= 0.0 for stamp in history_stamps)
            or any(right <= left + 1e-9 for left, right in zip(history_stamps, history_stamps[1:]))
        ):
            invalid_count = reset_depth_window(values, timestamps)
        else:
            # Keep the legacy queue numeric even if an older caller injected
            # integer/NumPy/string-coercible numbers rather than Python floats.
            values.clear()
            values.extend(history_values)
            timestamps.clear()
            timestamps.extend(history_stamps)

    def rejected(reason: str) -> DepthTemporalFilterResult:
        return DepthTemporalFilterResult(
            accepted=False, reason=reason, distance_m=None,
            unknown_time_count=unknown_count, invalid_history_count=invalid_count,
            window_count=len(values),
            oldest_sample_timestamp=float(timestamps[0]) if timestamps else None,
            newest_sample_timestamp=float(timestamps[-1]) if timestamps else None,
        )

    if values.maxlen != timestamps.maxlen or values.maxlen == 0:
        invalid_count += reset_depth_window(values, timestamps)
        return rejected("window_capacity_mismatch")
    distance = _finite_number(distance_m)
    stamp = _finite_number(sample_timestamp)
    current = _finite_number(now)
    max_age = _finite_number(max_age_sec)
    if distance is None or distance <= 0.0:
        return rejected("invalid_distance")
    if stamp is None or stamp <= 0.0:
        return rejected("invalid_sample_timestamp")
    if current is None or current <= 0.0:
        return rejected("invalid_now")
    if max_age is None or max_age < 0.0:
        return rejected("invalid_max_age")
    if stamp > current:
        return rejected("future_sample")
    if current - stamp > max_age + 1e-9:
        return rejected("stale_sample")
    if timestamps and stamp <= float(timestamps[-1]) + 1e-9:
        return rejected(
            "duplicate_sample" if abs(stamp - float(timestamps[-1])) <= 1e-9
            else "out_of_order_sample"
        )

    expired_count = 0
    while timestamps and stamp - float(timestamps[0]) > max_age + 1e-9:
        values.popleft()
        timestamps.popleft()
        expired_count += 1
    capacity_evicted = int(values.maxlen is not None and len(values) == values.maxlen)
    values.append(distance)
    timestamps.append(stamp)
    return DepthTemporalFilterResult(
        accepted=True, reason="accepted", distance_m=float(median(values)),
        expired_count=expired_count, unknown_time_count=unknown_count,
        invalid_history_count=invalid_count, capacity_evicted_count=capacity_evicted,
        window_count=len(values), oldest_sample_timestamp=float(timestamps[0]),
        newest_sample_timestamp=float(timestamps[-1]),
    )
