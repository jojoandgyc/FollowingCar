"""Immutable diagnostics for acknowledged zero-speed and STOP transactions.

These records are evidence of protocol writes, never motor authority or proof
of physical stillness.  Formatting is delayed until a logging handler consumes
the record, so the executor's deferred_diagnostics scope can emit outside its
motor lock.  Context providers must inspect caches only (no I/O or new locks).
"""
from dataclasses import dataclass
import json
import math


def freeze_context(value, depth=0):
    """Take a bounded primitive-only copy; never retain live runtime objects."""
    if value is None or isinstance(value, (bool, int, str)):
        return value if not isinstance(value, str) else value[:384]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if depth < 3 and isinstance(value, (list, tuple)):
        return tuple(freeze_context(item, depth + 1) for item in value[:16])
    if depth < 3 and isinstance(value, dict):
        return tuple((str(key)[:64], freeze_context(item, depth + 1))
                     for key, item in list(value.items())[:64])
    return "unsupported_context_value"


@dataclass(frozen=True)
class MotorZeroAuditEvent:
    event_id: int
    event_kind: str
    source: str
    phase: str
    requested_raw: tuple | None
    normalized_forward_yaw: tuple | None
    previous_acknowledged_raw: tuple | None
    previous_forward_yaw: tuple | None
    previous_acknowledged_at: float | None
    previous_receipt_sequence: int | None
    receipt_before: int | None
    receipt_after: int | None
    stop_generation: int
    generated_at: float
    completed_at: float
    outcome: str
    attempted_sides: tuple
    acknowledged_sides: tuple
    failed_sides: tuple
    stop_mode: int | None
    context: tuple

    def __str__(self):
        values = dict(vars(self))
        values["context"] = dict(self.context)
        values["elapsed_ms"] = max(0.0, self.completed_at - self.generated_at) * 1000.0
        values["writes_complete"] = self.outcome == "acknowledged"
        values["physical_stillness"] = "unverified"
        return json.dumps(values, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
