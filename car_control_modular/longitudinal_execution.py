"""Successful command evidence for PI continuity, never a motor lease."""
from dataclasses import dataclass
import math

from .depth_authority_timing import MAX_FORWARD_DEPTH_TTL_SEC


@dataclass(frozen=True)
class ForwardExecutionAnchor:
    uid: int
    sample_timestamp: float
    rpm: float
    sent_at: float
    receipt: object


@dataclass(frozen=True)
class ForwardRecoveryAnchor:
    """A still-current positive write, not an extension of its depth lease.

    The executor supplies and revalidates receipt identity and STOP generation.
    The controller must independently qualify a NEW range sample before use.
    A zero or any intervening write invalidates this proof; there is deliberately
    no permission to jump back across a completed zero transaction.
    """

    executed: ForwardExecutionAnchor
    current_receipt: object
    stop_generation: int
    checked_at: float

    def valid_for(self, uid, previous_sample, new_sample, now, memory_sec):
        anchor = self.executed
        numeric = (previous_sample, new_sample, now, memory_sec, self.checked_at)
        if (not isinstance(anchor, ForwardExecutionAnchor)
                or type(uid) is not int or uid <= 0 or type(anchor.uid) is not int
                or anchor.uid != uid or anchor.receipt is None
                or self.current_receipt is not anchor.receipt
                or type(self.stop_generation) is not int or self.stop_generation < 0
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           and math.isfinite(v) for v in numeric)):
            return False
        values = (anchor.sample_timestamp, anchor.sent_at, anchor.rpm)
        return bool(
            all(isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v) for v in values)
            and self.checked_at == now and anchor.rpm > 0
            and 0 < memory_sec <= .35
            and anchor.sample_timestamp <= previous_sample < new_sample <= now
            and anchor.sample_timestamp <= anchor.sent_at <= now
            and anchor.sent_at <= anchor.sample_timestamp+MAX_FORWARD_DEPTH_TTL_SEC
            and 0 <= now-anchor.sample_timestamp <= memory_sec
            and 0 <= now-anchor.sent_at <= memory_sec
            and 0 <= now-new_sample <= .18)
