"""Latest-value selection for an unsent, already guarded forward packet.

This is arithmetic and provenance only, never a motion authorization. The
executor must evaluate the selected sample's braking budget, identity,
feedback, physical deadlines and STOP ownership immediately before writing.
"""
from dataclasses import dataclass
import math

from .sample_braking import SampleBrakingAssessment
from .final_yaw_coalescing import contract_forward_speed


@dataclass(frozen=True)
class ForwardWriteSnapshot:
    linear: tuple
    grant: tuple
    pair: tuple[int, int]


def select_forward_write_snapshot(uid, previous, current, pair, now, maximum_rpm):
    """Use a newer same-UID positive sample without increasing either wheel.

    A changed immutable publication is not a STOP. A missing/zero/reverse,
    older, unassessed or already expired replacement is not admissible here.
    Keeping the old pair's yaw avoids inventing a new turn from a depth update.
    A higher new cap need not raise this already guarded pair; the next tick
    can plan the acceleration. A lower cap contracts both wheels immediately.
    """
    try:
        old, _old_timing = previous
        raw, timing = current
        assessment = timing.braking_assessment
        if (len(old) != 4 or len(raw) != 4 or len(pair) != 2
                or old[0] != raw[0] or raw[0] != "forward"
                or old[2] != uid or raw[2] != uid
                or timing.snapshot != raw
                or not isinstance(assessment, SampleBrakingAssessment)
                or not assessment.valid_for(uid, raw[3])
                or not all(math.isfinite(value) for value in (
                    old[3], raw[1], raw[3], now, timing.depth_expires_at,
                    maximum_rpm, *pair))
                or not 0 < old[3] <= raw[3] <= now <= timing.depth_expires_at
                or not 0 < raw[1] <= 100 or not 0 < old[1] <= 100
                or maximum_rpm <= 0
                or min(pair) < 0 or sum(pair) <= 0):
            return None
        bounded = contract_forward_speed(pair, raw[1] * maximum_rpm / 100.)
        if bounded is None:
            return None
        return ForwardWriteSnapshot(raw, current, bounded)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
