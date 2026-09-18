"""Explicit normal parking intent, separate from zero longitudinal speed.

No motor I/O or independent timeout extends movement authority here.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NearYawParkRequest:
    uid: int
    capture_frame_id: int
    capture_timestamp: float
    requested_at: float
    reason: str


def newer_visual_evidence(capture_id, stamp, reference, now, max_age):
    """Both physical provenance fields must advance; publication is not evidence."""
    return bool(
        isinstance(capture_id, int) and not isinstance(capture_id, bool)
        and isinstance(stamp, (int, float)) and not isinstance(stamp, bool)
        and math.isfinite(stamp) and stamp > 0
        and capture_id > reference[0] and stamp > reference[1]
        and 0 <= now - stamp <= max_age
    )


def predictive_or_center_stop(result):
    """A transient PID zero or direction guard is not a parking request."""
    return bool(
        result is not None
        and int(getattr(result, "correction_rpm", 1)) == 0
        and (getattr(result, "predictive_braking", False)
             or getattr(result, "output_floor_reason", "") in {
                 "predictive_brake_coast", "center_hold",
             })
    )
