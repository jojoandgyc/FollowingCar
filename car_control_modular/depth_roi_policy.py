"""A longer detector observation window is NOT a longer motor lease."""
import math
from dataclasses import dataclass
from typing import Optional, Tuple


def roi_age_status(capture_stamp, now, max_age, feedback=None):
    """Keep the original 180ms path; extra time requires fresh low-yaw feedback.

    Identity/geometry/foreground checks remain the caller's responsibility.
    This function neither moves a bbox nor authorizes a motor command.
    """
    values = (capture_stamp, now, max_age)
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in values):
        return "invalid"
    age = now - capture_stamp
    if capture_stamp <= 0 or age < 0 or age > min(.25, max_age):
        return "expired"
    if age <= .18:
        return "normal"
    if feedback is None or not getattr(feedback, "trustworthy", False):
        return "feedback_missing"
    stamp = getattr(feedback, "timestamp", None)
    yaw = getattr(feedback, "yaw_rate_right_dps", None)
    raw = getattr(feedback, "raw_yaw_rate_right_dps", None)
    raw = yaw if raw is None else raw
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in (stamp, yaw, raw)):
        return "feedback_invalid"
    if not 0 <= now-stamp <= .15:
        return "feedback_stale"
    if max(abs(yaw), abs(raw)) > 5.0:
        return "turning"
    return "extended"


def roi_age_allowed(capture_stamp, now, max_age, feedback=None):
    return roi_age_status(capture_stamp, now, max_age, feedback) in {"normal", "extended"}


def bounded_roi_history_allowed(capture_stamp, now, max_age=.25):
    """Eligibility to TRY historical ranging, never a latest-frame exemption.

    A delayed ROI may still have a real, fresh Depth frame from its original
    180ms association window. The backend must select that frame atomically;
    callers without this capability must keep the existing low-yaw gate.
    """
    values = (capture_stamp, now, max_age)
    return bool(
        all(not isinstance(v, bool) and isinstance(v, (int, float))
            and math.isfinite(v) for v in values)
        and capture_stamp > 0 and 0 <= now-capture_stamp <= min(.25, max_age)
    )


def bounded_depth_sample_allowed(capture_stamp, sample_stamp, now, max_sample_age=.18):
    """Admission to select a physical frame while its ROI can still start work."""
    return bool(
        bounded_roi_history_allowed(capture_stamp, now)
        and _bounded_physical_sample_current(capture_stamp, sample_stamp, now, max_sample_age)
    )


def _bounded_physical_sample_current(capture_stamp, sample_stamp, now, max_sample_age):
    """Fixed association plus current physical age, not a new ROI admission."""
    values = (capture_stamp, sample_stamp, now, max_sample_age)
    return bool(
        all(not isinstance(v, bool) and isinstance(v, (int, float))
                and math.isfinite(v) for v in values)
        and capture_stamp > 0 and sample_stamp > 0 and max_sample_age > 0
        and capture_stamp <= sample_stamp <= capture_stamp + .18
        and 0 <= now-sample_stamp <= min(.18, max_sample_age)
    )


@dataclass(frozen=True)
class BoundedDepthSelection:
    """Immutable proof of one admitted ROI-to-physical-frame association.

    This is measurement provenance, not an identity or motor authorization.
    The capture-age check belongs to ``selected_timestamp``; completion only
    ages the same physical frame. No consumer may replace these timestamps
    with completion/publication time or reuse the proof for another ROI/UID.
    """
    capture_timestamp: float
    sample_timestamp: float
    selected_timestamp: float
    target_id: Optional[int]
    capture_frame_id: Optional[int]
    bbox: Tuple[float, float, float, float]

    def valid_for(self, *, capture_timestamp, sample_timestamp, now,
                  target_id, capture_frame_id, bbox, max_sample_age=.18):
        return bool(
            all(isinstance(v, int) and not isinstance(v, bool) and v > 0
                for v in (self.target_id, self.capture_frame_id, target_id, capture_frame_id))
            and self.capture_timestamp == capture_timestamp
            and self.sample_timestamp == sample_timestamp
            and self.target_id == target_id
            and self.capture_frame_id == capture_frame_id
            and self.bbox == tuple(bbox)
            and bounded_depth_sample_allowed(
                self.capture_timestamp, self.sample_timestamp,
                self.selected_timestamp, max_sample_age)
            and isinstance(now, (int, float)) and not isinstance(now, bool)
            and math.isfinite(now) and self.selected_timestamp <= now
            and _bounded_physical_sample_current(
                self.capture_timestamp, self.sample_timestamp, now, max_sample_age)
        )
