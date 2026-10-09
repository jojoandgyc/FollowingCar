"""Bounded detector measurements between independent full identity checks.

These proofs never contain embeddings. Detection continuity can consume a
full verification's budget, but cannot renew it or teach an identity gallery.
"""
from dataclasses import dataclass
import math

from .camera_geometry import horizontal_center_displacement


# CAP171-370: the following full result often completes 400-583 ms after
# the preceding verified capture. A 350 ms proof expired during that ordinary
# full check and introduced a motor zero after an otherwise valid fast frame.
# This remains an absolute, non-renewable deadline from the FULL capture.
FULL_PROOF_TTL_SEC = .60
MAX_FULL_RESULT_AGE_SEC = .35
FULL_RECHECK_INTERVAL_SEC = .20
# CAP49/53/57: valid full checks were ~195/202ms apart. Allow that
# capture cadence without clearing the verification streak. This is NOT
# extra processing time, a sliding identity lease or a motor authorization.
MAX_DETECTION_GAP_SEC = .21
MAX_DETECTION_AGE_SEC = .18
MAX_FAST_FRAMES = 2
MAX_COLOR_DISTANCE = .04


@dataclass(frozen=True)
class DetectorObservation:
    capture: int
    timestamp: float
    yaw: float
    bbox: tuple
    score: float
    width: int
    height: int
    color: tuple = ()


@dataclass(frozen=True)
class DetectorBackgroundProof:
    """A full-check exclusion, never a newly inferred identity permission."""
    track_id: int
    verified: DetectorObservation
    previous: DetectorObservation
    eligibility: tuple


@dataclass(frozen=True)
class DetectorProof:
    uid: int
    track_id: int
    verified: DetectorObservation
    previous: DetectorObservation
    full_count: int
    fast_count: int = 0
    backgrounds: tuple = ()

    @property
    def deadline(self):
        return self.verified.timestamp + FULL_PROOF_TTL_SEC


@dataclass(frozen=True)
class DetectorContinuationPlan:
    proof: DetectorProof
    observation: DetectorObservation
    frame_index: int
    epoch: int
    planned_at: float
    context: tuple = ()
    source_index: int = 0
    backgrounds: tuple = ()  # (source index, background proof, fresh observation)


def separated_observations(first, second):
    """No overlap/crossing permission just because an old crop was excluded."""
    ax1, ay1, ax2, ay2 = first.bbox
    bx1, by1, bx2, by2 = second.bbox
    return (first.width == second.width and first.height == second.height
            and (max(bx1-ax2, ax1-bx2) >= .02*first.width
                 or max(by1-ay2, ay1-by2) >= .02*first.height))


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def color_signature(value):
    try:
        values = tuple(float(v) for v in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if len(values) != 16 or not all(math.isfinite(v) and v >= 0 for v in values):
        return None
    norm = math.sqrt(sum(v*v for v in values))
    return tuple(v/norm for v in values) if norm > 1e-12 else None


def color_matches(left, right):
    return bool(len(left) == len(right) == 16
                and 1.-sum(a*b for a, b in zip(left, right)) <= MAX_COLOR_DISTANCE)


def capture_observation(detection, context, width, height):
    """Freeze only physical capture fields; never accept predicted metadata."""
    if not isinstance(context, dict) or any(context.get(key) for key in (
        "search_reacquire_context_active", "searching", "identity_control_rejected",
        "identity_recheck_pending", "observation_only", "search_observation_only",
    )) or context.get("is_fresh", True) is not True:
        return None
    capture = finite_number(context.get("capture_frame_id"))
    stamp = finite_number(context.get("capture_timestamp"))
    yaw = finite_number(context.get("integrated_yaw_deg"))
    if (capture is None or capture <= 0 or not capture.is_integer()
            or stamp is None or stamp <= 0 or yaw is None
            or width <= 0 or height is None or height <= 0):
        return None
    try:
        bbox = tuple(float(v) for v in detection.bbox)
        score = float(detection.score)
    except (TypeError, ValueError, OverflowError):
        return None
    if (len(bbox) != 4 or not all(math.isfinite(v) for v in (*bbox, score))
            or int(detection.class_id) != 0 or not 0 <= score <= 1
            or not 0 <= bbox[0] < bbox[2] <= width
            or not 0 <= bbox[1] < bbox[3] <= height):
        return None
    return DetectorObservation(int(capture), stamp, yaw, bbox, score, int(width), int(height))


def geometry_matches(previous, current, hfov, *, anchor=False):
    """Check raw boxes, compensating camera yaw with the existing convention."""
    if (previous.width != current.width or previous.height != current.height
            or not math.isfinite(hfov) or hfov <= 0):
        return False
    px1, py1, px2, py2 = previous.bbox
    x1, y1, x2, y2 = current.bbox
    delta = ((x1+x2)-(px1+px2))/(2*current.width)
    yaw_delta = current.yaw-previous.yaw
    _, compensated = horizontal_center_displacement(
        current_center=(x1+x2)/(2*current.width),
        previous_center=(px1+px2)/(2*current.width),
        current_yaw=current.yaw, previous_yaw=previous.yaw, camera_hfov_deg=hfov,
    )
    area, old_area = (x2-x1)*(y2-y1), (px2-px1)*(py2-py1)
    intersection = max(0., min(x2, px2)-max(x1, px1))*max(0., min(y2, py2)-max(y1, py1))
    iou = intersection/(area+old_area-intersection)
    return bool(abs(yaw_delta) <= 15 and abs(delta) <= .15
                and iou >= (.25 if anchor else .35)
                and compensated <= (.15 if anchor else .08)
                and abs((y1+y2)-(py1+py2))/(2*current.height) <= .10
                and min(area, old_area)/max(area, old_area) >= .65
                and min(x2-x1, px2-px1)/max(x2-x1, px2-px1) >= .65
                and min(y2-y1, py2-py1)/max(y2-y1, py2-py1) >= .65)


def continuation_reason(proof, observation, now, hfov):
    now = finite_number(now)
    if now is None or now < observation.timestamp:
        return "invalid_clock"
    if now >= proof.deadline:
        return "verification_expired"
    if (observation.capture <= proof.previous.capture
            or observation.timestamp <= proof.previous.timestamp):
        return "nonnew_capture"
    if now >= observation.timestamp + MAX_DETECTION_AGE_SEC:
        return "detection_stale"
    if observation.timestamp > proof.previous.timestamp + MAX_DETECTION_GAP_SEC:
        return "detection_gap"
    if proof.full_count < 2:
        return "full_verification_streak"
    if proof.fast_count >= MAX_FAST_FRAMES:
        return "fast_budget_exhausted"
    if observation.timestamp >= proof.verified.timestamp+FULL_RECHECK_INTERVAL_SEC:
        return "full_recheck_due"
    if not geometry_matches(proof.previous, observation, hfov):
        return "adjacent_geometry"
    if not geometry_matches(proof.verified, observation, hfov, anchor=True):
        return "verified_geometry"
    return None
