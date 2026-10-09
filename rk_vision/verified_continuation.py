"""Pure, source-bound evidence for continuing an already confirmed identity.

This function does not establish the provenance of a binding or a template.
The owner must supply a previously confirmed source and a same-capture pair
from its approved, recent gallery.  A ``hold`` is explicitly *not* an identity
confirmation: it may preserve that source until a fixed deadline, but grants
neither a UID, a template update, nor motor authority.
"""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Optional


@dataclass(frozen=True)
class ContinuationDecision:
    status: str
    reason: str
    source: Optional[str]
    reference_cap: Optional[int]
    pair_cap: Optional[int]
    deadline: Optional[float]


def _number(value):
    # Booleans, numeric-looking strings and nonfinite telemetry fail closed.
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _identifier(value):
    number = _number(value)
    return int(number) if number is not None and number > 0 and number.is_integer() else None


def _strong_observation(metadata):
    return bool(
        metadata.get("is_fresh") is True
        and metadata.get("quality_bbox_ok") is True
        and metadata.get("bbox_quality_tier", metadata.get("quality_tier")) == "strong"
        and not metadata.get("search_observation_only")
        and not metadata.get("observation_only")
        and not metadata.get("preferred_search_low_confidence")
    )


def evaluate_continuation(
    *, uid, track_id, source, previous: dict, current: dict,
    geometry: dict, competition_ok: bool, blocked: bool, pair: dict,
    full_limit: float, confirm_limit: float, observe_limit: float,
    max_gap: float = .35, pending_deadline=None,
) -> ContinuationDecision:
    """Evaluate new evidence without modifying the supplied dictionaries.

    The current pair must refer to one previously approved capture, not two
    independently selected gallery minima. ``qualified_comparison`` certifies
    normal comparable regions (exact coverage or a normal vertical bridge),
    not temporary pose/scale retention.  The owner remains responsible for
    recent-gallery membership and same-capture descriptor pairing.

    A tentative torso score preserves only the last *accepted* reference.
    Deadlines are capture-time based and cannot be renewed by tentative frames.
    An expired pending interval rejects even otherwise acceptable new scores.
    The optional ``previous.continuation_watermark_cap`` / ``_timestamp`` pair
    identifies the newest already-consumed capture, including held captures.
    This prevents recomputing a held frame's descriptors to promote a duplicate
    without changing the accepted reference or its deadline.
    """
    safe_source = source if isinstance(source, str) and source in ("strong", "partial") else None
    reference_cap = _identifier(previous.get("capture_frame_id")) if isinstance(previous, dict) else None
    pair_cap = _identifier(pair.get("winner_cap")) if isinstance(pair, dict) else None
    deadline = None

    def reject(reason):
        return ContinuationDecision("reject", reason, safe_source, reference_cap, pair_cap, deadline)

    identity, track = _identifier(uid), _identifier(track_id)
    if identity is None or track is None or safe_source is None:
        return reject("invalid_binding")
    if not all(isinstance(value, dict) for value in (previous, current, geometry, pair)):
        return reject("missing_evidence")
    limits = tuple(_number(value) for value in (full_limit, confirm_limit, observe_limit, max_gap))
    if any(value is None for value in limits):
        return reject("invalid_limits")
    full_bound, confirm_bound, observe_bound, gap = limits
    if full_bound < 0 or confirm_bound < 0 or observe_bound < confirm_bound or gap <= 0:
        return reject("invalid_limits")
    gap = min(.35, gap)
    if blocked is not False:
        return reject("identity_blocked")
    if competition_ok is not True:
        return reject("competition_unverified")
    if not _strong_observation(previous) or not _strong_observation(current):
        return reject("observation_unverified")
    if _identifier(previous.get("track_id")) != track or _identifier(current.get("track_id")) != track:
        return reject("track_changed")
    capture = _identifier(current.get("capture_frame_id"))
    old_time, now = _number(previous.get("capture_timestamp")), _number(current.get("capture_timestamp"))
    if reference_cap is None or capture is None or old_time is None or now is None or old_time <= 0 or now <= 0:
        return reject("missing_capture")
    deadline = old_time + gap
    if not math.isfinite(deadline):
        deadline = None
        return reject("invalid_deadline")
    if pending_deadline is not None:
        pending = _number(pending_deadline)
        if pending is None or pending <= old_time:
            return reject("invalid_pending_deadline")
        deadline = min(deadline, pending)
    watermark_cap, watermark_time = reference_cap, old_time
    watermark_keys = ("continuation_watermark_cap", "continuation_watermark_timestamp")
    if any(key in previous for key in watermark_keys):
        watermark_cap = _identifier(previous.get(watermark_keys[0]))
        watermark_time = _number(previous.get(watermark_keys[1]))
        if (not all(key in previous for key in watermark_keys)
                or watermark_cap is None or watermark_time is None
                or watermark_cap < reference_cap or watermark_time < old_time):
            return reject("invalid_continuation_watermark")
    if capture <= watermark_cap or now <= watermark_time:
        return reject("nonnew_capture")
    # Expiry is exclusive. In particular, a good score at the pending deadline
    # cannot reactivate the old source; ordinary reacquisition must re-evaluate.
    if now >= deadline:
        return reject("continuation_expired")
    if any(_number(observation.get("integrated_yaw_deg")) is None for observation in (previous, current)):
        return reject("yaw_unavailable")
    jump = _number(geometry.get("yaw_compensated_center_jump_ratio"))
    area = _number(geometry.get("area_similarity"))
    if (geometry.get("ok") is not True or jump is None or not 0 <= jump <= .15
            or area is None or not .65 <= area <= 1):
        return reject("geometry_unverified")
    # The approval boundary supplies provenance; reject obvious self/future
    # evidence here as well, so this function cannot authorize learning itself.
    if (pair.get("qualified_comparison") is not True or pair_cap is None
            or pair_cap > reference_cap
            or ("comparison_mode" in pair and pair["comparison_mode"] not in
                ("exact_coverage", "vertical_border_bridge"))):
        return reject("pair_unverified")
    full = _number(pair.get("full_distance"))
    partial = _number(pair.get("partial_distance"))
    if full is None or partial is None or full < 0 or partial < 0:
        return reject("pair_unverified")
    if full > full_bound:
        return reject("full_distance_conflict")
    if partial > observe_bound:
        return reject("partial_distance_conflict")
    status = "accept" if partial <= confirm_bound else "hold"
    reason = "verified_pair" if status == "accept" else "partial_recheck_pending"
    return ContinuationDecision(status, reason, safe_source, reference_cap, pair_cap, deadline)
