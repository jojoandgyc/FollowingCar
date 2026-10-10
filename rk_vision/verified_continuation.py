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


@dataclass(frozen=True)
class PoseContinuationDecision:
    """A bounded tracking decision, never an identity or learning approval."""

    status: str
    reason: str
    uid: Optional[int]
    track_id: Optional[int]
    source: Optional[str]
    origin_cap: Optional[int]
    origin_timestamp: Optional[float]
    deadline: Optional[float]
    last_cap: Optional[int]
    last_timestamp: Optional[float]
    recovery_streak: int


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


def _pose_crop(metadata):
    """Validated crop geometry; this does not recognize anatomical pose."""
    box = metadata.get("detector_bbox")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    values = [_number(value) for value in (*box, metadata.get("image_width"),
                                         metadata.get("image_height"))]
    if any(value is None for value in values):
        return None
    x1, y1, x2, y2, width, height = values
    if (not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height
            or x2-x1 < 64 or y2-y1 < 160
            or x1 <= .02*width or width-x2 <= .02*width):
        return None
    aspect = (x2-x1)/(y2-y1)
    if not .25 <= aspect <= 1.20:
        return None
    return dict(width=width, height=height, body_height=y2-y1,
                aspect=aspect, vertical=(y1 <= .02*height, height-y2 <= .02*height),
                top=y1, bottom=y2)


def evaluate_pose_continuation(
    *, uid, track_id, source, origin: dict, previous: dict, current: dict,
    geometry: dict, competition_ok: bool, blocked: bool,
    full_distance, full_limit, partial_distance, confirm_limit,
    pair_comparable: bool, state=None, observe_limit=.45,
    max_gap=.35, max_duration=2., retention_full_limit=None,
) -> PoseContinuationDecision:
    """Keep a previously confirmed binding through one bounded crop change.

    The owner supplies the independently accepted ``origin`` and retains it
    unchanged for the episode. ``previous`` is the last consumed candidate,
    including uncertain frames; it is never a template or a new origin.
    ``state`` is the preceding continue decision as a dictionary. The owner
    must discard this lineage on any negative geometry or competition result.

    A grey torso score can enter only with measurable narrowing or a nearby
    vertical crop change. Later torso disagreement remains unknown during the
    fixed episode, including brief returns to the original crop shape.
    Full-body support, fresh same-track geometry,
    a fixed deadline and two comparable recovery samples prevent a weak torso
    sequence from becoming a rolling identity proof. All successful results
    preserve tracking only; template learning stays frozen, including recovery.
    """
    identity, track = _identifier(uid), _identifier(track_id)
    safe_source = source if isinstance(source, str) and source in ("strong", "partial") else None
    origin_cap = _identifier(origin.get("capture_frame_id")) if isinstance(origin, dict) else None
    origin_time = _number(origin.get("capture_timestamp")) if isinstance(origin, dict) else None
    deadline = None
    last_cap = last_time = None
    streak = 0

    def decision(status, reason):
        return PoseContinuationDecision(status, reason, identity, track, safe_source,
            origin_cap, origin_time, deadline, last_cap, last_time, streak)

    def reject(reason):
        return decision("reject", reason)

    if identity is None or track is None or safe_source is None:
        return reject("invalid_binding")
    if not all(isinstance(value, dict) for value in (origin, previous, current, geometry)):
        return reject("missing_evidence")
    numbers = tuple(_number(value) for value in
                    (full_distance, full_limit, partial_distance, confirm_limit,
                     observe_limit, max_gap, max_duration))
    if any(value is None for value in numbers):
        return reject("invalid_limits_or_scores")
    full, full_bound, partial, confirm, observe, gap, duration = numbers
    retention_bound = full_bound if retention_full_limit is None else _number(retention_full_limit)
    if (full < 0 or partial < 0 or full_bound < 0 or confirm < 0
            or observe < confirm or gap <= 0 or duration <= 0
            or retention_bound is None or retention_bound < 0):
        return reject("invalid_limits_or_scores")
    gap, duration = min(.35, gap), min(2., duration)
    # A caller may explicitly retain an established episode at its ordinary
    # full-body match bound. This never changes the stricter entry threshold.
    full_bound = (min(.30, full_bound) if state is None else
                  min(.30 if retention_full_limit is None else .38, retention_bound))
    if blocked is not False:
        return reject("identity_blocked")
    if competition_ok is not True:
        return reject("competition_unverified")
    if pair_comparable is not True and pair_comparable is not False:
        return reject("pair_unverified")
    if not all(_strong_observation(value) for value in (origin, previous, current)):
        return reject("observation_unverified")
    if any(_identifier(value.get("track_id")) != track for value in (origin, previous, current)):
        return reject("track_changed")
    old_cap, capture = (_identifier(value.get("capture_frame_id")) for value in (previous, current))
    old_time, now = (_number(value.get("capture_timestamp")) for value in (previous, current))
    if (origin_cap is None or origin_time is None or origin_time <= 0
            or old_cap is None or capture is None or old_time is None or now is None
            or old_cap < origin_cap or old_time < origin_time):
        return reject("missing_capture")
    deadline = origin_time + duration
    if not math.isfinite(deadline):
        deadline = None
        return reject("invalid_deadline")
    if state is None:
        if old_cap != origin_cap or old_time != origin_time:
            return reject("origin_unverified")
    else:
        if not isinstance(state, dict):
            return reject("invalid_pose_state")
        prior_deadline = _number(state.get("deadline"))
        prior_streak = _number(state.get("recovery_streak"))
        if (state.get("status") != "continue"
                or _identifier(state.get("uid")) != identity
                or _identifier(state.get("track_id")) != track
                or state.get("source") != safe_source
                or _identifier(state.get("origin_cap")) != origin_cap
                or _number(state.get("origin_timestamp")) != origin_time
                or _identifier(state.get("last_cap")) != old_cap
                or _number(state.get("last_timestamp")) != old_time
                or prior_deadline is None or not origin_time < prior_deadline <= origin_time+2.
                or prior_streak is None or prior_streak not in (0., 1.)):
            return reject("invalid_pose_state")
        deadline = min(deadline, prior_deadline)
        streak = int(prior_streak)
    if capture <= old_cap or now <= old_time:
        return reject("nonnew_capture")
    if now >= deadline:
        return reject("pose_continuation_expired")
    if now >= old_time+gap:
        return reject("continuation_gap")
    if any(_number(value.get("integrated_yaw_deg")) is None for value in (origin, previous, current)):
        return reject("yaw_unavailable")
    jump = _number(geometry.get("yaw_compensated_center_jump_ratio"))
    area = _number(geometry.get("area_similarity"))
    if (geometry.get("ok") is not True or jump is None or not 0 <= jump <= .15
            or area is None or not .65 <= area <= 1):
        return reject("geometry_unverified")
    if full > full_bound:
        return reject("full_distance_conflict")
    crops = [_pose_crop(value) for value in (origin, previous, current)]
    if any(value is None for value in crops):
        return reject("crop_unverified")
    initial, prior, crop = crops
    if any((value["width"], value["height"]) != (initial["width"], initial["height"])
           for value in crops[1:]):
        return reject("crop_unverified")
    height_similarity = min(crop["body_height"], prior["body_height"])/max(crop["body_height"], prior["body_height"])
    if height_similarity < .85:
        return reject("crop_discontinuity")
    narrowing = crop["aspect"]/initial["aspect"]
    vertical_changed = [index for index, values in enumerate(zip(crop["vertical"], initial["vertical"]))
                        if values[0] != values[1]]
    nearby_border = (len(vertical_changed) == 1
        and abs(crop[("top", "bottom")[vertical_changed[0]]]
                - initial[("top", "bottom")[vertical_changed[0]]]) <= .05*crop["height"])
    if state is None:
        if not confirm < partial <= observe:
            return reject("entry_torso_unverified")
        if narrowing > .85 and not nearby_border:
            return reject("pose_change_unverified")
        reason = "pose_change_torso_uncertain"
    else:
        if pair_comparable and partial <= confirm:
            streak += 1
        else:
            streak = 0
        reason = "pose_recovery_confirmed" if streak == 2 else (
            "pose_recovery_pending" if streak else "pose_change_torso_uncertain")
    last_cap, last_time = capture, now
    return decision("recover" if streak == 2 else "continue", reason)


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
