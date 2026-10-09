"""Narrow contraction of an already guarded forward pair, never authority."""
import math
from .lateral_intent import LateralControlIntent


def contract_straight_forward_handoff(uid, intent, applied, planned_axes,
                                      current_axes, depth_cap_rpm, now, *, cleared=False):
    """Bound an unsent straight packet by newer same-UID forward evidence.

    A visual park *candidate* only stops translation after the runtime has
    admitted a real parking owner. This arithmetic cannot admit yaw, reverse,
    a zero intent, or an increase of either already guarded wheel.
    """
    if (not isinstance(intent, LateralControlIntent)
            or intent.target_id != uid or intent.mode != "forward"
            or intent.bbox_quality != "reliable" or intent.hold_zero
            or intent.near_distance_mode or intent.forward_countersteer
            or intent.countersteer_rpm or intent.initial_correction_rpm != 0
            or planned_axes is None or current_axes is None
            or planned_axes[0] != current_axes[0] or current_axes[0] != uid
            or planned_axes[3] != 0 or current_axes[3] != 0
            or applied[0] != applied[1] or applied[0] <= 0):
        return None
    values = (now, intent.capture_timestamp, intent.published_at,
              intent.valid_until, applied[0], planned_axes[2],
              current_axes[2], depth_cap_rpm)
    if (not all(math.isfinite(v) for v in values)
            or not 0 < intent.capture_timestamp <= intent.published_at <= now
            or not 0 <= now-intent.capture_timestamp <= .50
            or (not cleared and not intent.valid(now))
            or (cleared and (intent.valid(now) or now-intent.valid_until > .10))
            or current_axes[2] <= 0 or depth_cap_rpm <= 0
            or applied[0] > planned_axes[2]):
        return None
    speed = math.floor(min(applied[0], current_axes[2], depth_cap_rpm))
    return (speed, speed) if speed > 0 else None


def contract_fresh_forward_handoff(uid, intent, applied, planned_axes,
                                   current_axes, depth_cap_rpm, now, *, yaw_zeroed=False):
    """Retarget an unsent forward pair to a newer grant without adding RPM.

    Only wheel arithmetic lives here. The caller must prove that the grant is
    *new*, that this is the same visible UID, and that STOP, encoder, intent,
    and physical write deadlines still pass. A zeroed visual yaw may preserve
    straight translation, but never reintroduce the old turn.
    """
    if (not isinstance(intent, LateralControlIntent)
            or intent.target_id != uid or intent.mode != "forward"
            or intent.bbox_quality != "reliable" or intent.hold_zero
            or intent.near_distance_mode or intent.forward_countersteer
            or intent.countersteer_rpm or planned_axes is None
            or current_axes is None or len(planned_axes) != 4
            or len(current_axes) != 4 or planned_axes[0] != uid
            or current_axes[0] != uid):
        return None
    old_left, old_right = applied
    old_base = .5 * (old_left + old_right)
    old_yaw = .5 * (old_left - old_right)
    new_base, new_yaw = current_axes[2:]
    values = (now, intent.capture_timestamp, intent.published_at,
              intent.valid_until, old_left, old_right,
              *planned_axes[2:], new_base, new_yaw, depth_cap_rpm)
    if (not all(math.isfinite(v) for v in values)
            or not 0 < intent.capture_timestamp <= intent.published_at <= now
            or not 0 <= now-intent.capture_timestamp <= .25
            or not intent.valid(now) or min(applied) < 0 or old_base <= 0
            or old_base > planned_axes[2] or old_yaw != planned_axes[3]
            or new_base <= 0 or depth_cap_rpm <= 0
            or abs(new_yaw) > new_base or old_yaw * new_yaw < 0
            or (yaw_zeroed and new_yaw != 0)
            or (intent.park_requested and (old_yaw != 0 or new_yaw != 0))):
        return None
    base = math.floor(min(old_base, new_base, depth_cap_rpm,
                          old_left-new_yaw, old_right+new_yaw))
    if base <= 0 or base < abs(new_yaw):
        return None
    pair = (int(round(base+new_yaw)), int(round(base-new_yaw)))
    if (min(pair) < 0 or sum(pair) <= 0
            or .5*(pair[0]-pair[1]) != new_yaw
            or any(new > old for new, old in zip(pair, applied))):
        return None
    return pair


def contract_fresh_forward_neutral_handoff(uid, intent, applied, planned_axes,
                                           current_axes, depth_cap_rpm, now, *,
                                           yaw_zeroed=False):
    """Bridge a genuine yaw sign change with a smaller *straight* pair.

    An old left turn cannot authorize a new right turn (or vice versa). The
    next ordinary tick must guard the new turn. For this one unsent packet,
    remove differential and reduce *both* guarded wheels under the current
    forward Depth cap. No identity, STOP or physical lease is granted here.
    """
    if (planned_axes is None or current_axes is None
            or len(planned_axes) != 4 or len(current_axes) != 4
            or not all(math.isfinite(v) for v in (*planned_axes[2:], *current_axes[2:]))
            or planned_axes[3]*current_axes[3] >= 0
            or abs(current_axes[3]) > current_axes[2]):
        return None
    neutral_axes = (current_axes[0], current_axes[1], current_axes[2], 0.)
    return contract_fresh_forward_handoff(
        uid, intent, applied, planned_axes, neutral_axes, depth_cap_rpm, now,
        yaw_zeroed=yaw_zeroed)


def straight_park_candidate_contraction(uid, intent, applied, planned_axes, current_axes):
    """A yaw-stop candidate may accompany a strictly smaller straight base.

    ``park_requested`` is visual evidence for the producer's measured-motion
    parking gate, not a whole-chassis STOP owner. This predicate grants no
    motion: the caller still binds the same Depth grant/receipt and checks
    actual parking owners, wheel feedback and physical deadlines. It cannot
    adopt a new yaw, pivot, reverse, near-mode intent or countersteer.
    """
    return bool(
        isinstance(intent, LateralControlIntent) and intent.target_id == uid
        and intent.mode == "forward" and intent.bbox_quality == "reliable"
        and intent.park_requested and intent.initial_correction_rpm == 0
        and not intent.near_distance_mode and not intent.forward_countersteer
        and not intent.countersteer_rpm
        and planned_axes is not None and current_axes is not None
        and planned_axes[0] == current_axes[0] == uid
        and planned_axes[3] == current_axes[3] == 0
        and applied[0] == applied[1] and applied[0] > 0
        and contract_forward_base(applied, planned_axes, current_axes) is not None
    )


def ordinary_intent_handoff(uid, current, previous, now):
    """Qualify a new capture, not a braking/identity handoff or an authority.

    The caller must still bind the same physical grant and completed write,
    contract BOTH wheels and recheck this exact object at the write boundary.
    The intent's base is only a visual snapshot, never longitudinal authority.
    """
    previous = tuple(intent for intent in previous if intent is not None)
    if not previous or not any(intent is not current for intent in previous):
        return False
    for intent in (current, *previous):
        if (not isinstance(intent, LateralControlIntent) or intent.target_id != uid
                or intent.mode != "forward" or intent.bbox_quality != "reliable"
                or any(getattr(intent, name) for name in (
                    "hold_zero", "near_distance_mode", "park_requested",
                    "forward_countersteer", "countersteer_rpm"))
                or not all(math.isfinite(v) for v in (
                    intent.capture_timestamp, intent.published_at, intent.valid_until,
                    intent.base_rpm, intent.correction_limit_rpm))
                or intent.base_rpm < 0 or intent.capture_timestamp <= 0
                or intent.capture_frame_id <= 0):
            return False
    if (not math.isfinite(now) or not 0 <= now-current.capture_timestamp <= .25
            or not current.capture_timestamp <= current.published_at <= now
            or not current.valid(now)):
        return False
    return all(intent is current or (
        current.sequence > intent.sequence
        and current.capture_frame_id > intent.capture_frame_id
        and current.capture_timestamp > intent.capture_timestamp
        and current.published_at >= intent.published_at
    ) for intent in previous)


def contract_forward_axes(applied, planned_axes, current_axes):
    """Adopt a live same-direction yaw by removing common forward speed.

    Unlike simply adding more differential, neither wheel can increase its
    already guarded RPM. Authority/receipt/intent checks belong to the caller.
    Opposite steering, pivots and an increased canonical base require a new
    full plan. Integer rounding must preserve the exact authorised yaw.
    """
    if (planned_axes is None or current_axes is None
            or planned_axes[0] != current_axes[0]):
        return None
    if not all(math.isfinite(v) for v in (*applied, *planned_axes[2:], *current_axes[2:])):
        return None
    old_base, old_yaw = .5 * sum(applied), .5 * (applied[0] - applied[1])
    allowed_base, yaw = current_axes[2:]
    if (min(applied) < 0 or old_base <= 0 or old_base > planned_axes[2]
            or old_yaw != planned_axes[3] or not 0 < allowed_base <= planned_axes[2]
            or old_yaw * yaw < 0 or abs(yaw) > allowed_base):
        return None
    base = min(old_base, allowed_base, applied[0] - yaw, applied[1] + yaw)
    if base <= 0 or base < abs(yaw):
        return None
    pair = (int(math.floor(base + yaw)), int(math.floor(base - yaw)))
    if (min(pair) < 0 or sum(pair) <= 0 or .5 * (pair[0] - pair[1]) != yaw
            or any(new > old for new, old in zip(pair, applied))):
        return None
    return pair


def contract_forward_speed(applied, max_base):
    """Final same-yaw speed contraction; never round above a live cap."""
    if not all(math.isfinite(v) for v in (*applied, max_base)):
        return None
    base = .5 * sum(applied)
    if min(applied) < 0 or base <= 0 or max_base <= 0:
        return None
    reduction = max(0, math.ceil(base - max_base))
    pair = (applied[0] - reduction, applied[1] - reduction)
    return pair if min(pair) >= 0 and sum(pair) > 0 else None


def contract_forward_yaw(applied, planned_axes, current_axes):
    """Keep the identical base and keep/reduce a same-direction differential.

    The caller still owns every identity, safety, feedback and lease check.
    Exclude earlier guard/assist changes and fractional rounding that would
    alter the approved mean speed. This performs no retry or state change.
    """
    if planned_axes is None or current_axes is None or planned_axes[0] != current_axes[0]:
        return None
    values = (*applied, *planned_axes[2:], *current_axes[2:])
    if not all(math.isfinite(value) for value in values):
        return None
    base = .5 * sum(applied)
    yaw = .5 * (applied[0] - applied[1])
    new_yaw = current_axes[3]
    if (min(applied) < 0 or base <= 0
            or planned_axes[2] != base or current_axes[2] != base
            or planned_axes[3] != yaw
            or abs(new_yaw) > abs(yaw) or yaw * new_yaw < 0):
        return None
    contracted = (int(round(base + new_yaw)), int(round(base - new_yaw)))
    if (min(contracted) < 0 or .5 * sum(contracted) != base
            or .5 * (contracted[0] - contracted[1]) != new_yaw):
        return None
    return contracted


def contract_forward_base(applied, planned_axes, current_axes):
    """Reduce an ordinary forward pair for a lower base on the same yaw axis.

    This is only arithmetic, not permission to move. The caller must verify
    the original depth grant, identity, feedback, stop ownership and deadline.
    No wheel may reverse or receive more RPM than the already guarded pair.
    """
    if (planned_axes is None or current_axes is None
            or planned_axes[0] != current_axes[0]
            # A lateral publication may advance its revision while both the
            # old and new commands remain exactly straight. That metadata
            # change alone does not turn a smaller *forward* base into a
            # reversal. Any nonzero yaw still needs the ordinary rebuild.
            or (planned_axes[1] != current_axes[1]
                and (planned_axes[3] != 0 or current_axes[3] != 0))):
        return None
    values = (*applied, *planned_axes[2:], *current_axes[2:])
    if not all(math.isfinite(value) for value in values):
        return None
    old_base = .5 * sum(applied)
    old_yaw = .5 * (applied[0] - applied[1])
    allowed_base, new_yaw = current_axes[2:]
    if (min(applied) < 0 or old_base <= 0 or allowed_base <= 0
            or allowed_base >= planned_axes[2] or old_base > planned_axes[2]
            or planned_axes[3] != old_yaw or new_yaw != old_yaw
            or abs(new_yaw) > allowed_base):
        return None
    # The initial fresh-depth read may already have reduced the guarded pair
    # below planned_axes before this final reduction is observed.
    new_base = min(old_base, allowed_base)
    contracted = (int(round(new_base + new_yaw)), int(round(new_base - new_yaw)))
    if (min(contracted) < 0 or any(new > old for new, old in zip(contracted, applied))
            or .5 * sum(contracted) != new_base
            or .5 * (contracted[0] - contracted[1]) != new_yaw):
        return None
    return contracted
