"""Narrow contraction of an already guarded forward pair, never authority."""
import math


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
