"""Shared correction ceiling for image-position visible following.

Limits are per-wheel corrections: differential = 2 * correction. No motion
authority or minimum output is created here; zero/braking always wins.
"""
import math


def effective_correction_limit(cfg, base_rpm, *, near_distance=False, policy_limit=None):
    if not getattr(cfg, "visible_steering_pid_image_error_only", False):
        return policy_limit
    limit = max(0.0, float(cfg.visible_steering_pid_max_correction_rpm))
    if near_distance or base_rpm <= 0:
        limit = min(limit, max(0.0, float(cfg.near_distance_rotation_only_max_rpm)))
    if policy_limit is not None:
        limit = min(limit, max(0.0, float(policy_limit)))
    return limit


def clamp_correction(value, limit):
    if limit is None:
        return value
    # Integer motor RPM must not round a fractional policy ceiling upward.
    bound = max(0, math.floor(limit))
    return max(-bound, min(bound, value))
