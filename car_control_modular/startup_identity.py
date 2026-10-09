"""Carry completed initial enrollment through an exact current-frame binding.

A cached UID, a tentative mapped UID, or a successful enrollment on a different
capture is not a current identity observation. This grants no motor authority.
"""
from collections.abc import Mapping

from .depth_target_geometry import resolve_depth_target_observation


def resolve_initial_identity_confirmation(**kwargs) -> bool:
    """Return provenance only for the same unique accepted detector association.

    The geometry resolver is independent of whether measured depth exists. It
    already rejects ambiguous boxes, UID/capture mismatches and weak/rejected
    identity observations. Current image age is checked by the controller.
    """
    try:
        records = tuple(kwargs["observations"])
    except (KeyError, TypeError):
        return False
    observation = resolve_depth_target_observation(**{**kwargs, "observations": records})
    if observation is None:
        return False
    for record in records:
        if not isinstance(record, Mapping):
            continue
        if record.get("raw_track_id") != observation.raw_track_id:
            continue
        assignment = record.get("assignment", {})
        competition = record.get("sample_metadata", {}).get("identity_competition") or {}
        return bool(isinstance(assignment, Mapping)
                    and assignment.get("initial_identity_confirmed") is True
                    and not assignment.get("identity_control_rejected")
                    and isinstance(competition, Mapping)
                    and competition.get("passed") is not False)
    return False
