"""Brake-only candidate continuity. Never assigns a UID or a search direction."""
from collections.abc import Mapping
import math


def provisional_reacquire_direction(assignment):
    """Identify only a newly armed late/soft binding, not all quarantined UIDs.

    CAP589 obtained a UID before its first post-binding verification. That
    single candidate must not replace a trusted search-direction observation.
    Later accepted continuation/recovery can provide direction evidence while
    gallery writes remain quarantined; no one-second gallery wait is imposed.

    False is *not* identity or motion authorization. The caller must still
    check current-capture provenance, quality, competition and exclusions.
    Missing legacy metadata preserves those existing checks unchanged.
    """
    if not isinstance(assignment, Mapping):
        return False
    if not (
        assignment.get("reason") in (
            "preferred_search_late_reacquire",
            "preferred_search_mapped_late_reacquire",
            "preferred_search_soft_reacquire",
        )
        and assignment.get("template_update_quarantined") is True
        and assignment.get("template_quarantine_reason") == "armed"
        and assignment.get("template_quarantine_streak") == 0
        and not isinstance(assignment.get("template_quarantine_streak"), bool)
    ):
        return False
    continuation = assignment.get("identity_continuation")
    continuation = continuation if isinstance(continuation, Mapping) else {}
    competition = assignment.get("identity_competition")
    competition = competition if isinstance(competition, Mapping) else {}
    # Contradictory positive flags must not override a current explicit veto.
    if (assignment.get("identity_control_rejected")
            or assignment.get("search_excluded")
            or assignment.get("reacquire_geometry_ok") is False
            or competition.get("passed") is False
            or continuation.get("status") == "reject"):
        return True
    return not (continuation.get("status") == "accept"
                or assignment.get("reacquire_control_recovered") is True)


class SearchBrakeObservation:
    def __init__(self):
        self.last = None

    def centered_candidate(self, observations, uid, cap, stamp, width, height, now):
        if (uid is None or width <= 0 or height <= 0 or not math.isfinite(stamp)
                or not 0 < stamp <= now or now-stamp > .25):
            return False
        candidates = []
        for obs in observations:
            meta, assignment = obs.get("sample_metadata") or {}, obs.get("assignment") or {}
            if (meta.get("capture_frame_id") != cap or meta.get("capture_timestamp") != stamp
                    or meta.get("is_fresh") is not True
                    or assignment.get("bbox_quality_ok") is not True
                    or assignment.get("search_excluded") or assignment.get("identity_control_rejected")
                    or (meta.get("identity_competition") or {}).get("passed") is False
                    or (assignment.get("identity_competition") or {}).get("passed") is False
                    or uid not in (obs.get("uid"), assignment.get("best_uid"))):
                continue
            bbox = obs.get("detector_bbox")
            if bbox is None or len(bbox) != 4:
                continue
            a,b,c,d = map(float, bbox)
            if not (all(math.isfinite(v) for v in (a,b,c,d))
                    and 0 <= a < c <= width and 0 <= b < d <= height
                    and c-a >= 24 and d-b >= 48):
                continue
            candidates.append((obs.get("raw_track_id"), (a+c)/2/width, (c-a)*(d-b), obs.get("uid") == uid))
        if len(candidates) != 1:
            self.last = None
            return False
        raw, x, area, confirmed = candidates[0]
        previous = self.last
        if previous is not None and (cap <= previous[0] or stamp <= previous[1]):
            return False  # repeated frames neither count nor rearm the window
        self.last = (cap, stamp, raw, x, area)
        continuous = bool(previous is not None and raw == previous[2]
                          and 0 < stamp-previous[1] <= .25
                          and abs(x-previous[3]) <= .20 and .6 <= area/previous[4] <= 1.67)
        # Wider than the center deadband: uncertain identity cannot authorize
        # an immediate restart just outside it. This veto has a bounded lifetime.
        return continuous and not confirmed and abs(x-.5) <= .08
