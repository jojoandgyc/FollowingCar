"""Current formal identity proof for search direction, not motion authority."""

import math
from collections.abc import Mapping


def confirmed_search_candidate_uid(
    *, active_uid, raw_track_id, capture_frame_id, capture_timestamp,
    records, observations, excluded=False, person_class_id=0,
):
    """Return the current assigned UID, never a retained mapping/ReID guess.

    The caller must uniquely bind the candidate's *detector* box to this raw
    track. Matching feature distance and repeated detector observations only
    justify observation; they cannot replace the bank's current UID decision.
    """
    try:
        uid, track, cap = int(active_uid), int(raw_track_id), int(capture_frame_id)
        stamp = float(capture_timestamp)
    except (TypeError, ValueError, OverflowError):
        return 0
    # The detector-probe track uses a negative synthetic ID. It is admissible
    # only with the very same formal UID/provenance checks as ordinary tracks.
    if uid <= 0 or cap <= 0 or not math.isfinite(stamp) or stamp <= 0 or excluded:
        return 0
    try:
        owned = [r for r in records if int(getattr(r, "track_id", -1)) == track]
        if len(owned) != 1:
            return 0
        rec = owned[0]
        if (int(getattr(rec, "class_id", -1)) != int(person_class_id)
                or int(getattr(rec, "time_since_update", -1)) != 0
                or int(getattr(rec, "reid_uid", 0)) != uid):
            return 0
        matched = []
        for observation in observations:
            if not isinstance(observation, Mapping):
                continue
            metadata = observation.get("sample_metadata") or {}
            if (int(observation.get("raw_track_id", -1)) == track
                    and int(metadata.get("capture_frame_id", -1)) == cap
                    and float(metadata.get("capture_timestamp", -1)) == stamp):
                matched.append(observation)
        if len(matched) != 1:
            return 0
        observation = matched[0]
        metadata = observation.get("sample_metadata") or {}
        assignment = observation.get("assignment") or {}
        geometry = assignment.get("reacquire_geometry") or {}
        if (int(observation.get("uid", 0)) != uid
                or int(assignment.get("uid", 0)) != uid
                or metadata.get("is_fresh") is not True
                or assignment.get("bbox_quality_ok") is not True
                or assignment.get("reacquire_geometry_ok") is False
                or geometry.get("ok") is False
                or assignment.get("identity_control_rejected")
                or assignment.get("search_excluded")
                or assignment.get("search_contradiction_retained")
                or geometry.get("search_cross_edge_conflict")
                or geometry.get("short_handoff_identity_conflict")):
            return 0
        for source in (assignment, metadata):
            competition = source.get("identity_competition") or {}
            if competition.get("passed") is False:
                return 0
    except (AttributeError, TypeError, ValueError, OverflowError):
        return 0
    return uid
