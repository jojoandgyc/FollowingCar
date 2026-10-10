"""Current associated position, not an identity or longitudinal renewal.

The tracker provides independently checked association evidence. This boundary
binds it to the actual current capture and existing identity publication before
any controller may consume it. Ordinary diagnostic dictionaries are not proof.
"""
from dataclasses import dataclass
import math


def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


@dataclass(frozen=True)
class AssociatedPosition:
    uid: int
    track_id: int
    capture: int
    timestamp: float
    bbox: tuple
    reference_capture: int
    reference_timestamp: float
    expires_at: float
    confidence: float
    # The owner of the original permission is not necessarily the detector's
    # current raw track. An unbound probe retains its negative observed id;
    # it must never pretend DeepSORT assigned it to the old positive track.
    observed_track_id: int = 0
    source: str = "low_score_existing_track"


def current_associated_position(observations, *, uid, capture, timestamp, width, height, now):
    """Accept one explicit current target-associated observation, never infer it.

    Strong appearance, ambiguity and identity contradictions are checked in the
    tracker. Re-check provenance here; a copied old diagnostic or malformed
    provider cannot extend the original half-second independent anchor budget.
    """
    if (type(uid) is not int or uid <= 0 or type(capture) is not int or capture <= 0
            or not all(_finite(v) for v in (timestamp, now, width, height))
            or not 0 < timestamp <= now or min(width, height) <= 0):
        return None
    admitted = []
    for item in observations or ():
        if not isinstance(item, dict):
            return None
        assignment = item.get('assignment') or {}
        metadata = item.get('sample_metadata') or {}
        if not isinstance(assignment, dict) or not isinstance(metadata, dict):
            return None
        proof = assignment.get('low_score_position_evidence')
        probe = assignment.get('follow_only_position_evidence')
        if proof is not None and probe is not None:
            return None
        is_probe = probe is not None
        if is_probe:
            proof = probe
        if proof is None:
            continue
        if not isinstance(proof, dict):
            return None
        ref_cap, ref_ts = proof.get('reference_capture_frame_id'), proof.get('reference_capture_timestamp')
        observed_track = proof.get('track_id')
        track = proof.get('reference_track_id') if is_probe else observed_track
        until = proof.get('expires_at')
        box, score = proof.get('bbox'), metadata.get('detector_confidence')
        if (proof.get('identity_authorized') is not False or proof.get('learning_allowed') is not False
                or proof.get('uid') != uid or type(track) is not int or track <= 0
                or type(observed_track) is not int
                or item.get('raw_track_id') != observed_track or item.get('uid') != 0
                or metadata.get('is_fresh') is not True
                or proof.get('capture_frame_id') != capture or metadata.get('capture_frame_id') != capture
                or proof.get('capture_timestamp') != timestamp or metadata.get('capture_timestamp') != timestamp
                or type(ref_cap) is not int or not 0 < ref_cap < capture
                or not all(_finite(v) for v in (ref_ts, until, score))
                or not 0 < ref_ts < timestamp <= now < until <= ref_ts + .5 + 1e-9
                or not 0 < score <= 1
                or not isinstance(box, (tuple, list)) or len(box) != 4
                or not all(_finite(v) for v in box)):
            return None
        if is_probe:
            gallery, jump, area = (proof.get(name) for name in
                                  ('gallery_distance', 'center_jump_ratio', 'area_similarity'))
            source_index = metadata.get('source_detection_index')
            if (proof.get('source') != 'follow_only_detector_probe'
                    or observed_track >= 0 or assignment.get('uid') != 0
                    or uid not in (assignment.get('best_uid'), assignment.get('mapped_uid'))
                    or metadata.get('quality_bbox_ok') is not True
                    or type(source_index) is not int or source_index < 0
                    or score < .50
                    or not all(_finite(v) for v in (gallery, jump, area))
                    or not 0 <= gallery <= .30 or not 0 <= jump <= .20 or not .55 <= area <= 1.
                    or assignment.get('search_excluded')
                    or assignment.get('search_contradiction_retained')
                    or assignment.get('reason') in ('mapped_geometry_reject', 'identity_center_jump_reject')):
                return None
        elif (proof.get('source') != 'low_score_existing_track'
                or assignment.get('reason') not in ('low_score_observation_only', 'similar_follow_observe')
                or assignment.get('mapped_uid') != uid
                or metadata.get('low_score_continuation') is not True):
            return None
        x1, y1, x2, y2 = box
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            return None
        if tuple(item.get('detector_bbox') or ()) != tuple(box):
            return None
        admitted.append(AssociatedPosition(uid, track, capture, timestamp, tuple(box),
            ref_cap, ref_ts, until, score, observed_track, proof['source']))
    return admitted[0] if len(admitted) == 1 else None
