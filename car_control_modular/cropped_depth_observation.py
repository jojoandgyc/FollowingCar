"""Bounded ranging permission for an accepted identity's edge crop, not motion."""
import math
from dataclasses import replace

from .depth_target_geometry import resolve_braking_target_observation


def resolve_cropped_depth_observation(*, bank, observations, uid, raw_track_id,
                                      display_bbox, capture_id, capture_timestamp,
                                      width, height, now):
    try:
        if (uid <= 0 or bank.track_to_uid.get(raw_track_id) != uid
                or uid in bank._reacquire_control_suspects
                or uid in bank._geometry_revoked_uids
                or raw_track_id in bank._mapped_geometry_conflicts
                or bank._reacquire_quarantine.is_held(uid)):
            return None, 'identity_unqualified'
        if not 0 <= now - capture_timestamp <= .25:
            return None, 'capture_stale'
        # Single-person first version: no weak candidate can nominate itself
        # or dismiss a rival to obtain this measurement permission.
        if len(observations) != 1:
            return None, 'multiple_observations'
        item = observations[0]
        m, a = item['sample_metadata'], item['assignment']
        reason = str(a.get('bbox_quality_reason', '')).replace('detector_crop:', '')
        if (a.get('reason') != 'mapped_weak_observed'
                or not reason or any(r.strip() != 'edge_touch>2' for r in reason.split(','))
                or m.get('search_reacquire_context_active')
                or a.get('reacquire_partial_state') == 'mismatch'):
            return None, 'not_edge_only'
        proof = m.get('identity_competition') or {}
        if (item.get('frame_index') is None
                or m.get('candidate_count') != 1 or proof.get('candidate_count') != 1
                or proof.get('uid') != uid or proof.get('passed') is not True
                or proof.get('frame_index') != item.get('frame_index')
                or proof.get('source_detection_index') != m.get('source_detection_index')):
            return None, 'competition_unqualified'
        recent = a.get('template_recent_evidence') or {}
        distance = float(recent.get('distance', float('inf')))
        if (recent.get('count', 0) <= 0 or not 0 <= distance <= .20
                or not .80 <= float(m.get('detector_confidence', 0)) <= 1.):
            return None, 'appearance_unqualified'
        ref = bank.identities[uid].last_strong_observation
        if (not ref or ref.get('geometry_source') != 'detector'
                or ref.get('track_id') != raw_track_id
                or not 0 < capture_timestamp - float(ref['capture_timestamp']) <= .75
                or not capture_id > int(ref['capture_frame_id'])):
            return None, 'identity_reference_stale'
        yaw_delta = float(m['integrated_yaw_deg']) - float(ref['integrated_yaw_deg'])
        if not math.isfinite(yaw_delta) or abs(yaw_delta) > 20:
            return None, 'turn_uncertain'
        observation = resolve_braking_target_observation(
            target_id=uid, display_bbox=display_bbox, capture_frame_id=capture_id,
            capture_timestamp=capture_timestamp, expected_raw_track_id=raw_track_id,
            observations=observations, width=width, height=height)
        if observation is None:
            return None, 'provenance_rejected'
        x1, y1, x2, y2 = observation.bbox
        ox1, oy1, ox2, oy2 = map(float, ref['bbox'])
        area, old_area = (x2-x1)*(y2-y1), (ox2-ox1)*(oy2-oy1)
        intersection = max(0., min(x2, ox2)-max(x1, ox1))*max(0., min(y2, oy2)-max(y1, oy1))
        if (not all(math.isfinite(v) for v in (old_area, ox1, oy1, ox2, oy2))
                or old_area <= 0 or not .15 <= area/(width*height) <= .75
                or (x2-x1)/width < .20 or (y2-y1)/height < .65
                or min(area, old_area)/max(area, old_area) < .60
                or intersection/(area+old_area-intersection) < .40):
            return None, 'body_geometry_rejected'
        return replace(observation, source='yolo_cropped_observation'), 'accepted'
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None, 'missing_or_invalid_evidence'
