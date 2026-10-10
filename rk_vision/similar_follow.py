"""Fresh, untrusted candidate continuity, separate from template learning.

This opt-in policy can follow a moderately similar person. It does not prove
identity, approve gallery writes, or change any motor authorization deadline.
The owner supplies current quality/competition decisions and local geometry
against ``state['observation']``, not against a protected historical anchor.
"""

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real
from typing import Optional

from .camera_geometry import yaw_image_shift_ratio


@dataclass(frozen=True)
class SimilarFollowDecision:
    status: str
    reason: str
    state: Optional[dict]
    learning_allowed: bool = False


def _number(value):
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


def _valid_state(state, uid, track):
    if not isinstance(state, dict) or not isinstance(state.get('observation'), dict):
        return False
    observation = state['observation']
    cap = _identifier(state.get('last_cap'))
    timestamp = _number(state.get('last_timestamp'))
    origin_cap = _identifier(state.get('origin_cap'))
    origin_timestamp = _number(state.get('origin_timestamp'))
    count = _number(state.get('count'))
    if state.get('position_only'):
        qualified_cap = _identifier(state.get('last_qualified_cap'))
        qualified_stamp = _number(state.get('last_qualified_timestamp'))
        if (qualified_cap is None or qualified_stamp is None or cap is None or timestamp is None
                or qualified_cap > cap or qualified_stamp > timestamp
                or not isinstance(state.get('position_previous_observation'), dict)):
            return False
    return bool(
        _identifier(state.get('uid')) == uid
        and _identifier(state.get('track_id')) == track
        and state.get('entry_direction_compatible') is True
        and state.get('active') in (True, False)
        and type(state.get('active')) is bool
        and count in (1., 2.)
        and state['active'] == (count == 2)
        and cap is not None and origin_cap is not None and origin_cap <= cap
        and timestamp is not None and origin_timestamp is not None
        and 0 <= origin_timestamp <= timestamp
        and _identifier(observation.get('capture_frame_id')) == cap
        and _number(observation.get('capture_timestamp')) == timestamp
        and _identifier(observation.get('track_id')) == track
    )


def evaluate_similar_follow(
    *, uid, track_id, current, geometry, competition_ok, blocked,
    full_distance, direction_compatible, partial_conflict=False, state=None,
    entry_limit=.50, retain_limit=.55, max_gap=.50, handoff_from_track_id=None,
    handoff_gallery_distance=None,
) -> SimilarFollowDecision:
    """Observe once, then follow using two fresh locally continuous captures.

    ``partial_conflict`` means a caller-established reliable hard contradiction,
    not missing/incomparable torso evidence or a slight threshold fluctuation.
    Unknown torso evidence alone cannot reject this opt-in following policy.

    Entry must be on the compatible side. A continuously observed candidate can
    then cross to the other side without becoming a new reverse-side entrant.
    Current full-gallery support is still required on every accepted capture;
    neither old scores nor candidate features may become their own reference.
    ``handoff_from_track_id`` accepts only a caller-vetted recent unique owner;
    the new raw track must pass entry appearance and current local geometry.
    A current independent gallery distance <= .30 allows up to .25 local
    displacement on this handoff only; ordinary continuation stays at .20.

    A duplicate or out-of-order capture returns ``ignore`` with an independent
    copy of the previous state, never ``follow``. All other failures return
    ``reject`` with no state. The caller must honor these state transitions and
    must not use ``observe`` or ``ignore`` as fresh positive identity evidence.
    """
    def reject(reason):
        return SimilarFollowDecision('reject', reason, None)

    identity, track = _identifier(uid), _identifier(track_id)
    if identity is None or track is None:
        return reject('invalid_binding')
    if not isinstance(current, dict):
        return reject('missing_observation')
    capture = _identifier(current.get('capture_frame_id'))
    timestamp = _number(current.get('capture_timestamp'))
    if capture is None or timestamp is None or timestamp < 0:
        return reject('invalid_capture')
    if _identifier(current.get('track_id')) != track:
        return reject('track_changed')
    if state is not None:
        previous_track = track if handoff_from_track_id is None else _identifier(handoff_from_track_id)
        if (previous_track is None or not _valid_state(state, identity, previous_track)
                or (previous_track != track and not state['active'])):
            return reject('invalid_state')
        if capture <= state['last_cap'] or timestamp <= state['last_timestamp']:
            return SimilarFollowDecision('ignore', 'nonnew_capture', deepcopy(state))
        if state.get('position_only') and timestamp - state['last_qualified_timestamp'] > .75:
            return reject('qualified_observation_gap')

    limits = tuple(_number(value) for value in
                   (full_distance, entry_limit, retain_limit, max_gap))
    if any(value is None for value in limits):
        return reject('invalid_limits_or_score')
    distance, entry, retention, gap = limits
    if distance < 0 or entry < 0 or retention < entry or gap <= 0:
        return reject('invalid_limits_or_score')
    # This policy is deliberately looser than trusted matching, not unlimited.
    entry, retention, gap = min(.50, entry), min(.55, retention), min(.50, gap)
    if blocked is not False:
        return reject('identity_blocked')
    if competition_ok is not True:
        return reject('competition_unverified')
    if partial_conflict is not False:
        return reject('reliable_partial_conflict')
    if (current.get('is_fresh') is not True
            or current.get('quality_bbox_ok') is not True
            or current.get('bbox_quality_tier', current.get('quality_tier')) != 'strong'
            or current.get('search_observation_only')
            or current.get('observation_only')
            or current.get('preferred_search_low_confidence')):
        return reject('observation_unverified')
    active = state is not None and state['active']
    if distance > (retention if active and handoff_from_track_id is None else entry):
        return reject('full_distance_conflict')

    if state is None:
        if direction_compatible is not True:
            return reject('entry_direction_unverified')
        count = 1
        origin_cap, origin_timestamp = capture, timestamp
    else:
        if timestamp - state['last_timestamp'] > gap:
            return reject('observation_gap')
        if not isinstance(geometry, dict):
            return reject('missing_local_geometry')
        jump = _number(geometry.get('yaw_compensated_center_jump_ratio'))
        area = _number(geometry.get('crop_continuity_area_similarity', geometry.get('area_similarity')))
        independent = _number(handoff_gallery_distance)
        jump_limit = (.25 if handoff_from_track_id is not None
                      and independent is not None and 0 <= independent <= .30 else .20)
        if (geometry.get('ok') is not True or jump is None
                or not 0 <= jump <= jump_limit or area is None or not .55 <= area <= 1.):
            return reject('local_geometry_conflict')
        count = min(2, state['count'] + 1)
        origin_cap, origin_timestamp = state['origin_cap'], state['origin_timestamp']

    next_state = dict(uid=identity, track_id=track, count=count, active=count == 2,
                      entry_direction_compatible=True,
                      origin_cap=origin_cap, origin_timestamp=origin_timestamp,
                      last_cap=capture, last_timestamp=timestamp,
                      last_qualified_cap=capture, last_qualified_timestamp=timestamp,
                      full_distance=distance, observation=deepcopy(current))
    return SimilarFollowDecision(
        'follow' if count == 2 else 'observe',
        'similar_candidate_continuous' if count == 2 else 'similar_candidate_observed',
        next_state,
    )


def observe_low_score_position(*, uid, track_id, current, state, geometry,
                               competition_ok, blocked, full_distance):
    """Retain an actual matched box, never another identity confirmation.

    Low detections cannot start this state, increment its confirmation count,
    roll the qualified-observation deadline, or authorize following. Association
    provenance must point to this candidate's last qualified physical capture.
    """
    if not _valid_state(state, _identifier(uid), _identifier(track_id)):
        return None
    cap, stamp = _identifier(current.get('capture_frame_id')), _number(current.get('capture_timestamp'))
    full, confidence = _number(full_distance), _number(current.get('detector_confidence'))
    confidence_limit = _number(current.get('association_confidence_limit', .50))
    qualified_cap = state.get('last_qualified_cap', state['last_cap'])
    qualified_stamp = state.get('last_qualified_timestamp', state['last_timestamp'])
    jump = _number((geometry or {}).get('yaw_compensated_center_jump_ratio'))
    area = _number((geometry or {}).get('area_similarity'))
    if (cap is None or stamp is None or cap <= state['last_cap'] or stamp <= state['last_timestamp']
            or current.get('track_id') != track_id or current.get('is_fresh') is not True
            or current.get('low_score_continuation') is not True
            or current.get('association_reason') != 'low_score_existing_track'
            or current.get('association_previous_capture_frame_id') != qualified_cap
            or current.get('association_previous_capture_timestamp') != qualified_stamp
            or not 0 < stamp-qualified_stamp <= .5 or stamp-state['last_timestamp'] > .5
            or confidence_limit is None or not .25 < confidence_limit <= 1.
            or confidence is None or not .25 <= confidence < confidence_limit
            or full is None or not 0 <= full <= .50
            or competition_ok is not True or blocked is not False
            or not geometry or geometry.get('ok') is not True
            or jump is None or not 0 <= jump <= .20
            or area is None or not .55 <= area <= 1.):
        return None
    result = deepcopy(state)
    result.update(position_only=True,
                  position_previous_observation=deepcopy(state['observation']),
                  last_qualified_cap=qualified_cap, last_qualified_timestamp=qualified_stamp,
                  last_cap=cap, last_timestamp=stamp, observation=deepcopy(current))
    return result


def candidate_motion_geometry(current, state, geometry, *, camera_hfov_deg):
    """Check a real new strong box against two real associated observations.

    Used only after a bounded low-score observation, with known camera yaw.
    The result is follow-only evidence, never a replacement trusted anchor.
    No extrapolated box or timestamp is ever committed as an observation.
    """
    if (not isinstance(state, dict) or not state.get('position_only') or not geometry
            or not _valid_state(state, _identifier(state.get('uid')), _identifier(state.get('track_id')))):
        return None
    first, second = state.get('position_previous_observation'), state.get('observation')
    if not isinstance(first, dict) or not isinstance(second, dict):
        return None
    rows = []
    for observation in (first, second, current):
        row = [_number(observation.get(key)) for key in
               ('capture_timestamp', 'detector_center_x_ratio', 'integrated_yaw_deg', 'detector_area_ratio')]
        if any(value is None for value in row) or not 0 <= row[1] <= 1 or row[3] <= 0:
            return None
        if observation.get('is_fresh') is not True or observation.get('track_id') != state['track_id']:
            return None
        rows.append(row)
    (t0, x0, y0, a0), (t1, x1, y1, a1), (t2, x2, y2, a2) = rows
    if (not .05 <= t1-t0 <= .5 or not 0 < t2-t1 <= .35
            or t2-state['last_qualified_timestamp'] > .75
            or second.get('low_score_continuation') is not True
            or current.get('quality_bbox_ok') is not True
            or current.get('bbox_quality_tier') != 'strong'
            or current.get('low_score_continuation')
            or min(a0, a1, a2)/max(a0, a1, a2) < .55):
        return None
    v = (x1-x0-yaw_image_shift_ratio(y0, y1, camera_hfov_deg))/(t1-t0)
    predicted = x1 + v*(t2-t1) + yaw_image_shift_ratio(y1, y2, camera_hfov_deg)
    residual = abs(x2-predicted)
    # Reuse the normal local .20 residual test; only the reference position
    # changes. The short horizon and measured speed bound prevent arbitrary
    # extrapolation, without introducing another tighter threshold to chatter.
    if not math.isfinite(v) or abs(v) > 1.0 or not math.isfinite(residual) or residual > .20:
        return None
    return dict(geometry, ok=True, reason='candidate_motion_continuous',
                candidate_motion_prediction=dict(reference_caps=[first['capture_frame_id'], second['capture_frame_id']],
                    predicted_center_x_ratio=predicted, residual=residual,
                    measured_center_jump=geometry.get('yaw_compensated_center_jump_ratio')),
                yaw_compensated_center_jump_ratio=residual)


def formal_detection_continuous(current, state, geometry, *, confidence, gallery_distance):
    """Recheck a formal .50+ detector result against an accepted candidate.

    Detector admission (.50) and template quality (.60) are different rights.
    This cannot initialize/rebind a candidate or use its own follow references
    as appearance proof. Both the last qualified capture and this new capture
    need independent gallery support. Low-score position observations keep the
    original qualified timestamp and cannot extend its bounded .75 s bridge.
    The caller still checks minimum area, competition and hard contradictions.
    """
    track = _identifier(current.get('track_id'))
    if (not _valid_state(state, _identifier((state or {}).get('uid')), track)
            or not state['active'] or current.get('is_fresh') is not True
            or current.get('quality_bbox_ok') is not True
            or current.get('bbox_quality_tier') != 'strong'
            or any(current.get(key) for key in ('low_score_continuation',
                'search_observation_only', 'observation_only', 'preferred_search_low_confidence'))
            or current.get('candidate_count') != 1):
        return False
    cap, stamp = _identifier(current.get('capture_frame_id')), _number(current.get('capture_timestamp'))
    score, detector_score = _number(confidence), _number(current.get('detector_confidence'))
    previous_gallery, current_gallery = _number(state.get('gallery_distance')), _number(gallery_distance)
    qualified_stamp = _number(state.get('last_qualified_timestamp', state.get('last_timestamp')))
    jump = _number((geometry or {}).get('yaw_compensated_center_jump_ratio'))
    area = _number((geometry or {}).get('area_similarity'))
    return bool(cap is not None and cap > state['last_cap'] and stamp is not None
        and 0 < stamp-state['last_timestamp'] <= .50
        and qualified_stamp is not None and 0 < stamp-qualified_stamp <= .75
        and score is not None and .50 <= score <= 1.
        and detector_score is not None and .50 <= detector_score <= 1.
        and previous_gallery is not None and 0 <= previous_gallery <= .30
        and current_gallery is not None and 0 <= current_gallery <= .30
        and geometry and geometry.get('ok') is True
        and jump is not None and 0 <= jump <= .20
        and area is not None and .55 <= area <= 1.)


def bounded_crop_follow_geometry(*, uid, track_id, current, state, geometry,
                                 gallery_distance, local_full_distance,
                                 local_partial_distance, competition_ok, blocked,
                                 partial_conflict):
    return _bounded_crop_geometry(uid=uid, track_id=track_id, current=current,
        state=state, geometry=geometry, gallery_distance=gallery_distance,
        local_full_distance=local_full_distance, local_partial_distance=local_partial_distance,
        competition_ok=competition_ok, blocked=blocked, partial_conflict=partial_conflict)


def bounded_crop_observation_retainable(**kwargs):
    """Keep the old candidate only; never grant UID or advance its clocks."""
    return _bounded_crop_geometry(**kwargs, observation_only=True) is not None


def _bounded_crop_geometry(*, uid, track_id, current, state, geometry,
                          gallery_distance, local_full_distance,
                          local_partial_distance, competition_ok, blocked,
                          partial_conflict, observation_only=False):
    """Finish a short same-raw crop episode using a fixed independent anchor.

    The bank supplies two current-to-anchor descriptor comparisons. The anchor
    was admitted by strong quality or the existing independent crop recheck,
    not by this permission. Current gallery support remains mandatory. Neither
    these weaker crops nor their descriptors can roll the anchor's .5 s age,
    become learning parents, or establish a new raw/UID binding.
    """
    uid, track = _identifier(uid), _identifier(track_id)
    anchor = (state or {}).get('crop_appearance_anchor')
    if (uid is None or track is None or not isinstance(current, dict)
            or not _valid_state(state, uid, track) or not state['active']
            or state.get('position_only') or not isinstance(anchor, dict)
            or anchor.get('uid') != uid or anchor.get('track_id') != track
            or anchor.get('source') not in ('strong_gallery', 'independent_crop')
            or current.get('track_id') != track or current.get('is_fresh') is not True
            or current.get('quality_bbox_ok') is not False
            or current.get('bbox_quality_tier') != 'weak'
            or competition_ok is not True or blocked is not False or partial_conflict is not False
            or any(current.get(key) for key in ('low_score_continuation', 'observation_only',
                'search_observation_only', 'preferred_search_low_confidence'))):
        return None
    cap, stamp = _identifier(current.get('capture_frame_id')), _number(current.get('capture_timestamp'))
    anchor_cap, anchor_stamp = _identifier(anchor.get('capture')), _number(anchor.get('timestamp'))
    distances = tuple(_number(value) for value in (gallery_distance,
        anchor.get('gallery_distance'), local_full_distance, local_partial_distance))
    score = _number(current.get('detector_confidence'))
    jump = _number((geometry or {}).get('yaw_compensated_center_jump_ratio'))
    area = _number((geometry or {}).get('area_similarity'))
    reference = anchor.get('observation')
    if (cap is None or anchor_cap is None or not anchor_cap <= state['last_cap'] < cap
            or stamp is None or anchor_stamp is None or not 0 < stamp-anchor_stamp <= .5
            or not 0 < stamp-state['last_timestamp'] <= .35
            or score is None or not .75 <= score <= 1.
            or any(value is None for value in distances)
            or not 0 <= distances[0] <= .35 or not 0 <= distances[1] <= .30
            or not 0 <= distances[2] <= .25 or not 0 <= distances[3] <= .25
            or not isinstance(reference, dict)
            or reference.get('track_id') != track or reference.get('capture_frame_id') != anchor_cap
            or reference.get('capture_timestamp') != anchor_stamp
            or not current.get('partial_feature_source')
            or current.get('partial_feature_source') != reference.get('partial_feature_source')
            or not geometry or geometry.get('ok') is not True
            or jump is None or not 0 <= jump <= .12
            or area is None or not .65 <= area <= 1.):
        return None
    allowed = {'aspect<0.18', 'edge_touch>2'}
    reasons = [str(current.get(key) or '').removeprefix('detector_crop:')
               for key in ('quality_bbox_reason', 'bbox_quality_reason')]
    if not any(reasons) or any(set(reason.split(','))-allowed for reason in reasons if reason):
        return None
    width, height = (_number(current.get(key)) for key in ('image_width', 'image_height'))
    if width is None or height is None or min(width, height) <= 0:
        return None
    boxes, sides = [], []
    for row in (reference, state['observation'], current):
        box = row.get('detector_bbox')
        if (row.get('image_width') != width or row.get('image_height') != height
                or not isinstance(box, (tuple, list)) or len(box) != 4):
            return None
        values = [_number(value) for value in box]
        if any(value is None for value in values):
            return None
        x1, y1, x2, y2 = values
        left, right = x1 <= .02*width, x2 >= .98*width
        if (not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height
                or left == right or x2-x1 < max(64., .10*width)
                or y2-y1 < .70*height or (x2-x1)*(y2-y1)/(width*height) < .08):
            return None
        boxes.append(values)
        sides.append(left)
    heights = [box[3]-box[1] for box in boxes]
    widths = [box[2]-box[0] for box in boxes]
    areas = [(box[2]-box[0])*(box[3]-box[1]) for box in boxes]
    height_similarity = min(heights)/max(heights)
    if len(set(sides)) != 1 or min(areas)/max(areas) < .60:
        return None
    # A tall same-side crop can gain/lose its upper or lower visible extent
    # without changing person or scale (CAP734 -> 738). Require the OTHER end
    # and the width/center to remain stable across the fixed anchor, previous
    # accepted box and current box. This is not a general height tolerance.
    tops, bottoms = [box[1] for box in boxes], [box[3] for box in boxes]
    centers = [(box[0]+box[2])/2 for box in boxes]
    end_changes = (max(tops)-min(tops), max(bottoms)-min(bottoms))
    stable_height = height_similarity >= .95
    edge_visibility_change = bool(not stable_height and height_similarity >= .85
        and min(widths)/max(widths) >= .80
        and max(centers)-min(centers) <= .05*width
        and min(end_changes) <= .03*height and max(end_changes) <= .12*height
        and distances[0] <= .30)
    # CAP298 -> 304: a clipped person's visible center barely moved, while
    # camera-yaw compensation contributed a .107 residual. Permit the small
    # .10 -> .12 band only with stronger independent AND paired appearance,
    # stable full-height same-edge boxes, and the existing fixed .5 s anchor.
    # This does not change the rolling independent crop-reverification gate:
    # this bounded permission cannot refresh its own anchor or learn.
    stable_edge_motion = bool(stable_height
        and min(widths)/max(widths) >= .90
        and max(centers)-min(centers) <= .03*width
        and all(box[1] <= .02*height and box[3] >= .98*height for box in boxes)
        and max(distances[:2]) <= .15 and max(distances[2:]) <= .10)
    if jump > .10 and not stable_edge_motion:
        return None
    local_limit = .16 if edge_visibility_change else .12
    if not (stable_height or edge_visibility_change):
        return None
    if observation_only:
        # Inconclusive appearance retains no *new* evidence. A bounded later
        # frame must still pass every normal gate against the untouched anchor.
        if distances[0] > .30:
            return None
    elif max(distances[2:]) > local_limit:
        return None
    return dict(geometry, bounded_crop_confirmation=dict(
        reference_cap=anchor_cap, reference_timestamp=anchor_stamp,
        expires_at=anchor_stamp+.5, reference_age_ms=1000*(stamp-anchor_stamp),
        full_distance=distances[2], partial_distance=distances[3],
        gallery_distance=distances[0], anchor_gallery_distance=distances[1],
        visibility_mode='single_end_change' if edge_visibility_change else 'stable_height',
        height_similarity=height_similarity, local_distance_limit=local_limit,
        compensated_jump_limit=.12 if jump > .10 else .10,
        fixed_anchor_motion_bridge=jump > .10,
        permission='observation_only' if observation_only else 'follow_only',
        learning_allowed=False, anchor_renewed=False))


def narrow_edge_follow_geometry(*, uid, track_id, current, state, geometry,
                               gallery_distance, competition_ok, blocked,
                               partial_conflict):
    """Retain a substantial same-person edge crop, without a quality upgrade.

    CAP936's 85 px, full-height crop narrowly crosses the aspect limit while
    keeping current independent appearance and the same physical trajectory.
    This permission is only a bounded continuation of an active raw track;
    it cannot enroll, hand off, learn, or renew the last strong-observation age.
    """
    uid, track_id = _identifier(uid), _identifier(track_id)
    if (uid is None or track_id is None or not isinstance(current, dict)
            or not _valid_state(state, uid, track_id) or not state['active']
            or state.get('position_only') or current.get('track_id') != track_id
            or current.get('is_fresh') is not True
            or current.get('quality_bbox_ok') is not False or current.get('bbox_quality_tier') != 'weak'
            or competition_ok is not True or blocked is not False or partial_conflict is not False
            or any(current.get(key) for key in ('low_score_continuation', 'observation_only',
                                                'search_observation_only', 'preferred_search_low_confidence'))):
        return None
    distance = _number(gallery_distance)
    cap, stamp = _identifier(current.get('capture_frame_id')), _number(current.get('capture_timestamp'))
    strong_stamp = _number(state.get('last_strong_timestamp'))
    jump = _number((geometry or {}).get('yaw_compensated_center_jump_ratio'))
    area = _number((geometry or {}).get('area_similarity'))
    if (distance is None or not 0 <= distance <= .30 or cap is None or cap <= state['last_cap']
            or stamp is None or not 0 < stamp-state['last_timestamp'] <= .35
            or strong_stamp is None or not 0 < stamp-strong_stamp <= .75
            or not geometry or geometry.get('ok') is not True
            or jump is None or not 0 <= jump <= .12
            or area is None or not .35 <= area <= 1.):
        return None
    previous = state['observation']
    allowed = {'aspect<0.18', 'edge_touch>2'}
    for row, permitted in ((current, (allowed,)), (previous, ({'edge_touch>2'}, allowed))):
        reasons = [str(row.get(key) or '').removeprefix('detector_crop:')
                   for key in ('quality_bbox_reason', 'bbox_quality_reason')]
        if not any(reasons):
            return None
        for reason in reasons:
            parts = set(reason.split(',')) if reason else set()
            if parts and parts not in permitted:
                return None
    width, height = (_number(current.get(key)) for key in ('image_width', 'image_height'))
    if (width is None or height is None or min(width, height) <= 0
            or _number(previous.get('image_width')) != width
            or _number(previous.get('image_height')) != height):
        return None
    boxes, sides = [], []
    for row in (previous, current):
        box = row.get('detector_bbox')
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            return None
        values = [_number(value) for value in box]
        if any(value is None for value in values):
            return None
        x1, y1, x2, y2 = values
        left, right = x1 <= .02*width, x2 >= .98*width
        if (not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height
                or left == right or y1 > .02*height or y2 < .98*height
                or x2-x1 < max(80., .12*width)
                or (x2-x1)*(y2-y1)/(width*height) < .12):
            return None
        boxes.append(values)
        sides.append(left)
    (px1, py1, px2, py2), (x1, y1, x2, y2) = boxes
    height_similarity = min(py2-py1, y2-y1)/max(py2-py1, y2-y1)
    if (sides[0] != sides[1] or height_similarity < .95 or x2-x1 > px2-px1
            or (sides[1] and x1+x2 > px1+px2)
            or (not sides[1] and x1+x2 < px1+px2)):
        return None
    return dict(geometry, crop_continuity_area_similarity=height_similarity,
                crop_visible_area_similarity=area,
                crop_visibility_reason='same_raw_narrow_edge_height_continuous')


def cropped_follow_continuous(current, state, geometry):
    """A recent accepted candidate may retain a substantial three-edge crop.

    This does not make the crop a trusted template. Only the exact edge-count
    failure is eligible, with current local geometry and at most .75 s since
    the last normally qualified capture (not a rolling motor/identity lease).
    """
    if not state or not state.get('active') or not isinstance(geometry, dict):
        return False
    reasons = [str(current.get(k) or '').removeprefix('detector_crop:')
               for k in ('quality_bbox_reason', 'bbox_quality_reason')]
    if not any(reasons) or any(r and r != 'edge_touch>2' for r in reasons):
        return False
    if geometry.get('ok') is not True:
        return False
    timestamp = _number(current.get('capture_timestamp'))
    origin = _number(state.get('last_strong_timestamp', state.get('last_timestamp')))
    width, height = (_number(current.get(k)) for k in ('image_width', 'image_height'))
    box = current.get('detector_bbox')
    if (timestamp is None or origin is None or not 0 < timestamp-origin <= .75
            or width is None or height is None or min(width, height) <= 0
            or not isinstance(box, (list, tuple)) or len(box) != 4):
        return False
    points = [_number(v) for v in box]
    if any(v is None for v in points):
        return False
    x1, y1, x2, y2 = points
    return bool(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
                and x2-x1 >= max(80., .12*width) and y2-y1 >= .5*height
                and (x2-x1)*(y2-y1)/(width*height) >= .12
                and not (x1 <= .02*width and x2 >= .98*width))


def crop_reverification_eligible(current, state, geometry):
    """A new, substantial same-side crop may be independently rechecked.

    This is not the .75 s historical crop bridge. Each admitted capture must
    carry new detector evidence, strict local geometry and (at the caller)
    current independent-gallery appearance and competition checks. Missing
    frames cannot roll this .35 s observation-gap limit. Neither a raw-ID
    handoff nor a position-only/low-score observation can enter this path.
    """
    uid, track = _identifier((state or {}).get('uid')), _identifier(current.get('track_id'))
    if (not _valid_state(state, uid, track) or not state['active']
            or state.get('position_only') or current.get('is_fresh') is not True
            or current.get('low_score_continuation') is True
            or any(current.get(key) for key in ('search_observation_only', 'observation_only',
                                                'preferred_search_low_confidence'))):
        return False
    cap, stamp = _identifier(current.get('capture_frame_id')), _number(current.get('capture_timestamp'))
    score = _number(current.get('detector_confidence'))
    if (cap is None or cap <= state['last_cap'] or stamp is None
            or not 0 < stamp-state['last_timestamp'] <= .35
            or score is None or not .75 <= score <= 1.):
        return False
    previous = state['observation']
    reasons = [str(row.get(key) or '').removeprefix('detector_crop:')
               for row in (previous, current)
               for key in ('quality_bbox_reason', 'bbox_quality_reason')]
    if (not any(reasons[:2]) or not any(reasons[2:])
            or any(reason and reason != 'edge_touch>2' for reason in reasons)):
        return False
    jump = _number((geometry or {}).get('yaw_compensated_center_jump_ratio'))
    area = _number((geometry or {}).get('area_similarity'))
    if (not geometry or geometry.get('ok') is not True or jump is None
            or not 0 <= jump <= .10 or area is None or not .70 <= area <= 1.):
        return False
    width, height = (_number(current.get(key)) for key in ('image_width', 'image_height'))
    if (width is None or height is None or min(width, height) <= 0
            or _number(previous.get('image_width')) != width
            or _number(previous.get('image_height')) != height):
        return False
    sides = []
    for row in (previous, current):
        box = row.get('detector_bbox')
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return False
        values = [_number(value) for value in box]
        if any(value is None for value in values):
            return False
        x1, y1, x2, y2 = values
        left, right = x1 <= .02*width, x2 >= .98*width
        if (not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height
                or left == right or x2-x1 < max(80., .12*width)
                or y2-y1 < .5*height or (x2-x1)*(y2-y1)/(width*height) < .12):
            return False
        sides.append(left)
    return sides[0] == sides[1]


def cropped_visibility_geometry(current, state, geometry):
    """Do not interpret same-edge visibility loss as a shrinking person.

    Both observations must be full-height same-side crops of the already
    accepted candidate. Existing position/absolute-size/crop-age checks still
    apply; only the visible-area comparison uses the unchanged vertical span.
    """
    if not cropped_follow_continuous(current, state, geometry):
        return geometry
    previous = state.get('observation') or {}
    boxes = [observation.get('detector_bbox') for observation in (previous, current)]
    width, height = _number(current.get('image_width')), _number(current.get('image_height'))
    if any(not isinstance(box, (list, tuple)) or len(box) != 4 for box in boxes):
        return geometry
    values = [[_number(v) for v in box] for box in boxes]
    if any(v is None for box in values for v in box):
        return geometry
    (px1, py1, px2, py2), (x1, y1, x2, y2) = values
    left = px1 <= .02*width and x1 <= .02*width
    right = px2 >= .98*width and x2 >= .98*width
    if (left == right or min(px2-px1, x2-x1, py2-py1, y2-y1) <= 0
            or max(py1, y1) > .02*height or min(py2, y2) < .98*height
            or x2-x1 > px2-px1):
        return geometry
    height_similarity = min(py2-py1, y2-y1)/max(py2-py1, y2-y1)
    if height_similarity < .90:
        return geometry
    return dict(geometry, crop_continuity_area_similarity=height_similarity,
                crop_visible_area_similarity=geometry.get('area_similarity'),
                crop_visibility_reason='same_edge_height_continuous')
