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
    entry_limit=.50, retain_limit=.55, max_gap=.50,
) -> SimilarFollowDecision:
    """Observe once, then follow using two fresh locally continuous captures.

    ``partial_conflict`` means a caller-established reliable hard contradiction,
    not missing/incomparable torso evidence or a slight threshold fluctuation.
    Unknown torso evidence alone cannot reject this opt-in following policy.

    Entry must be on the compatible side. A continuously observed candidate can
    then cross to the other side without becoming a new reverse-side entrant.
    Current full-gallery support is still required on every accepted capture;
    neither old scores nor candidate features may become their own reference.

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
        if not _valid_state(state, identity, track):
            return reject('invalid_state')
        if capture <= state['last_cap'] or timestamp <= state['last_timestamp']:
            return SimilarFollowDecision('ignore', 'nonnew_capture', deepcopy(state))

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
    if distance > (retention if active else entry):
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
        area = _number(geometry.get('area_similarity'))
        if (geometry.get('ok') is not True or jump is None
                or not 0 <= jump <= .20 or area is None or not .55 <= area <= 1.):
            return reject('local_geometry_conflict')
        count = min(2, state['count'] + 1)
        origin_cap, origin_timestamp = state['origin_cap'], state['origin_timestamp']

    next_state = dict(uid=identity, track_id=track, count=count, active=count == 2,
                      entry_direction_compatible=True,
                      origin_cap=origin_cap, origin_timestamp=origin_timestamp,
                      last_cap=capture, last_timestamp=timestamp,
                      full_distance=distance, observation=deepcopy(current))
    return SimilarFollowDecision(
        'follow' if count == 2 else 'observe',
        'similar_candidate_continuous' if count == 2 else 'similar_candidate_observed',
        next_state,
    )
