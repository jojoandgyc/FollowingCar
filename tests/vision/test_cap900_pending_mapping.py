"""Recorded CAP881/883/898/900 geometry, synthetic logged-distance vectors.

The existing confirmed-source fixture seeds identity before the recorded
missing-region interval. This is policy replay, not model or robot replay.
"""

from copy import deepcopy

import pytest

from test_cap828_verified_continuation import checkpoint, send as source_send
from test_cap225_pose_continuity import gallery


ROWS = {
    883: (361, 21759.666047589, (1.4616575, 139.7039185, 83.0461502, 402.9717102),
          129.673787970858, .2243629694, .34),
    896: (369, 21760.363655168, (.7365265, 134.8483276, 97.0161133, 411.0161743),
          129.32562427586348, .1582216620, .34),
    898: (370, 21760.462278003, (13.2001114, 135.3519592, 109.4488983, 408.9692078),
          128.9352152426219, .1727628708, .3421023488),
    900: (371, 21760.566386658, (31.9793930, 134.8690491, 124.5390396, 408.4611816),
          128.61387360440588, .2055376172, .3646860123),
}
PROTECTED = dict(frame_index=241, track_id=1, capture_frame_id=713,
    bbox=[504.8725281, 5.8448639, 639.2619019, 459.5390015],
    center_x_ratio=.8938550234, area=.1984754917, area_units="ratio",
    geometry_source="detector", capture_timestamp=21750.815550529,
    edge_touch_count=2., integrated_yaw_deg=2.3797468813, yaw_rate_dps=0.)


def send(bank, cap, **changes):
    frame, stamp, box, yaw, full, partial = ROWS[cap]
    kw = dict(capture_frame_id=cap, frame_index=frame, control_frame_id=frame,
              capture_timestamp=stamp, detector_bbox=list(box), integrated_yaw_deg=yaw,
              search_reacquire_context_active=True, search_direction="left",
              search_direction_compatible=True)
    kw.update(changes)
    return source_send(bank, 881, full=full, part=partial, **kw)


def before_seed():
    bank = checkpoint("strong", at=879)
    bank._reacquire_search_anchors[1] = deepcopy(PROTECTED)
    assert source_send(bank, 881, .218, .355, search_reacquire_context_active=True) == 1
    # Loss of comparable region is unknown, not a measured identity conflict.
    for cap in (883, 896):
        assert send(bank, cap) == 0
        assert bank.last_assignments[3]["reason"] == "secondary_evidence_unavailable"
    assert bank.track_to_uid[3] == 1
    assert 1 not in bank._appearance_verified
    return bank


def seeded():
    bank = before_seed()
    assert send(bank, 898) == 0
    assert bank.last_assignments[3]["reason"] == "preferred_search_mapped_late_wait"
    assert bank.pending_late_handoffs[3].streak == 1
    assert 3 not in bank.track_to_uid
    return bank


def test_waiting_mapping_becomes_pending_and_next_soft_sample_finishes_same_chain():
    bank = before_seed()
    frozen = gallery(bank)
    strong_anchor = deepcopy(bank.identities[1].last_strong_observation)
    isolated = deepcopy(bank._reacquire_quarantine._held[1])
    assert send(bank, 898) == 0
    assert 3 not in bank.track_to_uid
    assert bank.pending_late_handoffs[3].streak == 1
    assert bank._reacquire_search_anchors[1] == PROTECTED
    assert bank.identities[1].last_strong_observation == strong_anchor
    assert bank._reacquire_quarantine._held[1].armed_timestamp == isolated.armed_timestamp
    assert gallery(bank) == frozen
    assert send(bank, 900) == 1
    result = bank.last_assignments[3]
    assert result["reason"] == "preferred_search_soft_reacquire"
    assert result["late_candidate_streak"] == 2
    assert bank.track_to_uid[3] == 1 and 3 not in bank.pending_late_handoffs
    assert bank._reacquire_search_anchors[1] == PROTECTED
    assert bank._reacquire_quarantine.is_held(1)
    assert bank._reacquire_quarantine._held[1].armed_timestamp == ROWS[900][1]
    assert not result["bank_updated"] and gallery(bank) == frozen


@pytest.mark.parametrize("kind", ["opposite", "competition"])
def test_pending_demotion_does_not_widen_direction_or_identity_competition(kind):
    bank = seeded()
    changes = {}
    if kind == "opposite":
        changes["search_direction_compatible"] = False
    else:
        changes["identity_competition"] = dict(uid=1, frame_index=ROWS[900][0],
            source_detection_index=0, candidate_count=1, passed=False)
    assert send(bank, 900, **changes) == 0
    assert 3 not in bank.track_to_uid
    assert not bank.last_assignments[3]["bank_updated"]
    assert bank._reacquire_search_anchors[1] == PROTECTED


def test_repeated_capture_cannot_finish_pending_confirmation():
    bank = seeded()
    for frame in (371, 372, 373):
        assert send(bank, 898, frame_index=frame, control_frame_id=frame) == 0
        assert 3 not in bank.track_to_uid
        assert not bank.last_assignments[3]["bank_updated"]
        pending = bank.pending_late_handoffs.get(3)
        assert pending is None or pending.streak <= 1
    assert bank._reacquire_search_anchors[1] == PROTECTED


@pytest.mark.parametrize("kind", ["contradiction", "suspect", "revoked"])
def test_waiting_candidate_cannot_erase_known_identity_rejection(kind):
    bank = before_seed()
    if kind == "contradiction":
        bank._mapped_geometry_conflicts[3] = dict(uid=1, search_contradiction=True,
            reference=deepcopy(PROTECTED), rejected_capture=883, rejected_frame=361)
    elif kind == "suspect":
        bank._reacquire_control_suspects[1] = dict(track_id=3, reason="partial_conflict",
            streak=0, capture=896, timestamp=ROWS[896][1])
    else:
        bank._geometry_revoked_uids[1] = 361
    assert send(bank, 898) == 0
    assert not bank.last_assignments[3]["bank_updated"]
    if kind != "revoked":
        assert bank.last_assignments[3]["reason"] != "preferred_search_mapped_late_wait"
    if kind == "contradiction":
        assert bank._mapped_geometry_conflicts[3]["rejected_capture"] == 883
    elif kind == "suspect":
        assert 1 in bank._reacquire_control_suspects
    else:
        # This artificial mapped+revoked combination is not manufactured by
        # the normal geometry rejection (which removes the binding). Even if
        # it reaches the observation-only exit, demotion must not erase it.
        assert 1 in bank._geometry_revoked_uids
    assert bank._reacquire_search_anchors[1] == PROTECTED
    # A waiting frame alone is not enough: the following soft candidate must
    # not bind and silently remove negative evidence via late confirmation.
    assert send(bank, 900) == 0
    assert not bank.last_assignments[3]["bank_updated"]
    if kind == "contradiction":
        assert bank._mapped_geometry_conflicts[3]["rejected_capture"] == 883
    elif kind == "suspect":
        assert 1 in bank._reacquire_control_suspects
    else:
        assert 1 in bank._geometry_revoked_uids
        assert bank.last_assignments[3]["reacquire_geometry"]["late_candidate_rejection"] == (
            "revoked_owner_requires_trusted_geometry")


@pytest.mark.parametrize("distance", [.01, .25])
def test_real_geometry_revocation_cannot_age_into_strong_or_soft_late_reacquisition(distance):
    from test_mapped_identity_geometry import bank_with_anchor, assign, TRUE_341
    bank = bank_with_anchor()
    assert assign(bank) == 0  # Real geometry review revokes the wrong raw ID.
    revoked = deepcopy(bank._geometry_revoked_uids)
    negative = deepcopy(bank._mapped_geometry_conflicts)
    anchor = deepcopy(bank.identities[1].last_strong_observation)
    for frame in (200, 201):
        assert assign(bank, track=7, frame=frame, bbox=TRUE_341, distance=distance,
            preferred_uid=1, preferred_candidate_ok=True,
            metadata_extra=dict(search_reacquire_context_active=True,
                                search_direction_compatible=True)) == 0
        result = bank.last_assignments[7]
        assert result["reacquire_geometry"]["late_candidate_rejection"] == (
            "revoked_owner_requires_trusted_geometry")
        assert result["reacquire_geometry"]["old_anchor_ignored"] is False
        assert not result["bank_updated"]
        assert bank._geometry_revoked_uids == revoked
        assert bank._mapped_geometry_conflicts == negative
        assert bank.identities[1].last_strong_observation == anchor
        assert 7 not in bank.track_to_uid and not bank.pending_late_handoffs


def test_real_revocation_still_allows_independent_strong_geometry_owner_recovery():
    from test_mapped_identity_geometry import bank_with_anchor, assign, TRUE_341, TRUE_346
    bank = bank_with_anchor()
    assert assign(bank) == 0
    rejected = deepcopy(bank._mapped_geometry_conflicts[1])
    assert 1 in bank._geometry_revoked_uids
    assert assign(bank, track=3, frame=116, bbox=TRUE_341, distance=.05) == 0
    assert assign(bank, track=3, frame=117, bbox=TRUE_346, distance=.0475) == 1
    assert not bank._geometry_revoked_uids
    assert bank._mapped_geometry_conflicts[1] == rejected
    assert bank.track_to_uid == {3: 1}
    assert bank._reacquire_quarantine.is_held(1)


def test_correct_owner_return_uses_old_trusted_geometry_and_two_new_observations():
    from test_cap874_identity_reacquire import bank_at_857, assign, ROWS as WRONG_ROWS, ANCHOR
    bank = bank_at_857()
    assert assign(bank, WRONG_ROWS[0]) == 0
    negative = deepcopy(bank._mapped_geometry_conflicts[5])
    old_reference = deepcopy(bank.identities[1].last_strong_observation)
    for frame, cap, stamp, expected in [(372, 877, 10389.82, 0), (373, 881, 10390.02, 1)]:
        assert assign(bank, (frame, cap, stamp, ANCHOR, -66.329599, .08),
                      track=6, opposite=False) == expected
        result = bank.last_assignments[6]
        proof = result["reacquire_geometry"]["revoked_owner_reference_proof"]
        assert proof["verified"] and proof["reference_cap"] == 857
        assert proof["full_distance"] == pytest.approx(.08)
        assert proof["yaw_compensated_center_jump_ratio"] == pytest.approx(0.)
        assert result["late_candidate_streak"] == (1 if expected == 0 else 2)
        assert not result["bank_updated"]
        assert bank._mapped_geometry_conflicts[5] == negative
        if not expected:
            assert bank.identities[1].last_strong_observation == old_reference
            assert 1 in bank._geometry_revoked_uids
    assert bank.track_to_uid[6] == 1 and 5 not in bank.track_to_uid
    assert bank._reacquire_quarantine.is_held(1)


@pytest.mark.parametrize("kind", ["full_point_21", "soft", "long_time", "large_turn",
    "same_wrong_track", "inherited_wrong_track", "missing_full", "fake_metadata_full",
    "stale", "weak", "competition", "position", "scale", "missing_yaw", "strict_full_config"])
def test_revoked_return_requires_independent_bounded_trusted_reference_evidence(kind):
    from dataclasses import replace
    from test_cap874_identity_reacquire import bank_at_857, assign, ROWS as WRONG_ROWS, ANCHOR, metadata, feature
    bank = bank_at_857()
    assert assign(bank, WRONG_ROWS[0]) == 0
    negative = deepcopy(bank._mapped_geometry_conflicts)
    reference = deepcopy(bank.identities[1].last_strong_observation)
    m = metadata(881, 10390.02, ANCHOR, -66.329599, search=True, opposite=False)
    m.update(track_id=6, source_detection_index=0)
    source = "strong"
    body = feature(.08)
    track = 6
    if kind == "full_point_21": body = feature(.21)
    elif kind == "soft": source = "soft_strong"
    elif kind == "long_time": m["capture_timestamp"] = reference["capture_timestamp"] + 2.001
    elif kind == "large_turn": m["integrated_yaw_deg"] += 20.01
    elif kind == "same_wrong_track": track = 5
    elif kind == "inherited_wrong_track":
        wrong = WRONG_ROWS[0]
        m = metadata(877, wrong[2]+.10, wrong[3], wrong[4], search=True, opposite=False)
        m.update(track_id=6, source_detection_index=0)
    elif kind == "missing_full": body = None
    elif kind == "fake_metadata_full":
        body = None
        m.update(authorization_distance_uid=1, authorization_full_distance_floor=.01)
    elif kind == "stale": m["is_fresh"] = False
    elif kind == "weak": m["bbox_quality_tier"] = "weak"
    elif kind == "competition":
        m["identity_competition"] = dict(uid=1, frame_index=373, candidate_count=1,
                                       source_detection_index=0, passed=False)
    elif kind == "position":
        m["detector_bbox"] = [150., ANCHOR[1], 150.+ANCHOR[2], ANCHOR[3]]
        m["detector_center_x_ratio"] = (300.+ANCHOR[2])/1280
    elif kind == "scale": m["detector_area_ratio"] *= .60
    elif kind == "missing_yaw": m.pop("integrated_yaw_deg")
    elif kind == "strict_full_config":
        bank.config = replace(bank.config, preferred_search_reacquire_threshold=.05)
    geometry = bank._handoff_geometry(1, m, 373)
    result = bank._observe_late_search_candidate(
        track_id=track, candidate_uid=1, distance=.01, full_feature=body,
        partial_feature=feature(.01), match_source=source, candidate_count=1,
        bbox_quality_ok=True, sample_metadata=m, frame_index=373,
        geometry=geometry, distance_limit=.30)
    assert result is not None and result[0] == 0 and result[1] == 0
    assert result[2]["late_candidate_rejection"] == "revoked_owner_requires_trusted_geometry"
    assert result[2]["old_anchor_ignored"] is False
    assert not bank.pending_late_handoffs
    assert bank._geometry_revoked_uids
    assert bank._mapped_geometry_conflicts == negative
    assert bank.identities[1].last_strong_observation == reference
