"""Recorded CAP521/524/548 geometry and distances, without model or hardware.

Synthetic normalized vectors preserve each recorded distance to approved
CAP82 appearance. This replays identity policy, not OSNet inference.
"""
from copy import deepcopy

import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig, _geometry_observation
from test_cap874_identity_reacquire import feature, metadata
from test_cap225_pose_continuity import gallery


REFERENCE = dict(frame_index=157, track_id=4, capture_frame_id=459,
    bbox=[173.5485534668, 5.2400817871, 366.835723877, 473.139892578],
    center_x_ratio=.4221752166748, area=.2943978856225, area_units="ratio",
    geometry_source="detector", capture_timestamp=33182.07022432,
    edge_touch_count=2., integrated_yaw_deg=-28.9963692038, yaw_rate_dps=8.136)
ROWS = {
    # frame, stamp, bbox, yaw, full, reliable torso, competition gap, index, count
    521: (185, 33185.292871215, (137.842407227, 0., 291.990814209, 477.974853516),
          -52.1074330165, .1553558707, .2417996526, None, 0, 1),
    524: (186, 33185.457560822, (144.374847412, .717910767, 286.601989746, 475.947021484),
          -52.5746050375, .1314557791, .2366564274, .1212624907, 0, 2),
    527: (187, 33185.593249612, (136.593307495, 2.080078125, 266.947509766, 442.708312988),
          -53.7499522953, .1590969563, .1848025322, .083717525, 0, 2),
    529: (188, 33185.723911077, (140.873809814, 2.729125977, 262.958557129, 454.928344727),
          -55.1642857225, .1864485741, .2555696368, .0689532757, 0, 2),
    532: (189, 33185.887850537, (148.896911621, 3.005111694, 273.009796143, 472.081787109),
          -57.4666028556, .1581321359, .2238174677, .1914197803, 1, 2),
    536: (190, 33186.087830878, (192.538772583, 3.205978394, 317.147918701, 468.263366699),
          -60.2297363504, .1552354097, .2397136688, .1363612413, 1, 2),
    548: (194, 33186.755936685, (357.606719971, 4.58026123, 481.177886963, 470.970458984),
          -71.8955734630, .1956998110, .2459832430, .0772311687, 1, 2),
}
WRONG = {
    529: (188, 33185.723911077, (0., 43.984405518, 53.76581955, 397.287719727),
          -55.1642857225, .2554018497, .3848467767, -.0689532757, 1, 2),
    562: (199, 33187.489653216, (372.076965332, 54.248458862, 467.565093994, 385.608489990),
          -84.4207920132, .3562717438, .3715451658, -.0821461678, 0, 2),
}


def sample(cap, *, track=9, row=None, **changes):
    frame, stamp, box, yaw, full, part, gap, index, count = row or ROWS[cap]
    m = dict(metadata(cap, stamp, box, yaw, search=True, opposite=cap >= 548),
             frame_index=frame, track_id=track, source_detection_index=index,
             image_width=640, image_height=480, partial_feature_source="osnet_torso",
             partial_observation=True, candidate_count=count, detector_edge_touch_count=2)
    m["identity_competition"] = dict(uid=1, frame_index=frame, candidate_count=count,
        source_detection_index=index, passed=gap is None or gap >= .05, distance_gap=gap)
    m.update(changes)
    return m, full, part


def revoked_bank():
    b = IdentityBank(IdentityBankConfig(template_memory_enable=True,
        template_crosscheck_enable=True, appearance_region_safety_enable=True,
        partial_appearance_enable=True, similar_follow_enable=True,
        template_learning_guard_enable=True, partial_match_threshold=.45,
        partial_confirm_threshold=.40, controlled_handoff_enable=True,
        preferred_search_reacquire_threshold=.20, camera_hfov_deg=60.))
    box = [376.964263916, 0., 565.315307617, 479.204711914]
    seed = dict(metadata(82, 33162.232653698, box, -1.5500082066, search=False),
        track_id=1, frame_index=20, image_width=640, image_height=480,
        partial_feature_source="osnet_torso", partial_observation=True)
    b._create_identity(feature(0), 20, seed, feature(0))
    b.identities[1].last_strong_observation = deepcopy(REFERENCE)
    b._reacquire_search_anchors[1] = deepcopy(REFERENCE)
    # Preserve the actual old-run revocation, even when its producer is fixed.
    crop = dict(metadata(488, 33183.592120901,
        [.6617040634, 2.6116638184, 63.379524231, 472.995178223], -29.141520627),
        track_id=7, detector_edge_touch_count=3)
    b._geometry_revoked_uids[1] = 157
    b._mapped_geometry_conflicts[7] = dict(uid=1, reference=deepcopy(REFERENCE),
        search_contradiction=False, rejected_frame=167, rejected_capture=488,
        origin_geometry_reason="center_jump,area_change",
        candidate=_geometry_observation(crop, 167))
    return b


def send(b, cap, *, row=None, track=9, **changes):
    m, full, part = sample(cap, row=row, track=track, **changes)
    box = m["detector_bbox"]
    return b.assign(track_id=track, feature=feature(full), partial_feature=feature(part),
        confidence=.88, area=(box[2]-box[0])*(box[3]-box[1]),
        frame_index=m["frame_index"], candidate_count=m["candidate_count"],
        bbox_quality_ok=True, bbox_quality_tier="strong", sample_metadata=m,
        preferred_uid=1, preferred_candidate_ok=m["search_direction_compatible"])


def test_cap521_524_recovers_from_old_geometry_loss_without_learning():
    b = revoked_bank()
    original_gallery = gallery(b)
    conflict = deepcopy(b._mapped_geometry_conflicts[7])
    assert send(b, 521) == 0
    g = b.last_assignments[9]["reacquire_geometry"]
    assert not g["revoked_owner_reference_proof"]["verified"]
    assert g["independent_recovery_evidence"]["verified"]
    assert "independent_recovery_confirmation" not in g
    assert b.last_assignments[9]["late_candidate_streak"] == 1
    assert send(b, 524) == 1
    a = b.last_assignments[9]
    proof = a["reacquire_geometry"]["independent_recovery_confirmation"]
    assert proof == dict(uid=1, raw_track_id=9, capture_frame_id=524,
        capture_timestamp=ROWS[524][1], frame_index=186, count=2,
        completed_confirmation=True, learning_allowed=False)
    assert a["reacquire_geometry"]["ok"]
    assert not a["bank_updated"] and not a["recent_bank_updated"]
    assert not a.get("learning_written_tiers")
    assert b._reacquire_quarantine.is_held(1)
    assert b._mapped_geometry_conflicts[7] == conflict
    assert b._reacquire_search_anchors[1] == REFERENCE
    assert gallery(b) == original_gallery


def test_cap548_starts_independent_observation_despite_expired_rotated_reference():
    b = revoked_bank()
    assert send(b, 548) == 0
    a = b.last_assignments[9]
    g = a["reacquire_geometry"]
    assert g["revoked_owner_reference_proof"]["capture_delta_sec"] > 4.
    assert g["revoked_owner_reference_proof"]["yaw_delta_deg"] > 40.
    assert g["independent_recovery_evidence"]["verified"]
    assert a["late_candidate_streak"] == 1
    assert "independent_recovery_confirmation" not in g
    assert 9 not in b.track_to_uid


def test_recovery_continues_through_recorded_crop_changes_without_reconfirmation():
    b = revoked_bank()
    original_gallery = gallery(b)
    assert send(b, 521) == 0
    assert send(b, 524) == 1
    assert b._similar_follow_states[(1, 9)]["active"]
    assert not b._follow_references
    for cap in (527, 529, 532, 536):
        assert send(b, cap) == 1
        a = b.last_assignments[9]
        assert a["reason"] == "mapped_similar_follow"
        assert a["similar_follow"]["completed_confirmation"]
        assert not a["bank_updated"] and not a["recent_bank_updated"]
        assert a["reacquire_geometry"]["current"]["capture_frame_id"] == cap
        assert b._reacquire_search_anchors[1] == REFERENCE
        assert b._reacquire_quarantine.is_held(1)
        assert not b._follow_references
        assert b._mapped_position_observations[9]["last"]["capture_frame_id"] == cap
        assert gallery(b) == original_gallery


def test_wrong_candidate_after_recovery_cannot_replace_following_owner():
    b = revoked_bank()
    assert send(b, 521) == 0
    assert send(b, 524) == 1
    assert send(b, 527) == 1
    assert send(b, 529) == 1
    original_gallery = gallery(b)
    owner = deepcopy(b._similar_follow_states[(1, 9)])
    assert send(b, 529, row=WRONG[529], track=10) == 0
    assert b.track_to_uid[9] == 1 and 10 not in b.track_to_uid
    assert b._similar_follow_states[(1, 9)] == owner
    assert gallery(b) == original_gallery


def test_similar_binding_from_recovery_still_rejects_failed_current_competition():
    b = revoked_bank()
    assert send(b, 521) == 0
    assert send(b, 524) == 1
    m, _, _ = sample(527)
    m["identity_competition"]["passed"] = False
    assert send(b, 527, **m) == 0
    assert not b.last_assignments[9]["bank_updated"]


@pytest.mark.parametrize("cap", [529, 562])
def test_recorded_wrong_raw10_never_recovers_or_updates_templates(cap):
    b = revoked_bank()
    original_gallery = gallery(b)
    assert send(b, cap, row=WRONG[cap], track=10) == 0
    assert 10 not in b.track_to_uid and not b.pending_late_handoffs
    assert not b.last_assignments[10]["bank_updated"]
    assert gallery(b) == original_gallery


@pytest.mark.parametrize("change", ["same_raw", "identity_conflict", "unknown_origin",
    "stale", "weak", "no_torso", "unavailable_region", "ambiguous", "foreign_competition",
    "bad_full", "bad_partial", "expired_templates", "soft"])
def test_new_exit_requires_independent_current_paired_appearance(change):
    b = revoked_bank()
    m, full, part = sample(521)
    source, track = "partial", 9
    if change == "same_raw": track = 7
    elif change == "identity_conflict": b._mapped_geometry_conflicts[7]["search_contradiction"] = True
    elif change == "unknown_origin": b._mapped_geometry_conflicts.clear()
    elif change == "stale": m["is_fresh"] = False
    elif change == "weak": m["bbox_quality_tier"] = "weak"
    elif change == "no_torso": part = None
    elif change == "unavailable_region": m["detector_bbox"] = [136.6, 2.1, 266.9, 442.7]
    elif change == "ambiguous": m["identity_competition"]["passed"] = False
    elif change == "foreign_competition": m["identity_competition"]["frame_index"] -= 1
    elif change == "bad_full": full = .21
    elif change == "bad_partial": part = .41
    elif change == "expired_templates": m["capture_timestamp"] += 30.
    elif change == "soft": source = "soft_partial"
    r = b._observe_late_search_candidate(track_id=track, candidate_uid=1,
        distance=full, full_feature=feature(full), partial_feature=None if part is None else feature(part),
        match_source=source, candidate_count=1, bbox_quality_ok=True,
        sample_metadata=m, frame_index=185, geometry=b._handoff_geometry(1, m, 185))
    assert r[0:2] == (0, 0)
    assert r[2]["reason"] == "revoked_owner_candidate_unqualified"
    assert not b.pending_late_handoffs and not b.track_to_uid


def test_repeated_capture_cannot_complete_independent_confirmation():
    b = revoked_bank()
    assert send(b, 521) == 0
    m, _, _ = sample(524)
    m["capture_frame_id"] = 521
    assert send(b, 524, **m) == 0
    assert "independent_recovery_confirmation" not in b.last_assignments[9]["reacquire_geometry"]


def test_weak_pending_cannot_supply_first_independent_confirmation():
    from rk_vision.identity_bank import PendingLateHandoff
    b = revoked_bank()
    m, _, _ = sample(521)
    g = _geometry_observation(m, 184)
    b.pending_late_handoffs[9] = PendingLateHandoff(uid=1, last_frame=184, streak=5,
        center_ratio=g["center_x_ratio"], area=g["area"], area_units=g["area_units"],
        geometry_source=g["geometry_source"], capture_timestamp=g["capture_timestamp"]-.1)
    assert send(b, 521) == 0
    assert b.last_assignments[9]["late_candidate_streak"] == 1
