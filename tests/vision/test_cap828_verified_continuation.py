"""CAP828 policy integration, not an embedding-model or vehicle replay.

Boxes, capture times, yaw and the named full/torso distances below come from
run_20260929_234529_138351_7abb13d5. Synthetic unit vectors reproduce distances
to one paired, trusted CAP414 template. The two seed observations deliberately
isolate an already confirmed partial/strong reacquisition; they do not claim
to replay the complete earlier search or the original multi-template gallery.
"""
from copy import deepcopy
from dataclasses import replace

import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from test_cap225_pose_continuity import gallery
from test_cap874_identity_reacquire import feature, metadata


# processed frame, capture timestamp, detector box, integrated yaw
ROWS = {
    414: (145, 21735.270401774, (338.984802, 121.640793, 456.651917, 448.673218), -50.230918),
    826: (330, 21756.733782490, (434.653961, 128.936600, 553.157227, 456.817627), 100.235539),
    828: (331, 21756.842718689, (445.637421, 135.770691, 559.968018, 459.210693), 100.153731),
    830: (332, 21756.934999101, (457.986298, 135.827484, 571.680725, 444.679810), 99.900716),
    832: (333, 21757.035196022, (468.048767, 137.632462, 579.662903, 434.895630), 99.900716),
    837: (336, 21757.269903868, (472.374054, 142.459503, 581.747803, 426.792755), 99.900716),
    842: (339, 21757.570048098, (461.128174, 150.895889, 559.796509, 423.686615), 100.306181),
    843: (340, 21757.602237950, (459.645599, 151.478653, 557.253296, 420.768005), 100.582122),
    845: (341, 21757.702231000, (455.543488, 150.874359, 546.989685, 406.663330), 100.582122),
    847: (342, 21757.803009197, (457.653351, 153.097031, 546.251465, 404.960419), 100.582122),
    849: (343, 21757.907069437, (461.733612, 154.064911, 546.280151, 401.417572), 100.582122),
    851: (344, 21758.002903426, (468.491302, 156.466736, 547.725037, 402.754028), 100.582122),
    879: (359, 21759.466722388, (60.259285, 143.194550, 122.394104, 399.898834), 127.806524),
    881: (360, 21759.565890626, (30.330639, 141.029007, 99.156830, 401.866394), 129.203806),
}


def meta(cap, **changes):
    frame, stamp, box, yaw = ROWS[cap]
    result = dict(metadata(cap, stamp, box, yaw, search=False), frame_index=frame,
        control_frame_id=frame, track_id=3, image_width=640, image_height=480,
        partial_feature_source="osnet_torso", partial_observation=False,
        source_detection_index=0)
    result.update(changes)
    box = result["detector_bbox"]
    result.update(bbox=list(box), center_x_ratio=(box[0]+box[2])/1280,
        detector_center_x_ratio=(box[0]+box[2])/1280,
        detector_area_ratio=(box[2]-box[0])*(box[3]-box[1])/(640*480),
        area_ratio=(box[2]-box[0])*(box[3]-box[1])/(640*480))
    return result


def send(bank, cap, full=.326, part=.362, **changes):
    current = meta(cap, **changes)
    box = current["detector_bbox"]
    return bank.assign(track_id=current["track_id"], feature=feature(full),
        partial_feature=None if part is None else feature(part), confidence=.92,
        area=(box[2]-box[0])*(box[3]-box[1]), frame_index=current["frame_index"],
        candidate_count=current["candidate_count"], bbox_quality_ok=True,
        bbox_quality_tier="strong", sample_metadata=current,
        preferred_uid=1 if current["search_reacquire_context_active"] else None,
        preferred_candidate_ok=current.get("search_direction_compatible") is not False)


def empty_bank():
    bank = IdentityBank(IdentityBankConfig(template_memory_enable=True,
        template_crosscheck_enable=True, appearance_region_safety_enable=True,
        partial_match_threshold=.45, partial_confirm_threshold=.40,
        mapped_verify_threshold=.45, match_threshold=.38, controlled_handoff_enable=True,
        camera_hfov_deg=60., update_interval=1))
    bank._create_identity(feature(0), ROWS[414][0], meta(414, track_id=1), feature(0))
    return bank


def checkpoint(source="partial", at=830):
    bank = empty_bank()
    # Actual assign calls establish source provenance; no test manufactures
    # _appearance_verified or a continuation permission in private state.
    cap0, cap1 = (826, 828) if at == 830 else (879, 879)
    extra0 = {} if at == 830 else dict(capture_frame_id=875,
        capture_timestamp=ROWS[879][1]-.20, frame_index=357)
    extra1 = {} if at == 830 else dict(capture_frame_id=877,
        capture_timestamp=ROWS[879][1]-.10, frame_index=358)
    initial_full = .25 if source == "partial" else .17
    # The legacy partial-observation label selects the previously supported
    # reacquisition source here; every tested complete-body query clears it.
    assert send(bank, cap0, initial_full, .30, partial_observation=source == "partial",
        search_reacquire_context_active=True, **extra0) == 0
    assert send(bank, cap1, initial_full, .30, partial_observation=source == "partial",
        search_reacquire_context_active=True, **extra1) == 1
    assert bank._candidate_observations.rows[(1, 3)]["late_confirmed_source"] == source
    assert send(bank, at, .296 if source == "partial" else .186,
                .376 if source == "partial" else .275) == 1
    assert bank._appearance_verified[1]["metadata"]["capture_frame_id"] == at
    assert bank._reacquire_quarantine.is_held(1)
    return bank


def test_cap832_and_837_complete_body_can_use_confirmed_partial_source():
    bank = checkpoint()
    frozen = gallery(bank)
    for cap, full, part in [(832, .326, .362), (837, .371, .329)]:
        assert send(bank, cap, full, part) == 1
        result = bank.last_assignments[3]
        assert result["reason"] == "mapped_verified_continuation"
        assert result["identity_continuation"]["source"] == "partial"
        assert result["identity_continuation"]["pair_cap"] == 414
        assert not result["bank_updated"]
        assert bank._reacquire_quarantine.is_held(1)
        assert gallery(bank) == frozen


def test_real_grey_frames_recheck_without_rolling_deadline_or_template_learning():
    bank = checkpoint()
    frozen = gallery(bank)
    rows = [(832, .326, .362, 1), (837, .371, .329, 1), (842, .338, .381, 1),
            (843, .317, .408, 0), (845, .328, .397, 1), (847, .325, .435, 0),
            (849, .299, .420, 0), (851, .340, .394, 1)]
    last_good = 830
    for cap, full, part, expected in rows:
        assert send(bank, cap, full, part) == expected, (cap, bank.last_assignments[3])
        result = bank.last_assignments[3]
        if not expected:
            assert result["reason"] == "verified_continuation_recheck"
            assert result["identity_recheck_pending"]
            assert result["identity_continuation"]["deadline"] == pytest.approx(ROWS[last_good][1]+.35)
            assert bank._appearance_verified[1]["metadata"]["capture_frame_id"] == last_good
        else:
            last_good = cap
        assert not result["bank_updated"]
        assert bank._reacquire_quarantine.is_held(1)
        assert gallery(bank) == frozen
    # The two real grey samples must share CAP845's deadline. CAP851 is
    # ~301 ms after CAP845, not a fresh 350 ms allowance after CAP849.
    assert ROWS[851][1]-ROWS[845][1] == pytest.approx(.300672426)


def test_cap879_strong_reacquisition_survives_later_point_two_search_gate():
    bank = checkpoint("strong", at=879)
    frozen = gallery(bank)
    assert send(bank, 881, .218, .355, search_reacquire_context_active=True) == 1
    assert bank.last_assignments[3]["reason"] == "mapped_verified_continuation"
    assert bank.last_assignments[3]["identity_continuation"]["source"] == "strong"
    assert gallery(bank) == frozen
    assert bank._reacquire_quarantine.is_held(1)


@pytest.mark.parametrize("partial_label", [False, True])
def test_complete_vs_clipped_label_does_not_change_valid_current_descriptor_permission(partial_label):
    bank = checkpoint()
    assert send(bank, 832, .326, .362, partial_observation=partial_label) == 1
    assert bank.last_assignments[3]["reason"] == "mapped_verified_continuation"


def test_grey_replays_do_not_refresh_deadline_or_complete_identity():
    bank = checkpoint()
    for _ in range(3):
        assert send(bank, 832, .326, .420) == 0
        assert not bank.last_assignments[3]["bank_updated"]
    assert bank._appearance_verified[1]["metadata"]["capture_frame_id"] == 830
    assert send(bank, 837, .326, .362, capture_timestamp=ROWS[830][1]+.351) == 0
    assert bank.last_assignments[3]["reason"] != "mapped_verified_continuation"


@pytest.mark.parametrize("case", ["no_provenance", "new_track", "duplicate", "old_capture",
    "stale", "gap", "competition", "contradiction", "revoked", "suspect", "jump",
    "scale_jump", "side_crop", "missing_partial", "partial_conflict", "expired_template",
    "unpaired_template", "strict_full_config", "strict_match_config", "strict_partial_config", "weak_full"])
def test_continuation_does_not_turn_uncertainty_into_identity(case):
    bank = checkpoint()
    changes = {}; full=.326; part=.362
    if case == "no_provenance":
        bank._appearance_verified.clear()
        bank._candidate_observations.rows.clear()
    elif case == "new_track": changes["track_id"] = 4
    elif case == "duplicate": changes.update(capture_frame_id=830, capture_timestamp=ROWS[830][1])
    elif case == "old_capture": changes.update(capture_frame_id=829, capture_timestamp=ROWS[830][1]-.01)
    elif case == "stale": changes["is_fresh"] = False
    elif case == "gap": changes["capture_timestamp"] = ROWS[830][1]+.351
    elif case == "competition": changes["identity_competition"] = dict(uid=1, frame_index=333, passed=False)
    elif case == "contradiction":
        bank._mapped_geometry_conflicts[3] = dict(uid=1, search_contradiction=True,
            reference=deepcopy(bank.identities[1].last_strong_observation))
    elif case == "revoked": bank._geometry_revoked_uids[1] = ROWS[830][0]
    elif case == "suspect": bank._reacquire_control_suspects[1] = dict(track_id=3,
        reason="partial_conflict", streak=0, capture=830, timestamp=ROWS[830][1])
    elif case == "jump": changes["detector_bbox"] = [80., 137., 191., 435.]
    elif case == "scale_jump": changes["detector_bbox"] = [480., 240., 530., 370.]
    elif case == "side_crop": changes["detector_bbox"] = [0., 137., 111., 435.]
    elif case == "missing_partial": part = None
    elif case == "partial_conflict": part = .46
    elif case == "expired_template":
        bank.identities[1].template_memory.recent_sec = .1
    elif case == "unpaired_template":
        for _, row in bank.identities[1].template_memory.recent["partial"]:
            row["capture_frame_id"] = 415
    elif case == "strict_full_config": bank.config = replace(bank.config, mapped_verify_threshold=.30)
    elif case == "strict_match_config": bank.config = replace(bank.config, match_threshold=.30)
    elif case == "strict_partial_config": bank.config = replace(bank.config, partial_confirm_threshold=.34)
    elif case == "weak_full": full = .451
    assert send(bank, 832, full, part, **changes) == 0
    result = bank.last_assignments[changes.get("track_id", 3)]
    assert result["reason"] != "mapped_verified_continuation"
    assert not result["bank_updated"]


def test_strong_source_cannot_acquire_partial_sources_larger_full_permission():
    bank = checkpoint("strong")
    assert send(bank, 832, .326, .362) == 0
    assert bank.last_assignments[3]["reason"] != "mapped_verified_continuation"


def test_recomputed_grey_capture_cannot_turn_into_fresh_identity_permission():
    bank = checkpoint()
    frozen = gallery(bank)
    assert send(bank, 832, .326, .420) == 0
    deadline = bank.last_assignments[3]['identity_recheck_deadline']
    # Even a newly perfect embedding belongs to the SAME physical capture.
    assert send(bank, 832, .10, .10) == 0
    assert bank.last_assignments[3]['reason'] == 'verified_continuation_nonnew'
    assert bank.last_assignments[3]['identity_recheck_deadline'] == deadline
    assert bank._appearance_verified[1]['metadata']['capture_frame_id'] == 830
    assert gallery(bank) == frozen
    assert send(bank, 832, .339, .372, capture_frame_id=833,
                frame_index=ROWS[832][0]+1,
                capture_timestamp=ROWS[832][1]+.04) == 1


@pytest.mark.parametrize('recent,paired,expected', [(.352, .430, 1), (.390, .430, 0), (.352, .451, 0)])
def test_recent_support_and_same_capture_crosscheck_keep_distinct_limits(recent, paired, expected):
    bank = checkpoint()
    mem = bank.identities[1].template_memory
    # CAP840-like case: current recent support is .352, but the SAME torso
    # template's full distance is .430. Neither is called a strong match.
    original = deepcopy(mem.recent['strong'][0][1])
    other = dict(original, capture_frame_id=700, capture_timestamp=ROWS[830][1]-2.)
    mem.recent['strong'] = [(feature(paired), original), (feature(recent), other)]
    frozen = gallery(bank)
    assert send(bank, 832, 0., .366) == expected
    assert gallery(bank) == frozen
    if expected:
        result = bank.last_assignments[3]
        assert result['match_source'] == 'partial'
        assert result['identity_continuation_pair']['winner_cap'] == 414
        assert result['identity_continuation_pair']['full_distance'] == pytest.approx(paired)


def test_temporary_shape_retention_cannot_become_independent_continuation_pair():
    bank = checkpoint()
    mem = bank.identities[1].template_memory
    reliable = dict(comparison_mode='exact_coverage', comparable_caps=[414], shape_hysteresis_caps=[414])
    pair = mem.continuation_pair_evidence(feature(.10), feature(.20), meta(832),
                                          reliable, full_limit=.45)
    assert not pair['qualified_comparison']


def send_quality_contract(bank, *, argument_ok=True, argument_tier="strong",
                          reason="", metadata_changes=None, part=.362):
    current = meta(832, **(metadata_changes or {}))
    box = current["detector_bbox"]
    return bank.assign(track_id=3, feature=feature(.326), partial_feature=feature(part),
        confidence=.92, area=(box[2]-box[0])*(box[3]-box[1]), frame_index=current["frame_index"],
        candidate_count=1, bbox_quality_ok=argument_ok, bbox_quality_tier=argument_tier,
        bbox_quality_reason=reason, sample_metadata=current)


@pytest.mark.parametrize("part", [.362, .420])
@pytest.mark.parametrize("case", ["argument_false", "metadata_false", "weak_partial",
    "argument_weak", "argument_reject", "metadata_weak", "metadata_reject"])
def test_quality_veto_cannot_be_overruled_by_other_strong_flags(case, part):
    bank = checkpoint()
    changes = {}
    if case == "argument_false": changes["argument_ok"] = False
    elif case == "metadata_false": changes["metadata_changes"] = {"quality_bbox_ok": False}
    elif case.startswith("argument_"): changes["argument_tier"] = case.split("_")[1]
    elif case.startswith("metadata_"):
        changes["metadata_changes"] = {"bbox_quality_tier": case.split("_")[1]}
    else:
        changes.update(argument_ok=False, argument_tier="weak", reason="edge_touch>2",
                       metadata_changes=dict(partial_observation=True))
    assert send_quality_contract(bank, part=part, **changes) == 0
    result = bank.last_assignments[3]
    assert result["reason"] not in ("mapped_verified_continuation", "verified_continuation_recheck")
    assert not result.get("identity_recheck_pending")
    assert not result["bank_updated"]
    assert 1 not in bank._appearance_verified


@pytest.mark.parametrize("reason", ["identity_center_jump>0.30", "identity_swap_competing_track"])
@pytest.mark.parametrize("location", ["argument", "bbox_metadata", "detector_metadata"])
def test_explicit_swap_reason_precedes_positive_or_pending_continuation(reason, location):
    bank = checkpoint()
    # First establish a STOP-only grey state. A subsequent negative identity
    # signal must clear it even if both caller quality booleans still say true.
    assert send(bank, 832, .326, .420) == 0
    assert bank.last_assignments[3]["identity_recheck_pending"]
    changes = dict(capture_frame_id=833, capture_timestamp=ROWS[832][1]+.02,
                   frame_index=334)
    kwargs = {}
    if location == "argument": kwargs["reason"] = reason
    elif location == "bbox_metadata": changes["bbox_quality_reason"] = reason
    else: changes["quality_bbox_reason"] = reason
    assert send_quality_contract(bank, metadata_changes=changes, **kwargs) == 0
    result = bank.last_assignments[3]
    assert result["reason"] == "identity_center_jump_reject"
    assert result["identity_control_rejected"]
    assert not result.get("identity_recheck_pending")
    assert not result["bank_updated"]
    assert 3 not in bank.track_to_uid
    assert 1 not in bank._appearance_verified


@pytest.mark.parametrize("scene", ["cap874", "cap714", "cap2394"])
def test_prior_wrong_people_cannot_manufacture_verified_continuation(scene):
    if scene == "cap874":
        from test_cap874_identity_reacquire import bank_at_857, assign, ROWS as wrong
        bank = bank_at_857(); run=lambda row: assign(bank, row)
    elif scene == "cap714":
        from test_cap714_reacquire_geometry import seeded_bank, send as submit, ROWS as wrong
        bank = seeded_bank(); run=lambda row: submit(bank, row)
    else:
        from test_cap2394_handoff_conflict import bank as setup, send as submit, ROWS as wrong
        bank = setup(); run=lambda row: submit(bank, row)
    bank.config = replace(bank.config, partial_match_threshold=.45, partial_confirm_threshold=.40)
    for row in wrong:
        assert run(row) == 0
        assert all(a["reason"] != "mapped_verified_continuation" for a in bank.last_assignments.values())
