"""CAP221--252 recorded geometry/time with synthetic cosine descriptors.

This tests IdentityBank policy, not an OSNet embedding replay or vehicle
motion. Most descriptor distances isolate the coverage gate. The explicitly
named CAP63 comparison sequence uses synthetic vectors at independently
measured reliable-template distances, not the log's global gallery minimum.
"""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig, _geometry_observation
from rk_vision.template_memory import TemplateMemory
from test_cap874_identity_reacquire import feature, metadata


# frame, capture timestamp, detector bbox, integrated yaw, detector confidence.
# Source: run_20260929_231616_127903_d4784d55/request_0513_modular.log.
ROWS = {
    63: (15, 19963.520815260, (300.6915283, 26.9920197, 515.7787476, 474.4491577), 1.7486553, .9192646),
    217: (83, 19971.540593701, (228.9204407, 118.9599304, 336.8883667, 472.5247192), -53.5754304, .9004273),
    221: (84, 19971.741906026, (229.5108643, 121.1334381, 323.7209778, 474.8287964), -54.3045795, .8665199),
    225: (85, 19971.940085638, (229.5432281, 125.2802887, 319.2991943, 468.1743164), -47.4935038, .8702874),
    227: (86, 19972.072632563, (219.5072479, 122.9764099, 316.5052185, 457.8677979), -47.7564531, .8778224),
    229: (87, 19972.172362561, (209.8038025, 121.6083679, 314.3698120, 446.7741089), -47.5901779, .9041947),
    233: (88, 19972.372505041, (186.2838745, 117.6787415, 315.0707397, 453.6287842), -47.5901779, .9343346),
    235: (89, 19972.475857156, (186.1486206, 115.6791534, 311.9854736, 450.0037231), -47.3525967, .9079622),
    237: (90, 19972.572713143, (184.2525330, 114.9490662, 305.3135376, 452.2369385), -47.3525967, .8966598),
    239: (91, 19972.672397351, (182.7124023, 115.6426849, 296.6223145, 456.2716675), -47.3525967, .9267996),
    241: (92, 19972.773635520, (175.7096863, 117.6804047, 288.0493469, 452.6892700), -47.4292322, .9117297),
    242: (93, 19972.838246815, (170.1178284, 118.9117279, 283.0368347, 449.7794800), -47.5806311, .9154972),
    243: (94, 19972.904016874, (166.0560913, 118.9596252, 279.4552002, 446.1829834), -47.5806311, .9192646),
    246: (95, 19973.068888113, (159.9825134, 117.0344849, 272.9854431, 445.0178223), -48.2240908, .9154972),
    248: (96, 19973.169392758, (159.2259064, 119.5151672, 272.2004395, 444.0215454), -49.2900401, .9004273),
    250: (97, 19973.270530304, (163.6264343, 120.1028290, 274.6853027, 443.6994019), -50.3247868, .9079622),
    252: (98, 19973.371873471, (177.7063293, 122.1077881, 282.3963623, 441.5856934), -51.0325798, .9154972),
}
CHAIN = (225, 227, 229, 233, 235, 237, 239, 241, 242, 243, 246, 248, 250)
CAP63_DISTANCES = {225: .2936, 229: .3660, 233: .3620, 235: .3616,
    237: .3463, 239: .3809, 241: .3727, 243: .3629, 246: .3498,
    248: .3654, 250: .3990, 252: .4035}


def meta(cap, **changes):
    frame, stamp, box, yaw, score = ROWS[cap]
    result = dict(metadata(cap, stamp, box, yaw, search=False), frame_index=frame,
        control_frame_id=frame, track_id=1, image_width=640, image_height=480,
        partial_feature_source="osnet_torso", partial_observation=True,
        detector_confidence=score, source_detection_index=0)
    result.update(changes)
    x1, y1, x2, y2 = result["detector_bbox"]
    result.update(bbox=list(result["detector_bbox"]),
        center_x_ratio=(x1+x2)/1280., detector_center_x_ratio=(x1+x2)/1280.,
        area_ratio=(x2-x1)*(y2-y1)/(640*480), detector_area_ratio=(x2-x1)*(y2-y1)/(640*480))
    return result


def send(bank, cap, *, full=.25, part=.27, count=1, **changes):
    current = meta(cap, **changes)
    box = current["detector_bbox"]
    return bank.assign(track_id=current["track_id"], feature=feature(full),
        partial_feature=None if part is None else feature(part),
        confidence=current["detector_confidence"], area=(box[2]-box[0])*(box[3]-box[1]),
        frame_index=current["frame_index"], candidate_count=count,
        bbox_quality_ok=True, bbox_quality_tier="strong", sample_metadata=current)


def checkpoint():
    bank = IdentityBank(IdentityBankConfig(template_memory_enable=True,
        template_crosscheck_enable=True, appearance_region_safety_enable=True,
        partial_match_threshold=.45, partial_confirm_threshold=.40,
        mapped_verify_threshold=.45, controlled_handoff_enable=True,
        camera_hfov_deg=60., update_interval=1))
    bank._create_identity(feature(0), ROWS[63][0], meta(63), feature(0))
    bank._bind_reacquired_identity(1, 1, ROWS[217][0], meta(217))
    bank.identities[1].last_strong_observation = _geometry_observation(meta(217), ROWS[217][0])
    assert send(bank, 221) == 1  # A real assign establishes the checkpoint.
    assert bank._appearance_verified[1]["comparison_mode"] == "exact_coverage"
    assert bank._appearance_verified[1]["comparable_caps"] == [63]
    assert bank._reacquire_quarantine.is_held(1)
    return bank


def gallery(bank):
    entry = bank.identities[1]
    memory = entry.template_memory
    result = []
    for vectors, records in ((entry.features, entry.feature_metadata),
                             (entry.partial_features, entry.partial_feature_metadata),
                             (entry.weak_features, entry.weak_feature_metadata)):
        result.append([(v.tobytes(), json.dumps(m, sort_keys=True)) for v, m in zip(vectors, records)])
    result.append({tier: [(v.tobytes(), json.dumps(m, sort_keys=True)) for v, m in rows]
                   for tier, rows in memory.recent.items()})
    result.append(deepcopy(memory.last_learning))
    return result


def test_cap225_boundary_preserves_confirmed_uid_without_learning_or_quarantine_release():
    bank = checkpoint()
    frozen = gallery(bank)
    for cap in CHAIN:
        assert send(bank, cap) == 1
        assignment = bank.last_assignments[1]
        evidence = assignment["reacquire_recent_partial_evidence"]
        if cap in (225, 227, 229, 237):
            assert evidence["comparison_mode"] == "verified_pose_continuation"
            assert evidence["pose_bridge_caps"] == [63]
        assert bank._appearance_verified[1]["pose_started"] == ROWS[221][1]
        assert bank._appearance_verified[1]["pose_caps"] == [63]
        assert not assignment["bank_updated"]
        assert bank._reacquire_quarantine.is_held(1)
        assert gallery(bank) == frozen


def test_normal_bridge_at_cap233_cannot_restart_fixed_two_second_budget():
    bank = checkpoint()
    for cap in (225, 227, 229):
        assert send(bank, cap) == 1
    origin = deepcopy(bank._appearance_verified[1]["pose_origin"])
    assert send(bank, 233) == 1
    assert bank.last_assignments[1]["reacquire_recent_partial_evidence"]["comparison_mode"] == "vertical_border_bridge"
    assert bank._appearance_verified[1]["pose_started"] == ROWS[221][1]
    assert bank._appearance_verified[1]["pose_origin"] == origin
    for cap in CHAIN[4:]:
        assert send(bank, cap) == 1
    assert send(bank, 252) == 1
    start = ROWS[221][1]
    assert send(bank, 252, capture_frame_id=255, frame_index=99, capture_timestamp=start+1.80) == 1
    assert send(bank, 252, capture_frame_id=259, frame_index=100, capture_timestamp=start+1.99) == 1
    assert bank._appearance_verified[1]["pose_started"] == start
    assert send(bank, 252, capture_frame_id=260, frame_index=101, capture_timestamp=start+2.01) == 0
    assert bank.last_assignments[1]["reason"] == "secondary_evidence_unavailable"


def test_current_cap63_partial_comparison_is_not_replaced_by_global_gallery_minimum():
    bank = checkpoint()
    for earlier in CHAIN:
        # CAP227/242 have no independently measured distance in this fixture.
        assert send(bank, earlier, part=CAP63_DISTANCES.get(earlier, .27)) == 1
    frozen = gallery(bank)
    assert send(bank, 252, part=CAP63_DISTANCES[252]) == 0
    assert bank.last_assignments[1]["reacquire_partial_state"] != "match"
    assert not bank.last_assignments[1]["bank_updated"]
    assert gallery(bank) == frozen
    assert bank._reacquire_quarantine.is_held(1)


@pytest.mark.parametrize("case", ["search", "new_track", "contradiction", "competition", "crowd",
    "duplicate", "older", "missing_capture", "missing_timestamp", "gap", "stale", "no_proof",
    "tiny", "side_crop", "bad_full", "partial_conflict", "missing_partial", "jump", "large_jump", "yaw_missing",
    "low_confidence", "suspect"])
def test_pose_continuation_does_not_authorize_unverified_or_noncontinuous_candidate(case):
    bank = checkpoint()
    frozen = gallery(bank)
    changes = {}
    if case == "search": changes["search_reacquire_context_active"] = True
    if case == "new_track": changes["track_id"] = 8
    if case == "contradiction": bank._mapped_geometry_conflicts[1] = dict(uid=1,
        search_contradiction=True, reference=deepcopy(bank.identities[1].last_strong_observation))
    if case == "competition": changes["identity_competition"] = dict(uid=1, frame_index=85,
        candidate_count=1, source_detection_index=0, passed=False)
    if case == "crowd": changes["count"] = 2
    if case == "duplicate": changes.update(capture_frame_id=221, capture_timestamp=ROWS[221][1])
    if case == "older": changes["capture_timestamp"] = ROWS[221][1]-.01
    if case == "missing_capture": changes["capture_frame_id"] = None
    if case == "missing_timestamp": changes["capture_timestamp"] = None
    if case == "gap": changes["capture_timestamp"] = ROWS[221][1]+.251
    if case == "stale": changes["is_fresh"] = False
    if case == "no_proof": bank._appearance_verified.clear()
    if case == "tiny": changes["detector_bbox"] = [250., 170., 290., 280.]
    if case == "side_crop": changes["detector_bbox"] = [0., 125., 90., 468.]
    if case == "bad_full": changes["full"] = .31
    if case == "partial_conflict": changes["part"] = .46
    if case == "missing_partial": changes["part"] = None
    if case == "jump": changes["detector_bbox"] = [420., 125., 510., 468.]
    if case == "large_jump": changes["detector_bbox"] = [500., 125., 590., 468.]
    if case == "yaw_missing": changes["integrated_yaw_deg"] = None
    if case == "low_confidence": changes["detector_confidence"] = .79
    if case == "suspect": bank._reacquire_control_suspects[1] = dict(track_id=1, streak=0, reason="partial_conflict")
    assert send(bank, 225, **changes) == 0
    assert not bank.last_assignments[changes.get("track_id", 1)]["bank_updated"]
    assert gallery(bank) == frozen
    assert bank._reacquire_quarantine.is_held(1)


def test_global_template_comparison_remains_strict_without_verified_pose_proof():
    memory = TemplateMemory()
    memory.remember(feature(0), meta(63), "partial")
    frozen = deepcopy(memory.last_learning)
    for cap in (225, 227, 229, 237):
        assert not TemplateMemory.vertical_border_comparable(meta(cap), meta(63))
        result = memory.evidence(feature(.01), meta(cap), "partial", comparable_only=True)
        assert result["count"] == 0 and result["pose_bridge_caps"] == []
    assert memory.last_learning == frozen


def test_old_template_expiry_beats_live_pose_sequence():
    bank = checkpoint()
    assert send(bank, 225) == 1
    bank.identities[1].template_memory.recent["partial"][0][1]["capture_timestamp"] = ROWS[227][1]-30.1
    assert send(bank, 227) == 0
    assert bank.last_assignments[1]["reason"] == "secondary_evidence_unavailable"


@pytest.mark.parametrize("soft,full", [(.45, .31), (.15, .20)])
def test_pose_full_body_gate_is_capped_at_point_three_and_respects_stricter_config(soft, full):
    bank = checkpoint()
    bank.config = replace(bank.config, preferred_search_soft_candidate_threshold=soft)
    frozen = gallery(bank)
    assert send(bank, 225, full=full) == 0
    assert gallery(bank) == frozen


def test_repeated_high_similarity_pose_only_frames_do_not_release_quarantine_or_learn():
    bank = checkpoint()
    frozen = gallery(bank)
    for index in range(10):
        assert send(bank, 225, full=.10, part=.20, capture_frame_id=225+index,
            frame_index=85+index, capture_timestamp=ROWS[225][1]+.15*index) == 1
        evidence = bank.last_assignments[1]["reacquire_recent_partial_evidence"]
        assert evidence["comparison_mode"] == "verified_pose_continuation"
        assert evidence["pose_bridge_caps"] == [63]
        assert bank._reacquire_quarantine.is_held(1)
        assert not bank.last_assignments[1]["bank_updated"]
        assert gallery(bank) == frozen


def test_pose_epoch_cannot_be_extended_by_switching_to_scale_continuation():
    bank = checkpoint()
    for cap in (225, 227, 229, 233, 235, 237):
        assert send(bank, cap) == 1
    start = ROWS[221][1]
    for index, delta in enumerate((1., 1.2, 1.4, 1.6, 1.8, 1.99)):
        assert send(bank, 237, capture_frame_id=260+index, frame_index=100+index,
                    capture_timestamp=start+delta) == 1
        evidence = bank.last_assignments[1]["reacquire_recent_partial_evidence"]
        assert evidence["comparison_mode"] == "verified_pose_continuation"
        assert not evidence["scale_bridge_caps"]
        assert bank._appearance_verified[1]["pose_started"] == start
    assert send(bank, 237, capture_frame_id=266, frame_index=106,
                capture_timestamp=start+2.01) == 0


def test_expired_pose_epoch_survives_independent_normal_bridge_before_next_narrow_frame():
    bank = checkpoint()
    for cap in (225, 227, 229, 233, 235):
        assert send(bank, cap) == 1
    start = ROWS[221][1]
    frozen = gallery(bank)
    # A current ordinary bridge can independently match, but must not erase
    # the expired pose budget and thereby bootstrap a fresh scale budget.
    for index, delta in enumerate((.95, 1.15, 1.35, 1.55, 1.75, 1.95, 2.05)):
        assert send(bank, 233, capture_frame_id=260+index, frame_index=100+index,
                    capture_timestamp=start+delta) == 1
        evidence = bank.last_assignments[1]["reacquire_recent_partial_evidence"]
        assert evidence["comparison_mode"] == "vertical_border_bridge"
        assert bank._appearance_verified[1]["pose_started"] == start
        assert gallery(bank) == frozen
    assert send(bank, 237, capture_frame_id=267, frame_index=107,
                capture_timestamp=start+2.15) == 0
    assert bank.last_assignments[1]["reason"] == "secondary_evidence_unavailable"


def expired_pose_checkpoint():
    bank = checkpoint()
    start = ROWS[221][1]
    frozen = gallery(bank)
    for cap in (225, 227, 229, 233, 235, 237):
        assert send(bank, cap) == 1
    for index, delta in enumerate((1., 1.2, 1.4, 1.6, 1.8, 1.99, 2.01)):
        uid = send(bank, 237, capture_frame_id=300+index, frame_index=100+index,
            capture_timestamp=start+delta)
        assert uid == (0 if delta > 2 else 1)
    assert bank.last_assignments[1]['reason'] == 'secondary_evidence_unavailable'
    assert 1 not in bank._appearance_verified  # No expired positive approval survives.
    assert 1 in bank._pose_retention_blocked
    assert gallery(bank) == frozen
    return bank, start, frozen


def test_expiry_rejection_cannot_seed_new_scale_budget_through_normal_border():
    bank, start, frozen = expired_pose_checkpoint()
    assert send(bank, 233, capture_frame_id=400, frame_index=120,
        capture_timestamp=start+2.11) == 1
    assert bank.last_assignments[1]['reacquire_recent_partial_evidence']['comparison_mode'] == 'vertical_border_bridge'
    assert 1 in bank._pose_retention_blocked
    assert send(bank, 237, capture_frame_id=401, frame_index=121,
        capture_timestamp=start+2.21) == 0
    assert bank.last_assignments[1]['reason'] == 'secondary_evidence_unavailable'
    assert 1 not in bank._appearance_verified
    assert 1 in bank._pose_retention_blocked
    assert gallery(bank) == frozen


def test_independent_exact_confirmation_after_rejection_can_start_new_pose_epoch():
    bank, start, _ = expired_pose_checkpoint()
    assert send(bank, 221, capture_frame_id=400, frame_index=120,
        capture_timestamp=start+2.11, full=.1, part=.2) == 1
    assert bank.last_assignments[1]['reacquire_recent_partial_evidence']['comparison_mode'] == 'exact_coverage'
    assert 1 not in bank._pose_retention_blocked
    assert send(bank, 225, capture_frame_id=401, frame_index=121,
        capture_timestamp=start+2.21) == 1
    assert bank._appearance_verified[1]['pose_started'] == pytest.approx(start+2.11)


def test_rejected_exact_comparison_cannot_clear_exhausted_pose_restriction():
    bank, start, frozen = expired_pose_checkpoint()
    assert send(bank, 221, capture_frame_id=400, frame_index=120,
        capture_timestamp=start+2.11, part=.46) == 0
    assert bank.last_assignments[1]['reason'] == 'recent_partial_conflict'
    assert 1 in bank._pose_retention_blocked
    assert 1 not in bank._appearance_verified
    assert gallery(bank) == frozen


@pytest.mark.parametrize('capture,delta', [(306, 2.11), (400, 1.99)])
def test_consumed_capture_or_old_timestamp_cannot_clear_negative_pose_memory(capture, delta):
    bank, start, _ = expired_pose_checkpoint()
    # An ordinary mapped branch can still inspect the old exact descriptor;
    # it must not count as new evidence that clears the discarded epoch.
    send(bank, 221, capture_frame_id=capture, frame_index=120,
        capture_timestamp=start+delta, full=.1, part=.2)
    assert 1 in bank._pose_retention_blocked
    assert send(bank, 221, capture_frame_id=401, frame_index=121,
        capture_timestamp=start+2.21, full=.1, part=.2) == 1
    assert 1 not in bank._pose_retention_blocked


def test_independent_approved_learning_can_clear_exhausted_pose_restriction():
    bank, start, frozen = expired_pose_checkpoint()
    for index in range(3):
        assert send(bank, 233, capture_frame_id=400+index, frame_index=120+index,
            capture_timestamp=start+2.11+.1*index, full=.1, part=.2) == 1
        if index < 2:
            assert 1 in bank._pose_retention_blocked
            assert gallery(bank) == frozen
    assert bank.last_assignments[1]['bank_updated']
    assert 1 not in bank._pose_retention_blocked


def test_manual_identity_reset_clears_negative_pose_memory():
    bank, _, _ = expired_pose_checkpoint()
    bank.reset()
    assert not bank._pose_retention_blocked
    assert not bank._appearance_verified
