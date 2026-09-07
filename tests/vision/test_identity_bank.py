#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig, IdentityEntry


def _unit(values):
    arr = np.asarray(values, dtype="float32")
    return arr / max(float(np.linalg.norm(arr)), 1e-12)


def _x_axis_feature_at_distance(distance: float):
    cosine = 1.0 - float(distance)
    return _unit([cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine)), 0.0])


def _z_plane_feature_at_distance(distance: float):
    cosine = 1.0 - float(distance)
    return _unit([cosine, 0.0, math.sqrt(max(0.0, 1.0 - cosine * cosine))])


def _assert_multi_candidate_reacquire_rules() -> None:
    bank = IdentityBank(
        IdentityBankConfig(
            match_threshold=0.32,
            match_margin=0.03,
            update_threshold=0.30,
            update_interval=1,
            max_features=3,
            min_confidence=0.65,
            reacquire_threshold=0.50,
            reacquire_max_frames=10,
            reacquire_margin=0.08,
            reacquire_multi_candidate_threshold=0.40,
            reacquire_multi_candidate_margin=0.12,
            new_identity_confirm_frames=1,
        )
    )
    uid1 = bank.assign(track_id=1, feature=_unit([1.0, 0.0, 0.0]), confidence=0.9, area=1000, frame_index=1)
    uid2 = bank.assign(track_id=2, feature=_unit([0.0, 0.0, 1.0]), confidence=0.9, area=1000, frame_index=2)
    uid3 = bank.assign(
        track_id=3,
        feature=_x_axis_feature_at_distance(0.35),
        confidence=0.9,
        area=1000,
        frame_index=4,
        candidate_count=2,
    )
    state = bank.debug_state()
    if uid1 != 1 or uid2 != 2:
        raise AssertionError(f"expected seeded uids 1 and 2, got {uid1}, {uid2}")
    if uid3 != uid1:
        raise AssertionError(f"strict multi-candidate reacquire should keep uid {uid1}, got {uid3}")
    if state["last_assignments"]["3"]["reason"] != "reacquired_multi":
        raise AssertionError(f"expected reacquired_multi reason, got {state['last_assignments']['3']}")

    weak_bank = IdentityBank(
        IdentityBankConfig(
            match_threshold=0.32,
            update_threshold=0.30,
            update_interval=1,
            min_confidence=0.65,
            reacquire_threshold=0.50,
            reacquire_max_frames=10,
            reacquire_margin=0.08,
            reacquire_multi_candidate_threshold=0.40,
            reacquire_multi_candidate_margin=0.12,
            new_identity_confirm_frames=1,
        )
    )
    weak_uid1 = weak_bank.assign(
        track_id=1,
        feature=_unit([1.0, 0.0, 0.0]),
        confidence=0.9,
        area=1000,
        frame_index=1,
    )
    weak_bank.assign(
        track_id=2,
        feature=_unit([0.0, 0.0, 1.0]),
        confidence=0.9,
        area=1000,
        frame_index=2,
    )
    weak_uid = weak_bank.assign(
        track_id=3,
        feature=_x_axis_feature_at_distance(0.43),
        confidence=0.9,
        area=1000,
        frame_index=4,
        candidate_count=2,
    )
    if weak_uid == weak_uid1:
        raise AssertionError(f"multi-candidate distance over strict threshold should not reacquire uid {weak_uid1}")


def _assert_reacquire_uses_last_seen_frame() -> None:
    bank = IdentityBank(
        IdentityBankConfig(
            match_threshold=0.20,
            match_margin=0.01,
            update_threshold=0.10,
            update_interval=1,
            max_features=3,
            min_confidence=0.65,
            reacquire_threshold=0.50,
            reacquire_max_frames=10,
            reacquire_margin=0.01,
            new_identity_confirm_frames=1,
        )
    )
    uid1 = bank.assign(track_id=1, feature=_unit([1.0, 0.0, 0.0]), confidence=0.9, area=1000, frame_index=1)
    uid_seen = bank.assign(
        track_id=1,
        feature=_x_axis_feature_at_distance(0.25),
        confidence=0.9,
        area=1000,
        frame_index=8,
    )
    state_after_seen = bank.debug_state()
    identity = state_after_seen["identities"][0]
    if uid_seen != uid1:
        raise AssertionError(f"mapped track should keep uid {uid1}, got {uid_seen}")
    if identity["last_frame"] != 1:
        raise AssertionError(f"feature bank should not update over threshold, got {identity}")
    if identity["last_seen_frame"] != 8:
        raise AssertionError(f"mapped sighting should refresh last_seen_frame, got {identity}")

    uid_reacquired = bank.assign(
        track_id=2,
        feature=_x_axis_feature_at_distance(0.35),
        confidence=0.9,
        area=1000,
        frame_index=17,
    )
    state = bank.debug_state()
    if uid_reacquired != uid1:
        raise AssertionError(f"reacquire should use last_seen_frame and keep uid {uid1}, got {uid_reacquired}")
    if state["last_assignments"]["2"]["reason"] != "reacquired":
        raise AssertionError(f"expected reacquired reason, got {state['last_assignments']['2']}")
    identity = state["identities"][0]
    if identity["last_frame"] != 1 or identity["last_seen_frame"] != 17:
        raise AssertionError(f"reacquire should touch seen time without bank update, got {identity}")


def _assert_exclusive_claim_and_mapped_quality_guards() -> None:
    bank = IdentityBank(
        IdentityBankConfig(
            match_threshold=0.32,
            update_threshold=0.30,
            update_interval=1,
            min_confidence=0.65,
            new_identity_confirm_frames=1,
            mapped_verify_enable=True,
            mapped_verify_threshold=0.40,
            mapped_bad_quality_returns_unassigned=True,
            exclusive_uid_claim_enable=True,
            exclusive_uid_claim_frames=15,
        )
    )
    uid1 = bank.assign(track_id=1, feature=_unit([1.0, 0.0, 0.0]), confidence=0.9, area=1000, frame_index=1)
    uid2 = bank.assign(track_id=2, feature=_unit([1.0, 0.0, 0.0]), confidence=0.9, area=1000, frame_index=2)
    if uid2 == uid1:
        raise AssertionError("recently claimed uid should not be handed to a second raw track")

    bad_quality_uid = bank.assign(
        track_id=1,
        feature=_unit([1.0, 0.0, 0.0]),
        confidence=0.9,
        area=1000,
        frame_index=3,
        bbox_quality_ok=False,
        bbox_quality_reason="edge_touch>2",
    )
    if bad_quality_uid != 0:
        raise AssertionError(f"mapped bad-quality bbox should not output uid, got {bad_quality_uid}")
    state = bank.debug_state()
    if state["track_to_uid"].get("1") != uid1:
        raise AssertionError(f"bad-quality frame should keep internal mapping for later verification, got {state}")

    reject_uid = bank.assign(
        track_id=1,
        feature=_x_axis_feature_at_distance(0.65),
        confidence=0.9,
        area=1000,
        frame_index=4,
    )
    state = bank.debug_state()
    if reject_uid != 0:
        raise AssertionError(f"mapped verification reject should output 0, got {reject_uid}")
    if "1" in state["track_to_uid"]:
        raise AssertionError(f"mapped verification reject should clear stale track mapping, got {state}")


def _assert_controlled_handoff_requires_consecutive_matches() -> None:
    bank = IdentityBank(
        IdentityBankConfig(
            match_threshold=0.32,
            update_threshold=0.30,
            min_confidence=0.65,
            new_identity_confirm_frames=1,
            exclusive_uid_claim_enable=True,
            exclusive_uid_claim_frames=15,
            controlled_handoff_enable=True,
            controlled_handoff_confirm_frames=3,
            controlled_handoff_threshold=0.30,
            controlled_handoff_min_old_track_gap_frames=2,
        )
    )
    feature = _unit([1.0, 0.0, 0.0])
    uid = bank.assign(track_id=1, feature=feature, confidence=0.9, area=1000, frame_index=1)
    wait1 = bank.assign(track_id=2, feature=feature, confidence=0.9, area=1000, frame_index=3)
    wait2 = bank.assign(track_id=2, feature=feature, confidence=0.9, area=1000, frame_index=4)
    handed_off = bank.assign(track_id=2, feature=feature, confidence=0.9, area=1000, frame_index=5)
    state = bank.debug_state()
    if wait1 != 0 or wait2 != 0:
        raise AssertionError(f"handoff must remain unassigned until three consecutive frames, got {wait1}, {wait2}")
    if handed_off != uid:
        raise AssertionError(f"confirmed handoff should preserve uid {uid}, got {handed_off}")
    if state["track_to_uid"] != {"2": uid}:
        raise AssertionError(f"handoff should atomically transfer uid to the new raw track, got {state['track_to_uid']}")
    if state["last_assignments"]["2"]["reason"] != "controlled_handoff":
        raise AssertionError(f"expected controlled_handoff reason, got {state['last_assignments']['2']}")

    late_bank = IdentityBank(
        IdentityBankConfig(
            match_threshold=0.32,
            min_confidence=0.65,
            new_identity_confirm_frames=1,
            exclusive_uid_claim_enable=True,
            exclusive_uid_claim_frames=2,
            controlled_handoff_enable=True,
            controlled_handoff_confirm_frames=3,
            controlled_handoff_threshold=0.30,
            controlled_handoff_min_old_track_gap_frames=2,
        )
    )
    late_uid = late_bank.assign(track_id=10, feature=feature, confidence=0.9, area=1000, frame_index=1)
    late_wait = late_bank.assign(track_id=20, feature=feature, confidence=0.9, area=1000, frame_index=8)
    if late_wait != 0:
        raise AssertionError(f"a late direct match must still wait for controlled confirmation, got {late_wait}")
    late_bank.assign(track_id=20, feature=feature, confidence=0.9, area=1000, frame_index=9)
    late_handoff = late_bank.assign(track_id=20, feature=feature, confidence=0.9, area=1000, frame_index=10)
    if late_handoff != late_uid:
        raise AssertionError(f"late confirmed match should preserve uid {late_uid}, got {late_handoff}")


def _preferred_search_bank(target_distance: float, best_distance: float):
    bank = IdentityBank(
        IdentityBankConfig(
            match_threshold=0.10,
            reacquire_threshold=0.10,
            update_threshold=0.05,
            min_confidence=0.65,
            new_identity_confirm_frames=1,
            exclusive_uid_claim_enable=True,
            exclusive_uid_claim_frames=15,
            controlled_handoff_enable=True,
            controlled_handoff_confirm_frames=2,
            controlled_handoff_threshold=0.38,
            controlled_handoff_min_old_track_gap_frames=2,
            preferred_search_reacquire_enable=True,
            preferred_search_reacquire_threshold=0.36,
            preferred_search_reacquire_max_disadvantage=0.15,
            preferred_search_reacquire_confirm_frames=2,
        )
    )
    preferred_uid = bank.assign(
        track_id=1,
        feature=_x_axis_feature_at_distance(target_distance),
        confidence=0.9,
        area=1000,
        frame_index=1,
    )
    best_uid = bank.assign(
        track_id=2,
        feature=_z_plane_feature_at_distance(best_distance),
        confidence=0.9,
        area=1000,
        frame_index=2,
    )
    if preferred_uid <= 0 or best_uid <= 0 or preferred_uid == best_uid:
        raise AssertionError(f"expected two seeded identities, got {preferred_uid}, {best_uid}")
    return bank, preferred_uid, best_uid


def _assert_preferred_search_reacquire_rules() -> None:
    query = _unit([1.0, 0.0, 0.0])
    instant_bank, instant_uid, _ = _preferred_search_bank(0.25, 0.21)
    instant = instant_bank.assign(
        track_id=3,
        feature=query,
        confidence=0.9,
        area=1000,
        frame_index=5,
        candidate_count=1,
        preferred_uid=instant_uid,
        preferred_candidate_ok=True,
    )
    if instant != instant_uid:
        raise AssertionError(
            f"a unique strong search match must recover in one frame, got {instant}"
        )

    bank, preferred_uid, _ = _preferred_search_bank(0.35, 0.21)
    wait = bank.assign(
        track_id=3,
        feature=query,
        confidence=0.9,
        area=1000,
        frame_index=5,
        candidate_count=2,
        preferred_uid=preferred_uid,
        preferred_candidate_ok=True,
    )
    recovered = bank.assign(
        track_id=3,
        feature=query,
        confidence=0.9,
        area=1000,
        frame_index=6,
        candidate_count=2,
        preferred_uid=preferred_uid,
        preferred_candidate_ok=True,
    )
    state = bank.debug_state()
    if wait != 0 or recovered != preferred_uid:
        raise AssertionError(f"preferred search reacquire must wait two frames, got {wait}, {recovered}")
    if state["last_assignments"]["3"]["reason"] != "preferred_search_reacquire":
        raise AssertionError(f"unexpected preferred reacquire result: {state['last_assignments']['3']}")

    wrong_side, wrong_uid, _ = _preferred_search_bank(0.35, 0.21)
    wrong_side.assign(
        track_id=3,
        feature=query,
        confidence=0.9,
        area=1000,
        frame_index=5,
        preferred_uid=wrong_uid,
        preferred_candidate_ok=False,
    )
    wrong_result = wrong_side.assign(
        track_id=3,
        feature=query,
        confidence=0.9,
        area=1000,
        frame_index=6,
        preferred_uid=wrong_uid,
        preferred_candidate_ok=False,
    )
    if wrong_result == wrong_uid:
        raise AssertionError("a candidate from the wrong search direction must not recover the preferred uid")

    too_far, far_uid, _ = _preferred_search_bank(0.37, 0.21)
    too_far.assign(
        track_id=3,
        feature=query,
        confidence=0.9,
        area=1000,
        frame_index=5,
        preferred_uid=far_uid,
        preferred_candidate_ok=True,
    )
    far_result = too_far.assign(
        track_id=3,
        feature=query,
        confidence=0.9,
        area=1000,
        frame_index=6,
        preferred_uid=far_uid,
        preferred_candidate_ok=True,
    )
    if far_result == far_uid:
        raise AssertionError("preferred uid distance over 0.36 must be rejected")

    strict_bank = IdentityBank(
        IdentityBankConfig(
            match_threshold=0.38,
            reacquire_threshold=0.38,
            min_confidence=0.65,
            new_identity_confirm_frames=1,
            exclusive_uid_claim_enable=True,
            exclusive_uid_claim_frames=15,
            controlled_handoff_enable=True,
            controlled_handoff_confirm_frames=2,
            controlled_handoff_threshold=0.38,
            controlled_handoff_min_old_track_gap_frames=2,
            preferred_search_reacquire_enable=True,
            preferred_search_reacquire_threshold=0.36,
            preferred_search_reacquire_max_disadvantage=0.15,
            preferred_search_reacquire_confirm_frames=2,
        )
    )
    strict_uid = strict_bank.assign(
        track_id=1,
        feature=_x_axis_feature_at_distance(0.37),
        confidence=0.9,
        area=1000,
        frame_index=1,
    )
    strict_result = strict_bank.assign(
        track_id=3,
        feature=query,
        confidence=0.9,
        area=1000,
        frame_index=5,
        preferred_uid=strict_uid,
        preferred_candidate_ok=True,
    )
    strict_state = strict_bank.debug_state()
    if strict_result != 0:
        raise AssertionError("preferred distance limit must not fall back to the looser normal match threshold")
    if strict_state["last_assignments"]["3"]["reason"] != "preferred_search_reacquire_rejected":
        raise AssertionError(f"expected a hard preferred-search rejection, got {strict_state}")

    weak, weak_uid, _ = _preferred_search_bank(0.35, 0.19)
    weak.assign(
        track_id=3,
        feature=query,
        confidence=0.9,
        area=1000,
        frame_index=5,
        preferred_uid=weak_uid,
        preferred_candidate_ok=True,
    )
    weak_result = weak.assign(
        track_id=3,
        feature=query,
        confidence=0.9,
        area=1000,
        frame_index=6,
        preferred_uid=weak_uid,
        preferred_candidate_ok=True,
    )
    if weak_result == weak_uid:
        raise AssertionError("preferred uid more than 0.15 behind the best identity must be rejected")


def _assert_gallery_keeps_diverse_templates() -> None:
    entry = IdentityEntry(uid=1)
    anchor = _unit([1.0, 0.0, 0.0])
    entry.add(anchor, 1, 3, 0.02, 0.01)
    changed = entry.add(anchor, 2, 3, 0.02, 0.01)
    if changed or len(entry.features) != 1 or entry.duplicate_skip_count != 1:
        raise AssertionError(f"duplicate feature should not consume gallery capacity: {entry}")

    entry.add(_unit([0.8, 0.6, 0.0]), 3, 3, 0.02, 0.01)
    entry.add(_unit([0.6, 0.8, 0.0]), 4, 3, 0.02, 0.01)
    novel = _unit([0.0, 0.0, 1.0])
    changed = entry.add(novel, 5, 3, 0.02, 0.01)
    if not changed or len(entry.features) != 3 or entry.diversity_replace_count != 1:
        raise AssertionError(f"novel view should replace a redundant non-anchor template: {entry}")
    if float(np.dot(entry.features[0], anchor)) < 0.999:
        raise AssertionError("the first stable ReID anchor must never be evicted")
    if entry.distance(novel) > 1e-4:
        raise AssertionError("the diverse gallery should retain the novel template")


def _assert_weak_gallery_is_separate_and_control_safe() -> None:
    config = IdentityBankConfig(
        match_threshold=0.32,
        update_threshold=0.30,
        update_interval=1,
        max_features=2,
        max_weak_features=2,
        weak_update_threshold=0.42,
        weak_update_interval=1,
        weak_match_penalty=0.10,
        weak_reacquire_threshold=0.38,
        weak_reacquire_confirm_frames=3,
        weak_quality_weight=0.30,
        min_confidence=0.60,
        min_area=900,
        new_identity_confirm_frames=1,
        exclusive_uid_claim_enable=True,
        exclusive_uid_claim_frames=15,
        controlled_handoff_enable=True,
        controlled_handoff_confirm_frames=2,
        controlled_handoff_threshold=0.38,
        controlled_handoff_min_old_track_gap_frames=2,
        preferred_search_reacquire_enable=True,
        preferred_search_reacquire_threshold=0.36,
        preferred_search_reacquire_max_disadvantage=0.15,
    )
    bank = IdentityBank(config)
    anchor = _unit([1.0, 0.0, 0.0])
    narrow_view = _x_axis_feature_at_distance(0.20)
    uid = bank.assign(
        track_id=1,
        feature=anchor,
        confidence=0.95,
        area=4000,
        frame_index=1,
        sample_metadata={"bbox": [100, 20, 300, 440], "aspect_ratio": 0.48},
    )
    weak_output = bank.assign(
        track_id=1,
        feature=narrow_view,
        confidence=0.90,
        area=3000,
        frame_index=3,
        bbox_quality_ok=False,
        bbox_quality_reason="aspect<0.18,edge_touch>2",
        bbox_quality_tier="weak",
        sample_metadata={"bbox": [0, 20, 50, 440], "aspect_ratio": 0.12},
    )
    entry = bank.identities[uid]
    if weak_output != 0:
        raise AssertionError(f"a weak box must never expose a control uid, got {weak_output}")
    if len(entry.features) != 1 or len(entry.weak_features) != 0:
        raise AssertionError(
            "quality-rejected weak samples may be observed but must not update the identity gallery"
        )

    unknown_output = bank.assign(
        track_id=9,
        feature=narrow_view,
        confidence=0.90,
        area=3000,
        frame_index=4,
        bbox_quality_ok=False,
        bbox_quality_reason="aspect<0.18",
        bbox_quality_tier="weak",
    )
    if unknown_output != 0 or bank.debug_state()["identity_count"] != 1:
        raise AssertionError("an unbound weak box must not create a new identity")

    weak_results = []
    for frame_index in (5, 6, 7):
        weak_results.append(
            bank.assign(
                track_id=2,
                feature=narrow_view,
                confidence=0.90,
                area=3000,
                frame_index=frame_index,
                candidate_count=1,
                bbox_quality_ok=False,
                bbox_quality_reason="aspect<0.18,edge_touch>2",
                bbox_quality_tier="weak",
                preferred_uid=uid,
                preferred_candidate_ok=True,
            )
        )
    if weak_results != [0, 0, 0]:
        raise AssertionError(f"weak confirmation frames must all stay outside control: {weak_results}")
    state = bank.debug_state()
    if state["track_to_uid"] != {"2": uid}:
        raise AssertionError(f"three weak frames should only establish an internal mapping: {state}")
    if state["last_assignments"]["2"]["reason"] != "weak_preferred_reacquire_confirmed":
        raise AssertionError(f"expected weak confirmation reason, got {state['last_assignments']['2']}")
    if len(entry.weak_features) != 0:
        raise AssertionError("weak reacquisition confirmation must not write quality-rejected features")

    recovered = bank.assign(
        track_id=2,
        feature=narrow_view,
        confidence=0.95,
        area=5000,
        frame_index=8,
        bbox_quality_ok=True,
        bbox_quality_tier="strong",
    )
    if recovered != uid:
        raise AssertionError(f"a later strong box should activate the internally confirmed uid {uid}, got {recovered}")

    before_weak_count = len(entry.weak_features)
    rejected = bank.assign(
        track_id=3,
        feature=narrow_view,
        confidence=0.90,
        area=400,
        frame_index=9,
        bbox_quality_ok=False,
        bbox_quality_reason="area<900,width<16,aspect<0.18,area_shrink<0.30",
        bbox_quality_tier="reject",
        preferred_uid=uid,
        preferred_candidate_ok=True,
    )
    if rejected != 0 or len(entry.weak_features) != before_weak_count:
        raise AssertionError("tiny/area-collapse fragments must never enter either identity gallery")


def main() -> int:
    bank = IdentityBank(
        IdentityBankConfig(
            match_threshold=0.20,
            match_margin=0.01,
            update_threshold=0.30,
            update_interval=1,
            max_features=3,
            min_confidence=0.65,
            reacquire_threshold=0.50,
            reacquire_max_frames=10,
            reacquire_margin=0.01,
            new_identity_confirm_frames=2,
        )
    )

    uid1 = bank.assign(track_id=10, feature=_unit([1.0, 0.0, 0.0]), confidence=0.9, area=1000, frame_index=1)
    uid2 = bank.assign(track_id=20, feature=_unit([0.99, 0.01, 0.0]), confidence=0.9, area=1000, frame_index=2)
    uid3 = bank.assign(track_id=30, feature=_unit([0.56, 0.828, 0.0]), confidence=0.9, area=1000, frame_index=4)
    uid4 = bank.assign(
        track_id=40,
        feature=_unit([0.0, 1.0, 0.0]),
        confidence=0.9,
        area=1000,
        frame_index=5,
        candidate_count=2,
    )
    uid5 = bank.assign(
        track_id=40,
        feature=_unit([0.0, 1.0, 0.0]),
        confidence=0.9,
        area=1000,
        frame_index=6,
        candidate_count=2,
    )
    uid6 = bank.assign(track_id=50, feature=_unit([1.0, 0.0, 0.0]), confidence=0.2, area=1000, frame_index=7)
    uid7 = bank.assign(
        track_id=60,
        feature=_unit([1.0, 0.0, 0.0]),
        confidence=0.9,
        area=1000,
        frame_index=8,
        bbox_quality_ok=False,
        bbox_quality_reason="area_ratio>0.55",
    )
    print("uids", uid1, uid2, uid3, uid4, uid5, uid6, uid7)
    state = bank.debug_state()
    print("state", state)

    if uid1 != 1:
        raise AssertionError(f"first uid should be 1, got {uid1}")
    if uid2 != uid1:
        raise AssertionError(f"similar feature should match uid {uid1}, got {uid2}")
    if uid3 != uid1:
        raise AssertionError(f"recent single-candidate reacquire should keep uid {uid1}, got {uid3}")
    if state["last_assignments"]["30"]["reason"] != "reacquired":
        raise AssertionError(f"expected reacquired reason, got {state['last_assignments']['30']}")
    if uid4 != 0:
        raise AssertionError(f"first unmatched new track should be pending, got {uid4}")
    if uid5 == uid1 or uid5 <= 0:
        raise AssertionError(f"confirmed new track should create a different uid, got {uid5}")
    if uid6 != 0:
        raise AssertionError(f"low-confidence new track should stay unassigned, got {uid6}")
    if uid7 != 0:
        raise AssertionError(f"bad bbox-quality track should stay unassigned, got {uid7}")
    if state["last_assignments"]["60"]["reason"] != "bbox_quality_reject":
        raise AssertionError(f"expected bbox quality rejection, got {state['last_assignments']['60']}")
    same_distance = bank.distance_to_uid(uid1, _unit([1.0, 0.0, 0.0]))
    other_distance = bank.distance_to_uid(uid1, _unit([0.0, 1.0, 0.0]))
    missing_distance = bank.distance_to_uid(999, _unit([1.0, 0.0, 0.0]))
    if same_distance is None or same_distance > 1e-4:
        raise AssertionError(f"expected tiny same-uid distance, got {same_distance}")
    if other_distance is None or other_distance < 0.9:
        raise AssertionError(f"expected large different-feature distance, got {other_distance}")
    if missing_distance is not None:
        raise AssertionError(f"missing uid should return None, got {missing_distance}")
    if state["identity_count"] != 2:
        raise AssertionError(f"expected 2 identities, got {state['identity_count']}")
    uid1_state = next(item for item in state["identities"] if item["uid"] == uid1)
    if uid1_state["last_frame"] != 2 or uid1_state["last_seen_frame"] != 4:
        raise AssertionError(f"expected seen time to advance without feature update, got {uid1_state}")
    _assert_multi_candidate_reacquire_rules()
    _assert_reacquire_uses_last_seen_frame()
    _assert_exclusive_claim_and_mapped_quality_guards()
    _assert_controlled_handoff_requires_consecutive_matches()
    _assert_preferred_search_reacquire_rules()
    _assert_gallery_keeps_diverse_templates()
    _assert_weak_gallery_is_separate_and_control_safe()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
