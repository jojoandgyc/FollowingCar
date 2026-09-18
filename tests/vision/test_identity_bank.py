#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import logging
import math
import sys
from pathlib import Path
from unittest.mock import patch

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
            controlled_handoff_instant_threshold=0.0,
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


def _assert_center_jump_releases_stale_claim() -> None:
    bank = IdentityBank(
        IdentityBankConfig(
            min_confidence=0.65,
            new_identity_confirm_frames=1,
            controlled_handoff_enable=True,
            controlled_handoff_confirm_frames=1,
            controlled_handoff_threshold=0.30,
            controlled_handoff_min_old_track_gap_frames=2,
        )
    )
    feature = _unit([1.0, 0.0, 0.0])
    uid = bank.assign(track_id=1, feature=feature, confidence=0.9, area=1000, frame_index=1)
    rejected = bank.assign(
        track_id=1,
        feature=feature,
        confidence=0.9,
        area=1000,
        frame_index=2,
        bbox_quality_ok=False,
        bbox_quality_reason="identity_center_jump>0.30",
        bbox_quality_tier="reject",
    )
    if rejected != 0:
        raise AssertionError(f"center-jumped mapped track must be withheld, got {rejected}")
    state = bank.debug_state()
    if state["track_to_uid"].get("1") is not None:
        raise AssertionError(f"center-jumped track must release stale UID claim, got {state}")
    handed_off = bank.assign(
        track_id=2,
        feature=feature,
        confidence=0.9,
        area=1000,
        frame_index=3,
    )
    if handed_off != 0:
        raise AssertionError("a released UID without geometric reference must still wait")
    handed_off = bank.assign(
        track_id=2, feature=feature, confidence=0.9, area=1000, frame_index=4,
    )
    if handed_off != uid:
        raise AssertionError(f"released UID should be eligible after confirmation, got {handed_off}")
    swapped = bank.assign(
        track_id=3,
        feature=feature,
        confidence=0.9,
        area=1000,
        frame_index=5,
        bbox_quality_ok=False,
        bbox_quality_reason="identity_swap_competing_track",
        bbox_quality_tier="reject",
    )
    if swapped != 0:
        raise AssertionError(f"competing-track swap evidence must stay unassigned, got {swapped}")

    late_bank = IdentityBank(
        IdentityBankConfig(
            match_threshold=0.32,
            min_confidence=0.65,
            new_identity_confirm_frames=1,
            exclusive_uid_claim_enable=True,
            exclusive_uid_claim_frames=2,
            controlled_handoff_enable=True,
            controlled_handoff_confirm_frames=3,
            controlled_handoff_instant_threshold=0.0,
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
            preferred_search_reacquire_confirm_frames=3,
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
    instant_strong, instant_strong_uid, _ = _preferred_search_bank(0.10, 0.21)
    instant_strong_result = instant_strong.assign(
        track_id=3,
        feature=query,
        confidence=0.9,
        area=1000,
        frame_index=5,
        candidate_count=1,
        preferred_uid=instant_strong_uid,
        preferred_candidate_ok=True,
    )
    if instant_strong_result != 0:
        raise AssertionError(
            "preferred search distance <= 0.15 without geometry must not be instant, "
            f"got {instant_strong_result}"
        )
    for frame_index in (6, 7):
        instant_strong_result = instant_strong.assign(
            track_id=3, feature=query, confidence=0.9, area=1000,
            frame_index=frame_index, preferred_uid=instant_strong_uid,
            preferred_candidate_ok=True,
        )
    if instant_strong_result != 0:
        raise AssertionError("missing geometry must never bind a preferred search UID")
    if instant_strong.last_assignments[3]["instant_reacquire_allowed"]:
        raise AssertionError("ordinary confirmation must not advertise instant permission")

    instant_bank, instant_uid, _ = _preferred_search_bank(0.25, 0.21)
    instant_wait = instant_bank.assign(
        track_id=3,
        feature=query,
        confidence=0.9,
        area=1000,
        frame_index=5,
        candidate_count=1,
        preferred_uid=instant_uid,
        preferred_candidate_ok=True,
    )
    instant_second = instant_bank.assign(
        track_id=3,
        feature=query,
        confidence=0.9,
        area=1000,
        frame_index=6,
        candidate_count=1,
        preferred_uid=instant_uid,
        preferred_candidate_ok=True,
    )
    instant_recovered = instant_bank.assign(
        track_id=3,
        feature=query,
        confidence=0.9,
        area=1000,
        frame_index=7,
        candidate_count=1,
        preferred_uid=instant_uid,
        preferred_candidate_ok=True,
    )
    if instant_wait != 0 or instant_second != 0 or instant_recovered != 0:
        raise AssertionError(
            "a unique strong search match without geometry must remain unassigned, "
            f"got {instant_wait}, {instant_second}, {instant_recovered}"
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
        sample_metadata={"candidate_score_gap": 0.40},
    )
    recovered_wait = bank.assign(
        track_id=3,
        feature=query,
        confidence=0.9,
        area=1000,
        frame_index=6,
        candidate_count=2,
        preferred_uid=preferred_uid,
        preferred_candidate_ok=True,
        sample_metadata={"candidate_score_gap": 0.40},
    )
    recovered = bank.assign(
        track_id=3,
        feature=query,
        confidence=0.9,
        area=1000,
        frame_index=7,
        candidate_count=2,
        preferred_uid=preferred_uid,
        preferred_candidate_ok=True,
        sample_metadata={"candidate_score_gap": 0.40},
    )
    state = bank.debug_state()
    if wait != 0 or recovered_wait != 0 or recovered != 0:
        raise AssertionError(
            "preferred search reacquire without geometry must remain unassigned, "
            f"got {wait}, {recovered_wait}, {recovered}"
        )
    if state["last_assignments"]["3"]["reason"] != "preferred_search_reacquire_geometry_reject":
        raise AssertionError(f"unexpected preferred reacquire result: {state['last_assignments']['3']}")

    # Search reacquisition must survive a one-frame DeepSORT raw-track
    # replacement.  The spatial continuity check prevents an unrelated
    # person from inheriting the pending confirmation streak.
    bridged, bridged_uid, _ = _preferred_search_bank(0.35, 0.21)
    bridged_first = bridged.assign(
        track_id=3,
        feature=query,
        confidence=0.9,
        area=1000,
        frame_index=5,
        candidate_count=1,
        preferred_uid=bridged_uid,
        preferred_candidate_ok=True,
        sample_metadata={"center_x_ratio": 0.78},
    )
    bridged_weak = bridged.assign(
        track_id=3,
        feature=query,
        confidence=0.9,
        area=1000,
        frame_index=6,
        candidate_count=1,
        preferred_uid=bridged_uid,
        preferred_candidate_ok=True,
        bbox_quality_ok=True,
        bbox_quality_tier="weak",
        bbox_quality_reason="edge_touch=1",
        sample_metadata={"center_x_ratio": 0.79},
    )
    bridged_wait = bridged.assign(
        track_id=4,
        feature=query,
        confidence=0.9,
        area=1000,
        frame_index=7,
        candidate_count=1,
        preferred_uid=bridged_uid,
        preferred_candidate_ok=True,
        sample_metadata={"center_x_ratio": 0.80},
    )
    bridged_recovered = bridged.assign(
        track_id=4,
        feature=query,
        confidence=0.9,
        area=1000,
        frame_index=8,
        candidate_count=1,
        preferred_uid=bridged_uid,
        preferred_candidate_ok=True,
        sample_metadata={"center_x_ratio": 0.81},
    )
    if bridged_first != 0 or bridged_weak != 0 or bridged_wait != 0 or bridged_recovered != 0:
        raise AssertionError(
            "preferred search reacquire without positive geometry must remain unassigned, "
            f"got {bridged_first}, {bridged_weak}, {bridged_wait}, {bridged_recovered}"
        )

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
            preferred_search_reacquire_confirm_frames=3,
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


def _assert_search_candidate_competition_uses_confidence_gap() -> None:
    """Low-confidence detector fragments must not block a strong locked UID."""
    bank = IdentityBank(
        IdentityBankConfig(
            new_identity_confirm_frames=1,
            preferred_search_reacquire_enable=True,
            preferred_search_reacquire_min_score_gap=0.25,
        )
    )
    feature = _unit([1.0, 0.0, 0.0])
    uid = bank.assign(
        track_id=1,
        feature=feature,
        confidence=0.9,
        area=1000,
        frame_index=1,
    )
    accepted = bank._preferred_search_reacquire_candidate(
        feature=feature,
        partial_feature=None,
        preferred_uid=uid,
        candidate_ok=True,
        candidate_count=2,
        sample_metadata={"candidate_score_gap": 0.60},
    )
    if accepted is None:
        raise AssertionError("a clearly stronger candidate should pass competition gating")
    rejected = bank._preferred_search_reacquire_candidate(
        feature=feature,
        partial_feature=None,
        preferred_uid=uid,
        candidate_ok=True,
        candidate_count=2,
        sample_metadata={"candidate_score_gap": 0.08},
    )
    if rejected is not None:
        raise AssertionError("close-confidence candidates must remain gated")

    opposite_override = bank._preferred_search_reacquire_candidate(
        feature=feature,
        partial_feature=None,
        preferred_uid=uid,
        candidate_ok=False,
        candidate_count=2,
        sample_metadata={
            "candidate_score_gap": 0.08,
            "bbox_quality_tier": "strong",
            "search_reacquire_context_active": True,
            "search_direction_compatible": False,
        },
    )
    if opposite_override is None or opposite_override[-1] != "strong":
        raise AssertionError(
            "a strong opposite-side active-UID match should enter local confirmation"
        )


def _assert_preferred_search_blocks_global_fallback() -> None:
    """A search candidate must not inherit a UID through normal handoff."""
    bank = IdentityBank(
        IdentityBankConfig(
            match_threshold=0.30,
            reacquire_threshold=0.30,
            new_identity_confirm_frames=1,
            controlled_handoff_enable=True,
            controlled_handoff_confirm_frames=2,
            controlled_handoff_threshold=0.30,
            controlled_handoff_min_old_track_gap_frames=1,
            preferred_search_reacquire_enable=True,
            preferred_search_reacquire_threshold=0.10,
            preferred_search_reacquire_max_disadvantage=0.05,
            preferred_search_reacquire_confirm_frames=2,
        )
    )
    preferred_uid = bank.assign(
        track_id=1,
        feature=_unit([1.0, 0.0, 0.0]),
        confidence=0.9,
        area=1000,
        frame_index=1,
    )
    result = bank.assign(
        track_id=2,
        feature=_unit([1.0, 0.0, 0.0]),
        confidence=0.9,
        area=1000,
        frame_index=5,
        candidate_count=2,
        preferred_uid=preferred_uid,
        preferred_candidate_ok=False,
    )
    state = bank.debug_state()
    if result != 0:
        raise AssertionError(
            "active preferred search must reject a global controlled handoff, "
            f"got uid={result}, state={state}"
        )
    if state["last_assignments"]["2"]["reason"] != "preferred_search_reacquire_rejected":
        raise AssertionError(f"expected preferred-search rejection, got {state}")


def _assert_partial_search_reacquire_uses_torso_descriptor() -> None:
    """A partial view may use its separate torso descriptor, never the full gallery threshold."""
    bank = IdentityBank(
        IdentityBankConfig(
            match_threshold=0.20,
            update_threshold=0.20,
            min_confidence=0.65,
            new_identity_confirm_frames=1,
            controlled_handoff_enable=False,
            preferred_search_reacquire_enable=True,
            preferred_search_reacquire_threshold=0.20,
            partial_appearance_enable=True,
            partial_match_threshold=0.34,
        )
    )
    uid = bank.assign(
        track_id=1,
        feature=_unit([1.0, 0.0, 0.0]),
        partial_feature=_unit([1.0, 0.0, 0.0]),
        confidence=0.9,
        area=1000,
        frame_index=1,
    )
    candidate = bank.assign(
        track_id=2,
        feature=_unit([0.0, 1.0, 0.0]),
        # Above the full-body 0.20 threshold but inside the partial torso
        # threshold, reproducing a clipped-person observation.
        partial_feature=_unit([0.76, 0.65, 0.0]),
        confidence=0.9,
        area=1000,
        frame_index=2,
        candidate_count=1,
        preferred_uid=uid,
        preferred_candidate_ok=True,
        sample_metadata={"partial_observation": True},
    )
    if candidate != uid:
        raise AssertionError(f"partial search candidate should keep uid {uid}, got {candidate}")
    assignment = bank.last_assignments[2]
    if assignment["reason"] != "preferred_search_reacquire":
        raise AssertionError(f"expected partial preferred handoff, got {assignment}")
    if assignment["match_source"] != "partial":
        raise AssertionError(f"expected partial match source, got {assignment}")
    if assignment["bank_updated"]:
        raise AssertionError("partial handoff must not update the full-body gallery")


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


def _geometry_metadata(track_id, bbox, frame_index=1, detector_bbox=None, capture_timestamp=None):
    x1, y1, x2, y2 = bbox
    metadata = {
        "track_id": track_id,
        "bbox": list(bbox),
        "center_x_ratio": (x1 + x2) / 1280.0,
        "area_ratio": (x2 - x1) * (y2 - y1) / (640.0 * 480.0),
        "control_frame_id": frame_index,
        "capture_frame_id": frame_index * 2,
        "is_fresh": True,
    }
    if capture_timestamp is not None:
        metadata["capture_timestamp"] = float(capture_timestamp)
    if detector_bbox is not None:
        x1, y1, x2, y2 = detector_bbox
        metadata.update({
            "detector_bbox": list(detector_bbox),
            "detector_center_x_ratio": (x1 + x2) / 1280.0,
            "detector_area_ratio": (x2 - x1) * (y2 - y1) / (640.0 * 480.0),
        })
    return metadata


def _geometry_bank(**overrides):
    config = {
        "min_confidence": 0.60,
        "update_interval": 1,
        "controlled_handoff_enable": True,
        "controlled_handoff_min_old_track_gap_frames": 1,
        "controlled_handoff_confirm_frames": 2,
        "preferred_search_reacquire_confirm_frames": 2,
    }
    config.update(overrides)
    return IdentityBank(IdentityBankConfig(**config))


def _assert_handoff_geometry_rejects_implausible_reentry() -> None:
    anchor = _unit([1.0, 0.0, 0.0])
    old_bbox = [268.0, 0.0, 588.2, 479.0]
    new_bbox = [380.5, 187.1, 458.3, 401.5]
    bank = _geometry_bank()
    uid = bank.assign(
        track_id=79, feature=anchor, confidence=0.844, area=153375,
        frame_index=867, sample_metadata=_geometry_metadata(79, old_bbox, 867),
    )
    reference = dict(bank.identities[uid].last_strong_observation)
    for frame_index in (874, 875, 876):
        result = bank.assign(
            track_id=78, feature=_x_axis_feature_at_distance(0.037),
            confidence=0.622, area=16670, frame_index=frame_index,
            preferred_uid=uid, preferred_candidate_ok=True,
            sample_metadata=_geometry_metadata(78, new_bbox, frame_index),
        )
        assignment = bank.last_assignments[78]
        if result != 0 or assignment["reason"] != "handoff_geometry_reject":
            raise AssertionError(f"the small .037 candidate must not inherit U1: {assignment}")
        if assignment["reacquire_geometry_ok"] is not False or assignment["instant_reacquire_allowed"]:
            raise AssertionError("a short-gap area collapse must remain a hard rejection")
        if bank.identities[uid].last_strong_observation != reference:
            raise AssertionError("rejected observations must not become the next geometric reference")
        if len(bank.identities[uid].features) != 1 or 78 in bank.pending_handoffs:
            raise AssertionError("a repeated rejected box must neither update the gallery nor build a streak")
        evidence = assignment["match_evidence"]
        if evidence["winner"]["metadata"]["track_id"] != 79:
            raise AssertionError("the rejected candidate must retain the actual matched template provenance")

    # A weak-tier internal mapping must not bypass the same strong handoff guard.
    bank.track_to_uid[78] = uid
    result = bank.assign(
        track_id=78, feature=anchor, confidence=0.9, area=16670, frame_index=877,
        preferred_uid=uid, preferred_candidate_ok=True,
        sample_metadata=_geometry_metadata(78, new_bbox, 877),
    )
    if result != 0 or bank.identities[uid].last_strong_observation != reference:
        raise AssertionError("a weak-only mapping cannot turn a geometric rejection into a strong observation")

    jump_bank = _geometry_bank()
    jump_uid = jump_bank.assign(
        track_id=1, feature=anchor, confidence=0.9, area=10000, frame_index=1,
        sample_metadata=_geometry_metadata(1, [50, 100, 150, 300]),
    )
    jumped = jump_bank.assign(
        track_id=2, feature=anchor, confidence=0.9, area=10000, frame_index=5,
        preferred_uid=jump_uid, preferred_candidate_ok=True,
        sample_metadata=_geometry_metadata(2, [480, 100, 580, 300], 5),
    )
    if jumped != 0 or jump_bank.last_assignments[2]["reacquire_geometry_reason"] != "center_jump":
        raise AssertionError("a strong-looking query on the other side must not bypass UID geometry")


def _assert_instant_handoff_requires_trusted_geometry() -> None:
    anchor = _unit([1.0, 0.0, 0.0])
    bbox = [200, 40, 400, 440]
    bank = _geometry_bank()
    seed_metadata = _geometry_metadata(1, bbox, 1)
    uid = bank.assign(
        track_id=1, feature=anchor, confidence=0.9, area=80000,
        frame_index=1, sample_metadata=seed_metadata,
    )
    result = bank.assign(
        track_id=2, feature=_x_axis_feature_at_distance(0.037), confidence=0.9,
        area=78000, frame_index=5, preferred_uid=uid, preferred_candidate_ok=True,
        sample_metadata=_geometry_metadata(2, [205, 45, 405, 435], 5),
    )
    assignment = bank.last_assignments[2]
    if result != uid or not assignment["instant_reacquire_allowed"]:
        raise AssertionError(f"a recent continuous strong observation should allow instant identity handoff: {assignment}")
    if assignment["reacquire_geometry_ok"] is not True:
        raise AssertionError("instant permission must include explicit positive geometry evidence")
    evidence = assignment["match_evidence"]
    if len(bank.identities[uid].features) != 1 or assignment["bank_updated"]:
        raise AssertionError("successful identity handoff must not immediately learn the reacquired crop")
    if not assignment["template_update_quarantined"]:
        raise AssertionError("instant identity confirmation still requires template quarantine")
    if evidence["winner"]["metadata"]["frame_index"] != 1 or evidence["winner"]["index"] != 0:
        raise AssertionError("winning evidence must refer to the frozen, pre-handoff gallery")
    if abs(evidence["anchor_distance"] - 0.037) > 1e-5:
        raise AssertionError("anchor distance must describe the pre-update initial template")
    if evidence["anchor_metadata"]["control_frame_id"] != 1:
        raise AssertionError("the initial anchor must carry its image provenance")
    seed_metadata["bbox"][0] = 9999
    if evidence["winner"]["metadata"]["bbox"][0] != 200:
        raise AssertionError("evidence must not alias mutable sample metadata")

    for scenario in ("missing_reference", "missing_current"):
        fallback = _geometry_bank(
            controlled_handoff_confirm_frames=1,
            preferred_search_reacquire_confirm_frames=1,
        )
        fallback_uid = fallback.assign(
            track_id=1, feature=anchor, confidence=0.9, area=80000, frame_index=1,
            sample_metadata=None if scenario == "missing_reference" else _geometry_metadata(1, bbox),
        )
        first_frame = 20 if scenario == "stale_reference" else 5
        first = fallback.assign(
            track_id=2, feature=anchor, confidence=0.9, area=80000,
            frame_index=first_frame, preferred_uid=fallback_uid, preferred_candidate_ok=True,
            sample_metadata=None if scenario == "missing_current" else _geometry_metadata(2, bbox, first_frame),
        )
        if first != 0 or fallback.last_assignments[2]["reacquire_geometry_ok"] is not None:
            raise AssertionError(f"{scenario} must disable single-frame handoff")
        second = fallback.assign(
            track_id=2, feature=anchor, confidence=0.9, area=80000,
            frame_index=first_frame + 1, preferred_uid=fallback_uid, preferred_candidate_ok=True,
            sample_metadata=None if scenario == "missing_current" else _geometry_metadata(2, bbox, first_frame + 1),
        )
        if second != 0 or fallback.last_assignments[2]["reason"] != "preferred_search_reacquire_geometry_reject":
            raise AssertionError(f"{scenario} must never bind a preferred search UID without positive geometry")

    # Once the original geometry reference is stale, a strong sole candidate
    # enters an independent local observation chain.  It binds only on the
    # second consecutive frame and promotes that frame to the new reference.
    late = _geometry_bank(
        preferred_search_reacquire_confirm_frames=1,
        preferred_search_reacquire_late_candidate_enable=True,
    )
    late_uid = late.assign(
        track_id=1, feature=anchor, confidence=0.9, area=80000, frame_index=1,
        sample_metadata=_geometry_metadata(1, bbox, 1, capture_timestamp=1.0),
    )
    late_first = late.assign(
        track_id=2, feature=anchor, confidence=0.9, area=80000, frame_index=20,
        preferred_uid=late_uid, preferred_candidate_ok=True,
        sample_metadata=_geometry_metadata(2, bbox, 20, capture_timestamp=10.0),
    )
    if late_first != 0 or late.last_assignments[2]["reason"] != "preferred_search_late_candidate_wait":
        raise AssertionError(f"stale candidate must start a late observation chain: {late.last_assignments[2]}")
    late_second = late.assign(
        track_id=2, feature=anchor, confidence=0.9, area=80000, frame_index=21,
        preferred_uid=late_uid, preferred_candidate_ok=True,
        sample_metadata=_geometry_metadata(2, bbox, 21, capture_timestamp=10.1),
    )
    if late_second != late_uid or late.last_assignments[2]["reason"] != "preferred_search_late_reacquire":
        raise AssertionError(f"two local observations should bind the preferred UID: {late.last_assignments[2]}")
    if late.last_assignments[2]["reacquire_geometry_reason"] != "late_candidate_local_continuity":
        raise AssertionError("late handoff must report local continuity evidence")
    if late.identities[late_uid].last_strong_observation["frame_index"] != 21:
        raise AssertionError("late handoff must promote the confirmed candidate to the geometry reference")

    # A skipped frame cannot complete the chain, and multiple candidates remain
    # ambiguous even if their appearance matches the preferred UID.
    late_gap = _geometry_bank(preferred_search_reacquire_late_candidate_enable=True)
    gap_uid = late_gap.assign(
        track_id=1, feature=anchor, confidence=0.9, area=80000, frame_index=1,
        sample_metadata=_geometry_metadata(1, bbox, 1),
    )
    gap_first = late_gap.assign(
        track_id=2, feature=anchor, confidence=0.9, area=80000, frame_index=20,
        preferred_uid=gap_uid, preferred_candidate_ok=True,
        sample_metadata=_geometry_metadata(2, bbox, 20),
    )
    gap_reset = late_gap.assign(
        track_id=2, feature=anchor, confidence=0.9, area=80000, frame_index=22,
        preferred_uid=gap_uid, preferred_candidate_ok=True,
        sample_metadata=_geometry_metadata(2, bbox, 22),
    )
    if gap_first != 0 or gap_reset != 0 or 2 in late_gap.track_to_uid:
        raise AssertionError("late confirmation must require adjacent frames")


def _assert_opposite_side_strong_search_candidate_uses_local_confirmation() -> None:
    """A strong candidate may cross the frozen sweep side, but still needs two frames."""
    anchor = _unit([1.0, 0.0, 0.0])
    bank = _geometry_bank(
        preferred_search_reacquire_max_age_sec=0.35,
        preferred_search_reacquire_confirm_frames=2,
    )
    uid = bank.assign(
        track_id=1,
        feature=anchor,
        confidence=0.9,
        area=80000,
        frame_index=1,
        sample_metadata=_geometry_metadata(
            1, [100, 40, 300, 440], 1, capture_timestamp=1.0
        ),
    )
    candidate_metadata = {
        **_geometry_metadata(
            2, [500, 40, 700, 440], 20, capture_timestamp=2.0
        ),
        "bbox_quality_tier": "strong",
        "search_reacquire_context_active": True,
        "search_direction_compatible": False,
        "candidate_count": 2,
        "candidate_score_gap": 0.08,
    }
    first = bank.assign(
        track_id=2,
        feature=anchor,
        confidence=0.9,
        area=80000,
        frame_index=20,
        preferred_uid=uid,
        preferred_candidate_ok=False,
        sample_metadata=candidate_metadata,
    )
    if first != 0 or bank.last_assignments[2]["reason"] != "preferred_search_late_candidate_wait":
        raise AssertionError(
            "an opposite-side strong match must start local confirmation, got %s"
            % bank.last_assignments[2]
        )
    candidate_metadata = {
        **_geometry_metadata(
            2, [510, 40, 710, 440], 21, capture_timestamp=2.1
        ),
        "bbox_quality_tier": "strong",
        "search_reacquire_context_active": True,
        "search_direction_compatible": False,
        "candidate_count": 2,
        "candidate_score_gap": 0.08,
    }
    second = bank.assign(
        track_id=2,
        feature=anchor,
        confidence=0.9,
        area=80000,
        frame_index=21,
        preferred_uid=uid,
        preferred_candidate_ok=False,
        sample_metadata=candidate_metadata,
    )
    if second != uid or bank.last_assignments[2]["reason"] != "preferred_search_late_reacquire":
        raise AssertionError(
            "two continuous opposite-side frames should confirm the UID, got %s"
            % bank.last_assignments[2]
        )

    # The side exception must not widen the identity threshold.  A visually
    # weak match remains rejected even when its box is geometrically valid.
    weak = bank.assign(
        track_id=3,
        feature=_x_axis_feature_at_distance(0.28),
        confidence=0.9,
        area=80000,
        frame_index=22,
        preferred_uid=uid,
        preferred_candidate_ok=False,
        sample_metadata={
            **_geometry_metadata(3, [520, 40, 720, 440], 22, capture_timestamp=2.2),
            "bbox_quality_tier": "strong",
            "search_reacquire_context_active": True,
            "search_direction_compatible": False,
        },
    )
    if weak != 0 or bank.last_assignments[3]["reason"] != "preferred_search_reacquire_rejected":
        raise AssertionError(
            "opposite-side candidates above the preferred distance threshold must reject"
        )


def _assert_detector_geometry_and_rejected_reference_integrity() -> None:
    anchor = _unit([1.0, 0.0, 0.0])
    display_bbox = [200, 40, 400, 440]
    bank = _geometry_bank()
    uid = bank.assign(
        track_id=1, feature=anchor, confidence=0.9, area=80000, frame_index=1,
        sample_metadata=_geometry_metadata(1, display_bbox, 1, [210, 50, 390, 430]),
    )
    result = bank.assign(
        track_id=2, feature=anchor, confidence=0.9, area=80000, frame_index=5,
        preferred_uid=uid, preferred_candidate_ok=True,
        sample_metadata=_geometry_metadata(2, display_bbox, 5, [275, 200, 325, 300]),
    )
    if result != 0 or bank.last_assignments[2]["reacquire_geometry_reason"] != "area_change":
        raise AssertionError("an unchanged Kalman display box must not hide a shrunken feature crop")
    if bank.last_assignments[2]["reacquire_geometry"]["reference"]["geometry_source"] != "detector":
        raise AssertionError("the reference must prefer detector geometry whenever available")

    reference = dict(bank.identities[uid].last_strong_observation)
    for frame_index, tier, feature in (
        (6, "weak", anchor), (7, "reject", anchor), (8, "strong", _x_axis_feature_at_distance(0.8)),
    ):
        result = bank.assign(
            track_id=1, feature=feature, confidence=0.9, area=80000, frame_index=frame_index,
            bbox_quality_ok=tier == "strong", bbox_quality_tier=tier,
            bbox_quality_reason="" if tier == "strong" else "edge_touch>2",
            sample_metadata=_geometry_metadata(1, display_bbox, frame_index, [210, 50, 390, 430]),
        )
        if result != 0 or bank.identities[uid].last_strong_observation != reference:
            raise AssertionError("weak/quality/appearance rejections must not refresh the trusted reference")
        if bank.last_assignments[1]["match_evidence"] is None:
            raise AssertionError("mapped rejections must retain their feature-match evidence")

    mapped = _geometry_bank()
    mapped_uid = mapped.assign(
        track_id=1, feature=anchor, confidence=0.9, area=80000, frame_index=1,
        sample_metadata=_geometry_metadata(1, display_bbox),
    )
    trusted_reference = dict(mapped.identities[mapped_uid].last_strong_observation)
    mapped_result = mapped.assign(
        track_id=1, feature=_x_axis_feature_at_distance(0.037), confidence=0.9,
        area=5000, frame_index=5, sample_metadata=_geometry_metadata(1, [275, 200, 325, 300], 5),
    )
    assignment = mapped.last_assignments[1]
    if mapped_result != mapped_uid or assignment["reason"] != "skip_update_geometry":
        raise AssertionError("ordinary mapped output may be retained, but anomalous geometry must block learning")
    if assignment["bank_updated"] or len(mapped.identities[mapped_uid].features) != 1:
        raise AssertionError("a geometry-rejected mapped query must not become a strong template")
    if mapped.identities[mapped_uid].last_strong_observation != trusted_reference:
        raise AssertionError("a geometry-rejected mapped query must not replace the trusted reference")


def _assert_match_evidence_tracks_actual_uid_and_weak_winner() -> None:
    query = _unit([1.0, 0.0, 0.0])
    bank, preferred_uid, best_uid = _preferred_search_bank(0.25, 0.21)
    bank.assign(
        track_id=3, feature=query, confidence=0.9, area=1000, frame_index=5,
        preferred_uid=preferred_uid, preferred_candidate_ok=True,
    )
    assignment = bank.last_assignments[3]
    evidence = assignment["match_evidence"]
    if assignment["best_uid"] != best_uid or evidence["matched_uid"] != preferred_uid:
        raise AssertionError("preferred evidence must describe the selected UID, not the global nearest UID")
    if abs(evidence["anchor_distance"] - 0.25) > 1e-5:
        raise AssertionError("the preferred UID initial-anchor distance is wrong")
    rejected_bank, rejected_uid, nearest_uid = _preferred_search_bank(0.37, 0.21)
    rejected_bank.assign(
        track_id=3, feature=query, confidence=0.9, area=1000, frame_index=5,
        preferred_uid=rejected_uid, preferred_candidate_ok=True,
    )
    rejected = rejected_bank.last_assignments[3]
    if rejected["best_uid"] != nearest_uid or rejected["match_evidence"]["matched_uid"] != rejected_uid:
        raise AssertionError("a rejected preferred candidate still needs the locked UID evidence")
    if abs(rejected["distance"] - 0.37) > 1e-5 or abs(rejected["second_distance"] - 0.21) > 1e-5:
        raise AssertionError("rejected preferred distances must not be mislabeled with the global winner")

    weak_bank = _geometry_bank()
    uid = weak_bank.assign(
        track_id=1, feature=_x_axis_feature_at_distance(0.2), confidence=0.9,
        area=1000, frame_index=1,
    )
    entry = weak_bank.identities[uid]
    entry.add_weak(query, 2, 8, diversity_min_distance=0.0, metadata={"track_id": 10, "quality_weight": 0.0})
    entry.add_weak(
        _x_axis_feature_at_distance(0.01), 3, 8, diversity_min_distance=0.0,
        metadata={"track_id": 11, "quality_weight": 1.0},
    )
    weak_bank.assign(
        track_id=1, feature=query, confidence=0.9, area=1000, frame_index=4,
        bbox_quality_ok=False, bbox_quality_tier="weak", bbox_quality_reason="aspect<0.18",
    )
    evidence = weak_bank.last_assignments[1]["match_evidence"]
    winner = evidence["winner"]
    if winner["tier"] != "weak" or winner["index"] != 1 or winner["metadata"]["track_id"] != 11:
        raise AssertionError("the weak winner must include its quality penalty, not just raw cosine distance")
    if abs(evidence["weak_distance"]) > 1e-5 or abs(evidence["weak_weighted_distance"] - 0.01) > 1e-5:
        raise AssertionError("raw and penalized weak distances must be separately visible")
    if len(evidence["nearest_samples"]) > 3:
        raise AssertionError("match evidence must keep at most three neighbours")

    replacing = _geometry_bank(max_features=2)
    replacing_uid = replacing.assign(
        track_id=1, feature=query, confidence=0.9, area=1000, frame_index=1,
    )
    replacing.assign(
        track_id=1, feature=_x_axis_feature_at_distance(0.05), confidence=0.9,
        area=1000, frame_index=2,
    )
    replacing.assign(
        track_id=1, feature=_x_axis_feature_at_distance(0.20), confidence=0.9,
        area=1000, frame_index=3,
    )
    replaced_evidence = replacing.last_assignments[1]["match_evidence"]
    if not replacing.last_assignments[1]["bank_updated"]:
        raise AssertionError("the replacement test must exercise a real gallery change")
    if replacing.identities[replacing_uid].feature_metadata[1]["frame_index"] != 3:
        raise AssertionError("the redundant non-anchor should have been replaced on frame 3")
    if replaced_evidence["winner"]["metadata"]["frame_index"] != 2:
        raise AssertionError("winner metadata must survive replacement of its gallery slot")


def _assert_match_evidence_logging_is_sampled_and_vector_free() -> None:
    bank = _geometry_bank(update_interval=5)
    feature = _unit([1.0, 0.0, 0.0])
    bbox = [200, 40, 400, 440]
    bank.assign(
        track_id=1, feature=feature, confidence=0.9, area=80000, frame_index=1,
        sample_metadata=_geometry_metadata(1, bbox),
    )
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    log = logging.getLogger("PersonTracker")
    previous_level = log.level
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    try:
        for frame_index in (2, 3, 5):
            metadata = _geometry_metadata(1, bbox, frame_index)
            metadata["feature_vector"] = np.arange(512)
            bank.assign(
                track_id=1, feature=feature, confidence=0.9, area=80000,
                frame_index=frame_index, bbox_quality_ok=frame_index != 3,
                bbox_quality_reason="edge_touch>2" if frame_index == 3 else "",
                sample_metadata=metadata,
            )
    finally:
        log.removeHandler(handler)
        log.setLevel(previous_level)
    records = [
        json.loads(line.split("reid_match_evidence ", 1)[1])
        for line in stream.getvalue().splitlines() if "reid_match_evidence " in line
    ]
    if [record["frame_index"] for record in records] != [3, 5]:
        raise AssertionError("ordinary mapped evidence must be sampled, but quality rejection must be logged")
    if "feature_vector" in stream.getvalue() or not all(record["match_evidence"] for record in records):
        raise AssertionError("parseable evidence must include provenance without serializing feature vectors")

    nonfinite = _geometry_metadata(1, bbox, 10)
    nonfinite["capture_timestamp"] = float("nan")
    nonfinite["detector_confidence"] = float("inf")
    result = bank.assign(
        track_id=1, feature=feature, confidence=0.9, area=80000,
        frame_index=10, sample_metadata=nonfinite,
    )
    if result != 1:
        raise AssertionError("non-finite diagnostic metadata must not interrupt a valid assignment")
    with patch("rk_vision.identity_bank.json.dumps", side_effect=TypeError("diagnostic serialization")):
        result = bank.assign(
            track_id=1, feature=feature, confidence=0.9, area=80000,
            frame_index=15, sample_metadata=_geometry_metadata(1, bbox, 15),
        )
    if result != 1:
        raise AssertionError("diagnostic serialization errors must not interrupt the tracking path")


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
    _assert_center_jump_releases_stale_claim()
    _assert_preferred_search_reacquire_rules()
    _assert_search_candidate_competition_uses_confidence_gap()
    _assert_preferred_search_blocks_global_fallback()
    _assert_partial_search_reacquire_uses_torso_descriptor()
    _assert_gallery_keeps_diverse_templates()
    _assert_weak_gallery_is_separate_and_control_safe()
    _assert_handoff_geometry_rejects_implausible_reentry()
    _assert_instant_handoff_requires_trusted_geometry()
    _assert_opposite_side_strong_search_candidate_uses_local_confirmation()
    _assert_detector_geometry_and_rejected_reference_integrity()
    _assert_match_evidence_tracks_actual_uid_and_weak_winner()
    _assert_match_evidence_logging_is_sampled_and_vector_free()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
