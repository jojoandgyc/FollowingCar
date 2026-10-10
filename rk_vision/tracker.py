from __future__ import annotations
from .follow_bbox_policy import lower_compact_bbox_reason

import logging
import math
import time
from numbers import Integral
from dataclasses import dataclass, replace
from typing import Any, List, Optional, Sequence, Set, Tuple

from .deepsort import DeepSort, DeepSortConfig
from .deepsort.track import TrackState
from .deepsort.provisional import ProvisionalAssociationCache, capture_clock
from .similar_follow import cropped_follow_continuous
from .identity_bank import IdentityBank, IdentityBankConfig
from .candidate_competition import competition_evidence
from .competition_eligibility import anchored_competitor_exclusions
from .yolo11 import Detection
from .stage_timing import StageTiming
from .deepsort.detection import Detection as DeepSortDetection
from .detector_continuation import (
    DetectorProof, DetectorBackgroundProof, DetectorContinuationPlan,
    FULL_PROOF_TTL_SEC, MAX_FULL_RESULT_AGE_SEC, separated_observations,
    SIMILAR_MAX_GALLERY_DISTANCE,
    capture_observation, color_signature,
    color_matches, geometry_matches, continuation_reason, finite_number,
)


logger = logging.getLogger(__name__)

TRACK_STATE_NEW = 0
TRACK_STATE_UNSTABLE = 1
TRACK_STATE_STABLE = 2


def _template_learning_risk(detections, source_index, *, is_fresh, all_detections=None):
    """Describe crop mixing for learning only, without running ReID or gating UID.

    A small excluded person may still occlude most of its visible body inside
    the target crop. Intersection/min(area) deliberately detects that case;
    IoU alone misses it. Indices refer to the supplied all-detections snapshot.
    """
    result = dict(observed=False, risky=False, reason="no_fresh_detector_source",
                  overlap=0.0, other_indices=[])
    if (not is_fresh or isinstance(source_index, bool)
            or not isinstance(source_index, Integral)
            or not 0 <= int(source_index) < len(detections)):
        return result

    def person_box(item):
        try:
            values = tuple(float(v) for v in item.bbox)
            if (int(item.class_id) != 0 or len(values) != 4
                    or not all(math.isfinite(v) for v in values)
                    or values[2] <= values[0] or values[3] <= values[1]):
                return None
            return values
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None

    target = detections[int(source_index)]
    box = person_box(target)
    if box is None:
        return result
    candidates = detections if all_detections is None else all_detections
    # An optional broader snapshot must actually contain this observation.
    # Missing/older snapshots must not manufacture an "unoccluded" proof.
    matches = [i for i, item in enumerate(candidates) if item is target]
    if not matches:
        matches = [i for i, item in enumerate(candidates)
                   if person_box(item) == box and item.score == target.score]
    if len(matches) != 1:
        return result
    own_index = matches[0]
    area = (box[2] - box[0]) * (box[3] - box[1])
    result.update(observed=True, reason="clear")
    for index, item in enumerate(candidates):
        if index == own_index:
            continue
        other = person_box(item)
        if other is None:
            continue
        intersection = max(0.0, min(box[2], other[2]) - max(box[0], other[0])) * max(
            0.0, min(box[3], other[3]) - max(box[1], other[1]))
        denominator = min(area, (other[2] - other[0]) * (other[3] - other[1]))
        overlap = intersection / denominator
        result["overlap"] = max(result["overlap"], overlap)
        if overlap >= 0.15:
            result["other_indices"].append(index)
    if result["other_indices"]:
        result.update(risky=True, reason="overlapping_person_crop")
    return result


@dataclass(frozen=True)
class TrackRecord:
    track_id: int
    reid_uid: int
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    score: float
    cx: float
    cy: float
    area: float
    angle_deg: float
    tracker_state: int
    time_since_update: int = 0
    reid_verify_distance: Optional[float] = None
    reid_verify_passed: Optional[bool] = None


@dataclass(frozen=True)
class DeepSortTrackerConfig:
    max_age: int = 20
    max_output_age: int = 1
    n_init: int = 3
    max_iou_distance: float = 0.60
    max_cosine_distance: float = 0.35
    max_bbox_age: int = 2
    nn_budget: int = 15
    feature_update_interval: int = 1
    min_confidence: float = 0.50
    nms_max_overlap: float = 0.50
    bbox_expand_scale: float = 1.20
    hfov_deg: float = 90.0
    identity_bank_enable: bool = True
    identity_match_threshold: float = 0.32
    identity_match_margin: float = 0.03
    identity_update_threshold: float = 0.30
    identity_update_interval: int = 5
    identity_max_features: int = 20
    identity_template_memory_enable: bool = False
    identity_template_crosscheck_enable: bool = False
    identity_template_learning_guard_enable: bool = False
    identity_similar_follow_enable: bool = False
    identity_similar_follow_entry_threshold: float = 0.50
    identity_similar_follow_retain_threshold: float = 0.55
    identity_similar_follow_max_gap_sec: float = 0.50
    identity_appearance_region_safety_enable: bool = False
    identity_template_recent_sec: float = 30.0
    identity_template_archive_sec: float = 120.0
    identity_max_weak_features: int = 8
    identity_diversity_min_distance: float = 0.02
    identity_diversity_replace_margin: float = 0.01
    identity_weak_update_threshold: float = 0.42
    identity_weak_update_interval: int = 3
    identity_weak_match_penalty: float = 0.10
    identity_weak_reacquire_threshold: float = 0.38
    identity_weak_reacquire_confirm_frames: int = 2
    identity_weak_quality_weight: float = 0.30
    identity_min_confidence: float = 0.65
    identity_min_area: float = 0.0
    identity_min_width_px: float = 0.0
    identity_min_height_px: float = 0.0
    identity_max_single_frame_area_shrink_ratio: float = 0.30
    identity_area_shrink_max_gap_frames: int = 2
    # A raw DeepSORT track that jumps across a large part of the image in one
    # or two updates is usually a track swap during a person crossing.
    identity_max_center_jump_ratio: float = 0.30
    identity_center_jump_max_gap_frames: int = 2
    identity_swap_min_mapped_jump_ratio: float = 0.08
    identity_swap_max_replacement_distance_ratio: float = 0.12
    identity_max_area_ratio: float = 0.75
    identity_max_width_ratio: float = 0.85
    identity_max_height_ratio: float = 1.00
    identity_min_aspect_ratio: float = 0.18
    identity_max_aspect_ratio: float = 1.45
    identity_max_edge_touch_count: int = 2
    identity_edge_margin_ratio: float = 0.02
    identity_reacquire_threshold: float = 0.50
    identity_reacquire_max_frames: int = 90
    identity_reacquire_margin: float = 0.08
    identity_reacquire_single_candidate_only: bool = True
    identity_reacquire_multi_candidate_enable: bool = True
    identity_reacquire_multi_candidate_threshold: float = 0.40
    identity_reacquire_multi_candidate_margin: float = 0.12
    identity_new_confirm_frames: int = 2
    identity_mapped_verify_enable: bool = True
    identity_mapped_verify_threshold: float = 0.40
    identity_mapped_bad_quality_returns_unassigned: bool = True
    identity_exclusive_uid_claim_enable: bool = True
    identity_exclusive_uid_claim_frames: int = 15
    identity_controlled_handoff_enable: bool = True
    identity_controlled_handoff_confirm_frames: int = 2
    identity_controlled_handoff_instant_threshold: float = 0.15
    identity_controlled_handoff_threshold: float = 0.30
    identity_controlled_handoff_min_old_track_gap_frames: int = 5
    identity_handoff_geometry_max_gap_frames: int = 15
    identity_handoff_geometry_max_center_jump_ratio: float = 0.25
    identity_handoff_geometry_min_area_similarity: float = 0.35
    identity_preferred_search_reacquire_enable: bool = True
    identity_preferred_search_reacquire_threshold: float = 0.20
    identity_preferred_search_reacquire_max_disadvantage: float = 0.05
    identity_preferred_search_reacquire_min_confidence: float = 0.50
    identity_preferred_search_reacquire_observation_min_confidence: float = 0.25
    identity_preferred_search_reacquire_max_age_sec: float = 0.35
    identity_preferred_search_reacquire_late_candidate_enable: bool = True
    identity_preferred_search_reacquire_confirm_frames: int = 2
    identity_preferred_search_reacquire_instant_threshold: float = 0.15
    # Multiple detector boxes are acceptable when the active candidate is
    # clearly stronger than the remaining low-confidence fragments.  Close
    # competing person detections still require the normal rejection path.
    identity_preferred_search_reacquire_min_score_gap: float = 0.25
    identity_preferred_search_soft_candidate_enable: bool = True
    identity_preferred_search_soft_candidate_threshold: float = 0.30
    identity_preferred_search_soft_min_score_gap: float = 0.15
    identity_preferred_search_soft_min_area_ratio: float = 0.25
    identity_preferred_search_soft_min_confidence: float = 0.80
    identity_partial_appearance_enable: bool = True
    identity_partial_match_threshold: float = 0.34
    identity_partial_confirm_threshold: float = 0.34
    identity_partial_max_features: int = 8
    identity_partial_update_threshold: float = 0.30
    identity_preferred_search_reacquire_side_ratio: float = 0.05
    identity_duplicate_box_suppression_enable: bool = True
    identity_duplicate_iou_threshold: float = 0.55
    identity_duplicate_vertical_overlap_threshold: float = 0.88
    identity_duplicate_horizontal_overlap_threshold: float = 0.40
    identity_duplicate_large_height_ratio: float = 0.80
    identity_duplicate_large_width_ratio: float = 0.45
    identity_duplicate_max_area_ratio: float = 0.65
    identity_duplicate_bottom_gap_ratio: float = 0.10


class DeepSortTracker:
    """Thin adapter from rk_vision detections/features to the DeepSORT module."""

    def __init__(self, config: DeepSortTrackerConfig) -> None:
        self.config = config
        self.deepsort = DeepSort(
            DeepSortConfig(
                max_dist=config.max_cosine_distance,
                min_confidence=config.min_confidence,
                nms_max_overlap=config.nms_max_overlap,
                max_iou_distance=config.max_iou_distance,
                max_age=config.max_age,
                max_output_age=config.max_output_age,
                n_init=config.n_init,
                nn_budget=config.nn_budget,
                max_bbox_age=config.max_bbox_age,
                feature_update_interval=config.feature_update_interval,
                camera_hfov_deg=config.hfov_deg,
            )
        )
        self.identity_bank = IdentityBank(
            IdentityBankConfig(
                enabled=config.identity_bank_enable,
                match_threshold=config.identity_match_threshold,
                match_margin=config.identity_match_margin,
                update_threshold=config.identity_update_threshold,
                update_interval=config.identity_update_interval,
                max_features=config.identity_max_features,
                template_memory_enable=config.identity_template_memory_enable,
                template_crosscheck_enable=config.identity_template_crosscheck_enable,
                template_learning_guard_enable=config.identity_template_learning_guard_enable,
                similar_follow_enable=config.identity_similar_follow_enable,
                similar_follow_entry_threshold=config.identity_similar_follow_entry_threshold,
                similar_follow_retain_threshold=config.identity_similar_follow_retain_threshold,
                similar_follow_max_gap_sec=config.identity_similar_follow_max_gap_sec,
                appearance_region_safety_enable=config.identity_appearance_region_safety_enable,
                template_recent_sec=config.identity_template_recent_sec,
                template_archive_sec=config.identity_template_archive_sec,
                max_weak_features=config.identity_max_weak_features,
                diversity_min_distance=config.identity_diversity_min_distance,
                diversity_replace_margin=config.identity_diversity_replace_margin,
                weak_update_threshold=config.identity_weak_update_threshold,
                weak_update_interval=config.identity_weak_update_interval,
                weak_match_penalty=config.identity_weak_match_penalty,
                weak_reacquire_threshold=config.identity_weak_reacquire_threshold,
                weak_reacquire_confirm_frames=config.identity_weak_reacquire_confirm_frames,
                weak_quality_weight=config.identity_weak_quality_weight,
                min_confidence=config.identity_min_confidence,
                min_area=config.identity_min_area,
                reacquire_threshold=config.identity_reacquire_threshold,
                reacquire_max_frames=config.identity_reacquire_max_frames,
                reacquire_margin=config.identity_reacquire_margin,
                reacquire_single_candidate_only=config.identity_reacquire_single_candidate_only,
                reacquire_multi_candidate_enable=config.identity_reacquire_multi_candidate_enable,
                reacquire_multi_candidate_threshold=config.identity_reacquire_multi_candidate_threshold,
                reacquire_multi_candidate_margin=config.identity_reacquire_multi_candidate_margin,
                new_identity_confirm_frames=config.identity_new_confirm_frames,
                mapped_verify_enable=config.identity_mapped_verify_enable,
                mapped_verify_threshold=config.identity_mapped_verify_threshold,
                mapped_bad_quality_returns_unassigned=config.identity_mapped_bad_quality_returns_unassigned,
                exclusive_uid_claim_enable=config.identity_exclusive_uid_claim_enable,
                exclusive_uid_claim_frames=config.identity_exclusive_uid_claim_frames,
                controlled_handoff_enable=config.identity_controlled_handoff_enable,
                controlled_handoff_confirm_frames=config.identity_controlled_handoff_confirm_frames,
                controlled_handoff_instant_threshold=config.identity_controlled_handoff_instant_threshold,
                controlled_handoff_threshold=config.identity_controlled_handoff_threshold,
                controlled_handoff_min_old_track_gap_frames=config.identity_controlled_handoff_min_old_track_gap_frames,
                handoff_geometry_max_gap_frames=config.identity_handoff_geometry_max_gap_frames,
                handoff_geometry_max_center_jump_ratio=config.identity_handoff_geometry_max_center_jump_ratio,
                handoff_geometry_min_area_similarity=config.identity_handoff_geometry_min_area_similarity,
                max_edge_touch_count=config.identity_max_edge_touch_count,
                preferred_search_reacquire_enable=config.identity_preferred_search_reacquire_enable,
                preferred_search_reacquire_threshold=config.identity_preferred_search_reacquire_threshold,
                preferred_search_reacquire_max_disadvantage=config.identity_preferred_search_reacquire_max_disadvantage,
                preferred_search_reacquire_min_confidence=config.identity_preferred_search_reacquire_min_confidence,
                preferred_search_reacquire_observation_min_confidence=(
                    config.identity_preferred_search_reacquire_observation_min_confidence
                ),
                preferred_search_reacquire_max_age_sec=config.identity_preferred_search_reacquire_max_age_sec,
                preferred_search_reacquire_late_candidate_enable=config.identity_preferred_search_reacquire_late_candidate_enable,
                preferred_search_reacquire_confirm_frames=config.identity_preferred_search_reacquire_confirm_frames,
                preferred_search_reacquire_instant_threshold=config.identity_preferred_search_reacquire_instant_threshold,
                preferred_search_reacquire_min_score_gap=config.identity_preferred_search_reacquire_min_score_gap,
                preferred_search_soft_candidate_enable=config.identity_preferred_search_soft_candidate_enable,
                preferred_search_soft_candidate_threshold=config.identity_preferred_search_soft_candidate_threshold,
                preferred_search_soft_min_score_gap=config.identity_preferred_search_soft_min_score_gap,
                preferred_search_soft_min_area_ratio=config.identity_preferred_search_soft_min_area_ratio,
                preferred_search_soft_min_confidence=config.identity_preferred_search_soft_min_confidence,
                partial_appearance_enable=config.identity_partial_appearance_enable,
                partial_match_threshold=config.identity_partial_match_threshold,
                partial_confirm_threshold=config.identity_partial_confirm_threshold,
                partial_max_features=config.identity_partial_max_features,
                partial_update_threshold=config.identity_partial_update_threshold,
                camera_hfov_deg=config.hfov_deg,
            )
        )
        self._frame_index = 0
        self._search_reacquire_uid = 0
        self._search_reacquire_direction: Optional[str] = None
        self._search_reacquire_eligible_tracks: Set[int] = set()
        # A detector-only search observation must not become a normal
        # DeepSORT identity. Keep it on a private raw-track id until the
        # locked UID is confirmed, then discard it when a real track returns.
        self._search_probe_track_id = -1
        self._search_probe_bbox: Optional[Tuple[float, float, float, float]] = None
        self._search_probe_frame = -1
        self._last_quality_area_by_track_id = {}
        self._last_identity_center_by_track_id = {}
        self._last_identity_center_frame_by_track_id = {}
        self._frame_context = {}
        self._current_detections = ()
        self._identity_competition = {}
        self.last_identity_observations: List[dict] = []
        self._provisional_association = ProvisionalAssociationCache()
        self._detector_proof = None
        self._detector_full_bridge = None
        self._current_detector_position_bridge = None
        self._detector_epoch = 0
        self._detector_active_uid = 0
        self._low_score_position_anchor = None
        self._low_score_position_last = None
        self._follow_only_position_anchor = None
        self._follow_only_position_last = None
        self._detector_bank_binding = None
        self._detector_control_assignments = {}
        self.last_detector_continuation_reason = "no_full_verification"

    def _clear_detector_continuation(self, reason):
        self._detector_proof = None
        # A consumed detector position can still link the *next full* frame
        # captured before its fixed deadline. This does not keep a motion lease
        # alive after expiry, and negative evidence invalidates it immediately.
        if reason not in {"detection_gap", "detection_stale", "verification_expired"}:
            self._detector_full_bridge = None
        self._detector_bank_binding = None
        self._detector_control_assignments.clear()
        self._detector_epoch += 1
        self.last_detector_continuation_reason = reason

    def _detector_active(self, active_uid):
        uid = finite_number(active_uid)
        uid = int(uid) if uid is not None and uid > 0 and uid.is_integer() else 0
        if uid != self._detector_active_uid:
            self._clear_detector_continuation("active_uid_changed")
            self._low_score_position_anchor = None
            self._low_score_position_last = None
            anchor = getattr(self, "_follow_only_position_anchor", None)
            if anchor is not None and uid != anchor["uid"] and (
                    uid > 0 or self._search_reacquire_uid != anchor["uid"]):
                self._follow_only_position_anchor = None
                self._follow_only_position_last = None
            self._detector_active_uid = uid
        return uid

    def set_detector_continuation_context(self, *, active_uid, allowed):
        """Revoke promptly when control leaves established visible tracking."""
        self._detector_active(active_uid)
        if not allowed:
            self._clear_detector_continuation("control_context_requires_full")

    @staticmethod
    def _detector_verification_facts(entry, assignment):
        anchor = entry.last_strong_observation or {}
        return (tuple(anchor.get(key) for key in (
                    "track_id", "frame_index", "capture_frame_id", "capture_timestamp",
                    "integrated_yaw_deg", "geometry_source", "center_x_ratio", "area")),
                tuple(anchor.get("bbox", ())),
                tuple(assignment.get(key) for key in (
                    "uid", "reason", "distance", "match_source", "bbox_quality_ok",
                    "bbox_quality_tier", "identity_control_rejected", "identity_recheck_pending")))

    def _detector_background_eligibility(self, track_id, uid, target_track_id, stamp):
        """Read a full identity check's negative evidence, without refreshing it."""
        bank = self.identity_bank
        if bank.track_to_uid.get(track_id) == uid or track_id in bank._mapped_geometry_conflicts:
            return None
        exclusion = bank.search_exclusion_for(
            track_id, uid, frame_index=self._frame_index, capture_timestamp=stamp)
        if (exclusion and exclusion.get("reason") == "co_visible_distinct_person"
                and exclusion.get("reference_track_id") == target_track_id
                and exclusion.get("witness_geometry_valid") is True):
            return ("co_visible_distinct_person", exclusion.get("source_capture_frame_id"),
                    exclusion.get("source_frame"), target_track_id)
        assignment = bank.last_assignments.get(track_id, {})
        competition = assignment.get("identity_competition") or {}
        if (competition.get("uid") == uid
                and competition.get("competition_eligible") is False
                and competition.get("reason") == "ineligible_uid_competitor"
                and competition.get("eligibility_reference_kind") == "accepted_identity"
                and competition.get("eligibility_reference_cap") is not None):
            return ("ineligible_uid_competitor", competition.get("frame_index"),
                    competition.get("source_detection_index"),
                    competition.get("eligibility_reference_cap"))
        return None

    def _similar_detector_qualified(self, uid, track_id, observation):
        """Current independent-gallery evidence, never a learning permission.

        An accepted follow-only state already represents two observed captures.
        Require the current FULL frame's gallery support, not its optional
        self-reference distance, and preserve its exact raw-track provenance.
        """
        bank = self.identity_bank
        assignment = bank.last_assignments.get(track_id, {})
        detail = assignment.get("similar_follow") or {}
        state = bank._similar_follow_states.get((uid, track_id))
        gallery = finite_number(detail.get("gallery_distance"))
        crop_origin = finite_number((state or {}).get("last_strong_timestamp"))
        competition = assignment.get("identity_competition") or {}
        return bool(self.config.identity_similar_follow_enable and observation is not None
            and assignment.get("reason") == "mapped_similar_follow"
            and assignment.get("match_source") == "similar_follow"
            and assignment.get("identity_permission") == "similar_follow"
            and assignment.get("bank_updated") is False
            and assignment.get("reacquire_geometry_ok") is True
            and detail.get("status") == "follow"
            and detail.get("learning_allowed") is False
            and detail.get("capture_frame_id") == observation.capture
            and not detail.get("handoff_from_track_id")
            and (not detail.get("crop_continuation") or detail.get("crop_current_reverified") is True
                 or (crop_origin is not None and observation.timestamp < crop_origin + .75))
            and gallery is not None and 0 <= gallery <= SIMILAR_MAX_GALLERY_DISTANCE
            and competition.get("passed") is True and competition.get("candidate_count") == 1
            and isinstance(state, dict) and state.get("active") is True and state.get("count") == 2
            and state.get("uid") == uid and state.get("track_id") == track_id
            and state.get("entry_direction_compatible") is True
            and not state.get("position_only")
            and state.get("last_cap") == observation.capture
            and state.get("last_timestamp") == observation.timestamp
            and tuple((state.get("observation") or {}).get("detector_bbox", ())) == observation.bbox)

    def _detector_edge_mask(self, observation):
        margin = max(0., float(self.config.identity_edge_margin_ratio))
        x1, y1, x2, y2 = observation.bbox
        return (x1 <= observation.width*margin, y1 <= observation.height*margin,
                observation.width-x2 <= observation.width*margin,
                observation.height-y2 <= observation.height*margin)

    def _detector_quality_allowed(self, observation, proof=None):
        if observation.score < max(self.config.min_confidence, self.config.identity_min_confidence):
            return False
        good, reason = self._bbox_quality(observation.bbox, observation.width, observation.height)
        if proof is None or proof.permission != "similar_follow":
            return good
        # A new clipping boundary is an event requiring full appearance. Only
        # an already FULL-accepted same-edge crop can keep its weaker lane.
        if self._detector_edge_mask(observation) != proof.crop_edge_mask:
            return False
        return good or (reason == "edge_touch>2" and sum(proof.crop_edge_mask) == 3
                        and proof.crop_edge_mask[0] != proof.crop_edge_mask[2])

    def _detector_identity_block(self, uid, track_id, stamp, backgrounds=(), *, follow_observation=None):
        bank = self.identity_bank
        follow_only = follow_observation is not None
        if follow_only and (backgrounds or not self._similar_detector_qualified(uid, track_id, follow_observation)):
            return "similar_full_support_unavailable"
        if (not bank.config.enabled or uid <= 0 or self._detector_active_uid != uid
                or self._search_reacquire_uid > 0
                or bank.track_to_uid.get(track_id) != uid):
            return "identity_binding_changed"
        if (uid not in bank.identities or uid in bank._geometry_revoked_uids
                or uid in bank._reacquire_control_suspects
                or track_id in bank._mapped_geometry_conflicts
                or (not follow_only and bank._reacquire_quarantine.is_held(uid))):
            return "identity_blocked"
        if any(track_id in pending for pending in (
            bank.pending_new, bank.pending_handoffs,
            bank.pending_late_handoffs, bank.pending_weak_handoffs,
        )) or bank.search_exclusion_for(track_id, uid, frame_index=self._frame_index,
                                      capture_timestamp=stamp) is not None:
            return "identity_pending_or_excluded"
        assignment = bank.last_assignments.get(track_id, {})
        if (assignment.get("uid") != uid or any(assignment.get(key) for key in (
                "identity_control_rejected", "identity_recheck_pending", "search_excluded",
                "appearance_pose_retention"))
                or (not follow_only and assignment.get("template_update_quarantined"))
                or assignment.get("bbox_quality_ok") is not True
                or assignment.get("bbox_quality_tier") != "strong"
                or assignment.get("reacquire_geometry_ok") is False
                or (assignment.get("identity_competition") or {}).get("passed") is False):
            return "identity_assignment_rejected"
        tracks = self.deepsort.tracker.tracks
        expected = {track_id, *(item.track_id for item in backgrounds)}
        if (len(tracks) != len(expected) or {t.track_id for t in tracks} != expected
                or any(not t.is_confirmed() or t.time_since_update != 0 for t in tracks)):
            return "track_not_unique_current"
        if any(self._detector_background_eligibility(item.track_id, uid, track_id, stamp)
               != item.eligibility for item in backgrounds):
            return "background_exclusion_changed"
        return None

    def control_assignment_for_track(self, track_id):
        """Current detector-only control evidence; never overwrite bank history."""
        assignment = self._detector_control_assignments.get(int(track_id))
        return dict(assignment) if assignment is not None else None

    def _detector_transition_reject(self, track_id, observation):
        """Apply the adjacent size/center thresholds before skipping ReID.

        This is read-only: a failed proposal still runs the original full
        frame using the unmodified quality history.
        """
        cx, _cy, width, height = _xyxy_to_expanded_xywh(
            observation.bbox, self.config.bbox_expand_scale)
        next_frame = self._frame_index + 1
        previous_area = self._last_quality_area_by_track_id.get(track_id)
        if previous_area is not None:
            old_frame, old_area = previous_area
            max_gap = max(1, int(self.config.identity_area_shrink_max_gap_frames))
            ratio = max(0.05, min(1.0, float(
                self.config.identity_max_single_frame_area_shrink_ratio)))
            if (0 < next_frame - old_frame <= max_gap and old_area > 0
                    and width * height < old_area * ratio):
                return "detector_area_shrink"
        previous_center = self._last_identity_center_by_track_id.get(track_id)
        previous_frame = self._last_identity_center_frame_by_track_id.get(track_id)
        if previous_center is not None and previous_frame is not None:
            max_gap = max(1, int(self.config.identity_center_jump_max_gap_frames))
            max_jump = max(0.05, min(1.0, float(self.config.identity_max_center_jump_ratio)))
            if (0 < next_frame - previous_frame <= max_gap
                    and abs(cx / observation.width - float(previous_center)) > max_jump):
                return "detector_center_jump"
        return None

    def plan_detected_continuation(self, detections, *, image_width, image_height,
                                  frame_context, active_uid, now, color_features,
                                  raw_candidate_count):
        uid = self._detector_active(active_uid)
        self._detector_control_assignments.clear()
        proof = self._detector_proof
        if proof is None:
            self.last_detector_continuation_reason = "no_full_verification"
            return None
        if (isinstance(raw_candidate_count, bool) or raw_candidate_count != len(detections)
                or len(detections) != 1+len(proof.backgrounds)):
            self._clear_detector_continuation("competing_detections")
            return None
        observations = tuple(capture_observation(d, frame_context, image_width, image_height)
                             for d in detections)
        if any(item is None for item in observations):
            self._clear_detector_continuation("capture_metadata_invalid")
            return None
        # Singleton retains its existing diagnostic reason. With known
        # backgrounds, association must be unique for EVERY current detector
        # crop, not merely "one crop looks near the target".
        source_index = 0
        background_matches = []
        if proof.backgrounds:
            matches = [i for i, item in enumerate(observations)
                       if geometry_matches(proof.previous, item, self.config.hfov_deg)
                       and geometry_matches(proof.verified, item, self.config.hfov_deg, anchor=True)]
            if len(matches) != 1:
                self._clear_detector_continuation("target_detection_ambiguous")
                return None
            source_index = matches[0]
            for background in proof.backgrounds:
                matches = [i for i, item in enumerate(observations)
                           if i != source_index
                           and geometry_matches(background.previous, item, self.config.hfov_deg)
                           and geometry_matches(background.verified, item, self.config.hfov_deg, anchor=True)
                           and separated_observations(observations[source_index], item)]
                if len(matches) != 1:
                    self._clear_detector_continuation("background_detection_changed")
                    return None
                i = matches[0]
                background_matches.append((i, background, observations[i]))
            if (len({i for i, _, _ in background_matches}) != len(background_matches)
                    or any(not separated_observations(first, second)
                           for i, first in enumerate(observations) for second in observations[i+1:])):
                self._clear_detector_continuation("background_detection_ambiguous")
                return None
        observation = observations[source_index]
        reason = self._detector_identity_block(uid, proof.track_id, observation.timestamp, proof.backgrounds,
            follow_observation=proof.verified if proof.permission == "similar_follow" else None)
        if reason is None and not self._detector_quality_allowed(observation, proof):
            reason = "detector_quality"
        if reason is None:
            reason = continuation_reason(proof, observation, now, self.config.hfov_deg)
        if reason is None:
            reason = self._detector_transition_reject(proof.track_id, observation)
        if reason is not None:
            if reason in {"full_verification_streak", "fast_budget_exhausted", "full_recheck_due"}:
                self.last_detector_continuation_reason = reason
            else:
                self._clear_detector_continuation(reason)
            return None
        # Color is cheap evidence, but compute it only for an otherwise eligible
        # fast frame. Full frames use the extractor's existing color signature.
        try:
            colors = color_features() if callable(color_features) else color_features
            color = color_signature(colors[source_index]) if colors is not None and len(colors) == len(detections) else None
        except (TypeError, ValueError, IndexError, RuntimeError):
            color = None
        if color is None or not color_matches(proof.verified.color, color):
            self._clear_detector_continuation("color_conflict_or_unavailable")
            return None
        observation = replace(observation, color=color)
        self.last_detector_continuation_reason = "planned"
        current_context = tuple((key, frame_context[key]) for key in ("control_frame_id", "yaw_rate_dps")
                                if finite_number(frame_context.get(key)) is not None)
        return DetectorContinuationPlan(proof, observation, self._frame_index,
                                        self._detector_epoch, float(now), current_context,
                                        source_index, tuple(background_matches))

    def commit_detected_continuation(self, plan, *, now):
        if (not isinstance(plan, DetectorContinuationPlan) or plan.proof is not self._detector_proof
                or plan.epoch != self._detector_epoch or plan.frame_index != self._frame_index):
            self.last_detector_continuation_reason = "plan_superseded"
            return None
        proof, observation = plan.proof, plan.observation
        reason = self._detector_identity_block(proof.uid, proof.track_id, observation.timestamp, proof.backgrounds,
            follow_observation=proof.verified if proof.permission == "similar_follow" else None)
        if reason is None:
            reason = continuation_reason(proof, observation, now, self.config.hfov_deg)
        if reason is None:
            reason = self._detector_transition_reject(proof.track_id, observation)
        if reason is None and (finite_number(now) is None or now < plan.planned_at):
            reason = "clock_reversed"
        entry = self.identity_bank.identities.get(proof.uid)
        binding = self._detector_bank_binding
        if reason is None and (binding is None or entry is not binding[0]
                or entry.last_strong_observation is not binding[1]
                or self.identity_bank.last_assignments.get(proof.track_id) is not binding[2]
                or self._detector_verification_facts(entry, binding[2]) != binding[3]):
            reason = "full_verification_replaced"
        if reason is not None:
            self._clear_detector_continuation(reason)
            return None
        # A unique target plus only full-verified excluded backgrounds. Advance
        # real detector geometry once; neither embeddings nor identity memory
        # are learned and the original FULL verification deadline does not move.
        tracks = {track.track_id: track for track in self.deepsort.tracker.tracks}
        track = tracks[proof.track_id]
        xywh = _xyxy_to_expanded_xywh(observation.bbox, self.config.bbox_expand_scale)
        cx, cy, width, height = xywh
        detection = DeepSortDetection((cx-width/2, cy-height/2, width, height),
            observation.score, 0, None, store_feature=False, source_detection_index=plan.source_index)
        self.deepsort.tracker.predict()
        track.update(self.deepsort.tracker.kf, detection)
        for index, background, item in plan.backgrounds:
            cx, cy, width, height = _xyxy_to_expanded_xywh(item.bbox, self.config.bbox_expand_scale)
            tracks[background.track_id].update(self.deepsort.tracker.kf, DeepSortDetection(
                (cx-width/2, cy-height/2, width, height), item.score, 0, None,
                store_feature=False, source_detection_index=index))
        self.deepsort._frame_index += 1
        self._frame_index += 1
        # Detector measurements also advance local quality clocks. The
        # identity gallery and its trusted strong anchor remain untouched.
        tx1, ty1, tx2, ty2 = (float(value) for value in track.to_tlbr())
        self._last_quality_area_by_track_id[proof.track_id] = (
            self._frame_index, max(0., tx2-tx1) * max(0., ty2-ty1))
        self._last_identity_center_by_track_id[proof.track_id] = (
            (tx1+tx2) / (2*observation.width))
        self._last_identity_center_frame_by_track_id[proof.track_id] = self._frame_index
        self._detector_epoch += 1
        self._detector_proof = replace(proof, previous=observation, fast_count=proof.fast_count+1,
            backgrounds=tuple(replace(bg, previous=item) for _, bg, item in plan.backgrounds))
        if proof.permission == "similar_follow":
            self._detector_full_bridge = (self._detector_proof, self._detector_bank_binding, track,
                self.identity_bank._similar_follow_states.get((proof.uid, proof.track_id)))
        self._frame_context = dict(plan.context, capture_frame_id=observation.capture,
            capture_timestamp=observation.timestamp, integrated_yaw_deg=observation.yaw)
        current = [(plan.source_index, observation)] + [(i, item) for i, _, item in plan.backgrounds]
        self._current_detections = tuple(Detection(item.bbox, item.score, 0) for _, item in sorted(current))
        self._identity_competition = {}
        x1, y1, x2, y2 = observation.bbox
        cx, cy, area = (x1+x2)/2, (y1+y2)/2, (x2-x1)*(y2-y1)
        metadata = dict(self._frame_context, track_id=proof.track_id, frame_index=self._frame_index,
            is_fresh=True, detector_bbox=observation.bbox, bbox=observation.bbox,
            quality_bbox=observation.bbox, quality_bbox_source="detector", quality_bbox_ok=True,
            bbox_quality_tier="strong", source_detection_index=plan.source_index,
            candidate_count=len(current), detector_excluded_background_count=len(plan.backgrounds),
            image_width=observation.width, image_height=observation.height,
            detector_center_x_ratio=cx/observation.width,
            detector_area_ratio=area/(observation.width*observation.height),
            detector_confidence=observation.score)
        assignment = dict(uid=proof.uid, mapped_uid=proof.uid, reason="detector_continuation",
            identity_evidence_kind="detector_continuation", identity_verified_capture=proof.verified.capture,
            identity_verified_timestamp=proof.verified.timestamp, identity_valid_until=proof.deadline,
            initial_identity_confirmed=False, bank_updated=False, bbox_quality_ok=True,
            bbox_quality_tier="strong", identity_control_rejected=False,
            capture_frame_id=observation.capture, capture_timestamp=observation.timestamp)
        if proof.permission == "similar_follow":
            assignment.update(identity_permission="similar_follow", match_source="similar_follow",
                template_update_quarantined=True, learning_allowed=False,
                template_learning=dict(status="frozen", reason="similar_detector_continuation"),
                detector_continuation_permission="similar_follow")
        self._detector_control_assignments = {proof.track_id: assignment}
        self.last_identity_observations = [dict(frame_index=self._frame_index,
            raw_track_id=proof.track_id, uid=proof.uid, detector_bbox=observation.bbox,
            display_bbox=observation.bbox, sample_metadata=metadata, assignment=dict(assignment))]
        self.last_timing_ms = {"detector_continuation_used": 1.0}
        self.last_detector_continuation_reason = "accepted"
        records = [TrackRecord(proof.track_id, proof.uid, x1, y1, x2, y2, 0, observation.score,
                            cx, cy, area, (cx/observation.width-.5)*self.config.hfov_deg,
                            TRACK_STATE_STABLE, time_since_update=0)]
        # Retain other detected people for display/hazard consumers, but never
        # promote an exclusion into a positive UID or a fresh bank assignment.
        for index, background, item in plan.backgrounds:
            bx1, by1, bx2, by2 = item.bbox
            bcx, bcy = (bx1+bx2)/2, (by1+by2)/2
            background_assignment = dict(uid=0, reason="detector_excluded_background",
                identity_evidence_kind="detector_excluded_background", search_excluded=True,
                excluded_uid=proof.uid, exclusion_verified_capture=background.verified.capture,
                bank_updated=False, capture_frame_id=item.capture, capture_timestamp=item.timestamp)
            self._detector_control_assignments[background.track_id] = dict(background_assignment)
            self.last_identity_observations.append(dict(frame_index=self._frame_index,
                raw_track_id=background.track_id, uid=0, detector_bbox=item.bbox,
                display_bbox=item.bbox, assignment=background_assignment,
                sample_metadata=dict(self._frame_context, track_id=background.track_id,
                    frame_index=self._frame_index, is_fresh=True, source_detection_index=index,
                    candidate_count=len(current), detector_bbox=item.bbox,
                    search_excluded=True, identity_evidence_kind="detector_excluded_background")))
            records.append(TrackRecord(background.track_id, 0, bx1, by1, bx2, by2, 0,
                item.score, bcx, bcy, (bx2-bx1)*(by2-by1),
                (bcx/item.width-.5)*self.config.hfov_deg, TRACK_STATE_STABLE, time_since_update=0))
        return records

    def _prepare_detector_full_continuation(self, detections, features, color_features,
                                           image_width, image_height):
        """Consume measured fast geometry once, without retimestamping features.

        A skipped full frame must not make the next full appear unobserved for
        >500ms. Its *new* independent descriptor still has to pass the gallery;
        the earlier fast measurement only supplies local position continuity.
        """
        carried = self._detector_full_bridge
        self._detector_full_bridge = None
        if carried is None or len(detections) != 1 or len(self._learning_detections) != 1:
            return None
        proof, binding, track, state = carried
        if proof.permission != "similar_follow" or proof.fast_count < 1 or binding is None:
            return None
        observation = capture_observation(detections[0], self._frame_context, image_width, image_height)
        now = time.monotonic()
        entry, anchor, assignment, facts = binding
        bank = self.identity_bank
        if (observation is None or not 0 <= now-observation.timestamp < MAX_FULL_RESULT_AGE_SEC
                or not proof.previous.timestamp < observation.timestamp < proof.deadline
                or observation.capture <= proof.previous.capture
                or observation.timestamp-proof.previous.timestamp > .5
                or bank.identities.get(proof.uid) is not entry
                or entry.last_strong_observation is not anchor
                or bank.last_assignments.get(proof.track_id) is not assignment
                or bank._similar_follow_states.get((proof.uid, proof.track_id)) is not state
                or facts != self._detector_verification_facts(entry, assignment)
                or not any(t is track for t in self.deepsort.tracker.tracks)
                or self._detector_identity_block(proof.uid, proof.track_id, observation.timestamp,
                                                follow_observation=proof.verified)
                or not self._detector_quality_allowed(observation, proof)
                or not geometry_matches(proof.previous, observation, self.config.hfov_deg)
                or not geometry_matches(proof.verified, observation, self.config.hfov_deg, anchor=True)):
            return None
        gallery = finite_number(entry.distance(features[0])) if features[0] is not None else None
        color = color_signature(color_features[0]) if len(color_features) == 1 else None
        if gallery is None or not 0 <= gallery <= SIMILAR_MAX_GALLERY_DISTANCE or not color_matches(proof.verified.color, color or ()):
            return None
        return dict(proof=proof, identity=entry, assignment=assignment, state=state,
                    capture_frame_id=observation.capture, capture_timestamp=observation.timestamp, now=now)

    def note_full_identity_verification(self, records, *, detections, color_features,
                                        frame_context, image_width, image_height, now, active_uid,
                                        raw_candidate_count=None):
        uid = self._detector_active(active_uid)
        previous = self._detector_proof
        self._detector_control_assignments.clear()
        targets = [record for record in records if record.reid_uid == uid]
        if (len(targets) != 1 or not detections or uid <= 0
                or (raw_candidate_count is not None and (
                    isinstance(raw_candidate_count, bool) or raw_candidate_count != len(detections)))):
            self._clear_detector_continuation("full_identity_not_unique")
            return
        record = targets[0]
        evidence = self.last_identity_observations
        target_evidence = [item for item in evidence if item.get("raw_track_id") == record.track_id]
        if len(target_evidence) != 1:
            self._clear_detector_continuation("full_detection_binding_unavailable")
            return
        source_index = target_evidence[0].get("sample_metadata", {}).get("source_detection_index")
        if (not isinstance(source_index, int) or isinstance(source_index, bool)
                or not 0 <= source_index < len(detections)):
            self._clear_detector_continuation("full_detection_binding_unavailable")
            return
        observation = capture_observation(detections[source_index], frame_context, image_width, image_height)
        follow_only = bool(len(detections) == 1
                           and self._similar_detector_qualified(uid, record.track_id, observation))
        backgrounds = []
        for index, detection in enumerate(detections):
            if index == source_index:
                continue
            candidates = [item for item in evidence if
                item.get("sample_metadata", {}).get("source_detection_index") == index]
            item = capture_observation(detection, frame_context, image_width, image_height)
            if len(candidates) != 1 or item is None or observation is None:
                self._clear_detector_continuation("full_background_unverified")
                return
            candidate = candidates[0]
            meta = candidate.get("sample_metadata", {})
            track_id = candidate.get("raw_track_id")
            eligibility = self._detector_background_eligibility(
                track_id, uid, record.track_id, item.timestamp)
            if (candidate.get("uid") == uid or track_id == record.track_id
                    or meta.get("capture_frame_id") != item.capture
                    or meta.get("capture_timestamp") != item.timestamp
                    or tuple(candidate.get("detector_bbox", ())) != item.bbox
                    or not separated_observations(observation, item)
                    or eligibility is None
                    or (eligibility[0] == "co_visible_distinct_person" and eligibility[1] != item.capture)
                    or (eligibility[0] == "ineligible_uid_competitor" and eligibility[1] != self._frame_index)):
                self._clear_detector_continuation("full_background_unverified")
                return
            backgrounds.append(DetectorBackgroundProof(track_id, item, item, eligibility))
        if any(not separated_observations(first.verified, second.verified)
               for i, first in enumerate(backgrounds) for second in backgrounds[i+1:]):
            self._clear_detector_continuation("full_background_ambiguous")
            return
        clock = finite_number(now)
        assignment = self.identity_bank.last_assignments.get(record.track_id, {})
        distance = finite_number(assignment.get("distance"))
        evidence = target_evidence
        target_track = next((t for t in self.deepsort.tracker.tracks if t.track_id == record.track_id), None)
        if (observation is None or clock is None or not 0 <= clock-observation.timestamp < MAX_FULL_RESULT_AGE_SEC
                or record.reid_uid != uid or record.time_since_update != 0
                or record.tracker_state != TRACK_STATE_STABLE
                or self._detector_identity_block(uid, record.track_id,
                    observation.timestamp if observation else 0, backgrounds,
                    follow_observation=observation if follow_only else None)
                or (not follow_only and assignment.get("reason") not in {
                    "mapped", "updated_diverse", "skip_update_redundant", "skip_update_distance"})
                or (not follow_only and assignment.get("match_source") != "strong")
                or distance is None or not 0 <= distance <= min(.30, self.config.identity_mapped_verify_threshold)
                or len(evidence) != 1 or evidence[0].get("uid") != uid
                or evidence[0].get("raw_track_id") != record.track_id
                or evidence[0].get("sample_metadata", {}).get("capture_frame_id") != observation.capture
                or evidence[0].get("sample_metadata", {}).get("capture_timestamp") != observation.timestamp
                or tuple(evidence[0].get("detector_bbox", ())) != observation.bbox
                or target_track is None or target_track.last_feature is None):
            self._clear_detector_continuation("full_identity_unverified")
            return
        color = (color_signature(color_features[source_index])
                 if color_features is not None and len(color_features) == len(detections) else None)
        candidate = DetectorProof(uid, record.track_id, observation, observation, 2,
            permission="similar_follow", crop_edge_mask=self._detector_edge_mask(observation)) if follow_only else None
        if color is None or not self._detector_quality_allowed(observation, candidate):
            self._clear_detector_continuation("full_color_or_quality_unavailable")
            return
        observation = replace(observation, color=color)
        continuous = bool(previous is not None and clock < previous.deadline
            and previous.uid == uid and previous.track_id == record.track_id
            and previous.permission == ("similar_follow" if follow_only else "strong")
            and observation.capture > previous.previous.capture
            and 0 < observation.timestamp-previous.previous.timestamp <= FULL_PROOF_TTL_SEC
            and geometry_matches(previous.previous, observation, self.config.hfov_deg)
            and color_matches(previous.verified.color, color)
            and {bg.track_id for bg in previous.backgrounds} == {bg.track_id for bg in backgrounds})
        if previous is not None and (observation.capture <= previous.previous.capture
                                    or observation.timestamp <= previous.previous.timestamp):
            self._clear_detector_continuation("nonnew_full_capture")
            return
        count = 2 if follow_only else min(2, previous.full_count+1) if continuous else 1
        permission_deadline = None
        detail = assignment.get("similar_follow") or {}
        if follow_only and detail.get("crop_continuation") and not detail.get("crop_current_reverified"):
            # Historical crop acceptance cannot be promoted into a fresh
            # .6 s permission after its original .75 s bridge has expired.
            permission_deadline = self.identity_bank._similar_follow_states[(uid, record.track_id)]["last_strong_timestamp"] + .75
        self._detector_proof = DetectorProof(uid, record.track_id, observation, observation, count,
            backgrounds=tuple(backgrounds), permission="similar_follow" if follow_only else "strong",
            crop_edge_mask=self._detector_edge_mask(observation) if follow_only else (),
            permission_deadline=permission_deadline)
        entry = self.identity_bank.identities[uid]
        self._detector_bank_binding = (entry, entry.last_strong_observation, assignment,
                                       self._detector_verification_facts(entry, assignment))
        self._detector_epoch += 1
        self.last_detector_continuation_reason = "full_verified" if count == 2 else "full_verification_streak"

    def _observe_startup_track_lifecycle(self, before):
        bank = self.identity_bank
        if (bank.config.enabled and bank.config.appearance_region_safety_enable
                and not bank.identities):
            bank._initial_enrollment.observe_track_lifecycle(
                before, {int(track.track_id) for track in self.deepsort.tracker.tracks},
                frame_index=self._frame_index, metadata=self._frame_context,
                observed_at=time.monotonic())

    def update(
        self,
        detections: Sequence[Detection],
        features: Sequence[Optional[Any]],
        *,
        partial_features: Optional[Sequence[Optional[Any]]] = None,
        partial_feature_sources: Optional[Sequence[Optional[str]]] = None,
        color_features: Optional[Sequence[Optional[Any]]] = None,
        learning_detections: Optional[Sequence[Detection]] = None,
        image_width: int,
        image_height: Optional[int] = None,
        frame_context: Optional[dict] = None,
    ) -> List[TrackRecord]:
        timer = StageTiming()
        self._detector_control_assignments.clear()
        self._detector_epoch += 1
        self.last_timing_ms = {}
        if len(features) != len(detections):
            raise ValueError("features length must match detections length")
        if partial_features is None:
            partial_features = [None for _ in detections]
        if len(partial_features) != len(detections):
            raise ValueError("partial_features length must match detections length")
        if partial_feature_sources is None:
            partial_feature_sources = [None for _ in detections]
        if len(partial_feature_sources) != len(detections):
            raise ValueError("partial_feature_sources length must match detections length")
        # Optional startup evidence cannot be guessed or shifted between boxes.
        if color_features is None or len(color_features) != len(detections):
            color_features = ()
        self._frame_index += 1
        self._frame_context = dict(frame_context or {})
        self._current_detections = tuple(detections)
        # Learning needs all current person boxes, including observations that
        # were excluded from identity competition or the normal size filter.
        # This snapshot is read-only and never changes assignment eligibility.
        self._learning_detections = tuple(detections if learning_detections is None else learning_detections)
        self._identity_competition = {}
        self.last_identity_observations = []
        startup_tracks_before = {
            int(track.track_id) for track in self.deepsort.tracker.tracks}
        bridge = self._prepare_detector_full_continuation(
            detections, features, color_features, image_width, image_height)
        self._current_detector_position_bridge = bridge
        provisional_features = self._provisional_association.prepare(
            self.deepsort.tracker.tracks, self.identity_bank.track_to_uid,
            self.identity_bank.identities, self._frame_context,
            max_gap=self.config.identity_similar_follow_max_gap_sec,
            conflicts=self.identity_bank._mapped_geometry_conflicts,
            revoked=self.identity_bank._geometry_revoked_uids,
            detector_position_bridge=bridge,
        ) if self.config.identity_similar_follow_enable else None
        if not detections:
            self._search_reacquire_eligible_tracks.clear()
            outputs = self.deepsort.update([], [], [], [], image_shape=_image_shape(image_width, image_height))
            self._observe_startup_track_lifecycle(startup_tracks_before)
            timer.mark("association")
            self._observe_identity_frame_evidence([], image_width, image_height)
            timer.mark("evidence")
            records = [
                self._to_record(
                    out,
                    image_width,
                    image_height,
                    0,
                    partial_features=[],
                    partial_feature_sources=[],
                )
                for out in outputs
            ]
            timer.mark("records")
            self.last_timing_ms = timer.finish()
            return records

        bbox_xywh = [_xyxy_to_expanded_xywh(det.bbox, self.config.bbox_expand_scale) for det in detections]
        confidences = [float(det.score) for det in detections]
        classes = [int(det.class_id) for det in detections]
        # Association runs before UID verification. Preserve the raw-track
        # gallery so follow-only acceptance cannot silently train DeepSORT.
        # Arrays are immutable here; list copies suffice and avoid copying
        # full descriptors on every physical frame.
        association_gallery_before = (
            {track: list(values) for track, values in self.deepsort.tracker.metric.samples.items()}
            if self.config.identity_similar_follow_enable else None
        )
        outputs = self.deepsort.update(
            bbox_xywh,
            confidences,
            classes,
            features,
            image_shape=_image_shape(image_width, image_height),
            match_validator=lambda track_id, source_index: self._identity_match_allowed(
                track_id, source_index, image_width, image_height,
            ),
            provisional_features=provisional_features,
            capture_context=self._frame_context,
            low_score_validator=lambda track_id, source_index: self._low_score_match_allowed(
                track_id, source_index, image_width, image_height),
        )
        self._observe_startup_track_lifecycle(startup_tracks_before)
        timer.mark("association")
        if outputs:
            self.identity_bank.track_to_uid.pop(self._search_probe_track_id, None)
            self.identity_bank.track_last_seen_frame.pop(self._search_probe_track_id, None)
            self.identity_bank.pending_weak_handoffs.pop(self._search_probe_track_id, None)
        fresh_track_ids = {
            int(output.track_id)
            for output in outputs
            if int(output.time_since_update) == 0
        }
        self._search_reacquire_eligible_tracks.intersection_update(fresh_track_ids)
        suppressed_track_ids = self._duplicate_identity_track_ids(outputs, image_width, image_height)
        identity_swap_track_ids = self._identity_swap_track_ids(outputs, image_width)
        suppressed_indices = {
            getattr(out, "source_detection_index", None) for out in outputs
            if int(out.track_id) in suppressed_track_ids
        }
        timer.mark("geometry")
        self._identity_competition = self._frame_identity_competition(
            detections, features, suppressed_indices=suppressed_indices,
            outputs=[out for out in outputs if not getattr(out, "low_score_continuation", False)],
            image_width=image_width, image_height=image_height,
        )
        timer.mark("competition")
        self._observe_identity_frame_evidence(
            outputs, image_width, image_height,
            duplicate_track_ids=suppressed_track_ids,
            identity_swap_track_ids=identity_swap_track_ids,
        )
        timer.mark("evidence")
        candidate_count = max(0, len(detections) - len(suppressed_track_ids))
        records = [
            self._to_record(
                out,
                image_width,
                image_height,
                candidate_count,
                candidate_score_gap=self._candidate_score_gap(
                    getattr(out, "source_detection_index", None), detections
                ),
                partial_features=partial_features,
                partial_feature_sources=partial_feature_sources,
                color_features=color_features,
                duplicate_identity_box=int(out.track_id) in suppressed_track_ids,
                identity_swap_track=int(out.track_id) in identity_swap_track_ids,
            )
            for out in outputs
        ]
        # A weak match supplies candidate-position evidence, not a new control
        # observation. Previously these detections yielded no formal record;
        # emitting a fresh UID0 here would newly invalidate a still-live motor
        # proof. Keep diagnostics/identity observations, but preserve the empty
        # control boundary and do not recreate it through the detector probe.
        has_low_score_observation = any(getattr(out, "low_score_continuation", False)
                                        for out in outputs)
        records = [record for out, record in zip(outputs, records)
                   if not getattr(out, "low_score_continuation", False)]
        if association_gallery_before is not None:
            self._restore_follow_only_association_gallery(association_gallery_before)
        timer.mark("records")
        if not records and not has_low_score_observation:
            probe = self._search_probe_record(
                detections,
                features,
                partial_features=partial_features,
                partial_feature_sources=partial_feature_sources,
                image_width=image_width,
                image_height=image_height,
            )
            if probe is not None:
                records.append(probe)
        timer.mark("probe")
        self.last_timing_ms = timer.finish()
        return records

    def _restore_follow_only_association_gallery(self, before):
        """Freeze trusted galleries; separately retain bounded association evidence."""
        tracker = self.deepsort.tracker
        for track in tracker.tracks:
            assignment = self.identity_bank.last_assignments.get(int(track.track_id), {})
            proof = assignment.get("similar_follow") or {}
            uid = int(assignment.get("uid") or assignment.get("mapped_uid")
                      or self.identity_bank.track_to_uid.get(int(track.track_id), 0))
            fenced = uid in self.identity_bank._similar_learning_fences
            if (assignment.get("match_source") != "similar_follow"
                    and proof.get("status") not in ("observe", "follow") and not fenced):
                if (uid > 0 and assignment.get("bbox_quality_ok") is True
                        and not assignment.get("identity_control_rejected")
                        and proof.get("status") != "reject"
                        and getattr(track, "time_since_update", 1) == 0):
                    self._provisional_association.entries.pop(int(track.track_id), None)
                if int(track.track_id) not in self._provisional_association.entries:
                    continue
            if int(track.track_id) in before:
                tracker.metric.samples[int(track.track_id)] = before[int(track.track_id)]
            else:
                tracker.metric.samples.pop(int(track.track_id), None)
            # Tentative tracks can hold features until confirmation; clear
            # these too. last_feature is current evidence, not learned state.
            track.features = []
            assignment["association_gallery_frozen"] = True
            for observation in self.last_identity_observations:
                if observation.get("raw_track_id") == int(track.track_id):
                    observation["assignment"]["association_gallery_frozen"] = True
            self._update_provisional_association(track, assignment)

    def _update_provisional_association(self, track, assignment):
        raw = int(track.track_id)
        proof = assignment.get("similar_follow") or {}
        cache = self._provisional_association
        if (raw in self.identity_bank._mapped_geometry_conflicts
                or assignment.get("reason") == "identity_center_jump_reject"
                or assignment.get("search_excluded") is True
                or (assignment.get("identity_competition") or {}).get("passed") is False
                or proof.get("reason") in {
                    "identity_blocked", "competition_unverified", "reliable_partial_conflict",
                    "full_distance_conflict", "local_geometry_conflict", "invalid_state"}):
            cache.invalidate(raw, "identity_conflict")
        observations = [row for row in self.last_identity_observations
                        if row.get("raw_track_id") == raw]
        if len(observations) == 1:
            observation = observations[0]
            metadata = observation.get("sample_metadata") or {}
            clock = capture_clock(metadata)
            uid = assignment.get("uid")
            state = self.identity_bank._similar_follow_states.get((uid, raw))
            geometry = assignment.get("reacquire_geometry") or {}
            current_geometry = geometry.get("current") or {}
            crop_accepted = bool(
                proof.get("crop_continuation") is True
                and proof.get("original_quality_reason") == "edge_touch>2"
                and clock is not None and state
                and state.get("uid") == uid and state.get("track_id") == raw
                and state.get("last_cap") == clock[0] and state.get("last_timestamp") == clock[1]
                and current_geometry.get("track_id") == raw
                and capture_clock(current_geometry) == clock
                and cropped_follow_continuous(metadata, state, geometry))
            quality_accepted = bool(crop_accepted or (
                metadata.get("quality_bbox_ok") is True
                and metadata.get("bbox_quality_tier") == "strong"))
            if (getattr(track, "time_since_update", 1) == 0
                    and assignment.get("match_source") == "similar_follow"
                    and proof.get("status") == "follow" and clock is not None
                    and clock == capture_clock(self._frame_context)
                    and proof.get("capture_frame_id") == clock[0]
                    and metadata.get("is_fresh") is True
                    and quality_accepted
                    and assignment.get("bbox_quality_ok") is True
                    and not assignment.get("identity_control_rejected")
                    and not assignment.get("identity_recheck_pending")
                    and assignment.get("search_excluded") is not True
                    and assignment.get("reacquire_geometry_ok") is not False
                    and (assignment.get("identity_competition") or {}).get("passed") is not False
                    and isinstance(uid, int) and not isinstance(uid, bool) and uid > 0
                    and self.identity_bank.track_to_uid.get(raw) == uid
                    and uid in self.identity_bank.identities
                    and raw not in self.identity_bank._mapped_geometry_conflicts
                    and uid not in self.identity_bank._geometry_revoked_uids):
                remembered = cache.remember(track, uid, self.identity_bank.identities[uid],
                                            getattr(track, "last_feature", None), metadata)
                if remembered and crop_accepted:
                    # This current policy acceptance only refreshes local
                    # association evidence; its original strong/crop budget
                    # and both trusted galleries remain unchanged.
                    cache.entries[raw].reason = "accepted_crop_follow"
        assignment["provisional_association"] = cache.diagnostics(raw)
        for observation in observations:
            observation["assignment"]["provisional_association"] = cache.diagnostics(raw)

    def _frame_identity_competition(self, detections, features, *, suppressed_indices=(),
                                    outputs=(), image_width=None, image_height=None):
        uid = self._search_reacquire_uid
        if uid <= 0:
            claims = set(self.identity_bank.track_to_uid.values()) | set(
                getattr(self.identity_bank, "_geometry_revoked_uids", {})
            )
            uid = next(iter(claims)) if len(claims) == 1 else 0
        if uid <= 0:
            return {}
        distances = {
            i: self.reid_distance_to_uid(uid, features[i])
            for i, det in enumerate(detections)
            if int(det.class_id) == 0 and i not in suppressed_indices
        }
        # Snapshot BEFORE this frame's assignments or template updates. Never
        # let the best appearance score nominate its own geometry reference.
        entry = self.identity_bank.identities.get(uid)
        reference = None if entry is None else entry.last_strong_observation
        suspect = self.identity_bank._reacquire_control_suspects.get(uid)
        gap_recovery = bool(
            suspect and suspect.get('reason') == 'evidence_gap' and reference
            and suspect.get('track_id') == reference.get('track_id')
        )
        reference_kind = 'accepted_identity'
        if gap_recovery:
            reference_kind = 'gap_identity_reference'
            local = suspect.get('local_observation')
            # Only IdentityBank can establish a qualified first recovery
            # observation. Do not synthesize one from the current best ReID.
            if (suspect.get('streak', 0) > 0 and local
                    and local.get('track_id') == suspect.get('track_id')):
                reference = local
                reference_kind = 'qualified_gap_observation'
        mapped_indices = {
            o.source_detection_index for o in outputs
            if getattr(o, 'time_since_update', 1) == 0
            and getattr(o, 'source_detection_index', None) in distances
            and (self.identity_bank.track_to_uid.get(o.track_id) == uid
                 or (gap_recovery and suspect.get('track_id') == o.track_id
                     and self.identity_bank.track_to_uid.get(o.track_id, 0) == 0))
            and reference and reference.get('track_id') == o.track_id
            and o.track_id not in getattr(self.identity_bank, '_mapped_geometry_conflicts', {})
            and uid not in getattr(self.identity_bank, '_geometry_revoked_uids', {})
            and (suspect is None or gap_recovery)
        }
        exclusions, anchor_index = anchored_competitor_exclusions(
            detections, distances, reference=reference, context=self._frame_context,
            mapped_indices=mapped_indices, width=image_width, height=image_height,
            confidence_floor=max(self.config.min_confidence, self.config.identity_min_confidence),
            track_confidence_floor=self.config.min_confidence,
            appearance_limit=min(.30, self.identity_bank.config.mapped_verify_threshold),
            weak_only=gap_recovery)
        if exclusions:
            logger.info('identity_competition_eligibility capture_frame_id=%s uid=%s '
                        'reference_cap=%s anchor_detection=%s excluded=%s raw_count=%s '
                        'qualified_count=%s reference_kind=%s detections_retained=True',
                        self._frame_context.get('capture_frame_id'), uid,
                        reference.get('capture_frame_id'), anchor_index, exclusions,
                        len(distances), len(distances)-len(exclusions), reference_kind)
        proof = competition_evidence(
            distances, uid=uid, frame_index=self._frame_index,
            min_margin=self.identity_bank.config.preferred_search_candidate_min_margin,
            exclusions=exclusions,
            eligibility_reference_cap=reference.get('capture_frame_id') if reference else None,
        )
        if exclusions:
            for item in proof.values():
                item['eligibility_reference_kind'] = reference_kind
        return proof

    def _identity_match_allowed(self, track_id, source_index, width, height):
        """Pure pre-association guard: rejected pairs cannot train DeepSORT."""
        review = getattr(self.identity_bank, "review_mapped_geometry", None)
        if not callable(review) or source_index is None or not (
            0 <= int(source_index) < len(self._current_detections)
        ):
            return True
        metadata = dict(self._frame_context)
        metadata.update(self._detector_sample_metadata(
            self._current_detections[int(source_index)].bbox, width, height,
            self.config.identity_edge_margin_ratio,
        ))
        metadata.update(is_fresh=True, track_id=int(track_id))
        geometry = review(int(track_id), metadata, self._frame_index, commit=False)
        allowed = not geometry.get("mapped_geometry_blocked", False)
        if not allowed:
            logger.info(
                "deepsort_identity_pair_rejected frame=%d capture=%s track=%d "
                "source_detection_index=%s reason=%s reference_capture=%s "
                "compensated_jump=%s area_similarity=%s gallery_update=False",
                self._frame_index, metadata.get("capture_frame_id"), track_id, source_index,
                geometry.get("reason"), (geometry.get("reference") or {}).get("capture_frame_id"),
                geometry.get("yaw_compensated_center_jump_ratio"), geometry.get("area_similarity"),
            )
        return allowed

    def _low_score_match_allowed(self, track_id, source_index, width, height):
        """A weak detection may associate, but cannot overrule known conflict."""
        if (source_index is None or not 0 <= int(source_index) < len(self._current_detections)
                or not self.config.identity_bank_enable):
            return False
        detection = self._current_detections[int(source_index)]
        if (int(detection.class_id) != 0
                or not self._bbox_quality(detection.bbox, width, height)[0]):
            return False
        bank = self.identity_bank
        uid = int(bank.track_to_uid.get(int(track_id), 0) or self._search_reacquire_uid or 0)
        if (int(track_id) in bank._mapped_geometry_conflicts
                or uid in bank._geometry_revoked_uids
                or (uid > 0 and bank.search_exclusion_for(int(track_id), uid,
                    frame_index=self._frame_index,
                    capture_timestamp=self._frame_context.get("capture_timestamp")) is not None)):
            return False
        return self._identity_match_allowed(track_id, source_index, width, height)

    def associated_position_contradiction(self, uid, track_id, capture_frame_id, capture_timestamp):
        """Read reliable negative evidence for one previously observed person.

        Empty formal records can hide a rejected low-score observation. Do not
        confuse that transport boundary with absence of an identity conflict.
        Missing features, ordinary UID0/weak quality and expired references
        are unavailable evidence, not a new contradiction. This method neither
        changes identity state nor emits motor commands.
        """
        if (isinstance(uid, bool) or not isinstance(uid, Integral) or uid <= 0
                or isinstance(track_id, bool) or not isinstance(track_id, Integral) or track_id <= 0
                or capture_clock(dict(capture_frame_id=capture_frame_id,
                                      capture_timestamp=capture_timestamp)) is None):
            return None
        bank = self.identity_bank
        if uid in bank._geometry_revoked_uids:
            return "uid_geometry_revoked"
        held = bank._mapped_geometry_conflicts.get(track_id) or {}
        if held.get("uid") == uid:
            return "geometry_conflict"
        if bank.search_exclusion_for(track_id, uid, frame_index=self._frame_index,
                                     capture_timestamp=capture_timestamp) is not None:
            return "identity_excluded"
        for row in self.last_identity_observations:
            if row.get("raw_track_id") != track_id:
                continue
            metadata, assignment = row.get("sample_metadata") or {}, row.get("assignment") or {}
            if (metadata.get("is_fresh") is not True
                    or metadata.get("capture_frame_id") != capture_frame_id
                    or metadata.get("capture_timestamp") != capture_timestamp):
                continue
            competition = metadata.get("identity_competition") or assignment.get("identity_competition") or {}
            if (competition.get("uid") == uid and competition.get("passed") is False
                    and competition.get("frame_index") == row.get("frame_index") == self._frame_index
                    and metadata.get("source_detection_index") is not None
                    and competition.get("source_detection_index") == metadata.get("source_detection_index")
                    and competition.get("reason") == "reid_margin_insufficient"
                    and finite_number(competition.get("distance")) is not None
                    and finite_number(competition.get("competitor_distance")) is not None):
                return "competition_conflict"
            if uid not in (assignment.get("mapped_uid"), assignment.get("uid"),
                           assignment.get("best_uid"), assignment.get("excluded_uid")):
                continue
            if assignment.get("search_excluded") is True:
                return "identity_excluded"
            geometry = assignment.get("reacquire_geometry") or {}
            if (assignment.get("search_contradiction_retained") is True
                    or geometry.get("search_contradiction_retained") is True
                    or geometry.get("mapped_geometry_blocked") is True
                    or assignment.get("reason") in ("mapped_geometry_reject", "identity_center_jump_reject")):
                return "geometry_conflict"
        return None

    def _position_active_uid(self):
        # FULL proof creation is disabled during search and clears the fast
        # lane's UID. The independently locked search UID still owns position.
        return int(self._detector_active_uid or self._search_reacquire_uid or 0)

    def _observe_follow_only_position_anchor(self, output, metadata, assignment):
        """Remember accepted FULL follow permission, never a trusted template."""
        uid, raw = self._position_active_uid(), int(output.track_id)
        bank = self.identity_bank
        previous = getattr(self, "_follow_only_position_anchor", None)
        competition = metadata.get("identity_competition") or {}
        detail = assignment.get("similar_follow") or {}
        current_gallery = finite_number(competition.get("distance"))
        current_confidence = finite_number(metadata.get("detector_confidence"))
        full_conflict = bool(metadata.get("is_fresh") is True
            and not metadata.get("low_score_continuation")
            and metadata.get("quality_bbox_ok") is True
            and competition.get("uid") == uid and competition.get("frame_index") == self._frame_index
            and current_confidence is not None and current_confidence >= .50
            and current_gallery is not None
            and current_gallery > min(.55, bank.config.similar_follow_retain_threshold))
        if previous is not None and (uid != previous["uid"]
                or previous["track_id"] in bank._mapped_geometry_conflicts
                or uid in bank._geometry_revoked_uids
                or (raw == previous["track_id"] and (
                    full_conflict or competition.get("passed") is False or assignment.get("search_excluded")
                    or detail.get("reason") in ("identity_blocked", "competition_unverified",
                        "reliable_partial_conflict", "full_distance_conflict", "local_geometry_conflict")))):
            self._follow_only_position_anchor = None
            self._follow_only_position_last = None
        clock = capture_clock(metadata)
        gallery = finite_number(detail.get("gallery_distance"))
        state = bank._similar_follow_states.get((uid, raw)) or {}
        track = next((t for t in self.deepsort.tracker.tracks if t.track_id == raw), None)
        feature = getattr(output, "feature", None)
        if (uid <= 0 or raw <= 0 or assignment.get("uid") != uid
                or assignment.get("reason") not in ("similar_follow_reacquire", "mapped_similar_follow")
                or assignment.get("identity_permission") != "similar_follow"
                or detail.get("status") != "follow" or detail.get("learning_allowed") is not False
                or detail.get("capture_frame_id") != (clock[0] if clock else None)
                or gallery is None or not 0 <= gallery <= .30
                or assignment.get("bank_updated") is not False
                or assignment.get("reacquire_geometry_ok") is not True
                or any(assignment.get(k) for k in ("identity_control_rejected", "identity_recheck_pending", "search_excluded"))
                or metadata.get("is_fresh") is not True or metadata.get("low_score_continuation")
                or metadata.get("quality_bbox_ok") is not True
                or assignment.get("bbox_quality_ok") is not True
                or clock is None or clock != capture_clock(self._frame_context)
                or competition.get("passed") is not True or competition.get("candidate_count") != 1
                or state.get("active") is not True or state.get("count") != 2 or state.get("position_only")
                or state.get("uid") != uid or state.get("track_id") != raw
                or (state.get("last_cap"), state.get("last_timestamp")) != clock
                or bank.track_to_uid.get(raw) != uid or track is None or feature is None
                or raw in bank._mapped_geometry_conflicts or uid in bank._geometry_revoked_uids):
            return
        entry = bank.identities.get(uid)
        if entry is None or (previous and (clock[0] <= previous["clock"][0] or clock[1] <= previous["clock"][1])):
            return
        self._follow_only_position_anchor = dict(uid=uid, track_id=raw, track=track, entry=entry,
            clock=clock, metadata=dict(metadata, frame_index=self._frame_index), feature=feature.copy(),
            permission="similar_follow")

    def _follow_only_probe_position(self, raw, metadata, assignment, feature):
        """A current unassigned detector box can carry position, not raw/UID."""
        anchor = getattr(self, "_follow_only_position_anchor", None)
        bank, uid = self.identity_bank, self._position_active_uid()
        clock = capture_clock(metadata)
        if anchor is None or raw != -1 or uid != anchor["uid"] or clock is None:
            return None
        owner = anchor["track_id"]
        competition = metadata.get("identity_competition") or {}
        if (owner in bank._mapped_geometry_conflicts or raw in bank._mapped_geometry_conflicts
                or uid in bank._geometry_revoked_uids or uid in bank._reacquire_control_suspects
                or competition.get("passed") is False or assignment.get("search_excluded")
                or assignment.get("search_contradiction_retained")):
            self._follow_only_position_anchor = None
            return None
        previous = getattr(self, "_follow_only_position_last", None)
        distance = finite_number(competition.get("distance"))
        confidence = finite_number(metadata.get("detector_confidence"))
        if (bank.track_to_uid.get(owner) != uid or bank.identities.get(uid) is not anchor["entry"]
                or not any(t is anchor["track"] for t in self.deepsort.tracker.tracks)
                or assignment.get("uid") != 0 or assignment.get("mapped_uid") != uid
                or assignment.get("reason") not in ("secondary_evidence_unavailable", "similar_follow_observe",
                                                      "follow_only_position_probe")
                or metadata.get("is_fresh") is not True or metadata.get("quality_bbox_ok") is not True
                or confidence is None or not .50 <= confidence <= 1.
                or clock != capture_clock(self._frame_context)
                or clock[0] <= anchor["clock"][0] or not 0 < clock[1]-anchor["clock"][1] < .5
                or (previous and (clock[0] <= previous[0] or clock[1] <= previous[1]))
                or competition.get("uid") != uid or competition.get("passed") is not True
                or competition.get("candidate_count") != 1 or metadata.get("candidate_count") != 1
                or competition.get("frame_index") != self._frame_index
                or competition.get("source_detection_index") != metadata.get("source_detection_index")
                or distance is None or not 0 <= distance <= .30
                or bank.search_exclusion_for(owner, uid, frame_index=self._frame_index,
                    capture_timestamp=clock[1]) is not None
                or bank.search_exclusion_for(raw, uid, frame_index=self._frame_index,
                    capture_timestamp=clock[1]) is not None):
            return None
        import numpy as np
        if feature is None or feature.shape != anchor["feature"].shape:
            return None
        norm = float(np.linalg.norm(feature) * np.linalg.norm(anchor["feature"]))
        if not math.isfinite(norm) or norm <= 1e-12:
            return None
        appearance = 1.-float(np.dot(feature, anchor["feature"]))/norm
        geometry = bank._handoff_geometry(uid, metadata, self._frame_index,
            reference_override=bank._handoff_geometry(uid, anchor["metadata"],
                anchor["metadata"]["frame_index"])["current"])
        jump = finite_number(geometry.get("yaw_compensated_center_jump_ratio"))
        area = finite_number(geometry.get("area_similarity"))
        if (not math.isfinite(appearance) or not -.000001 <= appearance <= .35
                or geometry.get("ok") is not True or jump is None or jump > .20
                or area is None or area < .55):
            return None
        self._follow_only_position_last = clock
        return dict(source="follow_only_detector_probe", uid=uid, track_id=raw,
            reference_track_id=owner, capture_frame_id=clock[0], capture_timestamp=clock[1],
            bbox=tuple(metadata["detector_bbox"]), reference_capture_frame_id=anchor["clock"][0],
            reference_capture_timestamp=anchor["clock"][1], expires_at=anchor["clock"][1]+.5,
            gallery_distance=distance, appearance_distance=max(0., appearance),
            center_jump_ratio=jump, area_similarity=area, anchor_permission="similar_follow",
            identity_authorized=False, learning_allowed=False)

    def _low_score_position_evidence(self, output, metadata, assignment):
        """Expose a measured position, never a new UID or a renewable lease.

        The independent anchor is a full, accepted identity observation, not
        DeepSORT's latest high-score box (which may have failed identity).
        Weak frames cannot learn, replace that anchor, or prolong its budget.
        """
        bank, raw = self.identity_bank, int(output.track_id)
        uid = self._position_active_uid()
        weak = metadata.get("low_score_continuation") is True
        anchor = getattr(self, "_low_score_position_anchor", None)
        track = next((item for item in self.deepsort.tracker.tracks if item.track_id == raw), None)
        if weak:
            # An older trusted raw must not mask a newer follow-only owner.
            # Pick a live bound source, never relabel the old raw as current.
            clock = capture_clock(metadata)
            candidates = [item for item in (anchor, getattr(self, "_follow_only_position_anchor", None))
                if item is not None and item["uid"] == uid and item["track_id"] == raw
                and item["track"] is track and bank.identities.get(uid) is item["entry"]
                and bank.track_to_uid.get(raw) == uid and clock is not None
                and clock[0] > item["clock"][0] and 0 < clock[1] - item["clock"][1] <= .5]
            if candidates:
                anchor = max(candidates, key=lambda item: item["clock"][1])
        assigned = assignment.get("uid")
        competition = metadata.get("identity_competition") or {}
        # Only the caller's observation copy receives diagnostics. Do not
        # modify bank.last_assignments or recompute any appearance features.
        diagnostic = dict(reference_capture_frame_id=anchor["clock"][0] if anchor else None,
            gallery_distance=finite_number(competition.get("distance")),
            appearance_distance=None, center_jump_ratio=None, area_similarity=None)
        if weak:
            assignment["low_score_position_diagnostics"] = diagnostic
        def reject(reason):
            if weak:
                assignment["low_score_position_reject_reason"] = reason
            return None
        if competition.get("passed") is False and anchor is not None and anchor["track_id"] == raw:
            self._low_score_position_anchor = None
            self._follow_only_position_anchor = None
            return reject("competition_failed")
        if not weak:
            if anchor is not None and anchor["track_id"] == raw:
                self._low_score_position_anchor = None
            if (uid <= 0 or assigned != uid or bank.track_to_uid.get(raw) != uid
                    or metadata.get("is_fresh") is not True
                    or metadata.get("quality_bbox_ok") is not True
                    or assignment.get("bbox_quality_ok") is not True
                    or assignment.get("bbox_quality_tier") != "strong"
                    or assignment.get("match_source") == "similar_follow"
                    or any(assignment.get(key) for key in (
                        "identity_control_rejected", "identity_recheck_pending",
                        "search_excluded", "template_update_quarantined", "appearance_pose_retention"))
                    or assignment.get("reacquire_geometry_ok") is False
                    or (metadata.get("identity_competition") or {}).get("passed") is not True):
                return None
            clock = capture_clock(metadata)
            feature = getattr(output, "feature", None)
            entry = bank.identities.get(uid)
            if (clock is None or clock != capture_clock(self._frame_context)
                    or feature is None or entry is None or track is None
                    or (anchor is not None and (clock[0] <= anchor["clock"][0]
                        or clock[1] <= anchor["clock"][1]))):
                return None
            self._low_score_position_anchor = dict(uid=uid, track_id=raw, entry=entry, track=track,
                clock=clock, metadata=dict(metadata, frame_index=self._frame_index), feature=feature.copy())
            return None
        if (anchor is None or anchor["uid"] != uid or anchor["track_id"] != raw
                or anchor["track"] is not track
                or bank.identities.get(uid) is not anchor["entry"]
                or bank.track_to_uid.get(raw) != uid):
            return reject("trusted_anchor_unavailable")
        if (assigned != 0
                or assignment.get("reason") not in ("low_score_observation_only", "similar_follow_observe")
                or metadata.get("is_fresh") is not True
                or metadata.get("quality_bbox_ok") is not True
                or metadata.get("association_reason") != "low_score_existing_track"):
            return reject("observation_ineligible")
        if (raw in bank._mapped_geometry_conflicts or uid in bank._geometry_revoked_uids
                or uid in bank._reacquire_control_suspects
                or any(raw in pending for pending in (bank.pending_new, bank.pending_handoffs,
                    bank.pending_late_handoffs, bank.pending_weak_handoffs))
                or bank.search_exclusion_for(raw, uid, frame_index=self._frame_index,
                    capture_timestamp=metadata.get("capture_timestamp")) is not None):
            return reject("identity_review_or_conflict")
        clock = capture_clock(metadata)
        follow_only = anchor.get("permission") == "similar_follow"
        previous = getattr(self, "_follow_only_position_last" if follow_only else "_low_score_position_last", None)
        if (clock is None or clock != capture_clock(self._frame_context)
                or clock[0] <= anchor["clock"][0]
                or not 0 < clock[1] - anchor["clock"][1] <= .5
                or (previous is not None and (clock[0] <= previous[0] or clock[1] <= previous[1]))):
            return reject("capture_stale_or_out_of_order")
        distance = finite_number(competition.get("distance"))
        if (competition.get("uid") != uid or competition.get("frame_index") != self._frame_index
                or competition.get("source_detection_index") != metadata.get("source_detection_index")
                or competition.get("passed") is not True
                or distance is None or not 0 <= distance <= .35):
            return reject("competition_unavailable_or_weak")
        import numpy as np
        feature = getattr(output, "feature", None)
        if feature is None or feature.shape != anchor["feature"].shape:
            return reject("appearance_unavailable")
        norm = float(np.linalg.norm(feature) * np.linalg.norm(anchor["feature"]))
        if not math.isfinite(norm) or norm <= 1e-12:
            return reject("appearance_unavailable")
        appearance = 1. - float(np.dot(feature, anchor["feature"])) / norm
        diagnostic["appearance_distance"] = finite_number(appearance)
        if not math.isfinite(appearance) or not -.000001 <= appearance <= .35:
            return reject("anchor_appearance_mismatch")
        geometry = bank._handoff_geometry(uid, metadata, self._frame_index,
            reference_override=bank._handoff_geometry(uid, anchor["metadata"],
                anchor["metadata"].get("frame_index", self._frame_index))["current"])
        jump = finite_number(geometry.get("yaw_compensated_center_jump_ratio"))
        area = finite_number(geometry.get("area_similarity"))
        diagnostic.update(center_jump_ratio=jump, area_similarity=area,
                          geometry_reason=geometry.get("reason"))
        if geometry.get("ok") is not True or jump is None or jump > .20 or area is None or area < .55:
            return reject("anchor_geometry_rejected")
        self._low_score_position_last = clock
        if follow_only:
            self._follow_only_position_last = clock
        return dict(source="low_score_existing_track", uid=uid, track_id=raw,
            capture_frame_id=clock[0], capture_timestamp=clock[1], bbox=tuple(metadata["detector_bbox"]),
            reference_capture_frame_id=anchor["clock"][0], reference_capture_timestamp=anchor["clock"][1],
            expires_at=anchor["clock"][1] + .5, appearance_distance=max(0., appearance),
            center_jump_ratio=jump, area_similarity=area,
            anchor_permission="similar_follow" if follow_only else "strong",
            identity_authorized=False, learning_allowed=False)

    def _observe_identity_frame_evidence(
        self, outputs: Sequence[Any], image_width: int, image_height: Optional[int],
        *, duplicate_track_ids: Sequence[int] = (),
        identity_swap_track_ids: Sequence[int] = (),
    ) -> None:
        """Register current detector geometry before any per-track UID write."""
        observe = getattr(self.identity_bank, "observe_frame_evidence", None)
        if not callable(observe):
            return
        observations = []
        for output in outputs:
            source_index = getattr(output, "source_detection_index", None)
            if (
                int(output.time_since_update) != 0
                or getattr(output, "low_score_continuation", False)
                or source_index is None
                or not 0 <= int(source_index) < len(self._current_detections)
                or int(output.class_id) != 0
            ):
                continue
            detection = self._current_detections[int(source_index)]
            track_id = int(output.track_id)
            previous_center = self._last_identity_center_by_track_id.get(track_id)
            previous_frame = self._last_identity_center_frame_by_track_id.get(track_id)
            center_jump = bool(
                image_width > 0
                and previous_center is not None
                and previous_frame is not None
                and 0 < self._frame_index - previous_frame
                <= max(1, int(self.config.identity_center_jump_max_gap_frames))
                and abs(
                    (float(output.x1) + float(output.x2)) / (2.0 * image_width)
                    - float(previous_center)
                ) > max(0.05, min(1.0, float(self.config.identity_max_center_jump_ratio)))
            )
            observations.append({
                "raw_track_id": track_id,
                "detector_bbox": tuple(detection.bbox),
                "feature": getattr(output, "feature", None),
                "confidence": float(detection.score),
                "mapped_uid": int(self.identity_bank.track_to_uid.get(track_id, 0)),
                "capture_frame_id": self._frame_context.get("capture_frame_id"),
                "capture_timestamp": self._frame_context.get("capture_timestamp"),
                "integrated_yaw_deg": self._frame_context.get("integrated_yaw_deg"),
                "is_fresh": True,
                "duplicate": track_id in duplicate_track_ids,
                "identity_swap": track_id in identity_swap_track_ids or center_jump,
                "identity_competition": self._identity_competition.get(int(source_index)),
            })
        observe(
            frame_index=int(self._frame_index), observations=observations,
            width=image_width, height=image_height,
        )

    def set_search_reacquire_context(
        self,
        *,
        active_uid: Optional[int],
        searching: bool,
        direction: Optional[str],
    ) -> None:
        uid = 0 if active_uid is None else int(active_uid)
        self._detector_active(uid)
        if searching:
            self._clear_detector_continuation("searching")
        normalized_direction = str(direction or "").strip().lower()
        if not searching or uid <= 0 or normalized_direction not in ("left", "right"):
            if uid <= 0:
                self._follow_only_position_anchor = None
                self._follow_only_position_last = None
            self._search_reacquire_uid = 0
            self._search_reacquire_direction = None
            self._search_reacquire_eligible_tracks.clear()
            self.identity_bank.track_to_uid.pop(self._search_probe_track_id, None)
            self.identity_bank.track_last_seen_frame.pop(self._search_probe_track_id, None)
            self.identity_bank.pending_handoffs.pop(self._search_probe_track_id, None)
            self.identity_bank.pending_late_handoffs.pop(self._search_probe_track_id, None)
            self.identity_bank.pending_weak_handoffs.pop(self._search_probe_track_id, None)
            self._search_probe_bbox = None
            self._search_probe_frame = -1
            return
        if (
            uid != self._search_reacquire_uid
            or normalized_direction != self._search_reacquire_direction
        ):
            self._search_reacquire_eligible_tracks.clear()
        self._search_reacquire_uid = uid
        self._search_reacquire_direction = normalized_direction

    def reset(self) -> None:
        self._clear_detector_continuation("reset")
        self._low_score_position_anchor = None
        self._low_score_position_last = None
        self._follow_only_position_anchor = None
        self._follow_only_position_last = None
        self._detector_active_uid = 0
        self.deepsort.reset()
        self.identity_bank.reset()
        self._identity_competition = {}
        self._frame_index = 0
        self._search_reacquire_uid = 0
        self._search_reacquire_direction = None
        self._search_reacquire_eligible_tracks.clear()
        self._search_probe_bbox = None
        self._search_probe_frame = -1
        self._last_quality_area_by_track_id.clear()
        self._last_identity_center_by_track_id.clear()
        self._last_identity_center_frame_by_track_id.clear()
        self._frame_context = {}
        self._current_detections = ()
        self.last_identity_observations = []
        self._provisional_association.entries.clear()

    def debug_state(self) -> List[dict]:
        samples = self.deepsort.tracker.metric.samples
        state = [
            {
                "next_track_id": int(self.deepsort.tracker._next_id),
                "feature_update_interval": int(self.config.feature_update_interval),
                "max_output_age": int(self.config.max_output_age),
                "active_tracks": int(len(self.deepsort.tracker.tracks)),
                "identity_bank": self.identity_bank.debug_state(),
            }
        ]
        for track in self.deepsort.tracker.tracks:
            x1, y1, x2, y2 = [float(v) for v in track.to_tlbr()]
            state.append(
                {
                    "track_id": int(track.track_id),
                    "state": int(track.state),
                    "state_name": _track_state_name(int(track.state)),
                    "hits": int(track.hits),
                    "age": int(track.age),
                    "time_since_update": int(track.time_since_update),
                    "class_id": int(track.cls),
                    "confidence": float(track.confidence),
                    "pending_features": int(len(track.features)),
                    "gallery_features": int(len(samples.get(int(track.track_id), []))),
                    "provisional_association": self._provisional_association.diagnostics(int(track.track_id)),
                    "reid_uid": int(self.identity_bank.track_to_uid.get(int(track.track_id), 0)),
                    "bbox": [x1, y1, x2, y2],
                }
            )
        return state

    def reid_distance_to_uid(self, uid: int, feature: Any) -> Optional[float]:
        if not self.config.identity_bank_enable or int(uid) <= 0 or feature is None:
            return None
        return self.identity_bank.distance_to_uid(int(uid), feature)

    def _to_record(
        self,
        output,
        image_width: int,
        image_height: Optional[int],
        candidate_count: int,
        partial_features: Sequence[Optional[Any]],
        partial_feature_sources: Sequence[Optional[str]] = (),
        color_features: Sequence[Optional[Any]] = (),
        candidate_score_gap: Optional[float] = None,
        duplicate_identity_box: bool = False,
        identity_swap_track: bool = False,
    ) -> TrackRecord:
        x1, y1, x2, y2 = output.x1, output.y1, output.x2, output.y2
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        angle = 0.0
        if image_width > 0:
            norm = (cx - float(image_width) / 2.0) / (float(image_width) / 2.0)
            angle = norm * (float(self.config.hfov_deg) / 2.0)
        output_track_id = int(output.track_id)
        output_is_fresh = int(output.time_since_update) == 0
        source_index = getattr(output, "source_detection_index", None)
        detector_bbox = None
        partial_feature = None
        partial_feature_source = None
        if (
            output_is_fresh
            and source_index is not None
            and 0 <= int(source_index) < len(self._current_detections)
        ):
            detector_bbox = tuple(self._current_detections[int(source_index)].bbox)
            if 0 <= int(source_index) < len(partial_features):
                partial_feature = partial_features[int(source_index)]
            if 0 <= int(source_index) < len(partial_feature_sources):
                partial_feature_source = partial_feature_sources[int(source_index)]

        # DeepSORT receives an expanded box for association.  That box can be
        # clipped by the image boundary even when the original YOLO box was a
        # valid, complete person detection (for example, the expansion of a
        # box near the top edge).  Identity quality must describe the evidence
        # actually cropped for ReID, so use the fresh detector box whenever its
        # provenance is available.  Keep the expanded track box for display,
        # motion and association, and expose its quality separately below.
        quality_bbox = detector_bbox or (
            float(x1), float(y1), float(x2), float(y2)
        )
        bbox_quality_ok, bbox_quality_reason = self._bbox_quality(
            quality_bbox,
            image_width,
            image_height,
        )
        display_bbox_quality_ok, display_bbox_quality_reason = self._bbox_quality(
            (float(x1), float(y1), float(x2), float(y2)),
            image_width,
            image_height,
        )
        previous_center = self._last_identity_center_by_track_id.get(output_track_id)
        previous_center_frame = self._last_identity_center_frame_by_track_id.get(output_track_id)
        center_transition_ok, center_transition_reason = self._bbox_center_transition_quality(
            output_track_id,
            cx,
            image_width=image_width,
            is_fresh=output_is_fresh,
        )
        if not center_transition_ok:
            bbox_quality_ok = False
            bbox_quality_reason = ",".join(
                item
                for item in (bbox_quality_reason, center_transition_reason)
                if item
            )
        if identity_swap_track:
            bbox_quality_ok = False
            bbox_quality_reason = ",".join(
                item
                for item in (bbox_quality_reason, "identity_swap_competing_track")
                if item
            )
        transition_ok, transition_reason = self._bbox_transition_quality(
            output_track_id,
            area,
            is_fresh=int(output.time_since_update) == 0,
            base_quality_ok=bbox_quality_ok and not duplicate_identity_box,
        )
        if not transition_ok:
            bbox_quality_ok = False
            bbox_quality_reason = ",".join(
                item for item in (bbox_quality_reason, transition_reason) if item
            )
        if duplicate_identity_box:
            bbox_quality_ok = False
            bbox_quality_reason = "duplicate_person_box"
        bbox_quality_tier = self._bbox_identity_tier(
            bbox_quality_ok=bbox_quality_ok,
            bbox_quality_reason=bbox_quality_reason,
            confidence=float(output.confidence),
            bbox=quality_bbox,
        )
        sample_metadata = self._identity_sample_metadata(
            track_id=output_track_id,
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            confidence=float(output.confidence),
            image_width=image_width,
            image_height=image_height,
            bbox_quality_tier=bbox_quality_tier,
            bbox_quality_reason=bbox_quality_reason,
            is_fresh=output_is_fresh,
        )
        sample_metadata.update(self._frame_context)
        # These fields may only come from this call's detector association.
        sample_metadata.pop("initial_color_feature", None)
        sample_metadata.pop("initial_color_source", None)
        sample_metadata["is_fresh"] = output_is_fresh
        # Association proof is current detector evidence, not identity proof.
        # Do not accept these flags from caller-provided frame context.
        sample_metadata["low_score_continuation"] = bool(
            output_is_fresh and getattr(output, "low_score_continuation", False))
        if sample_metadata["low_score_continuation"]:
            sample_metadata.update(association_reason="low_score_existing_track",
                association_confidence_limit=float(self.config.min_confidence),
                association_previous_capture_frame_id=getattr(
                    output, "association_previous_capture_frame_id", None),
                association_previous_capture_timestamp=getattr(
                    output, "association_previous_capture_timestamp", None))
            logger.info("deepsort_low_score_continuation capture_frame_id=%s raw_track_id=%s "
                "source_detection_index=%s score=%.3f reference_capture=%s "
                "association_only=True identity_authorized=False gallery_update=False",
                self._frame_context.get("capture_frame_id"), output_track_id, source_index,
                float(output.confidence), sample_metadata["association_previous_capture_frame_id"])
        else:
            for field in ("association_reason", "association_previous_capture_frame_id",
                          "association_previous_capture_timestamp", "association_confidence_limit"):
                sample_metadata.pop(field, None)
        sample_metadata["template_learning_risk"] = _template_learning_risk(
            self._current_detections, source_index, is_fresh=output_is_fresh,
            all_detections=getattr(self, "_learning_detections", None),
        )
        sample_metadata["quality_bbox_source"] = (
            "detector" if detector_bbox is not None else "track"
        )
        sample_metadata["quality_bbox"] = list(quality_bbox)
        sample_metadata["quality_bbox_ok"] = bool(bbox_quality_ok)
        sample_metadata["quality_bbox_reason"] = str(bbox_quality_reason or "")
        sample_metadata["display_bbox_quality_ok"] = bool(display_bbox_quality_ok)
        sample_metadata["display_bbox_quality_reason"] = str(
            display_bbox_quality_reason or ""
        )
        sample_metadata["candidate_count"] = int(candidate_count)
        if candidate_score_gap is not None:
            sample_metadata["candidate_score_gap"] = float(candidate_score_gap)
        # The loss controller freezes the motor sweep direction, but a real
        # target may cross the optical center while that sweep is in progress.
        # Preserve that distinction for the identity bank: an opposite-side
        # observation may enter the strong-ReID confirmation path, while the
        # controller continues using the existing frozen direction until the
        # handoff is actually confirmed.
        search_context_active = bool(
            output_is_fresh and self._search_reacquire_uid > 0
        )
        search_center_x = cx
        if detector_bbox is not None:
            search_center_x = (float(detector_bbox[0]) + float(detector_bbox[2])) * 0.5
        sample_metadata["search_reacquire_context_active"] = search_context_active
        sample_metadata["search_direction"] = (
            self._search_reacquire_direction if search_context_active else None
        )
        sample_metadata["search_direction_compatible"] = (
            None
            if not search_context_active
            else bool(self._search_candidate_direction_ok(search_center_x, image_width))
        )
        sample_metadata["partial_observation"] = bool(
            partial_feature is not None
            and _partial_bbox_observation(
                (float(x1), float(y1), float(x2), float(y2)),
                image_width,
                image_height,
                self.config.identity_edge_margin_ratio,
            )
        )
        if partial_feature_source:
            sample_metadata["partial_feature_source"] = str(partial_feature_source)
        if detector_bbox is not None:
            sample_metadata.update(
                self._detector_sample_metadata(
                    detector_bbox,
                    image_width,
                    image_height,
                    self.config.identity_edge_margin_ratio,
                )
            )
            sample_metadata["source_detection_index"] = int(source_index)
            if (not self.identity_bank.identities
                    and 0 <= int(source_index) < len(color_features)):
                try:
                    color = [float(v) for v in color_features[int(source_index)]]
                    color_norm = math.sqrt(sum(v * v for v in color))
                    # The fallback six-dimensional BGR statistics are not an
                    # HSV histogram and must not carry this versioned source.
                    if (len(color) == 16 and all(math.isfinite(v) and v >= 0. for v in color)
                            and math.isfinite(color_norm) and color_norm > 1e-12):
                        sample_metadata["initial_color_feature"] = color
                        sample_metadata["initial_color_source"] = "hsv_crop_v1_bgr"
                except (TypeError, ValueError, OverflowError):
                    pass
            sample_metadata["partial_observation"] = bool(
                partial_feature is not None
                and _partial_bbox_observation(
                    detector_bbox,
                    image_width,
                    image_height,
                    self.config.identity_edge_margin_ratio,
                )
            )
        if not center_transition_ok:
            # Keep comparing against the last trusted position. If the jumped
            # center became the new baseline, the next frame would immediately
            # re-accept the swapped identity.
            if previous_center is None:
                self._last_identity_center_by_track_id.pop(output_track_id, None)
            else:
                self._last_identity_center_by_track_id[output_track_id] = previous_center
            if previous_center_frame is None:
                self._last_identity_center_frame_by_track_id.pop(output_track_id, None)
            else:
                self._last_identity_center_frame_by_track_id[output_track_id] = previous_center_frame
        if output_is_fresh and self._search_candidate_direction_ok(search_center_x, image_width):
            self._search_reacquire_eligible_tracks.add(output_track_id)
        preferred_candidate_ok = (
            output_is_fresh
            and output_track_id in self._search_reacquire_eligible_tracks
        )
        if source_index in self._identity_competition:
            sample_metadata["identity_competition"] = dict(self._identity_competition[source_index])
        reid_uid = self.identity_bank.assign(
            track_id=output_track_id,
            feature=output.feature,
            partial_feature=partial_feature,
            confidence=float(output.confidence),
            area=float(area),
            frame_index=int(self._frame_index),
            candidate_count=int(candidate_count),
            bbox_quality_ok=bbox_quality_ok,
            bbox_quality_reason=bbox_quality_reason,
            bbox_quality_tier=bbox_quality_tier,
            sample_metadata=sample_metadata,
            preferred_uid=self._search_reacquire_uid,
            preferred_candidate_ok=preferred_candidate_ok,
            detector_position_bridge=getattr(self, "_current_detector_position_bridge", None),
        )
        if output_is_fresh and detector_bbox is not None:
            assignment = dict(self.identity_bank.last_assignments.get(output_track_id, {}))
            self._observe_follow_only_position_anchor(output, sample_metadata, assignment)
            position = self._low_score_position_evidence(output, sample_metadata, assignment)
            if position is not None:
                assignment["low_score_position_evidence"] = position
            if sample_metadata.get("low_score_continuation") is True:
                diagnostic = assignment.get("low_score_position_diagnostics") or {}
                logger.info("low_score_position_evidence capture_frame_id=%s raw_track_id=%s "
                    "accepted=%s reason=%s reference_capture=%s gallery_distance=%s "
                    "anchor_distance=%s center_jump=%s expires_at=%s "
                    "identity_authorized=False learning_allowed=False",
                    sample_metadata.get("capture_frame_id"), output_track_id, position is not None,
                    assignment.get("low_score_position_reject_reason", "accepted"),
                    diagnostic.get("reference_capture_frame_id"), diagnostic.get("gallery_distance"),
                    diagnostic.get("appearance_distance"), diagnostic.get("center_jump_ratio"),
                    position.get("expires_at") if position is not None else None)
            self.last_identity_observations.append({
                "frame_index": int(self._frame_index),
                "raw_track_id": output_track_id,
                "uid": int(reid_uid),
                "detector_bbox": detector_bbox,
                "display_bbox": (float(x1), float(y1), float(x2), float(y2)),
                "sample_metadata": sample_metadata,
                "assignment": assignment,
            })
        return TrackRecord(
            track_id=int(output.track_id),
            reid_uid=int(reid_uid),
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
            class_id=int(output.class_id),
            score=float(output.confidence),
            cx=float(cx),
            cy=float(cy),
            area=float(area),
            angle_deg=float(angle),
            tracker_state=_map_state(int(output.state)),
            time_since_update=int(output.time_since_update),
        )

    @staticmethod
    def _candidate_score_gap(
        source_detection_index: Optional[int], detections: Sequence[Detection]
    ) -> Optional[float]:
        """Return current-vs-next-best detector confidence for competition gating."""
        if source_detection_index is None:
            return None
        index = int(source_detection_index)
        if index < 0 or index >= len(detections):
            return None
        current = detections[index]
        if int(current.class_id) != 0:
            return None
        competitors = [
            float(det.score)
            for idx, det in enumerate(detections)
            if idx != index and int(det.class_id) == 0
        ]
        if not competitors:
            return float(current.score)
        return float(current.score) - max(competitors)

    def _normal_follow_position_probe(self, detections, features, *, image_width, image_height):
        """Use a current unassigned crop after search release, without assign().

        Association may temporarily have no output even after a completed
        follow-only confirmation. Search direction is no longer meaningful;
        only the independently bound finite position anchor can admit this
        observation. No IdentityBank mutation or artificial search is needed.
        """
        anchor = getattr(self, "_follow_only_position_anchor", None)
        if (anchor is None or self._detector_active_uid != anchor["uid"]
                or self._search_reacquire_direction or image_width <= 0
                or image_height is None or image_height <= 0):
            return None
        people = [i for i, detection in enumerate(detections) if int(detection.class_id) == 0]
        if len(people) != 1:
            return None
        index = people[0]
        detection, descriptor = detections[index], features[index]
        confidence = finite_number(detection.score)
        box = tuple(float(value) for value in detection.bbox)
        quality, reason = self._bbox_quality(box, image_width, image_height)
        if confidence is None or not .50 <= confidence <= 1. or descriptor is None or not quality:
            return None
        raw, uid = self._search_probe_track_id, anchor["uid"]
        # is_fresh=False here avoids changing raw-track direction history.
        # The returned metadata explicitly describes the new detector crop.
        metadata = self._identity_sample_metadata(track_id=raw, bbox=box,
            confidence=confidence, image_width=image_width, image_height=image_height,
            bbox_quality_tier="strong", bbox_quality_reason=reason, is_fresh=False)
        metadata.update(self._frame_context)
        metadata.update(self._detector_sample_metadata(box, image_width, image_height,
                                                       self.config.identity_edge_margin_ratio))
        metadata.update(is_fresh=True, source_detection_index=index, candidate_count=1,
            candidate_score_gap=confidence, quality_bbox_source="detector", quality_bbox=list(box),
            quality_bbox_ok=True, quality_bbox_reason=reason, search_reacquire_context_active=False,
            search_observation_only=False, low_score_continuation=False,
            identity_competition=self._frame_identity_competition(detections, features).get(index, {}))
        assignment = dict(uid=0, mapped_uid=uid, best_uid=uid,
            reason="follow_only_position_probe", bank_updated=False, identity_permission="position_only")
        proof = self._follow_only_probe_position(raw, metadata, assignment, descriptor)
        if proof is None:
            return None
        assignment["follow_only_position_evidence"] = proof
        self.last_identity_observations.append(dict(frame_index=self._frame_index,
            raw_track_id=raw, uid=0, detector_bbox=box, display_bbox=box,
            sample_metadata=metadata, assignment=assignment))
        logger.info("follow_only_position_probe capture_frame_id=%s raw=%s reference_raw=%s "
            "uid=0 reference_uid=%s reference_capture=%s expires_at=%s "
            "bank_updated=False identity_renewed=False search_started=False",
            proof["capture_frame_id"], raw, anchor["track_id"], uid,
            proof["reference_capture_frame_id"], proof["expires_at"])
        x1, y1, x2, y2 = box
        return TrackRecord(track_id=raw, reid_uid=0, x1=x1, y1=y1, x2=x2, y2=y2,
            class_id=int(detection.class_id), score=confidence, cx=(x1+x2)*.5, cy=(y1+y2)*.5,
            area=(x2-x1)*(y2-y1), angle_deg=0., tracker_state=TRACK_STATE_STABLE, time_since_update=0)

    def _search_probe_record(
        self,
        detections: Sequence[Detection],
        features: Sequence[Optional[Any]],
        *,
        partial_features: Sequence[Optional[Any]],
        partial_feature_sources: Sequence[Optional[str]] = (),
        image_width: int,
        image_height: Optional[int],
    ) -> Optional[TrackRecord]:
        """Use one detector box as locked-UID evidence during a tracker gap."""
        if self._search_reacquire_uid <= 0 or not self._search_reacquire_direction:
            return self._normal_follow_position_probe(detections, features,
                image_width=image_width, image_height=image_height)
        # No formal assignments run on this path. Recompute here as this
        # entry point is also called independently by diagnostic/test callers.
        probe_competition = self._frame_identity_competition(detections, features)
        # This method is reached only when DeepSORT has no output and an
        # active UID is already being searched.  Use the search-only detector
        # floor here as well; otherwise a candidate such as CAP406 (conf
        # 0.573) is discarded before IdentityBank can apply its strict
        # ReID/geometry/confirmation gates.  Normal tracking still uses the
        # global identity_min_confidence in DeepSORT and IdentityBank.
        search_confidence_floor = min(
            float(self.config.identity_min_confidence),
            max(0.0, min(1.0, float(
                self.config.identity_preferred_search_reacquire_min_confidence
            ))),
        )
        observation_confidence_floor = min(
            float(search_confidence_floor),
            max(
                0.0,
                min(
                    1.0,
                    float(
                        self.config.identity_preferred_search_reacquire_observation_min_confidence
                    ),
                ),
            ),
        )

        def _area_ratio(detection: Detection) -> float:
            x1, y1, x2, y2 = (float(value) for value in detection.bbox)
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            image_area = float(image_width * max(1, int(image_height or 1)))
            return area / max(image_area, 1.0)

        person_indices = [
            index
            for index, detection in enumerate(detections)
            if int(detection.class_id) == 0
            and (
                float(detection.score) >= search_confidence_floor
                or (
                    float(detection.score) >= observation_confidence_floor
                    and _area_ratio(detection)
                    >= max(
                        0.0,
                        float(self.config.identity_preferred_search_soft_min_area_ratio),
                    )
                )
            )
            and features[index] is not None
        ]
        if len(person_indices) != 1 or image_width <= 0:
            self._search_probe_bbox = None
            self._search_probe_frame = -1
            self.identity_bank.pending_late_handoffs.pop(self._search_probe_track_id, None)
            return None
        detection_index = person_indices[0]
        detection = detections[detection_index]
        observation_only = float(detection.score) < float(search_confidence_floor)
        bbox = tuple(float(value) for value in detection.bbox)
        x1, y1, x2, y2 = bbox
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        area = box_width * box_height
        image_area = float(image_width * max(1, int(image_height or 1)))
        area_ratio = area / image_area
        if (
            box_width < float(self.config.identity_min_width_px)
            or box_height < float(self.config.identity_min_height_px)
            or area < float(self.config.identity_min_area)
            or area_ratio > float(self.config.identity_max_area_ratio)
            or lower_compact_bbox_reason(bbox, image_width, image_height)
        ):
            self._search_probe_bbox = None
            self._search_probe_frame = -1
            self.identity_bank.pending_late_handoffs.pop(self._search_probe_track_id, None)
            return None
        center_ratio = max(0.0, min(1.0, ((x1 + x2) * 0.5) / float(image_width)))
        if self._search_probe_bbox is not None and self._search_probe_frame >= 0:
            frame_gap = int(self._frame_index) - int(self._search_probe_frame)
            previous_center = (self._search_probe_bbox[0] + self._search_probe_bbox[2]) * 0.5
            center_jump = abs(((x1 + x2) * 0.5) - previous_center) / float(image_width)
            if frame_gap > 2 or center_jump > 0.25:
                self.identity_bank.pending_handoffs.pop(self._search_probe_track_id, None)
                self.identity_bank.pending_late_handoffs.pop(self._search_probe_track_id, None)
        self._search_probe_bbox = bbox
        self._search_probe_frame = int(self._frame_index)
        sample_metadata = self._identity_sample_metadata(
            track_id=self._search_probe_track_id,
            bbox=bbox,
            confidence=float(detection.score),
            image_width=image_width,
            image_height=image_height,
            bbox_quality_tier="strong",
            bbox_quality_reason="search_detector_probe",
            is_fresh=True,
        )
        probe_quality_ok, probe_quality_reason = self._bbox_quality(
            bbox, image_width, image_height
        )
        probe_quality_tier = self._bbox_identity_tier(
            bbox_quality_ok=probe_quality_ok,
            bbox_quality_reason=probe_quality_reason,
            confidence=float(detection.score),
            bbox=bbox,
        )
        # Keep a large low-score person box as weak identity evidence.  The
        # IdentityBank receives an explicit search-observation marker and
        # continues to hold output UID at zero until its existing weak
        # two-frame handoff is complete.
        observation_large = bool(
            observation_only
            and area_ratio
            >= max(
                0.0,
                float(self.config.identity_preferred_search_soft_min_area_ratio),
            )
        )
        if observation_large and probe_quality_tier == "reject":
            probe_quality_tier = "weak"
        sample_metadata["bbox_quality_tier"] = probe_quality_tier
        sample_metadata["bbox_quality_reason"] = probe_quality_reason
        sample_metadata.update(self._frame_context)
        sample_metadata.update(
            self._detector_sample_metadata(
                bbox,
                image_width,
                image_height,
                self.config.identity_edge_margin_ratio,
            )
        )
        # Same quality contract as formal track records. A detector-only
        # probe has no filtered display box; report the actual crop result,
        # never promote a weak/rejected crop to a passing one.
        sample_metadata.update(
            quality_bbox_source="detector", quality_bbox=list(bbox),
            quality_bbox_ok=bool(probe_quality_ok),
            quality_bbox_reason=str(probe_quality_reason or ""),
            display_bbox_quality_ok=bool(probe_quality_ok),
            display_bbox_quality_reason=str(probe_quality_reason or ""),
        )
        sample_metadata["is_fresh"] = True
        sample_metadata["source_detection_index"] = detection_index
        sample_metadata["template_learning_risk"] = _template_learning_risk(
            detections, detection_index, is_fresh=True,
            all_detections=(getattr(self, "_learning_detections", None)
                if len(detections) == len(self._current_detections)
                and all(a is b for a, b in zip(detections, self._current_detections))
                else None),
        )
        sample_metadata["identity_competition"] = probe_competition.get(detection_index, {})
        if 0 <= detection_index < len(partial_feature_sources):
            source = partial_feature_sources[detection_index]
            if source:
                sample_metadata["partial_feature_source"] = str(source)
        # Probe eligibility selects one useful crop, but competition still
        # includes every person in the current detector frame. Keep the count
        # bound to those detections, not to a proof's self-reported value.
        candidate_count = sum(int(item.class_id) == 0 for item in detections)
        sample_metadata["candidate_count"] = candidate_count
        sample_metadata["candidate_score_gap"] = float(detection.score)
        # Detector-only probes are part of the active preferred-UID search
        # path. Mark that context explicitly so IdentityBank can apply the
        # search-only confidence floor without weakening normal assignment.
        sample_metadata["search_reacquire_context_active"] = True
        sample_metadata["search_observation_only"] = bool(observation_large)
        sample_metadata["search_observation_large"] = bool(observation_large)
        sample_metadata["search_direction"] = self._search_reacquire_direction
        sample_metadata["search_direction_compatible"] = bool(
            self._search_candidate_direction_ok(
                (x1 + x2) * 0.5,
                image_width,
            )
        )
        sample_metadata["partial_observation"] = bool(
            partial_features[detection_index] is not None
            and _partial_bbox_observation(
                bbox,
                image_width,
                image_height,
                self.config.identity_edge_margin_ratio,
            )
        )
        observe = getattr(self.identity_bank, "observe_frame_evidence", None)
        if callable(observe):
            observe(
                frame_index=int(self._frame_index),
                observations=[{
                    "raw_track_id": int(self._search_probe_track_id),
                    "detector_bbox": bbox,
                    "feature": features[detection_index],
                    "confidence": float(detection.score),
                    "mapped_uid": int(self.identity_bank.track_to_uid.get(self._search_probe_track_id, 0)),
                    "capture_frame_id": self._frame_context.get("capture_frame_id"),
                    "capture_timestamp": self._frame_context.get("capture_timestamp"),
                    "integrated_yaw_deg": self._frame_context.get("integrated_yaw_deg"),
                    "is_fresh": True,
                    "duplicate": False,
                    "identity_swap": False,
                }],
                width=image_width, height=image_height,
            )
        uid = self.identity_bank.assign(
            track_id=self._search_probe_track_id,
            feature=features[detection_index],
            partial_feature=partial_features[detection_index],
            confidence=float(detection.score),
            area=float(area),
            frame_index=int(self._frame_index),
            candidate_count=candidate_count,
            bbox_quality_ok=probe_quality_ok,
            bbox_quality_tier=probe_quality_tier,
            bbox_quality_reason=probe_quality_reason or "search_detector_probe",
            sample_metadata=sample_metadata,
            preferred_uid=int(self._search_reacquire_uid),
            preferred_candidate_ok=True,
        )
        assignment = dict(self.identity_bank.last_assignments.get(self._search_probe_track_id, {}))
        position = self._follow_only_probe_position(self._search_probe_track_id, sample_metadata,
                                                   assignment, features[detection_index])
        if position is not None:
            assignment["follow_only_position_evidence"] = position
        self.last_identity_observations.append({
            "frame_index": int(self._frame_index),
            "raw_track_id": int(self._search_probe_track_id),
            "uid": int(uid),
            "detector_bbox": bbox,
            "display_bbox": bbox,
            "sample_metadata": sample_metadata,
            "assignment": dict(assignment),
        })
        logger.info(
            "search_detector_probe_reid frame=%d uid=%d output_uid=%d distance=%s "
            "streak=%d/%d center=%.3f reason=%s",
            int(self._frame_index),
            int(self._search_reacquire_uid),
            int(uid),
            "none" if assignment.get("distance") is None else "%.3f" % float(assignment["distance"]),
            int(assignment.get("handoff_streak", 0)),
            int(self.config.identity_preferred_search_reacquire_confirm_frames),
            center_ratio,
            assignment.get("reason", "unknown"),
        )
        return TrackRecord(
            track_id=int(self._search_probe_track_id),
            reid_uid=int(uid),
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            class_id=int(detection.class_id),
            score=float(detection.score),
            cx=(x1 + x2) * 0.5,
            cy=(y1 + y2) * 0.5,
            area=float(area),
            angle_deg=0.0,
            tracker_state=TRACK_STATE_STABLE,
            time_since_update=0,
        )

    @staticmethod
    def _detector_sample_metadata(
        bbox, image_width, image_height, edge_margin_ratio: float = 0.02
    ) -> dict:
        x1, y1, x2, y2 = [float(value) for value in bbox]
        margin_x = float(image_width) * max(0.0, float(edge_margin_ratio))
        margin_y = float(image_height or 0) * max(0.0, float(edge_margin_ratio))
        edge_touch_count = int(x1 <= margin_x) + int(float(image_width) - x2 <= margin_x)
        if image_height is not None and image_height > 0:
            edge_touch_count += int(y1 <= margin_y) + int(float(image_height) - y2 <= margin_y)
        return {
            "detector_bbox": [x1, y1, x2, y2],
            "image_width": image_width,
            "image_height": image_height,
            "detector_center_x_ratio": (x1 + x2) * 0.5 / max(1.0, float(image_width)),
            "detector_area_ratio": max(0.0, x2 - x1) * max(0.0, y2 - y1)
            / max(1.0, float(image_width * (image_height or 1))),
            "detector_edge_touch_count": int(edge_touch_count),
        }

    def _identity_swap_track_ids(
        self,
        outputs: Sequence[Any],
        image_width: int,
    ) -> Set[int]:
        """Find mapped tracks that jumped onto another fresh track's old spot.

        DeepSORT can preserve a raw track id through a person crossing even
        when the box has actually moved to the other person. If an unassigned
        fresh track remains near the mapped track's previous center, the mapped
        track is the likely swap and must not keep publishing its UID.
        """
        if image_width <= 0:
            return set()
        mapped = self.identity_bank.track_to_uid
        fresh = [
            output
            for output in outputs
            if int(output.time_since_update) == 0
        ]
        if len(fresh) < 2:
            return set()
        max_jump = max(
            0.03,
            min(0.50, float(self.config.identity_swap_min_mapped_jump_ratio)),
        )
        replacement_distance = max(
            0.05,
            min(0.35, float(self.config.identity_swap_max_replacement_distance_ratio)),
        )
        result: Set[int] = set()
        centers = {
            int(output.track_id): (float(output.x1) + float(output.x2)) / 2.0 / float(image_width)
            for output in fresh
        }
        for output in fresh:
            track_id = int(output.track_id)
            if int(mapped.get(track_id, 0)) <= 0:
                continue
            previous = self._last_identity_center_by_track_id.get(track_id)
            if previous is None:
                continue
            current = centers[track_id]
            if abs(current - float(previous)) < max_jump:
                continue
            for replacement in fresh:
                replacement_id = int(replacement.track_id)
                if replacement_id == track_id or int(mapped.get(replacement_id, 0)) > 0:
                    continue
                if abs(centers[replacement_id] - float(previous)) <= replacement_distance:
                    result.add(track_id)
                    break
        return result

    def _search_candidate_direction_ok(self, center_x: float, image_width: int) -> bool:
        if self._search_reacquire_uid <= 0 or image_width <= 0:
            return False
        direction = self._search_reacquire_direction
        offset = min(0.45, max(0.0, float(self.config.identity_preferred_search_reacquire_side_ratio)))
        center_ratio = float(center_x) / float(image_width)
        # A target crossing the optical center during the search turn is still
        # a plausible continuation.  Keep the narrow center band eligible so
        # ReID and the two-frame local geometry gate can confirm it instead of
        # rejecting a correct target solely because the chassis has moved.
        if abs(center_ratio - 0.5) <= offset:
            return True
        if direction == "left":
            return center_ratio <= 0.5 - offset
        if direction == "right":
            return center_ratio >= 0.5 + offset
        return False

    def _duplicate_identity_track_ids(
        self,
        outputs: Sequence[Any],
        image_width: int,
        image_height: Optional[int],
    ) -> Set[int]:
        cfg = self.config
        if (
            not bool(cfg.identity_duplicate_box_suppression_enable)
            or image_width <= 0
            or image_height is None
            or image_height <= 0
        ):
            return set()

        mapped = [
            output
            for output in outputs
            if int(output.time_since_update) == 0
            and int(self.identity_bank.track_to_uid.get(int(output.track_id), 0)) > 0
        ]
        if not mapped:
            return set()

        suppressed: Set[int] = set()
        for candidate in outputs:
            track_id = int(candidate.track_id)
            if int(candidate.time_since_update) != 0 or track_id in self.identity_bank.track_to_uid:
                continue
            candidate_bbox = (candidate.x1, candidate.y1, candidate.x2, candidate.y2)
            if any(
                _is_duplicate_person_box(
                    candidate_bbox,
                    (anchor.x1, anchor.y1, anchor.x2, anchor.y2),
                    image_width=image_width,
                    image_height=int(image_height),
                    iou_threshold=float(cfg.identity_duplicate_iou_threshold),
                    vertical_overlap_threshold=float(cfg.identity_duplicate_vertical_overlap_threshold),
                    horizontal_overlap_threshold=float(cfg.identity_duplicate_horizontal_overlap_threshold),
                    large_height_ratio=float(cfg.identity_duplicate_large_height_ratio),
                    large_width_ratio=float(cfg.identity_duplicate_large_width_ratio),
                    max_area_ratio=float(cfg.identity_duplicate_max_area_ratio),
                    bottom_gap_ratio=float(cfg.identity_duplicate_bottom_gap_ratio),
                )
                for anchor in mapped
                if int(anchor.track_id) != track_id
            ):
                suppressed.add(track_id)
        return suppressed

    def _bbox_quality(
        self,
        bbox: Tuple[float, float, float, float],
        image_width: int,
        image_height: Optional[int],
    ) -> Tuple[bool, str]:
        if image_width <= 0 or image_height is None or image_height <= 0:
            return True, ""

        x1, y1, x2, y2 = [float(v) for v in bbox]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width <= 1.0 or height <= 1.0:
            return False, "empty_bbox"

        frame_area = max(1.0, float(image_width) * float(image_height))
        area_ratio = (width * height) / frame_area
        width_ratio = width / max(1.0, float(image_width))
        height_ratio = height / max(1.0, float(image_height))
        aspect = width / max(1.0, height)

        reasons: List[str] = []
        shape_reason = lower_compact_bbox_reason(bbox, image_width, image_height)
        if shape_reason:
            reasons.append(shape_reason)
        if area_ratio > float(self.config.identity_max_area_ratio):
            reasons.append(f"area_ratio>{float(self.config.identity_max_area_ratio):.2f}")
        if width * height < max(0.0, float(self.config.identity_min_area)):
            reasons.append(f"area<{float(self.config.identity_min_area):.0f}")
        if width < max(0.0, float(self.config.identity_min_width_px)):
            reasons.append(f"width<{float(self.config.identity_min_width_px):.0f}")
        if height < max(0.0, float(self.config.identity_min_height_px)):
            reasons.append(f"height<{float(self.config.identity_min_height_px):.0f}")
        if width_ratio > float(self.config.identity_max_width_ratio):
            reasons.append(f"width_ratio>{float(self.config.identity_max_width_ratio):.2f}")
        if height_ratio > float(self.config.identity_max_height_ratio):
            reasons.append(f"height_ratio>{float(self.config.identity_max_height_ratio):.2f}")
        if aspect < float(self.config.identity_min_aspect_ratio):
            reasons.append(f"aspect<{float(self.config.identity_min_aspect_ratio):.2f}")
        if aspect > float(self.config.identity_max_aspect_ratio):
            reasons.append(f"aspect>{float(self.config.identity_max_aspect_ratio):.2f}")

        edge_margin_x = float(image_width) * max(0.0, float(self.config.identity_edge_margin_ratio))
        edge_margin_y = float(image_height) * max(0.0, float(self.config.identity_edge_margin_ratio))
        edge_count = 0
        if x1 <= edge_margin_x:
            edge_count += 1
        if y1 <= edge_margin_y:
            edge_count += 1
        if (float(image_width) - x2) <= edge_margin_x:
            edge_count += 1
        if (float(image_height) - y2) <= edge_margin_y:
            edge_count += 1
        max_edges = int(self.config.identity_max_edge_touch_count)
        if max_edges >= 0 and edge_count > max_edges:
            reasons.append(f"edge_touch>{max_edges}")

        return not reasons, ",".join(reasons)

    def _bbox_transition_quality(
        self,
        track_id: int,
        area: float,
        *,
        is_fresh: bool,
        base_quality_ok: bool,
    ) -> Tuple[bool, str]:
        """Reject an impossible one-frame collapse without replacing the baseline."""
        if not is_fresh:
            return bool(base_quality_ok), ""

        track_id = int(track_id)
        area = max(0.0, float(area))
        previous = self._last_quality_area_by_track_id.get(track_id)
        max_gap = max(1, int(self.config.identity_area_shrink_max_gap_frames))
        shrink_ratio = max(
            0.05,
            min(1.0, float(self.config.identity_max_single_frame_area_shrink_ratio)),
        )
        transition_ok = True
        reason = ""
        if previous is not None:
            previous_frame, previous_area = previous
            frame_gap = int(self._frame_index) - int(previous_frame)
            if (
                0 < frame_gap <= max_gap
                and float(previous_area) > 0.0
                and area < float(previous_area) * shrink_ratio
            ):
                transition_ok = False
                reason = "area_shrink<%.2f" % shrink_ratio

        if bool(base_quality_ok) and transition_ok:
            self._last_quality_area_by_track_id[track_id] = (
                int(self._frame_index),
                area,
            )
        return transition_ok, reason

    def _bbox_center_transition_quality(
        self,
        track_id: int,
        center_x: float,
        *,
        image_width: int,
        is_fresh: bool,
    ) -> Tuple[bool, str]:
        """Reject implausible center jumps that indicate a track swap.

        This is deliberately independent of bbox area. During a crossing the
        wrong track can have a perfectly plausible area, so the previous area
        guard cannot catch the swap. The center history is still updated so a
        later stable track can recover through the normal handoff path.
        """
        if not is_fresh:
            return True, ""
        track_id = int(track_id)
        center_ratio = float(center_x) / max(1.0, float(image_width))
        previous = self._last_identity_center_by_track_id.get(track_id)
        previous_frame = self._last_identity_center_frame_by_track_id.get(track_id)
        max_gap = max(1, int(self.config.identity_center_jump_max_gap_frames))
        max_jump = max(0.05, min(1.0, float(self.config.identity_max_center_jump_ratio)))
        if previous is not None and previous_frame is not None:
            frame_gap = int(self._frame_index) - int(previous_frame)
            if 0 < frame_gap <= max_gap and abs(center_ratio - float(previous)) > max_jump:
                return False, "identity_center_jump>%.2f" % max_jump
        return True, ""

    def _bbox_identity_tier(
        self,
        *,
        bbox_quality_ok: bool,
        bbox_quality_reason: str,
        confidence: float,
        bbox: Tuple[float, float, float, float],
    ) -> str:
        """Classify identity evidence separately from motion-control geometry."""
        if bbox_quality_ok:
            return "strong"
        x1, y1, x2, y2 = [float(value) for value in bbox]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height
        if (
            float(confidence) < float(self.config.identity_min_confidence)
            or area < float(self.config.identity_min_area)
            or width < float(self.config.identity_min_width_px)
            or height < float(self.config.identity_min_height_px)
        ):
            return "reject"
        reasons = [
            item.strip()
            for item in str(bbox_quality_reason or "").split(",")
            if item.strip()
        ]
        weak_prefixes = ("aspect<", "aspect>", "edge_touch>")
        # Detector provenance may prefix an otherwise recoverable edge/aspect
        # reason (for example ``detector_crop:edge_touch>2``).  Keep the
        # original reason for diagnostics, but classify only the underlying
        # quality signal.  A compound reason containing area/size/identity
        # evidence must remain reject so partial crops cannot bypass it.
        normalized_reasons = []
        for item in reasons:
            normalized = item
            while normalized.startswith("detector_crop:"):
                normalized = normalized[len("detector_crop:"):].strip()
            normalized_reasons.append(normalized)
        if normalized_reasons and all(
            item.startswith(weak_prefixes) for item in normalized_reasons
        ):
            return "weak"
        return "reject"

    def _identity_sample_metadata(
        self,
        *,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        confidence: float,
        image_width: int,
        image_height: Optional[int],
        bbox_quality_tier: str,
        bbox_quality_reason: str,
        is_fresh: bool,
    ) -> dict:
        x1, y1, x2, y2 = [float(value) for value in bbox]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        center_x = (x1 + x2) / 2.0
        center_ratio = center_x / max(1.0, float(image_width))
        frame_area = max(1.0, float(image_width) * float(image_height or 1))
        margin_x = float(image_width) * max(0.0, float(self.config.identity_edge_margin_ratio))
        margin_y = float(image_height or 0) * max(0.0, float(self.config.identity_edge_margin_ratio))
        edge_touch_count = int(x1 <= margin_x) + int(float(image_width) - x2 <= margin_x)
        if image_height is not None and image_height > 0:
            edge_touch_count += int(y1 <= margin_y) + int(float(image_height) - y2 <= margin_y)

        previous = self._last_identity_center_by_track_id.get(int(track_id))
        motion_direction = "unknown"
        if previous is not None:
            delta = center_ratio - float(previous)
            if delta <= -0.01:
                motion_direction = "left"
            elif delta >= 0.01:
                motion_direction = "right"
            else:
                motion_direction = "stable"
        if is_fresh:
            self._last_identity_center_by_track_id[int(track_id)] = center_ratio
            self._last_identity_center_frame_by_track_id[int(track_id)] = int(self._frame_index)
        quality_reason_items = [
            item.strip()
            for item in str(bbox_quality_reason or "").split(",")
            if item.strip()
        ]
        if bbox_quality_tier == "strong":
            quality_weight = 1.0
        elif any(item.startswith(("aspect<", "aspect>")) for item in quality_reason_items):
            quality_weight = 0.25 if edge_touch_count > 0 else 0.35
        else:
            quality_weight = 0.60
        return {
            "track_id": int(track_id),
            "bbox": [x1, y1, x2, y2],
            "aspect_ratio": width / max(1.0, height),
            "area_ratio": (width * height) / frame_area,
            "detector_confidence": float(confidence),
            "edge_touch_count": int(edge_touch_count),
            "center_x_ratio": float(center_ratio),
            "motion_direction": motion_direction,
            "bbox_quality_tier": str(bbox_quality_tier),
            "bbox_quality_reason": str(bbox_quality_reason or ""),
            "quality_weight": float(quality_weight),
        }


def _xyxy_to_expanded_xywh(
    bbox: Tuple[float, float, float, float],
    scale: float,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(0.0, x2 - x1) * max(1.0, float(scale))
    height = max(0.0, y2 - y1) * max(1.0, float(scale))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy, width, height


def _image_shape(image_width: int, image_height: Optional[int]) -> Optional[Tuple[int, int]]:
    if image_width <= 0 or image_height is None or image_height <= 0:
        return None
    return int(image_height), int(image_width)


def _partial_bbox_observation(
    bbox: Tuple[float, float, float, float],
    image_width: int,
    image_height: Optional[int],
    edge_margin_ratio: float,
) -> bool:
    """Identify views where a visible-torso descriptor is safer than full-body appearance."""
    x1, y1, x2, y2 = [float(value) for value in bbox]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    frame_height = max(1.0, float(image_height or 1))
    margin_x = max(0.0, float(image_width)) * max(0.0, float(edge_margin_ratio))
    margin_y = frame_height * max(0.0, float(edge_margin_ratio))
    edge_count = int(x1 <= margin_x) + int(float(image_width) - x2 <= margin_x)
    edge_count += int(y1 <= margin_y) + int(frame_height - y2 <= margin_y)
    return edge_count > 0 or height / frame_height >= 0.80


def _is_duplicate_person_box(
    candidate_bbox: Tuple[float, float, float, float],
    anchor_bbox: Tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
    iou_threshold: float,
    vertical_overlap_threshold: float,
    horizontal_overlap_threshold: float,
    large_height_ratio: float,
    large_width_ratio: float,
    max_area_ratio: float,
    bottom_gap_ratio: float,
) -> bool:
    ax1, ay1, ax2, ay2 = [float(value) for value in candidate_bbox]
    bx1, by1, bx2, by2 = [float(value) for value in anchor_bbox]
    aw, ah = max(0.0, ax2 - ax1), max(0.0, ay2 - ay1)
    bw, bh = max(0.0, bx2 - bx1), max(0.0, by2 - by1)
    if min(aw, ah, bw, bh) <= 1.0:
        return False

    overlap_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    overlap_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = overlap_w * overlap_h
    if intersection <= 0.0:
        return False
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - intersection
    if intersection / max(union, 1e-6) >= max(0.0, float(iou_threshold)):
        return True

    vertical_overlap = overlap_h / max(1.0, min(ah, bh))
    horizontal_overlap = overlap_w / max(1.0, min(aw, bw))
    largest_height = max(ah, bh) / max(1.0, float(image_height))
    largest_width = max(aw, bw) / max(1.0, float(image_width))
    area_ratio = min(area_a, area_b) / max(area_a, area_b)
    bottom_gap = abs(ay2 - by2) / max(1.0, float(image_height))
    return (
        vertical_overlap >= float(vertical_overlap_threshold)
        and horizontal_overlap >= float(horizontal_overlap_threshold)
        and largest_height >= float(large_height_ratio)
        and largest_width >= float(large_width_ratio)
        and area_ratio <= float(max_area_ratio)
        and bottom_gap <= float(bottom_gap_ratio)
    )


def _map_state(state: int) -> int:
    if state == TrackState.CONFIRMED:
        return TRACK_STATE_STABLE
    if state == TrackState.TENTATIVE:
        return TRACK_STATE_UNSTABLE
    return TRACK_STATE_NEW


def _track_state_name(state: int) -> str:
    if state == TrackState.CONFIRMED:
        return "confirmed"
    if state == TrackState.TENTATIVE:
        return "tentative"
    if state == TrackState.DELETED:
        return "deleted"
    return "unknown"


SimpleReIDTrackerConfig = DeepSortTrackerConfig
SimpleReIDTracker = DeepSortTracker
