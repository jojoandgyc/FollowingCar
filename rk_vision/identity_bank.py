from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .identity_exclusion import IdentityExclusionMemory
from .reacquire_quarantine import ReacquireQuarantine

logger = logging.getLogger("PersonTracker")


@dataclass(frozen=True)
class IdentityBankConfig:
    enabled: bool = True
    match_threshold: float = 0.32
    match_margin: float = 0.03
    update_threshold: float = 0.30
    update_interval: int = 5
    max_features: int = 20
    max_weak_features: int = 8
    diversity_min_distance: float = 0.02
    diversity_replace_margin: float = 0.01
    weak_update_threshold: float = 0.42
    weak_update_interval: int = 3
    weak_match_penalty: float = 0.10
    weak_reacquire_threshold: float = 0.38
    weak_reacquire_confirm_frames: int = 2
    weak_quality_weight: float = 0.30
    min_confidence: float = 0.65
    min_area: float = 0.0
    reacquire_threshold: float = 0.50
    reacquire_max_frames: int = 90
    reacquire_margin: float = 0.08
    reacquire_single_candidate_only: bool = True
    reacquire_multi_candidate_enable: bool = True
    reacquire_multi_candidate_threshold: float = 0.40
    reacquire_multi_candidate_margin: float = 0.12
    new_identity_confirm_frames: int = 2
    mapped_verify_enable: bool = True
    mapped_verify_threshold: float = 0.40
    mapped_bad_quality_returns_unassigned: bool = True
    exclusive_uid_claim_enable: bool = False
    exclusive_uid_claim_frames: int = 15
    controlled_handoff_enable: bool = False
    # A new raw track needs two consecutive matching observations before it
    # inherits an existing UID.  Very strong preferred-search matches may use
    # the dedicated instant threshold below.
    controlled_handoff_confirm_frames: int = 2
    controlled_handoff_instant_threshold: float = 0.15
    controlled_handoff_threshold: float = 0.30
    controlled_handoff_min_old_track_gap_frames: int = 5
    handoff_geometry_max_gap_frames: int = 15
    # IMU-compensated geometry still has detector jitter while the chassis is
    # rotating.  Keep normal handoffs bounded, but allow a modest residual
    # jump so a sole strong candidate can complete its two-frame handoff.
    handoff_geometry_max_center_jump_ratio: float = 0.30
    handoff_geometry_min_area_similarity: float = 0.35
    max_edge_touch_count: int = 2
    preferred_search_reacquire_enable: bool = True
    preferred_search_reacquire_threshold: float = 0.20
    preferred_search_reacquire_max_disadvantage: float = 0.05
    # Different PEOPLE compared to the active UID, not different gallery UIDs.
    preferred_search_candidate_min_margin: float = 0.05
    # Search recovery may use the detector confidence floor used by DeepSORT,
    # but only after an active UID, fresh candidate and search-side gate have
    # been established. Normal identity assignment keeps min_confidence.
    preferred_search_reacquire_min_confidence: float = 0.50
    # Large person-sized detector boxes may still provide useful ReID
    # evidence below the normal search floor. This applies only to the
    # explicit search-observation path; normal tracking keeps min_confidence.
    preferred_search_reacquire_observation_min_confidence: float = 0.25
    # Search reacquisition is only allowed close to the last visual anchor.
    # When capture timestamps are available this is a wall-clock limit;
    # frame-gap geometry remains the fallback for replay/unit-test metadata.
    preferred_search_reacquire_max_age_sec: float = 0.35
    # When the old geometry anchor is stale, observe a new candidate locally
    # before binding the locked UID.  This never creates a new identity.
    preferred_search_reacquire_late_candidate_enable: bool = True
    # Search reacquisition normally needs two consecutive observations.
    preferred_search_reacquire_confirm_frames: int = 2
    # A close match is instant only with a recent, geometrically continuous
    # strong observation of the UID. Missing geometry requires confirmation.
    preferred_search_reacquire_instant_threshold: float = 0.15
    # A second low-confidence detector fragment should not block a strong
    # preferred-UID candidate.  Close-confidence competitors remain gated.
    preferred_search_reacquire_min_score_gap: float = 0.25
    # When the strict preferred gate misses because of normal pose/viewpoint
    # variation, allow a bounded observation-only path.  It never binds a UID
    # on its first frame; the existing local two-frame geometry chain does so.
    preferred_search_soft_candidate_enable: bool = True
    preferred_search_soft_candidate_threshold: float = 0.30
    preferred_search_soft_min_score_gap: float = 0.15
    preferred_search_soft_min_area_ratio: float = 0.25
    preferred_search_soft_min_confidence: float = 0.80
    partial_appearance_enable: bool = True
    partial_match_threshold: float = 0.34
    partial_max_features: int = 8
    partial_update_threshold: float = 0.30
    camera_hfov_deg: float = 90.0


@dataclass
class IdentityEntry:
    uid: int
    features: List[Any] = field(default_factory=list)
    weak_features: List[Any] = field(default_factory=list)
    feature_metadata: List[dict] = field(default_factory=list)
    weak_feature_metadata: List[dict] = field(default_factory=list)
    partial_features: List[Any] = field(default_factory=list)
    partial_feature_metadata: List[dict] = field(default_factory=list)
    last_frame: int = 0
    last_weak_frame: int = 0
    last_seen_frame: int = 0
    update_count: int = 0
    weak_update_count: int = 0
    duplicate_skip_count: int = 0
    weak_duplicate_skip_count: int = 0
    diversity_replace_count: int = 0
    weak_diversity_replace_count: int = 0
    last_strong_observation: Optional[dict] = None

    def add_partial(
        self,
        feature: Any,
        frame_index: int,
        max_features: int,
        diversity_min_distance: float = 0.02,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Store a separate visible-torso descriptor without changing full-body templates."""
        if feature is None or max_features <= 0:
            return False
        np = _np()
        sample = _normalize_feature(feature)
        while len(self.partial_feature_metadata) < len(self.partial_features):
            self.partial_feature_metadata.append({"quality_tier": "partial", "quality_weight": 0.5})
        if not self.partial_features:
            self.partial_features.append(sample)
            self.partial_feature_metadata.append(
                _sample_metadata(metadata, frame_index, 0.5, "partial")
            )
            return True
        # A process configured with the learned torso branch must never try
        # to dot-product it with legacy histogram samples left by a caller or
        # replay fixture.  Keep only vectors with the query dimension.
        compatible = [
            index
            for index, value in enumerate(self.partial_features)
            if _feature_size(value) == _feature_size(sample)
        ]
        if len(compatible) != len(self.partial_features):
            self.partial_features = [self.partial_features[index] for index in compatible]
            self.partial_feature_metadata = [
                self.partial_feature_metadata[index]
                for index in compatible
                if index < len(self.partial_feature_metadata)
            ]
        if not self.partial_features:
            self.partial_features.append(sample)
            self.partial_feature_metadata.append(
                _sample_metadata(metadata, frame_index, 0.5, "partial")
            )
            return True
        samples = np.asarray(self.partial_features, dtype="float32")
        distances = 1.0 - samples.dot(sample.reshape(-1, 1)).reshape(-1)
        if float(distances.min()) < max(0.0, float(diversity_min_distance)):
            return False
        if len(self.partial_features) < int(max_features):
            self.partial_features.append(sample)
            self.partial_feature_metadata.append(
                _sample_metadata(metadata, frame_index, 0.5, "partial")
            )
            return True
        # Keep the first partial sample as a stable anchor and replace the
        # most redundant later sample when the gallery is full.
        pairwise = 1.0 - samples.dot(samples.T)
        np.fill_diagonal(pairwise, np.inf)
        replace_index = min(
            range(1, len(self.partial_features)),
            key=lambda index: float(pairwise[index].min()),
        )
        self.partial_features[replace_index] = sample
        _replace_metadata(
            self.partial_feature_metadata,
            replace_index,
            _sample_metadata(metadata, frame_index, 0.5, "partial"),
        )
        return True

    def add(
        self,
        feature: Any,
        frame_index: int,
        max_features: int,
        diversity_min_distance: float = 0.02,
        diversity_replace_margin: float = 0.01,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Keep a compact gallery with one stable anchor and diverse templates."""
        np = _np()
        sample = _normalize_feature(feature)
        limit = max(1, int(max_features))
        changed = False
        while len(self.feature_metadata) < len(self.features):
            self.feature_metadata.append({"quality_tier": "strong", "quality_weight": 1.0})

        if not self.features:
            self.features.append(sample)
            self.feature_metadata.append(_sample_metadata(metadata, frame_index, 1.0, "strong"))
            changed = True
        else:
            samples = np.asarray(self.features, dtype="float32")
            distances = 1.0 - samples.dot(sample.reshape(-1, 1)).reshape(-1)
            min_distance = float(distances.min())
            min_separation = max(0.0, float(diversity_min_distance))

            if min_distance < min_separation:
                self.duplicate_skip_count += 1
            elif len(self.features) < limit:
                self.features.append(sample)
                self.feature_metadata.append(_sample_metadata(metadata, frame_index, 1.0, "strong"))
                changed = True
            elif limit > 1:
                # 第 0 条是首次稳定录入的锚点，不参与淘汰。其余模板中找出
                # 与图库最重复的一条；新特征只有明显增加覆盖范围时才替换它。
                pairwise = 1.0 - samples.dot(samples.T)
                np.fill_diagonal(pairwise, np.inf)
                replace_index = min(
                    range(1, len(self.features)),
                    key=lambda index: float(pairwise[index].min()),
                )
                existing_novelty = float(pairwise[replace_index].min())
                remaining = [index for index in range(len(self.features)) if index != replace_index]
                new_novelty = float(distances[remaining].min()) if remaining else float("inf")
                margin = max(0.0, float(diversity_replace_margin))
                if new_novelty >= existing_novelty + margin:
                    self.features[replace_index] = sample
                    _replace_metadata(
                        self.feature_metadata,
                        replace_index,
                        _sample_metadata(metadata, frame_index, 1.0, "strong"),
                    )
                    self.diversity_replace_count += 1
                    changed = True
                else:
                    self.duplicate_skip_count += 1

        self.last_frame = int(frame_index)
        self.last_seen_frame = int(frame_index)
        self.update_count += 1
        return changed

    def add_weak(
        self,
        feature: Any,
        frame_index: int,
        max_features: int,
        diversity_min_distance: float = 0.02,
        diversity_replace_margin: float = 0.01,
        quality_weight: float = 0.30,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Store diverse weak views without consuming or replacing strong anchors."""
        np = _np()
        sample = _normalize_feature(feature)
        limit = max(0, int(max_features))
        changed = False
        if limit <= 0:
            return False
        while len(self.weak_feature_metadata) < len(self.weak_features):
            self.weak_feature_metadata.append({"quality_tier": "weak", "quality_weight": 0.30})

        metadata_weight = (metadata or {}).get("quality_weight", quality_weight)
        sample_meta = _sample_metadata(
            metadata,
            frame_index,
            max(0.0, min(1.0, float(metadata_weight))),
            "weak",
        )
        if not self.weak_features:
            self.weak_features.append(sample)
            self.weak_feature_metadata.append(sample_meta)
            changed = True
        else:
            samples = np.asarray(self.weak_features, dtype="float32")
            distances = 1.0 - samples.dot(sample.reshape(-1, 1)).reshape(-1)
            min_distance = float(distances.min())
            min_separation = max(0.0, float(diversity_min_distance))
            if min_distance < min_separation:
                self.weak_duplicate_skip_count += 1
            elif len(self.weak_features) < limit:
                self.weak_features.append(sample)
                self.weak_feature_metadata.append(sample_meta)
                changed = True
            else:
                pairwise = 1.0 - samples.dot(samples.T)
                np.fill_diagonal(pairwise, np.inf)
                replace_index = min(
                    range(len(self.weak_features)),
                    key=lambda index: float(pairwise[index].min()),
                )
                existing_novelty = float(pairwise[replace_index].min())
                remaining = [index for index in range(len(self.weak_features)) if index != replace_index]
                new_novelty = float(distances[remaining].min()) if remaining else float("inf")
                margin = max(0.0, float(diversity_replace_margin))
                if new_novelty >= existing_novelty + margin:
                    self.weak_features[replace_index] = sample
                    _replace_metadata(self.weak_feature_metadata, replace_index, sample_meta)
                    self.weak_diversity_replace_count += 1
                    changed = True
                else:
                    self.weak_duplicate_skip_count += 1

        self.last_weak_frame = int(frame_index)
        self.last_seen_frame = max(int(self.last_seen_frame), int(frame_index))
        self.weak_update_count += 1
        return changed

    def touch(self, frame_index: int) -> None:
        self.last_seen_frame = max(int(self.last_seen_frame), int(frame_index))

    def distance(self, feature: Any) -> float:
        if not self.features:
            return float("inf")
        np = _np()
        query = _normalize_feature(feature).reshape(1, -1)
        samples = np.asarray(self.features, dtype="float32")
        return float((1.0 - samples.dot(query.T)).min())

    def weak_distance(self, feature: Any) -> float:
        if not self.weak_features:
            return float("inf")
        np = _np()
        query = _normalize_feature(feature).reshape(1, -1)
        samples = np.asarray(self.weak_features, dtype="float32")
        return float((1.0 - samples.dot(query.T)).min())

    def partial_distance(self, feature: Any) -> float:
        if feature is None or not self.partial_features:
            return float("inf")
        np = _np()
        query = _normalize_feature(feature).reshape(1, -1)
        compatible = [
            value
            for value in self.partial_features
            if _feature_size(value) == int(query.shape[1])
        ]
        if not compatible:
            return float("inf")
        samples = np.asarray(compatible, dtype="float32")
        distances = np.sort(1.0 - samples.dot(query.T).reshape(-1))
        # Avoid accepting a partial view because of one accidental gallery
        # template match.  The closest three visible-torso templates provide
        # a small robust estimate while keeping the first observation useful.
        top_k = min(3, int(distances.size))
        return float(np.mean(distances[:top_k]))

    def weighted_distance(self, feature: Any, weak_penalty: float) -> Tuple[float, str]:
        """Return the best distance while penalizing every weak template by quality."""
        strong_distance = self.distance(feature)
        if not self.weak_features:
            return strong_distance, "strong"
        np = _np()
        query = _normalize_feature(feature).reshape(1, -1)
        samples = np.asarray(self.weak_features, dtype="float32")
        distances = 1.0 - samples.dot(query.T).reshape(-1)
        penalties = []
        base_penalty = max(0.0, float(weak_penalty))
        for index in range(len(self.weak_features)):
            metadata = self.weak_feature_metadata[index] if index < len(self.weak_feature_metadata) else {}
            weight = max(0.0, min(1.0, float(metadata.get("quality_weight", 0.30))))
            penalties.append(base_penalty * (1.0 - weight))
        weak_weighted = float((distances + np.asarray(penalties, dtype="float32")).min())
        if weak_weighted < strong_distance:
            return weak_weighted, "weak"
        return strong_distance, "strong"

    def match_evidence(self, feature: Any, weak_penalty: float) -> dict:
        """Snapshot the winning template before this observation can update it."""
        np = _np()
        query = _normalize_feature(feature).reshape(-1, 1)
        candidates = []
        winners = {}
        raw_distances = {}
        anchor_distance = None
        for tier, features, metadata in (
            ("strong", self.features, self.feature_metadata),
            ("weak", self.weak_features, self.weak_feature_metadata),
        ):
            if not features:
                winners[tier] = None
                raw_distances[tier] = None
                continue
            distances = 1.0 - np.asarray(features, dtype="float32").dot(query).reshape(-1)
            weighted = distances.copy()
            if tier == "weak":
                penalties = [
                    max(0.0, float(weak_penalty))
                    * (1.0 - max(0.0, min(1.0, float(
                        (metadata[index] if index < len(metadata) else {}).get("quality_weight", 0.30)
                    ))))
                    for index in range(len(features))
                ]
                weighted += np.asarray(penalties, dtype="float32")
            else:
                anchor_distance = _finite_float(distances[0])
            raw_distances[tier] = _finite_float(distances.min())
            nearest = []
            for index in np.argsort(weighted, kind="stable")[:3]:
                index = int(index)
                nearest.append({
                    "tier": tier,
                    "index": index,
                    "distance": _finite_float(distances[index]),
                    "weighted_distance": _finite_float(weighted[index]),
                    "metadata": _evidence_metadata(metadata[index] if index < len(metadata) else None),
                })
            winners[tier] = nearest[0]
            candidates.extend(nearest)
        candidates.sort(key=lambda item: (
            float("inf") if item["weighted_distance"] is None else item["weighted_distance"],
            item["tier"] != "strong",
            item["index"],
        ))
        winner = candidates[0] if candidates else None
        return {
            "matched_uid": int(self.uid),
            "distance": None if winner is None else winner["weighted_distance"],
            "match_source": None if winner is None else winner["tier"],
            "anchor_distance": anchor_distance,
            "anchor_metadata": _evidence_metadata(self.feature_metadata[0] if self.feature_metadata else None),
            "strong_distance": raw_distances["strong"],
            "weak_distance": raw_distances["weak"],
            "weak_weighted_distance": (
                None if winners["weak"] is None else winners["weak"]["weighted_distance"]
            ),
            "winner": winner,
            "strong_winner": winners["strong"],
            "weak_winner": winners["weak"],
            "nearest_samples": candidates[:3],
        }


@dataclass
class PendingIdentity:
    feature: Any
    partial_feature: Optional[Any]
    first_frame: int
    last_frame: int
    streak: int = 1


@dataclass
class PendingHandoff:
    uid: int
    last_frame: int
    streak: int = 1
    center_ratio: Optional[float] = None
    area: Optional[float] = None
    area_units: Optional[str] = None
    geometry_source: Optional[str] = None
    capture_timestamp: Optional[float] = None
    integrated_yaw_deg: Optional[float] = None


@dataclass
class PendingLateHandoff:
    uid: int
    last_frame: int
    streak: int = 1
    center_ratio: Optional[float] = None
    area: Optional[float] = None
    area_units: Optional[str] = None
    geometry_source: Optional[str] = None
    capture_timestamp: Optional[float] = None
    partial_feature: Optional[Any] = None
    integrated_yaw_deg: Optional[float] = None


class IdentityBank:
    """Long-lived ReID identity memory outside DeepSORT's active-track gallery."""

    def __init__(self, config: Optional[IdentityBankConfig] = None) -> None:
        self.config = config or IdentityBankConfig()
        self.identities: Dict[int, IdentityEntry] = {}
        self.track_to_uid: Dict[int, int] = {}
        self.track_last_seen_frame: Dict[int, int] = {}
        self.pending_new: Dict[int, PendingIdentity] = {}
        self.pending_handoffs: Dict[int, PendingHandoff] = {}
        self.pending_late_handoffs: Dict[int, PendingLateHandoff] = {}
        self.pending_weak_handoffs: Dict[int, PendingHandoff] = {}
        self.last_assignments: Dict[int, dict] = {}
        # A raw ID is not proof of identity. Keep a contradiction until that
        # track returns to trusted geometry, rather than aging it into a match.
        self._mapped_geometry_conflicts: Dict[int, dict] = {}
        self._geometry_revoked_uids: Dict[int, int] = {}
        self._identity_exclusion = IdentityExclusionMemory(
            camera_hfov_deg=self.config.camera_hfov_deg,
        )
        self._reacquire_quarantine = ReacquireQuarantine()
        self._quarantine_decisions: Dict[int, Any] = {}
        self._next_uid = 1

    def reset(self) -> None:
        self.identities.clear()
        self.track_to_uid.clear()
        self.track_last_seen_frame.clear()
        self.pending_new.clear()
        self.pending_handoffs.clear()
        self.pending_late_handoffs.clear()
        self.pending_weak_handoffs.clear()
        self.last_assignments.clear()
        self._mapped_geometry_conflicts.clear()
        self._geometry_revoked_uids.clear()
        self._identity_exclusion.reset()
        self._reacquire_quarantine.prune([])
        self._quarantine_decisions.clear()
        self._next_uid = 1

    def observe_frame_evidence(
        self, *, frame_index: int, observations: List[dict], width: int, height: int,
    ) -> None:
        """Read the whole detector frame before any UID assignment mutates it.

        A currently mapped, strong full-body match can witness another,
        spatially distinct person even when its own cropped box is weak.
        A newly reacquired UID is not a trusted witness until its template
        quarantine clears. Validate every old claim before creating negative
        evidence; invalid claims are revoked, but no gallery is written here.
        """
        if not self.config.enabled:
            return
        evidence = []
        reviews = {}
        claimed_counts: Dict[int, int] = {}
        for observation in observations:
            claim = int(observation.get("mapped_uid", 0) or 0)
            if claim > 0 and observation.get("is_fresh") is True:
                claimed_counts[claim] = claimed_counts.get(claim, 0) + 1
        for observation in observations:
            track_id = int(observation.get("raw_track_id", 0))
            bbox = observation.get("detector_bbox")
            metadata = dict(observation)
            if bbox is not None and width and height:
                x1, y1, x2, y2 = bbox
                metadata.update(
                    detector_center_x_ratio=(x1 + x2) / (2.0 * width),
                    detector_area_ratio=(x2 - x1) * (y2 - y1) / (width * height),
                )
            reviews[track_id] = self.review_mapped_geometry(
                track_id, metadata, frame_index, commit=True,
            )
        for observation in observations:
            item = dict(observation)
            item["trusted_uid"] = 0
            track_id = int(item.get("raw_track_id", 0))
            uid = int(item.get("mapped_uid", 0) or 0)
            entry = self.identities.get(uid)
            confidence = _finite_float(item.get("confidence"))
            feature = item.get("feature")
            geometry = reviews.get(track_id) or {}
            competition = item.get("identity_competition") or {}
            reference = geometry.get("reference") or {}
            item.update(
                witness_geometry_valid=geometry.get("ok") is True,
                witness_reference_frame_index=reference.get("frame_index"),
                witness_reference_capture_frame_id=reference.get("capture_frame_id"),
            )
            if (
                uid > 0 and entry is not None
                and self.track_to_uid.get(track_id) == uid
                and item.get("is_fresh") is True
                and not item.get("duplicate") and not item.get("identity_swap")
                and not self._reacquire_quarantine.is_held(uid)
                and confidence is not None
                and confidence >= max(0.60, float(self.config.min_confidence))
                and feature is not None
                and geometry.get("ok") is True
                and not geometry.get("mapped_geometry_blocked", False)
                and claimed_counts.get(uid, 0) == 1
                and not (
                    competition.get("frame_index") == int(frame_index)
                    and competition.get("uid") == uid
                    and competition.get("passed") is False
                )
            ):
                strong_distance = _finite_float(entry.distance(feature))
                _, match_source = entry.weighted_distance(
                    feature, self.config.weak_match_penalty,
                )
                if strong_distance is not None and strong_distance <= 0.20 and match_source == "strong":
                    item["trusted_uid"] = uid
            evidence.append(item)
        witnesses: Dict[int, int] = {}
        for item in evidence:
            uid = int(item["trusted_uid"])
            if uid > 0:
                witnesses[uid] = witnesses.get(uid, 0) + 1
        for item in evidence:
            if witnesses.get(int(item["trusted_uid"]), 0) > 1:
                # Multiple current claims for one UID are ambiguous, not
                # proof that either claimant can exclude the other person.
                item["trusted_uid"] = 0
        self._identity_exclusion.observe_frame(
            frame_index=int(frame_index), observations=evidence,
            width=width, height=height,
        )

    def review_mapped_geometry(
        self, track_id: int, metadata: Optional[dict], frame_index: int, *,
        commit: bool = False,
    ) -> dict:
        """Check raw detector geometry; pure during association cost building.

        Only simultaneous position AND scale contradictions revoke an ordinary
        mapped track. Edge/size changes alone keep their existing policy.
        """
        track_id = int(track_id)
        held = self._mapped_geometry_conflicts.get(track_id)
        uid = int(self.track_to_uid.get(track_id, 0) or (held or {}).get("uid", 0))
        if not self.config.enabled or uid <= 0:
            return {"ok": None, "reason": "unmapped", "mapped_geometry_blocked": False}
        geometry = self._handoff_geometry(
            uid, metadata, frame_index,
            reference_override=(held or {}).get("reference"),
        )
        reasons = set(str(geometry.get("reason", "")).split(","))
        severe = {"center_jump", "area_change"}.issubset(reasons)
        blocked = bool(severe or (held is not None and geometry.get("ok") is not True))
        geometry["mapped_geometry_blocked"] = blocked
        geometry["mapped_geometry_uid"] = uid
        if not commit:
            return geometry
        if blocked:
            reference = geometry.get("reference") or (held or {}).get("reference") or {}
            if held is None:
                self._mapped_geometry_conflicts[track_id] = {
                    "uid": uid, "reference": dict(reference),
                }
            revoked_mapping = self.track_to_uid.pop(track_id, None)
            self.track_last_seen_frame.pop(track_id, None)
            for pending in (self.pending_new, self.pending_handoffs,
                            self.pending_weak_handoffs, self.pending_late_handoffs):
                pending.pop(track_id, None)
            if revoked_mapping == uid:
                self._geometry_revoked_uids[uid] = int(reference.get("frame_index", frame_index))
            revoked = self._identity_exclusion.invalidate_witness(
                uid, track_id, after_frame=int(reference.get("frame_index", frame_index)),
            )
            logger.info(
                "mapped_identity_geometry_reject frame=%d capture=%s track=%d uid=%d "
                "reason=%s compensated_jump=%s area_similarity=%s reference_capture=%s "
                "exclusions_revoked=%d control_allowed=False gallery_update=False",
                frame_index, (metadata or {}).get("capture_frame_id"), track_id, uid,
                geometry.get("reason"), geometry.get("yaw_compensated_center_jump_ratio"),
                geometry.get("area_similarity"), reference.get("capture_frame_id"), revoked,
            )
        elif geometry.get("ok") is True:
            self._mapped_geometry_conflicts.pop(track_id, None)
        return geometry

    def search_exclusion_for(
        self, track_id: int, uid: int, *, frame_index: int,
        capture_timestamp: Optional[float] = None,
    ) -> Optional[dict]:
        return self._identity_exclusion.exclusion_for(
            track_id, uid, frame_index=frame_index,
            capture_timestamp=capture_timestamp,
        )

    def _reject_identity_exclusion(
        self, track_id: int, uid: Optional[int], frame_index: int,
        metadata: Optional[dict], diagnostics: dict,
    ) -> bool:
        if int(uid or 0) <= 0:
            return False
        exclusion = self.search_exclusion_for(
            track_id, int(uid), frame_index=frame_index,
            capture_timestamp=(metadata or {}).get("capture_timestamp"),
        )
        if exclusion is None:
            return False
        # Positive local continuity cannot cancel a previous co-visible
        # contradiction. Clear only this candidate's confirmation chains.
        for pending in (self.pending_new, self.pending_handoffs,
                        self.pending_late_handoffs, self.pending_weak_handoffs):
            pending.pop(int(track_id), None)
        if self.track_to_uid.get(int(track_id)) == int(uid):
            self.track_to_uid.pop(int(track_id), None)
            self.track_last_seen_frame.pop(int(track_id), None)
        diagnostics.update(
            search_excluded=True, excluded_uid=int(uid),
            search_exclusion=exclusion, instant_reacquire_allowed=False,
        )
        match_evidence = diagnostics.get("match_evidence") or {}
        self.last_assignments[int(track_id)] = {
            "uid": 0, "reason": "search_candidate_excluded", "best_uid": int(uid),
            "distance": _finite_float(match_evidence.get("distance")),
            "match_source": match_evidence.get("match_source"),
            "bank_updated": False, "bbox_quality_ok": False,
            "bbox_quality_tier": "reject",
            "bbox_quality_reason": "co_visible_distinct_person",
        }
        return True

    def _bind_reacquired_identity(
        self, uid: int, track_id: int, frame_index: int, metadata: Optional[dict],
    ) -> None:
        """Bind an existing UID without immediately learning the new person."""
        if self.track_to_uid.get(int(track_id)) != int(uid):
            source = metadata or {}
            self._quarantine_decisions[int(uid)] = self._reacquire_quarantine.arm(
                uid, track_id, capture_frame_id=source.get("capture_frame_id"),
                capture_timestamp=source.get("capture_timestamp"), frame_index=frame_index,
            )
        self.track_to_uid[int(track_id)] = int(uid)
        self._geometry_revoked_uids.pop(int(uid), None)
        self._mapped_geometry_conflicts.pop(int(track_id), None)

    def _observe_template_quarantine(
        self, *, uid: int, track_id: int, feature: Any, confidence: float,
        area: float, frame_index: int, bbox_quality_ok: bool,
        bbox_quality_tier: Optional[str], metadata: dict,
    ) -> None:
        if not self._reacquire_quarantine.is_held(uid):
            return
        entry = self.identities.get(int(uid))
        geometry = _geometry_observation(metadata, frame_index) or {}
        self._quarantine_decisions[int(uid)] = self._reacquire_quarantine.observe(
            uid=uid, track_id=track_id, frame_index=frame_index,
            capture_frame_id=metadata.get("capture_frame_id"),
            capture_timestamp=metadata.get("capture_timestamp"),
            is_fresh=metadata.get("is_fresh") is True,
            quality_ok=bool(bbox_quality_ok and self._quality_ok(confidence, area)),
            quality_tier=bbox_quality_tier or ("strong" if bbox_quality_ok else "reject"),
            match_source=self._match_source(uid, feature) if feature is not None else None,
            feature_available=feature is not None,
            strong_distance=entry.distance(feature) if entry is not None and feature is not None else None,
            center_x_ratio=geometry.get("center_x_ratio"),
            area_ratio=geometry.get("area") if geometry.get("area_units") == "ratio" else None,
        )

    def assign(
        self,
        *,
        track_id: int,
        feature: Optional[Any],
        partial_feature: Optional[Any] = None,
        confidence: float,
        area: float,
        frame_index: int,
        candidate_count: int = 1,
        bbox_quality_ok: bool = True,
        bbox_quality_reason: str = "",
        bbox_quality_tier: Optional[str] = None,
        sample_metadata: Optional[dict] = None,
        preferred_uid: Optional[int] = None,
        preferred_candidate_ok: bool = False,
    ) -> int:
        sample_metadata = dict(sample_metadata or {})
        sample_metadata["track_id"] = int(track_id)
        sample_metadata["frame_index"] = int(frame_index)
        diagnostics = {
            "match_evidence": None,
            "instant_reacquire_allowed": False,
            "reacquire_geometry_ok": None,
            "reacquire_geometry_reason": "not_evaluated",
            "search_excluded": False,
        }
        comparison_uid = self.track_to_uid.get(int(track_id), 0) or int(
            self._mapped_geometry_conflicts.get(int(track_id), {}).get("uid", 0)
        ) or int(preferred_uid or 0)
        mapped_uid = self.track_to_uid.get(int(track_id), 0)
        if self.config.enabled:
            self._capture_match_evidence(diagnostics, comparison_uid, feature)
            diagnostics["partial_distance"] = self._partial_distance_to_uid(
                comparison_uid, partial_feature
            )
        uid = self._assign(
            track_id=track_id,
            feature=feature,
            partial_feature=partial_feature,
            confidence=confidence,
            area=area,
            frame_index=frame_index,
            candidate_count=candidate_count,
            bbox_quality_ok=bbox_quality_ok,
            bbox_quality_reason=bbox_quality_reason,
            bbox_quality_tier=bbox_quality_tier,
            sample_metadata=sample_metadata,
            preferred_uid=preferred_uid,
            preferred_candidate_ok=preferred_candidate_ok,
            diagnostics=diagnostics,
        )
        assignment = self.last_assignments[int(track_id)]
        quarantine_evaluated = diagnostics.pop("_template_observation_evaluated", False)
        if (
            self.config.enabled and mapped_uid > 0 and not quarantine_evaluated
            and self._reacquire_quarantine.is_held(mapped_uid)
        ):
            # Any early return (exclusion, weak quality, missing feature,
            # geometry/identity rejection) breaks proof; it cannot release
            # the gallery just because its raw embedding happened to match.
            self._observe_template_quarantine(
                uid=mapped_uid, track_id=track_id, feature=feature,
                confidence=confidence, area=area, frame_index=frame_index,
                bbox_quality_ok=False, bbox_quality_tier=bbox_quality_tier,
                metadata=sample_metadata,
            )
        assignment.setdefault("bank_updated", False)
        quarantine_uid = int(uid or comparison_uid or 0)
        quarantine = self._quarantine_decisions.get(quarantine_uid)
        diagnostics["template_update_quarantined"] = bool(
            quarantine_uid > 0 and self._reacquire_quarantine.is_held(quarantine_uid)
        )
        if quarantine is not None:
            diagnostics.update(
                template_quarantine_reason=quarantine.reason,
                template_quarantine_streak=quarantine.streak,
                template_quarantine_elapsed_sec=_finite_float(quarantine.elapsed_sec),
            )
        if assignment.get("reason") in {"created", "created_confirmed"}:
            diagnostics["rejected_match_evidence"] = diagnostics["match_evidence"]
            diagnostics["match_evidence"] = None
        assignment.update(diagnostics)
        if int(uid) <= 0:
            assignment["instant_reacquire_allowed"] = False
        routine = assignment.get("reason") in {
            "mapped", "updated_diverse", "skip_update_redundant", "skip_update_distance",
            "mapped_weak_observed", "mapped_weak_no_feature", "no_feature",
        }
        if self.config.enabled and (
            not routine or self._should_update(frame_index)
            or diagnostics["template_update_quarantined"]
        ):
            try:
                payload = json.dumps({
                    "frame_index": int(frame_index),
                    "track_id": int(track_id),
                    "output_uid": int(uid),
                    "reason": assignment.get("reason"),
                    "bank_updated": bool(assignment["bank_updated"]),
                    "query_metadata": _evidence_metadata(sample_metadata),
                    "feature_available": feature is not None,
                    **diagnostics,
                    "instant_reacquire_allowed": assignment["instant_reacquire_allowed"],
                }, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
            except (TypeError, ValueError, OverflowError) as exc:
                logger.warning(
                    "reid_match_evidence_serialization_failed frame=%d track_id=%d error=%s",
                    int(frame_index), int(track_id), type(exc).__name__,
                )
            else:
                logger.info("reid_match_evidence %s", payload)
        return int(uid)

    def _assign(
        self,
        *,
        track_id: int,
        feature: Optional[Any],
        partial_feature: Optional[Any],
        confidence: float,
        area: float,
        frame_index: int,
        candidate_count: int,
        bbox_quality_ok: bool,
        bbox_quality_reason: str,
        bbox_quality_tier: Optional[str],
        sample_metadata: Optional[dict],
        preferred_uid: Optional[int],
        preferred_candidate_ok: bool,
        diagnostics: dict,
    ) -> int:
        track_id = int(track_id)
        if not self.config.enabled:
            self.track_to_uid[track_id] = track_id
            self.pending_new.pop(track_id, None)
            self.last_assignments[track_id] = {"uid": track_id, "reason": "disabled"}
            return track_id

        uid = self.track_to_uid.get(track_id, 0)
        metadata = sample_metadata if sample_metadata is not None else {}
        geometry_review = self.review_mapped_geometry(
            track_id, metadata, frame_index, commit=True,
        )
        if geometry_review.get("mapped_geometry_blocked"):
            self._set_geometry_diagnostics(diagnostics, geometry_review)
            self.last_assignments[track_id] = {
                "uid": 0, "mapped_uid": geometry_review["mapped_geometry_uid"],
                "reason": "mapped_geometry_reject", "bank_updated": False,
                "bbox_quality_ok": False, "bbox_quality_tier": "reject",
                "bbox_quality_reason": geometry_review.get("reason"),
            }
            return 0
        competition = metadata.get("identity_competition") or {}
        if competition.get("frame_index") == int(frame_index):
            diagnostics["identity_competition"] = dict(competition)
        if self._reject_identity_exclusion(
            track_id, uid or preferred_uid, frame_index, metadata, diagnostics,
        ):
            return 0
        if (
            metadata.get("search_reacquire_context_active")
            and metadata.get("is_fresh") is True
            and int(preferred_uid or 0) > 0
            and competition.get("uid") == int(preferred_uid)
            and competition.get("frame_index") == int(frame_index)
        ):
            if competition.get("passed") is False:
                # Clear accumulated proof: two ambiguous observations are
                # not two confirmations. Never write a template on this path.
                for pending in (self.pending_new, self.pending_handoffs,
                                self.pending_late_handoffs, self.pending_weak_handoffs):
                    pending.pop(track_id, None)
                if uid == int(preferred_uid):
                    self.track_to_uid.pop(track_id, None)
                    self.track_last_seen_frame.pop(track_id, None)
                self.last_assignments[track_id] = {
                    "uid": 0, "reason": "search_candidate_identity_ambiguous",
                    "best_uid": int(preferred_uid), "distance": competition.get("distance"),
                    "bank_updated": False, "bbox_quality_ok": False,
                    "bbox_quality_tier": "reject", "bbox_quality_reason": competition.get("reason"),
                }
                diagnostics["instant_reacquire_allowed"] = False
                return 0
        search_observation_only = bool(metadata.get("search_observation_only"))
        search_observation_override = bool(
            self.config.preferred_search_reacquire_enable
            and search_observation_only
            and int(preferred_uid or 0) > 0
            and bool(preferred_candidate_ok)
            and bool(metadata.get("search_reacquire_context_active"))
            and bool(metadata.get("is_fresh", True))
            and bool(metadata.get("search_observation_large"))
            and str(bbox_quality_tier or "").strip().lower() in {"strong", "weak"}
            and float(confidence) >= max(
                0.0,
                min(
                    1.0,
                    float(self.config.preferred_search_reacquire_observation_min_confidence),
                ),
            )
        )
        search_quality_override = bool(
            self.config.preferred_search_reacquire_enable
            and int(preferred_uid or 0) > 0
            and bool(preferred_candidate_ok)
            and bool(metadata.get("search_reacquire_context_active"))
            and bool(metadata.get("is_fresh", True))
            and bool(bbox_quality_ok)
        )
        quality_confidence_floor = float(self.config.min_confidence)
        if search_quality_override or search_observation_override:
            quality_confidence_floor = min(
                quality_confidence_floor,
                max(0.0, min(1.0, float(
                    self.config.preferred_search_reacquire_observation_min_confidence
                    if search_observation_override
                    else self.config.preferred_search_reacquire_min_confidence
                ))),
            )
            if float(confidence) < float(self.config.min_confidence):
                metadata["preferred_search_low_confidence"] = True
        base_quality_ok = self._quality_ok(
            confidence,
            area,
            min_confidence=quality_confidence_floor,
        )
        quality_tier = str(bbox_quality_tier or ("strong" if bbox_quality_ok else "reject")).strip().lower()
        if quality_tier not in ("strong", "weak", "reject"):
            quality_tier = "reject"
        # A raw detector box touching several image edges is a valid partial
        # person observation when the caller supplied a torso descriptor. The
        # tracker normally classifies this as weak; normalize direct callers
        # that still report it as reject, while keeping hard area/identity
        # failures rejected.
        if quality_tier == "reject" and self._recoverable_partial_quality(
            bbox_quality_reason, partial_feature, sample_metadata
        ):
            quality_tier = "weak"
        quality_ok = base_quality_ok and quality_tier == "strong"
        weak_quality_ok = base_quality_ok and quality_tier == "weak"
        quality_reason = str(bbox_quality_reason or "").strip()
        diagnostics["search_quality_override"] = bool(search_quality_override)
        diagnostics["search_observation_override"] = bool(search_observation_override)
        diagnostics["quality_confidence_floor"] = float(quality_confidence_floor)
        diagnostics["preferred_search_low_confidence"] = bool(
            (search_quality_override or search_observation_override)
            and float(confidence) < float(self.config.min_confidence)
        )
        if quality_tier != "weak" and not (
            int(preferred_uid or 0) > 0 and bool(preferred_candidate_ok)
        ):
            self.pending_weak_handoffs.pop(track_id, None)

        # A large center jump is a track-swap signal, not ordinary bbox
        # degradation. Drop the stale raw-track claim immediately so the
        # other person can pass through the normal multi-frame handoff gate.
        # Retaining the mapping here would keep refreshing its claim and could
        # block the correct track indefinitely.
        center_jump_rejected = (
            "identity_center_jump>" in quality_reason
            or "identity_swap_competing_track" in quality_reason
        )
        if uid > 0 and center_jump_rejected:
            self.track_to_uid.pop(track_id, None)
            self.track_last_seen_frame.pop(track_id, None)
            self.pending_handoffs.pop(track_id, None)
            self.pending_late_handoffs.pop(track_id, None)
            self.pending_weak_handoffs.pop(track_id, None)
            self.last_assignments[track_id] = {
                "uid": 0,
                "mapped_uid": int(uid),
                "reason": "identity_center_jump_reject",
                "distance": None,
                "bbox_quality_ok": False,
                "bbox_quality_tier": "reject",
                "bbox_quality_reason": quality_reason or None,
            }
            logger.info(
                "ReID身份交换保护: 帧=%d 轨迹ID=%d 原身份=%d 输出身份=0 原因代码=identity_center_jump_reject 质量原因=%s",
                int(frame_index),
                int(track_id),
                int(uid),
                quality_reason or "identity_center_jump",
            )
            return 0
        reason = "mapped" if uid > 0 else "unassigned"
        distance = None
        second_distance = None
        best_uid = None
        best_last_seen_frame = None
        best_frame_gap = None
        handoff_streak = 0
        match_source = None
        bank_updated = False

        if uid <= 0 and feature is not None and (
            weak_quality_ok or search_observation_override
        ):
            return self._assign_weak_search_candidate(
                track_id=track_id,
                feature=feature,
                partial_feature=partial_feature,
                frame_index=frame_index,
                candidate_count=candidate_count,
                quality_reason=quality_reason,
                sample_metadata=sample_metadata,
                preferred_uid=preferred_uid,
                preferred_candidate_ok=preferred_candidate_ok,
                diagnostics=diagnostics,
            )

        if uid > 0 and weak_quality_ok:
            self._remember_track_seen(track_id, uid, frame_index)
            entry = self.identities.get(uid)
            strong_distance = None
            weak_distance = None
            combined_distance = None
            source = None
            bank_updated = False
            if feature is not None and entry is not None:
                strong_distance = entry.distance(feature)
                weak_distance = entry.weak_distance(feature)
                combined_distance, source = entry.weighted_distance(
                    feature,
                    self.config.weak_match_penalty,
                )
                if combined_distance <= float(self.config.weak_update_threshold):
                    entry.touch(frame_index)
                    # Weak observations may help matching, but they never
                    # rewrite the long-lived identity gallery.  The weak tier
                    # is itself the quality rejection boundary, even when a
                    # caller omitted a textual reason.
                    reason = "mapped_weak_observed"
                else:
                    reason = "mapped_weak_distance_reject"
            elif entry is None:
                self.track_to_uid.pop(track_id, None)
                reason = "mapped_missing_identity"
            else:
                reason = "mapped_weak_no_feature"
            self.last_assignments[track_id] = {
                "uid": 0,
                "mapped_uid": int(uid),
                "reason": reason,
                "distance": _finite_float(combined_distance),
                "strong_distance": _finite_float(strong_distance),
                "weak_distance": _finite_float(weak_distance),
                "match_source": source,
                "bank_updated": bool(bank_updated),
                "bbox_quality_ok": False,
                "bbox_quality_tier": "weak",
                "bbox_quality_reason": quality_reason or None,
            }
            logger.info(
                "ReID弱特征观察: 帧=%d 轨迹ID=%d 映射身份=%d 输出身份=0 原因代码=%s "
                "综合距离=%s 强距离=%s 弱距离=%s 匹配来源=%s 更新弱库=%s 质量原因=%s",
                int(frame_index),
                int(track_id),
                int(uid),
                reason,
                _fmt_float(combined_distance),
                _fmt_float(strong_distance),
                _fmt_float(weak_distance),
                source or "none",
                bool(bank_updated),
                quality_reason or "none",
            )
            return 0

        if uid <= 0 and feature is not None and quality_ok:
            preferred_match = self._preferred_search_reacquire_candidate(
                feature=feature,
                partial_feature=partial_feature,
                preferred_uid=preferred_uid,
                candidate_ok=preferred_candidate_ok,
                candidate_count=candidate_count,
                sample_metadata=sample_metadata,
            )
            soft_preferred_handoff = False
            active_preferred_search = (
                bool(self.config.preferred_search_reacquire_enable)
                and preferred_uid is not None
                and int(preferred_uid) > 0
            )
            # The strict gate intentionally stays narrow.  If it rejects a
            # strong candidate during an active search, keep the locked UID as
            # a provisional hypothesis and let the local two-frame observer
            # decide.  This closes the old path where strict rejection erased
            # the candidate before geometry continuity could be evaluated.
            if preferred_match is None and active_preferred_search:
                preferred_match = self._soft_search_reacquire_candidate(
                    feature=feature,
                    partial_feature=partial_feature,
                    preferred_uid=preferred_uid,
                    candidate_count=candidate_count,
                    sample_metadata=sample_metadata,
                    bbox_quality_ok=bbox_quality_ok,
                    bbox_quality_tier=quality_tier,
                )
                soft_preferred_handoff = preferred_match is not None
                if soft_preferred_handoff:
                    diagnostics["soft_search_observation"] = True
                    diagnostics["soft_search_threshold"] = float(
                        self.config.preferred_search_soft_candidate_threshold
                    )
            preferred_handoff = preferred_match is not None
            if bool(
                (sample_metadata or {}).get(
                    "preferred_search_identity_competition_override"
                )
            ):
                diagnostics["preferred_search_identity_competition_override"] = True
            preferred_rejected = False
            if preferred_match is not None:
                (
                    match_uid,
                    distance,
                    second_distance,
                    best_uid,
                    best_last_seen_frame,
                    preferred_match_source,
                ) = preferred_match
                match_source = preferred_match_source
                match_reason = "preferred_search_reacquire"
            else:
                match_uid, distance, second_distance, match_reason, best_uid, best_last_seen_frame = self._match(
                    track_id=track_id,
                    feature=feature,
                    frame_index=frame_index,
                    candidate_count=candidate_count,
                )
                # While searching for a locked UID, never fall back to the
                # global matcher.  A visually similar bystander can otherwise
                # pass the wider controlled-handoff threshold and steal the
                # target after the strict preferred gate rejects it.
                if active_preferred_search:
                    match_uid = 0
                    match_reason = "preferred_search_reacquire_rejected"
                    preferred_rejected = True
            if best_last_seen_frame is not None:
                best_frame_gap = int(frame_index) - int(best_last_seen_frame)
            source_uid = int(match_uid) if int(match_uid) > 0 else best_uid
            if preferred_rejected:
                source_uid = int(preferred_uid)
            self._capture_match_evidence(diagnostics, source_uid, feature)
            if (
                (match_uid > 0 or match_reason == "uid_claimed_recently")
                and self._reject_identity_exclusion(
                    track_id, source_uid, frame_index, sample_metadata, diagnostics,
                )
            ):
                return 0
            if preferred_rejected and diagnostics["match_evidence"] is not None:
                if int(best_uid or 0) != int(source_uid):
                    second_distance = distance
                distance = diagnostics["match_evidence"]["distance"]
            if source_uid is not None and int(source_uid) > 0 and not (
                preferred_handoff
                and (match_source == "partial" or str(match_source or "").startswith("soft_"))
            ):
                match_source = self._match_source(int(source_uid), feature)
            handoff_uid = 0
            if bool(self.config.controlled_handoff_enable):
                if match_uid > 0:
                    handoff_uid = int(match_uid)
                elif match_reason == "uid_claimed_recently" and best_uid is not None:
                    handoff_uid = int(best_uid)

            if handoff_uid > 0:
                geometry = self._handoff_geometry(handoff_uid, sample_metadata, frame_index)
                if soft_preferred_handoff:
                    # Force every soft candidate through the local observer,
                    # even when the old anchor happens to look continuous.
                    # The old anchor is evidence for prediction only; it is
                    # never enough to bind a newly observed raw track.
                    geometry["ok"] = False
                    geometry["reason"] = "soft_candidate"
                    geometry["soft_candidate"] = True
                # A preferred-search candidate is an identity hint, not a new
                # anchor.  Stale or unavailable geometry must never bind the
                # locked UID; otherwise any visually similar bystander can
                # start a fresh confirmation chain after the real target is
                # gone.  A real capture timestamp also enforces the intended
                # short reacquisition window instead of relying on control
                # frame numbering alone.
                if preferred_handoff:
                    capture_age = geometry.get("capture_delta_sec")
                    if capture_age is not None and (
                        not math.isfinite(float(capture_age))
                        or float(capture_age)
                        > max(0.0, float(self.config.preferred_search_reacquire_max_age_sec))
                    ):
                        geometry["ok"] = False
                        geometry["reason"] = "search_reacquire_time_window"
                    # Search reacquisition must have a positive geometry
                    # result.  Missing metadata is not continuity evidence:
                    # accepting it would recreate the stale-anchor bug with
                    # a different reason code.
                    if geometry["ok"] is not True:
                        late_result = None
                        if preferred_handoff:
                            late_result = self._observe_late_search_candidate(
                                track_id=track_id,
                                candidate_uid=handoff_uid,
                                distance=distance,
                                partial_feature=partial_feature,
                                match_source=match_source,
                                candidate_count=candidate_count,
                                bbox_quality_ok=bbox_quality_ok,
                                sample_metadata=sample_metadata,
                                frame_index=frame_index,
                                geometry=geometry,
                                distance_limit=(
                                    float(self.config.preferred_search_soft_candidate_threshold)
                                    if soft_preferred_handoff and match_source == "soft_strong"
                                    else float(self.config.partial_match_threshold)
                                    if soft_preferred_handoff and match_source == "soft_partial"
                                    else float(self.config.controlled_handoff_threshold)
                                    if bool(
                                        (sample_metadata or {}).get(
                                            "preferred_search_geometry_relaxation"
                                        )
                                    )
                                    and match_source == "strong"
                                    else None
                                ),
                            )
                        if late_result is not None:
                            late_uid, late_streak, late_geometry = late_result
                            self._set_geometry_diagnostics(diagnostics, late_geometry)
                            self.last_assignments[track_id] = {
                                "uid": int(late_uid),
                                "reason": (
                                    (
                                        "preferred_search_soft_reacquire"
                                        if soft_preferred_handoff and late_uid > 0
                                        else "preferred_search_soft_candidate_wait"
                                        if soft_preferred_handoff
                                        else "preferred_search_late_reacquire"
                                        if late_uid > 0
                                        else "preferred_search_late_candidate_wait"
                                    )
                                ),
                                "distance": _finite_float(distance),
                                "second_distance": _finite_float(second_distance),
                                "best_uid": best_uid,
                                "best_frame_gap": None if best_frame_gap is None else int(best_frame_gap),
                                "pending_streak": 0,
                                "handoff_streak": int(late_streak),
                                "late_candidate_streak": int(late_streak),
                                "bank_updated": False,
                                "match_source": match_source,
                                "bbox_quality_ok": bool(bbox_quality_ok),
                                "bbox_quality_tier": quality_tier,
                                "bbox_quality_reason": quality_reason or None,
                                "partial_aggregate_distance": _finite_float(
                                    late_geometry.get("partial_aggregate_distance")
                                ),
                                "partial_aggregate_threshold": _finite_float(
                                    late_geometry.get("partial_aggregate_threshold")
                                ),
                                "partial_aggregate_override": bool(
                                    late_geometry.get("partial_aggregate_override", False)
                                ),
                                "late_candidate_rejection": late_geometry.get(
                                    "late_candidate_rejection"
                                ),
                            }
                            if late_uid > 0:
                                self._touch_identity(late_uid, frame_index)
                                current = late_geometry.get("current")
                                if current is not None:
                                    self._remember_geometry_observation(
                                        late_uid,
                                        track_id,
                                        current,
                                    )
                            logger.info(
                                "ReID晚到候选观察: 帧=%d 轨迹ID=%d 候选身份=%d 输出身份=%d "
                                "连续确认=%d/%d 距离=%s 局部中心跳变=%s 面积相似度=%s "
                                "局部时间间隔=%s 局部锚点模式=%s 旋转补偿跳变=%s "
                                "旧锚点已忽略=%s 原锚点原因=%s 局部聚合距离=%s "
                                "聚合阈值=%s 聚合豁免=%s",
                                int(frame_index),
                                int(track_id),
                                int(handoff_uid),
                                int(late_uid),
                                int(late_streak),
                                max(2, int(self.config.preferred_search_reacquire_confirm_frames)),
                                _fmt_float(distance),
                                _fmt_float(late_geometry.get("late_center_jump_ratio")),
                                _fmt_float(late_geometry.get("late_area_similarity")),
                                _fmt_float(late_geometry.get("late_capture_delta_sec")),
                                late_geometry.get("late_anchor_mode", "none"),
                                _fmt_float(late_geometry.get("yaw_compensated_center_jump_ratio")),
                                bool(late_geometry.get("old_anchor_ignored")),
                                geometry.get("reason", "none"),
                                _fmt_float(late_geometry.get("partial_aggregate_distance")),
                                _fmt_float(late_geometry.get("partial_aggregate_threshold")),
                                bool(late_geometry.get("partial_aggregate_override", False)),
                            )
                            return int(late_uid)
                        self.pending_handoffs.pop(track_id, None)
                        self.pending_new.pop(track_id, None)
                        reject_reason = (
                            "handoff_geometry_reject"
                            if geometry["ok"] is False
                            and geometry.get("reason") != "search_reacquire_time_window"
                            else "preferred_search_reacquire_geometry_reject"
                        )
                        self.last_assignments[track_id] = {
                            "uid": 0,
                            "reason": reject_reason,
                            "distance": _finite_float(distance),
                            "second_distance": _finite_float(second_distance),
                            "best_uid": best_uid,
                            "match_source": match_source,
                            "bbox_quality_ok": bool(bbox_quality_ok),
                            "bbox_quality_tier": quality_tier,
                            "reacquire_geometry_ok": geometry["ok"],
                            "reacquire_geometry_reason": geometry["reason"],
                            "partial_aggregate_distance": _finite_float(
                                geometry.get("partial_aggregate_distance")
                            ),
                            "partial_aggregate_threshold": _finite_float(
                                geometry.get("partial_aggregate_threshold")
                            ),
                            "partial_aggregate_override": bool(
                                geometry.get("partial_aggregate_override", False)
                            ),
                            "late_candidate_rejection": geometry.get(
                                "late_candidate_rejection"
                            ),
                            "bank_updated": False,
                        }
                        self._set_geometry_diagnostics(diagnostics, geometry)
                        return 0
                self._set_geometry_diagnostics(diagnostics, geometry)
                instant_threshold = float(
                    self.config.preferred_search_reacquire_instant_threshold
                    if preferred_handoff else self.config.controlled_handoff_instant_threshold
                )
                instant_allowed = bool(
                    geometry["ok"] is True
                    and bbox_quality_ok
                    and match_source == "strong"
                    and distance is not None
                    and instant_threshold > 0.0
                    and float(distance) <= instant_threshold
                )
                revoked_owner_handoff = bool(
                    int(handoff_uid or 0) in self._geometry_revoked_uids
                    and geometry.get("ok") is True and bbox_quality_ok
                    and match_source == "strong" and distance is not None
                    and float(distance) <= float(self.config.controlled_handoff_threshold)
                    and (int(candidate_count) == 1 or (
                        competition.get("uid") == int(handoff_uid)
                        and competition.get("frame_index") == int(frame_index)
                        and competition.get("passed") is True
                    ))
                )
                if int(handoff_uid or 0) in self._geometry_revoked_uids and not revoked_owner_handoff:
                    # Failing this recovery gate must not fall back to the
                    # ordinary d<=instant_threshold single-frame handoff.
                    for pending in (self.pending_new, self.pending_handoffs,
                                    self.pending_late_handoffs, self.pending_weak_handoffs):
                        pending.pop(track_id, None)
                    self.last_assignments[track_id] = {
                        "uid": 0, "reason": "revoked_owner_candidate_unqualified",
                        "best_uid": int(handoff_uid), "distance": _finite_float(distance),
                        "bank_updated": False, "bbox_quality_ok": bool(bbox_quality_ok),
                        "bbox_quality_tier": quality_tier,
                    }
                    return 0
                if revoked_owner_handoff:
                    instant_allowed = False
                preferred_confirm_frames = 1 if instant_allowed else max(
                    2,
                    int(self.config.preferred_search_reacquire_confirm_frames
                        if preferred_handoff else self.config.controlled_handoff_confirm_frames),
                )
                if match_source == "weak":
                    preferred_confirm_frames = max(
                        int(self.config.weak_reacquire_confirm_frames),
                        1 if preferred_confirm_frames is None else int(preferred_confirm_frames),
                    )
                if geometry["ok"] is False:
                    self.pending_handoffs.pop(track_id, None)
                    self.pending_new.pop(track_id, None)
                    self.last_assignments[track_id] = {
                        "uid": 0,
                        "reason": "handoff_geometry_reject",
                        "distance": _finite_float(distance),
                        "second_distance": _finite_float(second_distance),
                        "best_uid": best_uid,
                        "match_source": match_source,
                        "bbox_quality_ok": bool(bbox_quality_ok),
                        "bbox_quality_tier": quality_tier,
                        "bank_updated": False,
                    }
                    return 0
                # A sole, strong, geometrically continuous candidate is
                # already protected by the two-frame handoff below.  Do not
                # make it wait for the conservative five-frame UID vacancy
                # window: DeepSORT commonly creates the replacement track
                # after only two missed frames.  This exception is scoped to
                # strong evidence and never applies to weak/partial or
                # competing candidates.
                sole_strong_handoff = bool(
                    not preferred_handoff
                    and match_source == "strong"
                    and bbox_quality_ok
                    and int(candidate_count) == 1
                    and distance is not None
                    and float(distance) <= float(self.config.controlled_handoff_threshold)
                    and geometry.get("ok") is True
                )
                uid, handoff_streak = self._controlled_handoff_candidate(
                    track_id=track_id,
                    candidate_uid=handoff_uid,
                    distance=distance,
                    second_distance=second_distance,
                    frame_index=frame_index,
                    require_margin=not preferred_handoff,
                    confirm_frames=preferred_confirm_frames,
                    threshold=(
                        self.config.weak_reacquire_threshold
                        if match_source == "weak"
                        else self.config.partial_match_threshold
                        if match_source == "partial"
                        else None
                    ),
                    # During preferred search reacquisition, a DeepSORT
                    # track-id change is expected immediately after a target
                    # leaves/re-enters the frame.  Keep the normal handoff
                    # gap conservative, but allow the search path to bridge
                    # a one-frame gap and accumulate confirmation.
                    min_old_track_gap=(
                        1
                        if preferred_handoff or revoked_owner_handoff
                        else 2
                        if sole_strong_handoff
                        else None
                    ),
                    sample_metadata=sample_metadata,
                )
                bank_updated = False
                if uid > 0:
                    diagnostics["instant_reacquire_allowed"] = instant_allowed
                    reason = "preferred_search_reacquire" if preferred_handoff else "controlled_handoff"
                    self.pending_new.pop(track_id, None)
                    self._touch_identity(uid, frame_index)
                    bank_updated = False
                    if match_source != "partial":
                        bank_updated = self._maybe_add_to_identity(
                            uid,
                            feature,
                            partial_feature,
                            frame_index,
                            distance,
                            sample_metadata=sample_metadata,
                        )
                else:
                    reason = (
                        "preferred_search_soft_candidate_wait"
                        if soft_preferred_handoff
                        else "preferred_search_reacquire_wait"
                        if preferred_handoff
                        else "controlled_handoff_wait"
                    )
            elif match_uid > 0:
                if match_source == "weak":
                    uid = 0
                    reason = "weak_match_requires_handoff"
                    bank_updated = False
                    self.pending_new.pop(track_id, None)
                else:
                    uid = match_uid
                    self._bind_reacquired_identity(uid, track_id, frame_index, sample_metadata)
                    self.pending_new.pop(track_id, None)
                    reason = match_reason
                    self._touch_identity(uid, frame_index)
                    bank_updated = False
                    if match_source != "partial":
                        bank_updated = self._maybe_add_to_identity(
                            uid,
                            feature,
                            partial_feature,
                            frame_index,
                            distance,
                            sample_metadata=sample_metadata,
                        )
            elif preferred_rejected:
                self.pending_handoffs.pop(track_id, None)
                self.pending_late_handoffs.pop(track_id, None)
                self.pending_new.pop(track_id, None)
                uid = 0
                reason = "preferred_search_reacquire_rejected"
                bank_updated = False
            else:
                self.pending_handoffs.pop(track_id, None)
                uid, reason = self._assign_new_or_pending(
                    track_id,
                    feature,
                    partial_feature,
                    frame_index,
                    sample_metadata,
                )
                bank_updated = False
            if uid > 0:
                self.pending_late_handoffs.pop(track_id, None)
                self._remember_track_seen(track_id, uid, frame_index)
                if bbox_quality_ok:
                    self._remember_strong_observation(uid, track_id, sample_metadata, frame_index)
            self.last_assignments[track_id] = {
                "uid": int(uid),
                "reason": reason,
                "distance": _finite_float(distance),
                "partial_distance": self._partial_distance_to_uid(
                    preferred_uid if preferred_uid is not None else best_uid,
                    partial_feature,
                ),
                "second_distance": _finite_float(second_distance),
                "best_uid": None if best_uid is None else int(best_uid),
                "best_frame_gap": None if best_frame_gap is None else int(best_frame_gap),
                "pending_streak": self._pending_streak(track_id),
                "handoff_streak": (
                    int(handoff_streak)
                    if reason in (
                        "controlled_handoff",
                        "preferred_search_reacquire",
                        "preferred_search_soft_reacquire",
                    )
                    else self._handoff_streak(track_id)
                ),
                "reacquire_distance_limit": _finite_float(
                    (sample_metadata or {}).get("reacquire_distance_limit")
                ),
                "bank_updated": bool(bank_updated),
                "match_source": match_source,
                "bbox_quality_ok": bool(bbox_quality_ok),
                "bbox_quality_tier": quality_tier,
                "bbox_quality_reason": quality_reason or None,
            }
            logger.info(
                "ReID身份库分配: 帧=%d 轨迹ID=%d 身份编号=%d 原因代码=%s 最佳身份=%s 距离=%s 次佳距离=%s "
                "匹配来源=%s 局部距离=%s 接回距离阈值=%s 间隔帧=%s 候选数=%d 待确认次数=%d 交接确认次数=%d 更新身份库=%s 质量层级=%s%s",
                int(frame_index),
                int(track_id),
                int(uid),
                reason,
                "none" if best_uid is None else str(int(best_uid)),
                _fmt_float(distance),
                _fmt_float(second_distance),
                match_source or "none",
                _fmt_float(self.last_assignments.get(track_id, {}).get("partial_distance")),
                _fmt_float(
                    self.last_assignments.get(track_id, {}).get(
                        "reacquire_distance_limit"
                    )
                ),
                "none" if best_frame_gap is None else str(int(best_frame_gap)),
                int(candidate_count),
                self._pending_streak(track_id),
                self._handoff_streak(track_id),
                bool(bank_updated),
                quality_tier,
                "" if not quality_reason else f"({quality_reason})",
            )
            return uid

        if uid > 0:
            self._remember_track_seen(track_id, uid, frame_index)
            if not quality_ok:
                # A mapped track can occasionally have a detector confidence
                # just below the normal floor while its fresh, unique ReID
                # embedding is still a strong match.  Keep that observation
                # attached to the already-locked UID as evidence, but do not
                # update the gallery or use it to claim a new UID.  This is
                # deliberately narrower than the search overrides above.
                mapped_distance = None
                mapped_source = None
                mapped_entry = self.identities.get(uid)
                if feature is not None and mapped_entry is not None:
                    mapped_distance, mapped_source = mapped_entry.weighted_distance(
                        feature,
                        self.config.weak_match_penalty,
                    )
                low_confidence_strong_observation = bool(
                    base_quality_ok is False
                    and quality_tier == "strong"
                    and bbox_quality_ok
                    and mapped_source == "strong"
                    and mapped_distance is not None
                    and float(mapped_distance)
                    <= float(self.config.controlled_handoff_instant_threshold)
                    and int(candidate_count) == 1
                    and float(confidence) >= 0.50
                    and bool(metadata.get("is_fresh", True))
                )
                if low_confidence_strong_observation:
                    diagnostics["mapped_low_confidence_strong_observation"] = True
                    diagnostics["mapped_observation_distance"] = float(mapped_distance)
                    self._touch_identity(uid, frame_index)
                    self.last_assignments[track_id] = {
                        "uid": int(uid),
                        "mapped_uid": int(uid),
                        "reason": "mapped_low_confidence_strong_observation",
                        "distance": _finite_float(mapped_distance),
                        "match_source": mapped_source,
                        "bank_updated": False,
                        "bbox_quality_ok": bool(bbox_quality_ok),
                        "bbox_quality_tier": quality_tier,
                        "bbox_quality_reason": quality_reason or None,
                    }
                    logger.info(
                        "ReID身份库映射观察: 帧=%d 轨迹ID=%d 身份编号=%d 输出身份=%d "
                        "原因代码=mapped_low_confidence_strong_observation 距离=%s "
                        "检测置信度=%.3f 候选数=%d 更新身份库=False",
                        int(frame_index),
                        int(track_id),
                        int(uid),
                        int(uid),
                        _fmt_float(mapped_distance),
                        float(confidence),
                        int(candidate_count),
                    )
                    return int(uid)
                if not base_quality_ok:
                    reason = "mapped_low_quality"
                elif not bbox_quality_ok:
                    reason = "mapped_bbox_quality_reject"
                output_uid = 0 if bool(self.config.mapped_bad_quality_returns_unassigned) else uid
                self.last_assignments[track_id] = {
                    "uid": int(output_uid),
                    "mapped_uid": int(uid),
                    "reason": reason,
                    "distance": _finite_float(distance),
                    "bbox_quality_ok": bool(bbox_quality_ok),
                    "bbox_quality_reason": quality_reason or None,
                }
                logger.info(
                    "ReID身份库映射: 帧=%d 轨迹ID=%d 身份编号=%d 输出身份=%d 原因代码=%s 距离=%s 质量合格=%s%s",
                    int(frame_index),
                    int(track_id),
                    int(uid),
                    int(output_uid),
                    reason,
                    _fmt_float(distance),
                    bool(bbox_quality_ok),
                    "" if not quality_reason else f"({quality_reason})",
                )
                return int(output_uid)

            if feature is not None:
                entry = self.identities.get(uid)
                if entry is None:
                    self.track_to_uid.pop(track_id, None)
                    reason = "mapped_missing_identity"
                    self.last_assignments[track_id] = {
                        "uid": 0,
                        "mapped_uid": int(uid),
                        "reason": reason,
                        "distance": None,
                        "bbox_quality_ok": bool(bbox_quality_ok),
                        "bbox_quality_reason": quality_reason or None,
                    }
                    logger.info(
                        "ReID身份库映射: 帧=%d 轨迹ID=%d 身份编号=%d 输出身份=0 原因代码=%s 距离=none 质量合格=%s%s",
                        int(frame_index),
                        int(track_id),
                        int(uid),
                        reason,
                        bool(bbox_quality_ok),
                        "" if not quality_reason else f"({quality_reason})",
                    )
                    return 0

                mapped_distance, match_source = entry.weighted_distance(
                    feature,
                    self.config.weak_match_penalty,
                )
                distance = mapped_distance
                # A mapped track can survive a brief tracker gap, but while
                # search is active it must continue to satisfy the preferred
                # UID distance gate.  Do not let one accepted frame turn a
                # bystander's feature into a long-lived UID1 mapping/gallery
                # update on later frames.
                if (
                    preferred_uid is not None
                    and int(preferred_uid) == int(uid)
                    and mapped_distance > float(self.config.preferred_search_reacquire_threshold)
                ):
                    self.track_to_uid.pop(track_id, None)
                    self.pending_handoffs.pop(track_id, None)
                    reason = "preferred_search_mapped_distance_reject"
                    self.last_assignments[track_id] = {
                        "uid": 0,
                        "mapped_uid": int(uid),
                        "reason": reason,
                        "distance": _finite_float(distance),
                        "match_source": match_source,
                        "bbox_quality_ok": bool(bbox_quality_ok),
                        "bbox_quality_tier": quality_tier,
                        "bank_updated": False,
                    }
                    logger.info(
                        "ReID搜索映射复核拒绝: 帧=%d 轨迹ID=%d 身份编号=%d 距离=%s 阈值=%.3f",
                        int(frame_index), int(track_id), int(uid),
                        _fmt_float(distance),
                        float(self.config.preferred_search_reacquire_threshold),
                    )
                    return 0
                if (
                    bool(self.config.mapped_verify_enable)
                    and mapped_distance > float(self.config.mapped_verify_threshold)
                ):
                    self.track_to_uid.pop(track_id, None)
                    reason = "mapped_verify_reject"
                    self.last_assignments[track_id] = {
                        "uid": 0,
                        "mapped_uid": int(uid),
                        "reason": reason,
                        "distance": _finite_float(distance),
                        "match_source": match_source,
                        "bbox_quality_ok": bool(bbox_quality_ok),
                        "bbox_quality_tier": quality_tier,
                        "bbox_quality_reason": quality_reason or None,
                    }
                    logger.info(
                        "ReID身份库映射: 帧=%d 轨迹ID=%d 身份编号=%d 输出身份=0 原因代码=%s 距离=%s 阈值=%.3f 质量合格=%s%s",
                        int(frame_index),
                        int(track_id),
                        int(uid),
                        reason,
                        _fmt_float(distance),
                        float(self.config.mapped_verify_threshold),
                        bool(bbox_quality_ok),
                        "" if not quality_reason else f"({quality_reason})",
                    )
                    return 0

                geometry = self._handoff_geometry(uid, sample_metadata, frame_index)
                self._set_geometry_diagnostics(diagnostics, geometry)
                reference = entry.last_strong_observation
                preferred_geometry_reject = bool(
                    preferred_uid is not None
                    and int(preferred_uid) == int(uid)
                    and geometry["ok"] is not True
                )
                if (
                    preferred_geometry_reject
                    or (
                        geometry["ok"] is False
                        and reference is not None
                        and int(reference["track_id"]) != track_id
                    )
                ):
                    # A weak-only internal mapping is not a trusted handoff.
                    self.track_to_uid.pop(track_id, None)
                    self.pending_handoffs.pop(track_id, None)
                    geometry_reject_reason = (
                        "preferred_search_reacquire_geometry_reject"
                        if preferred_geometry_reject
                        else "handoff_geometry_reject"
                    )
                    self.last_assignments[track_id] = {
                        "uid": 0,
                        "mapped_uid": int(uid),
                        "reason": geometry_reject_reason,
                        "distance": _finite_float(distance),
                        "match_source": match_source,
                        "bbox_quality_ok": bool(bbox_quality_ok),
                        "bbox_quality_tier": quality_tier,
                        "bank_updated": False,
                    }
                    return 0
                instant_threshold = float(
                    self.config.preferred_search_reacquire_instant_threshold
                    if int(preferred_uid or 0) == uid
                    else self.config.controlled_handoff_instant_threshold
                )
                diagnostics["instant_reacquire_allowed"] = bool(
                    geometry["ok"] is True
                    and bbox_quality_ok
                    and match_source == "strong"
                    and instant_threshold > 0.0
                    and mapped_distance <= instant_threshold
                )
                if self._reacquire_quarantine.is_held(uid):
                    self._observe_template_quarantine(
                        uid=uid, track_id=track_id, feature=feature,
                        confidence=confidence, area=area, frame_index=frame_index,
                        bbox_quality_ok=bool(bbox_quality_ok and geometry["ok"] is True),
                        bbox_quality_tier=quality_tier, metadata=metadata,
                    )
                    diagnostics["_template_observation_evaluated"] = True
                self._touch_identity(uid, frame_index)
                if bbox_quality_ok:
                    self._remember_strong_observation(uid, track_id, sample_metadata, frame_index)
                if geometry["ok"] is False or preferred_geometry_reject:
                    reason = "skip_update_geometry"
                elif self._reacquire_quarantine.is_held(uid):
                    reason = "skip_update_reacquire_quarantine"
                elif self._should_update(frame_index):
                    update_distance = mapped_distance
                    if update_distance <= float(self.config.update_threshold):
                        changed = entry.add(
                            feature,
                            frame_index,
                            max(1, int(self.config.max_features)),
                            self.config.diversity_min_distance,
                            self.config.diversity_replace_margin,
                            sample_metadata,
                        )
                        bank_updated = bool(changed)
                        reason = "updated_diverse" if changed else "skip_update_redundant"
                    else:
                        reason = "skip_update_distance"
        elif uid <= 0:
            if feature is None:
                reason = "no_feature"
            elif not base_quality_ok:
                reason = "low_quality"
            elif not bbox_quality_ok:
                reason = "bbox_quality_reject"
        elif uid > 0 and feature is not None and not quality_ok:
            if not base_quality_ok:
                reason = "skip_update_low_quality"
            elif not bbox_quality_ok:
                reason = "skip_update_bbox_quality"

        self.last_assignments[track_id] = {
            "uid": int(uid),
            "reason": reason,
            "distance": _finite_float(distance),
            "match_source": match_source,
            "bank_updated": bool(bank_updated),
            "bbox_quality_ok": bool(bbox_quality_ok),
            "bbox_quality_tier": quality_tier,
            "bbox_quality_reason": quality_reason or None,
        }
        if reason == "bbox_quality_reject" and quality_reason == "duplicate_person_box":
            logger.info(
                "ReID重复人体框抑制: 帧=%d 轨迹ID=%d 原因代码=duplicate_person_box 候选数=%d",
                int(frame_index),
                int(track_id),
                int(candidate_count),
            )
        return int(uid)

    def _recoverable_partial_quality(
        self,
        quality_reason: str,
        partial_feature: Optional[Any],
        sample_metadata: Optional[dict],
    ) -> bool:
        """Return whether a rejected box can be treated as partial evidence.

        Edge contact by itself is not an identity failure: a person close to
        the camera can legitimately have the head or feet outside the image.
        Only the recoverable edge/aspect reasons are allowed here. Area
        collapse, identity jumps, duplicate boxes and transition failures
        remain hard rejects even when a partial descriptor is available.
        """
        cfg = self.config
        if not bool(cfg.partial_appearance_enable) or partial_feature is None:
            return False
        metadata = sample_metadata or {}
        if metadata.get("partial_observation") is not True:
            return False
        reasons = []
        for item in str(quality_reason or "").split(","):
            normalized = item.strip()
            while normalized.startswith("detector_crop:"):
                normalized = normalized[len("detector_crop:"):].strip()
            if normalized:
                reasons.append(normalized)
        if not reasons or not any(item.startswith("edge_touch>") for item in reasons):
            return False
        return all(item.startswith(("edge_touch>", "aspect<", "aspect>")) for item in reasons)

    def debug_state(self) -> dict:
        return {
            "enabled": bool(self.config.enabled),
            "next_uid": int(self._next_uid),
            "identity_count": int(len(self.identities)),
            "track_to_uid": {str(track_id): int(uid) for track_id, uid in sorted(self.track_to_uid.items())},
            "track_last_seen_frame": {
                str(track_id): int(frame) for track_id, frame in sorted(self.track_last_seen_frame.items())
            },
            "identities": [
                {
                    "uid": int(entry.uid),
                    "features": int(len(entry.features)),
                    "weak_features": int(len(entry.weak_features)),
                    "partial_features": int(len(entry.partial_features)),
                    "last_frame": int(entry.last_frame),
                    "last_weak_frame": int(entry.last_weak_frame),
                    "last_seen_frame": int(entry.last_seen_frame),
                    "updates": int(entry.update_count),
                    "weak_updates": int(entry.weak_update_count),
                    "duplicate_skips": int(entry.duplicate_skip_count),
                    "weak_duplicate_skips": int(entry.weak_duplicate_skip_count),
                    "diversity_replacements": int(entry.diversity_replace_count),
                    "weak_diversity_replacements": int(entry.weak_diversity_replace_count),
                    "last_strong_observation": (
                        None if entry.last_strong_observation is None else dict(entry.last_strong_observation)
                    ),
                }
                for entry in sorted(self.identities.values(), key=lambda item: item.uid)
            ],
            "pending_new": {
                str(track_id): {
                    "first_frame": int(pending.first_frame),
                    "last_frame": int(pending.last_frame),
                    "streak": int(pending.streak),
                }
                for track_id, pending in sorted(self.pending_new.items())
            },
            "pending_handoffs": {
                str(track_id): {
                    "uid": int(pending.uid),
                    "last_frame": int(pending.last_frame),
                    "streak": int(pending.streak),
                    "center_ratio": _finite_float(pending.center_ratio),
                    "area": _finite_float(pending.area),
                }
                for track_id, pending in sorted(self.pending_handoffs.items())
            },
            "pending_late_handoffs": {
                str(track_id): {
                    "uid": int(pending.uid),
                    "last_frame": int(pending.last_frame),
                    "streak": int(pending.streak),
                    "center_ratio": _finite_float(pending.center_ratio),
                    "area": _finite_float(pending.area),
                    "capture_timestamp": _finite_float(pending.capture_timestamp),
                }
                for track_id, pending in sorted(self.pending_late_handoffs.items())
            },
            "pending_weak_handoffs": {
                str(track_id): {
                    "uid": int(pending.uid),
                    "last_frame": int(pending.last_frame),
                    "streak": int(pending.streak),
                    "center_ratio": _finite_float(pending.center_ratio),
                    "area": _finite_float(pending.area),
                }
                for track_id, pending in sorted(self.pending_weak_handoffs.items())
            },
            "last_assignments": {
                str(track_id): assignment for track_id, assignment in sorted(self.last_assignments.items())
            },
        }

    def distance_to_uid(self, uid: int, feature: Any) -> Optional[float]:
        entry = self.identities.get(int(uid))
        if entry is None or not entry.features:
            return None
        distance, _ = entry.weighted_distance(feature, self.config.weak_match_penalty)
        return float(distance)

    def _partial_distance_to_uid(self, uid: int, partial_feature: Any) -> Optional[float]:
        uid = 0 if uid is None else int(uid)
        if (
            not bool(self.config.partial_appearance_enable)
            or uid <= 0
            or partial_feature is None
        ):
            return None
        entry = self.identities.get(int(uid))
        if entry is None or not entry.partial_features:
            return None
        distance = entry.partial_distance(partial_feature)
        return _finite_float(distance)

    def _capture_match_evidence(self, diagnostics: dict, uid: Optional[int], feature: Any) -> None:
        uid = int(uid or 0)
        previous = diagnostics.get("match_evidence")
        if previous is not None and int(previous["matched_uid"]) == uid:
            return
        entry = self.identities.get(uid)
        diagnostics["match_evidence"] = (
            None if feature is None or entry is None
            else entry.match_evidence(feature, self.config.weak_match_penalty)
        )

    @staticmethod
    def _set_geometry_diagnostics(diagnostics: dict, geometry: dict) -> None:
        diagnostics["reacquire_geometry"] = geometry
        diagnostics["reacquire_geometry_ok"] = geometry["ok"]
        diagnostics["reacquire_geometry_reason"] = geometry["reason"]

    def _observe_late_search_candidate(
        self,
        *,
        track_id: int,
        candidate_uid: int,
        distance: Optional[float],
        partial_feature: Optional[Any],
        match_source: Optional[str],
        candidate_count: int,
        bbox_quality_ok: bool,
        sample_metadata: Optional[dict],
        frame_index: int,
        geometry: dict,
        distance_limit: Optional[float] = None,
    ) -> Optional[Tuple[int, int, dict]]:
        """Build a short local tracklet after the original anchor expires.

        Stale UID geometry is not a useful position prediction, but an
        independent co-visible exclusion still forbids this handoff. Local
        continuity alone proves neither identity nor absence of contradiction.
        """
        uid = int(candidate_uid)
        if self._reject_identity_exclusion(
            track_id, uid, frame_index, sample_metadata, {},
        ):
            return 0, 0, {
                **geometry, "ok": False, "reason": "co_visible_distinct_person",
                "late_candidate_rejection": "search_candidate_excluded",
                "old_anchor_ignored": False,
            }
        cfg = self.config
        current = geometry.get("current")
        # A stale anchor is expected after a sweep.  During an active search,
        # a valid candidate can also differ from the old anchor because the
        # camera has rotated or the detector has rebuilt the box.  Those
        # reasons start a *local* observation chain; they never bind a UID by
        # themselves.  Keep the broader exception scoped to search metadata so
        # normal track handoff geometry remains a hard guard.
        search_context_active = bool(
            _evidence_metadata(sample_metadata).get("search_reacquire_context_active")
        )
        allowed_reasons = {"search_reacquire_time_window", "stale_reference"}
        if search_context_active:
            allowed_reasons.update(
                {
                    "center_jump",
                    "area_change",
                    "center_jump,area_change",
                    "incompatible_geometry",
                    "missing_reference",
                    "soft_candidate",
                }
            )
        source_name = str(match_source or "")
        if distance_limit is None:
            distance_limit = (
                float(cfg.partial_match_threshold)
                if source_name == "partial"
                else float(cfg.preferred_search_reacquire_threshold)
            )
        competition_ok = (
            self._soft_candidate_competition_ok(candidate_count, sample_metadata)
            if source_name.startswith("soft_")
            else self._candidate_competition_ok(candidate_count, sample_metadata)
        )
        # `_preferred_search_reacquire_candidate` has already verified that
        # this is the locked UID's unique, full-body strong match. Carry that
        # narrow decision into the late local observer; otherwise the observer
        # would immediately reapply the detector-score gap and discard the
        # very candidate that was allowed to start the chain.
        if bool(
            _evidence_metadata(sample_metadata).get(
                "preferred_search_identity_competition_override"
            )
        ):
            competition_ok = bool(
                source_name == "strong"
                and distance is not None
                and math.isfinite(float(distance))
                and float(distance)
                <= float(cfg.preferred_search_reacquire_instant_threshold)
                and bool(
                    _evidence_metadata(sample_metadata).get(
                        "search_reacquire_context_active"
                    )
                )
            )
        eligible = bool(
            cfg.preferred_search_reacquire_late_candidate_enable
            and uid > 0
            and competition_ok
            and bool(bbox_quality_ok)
            and source_name in {"strong", "partial", "soft_strong", "soft_partial"}
            and distance is not None
            and math.isfinite(float(distance))
            and float(distance) <= float(distance_limit)
            and geometry.get("reason") in allowed_reasons
            and current is not None
            and (
                _finite_float(current.get("edge_touch_count")) is None
                or int(float(current.get("edge_touch_count")))
                <= int(cfg.max_edge_touch_count)
            )
        )
        if not eligible:
            self._clear_late_handoff_for_uid(uid)
            return None

        # Do not let abandoned local anchors accumulate while the vehicle is
        # sweeping.  A local chain is only valid for adjacent observations;
        # an older entry cannot be used as an alternate path around this rule.
        self._prune_late_handoffs(uid, frame_index, current)

        current_center = _finite_float(current.get("center_x_ratio"))
        current_area = _finite_float(current.get("area"))
        current_timestamp = _finite_float(current.get("capture_timestamp"))
        current_yaw = _finite_float(current.get("integrated_yaw_deg"))
        if current_center is None or current_area is None or current_area <= 0.0:
            self._clear_late_handoff_for_uid(uid)
            return None

        pending = self.pending_late_handoffs.get(int(track_id))
        previous_track_id = int(track_id)
        # Weak (edge/partial) evidence and a later strong observation belong
        # to one confirmation chain.  Promote a continuous weak sample into
        # the late-candidate state instead of restarting at 1/2.
        if pending is None:
            for other_track_id, weak_pending in list(self.pending_weak_handoffs.items()):
                if int(weak_pending.uid) != uid:
                    continue
                if int(frame_index) - int(weak_pending.last_frame) != 1:
                    continue
                weak_observation = PendingLateHandoff(
                    uid=uid,
                    last_frame=int(weak_pending.last_frame),
                    streak=int(weak_pending.streak),
                    center_ratio=weak_pending.center_ratio,
                    area=weak_pending.area,
                    area_units=weak_pending.area_units,
                    geometry_source=weak_pending.geometry_source,
                    capture_timestamp=weak_pending.capture_timestamp,
                    integrated_yaw_deg=weak_pending.integrated_yaw_deg,
                )
                if not _pending_observation_continuous(
                    weak_pending,
                    current,
                    frame_index,
                    cfg,
                ):
                    continue
                pending = weak_observation
                previous_track_id = int(other_track_id)
                self.pending_weak_handoffs.pop(int(other_track_id), None)
                logger.info(
                    "ReID接回确认链统一: 身份=%d 弱证据轨迹=%d 晚到轨迹=%d 间隔帧=1 已确认次数=%d 来源=weak_to_late",
                    uid,
                    int(other_track_id),
                    int(track_id),
                    int(weak_pending.streak),
                )
                break
        if pending is None:
            for other_track_id, other_pending in list(self.pending_late_handoffs.items()):
                if int(other_pending.uid) != uid or int(other_track_id) == int(track_id):
                    continue
                gap = int(frame_index) - int(other_pending.last_frame)
                center_delta = float(current_center) - float(other_pending.center_ratio or current_center)
                center_gap = abs(center_delta)
                previous_yaw = _finite_float(other_pending.integrated_yaw_deg)
                if previous_yaw is not None and current_yaw is not None:
                    yaw_ratio = (float(current_yaw) - float(previous_yaw)) / max(
                        1.0, float(cfg.camera_hfov_deg)
                    )
                    center_gap = min(abs(center_delta - yaw_ratio), abs(center_delta + yaw_ratio))
                if gap == 1 and center_gap <= float(cfg.handoff_geometry_max_center_jump_ratio):
                    pending = other_pending
                    previous_track_id = int(other_track_id)
                    break

        local_gap = None if pending is None else int(frame_index) - int(pending.last_frame)
        local_capture_delta = None
        local_center_jump = None
        local_area_similarity = None
        local_ok = pending is None
        compensated_center_jump = None
        aggregate_partial_distance = None
        if pending is not None:
            local_center_jump = abs(float(current_center) - float(pending.center_ratio or current_center))
            local_area_similarity = min(float(current_area), float(pending.area or current_area)) / max(
                float(current_area), float(pending.area or current_area)
            )
            previous_timestamp = _finite_float(pending.capture_timestamp)
            if previous_timestamp is not None and current_timestamp is not None:
                local_capture_delta = float(current_timestamp - previous_timestamp)
                timestamp_ok = (
                    math.isfinite(local_capture_delta)
                    and local_capture_delta > 0.0
                    and local_capture_delta <= max(0.0, float(cfg.preferred_search_reacquire_max_age_sec))
                )
            else:
                timestamp_ok = local_gap == 1
            previous_yaw = _finite_float(pending.integrated_yaw_deg)
            if previous_yaw is not None and current_yaw is not None:
                yaw_ratio = (float(current_yaw) - float(previous_yaw)) / max(
                    1.0, float(cfg.camera_hfov_deg)
                )
                raw_center_delta = float(current_center) - float(pending.center_ratio or current_center)
                # The sign depends on the camera mounting convention.  Pick
                # the smaller residual, retaining a conservative bound.
                compensated_center_jump = min(
                    abs(raw_center_delta - yaw_ratio),
                    abs(raw_center_delta + yaw_ratio),
                )
            local_ok = bool(
                local_gap == 1
                and timestamp_ok
                and pending.area_units == current.get("area_units")
                and pending.geometry_source == current.get("geometry_source")
                and min(
                    float(local_center_jump),
                    float(compensated_center_jump)
                    if compensated_center_jump is not None
                    else float(local_center_jump),
                ) <= float(cfg.handoff_geometry_max_center_jump_ratio)
                and local_area_similarity >= float(cfg.handoff_geometry_min_area_similarity)
            )

        if not local_ok:
            self._clear_late_handoff_for_uid(uid)
            return None

        if pending is None:
            pending = PendingLateHandoff(
                uid=uid,
                last_frame=int(frame_index),
                streak=1,
                center_ratio=float(current_center),
                area=float(current_area),
                area_units=current.get("area_units"),
                geometry_source=current.get("geometry_source"),
                capture_timestamp=current_timestamp,
                partial_feature=(
                    None if partial_feature is None else _normalize_feature(partial_feature)
                ),
                integrated_yaw_deg=current_yaw,
            )
        else:
            aggregate_partial_distance = None
            if (
                partial_feature is not None
                and pending.partial_feature is not None
                and self.config.partial_appearance_enable
            ):
                np = _np()
                current_partial = _normalize_feature(partial_feature)
                previous_partial = _normalize_feature(pending.partial_feature)
                if current_partial.shape == previous_partial.shape:
                    aggregate = _normalize_feature(0.5 * previous_partial + 0.5 * current_partial)
                    entry = self.identities.get(uid)
                    if entry is not None:
                        aggregate_partial_distance = entry.partial_distance(aggregate)
                        if aggregate_partial_distance > float(cfg.partial_match_threshold):
                            # A partial descriptor is deliberately noisy when
                            # the head/feet are clipped.  Do not let it erase
                            # an otherwise strong, geometrically continuous
                            # full-body match.  This exception is intentionally
                            # narrow: the candidate must be the only detector
                            # candidate, use the normal strong source, and be
                            # within the strict preferred-UID threshold.
                            full_strong_continuity = bool(
                                source_name == "strong"
                                and distance is not None
                                and math.isfinite(float(distance))
                                and float(distance)
                                <= float(cfg.preferred_search_reacquire_threshold)
                                and int(candidate_count) == 1
                                and local_gap == 1
                            )
                            late_geometry_fields = {
                                "partial_aggregate_distance": _finite_float(
                                    aggregate_partial_distance
                                ),
                                "partial_aggregate_threshold": float(
                                    cfg.partial_match_threshold
                                ),
                                "partial_aggregate_rejection": (
                                    "overridden_full_strong"
                                    if full_strong_continuity
                                    else "threshold"
                                ),
                                "partial_aggregate_override": bool(
                                    full_strong_continuity
                                ),
                                "full_distance_for_override": _finite_float(distance),
                            }
                            geometry.update(late_geometry_fields)
                            if not full_strong_continuity:
                                self._clear_late_handoff_for_uid(uid)
                                geometry["late_candidate_rejection"] = (
                                    "partial_aggregate_threshold"
                                )
                                logger.info(
                                    "ReID晚到候选拒绝: 帧=%d 轨迹ID=%d 身份=%d 原因=partial_aggregate_threshold "
                                    "全身距离=%s 躯干聚合距离=%s 阈值=%s 候选数=%d 局部间隔帧=%s",
                                    int(frame_index),
                                    int(track_id),
                                    int(uid),
                                    _fmt_float(distance),
                                    _fmt_float(aggregate_partial_distance),
                                    _fmt_float(cfg.partial_match_threshold),
                                    int(candidate_count),
                                    "none" if local_gap is None else str(int(local_gap)),
                                )
                                return None
                            logger.info(
                                "ReID晚到候选躯干聚合豁免: 帧=%d 轨迹ID=%d 身份=%d "
                                "全身距离=%s 躯干聚合距离=%s 阈值=%s 原因=唯一候选同轨迹连续强匹配",
                                int(frame_index),
                                int(track_id),
                                int(uid),
                                _fmt_float(distance),
                                _fmt_float(aggregate_partial_distance),
                                _fmt_float(cfg.partial_match_threshold),
                            )
                    pending.partial_feature = aggregate
            pending.last_frame = int(frame_index)
            pending.streak += 1
            pending.center_ratio = float(current_center)
            pending.area = float(current_area)
            pending.capture_timestamp = current_timestamp
            pending.integrated_yaw_deg = current_yaw
            self.pending_late_handoffs.pop(previous_track_id, None)
        self.pending_late_handoffs[int(track_id)] = pending

        required = max(2, int(cfg.preferred_search_reacquire_confirm_frames))
        confirmed = int(pending.streak) >= required
        late_geometry = dict(geometry)
        late_geometry["ok"] = bool(confirmed)
        late_geometry["reason"] = (
            "late_candidate_local_continuity" if confirmed else "late_candidate_observation"
        )
        late_geometry["late_candidate"] = True
        late_geometry["late_center_jump_ratio"] = local_center_jump
        late_geometry["yaw_compensated_center_jump_ratio"] = compensated_center_jump
        late_geometry["late_area_similarity"] = local_area_similarity
        late_geometry["late_capture_delta_sec"] = local_capture_delta
        late_geometry["late_frame_gap"] = local_gap
        late_geometry["late_anchor_mode"] = (
            "seed" if local_gap is None else "rolling"
        )
        late_geometry["old_anchor_ignored"] = geometry.get("reason") in {
            "search_reacquire_time_window",
            "stale_reference",
            "center_jump",
            "area_change",
            "center_jump,area_change",
            "incompatible_geometry",
            "missing_reference",
            "soft_candidate",
        }
        late_geometry["old_anchor_frame_gap"] = geometry.get("frame_gap")
        late_geometry["old_anchor_capture_delta_sec"] = geometry.get(
            "capture_delta_sec"
        )
        if pending.partial_feature is not None:
            late_geometry["partial_aggregate_available"] = True
            late_geometry["partial_aggregate_distance"] = _finite_float(
                aggregate_partial_distance
            )
        if confirmed:
            for other_track_id, other_uid in list(self.track_to_uid.items()):
                if int(other_track_id) != int(track_id) and int(other_uid) == uid:
                    self.track_to_uid.pop(int(other_track_id), None)
            self._bind_reacquired_identity(uid, track_id, frame_index, sample_metadata)
            self._remember_track_seen(track_id, uid, frame_index)
            self.pending_late_handoffs.pop(int(track_id), None)
            self._clear_late_handoff_for_uid(uid, keep_track_id=int(track_id))
        return (uid if confirmed else 0), int(pending.streak), late_geometry

    def _prune_late_handoffs(
        self, uid: int, frame_index: int, current: Optional[dict]
    ) -> None:
        """Drop local anchors that can no longer form an adjacent-frame pair."""
        current_timestamp = None if current is None else _finite_float(
            current.get("capture_timestamp")
        )
        max_frame_gap = 1
        max_capture_age = max(
            0.0, float(self.config.preferred_search_reacquire_max_age_sec)
        )
        for track_id, pending in list(self.pending_late_handoffs.items()):
            if int(pending.uid) != int(uid):
                continue
            if int(frame_index) - int(pending.last_frame) > max_frame_gap:
                self.pending_late_handoffs.pop(int(track_id), None)
                continue
            previous_timestamp = _finite_float(pending.capture_timestamp)
            if (
                current_timestamp is not None
                and previous_timestamp is not None
                and current_timestamp - previous_timestamp > max_capture_age
            ):
                self.pending_late_handoffs.pop(int(track_id), None)

    def _clear_late_handoff_for_uid(self, uid: int, keep_track_id: Optional[int] = None) -> None:
        for track_id, pending in list(self.pending_late_handoffs.items()):
            if int(pending.uid) == int(uid) and (
                keep_track_id is None or int(track_id) != int(keep_track_id)
            ):
                self.pending_late_handoffs.pop(int(track_id), None)

    def _remember_geometry_observation(self, uid: int, track_id: int, current: dict) -> None:
        entry = self.identities.get(int(uid))
        if entry is None:
            return
        observation = dict(current)
        observation["track_id"] = int(track_id)
        entry.last_strong_observation = observation

    def _handoff_geometry(
        self, uid: int, metadata: Optional[dict], frame_index: int, *,
        reference_override: Optional[dict] = None,
    ) -> dict:
        entry = self.identities.get(int(uid))
        reference = reference_override if reference_override is not None else (
            None if entry is None else entry.last_strong_observation
        )
        current = _geometry_observation(metadata, frame_index)
        result = {
            "ok": None,
            "reason": "missing_reference",
            "reference": None if reference is None else dict(reference),
            "current": current,
            "frame_gap": None,
            "center_jump_ratio": None,
            "yaw_compensated_center_jump_ratio": None,
            "area_similarity": None,
            "capture_delta_sec": None,
        }
        if reference is None:
            return result
        gap = int(frame_index) - int(reference["frame_index"])
        result["frame_gap"] = gap
        reference_ts = _finite_float(reference.get("capture_timestamp"))
        current_ts = None if current is None else _finite_float(current.get("capture_timestamp"))
        if reference_ts is not None and current_ts is not None:
            result["capture_delta_sec"] = float(current_ts - reference_ts)
        if gap < 0 or (reference_override is None and gap > max(1, int(self.config.handoff_geometry_max_gap_frames))):
            result["reason"] = "stale_reference"
            return result
        if current is None:
            result["reason"] = "missing_current_geometry"
            return result
        if (
            current["geometry_source"] != reference["geometry_source"]
            or current["area_units"] != reference["area_units"]
        ):
            result["reason"] = "incompatible_geometry"
            return result
        center_jump = abs(float(current["center_x_ratio"]) - float(reference["center_x_ratio"]))
        yaw_compensated_center_jump = center_jump
        reference_yaw = _finite_float(reference.get("integrated_yaw_deg"))
        current_yaw = _finite_float(current.get("integrated_yaw_deg"))
        if reference_yaw is not None and current_yaw is not None:
            _, yaw_compensated_center_jump = _yaw_compensated_center_jump(
                current_center=float(current["center_x_ratio"]),
                previous_center=float(reference["center_x_ratio"]),
                current_yaw=current_yaw,
                previous_yaw=reference_yaw,
                camera_hfov_deg=self.config.camera_hfov_deg,
            )
        area_similarity = min(current["area"], reference["area"]) / max(current["area"], reference["area"])
        result["center_jump_ratio"] = center_jump
        result["yaw_compensated_center_jump_ratio"] = yaw_compensated_center_jump
        result["area_similarity"] = area_similarity
        reasons = []
        max_center_jump = max(
            0.0, float(self.config.handoff_geometry_max_center_jump_ratio)
        )
        # During a preferred-UID search, the camera is intentionally moving.
        # A strong, unique candidate may therefore cross a few extra pixels
        # after IMU compensation. Keep this exception scoped to the candidate
        # path that explicitly opted into it; ordinary handoffs retain the
        # stricter geometry limit.
        if bool((metadata or {}).get("preferred_search_geometry_relaxation")):
            max_center_jump = max(max_center_jump, 0.30)
        if yaw_compensated_center_jump > max_center_jump:
            reasons.append("center_jump")
        if area_similarity < max(0.0, min(1.0, float(self.config.handoff_geometry_min_area_similarity))):
            reasons.append("area_change")
        result["ok"] = not reasons
        result["reason"] = ",".join(reasons) if reasons else "continuous"
        return result

    def _remember_strong_observation(
        self, uid: int, track_id: int, metadata: Optional[dict], frame_index: int
    ) -> None:
        entry = self.identities.get(int(uid))
        if entry is None:
            return
        geometry = self._handoff_geometry(uid, metadata, frame_index)
        if geometry["ok"] is False or geometry["current"] is None:
            return
        observation = dict(geometry["current"])
        observation["track_id"] = int(track_id)
        entry.last_strong_observation = observation

    def _create_identity(
        self,
        feature: Any,
        frame_index: int,
        sample_metadata: Optional[dict] = None,
        partial_feature: Optional[Any] = None,
    ) -> int:
        uid = int(self._next_uid)
        self._next_uid += 1
        entry = IdentityEntry(uid)
        entry.add(
            feature,
            frame_index,
            max(1, int(self.config.max_features)),
            self.config.diversity_min_distance,
            self.config.diversity_replace_margin,
            sample_metadata,
        )
        if partial_feature is not None and self.config.partial_appearance_enable:
            entry.add_partial(
                partial_feature,
                frame_index,
                max(1, int(self.config.partial_max_features)),
                self.config.diversity_min_distance,
                sample_metadata,
            )
        self.identities[uid] = entry
        return uid

    def _match(
        self,
        track_id: int,
        feature: Any,
        *,
        frame_index: int,
        candidate_count: int,
    ) -> Tuple[int, Optional[float], Optional[float], str, Optional[int], Optional[int]]:
        distances = [
            (
                entry.uid,
                entry.weighted_distance(feature, self.config.weak_match_penalty)[0],
                entry.last_seen_frame,
            )
            for entry in self.identities.values()
            if entry.features
        ]
        if not distances:
            return 0, None, None, "unmatched", None, None
        distances.sort(key=lambda item: item[1])
        best_uid, best_distance, best_last_seen_frame = distances[0]
        second_distance = distances[1][1] if len(distances) > 1 else None
        normal_margin_ok = self._margin_ok(best_distance, second_distance, float(self.config.match_margin))
        if best_distance <= float(self.config.match_threshold) and normal_margin_ok:
            claim_track, claim_gap = self._recent_uid_claim(best_uid, track_id, frame_index)
            if claim_track is not None:
                logger.info(
                    "ReID身份占用冲突: 帧=%d 轨迹ID=%d 身份编号=%d 当前占用轨迹=%d 占用间隔帧=%d 距离=%s 原因代码=matched",
                    int(frame_index),
                    int(track_id),
                    int(best_uid),
                    int(claim_track),
                    int(claim_gap),
                    _fmt_float(best_distance),
                )
                return 0, best_distance, second_distance, "uid_claimed_recently", int(best_uid), int(best_last_seen_frame)
            return int(best_uid), best_distance, second_distance, "matched", int(best_uid), int(best_last_seen_frame)

        reacquire_ok, reacquire_reason = self._reacquire_allowed(
            best_distance=best_distance,
            second_distance=second_distance,
            best_last_seen_frame=best_last_seen_frame,
            frame_index=frame_index,
            candidate_count=candidate_count,
        )
        if reacquire_ok:
            claim_track, claim_gap = self._recent_uid_claim(best_uid, track_id, frame_index)
            if claim_track is not None:
                logger.info(
                    "ReID身份占用冲突: 帧=%d 轨迹ID=%d 身份编号=%d 当前占用轨迹=%d 占用间隔帧=%d 距离=%s 原因代码=%s",
                    int(frame_index),
                    int(track_id),
                    int(best_uid),
                    int(claim_track),
                    int(claim_gap),
                    _fmt_float(best_distance),
                    reacquire_reason,
                )
                return 0, best_distance, second_distance, "uid_claimed_recently", int(best_uid), int(best_last_seen_frame)
            return int(best_uid), best_distance, second_distance, reacquire_reason, int(best_uid), int(best_last_seen_frame)

        return 0, best_distance, second_distance, "unmatched", int(best_uid), int(best_last_seen_frame)

    def _partial_or_weighted_distance(
        self,
        entry: IdentityEntry,
        feature: Any,
        partial_feature: Optional[Any],
        partial_observation: bool,
    ) -> float:
        """Compare a partial crop against the same tier of every identity."""
        if (
            partial_observation
            and partial_feature is not None
            and bool(self.config.partial_appearance_enable)
            and entry.partial_features
        ):
            partial_distance = entry.partial_distance(partial_feature)
            if (
                math.isfinite(float(partial_distance))
                and float(partial_distance) <= float(self.config.partial_match_threshold)
            ):
                return float(partial_distance)
        return float(entry.weighted_distance(feature, self.config.weak_match_penalty)[0])

    def _assign_weak_search_candidate(
        self,
        *,
        track_id: int,
        feature: Any,
        partial_feature: Optional[Any],
        frame_index: int,
        candidate_count: int,
        quality_reason: str,
        sample_metadata: Optional[dict],
        preferred_uid: Optional[int],
        preferred_candidate_ok: bool,
        diagnostics: dict,
    ) -> int:
        """Use a weak box as identity evidence while keeping its control UID at zero."""
        uid = 0 if preferred_uid is None else int(preferred_uid)
        target_entry = self.identities.get(uid)
        reason = "weak_bbox_unassigned"
        distance = None
        strong_distance = None
        weak_distance = None
        second_distance = None
        source = None
        streak = 0
        confirmed = False
        bank_updated = False

        candidate_allowed = bool(
            bool(self.config.preferred_search_reacquire_enable)
            and bool(preferred_candidate_ok)
            and self._candidate_competition_ok(candidate_count, sample_metadata)
            and uid > 0
            and target_entry is not None
            and bool(target_entry.features)
        )
        if candidate_allowed and target_entry is not None:
            partial_observation = bool((sample_metadata or {}).get("partial_observation"))
            partial_distance = None
            if (
                partial_observation
                and partial_feature is not None
                and bool(self.config.partial_appearance_enable)
                and target_entry.partial_features
            ):
                partial_distance = target_entry.partial_distance(partial_feature)
            if (
                partial_distance is not None
                and math.isfinite(float(partial_distance))
                and float(partial_distance) <= float(self.config.partial_match_threshold)
            ):
                distance = float(partial_distance)
                source = "partial"
            else:
                distance, source = target_entry.weighted_distance(
                    feature,
                    self.config.weak_match_penalty,
                )
            strong_distance = target_entry.distance(feature)
            weak_distance = target_entry.weak_distance(feature)
            competing = sorted(
                float(self._partial_or_weighted_distance(
                    entry,
                    feature,
                    partial_feature,
                    partial_observation,
                ))
                for other_uid, entry in self.identities.items()
                if int(other_uid) != uid and entry.features
            )
            second_distance = competing[0] if competing else None
            disadvantage = (
                0.0
                if second_distance is None
                else max(0.0, float(distance) - float(second_distance))
            )
            search_observation_only = bool(
                (sample_metadata or {}).get("search_observation_only")
            )
            distance_limit = float(
                self.config.partial_match_threshold
                if source == "partial"
                else self.config.preferred_search_soft_candidate_threshold
                if search_observation_only
                else self.config.weak_reacquire_threshold
            )
            (sample_metadata or {})["reacquire_distance_limit"] = float(distance_limit)
            geometry = self._handoff_geometry(uid, sample_metadata, frame_index)
            self._set_geometry_diagnostics(diagnostics, geometry)
            if geometry["ok"] is False:
                self.pending_weak_handoffs.pop(track_id, None)
                reason = "weak_handoff_geometry_reject"
            elif (
                float(distance) <= distance_limit
                and disadvantage <= float(self.config.preferred_search_reacquire_max_disadvantage)
            ):
                self._bridge_weak_handoff_observation(
                    track_id=track_id,
                    candidate_uid=uid,
                    frame_index=frame_index,
                    sample_metadata=sample_metadata,
                )
                confirmed, streak = self._weak_handoff_candidate(
                    track_id=track_id,
                    candidate_uid=uid,
                    frame_index=frame_index,
                    sample_metadata=sample_metadata,
                )
                reason = (
                    "weak_preferred_reacquire_confirmed"
                    if confirmed
                    else "weak_preferred_reacquire_wait"
                )
                if confirmed:
                    target_entry.touch(frame_index)
                    # A weak reacquisition can confirm an existing identity,
                    # but it must not change the gallery used by future matches.
                    bank_updated = False
            else:
                self.pending_weak_handoffs.pop(track_id, None)
                reason = "weak_preferred_reacquire_rejected"
        else:
            self.pending_weak_handoffs.pop(track_id, None)

        self.last_assignments[track_id] = {
            "uid": 0,
            "mapped_uid": int(uid) if confirmed else 0,
            "reason": reason,
            "distance": _finite_float(distance),
            "strong_distance": _finite_float(strong_distance),
            "weak_distance": _finite_float(weak_distance),
            "second_distance": _finite_float(second_distance),
            "best_uid": int(uid) if candidate_allowed else None,
            "match_source": source,
            "search_observation_only": bool(
                (sample_metadata or {}).get("search_observation_only")
            ),
            "reacquire_distance_limit": _finite_float(
                distance_limit if candidate_allowed and target_entry is not None else None
            ),
            "handoff_streak": int(streak),
            "bank_updated": bool(bank_updated),
            "bbox_quality_ok": False,
            "bbox_quality_tier": "weak",
            "bbox_quality_reason": quality_reason or None,
        }
        logger.info(
            "ReID弱特征搜索证据: 帧=%d 轨迹ID=%d 候选身份=%d 输出身份=0 原因代码=%s "
            "连续确认次数=%d/%d 综合距离=%s 强距离=%s 弱距离=%s 次佳距离=%s 匹配来源=%s 更新弱库=%s 质量原因=%s",
            int(frame_index),
            int(track_id),
            int(uid),
            reason,
            int(streak),
            int(max(1, self.config.weak_reacquire_confirm_frames)),
            _fmt_float(distance),
            _fmt_float(strong_distance),
            _fmt_float(weak_distance),
            _fmt_float(second_distance),
            source or "none",
            bool(bank_updated),
            quality_reason or "none",
        )
        return 0

    def _bridge_weak_handoff_observation(
        self,
        *,
        track_id: int,
        candidate_uid: int,
        frame_index: int,
        sample_metadata: Optional[dict],
    ) -> None:
        """Continue weak confirmation when DeepSORT changes the raw track id.

        A detector probe uses ``-1`` while a formal DeepSORT observation may
        receive a new id on the next frame.  The candidate must still be a
        locally continuous, sole observation; otherwise the confirmation is
        intentionally reset.
        """
        current = _local_reacquire_observation(sample_metadata, frame_index)
        current_center = None if current is None else _finite_float(current.get("center_x_ratio"))
        current_area = None if current is None else _finite_float(current.get("area"))
        if current_center is None:
            return
        uid = int(candidate_uid)
        for previous_track_id, pending in list(self.pending_weak_handoffs.items()):
            if int(previous_track_id) == int(track_id) or int(pending.uid) != uid:
                continue
            if int(frame_index) - int(pending.last_frame) != 1:
                continue
            previous_center = _finite_float(pending.center_ratio)
            if previous_center is None:
                continue
            center_jump, compensated_center_jump = _yaw_compensated_center_jump(
                current_center=current_center,
                previous_center=previous_center,
                current_yaw=_finite_float(current.get("integrated_yaw_deg")),
                previous_yaw=_finite_float(pending.integrated_yaw_deg),
                camera_hfov_deg=self.config.camera_hfov_deg,
            )
            area_similarity = 1.0
            if pending.area is not None and current_area is not None and float(pending.area) > 0.0:
                if pending.area_units is not None and current.get("area_units") not in (None, pending.area_units):
                    continue
                area_similarity = min(float(current_area), float(pending.area)) / max(
                    float(current_area), float(pending.area), 1e-9
                )
            if (
                min(center_jump, compensated_center_jump)
                > float(self.config.handoff_geometry_max_center_jump_ratio)
                or area_similarity < float(self.config.handoff_geometry_min_area_similarity)
            ):
                continue
            self.pending_weak_handoffs[int(track_id)] = PendingHandoff(
                uid=uid,
                last_frame=int(pending.last_frame),
                streak=int(pending.streak),
                center_ratio=float(previous_center),
                area=pending.area,
                area_units=pending.area_units,
                geometry_source=pending.geometry_source,
                capture_timestamp=pending.capture_timestamp,
                integrated_yaw_deg=pending.integrated_yaw_deg,
            )
            self.pending_weak_handoffs.pop(int(previous_track_id), None)
            logger.info(
                "ReID弱证据接力: 身份=%d 旧轨迹=%d 新轨迹=%d 间隔帧=1 连续确认=%d 来源=weak",
                uid,
                int(previous_track_id),
                int(track_id),
                int(pending.streak),
            )
            return

        # A strong late-candidate observation and a weak cropped observation
        # are the same local confirmation stream. Bridge them when the frame
        # and geometry are continuous instead of starting a second counter.
        for previous_track_id, pending in list(self.pending_late_handoffs.items()):
            if int(previous_track_id) == int(track_id) or int(pending.uid) != uid:
                continue
            if int(frame_index) - int(pending.last_frame) != 1:
                continue
            previous_center = _finite_float(pending.center_ratio)
            previous_area = _finite_float(pending.area)
            if previous_center is None:
                continue
            center_jump, compensated_center_jump = _yaw_compensated_center_jump(
                current_center=current_center,
                previous_center=previous_center,
                current_yaw=_finite_float(current.get("integrated_yaw_deg")),
                previous_yaw=_finite_float(pending.integrated_yaw_deg),
                camera_hfov_deg=self.config.camera_hfov_deg,
            )
            area_similarity = 1.0
            if previous_area is not None and current_area is not None and previous_area > 0:
                area_similarity = min(float(current_area), float(previous_area)) / max(
                    float(current_area), float(previous_area), 1e-9
                )
            if (
                min(center_jump, compensated_center_jump)
                > float(self.config.handoff_geometry_max_center_jump_ratio)
                or area_similarity < float(self.config.handoff_geometry_min_area_similarity)
            ):
                continue
            self.pending_weak_handoffs[int(track_id)] = PendingHandoff(
                uid=uid,
                last_frame=int(pending.last_frame),
                streak=int(pending.streak),
                center_ratio=float(previous_center),
                area=pending.area,
                area_units=pending.area_units,
                geometry_source=pending.geometry_source,
                capture_timestamp=pending.capture_timestamp,
                integrated_yaw_deg=pending.integrated_yaw_deg,
            )
            self.pending_late_handoffs.pop(int(previous_track_id), None)
            logger.info(
                "ReID弱证据接力: 身份=%d 旧轨迹=%d 新轨迹=%d 间隔帧=1 连续确认=%d 来源=late_to_weak",
                uid,
                int(previous_track_id),
                int(track_id),
                int(pending.streak),
            )
            return

    def _weak_handoff_candidate(
        self,
        *,
        track_id: int,
        candidate_uid: int,
        frame_index: int,
        sample_metadata: Optional[dict] = None,
    ) -> Tuple[bool, int]:
        """Confirm weak identity evidence without exposing a control UID."""
        uid = int(candidate_uid)
        claim_last_seen = None
        for other_track_id, other_uid in self.track_to_uid.items():
            if int(other_track_id) == int(track_id) or int(other_uid) != uid:
                continue
            last_seen = self.track_last_seen_frame.get(int(other_track_id))
            if last_seen is not None and (
                claim_last_seen is None or int(last_seen) > int(claim_last_seen)
            ):
                claim_last_seen = int(last_seen)
        entry = self.identities.get(uid)
        if claim_last_seen is None and entry is not None:
            claim_last_seen = int(entry.last_seen_frame)
        claim_gap = None if claim_last_seen is None else int(frame_index) - int(claim_last_seen)
        min_gap = max(1, int(self.config.controlled_handoff_min_old_track_gap_frames))
        if claim_gap is None or int(claim_gap) < min_gap:
            self.pending_weak_handoffs.pop(int(track_id), None)
            return False, 0

        current = _local_reacquire_observation(sample_metadata, frame_index)
        current_center = None if current is None else _finite_float(current.get("center_x_ratio"))
        current_area = None if current is None else _finite_float(current.get("area"))
        current_units = None if current is None else current.get("area_units")
        current_source = None if current is None else current.get("geometry_source")
        current_timestamp = None if current is None else _finite_float(current.get("capture_timestamp"))
        pending = self.pending_weak_handoffs.get(int(track_id))
        if pending is None or int(pending.uid) != uid or int(frame_index) != int(pending.last_frame) + 1:
            pending = PendingHandoff(
                uid=uid,
                last_frame=int(frame_index),
                streak=1,
                center_ratio=current_center,
                area=current_area,
                area_units=current_units,
                geometry_source=current_source,
                capture_timestamp=current_timestamp,
                integrated_yaw_deg=(
                    None if current is None else _finite_float(current.get("integrated_yaw_deg"))
                ),
            )
            self.pending_weak_handoffs[int(track_id)] = pending
        else:
            if not _pending_observation_continuous(
                pending,
                current,
                frame_index,
                self.config,
            ):
                self.pending_weak_handoffs.pop(int(track_id), None)
                return False, 0
            pending.last_frame = int(frame_index)
            pending.streak += 1
            if current_center is not None:
                pending.center_ratio = float(current_center)
            if current is not None:
                pending.area = current_area
                pending.area_units = current_units
                pending.geometry_source = current_source
                pending.capture_timestamp = current_timestamp
                pending.integrated_yaw_deg = _finite_float(current.get("integrated_yaw_deg"))
        required = max(2, int(self.config.weak_reacquire_confirm_frames))
        if int(pending.streak) < required:
            return False, int(pending.streak)

        for other_track_id, other_uid in list(self.track_to_uid.items()):
            if int(other_track_id) != int(track_id) and int(other_uid) == uid:
                self.track_to_uid.pop(int(other_track_id), None)
        self._bind_reacquired_identity(uid, track_id, frame_index, sample_metadata)
        self.pending_weak_handoffs.pop(int(track_id), None)
        return True, int(pending.streak)

    def _match_source(self, uid: int, feature: Any) -> Optional[str]:
        entry = self.identities.get(int(uid))
        if entry is None:
            return None
        _, source = entry.weighted_distance(feature, self.config.weak_match_penalty)
        return source

    def _preferred_search_reacquire_candidate(
        self,
        *,
        feature: Any,
        partial_feature: Optional[Any],
        preferred_uid: Optional[int],
        candidate_ok: bool,
        candidate_count: int = 1,
        sample_metadata: Optional[dict] = None,
    ) -> Optional[Tuple[int, float, Optional[float], int, int, str]]:
        """Select the locked UID during search without widening global ReID matching."""
        cfg = self.config
        uid = 0 if preferred_uid is None else int(preferred_uid)
        metadata = sample_metadata or {}
        # Search direction is a motor policy, not an identity constraint.  A
        # target can cross the optical centre while the chassis is still
        # completing its frozen sweep.  Permit only a strong full-body ReID
        # candidate from the opposite side to enter the normal preferred-UID
        # geometry/confirmation path.  Weak/partial evidence still has to
        # come from the direction-compatible path, preserving the protection
        # against a visually similar bystander.
        opposite_strong_candidate = bool(
            not candidate_ok
            and metadata.get("search_reacquire_context_active") is True
            and metadata.get("search_direction_compatible") is False
            and str(metadata.get("bbox_quality_tier", "")).strip().lower() == "strong"
        )
        if (
            not bool(cfg.preferred_search_reacquire_enable)
            or not bool(candidate_ok or opposite_strong_candidate)
            or uid <= 0
        ):
            return None
        entry = self.identities.get(uid)
        if entry is None or not entry.features:
            return None

        # Candidate count is only a warning signal.  A strong preferred
        # candidate may coexist with low-confidence detector fragments; use
        # the score gap to distinguish that case from two equally plausible
        # people in the frame.  The decision is made after ReID distances are
        # available below so a locked UID can use a narrow opposite-side
        # exception without weakening normal/global matching.
        competition_ok = self._candidate_competition_ok(candidate_count, metadata)

        partial_allowed = bool(
            cfg.partial_appearance_enable
            and partial_feature is not None
            and bool((sample_metadata or {}).get("partial_observation"))
            and competition_ok
        )

        def score(item: IdentityEntry) -> Tuple[float, str]:
            full_distance = float(item.weighted_distance(feature, cfg.weak_match_penalty)[0])
            if partial_allowed and item.partial_features:
                partial_distance = float(item.partial_distance(partial_feature))
                if math.isfinite(partial_distance) and partial_distance <= float(cfg.partial_match_threshold):
                    return partial_distance, "partial"
            return full_distance, "strong"

        distances = [
            (
                int(item.uid),
                float(score(item)[0]),
                int(item.last_seen_frame),
                score(item)[1],
            )
            for item in self.identities.values()
            if item.features
        ]
        distances.sort(key=lambda item: item[1])
        if not distances:
            return None

        target_distance, target_source = score(entry)
        best_uid, best_distance, best_last_seen_frame, _ = distances[0]
        disadvantage = max(0.0, target_distance - float(best_distance))
        # Partial crops use their separate torso-gallery threshold. Applying
        # the full-body gate here would reject the partial-observation path
        # whenever the head or feet leave the frame.
        distance_limit = (
            float(cfg.partial_match_threshold)
            if target_source == "partial"
            else float(cfg.controlled_handoff_threshold)
            if opposite_strong_candidate
            else float(cfg.preferred_search_reacquire_threshold)
        )
        # Keep the effective search-only threshold visible in the structured
        # evidence log.  The ordinary global matcher must remain unchanged;
        # this value is only diagnostic for the locked-UID handoff path.
        metadata["reacquire_distance_limit"] = float(distance_limit)
        if target_distance > distance_limit:
            return None
        if disadvantage > float(cfg.preferred_search_reacquire_max_disadvantage):
            return None

        competing = [distance for candidate_uid, distance, _, _ in distances if candidate_uid != uid]
        second_distance = competing[0] if competing else None
        # During a frozen sweep the true person can reappear on the opposite
        # side while another detector box has a slightly higher YOLO score.
        # Let only a very strong, unambiguous match to the already locked UID
        # enter the existing two-frame geometry observer.  This is deliberately
        # narrower than the normal preferred threshold and never applies to
        # partial/soft features or to direction-compatible weak candidates.
        strong_identity_override = bool(
            opposite_strong_candidate
            and target_source == "strong"
            and target_distance <= float(cfg.preferred_search_reacquire_instant_threshold)
            and int(best_uid) == int(uid)
            and (
                second_distance is None
                or float(second_distance) - float(target_distance) >= 0.05
            )
        )
        if not competition_ok and not strong_identity_override:
            return None
        if strong_identity_override:
            metadata["preferred_search_identity_competition_override"] = True
        if opposite_strong_candidate and target_source == "strong":
            # Direction remains a motor/search policy. A unique strong match
            # to the locked UID may still start the normal two-frame handoff,
            # with a slightly wider IMU-compensated geometry tolerance.
            metadata["preferred_search_geometry_relaxation"] = True
        return (
            uid,
            target_distance,
            second_distance,
            int(best_uid),
            int(best_last_seen_frame),
            target_source,
        )

    def _soft_search_reacquire_candidate(
        self,
        *,
        feature: Any,
        partial_feature: Optional[Any],
        preferred_uid: Optional[int],
        candidate_count: int,
        sample_metadata: Optional[dict],
        bbox_quality_ok: bool,
        bbox_quality_tier: Optional[str],
    ) -> Optional[Tuple[int, float, Optional[float], int, int, str]]:
        """Return a bounded provisional preferred-UID match during search.

        This is deliberately narrower than normal reacquisition: it requires
        a fresh strong box, a useful detector confidence/size, and the locked
        UID must remain no worse than the configured identity margin.  The
        returned source is marked ``soft_*`` so the caller must send it through
        the local consecutive-observation confirmation path.
        """
        cfg = self.config
        metadata = sample_metadata or {}
        uid = int(preferred_uid or 0)
        if not (
            bool(cfg.preferred_search_soft_candidate_enable)
            and bool(metadata.get("search_reacquire_context_active"))
            # A soft match is an observation aid, not permission to cross the
            # frozen search side.  Opposite-side candidates must first pass
            # the strict strong-ReID path above; otherwise a visually similar
            # bystander could enter the two-frame observer merely because its
            # distance is below the wider soft threshold.
            and metadata.get("search_direction_compatible") is not False
            and uid > 0
            and bool(bbox_quality_ok)
            and str(bbox_quality_tier or "").strip().lower() == "strong"
            and bool(metadata.get("is_fresh", True))
            and self._soft_candidate_competition_ok(candidate_count, metadata)
        ):
            return None
        entry = self.identities.get(uid)
        if entry is None or not entry.features or feature is None:
            return None

        full_distance = _finite_float(
            entry.weighted_distance(feature, cfg.weak_match_penalty)[0]
        )
        target_distance = full_distance
        source = "soft_strong"
        partial_observation = bool(metadata.get("partial_observation"))
        if (
            (target_distance is None or target_distance > float(cfg.preferred_search_soft_candidate_threshold))
            and partial_observation
            and partial_feature is not None
            and bool(cfg.partial_appearance_enable)
            and entry.partial_features
        ):
            partial_distance = _finite_float(entry.partial_distance(partial_feature))
            if partial_distance is not None and partial_distance <= float(cfg.partial_match_threshold):
                target_distance = partial_distance
                source = "soft_partial"
        if target_distance is None or target_distance > (
            float(cfg.partial_match_threshold)
            if source == "soft_partial"
            else float(cfg.preferred_search_soft_candidate_threshold)
        ):
            return None
        metadata["reacquire_distance_limit"] = float(
            cfg.partial_match_threshold
            if source == "soft_partial"
            else cfg.preferred_search_soft_candidate_threshold
        )

        distances = []
        for item in self.identities.values():
            if not item.features:
                continue
            item_distance = _finite_float(
                item.weighted_distance(feature, cfg.weak_match_penalty)[0]
            )
            if item_distance is not None:
                distances.append((int(item.uid), float(item_distance), int(item.last_seen_frame)))
        distances.sort(key=lambda item: item[1])
        if not distances:
            return None
        best_uid, best_distance, best_last_seen_frame = distances[0]
        if int(best_uid) != uid:
            # A soft candidate can tolerate appearance drift, but not a clear
            # match to another identity in the gallery.
            return None
        disadvantage = max(0.0, float(target_distance) - float(best_distance))
        if disadvantage > float(cfg.preferred_search_reacquire_max_disadvantage):
            return None
        competing = [distance for candidate_uid, distance, _ in distances if candidate_uid != uid]
        second_distance = competing[0] if competing else None
        return (
            uid,
            float(target_distance),
            second_distance,
            int(best_uid),
            int(best_last_seen_frame),
            source,
        )

    def _candidate_competition_ok(
        self, candidate_count: int, sample_metadata: Optional[dict]
    ) -> bool:
        """Allow low-confidence fragments while retaining close-candidate gating."""
        if int(candidate_count) <= 1:
            return True
        gap = _finite_float((sample_metadata or {}).get("candidate_score_gap"))
        if gap is None:
            return False
        return float(gap) >= max(
            0.0, float(self.config.preferred_search_reacquire_min_score_gap)
        )

    def _soft_candidate_competition_ok(
        self, candidate_count: int, sample_metadata: Optional[dict]
    ) -> bool:
        """Ignore tiny detector fragments, but keep comparable people gated."""
        if int(candidate_count) <= 1:
            return True
        metadata = sample_metadata or {}
        gap = _finite_float(metadata.get("candidate_score_gap"))
        area_ratio = _finite_float(
            metadata.get("detector_area_ratio", metadata.get("area_ratio"))
        )
        confidence = _finite_float(metadata.get("detector_confidence"))
        return bool(
            gap is not None
            and gap >= max(0.0, float(self.config.preferred_search_soft_min_score_gap))
            and area_ratio is not None
            and area_ratio >= max(0.0, float(self.config.preferred_search_soft_min_area_ratio))
            and confidence is not None
            and confidence >= max(0.0, float(self.config.preferred_search_soft_min_confidence))
        )

    def _reacquire_allowed(
        self,
        *,
        best_distance: float,
        second_distance: Optional[float],
        best_last_seen_frame: int,
        frame_index: int,
        candidate_count: int,
    ) -> Tuple[bool, str]:
        match_threshold = float(self.config.match_threshold)
        threshold = float(self.config.reacquire_threshold)
        margin = float(self.config.reacquire_margin)
        reason = "reacquired"
        if threshold <= match_threshold:
            return False, ""

        if bool(self.config.reacquire_single_candidate_only) and int(candidate_count) > 1:
            if not bool(self.config.reacquire_multi_candidate_enable):
                return False, ""
            threshold = min(threshold, float(self.config.reacquire_multi_candidate_threshold))
            margin = max(margin, float(self.config.reacquire_multi_candidate_margin))
            reason = "reacquired_multi"
            if threshold <= match_threshold:
                return False, ""

        max_frames = int(self.config.reacquire_max_frames)
        if max_frames <= 0 or int(frame_index) - int(best_last_seen_frame) > max_frames:
            return False, ""
        if best_distance > threshold:
            return False, ""
        if not self._margin_ok(best_distance, second_distance, margin):
            return False, ""
        return True, reason

    @staticmethod
    def _margin_ok(best_distance: float, second_distance: Optional[float], margin: float) -> bool:
        if second_distance is None:
            return True
        return (float(second_distance) - float(best_distance)) >= float(margin)

    def _remember_track_seen(self, track_id: int, uid: int, frame_index: int) -> None:
        if int(uid) <= 0:
            return
        self.track_last_seen_frame[int(track_id)] = int(frame_index)

    def _recent_uid_claim(self, uid: int, track_id: int, frame_index: int) -> Tuple[Optional[int], Optional[int]]:
        if not bool(self.config.exclusive_uid_claim_enable):
            return None, None
        max_gap = int(self.config.exclusive_uid_claim_frames)
        if max_gap <= 0:
            return None, None
        for other_track_id, other_uid in self.track_to_uid.items():
            if int(other_track_id) == int(track_id) or int(other_uid) != int(uid):
                continue
            last_seen = self.track_last_seen_frame.get(int(other_track_id))
            if last_seen is None:
                continue
            gap = int(frame_index) - int(last_seen)
            if gap <= max_gap:
                return int(other_track_id), int(gap)
        return None, None

    def _assign_new_or_pending(
        self,
        track_id: int,
        feature: Any,
        partial_feature: Optional[Any],
        frame_index: int,
        sample_metadata: Optional[dict] = None,
    ) -> Tuple[int, str]:
        confirm_frames = max(1, int(self.config.new_identity_confirm_frames))
        if confirm_frames <= 1 or not self.identities:
            uid = self._create_identity(feature, frame_index, sample_metadata, partial_feature)
            self.track_to_uid[track_id] = uid
            self.pending_new.pop(track_id, None)
            return uid, "created"

        pending = self.pending_new.get(track_id)
        if pending is None or int(frame_index) > int(pending.last_frame) + 1:
            self.pending_new[track_id] = PendingIdentity(
                feature=_normalize_feature(feature),
                partial_feature=(
                    None if partial_feature is None else _normalize_feature(partial_feature)
                ),
                first_frame=int(frame_index),
                last_frame=int(frame_index),
                streak=1,
            )
            return 0, "pending_new"

        pending.feature = _normalize_feature(feature)
        pending.partial_feature = (
            None if partial_feature is None else _normalize_feature(partial_feature)
        )
        pending.last_frame = int(frame_index)
        pending.streak += 1
        if int(pending.streak) < confirm_frames:
            return 0, "pending_new"

        uid = self._create_identity(feature, frame_index, sample_metadata, partial_feature)
        self.track_to_uid[track_id] = uid
        self.pending_new.pop(track_id, None)
        return uid, "created_confirmed"

    def _maybe_add_to_identity(
        self,
        uid: int,
        feature: Any,
        partial_feature: Optional[Any],
        frame_index: int,
        distance: Optional[float],
        sample_metadata: Optional[dict] = None,
    ) -> bool:
        # A search-only low-confidence observation can help confirm an
        # existing UID, but it must never become a new gallery template.
        if self._reacquire_quarantine.is_held(uid):
            return False
        if bool((sample_metadata or {}).get("preferred_search_low_confidence")):
            return False
        if distance is None or float(distance) > float(self.config.update_threshold):
            return False
        if not self._should_update(frame_index):
            return False
        entry = self.identities.get(int(uid))
        if entry is None:
            return False
        changed = entry.add(
            feature,
            frame_index,
            max(1, int(self.config.max_features)),
            self.config.diversity_min_distance,
            self.config.diversity_replace_margin,
            sample_metadata,
        )
        if partial_feature is not None and self.config.partial_appearance_enable:
            partial_distance = entry.partial_distance(partial_feature)
            if (
                not entry.partial_features
                or partial_distance <= float(self.config.partial_update_threshold)
            ):
                changed = entry.add_partial(
                    partial_feature,
                    frame_index,
                    max(1, int(self.config.partial_max_features)),
                    self.config.diversity_min_distance,
                    sample_metadata,
                ) or changed
        return changed

    def _touch_identity(self, uid: int, frame_index: int) -> None:
        entry = self.identities.get(int(uid))
        if entry is not None:
            entry.touch(frame_index)

    def _pending_streak(self, track_id: int) -> int:
        pending = self.pending_new.get(int(track_id))
        return 0 if pending is None else int(pending.streak)

    def _handoff_streak(self, track_id: int) -> int:
        pending = self.pending_handoffs.get(int(track_id))
        return 0 if pending is None else int(pending.streak)

    def _controlled_handoff_candidate(
        self,
        *,
        track_id: int,
        candidate_uid: Optional[int],
        distance: Optional[float],
        second_distance: Optional[float],
        frame_index: int,
        require_margin: bool = True,
        confirm_frames: Optional[int] = None,
        threshold: Optional[float] = None,
        sample_metadata: Optional[dict] = None,
        min_old_track_gap: Optional[int] = None,
    ) -> Tuple[int, int]:
        cfg = self.config
        uid = 0 if candidate_uid is None else int(candidate_uid)
        threshold_value = float(
            cfg.controlled_handoff_threshold if threshold is None else threshold
        )
        if (
            not bool(cfg.controlled_handoff_enable)
            or uid <= 0
            or distance is None
            or float(distance) > threshold_value
            or (
                bool(require_margin)
                and not self._margin_ok(float(distance), second_distance, float(cfg.match_margin))
            )
        ):
            self.pending_handoffs.pop(int(track_id), None)
            return 0, 0

        claim_track = None
        claim_last_seen = None
        for other_track_id, other_uid in self.track_to_uid.items():
            if int(other_track_id) == int(track_id) or int(other_uid) != uid:
                continue
            last_seen = self.track_last_seen_frame.get(int(other_track_id))
            if last_seen is not None and (claim_last_seen is None or int(last_seen) > int(claim_last_seen)):
                claim_track = int(other_track_id)
                claim_last_seen = int(last_seen)
        entry = self.identities.get(uid)
        if claim_last_seen is None and entry is not None:
            claim_last_seen = int(entry.last_seen_frame)
        claim_gap = None if claim_last_seen is None else int(frame_index) - int(claim_last_seen)
        min_gap = max(
            1,
            int(
                cfg.controlled_handoff_min_old_track_gap_frames
                if min_old_track_gap is None
                else min_old_track_gap
            ),
        )
        pending = self.pending_handoffs.get(int(track_id))
        current_center = None
        if sample_metadata is not None:
            try:
                current_center = float(sample_metadata.get("center_x_ratio"))
            except (TypeError, ValueError):
                current_center = None
        # DeepSORT may briefly drop a detection and create a new raw track for
        # the same person. Preserve the confirmation chain only when the new
        # observation is adjacent in time and remains spatially continuous;
        # this prevents a different person from inheriting a stale streak.
        continuity_gap = False
        if pending is None:
            for previous_track_id, previous_pending in list(self.pending_handoffs.items()):
                if int(previous_track_id) == int(track_id) or int(previous_pending.uid) != uid:
                    continue
                gap = int(frame_index) - int(previous_pending.last_frame)
                center_ok = (
                    current_center is not None
                    and previous_pending.center_ratio is not None
                    and abs(current_center - float(previous_pending.center_ratio)) <= 0.25
                )
                if gap == 1 and center_ok:
                    pending = PendingHandoff(
                        uid=uid,
                        last_frame=int(previous_pending.last_frame),
                        streak=int(previous_pending.streak),
                        center_ratio=previous_pending.center_ratio,
                    )
                    self.pending_handoffs.pop(int(previous_track_id), None)
                    continuity_gap = True
                    logger.info(
                        "ReID重捕获确认链跨轨迹延续: 帧=%d 原轨迹=%d 新轨迹=%d 身份编号=%d 间隔帧=%d 已确认次数=%d 中心变化=%.3f",
                        int(frame_index),
                        int(previous_track_id),
                        int(track_id),
                        uid,
                        int(gap),
                        int(previous_pending.streak),
                        abs(current_center - float(previous_pending.center_ratio)),
                    )
                    break
        frame_gap = None if pending is None else int(frame_index) - int(pending.last_frame)
        # A missing or weak frame breaks the search confirmation chain.  It is
        # unsafe to count frame N and N+2 as consecutive identity evidence:
        # the intervening observation may belong to another person.
        if (
            pending is None
            or int(pending.uid) != uid
            or (
                int(frame_gap) != 1
                and not (continuity_gap and int(frame_gap) == 2)
            )
        ):
            pending = PendingHandoff(
                uid=uid,
                last_frame=int(frame_index),
                streak=1,
                center_ratio=current_center,
            )
            self.pending_handoffs[int(track_id)] = pending
        else:
            pending.last_frame = int(frame_index)
            pending.streak += 1
            pending.center_ratio = current_center
            # When the pending state was carried over from another raw track,
            # the old entry was removed above. Reattach it to the current
            # track so the next frame continues the same confirmation chain.
            self.pending_handoffs[int(track_id)] = pending

        if confirm_frames is None:
            required_confirm_frames = max(2, int(cfg.controlled_handoff_confirm_frames))
        else:
            required_confirm_frames = max(1, int(confirm_frames))
        if int(pending.streak) < required_confirm_frames:
            return 0, int(pending.streak)

        # Keep the old track's ownership guard for the actual UID transfer,
        # but allow observations to accumulate before the guard is satisfied.
        # This avoids discarding the first valid frame during a DeepSORT
        # track-id handoff while preserving the two-frame confirmation rule.
        if claim_gap is None or int(claim_gap) < min_gap:
            return 0, int(pending.streak)

        for other_track_id, other_uid in list(self.track_to_uid.items()):
            if int(other_track_id) != int(track_id) and int(other_uid) == uid:
                self.track_to_uid.pop(int(other_track_id), None)
        self._bind_reacquired_identity(uid, track_id, frame_index, sample_metadata)
        self.pending_handoffs.pop(int(track_id), None)
        logger.info(
            "ReID身份受控交接: 帧=%d 原轨迹=%d 新轨迹=%d 身份编号=%d 间隔帧=%d 连续确认次数=%d 距离=%s",
            int(frame_index),
            -1 if claim_track is None else int(claim_track),
            int(track_id),
            uid,
            int(claim_gap),
            int(pending.streak),
            _fmt_float(distance),
        )
        return uid, int(pending.streak)

    def _quality_ok(
        self,
        confidence: float,
        area: float,
        *,
        min_confidence: Optional[float] = None,
    ) -> bool:
        confidence_floor = (
            float(self.config.min_confidence)
            if min_confidence is None
            else float(min_confidence)
        )
        return (
            float(confidence) >= confidence_floor
            and float(area) >= float(self.config.min_area)
        )

    def _should_update(self, frame_index: int) -> bool:
        interval = max(1, int(self.config.update_interval))
        return int(frame_index) % interval == 0

    def _should_update_weak(self, frame_index: int) -> bool:
        interval = max(1, int(self.config.weak_update_interval))
        return int(frame_index) % interval == 0


def _normalize_feature(feature: Any):
    np = _np()
    arr = np.asarray(feature, dtype="float32").reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-12:
        arr = arr / norm
    return arr


def _feature_size(feature: Any) -> int:
    """Return a flat feature length for safe gallery compatibility checks."""
    try:
        return int(_normalize_feature(feature).size)
    except (TypeError, ValueError, AttributeError):
        return 0


def _evidence_metadata(metadata: Optional[dict]) -> dict:
    """Keep provenance JSON-safe without serializing any feature vectors."""
    source = metadata or {}
    result = {}
    for key in (
        "track_id", "frame_index", "control_frame_id", "capture_frame_id",
        "capture_timestamp", "source_detection_index", "center_x_ratio", "area_ratio",
        "detector_center_x_ratio", "detector_area_ratio", "detector_confidence",
        "aspect_ratio", "edge_touch_count", "quality_weight", "image_width", "image_height",
        "partial_observation", "detector_edge_touch_count", "yaw_rate_dps", "integrated_yaw_deg",
        "candidate_count", "candidate_score_gap",
        "search_reacquire_context_active", "search_direction_compatible",
        "reacquire_distance_limit",
        "preferred_search_identity_competition_override",
        "preferred_search_geometry_relaxation",
        "preferred_search_low_confidence", "search_quality_override",
        "quality_bbox_source", "quality_bbox_ok", "display_bbox_quality_ok",
    ):
        if key in source:
            if isinstance(source[key], bool):
                result[key] = source[key]
                continue
            value = _finite_float(source[key])
            if value is not None:
                result[key] = int(value) if key.endswith("_id") or key.endswith("_index") else value
    for key in ("bbox", "detector_bbox", "quality_bbox"):
        try:
            values = list(source[key])
            bbox = [_finite_float(value) for value in values]
        except (KeyError, TypeError, ValueError):
            continue
        if len(bbox) == 4 and all(value is not None for value in bbox):
            result[key] = bbox
    for key in (
        "quality_tier", "bbox_quality_tier", "bbox_quality_reason",
        "quality_bbox_reason", "display_bbox_quality_reason", "motion_direction",
        "partial_feature_source",
    ):
        if isinstance(source.get(key), str):
            result[key] = source[key]
    if isinstance(source.get("is_fresh"), bool):
        result["is_fresh"] = source["is_fresh"]
    if isinstance(source.get("search_reacquire_context_active"), bool):
        result["search_reacquire_context_active"] = source[
            "search_reacquire_context_active"
        ]
    if isinstance(source.get("search_direction_compatible"), bool):
        result["search_direction_compatible"] = source[
            "search_direction_compatible"
        ]
    if isinstance(source.get("partial_observation"), bool):
        result["partial_observation"] = source["partial_observation"]
    return result


def _geometry_observation(metadata: Optional[dict], frame_index: int) -> Optional[dict]:
    source = _evidence_metadata(metadata)
    if source.get("is_fresh") is False:
        return None
    detector = "detector_bbox" in source
    bbox = source.get("detector_bbox" if detector else "bbox")
    center = source.get("detector_center_x_ratio" if detector else "center_x_ratio")
    area = source.get("detector_area_ratio" if detector else "area_ratio")
    if bbox is None or center is None or not 0.0 <= float(center) <= 1.0:
        return None
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    area_units = "ratio" if area is not None else "pixels"
    area = _finite_float((x2 - x1) * (y2 - y1) if area is None else area)
    if area is None or area <= 0.0:
        return None
    return {
        "frame_index": int(frame_index),
        "track_id": source.get("track_id"),
        "capture_frame_id": source.get("capture_frame_id"),
        "bbox": list(bbox),
        "center_x_ratio": float(center),
        "area": float(area),
        "area_units": area_units,
        "geometry_source": "detector" if detector else "track",
        "capture_timestamp": source.get("capture_timestamp"),
        "edge_touch_count": source.get(
            "detector_edge_touch_count", source.get("edge_touch_count")
        ),
        "integrated_yaw_deg": source.get("integrated_yaw_deg"),
        "yaw_rate_dps": source.get("yaw_rate_dps"),
    }


def _local_reacquire_observation(
    metadata: Optional[dict], frame_index: int
) -> Optional[dict]:
    """Extract geometry for weak/partial confirmation, including sparse probes."""
    geometry = _geometry_observation(metadata, frame_index)
    if geometry is not None:
        return geometry
    source = _evidence_metadata(metadata)
    center = _finite_float(source.get("center_x_ratio"))
    if center is None:
        center = _finite_float(source.get("detector_center_x_ratio"))
    area = _finite_float(source.get("area"))
    area_units = "pixels"
    if area is None:
        area = _finite_float(source.get("area_ratio"))
        area_units = "ratio"
    if area is None or area <= 0.0 or center is None or not 0.0 <= center <= 1.0:
        return None
    return {
        "frame_index": int(frame_index),
        "track_id": source.get("track_id"),
        "capture_frame_id": source.get("capture_frame_id"),
        "bbox": source.get("bbox"),
        "center_x_ratio": float(center),
        "area": float(area),
        "area_units": area_units,
        "geometry_source": "metadata",
        "capture_timestamp": source.get("capture_timestamp"),
        "edge_touch_count": source.get("edge_touch_count"),
        "integrated_yaw_deg": source.get("integrated_yaw_deg"),
        "yaw_rate_dps": source.get("yaw_rate_dps"),
    }


def _pending_observation_continuous(
    pending: PendingHandoff,
    current: Optional[dict],
    frame_index: int,
    config: IdentityBankConfig,
) -> bool:
    """Require adjacent, spatially and (when available) size-continuous samples."""
    if int(frame_index) - int(pending.last_frame) != 1:
        return False
    # Legacy replay/unit-test callers may not provide geometry at all. In that
    # case the existing ReID/candidate gates still apply; enforce continuity
    # whenever a position or size observation is actually available.
    if current is None:
        return True
    current_center = _finite_float(current.get("center_x_ratio"))
    previous_center = _finite_float(pending.center_ratio)
    if current_center is None or previous_center is None:
        return False
    center_jump, compensated_center_jump = _yaw_compensated_center_jump(
        current_center=current_center,
        previous_center=previous_center,
        current_yaw=_finite_float(current.get("integrated_yaw_deg")),
        previous_yaw=_finite_float(pending.integrated_yaw_deg),
        camera_hfov_deg=config.camera_hfov_deg,
    )
    if min(center_jump, compensated_center_jump) > float(
        config.handoff_geometry_max_center_jump_ratio
    ):
        return False
    current_area = _finite_float(current.get("area"))
    previous_area = _finite_float(pending.area)
    if current_area is not None and previous_area is not None and previous_area > 0.0:
        if pending.area_units is not None and current.get("area_units") not in (None, pending.area_units):
            return False
        area_similarity = min(current_area, previous_area) / max(current_area, previous_area)
        if area_similarity < float(config.handoff_geometry_min_area_similarity):
            return False
    previous_timestamp = _finite_float(pending.capture_timestamp)
    current_timestamp = _finite_float(current.get("capture_timestamp"))
    if previous_timestamp is not None and current_timestamp is not None:
        delta = float(current_timestamp - previous_timestamp)
        if delta <= 0.0 or delta > max(0.0, float(config.preferred_search_reacquire_max_age_sec)):
            return False
    return True


def _yaw_compensated_center_jump(
    *,
    current_center: float,
    previous_center: float,
    current_yaw: Optional[float],
    previous_yaw: Optional[float],
    camera_hfov_deg: float,
) -> Tuple[float, float]:
    """Return raw and yaw-compensated horizontal displacement.

    The sign of integrated yaw depends on camera mounting.  Taking the
    smaller residual supports either mounting convention while still keeping
    the configured center-jump bound as the final safety gate.
    """
    raw_jump = abs(float(current_center) - float(previous_center))
    if current_yaw is None or previous_yaw is None:
        return raw_jump, raw_jump
    yaw_ratio = (float(current_yaw) - float(previous_yaw)) / max(
        1.0, float(camera_hfov_deg)
    )
    delta = float(current_center) - float(previous_center)
    compensated = min(abs(delta - yaw_ratio), abs(delta + yaw_ratio))
    return raw_jump, compensated


def _sample_metadata(
    metadata: Optional[dict],
    frame_index: int,
    quality_weight: float,
    quality_tier: str,
) -> dict:
    result = dict(metadata or {})
    result["frame_index"] = int(frame_index)
    result["quality_weight"] = max(0.0, min(1.0, float(quality_weight)))
    result["quality_tier"] = str(quality_tier)
    return result


def _replace_metadata(items: List[dict], index: int, metadata: dict) -> None:
    while len(items) <= int(index):
        items.append({})
    items[int(index)] = metadata


def _finite_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def _fmt_float(value: Optional[float]) -> str:
    value_f = _finite_float(value)
    if value_f is None:
        return "none"
    return f"{value_f:.3f}"


def _np():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("numpy is required for ReID identity banking") from exc
    return np
