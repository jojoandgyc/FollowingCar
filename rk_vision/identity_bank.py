from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    weak_reacquire_confirm_frames: int = 3
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
    controlled_handoff_confirm_frames: int = 3
    controlled_handoff_threshold: float = 0.30
    controlled_handoff_min_old_track_gap_frames: int = 2
    preferred_search_reacquire_enable: bool = True
    preferred_search_reacquire_threshold: float = 0.36
    preferred_search_reacquire_max_disadvantage: float = 0.15
    preferred_search_reacquire_confirm_frames: int = 2
    preferred_search_reacquire_instant_threshold: float = 0.28


@dataclass
class IdentityEntry:
    uid: int
    features: List[Any] = field(default_factory=list)
    weak_features: List[Any] = field(default_factory=list)
    feature_metadata: List[dict] = field(default_factory=list)
    weak_feature_metadata: List[dict] = field(default_factory=list)
    last_frame: int = 0
    last_weak_frame: int = 0
    last_seen_frame: int = 0
    update_count: int = 0
    weak_update_count: int = 0
    duplicate_skip_count: int = 0
    weak_duplicate_skip_count: int = 0
    diversity_replace_count: int = 0
    weak_diversity_replace_count: int = 0

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


@dataclass
class PendingIdentity:
    feature: Any
    first_frame: int
    last_frame: int
    streak: int = 1


@dataclass
class PendingHandoff:
    uid: int
    last_frame: int
    streak: int = 1


class IdentityBank:
    """Long-lived ReID identity memory outside DeepSORT's active-track gallery."""

    def __init__(self, config: Optional[IdentityBankConfig] = None) -> None:
        self.config = config or IdentityBankConfig()
        self.identities: Dict[int, IdentityEntry] = {}
        self.track_to_uid: Dict[int, int] = {}
        self.track_last_seen_frame: Dict[int, int] = {}
        self.pending_new: Dict[int, PendingIdentity] = {}
        self.pending_handoffs: Dict[int, PendingHandoff] = {}
        self.pending_weak_handoffs: Dict[int, PendingHandoff] = {}
        self.last_assignments: Dict[int, dict] = {}
        self._next_uid = 1

    def reset(self) -> None:
        self.identities.clear()
        self.track_to_uid.clear()
        self.track_last_seen_frame.clear()
        self.pending_new.clear()
        self.pending_handoffs.clear()
        self.pending_weak_handoffs.clear()
        self.last_assignments.clear()
        self._next_uid = 1

    def assign(
        self,
        *,
        track_id: int,
        feature: Optional[Any],
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
        track_id = int(track_id)
        if not self.config.enabled:
            self.track_to_uid[track_id] = track_id
            self.pending_new.pop(track_id, None)
            self.last_assignments[track_id] = {"uid": track_id, "reason": "disabled"}
            return track_id

        uid = self.track_to_uid.get(track_id, 0)
        base_quality_ok = self._quality_ok(confidence, area)
        quality_tier = str(bbox_quality_tier or ("strong" if bbox_quality_ok else "reject")).strip().lower()
        if quality_tier not in ("strong", "weak", "reject"):
            quality_tier = "reject"
        quality_ok = base_quality_ok and quality_tier == "strong"
        weak_quality_ok = base_quality_ok and quality_tier == "weak"
        quality_reason = str(bbox_quality_reason or "").strip()
        if quality_tier != "weak":
            self.pending_weak_handoffs.pop(track_id, None)
        reason = "mapped" if uid > 0 else "unassigned"
        distance = None
        second_distance = None
        best_uid = None
        best_last_seen_frame = None
        best_frame_gap = None
        handoff_streak = 0
        match_source = None

        if uid <= 0 and feature is not None and weak_quality_ok:
            return self._assign_weak_search_candidate(
                track_id=track_id,
                feature=feature,
                frame_index=frame_index,
                candidate_count=candidate_count,
                quality_reason=quality_reason,
                sample_metadata=sample_metadata,
                preferred_uid=preferred_uid,
                preferred_candidate_ok=preferred_candidate_ok,
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
                preferred_uid=preferred_uid,
                candidate_ok=preferred_candidate_ok,
            )
            preferred_handoff = preferred_match is not None
            preferred_rejected = False
            if preferred_match is not None:
                (
                    match_uid,
                    distance,
                    second_distance,
                    best_uid,
                    best_last_seen_frame,
                ) = preferred_match
                match_reason = "preferred_search_reacquire"
            else:
                match_uid, distance, second_distance, match_reason, best_uid, best_last_seen_frame = self._match(
                    track_id=track_id,
                    feature=feature,
                    frame_index=frame_index,
                    candidate_count=candidate_count,
                )
                if (
                    bool(self.config.preferred_search_reacquire_enable)
                    and bool(preferred_candidate_ok)
                    and preferred_uid is not None
                    and int(preferred_uid) > 0
                    and (
                        int(match_uid) == int(preferred_uid)
                        or (
                            match_reason == "uid_claimed_recently"
                            and best_uid is not None
                            and int(best_uid) == int(preferred_uid)
                        )
                    )
                ):
                    match_uid = 0
                    match_reason = "preferred_search_reacquire_rejected"
                    preferred_rejected = True
            if best_last_seen_frame is not None:
                best_frame_gap = int(frame_index) - int(best_last_seen_frame)
            source_uid = int(match_uid) if int(match_uid) > 0 else best_uid
            if source_uid is not None and int(source_uid) > 0:
                match_source = self._match_source(int(source_uid), feature)
            handoff_uid = 0
            if bool(self.config.controlled_handoff_enable):
                if match_uid > 0:
                    handoff_uid = int(match_uid)
                elif match_reason == "uid_claimed_recently" and best_uid is not None:
                    handoff_uid = int(best_uid)

            if handoff_uid > 0:
                preferred_confirm_frames = None
                if preferred_handoff:
                    preferred_confirm_frames = int(
                        self.config.preferred_search_reacquire_confirm_frames
                    )
                    instant_threshold = min(
                        float(self.config.preferred_search_reacquire_threshold),
                        max(
                            0.0,
                            float(self.config.preferred_search_reacquire_instant_threshold),
                        ),
                    )
                    if (
                        match_source != "weak"
                        and int(candidate_count) == 1
                        and distance is not None
                        and float(distance) <= instant_threshold
                    ):
                        preferred_confirm_frames = 1
                if match_source == "weak":
                    preferred_confirm_frames = max(
                        int(self.config.weak_reacquire_confirm_frames),
                        1 if preferred_confirm_frames is None else int(preferred_confirm_frames),
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
                        else None
                    ),
                )
                bank_updated = False
                if uid > 0:
                    reason = "preferred_search_reacquire" if preferred_handoff else "controlled_handoff"
                    self.pending_new.pop(track_id, None)
                    self._touch_identity(uid, frame_index)
                    bank_updated = self._maybe_add_to_identity(
                        uid,
                        feature,
                        frame_index,
                        distance,
                        sample_metadata=sample_metadata,
                    )
                else:
                    reason = "preferred_search_reacquire_wait" if preferred_handoff else "controlled_handoff_wait"
            elif match_uid > 0:
                if match_source == "weak":
                    uid = 0
                    reason = "weak_match_requires_handoff"
                    bank_updated = False
                    self.pending_new.pop(track_id, None)
                else:
                    uid = match_uid
                    self.track_to_uid[track_id] = uid
                    self.pending_new.pop(track_id, None)
                    reason = match_reason
                    self._touch_identity(uid, frame_index)
                    bank_updated = self._maybe_add_to_identity(
                        uid,
                        feature,
                        frame_index,
                        distance,
                        sample_metadata=sample_metadata,
                    )
            elif preferred_rejected:
                self.pending_handoffs.pop(track_id, None)
                self.pending_new.pop(track_id, None)
                uid = 0
                reason = "preferred_search_reacquire_rejected"
                bank_updated = False
            else:
                self.pending_handoffs.pop(track_id, None)
                uid, reason = self._assign_new_or_pending(
                    track_id,
                    feature,
                    frame_index,
                    sample_metadata,
                )
                bank_updated = False
            if uid > 0:
                self._remember_track_seen(track_id, uid, frame_index)
            self.last_assignments[track_id] = {
                "uid": int(uid),
                "reason": reason,
                "distance": _finite_float(distance),
                "second_distance": _finite_float(second_distance),
                "best_uid": None if best_uid is None else int(best_uid),
                "best_frame_gap": None if best_frame_gap is None else int(best_frame_gap),
                "pending_streak": self._pending_streak(track_id),
                "handoff_streak": (
                    int(handoff_streak)
                    if reason in ("controlled_handoff", "preferred_search_reacquire")
                    else self._handoff_streak(track_id)
                ),
                "bank_updated": bool(bank_updated),
                "match_source": match_source,
                "bbox_quality_ok": bool(bbox_quality_ok),
                "bbox_quality_tier": quality_tier,
                "bbox_quality_reason": quality_reason or None,
            }
            logger.info(
                "ReID身份库分配: 帧=%d 轨迹ID=%d 身份编号=%d 原因代码=%s 最佳身份=%s 距离=%s 次佳距离=%s "
                "匹配来源=%s 间隔帧=%s 候选数=%d 待确认次数=%d 交接确认次数=%d 更新身份库=%s 质量层级=%s%s",
                int(frame_index),
                int(track_id),
                int(uid),
                reason,
                "none" if best_uid is None else str(int(best_uid)),
                _fmt_float(distance),
                _fmt_float(second_distance),
                match_source or "none",
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

                self._touch_identity(uid, frame_index)
                if self._should_update(frame_index):
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
                    "last_frame": int(entry.last_frame),
                    "last_weak_frame": int(entry.last_weak_frame),
                    "last_seen_frame": int(entry.last_seen_frame),
                    "updates": int(entry.update_count),
                    "weak_updates": int(entry.weak_update_count),
                    "duplicate_skips": int(entry.duplicate_skip_count),
                    "weak_duplicate_skips": int(entry.weak_duplicate_skip_count),
                    "diversity_replacements": int(entry.diversity_replace_count),
                    "weak_diversity_replacements": int(entry.weak_diversity_replace_count),
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
                }
                for track_id, pending in sorted(self.pending_handoffs.items())
            },
            "pending_weak_handoffs": {
                str(track_id): {
                    "uid": int(pending.uid),
                    "last_frame": int(pending.last_frame),
                    "streak": int(pending.streak),
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

    def _create_identity(
        self,
        feature: Any,
        frame_index: int,
        sample_metadata: Optional[dict] = None,
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

    def _assign_weak_search_candidate(
        self,
        *,
        track_id: int,
        feature: Any,
        frame_index: int,
        candidate_count: int,
        quality_reason: str,
        sample_metadata: Optional[dict],
        preferred_uid: Optional[int],
        preferred_candidate_ok: bool,
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
            and int(candidate_count) == 1
            and uid > 0
            and target_entry is not None
            and bool(target_entry.features)
        )
        if candidate_allowed and target_entry is not None:
            distance, source = target_entry.weighted_distance(
                feature,
                self.config.weak_match_penalty,
            )
            strong_distance = target_entry.distance(feature)
            weak_distance = target_entry.weak_distance(feature)
            competing = sorted(
                float(entry.weighted_distance(feature, self.config.weak_match_penalty)[0])
                for other_uid, entry in self.identities.items()
                if int(other_uid) != uid and entry.features
            )
            second_distance = competing[0] if competing else None
            disadvantage = (
                0.0
                if second_distance is None
                else max(0.0, float(distance) - float(second_distance))
            )
            if (
                float(distance) <= float(self.config.weak_reacquire_threshold)
                and disadvantage <= float(self.config.preferred_search_reacquire_max_disadvantage)
            ):
                confirmed, streak = self._weak_handoff_candidate(
                    track_id=track_id,
                    candidate_uid=uid,
                    frame_index=frame_index,
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

    def _weak_handoff_candidate(
        self,
        *,
        track_id: int,
        candidate_uid: int,
        frame_index: int,
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

        pending = self.pending_weak_handoffs.get(int(track_id))
        if pending is None or int(pending.uid) != uid or int(frame_index) != int(pending.last_frame) + 1:
            pending = PendingHandoff(uid=uid, last_frame=int(frame_index), streak=1)
            self.pending_weak_handoffs[int(track_id)] = pending
        else:
            pending.last_frame = int(frame_index)
            pending.streak += 1
        required = max(2, int(self.config.weak_reacquire_confirm_frames))
        if int(pending.streak) < required:
            return False, int(pending.streak)

        for other_track_id, other_uid in list(self.track_to_uid.items()):
            if int(other_track_id) != int(track_id) and int(other_uid) == uid:
                self.track_to_uid.pop(int(other_track_id), None)
        self.track_to_uid[int(track_id)] = uid
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
        preferred_uid: Optional[int],
        candidate_ok: bool,
    ) -> Optional[Tuple[int, float, Optional[float], int, int]]:
        """Select the locked UID during search without widening global ReID matching."""
        cfg = self.config
        uid = 0 if preferred_uid is None else int(preferred_uid)
        if not bool(cfg.preferred_search_reacquire_enable) or not bool(candidate_ok) or uid <= 0:
            return None
        entry = self.identities.get(uid)
        if entry is None or not entry.features:
            return None

        distances = [
            (
                int(item.uid),
                float(item.weighted_distance(feature, cfg.weak_match_penalty)[0]),
                int(item.last_seen_frame),
            )
            for item in self.identities.values()
            if item.features
        ]
        distances.sort(key=lambda item: item[1])
        if not distances:
            return None

        target_distance = float(entry.weighted_distance(feature, cfg.weak_match_penalty)[0])
        best_uid, best_distance, best_last_seen_frame = distances[0]
        disadvantage = max(0.0, target_distance - float(best_distance))
        if target_distance > float(cfg.preferred_search_reacquire_threshold):
            return None
        if disadvantage > float(cfg.preferred_search_reacquire_max_disadvantage):
            return None

        competing = [distance for candidate_uid, distance, _ in distances if candidate_uid != uid]
        second_distance = competing[0] if competing else None
        return uid, target_distance, second_distance, int(best_uid), int(best_last_seen_frame)

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
        frame_index: int,
        sample_metadata: Optional[dict] = None,
    ) -> Tuple[int, str]:
        confirm_frames = max(1, int(self.config.new_identity_confirm_frames))
        if confirm_frames <= 1 or not self.identities:
            uid = self._create_identity(feature, frame_index, sample_metadata)
            self.track_to_uid[track_id] = uid
            self.pending_new.pop(track_id, None)
            return uid, "created"

        pending = self.pending_new.get(track_id)
        if pending is None or int(frame_index) > int(pending.last_frame) + 1:
            self.pending_new[track_id] = PendingIdentity(
                feature=_normalize_feature(feature),
                first_frame=int(frame_index),
                last_frame=int(frame_index),
                streak=1,
            )
            return 0, "pending_new"

        pending.feature = _normalize_feature(feature)
        pending.last_frame = int(frame_index)
        pending.streak += 1
        if int(pending.streak) < confirm_frames:
            return 0, "pending_new"

        uid = self._create_identity(feature, frame_index, sample_metadata)
        self.track_to_uid[track_id] = uid
        self.pending_new.pop(track_id, None)
        return uid, "created_confirmed"

    def _maybe_add_to_identity(
        self,
        uid: int,
        feature: Any,
        frame_index: int,
        distance: Optional[float],
        sample_metadata: Optional[dict] = None,
    ) -> bool:
        if distance is None or float(distance) > float(self.config.update_threshold):
            return False
        if not self._should_update(frame_index):
            return False
        entry = self.identities.get(int(uid))
        if entry is None:
            return False
        return entry.add(
            feature,
            frame_index,
            max(1, int(self.config.max_features)),
            self.config.diversity_min_distance,
            self.config.diversity_replace_margin,
            sample_metadata,
        )

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
        min_gap = max(1, int(cfg.controlled_handoff_min_old_track_gap_frames))
        if claim_gap is None or int(claim_gap) < min_gap:
            self.pending_handoffs.pop(int(track_id), None)
            return 0, 0

        pending = self.pending_handoffs.get(int(track_id))
        if pending is None or int(pending.uid) != uid or int(frame_index) != int(pending.last_frame) + 1:
            pending = PendingHandoff(uid=uid, last_frame=int(frame_index), streak=1)
            self.pending_handoffs[int(track_id)] = pending
        else:
            pending.last_frame = int(frame_index)
            pending.streak += 1

        required_confirm_frames = max(
            1,
            int(cfg.controlled_handoff_confirm_frames if confirm_frames is None else confirm_frames),
        )
        if int(pending.streak) < required_confirm_frames:
            return 0, int(pending.streak)

        for other_track_id, other_uid in list(self.track_to_uid.items()):
            if int(other_track_id) != int(track_id) and int(other_uid) == uid:
                self.track_to_uid.pop(int(other_track_id), None)
        self.track_to_uid[int(track_id)] = uid
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

    def _quality_ok(self, confidence: float, area: float) -> bool:
        return (
            float(confidence) >= float(self.config.min_confidence)
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
    value_f = float(value)
    if value_f == float("inf") or value_f == float("-inf"):
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
