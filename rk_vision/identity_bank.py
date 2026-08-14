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


@dataclass
class IdentityEntry:
    uid: int
    features: List[Any] = field(default_factory=list)
    last_frame: int = 0
    last_seen_frame: int = 0
    update_count: int = 0

    def add(self, feature: Any, frame_index: int, max_features: int) -> None:
        self.features.append(_normalize_feature(feature))
        if max_features > 0:
            self.features = self.features[-int(max_features) :]
        self.last_frame = int(frame_index)
        self.last_seen_frame = int(frame_index)
        self.update_count += 1

    def touch(self, frame_index: int) -> None:
        self.last_seen_frame = max(int(self.last_seen_frame), int(frame_index))

    def distance(self, feature: Any) -> float:
        if not self.features:
            return float("inf")
        np = _np()
        query = _normalize_feature(feature).reshape(1, -1)
        samples = np.asarray(self.features, dtype="float32")
        return float((1.0 - samples.dot(query.T)).min())


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
        self.last_assignments: Dict[int, dict] = {}
        self._next_uid = 1

    def reset(self) -> None:
        self.identities.clear()
        self.track_to_uid.clear()
        self.track_last_seen_frame.clear()
        self.pending_new.clear()
        self.pending_handoffs.clear()
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
    ) -> int:
        track_id = int(track_id)
        if not self.config.enabled:
            self.track_to_uid[track_id] = track_id
            self.pending_new.pop(track_id, None)
            self.last_assignments[track_id] = {"uid": track_id, "reason": "disabled"}
            return track_id

        uid = self.track_to_uid.get(track_id, 0)
        base_quality_ok = self._quality_ok(confidence, area)
        quality_ok = base_quality_ok and bool(bbox_quality_ok)
        quality_reason = str(bbox_quality_reason or "").strip()
        reason = "mapped" if uid > 0 else "unassigned"
        distance = None
        second_distance = None
        best_uid = None
        best_last_seen_frame = None
        best_frame_gap = None
        handoff_streak = 0

        if uid <= 0 and feature is not None and quality_ok:
            match_uid, best_distance, second_distance, match_reason, best_uid, best_last_seen_frame = self._match(
                track_id=track_id,
                feature=feature,
                frame_index=frame_index,
                candidate_count=candidate_count,
            )
            distance = best_distance
            if best_last_seen_frame is not None:
                best_frame_gap = int(frame_index) - int(best_last_seen_frame)
            handoff_uid = 0
            if bool(self.config.controlled_handoff_enable):
                if match_uid > 0:
                    handoff_uid = int(match_uid)
                elif match_reason == "uid_claimed_recently" and best_uid is not None:
                    handoff_uid = int(best_uid)

            if handoff_uid > 0:
                uid, handoff_streak = self._controlled_handoff_candidate(
                    track_id=track_id,
                    candidate_uid=handoff_uid,
                    distance=distance,
                    second_distance=second_distance,
                    frame_index=frame_index,
                )
                bank_updated = False
                if uid > 0:
                    reason = "controlled_handoff"
                    self.pending_new.pop(track_id, None)
                    self._touch_identity(uid, frame_index)
                    self._maybe_add_to_identity(uid, feature, frame_index, distance)
                else:
                    reason = "controlled_handoff_wait"
            elif match_uid > 0:
                uid = match_uid
                self.track_to_uid[track_id] = uid
                self.pending_new.pop(track_id, None)
                reason = match_reason
                self._touch_identity(uid, frame_index)
                bank_updated = self._maybe_add_to_identity(uid, feature, frame_index, distance)
            else:
                self.pending_handoffs.pop(track_id, None)
                uid, reason = self._assign_new_or_pending(track_id, feature, frame_index)
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
                "handoff_streak": int(handoff_streak) if reason == "controlled_handoff" else self._handoff_streak(track_id),
                "bank_updated": bool(bank_updated),
                "bbox_quality_ok": bool(bbox_quality_ok),
                "bbox_quality_reason": quality_reason or None,
            }
            logger.info(
                "identity_bank assign frame=%d track_id=%d uid=%d reason=%s best_uid=%s dist=%s second=%s gap=%s candidates=%d pending=%d handoff=%d update=%s quality=%s%s",
                int(frame_index),
                int(track_id),
                int(uid),
                reason,
                "none" if best_uid is None else str(int(best_uid)),
                _fmt_float(distance),
                _fmt_float(second_distance),
                "none" if best_frame_gap is None else str(int(best_frame_gap)),
                int(candidate_count),
                self._pending_streak(track_id),
                self._handoff_streak(track_id),
                bool(bank_updated),
                bool(bbox_quality_ok),
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
                    "identity_bank mapped frame=%d track_id=%d uid=%d output_uid=%d reason=%s dist=%s quality=%s%s",
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
                        "identity_bank mapped frame=%d track_id=%d uid=%d output_uid=0 reason=%s dist=none quality=%s%s",
                        int(frame_index),
                        int(track_id),
                        int(uid),
                        reason,
                        bool(bbox_quality_ok),
                        "" if not quality_reason else f"({quality_reason})",
                    )
                    return 0

                mapped_distance = entry.distance(feature)
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
                        "bbox_quality_ok": bool(bbox_quality_ok),
                        "bbox_quality_reason": quality_reason or None,
                    }
                    logger.info(
                        "identity_bank mapped frame=%d track_id=%d uid=%d output_uid=0 reason=%s dist=%s threshold=%.3f quality=%s%s",
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
                        entry.add(feature, frame_index, max(1, int(self.config.max_features)))
                        reason = "updated"
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
            "bbox_quality_ok": bool(bbox_quality_ok),
            "bbox_quality_reason": quality_reason or None,
        }
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
                    "last_frame": int(entry.last_frame),
                    "last_seen_frame": int(entry.last_seen_frame),
                    "updates": int(entry.update_count),
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
            "last_assignments": {
                str(track_id): assignment for track_id, assignment in sorted(self.last_assignments.items())
            },
        }

    def distance_to_uid(self, uid: int, feature: Any) -> Optional[float]:
        entry = self.identities.get(int(uid))
        if entry is None or not entry.features:
            return None
        return float(entry.distance(feature))

    def _create_identity(self, feature: Any, frame_index: int) -> int:
        uid = int(self._next_uid)
        self._next_uid += 1
        entry = IdentityEntry(uid)
        entry.add(feature, frame_index, max(1, int(self.config.max_features)))
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
            (entry.uid, entry.distance(feature), entry.last_seen_frame)
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
                    "identity_bank claim_reject frame=%d track_id=%d uid=%d claimed_by=%d claim_gap=%d dist=%s reason=matched",
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
                    "identity_bank claim_reject frame=%d track_id=%d uid=%d claimed_by=%d claim_gap=%d dist=%s reason=%s",
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

    def _assign_new_or_pending(self, track_id: int, feature: Any, frame_index: int) -> Tuple[int, str]:
        confirm_frames = max(1, int(self.config.new_identity_confirm_frames))
        if confirm_frames <= 1 or not self.identities:
            uid = self._create_identity(feature, frame_index)
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

        uid = self._create_identity(feature, frame_index)
        self.track_to_uid[track_id] = uid
        self.pending_new.pop(track_id, None)
        return uid, "created_confirmed"

    def _maybe_add_to_identity(
        self,
        uid: int,
        feature: Any,
        frame_index: int,
        distance: Optional[float],
    ) -> bool:
        if distance is None or float(distance) > float(self.config.update_threshold):
            return False
        if not self._should_update(frame_index):
            return False
        entry = self.identities.get(int(uid))
        if entry is None:
            return False
        entry.add(feature, frame_index, max(1, int(self.config.max_features)))
        return True

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
    ) -> Tuple[int, int]:
        cfg = self.config
        uid = 0 if candidate_uid is None else int(candidate_uid)
        threshold = float(cfg.controlled_handoff_threshold)
        if (
            not bool(cfg.controlled_handoff_enable)
            or uid <= 0
            or distance is None
            or float(distance) > threshold
            or not self._margin_ok(float(distance), second_distance, float(cfg.match_margin))
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

        confirm_frames = max(1, int(cfg.controlled_handoff_confirm_frames))
        if int(pending.streak) < confirm_frames:
            return 0, int(pending.streak)

        for other_track_id, other_uid in list(self.track_to_uid.items()):
            if int(other_track_id) != int(track_id) and int(other_uid) == uid:
                self.track_to_uid.pop(int(other_track_id), None)
        self.track_to_uid[int(track_id)] = uid
        self.pending_handoffs.pop(int(track_id), None)
        logger.info(
            "identity_bank controlled_handoff frame=%d old_track=%d new_track=%d uid=%d gap=%d streak=%d distance=%s",
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


def _normalize_feature(feature: Any):
    np = _np()
    arr = np.asarray(feature, dtype="float32").reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-12:
        arr = arr / norm
    return arr


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
