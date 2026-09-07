from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Set, Tuple

from .deepsort import DeepSort, DeepSortConfig
from .deepsort.track import TrackState
from .identity_bank import IdentityBank, IdentityBankConfig
from .yolo11 import Detection


TRACK_STATE_NEW = 0
TRACK_STATE_UNSTABLE = 1
TRACK_STATE_STABLE = 2


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
    identity_max_weak_features: int = 8
    identity_diversity_min_distance: float = 0.02
    identity_diversity_replace_margin: float = 0.01
    identity_weak_update_threshold: float = 0.42
    identity_weak_update_interval: int = 3
    identity_weak_match_penalty: float = 0.10
    identity_weak_reacquire_threshold: float = 0.38
    identity_weak_reacquire_confirm_frames: int = 3
    identity_weak_quality_weight: float = 0.30
    identity_min_confidence: float = 0.65
    identity_min_area: float = 0.0
    identity_min_width_px: float = 0.0
    identity_min_height_px: float = 0.0
    identity_max_single_frame_area_shrink_ratio: float = 0.30
    identity_area_shrink_max_gap_frames: int = 2
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
    identity_controlled_handoff_confirm_frames: int = 3
    identity_controlled_handoff_threshold: float = 0.30
    identity_controlled_handoff_min_old_track_gap_frames: int = 2
    identity_preferred_search_reacquire_enable: bool = True
    identity_preferred_search_reacquire_threshold: float = 0.36
    identity_preferred_search_reacquire_max_disadvantage: float = 0.15
    identity_preferred_search_reacquire_confirm_frames: int = 2
    identity_preferred_search_reacquire_instant_threshold: float = 0.28
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
                controlled_handoff_threshold=config.identity_controlled_handoff_threshold,
                controlled_handoff_min_old_track_gap_frames=config.identity_controlled_handoff_min_old_track_gap_frames,
                preferred_search_reacquire_enable=config.identity_preferred_search_reacquire_enable,
                preferred_search_reacquire_threshold=config.identity_preferred_search_reacquire_threshold,
                preferred_search_reacquire_max_disadvantage=config.identity_preferred_search_reacquire_max_disadvantage,
                preferred_search_reacquire_confirm_frames=config.identity_preferred_search_reacquire_confirm_frames,
                preferred_search_reacquire_instant_threshold=config.identity_preferred_search_reacquire_instant_threshold,
            )
        )
        self._frame_index = 0
        self._search_reacquire_uid = 0
        self._search_reacquire_direction: Optional[str] = None
        self._search_reacquire_eligible_tracks: Set[int] = set()
        self._last_quality_area_by_track_id = {}
        self._last_identity_center_by_track_id = {}

    def update(
        self,
        detections: Sequence[Detection],
        features: Sequence[Optional[Any]],
        *,
        image_width: int,
        image_height: Optional[int] = None,
    ) -> List[TrackRecord]:
        if len(features) != len(detections):
            raise ValueError("features length must match detections length")
        self._frame_index += 1
        if not detections:
            self._search_reacquire_eligible_tracks.clear()
            outputs = self.deepsort.update([], [], [], [], image_shape=_image_shape(image_width, image_height))
            return [self._to_record(out, image_width, image_height, 0) for out in outputs]

        bbox_xywh = [_xyxy_to_expanded_xywh(det.bbox, self.config.bbox_expand_scale) for det in detections]
        confidences = [float(det.score) for det in detections]
        classes = [int(det.class_id) for det in detections]
        outputs = self.deepsort.update(
            bbox_xywh,
            confidences,
            classes,
            features,
            image_shape=_image_shape(image_width, image_height),
        )
        fresh_track_ids = {
            int(output.track_id)
            for output in outputs
            if int(output.time_since_update) == 0
        }
        self._search_reacquire_eligible_tracks.intersection_update(fresh_track_ids)
        suppressed_track_ids = self._duplicate_identity_track_ids(outputs, image_width, image_height)
        candidate_count = max(0, len(detections) - len(suppressed_track_ids))
        return [
            self._to_record(
                out,
                image_width,
                image_height,
                candidate_count,
                duplicate_identity_box=int(out.track_id) in suppressed_track_ids,
            )
            for out in outputs
        ]

    def set_search_reacquire_context(
        self,
        *,
        active_uid: Optional[int],
        searching: bool,
        direction: Optional[str],
    ) -> None:
        uid = 0 if active_uid is None else int(active_uid)
        normalized_direction = str(direction or "").strip().lower()
        if not searching or uid <= 0 or normalized_direction not in ("left", "right"):
            self._search_reacquire_uid = 0
            self._search_reacquire_direction = None
            self._search_reacquire_eligible_tracks.clear()
            return
        if (
            uid != self._search_reacquire_uid
            or normalized_direction != self._search_reacquire_direction
        ):
            self._search_reacquire_eligible_tracks.clear()
        self._search_reacquire_uid = uid
        self._search_reacquire_direction = normalized_direction

    def reset(self) -> None:
        self.deepsort.reset()
        self.identity_bank.reset()
        self._frame_index = 0
        self._search_reacquire_uid = 0
        self._search_reacquire_direction = None
        self._search_reacquire_eligible_tracks.clear()
        self._last_quality_area_by_track_id.clear()
        self._last_identity_center_by_track_id.clear()

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
        duplicate_identity_box: bool = False,
    ) -> TrackRecord:
        x1, y1, x2, y2 = output.x1, output.y1, output.x2, output.y2
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        angle = 0.0
        if image_width > 0:
            norm = (cx - float(image_width) / 2.0) / (float(image_width) / 2.0)
            angle = norm * (float(self.config.hfov_deg) / 2.0)
        bbox_quality_ok, bbox_quality_reason = self._bbox_quality(
            (float(x1), float(y1), float(x2), float(y2)),
            image_width,
            image_height,
        )
        output_track_id = int(output.track_id)
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
        output_is_fresh = int(output.time_since_update) == 0
        bbox_quality_tier = self._bbox_identity_tier(
            bbox_quality_ok=bbox_quality_ok,
            bbox_quality_reason=bbox_quality_reason,
            confidence=float(output.confidence),
            bbox=(float(x1), float(y1), float(x2), float(y2)),
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
        if output_is_fresh and self._search_candidate_direction_ok(cx, image_width):
            self._search_reacquire_eligible_tracks.add(output_track_id)
        preferred_candidate_ok = (
            output_is_fresh
            and output_track_id in self._search_reacquire_eligible_tracks
        )
        reid_uid = self.identity_bank.assign(
            track_id=output_track_id,
            feature=output.feature,
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
        )
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

    def _search_candidate_direction_ok(self, center_x: float, image_width: int) -> bool:
        if self._search_reacquire_uid <= 0 or image_width <= 0:
            return False
        direction = self._search_reacquire_direction
        offset = min(0.45, max(0.0, float(self.config.identity_preferred_search_reacquire_side_ratio)))
        center_ratio = float(center_x) / float(image_width)
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
        if reasons and all(item.startswith(weak_prefixes) for item in reasons):
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
