from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

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
    identity_min_confidence: float = 0.65
    identity_min_area: float = 0.0
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
            )
        )
        self._frame_index = 0

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
        return [self._to_record(out, image_width, image_height, len(detections)) for out in outputs]

    def reset(self) -> None:
        self.deepsort.reset()
        self.identity_bank.reset()
        self._frame_index = 0

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
        reid_uid = self.identity_bank.assign(
            track_id=int(output.track_id),
            feature=output.feature,
            confidence=float(output.confidence),
            area=float(area),
            frame_index=int(self._frame_index),
            candidate_count=int(candidate_count),
            bbox_quality_ok=bbox_quality_ok,
            bbox_quality_reason=bbox_quality_reason,
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
