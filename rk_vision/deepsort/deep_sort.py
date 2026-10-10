from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from .detection import Detection
from .nn_matching import NearestNeighborDistanceMetric
from .preprocessing import non_max_suppression
from .track import TrackState
from .tracker import MatchValidator, Tracker
from .low_score import capture_anchor


@dataclass(frozen=True)
class DeepSortConfig:
    max_dist: float = 0.35
    min_confidence: float = 0.50
    nms_max_overlap: float = 0.50
    max_iou_distance: float = 0.60
    max_age: int = 20
    max_output_age: int = 1
    n_init: int = 3
    nn_budget: int = 15
    max_bbox_age: int = 2
    feature_update_interval: int = 1
    low_score_min_confidence: float = 0.25
    camera_hfov_deg: float = 60.0


@dataclass(frozen=True)
class DeepSortOutput:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    confidence: float
    state: int
    time_since_update: int = 0
    hits: int = 0
    age: int = 0
    feature: Optional[Any] = None
    source_detection_index: Optional[int] = None
    low_score_continuation: bool = False
    association_previous_capture_frame_id: Optional[int] = None
    association_previous_capture_timestamp: Optional[float] = None


class DeepSort:
    def __init__(self, config: Optional[DeepSortConfig] = None) -> None:
        self.config = config or DeepSortConfig()
        metric = NearestNeighborDistanceMetric("cosine", self.config.max_dist, self.config.nn_budget)
        self.tracker = Tracker(
            metric,
            max_iou_distance=self.config.max_iou_distance,
            max_age=self.config.max_age,
            n_init=self.config.n_init,
            max_bbox_age=self.config.max_bbox_age,
        )
        self._frame_index = 0

    def update(
        self,
        bbox_xywh: Any,
        confidences: Sequence[float],
        classes: Sequence[int],
        features: Sequence[Optional[Any]],
        *,
        image_shape: Optional[Tuple[int, int]] = None,
        match_validator: Optional[MatchValidator] = None,
        provisional_features=None,
        capture_context=None,
        low_score_validator=None,
    ) -> List[DeepSortOutput]:
        store_feature = self._should_store_feature()
        detections = self._make_detections(bbox_xywh, confidences, classes, features, store_feature=store_feature)
        if detections:
            np = _np()
            boxes = np.asarray([det.tlwh for det in detections], dtype="float32")
            scores = np.asarray([det.confidence for det in detections], dtype="float32")
            keep = non_max_suppression(boxes, self.config.nms_max_overlap, scores)
            detections = [detections[i] for i in keep]

        self.tracker.predict()
        self.tracker.update(detections, match_validator=match_validator,
                            provisional_features=provisional_features,
                            low_score_validator=low_score_validator,
                            capture_context=capture_context, image_shape=image_shape,
                            camera_hfov_deg=self.config.camera_hfov_deg)
        # Only a real high-score match can establish/renew the finite bridge.
        # Weak observations never learn features or extend this deadline.
        by_source = {d.source_detection_index: d for d in detections}
        for track in self.tracker.tracks:
            if track.time_since_update == 0 and not track.low_score_continuation:
                detection = by_source.get(track.source_detection_index)
                track.low_score_anchor = capture_anchor(detection, capture_context)
        self._frame_index += 1

        outputs: List[DeepSortOutput] = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > self.config.max_output_age:
                continue
            x1, y1, x2, y2 = [float(v) for v in track.to_tlbr()]
            if image_shape is not None:
                height, width = image_shape
                x1 = max(0.0, min(float(width - 1), x1))
                x2 = max(0.0, min(float(width - 1), x2))
                y1 = max(0.0, min(float(height - 1), y1))
                y2 = max(0.0, min(float(height - 1), y2))
            outputs.append(
                DeepSortOutput(
                    track_id=int(track.track_id),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    class_id=int(track.cls),
                    confidence=float(track.confidence),
                    state=int(track.state),
                    time_since_update=int(track.time_since_update),
                    hits=int(track.hits),
                    age=int(track.age),
                    feature=track.last_feature if track.time_since_update == 0 else None,
                    source_detection_index=(
                        track.source_detection_index if track.time_since_update == 0 else None
                    ),
                    low_score_continuation=bool(track.time_since_update == 0 and track.low_score_continuation),
                    association_previous_capture_frame_id=(track.low_score_anchor or {}).get("capture_frame_id")
                        if track.low_score_continuation else None,
                    association_previous_capture_timestamp=(track.low_score_anchor or {}).get("capture_timestamp")
                        if track.low_score_continuation else None,
                )
            )
        return outputs

    def reset(self) -> None:
        self.tracker.tracks.clear()
        self.tracker.metric.samples.clear()
        self.tracker._next_id = 1
        self._frame_index = 0

    def _make_detections(
        self,
        bbox_xywh: Any,
        confidences: Sequence[float],
        classes: Sequence[int],
        features: Sequence[Optional[Any]],
        *,
        store_feature: bool = True,
    ) -> List[Detection]:
        np = _np()
        bbox_np = np.asarray(bbox_xywh, dtype="float32")
        if bbox_np.size == 0:
            return []
        bbox_np = bbox_np.reshape(-1, 4)
        detections: List[Detection] = []
        for source_index, (box, confidence, class_id, feature) in enumerate(
            zip(bbox_np, confidences, classes, features)
        ):
            if not np.isfinite(confidence) or float(confidence) < min(
                    self.config.min_confidence, self.config.low_score_min_confidence):
                continue
            tlwh = box.copy()
            tlwh[0] = box[0] - box[2] / 2.0
            tlwh[1] = box[1] - box[3] / 2.0
            if not np.isfinite(tlwh).all() or tlwh[2] <= 0.0 or tlwh[3] <= 0.0:
                continue
            detections.append(
                Detection(
                    tlwh,
                    float(confidence),
                    int(class_id),
                    feature,
                    store_feature=store_feature and float(confidence) >= self.config.min_confidence,
                    source_detection_index=source_index,
                    low_score_continuation=float(confidence) < self.config.min_confidence,
                )
            )
        return detections

    def _should_store_feature(self) -> bool:
        interval = max(1, int(self.config.feature_update_interval))
        return self._frame_index % interval == 0


def _np():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("numpy is required for DeepSORT") from exc
    return np
