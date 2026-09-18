from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from typing import Any, List, Optional, Sequence, Tuple

from .frames import FramePacket, numpy_from_frame
from .reid import OSNetConfig, OSNetRKNNExtractor
from .reid_diagnostics import ReIDDiagnosticsWriter
from .tracker import DeepSortTracker, DeepSortTrackerConfig, TrackRecord
from .yolo11 import Detection, YOLO11Config, YOLO11RKNNDetector


@dataclass(frozen=True)
class SearchCandidateEvidence:
    """Read-only detector evidence; it carries no target or motion decision."""

    formal_persons: Tuple[Detection, ...] = ()
    probe_persons: Tuple[Detection, ...] = ()
    # Raw probe boxes remain available through the detector for video/debug
    # output; this describes the conservative clustering decision exposed to
    # the search gate.
    probe_cluster_diagnostics: Tuple[dict, ...] = ()


def _bbox_iou(first: Tuple[float, float, float, float], second: Tuple[float, float, float, float]) -> float:
    """Return IoU for detector boxes without pulling a CV dependency into the pipeline."""
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _probe_boxes_should_cluster(
    first: Detection,
    second: Detection,
    *,
    iou_threshold: float,
    center_distance_ratio: float,
) -> bool:
    """Recognize overlapping fragments of one detector hypothesis.

    Low-confidence YOLO decoding can emit boxes with slightly different
    extents for the same person. IoU handles normal overlap; the center/height
    fallback handles a narrow fragment nested in a much wider hypothesis.
    It deliberately does not use score, so two nearby people are not merged
    merely because one has a higher confidence.
    """
    first_box = tuple(float(value) for value in first.bbox)
    second_box = tuple(float(value) for value in second.bbox)
    if _bbox_iou(first_box, second_box) >= max(0.0, float(iou_threshold)):
        return True
    first_width = max(0.0, first_box[2] - first_box[0])
    first_height = max(0.0, first_box[3] - first_box[1])
    second_width = max(0.0, second_box[2] - second_box[0])
    second_height = max(0.0, second_box[3] - second_box[1])
    if min(first_width, first_height, second_width, second_height) <= 0.0:
        return False
    first_center = (
        (first_box[0] + first_box[2]) * 0.5,
        (first_box[1] + first_box[3]) * 0.5,
    )
    second_center = (
        (second_box[0] + second_box[2]) * 0.5,
        (second_box[1] + second_box[3]) * 0.5,
    )
    center_distance = (
        (first_center[0] - second_center[0]) ** 2
        + (first_center[1] - second_center[1]) ** 2
    ) ** 0.5
    scale = max(first_height, second_height, first_width, second_width, 1.0)
    vertical_overlap = max(
        0.0,
        min(first_box[3], second_box[3]) - max(first_box[1], second_box[1]),
    )
    vertical_overlap /= max(min(first_height, second_height), 1e-6)
    return (
        center_distance <= max(0.0, float(center_distance_ratio)) * scale
        and vertical_overlap >= 0.55
    )


def cluster_probe_detections(
    detections: Sequence[Detection],
    *,
    person_class_id: int = 0,
    iou_threshold: float = 0.45,
    center_distance_ratio: float = 0.18,
    min_score_gap: float = 0.03,
) -> Tuple[Tuple[Detection, ...], Tuple[dict, ...]]:
    """Collapse overlapping below-threshold boxes into conservative candidates.

    The returned detections contain at most one representative when a single
    cluster is unambiguous. If multiple spatial clusters compete with a small
    confidence gap, no probe is exposed to control/ReID. The second return
    value is diagnostic metadata for the raw cluster structure.
    """
    candidates = [
        detection
        for detection in detections
        if int(detection.class_id) == int(person_class_id)
        and float(detection.score) > 0.0
    ]
    if not candidates:
        return (), ()
    ordered = sorted(candidates, key=lambda item: float(item.score), reverse=True)
    clusters: List[List[Detection]] = []
    for detection in ordered:
        matching = [
            index
            for index, cluster in enumerate(clusters)
            if any(
                _probe_boxes_should_cluster(
                    detection,
                    member,
                    iou_threshold=iou_threshold,
                    center_distance_ratio=center_distance_ratio,
                )
                for member in cluster
            )
        ]
        if not matching:
            clusters.append([detection])
            continue
        target = matching[0]
        clusters[target].append(detection)
        # Connected components are useful for decoder fragments that form a
        # chain of slightly shifted boxes. Merge all touched components.
        for index in reversed(matching[1:]):
            clusters[target].extend(clusters.pop(index))
    clusters.sort(
        key=lambda cluster: max(float(item.score) for item in cluster),
        reverse=True,
    )
    representatives = [
        max(cluster, key=lambda item: float(item.score)) for cluster in clusters
    ]
    cluster_scores = [float(item.score) for item in representatives]
    score_gap = (
        cluster_scores[0] - cluster_scores[1]
        if len(cluster_scores) > 1
        else cluster_scores[0]
    )
    ambiguous = len(representatives) > 1 and score_gap < max(0.0, float(min_score_gap))
    metadata = tuple(
        {
            "cluster_index": int(index),
            "member_count": int(len(cluster)),
            "score": float(representative.score),
            "bbox": [float(value) for value in representative.bbox],
            "score_gap": float(score_gap),
            "ambiguous": bool(ambiguous),
        }
        for index, (cluster, representative) in enumerate(zip(clusters, representatives))
    )
    if ambiguous:
        return (), metadata
    return (representatives[0],), metadata


@dataclass(frozen=True)
class RKNNVisionConfig:
    yolo_model_path: str
    reid_model_path: str = ""
    reid_enable: bool = True
    person_class_id: int = 0
    conf_threshold: float = 0.25
    search_diagnostic_conf_threshold: float = 0.10
    # Search-only low-confidence boxes are clustered before being exposed to
    # the candidate gate. Formal detector output is never changed by these
    # settings.
    search_probe_cluster_enable: bool = True
    search_probe_cluster_iou_threshold: float = 0.45
    search_probe_cluster_center_distance_ratio: float = 0.18
    search_probe_cluster_min_score_gap: float = 0.03
    nms_threshold: float = 0.45
    yolo_input_size: int = 640
    yolo_num_classes: int = 80
    yolo_box_format: str = "xywh"
    yolo_input_format: str = "RGB"
    reid_input_width: int = 64
    reid_input_height: int = 128
    reid_input_format: str = "BGR"
    reid_input_dtype: str = "float32"
    reid_input_layout: str = "NCHW"
    reid_normalize: str = "imagenet"
    reid_color_fusion_enable: bool = True
    reid_color_fusion_weight: float = 0.35
    reid_partial_appearance_enable: bool = True
    reid_partial_osnet_enable: bool = True
    reid_diagnostics_enable: bool = False
    reid_diagnostics_dir: str = ""
    reid_diagnostics_max_samples: int = 2000
    reid_diagnostics_queue_capacity: int = 16
    reid_diagnostics_mapped_interval: int = 30
    frame_width: int = 1920
    frame_height: int = 1080
    hfov_deg: float = 90.0
    max_unmatched_num: int = 20
    max_output_age: int = 1
    accreditation_threshold: int = 3
    max_unmatched_bbox_matching: int = 2
    max_distance_iou: float = 0.60
    max_distance_cosine: float = 0.35
    feature_budget_size: int = 15
    feature_update_interval: int = 1
    deepsort_min_confidence: float = 0.50
    deepsort_nms_max_overlap: float = 0.50
    deepsort_bbox_expand_scale: float = 1.20
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
    identity_preferred_search_reacquire_min_score_gap: float = 0.25
    identity_preferred_search_soft_candidate_enable: bool = True
    identity_preferred_search_soft_candidate_threshold: float = 0.30
    identity_preferred_search_soft_min_score_gap: float = 0.15
    identity_preferred_search_soft_min_area_ratio: float = 0.25
    identity_preferred_search_soft_min_confidence: float = 0.80
    identity_partial_appearance_enable: bool = True
    identity_partial_match_threshold: float = 0.34
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
    identity_suppress_duplicate_uids: bool = True
    predicted_reid_verify_enable: bool = False
    predicted_reid_verify_threshold: float = 0.30
    predicted_reid_duplicate_iou_threshold: float = 0.30
    predicted_reid_duplicate_overlap_threshold: float = 0.45
    target: str = "rk3588"
    core_mask: str = "auto"
    backend: str = "auto"

    @classmethod
    def from_env(cls) -> "RKNNVisionConfig":
        return cls(
            yolo_model_path=os.environ.get("VISION_MODEL_PATH", "models/yolo11s.rknn").strip(),
            reid_model_path=os.environ.get("VISION_REID_MODEL_PATH", "models/deepsort.rknn").strip(),
            reid_enable=os.environ.get("VISION_REID_ENABLE", "1").strip() != "0",
            person_class_id=int(os.environ.get("PERSON_CLASS_ID", "0")),
            conf_threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.25")),
            search_diagnostic_conf_threshold=float(
                os.environ.get("RKNN_SEARCH_DIAGNOSTIC_CONF_THRESHOLD", "0.10")
            ),
            search_probe_cluster_enable=os.environ.get(
                "RKNN_SEARCH_PROBE_CLUSTER_ENABLE", "1"
            ).strip()
            != "0",
            search_probe_cluster_iou_threshold=float(
                os.environ.get("RKNN_SEARCH_PROBE_CLUSTER_IOU_THRESHOLD", "0.45")
            ),
            search_probe_cluster_center_distance_ratio=float(
                os.environ.get(
                    "RKNN_SEARCH_PROBE_CLUSTER_CENTER_DISTANCE_RATIO", "0.18"
                )
            ),
            search_probe_cluster_min_score_gap=float(
                os.environ.get("RKNN_SEARCH_PROBE_CLUSTER_MIN_SCORE_GAP", "0.03")
            ),
            nms_threshold=float(os.environ.get("RKNN_YOLO_NMS_THRESHOLD", "0.45")),
            yolo_input_size=int(os.environ.get("RKNN_YOLO_INPUT_SIZE", "640")),
            yolo_num_classes=int(os.environ.get("RKNN_YOLO_NUM_CLASSES", "80")),
            yolo_box_format=os.environ.get("RKNN_YOLO_BOX_FORMAT", "xywh").strip(),
            yolo_input_format=os.environ.get("RKNN_YOLO_INPUT_FORMAT", "RGB").strip(),
            reid_input_width=int(os.environ.get("RKNN_REID_INPUT_WIDTH", "64")),
            reid_input_height=int(os.environ.get("RKNN_REID_INPUT_HEIGHT", "128")),
            reid_input_format=os.environ.get("RKNN_REID_INPUT_FORMAT", "BGR").strip(),
            reid_input_dtype=os.environ.get("RKNN_REID_INPUT_DTYPE", "float32").strip(),
            reid_input_layout=os.environ.get("RKNN_REID_INPUT_LAYOUT", "NCHW").strip(),
            reid_normalize=os.environ.get("RKNN_REID_NORMALIZE", "imagenet").strip(),
            reid_color_fusion_enable=os.environ.get("RKNN_REID_COLOR_FUSION_ENABLE", "1").strip() != "0",
            reid_color_fusion_weight=float(os.environ.get("RKNN_REID_COLOR_FUSION_WEIGHT", "0.35")),
            reid_partial_appearance_enable=os.environ.get("RKNN_REID_PARTIAL_APPEARANCE_ENABLE", "1").strip() != "0",
            reid_partial_osnet_enable=os.environ.get("RKNN_REID_PARTIAL_OSNET_ENABLE", "1").strip() != "0",
            reid_diagnostics_enable=os.environ.get("RKNN_REID_DIAGNOSTICS_ENABLE", "0").strip() != "0",
            reid_diagnostics_dir=(
                os.path.join(os.environ["FOLLOW_LOG_DIR"], "reid_diagnostics")
                if os.environ.get("FOLLOW_LOG_DIR") else ""
            ),
            reid_diagnostics_max_samples=max(0, int(os.environ.get("RKNN_REID_DIAGNOSTICS_MAX_SAMPLES", "2000"))),
            reid_diagnostics_queue_capacity=max(1, int(os.environ.get("RKNN_REID_DIAGNOSTICS_QUEUE_CAPACITY", "16"))),
            reid_diagnostics_mapped_interval=max(1, int(os.environ.get("RKNN_REID_DIAGNOSTICS_MAPPED_INTERVAL", "30"))),
            frame_width=int(os.environ.get("VISION_FRAME_WIDTH", "1920")),
            frame_height=int(os.environ.get("VISION_FRAME_HEIGHT", "1080")),
            hfov_deg=float(os.environ.get("VISION_HFOV_DEG", "90.0")),
            max_unmatched_num=max(0, int(os.environ.get("Y8_DEEPSORT_MAX_UNMATCHED_NUM", "20"))),
            max_output_age=max(0, int(os.environ.get("Y8_DEEPSORT_MAX_OUTPUT_AGE", "1"))),
            accreditation_threshold=max(1, int(os.environ.get("Y8_DEEPSORT_ACCREDITATION_THRESHOLD", "3"))),
            max_unmatched_bbox_matching=max(
                0,
                int(os.environ.get("Y8_DEEPSORT_MAX_UNMATCHED_TIMES_FOR_BBOX_MATCHING", "2")),
            ),
            max_distance_iou=float(os.environ.get("Y8_DEEPSORT_MAX_DISTANCE_IOU", "0.60")),
            max_distance_cosine=float(os.environ.get("Y8_DEEPSORT_MAX_DISTANCE_CONSINE", "0.35")),
            feature_budget_size=max(1, int(os.environ.get("Y8_DEEPSORT_FEATURE_BUDGET_SIZE", "15"))),
            feature_update_interval=max(1, int(os.environ.get("Y8_DEEPSORT_FEATURE_UPDATE_INTERVAL", "1"))),
            deepsort_min_confidence=float(os.environ.get("Y8_DEEPSORT_MIN_CONFIDENCE", "0.50")),
            deepsort_nms_max_overlap=float(os.environ.get("Y8_DEEPSORT_NMS_MAX_OVERLAP", "0.50")),
            deepsort_bbox_expand_scale=float(os.environ.get("Y8_DEEPSORT_BBOX_EXPAND_SCALE", "1.20")),
            identity_bank_enable=os.environ.get("Y8_IDENTITY_BANK_ENABLE", "1").strip() != "0",
            identity_match_threshold=float(os.environ.get("Y8_IDENTITY_MATCH_THRESHOLD", "0.32")),
            identity_match_margin=float(os.environ.get("Y8_IDENTITY_MATCH_MARGIN", "0.03")),
            identity_update_threshold=float(os.environ.get("Y8_IDENTITY_UPDATE_THRESHOLD", "0.30")),
            identity_update_interval=max(1, int(os.environ.get("Y8_IDENTITY_UPDATE_INTERVAL", "5"))),
            identity_max_features=max(1, int(os.environ.get("Y8_IDENTITY_MAX_FEATURES", "20"))),
            identity_max_weak_features=max(0, int(os.environ.get("Y8_IDENTITY_MAX_WEAK_FEATURES", "8"))),
            identity_diversity_min_distance=max(
                0.0, float(os.environ.get("Y8_IDENTITY_DIVERSITY_MIN_DISTANCE", "0.02"))
            ),
            identity_diversity_replace_margin=max(
                0.0, float(os.environ.get("Y8_IDENTITY_DIVERSITY_REPLACE_MARGIN", "0.01"))
            ),
            identity_weak_update_threshold=float(
                os.environ.get("Y8_IDENTITY_WEAK_UPDATE_THRESHOLD", "0.42")
            ),
            identity_weak_update_interval=max(
                1, int(os.environ.get("Y8_IDENTITY_WEAK_UPDATE_INTERVAL", "3"))
            ),
            identity_weak_match_penalty=max(
                0.0, float(os.environ.get("Y8_IDENTITY_WEAK_MATCH_PENALTY", "0.10"))
            ),
            identity_weak_reacquire_threshold=float(
                os.environ.get("Y8_IDENTITY_WEAK_REACQUIRE_THRESHOLD", "0.38")
            ),
            identity_weak_reacquire_confirm_frames=max(
                2, int(os.environ.get("Y8_IDENTITY_WEAK_REACQUIRE_CONFIRM_FRAMES", "3"))
            ),
            identity_weak_quality_weight=max(
                0.0,
                min(1.0, float(os.environ.get("Y8_IDENTITY_WEAK_QUALITY_WEIGHT", "0.30"))),
            ),
            identity_min_confidence=float(os.environ.get("Y8_IDENTITY_MIN_CONFIDENCE", "0.65")),
            identity_min_area=float(os.environ.get("Y8_IDENTITY_MIN_AREA", "0.0")),
            identity_min_width_px=float(os.environ.get("Y8_IDENTITY_MIN_WIDTH_PX", "0.0")),
            identity_min_height_px=float(os.environ.get("Y8_IDENTITY_MIN_HEIGHT_PX", "0.0")),
            identity_max_single_frame_area_shrink_ratio=float(
                os.environ.get("Y8_IDENTITY_MAX_SINGLE_FRAME_AREA_SHRINK_RATIO", "0.30")
            ),
            identity_area_shrink_max_gap_frames=max(
                1,
                int(os.environ.get("Y8_IDENTITY_AREA_SHRINK_MAX_GAP_FRAMES", "2")),
            ),
            identity_max_center_jump_ratio=float(
                os.environ.get("Y8_IDENTITY_MAX_CENTER_JUMP_RATIO", "0.30")
            ),
            identity_center_jump_max_gap_frames=max(
                1,
                int(os.environ.get("Y8_IDENTITY_CENTER_JUMP_MAX_GAP_FRAMES", "2")),
            ),
            identity_swap_min_mapped_jump_ratio=float(
                os.environ.get("Y8_IDENTITY_SWAP_MIN_MAPPED_JUMP_RATIO", "0.08")
            ),
            identity_swap_max_replacement_distance_ratio=float(
                os.environ.get("Y8_IDENTITY_SWAP_MAX_REPLACEMENT_DISTANCE_RATIO", "0.12")
            ),
            identity_max_area_ratio=float(os.environ.get("Y8_IDENTITY_MAX_AREA_RATIO", "0.75")),
            identity_max_width_ratio=float(os.environ.get("Y8_IDENTITY_MAX_WIDTH_RATIO", "0.85")),
            identity_max_height_ratio=float(os.environ.get("Y8_IDENTITY_MAX_HEIGHT_RATIO", "1.00")),
            identity_min_aspect_ratio=float(os.environ.get("Y8_IDENTITY_MIN_ASPECT_RATIO", "0.18")),
            identity_max_aspect_ratio=float(os.environ.get("Y8_IDENTITY_MAX_ASPECT_RATIO", "1.45")),
            identity_max_edge_touch_count=int(os.environ.get("Y8_IDENTITY_MAX_EDGE_TOUCH_COUNT", "2")),
            identity_edge_margin_ratio=float(os.environ.get("Y8_IDENTITY_EDGE_MARGIN_RATIO", "0.02")),
            identity_reacquire_threshold=float(os.environ.get("Y8_IDENTITY_REACQUIRE_THRESHOLD", "0.50")),
            identity_reacquire_max_frames=max(0, int(os.environ.get("Y8_IDENTITY_REACQUIRE_MAX_FRAMES", "90"))),
            identity_reacquire_margin=float(os.environ.get("Y8_IDENTITY_REACQUIRE_MARGIN", "0.08")),
            identity_reacquire_single_candidate_only=os.environ.get(
                "Y8_IDENTITY_REACQUIRE_SINGLE_CANDIDATE_ONLY",
                "1",
            ).strip()
            != "0",
            identity_reacquire_multi_candidate_enable=os.environ.get(
                "Y8_IDENTITY_REACQUIRE_MULTI_CANDIDATE_ENABLE",
                "1",
            ).strip()
            != "0",
            identity_reacquire_multi_candidate_threshold=float(
                os.environ.get("Y8_IDENTITY_REACQUIRE_MULTI_CANDIDATE_THRESHOLD", "0.40")
            ),
            identity_reacquire_multi_candidate_margin=float(
                os.environ.get("Y8_IDENTITY_REACQUIRE_MULTI_CANDIDATE_MARGIN", "0.12")
            ),
            identity_new_confirm_frames=max(1, int(os.environ.get("Y8_IDENTITY_NEW_CONFIRM_FRAMES", "2"))),
            identity_mapped_verify_enable=os.environ.get("Y8_IDENTITY_MAPPED_VERIFY_ENABLE", "1").strip() != "0",
            identity_mapped_verify_threshold=float(os.environ.get("Y8_IDENTITY_MAPPED_VERIFY_THRESHOLD", "0.40")),
            identity_mapped_bad_quality_returns_unassigned=os.environ.get(
                "Y8_IDENTITY_MAPPED_BAD_QUALITY_RETURNS_UNASSIGNED",
                "1",
            ).strip()
            != "0",
            identity_exclusive_uid_claim_enable=os.environ.get("Y8_IDENTITY_EXCLUSIVE_UID_CLAIM_ENABLE", "1").strip()
            != "0",
            identity_exclusive_uid_claim_frames=max(0, int(os.environ.get("Y8_IDENTITY_EXCLUSIVE_UID_CLAIM_FRAMES", "15"))),
            identity_controlled_handoff_enable=os.environ.get("Y8_IDENTITY_CONTROLLED_HANDOFF_ENABLE", "1").strip()
            != "0",
            identity_controlled_handoff_confirm_frames=max(
                1, int(os.environ.get("Y8_IDENTITY_CONTROLLED_HANDOFF_CONFIRM_FRAMES", "2"))
            ),
            identity_controlled_handoff_instant_threshold=float(
                os.environ.get("Y8_IDENTITY_CONTROLLED_HANDOFF_INSTANT_THRESHOLD", "0.15")
            ),
            identity_controlled_handoff_threshold=float(
                os.environ.get("Y8_IDENTITY_CONTROLLED_HANDOFF_THRESHOLD", "0.30")
            ),
            identity_controlled_handoff_min_old_track_gap_frames=max(
                1, int(os.environ.get("Y8_IDENTITY_CONTROLLED_HANDOFF_MIN_OLD_TRACK_GAP_FRAMES", "5"))
            ),
            identity_handoff_geometry_max_gap_frames=max(
                1, int(os.environ.get("Y8_IDENTITY_HANDOFF_GEOMETRY_MAX_GAP_FRAMES", "15"))
            ),
            identity_handoff_geometry_max_center_jump_ratio=max(
                0.0, min(1.0, float(os.environ.get("Y8_IDENTITY_HANDOFF_GEOMETRY_MAX_CENTER_JUMP_RATIO", "0.25")))
            ),
            identity_handoff_geometry_min_area_similarity=max(
                0.0, min(1.0, float(os.environ.get("Y8_IDENTITY_HANDOFF_GEOMETRY_MIN_AREA_SIMILARITY", "0.35")))
            ),
            identity_preferred_search_reacquire_enable=os.environ.get(
                "Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_ENABLE",
                "1",
            ).strip()
            != "0",
            identity_preferred_search_reacquire_threshold=float(
                os.environ.get("Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_THRESHOLD", "0.20")
            ),
            identity_preferred_search_reacquire_max_disadvantage=float(
                os.environ.get("Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_MAX_DISADVANTAGE", "0.05")
            ),
            identity_preferred_search_reacquire_min_confidence=max(
                0.0,
                min(
                    1.0,
                    float(os.environ.get(
                        "Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_MIN_CONFIDENCE",
                        "0.50",
                    )),
                ),
            ),
            identity_preferred_search_reacquire_observation_min_confidence=max(
                0.0,
                min(
                    1.0,
                    float(
                        os.environ.get(
                            "Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_OBSERVATION_MIN_CONFIDENCE",
                            "0.25",
                        )
                    ),
                ),
            ),
            identity_preferred_search_reacquire_max_age_sec=max(
                0.0,
                float(os.environ.get("Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_MAX_AGE_SEC", "0.35")),
            ),
            identity_preferred_search_reacquire_late_candidate_enable=os.environ.get(
                "Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_LATE_CANDIDATE_ENABLE",
                "1",
            ).strip()
            != "0",
            identity_preferred_search_reacquire_confirm_frames=max(
                2,
                int(os.environ.get("Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_CONFIRM_FRAMES", "2")),
            ),
            identity_preferred_search_reacquire_instant_threshold=float(
                os.environ.get(
                    "Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_INSTANT_THRESHOLD",
                    "0.15",
                )
            ),
            identity_preferred_search_reacquire_min_score_gap=max(
                0.0,
                min(
                    1.0,
                    float(
                        os.environ.get(
                            "Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_MIN_SCORE_GAP",
                            "0.25",
                        )
                    ),
                ),
            ),
            identity_preferred_search_soft_candidate_enable=os.environ.get(
                "Y8_IDENTITY_PREFERRED_SEARCH_SOFT_CANDIDATE_ENABLE", "1"
            ).strip() != "0",
            identity_preferred_search_soft_candidate_threshold=max(
                0.0,
                float(
                    os.environ.get(
                        "Y8_IDENTITY_PREFERRED_SEARCH_SOFT_CANDIDATE_THRESHOLD", "0.30"
                    )
                ),
            ),
            identity_preferred_search_soft_min_score_gap=max(
                0.0,
                min(
                    1.0,
                    float(
                        os.environ.get(
                            "Y8_IDENTITY_PREFERRED_SEARCH_SOFT_MIN_SCORE_GAP", "0.15"
                        )
                    ),
                ),
            ),
            identity_preferred_search_soft_min_area_ratio=max(
                0.0,
                min(
                    1.0,
                    float(
                        os.environ.get(
                            "Y8_IDENTITY_PREFERRED_SEARCH_SOFT_MIN_AREA_RATIO", "0.25"
                        )
                    ),
                ),
            ),
            identity_preferred_search_soft_min_confidence=max(
                0.0,
                min(
                    1.0,
                    float(
                        os.environ.get(
                            "Y8_IDENTITY_PREFERRED_SEARCH_SOFT_MIN_CONFIDENCE", "0.80"
                        )
                    ),
                ),
            ),
            identity_preferred_search_reacquire_side_ratio=float(
                os.environ.get("Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_SIDE_RATIO", "0.05")
            ),
            identity_partial_appearance_enable=os.environ.get(
                "Y8_IDENTITY_PARTIAL_APPEARANCE_ENABLE", "1"
            ).strip() != "0",
            identity_partial_match_threshold=float(
                os.environ.get("Y8_IDENTITY_PARTIAL_MATCH_THRESHOLD", "0.34")
            ),
            identity_partial_max_features=max(
                1, int(os.environ.get("Y8_IDENTITY_PARTIAL_MAX_FEATURES", "8"))
            ),
            identity_partial_update_threshold=float(
                os.environ.get("Y8_IDENTITY_PARTIAL_UPDATE_THRESHOLD", "0.30")
            ),
            identity_duplicate_box_suppression_enable=os.environ.get(
                "Y8_IDENTITY_DUPLICATE_BOX_SUPPRESSION_ENABLE",
                "1",
            ).strip()
            != "0",
            identity_duplicate_iou_threshold=float(
                os.environ.get("Y8_IDENTITY_DUPLICATE_IOU_THRESHOLD", "0.55")
            ),
            identity_duplicate_vertical_overlap_threshold=float(
                os.environ.get("Y8_IDENTITY_DUPLICATE_VERTICAL_OVERLAP_THRESHOLD", "0.88")
            ),
            identity_duplicate_horizontal_overlap_threshold=float(
                os.environ.get("Y8_IDENTITY_DUPLICATE_HORIZONTAL_OVERLAP_THRESHOLD", "0.40")
            ),
            identity_duplicate_large_height_ratio=float(
                os.environ.get("Y8_IDENTITY_DUPLICATE_LARGE_HEIGHT_RATIO", "0.80")
            ),
            identity_duplicate_large_width_ratio=float(
                os.environ.get("Y8_IDENTITY_DUPLICATE_LARGE_WIDTH_RATIO", "0.45")
            ),
            identity_duplicate_max_area_ratio=float(
                os.environ.get("Y8_IDENTITY_DUPLICATE_MAX_AREA_RATIO", "0.65")
            ),
            identity_duplicate_bottom_gap_ratio=float(
                os.environ.get("Y8_IDENTITY_DUPLICATE_BOTTOM_GAP_RATIO", "0.10")
            ),
            identity_suppress_duplicate_uids=os.environ.get("Y8_IDENTITY_SUPPRESS_DUPLICATE_UIDS", "1").strip() != "0",
            predicted_reid_verify_enable=os.environ.get("Y8_PREDICTED_REID_VERIFY_ENABLE", "0").strip() != "0",
            predicted_reid_verify_threshold=float(os.environ.get("Y8_PREDICTED_REID_VERIFY_THRESHOLD", "0.30")),
            predicted_reid_duplicate_iou_threshold=float(
                os.environ.get("Y8_PREDICTED_REID_DUPLICATE_IOU_THRESHOLD", "0.30")
            ),
            predicted_reid_duplicate_overlap_threshold=float(
                os.environ.get("Y8_PREDICTED_REID_DUPLICATE_OVERLAP_THRESHOLD", "0.45")
            ),
            target=os.environ.get("RKNN_TARGET", "rk3588").strip(),
            core_mask=os.environ.get("RKNN_CORE_MASK", "auto").strip(),
            backend=os.environ.get("RKNN_BACKEND", "auto").strip(),
        )


class RKNNVisionPipeline:
    """YOLO11 + optional ReID embedding + tracker pipeline for externally supplied frames."""

    def __init__(self, config: Optional[RKNNVisionConfig] = None, logger: Any = None) -> None:
        self.config = config or RKNNVisionConfig.from_env()
        self.logger = logger
        self.detector = YOLO11RKNNDetector(
            YOLO11Config(
                model_path=self.config.yolo_model_path,
                input_size=self.config.yolo_input_size,
                conf_threshold=self.config.conf_threshold,
                search_diagnostic_conf_threshold=self.config.search_diagnostic_conf_threshold,
                search_diagnostic_class_id=self.config.person_class_id,
                nms_threshold=self.config.nms_threshold,
                num_classes=self.config.yolo_num_classes,
                input_format=self.config.yolo_input_format,
                output_box_format=self.config.yolo_box_format,
                target=self.config.target,
                core_mask=self.config.core_mask,
                backend=self.config.backend,
            )
        )
        self.reid = OSNetRKNNExtractor(
            OSNetConfig(
                model_path=self.config.reid_model_path,
                enabled=self.config.reid_enable,
                input_width=self.config.reid_input_width,
                input_height=self.config.reid_input_height,
                input_format=self.config.reid_input_format,
                input_dtype=self.config.reid_input_dtype,
                input_layout=self.config.reid_input_layout,
                normalize=self.config.reid_normalize,
                color_fusion_enable=self.config.reid_color_fusion_enable,
                color_fusion_weight=self.config.reid_color_fusion_weight,
                partial_appearance_enable=self.config.reid_partial_appearance_enable,
                partial_osnet_enable=self.config.reid_partial_osnet_enable,
                target=self.config.target,
                core_mask=self.config.core_mask,
                backend=self.config.backend,
            )
        )
        self.tracker = DeepSortTracker(
            DeepSortTrackerConfig(
                max_age=self.config.max_unmatched_num,
                max_output_age=self.config.max_output_age,
                n_init=self.config.accreditation_threshold,
                max_iou_distance=self.config.max_distance_iou,
                max_cosine_distance=self.config.max_distance_cosine,
                max_bbox_age=self.config.max_unmatched_bbox_matching,
                nn_budget=self.config.feature_budget_size,
                feature_update_interval=self.config.feature_update_interval,
                min_confidence=self.config.deepsort_min_confidence,
                nms_max_overlap=self.config.deepsort_nms_max_overlap,
                bbox_expand_scale=self.config.deepsort_bbox_expand_scale,
                hfov_deg=self.config.hfov_deg,
                identity_bank_enable=self.config.identity_bank_enable,
                identity_match_threshold=self.config.identity_match_threshold,
                identity_match_margin=self.config.identity_match_margin,
                identity_update_threshold=self.config.identity_update_threshold,
                identity_update_interval=self.config.identity_update_interval,
                identity_max_features=self.config.identity_max_features,
                identity_max_weak_features=self.config.identity_max_weak_features,
                identity_diversity_min_distance=self.config.identity_diversity_min_distance,
                identity_diversity_replace_margin=self.config.identity_diversity_replace_margin,
                identity_weak_update_threshold=self.config.identity_weak_update_threshold,
                identity_weak_update_interval=self.config.identity_weak_update_interval,
                identity_weak_match_penalty=self.config.identity_weak_match_penalty,
                identity_weak_reacquire_threshold=self.config.identity_weak_reacquire_threshold,
                identity_weak_reacquire_confirm_frames=self.config.identity_weak_reacquire_confirm_frames,
                identity_weak_quality_weight=self.config.identity_weak_quality_weight,
                identity_min_confidence=self.config.identity_min_confidence,
                identity_min_area=self.config.identity_min_area,
                identity_min_width_px=self.config.identity_min_width_px,
                identity_min_height_px=self.config.identity_min_height_px,
                identity_max_single_frame_area_shrink_ratio=(
                    self.config.identity_max_single_frame_area_shrink_ratio
                ),
                identity_area_shrink_max_gap_frames=(
                    self.config.identity_area_shrink_max_gap_frames
                ),
                identity_max_center_jump_ratio=self.config.identity_max_center_jump_ratio,
                identity_center_jump_max_gap_frames=self.config.identity_center_jump_max_gap_frames,
                identity_swap_min_mapped_jump_ratio=self.config.identity_swap_min_mapped_jump_ratio,
                identity_swap_max_replacement_distance_ratio=self.config.identity_swap_max_replacement_distance_ratio,
                identity_max_area_ratio=self.config.identity_max_area_ratio,
                identity_max_width_ratio=self.config.identity_max_width_ratio,
                identity_max_height_ratio=self.config.identity_max_height_ratio,
                identity_min_aspect_ratio=self.config.identity_min_aspect_ratio,
                identity_max_aspect_ratio=self.config.identity_max_aspect_ratio,
                identity_max_edge_touch_count=self.config.identity_max_edge_touch_count,
                identity_edge_margin_ratio=self.config.identity_edge_margin_ratio,
                identity_reacquire_threshold=self.config.identity_reacquire_threshold,
                identity_reacquire_max_frames=self.config.identity_reacquire_max_frames,
                identity_reacquire_margin=self.config.identity_reacquire_margin,
                identity_reacquire_single_candidate_only=self.config.identity_reacquire_single_candidate_only,
                identity_reacquire_multi_candidate_enable=self.config.identity_reacquire_multi_candidate_enable,
                identity_reacquire_multi_candidate_threshold=self.config.identity_reacquire_multi_candidate_threshold,
                identity_reacquire_multi_candidate_margin=self.config.identity_reacquire_multi_candidate_margin,
                identity_new_confirm_frames=self.config.identity_new_confirm_frames,
                identity_mapped_verify_enable=self.config.identity_mapped_verify_enable,
                identity_mapped_verify_threshold=self.config.identity_mapped_verify_threshold,
                identity_mapped_bad_quality_returns_unassigned=self.config.identity_mapped_bad_quality_returns_unassigned,
                identity_exclusive_uid_claim_enable=self.config.identity_exclusive_uid_claim_enable,
                identity_exclusive_uid_claim_frames=self.config.identity_exclusive_uid_claim_frames,
                identity_controlled_handoff_enable=self.config.identity_controlled_handoff_enable,
                identity_controlled_handoff_confirm_frames=self.config.identity_controlled_handoff_confirm_frames,
                identity_controlled_handoff_instant_threshold=self.config.identity_controlled_handoff_instant_threshold,
                identity_controlled_handoff_threshold=self.config.identity_controlled_handoff_threshold,
                identity_controlled_handoff_min_old_track_gap_frames=self.config.identity_controlled_handoff_min_old_track_gap_frames,
                identity_handoff_geometry_max_gap_frames=self.config.identity_handoff_geometry_max_gap_frames,
                identity_handoff_geometry_max_center_jump_ratio=self.config.identity_handoff_geometry_max_center_jump_ratio,
                identity_handoff_geometry_min_area_similarity=self.config.identity_handoff_geometry_min_area_similarity,
                identity_preferred_search_reacquire_enable=self.config.identity_preferred_search_reacquire_enable,
                identity_preferred_search_reacquire_threshold=self.config.identity_preferred_search_reacquire_threshold,
                identity_preferred_search_reacquire_max_disadvantage=self.config.identity_preferred_search_reacquire_max_disadvantage,
                identity_preferred_search_reacquire_min_confidence=self.config.identity_preferred_search_reacquire_min_confidence,
                identity_preferred_search_reacquire_observation_min_confidence=(
                    self.config.identity_preferred_search_reacquire_observation_min_confidence
                ),
                identity_preferred_search_reacquire_max_age_sec=self.config.identity_preferred_search_reacquire_max_age_sec,
                identity_preferred_search_reacquire_late_candidate_enable=self.config.identity_preferred_search_reacquire_late_candidate_enable,
                identity_preferred_search_reacquire_confirm_frames=self.config.identity_preferred_search_reacquire_confirm_frames,
                identity_preferred_search_reacquire_instant_threshold=self.config.identity_preferred_search_reacquire_instant_threshold,
                identity_preferred_search_reacquire_min_score_gap=self.config.identity_preferred_search_reacquire_min_score_gap,
                identity_preferred_search_soft_candidate_enable=self.config.identity_preferred_search_soft_candidate_enable,
                identity_preferred_search_soft_candidate_threshold=self.config.identity_preferred_search_soft_candidate_threshold,
                identity_preferred_search_soft_min_score_gap=self.config.identity_preferred_search_soft_min_score_gap,
                identity_preferred_search_soft_min_area_ratio=self.config.identity_preferred_search_soft_min_area_ratio,
                identity_preferred_search_soft_min_confidence=self.config.identity_preferred_search_soft_min_confidence,
                identity_preferred_search_reacquire_side_ratio=self.config.identity_preferred_search_reacquire_side_ratio,
                identity_partial_appearance_enable=self.config.identity_partial_appearance_enable,
                identity_partial_match_threshold=self.config.identity_partial_match_threshold,
                identity_partial_max_features=self.config.identity_partial_max_features,
                identity_partial_update_threshold=self.config.identity_partial_update_threshold,
                identity_duplicate_box_suppression_enable=self.config.identity_duplicate_box_suppression_enable,
                identity_duplicate_iou_threshold=self.config.identity_duplicate_iou_threshold,
                identity_duplicate_vertical_overlap_threshold=self.config.identity_duplicate_vertical_overlap_threshold,
                identity_duplicate_horizontal_overlap_threshold=self.config.identity_duplicate_horizontal_overlap_threshold,
                identity_duplicate_large_height_ratio=self.config.identity_duplicate_large_height_ratio,
                identity_duplicate_large_width_ratio=self.config.identity_duplicate_large_width_ratio,
                identity_duplicate_max_area_ratio=self.config.identity_duplicate_max_area_ratio,
                identity_duplicate_bottom_gap_ratio=self.config.identity_duplicate_bottom_gap_ratio,
            )
        )
        self.last_detections: List[Detection] = []
        self._frame_context: dict = {}
        self._reid_diagnostics = None
        if self.config.reid_diagnostics_enable and self.config.reid_diagnostics_dir:
            self._reid_diagnostics = ReIDDiagnosticsWriter(
                self.config.reid_diagnostics_dir,
                max_samples=self.config.reid_diagnostics_max_samples,
                queue_capacity=self.config.reid_diagnostics_queue_capacity,
                logger=self.logger,
            )
        self.last_search_diagnostic_detections: List[Detection] = []
        self.last_search_probe_clusters: List[Detection] = []
        self.last_search_probe_cluster_diagnostics: Tuple[dict, ...] = ()
        self.last_search_candidate_evidence = SearchCandidateEvidence()
        self.last_predicted_reid_verifications: List[dict] = []
        self.last_frame_width = int(self.config.frame_width)
        self.last_frame_height = int(self.config.frame_height)
        self.last_timing_ms = {
            "yolo_preprocess": 0.0,
            "yolo_inference": 0.0,
            "yolo_decode": 0.0,
            "yolo_nms": 0.0,
            "yolo_postprocess": 0.0,
            "yolo_total": 0.0,
            "reid_preprocess": 0.0,
            "reid_inference": 0.0,
            "reid_partial_inference": 0.0,
            "reid_postprocess": 0.0,
            "reid_total": 0.0,
            "tracker": 0.0,
            "total": 0.0,
        }

    def _follow_size_candidates(self, detections, *, source: str):
        """Filter control evidence, not the raw detector/recording output.

        Apply the identity size limits before either ReID or search observation.
        A track's expanded box, a high detector score, or a tiny ReID distance
        must not grant a small raw detection permission to stop/redirect search.
        """
        accepted = []
        for detection in detections:
            x1, y1, x2, y2 = (float(v) for v in detection.bbox)
            width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
            area = width * height
            limits = (
                ("width", width, float(getattr(self.config, "identity_min_width_px", 0.0))),
                ("height", height, float(getattr(self.config, "identity_min_height_px", 0.0))),
                ("area", area, float(getattr(self.config, "identity_min_area", 0.0))),
            )
            reasons = [f"{name}<{limit:g}" for name, value, limit in limits if value < limit]
            if not reasons:
                accepted.append(detection)
                continue
            item = {
                "source": source, "bbox": tuple(detection.bbox),
                "width_px": width, "height_px": height, "area_px": area,
                "score": float(detection.score), "reason": ",".join(reasons),
            }
            self.last_follow_size_rejections.append(item)
            logger = getattr(self, "logger", None)
            if logger is not None:
                logger.info(
                    "follow_bbox_size_rejected capture_frame_id=%s source=%s "
                    "bbox=%s width_px=%.2f height_px=%.2f area_px=%.2f "
                    "score=%.3f reason=%s identity_allowed=False observation_allowed=False",
                    getattr(self, "_frame_context", {}).get("capture_frame_id"),
                    source, item["bbox"], width, height, area,
                    item["score"], item["reason"],
                )
        return accepted

    def _record_detector_output(self, detections: List[Detection]) -> List[Detection]:
        self.last_detections = detections
        self.last_follow_size_rejections = []
        self.last_search_diagnostic_detections = list(
            self.detector.last_search_diagnostic_detections
        )
        persons = [
            det
            for det in detections
            if int(det.class_id) == int(self.config.person_class_id)
            and float(det.score) >= float(self.config.conf_threshold)
        ]
        persons = self._follow_size_candidates(persons, source="formal")
        raw_probe_persons = tuple(
            det
            for det in self.last_search_diagnostic_detections
            if int(det.class_id) == int(self.config.person_class_id)
            and float(det.score) < float(self.config.conf_threshold)
        )
        # Filter before clustering so tiny fragments cannot suppress a valid
        # candidate through cluster competition or grow into an eligible box.
        raw_probe_persons = tuple(self._follow_size_candidates(raw_probe_persons, source="probe"))
        if bool(getattr(self.config, "search_probe_cluster_enable", True)):
            probe_persons, cluster_diagnostics = cluster_probe_detections(
                raw_probe_persons,
                person_class_id=int(self.config.person_class_id),
                iou_threshold=float(
                    getattr(self.config, "search_probe_cluster_iou_threshold", 0.45)
                ),
                center_distance_ratio=float(
                    getattr(
                        self.config,
                        "search_probe_cluster_center_distance_ratio",
                        0.18,
                    )
                ),
                min_score_gap=float(
                    getattr(self.config, "search_probe_cluster_min_score_gap", 0.03)
                ),
            )
        else:
            probe_persons = raw_probe_persons
            cluster_diagnostics = ()
        self.last_search_probe_clusters = list(probe_persons)
        self.last_search_probe_cluster_diagnostics = tuple(cluster_diagnostics)
        self.last_search_candidate_evidence = SearchCandidateEvidence(
            formal_persons=tuple(persons),
            probe_persons=probe_persons,
            probe_cluster_diagnostics=tuple(cluster_diagnostics),
        )
        return persons

    def process_search_probe_frame(
        self,
        frame: Any,
        frame_format: Optional[str] = None,
    ) -> List[TrackRecord]:
        """Run detector-only recovery without advancing ReID or tracker state."""
        frame_start = time.perf_counter()
        arr, width, height, fmt = numpy_from_frame(frame, frame_format)
        self.last_frame_width = width
        self.last_frame_height = height
        self.last_predicted_reid_verifications = []
        packet = FramePacket(arr, width=width, height=height, format=fmt)

        detections = self.detector.detect(packet, fmt)
        detect_end = time.perf_counter()
        persons = self._record_detector_output(detections)
        yolo_timing = self.detector.last_timing_ms
        frame_wrap_ms = _elapsed_ms(frame_start, detect_end) - float(
            yolo_timing.get("total", 0.0)
        )
        self.last_timing_ms = {
            "frame_wrap": max(0.0, frame_wrap_ms),
            "yolo_preprocess": float(yolo_timing.get("preprocess", 0.0)),
            "yolo_inference": float(yolo_timing.get("inference", 0.0)),
            "yolo_decode": float(yolo_timing.get("decode", 0.0)),
            "yolo_nms": float(yolo_timing.get("nms", 0.0)),
            "yolo_postprocess": float(yolo_timing.get("postprocess", 0.0)),
            "yolo_total": float(yolo_timing.get("total", 0.0)),
            "reid_preprocess": 0.0,
            "reid_inference": 0.0,
            "reid_partial_inference": 0.0,
            "reid_postprocess": 0.0,
            "reid_total": 0.0,
            "reid_detections": 0.0,
            "reid_features": 0.0,
            "reid_verify_preprocess": 0.0,
            "reid_verify_inference": 0.0,
            "reid_verify_postprocess": 0.0,
            "reid_verify_total": 0.0,
            "reid_verify_detections": 0.0,
            "reid_verify_features": 0.0,
            "tracker": 0.0,
            "predicted_reid_verify": 0.0,
            "total": _elapsed_ms(frame_start, detect_end),
        }
        if self.logger is not None:
            self.logger.debug(
                "rknn detector-only recovery frame processed "
                "width=%d height=%d detections=%d persons=%d",
                width,
                height,
                len(detections),
                len(persons),
            )
        return []

    def process_frame(self, frame: Any, frame_format: Optional[str] = None) -> List[TrackRecord]:
        frame_start = time.perf_counter()
        arr, width, height, fmt = numpy_from_frame(frame, frame_format)
        self.last_frame_width = width
        self.last_frame_height = height
        self.last_predicted_reid_verifications = []
        packet = FramePacket(arr, width=width, height=height, format=fmt)

        detections = self.detector.detect(packet, fmt)
        detect_end = time.perf_counter()
        persons = self._record_detector_output(detections)
        features = self.reid.extract(packet, persons, fmt)
        partial_features = list(getattr(self.reid, "last_partial_features", ()))
        if len(partial_features) != len(persons):
            partial_features = [None for _ in persons]
        partial_feature_sources = list(
            getattr(self.reid, "last_partial_feature_sources", ())
        )
        if len(partial_feature_sources) != len(persons):
            partial_feature_sources = [None for _ in persons]
        reid_end = time.perf_counter()
        records = self.tracker.update(
            persons,
            features,
            partial_features=partial_features,
            partial_feature_sources=partial_feature_sources,
            image_width=width, image_height=height,
            frame_context=self._frame_context,
        )
        tracker_end = time.perf_counter()
        yolo_timing = self.detector.last_timing_ms
        reid_timing = dict(self.reid.last_timing_ms)
        verify_timing = None
        verify_start = tracker_end
        if (
            self.config.predicted_reid_verify_enable
            and self.config.identity_bank_enable
            and self.config.reid_enable
            and records
        ):
            records, verify_timing = self._verify_predicted_records(packet, records, fmt)
        if self.config.identity_suppress_duplicate_uids and records:
            records = self._suppress_duplicate_reid_uids(records)
        diagnostics_start = time.perf_counter()
        self._record_reid_diagnostics(packet, fmt)
        diagnostics_ms = _elapsed_ms(diagnostics_start, time.perf_counter())
        verify_end = time.perf_counter()
        combined_reid_timing = dict(reid_timing)
        if verify_timing is not None:
            for key in (
                "preprocess", "inference", "partial_inference", "postprocess",
                "total", "detections", "features",
            ):
                combined_reid_timing[key] = float(combined_reid_timing.get(key, 0.0)) + float(verify_timing.get(key, 0.0))
        frame_wrap_ms = _elapsed_ms(frame_start, detect_end) - float(yolo_timing.get("total", 0.0))
        self.last_timing_ms = {
            "frame_wrap": max(0.0, frame_wrap_ms),
            "yolo_preprocess": float(yolo_timing.get("preprocess", 0.0)),
            "yolo_inference": float(yolo_timing.get("inference", 0.0)),
            "yolo_decode": float(yolo_timing.get("decode", 0.0)),
            "yolo_nms": float(yolo_timing.get("nms", 0.0)),
            "yolo_postprocess": float(yolo_timing.get("postprocess", 0.0)),
            "yolo_total": float(yolo_timing.get("total", 0.0)),
            "reid_preprocess": float(combined_reid_timing.get("preprocess", 0.0)),
            "reid_inference": float(combined_reid_timing.get("inference", 0.0)),
            "reid_partial_inference": float(combined_reid_timing.get("partial_inference", 0.0)),
            "reid_postprocess": float(combined_reid_timing.get("postprocess", 0.0)),
            "reid_total": float(combined_reid_timing.get("total", 0.0)),
            "reid_detections": float(combined_reid_timing.get("detections", 0.0)),
            "reid_features": float(combined_reid_timing.get("features", 0.0)),
            "reid_verify_preprocess": float((verify_timing or {}).get("preprocess", 0.0)),
            "reid_verify_inference": float((verify_timing or {}).get("inference", 0.0)),
            "reid_verify_postprocess": float((verify_timing or {}).get("postprocess", 0.0)),
            "reid_verify_total": float((verify_timing or {}).get("total", 0.0)),
            "reid_verify_detections": float((verify_timing or {}).get("detections", 0.0)),
            "reid_verify_features": float((verify_timing or {}).get("features", 0.0)),
            "tracker": _elapsed_ms(reid_end, tracker_end),
            "reid_diagnostics": diagnostics_ms,
            "predicted_reid_verify": _elapsed_ms(verify_start, verify_end) if verify_timing is not None else 0.0,
            "total": _elapsed_ms(frame_start, verify_end),
        }
        if self.logger is not None:
            self.logger.debug(
                "rknn vision frame processed width=%d height=%d detections=%d persons=%d tracks=%d",
                width,
                height,
                len(detections),
                len(persons),
                len(records),
            )
        return records

    def set_frame_context(
        self,
        *,
        control_frame_id: int,
        capture_frame_id: int,
        capture_timestamp: float,
        yaw_rate_dps: Optional[float] = None,
        integrated_yaw_deg: Optional[float] = None,
    ) -> None:
        self._frame_context = {
            "control_frame_id": int(control_frame_id),
            "capture_frame_id": int(capture_frame_id),
            "capture_timestamp": float(capture_timestamp),
        }
        if yaw_rate_dps is not None:
            self._frame_context["yaw_rate_dps"] = float(yaw_rate_dps)
        if integrated_yaw_deg is not None:
            self._frame_context["integrated_yaw_deg"] = float(integrated_yaw_deg)

    def _record_reid_diagnostics(self, frame: Any, frame_format: str) -> None:
        writer = self._reid_diagnostics
        if writer is None:
            return
        # The tracker carries detector indices through association and NMS;
        # never infer the ReID crop from the expanded, filtered display box.
        for observation in self.tracker.last_identity_observations:
            assignment = observation["assignment"]
            reason = str(assignment.get("reason", ""))
            ordinary = reason in ("mapped", "skip_update_redundant", "skip_update_distance")
            if (
                ordinary
                and not assignment.get("bank_updated", False)
                and observation["frame_index"] % self.config.reid_diagnostics_mapped_interval != 0
            ):
                continue
            metadata = dict(observation)
            metadata.update(self._frame_context)
            metadata.setdefault("control_frame_id", observation["frame_index"])
            evidence = assignment.get("match_evidence") or {}
            winner_metadata = (evidence.get("winner") or {}).get("metadata") or {}
            anchor_metadata = evidence.get("anchor_metadata") or {}
            metadata["matched_template_sample_path"] = writer.sample_path(
                winner_metadata.get("control_frame_id", winner_metadata.get("frame_index")),
                winner_metadata.get("track_id"),
            )
            metadata["anchor_sample_path"] = writer.sample_path(
                anchor_metadata.get("control_frame_id", anchor_metadata.get("frame_index")),
                anchor_metadata.get("track_id"),
            )
            writer.submit(frame, observation["detector_bbox"], metadata, frame_format=frame_format)

    def set_identity_reacquire_context(
        self,
        *,
        active_uid: Optional[int],
        searching: bool,
        direction: Optional[str],
    ) -> None:
        self.tracker.set_search_reacquire_context(
            active_uid=active_uid,
            searching=searching,
            direction=direction,
        )

    def set_search_diagnostic_mode(self, enabled: bool) -> None:
        """Enable passive low-confidence output without touching tracker state."""
        self.detector.set_search_diagnostic_active(bool(enabled))
        if not enabled:
            self.last_search_candidate_evidence = SearchCandidateEvidence()
            self.last_search_probe_clusters = []
            self.last_search_probe_cluster_diagnostics = ()

    def get_search_candidate_evidence(self) -> SearchCandidateEvidence:
        """Return the latest immutable detector evidence for the control gate."""
        return self.last_search_candidate_evidence

    def set_search_reacquire_context(
        self,
        *,
        active_uid: Optional[int],
        searching: bool,
        direction: Optional[str],
    ) -> None:
        """Compatibility alias for identity reacquisition only."""
        self.set_identity_reacquire_context(
            active_uid=active_uid,
            searching=searching,
            direction=direction,
        )

    def _suppress_duplicate_reid_uids(self, records: List[TrackRecord]) -> List[TrackRecord]:
        counts = {}
        for rec in records:
            uid = int(rec.reid_uid)
            if uid > 0:
                counts[uid] = counts.get(uid, 0) + 1
        duplicate_uids = {uid for uid, count in counts.items() if count > 1}
        if not duplicate_uids:
            return records

        if self.logger is not None:
            self.logger.info(
                "identity_bank duplicate uid suppression: uids=%s tracks=%s",
                sorted(duplicate_uids),
                [
                    {"track_id": int(rec.track_id), "reid_uid": int(rec.reid_uid)}
                    for rec in records
                    if int(rec.reid_uid) in duplicate_uids
                ],
            )
        return [
            replace(rec, reid_uid=0) if int(rec.reid_uid) in duplicate_uids else rec
            for rec in records
        ]

    def _verify_predicted_records(self, packet: FramePacket, records: List[TrackRecord], frame_format: str):
        predicted = [
            rec for rec in records
            if int(rec.time_since_update) > 0 and int(rec.reid_uid) > 0
        ]
        if not predicted:
            return records, None

        detections = [
            Detection(
                bbox=(float(rec.x1), float(rec.y1), float(rec.x2), float(rec.y2)),
                score=float(rec.score),
                class_id=int(rec.class_id),
            )
            for rec in predicted
        ]
        try:
            features = self.reid.extract(
                packet, detections, frame_format, compute_partial=False
            )
        except TypeError:
            # Test/detector adapters predating the optional keyword still
            # expose the original three-argument extractor API.
            features = self.reid.extract(packet, detections, frame_format)
        threshold = float(self.config.predicted_reid_verify_threshold)
        verified_by_track = {}
        verification_items = []
        for rec, feature in zip(predicted, features):
            distance = self.tracker.reid_distance_to_uid(int(rec.reid_uid), feature)
            passed = distance is not None and float(distance) <= threshold
            verified_by_track[int(rec.track_id)] = (distance, passed)
            verification_items.append(
                {
                    "track_id": int(rec.track_id),
                    "reid_uid": int(rec.reid_uid),
                    "time_since_update": int(rec.time_since_update),
                    "distance": None if distance is None else float(distance),
                    "threshold": float(threshold),
                    "passed": bool(passed),
                    "bbox": [float(rec.x1), float(rec.y1), float(rec.x2), float(rec.y2)],
                }
            )
        self.last_predicted_reid_verifications = verification_items

        filtered: List[TrackRecord] = []
        for rec in records:
            decision = verified_by_track.get(int(rec.track_id))
            if decision is None:
                filtered.append(rec)
                continue
            distance, passed = decision
            if passed:
                filtered.append(
                    replace(
                        rec,
                        reid_verify_distance=float(distance) if distance is not None else None,
                        reid_verify_passed=True,
                    )
                )
        return self._suppress_duplicate_predicted_records(filtered), dict(self.reid.last_timing_ms)

    def _suppress_duplicate_predicted_records(self, records: List[TrackRecord]) -> List[TrackRecord]:
        updated = [rec for rec in records if int(rec.time_since_update) <= 0]
        if not updated:
            return records

        iou_threshold = float(self.config.predicted_reid_duplicate_iou_threshold)
        overlap_threshold = float(self.config.predicted_reid_duplicate_overlap_threshold)
        filtered: List[TrackRecord] = []
        for rec in records:
            if int(rec.time_since_update) <= 0:
                filtered.append(rec)
                continue
            duplicate = any(
                int(rec.class_id) == int(other.class_id)
                and _is_predicted_duplicate(rec, other, iou_threshold, overlap_threshold)
                for other in updated
            )
            if not duplicate:
                filtered.append(rec)
        return filtered

    def reset_tracker(self) -> None:
        self.tracker.reset()

    def debug_state(self) -> List[dict]:
        return self.tracker.debug_state()

    def load(self) -> None:
        self.detector.load()
        self.reid.load()

    def close(self) -> None:
        if self._reid_diagnostics is not None:
            self._reid_diagnostics.close()
            if self.logger is not None:
                self.logger.info(
                    "reid_diagnostics_summary written=%d dropped=%d failed=%d dir=%s",
                    self._reid_diagnostics.written_samples,
                    self._reid_diagnostics.dropped_samples,
                    self._reid_diagnostics.failed_samples,
                    self.config.reid_diagnostics_dir,
                )
        self.detector.release()
        self.reid.release()


def _elapsed_ms(start: float, end: float) -> float:
    return max(0.0, (end - start) * 1000.0)


def _is_predicted_duplicate(
    predicted: TrackRecord,
    updated: TrackRecord,
    iou_threshold: float,
    overlap_threshold: float,
) -> bool:
    pred_box = (predicted.x1, predicted.y1, predicted.x2, predicted.y2)
    updated_box = (updated.x1, updated.y1, updated.x2, updated.y2)
    intersection = _intersection_area(pred_box, updated_box)
    if intersection <= 0.0:
        return False
    pred_area = max(0.0, float(predicted.x2) - float(predicted.x1)) * max(
        0.0,
        float(predicted.y2) - float(predicted.y1),
    )
    updated_area = max(0.0, float(updated.x2) - float(updated.x1)) * max(
        0.0,
        float(updated.y2) - float(updated.y1),
    )
    union = pred_area + updated_area - intersection
    iou = intersection / max(union, 1e-6)
    predicted_overlap = intersection / max(pred_area, 1e-6)
    center_inside = (
        float(updated.x1) <= float(predicted.cx) <= float(updated.x2)
        and float(updated.y1) <= float(predicted.cy) <= float(updated.y2)
    )
    return iou >= iou_threshold or predicted_overlap >= overlap_threshold or center_inside


def _intersection_area(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
