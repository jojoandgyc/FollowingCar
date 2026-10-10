from __future__ import annotations

import os
import math
import time
from .follow_bbox_policy import lower_compact_bbox_reason
from dataclasses import dataclass, replace
from typing import Any, List, Optional, Sequence, Tuple

from .deepsort.linear_assignment import prepare_assignment_backend
from .frames import FramePacket, numpy_from_frame
from .reid import OSNetConfig, OSNetRKNNExtractor, _color_signature
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
    # Only stable, already verified bindings may use detector-only geometry.
    # The board config opts in; legacy callers keep the full pipeline.
    detector_continuation_enable: bool = False
    # Diagnostic only, fed at processed-frame cadence; no camera-loop hook.
    lk_shadow_enable: bool = False
    lk_shadow_width: int = 320
    lk_shadow_correction_interval_sec: float = 0.30
    lk_shadow_max_gap_sec: float = 0.25
    lk_shadow_max_seed_age_sec: float = 0.75
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
            identity_similar_follow_enable=os.environ.get(
                "Y8_IDENTITY_SIMILAR_FOLLOW_ENABLE", "0").strip().lower() in {"1", "true", "yes"},
            identity_similar_follow_entry_threshold=float(os.environ.get(
                "Y8_IDENTITY_SIMILAR_FOLLOW_ENTRY_THRESHOLD", "0.50")),
            identity_similar_follow_retain_threshold=float(os.environ.get(
                "Y8_IDENTITY_SIMILAR_FOLLOW_RETAIN_THRESHOLD", "0.55")),
            identity_similar_follow_max_gap_sec=float(os.environ.get(
                "Y8_IDENTITY_SIMILAR_FOLLOW_MAX_GAP_SEC", "0.50")),
            yolo_model_path=os.environ.get("VISION_MODEL_PATH", "models/yolo11s.rknn").strip(),
            reid_model_path=os.environ.get("VISION_REID_MODEL_PATH", "models/deepsort.rknn").strip(),
            reid_enable=os.environ.get("VISION_REID_ENABLE", "1").strip() != "0",
            detector_continuation_enable=os.environ.get(
                "Y8_DETECTOR_CONTINUATION_ENABLE", "0").strip().lower() in {"1", "true", "yes"},
            lk_shadow_enable=os.environ.get(
                "Y8_LK_SHADOW_ENABLE", "0").strip().lower() in {"1", "true", "yes", "on"},
            lk_shadow_width=int(os.environ.get("Y8_LK_SHADOW_WIDTH", "320")),
            lk_shadow_correction_interval_sec=float(os.environ.get(
                "Y8_LK_SHADOW_CORRECTION_INTERVAL_SEC", "0.30")),
            lk_shadow_max_gap_sec=float(os.environ.get("Y8_LK_SHADOW_MAX_GAP_SEC", "0.25")),
            lk_shadow_max_seed_age_sec=float(os.environ.get("Y8_LK_SHADOW_MAX_SEED_AGE_SEC", "0.75")),
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
            identity_template_memory_enable=os.environ.get("Y8_IDENTITY_TEMPLATE_MEMORY_ENABLE", "0").strip() == "1",
            identity_template_crosscheck_enable=os.environ.get("Y8_IDENTITY_TEMPLATE_CROSSCHECK_ENABLE", "0").strip() == "1",
            identity_template_learning_guard_enable=os.environ.get("Y8_IDENTITY_TEMPLATE_LEARNING_GUARD_ENABLE", "0").strip() == "1",
            identity_appearance_region_safety_enable=os.environ.get("Y8_IDENTITY_APPEARANCE_REGION_SAFETY_ENABLE", "0").strip() == "1",
            identity_template_recent_sec=max(1.0, float(os.environ.get("Y8_IDENTITY_TEMPLATE_RECENT_SEC", "30"))),
            identity_template_archive_sec=max(1.0, float(os.environ.get("Y8_IDENTITY_TEMPLATE_ARCHIVE_SEC", "120"))),
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
            identity_partial_confirm_threshold=float(
                os.environ.get("Y8_IDENTITY_PARTIAL_CONFIRM_THRESHOLD", "0.34")
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
        # The runtime constructs this pipeline before starting frame/control
        # work and does not necessarily call load(). Prepare the optional CPU
        # solver here, not on the first multi-person frame during motion.
        assignment_start = time.perf_counter()
        self.assignment_backend = prepare_assignment_backend()
        self.assignment_prepare_ms = _elapsed_ms(assignment_start, time.perf_counter())
        if self.logger is not None:
            self.logger.info(
                "deepsort_assignment_prepared backend=%s elapsed_ms=%.3f",
                self.assignment_backend,
                self.assignment_prepare_ms,
            )
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
                identity_template_memory_enable=self.config.identity_template_memory_enable,
                identity_template_crosscheck_enable=self.config.identity_template_crosscheck_enable,
                identity_template_learning_guard_enable=self.config.identity_template_learning_guard_enable,
                identity_similar_follow_enable=self.config.identity_similar_follow_enable,
                identity_similar_follow_entry_threshold=self.config.identity_similar_follow_entry_threshold,
                identity_similar_follow_retain_threshold=self.config.identity_similar_follow_retain_threshold,
                identity_similar_follow_max_gap_sec=self.config.identity_similar_follow_max_gap_sec,
                identity_appearance_region_safety_enable=self.config.identity_appearance_region_safety_enable,
                identity_template_recent_sec=self.config.identity_template_recent_sec,
                identity_template_archive_sec=self.config.identity_template_archive_sec,
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
                identity_partial_confirm_threshold=self.config.identity_partial_confirm_threshold,
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
        self._detector_continuation_active_uid = None
        self._detector_continuation_allowed = False
        self._lk_shadow_worker = None
        self._lk_shadow_closed = False
        self._lk_shadow_failed = False
        self._lk_shadow_last_seed = None
        self._lk_shadow_logged_capture = None
        self.last_lk_shadow_result = None
        self.last_lk_shadow_diagnostic = None
        self.last_identity_processing = {"mode": "full", "reason": "startup"}
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
            shape_reason = lower_compact_bbox_reason(
                detection.bbox, getattr(self, "last_frame_width", None),
                getattr(self, "last_frame_height", None),
            )
            if shape_reason:
                reasons.append(shape_reason)
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
        # A rejected small person is still a competing detector hypothesis.
        # Do not turn a multi-person image into a singleton fast-path proof.
        self.last_raw_person_count = len(persons)
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
        probe_persons = tuple(self._follow_size_candidates(probe_persons, source="probe_cluster"))
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
        self._reset_identity_processing(mode="probe", reason="search_observation_only")
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
        self._submit_lk_shadow(arr, fmt, (), allow_seed=False)
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
        # Reset before frame conversion or inference: any exception must not
        # expose the previous capture's completed empty-detection contract.
        self._reset_identity_processing(mode="full", reason="processing_pending")
        frame_start = time.perf_counter()
        arr, width, height, fmt = numpy_from_frame(frame, frame_format)
        self.last_frame_width = width
        self.last_frame_height = height
        self.last_predicted_reid_verifications = []
        packet = FramePacket(arr, width=width, height=height, format=fmt)

        detections = self.detector.detect(packet, fmt)
        detect_end = time.perf_counter()
        persons = self._record_detector_output(detections)
        # Count hypotheses before size/appearance filtering. Low-score person
        # diagnostics are not identity evidence, but must not turn a frame with
        # a possible person into a certified empty one. Formal detections are
        # already included, so count only sub-threshold diagnostics here.
        detector_person_count = sum(
            int(det.class_id) == int(self.config.person_class_id) for det in detections
        ) + sum(
            int(det.class_id) == int(self.config.person_class_id)
            and float(det.score) < float(self.config.conf_threshold)
            for det in self.last_search_diagnostic_detections
        )
        check_start = time.perf_counter()
        records = self._try_detected_continuation(packet, persons, fmt)
        fast_check_ms = _elapsed_ms(check_start, time.perf_counter())
        self.last_identity_processing.update(
            capture_frame_id=self._frame_context.get("capture_frame_id"),
            capture_timestamp=self._frame_context.get("capture_timestamp"))
        detector_only = records is not None
        if not detector_only:
            # No tracker/bank state has been advanced by a failed fast plan.
            # Process this same physical frame once through the normal path.
            features = self.reid.extract(packet, persons, fmt)
            self.last_identity_processing["full_features_current"] = bool(
                persons and len(features) == len(persons) and all(f is not None for f in features))
            partial_features = list(getattr(self.reid, "last_partial_features", ()))
            if len(partial_features) != len(persons):
                partial_features = [None for _ in persons]
            partial_feature_sources = list(getattr(self.reid, "last_partial_feature_sources", ()))
            if len(partial_feature_sources) != len(persons):
                partial_feature_sources = [None for _ in persons]
            color_features = getattr(self.reid, "last_color_features", None)
            if color_features is not None and len(color_features) != len(persons):
                color_features = None
            reid_end = time.perf_counter()
            color_kwargs = {} if color_features is None else {"color_features": color_features}
            if getattr(self.config, "identity_template_learning_guard_enable", False):
                color_kwargs["learning_detections"] = detections
            records = self.tracker.update(
                persons, features, partial_features=partial_features,
                partial_feature_sources=partial_feature_sources,
                image_width=width, image_height=height,
                frame_context=self._frame_context, **color_kwargs,
            )
            reid_timing = dict(self.reid.last_timing_ms)
        else:
            reid_end = time.perf_counter()
            # Never expose the last full frame's embedding timings/features as
            # fresh evidence. The detector path carries its own provenance.
            reid_timing = {}
        tracker_end = time.perf_counter()
        yolo_timing = self.detector.last_timing_ms
        verify_timing = None
        verify_start = tracker_end
        if (
            self.config.predicted_reid_verify_enable
            and not detector_only
            and self.config.identity_bank_enable
            and self.config.reid_enable
            and records
        ):
            records, verify_timing = self._verify_predicted_records(packet, records, fmt)
        if self.config.identity_suppress_duplicate_uids and records:
            records = self._suppress_duplicate_reid_uids(records)
        if (not detector_only and getattr(self.config, "detector_continuation_enable", False)
                and callable(getattr(self.tracker, "note_full_identity_verification", None))):
            self.tracker.note_full_identity_verification(
                records, detections=persons, color_features=color_features,
                frame_context=self._frame_context, image_width=width, image_height=height,
                now=time.monotonic(), active_uid=(
                    getattr(self, "_detector_continuation_active_uid", None)
                    if getattr(self, "_detector_continuation_allowed", False) else None),
                raw_candidate_count=self.last_raw_person_count,
            )
        diagnostics_start = time.perf_counter()
        if not detector_only:
            self._record_reid_diagnostics(packet, fmt)
        diagnostics_ms = _elapsed_ms(diagnostics_start, time.perf_counter())
        verify_end = time.perf_counter()
        combined_reid_timing = dict(reid_timing)
        if verify_timing is not None:
            for key in (
                "preprocess", "inference", "partial_inference", "postprocess",
                "total", "detections", "features",
                "color", "postprocess_exclusive",
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
            "reid_color": float(combined_reid_timing.get("color", 0.0)),
            "reid_postprocess_exclusive": float(combined_reid_timing.get("postprocess_exclusive", 0.0)),
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
            "detector_continuation_check": fast_check_ms,
            "detector_continuation_used": float(detector_only),
            "reid_diagnostics": diagnostics_ms,
            "predicted_reid_verify": _elapsed_ms(verify_start, verify_end) if verify_timing is not None else 0.0,
            "total": _elapsed_ms(frame_start, verify_end),
        }
        for key, value in ({} if detector_only else getattr(self.tracker, "last_timing_ms", {})).items():
            self.last_timing_ms["tracker_" + key] = value
        inner_tracker = getattr(getattr(self.tracker, "deepsort", None), "tracker", None)
        for key, value in ({} if detector_only else getattr(inner_tracker, "last_timing_ms", {})).items():
            self.last_timing_ms["deepsort_" + key] = value
        bank = getattr(self.tracker, "identity_bank", None)
        if not detector_only and getattr(bank, "_assign_timing_frame", None) == getattr(self.tracker, "_frame_index", -1):
            for key, value in getattr(bank, "last_assign_timing_ms", {}).items():
                self.last_timing_ms["identity_" + key] = value
        self._submit_lk_shadow(arr, fmt, records)
        if self.logger is not None:
            if getattr(self.config, "detector_continuation_enable", False):
                processing = self.last_identity_processing
                obs = getattr(self.tracker, "last_identity_observations", ())
                current = [item.get("assignment", {}) for item in obs
                           if item.get("uid") == getattr(self, "_detector_continuation_active_uid", None)]
                assignment = current[0] if len(current) == 1 else {}
                self.logger.info(
                    "identity_processing capture_frame_id=%s mode=%s reason=%s "
                    "verified_capture=%s identity_valid_until=%s fast_check_ms=%.2f "
                    "reid_ms=%.2f total_ms=%.2f next_proof_reason=%s known_background_count=%d "
                    "continuation_permission=%s",
                    self._frame_context.get("capture_frame_id"), processing["mode"],
                    processing["reason"], assignment.get("identity_verified_capture"),
                    assignment.get("identity_valid_until"), fast_check_ms,
                    self.last_timing_ms["reid_total"], self.last_timing_ms["total"],
                    getattr(self.tracker, "last_detector_continuation_reason", "unavailable"),
                    len(getattr(getattr(self.tracker, "_detector_proof", None), "backgrounds", ())),
                    processing.get("permission", "full"),
                )
            self.logger.debug(
                "rknn vision frame processed width=%d height=%d detections=%d persons=%d tracks=%d",
                width,
                height,
                len(detections),
                len(persons),
                len(records),
            )
        # This reports only successful current-frame detection processing, not
        # an identity proof. Empty frames keep full_features_current=False;
        # consumers must also match both capture fields and bound old proof TTL.
        # Publish last so tracker/ReID/diagnostic failures remain incomplete.
        self.last_identity_processing.update(
            detector_result_complete=True,
            detector_person_count=detector_person_count,
        )
        return records

    def _lk_shadow_seed(self, records, capture_id, timestamp):
        """Resolve a detector crop on this exact image, without changing identity."""
        from .lk_shadow import LKShadowSeed

        active_uid = getattr(self, "_detector_continuation_active_uid", None)
        current = [r for r in records if int(r.reid_uid) > 0
                   and r.time_since_update == 0 and r.class_id == self.config.person_class_id
                   and (active_uid is None or int(r.reid_uid) == int(active_uid))]
        if len(current) != 1:
            return None
        record = current[0]
        uid, raw = int(record.reid_uid), int(record.track_id)
        observations = []
        for observation in getattr(self.tracker, "last_identity_observations", ()):
            metadata = observation.get("sample_metadata") or {}
            assignment = observation.get("assignment") or {}
            if (observation.get("uid") == uid and observation.get("raw_track_id") == raw
                    and metadata.get("capture_frame_id") == capture_id
                    and metadata.get("capture_timestamp") == timestamp
                    and metadata.get("is_fresh") is True
                    and assignment.get("uid") == uid
                    and not assignment.get("identity_control_rejected")
                    and not assignment.get("search_excluded")
                    and not assignment.get("search_contradiction_retained")
                    and assignment.get("bbox_quality_ok") is not False
                    and assignment.get("reacquire_geometry_ok") is not False
                    and (assignment.get("identity_competition") or {}).get("passed") is not False
                    and observation.get("detector_bbox") is not None):
                observations.append(observation)
        if len(observations) != 1:
            return None
        previous = getattr(self, "_lk_shadow_last_seed", None)
        if (previous is not None and previous[:2] == (uid, raw)
                and timestamp - previous[2] < self.config.lk_shadow_correction_interval_sec):
            return None
        return LKShadowSeed(uid, raw, capture_id, timestamp, observations[0]["detector_bbox"])

    def _submit_lk_shadow(self, arr, frame_format, records, *, allow_seed=True):
        """Submit once after real processing; never replace a TrackRecord.

        This first trial runs only at process_frame/probe cadence. It does not
        see intervening camera captures or replay delayed YOLO keyframes, and
        makes no camera-FPS claim. Corrections stay on their own raw images.
        """
        if (not getattr(self.config, "lk_shadow_enable", False)
                or getattr(self, "_lk_shadow_closed", False)
                or getattr(self, "_lk_shadow_failed", False)):
            return
        start = time.perf_counter()
        try:
            context = self._frame_context
            cap, timestamp = context.get("capture_frame_id"), context.get("capture_timestamp")
            if (isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0
                    or isinstance(timestamp, bool) or timestamp is None
                    or not math.isfinite(timestamp) or timestamp <= 0):
                self.last_lk_shadow_diagnostic = {"status": "not_submitted", "reason": "invalid_capture_context"}
                return
            from .lk_shadow import LKShadowConfig, LKShadowWorker
            worker = getattr(self, "_lk_shadow_worker", None)
            if worker is None:
                worker = LKShadowWorker(LKShadowConfig(
                    width=self.config.lk_shadow_width,
                    max_gap_sec=self.config.lk_shadow_max_gap_sec,
                    max_seed_age_sec=self.config.lk_shadow_max_seed_age_sec))
                self._lk_shadow_worker = worker
            # Poll already-computed immutable diagnostics; never wait on flow.
            result = worker.latest_result()
            if result is not None:
                self.last_lk_shadow_result = result
                if result.capture_id != getattr(self, "_lk_shadow_logged_capture", None):
                    self._lk_shadow_logged_capture = result.capture_id
                    if self.logger is not None:
                        import json
                        from dataclasses import asdict
                        payload = asdict(result)
                        payload.update(observed_at_capture_id=cap,
                            observed_capture_age_ms=(timestamp - result.capture_timestamp) * 1000.,
                            cadence="processed_frames_only", identity_authority=False,
                            motion_authority=False, worker=worker.stats())
                        self.logger.info("lk_shadow_result %s", json.dumps(payload, allow_nan=False))
            seed = self._lk_shadow_seed(records, cap, timestamp) if allow_seed else None
            submitted = worker.submit(arr, cap, timestamp, seed=seed, frame_format=frame_format)
            if submitted and seed is not None:
                self._lk_shadow_last_seed = (seed.uid, seed.raw_track_id, seed.capture_timestamp)
            self.last_lk_shadow_diagnostic = {
                "status": "submitted" if submitted else "not_submitted",
                "capture_id": cap, "capture_timestamp": timestamp,
                "seed_capture_id": seed.capture_id if seed else None,
                "cadence": "processed_frames_only", "worker": worker.stats(),
            }
        except Exception as exc:
            # Optional diagnostics must not become a vision failure/motor stop.
            self._lk_shadow_failed = True
            self.last_lk_shadow_diagnostic = {"status": "disabled_after_error", "reason": type(exc).__name__}
            try:
                if self.logger is not None:
                    self.logger.warning("lk_shadow_disabled reason=%s", type(exc).__name__)
            except Exception:
                pass
        finally:
            submit_ms = _elapsed_ms(start, time.perf_counter())
            self.last_timing_ms["lk_shadow_submit"] = submit_ms
            # Only this synchronous cost belongs to pipeline total. Worker
            # compute is reported by result.wall_ms, never as model inference.
            self.last_timing_ms["total"] = self.last_timing_ms.get("total", 0.) + submit_ms

    def set_detector_continuation_context(self, *, active_uid, allowed: bool) -> None:
        self._detector_continuation_active_uid = active_uid
        self._detector_continuation_allowed = bool(allowed)
        setter = getattr(self.tracker, "set_detector_continuation_context", None)
        if callable(setter):
            setter(active_uid=active_uid, allowed=allowed)

    def _reset_identity_processing(self, *, mode: str, reason: str) -> None:
        context = getattr(self, "_frame_context", {})
        self.last_identity_processing = {
            "mode": mode,
            "reason": reason,
            "full_features_current": False,
            "capture_frame_id": context.get("capture_frame_id"),
            "capture_timestamp": context.get("capture_timestamp"),
            "detector_result_complete": False,
            "detector_person_count": None,
            "detector_result_capture_frame_id": context.get("capture_frame_id"),
            "detector_result_capture_timestamp": context.get("capture_timestamp"),
        }

    def _try_detected_continuation(self, packet, persons, fmt):
        self.last_identity_processing.update(mode="full", reason="disabled")
        if not (getattr(self.config, "detector_continuation_enable", False)
                and self.config.reid_enable and self.config.identity_bank_enable):
            return None
        if not getattr(self, "_detector_continuation_allowed", False):
            self.last_identity_processing["reason"] = "control_context_requires_full"
            return None
        plan_fn = getattr(self.tracker, "plan_detected_continuation", None)
        if not callable(plan_fn):
            self.last_identity_processing["reason"] = "tracker_unsupported"
            return None
        plan = plan_fn(
            persons, image_width=self.last_frame_width, image_height=self.last_frame_height,
            frame_context=self._frame_context,
            active_uid=getattr(self, "_detector_continuation_active_uid", None),
            now=time.monotonic(), raw_candidate_count=self.last_raw_person_count,
            color_features=lambda: _detector_color_features(packet, persons, fmt),
        )
        records = None if plan is None else self.tracker.commit_detected_continuation(
            plan, now=time.monotonic())
        self.last_identity_processing.update(
            mode="detector_continuation" if records is not None else "full",
            reason=getattr(self.tracker, "last_detector_continuation_reason", "unavailable"),
            permission=plan.proof.permission if records is not None else "full",
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
                and not assignment.get("recent_bank_updated", False)
                and not assignment.get("learning_written_tiers")
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

        # A successful current handoff can coexist with an old predicted
        # record assembled earlier in this frame. The bank has already
        # revoked that raw-track claim; it is not a second current person.
        current_handoffs = {}
        for uid in duplicate_uids:
            owner = RKNNVisionPipeline._current_similar_handoff_owner(self, uid, records)
            if owner is not None:
                current_handoffs[uid] = owner

        if self.logger is not None:
            self.logger.info(
                "identity_bank duplicate uid suppression: uids=%s tracks=%s current_handoff_owners=%s",
                sorted(duplicate_uids),
                [
                    {"track_id": int(rec.track_id), "reid_uid": int(rec.reid_uid)}
                    for rec in records
                    if int(rec.reid_uid) in duplicate_uids
                ],
                current_handoffs,
            )
        return [
            replace(rec, reid_uid=0)
            if (int(rec.reid_uid) in duplicate_uids
                and current_handoffs.get(int(rec.reid_uid)) != int(rec.track_id)) else rec
            for rec in records
        ]

    def _current_similar_handoff_owner(self, uid, records):
        """Resolve only an accepted new capture versus revoked predictions.

        Never choose between two fresh claims, use a retained mapping as
        identity proof, or grant a UID which IdentityBank did not publish.
        """
        if not getattr(getattr(self, 'config', None), 'identity_similar_follow_enable', False):
            return None
        owned = [r for r in records if int(r.reid_uid) == uid]
        fresh = [r for r in owned if r.time_since_update == 0]
        if len(fresh) != 1 or fresh[0].class_id != self.config.person_class_id:
            return None
        current = fresh[0]
        bank = self.tracker.identity_bank
        if bank.track_to_uid.get(int(current.track_id)) != uid:
            return None
        others = [r for r in owned if r is not current]
        if (not others or any(r.time_since_update <= 0
                or int(r.track_id) == int(current.track_id)
                or bank.track_to_uid.get(int(r.track_id)) == uid for r in others)):
            return None
        context = self._frame_context
        cap, stamp = context.get('capture_frame_id'), context.get('capture_timestamp')
        try:
            valid_capture = (not isinstance(cap, bool) and int(cap) == cap and cap > 0
                             and not isinstance(stamp, bool) and math.isfinite(stamp) and stamp > 0)
        except (TypeError, ValueError, OverflowError):
            valid_capture = False
        if not valid_capture:
            return None
        observations = [o for o in self.tracker.last_identity_observations
            if o.get('raw_track_id') == int(current.track_id)
            and o.get('uid') == uid
            and (o.get('sample_metadata') or {}).get('capture_frame_id') == cap
            and (o.get('sample_metadata') or {}).get('capture_timestamp') == stamp
            and (o.get('sample_metadata') or {}).get('is_fresh') is True]
        if len(observations) != 1:
            return None
        observation = observations[0]
        for assignment in (observation.get('assignment') or {},
                           bank.last_assignments.get(int(current.track_id), {})):
            proof = assignment.get('similar_follow') or {}
            if (assignment.get('uid') != uid or assignment.get('match_source') != 'similar_follow'
                    or proof.get('status') != 'follow' or proof.get('capture_frame_id') != cap
                    or assignment.get('bbox_quality_ok') is not True
                    or assignment.get('identity_control_rejected')
                    or assignment.get('search_excluded')
                    or assignment.get('search_contradiction_retained')
                    or assignment.get('reacquire_geometry_ok') is False
                    or (assignment.get('reacquire_geometry') or {}).get('ok') is False
                    or (assignment.get('identity_competition') or {}).get('passed') is False):
                return None
        superseded = [int(r.track_id) for r in others]
        for assignment in (observation['assignment'], bank.last_assignments[int(current.track_id)]):
            assignment['superseded_predicted_track_ids'] = superseded
        return int(current.track_id)

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
        self._lk_shadow_closed = True
        worker = getattr(self, "_lk_shadow_worker", None)
        if worker is not None:
            try:
                stopped = worker.close()
                if not stopped and self.logger is not None:
                    self.logger.warning("lk_shadow_close worker_still_finishing=true")
                if self.logger is not None:
                    import json
                    self.logger.info("lk_shadow_summary %s", json.dumps(dict(
                        cadence="processed_frames_only", identity_authority=False,
                        motion_authority=False, complete=stopped, worker=worker.stats()),
                        allow_nan=False))
            except Exception as exc:
                try:
                    if self.logger is not None:
                        self.logger.warning("lk_shadow_close reason=%s", type(exc).__name__)
                except Exception:
                    pass
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


def _detector_color_features(frame, detections, frame_format):
    """Small CPU-only check, in the same BGR/crop convention as full ReID.

    No embedding extraction, gallery mutation, model call or old-feature reuse.
    Called lazily, only after the tracker's cheap continuation preconditions.
    """
    arr, width, height, fmt = numpy_from_frame(frame, frame_format)
    colors = []
    for detection in detections:
        x1, y1, x2, y2 = [int(round(float(v))) for v in detection.bbox]
        x1, y1 = max(0, min(width-1, x1)), max(0, min(height-1, y1))
        x2, y2 = max(0, min(width, x2)), max(0, min(height, y2))
        crop = arr[y1:y2, x1:x2]
        colors.append(_color_signature(crop[..., ::-1] if fmt == "RGB" else crop))
    return colors


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
