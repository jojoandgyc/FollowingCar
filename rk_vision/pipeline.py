from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from typing import Any, List, Optional, Tuple

from .frames import FramePacket, numpy_from_frame
from .reid import OSNetConfig, OSNetRKNNExtractor
from .tracker import DeepSortTracker, DeepSortTrackerConfig, TrackRecord
from .yolo11 import Detection, YOLO11Config, YOLO11RKNNDetector


@dataclass(frozen=True)
class SearchCandidateEvidence:
    """Read-only detector evidence; it carries no target or motion decision."""

    formal_persons: Tuple[Detection, ...] = ()
    probe_persons: Tuple[Detection, ...] = ()


@dataclass(frozen=True)
class RKNNVisionConfig:
    yolo_model_path: str
    reid_model_path: str = ""
    reid_enable: bool = True
    person_class_id: int = 0
    conf_threshold: float = 0.25
    search_diagnostic_conf_threshold: float = 0.10
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
                1, int(os.environ.get("Y8_IDENTITY_CONTROLLED_HANDOFF_CONFIRM_FRAMES", "3"))
            ),
            identity_controlled_handoff_threshold=float(
                os.environ.get("Y8_IDENTITY_CONTROLLED_HANDOFF_THRESHOLD", "0.30")
            ),
            identity_controlled_handoff_min_old_track_gap_frames=max(
                1, int(os.environ.get("Y8_IDENTITY_CONTROLLED_HANDOFF_MIN_OLD_TRACK_GAP_FRAMES", "2"))
            ),
            identity_preferred_search_reacquire_enable=os.environ.get(
                "Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_ENABLE",
                "1",
            ).strip()
            != "0",
            identity_preferred_search_reacquire_threshold=float(
                os.environ.get("Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_THRESHOLD", "0.36")
            ),
            identity_preferred_search_reacquire_max_disadvantage=float(
                os.environ.get("Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_MAX_DISADVANTAGE", "0.15")
            ),
            identity_preferred_search_reacquire_confirm_frames=max(
                1,
                int(os.environ.get("Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_CONFIRM_FRAMES", "2")),
            ),
            identity_preferred_search_reacquire_instant_threshold=float(
                os.environ.get(
                    "Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_INSTANT_THRESHOLD",
                    "0.28",
                )
            ),
            identity_preferred_search_reacquire_side_ratio=float(
                os.environ.get("Y8_IDENTITY_PREFERRED_SEARCH_REACQUIRE_SIDE_RATIO", "0.05")
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
                identity_controlled_handoff_threshold=self.config.identity_controlled_handoff_threshold,
                identity_controlled_handoff_min_old_track_gap_frames=self.config.identity_controlled_handoff_min_old_track_gap_frames,
                identity_preferred_search_reacquire_enable=self.config.identity_preferred_search_reacquire_enable,
                identity_preferred_search_reacquire_threshold=self.config.identity_preferred_search_reacquire_threshold,
                identity_preferred_search_reacquire_max_disadvantage=self.config.identity_preferred_search_reacquire_max_disadvantage,
                identity_preferred_search_reacquire_confirm_frames=self.config.identity_preferred_search_reacquire_confirm_frames,
                identity_preferred_search_reacquire_instant_threshold=self.config.identity_preferred_search_reacquire_instant_threshold,
                identity_preferred_search_reacquire_side_ratio=self.config.identity_preferred_search_reacquire_side_ratio,
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
        self.last_search_diagnostic_detections: List[Detection] = []
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
            "reid_postprocess": 0.0,
            "reid_total": 0.0,
            "tracker": 0.0,
            "total": 0.0,
        }

    def _record_detector_output(self, detections: List[Detection]) -> List[Detection]:
        self.last_detections = detections
        self.last_search_diagnostic_detections = list(
            self.detector.last_search_diagnostic_detections
        )
        persons = [
            det
            for det in detections
            if int(det.class_id) == int(self.config.person_class_id)
            and float(det.score) >= float(self.config.conf_threshold)
        ]
        probe_persons = tuple(
            det
            for det in self.last_search_diagnostic_detections
            if int(det.class_id) == int(self.config.person_class_id)
            and float(det.score) < float(self.config.conf_threshold)
        )
        self.last_search_candidate_evidence = SearchCandidateEvidence(
            formal_persons=tuple(persons),
            probe_persons=probe_persons,
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
        reid_end = time.perf_counter()
        records = self.tracker.update(persons, features, image_width=width, image_height=height)
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
        verify_end = time.perf_counter()
        combined_reid_timing = dict(reid_timing)
        if verify_timing is not None:
            for key in ("preprocess", "inference", "postprocess", "total", "detections", "features"):
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
