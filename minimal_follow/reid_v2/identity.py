"""State machine for an independently implemented one-target ReID profile."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .association import BBox, ReidCandidate, associate, iou
from .profile import Match, TargetProfile
from .worker import ReidRequest, ReidResult, ReidWorker


@dataclass(frozen=True)
class ReidConfig:
    enabled: bool = True
    stable_frames: int = 10
    enroll_interval_sec: float = 1.0
    reacquire_interval_sec: float = 0.20
    result_max_age_sec: float = 0.30
    # One high-quality template is enough to lock a first-session target;
    # later views are retained and make matching progressively more robust.
    min_full_templates: int = 1
    min_torso_templates: int = 2
    max_full_templates: int = 8
    max_torso_templates: int = 4
    full_threshold: float = 0.78
    torso_threshold: float = 0.84
    confirm_hits: int = 2
    confirm_window: int = 3
    min_iou: float = 0.12
    max_center_distance_ratio: float = 0.16
    min_confidence: float = 0.65
    min_height_px: float = 120.0
    min_area_px: float = 8000.0
    edge_margin_ratio: float = 0.015


@dataclass(frozen=True)
class ReidDecision:
    accepted: bool
    candidate: Optional[ReidCandidate]
    state: str
    reason: str
    match_score: Optional[float]
    match_source: Optional[str]
    result_age_ms: Optional[float]
    worker_timing_ms: Optional[dict]
    full_templates: int = 0
    torso_templates: int = 0


class ReidPolicy:
    """Target lock + asynchronous ReID. The control thread never waits on NPU."""

    def __init__(self, config: ReidConfig, worker: Optional[ReidWorker]) -> None:
        self.config = config
        self.worker = worker
        self.profile = TargetProfile(max_full=config.max_full_templates, max_torso=config.max_torso_templates)
        self.state = "INIT"
        self._last_bbox: Optional[BBox] = None
        self._stable_frames = 0
        self._latest_match: Optional[tuple[ReidResult, Match, float]] = None
        self._confirmations: list[bool] = []
        self._last_worker_timing: Optional[dict] = None

    @staticmethod
    def _view_bin(bbox: BBox, frame_width: int) -> int:
        center = (bbox[0] + bbox[2]) * 0.5 / max(1.0, float(frame_width))
        return max(0, min(2, int(center * 3.0)))

    def _quality(self, candidate: ReidCandidate, frame_width: int, frame_height: int) -> Optional[float]:
        x1, y1, x2, y2 = candidate.bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height
        margin_x = float(frame_width) * self.config.edge_margin_ratio
        margin_y = float(frame_height) * self.config.edge_margin_ratio
        if candidate.score < self.config.min_confidence or height < self.config.min_height_px or area < self.config.min_area_px:
            return None
        if x1 <= margin_x or y1 <= margin_y or x2 >= frame_width - margin_x or y2 >= frame_height - margin_y:
            return None
        return min(1.0, candidate.score) * min(1.0, area / (float(frame_width * frame_height) * 0.25))

    @staticmethod
    def _crop(frame, bbox: BBox):
        height, width = frame.shape[:2]
        x1 = max(0, min(width - 1, int(round(bbox[0]))))
        y1 = max(0, min(height - 1, int(round(bbox[1]))))
        x2 = max(0, min(width, int(round(bbox[2]))))
        y2 = max(0, min(height, int(round(bbox[3]))))
        return None if x2 <= x1 or y2 <= y1 else frame[y1:y2, x1:x2].copy()

    def _submit(self, *, frame, candidate: ReidCandidate, frame_id: int, now: float, purpose: str, frame_width: int, frame_height: int) -> bool:
        if self.worker is None:
            return False
        quality = self._quality(candidate, frame_width, frame_height)
        if quality is None:
            return False
        crop = self._crop(frame, candidate.bbox)
        if crop is None:
            return False
        interval = self.config.enroll_interval_sec if purpose == "enroll" else self.config.reacquire_interval_sec
        return self.worker.submit(
            ReidRequest(frame_id, now, purpose, candidate.bbox, quality, crop, frame_width), min_interval_sec=interval,
        )

    def _consume(self, now: float) -> None:
        if self.worker is None:
            return
        result = self.worker.poll_latest()
        if result is None:
            return
        self._last_worker_timing = dict(result.timings_ms)
        if result.error:
            self._latest_match = None
            return
        if result.purpose == "enroll":
            view_bin = self._view_bin(result.bbox, result.frame_width)
            full_added = self.profile.add(result.full_feature, source="full", quality=result.quality, captured_at=result.completed_at, view_bin=view_bin)
            torso_added = self.profile.add(result.torso_feature, source="torso", quality=result.quality, captured_at=result.completed_at, view_bin=view_bin)
            self._latest_match = None
            return
        age = max(0.0, now - result.submitted_at)
        if result.purpose != "reacquire" or age > self.config.result_max_age_sec:
            self._latest_match = None
            return
        match = self.profile.match(result.full_feature, result.torso_feature)
        self._latest_match = (result, match, age * 1000.0)

    def _decision(self, accepted: bool, candidate: Optional[ReidCandidate], *, state: str, reason: str,
                  match: Optional[Match] = None, age_ms: Optional[float] = None) -> ReidDecision:
        return ReidDecision(
            accepted, candidate, state, reason,
            None if match is None else match.score,
            None if match is None else match.source,
            age_ms, self._last_worker_timing, self.profile.full_count, self.profile.torso_count,
        )

    def _same_result_candidate(self, evidence_bbox: BBox, candidate: ReidCandidate, frame_width: int) -> bool:
        if iou(evidence_bbox, candidate.bbox) >= 0.15:
            return True
        ax = (evidence_bbox[0] + evidence_bbox[2]) * 0.5
        bx = (candidate.bbox[0] + candidate.bbox[2]) * 0.5
        return abs(ax - bx) <= max(16.0, float(frame_width) * 0.18)

    def observe(self, *, frame, candidates: list[ReidCandidate], frame_id: int, now: float,
                frame_width: int, frame_height: int) -> ReidDecision:
        self._consume(now)
        if not self.config.enabled:
            candidate = associate(candidates, self._last_bbox, frame_width=frame_width, frame_height=frame_height,
                                  min_iou=self.config.min_iou, max_center_distance_ratio=self.config.max_center_distance_ratio)
            self._last_bbox = None if candidate is None else candidate.bbox
            return self._decision(candidate is not None, candidate, state="DISABLED", reason="reid_disabled")

        associated = associate(candidates, self._last_bbox, frame_width=frame_width, frame_height=frame_height,
                               min_iou=self.config.min_iou, max_center_distance_ratio=self.config.max_center_distance_ratio)
        if self.state in {"INIT", "ENROLLING"}:
            if associated is None:
                self._stable_frames = 0
                self._last_bbox = None
                return self._decision(False, None, state="INIT", reason="no_candidate")
            self._last_bbox = associated.bbox
            self._stable_frames += 1
            if self._stable_frames >= self.config.stable_frames:
                self._submit(frame=frame, candidate=associated, frame_id=frame_id, now=now, purpose="enroll",
                             frame_width=frame_width, frame_height=frame_height)
                self.state = "LOCKED" if self.profile.ready(
                    min_full=self.config.min_full_templates, min_torso=self.config.min_torso_templates,
                ) else "ENROLLING"
            return self._decision(True, associated, state=self.state, reason="initial_target")

        if self.state == "LOCKED" and associated is not None:
            self._last_bbox = associated.bbox
            return self._decision(True, associated, state="LOCKED", reason="geometry_associated")

        # No continuous target: do not hand control to any new detection until
        # its ReID result is fresh and repeatedly confirmed.
        self.state = "SEARCHING"
        candidate = associated or associate(candidates, None, frame_width=frame_width, frame_height=frame_height,
                                             min_iou=self.config.min_iou, max_center_distance_ratio=self.config.max_center_distance_ratio)
        if candidate is None:
            self._confirmations.clear()
            return self._decision(False, None, state="SEARCHING", reason="no_candidate")
        self._submit(frame=frame, candidate=candidate, frame_id=frame_id, now=now, purpose="reacquire",
                     frame_width=frame_width, frame_height=frame_height)
        if self._latest_match is None:
            return self._decision(False, None, state="SEARCHING", reason="reid_pending")
        evidence, match, age_ms = self._latest_match
        self._latest_match = None
        threshold = self.config.full_threshold if match.source == "full" else self.config.torso_threshold
        hit = bool(match.score is not None and match.score >= threshold and self._same_result_candidate(evidence.bbox, candidate, frame_width))
        self._confirmations.append(hit)
        del self._confirmations[:-max(1, self.config.confirm_window)]
        if sum(self._confirmations) >= max(1, self.config.confirm_hits):
            self.state = "LOCKED"
            self._last_bbox = candidate.bbox
            self._confirmations.clear()
            return self._decision(True, candidate, state="LOCKED", reason="reid_confirmed", match=match, age_ms=age_ms)
        return self._decision(False, None, state="SEARCHING", reason="reid_mismatch" if not hit else "reid_confirming", match=match, age_ms=age_ms)
