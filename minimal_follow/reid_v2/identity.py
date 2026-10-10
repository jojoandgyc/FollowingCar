"""State machine for an independently implemented one-target ReID profile."""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from .association import BBox, ReidCandidate, associate, center_distance_ratio, iou
from .profile import Match, TargetProfile
from .worker import ReidRequest, ReidResult, ReidWorker


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReidConfig:
    enabled: bool = True
    stable_frames: int = 10
    enroll_interval_sec: float = 1.0
    reacquire_interval_sec: float = 0.20
    result_max_age_sec: float = 0.75
    # One high-quality template is enough to lock a first-session target;
    # later views are retained and make matching progressively more robust.
    min_full_templates: int = 1
    min_torso_templates: int = 2
    max_full_templates: int = 8
    max_torso_templates: int = 4
    full_threshold: float = 0.70
    torso_threshold: float = 0.76
    confirm_hits: int = 2
    confirm_window: int = 3
    min_iou: float = 0.12
    max_center_distance_ratio: float = 0.16
    min_confidence: float = 0.65
    min_height_px: float = 120.0
    min_area_px: float = 8000.0
    edge_margin_ratio: float = 0.015
    # Deprecated A/B-test mode retained for launch compatibility.
    freeze_after_first_enrollment: bool = False
    # Bootstrap is the only time the target gallery may change. It samples at
    # a bounded high rate until all requested visual views have evidence, then
    # freezes permanently for the remainder of this process.
    bootstrap_enabled: bool = True
    bootstrap_interval_sec: float = 0.20
    bootstrap_required_views: int = 4
    freeze_after_bootstrap: bool = True
    allow_partial_enrollment: bool = True
    enrollment_min_aspect_ratio: float = 0.22
    enrollment_max_aspect_ratio: float = 0.95
    enrollment_max_competitor_overlap: float = 0.12
    enrollment_duplicate_similarity: float = 0.97
    enrollment_owner_min_iou: float = 0.20
    enrollment_owner_max_center_distance_ratio: float = 0.10
    view_capture_enabled: bool = True
    view_change_threshold: float = 0.90
    max_templates_per_view: int = 2


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
    probe_candidates: int = 0
    probe_attempts: int = 0
    view_templates: Optional[dict] = None
    profile_frozen: bool = False
    enrollment_status: str = "idle"
    target_track_id: Optional[int] = None


class ReidPolicy:
    """Target lock + asynchronous ReID. The control thread never waits on NPU."""

    def __init__(self, config: ReidConfig, worker: Optional[ReidWorker]) -> None:
        self.config = config
        self.worker = worker
        self.profile = TargetProfile(
            max_full=config.max_full_templates, max_torso=config.max_torso_templates,
            max_templates_per_view=config.max_templates_per_view,
            duplicate_similarity=config.enrollment_duplicate_similarity,
        )
        self.state = "INIT"
        self._last_bbox: Optional[BBox] = None
        self._stable_frames = 0
        self._latest_match: Optional[tuple[ReidResult, Match, float]] = None
        self._last_worker_timing: Optional[dict] = None
        self._probe_bbox: Optional[BBox] = None
        self._probe_cursor = 0
        self._probe_attempts = 0
        self._probe_hits = 0
        self._probe_candidates = 0
        self._enroll_view_index = 0
        self._profile_frozen = False
        self._enrollment_status = "idle"
        self._enrollment_owner_bbox: Optional[BBox] = None
        # This is deliberately separate from the detector's raw order.  Once
        # initialized, normal following may only use this ByteTrack ID.  A
        # different ID must pass asynchronous ReID confirmation in SEARCHING.
        self._target_track_id: Optional[int] = None

    def _enrollment_view(self, feature) -> str:
        """Assign samples to an ordered four-view capture session.

        OSNet is an identity extractor, not a body-orientation classifier.  We
        therefore use it only to detect a sufficiently new visual view.  The
        labels are reliable when the initial user-facing capture is performed
        in the documented order: front, left, right, back.  Regardless of the
        label, matching always compares every retained template.
        """
        if not self.config.view_capture_enabled:
            return "unknown"
        order = TargetProfile.VIEW_ORDER
        while self._enroll_view_index < len(order):
            active = order[self._enroll_view_index]
            similarity = self.profile.best_view_similarity(feature, active)
            if similarity is None or similarity >= self.config.view_change_threshold:
                return active
            self._enroll_view_index += 1
        scores = [(self.profile.best_view_similarity(feature, view), view) for view in order]
        usable = [item for item in scores if item[0] is not None]
        return max(usable, key=lambda item: item[0])[1] if usable else order[-1]

    @staticmethod
    def _view_bin(bbox: BBox, frame_width: int) -> int:
        center = (bbox[0] + bbox[2]) * 0.5 / max(1.0, float(frame_width))
        return max(0, min(2, int(center * 3.0)))

    def _quality(self, candidate: ReidCandidate, frame_width: int, frame_height: int, *, purpose: str) -> Optional[tuple[float, bool]]:
        x1, y1, x2, y2 = candidate.bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height
        aspect_ratio = width / max(1.0, height)
        margin_x = float(frame_width) * self.config.edge_margin_ratio
        margin_y = float(frame_height) * self.config.edge_margin_ratio
        if candidate.score < self.config.min_confidence or height < self.config.min_height_px or area < self.config.min_area_px:
            return None
        if purpose == "enroll" and not (
            self.config.enrollment_min_aspect_ratio <= aspect_ratio <= self.config.enrollment_max_aspect_ratio
        ):
            return None
        quality = min(1.0, candidate.score) * min(1.0, area / (float(frame_width * frame_height) * 0.25))
        if x1 <= margin_x or y1 <= margin_y or x2 >= frame_width - margin_x or y2 >= frame_height - margin_y:
            # A partial person entering from an edge is useless for building a
            # full-body gallery, but its torso is often enough for search.
            if purpose == "enroll" and not self.config.allow_partial_enrollment:
                return None
            return quality * 0.70, False
        return quality, True

    @staticmethod
    def _overlap_ratio(left: BBox, right: BBox) -> float:
        intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
            0.0, min(left[3], right[3]) - max(left[1], right[1])
        )
        left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        return intersection / max(1.0, min(left_area, right_area))

    def _enrollment_allowed(self, candidate: ReidCandidate, candidates: list[ReidCandidate], *, frame_width: int,
                            frame_height: int) -> bool:
        """Protect the immutable target gallery from nearby/ambiguous people."""
        if self._profile_frozen:
            self._enrollment_status = "profile_frozen"
            return False
        if self._target_track_id is not None and candidate.track_id != self._target_track_id:
            self._enrollment_status = "target_track_rejected"
            return False
        if self._enrollment_owner_bbox is not None:
            continuous = (
                iou(self._enrollment_owner_bbox, candidate.bbox) >= self.config.enrollment_owner_min_iou
                or center_distance_ratio(
                    self._enrollment_owner_bbox, candidate.bbox, frame_width, frame_height,
                ) <= self.config.enrollment_owner_max_center_distance_ratio
            )
            if not continuous:
                self._enrollment_status = "owner_geometry_rejected"
                return False
        for other in candidates:
            if other is candidate or other.bbox == candidate.bbox:
                continue
            if self._overlap_ratio(candidate.bbox, other.bbox) > self.config.enrollment_max_competitor_overlap:
                self._enrollment_status = "overlapping_person_rejected"
                return False
        self._enrollment_owner_bbox = candidate.bbox
        return True

    @staticmethod
    def _crop(frame, bbox: BBox):
        height, width = frame.shape[:2]
        x1 = max(0, min(width - 1, int(round(bbox[0]))))
        y1 = max(0, min(height - 1, int(round(bbox[1]))))
        x2 = max(0, min(width, int(round(bbox[2]))))
        y2 = max(0, min(height, int(round(bbox[3]))))
        return None if x2 <= x1 or y2 <= y1 else frame[y1:y2, x1:x2].copy()

    def _submit(self, *, frame, candidate: ReidCandidate, frame_id: int, now: float, purpose: str, frame_width: int,
                frame_height: int, candidates: Optional[list[ReidCandidate]] = None) -> bool:
        if self.worker is None:
            if purpose == "enroll":
                self._enrollment_status = "worker_unavailable"
            return False
        if purpose == "enroll" and not self._enrollment_allowed(
            candidate, candidates or [candidate], frame_width=frame_width, frame_height=frame_height,
        ):
            return False
        quality_evidence = self._quality(candidate, frame_width, frame_height, purpose=purpose)
        if quality_evidence is None:
            if purpose == "enroll":
                self._enrollment_status = "quality_rejected"
            return False
        quality, allow_full = quality_evidence
        crop = self._crop(frame, candidate.bbox)
        if crop is None:
            if purpose == "enroll":
                self._enrollment_status = "crop_rejected"
            return False
        interval = self.config.reacquire_interval_sec
        if purpose == "enroll":
            interval = (self.config.bootstrap_interval_sec if self.config.bootstrap_enabled and not self._profile_frozen
                        else self.config.enroll_interval_sec)
        submitted = self.worker.submit(
            ReidRequest(
                frame_id=frame_id, submitted_at=now, purpose=purpose, bbox=candidate.bbox,
                quality=quality, crop=crop, frame_width=frame_width, allow_full=allow_full,
                track_id=candidate.track_id,
            ), min_interval_sec=interval,
        )
        if purpose == "enroll":
            self._enrollment_status = "submitted" if submitted else "rate_limited"
        return submitted

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
            # A fixed-profile run has exactly one enrollment result. Results
            # queued just before the first one completed are discarded here,
            # so they cannot silently replace or supplement the target.
            if self.config.freeze_after_first_enrollment and self._profile_frozen:
                self._latest_match = None
                return
            view_bin = self._view_bin(result.bbox, result.frame_width)
            # A clipped person does not enter the full-body gallery, but its
            # torso feature is still valid evidence for a bootstrap view.
            view_feature = result.full_feature if result.full_feature is not None else result.torso_feature
            source = "full" if result.full_feature is not None else "torso"
            view = self._enrollment_view(view_feature)
            if self.profile.is_duplicate(
                view_feature, source=source, similarity=self.config.enrollment_duplicate_similarity, view=view,
            ):
                self._enrollment_status = "duplicate_rejected"
                self._latest_match = None
                return
            full_added = self.profile.add(result.full_feature, source="full", quality=result.quality, captured_at=result.completed_at, view_bin=view_bin, view=view)
            torso_added = self.profile.add(result.torso_feature, source="torso", quality=result.quality, captured_at=result.completed_at, view_bin=view_bin, view=view)
            # Advance only after a distinct view has been stored. This avoids
            # rapidly consuming all four slots when the target is standing.
            if full_added and self._enroll_view_index < len(TargetProfile.VIEW_ORDER) - 1:
                active = TargetProfile.VIEW_ORDER[self._enroll_view_index]
                if view == active and self.profile.view_count(view) >= self.profile.max_templates_per_view:
                    # Keep collecting the current view until a visual change;
                    # the next result will remain here if it is still similar.
                    pass
            if self.config.freeze_after_first_enrollment and (full_added or torso_added):
                self._profile_frozen = True
                self._enrollment_status = "frozen_first_template"
            elif (self.config.bootstrap_enabled and self.config.freeze_after_bootstrap
                    and self.profile.covered_view_count() >= max(1, int(self.config.bootstrap_required_views))):
                self._profile_frozen = True
                self._enrollment_status = "frozen_bootstrap_complete"
            else:
                self._enrollment_status = "bootstrap_collecting"
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
            self._probe_candidates, self._probe_attempts, self.profile.combined_view_counts(), self._profile_frozen,
            self._enrollment_status, self._target_track_id,
        )

    def _candidate_with_target_track(self, candidates: list[ReidCandidate]) -> Optional[ReidCandidate]:
        if self._target_track_id is None:
            return None
        return next((item for item in candidates if item.track_id == self._target_track_id), None)

    def _same_result_candidate(self, evidence_bbox: BBox, candidate: ReidCandidate, frame_width: int) -> bool:
        if iou(evidence_bbox, candidate.bbox) >= 0.15:
            return True
        ax = (evidence_bbox[0] + evidence_bbox[2]) * 0.5
        bx = (candidate.bbox[0] + candidate.bbox[2]) * 0.5
        return abs(ax - bx) <= max(16.0, float(frame_width) * 0.18)

    def _reset_probe(self, *, advance: bool = False) -> None:
        if advance:
            self._probe_cursor += 1
        self._probe_bbox = None
        self._probe_attempts = 0
        self._probe_hits = 0

    @staticmethod
    def _ordered_candidates(candidates: list[ReidCandidate]) -> list[ReidCandidate]:
        # Stable left-to-right ordering makes round-robin evidence clear in
        # logs and guarantees a large bystander cannot starve other people.
        return sorted(candidates, key=lambda item: ((item.bbox[0] + item.bbox[2]) * 0.5, -item.score))

    def _probe_candidate(self, candidates: list[ReidCandidate], frame_width: int) -> Optional[ReidCandidate]:
        ordered = self._ordered_candidates(candidates)
        self._probe_candidates = len(ordered)
        if not ordered:
            return None
        if self._probe_bbox is not None:
            for candidate in ordered:
                if self._same_result_candidate(self._probe_bbox, candidate, frame_width):
                    return candidate
            self._reset_probe(advance=True)
        return ordered[self._probe_cursor % len(ordered)]

    def _candidate_for_evidence(
        self, evidence_bbox: BBox, candidates: list[ReidCandidate], frame_width: int,
    ) -> Optional[ReidCandidate]:
        for candidate in self._ordered_candidates(candidates):
            if self._same_result_candidate(evidence_bbox, candidate, frame_width):
                return candidate
        return None

    def _consume_reacquire_result(
        self, candidates: list[ReidCandidate], frame_width: int,
    ) -> tuple[Optional[ReidCandidate], Optional[Match], Optional[float], Optional[str]]:
        if self._latest_match is None:
            return None, None, None, None
        evidence, match, age_ms = self._latest_match
        self._latest_match = None
        candidate = self._candidate_for_evidence(evidence.bbox, candidates, frame_width)
        same_probe = candidate is not None and self._probe_bbox is not None and self._same_result_candidate(
            self._probe_bbox, candidate, frame_width,
        )
        if not same_probe:
            self._reset_probe(advance=True)
            return None, match, age_ms, "reid_candidate_left"
        self._probe_attempts += 1
        threshold = self.config.full_threshold if match.source == "full" else self.config.torso_threshold
        hit = bool(match.score is not None and match.score >= threshold)
        if hit:
            self._probe_hits += 1
            if self._probe_hits >= max(1, self.config.confirm_hits):
                self._reset_probe()
                return candidate, match, age_ms, "reid_confirmed"
        if self._probe_attempts >= max(1, self.config.confirm_window):
            self._reset_probe(advance=True)
            return None, match, age_ms, "reid_mismatch"
        return None, match, age_ms, "reid_confirming" if hit else "reid_probe_retry"

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
            if self._target_track_id is None:
                self._target_track_id = associated.track_id
            elif associated.track_id != self._target_track_id:
                # Do not replace an unfinished initial enrollment with a
                # person who merely became geometrically more convenient.
                associated = self._candidate_with_target_track(candidates)
                if associated is None:
                    self._stable_frames = 0
                    self._last_bbox = None
                    return self._decision(False, None, state="ENROLLING", reason="initial_track_missing")
            self._last_bbox = associated.bbox
            self._stable_frames += 1
            if self._stable_frames >= self.config.stable_frames:
                if self.profile.ready(min_full=self.config.min_full_templates, min_torso=self.config.min_torso_templates):
                    self.state = "LOCKED"
                else:
                    self._submit(frame=frame, candidate=associated, frame_id=frame_id, now=now, purpose="enroll",
                                 frame_width=frame_width, frame_height=frame_height, candidates=candidates)
                    self.state = "ENROLLING"
            return self._decision(True, associated, state=self.state, reason="initial_target")

        locked_candidate = self._candidate_with_target_track(candidates)
        if self.state == "LOCKED" and locked_candidate is not None:
            self._last_bbox = locked_candidate.bbox
            # Keep collecting a small number of views after lock. Otherwise a
            # target enrolled only front-facing would be very hard to recover
            # after the vehicle rotates during a loss episode.
            if not self._profile_frozen:
                self._submit(frame=frame, candidate=locked_candidate, frame_id=frame_id, now=now, purpose="enroll",
                             frame_width=frame_width, frame_height=frame_height, candidates=candidates)
            return self._decision(True, locked_candidate, state="LOCKED", reason="track_associated")

        # No continuous target: probe every detected person in turn. A large
        # bystander gets at most one confirmation window before the next
        # candidate is sampled, while the normal control loop stays nonblocking.
        if self.state != "SEARCHING":
            self._reset_probe()
        self.state = "SEARCHING"
        confirmed, match, age_ms, result_reason = self._consume_reacquire_result(candidates, frame_width)
        if confirmed is not None:
            self.state = "LOCKED"
            self._last_bbox = confirmed.bbox
            self._target_track_id = confirmed.track_id
            self._enrollment_owner_bbox = confirmed.bbox
            return self._decision(True, confirmed, state="LOCKED", reason="reid_confirmed", match=match, age_ms=age_ms)
        candidate = self._probe_candidate(candidates, frame_width)
        if candidate is None:
            self._reset_probe()
            return self._decision(False, None, state="SEARCHING", reason=result_reason or "no_candidate", match=match, age_ms=age_ms)
        submitted = self._submit(frame=frame, candidate=candidate, frame_id=frame_id, now=now, purpose="reacquire",
                                 frame_width=frame_width, frame_height=frame_height)
        if submitted:
            self._probe_bbox = candidate.bbox
        return self._decision(False, None, state="SEARCHING", reason=result_reason or ("reid_pending" if submitted else "reid_rate_limited"), match=match, age_ms=age_ms)
