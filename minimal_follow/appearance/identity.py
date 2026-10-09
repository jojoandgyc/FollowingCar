"""Main-thread identity policy: enrollment is opportunistic, re-ID is gated."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

from .bank import AppearanceTemplateBank
from .quality import AppearanceQuality, AppearanceQualityGate
from .worker import AppearanceRequest, AppearanceResult, AppearanceWorker


@dataclass(frozen=True)
class AppearanceIdentityConfig:
    enabled: bool = True
    stable_enroll_frames: int = 3
    enroll_interval_sec: float = 0.80
    reacquire_interval_sec: float = 0.20
    result_max_age_sec: float = 0.35
    full_match_threshold: float = 0.72
    partial_match_threshold: float = 0.80
    reacquire_confirm_results: int = 2
    max_full_templates: int = 5
    max_partial_templates: int = 3


@dataclass(frozen=True)
class AppearanceDecision:
    accepted: bool
    state: str
    reason: str
    match_score: Optional[float]
    match_source: Optional[str]
    result_age_ms: Optional[float]
    worker_timing_ms: Optional[dict]


class AppearanceIdentityPolicy:
    def __init__(
        self,
        config: AppearanceIdentityConfig,
        quality_gate: AppearanceQualityGate,
        worker: Optional[AppearanceWorker],
    ) -> None:
        self.config = config
        self.quality_gate = quality_gate
        self.worker = worker
        self.bank = AppearanceTemplateBank(
            max_full=config.max_full_templates, max_partial=config.max_partial_templates,
        )
        self._visible_streak = 0
        self._reacquire_confirmations = 0
        self._latest_reacquire: Optional[AppearanceDecision] = None
        self._last_worker_timing: Optional[dict] = None
        self._enroll_requests = 0

    @property
    def enrolled(self) -> bool:
        return self.bank.enrolled

    def _consume_result(self, now: float) -> None:
        if self.worker is None:
            return
        result = self.worker.poll_latest()
        if result is None:
            return
        self._last_worker_timing = dict(result.timings_ms)
        age = max(0.0, now - float(result.submitted_at))
        age_ms = age * 1000.0
        if result.error:
            self._latest_reacquire = AppearanceDecision(False, "error", result.error, None, None, age_ms, self._last_worker_timing)
            return
        if result.purpose == "enroll":
            if result.full_feature is not None:
                self.bank.add(result.full_feature, source="full")
            if result.partial_feature is not None:
                self.bank.add(result.partial_feature, source="partial")
            return
        if result.purpose != "reacquire" or age > float(self.config.result_max_age_sec):
            self._latest_reacquire = AppearanceDecision(False, "stale", "result_stale", None, None, age_ms, self._last_worker_timing)
            return
        match = self.bank.match(result.full_feature, result.partial_feature)
        threshold = (
            self.config.full_match_threshold if match.source == "full"
            else self.config.partial_match_threshold
        )
        if match.score is not None and match.score >= threshold:
            self._reacquire_confirmations += 1
            accepted = self._reacquire_confirmations >= max(1, int(self.config.reacquire_confirm_results))
            self._latest_reacquire = AppearanceDecision(
                accepted, "reacquire_match", "confirmed" if accepted else "confirming",
                match.score, match.source, age_ms, self._last_worker_timing,
            )
        else:
            self._reacquire_confirmations = 0
            self._latest_reacquire = AppearanceDecision(
                False, "reacquire_mismatch", "threshold", match.score, match.source,
                age_ms, self._last_worker_timing,
            )

    def _submit(
        self, *, frame, bbox: Tuple[float, float, float, float], score: float,
        frame_id: int, now: float, purpose: str, quality: AppearanceQuality,
    ) -> bool:
        if self.worker is None or not (quality.full_ok or quality.partial_ok):
            return False
        # Copy only on rare feature events. OpenCV capture buffers may be
        # reused immediately after the control loop advances to the next frame.
        # Full enrollment also collects a torso descriptor.  This runs just a
        # few times at startup and is what makes a later edge-clipped target
        # comparable to the separate partial gallery.
        compute_partial = quality.partial_ok and (
            purpose == "enroll" or not quality.full_ok
        )
        request = AppearanceRequest(
            frame_id=frame_id,
            submitted_at=now,
            bbox=bbox,
            score=score,
            purpose=purpose,
            allow_full=quality.full_ok,
            compute_partial=compute_partial,
            frame=frame.copy(),
        )
        interval = self.config.enroll_interval_sec if purpose == "enroll" else self.config.reacquire_interval_sec
        submitted = self.worker.submit(request, min_interval_sec=interval)
        if submitted and purpose == "enroll":
            self._enroll_requests += 1
        return submitted

    def visible_target(
        self, *, frame, bbox, score: float, frame_id: int, now: float,
        frame_width: int, frame_height: int,
    ) -> AppearanceDecision:
        self._consume_result(now)
        self._visible_streak += 1
        self._reacquire_confirmations = 0
        quality = self.quality_gate.evaluate(
            bbox, score=score, frame_width=frame_width, frame_height=frame_height,
        )
        if (self.config.enabled and self._visible_streak >= max(1, int(self.config.stable_enroll_frames))
                # Bound expensive enrollment even if several near-identical
                # samples collapse to one template in the de-duplication bank.
                and self._enroll_requests < self.config.max_full_templates):
            self._submit(
                frame=frame, bbox=bbox, score=score, frame_id=frame_id, now=now,
                purpose="enroll", quality=quality,
            )
        return AppearanceDecision(True, "visible", "normal_follow", None, None, None, self._last_worker_timing)

    def target_missing(self, now: float) -> None:
        self._consume_result(now)
        if self._visible_streak:
            # A result from an older loss episode must never authorize a new
            # search episode just because it arrived late.
            self._latest_reacquire = None
            self._reacquire_confirmations = 0
        self._visible_streak = 0

    def search_candidate(
        self, *, frame, bbox, score: float, frame_id: int, now: float,
        frame_width: int, frame_height: int,
    ) -> AppearanceDecision:
        self._consume_result(now)
        if not self.config.enabled or not self.bank.enrolled:
            return AppearanceDecision(True, "fallback", "bank_unavailable", None, None, None, self._last_worker_timing)
        if self._latest_reacquire is not None and self._latest_reacquire.accepted:
            return self._latest_reacquire
        quality = self.quality_gate.evaluate(
            bbox, score=score, frame_width=frame_width, frame_height=frame_height,
        )
        submitted = self._submit(
            frame=frame, bbox=bbox, score=score, frame_id=frame_id, now=now,
            purpose="reacquire", quality=quality,
        )
        if not (quality.full_ok or quality.partial_ok):
            return AppearanceDecision(False, "candidate_rejected", quality.reason, None, None, None, self._last_worker_timing)
        if self._latest_reacquire is not None:
            return self._latest_reacquire
        return AppearanceDecision(False, "candidate_pending", "submitted" if submitted else "rate_limited", None, None, None, self._last_worker_timing)
