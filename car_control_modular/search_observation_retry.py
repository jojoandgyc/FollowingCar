"""One extra zero-speed observation per search episode; never identity authority."""
from __future__ import annotations

import math
from typing import Optional

from .search_candidate_gate import BBox, SearchCandidateGate, SearchCandidateGateDecision


class SearchObservationRetry:
    """Observe a continuous credible candidate that exhausted the detector gate.

    Entry requires two distinct capture timestamps with local box continuity.
    The budget is spent once per (search epoch, UID), even if a track changes or
    a candidate vanishes. Completion needs a capture after a successful motor
    zero write plus a short settling interval; timeout never claims a target.
    """

    def __init__(self, max_hold_sec: float = 0.30) -> None:
        self.max_hold_sec = max(.10, min(.30, float(max_hold_sec)))
        self.reset()

    def reset(self) -> None:
        self.session = None
        self.spent = False
        self.active = False
        self.started_at = 0.0
        self.deadline = 0.0
        self.previous = None
        self.last_capture = None
        self.bbox = None
        self.score = 0.0
        self.reason = "inactive"

    def release(self) -> None:
        self.active = False
        self.previous = None

    @staticmethod
    def continuous(first: BBox, second: BBox) -> bool:
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in (first, second)]
        return bool(min(areas) > 0 and min(areas) / max(areas) >= .55
                    and SearchCandidateGate._iou(first, second) >= .35)

    def _decision(self, reason: str, *, entered=False, completed=False):
        self.reason = reason
        return SearchCandidateGateDecision(
            pause_rotation=not completed, entered=entered, completed=completed,
            source="credible_retry", reason="search_retry_" + reason,
            bbox=self.bbox, score=self.score,
            hold_frame=2 if completed else 1, hold_frames=2,
        )

    def update(self, *, now: float, session, capture_id: int, capture_timestamp: float,
               eligible: bool, bbox: Optional[BBox], score: float,
               blocked: bool, zero_sent_at: Optional[float] = None):
        if session is None:
            self.reset()
            return None
        if session != self.session:
            self.reset()
            self.session = session
        if self.active and now >= self.deadline:
            self.active = False
            return self._decision("timeout", completed=True)
        fresh = bool(math.isfinite(capture_timestamp) and 0 <= now - capture_timestamp <= .19
                     and (self.last_capture is None or (
                         capture_id > self.last_capture[0] and capture_timestamp > self.last_capture[1])))
        if not fresh:
            return self._decision("await_new_capture") if self.active else None
        self.last_capture = (capture_id, capture_timestamp)
        if not eligible or bbox is None:
            self.previous = None
            if self.active:
                self.active = False
                return self._decision("candidate_lost_or_ambiguous", completed=True)
            self.reason = "no_credible_candidate"
            return None
        if self.active:
            if not self.continuous(self.bbox, bbox):
                self.active = False
                return self._decision("candidate_changed", completed=True)
            self.bbox, self.score = bbox, score
            post_zero = bool(zero_sent_at is not None and math.isfinite(zero_sent_at)
                             and self.started_at <= zero_sent_at <= now
                             and capture_timestamp >= zero_sent_at + .08)
            if post_zero:
                self.active = False
                return self._decision("post_zero_capture", completed=True)
            return self._decision("await_post_zero_capture")
        continuous = bool(self.previous is not None
                          and 0 < capture_timestamp - self.previous[0] <= .25
                          and self.continuous(self.previous[1], bbox))
        self.previous = (capture_timestamp, bbox)
        if self.spent or not blocked or not continuous:
            self.reason = "budget_spent" if self.spent else "await_continuity"
            return None
        self.spent = self.active = True
        self.bbox, self.score = bbox, score
        self.started_at, self.deadline = now, now + self.max_hold_sec
        return self._decision("start", entered=True)
