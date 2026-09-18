from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class CandidateObservation:
    bbox: BBox
    score: float


@dataclass(frozen=True)
class SearchCandidateGateConfig:
    enabled: bool = True
    formal_min_score: float = 0.25
    probe_min_score: float = 0.10
    min_area_ratio: float = 0.01
    max_area_ratio: float = 0.75
    min_aspect_ratio: float = 0.12
    max_aspect_ratio: float = 1.50
    probe_confirm_frames: int = 1
    # Hold the current search motion for a bounded number of fresh detector
    # observations before handing control back to normal following. Runtime
    # defaults to one frame; tests and deployments may raise it explicitly.
    hold_frames: int = 1
    max_hold_sec: float = 0.30
    consistency_iou: float = 0.20
    # Detector confidence often flickers for a few frames around the formal
    # threshold. Keep an observed candidate blocked across that short gap so
    # the same person cannot repeatedly stop an otherwise bounded sweep.
    blocked_reset_missing_frames: int = 8
    # A detector fragment can consume the observation hold just before the
    # real person box appears.  A large, valid area change is evidence of a
    # new observation candidate; re-arm once, while ordinary coordinate drift
    # remains blocked by the missing-frame rule above.
    blocked_rearm_area_ratio: float = 3.0


@dataclass(frozen=True)
class SearchCandidateGateDecision:
    pause_rotation: bool = False
    entered: bool = False
    completed: bool = False
    source: str = "none"
    reason: str = "no_candidate"
    score: float = 0.0
    bbox: Optional[BBox] = None
    probe_streak: int = 0
    hold_frame: int = 0
    hold_frames: int = 0
    defer_sec: float = 0.0
    preferred_target_match: bool = False


class SearchCandidateGate:
    """Turn recognition evidence into a bounded search observation hold.

    This state machine has no motor, PID, tracker, or ReID dependency. It can
    request a temporary pause, but it cannot select a target or produce motion.
    """

    def __init__(self, config: SearchCandidateGateConfig) -> None:
        self.config = config
        self._probe_bbox: Optional[BBox] = None
        self._probe_streak = 0
        self._hold_remaining = 0
        self._hold_source = "none"
        self._hold_score = 0.0
        self._hold_bbox: Optional[BBox] = None
        self._hold_started_at: Optional[float] = None
        self._blocked_bbox: Optional[BBox] = None
        self._blocked_missing_frames = 0
        self._last_timestamp: Optional[float] = None

    @property
    def hold_active(self) -> bool:
        return self._hold_remaining > 0

    def reset(self) -> None:
        self._probe_bbox = None
        self._probe_streak = 0
        self._hold_remaining = 0
        self._hold_source = "none"
        self._hold_score = 0.0
        self._hold_bbox = None
        self._hold_started_at = None
        self._blocked_bbox = None
        self._blocked_missing_frames = 0
        self._last_timestamp = None

    @staticmethod
    def _iou(first: BBox, second: BBox) -> float:
        x1 = max(float(first[0]), float(second[0]))
        y1 = max(float(first[1]), float(second[1]))
        x2 = min(float(first[2]), float(second[2]))
        y2 = min(float(first[3]), float(second[3]))
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        return 0.0 if union <= 0.0 else intersection / union

    def _valid_candidates(
        self,
        candidates: Tuple[CandidateObservation, ...],
        *,
        width: int,
        height: int,
        min_score: float,
        below_score: Optional[float] = None,
    ) -> Tuple[CandidateObservation, ...]:
        frame_area = float(max(1, int(width) * int(height)))
        valid = []
        for candidate in candidates:
            score = float(candidate.score)
            if score < float(min_score):
                continue
            if below_score is not None and score >= float(below_score):
                continue
            x1, y1, x2, y2 = (float(value) for value in candidate.bbox)
            box_width = max(0.0, x2 - x1)
            box_height = max(0.0, y2 - y1)
            if box_width <= 0.0 or box_height <= 0.0:
                continue
            area_ratio = box_width * box_height / frame_area
            aspect_ratio = box_width / box_height
            if not (
                float(self.config.min_area_ratio)
                <= area_ratio
                <= float(self.config.max_area_ratio)
            ):
                continue
            if not (
                float(self.config.min_aspect_ratio)
                <= aspect_ratio
                <= float(self.config.max_aspect_ratio)
            ):
                continue
            valid.append(CandidateObservation((x1, y1, x2, y2), score))
        return tuple(sorted(valid, key=lambda item: item.score, reverse=True))

    def _select_current(
        self,
        formal: Tuple[CandidateObservation, ...],
        probe: Tuple[CandidateObservation, ...],
        preferred_bbox: Optional[BBox],
    ) -> Tuple[Optional[CandidateObservation], str, bool]:
        if preferred_bbox is not None:
            matches = [
                (self._iou(preferred_bbox, item.bbox), source, item)
                for source, candidates in (("formal", formal), ("probe", probe))
                for item in candidates
                if self._iou(preferred_bbox, item.bbox)
                >= float(self.config.consistency_iou)
            ]
            if matches:
                _iou, source, candidate = max(
                    matches,
                    key=lambda entry: (
                        float(entry[0]),
                        entry[1] == "formal",
                        float(entry[2].score),
                    ),
                )
                return candidate, source, True
        if formal:
            return formal[0], "formal", False
        if probe:
            return probe[0], "probe", False
        return None, "none", False

    def _start_hold(
        self,
        candidate: CandidateObservation,
        *,
        source: str,
        defer_sec: float,
        timestamp: float,
    ) -> SearchCandidateGateDecision:
        hold_frames = max(1, int(self.config.hold_frames))
        self._hold_remaining = hold_frames - 1
        self._hold_source = str(source)
        self._hold_score = float(candidate.score)
        self._hold_bbox = candidate.bbox
        self._hold_started_at = float(timestamp)
        self._probe_bbox = None
        self._probe_streak = 0
        if self._hold_remaining <= 0:
            self._blocked_bbox = candidate.bbox
        return SearchCandidateGateDecision(
            pause_rotation=True,
            entered=True,
            completed=self._hold_remaining <= 0,
            source=self._hold_source,
            reason="%s_candidate_observe_start" % self._hold_source,
            score=self._hold_score,
            bbox=self._hold_bbox,
            hold_frame=1,
            hold_frames=hold_frames,
            defer_sec=defer_sec,
        )

    def select_current_candidate(
        self,
        *,
        width: int,
        height: int,
        formal_candidates: Tuple[CandidateObservation, ...] = (),
        probe_candidates: Tuple[CandidateObservation, ...] = (),
        preferred_bbox: Optional[BBox] = None,
    ) -> SearchCandidateGateDecision:
        """Return the best valid candidate without changing gate state."""
        if not self.config.enabled:
            return SearchCandidateGateDecision(reason="gate_disabled")
        formal = self._valid_candidates(
            formal_candidates,
            width=width,
            height=height,
            min_score=self.config.formal_min_score,
        )
        probe = self._valid_candidates(
            probe_candidates,
            width=width,
            height=height,
            min_score=self.config.probe_min_score,
            below_score=self.config.formal_min_score,
        )
        candidate, source, preferred_match = self._select_current(
            formal,
            probe,
            preferred_bbox,
        )
        if candidate is None:
            return SearchCandidateGateDecision(reason="no_candidate")
        return SearchCandidateGateDecision(
            source=source,
            reason=(
                "current_%s_active_target_candidate" % source
                if preferred_match
                else "current_%s_candidate" % source
            ),
            score=float(candidate.score),
            bbox=candidate.bbox,
            preferred_target_match=preferred_match,
        )

    def update(
        self,
        *,
        timestamp: float,
        search_active: bool,
        width: int,
        height: int,
        formal_candidates: Tuple[CandidateObservation, ...] = (),
        probe_candidates: Tuple[CandidateObservation, ...] = (),
        preferred_bbox: Optional[BBox] = None,
    ) -> SearchCandidateGateDecision:
        now = float(timestamp)
        defer_sec = (
            0.0
            if self._last_timestamp is None
            else max(0.0, now - float(self._last_timestamp))
        )
        self._last_timestamp = now
        if not self.config.enabled or not search_active:
            self.reset()
            return SearchCandidateGateDecision(reason="search_inactive")

        formal = self._valid_candidates(
            formal_candidates,
            width=width,
            height=height,
            min_score=self.config.formal_min_score,
        )
        probe = self._valid_candidates(
            probe_candidates,
            width=width,
            height=height,
            min_score=self.config.probe_min_score,
            below_score=self.config.formal_min_score,
        )
        current, current_source, preferred_match = self._select_current(
            formal,
            probe,
            preferred_bbox,
        )

        if self._hold_remaining > 0:
            # Keep the observation attached to the same moving person. The
            # first bbox is only entry evidence; after several frames it may
            # describe the side the person has already left.
            hold_candidates = formal + probe
            if self._hold_bbox is not None and hold_candidates:
                matched = max(
                    hold_candidates,
                    key=lambda item: self._iou(self._hold_bbox, item.bbox),
                )
                if self._iou(self._hold_bbox, matched.bbox) >= float(
                    self.config.consistency_iou
                ):
                    self._hold_bbox = matched.bbox
                    self._hold_score = float(matched.score)
            hold_frames = max(1, int(self.config.hold_frames))
            hold_frame = hold_frames - self._hold_remaining + 1
            hold_elapsed = (
                0.0
                if self._hold_started_at is None
                else max(0.0, now - float(self._hold_started_at))
            )
            if hold_elapsed >= max(0.0, float(self.config.max_hold_sec)):
                self._hold_remaining = 0
                self._blocked_bbox = self._hold_bbox
                self._blocked_missing_frames = 0
                return SearchCandidateGateDecision(
                    pause_rotation=False,
                    completed=True,
                    source=self._hold_source,
                    reason="%s_candidate_observe_timeout" % self._hold_source,
                    score=self._hold_score,
                    bbox=self._hold_bbox,
                    hold_frame=hold_frame,
                    hold_frames=hold_frames,
                    defer_sec=defer_sec,
                )
            self._hold_remaining -= 1
            completed = self._hold_remaining <= 0
            if completed:
                self._blocked_bbox = self._hold_bbox
                self._blocked_missing_frames = 0
            return SearchCandidateGateDecision(
                pause_rotation=True,
                completed=completed,
                source=self._hold_source,
                reason="%s_candidate_observe_hold" % self._hold_source,
                score=self._hold_score,
                bbox=self._hold_bbox,
                hold_frame=hold_frame,
                hold_frames=hold_frames,
                defer_sec=defer_sec,
            )

        if self._blocked_bbox is not None:
            # A fresh, strongly matched active-UID bbox is allowed to replace
            # a previously blocked detector candidate.  This is the handoff
            # path for a target reappearing on the opposite side of a frozen
            # sweep; ordinary detector boxes still follow the bounded missing
            # frame reset below.
            if preferred_bbox is not None and self._iou(self._blocked_bbox, preferred_bbox) < float(
                self.config.consistency_iou
            ):
                self._blocked_bbox = None
                self._blocked_missing_frames = 0

        if self._blocked_bbox is not None:
            # Once a candidate has completed its observation hold, changing
            # bbox coordinates does not make it a new candidate.  Re-arm only
            # after the candidate is actually absent for the configured gap;
            # otherwise a moving person can restart the pause every frame and
            # leave the sweep stuck at a tiny encoder angle.
            if current is not None:
                self._blocked_missing_frames = 0
                blocked_candidates = formal + probe
                matched = max(
                    blocked_candidates,
                    key=lambda item: self._iou(self._blocked_bbox, item.bbox),
                )
                blocked_area = max(
                    0.0,
                    (float(self._blocked_bbox[2]) - float(self._blocked_bbox[0]))
                    * (float(self._blocked_bbox[3]) - float(self._blocked_bbox[1])),
                )
                current_area = max(
                    0.0,
                    (float(current.bbox[2]) - float(current.bbox[0]))
                    * (float(current.bbox[3]) - float(current.bbox[1])),
                )
                area_ratio = (
                    max(current_area, blocked_area) / min(current_area, blocked_area)
                    if min(current_area, blocked_area) > 0.0
                    else 1.0
                )
                if area_ratio >= max(
                    1.0, float(self.config.blocked_rearm_area_ratio)
                ):
                    # Re-arm only for a valid, large scale transition. The
                    # normal gate will still require its configured hold and
                    # never grants identity or motor authority by itself.
                    self._blocked_bbox = None
                    self._blocked_missing_frames = 0
                else:
                    # A person can move a large fraction of the frame between two
                    # detector results. Once the gate is already blocked, that
                    # current box is still useful for active candidate centering;
                    # withholding it would turn a visible moving target into a
                    # false missing frame and keep the old search direction alive.
                    matched_bbox = current.bbox
                    matched_score = float(current.score)
                    if self._iou(self._blocked_bbox, matched.bbox) >= float(
                        self.config.consistency_iou
                    ):
                        self._blocked_bbox = matched.bbox
                        matched_bbox = matched.bbox
                        matched_score = float(matched.score)
                    return SearchCandidateGateDecision(
                        source="blocked",
                        reason="candidate_already_observed",
                        score=matched_score,
                        bbox=matched_bbox,
                    )
            if current is None:
                self._blocked_missing_frames += 1
                if self._blocked_missing_frames < max(
                    1, int(self.config.blocked_reset_missing_frames)
                ):
                    return SearchCandidateGateDecision(reason="candidate_block_reset_wait")
                self._blocked_bbox = None
                self._blocked_missing_frames = 0

        if current is not None and current_source == "formal":
            return self._start_hold(
                current, source="formal", defer_sec=defer_sec, timestamp=now
            )

        if current is None:
            self._probe_bbox = None
            self._probe_streak = 0
            return SearchCandidateGateDecision(reason="no_candidate")

        candidate = current
        if self._probe_bbox is not None and self._iou(
            self._probe_bbox, candidate.bbox
        ) >= float(self.config.consistency_iou):
            self._probe_streak += 1
        else:
            self._probe_streak = 1
        self._probe_bbox = candidate.bbox
        if self._probe_streak < max(1, int(self.config.probe_confirm_frames)):
            return SearchCandidateGateDecision(
                source="probe",
                reason=(
                    "probe_active_target_candidate_confirming"
                    if preferred_match
                    else "probe_candidate_confirming"
                ),
                score=candidate.score,
                bbox=candidate.bbox,
                probe_streak=self._probe_streak,
                preferred_target_match=preferred_match,
            )
        return self._start_hold(
            candidate, source="probe", defer_sec=defer_sec, timestamp=now
        )
