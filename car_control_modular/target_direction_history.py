"""Capture-ordered target evidence used to choose a lost-target search side."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class TargetFrameEvidence:
    capture_frame_id: int
    timestamp: float
    state: str
    target_id: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    center_x_ratio: Optional[float] = None
    edge_side: str = "none"
    motion_direction: str = "unknown"
    velocity_ratio_s: Optional[float] = None
    vehicle_yaw_deg: Optional[float] = None
    confidence: float = 0.0
    reason: str = "none"


@dataclass(frozen=True)
class TargetDirectionDecision:
    direction: Optional[str]
    confidence: float
    reason: str
    missing_frames: int
    last_visible_capture_frame_id: Optional[int] = None
    visible_samples: int = 0


class TargetDirectionHistory:
    """A small upsertable timeline keyed by camera capture sequence.

    Entries describe acquisition order, not completion order. A stale inference
    therefore updates its original capture slot and can never masquerade as a
    current target observation.
    """

    def __init__(
        self,
        *,
        max_frames: int = 60,
        lookback_samples: int = 6,
        max_capture_gap: int = 12,
        edge_margin_ratio: float = 0.04,
        outer_center_ratio: float = 0.60,
        min_motion_ratio: float = 0.015,
        min_consistent_steps: int = 1,
    ) -> None:
        self.max_frames = max(12, int(max_frames))
        self.lookback_samples = max(2, int(lookback_samples))
        self.max_capture_gap = max(3, int(max_capture_gap))
        self.edge_margin_ratio = max(0.0, min(0.20, float(edge_margin_ratio)))
        self.outer_center_ratio = max(0.51, min(0.90, float(outer_center_ratio)))
        self.min_motion_ratio = max(0.0, min(0.25, float(min_motion_ratio)))
        self.min_consistent_steps = max(1, int(min_consistent_steps))
        self._entries: List[TargetFrameEvidence] = []
        # Preserve visible geometry outside the bounded capture timeline so a
        # delayed search cannot erase the final exit trace.
        self._visible_entries: List[TargetFrameEvidence] = []
        self._cached_decision: Optional[TargetDirectionDecision] = None

    @property
    def entries(self) -> Tuple[TargetFrameEvidence, ...]:
        return tuple(self._entries)

    def record(self, evidence: TargetFrameEvidence) -> None:
        capture_id = int(evidence.capture_frame_id)
        if capture_id <= 0:
            return
        replacement = TargetFrameEvidence(
            capture_frame_id=capture_id,
            timestamp=float(evidence.timestamp),
            state=str(evidence.state),
            target_id=(None if evidence.target_id is None else int(evidence.target_id)),
            bbox=evidence.bbox,
            center_x_ratio=evidence.center_x_ratio,
            edge_side=str(evidence.edge_side),
            motion_direction=str(evidence.motion_direction),
            velocity_ratio_s=(
                None if evidence.velocity_ratio_s is None else float(evidence.velocity_ratio_s)
            ),
            vehicle_yaw_deg=(
                None if evidence.vehicle_yaw_deg is None else float(evidence.vehicle_yaw_deg)
            ),
            confidence=max(0.0, min(1.0, float(evidence.confidence))),
            reason=str(evidence.reason),
        )
        for index, item in enumerate(self._entries):
            if item.capture_frame_id == capture_id:
                self._entries[index] = replacement
                break
        else:
            self._entries.append(replacement)
            self._entries.sort(key=lambda item: item.capture_frame_id)
        visible_replaced = False
        for index, item in enumerate(self._visible_entries):
            if item.capture_frame_id == capture_id:
                visible_replaced = True
                if replacement.state == "visible":
                    self._visible_entries[index] = replacement
                else:
                    del self._visible_entries[index]
                break
        if not visible_replaced and replacement.state == "visible":
            self._visible_entries.append(replacement)
            self._visible_entries.sort(key=lambda item: item.capture_frame_id)
        if replacement.state == "visible":
            # A fresh visible slot starts a new loss episode.
            self._cached_decision = None
        visible_limit = max(self.max_frames, self.lookback_samples * 4)
        if len(self._visible_entries) > visible_limit:
            del self._visible_entries[: len(self._visible_entries) - visible_limit]
        if len(self._entries) > self.max_frames:
            del self._entries[: len(self._entries) - self.max_frames]

    def record_unknown(self, capture_frame_id: int, timestamp: float, reason: str) -> None:
        self.record(
            TargetFrameEvidence(
                capture_frame_id=int(capture_frame_id),
                timestamp=float(timestamp),
                state="unknown",
                reason=str(reason),
            )
        )

    def record_stale(self, capture_frame_id: int, timestamp: float) -> None:
        self.record(
            TargetFrameEvidence(
                capture_frame_id=int(capture_frame_id),
                timestamp=float(timestamp),
                state="stale",
                reason="result_age_exceeded",
            )
        )

    def record_missing(self, capture_frame_id: int, timestamp: float) -> None:
        self.record(
            TargetFrameEvidence(
                capture_frame_id=int(capture_frame_id),
                timestamp=float(timestamp),
                state="missing",
                reason="fresh_detector_missing",
            )
        )

    def record_visible(
        self,
        capture_frame_id: int,
        timestamp: float,
        *,
        target_id: int,
        bbox: Iterable[float],
        frame_width: int,
        confidence: float,
        vehicle_yaw_deg: Optional[float] = None,
        reason: str = "reliable_target",
    ) -> None:
        values = tuple(float(value) for value in bbox)
        if len(values) != 4 or int(frame_width) <= 0:
            return
        x1, y1, x2, y2 = values
        width = float(frame_width)
        center_ratio = max(0.0, min(1.0, (x1 + x2) / (2.0 * float(frame_width))))
        touches_left = x1 / width <= self.edge_margin_ratio
        touches_right = x2 / width >= 1.0 - self.edge_margin_ratio
        edge_side = (
            "none"
            if touches_left and touches_right
            else "left"
            if touches_left
            else "right"
            if touches_right
            else "none"
        )
        previous = next(
            (
                item
                for item in reversed(self._visible_entries)
                if item.state == "visible"
                and item.target_id == int(target_id)
                and item.center_x_ratio is not None
                and item.capture_frame_id < int(capture_frame_id)
            ),
            None,
        )
        velocity_ratio_s = None
        motion_direction = "unknown"
        if previous is not None:
            dt = float(timestamp) - float(previous.timestamp)
            delta = center_ratio - float(previous.center_x_ratio)
            if dt > 0.0:
                velocity_ratio_s = delta / dt
            motion_direction = (
                "hold"
                if abs(delta) < self.min_motion_ratio
                else "left"
                if delta < 0.0
                else "right"
            )
        self.record(
            TargetFrameEvidence(
                capture_frame_id=int(capture_frame_id),
                timestamp=float(timestamp),
                state="visible",
                target_id=int(target_id),
                bbox=(x1 / width, y1, x2 / width, y2),
                center_x_ratio=center_ratio,
                edge_side=edge_side,
                motion_direction=motion_direction,
                velocity_ratio_s=velocity_ratio_s,
                vehicle_yaw_deg=vehicle_yaw_deg,
                confidence=float(confidence),
                reason=str(reason),
            )
        )

    def _recent_missing_count(self) -> int:
        count = 0
        for item in reversed(self._entries):
            if item.state != "missing":
                break
            count += 1
        return count

    def resolve(self, *, required_missing_frames: int) -> TargetDirectionDecision:
        required = max(1, int(required_missing_frames))
        missing_count = self._recent_missing_count()
        if missing_count < required:
            return TargetDirectionDecision(
                direction=None,
                confidence=0.0,
                reason="missing_confirmation_pending",
                missing_frames=missing_count,
            )

        last_missing = self._entries[-1]
        first_missing_index = len(self._entries) - missing_count
        first_missing_capture_id = (
            int(self._entries[first_missing_index].capture_frame_id)
            if first_missing_index > 0
            else int(last_missing.capture_frame_id) - missing_count + 1
        )
        lower_capture_id = int(last_missing.capture_frame_id) - self.max_capture_gap
        visible = []
        if first_missing_index > 0:
            visible = [
                item
                for item in self._entries[:first_missing_index]
                if item.state == "visible"
                and item.center_x_ratio is not None
                and int(item.capture_frame_id) >= lower_capture_id
            ][-self.lookback_samples :]
        if not visible:
            # The controller may service the loss only after the ring has
            # rolled past the final visible frame. Recover that episode's
            # visible samples from the side buffer.
            visible = [
                item
                for item in self._visible_entries
                if item.center_x_ratio is not None
                and int(item.capture_frame_id) < first_missing_capture_id
            ][-self.lookback_samples :]
        if not visible:
            cached = self._cached_decision
            if (
                cached is not None
                and cached.direction in ("left", "right")
                and (
                    cached.last_visible_capture_frame_id is None
                    or cached.last_visible_capture_frame_id < first_missing_capture_id
                )
            ):
                return TargetDirectionDecision(
                    cached.direction,
                    cached.confidence,
                    "cached_exit_direction",
                    missing_count,
                    cached.last_visible_capture_frame_id,
                    cached.visible_samples,
                )
            return TargetDirectionDecision(None, 0.0, "no_recent_reliable_visible", missing_count)

        last = visible[-1]
        last_target = last.target_id
        same_target = [item for item in visible if item.target_id == last_target]
        # Search direction is a spatial decision.  Once the target is lost,
        # motion velocity and the sign of the final step are stale control
        # signals; they must not veto the side selected from the capture
        # timeline.  Classify every usable sample around the image midpoint,
        # then use the most recent sample as the tie breaker.
        midpoint = 0.5
        side_votes = [
            "left" if float(item.center_x_ratio) < midpoint else "right"
            for item in same_target
            if item.center_x_ratio is not None
        ]
        if not side_votes:
            return TargetDirectionDecision(
                None,
                0.0,
                "no_side_geometry",
                missing_count,
                last.capture_frame_id,
                len(same_target),
            )
        latest_side = side_votes[-1]
        left_votes = side_votes.count("left")
        right_votes = side_votes.count("right")
        side = latest_side if left_votes == right_votes else (
            "left" if left_votes > right_votes else "right"
        )
        side_votes_count = max(left_votes, right_votes)
        confidence = min(
            0.95,
            0.60 + 0.08 * min(3, side_votes_count) +
            (0.08 if last.edge_side == side else 0.0),
        )
        reason = "capture_timeline_exit_side"

        decision = TargetDirectionDecision(
            side,
            confidence,
            reason,
            missing_count,
            last.capture_frame_id,
            len(same_target),
        )
        self._cached_decision = decision
        return decision

    def latest_reliable_side(self) -> TargetDirectionDecision:
        """Return the side of the newest reliable visible capture.

        The controller calls this only after its fresh-loss threshold has
        elapsed. It prevents delayed or unknown worker slots from trapping the
        vehicle in an endless direction-probe state.
        """
        visible = [
            item
            for item in self._visible_entries
            if item.state == "visible" and item.center_x_ratio is not None
        ]
        if not visible:
            return TargetDirectionDecision(None, 0.0, "no_recent_reliable_visible", 0)
        latest = visible[-1]
        side = "left" if float(latest.center_x_ratio) < 0.5 else "right"
        confidence = min(0.95, max(0.50, float(latest.confidence)))
        return TargetDirectionDecision(
            side,
            confidence,
            "latest_reliable_capture_side",
            0,
            int(latest.capture_frame_id),
            1,
        )

    def latest_visible_evidence(self) -> Optional[TargetFrameEvidence]:
        """Return the newest target-owned geometry without changing history state."""
        return self._visible_entries[-1] if self._visible_entries else None
