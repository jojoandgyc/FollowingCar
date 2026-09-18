#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Short-horizon target distance fusion for visual person following."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from statistics import median
from typing import Optional

from .control_types import DepthJumpConfirmation, PersonTarget, SteeringFeedback


@dataclass(frozen=True)
class DistanceFusionConfig:
    enabled: bool = False
    radar_median_window: int = 3
    visual_weight: float = 0.80
    encoder_wheel_circumference_m: float = 0.60
    encoder_max_step_m: float = 0.12
    bbox_min_height_ratio: float = 0.08
    bbox_max_height_ratio: float = 0.90
    radar_recovery_alpha: float = 0.35
    max_distance_increase_mps: float = 2.0
    hold_max_sec: float = 2.50
    min_confidence: float = 0.40
    min_distance_m: float = 0.03
    max_distance_m: float = 10.0
    # Fresh far returns are accepted only after consecutive confirmation. A
    # sudden closer return remains immediate for collision safety.
    fresh_far_jump_m: float = 0.0
    fresh_far_jump_confirm_frames: int = 3
    # Only the Astra instance may consume an upstream spatial/temporal proof.
    allow_depth_jump_confirmation: bool = False
    depth_confirmation_max_age_sec: float = 0.25


@dataclass(frozen=True)
class DistanceFusionResult:
    distance_m: Optional[float]
    mode: str
    confidence: float
    radar_distance_m: Optional[float] = None
    visual_distance_m: Optional[float] = None
    encoder_delta_m: float = 0.0
    anchor_age_sec: Optional[float] = None
    depth_confirmation_used: bool = False
    depth_confirmation_reason: str = "none"
    sample_timestamp_rejected: bool = False
    sample_timestamp_reason: str = "legacy"


class VisionRadarEncoderDistanceFusion:
    """Use radar as an absolute anchor and vision/encoders during short gaps."""

    def __init__(self, config: DistanceFusionConfig) -> None:
        self.config = config
        window = max(1, int(config.radar_median_window))
        self._radar_samples = deque(maxlen=window)
        # Keep the replay watermark across target/cache resets: resetting an
        # anchor does not turn an already consumed sensor sample into a new one.
        self._last_depth_confirmation_sample_ts: Optional[float] = None
        self.reset()

    def reset(self) -> None:
        self._active_track_id: Optional[int] = None
        self._anchor_distance_m: Optional[float] = None
        self._anchor_bbox_height_px: Optional[float] = None
        self._last_distance_m: Optional[float] = None
        self._last_mode = "unavailable"
        self._last_update_ts: Optional[float] = None
        self._last_anchor_ts: Optional[float] = None
        self._last_accepted_depth_sample_ts: Optional[float] = None
        # Exact-sample deduplication, not a latest-attempt watermark: an
        # unseen RGB-aligned sample may precede a newer unsuccessful attempt.
        self._attempted_depth_sample_ts = deque(maxlen=64)
        self._last_feedback_ts: Optional[float] = None
        self._radar_samples.clear()
        self._pending_far_distance_m: Optional[float] = None
        self._pending_far_count = 0
        self._pending_far_sample_ts: Optional[float] = None

    def _clamp_distance(self, distance_m: float) -> float:
        low = max(0.01, float(self.config.min_distance_m))
        high = max(low, float(self.config.max_distance_m))
        return max(low, min(high, float(distance_m)))

    def _bbox_height(self, target: PersonTarget, frame_height: int) -> Optional[float]:
        if frame_height <= 0:
            return None
        _x1, y1, _x2, y2 = target.bbox
        height = max(0.0, float(y2) - float(y1))
        ratio = height / float(frame_height)
        if not math.isfinite(ratio):
            return None
        if ratio < max(0.01, float(self.config.bbox_min_height_ratio)):
            return None
        if ratio > min(0.99, float(self.config.bbox_max_height_ratio)):
            return None
        return height

    def _consume_encoder_delta(self, feedback: Optional[SteeringFeedback]) -> tuple[float, bool]:
        if (
            feedback is None
            or not feedback.trustworthy
            or float(self.config.encoder_wheel_circumference_m) <= 0.0
        ):
            return 0.0, False
        sample_ts = float(feedback.timestamp)
        if not math.isfinite(sample_ts):
            return 0.0, False
        previous_ts = self._last_feedback_ts
        self._last_feedback_ts = sample_ts
        if previous_ts is None or sample_ts <= previous_ts:
            return 0.0, True
        dt = sample_ts - previous_ts
        if dt > 0.50:
            return 0.0, False
        forward_rpm = 0.5 * (
            float(feedback.left_forward_rpm) + float(feedback.right_forward_rpm)
        )
        if not math.isfinite(forward_rpm):
            return 0.0, False
        delta_m = (
            forward_rpm
            * float(self.config.encoder_wheel_circumference_m)
            * dt
            / 60.0
        )
        limit = max(0.0, float(self.config.encoder_max_step_m))
        if limit > 0.0:
            delta_m = max(-limit, min(limit, delta_m))
        return float(delta_m), True

    def _begin_target(self, track_id: int) -> None:
        if self._active_track_id == int(track_id):
            return
        self.reset()
        self._active_track_id = int(track_id)

    def _fresh_radar_result(
        self,
        radar_distance_m: float,
        bbox_height: Optional[float],
        now: float,
        *,
        depth_confirmed: bool = False,
        sample_timestamp: Optional[float] = None,
    ) -> DistanceFusionResult:
        raw = self._clamp_distance(radar_distance_m)

        last = self._last_distance_m
        far_jump_limit = max(0.0, float(self.config.fresh_far_jump_m))
        far_jump = bool(
            last is not None
            and far_jump_limit > 0.0
            and raw - float(last) > far_jump_limit
        )
        if far_jump and not depth_confirmed:
            pending = self._pending_far_distance_m
            tolerance = max(0.15, far_jump_limit * 0.35)
            if pending is not None and abs(raw - float(pending)) <= tolerance:
                self._pending_far_count += 1
            else:
                self._pending_far_distance_m = raw
                self._pending_far_count = 1
            self._pending_far_sample_ts = sample_timestamp
            required = max(2, int(self.config.fresh_far_jump_confirm_frames))
            if self._pending_far_count < required:
                anchor_age = (
                    None
                    if self._last_anchor_ts is None
                    else max(0.0, float(now) - float(self._last_anchor_ts))
                )
                self._last_mode = "radar_jump_pending"
                self._last_update_ts = float(now)
                return DistanceFusionResult(
                    distance_m=last,
                    mode="radar_jump_pending",
                    confidence=min(0.45, float(self.config.min_confidence)),
                    radar_distance_m=raw,
                    anchor_age_sec=anchor_age,
                )
            self._pending_far_distance_m = None
            self._pending_far_count = 0
            self._pending_far_sample_ts = None
        else:
            # An older ordinary observation may recover a trusted anchor,
            # but cannot erase confirmation evidence from a newer sample.
            preserve_newer_pending = bool(
                sample_timestamp is not None
                and self._pending_far_sample_ts is not None
                and sample_timestamp < self._pending_far_sample_ts
            )
            if not preserve_newer_pending:
                self._pending_far_distance_m = None
                self._pending_far_count = 0
                self._pending_far_sample_ts = None

        if depth_confirmed:
            self._radar_samples.clear()
        self._radar_samples.append(raw)
        filtered = float(median(self._radar_samples))

        # A suddenly closer return may be a collision risk. Never let the
        # median or recovery blend hide it; farther corrections are smoothed.
        if self._last_distance_m is not None and raw < self._last_distance_m:
            filtered = raw
        mode = "radar_depth_confirmed" if depth_confirmed else "radar"
        used = filtered
        if (
            not depth_confirmed
            and self._last_distance_m is not None
            and self._last_mode in ("visual", "visual_encoder", "radar_hold")
            and filtered > self._last_distance_m
            # Timestamped Depth recovery must not blend with an expired
            # prediction. The far-jump gate above remains authoritative;
            # legacy callers and live, bounded holds retain their smoothing.
            and (
                sample_timestamp is None
                or (
                    self._last_anchor_ts is not None
                    and 0.0 <= float(now) - self._last_anchor_ts
                    <= max(0.01, float(self.config.hold_max_sec))
                )
            )
        ):
            alpha = max(0.05, min(1.0, float(self.config.radar_recovery_alpha)))
            used = self._last_distance_m + alpha * (filtered - self._last_distance_m)
            mode = "radar_recover"

        used = self._clamp_distance(used)
        self._anchor_distance_m = used
        # A temporarily clipped frame must not erase the last valid visual
        # anchor. It can still be useful when the next frame becomes valid.
        if bbox_height is not None:
            self._anchor_bbox_height_px = bbox_height
        self._last_distance_m = used
        self._last_mode = mode
        self._last_update_ts = float(now)
        self._last_anchor_ts = float(now) if sample_timestamp is None else sample_timestamp
        if sample_timestamp is not None:
            self._last_accepted_depth_sample_ts = sample_timestamp
        return DistanceFusionResult(
            distance_m=used,
            mode=mode,
            confidence=1.0,
            radar_distance_m=raw,
            anchor_age_sec=max(0.0, float(now) - self._last_anchor_ts),
        )

    def _depth_sample_timeline_reason(
        self, sample_timestamp: float, radar_distance_m: float, now: float,
    ) -> str:
        """Check sample identity/order before it can mutate distance state."""
        try:
            stamp = float(sample_timestamp)
            age = float(now) - stamp
            if not math.isfinite(stamp) or not math.isfinite(age) or stamp <= 0.0:
                return "invalid_timestamp"
            if not 0.0 <= age <= float(self.config.depth_confirmation_max_age_sec):
                return "stale_or_future"
        except (TypeError, ValueError, OverflowError):
            return "invalid_timestamp"
        accepted = self._last_accepted_depth_sample_ts
        if accepted is not None and stamp <= accepted:
            return "sample_reused" if stamp == accepted else "before_accepted_anchor"
        if stamp in self._attempted_depth_sample_ts:
            return "sample_reused"
        self._attempted_depth_sample_ts.append(stamp)
        pending_stamp = self._pending_far_sample_ts
        far_limit = max(0.0, float(self.config.fresh_far_jump_m))
        needs_far_confirmation = bool(
            self._last_distance_m is not None and far_limit > 0.0
            and self._clamp_distance(radar_distance_m) - self._last_distance_m > far_limit
        )
        if pending_stamp is not None and stamp <= pending_stamp and needs_far_confirmation:
            return "before_pending_sample"
        return "fresh"

    def _depth_confirmation_reason(
        self,
        proof: Optional[DepthJumpConfirmation],
        *,
        target: PersonTarget,
        radar_distance_m: Optional[float],
        radar_fresh: bool,
        sample_age_sec: Optional[float],
        now: float,
    ) -> str:
        if proof is None:
            return "none"
        if not self.config.allow_depth_jump_confirmation:
            return "disabled"
        if not isinstance(proof, DepthJumpConfirmation):
            return "invalid_type"
        if not radar_fresh or radar_distance_m is None:
            return "not_fresh"
        try:
            integer_values = (
                proof.target_id, target.track_id, proof.confirm_count,
                proof.required_confirm_frames, proof.region_count,
            )
            if any(
                isinstance(value, bool) or not math.isfinite(float(value))
                or int(value) != float(value) or int(value) <= 0
                for value in integer_values
            ):
                return "invalid_identity_or_count"
            if int(proof.target_id) != int(target.track_id):
                return "target_mismatch"
            if not (
                int(proof.confirm_count) >= int(proof.required_confirm_frames) >= 2
                and int(proof.region_count) >= 3
            ):
                return "insufficient_confirmation"
            if proof.kind not in ("far_jump", "reanchor", "large_clipped_consensus"):
                return "invalid_kind"
            distance = float(proof.distance_m)
            measured = float(radar_distance_m)
            stamp = float(proof.sample_timestamp)
            age = float(sample_age_sec)
            current = float(now)
            if not all(math.isfinite(value) for value in (distance, measured, stamp, age, current)):
                return "invalid_numeric"
            if distance <= 0.0 or measured <= 0.0 or stamp <= 0.0 or age < 0.0:
                return "invalid_numeric"
            if abs(distance - measured) > 1e-6:
                return "distance_mismatch"
            proof_age = current - stamp
            max_age = max(0.0, float(self.config.depth_confirmation_max_age_sec))
            if proof_age < 0.0 or proof_age > max_age or age > max_age:
                return "stale_or_future"
            # Astra and this layer take monotonic timestamps a few ms apart.
            if abs(proof_age - age) > 0.02:
                return "sample_timestamp_mismatch"
            if (
                self._last_depth_confirmation_sample_ts is not None
                and stamp <= self._last_depth_confirmation_sample_ts
            ):
                return "sample_reused"
        except (TypeError, ValueError, OverflowError):
            return "invalid_numeric"
        return "accepted"

    def _hold_result(
        self,
        radar_distance_m: float,
        bbox_height: Optional[float],
        encoder_delta_m: float,
        encoder_valid: bool,
        sample_age_sec: Optional[float],
        now: float,
    ) -> DistanceFusionResult:
        stale_radar = self._clamp_distance(radar_distance_m)
        anchor_age = (
            None
            if self._last_anchor_ts is None
            else max(0.0, float(now) - float(self._last_anchor_ts))
        )
        hold_sec = max(0.01, float(self.config.hold_max_sec))
        if anchor_age is not None and anchor_age > hold_sec:
            return DistanceFusionResult(
                distance_m=None,
                mode="expired",
                confidence=0.0,
                radar_distance_m=stale_radar,
                encoder_delta_m=encoder_delta_m if encoder_valid else 0.0,
                anchor_age_sec=anchor_age,
            )
        anchor = self._anchor_distance_m
        if anchor is None:
            anchor = stale_radar

        visual_distance: Optional[float] = None
        if (
            bbox_height is not None
            and self._anchor_bbox_height_px is not None
            and self._anchor_bbox_height_px > 0.0
        ):
            visual_distance = self._clamp_distance(
                anchor * self._anchor_bbox_height_px / bbox_height
            )

        previous = self._last_distance_m if self._last_distance_m is not None else stale_radar
        encoder_prediction = self._clamp_distance(previous - encoder_delta_m)
        if visual_distance is None and encoder_valid and abs(encoder_delta_m) > 1e-6:
            # Even when the person box is clipped, the encoder still tells us
            # how much distance the vehicle itself consumed. This prevents a
            # radar hold from freezing the old distance while driving forward.
            fused = encoder_prediction
            mode = "encoder_hold"
        elif visual_distance is None:
            fused = stale_radar
            mode = "radar_hold"
        elif encoder_valid:
            visual_weight = max(0.0, min(1.0, float(self.config.visual_weight)))
            fused = visual_weight * visual_distance + (1.0 - visual_weight) * encoder_prediction
            mode = "visual_encoder"
        else:
            fused = visual_distance
            mode = "visual"

        # A growing person box means the target is approaching and must take
        # effect immediately. Only implausibly fast distance increases are
        # limited, because those can incorrectly release a latched stop.
        elapsed = 0.0 if self._last_update_ts is None else max(0.0, float(now) - self._last_update_ts)
        if fused > previous and elapsed > 0.0:
            increase_limit = max(0.0, float(self.config.max_distance_increase_mps)) * elapsed
            fused = min(fused, previous + increase_limit)
        fused = self._clamp_distance(fused)

        age = max(0.0, float(sample_age_sec or 0.0), float(anchor_age or 0.0))
        progress = max(0.0, min(1.0, age / hold_sec))
        min_confidence = max(0.0, min(0.90, float(self.config.min_confidence)))
        confidence = 0.90 - (0.90 - min_confidence) * progress
        if visual_distance is None:
            confidence = min(confidence, min_confidence)
        elif not encoder_valid:
            confidence = max(min_confidence, confidence - 0.05)

        self._last_distance_m = fused
        self._last_mode = mode
        self._last_update_ts = float(now)
        return DistanceFusionResult(
            distance_m=fused,
            mode=mode,
            confidence=confidence,
            radar_distance_m=stale_radar,
            visual_distance_m=visual_distance,
            encoder_delta_m=encoder_delta_m if encoder_valid else 0.0,
            anchor_age_sec=anchor_age,
        )

    def update(
        self,
        *,
        target: Optional[PersonTarget],
        frame_height: int,
        radar_distance_m: Optional[float],
        radar_fresh: bool,
        sample_age_sec: Optional[float],
        steering_feedback: Optional[SteeringFeedback],
        now: float,
        jump_confirmation: Optional[DepthJumpConfirmation] = None,
        sample_timestamp: Optional[float] = None,
        discard_observation: bool = False,
    ) -> DistanceFusionResult:
        if not self.config.enabled:
            return DistanceFusionResult(
                distance_m=radar_distance_m,
                mode="disabled",
                confidence=1.0 if radar_fresh and radar_distance_m is not None else 0.0,
                radar_distance_m=radar_distance_m,
            )
        if target is None:
            self.reset()
            return DistanceFusionResult(None, "unavailable", 0.0)

        self._begin_target(int(target.track_id))
        bbox_height = self._bbox_height(target, frame_height)
        timeline_reason = "legacy" if sample_timestamp is None else "not_fresh"
        timeline_rejected = bool(discard_observation)
        if timeline_rejected:
            # Astra also rejects old/duplicate observations that carry only
            # a held anchor (no accepted sample_timestamp and no raw value).
            timeline_reason = "observation_discarded"
            radar_fresh = False
            radar_distance_m = None
        elif sample_timestamp is not None and radar_fresh and radar_distance_m is not None:
            timeline_reason = self._depth_sample_timeline_reason(
                sample_timestamp, float(radar_distance_m), float(now),
            )
            timeline_rejected = timeline_reason != "fresh"
            if timeline_rejected:
                # Never use the rejected sample as a fallback distance. Only
                # an existing trusted anchor may support a bounded hold.
                radar_fresh = False
                radar_distance_m = None
            else:
                sample_timestamp = float(sample_timestamp)
        confirmation_reason = self._depth_confirmation_reason(
            jump_confirmation,
            target=target,
            radar_distance_m=radar_distance_m,
            radar_fresh=radar_fresh,
            sample_age_sec=sample_age_sec,
            now=now,
        )
        if timeline_rejected:
            # A stale/duplicate observation must not mutate prediction via its
            # old RGB bbox either. Report the current cache and its real age;
            # neither measurement state nor encoder-consumption state moves.
            anchor_age = (
                None if self._last_anchor_ts is None
                else max(0.0, float(now) - self._last_anchor_ts)
            )
            available = bool(
                self._last_distance_m is not None and anchor_age is not None
                and anchor_age <= max(0.01, float(self.config.hold_max_sec))
            )
            return DistanceFusionResult(
                distance_m=self._last_distance_m if available else None,
                mode="radar_hold" if available else "expired" if anchor_age is not None else "unavailable",
                confidence=min(0.45, float(self.config.min_confidence)) if available else 0.0,
                radar_distance_m=self._anchor_distance_m,
                anchor_age_sec=anchor_age,
                depth_confirmation_reason=confirmation_reason,
                sample_timestamp_rejected=True,
                sample_timestamp_reason=timeline_reason,
            )
        encoder_delta_m, encoder_valid = self._consume_encoder_delta(steering_feedback)
        if radar_distance_m is not None and radar_fresh:
            confirmed = confirmation_reason == "accepted"
            if confirmed:
                self._last_depth_confirmation_sample_ts = float(jump_confirmation.sample_timestamp)
            result = self._fresh_radar_result(
                float(radar_distance_m), bbox_height, float(now),
                depth_confirmed=confirmed,
                sample_timestamp=sample_timestamp,
            )
            return replace(
                result,
                depth_confirmation_used=confirmed,
                depth_confirmation_reason=confirmation_reason,
                sample_timestamp_reason=timeline_reason,
            )
        if radar_distance_m is not None:
            result = self._hold_result(
                float(radar_distance_m),
                bbox_height,
                encoder_delta_m,
                encoder_valid,
                sample_age_sec,
                float(now),
            )
            return replace(
                result, depth_confirmation_reason=confirmation_reason,
                sample_timestamp_rejected=timeline_rejected,
                sample_timestamp_reason=timeline_reason,
            )
        # A missing current radar point is still recoverable when this target
        # has a previous radar anchor. Continue with visual/encoder prediction
        # instead of converting a short radar gap into an unavailable distance.
        hold_anchor = self._anchor_distance_m
        if hold_anchor is None:
            hold_anchor = self._last_distance_m
        if hold_anchor is not None:
            result = self._hold_result(
                float(hold_anchor),
                bbox_height,
                encoder_delta_m,
                encoder_valid,
                sample_age_sec,
                float(now),
            )
            return replace(
                result, depth_confirmation_reason=confirmation_reason,
                sample_timestamp_rejected=timeline_rejected,
                sample_timestamp_reason=timeline_reason,
            )
        return DistanceFusionResult(
            distance_m=None,
            mode="unavailable",
            confidence=0.0,
            encoder_delta_m=encoder_delta_m if encoder_valid else 0.0,
            depth_confirmation_reason=confirmation_reason,
            sample_timestamp_rejected=timeline_rejected,
            sample_timestamp_reason=timeline_reason,
        )
