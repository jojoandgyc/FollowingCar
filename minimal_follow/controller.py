"""Pure control law for the first person-follow closed loop."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from .commands import MinimalFollowCommand

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class MinimalFollowConfig:
    target_distance_m: float = 1.4
    distance_deadband_m: float = 0.10
    distance_kp_percent_per_m: float = 30.0
    min_forward_percent: int = 8
    max_forward_percent: int = 20
    center_deadband_ratio: float = 0.08
    steering_delta_max_percent: int = 8


class MinimalFollowController:
    """No identity memory, search, reverse, prediction, or stale-command hold."""

    def __init__(self, config: MinimalFollowConfig) -> None:
        self.config = config

    def step(
        self,
        *,
        frame_width: int,
        bbox: Optional[BBox],
        distance_m: Optional[float],
        front_obstacle: bool,
    ) -> MinimalFollowCommand:
        cfg = self.config
        if front_obstacle:
            return MinimalFollowCommand.stop("front_ir")
        if bbox is None:
            return MinimalFollowCommand.stop("person_missing")
        if frame_width <= 0:
            return MinimalFollowCommand.stop("invalid_frame_width")
        if distance_m is None or not math.isfinite(float(distance_m)) or float(distance_m) <= 0.0:
            return MinimalFollowCommand.stop("distance_unavailable")

        error_m = float(distance_m) - float(cfg.target_distance_m)
        if error_m <= float(cfg.distance_deadband_m) + 1e-9:
            return MinimalFollowCommand.stop("target_distance_reached")

        base = int(round(error_m * float(cfg.distance_kp_percent_per_m)))
        base = max(int(cfg.min_forward_percent), min(int(cfg.max_forward_percent), base))
        x1, _y1, x2, _y2 = bbox
        x_ratio = ((float(x1) + float(x2)) * 0.5) / float(frame_width)
        lateral_error = x_ratio - 0.5
        if abs(lateral_error) <= float(cfg.center_deadband_ratio):
            return MinimalFollowCommand(base, base, "forward")

        normalized = min(
            1.0,
            (abs(lateral_error) - float(cfg.center_deadband_ratio))
            / max(1e-6, 0.5 - float(cfg.center_deadband_ratio)),
        )
        delta = max(1, int(round(normalized * int(cfg.steering_delta_max_percent))))
        inner = max(0, base - delta)
        if lateral_error < 0.0:
            return MinimalFollowCommand(inner, base, "steer_left")
        return MinimalFollowCommand(base, inner, "steer_right")
