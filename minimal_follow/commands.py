"""Small hardware-independent drive command vocabulary for minimal follow."""

from __future__ import annotations

from dataclasses import dataclass


FORWARD_STATE = 0x01
REVERSE_STATE = 0x02


@dataclass(frozen=True)
class MinimalFollowCommand:
    """Wheel magnitudes plus direction states understood by ``MssdMotorBackend``."""

    left_percent: int = 0
    right_percent: int = 0
    reason: str = "stop"
    left_state: int = FORWARD_STATE
    right_state: int = FORWARD_STATE

    @property
    def moving(self) -> bool:
        return self.left_percent > 0 or self.right_percent > 0

    @staticmethod
    def stop(reason: str) -> "MinimalFollowCommand":
        return MinimalFollowCommand(reason=reason)

    @staticmethod
    def rotate_left(percent: int, reason: str = "search_rotate_left") -> "MinimalFollowCommand":
        # This mapping matches the existing LZ30EMA action runtime:
        # left wheel reverse + right wheel forward = chassis turns left.
        speed = max(0, int(percent))
        return MinimalFollowCommand(
            left_percent=speed, right_percent=speed, reason=reason,
            left_state=REVERSE_STATE, right_state=FORWARD_STATE,
        )

    @staticmethod
    def rotate_right(percent: int, reason: str = "search_rotate_right") -> "MinimalFollowCommand":
        speed = max(0, int(percent))
        return MinimalFollowCommand(
            left_percent=speed, right_percent=speed, reason=reason,
            left_state=FORWARD_STATE, right_state=REVERSE_STATE,
        )
