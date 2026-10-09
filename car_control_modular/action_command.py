"""Immutable queue provenance; publication is distinct from motor execution."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ActionCommandSnapshot:
    action: int
    revision: int
    enqueued_at: float
    control_frame: int
    capture_frame_id: int
    capture_timestamp: float
    reason: str
    soft_stop: bool
    protected_stop: bool = False
    source_module: str = "unknown"
    uid: Optional[int] = None
    # Ordinary controller wait/settle requests are not external emergency
    # latches. Unknown historical protected stops remain fail-closed.
    stop_origin: str = "unknown"

    def __int__(self):
        return self.action


@dataclass(frozen=True)
class SearchReacquireBrakeRequest:
    capture_frame_id: int
    capture_timestamp: float
    requested_at: float
    reason: str
