"""Successful command evidence for PI continuity, never a motor lease."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ForwardExecutionAnchor:
    uid: int
    sample_timestamp: float
    rpm: float
    sent_at: float
    receipt: object
