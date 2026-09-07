"""RK3588 vision model pipeline.

This package intentionally does not open cameras.  Callers pass frames in and
receive tracked person records out.
"""

from .frames import FramePacket
from .pipeline import RKNNVisionConfig, RKNNVisionPipeline, SearchCandidateEvidence
from .tracker import TrackRecord
from .yolo11 import Detection

__all__ = [
    "Detection",
    "FramePacket",
    "RKNNVisionConfig",
    "RKNNVisionPipeline",
    "SearchCandidateEvidence",
    "TrackRecord",
]
