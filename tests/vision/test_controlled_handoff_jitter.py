from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig


def _metadata(track_id: int, frame_index: int, bbox):
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return {
        "track_id": int(track_id),
        "frame_index": int(frame_index),
        "bbox": [x1, y1, x2, y2],
        "detector_bbox": [x1, y1, x2, y2],
        "center_x_ratio": (x1 + x2) / 1280.0,
        "detector_center_x_ratio": (x1 + x2) / 1280.0,
        "area_ratio": (x2 - x1) * (y2 - y1) / (640.0 * 480.0),
        "detector_area_ratio": (x2 - x1) * (y2 - y1) / (640.0 * 480.0),
        "area_units": "ratio",
        "geometry_source": "detector",
        "is_fresh": True,
    }


def test_sole_strong_handoff_bridges_two_frame_track_gap_with_small_yaw_residual():
    bank = IdentityBank(
        IdentityBankConfig(
            min_confidence=0.60,
            new_identity_confirm_frames=1,
            controlled_handoff_enable=True,
            controlled_handoff_confirm_frames=2,
            controlled_handoff_min_old_track_gap_frames=5,
            handoff_geometry_max_center_jump_ratio=0.30,
        )
    )
    feature = np.asarray([1.0, 0.0, 0.0], dtype="float32")
    query = np.asarray([0.8, 0.6, 0.0], dtype="float32")
    uid = bank.assign(
        track_id=6,
        feature=feature,
        confidence=0.95,
        area=0.20,
        frame_index=314,
        sample_metadata=_metadata(6, 314, (160, 40, 410, 390)),
    )

    # The replacement DeepSORT track appears after two missed frames.  Its
    # center shifts by roughly 0.26 of the image width, but remains a sole
    # strong match and should still use the two-frame confirmation chain.
    first = bank.assign(
        track_id=7,
        feature=query,
        confidence=0.92,
        area=0.20,
        frame_index=316,
        candidate_count=1,
        sample_metadata=_metadata(7, 316, (280, 40, 575, 475)),
    )
    first_reason = bank.last_assignments[7]["reason"]
    second = bank.assign(
        track_id=7,
        feature=query,
        confidence=0.92,
        area=0.20,
        frame_index=317,
        candidate_count=1,
        sample_metadata=_metadata(7, 317, (325, 40, 620, 475)),
    )

    assert uid == 1
    assert first == 0
    assert first_reason == "controlled_handoff_wait"
    assert second == uid
    assert bank.last_assignments[7]["reason"] == "controlled_handoff"
    assert bank.track_to_uid == {7: uid}
