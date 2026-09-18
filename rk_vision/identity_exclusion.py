"""Short-lived, geometry-backed evidence that two observations are different people.

This module never infers identity from appearance.  The caller supplies trusted
UIDs for current observations; only simultaneous, separated detector boxes can
create an exclusion.  Continuity keeps that evidence attached to the observed
person rather than to an old target anchor or a reusable detector-probe ID.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple


BBox = Tuple[float, float, float, float]


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _optional_int(value: Any) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError, OverflowError):
        return None


@dataclass
class _Observation:
    track_id: int
    bbox: BBox
    detector_bbox: BBox
    frame: int
    capture_id: Optional[int]
    timestamp: Optional[float]
    yaw: Optional[float]
    trusted_uid: int = 0
    identity_swap: bool = False
    witness_reference_capture_frame_id: Optional[int] = None
    witness_reference_frame_index: Optional[int] = None
    witness_geometry_valid: Optional[bool] = None
    exclusions: Dict[int, dict] = field(default_factory=dict)


class IdentityExclusionMemory:
    """Keep co-visible exclusions for continuous candidate tracklets only."""

    def __init__(self, camera_hfov_deg: float = 90.0) -> None:
        hfov = _finite(camera_hfov_deg)
        self.camera_hfov_deg = max(1.0, hfov if hfov is not None else 90.0)
        self._observations: Dict[int, _Observation] = {}
        self._last_frame = -1

    def reset(self) -> None:
        self._observations.clear()
        self._last_frame = -1

    def invalidate_witness(self, uid: int, track_id: int, *, after_frame: int) -> int:
        """Revoke this witness's evidence created after its last trusted frame.

        Search all observations because a candidate may have inherited its
        exclusion under a different raw track ID through geometry continuity.
        """
        uid, track_id, after_frame = int(uid), int(track_id), int(after_frame)
        removed = 0
        for observation in self._observations.values():
            evidence = observation.exclusions.get(uid)
            if (
                evidence is not None
                and evidence["reference_track_id"] == track_id
                and evidence["source_frame"] > after_frame
            ):
                del observation.exclusions[uid]
                removed += 1
        return removed

    @staticmethod
    def _expired(
        previous: _Observation, frame: int, timestamp: Optional[float],
        *, max_frames: int = 15, max_seconds: float = 1.0,
    ) -> bool:
        gap = int(frame) - previous.frame
        if gap < 0 or gap > max_frames:
            return True
        if timestamp is not None and previous.timestamp is not None:
            delta = timestamp - previous.timestamp
            return delta < 0.0 or delta > max_seconds
        return False

    def _continuous(self, previous: _Observation, current: _Observation, *, strict: bool) -> bool:
        px1, py1, px2, py2 = previous.bbox
        cx1, cy1, cx2, cy2 = current.bbox
        if previous.yaw is not None and current.yaw is not None:
            # Positive encoder yaw is a right turn: a stationary person's
            # image position moves left, not in either arbitrary direction.
            shift = -(current.yaw - previous.yaw) / self.camera_hfov_deg
            px1 += shift
            px2 += shift
        pw, ph = px2 - px1, py2 - py1
        cw, ch = cx2 - cx1, cy2 - cy1
        dx = abs((px1 + px2 - cx1 - cx2) * 0.5)
        dy = abs((py1 + py2 - cy1 - cy2) * 0.5)
        area_ratio = min(pw * ph, cw * ch) / max(pw * ph, cw * ch)
        if not strict:
            return dx <= 0.25 and dy <= 0.20 and area_ratio >= 0.25
        if dx > 0.025 or dy > 0.04 or area_ratio < 0.65:
            return False
        if min(pw, cw) / max(pw, cw) < 0.65 or min(ph, ch) / max(ph, ch) < 0.65:
            return False
        intersection = max(0.0, min(px2, cx2) - max(px1, cx1)) * max(
            0.0, min(py2, cy2) - max(py1, cy1)
        )
        union = pw * ph + cw * ch - intersection
        return union > 0.0 and intersection / union >= 0.50

    @staticmethod
    def _separated(first: _Observation, second: _Observation) -> bool:
        if first.capture_id is None or first.capture_id != second.capture_id:
            return False
        ax1, ay1, ax2, ay2 = first.bbox
        bx1, by1, bx2, by2 = second.bbox
        # A positive gap excludes overlap, containment, and nested duplicate
        # boxes without depending on a person's size or image-edge quality.
        horizontal_gap = max(bx1 - ax2, ax1 - bx2)
        vertical_gap = max(by1 - ay2, ay1 - by2)
        return horizontal_gap >= 0.02 or vertical_gap >= 0.02

    @staticmethod
    def _parse(item: dict, frame: int, width: float, height: float) -> Optional[_Observation]:
        if not item.get("is_fresh", False) or item.get("duplicate"):
            return None
        try:
            track_id = int(item["raw_track_id"])
            raw = tuple(_finite(value) for value in item["detector_bbox"])
            trusted_uid = max(0, int(item.get("trusted_uid", 0)))
            capture = item.get("capture_frame_id")
            capture_id = None if capture is None else int(capture)
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if len(raw) != 4 or any(value is None for value in raw):
            return None
        x1, y1, x2, y2 = raw
        if x2 <= x1 or y2 <= y1:
            return None
        return _Observation(
            track_id=track_id,
            bbox=(x1 / width, y1 / height, x2 / width, y2 / height),
            detector_bbox=raw,
            frame=frame,
            capture_id=capture_id,
            timestamp=_finite(item.get("capture_timestamp")),
            yaw=_finite(item.get("integrated_yaw_deg")),
            trusted_uid=0 if item.get("identity_swap") else trusted_uid,
            identity_swap=bool(item.get("identity_swap")),
            witness_reference_capture_frame_id=_optional_int(
                item.get("witness_reference_capture_frame_id")
            ),
            witness_reference_frame_index=_optional_int(
                item.get("witness_reference_frame_index")
            ),
            witness_geometry_valid=(
                None if item.get("witness_geometry_valid") is None
                else bool(item["witness_geometry_valid"])
            ),
        )

    def observe_frame(
        self, *, frame_index: int, observations: Sequence[dict], width: int,
        height: Optional[int],
    ) -> None:
        frame = int(frame_index)
        if frame < self._last_frame:
            return
        self._last_frame = frame
        valid_width, valid_height = _finite(width), _finite(height)
        if valid_width is None or valid_height is None or min(valid_width, valid_height) <= 0:
            return
        parsed: Dict[int, _Observation] = {}
        ambiguous_ids = set()
        for item in observations:
            current = self._parse(item, frame, valid_width, valid_height)
            if current is None:
                continue
            if current.track_id in parsed:
                ambiguous_ids.add(current.track_id)
            parsed[current.track_id] = current
        for track_id in ambiguous_ids:
            parsed.pop(track_id, None)
        timestamps = [item.timestamp for item in parsed.values() if item.timestamp is not None]
        now = max(timestamps) if timestamps else None
        self._observations = {
            track_id: previous for track_id, previous in self._observations.items()
            if not self._expired(previous, frame, now)
        }
        previous_frame = dict(self._observations)
        replacement_matches: Dict[int, list] = {}
        donor_matches: Dict[int, list] = {}
        for track_id, current in parsed.items():
            for donor_id, donor in previous_frame.items():
                if donor_id in ambiguous_ids:
                    continue
                if self._expired(donor, frame, current.timestamp, max_frames=2, max_seconds=0.35):
                    continue
                if frame <= donor.frame or not self._continuous(donor, current, strict=True):
                    continue
                # Match the complete frame, including still-present IDs.
                # DeepSORT can swap two IDs without either ID disappearing.
                replacement_matches.setdefault(track_id, []).append(donor_id)
                donor_matches.setdefault(donor_id, []).append(track_id)
        transferred = set()
        associations = {}
        for track_id, matches in replacement_matches.items():
            if len(matches) != 1 or len(donor_matches[matches[0]]) != 1:
                continue
            donor_id = matches[0]
            associations[track_id] = donor_id
        for track_id, current in parsed.items():
            donor_id = associations.get(track_id)
            previous = previous_frame.get(track_id)
            if donor_id is None and (
                previous is not None and not current.identity_swap
                and track_id not in associations.values()
                and not any(i != track_id for i in replacement_matches.get(track_id, []))
                and self._continuous(previous, current, strict=False)
            ):
                # Preserve the legacy same-ID short-gap path only if a
                # different local observation does not contradict it.
                donor_id = track_id
            if donor_id is None:
                continue
            parsed[track_id].exclusions = {
                uid: dict(evidence) for uid, evidence in previous_frame[donor_id].exclusions.items()
            }
            if donor_id != track_id:
                current.trusted_uid = 0  # transferred geometry is not a new witness
                for evidence in current.exclusions.values():
                    evidence.update(
                        association_reason="unique_geometry_track_transfer",
                        transferred_from_track_id=donor_id,
                        transfer_capture_frame_id=current.capture_id,
                    )
                transferred.add(donor_id)
        # Pair generation uses the complete snapshot, before any assignment
        # can change a mapped UID. Reversing input order has no effect.
        for candidate in sorted(parsed.values(), key=lambda item: item.track_id):
            # Swap-flagged raw crops may CARRY an existing negative label,
            # but neither create new exclusions nor serve as trusted witnesses.
            if candidate.identity_swap:
                continue
            for reference in sorted(parsed.values(), key=lambda item: item.track_id):
                uid = reference.trusted_uid
                if candidate.track_id == reference.track_id or uid <= 0:
                    continue
                if self._separated(reference, candidate):
                    candidate.exclusions[uid] = {
                        "reason": "co_visible_distinct_person",
                        "uid": uid,
                        "source_capture_frame_id": reference.capture_id,
                        "source_frame": frame,
                        "reference_track_id": reference.track_id,
                        "reference_bbox": list(reference.detector_bbox),
                        "source_candidate_bbox": list(candidate.detector_bbox),
                        "witness_reference_capture_frame_id": reference.witness_reference_capture_frame_id,
                        "witness_reference_frame_index": reference.witness_reference_frame_index,
                        "witness_geometry_valid": reference.witness_geometry_valid,
                    }
        for donor_id in transferred:
            if donor_id not in parsed:
                self._observations.pop(donor_id, None)
        for track_id in ambiguous_ids:
            self._observations.pop(track_id, None)
        self._observations.update(parsed)

    def exclusion_for(
        self, track_id: int, uid: int, *, frame_index: int,
        capture_timestamp: Optional[float] = None,
    ) -> Optional[dict]:
        current = self._observations.get(int(track_id))
        if current is None or self._expired(current, int(frame_index), _finite(capture_timestamp)):
            return None
        evidence = current.exclusions.get(int(uid))
        if evidence is None:
            return None
        return {
            **evidence,
            "reference_bbox": list(evidence["reference_bbox"]),
            "source_candidate_bbox": list(evidence["source_candidate_bbox"]),
            "candidate_track_id": int(track_id),
            "bbox": list(current.detector_bbox),
            "last_frame": current.frame,
            "last_capture_frame_id": current.capture_id,
            "last_capture_timestamp": current.timestamp,
        }
