"""Small ByteTrack-style motion tracker for the minimal follow runtime.

It intentionally owns only detector boxes and IDs.  Appearance is handled by
``identity.py`` so a temporary ID switch can never write an arbitrary person
into the target gallery.
"""

from __future__ import annotations

from dataclasses import dataclass

from .association import ReidCandidate, iou


@dataclass(frozen=True)
class ByteTrackConfig:
    high_confidence: float = 0.55
    low_confidence: float = 0.25
    match_iou: float = 0.25
    low_match_iou: float = 0.15
    max_lost_frames: int = 15
    min_confirmed_hits: int = 2


@dataclass
class _Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    score: float
    hits: int = 1
    missed: int = 0


class ByteTrack:
    """Deterministic high-then-low confidence association with stable IDs."""

    def __init__(self, config: ByteTrackConfig | None = None) -> None:
        self.config = config or ByteTrackConfig()
        self._tracks: list[_Track] = []
        self._next_id = 1

    @staticmethod
    def _greedy_match(tracks: list[_Track], detections: list[ReidCandidate], threshold: float):
        pairs = []
        for track_index, track in enumerate(tracks):
            for detection_index, detection in enumerate(detections):
                score = iou(track.bbox, detection.bbox)
                if score >= threshold:
                    pairs.append((score, track_index, detection_index))
        pairs.sort(reverse=True)
        used_tracks, used_detections, result = set(), set(), []
        for _, track_index, detection_index in pairs:
            if track_index in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_index)
            used_detections.add(detection_index)
            result.append((track_index, detection_index))
        return result, used_tracks, used_detections

    def update(self, detections: list[ReidCandidate]) -> list[ReidCandidate]:
        high = [item for item in detections if item.score >= self.config.high_confidence]
        low = [item for item in detections if self.config.low_confidence <= item.score < self.config.high_confidence]
        matches, used_track_indexes, used_high_indexes = self._greedy_match(self._tracks, high, self.config.match_iou)
        for track_index, detection_index in matches:
            track, detection = self._tracks[track_index], high[detection_index]
            track.bbox, track.score, track.hits, track.missed = detection.bbox, detection.score, track.hits + 1, 0

        unmatched_tracks = [track for index, track in enumerate(self._tracks) if index not in used_track_indexes]
        low_matches, low_track_indexes, _ = self._greedy_match(unmatched_tracks, low, self.config.low_match_iou)
        for local_track_index, detection_index in low_matches:
            track, detection = unmatched_tracks[local_track_index], low[detection_index]
            track.bbox, track.score, track.hits, track.missed = detection.bbox, detection.score, track.hits + 1, 0
        matched_low_tracks = {id(unmatched_tracks[index]) for index in low_track_indexes}
        for index, track in enumerate(self._tracks):
            if index not in used_track_indexes and id(track) not in matched_low_tracks:
                track.missed += 1

        for index, detection in enumerate(high):
            if index in used_high_indexes:
                continue
            self._tracks.append(_Track(self._next_id, detection.bbox, detection.score))
            self._next_id += 1
        self._tracks = [track for track in self._tracks if track.missed <= max(0, int(self.config.max_lost_frames))]
        return [
            ReidCandidate(track.bbox, track.track_id,
                          max(0.0, track.bbox[2] - track.bbox[0]) * max(0.0, track.bbox[3] - track.bbox[1]),
                          track.score)
            for track in self._tracks
            if track.missed == 0 and track.hits >= max(1, int(self.config.min_confirmed_hits))
        ]
