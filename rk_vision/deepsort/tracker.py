from __future__ import annotations

from typing import Sequence

from . import iou_matching, kalman_filter, linear_assignment
from .nn_matching import NearestNeighborDistanceMetric
from .track import Track


class Tracker:
    def __init__(
        self,
        metric: NearestNeighborDistanceMetric,
        max_iou_distance: float = 0.7,
        max_age: int = 70,
        n_init: int = 3,
        max_bbox_age: int = 2,
    ) -> None:
        self.metric = metric
        self.max_iou_distance = float(max_iou_distance)
        self.max_age = int(max_age)
        self.n_init = int(n_init)
        self.max_bbox_age = int(max_bbox_age)
        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self) -> None:
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections: Sequence) -> None:
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [track for track in self.tracks if not track.is_deleted()]

        active_targets = [track.track_id for track in self.tracks if track.is_confirmed()]
        features = []
        targets = []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            for feature in track.features:
                if feature is not None:
                    features.append(feature)
                    targets.append(track.track_id)
            track.features = []
        self.metric.partial_fit(features, targets, active_targets)

    def _match(self, detections: Sequence):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = [dets[i].feature for i in detection_indices]
            targets = [tracks[i].track_id for i in track_indices]
            cost_matrix = self.metric.distance(features, targets)
            for row, track_idx in enumerate(track_indices):
                for col, detection_idx in enumerate(detection_indices):
                    if int(tracks[track_idx].cls) != int(dets[detection_idx].cls):
                        cost_matrix[row, col] = linear_assignment.INFTY_COST
            return linear_assignment.gate_cost_matrix(
                self.kf,
                cost_matrix,
                tracks,
                dets,
                track_indices,
                detection_indices,
            )

        confirmed_tracks = [i for i, track in enumerate(self.tracks) if track.is_confirmed()]
        unconfirmed_tracks = [i for i, track in enumerate(self.tracks) if not track.is_confirmed()]

        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks,
        )

        iou_track_candidates = unconfirmed_tracks + [
            idx
            for idx in unmatched_tracks_a
            if self.tracks[idx].time_since_update <= self.max_bbox_age
        ]
        unmatched_tracks_a = [
            idx
            for idx in unmatched_tracks_a
            if self.tracks[idx].time_since_update > self.max_bbox_age
        ]
        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_distance,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection) -> None:
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(
                mean,
                covariance,
                self._next_id,
                self.n_init,
                self.max_age,
                detection.feature,
                detection.cls,
                detection.confidence,
            )
        )
        self._next_id += 1
