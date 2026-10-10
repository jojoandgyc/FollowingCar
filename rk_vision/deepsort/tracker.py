from __future__ import annotations

from typing import Callable, Optional, Sequence

from . import iou_matching, kalman_filter, linear_assignment
from .nn_matching import NearestNeighborDistanceMetric
from .track import Track
from .low_score import match_low_score
from ..stage_timing import StageTiming


MatchValidator = Callable[[int, Optional[int]], bool]


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
        self.kf.timing = {}  # one physical frame, including all track predictions
        for track in self.tracks:
            track.predict(self.kf)

    def update(
        self,
        detections: Sequence,
        *,
        match_validator: Optional[MatchValidator] = None,
        provisional_features=None,
        low_score_validator=None,
        capture_context=None,
        image_shape=None,
        camera_hfov_deg=60.,
    ) -> None:
        timer = StageTiming()
        matches, unmatched_tracks, unmatched_detections = self._match(
            detections, match_validator, provisional_features)
        low_indices = [i for i, detection in enumerate(detections)
                       if getattr(detection, "low_score_continuation", False)]
        low_matches = match_low_score(self.tracks, detections, unmatched_tracks, low_indices,
            validator=low_score_validator, context=capture_context,
            image_shape=image_shape, camera_hfov_deg=camera_hfov_deg)
        matches.extend(low_matches)
        low_matched_tracks = {ti for ti, _ in low_matches}
        unmatched_tracks = [ti for ti in unmatched_tracks if ti not in low_matched_tracks]
        timer.mark("match")

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [track for track in self.tracks if not track.is_deleted()]
        timer.mark("kalman_update")

        active_targets = [track.track_id for track in self.tracks if track.is_confirmed()]
        features = []
        targets = []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            # Existing provisional tracks never enter partial_fit, even
            # transiently before the adapter restores a frozen gallery.
            pending = [] if track.track_id in (provisional_features or {}) else track.features
            for feature in pending:
                if feature is not None:
                    features.append(feature)
                    targets.append(track.track_id)
            track.features = []
        self.metric.partial_fit(features, targets, active_targets)
        timer.mark("metric_update")
        self.last_timing_ms = timer.finish()
        self.last_timing_ms.update({"kf_" + key: value for key, value in self.kf.timing.items()})
        self.last_timing_ms.update(tracks_count=len(self.tracks), detections_count=len(detections),
                                   matched_count=len(matches), initiated_count=len(unmatched_detections),
                                   low_score_matched_count=len(low_matches))

    def _match(self, detections: Sequence, match_validator: Optional[MatchValidator] = None,
               provisional_features=None):
        validation_results = {}
        # These descriptors have capture-bounded, UID-checked provenance from
        # the adapter. They are never fitted into the trusted metric.
        provisional = NearestNeighborDistanceMetric("cosine", self.metric.matching_threshold)
        provisional.samples = provisional_features or {}

        def provisional_cost(tracks, dets, track_indices, detection_indices):
            costs = provisional.distance([dets[i].feature for i in detection_indices],
                                         [tracks[i].track_id for i in track_indices])
            geometry = iou_matching.iou_cost(tracks, dets, track_indices, detection_indices,
                                            max_bbox_age=self.max_age)
            for row, track_idx in enumerate(track_indices):
                for col in range(len(detection_indices)):
                    if (match_validator is None or geometry[row, col] > self.max_iou_distance):
                        costs[row, col] = linear_assignment.INFTY_COST
            return linear_assignment.gate_cost_matrix(
                self.kf, costs, tracks, dets, track_indices, detection_indices)

        def guard_cost_matrix(cost_matrix, tracks, dets, track_indices, detection_indices):
            if match_validator is None:
                return cost_matrix
            for row, track_idx in enumerate(track_indices):
                for col, detection_idx in enumerate(detection_indices):
                    pair = (track_idx, detection_idx)
                    if pair not in validation_results:
                        validation_results[pair] = bool(
                            match_validator(
                                int(tracks[track_idx].track_id),
                                dets[detection_idx].source_detection_index,
                            )
                        )
                    if not validation_results[pair]:
                        # Reject before assignment: a bad candidate must not consume
                        # a valid match or update this track's state and gallery.
                        cost_matrix[row, col] = linear_assignment.INFTY_COST
            return cost_matrix

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = [dets[i].feature for i in detection_indices]
            targets = [tracks[i].track_id for i in track_indices]
            cost_matrix = self.metric.distance(features, targets)
            soft_cost = provisional_cost(tracks, dets, track_indices, detection_indices) if provisional.samples else None
            for row, track_idx in enumerate(track_indices):
                for col, detection_idx in enumerate(detection_indices):
                    if tracks[track_idx].track_id in provisional.samples:
                        # A provisional raw track cannot use an unrelated
                        # frozen template to bypass the local appearance gate.
                        cost_matrix[row, col] = soft_cost[row, col]
                    if int(tracks[track_idx].cls) != int(dets[detection_idx].cls):
                        cost_matrix[row, col] = linear_assignment.INFTY_COST
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf,
                cost_matrix,
                tracks,
                dets,
                track_indices,
                detection_indices,
            )
            return guard_cost_matrix(cost_matrix, tracks, dets, track_indices, detection_indices)

        def guarded_iou_cost(tracks, dets, track_indices, detection_indices):
            cost_matrix = iou_matching.iou_cost(tracks, dets, track_indices, detection_indices,
                                                max_bbox_age=self.max_bbox_age)
            soft_cost = provisional_cost(tracks, dets, track_indices, detection_indices) if provisional.samples else None
            for row, track_idx in enumerate(track_indices):
                if tracks[track_idx].track_id not in provisional.samples:
                    continue
                for col in range(len(detection_indices)):
                    if soft_cost[row, col] > self.metric.matching_threshold:
                        cost_matrix[row, col] = linear_assignment.INFTY_COST
            return guard_cost_matrix(cost_matrix, tracks, dets, track_indices, detection_indices)

        confirmed_tracks = [i for i, track in enumerate(self.tracks) if track.is_confirmed()]
        unconfirmed_tracks = [i for i, track in enumerate(self.tracks) if not track.is_confirmed()]

        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks,
            [i for i, detection in enumerate(detections)
             if not getattr(detection, "low_score_continuation", False)],
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
            guarded_iou_cost,
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
                source_detection_index=detection.source_detection_index,
            )
        )
        self._next_id += 1
