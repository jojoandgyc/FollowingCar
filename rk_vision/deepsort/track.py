from __future__ import annotations


class TrackState:
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


class Track:
    def __init__(
        self,
        mean,
        covariance,
        track_id: int,
        n_init: int,
        max_age: int,
        feature=None,
        cls: int = 0,
        confidence: float = 0.0,
        source_detection_index=None,
    ) -> None:
        self.mean = mean
        self.covariance = covariance
        self.track_id = int(track_id)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.TENTATIVE
        self.cls = int(cls)
        self.confidence = float(confidence)
        self.features = []
        self.last_feature = feature
        self.source_detection_index = source_detection_index
        if feature is not None:
            self.features.append(feature)
        self._n_init = int(n_init)
        self._max_age = int(max_age)

    def to_tlwh(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2.0
        return ret

    def to_tlbr(self):
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf) -> None:
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection) -> None:
        was_confirmed = self.is_confirmed()
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        self.last_feature = detection.feature
        self.source_detection_index = detection.source_detection_index
        if detection.feature is not None and (not was_confirmed or bool(getattr(detection, "store_feature", True))):
            self.features.append(detection.feature)
        self.cls = int(detection.cls)
        self.confidence = float(detection.confidence)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.TENTATIVE and self.hits >= self._n_init:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.time_since_update > self._max_age:
            self.state = TrackState.DELETED

    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED
