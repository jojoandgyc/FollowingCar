import numpy as np
import pytest

from rk_vision.deepsort.deep_sort import DeepSort, DeepSortConfig
from rk_vision.deepsort.detection import Detection
from rk_vision.deepsort.nn_matching import NearestNeighborDistanceMetric
from rk_vision.deepsort.tracker import Tracker


BASE_FEATURE = np.asarray([1.0, 0.0, 0.0], dtype="float32")


def detection(source_index=7, *, x=100.0, feature=BASE_FEATURE):
    return Detection(
        [x, 80.0, 80.0, 180.0],
        0.9,
        0,
        feature,
        source_detection_index=source_index,
    )


def confirmed_tracker(route, *, two_tracks=False):
    tracker = Tracker(
        NearestNeighborDistanceMetric("cosine", 0.3, 15),
        n_init=2,
        max_age=5,
        # An appearance test must succeed without the IoU rescue path.
        max_bbox_age=0 if route == "appearance" else 2,
    )
    tracker._next_id = 41
    detections = [detection()]
    if two_tracks:
        detections.append(detection(8, x=120.0))
    tracker.update(detections)
    tracker.predict()
    tracker.update(detections)
    assert all(track.is_confirmed() for track in tracker.tracks)
    return tracker


def candidate_feature(route, *, alternate=False):
    if route == "appearance":
        return np.asarray([1.0, 0.2 if alternate else 0.1, 0.0], dtype="float32")
    # Outside the appearance threshold, so only IoU can associate these.
    return np.asarray([0.0, 0.0, 1.0] if alternate else [0.0, 1.0, 0.0], dtype="float32")


@pytest.mark.parametrize("route", ["appearance", "iou"])
def test_rejected_match_leaves_old_kalman_and_gallery_intact_and_starts_new_track(route):
    tracker = confirmed_tracker(route)
    track = tracker.tracks[0]
    tracker.predict()
    predicted_mean = track.mean.copy()
    predicted_covariance = track.covariance.copy()
    previous_feature = track.last_feature.copy()
    previous_samples = np.asarray(tracker.metric.samples[41]).copy()
    bad_feature = candidate_feature(route)
    calls = []

    def reject(raw_track_id, source_detection_index):
        calls.append((raw_track_id, source_detection_index))
        return False

    tracker.update([detection(17, x=102.0, feature=bad_feature)], match_validator=reject)

    assert calls == [(41, 17)]
    np.testing.assert_array_equal(track.mean, predicted_mean)
    np.testing.assert_array_equal(track.covariance, predicted_covariance)
    np.testing.assert_array_equal(track.last_feature, previous_feature)
    np.testing.assert_array_equal(tracker.metric.samples[41], previous_samples)
    assert track.features == []
    assert track.hits == 2 and track.time_since_update == 1
    assert track.source_detection_index == 7
    assert len(tracker.tracks) == 2
    new_track = tracker.tracks[1]
    assert new_track.track_id == 42 and new_track.source_detection_index == 17
    assert new_track.hits == 1 and new_track.time_since_update == 0
    np.testing.assert_array_equal(new_track.last_feature, bad_feature)
    assert 42 not in tracker.metric.samples


@pytest.mark.parametrize("route", ["appearance", "iou"])
def test_rejected_best_candidate_does_not_hide_a_legal_candidate(route):
    tracker = confirmed_tracker(route)
    track = tracker.tracks[0]
    samples_before = len(tracker.metric.samples[41])
    rejected_feature = candidate_feature(route)
    accepted_feature = candidate_feature(route, alternate=True)
    tracker.predict()
    tracker.update(
        [
            detection(17, feature=rejected_feature),
            detection(23, x=102.0, feature=accepted_feature),
        ],
        match_validator=lambda track_id, source_index: source_index != 17,
    )

    assert track.track_id == 41 and track.source_detection_index == 23
    assert track.hits == 3 and track.time_since_update == 0
    np.testing.assert_array_equal(track.last_feature, accepted_feature)
    samples = tracker.metric.samples[41]
    assert len(samples) == samples_before + 1
    np.testing.assert_array_equal(samples[-1], accepted_feature)
    assert not any(np.array_equal(sample, rejected_feature) for sample in samples)
    assert tracker.tracks[1].track_id == 42
    assert tracker.tracks[1].source_detection_index == 17


@pytest.mark.parametrize("route", ["appearance", "iou"])
def test_validation_is_per_track_detection_pair_not_a_global_detection_drop(route):
    tracker = confirmed_tracker(route, two_tracks=True)
    tracker.predict()
    tracker.update(
        [
            detection(17, feature=candidate_feature(route)),
            detection(23, x=120.0, feature=candidate_feature(route, alternate=True)),
        ],
        match_validator=lambda track_id, source_index: (track_id, source_index)
        in {(41, 23), (42, 17)},
    )

    assert [(track.track_id, track.source_detection_index) for track in tracker.tracks] == [
        (41, 23),
        (42, 17),
    ]
    assert all(track.time_since_update == 0 for track in tracker.tracks)


@pytest.mark.parametrize("route", ["appearance", "iou"])
@pytest.mark.parametrize("validator_option", ["omitted", "none", "accept"])
def test_default_and_accepting_validator_preserve_normal_matching(route, validator_option):
    tracker = confirmed_tracker(route)
    track = tracker.tracks[0]
    samples_before = len(tracker.metric.samples[41])
    feature = candidate_feature(route)
    options = {}
    if validator_option != "omitted":
        options["match_validator"] = None if validator_option == "none" else lambda *_: True
    tracker.predict()
    tracker.update([detection(17, x=102.0, feature=feature)], **options)

    assert len(tracker.tracks) == 1
    assert track.track_id == 41 and track.source_detection_index == 17
    assert track.hits == 3 and track.time_since_update == 0
    assert len(tracker.metric.samples[41]) == samples_before + 1
    np.testing.assert_array_equal(tracker.metric.samples[41][-1], feature)


def test_tentative_track_iou_match_is_also_guarded_with_missing_source_index():
    tracker = Tracker(NearestNeighborDistanceMetric("cosine", 0.3), n_init=2)
    tracker.update([detection()])
    old_track = tracker.tracks[0]
    tracker.predict()
    previous_mean = old_track.mean.copy()
    calls = []

    def reject(raw_track_id, source_index):
        calls.append((raw_track_id, source_index))
        return False

    tracker.update([detection(None, x=102.0)], match_validator=reject)

    assert calls == [(1, None)]
    np.testing.assert_array_equal(old_track.mean, previous_mean)
    assert old_track.is_deleted() and old_track.hits == 1
    assert [track.track_id for track in tracker.tracks] == [2]
    assert not tracker.metric.samples


def test_deep_sort_passes_original_detection_indices_after_confidence_filter_and_nms():
    deep_sort = DeepSort(DeepSortConfig(n_init=2, min_confidence=0.5, nms_max_overlap=0.5))
    original_box = [140.0, 170.0, 80.0, 180.0]
    for _ in range(2):
        deep_sort.update([original_box], [0.9], [0], [BASE_FEATURE])
    calls = []

    def reject(raw_track_id, source_index):
        calls.append((raw_track_id, source_index))
        return False

    outputs = deep_sort.update(
        [
            [700.0, 170.0, 80.0, 180.0],  # index 0: below confidence threshold
            original_box,                 # index 1: retained, second after NMS
            original_box,                 # index 2: suppressed by index 1
            [500.0, 170.0, 80.0, 180.0],  # index 3: retained, first after NMS
        ],
        [0.1, 0.9, 0.6, 0.95],
        [0, 0, 0, 0],
        [BASE_FEATURE] * 4,
        match_validator=reject,
    )

    assert calls == [(1, 3), (1, 1)]
    assert len(outputs) == 1
    assert outputs[0].track_id == 1 and outputs[0].time_since_update == 1
    assert outputs[0].source_detection_index is None and outputs[0].feature is None
    assert [(track.track_id, track.source_detection_index) for track in deep_sort.tracker.tracks] == [
        (1, 0),
        (2, 3),
        (3, 1),
    ]


def test_validator_applies_only_to_the_update_that_supplied_it():
    tracker = confirmed_tracker("iou")
    tracker.predict()
    tracker.update([detection(17)], match_validator=lambda *_: False)
    assert [track.track_id for track in tracker.tracks] == [41, 42]

    tracker.predict()
    tracker.update([detection(23)])

    assert [track.track_id for track in tracker.tracks] == [41]
    assert tracker.tracks[0].source_detection_index == 23
    assert tracker.tracks[0].hits == 3
