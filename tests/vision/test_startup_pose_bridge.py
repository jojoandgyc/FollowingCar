"""Startup pose regression without images, RKNN, cameras, or motor hardware.

Boxes, capture times, and HSV vectors below come from CAP17/24/25/27/29/30/32/33
of run_20260929_000932_72925_daee01d8. Colors were computed on its saved BGR
detector crops with rk_vision.reid._color_signature. These are embedded so this
regression does not depend on runtime logs or their retention policy.

The full appearance vectors are deliberately SYNTHETIC: the failed run did not
save OSNet embeddings. A distance of .30 represents a moderate first-pose
mismatch; it is not a measurement of the recorded person's OSNet distance.
"""

import copy

import numpy as np
import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from rk_vision.initial_enrollment import InitialEnrollment


FIRST_FULL = np.array([1., 0., 0.], dtype=np.float32)
POSE_FULL = np.array([.7, np.sqrt(.51), 0.], dtype=np.float32)
OTHER_FULL = np.array([0., 1., 0.], dtype=np.float32)

# frame, physical capture, capture timestamp, raw detector box, HSV descriptor
RECORDED = (
    (2, 17, 10978.139681840,
     (86.6504516602, 1.9225006104, 446.4589843750, 475.8035278320),
     (0.6008152962, 0.1795378178, 0.1326956451, 0.0015979764,
      0.0000159003, 0.0000159003, 0.0000159003, 0.0002464541,
      0.1824793667, 0.1036458611, 0.4399682283, 0.2214668095,
      0.1953347325, 0.4754098952, 0.1344128698, 0.1424027532)),
    (3, 24, 10978.477029822,
     (1.1824798584, 0.6176605225, 450.3737792969, 476.3763427734),
     (0.5360259414, 0.1113413274, 0.2736473680, 0.0020482873,
      0.0000382857, 0.0000510477, 0.0000191429, 0.0004658099,
      0.3135857582, 0.0993068367, 0.4502913952, 0.0908775926,
      0.1242117137, 0.4422131181, 0.0936660692, 0.2939707041)),
    (4, 25, 10978.538237780,
     (4.4061279297, 1.1282196045, 439.2241210938, 473.6169433594),
     (0.5652081370, 0.1184369400, 0.2511048913, 0.0001724166,
      0., 0., 0., 0.0000265256,
      0.2847393751, 0.0888674930, 0.4127122760, 0.1685040742,
      0.1219051629, 0.4663670063, 0.0932176933, 0.2733333707)),
    (5, 27, 10978.670579453,
     (0.8855590820, 0.1688385010, 462.5520629883, 473.9514160156),
     (0.5401517749, 0.1260519624, 0.2269035876, 0.0006534120,
      0., 0.0000251312, 0.0000251312, 0.0005151902,
      0.3116712272, 0.0996830240, 0.4578281641, 0.0944997072,
      0.1141020656, 0.4499746561, 0.1000285745, 0.2995768189)),
    (6, 29, 10978.769937934,
     (4.5169982910, 1.7133483887, 476.7910156250, 475.1935424805),
     (0.5334326029, 0.1254121959, 0.2487810701, 0.0006213281,
      0., 0., 0., 0.,
      0.3227831721, 0.0926291347, 0.3980471492, 0.1872759908,
      0.1397347748, 0.4501298070, 0.0936411917, 0.3172296584)),
    (7, 30, 10978.805972176,
     (7.0979614258, 2.1549224854, 473.8451232910, 474.4187011719),
     (0.5272442102, 0.1143405810, 0.2477549613, 0.0045289090,
      0.0000511019, 0.0000638774, 0.0000127755, 0.0001852445,
      0.3154202998, 0.0990483239, 0.4567107856, 0.1132355034,
      0.1486044228, 0.4411949515, 0.0918748900, 0.3027406335)),
    (8, 32, 10978.905843978,
     (34.5904235840, 1.4394531250, 471.0308532715, 472.4989624023),
     (0.5317903757, 0.1326820105, 0.2425062507, 0.0051867426,
      0.0000694343, 0., 0., 0.0001527555,
      0.2805840075, 0.1025544629, 0.4615367651, 0.1513181776,
      0.1845147163, 0.4497815371, 0.0976662934, 0.2640308738)),
    (9, 33, 10978.971583450,
     (50.5609893799, 2.0851898193, 462.2121582031, 471.8218994141),
     (0.5340925455, 0.1525832415, 0.2428112179, 0.0040873121,
      0.0000292473, 0., 0., 0.0000292473,
      0.2568060458, 0.1052976474, 0.4614495337, 0.1681501269,
      0.2007682025, 0.4546202719, 0.0950099081, 0.2413049787)),
)


def recorded_metadata():
    observations = []
    for frame, cap, stamp, box, color in RECORDED:
        weak = cap in (24, 25, 27, 29, 30)
        area = (box[2] - box[0]) * (box[3] - box[1])
        observations.append(dict(
            frame_index=frame, control_frame_id=frame, capture_frame_id=cap,
            capture_timestamp=stamp, track_id=1, candidate_count=1,
            detector_bbox=box, bbox=box, quality_bbox=box,
            image_width=640, image_height=480, detector_confidence=.90,
            area_ratio=area / (640 * 480), detector_area_ratio=area / (640 * 480),
            detector_edge_touch_count=3 if weak else 2,
            quality_bbox_ok=not weak, bbox_quality_tier='weak' if weak else 'strong',
            bbox_quality_reason='edge_touch>2' if weak else '',
            quality_bbox_reason='edge_touch>2' if weak else '',
            is_fresh=True, partial_observation=True,
            initial_color_feature=list(color), initial_color_source='hsv_crop_v1_bgr',
        ))
    return observations


def run_sequence(enrollment, observations, *, feature=POSE_FULL, track=1):
    results = []
    for index, metadata in enumerate(observations):
        results.append(enrollment.observe(
            1 if index == 0 else track,
            FIRST_FULL if index == 0 else feature,
            metadata,
        )[0])
    return results


def assign(bank, metadata, feature, track=1):
    box = metadata['detector_bbox']
    return bank.assign(
        track_id=track, feature=feature, confidence=metadata['detector_confidence'],
        area=(box[2] - box[0]) * (box[3] - box[1]),
        frame_index=metadata['frame_index'], candidate_count=metadata['candidate_count'],
        bbox_quality_ok=metadata['quality_bbox_ok'],
        bbox_quality_tier=metadata['bbox_quality_tier'],
        bbox_quality_reason=metadata['bbox_quality_reason'], sample_metadata=metadata,
    )


def test_recorded_pose_change_keeps_first_owner_and_needs_two_strong_captures():
    enrollment = InitialEnrollment()
    assert run_sequence(enrollment, recorded_metadata()) == [False] * 7 + [True]
    assert enrollment.candidate_track_id == 1
    np.testing.assert_array_equal(enrollment.candidate_feature, FIRST_FULL)
    np.testing.assert_allclose(enrollment.candidate_color, RECORDED[0][-1], atol=1e-7)
    evidence = enrollment.last_evidence
    assert evidence['seed_cap'] == 17
    assert evidence['anchor_full_distance'] == pytest.approx(.30, abs=1e-6)
    assert evidence['anchor_color_distance'] == pytest.approx(.018969, abs=1e-6)
    assert evidence['pose_continuous']
    assert evidence['confirmation_streak'] == 2


@pytest.mark.parametrize('template_memory', [False, True])
def test_weak_bridge_is_observation_only_and_only_confirming_crop_enters_gallery(template_memory):
    bank = IdentityBank(IdentityBankConfig(
        appearance_region_safety_enable=True, template_memory_enable=template_memory))
    observations = recorded_metadata()
    for index, metadata in enumerate(observations[:-1]):
        assert assign(bank, metadata, FIRST_FULL if index == 0 else POSE_FULL) == 0
        assert not bank.identities
        assert not bank.track_to_uid
    assert assign(bank, observations[-1], POSE_FULL) == 1
    entry = bank.identities[1]
    assert [item['capture_frame_id'] for item in entry.feature_metadata] == [33]
    assert not entry.weak_features
    assert not entry.partial_features


def test_weak_captures_cannot_choose_the_first_candidate():
    enrollment = InitialEnrollment()
    for metadata in recorded_metadata()[1:6]:
        assert not enrollment.observe(1, POSE_FULL, metadata)[0]
        assert enrollment.candidate_track_id is None


@pytest.mark.parametrize('interruption', [
    'multiple_people', 'stale', 'capture_gap', 'processing_gap',
    'other_weak_reason', 'cross_side', 'small_fragment', 'color_conflict',
])
def test_interrupted_bridge_cannot_be_restarted_by_repeating_the_new_appearance(interruption):
    observations = recorded_metadata()
    interrupted = observations[2]
    if interruption == 'multiple_people':
        interrupted['candidate_count'] = 2
    elif interruption == 'stale':
        interrupted['is_fresh'] = False
    elif interruption == 'capture_gap':
        for metadata in observations[2:]:
            metadata['capture_timestamp'] += .5
    elif interruption == 'processing_gap':
        for metadata in observations[2:]:
            metadata['frame_index'] += 1
            metadata['control_frame_id'] += 1
    elif interruption == 'other_weak_reason':
        interrupted['bbox_quality_reason'] = interrupted['quality_bbox_reason'] = 'low_confidence'
    elif interruption == 'cross_side':
        x1, y1, x2, y2 = interrupted['detector_bbox']
        interrupted['detector_bbox'] = (640 - x2, y1, 640 - x1, y2)
    elif interruption == 'small_fragment':
        interrupted['detector_bbox'] = (0., 100., 60., 180.)
    elif interruption == 'color_conflict':
        interrupted['initial_color_feature'] = [0.] * 7 + [1.] + [0.] * 8
    enrollment = InitialEnrollment()
    assert run_sequence(enrollment, observations) == [False] * len(observations)
    last = observations[-1]
    for offset in (1, 2, 3):
        repeated = dict(last, frame_index=last['frame_index'] + offset,
                        control_frame_id=last['control_frame_id'] + offset,
                        capture_frame_id=last['capture_frame_id'] + offset,
                        capture_timestamp=last['capture_timestamp'] + offset * .1)
        assert not enrollment.observe(1, POSE_FULL, repeated)[0]
    assert enrollment.candidate_track_id == 1
    np.testing.assert_array_equal(enrollment.candidate_feature, FIRST_FULL)


@pytest.mark.parametrize('feature,track', [(OTHER_FULL, 1), (POSE_FULL, 2)])
def test_bridge_cannot_replace_first_full_appearance_or_raw_track(feature, track):
    enrollment = InitialEnrollment()
    observations = recorded_metadata()
    assert run_sequence(enrollment, observations, feature=feature, track=track) == [False] * 8
    assert enrollment.candidate_track_id == 1
    np.testing.assert_array_equal(enrollment.candidate_feature, FIRST_FULL)


@pytest.mark.parametrize('color,source', [
    (None, 'hsv_crop_v1_bgr'), ([0.] * 16, 'hsv_crop_v1_bgr'),
    ([1.] * 15, 'hsv_crop_v1_bgr'), ([float('nan')] * 16, 'hsv_crop_v1_bgr'),
    (RECORDED[0][-1], 'unknown_color_source'),
])
def test_unavailable_color_provenance_cannot_enable_pose_bridge(color, source):
    observations = recorded_metadata()
    for metadata in observations:
        metadata['initial_color_feature'] = copy.deepcopy(color)
        metadata['initial_color_source'] = source
    assert run_sequence(InitialEnrollment(), observations) == [False] * 8


def test_bridge_deadline_is_measured_from_first_capture_not_refreshed_each_frame():
    observations = recorded_metadata()
    stamp = observations[0]['capture_timestamp']
    for index, metadata in enumerate(observations):
        metadata['capture_timestamp'] = stamp + index * .25
    assert run_sequence(InitialEnrollment(), observations) == [False] * 8


def test_matching_full_feature_cannot_excuse_conflicting_color_on_a_weak_bridge():
    observations = recorded_metadata()
    enrollment = InitialEnrollment()
    assert not enrollment.observe(1, FIRST_FULL, observations[0])[0]
    for metadata in observations[1:6]:
        metadata['initial_color_feature'] = [0.] * 7 + [1.] + [0.] * 8
        assert not enrollment.observe(1, FIRST_FULL, metadata)[0]
    # These two mutually identical synthetic poses still disagree with the
    # first full feature by .30. The conflicting weak corridor cannot bless it.
    for metadata in observations[6:]:
        assert not enrollment.observe(1, POSE_FULL, metadata)[0]


def test_cross_side_bridge_is_rejected_even_when_adjacent_boxes_still_overlap():
    source = recorded_metadata()
    observations = [source[index] for index in (0, 1, 2, 6, 7)]
    seed_box = (80., 0., 560., 471.)
    # Left -> right weak boxes have IoU exactly .50 and equal area. Both also
    # overlap the seed by .714, so geometry alone cannot reject this crossing.
    boxes = (seed_box, (0., 0., 480., 471.), (160., 0., 640., 471.),
             seed_box, seed_box)
    for index, (metadata, box) in enumerate(zip(observations, boxes)):
        metadata.update(
            frame_index=2 + index, control_frame_id=2 + index,
            capture_frame_id=17 + index,
            capture_timestamp=RECORDED[0][2] + index * .1,
            detector_bbox=box, bbox=box, quality_bbox=box,
            area_ratio=480. * 471. / (640 * 480),
            detector_area_ratio=480. * 471. / (640 * 480),
            initial_color_feature=list(RECORDED[0][-1]))
    enrollment = InitialEnrollment()
    for index, metadata in enumerate(observations):
        # Full matches on both weak frames must not override the side change.
        feature = FIRST_FULL if index <= 2 else POSE_FULL
        assert not enrollment.observe(1, feature, metadata)[0]
    assert not enrollment.pose_continuous
    np.testing.assert_array_equal(enrollment.candidate_feature, FIRST_FULL)


def test_replayed_capture_does_not_count_as_second_strong_observation():
    enrollment = InitialEnrollment()
    observations = recorded_metadata()
    assert run_sequence(enrollment, observations[:-1]) == [False] * 7
    for _ in range(3):
        assert not enrollment.observe(1, POSE_FULL, observations[-2])[0]
    assert enrollment.observe(1, POSE_FULL, observations[-1])[0]


def test_two_strong_captures_still_need_mutual_full_appearance_agreement():
    enrollment = InitialEnrollment()
    observations = recorded_metadata()
    assert run_sequence(enrollment, observations[:-1]) == [False] * 7
    opposite_pose = np.array([.7, -np.sqrt(.51), 0.], dtype=np.float32)
    assert not enrollment.observe(1, opposite_pose, observations[-1])[0]


def test_existing_gallery_never_calls_startup_pose_bridge(monkeypatch):
    bank = IdentityBank(IdentityBankConfig(appearance_region_safety_enable=True))
    seed = recorded_metadata()[0]
    confirm = dict(seed, frame_index=3, control_frame_id=3,
                   capture_frame_id=18, capture_timestamp=seed['capture_timestamp'] + .1)
    assert assign(bank, seed, FIRST_FULL) == 0
    assert assign(bank, confirm, FIRST_FULL) == 1

    def unexpected_startup_call(*args, **kwargs):
        pytest.fail('An existing gallery must not use first-person pose enrollment')

    monkeypatch.setattr(bank._initial_enrollment, 'observe', unexpected_startup_call)
    for metadata in recorded_metadata()[1:]:
        metadata['frame_index'] += 10
        metadata['control_frame_id'] += 10
        metadata['capture_frame_id'] += 100
        metadata['capture_timestamp'] += 1.
        assign(bank, metadata, POSE_FULL)
    assert bank._initial_enrollment.candidate_track_id is None


def test_recorded_pose_bridge_locks_controller_on_creation_without_depth_or_motion(monkeypatch):
    from types import SimpleNamespace

    import request_0513_modular as runtime
    from car_control_modular.control_types import SensorFrame
    from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
    from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
    from rk_vision.yolo11 import Detection

    tracker = DeepSortTracker(DeepSortTrackerConfig(
        n_init=1, identity_template_memory_enable=True,
        identity_template_crosscheck_enable=True,
        identity_appearance_region_safety_enable=True))
    controller = FollowSafetyController(FollowPolicyConfig(initial_target_confirm_frames=2))
    owner = object.__new__(runtime.PersonTracker)
    owner._rknn_pipeline = SimpleNamespace(tracker=tracker)
    owner._follow_controller = controller
    observations = recorded_metadata()
    # One synthetic tracker initialization precedes the recorded seed. Like
    # the original run's tentative first track, it produces no enrolled UID.
    warmup = dict(observations[0], frame_index=1, control_frame_id=1,
                  capture_frame_id=16,
                  capture_timestamp=observations[0]['capture_timestamp'] - .1)
    for metadata in [warmup] + observations:
        cap, stamp = metadata['capture_frame_id'], metadata['capture_timestamp']
        full = FIRST_FULL if cap in (16, 17) else POSE_FULL
        monkeypatch.setattr(runtime.time, 'monotonic', lambda stamp=stamp: stamp + .05)
        tracker.update(
            [Detection(metadata['detector_bbox'], .90, 0)], [full],
            partial_features=[None], partial_feature_sources=[None],
            color_features=[metadata['initial_color_feature']],
            image_width=640, image_height=480,
            frame_context=dict(control_frame_id=metadata['control_frame_id'],
                               capture_frame_id=cap, capture_timestamp=stamp))
        owner._active_capture_frame_id = cap
        owner._active_capture_timestamp = stamp
        persons = [
            (item['display_bbox'], item['uid'], .90,
             (item['display_bbox'][2] - item['display_bbox'][0]) *
             (item['display_bbox'][3] - item['display_bbox'][1]))
            for item in tracker.last_identity_observations if item['uid'] > 0
        ]
        targets = owner._persons_to_targets(persons, width=640, height=480)
        decision = controller.decide(metadata['control_frame_id'], SensorFrame(
            width=640, height=480, persons=targets,
            capture_frame_id=cap, capture_timestamp=stamp))
        assert all(action.kind not in ('forward', 'backward', 'steer_left', 'steer_right')
                   for action in decision.actions)
        if cap != 33:
            assert controller.active_target_id is None
            assert not targets
            assert all(action.kind == 'stop' for action in decision.actions)
        else:
            assert len(targets) == 1 and targets[0].initial_identity_confirmed
            assert controller.active_target_id == 1 and controller._has_seen_person
            observation = tracker.last_identity_observations[0]
            assert observation['assignment']['reason'] == 'created_confirmed'
            assert observation['assignment']['initial_enrollment_state']['seed_cap'] == 17
