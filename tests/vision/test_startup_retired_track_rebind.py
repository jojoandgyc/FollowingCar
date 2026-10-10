"""Bounded recovery of the first person's retired raw track; no hardware."""
import numpy as np
import pytest

from car_control_modular.video_recorder import VideoFrameOverlay, VideoTrackOverlay, startup_status
from rk_vision.initial_enrollment import InitialEnrollment
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection


BOX = (200., 40., 400., 460.)
FEATURE = np.array([1., 0., 0.], dtype=np.float32)
OTHER = np.array([0., 1., 0.], dtype=np.float32)
COLOR = np.array([1.] + [0.] * 15, dtype=np.float32)
OTHER_COLOR = np.roll(COLOR, 1)


@pytest.fixture
def clock(monkeypatch):
    value = [10.]
    monkeypatch.setattr('rk_vision.tracker.time.monotonic', lambda: value[0])
    return value


def tracker(max_age=20):
    return DeepSortTracker(DeepSortTrackerConfig(
        n_init=2, max_age=max_age, identity_appearance_region_safety_enable=True,
        identity_template_memory_enable=True, identity_template_crosscheck_enable=True))


def update(t, clock, cap, *, detected=True, feature=FEATURE, color=COLOR, box=BOX,
           context=None, age=.03, extra=False):
    stamp = 10. + cap * .1
    clock[0] = stamp + age
    detections = [Detection(box, .94, 0)] if detected else []
    features = [feature] if detected else []
    colors = [color] if detected else []
    if extra:
        detections.append(Detection((430., 40., 620., 460.), .94, 0))
        features.append(OTHER)
        colors.append(OTHER_COLOR)
    ctx = dict(capture_frame_id=cap, capture_timestamp=stamp, control_frame_id=cap)
    ctx.update(context or {})
    rows = t.update(detections, features, color_features=colors,
                    image_width=640, image_height=480, frame_context=ctx)
    return [(r.track_id, r.reid_uid) for r in rows if r.time_since_update == 0]


def retired(clock, *, max_age=20, color=COLOR):
    t = tracker(max_age)
    assert update(t, clock, 1, color=color) == []
    assert update(t, clock, 2, color=color) == [(1, 0)]
    enrollment = t.identity_bank._initial_enrollment
    assert enrollment.candidate_track_id == 1
    for cap in range(3, max_age + 4):
        update(t, clock, cap, detected=False)
    assert t.deepsort.tracker.tracks == []
    assert enrollment._candidate_retired is not None
    return t, max_age + 4


def test_same_person_recovers_after_real_20_frame_track_retirement(clock):
    t, cap = retired(clock)
    enrollment = t.identity_bank._initial_enrollment
    anchor = enrollment.seed
    assert update(t, clock, cap) == []
    assert update(t, clock, cap + 1) == [(2, 0)]
    assert not t.identity_bank.identities and not t.identity_bank.track_to_uid
    assert enrollment.seed is anchor and enrollment.candidate_track_id == 1
    assert enrollment.last_evidence['path'] == 'retired_track_rebind'
    assert enrollment.last_evidence['confirmation_streak'] == 1
    assert update(t, clock, cap + 2) == [(2, 1)]
    assert t.identity_bank.track_to_uid == {2: 1}
    assert t.identity_bank.last_assignments[2]['initial_identity_confirmed']
    assert enrollment._candidate_retired is None  # Successful creation clears receipts.


@pytest.mark.parametrize('change,blocker', [
    ({'color': None}, 'rebind_color_unavailable'),
    ({'color': np.ones(6)}, 'rebind_color_unavailable'),
    ({'color': OTHER_COLOR}, 'rebind_color_conflict'),
    ({'feature': OTHER}, 'rebind_appearance_conflict'),
    ({'box': (410., 40., 610., 460.)}, 'rebind_geometry_conflict'),
    ({'age': -.01}, 'track_lifecycle_not_current'),
    ({'age': .36}, 'track_lifecycle_not_current'),
])
def test_strong_looking_replacement_still_needs_independent_proof(clock, change, blocker):
    t, cap = retired(clock)
    for frame in range(cap, cap + 6):
        assert not any(uid for _, uid in update(t, clock, frame, **change))
    assert not t.identity_bank.identities
    evidence = t.identity_bank._initial_enrollment.last_evidence
    assert evidence['bridge_blocker'] == blocker
    assert evidence['seed_cap'] == 2


def test_missing_original_color_cannot_be_filled_by_replacement(clock):
    t, cap = retired(clock, color=None)
    for frame in range(cap, cap + 6):
        assert not any(uid for _, uid in update(t, clock, frame))
    assert t.identity_bank._initial_enrollment.candidate_color is None


def test_original_track_must_be_actually_deleted_not_merely_absent(clock):
    t = tracker()
    update(t, clock, 1)
    update(t, clock, 2)
    for cap in range(3, 7):
        update(t, clock, cap, box=(410., 40., 610., 460.), feature=OTHER)
    assert 1 in {r.track_id for r in t.deepsort.tracker.tracks}
    assert not t.identity_bank.identities
    assert t.identity_bank._initial_enrollment.last_evidence['bridge_blocker'] == 'original_track_not_retired'


def test_live_competitor_blocks_even_when_current_detection_is_unique(clock):
    t, cap = retired(clock)
    update(t, clock, cap, extra=True)
    update(t, clock, cap + 1, extra=True)
    for frame in (cap + 2, cap + 3):
        update(t, clock, frame)
        assert not t.identity_bank.identities
    assert t.identity_bank._initial_enrollment.last_evidence['bridge_blocker'] == 'other_live_tracks'


@pytest.mark.parametrize('interruption', ['color', 'full', 'ambiguous', 'missing', 'replay'])
@pytest.mark.parametrize('max_bbox_age', [1, 2])
def test_interrupted_rebind_requires_two_new_frames_again(clock, interruption, max_bbox_age):
    t, cap = retired(clock, max_age=1)
    t.deepsort.tracker.max_bbox_age = max_bbox_age
    update(t, clock, cap)
    assert update(t, clock, cap + 1) == [(2, 0)]
    changes = {
        'color': dict(color=OTHER_COLOR),
        'full': dict(feature=np.array([.7, np.sqrt(.51), 0.])),
        'ambiguous': dict(extra=True),
        'missing': dict(detected=False),
        'replay': dict(context={'capture_frame_id': cap + 1,
                               'capture_timestamp': 10. + (cap + 1) * .1}),
    }[interruption]
    update(t, clock, cap + 2, **changes)
    assert not t.identity_bank.identities
    # A tentative competing track from the ambiguous frame disappears here.
    if interruption == 'missing' and max_bbox_age == 1:
        # An explicitly one-frame IoU window retires the replacement here.
        # With the configured two-frame window it may keep raw2, but must
        # still accumulate two new enrollment observations after the gap.
        assert update(t, clock, cap + 3) == []
        assert update(t, clock, cap + 4) == [(3, 0)]
        assert update(t, clock, cap + 5) == [(3, 1)]
        return
    assert update(t, clock, cap + 3) == [(2, 0)]
    assert update(t, clock, cap + 4) == [(2, 1)]


def test_rebind_deadline_is_absolute_and_rejected_frames_do_not_extend_it(clock):
    t, cap = retired(clock)
    for frame in range(cap, 44):
        update(t, clock, frame, color=OTHER_COLOR)
    for frame in range(44, 48):
        assert update(t, clock, frame) == [(2, 0)]
        evidence = t.identity_bank._initial_enrollment.last_evidence
        assert evidence['bridge_blocker'] == 'startup_anchor_expired'
        assert evidence['seed_cap'] == 2


def test_tracker_reset_clears_retirement_and_restarts_two_frame_enrollment(clock):
    t, cap = retired(clock)
    t.reset()
    e = t.identity_bank._initial_enrollment
    assert e._candidate_retired is e._track_lifecycle is e._lifecycle_watermark is None
    assert e.candidate_track_id is None
    assert update(t, clock, cap) == []
    assert update(t, clock, cap + 1) == [(1, 0)]
    assert e._candidate_retired is None
    assert update(t, clock, cap + 2) == [(1, 1)]


def test_metadata_cannot_invent_a_retirement_receipt():
    e = InitialEnrollment()
    def m(cap):
        return dict(capture_frame_id=cap, capture_timestamp=10. + .1*cap,
                    frame_index=cap, candidate_count=1, is_fresh=True,
                    quality_bbox_ok=True, bbox_quality_tier='strong', detector_bbox=BOX,
                    image_width=640, image_height=480, initial_color_feature=COLOR,
                    initial_color_source='hsv_crop_v1_bgr', original_track_retired=True,
                    active_track_ids=[2], retired_track_ids=[1])
    e.observe(1, FEATURE, m(1))
    for cap in (2, 3, 4):
        assert e.observe(2, FEATURE, m(cap)) == (False, 'initial_candidate_track_changed')
        assert e.last_evidence['bridge_blocker'] == 'original_track_not_retired'


@pytest.mark.parametrize('context', [
    {'capture_frame_id': 1}, {'capture_timestamp': 10.1},
    {'capture_timestamp': float('nan')}, {'capture_frame_id': 0},
    {'capture_timestamp': None},
])
def test_bad_capture_cannot_finish_pending_rebind(clock, context):
    t, cap = retired(clock)
    update(t, clock, cap)
    update(t, clock, cap + 1)
    for frame in (cap + 2, cap + 3):
        update(t, clock, frame, context=context)
        assert not t.identity_bank.identities


@pytest.mark.parametrize('reason,blocker,label', [
    ('pending_initial_track_rebind', '', 'NEW TRACK 1/2'),
    ('initial_candidate_track_changed', 'startup_anchor_expired', 'RESTART TO RESELECT'),
])
def test_rebind_overlay_explains_pending_or_expired(reason, blocker, label):
    overlay = VideoFrameOverlay(tracks=(VideoTrackOverlay(
        BOX, 2, assignment_reason=reason, initial_enrollment_blocker=blocker),))
    assert label in startup_status(overlay)


def test_real_tracker_rebind_provenance_reaches_startup_control_without_early_motion(clock):
    from types import SimpleNamespace
    import request_0513_modular as runtime
    from car_control_modular.control_types import SensorFrame
    from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController

    t = tracker()
    controller = FollowSafetyController(FollowPolicyConfig(initial_target_confirm_frames=2))
    owner = object.__new__(runtime.PersonTracker)
    owner._rknn_pipeline = SimpleNamespace(tracker=t)
    owner._follow_controller = controller
    for cap in range(1, 27):
        update(t, clock, cap, detected=cap < 3 or cap > 23)
        owner._active_capture_frame_id = cap
        owner._active_capture_timestamp = 10. + .1 * cap
        persons = [(r['display_bbox'], r['uid'], .94, 84000.)
                   for r in t.last_identity_observations if r['uid'] > 0]
        targets = owner._persons_to_targets(persons, width=640, height=480)
        decision = controller.decide(cap, SensorFrame(
            width=640, height=480, persons=targets,
            capture_frame_id=cap, capture_timestamp=10. + .1 * cap))
        if cap < 26:
            assert controller.active_target_id is None
            assert controller.search_state == 'none'
            assert all(a.kind == 'stop' for a in decision.actions)
            assert not decision.is_forwarding
        else:
            assert targets[0].initial_identity_confirmed
            assert controller.active_target_id == 1
            assert controller._has_seen_person
