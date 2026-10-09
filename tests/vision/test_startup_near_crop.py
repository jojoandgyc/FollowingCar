"""21:17 run CAP26/27 geometry, synthetic features; no model/hardware replay."""
import numpy as np
import pytest

from test_startup_enrollment_20260924 import meta, bank, assign, V, OLD27
from rk_vision.template_memory import TemplateMemory
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection
from car_control_modular.video_recorder import VideoFrameOverlay, VideoTrackOverlay, startup_status

CAP26 = (94.50535583496094, 0., 547.6231079101562, 475.04461669921875)
CAP27 = (88.46308898925781, 1.164703369140625, 541.38330078125, 474.5960693359375)


def test_actual_wide_near_crops_can_enroll_but_never_on_first_capture():
    b = bank()
    first = meta(26, 32562.679879077, 3, CAP26)
    assert not TemplateMemory.initial_crop_usable(first)  # Other new identities unchanged.
    assert assign(b, first) == 0
    evidence = b.last_assignments[1]['initial_enrollment_evidence']
    assert evidence['path'] == 'near_vertical_crop' and evidence['usable']
    assert evidence['aspect_ratio'] == pytest.approx(.9538425)
    assert not b.identities and not b.track_to_uid
    assert assign(b, meta(27, 32562.744457884, 4, CAP27)) == 1
    assert b.last_assignments[1]['reason'] == 'created_confirmed'


@pytest.mark.parametrize('box,reason', [
    (OLD27, 'lateral_crop'),
    ((0., 0., 600., 480.), 'lateral_crop'),
    ((30., 0., 610., 480.), 'aspect_ratio_out_of_range'),  # >1.20
    ((100., 35., 500., 455.), 'aspect_ratio_out_of_range'),  # No vertical crop
    ((100., 0., 470., 400.), 'aspect_ratio_out_of_range'),  # Not tall enough
    ((200., 0., 240., 470.), 'body_too_small'),
    ((210., 50., 300., 190.), 'body_too_small'),
    ((float('nan'), 0., 500., 480.), 'geometry_unavailable'),
])
def test_wide_path_keeps_crop_boundaries(box, reason):
    m = meta(box=box)
    evidence = TemplateMemory.initial_crop_evidence(m, allow_near_vertical_crop=True)
    assert not evidence['usable'] and evidence['reason'] == reason
    b = bank()
    # Malformed geometry only tested at gate; bank normalizes boxes upstream.
    if reason == 'geometry_unavailable':
        return
    for i in range(3):
        assert assign(b, meta(26+i, 100.+i*.1, 3+i, box)) == 0
    assert not b.identities


@pytest.mark.parametrize('case', ['multi', 'stale', 'weak', 'duplicate', 'feature', 'gap', 'jump'])
def test_near_crop_does_not_bypass_identity_proof(case):
    b = bank()
    first = meta(26, 100., 3, CAP26)
    assert assign(b, first) == 0
    m = meta(27, 100.1, 4, CAP27)
    vector = V
    if case == 'multi': m['candidate_count'] = 2
    if case == 'stale': m['is_fresh'] = False
    if case == 'weak': m.update(quality_bbox_ok=False, bbox_quality_tier='weak')
    if case == 'duplicate': m = first
    if case == 'feature': vector = np.array([0., 1., 0.])
    if case == 'gap': m['capture_timestamp'] = 100.5
    if case == 'jump': m['detector_bbox'] = (380., 2., 625., 475.)
    assert assign(b, m, vector=vector) == 0
    assert not b.identities


def test_real_tracker_metadata_and_bank_diagnostics_for_near_crops():
    tracker = DeepSortTracker(DeepSortTrackerConfig(n_init=1,
        identity_template_memory_enable=True, identity_template_crosscheck_enable=True,
        identity_appearance_region_safety_enable=True))
    uids = []
    for i, box in enumerate([CAP26, CAP26, CAP27, CAP27]):
        records = tracker.update([Detection(box, .94, 0)], [V], partial_features=[V],
            partial_feature_sources=['osnet_torso'], image_width=640, image_height=480,
            frame_context=dict(control_frame_id=3+i, capture_frame_id=26+i,
                               capture_timestamp=100.+i*.07))
        uids.append([r.reid_uid for r in records])
        if i == 1:
            assert tracker.identity_bank.last_assignments[1]['initial_enrollment_evidence']['path'] == 'near_vertical_crop'
    assert uids == [[], [0], [1], [1]]


@pytest.mark.parametrize('detail,label', [
    ('lateral_crop', 'LEFT / RIGHT CROP'),
    ('aspect_ratio_out_of_range', 'ASPECT RATIO'),
    ('multiple_candidates', 'ONE PERSON REQUIRED'),
])
def test_video_specific_rejection(detail, label):
    overlay = VideoFrameOverlay(tracks=(VideoTrackOverlay(CAP26, 1,
        assignment_reason='initial_crop_incomplete', initial_enrollment_detail=detail),))
    assert label in startup_status(overlay)


def test_bank_logs_crop_and_ambiguity_without_changing_generic_rejection():
    b = bank()
    assert assign(b, meta(box=OLD27)) == 0
    assert b.last_assignments[1]['reason'] == 'initial_crop_incomplete'
    assert b.last_assignments[1]['initial_enrollment_evidence']['reason'] == 'lateral_crop'
    assert assign(b, meta(box=CAP26, candidate_count=2)) == 0
    assert b.last_assignments[1]['initial_enrollment_evidence']['reason'] == 'multiple_candidates'
