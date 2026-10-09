"""Startup-only bridge boundaries and operator diagnostics; no hardware."""
import numpy as np
import pytest

from rk_vision.initial_enrollment import InitialEnrollment
from car_control_modular.video_recorder import VideoFrameOverlay, VideoTrackOverlay, startup_status


def metadata(frame):
    return dict(frame_index=frame, capture_frame_id=frame, capture_timestamp=100.+frame*.1,
                detector_bbox=(100., 1., 450., 476.), image_width=640, image_height=480,
                is_fresh=True, candidate_count=1, quality_bbox_ok=True,
                bbox_quality_tier='strong', initial_color_source='hsv_crop_v1_bgr',
                initial_color_feature=[1.] + [0.]*15)


def test_pose_change_after_first_capture_can_start_a_new_pair_without_weak_frame():
    enrollment = InitialEnrollment()
    first = np.array([1., 0., 0.])
    pose = np.array([.7, np.sqrt(.51), 0.])
    assert enrollment.observe(1, first, metadata(1)) == (False, 'pending_initial_identity')
    assert enrollment.observe(1, pose, metadata(2)) == (False, 'pending_initial_identity')
    assert enrollment.observe(1, pose, metadata(3)) == (True, 'created_confirmed')
    assert enrollment.last_evidence['path'] == 'anchored_pose_pair'
    assert enrollment.last_evidence['anchor_full_distance'] == pytest.approx(.30)
    assert enrollment.last_evidence['seed_cap'] == 1


@pytest.mark.parametrize('blocker', ['continuity_broken', 'pose_window_expired'])
@pytest.mark.parametrize('reason', ['initial_candidate_appearance_mismatch', 'weak_bbox_unassigned'])
def test_video_explains_unresolved_startup_instead_of_generic_wait(blocker, reason):
    overlay = VideoFrameOverlay(tracks=(VideoTrackOverlay(
        (100., 1., 450., 476.), 1,
        assignment_reason=reason,
        initial_enrollment_blocker=blocker),))
    assert 'RESTART TO RESELECT' in startup_status(overlay)
