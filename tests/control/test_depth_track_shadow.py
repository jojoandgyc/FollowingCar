"""No hardware. Independent depth association and source-time safety gates."""
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from car_control_modular.depth_track_shadow import CameraPose, DepthTrackShadow, Intrinsics
from car_control_modular.astra_depth import AstraDepthConfig, AstraDepthRuntime
from tools.depth_track_shadow_eval import replay

K = Intrinsics(160, 120, 140, 140, 80, 60)


def depth(cx=80, z=2000):
    image = np.full((120, 160), 5000, dtype=np.uint16)
    image[40:80, cx-10:cx+10] = z
    return image


def seed(tracker, **overrides):
    args = dict(timestamp=10., pose=CameraPose(10., 0, 0, 0), uid=1,
                visual_timestamp=10., torso_bbox=(60, 30, 100, 90),
                trusted_distance_m=2., now=10.15, identity_confirmed=True)
    args.update(overrides)
    return tracker.seed(depth(), **args)


def update(tracker, t, image=None, **overrides):
    args = dict(timestamp=t, pose=CameraPose(t, 0, 0, 0), now=t+.01, active_uid=1)
    args.update(overrides)
    return tracker.update(depth() if image is None else image, **args)


def test_ten_depth_updates_without_any_new_visual_bbox():
    tracker = DepthTrackShadow(K)
    assert seed(tracker).status == "seeded"
    for i in range(1, 11):
        result = update(tracker, 10+i/30, depth(cx=80+i))
        assert result.status == "tracked"
        assert result.visual_timestamp == 10.
        assert result.control_allowed is False
    assert result.camera_xyz[0] > .1


def test_delayed_visual_seed_replays_intermediate_physical_frames():
    tracker = DepthTrackShadow(K)
    assert seed(tracker, now=10.16).status == "seeded"
    for i in range(1, 6):
        result = update(tracker, 10+i/30, depth(cx=80+i), now=10.17)
        assert result.status == "tracked"
    assert tracker.last_ts == pytest.approx(10+5/30)


def test_depth_updates_do_not_renew_visual_lease():
    tracker = DepthTrackShadow(K); seed(tracker)
    for t in (10.1, 10.2, 10.3):
        assert update(tracker, t).status == "tracked"
    assert update(tracker, 10.36).status == "visual_lease_expired"
    assert update(tracker, 10.37).status == "needs_visual_seed"


@pytest.mark.parametrize("yaw", [-.07, .07])
def test_camera_rotation_and_translation_preserve_world_position(yaw):
    tracker = DepthTrackShadow(K); seed(tracker)
    point = tracker.position.copy()
    pose = CameraPose(10.05, .005, .02, yaw)
    p = pose.camera(point)
    cx = round(K.fx*p[0]/p[2]+K.cx+.5)
    result = update(tracker, 10.05, depth(cx=cx, z=round(p[2]*1000)), pose=pose)
    assert result.status == "tracked"
    assert np.linalg.norm(np.array(result.world_xyz)-point) < .02


@pytest.mark.parametrize("kwargs,status", [
    ({"active_uid": 2}, "identity_or_search_revoked"),
    ({"searching": True}, "identity_or_search_revoked"),
    ({"pose": None}, "pose_missing_or_stale"),
    ({"pose": CameraPose(9., 0, 0, 0)}, "pose_missing_or_stale"),
    ({"now": 10.30}, "depth_expired_or_gap"),
])
def test_revocation_and_invalid_motion_feedback(kwargs, status):
    tracker = DepthTrackShadow(K); seed(tracker)
    assert update(tracker, 10.05, **kwargs).status == status
    assert update(tracker, 10.06).status == "needs_visual_seed"


def test_duplicate_cannot_accumulate_velocity_or_move_timestamps():
    tracker = DepthTrackShadow(K); seed(tracker)
    before = tracker.position.copy()
    for _ in range(3):
        assert update(tracker, 10., depth(cx=100)).status == "duplicate_or_old"
    assert tracker.last_ts == tracker.visual_ts == 10.
    assert np.array_equal(before, tracker.position)
    assert update(tracker, 10., now=10.36).status == 'visual_lease_expired'


def test_continuous_wall_instead_of_torso_is_rejected():
    tracker = DepthTrackShadow(K); seed(tracker)
    wall = np.full((120, 160), 2000, dtype=np.uint16)
    assert update(tracker, 10.033, wall).status == 'association_ambiguous_or_lost'


def test_missing_intermediate_physical_frames_require_new_seed():
    tracker = DepthTrackShadow(K); seed(tracker)
    assert update(tracker, 10.20).status == 'depth_expired_or_gap'


@pytest.mark.parametrize("image", [np.zeros((120, 160), dtype=np.uint16),
                                 np.full((120, 160), 5000, dtype=np.uint16),
                                 depth(cx=110), depth(z=3000)])
def test_lost_target_does_not_bind_background_or_far_jump(image):
    tracker = DepthTrackShadow(K); seed(tracker)
    result = update(tracker, 10.033, image)
    assert result.status == "association_ambiguous_or_lost"
    assert result.camera_xyz is None


def test_two_plausible_components_are_not_resolved_by_nearest_depth():
    tracker = DepthTrackShadow(K); seed(tracker)
    image = depth(); image[40:80, 79:81] = 0
    result = update(tracker, 10.1, image)
    assert result.status == "association_ambiguous_or_lost"
    assert result.candidate_count == 2


@pytest.mark.parametrize("kwargs,status", [
    ({"identity_confirmed": False}, "unconfirmed_identity"),
    ({"uid": 0}, "unconfirmed_identity"),
    ({"visual_timestamp": 9.8}, "invalid_visual_anchor"),
    ({"trusted_distance_m": 7.}, "seed_ambiguous_or_empty"),
    ({"torso_bbox": (-1, 0, 40, 40)}, "invalid_torso_roi"),
])
def test_seed_requires_identity_alignment_and_distance(kwargs, status):
    tracker = DepthTrackShadow(K)
    assert seed(tracker, **kwargs).status == status
    assert tracker.position is None


def test_depth_history_copies_do_not_modify_ranging_or_capture_state():
    sensor = AstraDepthRuntime(AstraDepthConfig())
    images = [depth() for _ in range(4)]
    sensor._depth_history.extend(zip([10., 10.033, 10.066, 10.1], images))
    before = sensor._last_accepted_ts
    result = sensor.copy_depth_history(after_timestamp=10., max_frames=2)
    assert [t for t, _ in result] == [10.066, 10.1]
    assert not result[0][1].flags.writeable
    assert not np.shares_memory(result[0][1], images[2])
    assert sensor._last_accepted_ts == before
    with pytest.raises(ValueError): sensor.copy_depth_history(after_timestamp=0, max_frames=17)


def test_manifest_replay_reports_shadow_only(tmp_path):
    np.savez(tmp_path / 'd.npz', depth_mm=depth())
    manifest = dict(schema_version=1, depth_geometry='registered_rectified_orientation_verified',
                    intrinsics=asdict(K), frames=[dict(frame_id=1, depth_file='d.npz',
                    timestamp=10., observed_at=10.1, active_uid=1,
                    pose=asdict(CameraPose(10., 0, 0, 0)), anchor=dict(uid=1,
                    visual_timestamp=10., torso_bbox=[60,30,100,90],
                    trusted_distance_m=2., identity_confirmed=True))])
    result = list(replay(manifest, tmp_path))[0]
    assert result['status'] == 'seeded'
    assert not result['control_allowed'] and not result['motor_authority_created']
