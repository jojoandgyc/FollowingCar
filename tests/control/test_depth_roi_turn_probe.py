"""Projection is an observation-only hypothesis, never identity/drive proof."""
import math
from dataclasses import replace
import numpy as np
import pytest

from car_control_modular.depth_roi_turn_probe import project_bbox, validate_regions
from car_control_modular.depth_track_shadow import CameraPose
from test_depth_track_online import session, anchor, feedback


def projection(**changes):
    args = dict(bbox=(40., 20., 120., 100.), width=160, height=120, hfov_deg=60,
                distance=2., source_pose=CameraPose(10, 0, 0, 0),
                depth_pose=CameraPose(10.2, 0, 0, math.radians(4)))
    args.update(changes)
    return project_bbox(**args)


def test_right_turn_projects_target_left_and_inverse_turn_right():
    b, z, status = projection()
    assert status == "projected" and (b[0]+b[2])/2 < 80
    b, _, _ = projection(depth_pose=CameraPose(10.2, 0, 0, math.radians(-4)))
    assert (b[0]+b[2])/2 > 80


def test_translation_changes_depth_and_bbox_scale():
    b, z, _ = projection(depth_pose=CameraPose(10.2, 0, .1, 0))
    assert z == pytest.approx(1.9)
    assert b[2]-b[0] > 80


@pytest.mark.parametrize("kwargs", [dict(source_pose=None), dict(distance=float("nan")),
    dict(distance=.1), dict(hfov_deg=0), dict(width=0),
    dict(depth_pose=CameraPose(10.3,0,0,0)), dict(depth_pose=CameraPose(9.9,0,0,0)),
    dict(depth_pose=CameraPose(10.2,0,0,math.radians(9))),
    dict(depth_pose=CameraPose(10.2,0,.5,0))])
def test_invalid_or_excess_motion_never_projects(kwargs):
    assert projection(**kwargs)[0] is None


def test_current_depth_regions_reject_far_background_and_single_strip():
    box, z, _ = projection(depth_pose=CameraPose(10.2,0,0,0))
    d = np.full((120,160), 5000, dtype=np.uint16)
    assert not validate_regions(d, box, z)[0]
    d[40:72,56:72] = 2000
    assert not validate_regions(d, box, z)[0]
    d[40:72,72:88] = 2000
    assert validate_regions(d, box, z)[0]
    assert d[0,0] == 5000


def test_probe_is_deduplicated_and_never_changes_tracker_or_motor_authority():
    s = session(10.24)
    s.add_frames([(10.2, np.full((120,160), 2000, dtype=np.uint16))])
    before = s.tracker, s.anchor_key, s.last_success, s.last_emitted
    rows = s.probe_turn_roi(anchor(), (1,10.,2.),10.21)
    assert len(rows) == 1 and rows[0]["roi_probe_status"] == "candidate_consistent"
    assert not rows[0]["control_allowed"] and not rows[0]["motor_authority_created"]
    assert rows[0]["geometry_verified"] is False
    assert (s.tracker,s.anchor_key,s.last_success,s.last_emitted) == before
    assert s.probe_turn_roi(anchor(), (1,10.,2.),10.22) == []


def test_probe_does_not_extrapolate_missing_pose_or_allow_wrong_uid():
    s = session(10.24)
    s.add_frames([(10.2, np.full((120,160),2000,dtype=np.uint16))])
    s.poses.samples.clear()
    row = s.probe_turn_roi(anchor(), (1,10.,2.),10.27)
    assert row == []  # never extends the 250ms observation window
    row = s.probe_turn_roi(anchor(), (2,10.,2.),10.21)
    assert row == []  # briefly wait for a bracket, never extrapolate


def test_probe_rejects_unrelated_uid_with_bracketed_poses():
    s = session(10.24)
    s.add_frames([(10.2,np.full((120,160),2000,dtype=np.uint16))])
    rows = s.probe_turn_roi(anchor(), (2,10.,2.),10.21)
    assert rows[0]["roi_probe_status"] == "pose_or_range_missing"
    assert not rows[0]["motor_authority_created"]
