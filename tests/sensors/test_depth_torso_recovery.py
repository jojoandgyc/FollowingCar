"""Pure bounded recovery policy; no camera, motor, or mutable runtime state."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from car_control_modular.depth_torso_recovery import (
    TorsoRecoveryEvidence, assess_torso_recovery, torso_evidence_continuous,
)
from car_control_modular.depth_torso_selection import select_torso_candidate_group


BBOX = (300.0, 70.0, 450.0, 478.0)


def selection(names=("left_torso",), distance=2.721):
    return select_torso_candidate_group([
        dict(distance_m=distance, pixels=209, valid_pixels=209,
             required_pixels=64, spatial_support_fraction=1.0, region_name=name)
        for name in names
    ], anchor_distance_m=2.136719, anchor_age_sec=1.9,
        cluster_span_m=0.4, max_distance_jump_m=0.8,
        anchor_strict_age_sec=0.6, anchor_expire_age_sec=1.5)


def region_geometry(bbox=BBOX, *, width=640, height=480, depth_width=640, depth_height=480):
    x1, y1, x2, y2 = bbox
    scale_x, scale_y = depth_width / width, depth_height / height
    patch_width = max(16, min(64, round((x2 - x1) * scale_x * 0.22)))
    patch_height = max(16, min(64, round((y2 - y1) * scale_y * 0.16)))
    result = []
    for name, x_ratio, y_ratio in (
        ("chest_center", 0.50, 0.30), ("abdomen_center", 0.50, 0.50),
        ("left_torso", 0.36, 0.42), ("right_torso", 0.64, 0.42),
        ("lower_abdomen", 0.50, 0.68),
    ):
        left = round((x1 + (x2 - x1) * x_ratio) * scale_x - patch_width / 2)
        top = round((y1 + (y2 - y1) * y_ratio) * scale_y - patch_height / 2)
        result.append((name, left, top, left + patch_width, top + patch_height))
    return result


def assess(**changes):
    arguments = dict(
        selection=selection(), bbox=BBOX, frame_width=640, frame_height=480,
        depth_width=640, depth_height=480, regions=region_geometry(),
        anchor_distance_m=2.136719, anchor_age_sec=1.9, sample_timestamp=100.0,
        max_distance_jump_m=0.8, max_anchor_age_sec=3.0, edge_margin_ratio=0.02,
        large_bbox_area_ratio=0.35, large_bbox_height_ratio=0.90,
        min_spatial_support_fraction=0.55,
    )
    arguments.update(changes)
    return assess_torso_recovery(**arguments)


def evidence(**changes):
    accepted, reason = assess()
    assert accepted is not None, reason
    return replace(accepted, **changes)


def continuous(previous, current, **changes):
    config = dict(max_gap_sec=0.25, max_distance_delta_m=0.28, max_rate_m_s=3.0)
    config.update(changes)
    return torso_evidence_continuous(previous, current, **config)


@pytest.mark.parametrize("names", [
    ("left_torso",), ("chest_center",), ("abdomen_center",), ("right_torso",),
    ("abdomen_center", "left_torso"),
])
def test_old_anchor_and_bottom_clipped_core_patch_can_supply_bounded_evidence(names):
    result, reason = assess(selection=selection(names))
    assert isinstance(result, TorsoRecoveryEvidence)
    assert result.distance_m == pytest.approx(2.721)
    assert result.sample_timestamp == 100.0 and result.bbox == BBOX
    assert result.frame_width == 640 and result.frame_height == 480
    assert result.region_names == names
    assert reason == "bottom_only_core_torso"
    with pytest.raises(FrozenInstanceError):
        result.distance_m = 4.2


def test_scaled_depth_geometry_is_checked_in_depth_coordinates():
    result, reason = assess(
        depth_width=320, depth_height=240,
        regions=region_geometry(depth_width=320, depth_height=240),
    )
    assert result is not None, reason


@pytest.mark.parametrize("bbox,reason", [
    ((0.0, 70.0, 450.0, 478.0), "not_bottom_only"),
    ((300.0, 0.0, 450.0, 478.0), "not_bottom_only"),
    ((300.0, 70.0, 640.0, 478.0), "not_bottom_only"),
    ((300.0, 70.0, 450.0, 450.0), "not_bottom_only"),
    ((12.8, 70.0, 450.0, 478.0), "not_bottom_only"),
    ((300.0, 9.6, 450.0, 478.0), "not_bottom_only"),
    ((300.0, 20.0, 450.0, 478.0), "large_bbox"),
    ((100.0, 70.0, 500.0, 478.0), "large_bbox"),
    ((300.0, 70.0, 450.0, 481.0), "invalid_bbox"),
    ((450.0, 70.0, 300.0, 478.0), "invalid_bbox"),
    ((300.0, 70.0, float("nan"), 478.0), "invalid_bbox"),
])
def test_other_clips_large_camera_fill_and_invalid_boxes_do_not_get_exception(bbox, reason):
    assert assess(bbox=bbox) == (None, reason)


@pytest.mark.parametrize("changes", [
    {"anchor_distance_m": None}, {"anchor_distance_m": 0.0},
    {"anchor_distance_m": float("nan")}, {"anchor_age_sec": None},
    {"anchor_age_sec": -0.01}, {"anchor_age_sec": 3.0001},
    {"anchor_age_sec": 2.01, "max_anchor_age_sec": 2.0},
    {"anchor_age_sec": 3.01, "max_anchor_age_sec": 10.0},
    {"sample_timestamp": 0.0}, {"sample_timestamp": float("inf")},
    {"sample_timestamp": None}, {"sample_timestamp": True},
    {"frame_width": 0}, {"frame_height": None}, {"depth_height": True},
    {"depth_width": 640.1}, {"max_distance_jump_m": float("nan")},
    {"max_distance_jump_m": -1.0}, {"max_anchor_age_sec": 0},
    {"edge_margin_ratio": -0.1}, {"large_bbox_area_ratio": 0.0},
    {"large_bbox_height_ratio": 2.0}, {"min_spatial_support_fraction": 1.1},
])
def test_missing_or_invalid_geometry_time_anchor_and_config_fail_closed(changes):
    assert assess(**changes)[0] is None


def test_anchor_age_and_distance_boundaries_are_inclusive_but_not_expandable():
    assert assess(anchor_age_sec=3.0)[0] is not None
    assert assess(selection=selection(distance=2.736719))[0] is not None
    assert assess(selection=selection(distance=2.73672))[0] is None
    assert assess(max_distance_jump_m=0.5)[0] is None
    assert assess(selection=selection(distance=3.0), anchor_distance_m=2.5)[0] is not None
    assert assess(selection=selection(distance=3.0001), anchor_distance_m=2.5)[0] is None
    assert assess(selection=selection(distance=4.2), anchor_distance_m=4.2)[0] is None


@pytest.mark.parametrize("names", [
    ("lower_abdomen",), ("lower_abdomen", "left_torso"),
    ("chest_center", "abdomen_center", "left_torso"),
])
def test_lower_background_and_three_region_paths_are_not_reclassified(names):
    assert assess(selection=selection(names))[0] is None


@pytest.mark.parametrize("changes", [
    {"pixels": 999}, {"valid_pixels": 999}, {"required_pixels": 1},
    {"region_count": 2}, {"region_count": True},
    {"region_names": ("whole_torso",)}, {"region_names": ("left_torso", "left_torso")},
    {"distance_m": 2.0}, {"distance_m": float("nan")},
    {"_region_evidence": ()}, {"_region_evidence": None},
])
def test_forged_aggregate_is_not_spatial_evidence(changes):
    assert assess(selection=replace(selection(), **changes))[0] is None


@pytest.mark.parametrize("changes", [
    {"pixels": 63}, {"valid_pixels": 208}, {"required_pixels": 210},
    {"spatial_support_fraction": 0.54}, {"spatial_support_fraction": float("nan")},
    {"distance_m": float("nan")}, {"region_name": "whole_torso"},
])
def test_per_region_pixel_and_spatial_gates_are_revalidated(changes):
    selected = selection()
    bad_region = replace(selected._region_evidence[0], **changes)
    assert assess(selection=replace(selected, _region_evidence=(bad_region,)))[0] is None


def test_lower_configured_support_does_not_bypass_actual_spatial_pixels():
    selected = selection()
    item = replace(selected._region_evidence[0], valid_pixels=400, spatial_support_fraction=0.6)
    selected = replace(selected, valid_pixels=400, _region_evidence=(item,))
    assert assess(selection=selected)[0] is not None
    assert assess(selection=selected, min_spatial_support_fraction=0.7)[0] is None


def test_average_cannot_hide_one_region_beyond_anchor_delta():
    selected = selection(("abdomen_center", "left_torso"))
    first, second = selected._region_evidence
    selected = replace(selected, distance_m=2.7, _region_evidence=(
        replace(first, distance_m=2.55), replace(second, distance_m=2.85),
    ))
    assert assess(selection=selected) == (None, "anchor_distance_discontinuity")


@pytest.mark.parametrize("bad_region,reason", [
    (("left_torso", 0, 200, 33, 264), "region_not_complete"),
    (("left_torso", 620, 200, 640, 264), "region_not_complete"),
    (("left_torso", 338, 0, 371, 64), "region_not_complete"),
    (("left_torso", 338, 416, 371, 480), "region_not_complete"),
    (("left_torso", 280, 200, 313, 264), "region_not_complete"),
    (("left_torso", 345, 200, 378, 264), "region_shifted"),
    (("left_torso", 338.5, 200, 371, 264), "invalid_region_geometry"),
    (("left_torso", None, 200, 371, 264), "invalid_region_geometry"),
])
def test_selected_patch_must_be_complete_unshifted_and_inside_raw_box(bad_region, reason):
    regions = [bad_region if item[0] == "left_torso" else item for item in region_geometry()]
    assert assess(regions=regions) == (None, reason)


def test_missing_duplicate_and_incomplete_region_metadata_are_rejected():
    regions = region_geometry()
    assert assess(regions=[item for item in regions if item[0] != "left_torso"])[0] is None
    assert assess(regions=regions + [regions[0]])[0] is None
    assert assess(regions=[("left_torso", 1, 2, 3, 4, 5)])[0] is None
    assert assess(regions=None)[0] is None
    assert assess(selection=None)[0] is None


def test_two_physical_samples_with_shared_core_patch_and_stable_geometry_are_continuous():
    first = evidence()
    second = evidence(sample_timestamp=100.10, distance_m=2.74,
                      region_names=("abdomen_center", "left_torso"),
                      bbox=(301.0, 72.0, 451.0, 478.0))
    assert continuous(first, second)
    assert first.sample_timestamp == 100.0  # The helper owns no watermark/state.


@pytest.mark.parametrize("changes", [
    {"sample_timestamp": 100.0}, {"sample_timestamp": 99.99},
    {"sample_timestamp": 100.251}, {"sample_timestamp": float("nan")},
    {"sample_timestamp": None}, {"sample_timestamp": True},
    {"distance_m": 3.01}, {"distance_m": float("inf")},
    {"distance_m": 2.44}, {"distance_m": 0.0},
    {"region_names": ("right_torso",)}, {"region_names": ("whole_torso",)},
    {"region_names": ("left_torso", "left_torso")},
    {"frame_width": 1280}, {"frame_height": None},
    {"bbox": (355.0, 70.0, 505.0, 478.0)},
    {"bbox": (320.0, 150.0, 430.0, 478.0)},
    {"bbox": (float("nan"), 70.0, 450.0, 478.0)},
])
def test_duplicate_old_gapped_far_unrelated_or_changed_geometry_breaks_continuity(changes):
    first = evidence()
    second = evidence(sample_timestamp=100.05)
    assert not continuous(first, replace(second, **changes))


def test_physical_rate_and_tighter_callers_limits_are_preserved():
    first = evidence()
    assert not continuous(first, evidence(sample_timestamp=100.033, distance_m=2.90))
    assert continuous(first, evidence(sample_timestamp=100.10, distance_m=2.90))
    assert not continuous(first, evidence(sample_timestamp=100.10), max_gap_sec=0.05)
    assert not continuous(first, evidence(sample_timestamp=100.10, distance_m=2.74),
                          max_distance_delta_m=0.01)
    assert not continuous(first, evidence(sample_timestamp=100.26), max_gap_sec=10.0)
    assert continuous(first, evidence(sample_timestamp=100.25))


def test_normalized_center_and_area_boundaries():
    first = evidence()
    assert continuous(first, evidence(sample_timestamp=100.1,
                                     bbox=(351.2, 70.0, 501.2, 478.0)))
    assert not continuous(first, evidence(sample_timestamp=100.1,
                                         bbox=(351.21, 70.0, 501.21, 478.0)))
    assert continuous(first, evidence(sample_timestamp=100.1,
                                     bbox=(318.75, 70.0, 431.25, 478.0)))
    assert not continuous(first, evidence(sample_timestamp=100.1,
                                         bbox=(318.76, 70.0, 431.24, 478.0)))


@pytest.mark.parametrize("changes", [
    {"max_gap_sec": None}, {"max_gap_sec": 0},
    {"max_distance_delta_m": -0.1}, {"max_distance_delta_m": float("nan")},
    {"max_rate_m_s": 0}, {"max_rate_m_s": float("inf")},
])
def test_invalid_continuity_config_fails_closed(changes):
    assert not continuous(evidence(), evidence(sample_timestamp=100.1), **changes)


def test_continuity_never_accepts_untyped_summaries():
    assert not continuous(None, evidence())
    assert not continuous(evidence(), vars(evidence()))
