"""Pure policy tests: no camera, serial I/O, or runtime state mutation."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from car_control_modular.depth_torso_selection import (
    TORSO_REGION_NAMES, TorsoSelection, allow_sparse_torso_continuation,
    select_torso_candidate_group,
)


def candidate(distance=1.98, pixels=120, *, name="chest_center", required=68,
              valid=None, spatial=1.0):
    return SimpleNamespace(
        distance_m=distance, pixels=pixels, valid_pixels=pixels if valid is None else valid,
        required_pixels=required, spatial_support_fraction=spatial, region_name=name,
    )


def select(candidates, *, anchor=1.96, age=0.1, **kwargs):
    return select_torso_candidate_group(
        candidates, anchor_distance_m=anchor, anchor_age_sec=age,
        cluster_span_m=0.4, max_distance_jump_m=0.8,
        anchor_strict_age_sec=0.6, anchor_expire_age_sec=1.5, **kwargs,
    )


def sparse(selection, *, anchor=1.96, age=0.1, **kwargs):
    return allow_sparse_torso_continuation(
        selection, anchor_distance_m=anchor, anchor_age_sec=age,
        strict_age_sec=0.6, max_near_distance_m=2.5,
        cluster_span_m=0.4, max_distance_jump_m=0.8, **kwargs,
    )


def competing_surfaces():
    return [candidate()] + [candidate(2.65, 2200, name=name)
                            for name in sorted(TORSO_REGION_NAMES)]


def test_fresh_anchor_beats_many_background_regions_inside_jump_threshold():
    result = select(competing_surfaces())
    assert result.distance_m == pytest.approx(1.98)
    assert result.region_count == 1
    assert result.region_names == ("chest_center",)
    assert result.pixels == 120 and result.required_pixels == 68
    assert result.anchor_consistent
    assert result.selection_reason == "fresh_anchor_consensus"
    assert "1.980m" in result.candidate_summary and "2.650m" in result.candidate_summary
    assert "pixels=120 required=68" in result.candidate_summary
    assert sparse(result)


@pytest.mark.parametrize("age", [0.600001, 1.0, 1.5, 5.0, None, -0.1, float("nan")])
def test_old_or_unknown_anchor_does_not_get_hard_precedence(age):
    result = select(competing_surfaces(), age=age)
    assert result.distance_m == pytest.approx(2.65)
    assert result.region_count == 5
    assert not result.anchor_consistent
    assert result.selection_reason == "group_consensus"
    assert not sparse(result, age=age)


def test_strict_anchor_boundary_is_inclusive():
    result = select(competing_surfaces(), age=0.6)
    assert result.distance_m == pytest.approx(1.98)
    assert sparse(result, age=0.6)


def test_closer_safety_precedence_does_not_bootstrap_sparse_distance():
    near_anchor = candidate(1.96, 1000)
    sudden_closer = candidate(0.9, 120, name="left_torso")
    result = select([near_anchor, sudden_closer])
    assert result.distance_m == 0.9
    assert result.selection_reason == "nearer_safety_override"
    assert not result.anchor_consistent
    assert not sparse(result)


def test_two_supported_near_regions_can_continue_existing_anchor():
    result = select([candidate(), candidate(2.0, 80, name="left_torso")])
    assert result.region_count == 2
    assert result.pixels == 200
    assert sparse(result)


@pytest.mark.parametrize("change", [
    {"pixels": 67}, {"required_pixels": 121}, {"valid_pixels": 119},
    {"pixels": -1}, {"pixels": 120.5}, {"pixels": True},
    {"spatial_support_fraction": 0.54, "valid_pixels": 300},
    {"spatial_support_fraction": 1.01},
    {"spatial_support_fraction": float("nan")},
    {"distance_m": float("inf")}, {"distance_m": 0},
    {"region_name": "whole_torso"}, {"region_name": "none"},
])
def test_invalid_or_insufficient_candidate_does_not_lower_any_gate(change):
    record = vars(candidate()) | change
    assert select([record]) is None


def test_duplicate_region_bands_do_not_inflate_count_or_support():
    result = select([candidate(), candidate(), candidate(2.0, 100)])
    assert result.region_names == ("chest_center",)
    assert result.region_count == 1
    assert result.pixels == 120 and result.valid_pixels == 120
    assert sparse(result)


@pytest.mark.parametrize("anchor,age", [
    (None, 0.1), (1.96, None), (1.96, 0.61), (1.96, -0.01),
    (1.5, 0.1), (3.0, 0.1), (float("nan"), 0.1),
])
def test_sparse_override_requires_current_recent_near_anchor(anchor, age):
    result = select([candidate()])
    assert not sparse(result, anchor=anchor, age=age)


def test_far_continuity_cannot_use_sparse_near_gate():
    result = select([candidate(4.0)], anchor=4.0)
    assert result.anchor_consistent
    assert not sparse(result, anchor=4.0)


@pytest.mark.parametrize("change", [
    {"pixels": 999}, {"valid_pixels": 999}, {"required_pixels": 1},
    {"region_count": 3}, {"region_count": True}, {"region_names": ("whole_torso",)},
    {"distance_m": 1.0}, {"distance_m": float("nan")}, {"distance_m": "bad"},
    {"_region_evidence": ()},
])
def test_forged_selection_aggregate_cannot_authorize_sparse_gate(change):
    result = select([candidate()])
    assert not sparse(replace(result, **change))


def test_sparse_gate_does_not_accept_duck_typed_fake_summary():
    assert not sparse(SimpleNamespace(
        distance_m=1.98, pixels=1000, required_pixels=20,
        region_names=("chest_center",), region_count=1, anchor_consistent=True,
    ))
    assert not sparse(TorsoSelection(
        distance_m=1.98, pixels=1000, valid_pixels=1000, required_pixels=20,
        region_names=("chest_center",), region_count=1, anchor_consistent=True,
        selection_reason="fake", candidate_summary="fake",
    ))


def test_candidate_order_does_not_change_selection():
    candidates = competing_surfaces()
    first, reversed_result = select(candidates), select(list(reversed(candidates)))
    assert first == reversed_result


def test_configured_spatial_support_gate_is_preserved():
    record = candidate(valid=240, spatial=0.6)
    assert select([record]) is not None
    assert select([record], minimum_spatial_support_fraction=0.7) is None
