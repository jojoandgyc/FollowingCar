"""Qualified low-score position updates lateral history without claiming UID.

CAP2083 was right; CAP2087 had strong independent appearance but its low
detector score only kept raw track 22 alive. Detector-only backfill must still
not reverse search; the explicit, bounded target-associated publication can.
"""
from dataclasses import replace

import pytest

from car_control_modular.control_types import PersonTarget, SensorFrame
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.low_quality_lateral import LimitedYawSource


@pytest.fixture
def scene(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("car_control_modular.controllers.time.monotonic", lambda: clock[0])
    controller = FollowSafetyController(FollowPolicyConfig(
        direction_history_enable=True, lost_confirm_frames=3))
    controller.active_target_id = 1
    controller._has_seen_person = True
    controller._direction_latest_visible_capture_id = 2083
    controller._direction_loss_capture_id = 2089
    controller._target_direction_history.record_visible(
        2083, 99.70, target_id=1, frame_width=640, confidence=.885,
        bbox=(256.6964, 0., 564.3388, 479.))
    source = LimitedYawSource(1, 22, 2087, 99.90,
        (114.3334, 2.6674, 433.2266, 476.3291), object())
    return controller, source, clock


def publish(controller, source, **kwargs):
    values = dict(frame_width=640, confidence=.4483, expires_at=100.20)
    values.update(kwargs)
    return controller.note_limited_yaw_direction(source, **values)


def missing(cap=2091, timestamp=100.0):
    return SensorFrame(width=640, height=480, capture_frame_id=cap,
                       capture_timestamp=timestamp)


def test_cap2087_updates_search_side_not_trusted_identity_history(scene):
    c, source, _ = scene
    original = c._target_direction_history.entries
    assert publish(c, source)
    assert c.active_target_id == 1
    assert c._direction_loss_capture_id == 2089
    assert c._direction_latest_visible_capture_id == 2083
    assert c._target_direction_history.entries == original
    c.lost_confirm_frames = 3
    c._capture_lost_exit_direction(missing(), search_entry=True)
    assert c._lost_exit_direction == "left"
    assert c._lost_hint_source == "associated_low_score_position"
    assert c._latest_lateral_direction_side().last_visible_capture_frame_id == 2087


def test_lost_confirmation_uses_new_side_without_extra_stop(scene):
    c, source, _ = scene
    assert publish(c, source)
    c.lost_confirm_frames = 1
    decision = c._lost_confirm_wait_decision(True, missing(2089))
    assert [action.kind for action in decision.actions] == ["rotate_left"]
    assert decision.evidence_capture_frame_id == 2087
    assert not decision.explicit_stop_requested
    assert not decision.soft_stop_requested


def test_new_position_replaces_old_loss_choice_at_actual_search_entry(scene):
    c, source, clock = scene
    c._capture_lost_exit_direction(missing(2089), search_entry=True)
    assert c._lost_exit_direction == "right"
    assert publish(c, source)
    c.lost_confirm_frames = 3
    c._ensure_search_state(missing())
    assert c.search_state == "searching"
    assert c.search_direction == "left"
    # The finite search keeps its chosen side; expiry is not another turn or
    # parking request, and it does not restart the scan's rotation budget.
    c._search_rotation_accumulated_deg = 12.
    clock[0] = 101.
    c._ensure_search_state(missing(2110, 101.))
    assert c.search_direction == "left"
    assert c._search_rotation_accumulated_deg == 12.


@pytest.mark.parametrize("delay", [.05, .10, .20])
def test_admitted_position_history_survives_motion_expiry_for_same_search(scene, delay):
    c, source, clock = scene
    assert publish(c, source)
    c._capture_lost_exit_direction(missing(2089), search_entry=True)
    assert c._lost_exit_direction == "left"
    clock[0] = 100.20 + delay
    c._ensure_search_state(missing(2091, clock[0]))
    assert c.search_direction == "left"
    assert c._lost_hint_source == "associated_low_score_position"
    assert c._limited_yaw_direction.expires_at == 100.20
    # Its position authorization cannot be renewed by another weak frame.
    assert not publish(c, replace(source, capture=2092, timestamp=clock[0]),
                       expires_at=clock[0]+.5)


def test_detector_only_backfill_cannot_reverse_trusted_direction(scene):
    c, _, _ = scene
    for cap, stamp in ((2084, 99.75), (2085, 99.8), (2087, 99.9)):
        c.note_direction_classifier_evidence(cap, stamp, state="visible",
            bbox=(30., 0., 280., 479.), frame_width=640, confidence=.95)
    assert c.note_historical_direction_hint("left", active_target_id=1,
        first_capture_frame_id=2084, last_capture_frame_id=2087,
        selected_capture_frame_ids=(2084, 2085, 2087), confidence=.67,
        loss_capture_frame_id=2089, evidence_timestamp=99.75)
    c._capture_lost_exit_direction(missing(), search_entry=True)
    assert c._lost_exit_direction == "right"
    assert c._lost_hint_source == "latest_reliable_capture_side"


@pytest.mark.parametrize("change", [
    {"uid": 2}, {"uid": 0}, {"track_id": -1}, {"capture": 0},
    {"capture": 2083}, {"timestamp": 99.7}, {"timestamp": 100.1},
    {"timestamp": float("nan")}, {"identity_publication": None},
    {"bbox": (-1., 0., 300., 479.)}, {"bbox": (0., 0., 641., 479.)},
    {"bbox": (1., 0., 1., 479.)}, {"bbox": (1., 5., 100., 4.)},
    {"bbox": (1., 0., 100., float("inf"))},
])
def test_invalid_or_foreign_publication_is_not_direction_evidence(scene, change):
    c, source, _ = scene
    assert not publish(c, replace(source, **change))
    assert c._latest_lateral_direction_side().direction == "right"


@pytest.mark.parametrize("kwargs", [
    {"expires_at": 100.}, {"expires_at": float("inf")},
    {"confidence": -1.}, {"confidence": float("nan")},
    {"frame_width": 0}, {"now": 99.8}, {"now": float("nan")},
])
def test_invalid_deadline_or_frame_not_admitted(scene, kwargs):
    c, source, _ = scene
    assert not publish(c, source, **kwargs)


def test_duplicate_and_older_frames_do_not_renew_deadline(scene):
    c, source, _ = scene
    assert publish(c, source)
    assert not publish(c, source, expires_at=100.5)
    assert not publish(c, replace(source, capture=2086, timestamp=99.95))
    assert not publish(c, replace(source, capture=2088, timestamp=99.85))
    assert c._limited_yaw_direction.expires_at == 100.20


def test_new_weak_frames_cannot_extend_parent_identity_deadline(scene):
    c, source, clock = scene
    assert publish(c, source)
    clock[0] = 100.10
    assert publish(c, replace(source, capture=2089, timestamp=100.05), expires_at=100.6)
    assert c._limited_yaw_direction.expires_at == 100.20
    clock[0] = 100.25
    assert not publish(c, replace(source, capture=2092, timestamp=100.22), expires_at=100.7)
    assert c._limited_yaw_direction.expires_at == 100.20
    assert c._latest_lateral_direction_side().direction == "left"  # Historical side only.


def test_callers_cannot_create_unbounded_position_deadline(scene):
    c, source, _ = scene
    assert publish(c, source, expires_at=999.)
    assert c._limited_yaw_direction.expires_at == pytest.approx(source.timestamp+.5)


def test_new_formal_position_retires_weak_chain(scene):
    c, source, clock = scene
    assert publish(c, source)
    clock[0] = 100.1
    target = PersonTarget(track_id=1, bbox=(400., 0., 600., 479.),
                          confidence=.9, area=95800.)
    frame = missing(2089, 100.05)
    c._record_target_direction_evidence(frame, target, reliable=True)
    assert c._limited_yaw_direction is None
    assert c._latest_lateral_direction_side().direction == "right"
    assert publish(c, replace(source, capture=2090, timestamp=100.08), expires_at=100.55)
    assert c._limited_yaw_direction.expires_at == 100.55


def test_hard_identity_reset_removes_position_evidence(scene):
    c, source, _ = scene
    assert publish(c, source)
    c.clear_active_target("test_identity_reset")
    assert c._limited_yaw_direction is None
    assert not publish(c, replace(source, capture=2088, timestamp=99.95))


def test_pre_brake_position_cannot_undo_direction_retirement(scene):
    c, source, _ = scene
    assert publish(c, source)
    c._target_direction_history.discard_through(99.95)
    assert c._latest_lateral_direction_side().direction is None
    assert not publish(c, replace(source, capture=2088, timestamp=99.94))


def test_explicit_rejection_can_clear_associated_position_only(scene):
    c, source, _ = scene
    assert publish(c, source)
    c.clear_limited_yaw_direction("identity_conflict")
    assert c._latest_lateral_direction_side().direction == "right"
    assert c.active_target_id == 1


def test_new_formal_uid_without_clear_cannot_use_foreign_hint(scene):
    c, source, _ = scene
    assert publish(c, source)
    c.active_target_id = 2
    assert c._latest_lateral_direction_side().reason != "associated_low_score_position"


def test_raw_track_switch_cannot_extend_weak_observation_chain(scene):
    c, source, _ = scene
    assert publish(c, source)
    assert not publish(c, replace(source, track_id=23, capture=2088, timestamp=99.95))


def test_pending_weak_side_does_not_change_active_search_budget(scene):
    c, source, _ = scene
    c.search_state, c.search_direction = "searching", "right"
    c._search_rotation_accumulated_deg = 16.
    assert publish(c, source)
    assert c.search_direction == "right"
    assert c._search_rotation_accumulated_deg == 16.
    assert c._lost_exit_direction is None


def test_position_admitted_before_loss_binds_only_the_first_loss_episode(scene):
    c, source, clock = scene
    c._direction_loss_capture_id = None
    assert publish(c, source)
    assert c._limited_yaw_direction.loss_capture_id is None
    clock[0] = 100.4
    c._capture_lost_exit_direction(missing(), search_entry=True)
    assert c._lost_exit_direction == "left"
    assert c._limited_yaw_direction.loss_capture_id == 2091
    c._direction_loss_capture_id = 2200
    c._capture_lost_exit_direction(missing(2200, 100.4), search_entry=True)
    assert c._lost_exit_direction == "right"


def test_identity_confirmation_ends_weak_direction_episode(scene):
    c, source, clock = scene
    assert publish(c, source)
    clock[0] = 100.4
    c.release_search_on_confirmed_target()
    assert c._limited_yaw_direction is None


@pytest.mark.parametrize("end", ["timeout", "revolution"])
def test_finite_search_completion_retires_admitted_position_history(scene, end):
    c, source, clock = scene
    assert publish(c, source)
    c.search_state, c.search_direction = "searching", "left"
    c._lost_started_at = 100.
    c._search_rotation_started_at = 100.
    if end == "timeout":
        clock[0] = 100.+c.cfg.search_timeout_sec+.1
        decision = c._search_timeout_decision(clock[0])
    else:
        c._search_rotation_accumulated_deg = c.cfg.search_revolution_deg
        decision = c._search_revolution_complete_decision(100.4)
    assert decision is not None
    assert c._limited_yaw_direction is None
