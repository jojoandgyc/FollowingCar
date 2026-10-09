"""One visual owner updates yaw; held distance never re-enters depth control."""
from dataclasses import replace
import pickle

import pytest

from car_control_modular.control_types import (
    DistanceState, HazardState, ObstacleState, PersonTarget, SensorFrame, SteeringFeedback,
)
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController


@pytest.fixture
def lateral(monkeypatch):
    monkeypatch.setattr("car_control_modular.controllers.time.monotonic", lambda: 100.0)
    controller = FollowSafetyController(FollowPolicyConfig(
        distance_pid_enable=True, distance_control_mode="distance_pi",
        distance_target_motion_control_enable=False,
        visible_steering_pid_enable=True, forward_max_rpm=200,
        direction_history_enable=True,
    ))
    controller._has_seen_person = True
    controller.active_target_id = 1
    controller._live_longitudinal_authority_reader = lambda uid: ("forward", 30, uid, 99.96)
    target = PersonTarget((440.0, 50.0, 600.0, 460.0), 1, .95, 65600.0)
    frame = SensorFrame(
        width=640, height=480, persons=[target],
        capture_frame_id=287, capture_timestamp=99.98,
        distance_m=None,
        distance_state=DistanceState(source="vision_depth", source_detail="depth_hold",
                                     sample_timestamp=99.4, used_distance_m=2.5),
        steering_feedback=SteeringFeedback(timestamp=99.99, trustworthy=True,
                                           left_forward_rpm=40, right_forward_rpm=40),
    )
    return controller, frame


def _longitudinal_state(controller):
    names = [name for name in vars(controller) if name.startswith((
        "_distance", "_depth", "_longitudinal", "_reverse", "_forward", "_target_stop",
        "_normal_parking", "_last_mmwave", "_mmwave",
    )) or name == "last_distance_pid_result"]
    return pickle.dumps({name: getattr(controller, name) for name in names})


@pytest.mark.parametrize("distance", [None, .1, 1.4, 1.8, 8.0])
@pytest.mark.parametrize("grant", [None, ("forward", 30, 1, 99.96)])
def test_yaw_reads_live_grant_without_any_longitudinal_state_change(lateral, monkeypatch, distance, grant):
    controller, frame = lateral
    controller._live_longitudinal_authority_reader = lambda uid: grant
    frame = replace(frame, distance_m=distance)
    before = _longitudinal_state(controller)
    def forbidden(*args, **kwargs):
        pytest.fail("visual-only frame must not run or reset longitudinal control")
    for name in ("_observe_longitudinal_motion", "_update_distance_pid", "_pause_distance_pi",
                 "_reset_distance_pid", "_remember_target_distance", "_visible_base_forward_percent",
                 "_reverse_control_decision", "_target_distance_lock_decision"):
        monkeypatch.setattr(controller, name, forbidden)
    decision = controller.decide(30, frame, lateral_only=True)
    assert decision.lateral_only and not decision.explicit_stop_requested
    assert not decision.soft_stop_requested and not decision.stop_action_execution
    assert len(decision.actions) == 1 and decision.actions[0].kind == "steer_right"
    assert decision.actions[0].steer_correction_rpm > 0
    assert decision.current_forward_percent <= (30 if grant is not None else 0)
    assert controller.last_steering_pid_result is not None
    assert controller.last_person_center_x == 520
    assert controller.last_selected_target is frame.persons[0]
    assert controller._direction_latest_visible_capture_id == 287
    # Methods were patched, but none of the owned state may be changed.
    for name in ("_observe_longitudinal_motion", "_update_distance_pid", "_pause_distance_pi",
                 "_reset_distance_pid", "_remember_target_distance", "_visible_base_forward_percent",
                 "_reverse_control_decision", "_target_distance_lock_decision"):
        monkeypatch.delattr(controller, name)
    assert _longitudinal_state(controller) == before


def test_center_zero_only_clears_yaw_without_stop_or_distance_pause(lateral):
    controller, frame = lateral
    controller._live_longitudinal_authority_reader = lambda uid: None
    centered = replace(frame.persons[0], bbox=(240., 50., 400., 460.))
    before = _longitudinal_state(controller)
    decision = controller.decide(31, replace(frame, persons=[centered]), lateral_only=True)
    assert decision.lateral_only and decision.actions[0].kind == "forward"
    assert decision.actions[0].speed_percent == 0
    assert not any((decision.explicit_stop_requested, decision.soft_stop_requested,
                    decision.stop_action_execution, decision.clear_action_queue))
    assert _longitudinal_state(controller) == before


@pytest.mark.parametrize("grant", [
    ("forward", 30, 2, 99.9), ("backward", 30, 1, 99.9),
    ("forward", 30, 1, 101.), ("forward", float("nan"), 1, 99.9),
    ("forward", 30, 1, float("nan")), (), ("forward",),
])
def test_invalid_live_grant_cannot_supply_forward_base(lateral, grant):
    controller, frame = lateral
    controller._live_longitudinal_authority_reader = lambda uid: grant
    decision = controller.decide(32, frame, lateral_only=True)
    assert decision.lateral_only and decision.current_forward_percent == 0
    assert not decision.explicit_stop_requested


@pytest.mark.parametrize("event", ["hazard", "front", "left", "right"])
def test_real_safety_stops_precede_lateral_only(lateral, event):
    controller, frame = lateral
    if event == "hazard":
        frame = replace(frame, hazard=HazardState(active=True, reason="danger"))
    else:
        frame = replace(frame, obstacles=ObstacleState(**{event: True}))
    decision = controller.decide(33, frame, lateral_only=True)
    assert not decision.lateral_only
    assert decision.explicit_stop_requested and decision.stop_action_execution
    assert decision.actions == []


@pytest.mark.parametrize("event", ["startup", "search", "lost", "uid", "low_quality", "unsteerable", "rotation"])
def test_identity_startup_and_special_modes_cannot_use_lateral_bypass(lateral, event):
    controller, frame = lateral
    flags = {}
    if event == "startup":
        controller._has_seen_person = False
        controller.active_target_id = None
    elif event == "search":
        controller.search_state = "searching"
    elif event == "lost":
        frame = replace(frame, persons=[])
    elif event == "uid":
        frame = replace(frame, persons=[replace(frame.persons[0], track_id=2)])
    elif event == "low_quality":
        flags["low_quality_visible"] = True
    elif event == "unsteerable":
        flags["target_steerable"] = False
    elif event == "rotation":
        flags["rotation_only"] = True
    decision = controller.decide(34, frame, lateral_only=True, **flags)
    assert not decision.lateral_only


def test_longitudinal_and_lateral_only_are_mutually_exclusive(lateral):
    controller, frame = lateral
    with pytest.raises(ValueError, match="mutually exclusive"):
        controller.decide(35, frame, lateral_only=True, longitudinal_only=True)


def test_existing_real_pi_history_survives_visual_hold_and_new_yaw(lateral, monkeypatch):
    controller, frame = lateral
    controller._live_longitudinal_authority_reader = None
    for index in range(8):
        now = 99.5 + index * .05
        monkeypatch.setattr("car_control_modular.controllers.time.monotonic", lambda: now)
        measured = replace(
            frame, distance_m=2.5, capture_timestamp=now, capture_frame_id=270+index,
            distance_state=DistanceState(
                source="vision_depth", source_detail="depth_multiregion",
                sample_timestamp=now, sample_age_sec=0., raw_distance_m=2.5,
                filtered_distance_m=2.5, used_distance_m=2.5, fusion_confidence=1.,
            ),
            steering_feedback=replace(frame.steering_feedback, timestamp=now),
        )
        controller.decide(index, measured, longitudinal_only=True)
    assert controller.last_distance_pid_result is not None
    assert controller.last_distance_pid_result.output_rpm > 0
    assert controller._distance_pid_last_sample_timestamp == now
    controller._live_longitudinal_authority_reader = lambda uid: ("forward", 30, uid, now)
    before = _longitudinal_state(controller)
    monkeypatch.setattr("car_control_modular.controllers.time.monotonic", lambda: 100.0)
    decision = controller.decide(287, frame, lateral_only=True)
    assert decision.lateral_only and decision.actions[0].kind == "steer_right"
    assert _longitudinal_state(controller) == before


@pytest.mark.parametrize("event", ["reverse", "target_stop", "pid_disabled", "waiting_loss", "stale_recovery"])
def test_preflight_excludes_modes_that_need_synchronous_full_control(lateral, event):
    controller, frame = lateral
    assert controller.can_decide_lateral_only(frame)
    if event == "reverse":
        controller._reverse_active = True
    elif event == "target_stop":
        controller._target_stop_latched = True
    elif event == "pid_disabled":
        controller.cfg = replace(controller.cfg, visible_steering_pid_enable=False)
    elif event == "waiting_loss":
        controller._lost_started_at = 99.9
    elif event == "stale_recovery":
        controller._stale_direction_recovery_active = True
    before = pickle.dumps({k: v for k, v in vars(controller).items() if not callable(v)})
    assert not controller.can_decide_lateral_only(frame)
    assert pickle.dumps({k: v for k, v in vars(controller).items() if not callable(v)}) == before


@pytest.mark.parametrize("older_axis", ["both", "capture_id", "timestamp"])
def test_older_inflight_depth_uses_own_roi_without_rolling_back_visual_target(lateral, monkeypatch, older_axis):
    controller, visual = lateral
    controller.decide(50, visual, lateral_only=True)
    latest_target = controller.last_selected_target
    lateral_before = pickle.dumps((
        controller._visual_steering_pid, controller._target_direction_history,
        controller.last_person_center_x, controller.last_steering_pid_result,
    ))
    roi = replace(visual.persons[0], bbox=(220., 50., 380., 460.))
    depth = replace(
        visual, persons=[roi], distance_m=2.5,
        capture_frame_id=280 if older_axis != "timestamp" else visual.capture_frame_id,
        capture_timestamp=99.90 if older_axis != "capture_id" else visual.capture_timestamp,
        distance_state=DistanceState(
            source="vision_depth", source_detail="depth_multiregion",
            sample_timestamp=99.99, sample_age_sec=.01,
            raw_distance_m=2.5, filtered_distance_m=2.5, used_distance_m=2.5,
            fusion_confidence=1.,
        ),
    )
    selected = []
    original = controller._longitudinal_only_decision
    def capture_target(frame_index, frame, target, **kwargs):
        selected.append(target)
        return original(frame_index, frame, target, **kwargs)
    monkeypatch.setattr(controller, "_longitudinal_only_decision", capture_target)
    decision = controller.decide(51, depth, longitudinal_only=True)
    assert not decision.lateral_only
    assert selected == [roi]  # Fresh measurement remains bound to its measured old ROI.
    assert controller.last_selected_target is latest_target
    assert controller._distance_pid_last_sample_timestamp == 99.99
    assert pickle.dumps((
        controller._visual_steering_pid, controller._target_direction_history,
        controller.last_person_center_x, controller.last_steering_pid_result,
    )) == lateral_before


def test_inflight_depth_cannot_resurrect_target_after_newer_missing_visual(lateral):
    controller, frame = lateral
    controller.decide(50, replace(frame, persons=[]), lateral_only=True)
    assert controller.last_selected_target is None
    assert controller.active_target_id == 1
    controller.decide(51, replace(frame, capture_frame_id=280, capture_timestamp=99.90),
                      longitudinal_only=True)
    assert controller.last_selected_target is None


@pytest.mark.parametrize("legacy", [False, True])
def test_same_capture_and_legacy_depth_selection_contract_is_unchanged(lateral, legacy):
    controller, frame = lateral
    controller.decide(50, frame, lateral_only=True)
    roi = replace(frame.persons[0], bbox=(220., 50., 380., 460.))
    depth = replace(frame, persons=[roi])
    if legacy:
        depth = replace(depth, capture_frame_id=0, capture_timestamp=0.)
    controller.decide(51, depth, longitudinal_only=True)
    assert controller.last_selected_target is roi
