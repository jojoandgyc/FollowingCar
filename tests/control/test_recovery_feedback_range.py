"""CAP188: recovery shares the estimator's measured-wheel validity range."""
from dataclasses import replace

import pytest

from car_control_modular.controllers import FollowSafetyController
from test_distance_tracking_response import setup
from test_scheduling_gap_evidence import missing


def recovery_case(setup, *, approach=True, maximum=200):
    clock, old, frame = setup
    controller = FollowSafetyController(replace(
        old.cfg, forward_max_rpm=maximum, distance_approach_enable=approach,
        distance_matching_base_max_rpm=80,
        distance_feedforward_wheel_circumference_m=.816814,
        depth_measured_recovery_enable=True,
        depth_recovery_stage1_rpm=24, depth_recovery_stage1_sec=.2,
        depth_recovery_stage2_sec=.4, distance_pid_output_rise_rpm_per_sec=240,
    ))
    controller.active_target_id = 1
    controller._has_seen_person = True
    controller._depth_quality_degraded = False
    controller._depth_recovery_anchor = (1, 100., 2.60)
    controller._depth_last_approved_forward_rpm = 84.
    clock.now = 100.10
    controller._note_depth_quality_failure(missing(frame), clock.now)
    clock.now = 100.23
    current = frame(2.539, rpm=99., stamp=100.17)
    current = replace(current, steering_feedback=replace(
        current.steering_feedback, right_forward_rpm=103.,
    ))
    return clock, controller, current


def check_path(controller, frame, now, path):
    hint = controller._depth_gap_resume_hint
    if path == "far_closing":
        return controller._far_closing_recovery_continuous(frame, hint, now)
    if path == "scheduled":
        return controller._scheduling_recovery_cap(frame, 42, now, hint)
    return controller._limit_depth_quality_forward_percent(frame, 42, now)


@pytest.mark.parametrize("path", ["far_closing", "scheduled", "continuity"])
@pytest.mark.parametrize("wheels", ["cap188", "limit"])
def test_legal_feedback_above_100_preserves_recovery(setup, path, wheels):
    clock, controller, current = recovery_case(setup)
    limit = controller._longitudinal_feedforward.config.max_abs_ego_rpm
    assert 129. < limit < 130.
    if wheels == "limit":
        current = replace(current, steering_feedback=replace(
            current.steering_feedback, left_forward_rpm=limit, right_forward_rpm=limit,
        ))
    result = check_path(controller, current, clock.now, path)
    assert result is True if path == "far_closing" else result == 42
    if path == "continuity":
        assert controller._depth_recovery_started_at is None
        assert controller._depth_schedule_recovery is None
        assert controller._depth_last_approved_forward_rpm == 84.
    assert controller.cfg.forward_max_rpm == 200
    assert controller.cfg.distance_matching_base_max_rpm == 80


@pytest.mark.parametrize("path", ["far_closing", "scheduled", "continuity"])
@pytest.mark.parametrize("invalid", [
    "over_limit", "nan", "infinity", "reverse", "feedback_expired",
    "feedback_future", "feedback_untrusted", "depth_expired", "depth_future",
])
def test_invalid_feedback_or_time_cannot_restore_high_speed(setup, path, invalid):
    clock, controller, current = recovery_case(setup)
    feedback = current.steering_feedback
    if invalid in {"over_limit", "nan", "infinity", "reverse"}:
        value = {
            "over_limit": controller._longitudinal_feedforward.config.max_abs_ego_rpm + .01,
            "nan": float("nan"), "infinity": float("inf"), "reverse": -.01,
        }[invalid]
        current = replace(current, steering_feedback=replace(feedback, right_forward_rpm=value))
    elif invalid in {"feedback_expired", "feedback_future"}:
        stamp = clock.now + (.01 if invalid == "feedback_future" else -.151)
        current = replace(current, steering_feedback=replace(feedback, timestamp=stamp))
    elif invalid == "feedback_untrusted":
        current = replace(current, steering_feedback=replace(feedback, trustworthy=False))
    else:
        stamp = clock.now + (.01 if invalid == "depth_future" else -.181)
        current = replace(current, distance_state=replace(
            current.distance_state, sample_timestamp=stamp,
        ))
    result = check_path(controller, current, clock.now, path)
    if path == "far_closing":
        assert result is False
    elif path == "scheduled":
        assert result is None
    else:
        assert result <= 12  # Launch cap remains 24 RPM on the 200 RPM scale.
        assert controller._depth_recovery_resume_base_rpm == 0.
    assert controller._depth_schedule_recovery is None


@pytest.mark.parametrize("path", ["scheduled", "continuity"])
@pytest.mark.parametrize("invalid", ["hazard", "obstacle", "uid", "search", "brake"])
def test_legal_fast_feedback_does_not_bypass_safety(setup, path, invalid):
    clock, controller, current = recovery_case(setup)
    if invalid == "hazard":
        current = replace(current, hazard=replace(current.hazard, active=True))
    elif invalid == "obstacle":
        current = replace(current, obstacles=replace(current.obstacles, front=True))
    elif invalid == "uid":
        controller.active_target_id = 2
    elif invalid == "search":
        controller.search_state = "searching"
    else:
        current = replace(current, distance_state=replace(current.distance_state, brake_latched=True))
    result = check_path(controller, current, clock.now, path)
    assert result is None if path == "scheduled" else result <= 12
    assert controller._depth_schedule_recovery is None


@pytest.mark.parametrize("approach,maximum,limit", [(False, 100, 100.), (False, 200, 105.)])
@pytest.mark.parametrize("path", ["far_closing", "scheduled", "continuity"])
def test_legacy_feedback_bounds_follow_existing_estimator(setup, approach, maximum, limit, path):
    clock, controller, current = recovery_case(setup, approach=approach, maximum=maximum)
    assert controller._longitudinal_feedforward.config.max_abs_ego_rpm == limit
    current = replace(current, steering_feedback=replace(
        current.steering_feedback, left_forward_rpm=limit, right_forward_rpm=limit,
    ))
    result = check_path(controller, current, clock.now, path)
    assert result is True if path == "far_closing" else result == 42
    assert controller.cfg.forward_max_rpm == maximum


def test_high_valid_feedback_keeps_requested_speed_and_ramp_bound(setup):
    clock, controller, current = recovery_case(setup)
    hint = controller._depth_gap_resume_hint
    approved = controller._scheduling_recovery_cap(current, 100, clock.now, hint)
    # Prior approval 84 RPM plus the existing 240 RPM/s * 50 ms step.
    assert approved == 48
    assert controller._depth_schedule_recovery[4] == 96.
    assert approved < 100  # Feedback validity did not change command limits.
