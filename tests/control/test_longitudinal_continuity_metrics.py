"""Regression for CAP938–1088: no hardware and no relaxed motor deadlines."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from test_distance_tracking_response import setup, decide
from test_lateral_zero_runtime import owner
from test_longitudinal_authority_runtime import _commit
from test_depth_target_snapshot import NOW


def seed(setup, distance=1.9):
    clock, controller, frame = setup
    decide(controller, frame(distance, rpm=30))
    clock.now += .1
    decide(controller, frame(distance, rpm=30))
    assert controller._longitudinal_motion_evidence.eligible
    return clock, controller, frame


def test_short_turn_decays_prior_base_without_accelerating_or_renewing_origin(setup):
    clock, controller, frame = seed(setup)
    origin = controller._longitudinal_bridge.origin.sample_timestamp
    previous = controller.last_distance_pid_result.output_rpm
    values = []
    for age in (.03, .07, .12, .17):
        clock.now = origin + age
        decide(controller, frame(1.93, rpm=30, yaw=10))
        result = controller.last_distance_pid_result
        assert result.output_rpm <= previous
        assert controller._longitudinal_bridge.origin.sample_timestamp == origin
        assert controller._longitudinal_motion_evidence.status == "transient_bridge"
        values.append(result.tracking_base_rpm)
        previous = result.output_rpm
    assert values == sorted(values, reverse=True)
    clock.now = origin + .181
    decide(controller, frame(1.93, rpm=30, yaw=10))
    assert controller._tracking_base_rpm(1.93, clock.now) is None


def test_bridge_uses_final_approved_rpm_not_unlimited_pid_request(setup):
    clock, controller, frame = seed(setup)
    controller.accept_longitudinal_limit(clock.now, 25)
    clock.now += .04
    decide(controller, frame(2.0, rpm=30, yaw=12))
    assert controller.last_distance_pid_result.output_rpm <= 25


def test_replay_during_short_turn_cannot_authorize_but_preserves_next_sample_prior(setup):
    clock, controller, frame = seed(setup)
    origin = controller._longitudinal_bridge.origin
    clock.now += .03
    decide(controller, frame(1.9, rpm=30, yaw=10))
    stamp = clock.now
    clock.now += .01
    current = frame(1.9, rpm=30, yaw=10)
    current = replace(current, distance_state=replace(current.distance_state,
        raw_distance_m=None, sample_timestamp=None, observation_timestamp=stamp,
        temporal_status="duplicate", source_detail="depth_sample_observation_discarded"))
    controller._observe_longitudinal_motion(current, current.persons[0])
    assert controller._tracking_base_rpm(1.9, clock.now) is None
    assert controller._longitudinal_bridge.origin is origin
    clock.now += .03
    controller._observe_longitudinal_motion(frame(1.9, rpm=30, yaw=10), current.persons[0])
    assert controller._longitudinal_motion_evidence.status == "transient_bridge"


@pytest.mark.parametrize("change", ["expired_depth", "future_depth", "encoder_stale", "encoder_untrusted",
                                    "closing", "jump", "too_close", "yaw", "bearing", "hazard", "search"])
def test_bridge_never_bypasses_invalid_or_contradictory_evidence(setup, change):
    clock, controller, frame = seed(setup)
    clock.now += .04
    current = frame(1.9, rpm=30, yaw=10)
    if change in {"closing", "jump", "too_close", "yaw"}:
        current = frame({"closing": 1.85, "jump": 2.3, "too_close": 1.4}.get(change, 1.9),
                        rpm=30, yaw=20 if change == "yaw" else 10)
    elif change in {"expired_depth", "future_depth"}:
        current = frame(1.9, rpm=30, yaw=10, stamp=clock.now + (.02 if change == "future_depth" else -.2))
    elif change.startswith("encoder"):
        current = replace(current, steering_feedback=replace(current.steering_feedback,
            timestamp=clock.now-.2 if change == "encoder_stale" else clock.now,
            trustworthy=change != "encoder_untrusted"))
    elif change == "bearing":
        current = replace(current, persons=[replace(current.persons[0], bbox=(450, 50, 630, 460))])
    elif change == "hazard":
        current = replace(current, hazard=replace(current.hazard, active=True))
    elif change == "search":
        controller.search_state = "searching"
    controller._observe_longitudinal_motion(current, current.persons[0])
    assert controller._tracking_base_rpm(current.distance_m, clock.now) is None


def test_bridge_cannot_start_after_zero_pid(setup):
    clock, controller, frame = seed(setup)
    controller.accept_longitudinal_limit(clock.now, 0)
    clock.now += .03
    decide(controller, frame(1.9, rpm=30, yaw=10))
    assert controller._tracking_base_rpm(1.9, clock.now) is None


def test_turn_exit_rewarms_estimator_then_replaces_bridge_with_new_evidence(setup):
    clock, controller, frame = seed(setup)
    clock.now += .03
    decide(controller, frame(1.9, rpm=30, yaw=10))
    clock.now += .03
    decide(controller, frame(1.9, rpm=30))
    assert controller._longitudinal_motion_evidence.status == "transient_bridge"
    clock.now += .03
    decide(controller, frame(1.9, rpm=30))
    assert controller._longitudinal_motion_evidence.status == "ready"
    assert controller._longitudinal_bridge_output_cap is None


@pytest.mark.parametrize("previous", [None, ("forward", 19, 1, NOW-.04),
                                      ("forward", 40, 1, NOW-.19), ("backward", 30, 1, NOW-.04)])
def test_runtime_bridge_cannot_revive_expired_reversed_or_revoked_authority(owner, previous):
    owner._depth30_linear_snapshot = previous
    owner._follow_controller._longitudinal_motion_evidence = SimpleNamespace(status="transient_bridge")
    owner._follow_controller._longitudinal_bridge = SimpleNamespace(origin=SimpleNamespace(sample_timestamp=NOW-.04))
    actions, _accepted = _commit(owner, stamp=NOW-.02, percent=50)
    expected = 19 if previous and previous[0] == "forward" and previous[1] == 19 else 0
    assert actions[0].speed_percent == expected


def test_runtime_bridge_expires_extra_speed_not_fresh_depth(owner, monkeypatch):
    import request_0513_modular as runtime
    owner._depth30_linear_snapshot = ("forward", 50, 1, NOW-.04)
    owner._follow_controller._longitudinal_motion_evidence = SimpleNamespace(status="transient_bridge")
    owner._follow_controller._longitudinal_bridge = SimpleNamespace(origin=SimpleNamespace(sample_timestamp=NOW-.10))
    owner._follow_controller.distance_only_forward_percent = lambda frame, stamp: 25
    actions, accepted = _commit(owner, stamp=NOW-.02, percent=45)
    assert accepted and actions[0].speed_percent == 45
    assert owner._depth30_linear_snapshot[3] == NOW-.02
    # Fresh depth does not carry old FF, but can retain the bounded PID part.
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.081)
    assert owner._fresh_depth_linear_snapshot(1)[1] == 25
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.161)
    assert owner._fresh_depth_linear_snapshot(1) is None


def test_short_ff_budget_is_not_mistaken_for_short_depth_budget(owner):
    owner._depth30_linear_snapshot = ("forward", 50, 1, NOW-.04)
    owner._follow_controller._longitudinal_motion_evidence = SimpleNamespace(status="transient_bridge")
    owner._follow_controller._longitudinal_bridge = SimpleNamespace(origin=SimpleNamespace(sample_timestamp=NOW-.16))
    actions, accepted = _commit(owner, stamp=NOW-.02, percent=40)
    assert accepted and actions[0].speed_percent == 40
    assert owner._depth30_linear_snapshot[3] == NOW-.02
    assert owner._depth30_linear_timing.feedforward_timestamp == NOW-.16


def gap_frame(frame):
    current = frame(None)
    return replace(current, distance_state=replace(current.distance_state,
        source_detail="depth_detector_bbox_stale", sample_timestamp=None))


def recovery_seed(setup):
    clock, controller, frame = setup
    controller.cfg = replace(controller.cfg, depth_recovery_stage1_sec=.2, depth_recovery_stage2_sec=.4)
    assert controller._limit_depth_quality_forward_percent(frame(2.1), 55, clock.now) == 25
    return clock, controller, frame


def test_short_scheduling_gap_resumes_original_ramp_not_another_200ms(setup, caplog):
    clock, controller, frame = recovery_seed(setup)
    origin = clock.now
    clock.now += .1
    controller._limit_depth_quality_forward_percent(frame(2.1), 55, clock.now)
    clock.now += .04
    controller._note_depth_quality_failure(gap_frame(frame), clock.now)
    clock.now += .08
    assert controller._limit_depth_quality_forward_percent(frame(2.12), 55, clock.now) == 45
    assert controller._depth_recovery_started_at == origin
    assert "action=resume" in caplog.text


@pytest.mark.parametrize("change", ["expired", "jump", "uid", "invalid_pixels", "hazard"])
def test_real_loss_or_bad_depth_still_restarts_recovery(setup, change):
    clock, controller, frame = recovery_seed(setup)
    clock.now += .5
    controller._limit_depth_quality_forward_percent(frame(2.1), 55, clock.now)
    clock.now += .03
    gap = gap_frame(frame)
    if change == "invalid_pixels":
        gap = replace(gap, distance_state=replace(gap.distance_state, source_detail="depth_invalid_pixels"))
    if change == "hazard":
        gap = replace(gap, hazard=replace(gap.hazard, active=True))
    controller._note_depth_quality_failure(gap, clock.now)
    clock.now += .2 if change == "expired" else .03
    if change == "uid":
        controller.active_target_id = 2
    output = controller._limit_depth_quality_forward_percent(frame(2.6 if change == "jump" else 2.1), 55, clock.now)
    assert output == 25
    assert controller._depth_recovery_started_at == clock.now


def test_gap_does_not_remove_existing_zero_longitudinal_decision(setup):
    clock, controller, frame = recovery_seed(setup)
    clock.now += .04
    decision = decide(controller, gap_frame(frame))
    assert not any(a.kind == "forward" and a.speed_percent > 0 for a in decision.actions)
