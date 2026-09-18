"""Controller integration for the opt-in distance PI path; no hardware."""
from dataclasses import replace

import pytest

from car_control_modular.controllers import FollowSafetyController
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


def configured(setup, **changes):
    clock, legacy, frame = setup
    cfg = replace(
        legacy.cfg,
        distance_control_mode="distance_pi",
        distance_pi_kp_per_sec=1.0,
        distance_pi_ki_per_sec2=0.4,
        distance_pi_integral_max_m_s=0.8,
        distance_pi_memory_sec=0.35,
        distance_approach_enable=True,
        distance_matching_base_max_rpm=80.0,
        distance_feedforward_wheel_circumference_m=0.816814,
        distance_pid_output_rise_rpm_per_sec=240.0,
        forward_max_rpm=200,
        distance_parking_enable=False,
        depth_measured_recovery_enable=True,
        depth_recovery_stage1_sec=0.2,
        depth_recovery_stage2_sec=0.4,
    )
    controller = FollowSafetyController(replace(cfg, **changes))
    controller.active_target_id = 1
    controller._has_seen_person = True
    return clock, controller, frame


def step(controller, observation, *, visual=False):
    return controller.decide(10, observation, longitudinal_only=not visual)


def integral(controller):
    return controller._distance_pid._distance_pi.integral_m_s


def accumulated(setup, **changes):
    clock, controller, frame = configured(setup, **changes)
    for _ in range(9):
        step(controller, frame(1.8, rpm=20.0))
        clock.now += 0.05
    assert integral(controller) > 0.0
    return clock, controller, frame


def missing(frame):
    current = frame(None)
    return replace(current, distance_state=replace(
        current.distance_state, sample_timestamp=None,
        source_detail="depth_detector_bbox_stale",
    ))


def test_pi_control_is_identical_with_human_speed_diagnostics_enabled_or_disabled(setup):
    clock, enabled, frame = configured(setup)
    disabled = FollowSafetyController(replace(enabled.cfg, distance_feedforward_enable=False))
    disabled.active_target_id = 1
    disabled._has_seen_person = True
    outputs = []
    for _ in range(12):
        current = frame(1.8, rpm=30.0)
        left, right = step(enabled, current), step(disabled, current)
        assert left.current_forward_percent == right.current_forward_percent
        assert enabled.last_distance_pid_result.tracking_base_rpm == 0.0
        assert integral(enabled) == pytest.approx(integral(disabled))
        outputs.append(left.current_forward_percent)
        clock.now += 0.05
    assert max(outputs) > 0
    assert enabled._longitudinal_motion_evidence is not None


@pytest.mark.parametrize("visual", [False, True])
def test_near_target_without_feedforward_does_not_erase_pi_integral(setup, visual):
    clock, controller, frame = accumulated(
        setup, distance_feedforward_enable=False,
        visible_steering_pid_enable=visual,
    )
    # The target still moves forward: close only 0.05 m/s while wheels report
    # 0.27 m/s. A fast approach legitimately releases I through the brake.
    for millimetres in range(1795, 1519, -5):
        step(controller, frame(millimetres / 1000.0, rpm=20.0), visual=visual)
        clock.now += 0.10
    assert integral(controller) > 0.0
    assert controller.last_distance_pid_result.approach_mode == "distance_pi"


def test_duplicate_physical_sample_cannot_integrate_twice(setup):
    clock, controller, frame = accumulated(setup)
    current = frame(1.8, rpm=20.0)
    step(controller, current)
    before = integral(controller)
    original_stamp = controller._distance_pid_last_sample_timestamp
    for _ in range(3):
        clock.now += 0.025
        step(controller, current)
        assert integral(controller) == pytest.approx(before)
        assert controller._distance_pid_last_sample_timestamp == original_stamp


def test_visual_and_depth_loops_share_one_pi_update_for_physical_sample(setup):
    clock, controller, frame = accumulated(setup, visible_steering_pid_enable=True)
    current = frame(1.8, rpm=20.0)
    step(controller, current, visual=True)
    before = integral(controller)
    clock.now += 0.01
    step(controller, current)
    assert integral(controller) == pytest.approx(before)
    assert controller._distance_pid_last_sample_timestamp == current.distance_state.sample_timestamp


def test_actual_decisions_and_two_rpm_approvals_allow_integral_to_grow(setup):
    clock, controller, frame = configured(setup)
    outputs = []
    for _ in range(40):
        decision = step(controller, frame(1.8, rpm=20.0))
        # Exercise the real runtime approval feedback at its 200 RPM scale.
        approved = decision.current_forward_percent * 2.0
        controller.accept_longitudinal_limit(clock.now, approved)
        outputs.append(approved)
        clock.now += 0.05
    # Positive distance error must cross the first quantization step. Otherwise
    # every request of 21 RPM is clipped to 20 and rolls back the same I update.
    assert integral(controller) > 0.15
    assert outputs[-1] > 30.0


def test_percent_quantization_cannot_round_above_pi_braking_envelope(setup):
    clock, controller, frame = configured(setup)
    for distance in (1.8, 1.82, 1.79, 1.77, 1.73, 1.70):
        decision = step(controller, frame(distance, rpm=20.0))
        result = controller.last_distance_pid_result
        approved = decision.current_forward_percent * 2.0
        assert approved <= result.output_rpm
        assert approved <= result.approach_cap_rpm
        clock.now += 0.05


@pytest.mark.parametrize("visual", [False, True])
def test_same_target_short_no_observation_pauses_integral(setup, visual):
    clock, controller, frame = accumulated(setup, visible_steering_pid_enable=visual)
    before = integral(controller)
    physical_stamp = controller._distance_pid_last_sample_timestamp
    for _ in range(3):
        step(controller, missing(frame), visual=visual)
        assert integral(controller) == pytest.approx(before)
        assert controller._distance_pid_last_sample_timestamp in (None, physical_stamp)
        clock.now += 0.04
    step(controller, frame(1.8, rpm=20.0))
    assert integral(controller) >= before


def test_expired_depth_can_retain_short_pi_memory_without_forward_command(setup):
    clock, controller, frame = accumulated(setup)
    before = integral(controller)
    original_stamp = controller._distance_pid_last_sample_timestamp
    clock.now = original_stamp + 0.22
    decision = step(controller, missing(frame))
    assert integral(controller) == pytest.approx(before)
    assert not any(a.kind == "forward" and a.speed_percent > 0 for a in decision.actions)
    clock.now = original_stamp + 0.26
    step(controller, frame(1.8, rpm=20.0))
    assert integral(controller) == pytest.approx(before)  # Never integrate the observation gap.


def test_old_integral_expires_after_memory_window(setup):
    clock, controller, frame = accumulated(setup)
    stamp = controller._distance_pid_last_sample_timestamp
    clock.now = stamp + 0.36
    step(controller, missing(frame))
    assert integral(controller) == 0.0


def test_early_authority_revocation_keeps_memory_but_resumes_from_measured_wheels(setup):
    clock, controller, frame = accumulated(setup)
    before = integral(controller)
    physical_stamp = controller._distance_pid_last_sample_timestamp
    previous_output = controller.last_distance_pid_result.output_rpm
    controller.suspend_longitudinal_authority(clock.now, "test_revoked_grant")
    assert integral(controller) == pytest.approx(before)
    assert controller._distance_pid_last_sample_timestamp == physical_stamp
    clock.now += .025
    resumed = step(controller, frame(1.8, rpm=5.0))
    result = controller.last_distance_pid_result
    assert result.pi_status == "recovering"
    assert result.pi_sample_dt_sec == 0.0
    assert integral(controller) == pytest.approx(before)
    assert result.output_rpm <= 5.0 < previous_output
    assert resumed.current_forward_percent * 2 <= 5.0


@pytest.mark.parametrize("cause", ["hazard", "obstacle", "identity", "lost", "search", "clear"])
def test_unsafe_or_different_target_does_not_inherit_pi_integral(setup, cause):
    clock, controller, frame = accumulated(setup)
    current = frame(1.8, rpm=20.0)
    if cause == "hazard":
        current = replace(current, hazard=replace(current.hazard, active=True, reason="test"))
    elif cause == "obstacle":
        current = replace(current, obstacles=replace(current.obstacles, front=True))
    elif cause == "identity":
        controller.active_target_id = 2
        current = frame(1.8, rpm=20.0, uid=2)
    elif cause == "lost":
        current = replace(current, persons=[])
    elif cause == "search":
        controller.search_state = "searching"
    else:
        controller.clear_active_target("test")
        assert integral(controller) == 0.0
        return
    step(controller, current)
    assert integral(controller) == 0.0


def test_recovery_state_cannot_reapply_legacy_25_45_caps_to_pi(setup):
    clock, controller, frame = configured(setup)
    current = frame(2.5, rpm=60.0)
    step(controller, current)
    controller._depth_quality_degraded = True
    controller._depth_recovery_started_at = clock.now
    controller._depth_recovery_pending_gap = True
    controller._depth_recovery_anchor = (1, clock.now, 2.5)
    controller._depth_gap_resume_hint = (1, clock.now, 2.5, 60.0, clock.now, False)
    clock.now += 0.05
    decision = step(controller, frame(2.5, rpm=60.0))
    requested = controller.last_distance_pid_result.output_rpm
    assert requested > 45
    assert abs(decision.current_forward_percent * 2.0 - requested) <= 1.0


def test_fresh_high_wheel_speed_ignores_all_stale_recovery_flags(setup):
    clock, controller, frame = configured(setup)
    for _ in range(3):
        step(controller, frame(2.7, rpm=105.0))
        clock.now += 0.05
    old_stamp = controller._distance_pid_last_sample_timestamp
    controller._depth_quality_degraded = True
    controller._depth_recovery_started_at = clock.now
    controller._depth_recovery_pending_gap = True
    controller._depth_recovery_anchor = (1, old_stamp, 2.7)
    controller._depth_gap_resume_hint = (1, old_stamp, 2.7, 24.0, clock.now, True)
    controller._depth_schedule_recovery = (1, old_stamp, 2.7, 24.0, 24.0, clock.now + 0.4)
    decision = step(controller, frame(2.7, rpm=105.0))
    assert decision.current_forward_percent * 2.0 > 60.0
    assert controller._depth_recovery_started_at is None
    assert controller._depth_schedule_recovery is None
    assert not controller._depth_recovery_pending_gap
    assert controller._depth_gap_resume_hint is None


def test_20260918_cap164_fresh_depth_bypasses_prior_24rpm_recovery_state(setup):
    clock, controller, frame = configured(setup)
    # run_20260918_000408_45885_d1453311: a valid current sample was
    # previously reduced to 24 RPM by the separate recovery state machine.
    old_stamp = 43751.74326172
    sample = 43751.80629367
    previous_raw = 2.3129983022071308
    clock.now = old_stamp + 0.01
    step(controller, frame(previous_raw, rpm=61.5, stamp=old_stamp))
    clock.now = sample + 0.0098
    current = frame(previous_raw, rpm=61.5, stamp=sample, yaw=4.8816)
    current = replace(
        current, capture_frame_id=164,
        distance_state=replace(current.distance_state, raw_distance_m=2.272736488812392),
        steering_feedback=replace(current.steering_feedback, timestamp=sample - 0.0479,
                                  left_forward_rpm=63.0, right_forward_rpm=60.0),
    )
    controller._depth_quality_degraded = True
    controller._depth_recovery_started_at = clock.now
    controller._depth_recovery_pending_gap = True
    controller._depth_recovery_anchor = (1, old_stamp, previous_raw)
    controller._depth_gap_resume_hint = (1, old_stamp, previous_raw, 24.0, clock.now, True)
    controller._depth_schedule_recovery = (1, old_stamp, previous_raw, 24.0, 24.0, clock.now + 0.4)
    decision = step(controller, current)
    assert controller._distance_pid_last_sample_timestamp == sample
    assert controller.last_distance_pid_result.output_rpm > 45
    assert 0 <= controller.last_distance_pid_result.output_rpm - decision.current_forward_percent * 2 < 2
    assert controller._depth_recovery_started_at is None
    assert controller._depth_schedule_recovery is None
    assert controller._depth_gap_resume_hint is None
    assert not controller._depth_recovery_pending_gap


def test_feedforward_eligibility_loss_does_not_reset_pi_integral(setup):
    clock, controller, frame = accumulated(setup)
    before = integral(controller)
    current = frame(1.8, rpm=20.0)
    controller._clear_longitudinal_velocity_evidence(current, "bearing_limit")
    assert integral(controller) == pytest.approx(before)
    step(controller, current)
    assert integral(controller) >= before


@pytest.mark.parametrize("near_rotation", [False, True])
def test_pi_mode_keeps_protected_reverse_and_clears_forward_integral(setup, near_rotation):
    clock, controller, frame = accumulated(
        setup, reverse_enable=True, near_distance_rotate_only_enable=near_rotation,
        reverse_start_distance_m=1.3, reverse_immediate_distance_m=1.3,
    )
    decision = step(controller, frame(1.2, rpm=0.0))
    if near_rotation:
        # Preserve the existing too-close rotation policy, which precedes
        # reverse; the new mode must not introduce reverse in that policy.
        assert decision.reason == "near_distance_rotation_only"
        assert not any(a.kind == "backward" and a.speed_percent > 0 for a in decision.actions)
    else:
        assert any(a.kind == "backward" and a.speed_percent > 0 for a in decision.actions)
    assert integral(controller) == 0.0


@pytest.mark.parametrize("visual", [False, True])
@pytest.mark.parametrize("distance", [1.5, 1.45])
def test_pi_near_band_does_not_trigger_legacy_no_ff_hold_or_reverse(setup, visual, distance):
    clock, controller, frame = configured(
        setup, reverse_enable=True, near_distance_rotate_only_enable=True,
        visible_steering_pid_enable=visual,
        reverse_start_distance_m=1.3, reverse_immediate_distance_m=1.3,
    )
    decision = step(controller, frame(distance, rpm=0.0), visual=visual)
    assert not any(a.kind == "backward" and a.speed_percent > 0 for a in decision.actions)
    assert decision.current_forward_percent == 0
    if distance == 1.5:
        assert decision.reason not in {"near_distance_rotation_only", "longitudinal_near_rotation_hold"}
        assert controller.last_distance_pid_result.approach_mode == "distance_pi"
    else:
        assert decision.reason == "near_distance_rotation_only"


def test_same_physical_stamp_switches_between_forward_pi_and_legacy_reverse(setup):
    clock, controller, frame = configured(setup)
    step(controller, frame(1.8, rpm=20.0))
    reverse = controller._update_distance_pid(1.2, now=clock.now, forward_control=False)
    assert reverse.output_rpm < 0
    assert reverse.approach_mode == "legacy_pid"
    forward = controller._update_distance_pid(1.8, now=clock.now, forward_control=True)
    assert forward.output_rpm >= 0
    assert forward.approach_mode == "distance_pi"


@pytest.mark.parametrize("visual", [False, True])
@pytest.mark.parametrize("distance", [1.2, 2.0])
@pytest.mark.parametrize("hazard", [False, True])
def test_unsteerable_reverse_exception_never_authorizes_forward_or_unsafe_reverse(setup, visual, distance, hazard):
    clock, controller, frame = accumulated(
        setup, reverse_enable=True, near_distance_rotate_only_enable=False,
        reverse_start_distance_m=1.3, reverse_immediate_distance_m=1.3,
    )
    current = frame(distance, rpm=0.0)
    if hazard:
        current = replace(current, hazard=replace(current.hazard, active=True, reason="test"))
    decision = controller.decide(
        10, current, longitudinal_only=not visual, target_steerable=False,
    )
    assert not any(a.kind == "forward" and a.speed_percent > 0 for a in decision.actions)
    assert integral(controller) == 0.0
    if hazard or distance > 1.3:
        assert not any(a.kind == "backward" and a.speed_percent > 0 for a in decision.actions)
    else:
        assert any(a.kind == "backward" and a.speed_percent > 0 for a in decision.actions)


@pytest.mark.parametrize("visual", [False, True])
@pytest.mark.parametrize("target_steerable", [False, True])
def test_consecutive_reverse_samples_and_release_keep_original_reverse_policy(setup, visual, target_steerable):
    clock, controller, frame = configured(
        setup, reverse_enable=True, near_distance_rotate_only_enable=False,
        reverse_start_distance_m=1.3, reverse_immediate_distance_m=1.3,
        reverse_stop_distance_m=1.45,
    )
    for distance in (1.2, 1.24, 1.28):
        decision = controller.decide(
            10, frame(distance, rpm=0.0), longitudinal_only=not visual,
            target_steerable=target_steerable,
        )
        assert any(a.kind == "backward" and a.speed_percent > 0 for a in decision.actions)
        assert controller._reverse_active
        assert integral(controller) == 0
        clock.now += 0.05
    for _ in range(3):
        decision = controller.decide(
            10, frame(1.46, rpm=0.0), longitudinal_only=not visual,
            target_steerable=target_steerable,
        )
        clock.now += 0.05
    assert not controller._reverse_active
    assert not any(a.kind == "backward" and a.speed_percent > 0 for a in decision.actions)
    # A previously unsteerable target becomes steerable again before forward
    # pursuit resumes; reverse protection alone must not authorize forward.
    for _ in range(3):
        decision = controller.decide(
            10, frame(1.8, rpm=0.0), longitudinal_only=not visual,
            target_steerable=True,
        )
        clock.now += 0.05
    assert controller.last_distance_pid_result.approach_mode == "distance_pi"
    assert any(a.kind == "forward" and a.speed_percent > 0 for a in decision.actions)


def test_real_pi_controller_runtime_commit_quantization_and_original_depth_expiry(setup, owner, monkeypatch):
    import request_0513_modular as runtime

    clock, controller, frame = configured(setup)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(runtime, "FORWARD_MAX_RPM", 200)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT", 100)
    monkeypatch.setattr(runtime, "MOTOR_RS485_TARGET_MIN_INTERVAL_SEC", .05)
    owner._follow_controller = controller
    for _ in range(40):
        current = frame(1.8, rpm=20.0)
        decision = step(controller, current)
        actions, accepted = owner._commit_depth_linear_decision(
            decision, current, 1, is_fresh_depth=True,
        )
        assert accepted
        assert actions[0].speed_percent == decision.current_forward_percent
        timing = owner._depth30_linear_timing
        assert timing.feedforward_expires_at is None
        assert timing.depth_expires_at == pytest.approx(clock.now + .18)
        assert controller.distance_only_forward_percent(current, clock.now) == actions[0].speed_percent
        clock.now += 0.05
    assert integral(controller) > 0.15
    assert owner._depth30_linear_snapshot[1] * 2 > 30
    clock.now = current.distance_state.sample_timestamp + .181
    assert owner._fresh_depth_linear_snapshot(1) is None


def test_legacy_mode_remains_default(setup):
    _, controller, _ = setup
    assert controller.cfg.distance_control_mode == "legacy"
    assert controller._distance_pid._distance_pi is None
