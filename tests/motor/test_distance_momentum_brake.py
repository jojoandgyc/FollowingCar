"""Physical-distance momentum STOP lifecycle, real backend and fake I/O only.

The budget is calculated from an actual SampleBrakingAssessment, not a stubbed
reason string. Encoder values exercise state transitions, not a prediction of
how the real vehicle would move after the new STOP command.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.sample_braking import SampleBrakingAssessment
from car_control_modular.control_types import DepthLinearTiming
from test_unified_forward_snapshot import feedback, writer


def braking_writer(monkeypatch):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=0.)

    def sample(distance=2.25, *, base=60., outer=80., yaw=0., stamp=None):
        publish(base, yaw)
        when = clock[0] if stamp is None else stamp
        assessment = SampleBrakingAssessment(
            1, when, when, distance, 80., 80., when, 0., 1.1,
            .8168, .7, .1, 100., outer_allowance_rpm=10.)
        raw = ("forward", base, 1, when)
        timing = replace(owner._depth30_linear_timing,
            snapshot=raw, accepted_depth_timestamp=when,
            depth_expires_at=when+.25, braking_assessment=assessment,
            continuation_speed_bound_m_s=80.*.8168/60.,
            continuation_distance_m=distance)
        owner._depth30_linear_snapshot = raw
        owner._depth30_prepared_timing = owner._depth30_linear_timing = timing
        rt._steering_feedback = feedback(clock[0], outer, outer)
        return assessment

    def limit(raw, timing, now, *, feedback, quiet):
        assert quiet
        budget = timing.braking_assessment.budget(now,
            max(abs(feedback.left_forward_rpm), abs(feedback.right_forward_rpm)),
            authorized_rpm=raw[1],
            execution_bound_rpm=timing.continuation_speed_bound_m_s*60./.8168,
            tightened_distance_m=timing.continuation_distance_m)
        return budget.cap_rpm, budget.reason

    owner._depth_forward_continuation_limit = limit
    assessment = sample()
    assert assessment.budget(clock[0], 80.).cap_rpm > 60.
    clock[0] += .03
    rt._steering_feedback = feedback(clock[0], 80., 80.)
    proof = rt._distance_brake_evidence(clock[0])
    assert proof[0] is assessment and proof[1:3] == (0., "shared_braking_momentum")
    return rt, owner, driver, clock, sample, assessment


def start(monkeypatch):
    rt, owner, driver, clock, sample, assessment = braking_writer(monkeypatch)
    rt._service_follow_wheels()
    assert rt._distance_brake_episode is not None
    assert driver.stops == [1] and driver.pairs == [(60, -60)]
    return rt, owner, driver, clock, sample, assessment


@pytest.mark.parametrize("entrance", ["periodic", "zero_packet"])
def test_real_momentum_budget_writes_stop_not_speed_zero_or_parking_current(monkeypatch, entrance):
    rt, owner, driver, clock, _, assessment = braking_writer(monkeypatch)
    old_timing = owner._depth30_linear_timing
    receipt_sequence = rt.backend._speed_receipt_sequence
    if entrance == "periodic":
        rt._service_follow_wheels()
    else:
        with owner.motor_io_lock:
            assert not rt._follow_motor_call("send_targets", 0, 0, "FOLLOW_SNAPSHOT_REVOKED")
    episode = rt._distance_brake_episode
    assert episode[0] is assessment and episode[1].sent_at == clock[0]
    assert driver.stops == [1] and driver.pairs == [(60, -60)]
    assert driver.register_writes == [] and rt.backend.parking_current_a == 0.
    assert rt.backend.normal_zero_hold and not rt.backend.motion_armed
    assert rt.backend.last_speed_receipt is None
    assert rt.backend._speed_receipt_sequence == receipt_sequence
    assert owner._depth30_linear_timing is old_timing
    assert rt._forward_execution_anchor is None


def test_no_grant_pi_momentum_can_brake_but_never_authorize_resume(monkeypatch):
    rt, owner, driver, clock, _, _ = braking_writer(monkeypatch)
    pi = SampleBrakingAssessment(1, clock[0], clock[0], 2.2, 80., 80.,
        clock[0], 0., 1.1, .8168, .7, .1, 100., outer_allowance_rpm=10.)
    owner._depth30_linear_snapshot = None
    owner._follow_controller.last_distance_pid_result = SimpleNamespace(
        pi_braking_assessment=pi, output_rpm=0, pi_stationary_preview_status="momentum_brake")
    owner._follow_controller._distance_pid_last_sample_timestamp = pi.sample_timestamp
    assert rt._distance_brake_evidence(clock[0])[2:] == ("shared_braking_momentum", None)
    rt._service_follow_wheels()
    assert driver.stops == [1] and driver.pairs == [(60, -60)]
    clock[0] += .03
    safe_pi = replace(pi, sample_timestamp=clock[0], checked_at=clock[0],
                      feedback_timestamp=clock[0], distance_m=3.)
    owner._follow_controller.last_distance_pid_result = SimpleNamespace(
        pi_braking_assessment=safe_pi, output_rpm=32, pi_stationary_preview_status="bounded")
    owner._follow_controller._distance_pid_last_sample_timestamp = safe_pi.sample_timestamp
    rt._steering_feedback = feedback(clock[0], 60., 60.)
    rt._service_follow_wheels()
    assert rt._distance_brake_episode is not None  # Safe PI preview is not an admitted grant.
    assert driver.stops == [1] and driver.pairs == [(60, -60)]


@pytest.mark.parametrize("command", ["periodic", "zero", "yaw_left", "yaw_right", "direct_follow"])
def test_old_commands_do_not_exit_active_braking(monkeypatch, command):
    rt, owner, driver, clock, _, _ = start(monkeypatch)
    episode = rt._distance_brake_episode
    for _ in range(3):
        clock[0] += .02
        rt._steering_feedback = feedback(clock[0], 60., 60.)
        if command == "periodic":
            rt._service_follow_wheels()
        elif command == "zero":
            with owner.motor_io_lock:
                assert not rt._follow_motor_call("send_targets", 0, 0, "old_zero")
        elif command == "direct_follow":
            rt.send_percent_drive(60, allow_below_min=True)
        else:
            rt.send_yaw_only(-4 if command == "yaw_left" else 4)
    assert rt._distance_brake_episode is episode
    assert driver.pairs == [(60, -60)] and driver.stops == [1]
    assert rt.backend.last_speed_receipt is None and driver.register_writes == []


@pytest.mark.parametrize("yaw", [-4., 0., 4.])
def test_new_independently_safe_post_stop_sample_resumes_through_normal_writer(monkeypatch, yaw):
    rt, owner, driver, clock, sample, _ = start(monkeypatch)
    sent_at = rt._distance_brake_episode[1].sent_at
    clock[0] += .03
    new = sample(2.6, outer=60., yaw=yaw)
    proof = rt._distance_brake_evidence(clock[0])
    assert new.sample_timestamp > sent_at and proof[1] > 0 and proof[2] == "shared_braking_cap"
    rt._service_follow_wheels()
    assert rt._distance_brake_episode is None
    assert driver.pairs == [(60, -60), (int(60+yaw), -int(60-yaw))]
    assert driver.stops == [1] and driver.register_writes == []
    assert rt.backend.last_speed_receipt.completed_at == clock[0]
    assert rt._forward_execution_anchor.sample_timestamp == new.sample_timestamp


@pytest.mark.parametrize("fault", ["same_sample", "at_stop", "expired", "feedback",
                                    "identity", "visual", "uid", "not_admitted",
                                    "continuation_veto", "read_veto"])
def test_nonqualifying_new_motion_cannot_release_brake(monkeypatch, fault):
    rt, owner, driver, clock, sample, old = start(monkeypatch)
    episode = rt._distance_brake_episode
    sent_at = episode[1].sent_at
    clock[0] += .03
    stamp = old.sample_timestamp if fault == "same_sample" else sent_at if fault == "at_stop" else None
    sample(2.6, outer=60., stamp=stamp)
    if fault == "expired":
        clock[0] += .251
        rt._steering_feedback = feedback(clock[0], 60., 60.)
    elif fault == "feedback":
        clock[0] += .151
    elif fault == "identity":
        owner._detector_identity_lease = False
    elif fault == "visual":
        owner._validated_visual_observation = False
    elif fault == "uid":
        owner._follow_controller.active_target_id = 2
    elif fault == "not_admitted":
        owner._depth30_linear_snapshot = None
    elif fault == "continuation_veto":
        owner._depth30_continuation_veto = (1, clock[0])
    elif fault == "read_veto":
        owner._depth30_read_veto = (1, clock[0], "visibility_expired")
    rt._service_follow_wheels()
    assert rt._distance_brake_episode is episode
    assert driver.pairs == [(60, -60)] and driver.stops == [1]


def test_quiet_requires_two_distinct_post_stop_encoder_samples_and_no_old_grant_replay(monkeypatch):
    rt, owner, driver, clock, sample, old = start(monkeypatch)
    episode = rt._distance_brake_episode
    clock[0] += .02
    rt._steering_feedback = feedback(clock[0], 0., 0.)
    rt._service_follow_wheels()
    assert episode[1].quiet_count == 1
    clock[0] += .05  # Duplicate report must not count again despite waiting.
    rt._service_follow_wheels()
    assert rt._distance_brake_episode is episode and episode[1].quiet_count == 1
    rt._steering_feedback = feedback(clock[0], 0., 0.)
    rt._service_follow_wheels()
    assert rt._distance_brake_episode is None
    assert driver.pairs == [(60, -60)]  # Old still-in-TTL depth cannot restart after STOP.
    assert owner._depth30_linear_snapshot[3] == old.sample_timestamp
    clock[0] += .051  # Resume on the next ordinary 50ms send opportunity.
    sample(2.6, outer=0.)
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (60, -60) and len(driver.pairs) == 2


@pytest.mark.parametrize("fault", ["one_wheel", "untrusted", "pre_stop", "future", "stale", "error"])
def test_bad_encoder_evidence_cannot_complete_braking(monkeypatch, fault):
    rt, _, driver, clock, _, _ = start(monkeypatch)
    episode = rt._distance_brake_episode
    for _ in range(2):
        clock[0] += .05
        fb = feedback(clock[0], 0., 0.)
        if fault == "one_wheel":
            fb = replace(fb, right_forward_rpm=2.)
        elif fault == "untrusted":
            fb = replace(fb, trustworthy=False)
        elif fault == "pre_stop":
            fb = replace(fb, timestamp=episode[1].sent_at)
        elif fault == "future":
            fb = replace(fb, timestamp=clock[0]+.01)
        elif fault == "error":
            fb = replace(fb, right_error=1)
        else:
            fb = replace(fb, timestamp=clock[0]-.151)
        rt._steering_feedback = fb
        rt._service_follow_wheels()
    assert rt._distance_brake_episode is episode
    assert driver.stops == [1] and driver.pairs == [(60, -60)]


@pytest.mark.parametrize("stop_owner", ["explicit", "brake_hold", "park", "search_brake"])
def test_higher_priority_stop_owner_keeps_its_mode(monkeypatch, stop_owner):
    rt, owner, driver, clock, sample, _ = start(monkeypatch)
    episode = rt._distance_brake_episode
    rt.backend.send_stop("higher_priority_stop", mode="emergency", preserve_zero=True)
    if stop_owner == "explicit":
        owner._explicit_stop_requested = True
    elif stop_owner == "brake_hold":
        owner._brake_hold_active = True
    elif stop_owner == "park":
        owner._near_yaw_park_request = object()
    else:
        rt._search_reacquire_brake_request = object()
    clock[0] += .03
    sample(2.6, outer=0.)
    rt._service_distance_brake()
    assert rt._distance_brake_episode is episode
    assert driver.stops == [1, 1] and driver.pairs == [(60, -60)]
    assert rt.backend.last_speed_receipt is None


def test_hard_stop_is_checked_during_active_episode_before_new_budget_resume(monkeypatch):
    rt, _, driver, clock, sample, _ = start(monkeypatch)
    clock[0] += .03
    sample(2.6, outer=0.)
    rt.hard_stop_check = lambda _: True
    rt._service_follow_wheels()
    assert rt._distance_brake_episode is not None
    assert driver.stops == [1, 1] and driver.pairs == [(60, -60)]


@pytest.mark.parametrize("preserve_zero", [False, True])
def test_new_stop_generation_invalidates_pre_stop_resume_before_state_publication(monkeypatch, preserve_zero):
    rt, _, driver, clock, sample, _ = start(monkeypatch)
    clock[0] += .02
    rt._steering_feedback = feedback(clock[0])
    rt._service_follow_wheels()
    assert rt._distance_brake_episode[1].quiet_count == 1
    clock[0] += .03
    sample(2.6, outer=60.)  # Later than first STOP, but BEFORE the new STOP.
    clock[0] += .01
    rt.backend.send_stop("external_stop_before_owner_state_publish",
                         mode="emergency", preserve_zero=preserve_zero)
    rt._service_follow_wheels()
    assert rt._distance_brake_episode is not None
    assert rt._distance_brake_episode[1].quiet_count == 0
    assert driver.stops == [1, 1] and driver.pairs == [(60, -60)]
    assert rt.backend.last_speed_receipt is None
    floor = rt._distance_brake_sample_floor
    assert floor >= clock[0]
    clock[0] += .02
    rt._steering_feedback = feedback(clock[0], 60., 60.)
    rt._service_follow_wheels()
    assert rt._distance_brake_episode is not None
    assert driver.pairs == [(60, -60)]  # Polling/reusing the old grant cannot resume.
    clock[0] += .03
    new = sample(2.6, outer=60.)
    assert new.sample_timestamp > floor
    rt._service_follow_wheels()
    assert rt._distance_brake_episode is None
    assert driver.pairs == [(60, -60), (60, -60)]


@pytest.mark.parametrize("invalid", ["old_reason", "sample_expired", "feedback_stale", "wrong_uid", "no_budget"])
def test_old_veto_or_invalid_evidence_does_not_start_momentum_mode(monkeypatch, invalid):
    rt, owner, driver, clock, _, _ = braking_writer(monkeypatch)
    if invalid == "old_reason":
        owner._depth30_linear_snapshot = None
        owner._depth30_read_veto = "shared_braking_momentum"
    elif invalid == "sample_expired":
        clock[0] += .251
        rt._steering_feedback = feedback(clock[0], 80., 80.)
    elif invalid == "feedback_stale":
        rt._steering_feedback = feedback(clock[0]-.151, 80., 80.)
    elif invalid == "wrong_uid":
        owner._follow_controller.active_target_id = 2
    else:
        old = owner._depth30_linear_timing
        owner._depth30_prepared_timing = owner._depth30_linear_timing = replace(
            old, continuation_speed_bound_m_s=0.)
    assert not rt._service_distance_brake()
    assert rt._distance_brake_episode is None
    assert driver.stops == [] and driver.pairs == [(60, -60)]


def test_cap119_recorded_budget_stop_and_cap122_fresh_32rpm_recovery(monkeypatch):
    """Recorded measurement clocks/speeds, not counterfactual vehicle dynamics."""
    rt, owner, driver, clock, _, _ = writer(monkeypatch, base=60., yaw=0.)
    rt.config.motor_forward_max_target_rpm = 200
    rt.backend.config = replace(rt.backend.config, max_target=200)

    def adopt(*, sample_ts, checked, distance, bound, outer, feedback_ts, rpm):
        clock[0] = checked
        assessment = SampleBrakingAssessment(1, sample_ts, checked, distance,
            bound, outer, feedback_ts, 0., 1.1, .816814, 1., .15, 200.,
            outer_allowance_rpm=10.)
        raw = ("forward", rpm/2., 1, sample_ts)
        timing = DepthLinearTiming(raw, sample_ts, sample_ts+.25,
            continuation_distance_m=distance,
            continuation_speed_bound_m_s=max(bound, rpm+10.)*.816814/60.,
            braking_assessment=assessment)
        owner._depth30_linear_snapshot = raw
        owner._depth30_prepared_timing = owner._depth30_linear_timing = timing
        owner._lateral_yaw_revision += 1
        rt._steering_feedback = feedback(feedback_ts, outer-1., outer)
        return assessment

    def axes(now):
        raw = owner._depth30_linear_snapshot
        return (1, owner._lateral_yaw_revision,
                raw[1]*2. if 0 <= now-raw[3] <= .25 else 0., 0.)

    def depth(uid, now=None):
        raw = owner._depth30_linear_snapshot
        return raw if raw[2] == uid and 0 <= clock[0]-raw[3] <= .25 else None

    def limit(raw, timing, now, *, feedback, quiet):
        assert quiet
        budget = timing.braking_assessment.budget(now,
            max(feedback.left_forward_rpm, feedback.right_forward_rpm),
            authorized_rpm=raw[1]*2.,
            execution_bound_rpm=timing.continuation_speed_bound_m_s*60./.816814,
            tightened_distance_m=timing.continuation_distance_m)
        return budget.cap_rpm/2., budget.reason

    owner._follow_wheel_axes = axes
    owner._fresh_depth_linear_snapshot = depth
    owner._depth_forward_continuation_limit = limit
    old = adopt(sample_ts=17362.178183768, checked=17362.217912701,
        distance=2.237169562, bound=86., outer=86., feedback_ts=17362.19283614, rpm=76.)
    assert rt._distance_brake_evidence(clock[0])[2] == "shared_braking_cap"
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (76, -76)
    clock[0] = 17362.254205307
    assert rt._distance_brake_evidence(clock[0])[1:3] == (0., "shared_braking_momentum")
    rt._service_follow_wheels()
    assert driver.stops == [1] and driver.pairs == [(60, -60), (76, -76)]
    assert rt.backend.last_speed_receipt is None
    assert rt._distance_brake_episode[0] is old

    new = adopt(sample_ts=17362.397264993, checked=17362.515138782,
        distance=2.05255892, bound=76., outer=61., feedback_ts=17362.484638782, rpm=32.)
    proof = rt._distance_brake_evidence(clock[0])
    assert proof[0] is new and proof[1] >= 32. and proof[2] == "shared_braking_cap"
    rt._service_follow_wheels()
    assert rt._distance_brake_episode is None
    assert driver.stops == [1] and driver.pairs[-1] == (32, -32)
    assert len(driver.pairs) == 3 and all(pair != (0, 0) for pair in driver.pairs)
    assert driver.register_writes == []
    assert rt._forward_execution_anchor.sample_timestamp == new.sample_timestamp


def test_unacknowledged_momentum_stop_latches_fault_without_claiming_an_episode(monkeypatch):
    rt, owner, driver, _, _, _ = braking_writer(monkeypatch)

    def failed_stop(_mode):
        raise OSError("offline STOP acknowledgement missing")

    monkeypatch.setattr(driver, "stop_all", failed_stop)
    with pytest.raises(OSError):
        rt._service_follow_wheels()
    assert rt.backend.motion_write_fault
    assert rt._distance_brake_episode is None
    assert driver.pairs == [(60, -60)] and driver.register_writes == []
    assert rt.backend.last_speed_receipt is None
    assert not owner.motor_io_lock.locked()


@pytest.mark.parametrize("failed_side", ["left", "right", "both"])
def test_partial_dual_stop_failure_still_attempts_both_wheels_and_latches_fault(monkeypatch, failed_side):
    rt, owner, driver, _, _, _ = braking_writer(monkeypatch)
    attempts = []

    def stop(side, mode):
        attempts.append((side, mode))
        if side == failed_side or failed_side == "both":
            raise OSError("offline STOP acknowledgement missing: " + side)

    monkeypatch.setattr(driver, "stop", stop, raising=False)
    with pytest.raises(OSError):
        rt._service_follow_wheels()
    assert attempts == [("right", 1), ("left", 1)]
    assert rt.backend.motion_write_fault and rt.backend.last_speed_receipt is None
    assert rt._distance_brake_episode is None
    assert driver.pairs == [(60, -60)] and driver.register_writes == []
    assert not owner.motor_io_lock.locked()


@pytest.mark.parametrize("scope", ["pivot", "search", "rotation_only"])
def test_nonforward_control_scope_does_not_start_longitudinal_momentum_mode(monkeypatch, scope):
    rt, owner, driver, clock, _, _ = braking_writer(monkeypatch)
    if scope == "pivot":
        # This is genuinely braking evidence at 40 RPM too, not merely a
        # far-distance budget that happens not to ask for STOP.
        assessment = SampleBrakingAssessment(1, clock[0], clock[0], 1.4,
            40., 40., clock[0], 0., 1.1, .8168, .7, .1, 100.)
        owner._follow_controller.last_distance_pid_result = SimpleNamespace(
            pi_braking_assessment=assessment, output_rpm=0,
            pi_stationary_preview_status="momentum_brake")
        owner._follow_controller._distance_pid_last_sample_timestamp = clock[0]
        rt._steering_feedback = feedback(clock[0], 40., -40.)
    elif scope == "search":
        owner.search_state = owner._follow_controller.search_state = "searching"
    else:
        rt.config.rotation_only = True
    assert rt._distance_brake_evidence(clock[0])[2] == "shared_braking_momentum"
    assert not rt._service_distance_brake()
    assert rt._distance_brake_episode is None
    assert driver.stops == [] and driver.pairs == [(60, -60)]


def test_ordinary_safe_follow_adds_no_motor_lock_or_publisher_callback(monkeypatch):
    rt, owner, _, clock, sample, _ = braking_writer(monkeypatch)
    sample(3., outer=60.)

    class UnavailableLock:
        def __enter__(self):
            raise AssertionError("normal motion acquired an extra motor lock")

        def __exit__(self, *_args):
            return False

    def forbidden(*_args, **_kwargs):
        raise AssertionError("brake evidence called a live publisher reader")

    monkeypatch.setattr(owner, "motor_io_lock", UnavailableLock())
    monkeypatch.setattr(owner, "_depth_forward_continuation_limit", forbidden)
    assert rt._distance_brake_evidence(clock[0])[2] == "shared_braking_cap"
    assert not rt._service_distance_brake()


@pytest.mark.parametrize("preserve_zero", [False, True])
def test_stop_between_brake_release_and_next_plan_retains_hardware_ownership(monkeypatch, preserve_zero):
    rt, owner, driver, clock, sample, _ = start(monkeypatch)
    clock[0] += .03
    sample(3., outer=60.)
    original = rt._service_distance_brake
    injected = []

    def release_then_external_stop():
        held = original()
        if not held and not injected:
            injected.append(True)
            clock[0] += .001
            rt.backend.send_stop("new_stop_before_flags", mode="emergency", preserve_zero=preserve_zero)
        return held

    monkeypatch.setattr(rt, "_service_distance_brake", release_then_external_stop)
    rt._service_follow_wheels()
    assert injected == [True] and rt._distance_brake_episode is None
    assert driver.pairs == [(60, -60)] and driver.stops == [1, 1]
    rt._service_follow_wheels()  # Retire the observation preceding the new STOP.
    assert driver.pairs == [(60, -60)]
    assert rt._distance_brake_sample_floor >= clock[0]
    clock[0] += .06
    sample(3., outer=60.)
    rt._service_follow_wheels()
    assert driver.pairs == [(60, -60), (60, -60)]
