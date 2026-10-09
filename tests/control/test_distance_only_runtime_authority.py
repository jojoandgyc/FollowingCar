"""Disabled target motion cannot influence the real runtime's depth grants."""
from dataclasses import replace
from copy import copy
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction, ControlDecision, SteeringFeedback
from car_control_modular.sample_braking import SampleBrakingAssessment
from car_control_modular.controllers import FollowSafetyController
from car_control_modular.longitudinal_approach import RawDepthMotionEvidence
from car_control_modular.mssd_motor import MotorSpeedReceipt
from test_depth_authority_250 import authority, decide_commit, advance, writer
from test_distance_tracking_response import setup
from test_distance_pi_runtime import pi_owner
from test_lateral_zero_runtime import owner, NOW
from test_longitudinal_authority_runtime import _frame


@pytest.fixture
def distance_only(pi_owner, monkeypatch):
    obj = pi_owner
    clock = SimpleNamespace(now=NOW)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE", True)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC", .25)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_CONTROL_SAMPLE_MAX_AGE_SEC", .18)
    monkeypatch.setattr(runtime, "MOTOR_FORWARD_MAX_TARGET_RPM", 200)
    monkeypatch.setattr(runtime, "DISTANCE_APPROACH_DECELERATION_M_S2", .7)
    monkeypatch.setattr(runtime, "DISTANCE_APPROACH_RESPONSE_DELAY_SEC", .2)
    obj._follow_controller.cfg = SimpleNamespace(distance_target_motion_control_enable=False)
    obj._follow_controller._braking_execution_bound_reader = lambda uid, now: 20.
    feedback = [SteeringFeedback(timestamp=NOW, trustworthy=True,
                                 left_forward_rpm=20., right_forward_rpm=20.)]
    obj._action_runtime = SimpleNamespace(
        get_steering_feedback=lambda: feedback[0],
        continuation_executed_speed_bound_rpm=lambda uid, now: 20.,
    )
    # Even a diagnostic motion object must not enter the disabled policy.
    obj._depth_continuation_motion_evidence = lambda *a, **kw: pytest.fail("motion consulted")
    monkeypatch.setattr(runtime, "relative_continuation_speed_cap",
                        lambda *a, **kw: pytest.fail("relative fallback consulted"))
    stamp = NOW-.02
    shared = SampleBrakingAssessment(
        1, stamp, NOW, 3.5, 20., 20., NOW, 0., 1.2, .816814, .7, .2, 200.,
        outer_allowance_rpm=10.)
    frame = replace(_frame(stamp, 3.5), steering_feedback=feedback[0])
    return SimpleNamespace(owner=obj, controller=obj._follow_controller, clock=clock,
                           feedback=feedback, shared=shared, frame=frame, stamp=stamp)


def publish(a, *, shared=..., motion=None, closing=0., fresh=True, percent=40):
    a.controller._distance_pid_last_sample_timestamp = a.stamp
    a.controller._longitudinal_motion_evidence = motion
    a.controller._braking_motion_evidence = motion
    a.controller._braking_range_rate = closing
    a.controller.last_distance_pid_result = SimpleNamespace(
        approach_mode="distance_pi", output_rpm=percent*2., approach_closing_m_s=closing,
        pi_braking_assessment=a.shared if shared is ... else shared,
        pi_motion_window_used=True, pi_brake_source="raw_relative_motion")
    return a.owner._commit_depth_linear_decision(
        ControlDecision(actions=[ControlAction.forward(percent, "distance_pi")], reason="distance_pi"),
        a.frame, 1, is_fresh_depth=fresh)


@pytest.mark.parametrize("target_speed,closing", [(None, 0.), (1000., -1000.), (-1000., 1000.)])
def test_real_admission_and_continuation_ignore_missing_or_extreme_target_estimates(
        distance_only, target_speed, closing):
    a = distance_only
    motion = (None if target_speed is None else SimpleNamespace(
        status="transient_bridge", target_speed_bound_m_s=target_speed,
        range_rate_m_s=-closing, span_sec=.16, sample_count=3))
    actions, accepted = publish(a, motion=motion, closing=closing)
    assert accepted and actions[0].speed_percent == 40
    original = a.owner._depth30_linear_snapshot
    timing = a.owner._depth30_linear_timing
    assert timing.braking_assessment is a.shared
    assert timing.continuation_motion is None
    assert timing.feedforward_timestamp is timing.feedforward_expires_at is None
    assert timing.depth_expires_at == pytest.approx(a.stamp+.25)
    approvals = list(a.owner.approvals)
    values = []
    for age in (.10, .179, .181, .21, .249):
        a.clock.now = a.stamp+age
        a.feedback[0] = replace(a.feedback[0], timestamp=a.clock.now)
        a.owner._last_vision_control_ts = a.clock.now-.01
        current = a.owner._fresh_depth_linear_snapshot(1, quiet=True)
        assert current is not None and current[3] == a.stamp
        values.append(current[1])
    assert values == [40]*5
    assert a.owner.approvals == approvals
    assert a.owner._depth30_linear_snapshot is original
    assert a.owner._depth30_linear_timing is timing
    a.clock.now = a.stamp+.250001
    assert a.owner._fresh_depth_linear_snapshot(1, quiet=True) is None


@pytest.mark.parametrize("fault", ["missing", "uid", "timestamp", "nonzero_target", "untyped"])
def test_missing_or_wrong_shared_assessment_fails_closed_without_target_fallback(distance_only, fault):
    a = distance_only
    shared = {"missing": None, "uid": replace(a.shared, uid=2),
              "timestamp": replace(a.shared, sample_timestamp=a.stamp-.01),
              "nonzero_target": replace(a.shared, target_speed_m_s=-.4),
              "untyped": object()}[fault]
    actions, accepted = publish(a, shared=shared, motion=SimpleNamespace(
        status="transient_bridge", target_speed_bound_m_s=1000.), closing=-1000.)
    assert all(action.speed_percent == 0 for action in actions)
    assert a.owner._depth30_linear_snapshot is None
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    reason = "missing_shared_braking_evidence" if fault == "missing" else "invalid_shared_braking_evidence"
    assert a.owner.rejections == [(a.stamp, reason)]
    assert not a.owner.approvals


@pytest.mark.parametrize("fault", ["missing", "wrong_uid", "nonzero_target"])
def test_final_reader_rejects_corrupt_or_missing_zero_target_assessment(distance_only, fault):
    a = distance_only
    publish(a)
    timing = a.owner._depth30_linear_timing
    assessment = (None if fault == "missing" else
                  replace(a.shared, uid=2) if fault == "wrong_uid" else
                  replace(a.shared, target_speed_m_s=-.4))
    a.owner._depth30_linear_timing = replace(timing, braking_assessment=assessment)
    assert a.owner._depth_forward_continuation_required(
        timing.snapshot, a.owner._depth30_linear_timing, a.clock.now)
    assert a.owner._fresh_depth_linear_snapshot(1, quiet=True) is None
    expected = "missing_shared_braking_evidence" if fault == "missing" else "invalid_shared_braking_evidence"
    assert a.owner._last_quiet_depth_veto[2] == expected
    # Later telemetry and even repaired metadata cannot revive this vetoed
    # same physical sample; only a new qualified Depth grant may do so.
    a.owner._depth30_linear_timing = timing
    a.feedback[0] = replace(a.feedback[0], timestamp=NOW)
    assert a.owner._fresh_depth_linear_snapshot(1, quiet=True) is None


@pytest.mark.parametrize("fault", ["hazard", "obstacle", "feedback_stale", "sample_stale"])
def test_zero_target_shared_assessment_never_bypasses_physical_or_hazard_gates(distance_only, fault):
    a = distance_only
    if fault == "hazard": a.frame = replace(a.frame, hazard=replace(a.frame.hazard, active=True))
    elif fault == "obstacle": a.frame = replace(a.frame, obstacles=replace(a.frame.obstacles, front=True))
    elif fault == "feedback_stale":
        a.feedback[0] = replace(a.feedback[0], timestamp=NOW-.151)
        a.frame = replace(a.frame, steering_feedback=a.feedback[0])
    else: a.clock.now = a.stamp+.181
    actions, accepted = publish(a)
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    assert a.owner._depth30_linear_snapshot is None


def test_same_grant_reduction_and_replay_keep_original_zero_target_assessment(distance_only):
    a = distance_only
    publish(a)
    timing = a.owner._depth30_linear_timing
    actions, accepted = publish(a, fresh=False, percent=25, motion=SimpleNamespace(
        status="transient_bridge", target_speed_bound_m_s=1000.), closing=1000.)
    assert accepted and actions[0].speed_percent == 25
    current = a.owner._depth30_linear_timing
    assert current.braking_assessment is a.shared
    assert current.continuation_motion is None
    assert current.depth_expires_at == timing.depth_expires_at
    assert current.accepted_depth_timestamp == timing.accepted_depth_timestamp


def test_non_pi_distance_adapter_still_ignores_closing_estimate_when_disabled(distance_only):
    a = distance_only
    a.controller.distance_pi_enabled = False
    a.controller._braking_execution_bound_reader = None
    a.controller.last_distance_pid_result = SimpleNamespace(approach_closing_m_s=1000.)
    distance, speed = a.owner._depth_continuation_evidence(a.frame, 1, 40, NOW)
    assert distance == 3.5
    assert speed == pytest.approx(80.*runtime.VISION_MMWAVE_FUSION_ENCODER_WHEEL_CIRCUMFERENCE_M/60.)


def test_missing_execution_reader_never_downgrades_pi_to_unpaired_static_admission(distance_only):
    a = distance_only
    a.controller._braking_execution_bound_reader = None
    assert a.owner._depth_shared_braking_required()
    actions, accepted = publish(a, shared=None, closing=1000.)
    assert not any(action.speed_percent > 0 for action in actions)
    assert a.owner.rejections == [(a.stamp, "missing_shared_braking_evidence")]
    assert a.owner._depth30_linear_snapshot is None


@pytest.mark.parametrize("parameter", ["DISTANCE_APPROACH_DECELERATION_M_S2",
                                      "DISTANCE_APPROACH_RESPONSE_DELAY_SEC"])
def test_shared_grant_uses_its_immutable_model_not_second_legacy_model(
        distance_only, monkeypatch, parameter):
    a = distance_only
    publish(a)
    linear = a.owner._depth30_linear_snapshot
    timing = a.owner._depth30_linear_timing
    monkeypatch.setattr(runtime, parameter, float("nan"))
    # A shared grant carries validated model inputs. It must not read another
    # mutable global stopping model and then silently discard that answer.
    assert a.owner._fresh_depth_linear_snapshot(1) == linear
    # The old API's default retains its own model validation for callers that
    # have no paired shared assessment; this is not a safety bypass for them.
    assert a.owner._depth_forward_continuation_safe(
        linear, timing, a.clock.now, feedback=a.feedback[0], relative_feedback=True
    ) == (False, "invalid_braking_model")


def test_late_closer_distance_tightens_without_replacing_shared_assessment_or_deadline(distance_only):
    a = distance_only
    publish(a)
    original = a.owner._depth30_linear_snapshot
    timing = a.owner._depth30_linear_timing
    a.clock.now = a.stamp+.22
    a.owner._last_vision_control_ts = a.clock.now-.01
    a.feedback[0] = replace(a.feedback[0], timestamp=a.clock.now)
    late_stamp = a.stamp+.01
    closer = replace(_frame(late_stamp, 2.), steering_feedback=a.feedback[0])
    actions, accepted = a.owner._commit_depth_linear_decision(
        ControlDecision(actions=[ControlAction.forward(40, "late_distance")], reason="late_distance"),
        closer, 1, is_fresh_depth=True)
    assert accepted and 0 < actions[0].speed_percent < 40
    current = a.owner._depth30_linear_snapshot
    current_timing = a.owner._depth30_linear_timing
    assert current[3] == original[3] and current[1] < original[1]
    assert current_timing.braking_assessment is a.shared
    assert current_timing.braking_assessment.distance_m == 3.5
    assert current_timing.continuation_distance_m == 2.
    assert current_timing.continuation_motion is None
    assert current_timing.depth_expires_at == timing.depth_expires_at
    assert current_timing.accepted_depth_timestamp == timing.accepted_depth_timestamp
    assert a.owner._depth30_linear_sample_watermark == (1, a.stamp)
    assert a.owner._fresh_depth_linear_snapshot(1)[1] <= current[1]


@pytest.mark.parametrize("feedback_age", [.10, .12, .149])
def test_one_mm_late_closer_keeps_same_encoder_policy(distance_only, monkeypatch, feedback_age):
    a = distance_only
    publish(a)
    original = a.owner._depth30_linear_snapshot
    timing = a.owner._depth30_linear_timing
    a.clock.now = a.stamp+.22
    a.owner._last_vision_control_ts = a.clock.now-.01
    a.feedback[0] = replace(a.feedback[0], timestamp=a.clock.now-feedback_age)
    assert a.owner._fresh_depth_linear_snapshot(1) == original
    monkeypatch.setattr(runtime, "continuation_speed_cap",
                        lambda **kw: pytest.fail("shared grant used old 100 ms policy"))
    closer = replace(_frame(a.stamp+.01, 3.499), steering_feedback=a.feedback[0])
    actions, accepted = a.owner._commit_depth_linear_decision(
        ControlDecision(actions=[ControlAction.forward(40, "late_distance")], reason="late_distance"),
        closer, 1, is_fresh_depth=True)
    assert accepted and actions[0].speed_percent == 40
    current = a.owner._depth30_linear_timing
    assert current.braking_assessment is a.shared
    assert current.continuation_distance_m == 3.499
    assert current.continuation_speed_bound_m_s == timing.continuation_speed_bound_m_s
    assert current.accepted_depth_timestamp == timing.accepted_depth_timestamp
    assert current.depth_expires_at == timing.depth_expires_at
    assert a.owner._fresh_depth_linear_snapshot(1) == original


def _recording_writer(a):
    """Real writer/receipt history; only serial I/O and clocks are replaced."""
    a.controller.search_state = "none"
    a.owner._lateral_yaw_revision = 1
    action, backend = writer(a)
    action.get_steering_feedback = lambda: a.feedback[0]
    backend.last_speed_receipt = None
    original_send = backend.send_targets

    def send(left, right, label, **kwargs):
        original_send(left, right, label, **kwargs)
        backend.last_speed_receipt = MotorSpeedReceipt(
            len(backend.pairs), left, right, a.clock.now)

    backend.send_targets = send
    return action, backend


def test_real_completed_80_then_lower_50_writes_50_not_zero(distance_only):
    a = distance_only
    publish(a)
    timing = a.owner._depth30_linear_timing
    action, backend = _recording_writer(a)
    action._service_follow_wheels()
    assert backend.pairs[-1][:2] == (80, -80)
    assert action.continuation_executed_speed_bound_rpm(1, a.clock.now) == 80.

    actions, accepted = publish(a, fresh=False, percent=25)
    assert accepted and actions[0].speed_percent == 25
    lower = a.owner._depth30_linear_timing
    assert lower.braking_assessment is timing.braking_assessment
    assert lower.continuation_speed_bound_m_s == timing.continuation_speed_bound_m_s
    assert lower.depth_expires_at == timing.depth_expires_at
    assert getattr(a.owner, "_depth30_continuation_veto", None) is None
    a.clock.now += .05
    a.feedback[0] = replace(a.feedback[0], timestamp=a.clock.now,
                            left_forward_rpm=80., right_forward_rpm=80.)
    a.owner._last_vision_control_ts = a.clock.now-.01
    action._service_follow_wheels()
    assert backend.pairs[-1][:2] == (50, -50)
    assert a.owner._fresh_depth_linear_snapshot(1)[1] == 25
    # The old execution budget is a historical cost, never permission to
    # restore the old command on a held/replayed depth observation.
    actions, accepted = publish(a, fresh=False, percent=40)
    assert accepted and actions[0].speed_percent == 25
    assert a.owner._depth30_linear_timing.continuation_speed_bound_m_s == timing.continuation_speed_bound_m_s


def test_late_closer_with_120ms_feedback_preserves_real_motor_packet(distance_only):
    a = distance_only
    publish(a)
    original = a.owner._depth30_linear_timing
    action, backend = _recording_writer(a)
    action._service_follow_wheels()
    assert backend.pairs[-1][:2] == (80, -80)
    a.clock.now = a.stamp+.22
    a.owner._last_vision_control_ts = a.clock.now-.01
    a.feedback[0] = replace(a.feedback[0], timestamp=a.clock.now-.12)
    closer = replace(_frame(a.stamp+.01, 3.499), steering_feedback=a.feedback[0])
    actions, accepted = a.owner._commit_depth_linear_decision(
        ControlDecision(actions=[ControlAction.forward(40, "late_distance")], reason="late_distance"),
        closer, 1, is_fresh_depth=True)
    assert accepted and actions[0].speed_percent == 40
    action._service_follow_wheels()
    assert len(backend.pairs) == 2
    assert all(pair[:2] == (80, -80) for pair in backend.pairs)
    current = a.owner._depth30_linear_timing
    assert current.braking_assessment is original.braking_assessment
    assert current.continuation_speed_bound_m_s == original.continuation_speed_bound_m_s
    assert current.depth_expires_at == original.depth_expires_at


@pytest.mark.parametrize("fault", ["higher_write", "feedback_stale", "expired", "identity", "stop"])
def test_reduced_grant_retains_final_writer_rejection_of_real_faults(distance_only, fault):
    a = distance_only
    publish(a)
    action, backend = _recording_writer(a)
    action._service_follow_wheels()
    assert backend.pairs[-1][:2] == (80, -80)
    publish(a, fresh=False, percent=25)
    a.clock.now += .05
    a.feedback[0] = replace(a.feedback[0], timestamp=a.clock.now)
    a.owner._last_vision_control_ts = a.clock.now-.01
    if fault == "higher_write":
        # Unlike the original legal 80 RPM write, 91 RPM exceeds the
        # immutable 80 + 10 RPM admitted outer-wheel budget.
        action.continuation_executed_speed_bound_rpm = lambda uid, now: 91.
    elif fault == "feedback_stale":
        a.feedback[0] = replace(a.feedback[0], timestamp=a.clock.now-.151)
    elif fault == "expired":
        a.clock.now = a.stamp+.250001
        a.owner._last_vision_control_ts = a.clock.now-.01
        a.feedback[0] = replace(a.feedback[0], timestamp=a.clock.now)
    elif fault == "identity":
        a.controller.active_target_id = 2
    else:
        a.owner._explicit_stop_requested = True
    action._service_follow_wheels()
    assert backend.pairs[-1][:2] == (0, 0)


@pytest.mark.parametrize("diagnostics", ["warming", "receding", "approaching", "nan"])
def test_real_controller_runtime_and_motor_writer_match_without_target_motion(
        authority, monkeypatch, diagnostics):
    """Same distance/ego samples, different raw windows, through real layers."""
    template = authority
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE", False)
    monkeypatch.setattr(runtime, "DISTANCE_APPROACH_DECELERATION_M_S2", .7)
    results = []
    for scenario in ("baseline", diagnostics):
        template.clock.now = 100.
        obj = copy(template.owner)
        controller = FollowSafetyController(replace(
            template.controller.cfg, distance_target_motion_control_enable=False,
            distance_pi_stationary_stop_preview_enabled=False,
            distance_approach_deceleration_m_s2=.7))
        controller.active_target_id = 1
        controller._has_seen_person = True
        obj._follow_controller = controller
        controller._live_longitudinal_authority_reader = obj._fresh_depth_linear_snapshot
        controller._braking_execution_bound_reader = lambda uid, now: 20.
        a = SimpleNamespace(owner=obj, controller=controller, clock=template.clock,
                            frame=template.frame, feedback=None)
        obj._action_runtime = SimpleNamespace(
            get_steering_feedback=lambda a=a: a.feedback,
            continuation_executed_speed_bound_rpm=lambda uid, now: 20.)
        original = controller._fresh_braking_assessment

        def inject(frame, now, stamp, *, controller=controller, original=original, scenario=scenario):
            rate = {"baseline": 0., "warming": None, "receding": 1000.,
                    "approaching": -1000., "nan": float("nan")}[scenario]
            controller._braking_range_rate = rate
            controller._braking_rate_source = "warming" if rate is None else "raw_depth_window"
            controller._braking_motion_evidence = (None if rate is None else RawDepthMotionEvidence(
                stamp, rate, rate, .16, 3))
            return original(frame, now, stamp)

        controller._fresh_braking_assessment = inject
        approvals = []
        for _ in range(4):
            current = a.frame(3.5, rpm=20.)
            _, actions, accepted = decide_commit(a, current)
            assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
            shared = controller.last_distance_pid_result.pi_braking_assessment
            assert isinstance(shared, SampleBrakingAssessment) and shared.target_speed_m_s == 0.
            assert obj._depth30_linear_timing.braking_assessment is shared
            assert obj._depth30_linear_timing.continuation_motion is None
            approvals.append(obj._depth30_linear_snapshot[1])
            a.clock.now += .05
        stamp = obj._depth30_linear_snapshot[3]
        frozen = obj._depth30_linear_timing
        watermark = obj._depth30_linear_sample_watermark
        action, backend = writer(a)
        action.get_steering_feedback = lambda a=a: a.feedback
        action.continuation_executed_speed_bound_rpm = lambda uid, now: 20.
        caps = []
        for age in (.10, .179, .181, .21, .249):
            advance(a, stamp+age)
            current = obj._fresh_depth_linear_snapshot(1, quiet=True)
            assert current is not None and current[3] == stamp
            caps.append(current[1])
            action._service_follow_wheels()
            assert backend.pairs and backend.pairs[-1][0] > 0 and backend.pairs[-1][1] < 0
            assert obj._depth30_linear_timing is frozen
            assert frozen.braking_assessment is shared
            assert frozen.depth_expires_at == pytest.approx(stamp+.25)
            assert obj._depth30_linear_sample_watermark == watermark
        advance(a, stamp+.250001)
        action._service_follow_wheels()
        assert backend.pairs[-1][:2] == (0, 0)
        assert obj._fresh_depth_linear_snapshot(1) is None
        results.append((approvals, caps, backend.pairs))
    assert results[0] == results[1]
