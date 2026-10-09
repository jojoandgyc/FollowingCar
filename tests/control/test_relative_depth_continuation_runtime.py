"""One physical grant uses one braking model, including final fake motor I/O.

No PersonTracker constructor, camera, serial port, or motor thread is started.
The fixtures exercise the real commit, reader, late-depth, and motor writers.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction, ControlDecision, SteeringFeedback
from car_control_modular.longitudinal_approach import RawDepthMotionEvidence
from test_depth_authority_250 import writer
from test_distance_pi_runtime import pi_owner
from test_distance_pi_controller import configured
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import NOW, owner
from test_longitudinal_authority_runtime import _frame


CIRCUMFERENCE = .816814


@pytest.fixture
def relative(pi_owner, monkeypatch):
    clock = SimpleNamespace(now=NOW)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock.now)
    for key, value in {
        "ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE": True,
        "ASTRA_DEPTH_CONTINUATION_OVERSHOOT_M": .20,
        "ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC": .25,
        "ASTRA_DEPTH_LONGITUDINAL_CONTROL_SAMPLE_MAX_AGE_SEC": .18,
        "ASTRA_DEPTH_CONTINUATION_SPEED_CAP_ENABLE": True,
        "TARGET_DISTANCE": 1.4,
        "FOLLOW_REVERSE_START_DISTANCE_M": 1.2,
        "DISTANCE_PID_DEADBAND_M": .03,
        "DISTANCE_APPROACH_DECELERATION_M_S2": .4,
        "DISTANCE_APPROACH_RESPONSE_DELAY_SEC": .2,
        "VISION_MMWAVE_FUSION_ENCODER_WHEEL_CIRCUMFERENCE_M": CIRCUMFERENCE,
        "MOTOR_FORWARD_MAX_TARGET_RPM": 200,
    }.items():
        monkeypatch.setattr(runtime, key, value)
    state = SimpleNamespace(clock=clock, owner=pi_owner, feedback=None)
    pi_owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: state.feedback)
    pi_owner._follow_controller.search_state = "none"
    pi_owner._follow_controller.distance_only_forward_percent = lambda *_args: 100

    def frame(distance, rpm=16., stamp=None):
        stamp = clock.now if stamp is None else stamp
        return replace(_frame(stamp, distance), steering_feedback=SteeringFeedback(
            timestamp=clock.now, trustworthy=True,
            left_forward_rpm=rpm, right_forward_rpm=rpm))

    state.frame = frame
    return state


def publish(state, *, distance=2.5648, percent=58, rpm=16., stamp=None,
            target_speed=.764, range_rate=.546, raw_changes=None, result_changes=None):
    """Attach the same-window source that the real distance PI publishes."""
    stamp = state.clock.now-.02 if stamp is None else stamp
    controller = state.owner._follow_controller
    controller._distance_pid_last_sample_timestamp = stamp
    controller._braking_rate_source = "raw_depth_window"
    controller._braking_range_rate = range_rate
    source = RawDepthMotionEvidence(stamp, range_rate, target_speed, .16, 3)
    controller._braking_motion_evidence = replace(source, **(raw_changes or {}))
    result = dict(approach_mode="distance_pi", output_rpm=percent*2,
                  pi_motion_window_used=True, pi_brake_source="raw_relative_motion",
                  approach_closing_m_s=max(0., -range_rate),
                  pi_motion_window_target_speed_m_s=target_speed,
                  pi_motion_window_range_rate_m_s=range_rate,
                  pi_motion_window_span_sec=.16)
    result.update(result_changes or {})
    controller.last_distance_pid_result = SimpleNamespace(**result)
    current = state.frame(distance, rpm=rpm, stamp=stamp)
    state.feedback = current.steering_feedback
    actions, accepted = state.owner._commit_depth_linear_decision(
        ControlDecision(actions=[ControlAction.forward(percent, "distance_pi")], reason="distance_pi"),
        current, 1, is_fresh_depth=True)
    assert accepted and actions and actions[0].speed_percent > 0
    return stamp, state.owner._depth30_linear_snapshot


def at_age(state, stamp, age, *, rpm=16.):
    state.clock.now = stamp+age
    state.owner._last_vision_control_ts = state.clock.now-.01
    state.feedback = SteeringFeedback(timestamp=state.clock.now, trustworthy=True,
                                     left_forward_rpm=rpm, right_forward_rpm=rpm)
    return state.owner._fresh_depth_linear_snapshot(1)


def test_cap238_has_no_model_cliff_at_180ms(relative):
    stamp, original = publish(relative)
    timing = relative.owner._depth30_linear_timing
    approvals = list(relative.owner.approvals)
    values = [at_age(relative, stamp, age) for age in (.020, .179, .181, .210, .249)]
    assert all(value is not None for value in values)
    rpms = [value[1]*2 for value in values]
    assert rpms == sorted(rpms, reverse=True)
    assert rpms[1]-rpms[2] <= 2  # Quantization, not the old 116 -> 46 RPM cliff.
    assert rpms[2] >= 70
    assert all(value[3] == stamp and value[1] <= original[1] for value in values)
    assert relative.owner._depth30_linear_timing is timing
    assert relative.owner.approvals == approvals
    assert timing.depth_expires_at == pytest.approx(stamp+.25)
    assert at_age(relative, stamp, .251) is None


def test_quiet_motor_reader_retains_reason_for_in_grant_feedback_veto(relative):
    stamp, _linear = publish(relative)
    relative.clock.now = stamp+.19
    relative.owner._last_vision_control_ts = relative.clock.now-.01
    relative.feedback = SteeringFeedback(
        timestamp=relative.clock.now-.151, trustworthy=True,
        left_forward_rpm=16., right_forward_rpm=16.)
    assert relative.owner._fresh_depth_linear_snapshot(
        1, now=relative.clock.now, quiet=True) is None
    audit = relative.owner._last_quiet_depth_veto
    assert audit[0:3] == (1, stamp, "continuation_feedback_stale")
    assert audit[4] == relative.feedback.timestamp
    # Fresh feedback does not revive this already-vetoed old grant. The
    # original cause remains visible in the next motor-dispatch diagnostic.
    relative.feedback = SteeringFeedback(
        timestamp=relative.clock.now, trustworthy=True,
        left_forward_rpm=16., right_forward_rpm=16.)
    assert relative.owner._fresh_depth_linear_snapshot(
        1, now=relative.clock.now, quiet=True) is None
    assert relative.owner._last_quiet_depth_veto == audit


def test_quiet_veto_diagnostic_does_not_inherit_previous_grant_reason(relative):
    stamp, _ = publish(relative)
    relative.clock.now = stamp+.19
    relative.owner._last_vision_control_ts = relative.clock.now-.01
    relative.feedback = SteeringFeedback(
        timestamp=relative.clock.now-.151, trustworthy=True,
        left_forward_rpm=16., right_forward_rpm=16.)
    assert relative.owner._fresh_depth_linear_snapshot(
        1, now=relative.clock.now, quiet=True) is None
    old_reason = relative.owner._last_quiet_depth_veto
    assert old_reason[2] == 'continuation_feedback_stale'

    relative.clock.now = stamp+.23
    new_stamp, _ = publish(relative, distance=2.7)
    relative.owner._depth30_continuation_veto = (1, new_stamp)
    relative.clock.now = new_stamp+.19
    relative.owner._last_vision_control_ts = relative.clock.now-.01
    assert relative.owner._fresh_depth_linear_snapshot(
        1, now=relative.clock.now, quiet=True) is None
    assert relative.owner._last_quiet_depth_veto[0:3] == (
        1, new_stamp, 'sticky_continuation_veto')


def test_newer_lower_encoder_cannot_raise_same_grant(relative):
    stamp, _ = publish(relative)
    first = at_age(relative, stamp, .179, rpm=55.)
    assert first is not None
    later = [at_age(relative, stamp, age, rpm=rpm)
             for age, rpm in ((.181, 35.), (.200, 16.), (.220, 0.))]
    values = [first[1]]+[0 if value is None else value[1] for value in later]
    assert values == sorted(values, reverse=True)


def test_first_aged_admission_reports_reduced_limit_but_keeps_original_travel_bound(relative):
    stamp, original = publish(relative, stamp=NOW-.10)
    timing = relative.owner._depth30_linear_timing
    assert 0 < original[1] < 58
    assert relative.owner.approvals[-1] == (stamp, original[1]*2)
    assert timing.continuation_speed_bound_m_s == pytest.approx(116*CIRCUMFERENCE/60.)
    values = [at_age(relative, stamp, age) for age in (.10, .179, .181, .249)]
    percents = [0 if value is None else value[1] for value in values]
    assert percents == sorted(percents, reverse=True)
    assert timing.depth_expires_at == pytest.approx(stamp+.25)


@pytest.mark.parametrize("fault", ["cache_stale", "outer_momentum"])
def test_fresh_envelope_veto_clears_positive_action_and_rejects_pi_sample(relative, fault):
    stamp = NOW-.02
    controller = relative.owner._follow_controller
    controller._distance_pid_last_sample_timestamp = stamp
    controller._braking_rate_source = "raw_depth_window"
    controller._braking_range_rate = .546
    controller._braking_motion_evidence = RawDepthMotionEvidence(stamp, .546, .764, .16, 3)
    controller.last_distance_pid_result = SimpleNamespace(
        approach_mode="distance_pi", output_rpm=116,
        pi_motion_window_used=True, pi_brake_source="raw_relative_motion")
    frame = relative.frame(2.5648, stamp=stamp)
    relative.feedback = (replace(frame.steering_feedback, timestamp=NOW-.151)
                         if fault == "cache_stale" else
                         replace(frame.steering_feedback, left_forward_rpm=160., right_forward_rpm=16.))
    actions, accepted = relative.owner._commit_depth_linear_decision(
        ControlDecision(actions=[ControlAction.forward(58, "distance_pi")], reason="distance_pi"),
        frame, 1, is_fresh_depth=True)
    assert accepted
    assert [(action.kind, action.speed_percent, action.reason) for action in actions] == [
        ("forward", 0, "relative_depth_admission_veto")]
    assert relative.owner.rejections[-1] == (stamp, "relative_depth_admission_veto")
    assert not any(rpm == 0 for _stamp, rpm in relative.owner.approvals)
    assert relative.owner._depth30_linear_snapshot is None
    assert relative.owner._depth30_linear_timing is None
    assert relative.owner._current_forward_percent == 0
    assert not relative.owner._current_forward_allow_below_min
    relative.feedback = frame.steering_feedback
    assert relative.owner._fresh_depth_linear_snapshot(1) is None


def test_reader_between_snapshot_and_timing_publication_uses_new_complete_envelope(relative, monkeypatch):
    stamp = NOW-.10
    old_setattr = runtime.PersonTracker.__setattr__
    observed = []
    reading = False

    def intercept(obj, name, value):
        nonlocal reading
        old_setattr(obj, name, value)
        if (obj is relative.owner and name == "_depth30_linear_snapshot"
                and value is not None and value[3] == stamp and not reading):
            reading = True
            try:
                before_timing = getattr(obj, "_depth30_linear_timing", None)
                live = obj._fresh_depth_linear_snapshot(1)
                observed.append((value, before_timing, live))
            finally:
                reading = False

    monkeypatch.setattr(runtime.PersonTracker, "__setattr__", intercept)
    _, final = publish(relative, stamp=stamp)
    assert observed and observed[0][1] is None
    # Arithmetic requests are now previewed before publication; no reader
    # should ever see the unqualified 58% request, even momentarily.
    assert observed[0][0] == final
    assert all(live is not None and 0 < live[1] < 58 for _value, _timing, live in observed)
    assert all(live[3] == stamp for _value, _timing, live in observed)
    assert observed[0][2] == final
    assert getattr(relative.owner, "_depth30_prepared_timing", None) is None
    assert relative.owner._depth30_linear_timing.snapshot == final


def test_building_next_motion_evidence_keeps_previous_complete_grant_readable(relative, monkeypatch):
    # This test exercises atomic publication, not the late-admission budget.
    # Without qualified continuation evidence a new grant at 160ms has less
    # than the required dispatch budget before its 180ms control deadline.
    old_stamp, _ = publish(relative, stamp=NOW-.10)
    previous = relative.owner._fresh_depth_linear_snapshot(1)
    previous_timing = relative.owner._depth30_linear_timing
    evidence_reader = relative.owner._depth_continuation_evidence
    during_build = []

    def intercept(frame, target_id, percent, now):
        during_build.append((relative.owner._fresh_depth_linear_snapshot(1),
                             relative.owner._depth30_linear_timing))
        return evidence_reader(frame, target_id, percent, now)

    monkeypatch.setattr(relative.owner, "_depth_continuation_evidence", intercept)
    new_stamp, latest = publish(relative, stamp=NOW-.04)
    assert new_stamp > old_stamp
    assert during_build == [(previous, previous_timing)]
    assert previous is not None and latest is not None
    assert latest[3] == new_stamp
    assert relative.owner._depth30_linear_timing.continuation_motion.sample_timestamp == new_stamp
    assert getattr(relative.owner, "_depth30_continuation_veto", None) is None


def test_old_grant_does_not_consult_new_controller_motion(relative):
    stamp, _ = publish(relative)
    timing = relative.owner._depth30_linear_timing
    proof = timing.continuation_motion
    assert proof is not None and proof.sample_timestamp == stamp and proof.uid == 1
    before = at_age(relative, stamp, .181)
    controller = relative.owner._follow_controller
    controller._braking_motion_evidence = RawDepthMotionEvidence(stamp+.10, 2., 3., .16, 3)
    controller.last_distance_pid_result = SimpleNamespace(
        pi_motion_window_used=True, pi_brake_source="raw_relative_motion",
        pi_motion_window_target_speed_m_s=3.)
    assert relative.owner._fresh_depth_linear_snapshot(1) == before
    assert relative.owner._depth30_linear_timing.continuation_motion is proof


@pytest.mark.parametrize("changes", [
    {"sample_timestamp": NOW-.021}, {"sample_timestamp": NOW+.01},
    {"target_speed_bound_m_s": float("nan")}, {"range_rate_m_s": float("inf")},
    {"span_sec": 0.}, {"sample_count": 1},
])
def test_bad_motion_is_not_captured_as_relative_permission(relative, changes):
    publish(relative, raw_changes=changes)
    assert relative.owner._depth30_linear_timing.continuation_motion is None


@pytest.mark.parametrize("changes", [
    {"pi_motion_window_used": False}, {"pi_brake_source": "relative_motion_memory"},
    {"pi_brake_source": "stationary_fallback"},
])
def test_memory_or_unused_motion_cannot_donate_proof(relative, changes):
    publish(relative, result_changes=changes)
    assert relative.owner._depth30_linear_timing.continuation_motion is None


def test_new_fresh_measurement_owns_its_own_proof_and_deadline(relative):
    first_stamp, _ = publish(relative)
    first = relative.owner._depth30_linear_timing
    at_age(relative, first_stamp, .10)
    next_stamp, _ = publish(relative, target_speed=.4, range_rate=.182)
    latest = relative.owner._depth30_linear_timing
    assert latest.continuation_motion is not first.continuation_motion
    assert latest.continuation_motion.sample_timestamp == next_stamp
    assert latest.continuation_motion.target_speed_bound_m_s == pytest.approx(.4)
    assert latest.depth_expires_at == pytest.approx(next_stamp+.25)


def test_nonfresh_reduction_inherits_original_motion_without_new_deadline(relative):
    stamp, _ = publish(relative)
    first = relative.owner._depth30_linear_timing
    at_age(relative, stamp, .10)
    relative.owner._follow_controller._braking_motion_evidence = RawDepthMotionEvidence(
        relative.clock.now, 2., 3., .16, 3)
    actions, accepted = relative.owner._commit_depth_linear_decision(
        ControlDecision(actions=[ControlAction.forward(30, "held")]),
        relative.frame(None), 1, is_fresh_depth=False)
    assert accepted and actions[0].speed_percent <= 30
    latest = relative.owner._depth30_linear_timing
    assert latest.continuation_motion is first.continuation_motion
    assert latest.depth_expires_at == first.depth_expires_at
    assert latest.accepted_depth_timestamp == first.accepted_depth_timestamp


@pytest.mark.parametrize("distance", [1.0, 1.39, 1.43])
def test_new_late_near_measurement_still_revokes_moving_target_proof(relative, distance):
    stamp, _ = publish(relative)
    at_age(relative, stamp, .210)
    actions, accepted = relative.owner._commit_depth_linear_decision(
        ControlDecision(), relative.frame(distance, stamp=stamp+.001), 1, is_fresh_depth=True)
    assert accepted and actions[0].speed_percent == 0
    assert relative.owner._fresh_depth_linear_snapshot(1) is None


def test_late_far_sample_never_renews_or_replaces_motion(relative):
    stamp, _ = publish(relative)
    original = relative.owner._depth30_linear_timing
    watermark = relative.owner._depth30_linear_sample_watermark
    before = at_age(relative, stamp, .210)
    actions, accepted = relative.owner._commit_depth_linear_decision(
        ControlDecision(actions=[ControlAction.forward(100, "late")]),
        relative.frame(10., stamp=stamp+.001), 1, is_fresh_depth=True)
    assert accepted and 0 < actions[0].speed_percent <= before[1]
    current = relative.owner._depth30_linear_timing
    assert current.continuation_motion is original.continuation_motion
    assert current.continuation_distance_m <= original.continuation_distance_m
    assert current.depth_expires_at == original.depth_expires_at
    assert relative.owner._depth30_linear_sample_watermark == watermark


@pytest.mark.parametrize("age", [.10, .210])
@pytest.mark.parametrize("field,value", [
    ("_explicit_stop_requested", True), ("_runtime_shutdown_requested", True),
    ("_brake_hold_active", True), ("search_state", "searching"),
    ("_vision_control_state", "target_lost"),
    ("_vision_control_state", "target_visible_low_quality"),
])
def test_owner_veto_cannot_revive_relative_grant_at_any_age(relative, age, field, value):
    stamp, _ = publish(relative)
    assert at_age(relative, stamp, age)
    old = getattr(relative.owner, field)
    setattr(relative.owner, field, value)
    assert relative.owner._fresh_depth_linear_snapshot(1) is None
    setattr(relative.owner, field, old)
    assert relative.owner._fresh_depth_linear_snapshot(1) is None


def test_uid_change_cannot_be_undone_to_restore_same_grant(relative):
    stamp, _ = publish(relative)
    assert at_age(relative, stamp, .10)
    relative.owner._follow_controller.active_target_id = 2
    assert relative.owner._fresh_depth_linear_snapshot(2) is None
    relative.owner._follow_controller.active_target_id = 1
    assert relative.owner._fresh_depth_linear_snapshot(1) is None


@pytest.mark.parametrize("fault", ["missing", "stale", "future", "untrusted", "reverse", "overspeed"])
def test_current_feedback_veto_never_repaired_by_only_encoder_refresh(relative, fault):
    stamp, _ = publish(relative)
    assert at_age(relative, stamp, .10)
    feedback = relative.feedback
    if fault == "missing":
        relative.feedback = None
    else:
        kwargs = {
            "stale": {"timestamp": relative.clock.now-.151},
            "future": {"timestamp": relative.clock.now+.001},
            "untrusted": {"trustworthy": False},
            "reverse": {"left_forward_rpm": -1.},
            "overspeed": {"left_forward_rpm": 201.},
        }[fault]
        relative.feedback = replace(feedback, **kwargs)
    assert relative.owner._fresh_depth_linear_snapshot(1) is None
    relative.feedback = feedback
    assert relative.owner._fresh_depth_linear_snapshot(1) is None


def test_relative_proof_does_not_extend_visibility(relative):
    stamp, _ = publish(relative)
    relative.clock.now = stamp+.210
    relative.owner._last_vision_control_ts = relative.clock.now-.251
    relative.feedback = replace(relative.feedback, timestamp=relative.clock.now)
    assert relative.owner._fresh_depth_linear_snapshot(1) is None


@pytest.mark.parametrize("left,right", [(30.,70.), (70.,30.)])
def test_relative_turning_must_still_fit_faster_wheel_braking(relative, left, right):
    stamp, _ = publish(relative, distance=2.05, percent=30, target_speed=.3, range_rate=.082)
    assert at_age(relative, stamp, .181, rpm=40.) is not None
    relative.feedback = replace(relative.feedback, left_forward_rpm=left, right_forward_rpm=right)
    assert relative.owner._fresh_depth_linear_snapshot(1) is None


def test_switch_off_preserves_old_stationary_policy(relative, monkeypatch):
    stamp, _ = publish(relative)
    fresh = at_age(relative, stamp, .181)
    assert fresh is not None
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE", False)
    legacy = relative.owner._fresh_depth_linear_snapshot(1)
    assert legacy is not None and legacy[1] < fresh[1]
    assert 40 <= legacy[1]*2 <= 55
    assert legacy[3] == stamp


def test_no_motion_proof_keeps_old_policy_without_new_permission(relative):
    stamp, _ = publish(relative, result_changes={"pi_motion_window_used": False})
    assert relative.owner._depth30_linear_timing.continuation_motion is None
    legacy = at_age(relative, stamp, .181)
    assert legacy is not None and 40 <= legacy[1]*2 <= 55


def test_real_controller_window_is_captured_by_runtime_and_replay_cannot_refresh_it(relative, setup, monkeypatch):
    clock, controller, frame = configured(setup,
        target_distance_m=1.4, depth_longitudinal_sample_max_age_sec=.25,
        distance_feedforward_enable=False)
    relative.clock = clock
    relative.frame = frame
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock.now)
    relative.owner._follow_controller = controller
    controller._live_longitudinal_authority_reader = relative.owner._fresh_depth_linear_snapshot
    for index in range(3):
        clock.now = 100.+.08*index
        observation = frame(2.6+.05*index, rpm=20.)
        relative.feedback = observation.steering_feedback
        relative.owner._last_vision_control_ts = clock.now-.01
        decision = controller.decide(10, observation, longitudinal_only=True)
        actions, accepted = relative.owner._commit_depth_linear_decision(
            decision, observation, 1, is_fresh_depth=True)
        assert accepted and actions[0].speed_percent > 0
    timing = relative.owner._depth30_linear_timing
    source = controller._braking_motion_evidence
    assert controller.last_distance_pid_result.pi_motion_window_used
    assert timing.continuation_motion is not None
    assert timing.continuation_motion.sample_timestamp == observation.distance_state.sample_timestamp
    assert timing.continuation_motion.target_speed_bound_m_s == source.target_speed_bound_m_s
    before_i = controller._distance_pid._distance_pi.integral_m_s
    clock.now += .04
    relative.feedback = replace(relative.feedback, timestamp=clock.now)
    replay = controller.decide(10, observation, longitudinal_only=True)
    relative.owner._commit_depth_linear_decision(replay, observation, 1, is_fresh_depth=True)
    assert relative.owner._depth30_linear_timing is timing
    assert controller._distance_pid._distance_pi.integral_m_s == before_i


def test_relative_periodic_writer_preserves_motion_across_boundary_but_expires(relative):
    stamp, _ = publish(relative)
    action, backend = writer(relative)
    pairs = []
    for age in (.179, .181, .210, .249):
        at_age(relative, stamp, age)
        action._service_follow_wheels()
        pairs.append(backend.pairs[-1][:2])
    assert all(left > 0 and right < 0 for left, right in pairs)
    assert abs(pairs[1][0]-pairs[0][0]) <= 2
    at_age(relative, stamp, .251)
    action._service_follow_wheels()
    assert backend.pairs[-1][:2] == (0, 0)


@pytest.mark.parametrize("direct", [False, True])
def test_relative_final_writer_rechecks_expiry_after_feedback_read(relative, direct):
    stamp, original = publish(relative)
    action, backend = writer(relative)
    if direct:
        action.config.follow_wheel_period_sec = 0.
    at_age(relative, stamp, .249)

    def read_feedback():
        relative.clock.now = stamp+.251
        return relative.frame(2.5, rpm=40).steering_feedback

    action.get_steering_feedback = read_feedback
    if direct:
        with relative.owner.motor_io_lock:
            action._send_follow_wheel_targets(original[1]*2, -original[1]*2, "TEST", visible_required=True)
    else:
        action._service_follow_wheels()
    assert not backend.pairs or all(pair[:2] == (0, 0) for pair in backend.pairs)
