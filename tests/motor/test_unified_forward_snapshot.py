"""Shared-braking production capability, complete fake motor execution chain."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.control_types import DepthLinearTiming, SteeringFeedback
from car_control_modular.detector_identity_lease import ValidatedVisualObservation
from car_control_modular.lateral_intent import LateralIntentStore
from car_control_modular.sample_braking import SampleBrakingAssessment
from test_cap331_intent_handoff import ordinary_intent
from test_follow_deferred_diagnostics import live_writer


def feedback(stamp, left=0., right=0.):
    return SteeringFeedback(timestamp=stamp, left_forward_rpm=left,
        right_forward_rpm=right, trustworthy=True, raw_yaw_rate_right_dps=0.)


def writer(monkeypatch, *, base=90., yaw=-6.):
    rt, owner, driver, clock, logger = live_writer(monkeypatch)
    store = owner._lateral_intent_store = LateralIntentStore()
    values = [base, yaw]
    reads = []
    owner._follow_controller.cfg = SimpleNamespace(
        visible_steering_pid_image_error_only=True,
        visible_steering_pid_execution_response_trial_sec=.35,
        visible_steering_pid_max_correction_rpm=10., near_distance_rotation_only_max_rpm=6.)

    def publish(base, yaw, *, new_depth=True, park_hint=False, uid=1):
        values[:] = [base, yaw]
        stamp = clock[0]
        if new_depth:
            raw = ("forward", float(base), uid, stamp)
            assessment = SampleBrakingAssessment(uid, stamp, stamp, 10., 24., 24.,
                stamp, 0., 1.1, .8168, 1., .15, 100., outer_allowance_rpm=10.)
            timing = DepthLinearTiming(raw, stamp, stamp+.25, braking_assessment=assessment)
            owner._depth30_prepared_timing = timing
            owner._depth30_linear_snapshot = raw
            owner._depth30_linear_timing = timing
        owner._lateral_yaw_revision += 1
        intent = store.publish(ordinary_intent(
            sign=-1 if yaw < 0 else 1,
            capture=stamp, published=stamp, initial_correction_rpm=yaw,
            base_rpm=base, base_percent=base, park_requested=park_hint,
            forward_countersteer=park_hint))
        owner._lateral_turn_response_policy = (intent.sequence, False)
        owner._lateral_intent_zero_sequence = intent.sequence if yaw == 0 else -1
        rt._steering_feedback = feedback(stamp, 24., 24.)
        return intent

    def depth(uid, now=None):
        assert not owner.motor_io_lock.locked()
        reads.append("depth")
        raw = owner._depth30_linear_snapshot
        return raw if raw and raw[2] == uid and 0 <= clock[0]-raw[3] <= .25 else None

    def axes(now):
        assert not owner.motor_io_lock.locked()
        reads.append("axes")
        raw = owner._depth30_linear_snapshot
        valid = raw and raw[2] == owner._follow_controller.active_target_id and 0 <= now-raw[3] <= .25
        return (1, owner._lateral_yaw_revision, values[0] if valid else 0., values[1])

    def limit(raw, timing, now, *, feedback, quiet):
        assert quiet
        budget = timing.braking_assessment.budget(now,
            max(feedback.left_forward_rpm, feedback.right_forward_rpm),
            authorized_rpm=raw[1], execution_bound_rpm=raw[1]+10.)
        return budget.cap_rpm, budget.reason  # max forward=100, RPM == percent.

    owner._fresh_depth_linear_snapshot = depth
    owner._follow_wheel_axes = axes
    owner._has_fresh_lateral_yaw = lambda uid: bool(values[1] and store.snapshot().valid(clock[0]))
    owner._depth_forward_continuation_required = lambda *_: True
    owner._depth_forward_continuation_limit = limit
    rt.get_steering_feedback = lambda: rt._steering_feedback
    publish(base, yaw)
    rt._service_follow_wheels()
    assert len(driver.pairs) == 1 and all(driver.pairs[0])
    clock[0] = 10.05
    return rt, owner, driver, clock, publish, reads


def inject(monkeypatch, rt, stage, change):
    calls = []
    if stage == "feedback":
        original = rt.get_steering_feedback

        def hook():
            value = original()
            if not calls:
                calls.append(True)
                change()
            return value

        monkeypatch.setattr(rt, "get_steering_feedback", hook)
    elif stage == "guard":
        original = rt._visible_wheel_guard.limit

        def hook(*args, **kwargs):
            result = original(*args, **kwargs)
            if not calls:
                calls.append(True)
                change()
            return result

        monkeypatch.setattr(rt._visible_wheel_guard, "limit", hook)
    elif stage == "commit":
        original = rt._begin_follow_commit

        def hook():
            if not calls:
                calls.append(True)
                change()
            return original()

        monkeypatch.setattr(rt, "_begin_follow_commit", hook)
    else:
        original = rt._linear_packet_write_limit

        def hook(*args, **kwargs):
            if not calls:
                calls.append(True)
                change()
            return original(*args, **kwargs)

        monkeypatch.setattr(rt, "_linear_packet_write_limit", hook)
    return calls


@pytest.mark.parametrize("stage", ["feedback", "guard", "commit", "terminal"])
@pytest.mark.parametrize("new_base,new_yaw", [(88., -6.), (94., 0.), (94., 4.), (60., 0.)])
def test_normal_updates_at_every_gate_commit_latest_nonzero_packet(
        monkeypatch, stage, new_base, new_yaw):
    rt, owner, driver, clock, publish, reads = writer(monkeypatch)
    old = owner._depth30_linear_timing
    calls = inject(monkeypatch, rt, stage, lambda: publish(
        new_base, new_yaw, park_hint=new_yaw == 0))
    rt._service_follow_wheels()
    assert calls == [True]
    assert len(driver.pairs) == 2 and all(driver.pairs[-1])
    left, raw_right = driver.pairs[-1]
    assert .5*(left-raw_right) <= new_base
    assert .5*(left+raw_right) == new_yaw
    assert owner._depth30_linear_timing is not old
    assert rt._forward_execution_anchor.sample_timestamp == clock[0]
    assert not owner.motor_io_lock.locked() and not driver.stops


def test_cap186_end_yaw_does_not_require_lower_than_last_executed_inner_wheel(monkeypatch):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=54., yaw=-9.)
    assert driver.pairs == [(45, -63)]
    publish(94., 5.)
    calls = inject(monkeypatch, rt, "feedback", lambda: publish(
        94., 0., new_depth=False, park_hint=True))
    rt._service_follow_wheels()
    assert calls and driver.pairs == [(45, -63), (94, -94)]
    assert owner._depth30_linear_snapshot[3] == 10.05


def test_early_then_guard_then_terminal_updates_use_short_refresh_not_full_restarts(monkeypatch):
    rt, owner, driver, clock, publish, reads = writer(monkeypatch)
    full_plans = []
    plan = rt._plan_follow_wheel_targets

    def counted(*args, **kwargs):
        full_plans.append(True)
        return plan(*args, **kwargs)

    monkeypatch.setattr(rt, "_plan_follow_wheel_targets", counted)
    inject(monkeypatch, rt, "feedback", lambda: publish(90., -4.))
    inject(monkeypatch, rt, "guard", lambda: publish(88., -6.))
    inject(monkeypatch, rt, "terminal", lambda: publish(86., 0., park_hint=True))
    reads.clear()
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (86, -86)
    assert len(driver.pairs) == 2 and len(full_plans) == 1
    assert reads.count("axes") <= 6 and reads.count("depth") <= 5


@pytest.mark.parametrize("fault", ["uid", "identity", "near", "feedback", "reverse",
                                    "expiry", "stop", "hazard", "assessment"])
@pytest.mark.parametrize("stage", ["guard", "commit", "terminal"])
def test_real_revocations_never_escape_short_commit(monkeypatch, stage, fault):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch)

    def change():
        publish(88., -6.)
        if fault == "uid":
            owner._follow_controller.active_target_id = 2
        elif fault == "identity":
            owner._vision_control_state = "target_lost"
        elif fault == "near":
            owner._depth30_linear_snapshot = None
        elif fault == "feedback":
            rt._steering_feedback = feedback(clock[0]-.151, 24., 24.)
        elif fault == "reverse":
            rt._steering_feedback = feedback(clock[0], -20., 24.)
        elif fault == "expiry":
            clock[0] += .251
            rt._steering_feedback = feedback(clock[0], 24., 24.)
        elif fault == "stop":
            rt.backend.send_stop("test-stop", mode="emergency", preserve_zero=True)
        elif fault == "hazard":
            rt.hard_stop_check = lambda _: True
        else:
            timing = replace(owner._depth30_linear_timing, braking_assessment=None)
            owner._depth30_prepared_timing = owner._depth30_linear_timing = timing

    inject(monkeypatch, rt, stage, change)
    rt._service_follow_wheels()
    assert all(pair == (0, 0) for pair in driver.pairs[1:])
    if fault in {"stop", "hazard"}:
        assert driver.stops == [1] and len(driver.pairs) == 1


def test_continuous_fresh_publication_is_adopted_without_zero_or_lease_extension(monkeypatch):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch)
    original = rt._linear_packet_write_limit
    calls = []

    def change(*args, **kwargs):
        calls.append(True)
        clock[0] += .005
        publish(88., -6.)
        return original(*args, **kwargs)

    monkeypatch.setattr(rt, "_linear_packet_write_limit", change)
    rt._service_follow_wheels()
    assert len(calls) == 1  # Fresh equivalent evidence is not a failed plan.
    assert driver.pairs[-1] == (82, -94)
    assert not owner.motor_io_lock.locked()
    history = rt._continuation_executed_speed_history
    assert len(history) == 2
    assert history[-1].outer_rpm == 94
    assert history[-1].receipt is rt.backend.last_speed_receipt
    # A newer grant doesn't erase previously acknowledged high momentum.
    assert history[0].outer_rpm > 0
    assert rt._forward_execution_anchor.sample_timestamp == clock[0]
    assert rt._turn_response_trial.history[-1][1:] == (88., -6.)
    assert owner._depth30_linear_timing.depth_expires_at == clock[0] + .25
    # No new publication: the physical deadline still stops translation.
    monkeypatch.setattr(rt, "_linear_packet_write_limit", original)
    clock[0] += .251
    rt._steering_feedback = feedback(clock[0], 24., 24.)
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (0, 0)


def test_short_veto_without_completed_zero_receipt_cannot_fabricate_history(monkeypatch):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch)
    history = rt._continuation_executed_speed_history
    trial_history = tuple(rt._turn_response_trial.history)
    original = rt.backend.send_targets

    def no_zero_receipt(left, right, label, **kwargs):
        if left == right == 0:
            # A backend hold/fault path did not complete a dual-wheel zero.
            rt.backend.last_speed_receipt = None
            return
        original(left, right, label, **kwargs)

    monkeypatch.setattr(rt.backend, "send_targets", no_zero_receipt)
    inject(monkeypatch, rt, "terminal", lambda: setattr(
        owner, "_validated_visual_observation", False))
    rt._service_follow_wheels()
    assert len(driver.pairs) == 1
    assert rt._continuation_executed_speed_history == history
    assert tuple(rt._turn_response_trial.history) == trial_history
    assert rt._forward_execution_anchor is None


@pytest.mark.parametrize("fault", ["rejected", "uid", "expired", "new_depth_on_grey"])
@pytest.mark.parametrize("cap_number", [1, 2])
def test_visual_rejection_after_either_quiet_cap_cannot_write_forward(
        monkeypatch, fault, cap_number):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch)
    publish(88., -6.)
    proof = ValidatedVisualObservation(1, 3, 200, 10.05, 10.05, 10.2, "full")
    owner._validated_visual_observation = proof
    original = owner._depth_forward_continuation_limit
    calls = []

    def late_rejection(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(True)
        if len(calls) == cap_number:
            if fault == "rejected":
                owner._validated_visual_observation = False
            elif fault == "uid":
                owner._validated_visual_observation = replace(proof, uid=2)
            elif fault == "expired":
                owner._validated_visual_observation = replace(proof, expires_at=clock[0])
            else:
                owner._validated_visual_observation = replace(proof,
                    continuation_sample_timestamp=10.)
        return result

    owner._depth_forward_continuation_limit = late_rejection
    rt._service_follow_wheels()
    assert len(calls) >= cap_number and driver.pairs[-1] == (0, 0)
    assert len(driver.pairs) == 2


def test_grey_visual_proof_only_finishes_its_bound_existing_depth(monkeypatch):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch)
    owner._validated_visual_observation = ValidatedVisualObservation(
        1, 3, 200, 10., 10., 10.2, "full", continuation_sample_timestamp=10.)
    publish(90., -4., new_depth=False)
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (86, -94)
    assert rt._linear_packet_write_limit(None, 1, 0).limit_rpm == 0.
    clock[0] += .05
    publish(88., -4.)
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (0, 0)


@pytest.mark.parametrize("elapsed", [0., .081])
def test_short_refresh_does_not_renew_or_double_consume_countersteer_time(
        monkeypatch, elapsed):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=-5.)
    publish(60., 5.)
    store = owner._lateral_intent_store
    intent = store.publish(replace(store.snapshot(), x_ratio=.35,
        target_image_rate_dps=12., forward_countersteer=True, park_requested=True))
    owner._lateral_turn_response_policy = (intent.sequence, False)
    rt._steering_feedback = replace(feedback(clock[0], 20., 26.),
        yaw_rate_right_dps=-10., raw_yaw_rate_right_dps=-10.)
    trial = rt._turn_response_trial
    original = trial.adjust
    calls = []

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((clock[0], trial.brake_started, result))
        return result

    def advance():
        clock[0] += elapsed
        rt._steering_feedback = replace(rt._steering_feedback, timestamp=clock[0])

    monkeypatch.setattr(trial, "adjust", record)
    inject(monkeypatch, rt, "guard", advance)
    rt._service_follow_wheels()
    assert len(calls) == 2
    assert [call[1] for call in calls] == [10.05, 10.05]
    expected_yaw = 5 if not elapsed else 0
    assert driver.pairs[-1] == (60+expected_yaw, -(60-expected_yaw))
    assert len(trial.history) == 2  # completed packets, not repeated plans
