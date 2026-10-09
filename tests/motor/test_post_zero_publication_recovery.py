"""Complete loss/zero/regrant and pivot-publication lifecycles, fake I/O only."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.detector_identity_lease import ValidatedVisualObservation
from car_control_modular.lateral_intent import LateralIntentStore
from test_cap331_intent_handoff import ordinary_intent
from test_follow_wheel_periodic import setup_periodic
from test_unified_forward_snapshot import feedback, writer
from test_zero_publication_lifecycle import churn_at_three_terminal_checks


def stopped_writer(monkeypatch):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=0.)
    owner._validated_visual_observation = ValidatedVisualObservation(
        1, 1, 100, 10., 10., 10.5, "full")
    clock[0] = 10.251
    rt._steering_feedback = feedback(clock[0], 0., 0.)
    rt._service_follow_wheels()
    assert driver.pairs == [(60, -60), (0, 0)]
    zero = rt.backend.last_speed_receipt
    assert zero.left_rpm == zero.right_rpm == 0
    assert rt._forward_execution_anchor is None
    clock[0] = 10.30
    publish(60., -4.)
    return rt, owner, driver, clock, publish


@pytest.mark.parametrize("yaw", [-4., 0., 4.])
@pytest.mark.parametrize("previous", ["zero", "untracked_zero", "pivot", "missing_clock"])
def test_fresh_grant_after_actual_expiry_zero_gets_bounded_independent_admission(
        monkeypatch, yaw, previous):
    rt, owner, driver, clock, publish = stopped_writer(monkeypatch)
    if previous == "pivot":
        owner._depth30_linear_snapshot = owner._depth30_prepared_timing = owner._depth30_linear_timing = None
        publish(0., -7., new_depth=False)
        rt._steering_feedback = feedback(clock[0], -7., 7.)
        rt._service_follow_wheels()
        left, right = driver.pairs[-1]
        assert left == right < 0  # Real pure-yaw ACK, not an invented clock entry.
        clock[0] += .05
        publish(60., -4.)
    elif previous == "missing_clock":
        rt._service_follow_wheels()
        assert driver.pairs[-1][0] > 0 > driver.pairs[-1][1]
        rt._follow_wheel_clock.reset()
        clock[0] += .05
        publish(60., -4.)
    elif previous == "untracked_zero":
        # Generic revoke zeros have an actual ACK but do not advance the
        # success-only clock/receipt. Fresh authority is independent of that.
        rt._follow_wheel_clock.reset()
        rt._follow_wheel_last_receipt = None
    prior_pairs = list(driver.pairs)
    calls = churn_at_three_terminal_checks(monkeypatch, rt, clock, publish, base=70., yaw=yaw)
    rt._service_follow_wheels()
    assert len(calls) == 4
    assert rt._follow_terminal_forward_used
    assert driver.pairs == prior_pairs + [(70, -70)]
    assert owner._depth30_linear_snapshot[3] == pytest.approx(clock[0])
    assert owner._depth30_linear_timing.depth_expires_at == pytest.approx(clock[0]+.25)
    assert rt._forward_execution_anchor.sample_timestamp == owner._depth30_linear_snapshot[3]
    assert rt.backend.last_speed_receipt.left_rpm == 70
    assert not driver.stops and not owner.motor_io_lock.locked()


@pytest.mark.parametrize("fault", ["stop", "explicit_stop", "hazard", "uid", "identity",
                                    "depth", "feedback", "reverse", "budget", "receipt",
                                    "external_zero", "commanded_reverse", "full_reverse"])
def test_post_zero_readmission_rechecks_all_current_authority(monkeypatch, fault):
    rt, owner, driver, clock, publish = stopped_writer(monkeypatch)
    churn_at_three_terminal_checks(monkeypatch, rt, clock, publish, base=70., yaw=-4.)
    original = rt._plan_ordinary_snapshot_forward
    changed = []

    def inject(*args, **kwargs):
        if kwargs.get("straight_fallback") and not changed:
            changed.append(True)
            if fault == "stop":
                rt.backend.send_stop("newer-stop", mode="emergency", preserve_zero=True)
            elif fault == "explicit_stop":
                owner._explicit_stop_requested = True
            elif fault == "hazard":
                rt.hard_stop_check = lambda _: True
            elif fault == "uid":
                owner._follow_controller.active_target_id = 2
            elif fault == "identity":
                owner._validated_visual_observation = False
            elif fault == "depth":
                clock[0] += .251
                rt._steering_feedback = feedback(clock[0], 0., 0.)
            elif fault == "feedback":
                rt._steering_feedback = feedback(clock[0]-.151, 0., 0.)
            elif fault == "reverse":
                rt._steering_feedback = feedback(clock[0], -8., -8.)
            elif fault == "commanded_reverse":
                rt._visible_wheel_guard.commanded_reverse = True
            elif fault == "full_reverse":
                rt._visible_wheel_guard.pending_full_reverse = True
            elif fault == "budget":
                old = owner._depth30_linear_timing
                assessment = replace(old.braking_assessment, distance_m=.1)
                owner._depth30_prepared_timing = owner._depth30_linear_timing = replace(
                    old, braking_assessment=assessment)
            elif fault == "external_zero":
                rt.backend.send_targets(0, 0, "another-writer-zero")
            else:
                rt.backend.send_targets(20, -20, "another-writer")
        return original(*args, **kwargs)

    monkeypatch.setattr(rt, "_plan_ordinary_snapshot_forward", inject)
    rt._service_follow_wheels()
    assert changed == [True]
    own_pairs = driver.pairs[3:] if fault in {"receipt", "external_zero"} else driver.pairs[2:]
    assert all(pair == (0, 0) for pair in own_pairs)
    if fault in {"receipt", "external_zero"}:
        assert len(driver.pairs) == 3  # Only the intervening writer may append.
    if fault in {"stop", "hazard"}:
        assert driver.stops == [1] and len(driver.pairs) == 2
    assert not owner.motor_io_lock.locked()


@pytest.mark.parametrize("fault", ["unknown_uid", "wrong_uid", "reverse"])
def test_missing_execution_clock_does_not_admit_unknown_or_reverse_packet(monkeypatch, fault):
    rt, owner, driver, clock, publish = stopped_writer(monkeypatch)
    rt._service_follow_wheels()
    assert driver.pairs[-1][0] > 0 > driver.pairs[-1][1]
    rt._follow_wheel_clock.reset()
    if fault == "reverse":
        rt.backend.send_targets(-8, 8, "actual-reverse", history_uid=1)
        rt._follow_wheel_last_receipt = rt.backend.last_speed_receipt
        rt._visible_wheel_guard.note_sent((-8, -8), clock[0])
    else:
        rt.backend.last_speed_write = replace(
            rt.backend.last_speed_write, uid=None if fault == "unknown_uid" else 2)
    prior = len(driver.pairs)
    clock[0] += .05
    publish(70., -4.)
    churn_at_three_terminal_checks(monkeypatch, rt, clock, publish, base=70., yaw=-4.)
    rt._service_follow_wheels()
    assert all(pair == (0, 0) for pair in driver.pairs[prior:])
    assert not owner.motor_io_lock.locked()


def test_post_zero_continuous_publication_cannot_restart_the_admission_budget(monkeypatch):
    rt, owner, driver, clock, publish = stopped_writer(monkeypatch)
    original = rt._linear_packet_write_limit
    calls = []

    def refresh(*args, **kwargs):
        calls.append(clock[0])
        clock[0] += .001
        publish(70., -4.)
        return original(*args, **kwargs)

    monkeypatch.setattr(rt, "_linear_packet_write_limit", refresh)
    rt._service_follow_wheels()
    assert len(calls) == 4
    assert driver.pairs == [(60, -60), (0, 0), (0, 0)]
    assert not driver.stops and not owner.motor_io_lock.locked()


def pivot_writer(monkeypatch, yaw=-7.):
    rt, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    state[:] = [0., yaw, 10.5, 10.5]
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
    store = owner._lateral_intent_store = LateralIntentStore()
    owner._follow_controller.cfg = SimpleNamespace(
        visible_steering_pid_image_error_only=True,
        visible_steering_pid_execution_response_trial_sec=.35,
        visible_steering_pid_max_correction_rpm=10., near_distance_rotation_only_max_rpm=7.)

    def publish(**changes):
        value = ordinary_intent(
            capture=clock[0], published=clock[0], cap=int(clock[0]*100),
            sign=-1 if yaw < 0 else 1, initial_correction_rpm=yaw,
            base_rpm=0., base_percent=0, mode="yaw_only", correction_limit_rpm=7.)
        intent = store.publish(replace(value, **changes))
        owner._lateral_turn_response_policy = (intent.sequence, False)
        rt._steering_feedback = feedback(clock[0], yaw, -yaw)
        return intent

    rt.get_steering_feedback = lambda: rt._steering_feedback
    publish()
    rt._service_follow_wheels()
    assert driver.pairs == [(int(yaw), int(yaw))]
    clock[0] = 10.05
    rt._steering_feedback = feedback(clock[0], yaw, -yaw)
    return rt, owner, driver, clock, publish


@pytest.mark.parametrize("yaw", [-7., 7.])
@pytest.mark.parametrize("new_revision", [False, True])
def test_equivalent_pivot_publication_at_commit_replans_without_zero(monkeypatch, yaw, new_revision):
    rt, owner, driver, clock, publish = pivot_writer(monkeypatch, yaw)
    begin = rt._begin_follow_commit
    changed = []

    def refresh():
        if not changed:
            changed.append(publish())
            if new_revision:
                owner._lateral_yaw_revision += 1
        return begin()

    monkeypatch.setattr(rt, "_begin_follow_commit", refresh)
    rt._service_follow_wheels()
    assert driver.pairs == [(int(yaw), int(yaw))] * 2
    assert len(changed) == 1 and changed[0].valid(clock[0])
    assert not driver.stops and not owner.motor_io_lock.locked()


@pytest.mark.parametrize("fault", ["stop", "explicit_stop", "hazard", "uid", "identity",
                                    "expired", "feedback", "reverse", "zero", "park",
                                    "opposite_intent", "lower_limit", "policy", "receipt"])
def test_pivot_refresh_rechecks_stop_identity_expiry_and_wheel_direction(monkeypatch, fault):
    rt, owner, driver, clock, publish = pivot_writer(monkeypatch)
    begin = rt._begin_follow_commit
    changed = []

    def refresh():
        if not changed:
            changed.append(publish(hold_zero=fault == "zero", park_requested=fault == "park"))
            if fault == "stop":
                rt.backend.send_stop("newer-stop", mode="emergency", preserve_zero=True)
            elif fault == "explicit_stop":
                owner._explicit_stop_requested = True
            elif fault == "hazard":
                rt.hard_stop_check = lambda _: True
            elif fault == "uid":
                owner._follow_controller.active_target_id = 2
            elif fault == "identity":
                owner._detector_identity_lease = False
            elif fault == "expired":
                clock[0] += .151
                rt._steering_feedback = feedback(clock[0], -7., 7.)
            elif fault == "feedback":
                rt._steering_feedback = feedback(clock[0]-.151, -7., 7.)
            elif fault == "reverse":
                rt._steering_feedback = feedback(clock[0], 7., -7.)
            elif fault == "opposite_intent":
                publish(initial_correction_rpm=7.)
            elif fault == "lower_limit":
                publish(correction_limit_rpm=3.)
            elif fault == "policy":
                owner._lateral_turn_response_policy = (changed[0].sequence-1, False)
            elif fault == "receipt":
                rt.backend.send_targets(20, -20, "another-writer")
        return begin()

    monkeypatch.setattr(rt, "_begin_follow_commit", refresh)
    rt._service_follow_wheels()
    own_pairs = driver.pairs[2:] if fault == "receipt" else driver.pairs[1:]
    assert all(pair == (0, 0) for pair in own_pairs)
    if fault in {"stop", "hazard"}:
        assert driver.stops == [1] and len(driver.pairs) == 1
    assert not owner.motor_io_lock.locked()


def test_continuous_pivot_publications_share_three_attempt_budget(monkeypatch):
    rt, owner, driver, clock, publish = pivot_writer(monkeypatch)
    begin = rt._begin_follow_commit
    changed = []

    def refresh():
        if not rt._follow_commit_locked:
            clock[0] += .001
            changed.append(publish())
        return begin()

    monkeypatch.setattr(rt, "_begin_follow_commit", refresh)
    rt._service_follow_wheels()
    assert len(changed) == 3
    assert driver.pairs == [(-7, -7), (0, 0)]
    assert not driver.stops and not owner.motor_io_lock.locked()
