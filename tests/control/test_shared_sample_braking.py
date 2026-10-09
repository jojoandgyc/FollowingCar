"""Shared fresh-PI/final-grant safety assessment; CPU only, no hardware."""
from dataclasses import replace
import ast
import inspect
import math
import textwrap
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.sample_braking import SampleBrakingAssessment
from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.longitudinal_approach import RawDepthMotionEvidence
from car_control_modular.executed_speed_budget import (
    executed_speed_bound_rpm, observe_completed_speed_response, record_completed_speed,
)
from car_control_modular.mssd_motor import MotorSpeedReceipt
from test_depth_authority_250 import authority, advance, decide_commit, seed
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


def assessment(distance=2.5, *, bound=40., outer=30., target=0., age=.06):
    return SampleBrakingAssessment(1, 100., 100.+age, distance, bound, outer,
                                   100.+age-.02, target, 1.2, .816814, .7, .2, 200.)


@pytest.mark.parametrize("bound,outer", [(0., 0.), (20., 10.), (60., 50.), (90., 85.)])
@pytest.mark.parametrize("distance", [1.5, 1.8, 2.2, 3.0, 4.])
@pytest.mark.parametrize("target", [0., -.3])
def test_same_clock_final_admission_does_not_contradict_positive_fresh_budget(bound, outer, distance, target):
    evidence = assessment(distance, bound=bound, outer=outer, target=target)
    fresh = evidence.budget(evidence.checked_at, outer)
    if fresh.cap_rpm < 2:
        return
    approved = 2*math.floor(fresh.cap_rpm/2)
    admitted = evidence.budget(evidence.checked_at, outer, authorized_rpm=approved,
                               completed_rpm=bound)
    assert admitted.cap_rpm == pytest.approx(approved)
    assert admitted.required_stop_m <= admitted.margin_m+1e-9


def test_unsent_candidate_is_not_retroactive_motion_or_a_zero_start_bound():
    value = assessment(2.2, bound=0., outer=0., age=.15)
    fresh = value.budget(value.checked_at, 0.)
    assert fresh.cap_rpm > 30
    # At the same physical clock, a permitted launch remains permitted even
    # though no motion was measured or completed before Depth capture.
    admitted = value.budget(value.checked_at, 0., authorized_rpm=30.)
    assert admitted.cap_rpm == 30.
    later = value.budget(100.24, 20., authorized_rpm=30., completed_rpm=30.)
    assert later.margin_m < admitted.margin_m
    assert later.cap_rpm <= 30.


def test_fixed_grant_budget_only_shrinks_with_age_despite_fresh_slower_feedback():
    value = assessment(2.5, bound=40., outer=30.)
    cap = math.floor(value.budget(value.checked_at, 30.).cap_rpm)
    samples = [value.budget(now, max(0., 30.-i*2), authorized_rpm=cap, completed_rpm=cap)
               for i, now in enumerate((100.06, 100.10, 100.15, 100.20, 100.249))]
    assert all(a.margin_m >= b.margin_m for a, b in zip(samples, samples[1:]))
    assert all(a.cap_rpm >= b.cap_rpm for a, b in zip(samples, samples[1:]))
    assert value.budget(100.300001, 0., authorized_rpm=cap).cap_rpm == 0


def test_negative_target_and_new_higher_write_are_not_laundered_by_a_small_request():
    toward = assessment(2.1417356917, bound=74., outer=49., target=-.2727073143, age=.0421)
    assert toward.budget(toward.checked_at, 49.).reason == "shared_braking_momentum"
    value = assessment(2.5)
    assert value.budget(value.checked_at, 30., authorized_rpm=30., completed_rpm=80.).cap_rpm == 0
    assert not value.valid_for(2, 100.)
    assert not value.valid_for(1, 100.001)


def test_future_outer_wheel_allowance_accepts_a_legal_turn_not_an_unbudgeted_write():
    evidence = replace(assessment(2.8, bound=20., outer=20.), outer_allowance_rpm=10.)
    budget = evidence.budget(evidence.checked_at, 20.)
    approved = 2*math.floor(budget.cap_rpm/2)
    assert approved > 20
    legal = evidence.budget(evidence.checked_at+.02, 20., authorized_rpm=approved,
                            completed_rpm=approved+10.)
    assert legal.cap_rpm > 0
    assert legal.reason == "shared_braking_cap"
    excessive = evidence.budget(evidence.checked_at+.02, 20., authorized_rpm=approved,
                                completed_rpm=approved+10.1)
    assert excessive.cap_rpm == 0
    assert excessive.reason == "shared_braking_new_higher_write"
    # A not-yet-executed steering request cannot invent observed momentum.
    assert evidence.outer_rpm == 20.
    assert budget.required_stop_m == pytest.approx(
        assessment(2.8, bound=20., outer=20.).budget(evidence.checked_at, 20.).required_stop_m)


def test_cap233_verified_low_response_and_stale_high_history_have_distinct_budgets():
    old = assessment(1.7786, bound=74., outer=19., target=-.3950538937, age=.0307)
    responded = replace(old, travel_bound_rpm=19.)
    assert old.budget(old.checked_at, 19.).cap_rpm == 0
    assert responded.budget(responded.checked_at, 19.).cap_rpm >= 10
    # Only external response proof may produce the new lower-history sample;
    # a small candidate alone does not erase the old high command.
    assert old.budget(old.checked_at, 19., authorized_rpm=10.).cap_rpm == 0


def test_cap233_new_sample_uses_feedback_retirement_not_a_hypothetical_small_request():
    high = MotorSpeedReceipt(1, 74, -66, 99.6)
    low = MotorSpeedReceipt(2, 19, -14, 99.9)
    history = record_completed_speed(
        (), uid=1, applied=(74, 66), signs=(1, -1), receipt=high,
        previous_receipt=None, now=99.6, packet_written=True)
    history = record_completed_speed(
        history, uid=1, applied=(19, 14), signs=(1, -1), receipt=low,
        previous_receipt=high, now=99.9, packet_written=True)
    before = replace(assessment(1.7786, bound=74., outer=19., target=-.3950538937,
                               age=.0307), outer_allowance_rpm=10.)
    assert before.budget(before.checked_at, 19.).cap_rpm == 0
    for stamp, left, right in [(99.95, 19., 14.), (100.005, 19., 13.)]:
        history = observe_completed_speed_response(
            history, feedback=SimpleNamespace(timestamp=stamp, trustworthy=True,
                left_forward_rpm=left, right_forward_rpm=right), now=stamp+.005, receipt=low)
    bound = executed_speed_bound_rpm(history, uid=1, now=before.checked_at)
    assert bound == 19.
    after = replace(before, travel_bound_rpm=bound)
    assert after.budget(after.checked_at, 19.).cap_rpm == pytest.approx(12.5517593604)
    assert after.budget(after.checked_at, 19., authorized_rpm=10., completed_rpm=bound).cap_rpm == 10.
    # Retirement supplies a new physical assessment only. The older frozen
    # grant is not made cheaper by this later low-speed report.
    assert before.budget(before.checked_at, 19., authorized_rpm=10., completed_rpm=bound).cap_rpm == 0


def test_pi_uses_shared_negative_motion_assessment_and_preserves_exact_object():
    c = DistancePiController(DistancePiConfig(
        kp_per_sec=3., deceleration_m_s2=.7, physical_ttl_sec=.25,
        stationary_stop_preview_enabled=True))
    shared = assessment(2.1417356917, bound=74., outer=49., target=-.2727073143, age=.0421)
    rate = -.2727073143-46.*.816814/60.
    result = c.update(2.254, 1.4, sample_timestamp=100., execution_now=shared.checked_at,
                      deadband_m=.03, max_output_rpm=200., ego_forward_rpm=46.,
                      preview_outer_forward_rpm=49., preview_feedback_timestamp=shared.feedback_timestamp,
                      raw_distance_m=shared.distance_m, raw_closure_valid=True,
                      range_rate_m_s=rate, raw_motion_evidence=RawDepthMotionEvidence(
                          100., rate, -.2727073143, .13, 2), braking_assessment=shared)
    assert result.braking_assessment is shared
    assert result.output_rpm == 0
    assert result.stationary_preview_status == "momentum_brake"


@pytest.mark.parametrize("relative, motion, required", [
    (False, False, True), (True, False, True), (True, True, True), (False, True, False),
])
def test_real_execution_budget_binding_remains_enabled_without_target_motion(relative, motion, required):
    calls = []
    owner = SimpleNamespace(_follow_controller=SimpleNamespace(),
        _action_runtime=SimpleNamespace(continuation_executed_speed_bound_rpm=
            lambda uid, now: calls.append((uid, now)) or 37.))
    tree = ast.parse(textwrap.dedent(inspect.getsource(runtime.PersonTracker.__init__)))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "_braking_execution_bound_reader"
                for target in node.targets)]
    assert len(assignments) == 1
    exec(compile(ast.Module(body=assignments, type_ignores=[]), __file__, "exec"),
         {"self": owner, "ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE": relative,
          "DISTANCE_TARGET_MOTION_CONTROL_ENABLE": motion})
    reader = owner._follow_controller._braking_execution_bound_reader
    assert callable(reader) is required
    if required:
        assert reader(8, 102.) == 37.
        assert calls == [(8, 102.)]


@pytest.fixture
def shared_runtime(authority, monkeypatch):
    a = authority
    a.controller.cfg = replace(a.controller.cfg, distance_pi_stationary_stop_preview_enabled=True)
    pi = a.controller._distance_pid._distance_pi
    pi.config = replace(pi.config, stationary_stop_preview_enabled=True, deceleration_m_s2=.7)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE", True)
    history = [20.]
    a.owner._action_runtime.continuation_executed_speed_bound_rpm = lambda uid, now: history[0]
    tree = ast.parse(textwrap.dedent(inspect.getsource(runtime.PersonTracker.__init__)))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Attribute)
                           and target.attr == "_braking_execution_bound_reader"
                           for target in node.targets)]
    assert len(assignments) == 1
    exec(compile(ast.Module(body=assignments, type_ignores=[]), __file__, "exec"),
         {"self": a.owner, "ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE": True})
    stamp, _ = seed(a, distance=2.5, rpm=20.)
    advance(a, stamp+.1)
    current = a.frame(2.5, rpm=20.)
    _, actions, accepted = decide_commit(a, current)
    assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    shared = a.controller.last_distance_pid_result.pi_braking_assessment
    assert isinstance(shared, SampleBrakingAssessment)
    assert a.owner._depth30_linear_timing.braking_assessment is shared
    return a, history, shared


def test_runtime_carries_exact_assessment_into_new_depth_grant(shared_runtime):
    a, _, shared = shared_runtime
    timing = a.owner._depth30_linear_timing
    linear = a.owner._depth30_linear_snapshot
    assert shared.valid_for(linear[2], linear[3])
    approved = a.owner._fresh_depth_linear_snapshot(1)
    assert approved is not None and approved[3] == shared.sample_timestamp
    assert a.owner._depth30_linear_timing is timing
    assert timing.depth_expires_at == pytest.approx(shared.sample_timestamp+.25)


@pytest.mark.parametrize("excess", [0., .1])
def test_runtime_allows_reserved_completed_outer_wheel_not_an_unbudgeted_turn(shared_runtime, excess):
    a, history, shared = shared_runtime
    linear = a.owner._depth30_linear_snapshot
    allowance = shared.outer_allowance_rpm
    assert allowance == a.controller.cfg.visible_steering_pid_max_correction_rpm
    history[0] = max(shared.travel_bound_rpm, linear[1]*2.+allowance)+excess
    current = a.owner._fresh_depth_linear_snapshot(1)
    assert (current is not None) is (excess == 0.)


@pytest.mark.parametrize("condition", ["higher_write", "wrong_assessment", "stale_feedback", "expired", "hazard"])
def test_shared_runtime_cannot_override_final_physical_or_stop_veto(shared_runtime, condition):
    a, history, shared = shared_runtime
    if condition == "higher_write":
        history[0] = 190.
    elif condition == "wrong_assessment":
        a.owner._depth30_linear_timing = replace(
            a.owner._depth30_linear_timing, braking_assessment=replace(shared, uid=2))
    elif condition == "stale_feedback":
        a.feedback = replace(a.feedback, timestamp=a.clock.now-.151)
    elif condition == "expired":
        advance(a, shared.sample_timestamp+.250001)
    else:
        a.owner._explicit_stop_requested = True
    assert a.owner._fresh_depth_linear_snapshot(1) is None
