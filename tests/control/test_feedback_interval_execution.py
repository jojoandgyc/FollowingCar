"""Real history -> PI -> depth admission -> executor, with fake serial only.

No final motion/age/braking predicate is replaced. The pre-existing command is
acknowledged by the real Mssd backend before its completed receipt is recorded;
the new sample and all following wheel packets use the production control chain.
"""
import ast
import inspect
import textwrap
from dataclasses import replace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction, ControlDecision
from car_control_modular.controllers import FollowSafetyController
from car_control_modular.executed_speed_budget import record_completed_speed
from car_control_modular.mssd_motor import MssdMotorBackend, MssdMotorConfig
from test_depth_authority_250 import authority, advance, decide_commit, writer
from test_distance_tracking_response import setup
from test_execution_anchor_admission import AdmissionDriver
from test_lateral_zero_runtime import owner


def bind_production_readers(tracker):
    names = {"_braking_execution_bound_reader", "_braking_interval_speed_bound_reader",
             "_live_longitudinal_authority_reader"}
    tree = ast.parse(textwrap.dedent(inspect.getsource(runtime.PersonTracker.__init__)))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Attribute) and t.attr in names for t in node.targets)]
    assert len(assignments) == len(names)
    # Use the production lambdas, but not the constructor that starts devices.
    exec(compile(ast.Module(body=assignments, type_ignores=[]), __file__, "exec"),
         {"self": tracker, "PersonTracker": runtime.PersonTracker,
          "ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE": False,
          "DISTANCE_TARGET_MOTION_CONTROL_ENABLE": False})


@pytest.fixture
def execution_case(authority, monkeypatch):
    def build(enabled=True, *, history=True):
        a = authority
        a.controller = FollowSafetyController(replace(a.controller.cfg,
            target_distance_m=1.4, distance_pi_kp_per_sec=3.,
            distance_target_motion_control_enable=False,
            depth_longitudinal_sample_max_age_sec=.30,
            distance_pi_braking_stop_distance_m=1.1,
            distance_pi_observed_feedback_reserve=True,
            distance_pi_feedback_interval_deduplication=enabled,
            distance_approach_deceleration_m_s2=1.,
            distance_approach_response_delay_sec=.15,
            distance_pi_launch_request_rpm=180., distance_pi_launch_full_error_m=.5,
            visible_steering_pid_max_correction_rpm=10.))
        a.controller.active_target_id = 1
        a.controller._has_seen_person = True
        a.owner._follow_controller = a.controller
        monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC", .30)
        monkeypatch.setattr(runtime, "ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE", False)
        a.action, _ = writer(a)
        a.backend = MssdMotorBackend(MssdMotorConfig(
            port="unused-offline", slave_id=1, baudrate=115200, timeout=.1,
            lib_dir="unused-offline", max_target=200, percent_limit=100,
            left_sign=-1, right_sign=1, forward_target_sign=-1,
            m1_is_left_wheel=True, exit_parking_mode_on_arm=False,
            stop_mode="emergency", stop_zero_delay_sec=0., startup_parking_enabled=False))
        a.backend.driver = AdmissionDriver()
        a.action.backend = a.backend
        a.action.get_steering_feedback = lambda: a.feedback
        a.owner.motor_io_lock = a.backend.io_lock
        bind_production_readers(a.owner)

        a.clock.now = 99.99
        a.backend.send_targets(76, -76, "PRIOR_COMPLETED_FOLLOW")
        receipt = a.backend.last_speed_receipt
        assert receipt is not None and receipt.completed_at == a.clock.now
        if history:
            a.action._continuation_executed_speed_history = record_completed_speed(
                (), uid=1, applied=(76, 76), signs=(1, -1), receipt=receipt,
                previous_receipt=None, now=a.clock.now, packet_written=True)
        a.stamp = 100.
        a.clock.now = a.stamp + .122070536
        a.owner._last_vision_control_ts = a.clock.now-.01
        a.current = a.frame(2.225378219278882, rpm=76., stamp=a.stamp)
        a.current = replace(a.current, steering_feedback=replace(
            a.current.steering_feedback, timestamp=a.stamp-.000741398))
        a.feedback = a.current.steering_feedback
        a.initial_receipt = receipt
        a.enabled, a.has_history = enabled, history
        return a
    return build


def admit(a):
    before = tuple(a.backend.driver.pairs)
    decision, actions, accepted = decide_commit(a, a.current)
    assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    model = a.controller.last_distance_pid_result.pi_braking_assessment
    assert model is a.owner._depth30_linear_timing.braking_assessment
    assert model.feedback_interval_covered is (a.enabled and a.has_history)
    assert model.travel_bound_rpm == 76.
    assert model.sample_timestamp == a.stamp
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(a.stamp+.30)
    assert tuple(a.backend.driver.pairs) == before  # Admission itself does no I/O.
    a.grant = a.owner._depth30_linear_snapshot
    a.model = model
    a.timing = a.owner._depth30_linear_timing
    return decision


def terminal(a, age=.275473165, *, rpm=75., publish_cache=False):
    advance(a, a.stamp+age)
    a.feedback = replace(a.feedback, timestamp=a.clock.now-.01565,
                         left_forward_rpm=rpm, right_forward_rpm=rpm-1.)
    if publish_cache:
        # The real feedback publisher supplies both synchronous control reads
        # and the cache-only emergency-braking path.
        a.action._steering_feedback = a.feedback
    before = len(a.backend.driver.pairs)
    a.action._service_follow_wheels()
    return a.backend.driver.pairs[before:]


@pytest.mark.parametrize("enabled", [False, True])
def test_cap135_overlap_policy_changes_real_terminal_zero_to_bounded_forward(execution_case, enabled):
    a = execution_case(enabled)
    admit(a)
    a.action._service_follow_wheels()
    admitted_pair = a.backend.driver.pairs[-1]
    assert admitted_pair[0] > 0 and admitted_pair[1] < 0
    assert a.action._continuation_executed_speed_history[-1].receipt is a.backend.last_speed_receipt
    pairs = terminal(a)
    assert pairs
    if enabled:
        assert pairs[-1] == (68, -68)
        assert max(map(abs, pairs[-1])) < max(map(abs, admitted_pair))
    else:
        assert pairs[-1] == (0, 0)
    assert a.owner._depth30_linear_snapshot[3] == a.stamp
    assert a.owner._depth30_linear_timing.braking_assessment is a.model
    assert a.timing.depth_expires_at == pytest.approx(a.stamp+.30)


@pytest.mark.parametrize("enabled", [False, True])
def test_cache_visible_physical_motion_starts_old_brake_episode_but_not_new(execution_case, enabled):
    a = execution_case(enabled)
    admit(a)
    a.action._steering_feedback = a.feedback
    a.action._service_follow_wheels()
    assert not a.backend.driver.stops
    pairs = terminal(a, publish_cache=True)
    if enabled:
        assert pairs[-1] == (68, -68)
        assert not a.backend.driver.stops
    else:
        assert a.backend.driver.stops
        assert not any(left > 0 and right < 0 for left, right in pairs)
    assert a.owner._depth30_linear_timing.braking_assessment is a.model
    assert a.timing.depth_expires_at == pytest.approx(a.stamp+.30)


@pytest.mark.parametrize("fault", ["missing_record", "untracked_same_pair", "stop", "no_anchor"])
def test_incomplete_real_receipt_chain_does_not_enable_overlap_correction(execution_case, fault):
    a = execution_case()
    a.clock.now = a.stamp+.005
    if fault == "missing_record":
        a.action._continuation_executed_speed_history = ()
    elif fault == "untracked_same_pair":
        a.backend.send_targets(76, -76, "UNTRACKED_SAME_PAIR")
    elif fault == "stop":
        a.backend.send_stop("STOP_BEFORE_ASSESSMENT", mode="emergency")
    else:
        previous = a.backend.last_speed_receipt
        a.backend.send_targets(76, -76, "FIRST_RECORDED_AFTER_DEPTH")
        a.action._continuation_executed_speed_history = record_completed_speed(
            (), uid=1, applied=(76, 76), signs=(1, -1),
            receipt=a.backend.last_speed_receipt, previous_receipt=previous,
            now=a.clock.now, packet_written=True)
    a.clock.now = a.stamp+.122070536
    a.has_history = False  # The original sample interval is not covered.
    assert a.controller._braking_interval_speed_bound_reader(1, a.stamp, a.clock.now) is None
    admit(a)
    assert a.model.effective_feedback_reserve_sec == pytest.approx(.122811934)
    a.action._service_follow_wheels()
    assert terminal(a)[-1] == (0, 0)


@pytest.mark.parametrize("age", [.180001, .210, .299])
def test_completed_history_does_not_relax_first_measurement_admission(execution_case, age):
    a = execution_case()
    advance(a, a.stamp+age)
    a.current = a.frame(2.225378219278882, rpm=76., stamp=a.stamp)
    decision, actions, _ = decide_commit(a, a.current)
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    before = len(a.backend.driver.pairs)
    a.action._service_follow_wheels()
    assert not any(left > 0 and right < 0 for left, right in a.backend.driver.pairs[before:])


def test_feedback_read_crossing_original_deadline_still_sends_no_forward(execution_case):
    a = execution_case()
    admit(a)
    a.action._service_follow_wheels()
    advance(a, a.stamp+.299)
    reads = []

    def delayed_feedback():
        reads.append(a.clock.now)
        advance(a, a.stamp+.300001)
        return a.feedback

    a.action.get_steering_feedback = delayed_feedback
    before = len(a.backend.driver.pairs)
    a.action._service_follow_wheels()
    assert reads
    assert not any(left > 0 and right < 0 for left, right in a.backend.driver.pairs[before:])
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    assert a.timing.depth_expires_at == pytest.approx(a.stamp+.30)


def test_enabled_policy_without_recorded_history_uses_original_budget(execution_case):
    a = execution_case(history=False)
    assert a.backend.last_speed_receipt is a.initial_receipt  # Receipt alone is insufficient.
    assert a.controller._braking_interval_speed_bound_reader(1, a.stamp, a.clock.now) is None
    admit(a)
    assert a.model.effective_feedback_reserve_sec == pytest.approx(.122811934)
    a.action._service_follow_wheels()
    assert terminal(a)[-1] == (0, 0)
    assert not a.model.feedback_interval_covered  # A later write cannot retrofit old evidence.


@pytest.mark.parametrize("event", ["explicit_stop", "revoke", "ttl", "feedback_missing", "uid",
                                  "higher_write", "untracked_high_then_low", "broken_then_recorded"])
def test_corrected_budget_does_not_bypass_terminal_revocation(execution_case, event):
    a = execution_case()
    admit(a)
    a.action._service_follow_wheels()
    advance(a, a.stamp+.20)
    if event == "explicit_stop":
        a.owner._explicit_stop_requested = True
        a.backend.send_stop("EXPLICIT", mode="emergency")
    elif event == "revoke":
        a.owner._revoke_depth_linear_authority("identity_rejected")
    elif event == "uid":
        a.controller.active_target_id = 2
    elif event in {"higher_write", "untracked_high_then_low", "broken_then_recorded"}:
        a.backend.send_targets(110, -110, "UNTRACKED_HIGHER_WRITE")
        if event != "higher_write":
            a.backend.send_targets(40, -40, "UNTRACKED_LOWER_WRITE")
        if event == "broken_then_recorded":
            # Exercise production bookkeeping after a normal writer resumes;
            # an actual new low receipt cannot erase the untracked interval.
            previous = a.backend.last_speed_receipt
            history = a.action._continuation_executed_speed_history
            a.backend.send_targets(40, -40, "FOLLOW_AFTER_INTERRUPTION")
            a.action._continuation_executed_speed_history = record_completed_speed(
                history, uid=1, applied=(40, 40), signs=(1, -1),
                receipt=a.backend.last_speed_receipt, previous_receipt=previous,
                now=a.clock.now, packet_written=True)
            assert not a.action._continuation_executed_speed_history[0].response_anchor_valid
    elif event == "feedback_missing":
        a.feedback = None
    age = .300001 if event == "ttl" else .275473165
    if event == "feedback_missing":
        advance(a, a.stamp+age)
        before = len(a.backend.driver.pairs)
        a.action._service_follow_wheels()
        pairs = a.backend.driver.pairs[before:]
    else:
        pairs = terminal(a, age)
    assert not any(left > 0 and right < 0 for left, right in pairs)
    assert a.owner._fresh_depth_linear_snapshot(a.controller.active_target_id) is None
    assert a.timing.depth_expires_at == pytest.approx(a.stamp+.30)


def test_contiguous_normal_reduction_keeps_old_sample_budget_and_nonzero_packets(execution_case):
    a = execution_case()
    admit(a)
    a.action._service_follow_wheels()
    original_model, original_timing = a.model, a.timing
    advance(a, a.stamp+.20)
    a.feedback = replace(a.feedback, timestamp=a.clock.now,
                         left_forward_rpm=75., right_forward_rpm=74.)
    observation = a.frame(2.225378219278882, rpm=75., stamp=a.stamp+.001)
    actions, _ = a.owner._commit_depth_linear_decision(
        ControlDecision(actions=[ControlAction.forward(30, "normal_contraction")], reason="held"),
        observation, 1, is_fresh_depth=True)
    assert actions and all(x.speed_percent <= 30 for x in actions)
    assert a.owner._depth30_linear_snapshot[3] == a.stamp
    assert a.owner._depth30_linear_timing.braking_assessment is original_model
    before = len(a.backend.driver.pairs)
    a.action._service_follow_wheels()
    assert a.backend.driver.pairs[before:] == [(60, -60)]
    pairs = terminal(a, rpm=60.)
    assert pairs and all(0 < left <= 60 and -60 <= right < 0 for left, right in pairs)
    assert a.action._continuation_executed_speed_history[-1].receipt is a.backend.last_speed_receipt
    assert a.owner._depth30_linear_timing.braking_assessment is original_model
    assert original_timing.depth_expires_at == pytest.approx(a.stamp+.30)
