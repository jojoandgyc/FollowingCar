"""CAP333 -> quiet -> centered target/new depth -> forward admission.

No camera, serial client or motor threads. Production parking and visual
admission functions; canonical PID result/commit are spies for this handoff.
"""
import queue
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "control"))
from test_lateral_zero_runtime import owner
from test_depth_drive_rpm import make_runtime
from test_search_handoff_execution import quiet, command
from car_control_modular.control_types import (
    ControlAction, ControlDecision, DepthTargetObservation, DistanceState,
    PersonTarget, SensorFrame,
)


@pytest.fixture(params=["normal", "emergency"])
def case(owner, monkeypatch, request):
    clock = [99.9]
    monkeypatch.setattr("request_0513_modular.time.monotonic", lambda: clock[0])
    rt, _, driver, symbols = make_runtime(max_rpm=200)
    rt.backend.config = replace(rt.backend.config, stop_mode=request.param,
                                parking_current_a=0. if request.param == "emergency" else 5.)
    rt.owner = owner
    owner.motor_io_lock = rt.backend.io_lock
    owner.action_queue = queue.Queue()
    owner._action_runtime = rt
    owner._follow_controller.search_state = "none"
    rt.get_steering_feedback = lambda: quiet(clock[0])
    rt.request_search_reacquire_brake(333, 99.8, "candidate_predictive_stop")
    rt._service_follow_wheels()
    for t in (100.40, 100.45, 100.50):
        clock[0] = t
        rt._service_follow_wheels()
    clock[0] = 100.55
    assert not rt.search_reacquire_brake_pending(capture_timestamp=100.53)
    assert owner._brake_hold_active  # settled is NOT yet permission to drive
    bbox = (200., 20., 440., 460.)
    obs = DepthTargetObservation(bbox, 1, 2, 361, 100.53)
    target = PersonTarget(bbox, 1, .9, 105600., obs)
    frame = SensorFrame(width=640, height=480, persons=[target], distance_m=3.136,
        distance_state=DistanceState(source="vision_depth", raw_distance_m=3.136,
            used_distance_m=3.136, sample_timestamp=100.52),
        capture_frame_id=361, capture_timestamp=100.53)
    decision = ControlDecision(actions=[ControlAction.forward(38, "fresh_distance_pi")],
                               reason="visual_pid_center_hold")
    commits = []
    owner._follow_controller._longitudinal_only_decision = lambda *a, **kw: decision
    def commit(new_decision, new_frame, uid, **kw):
        commits.append((new_decision, new_frame, uid))
        owner._depth30_linear_snapshot = ("forward", 38, uid, new_frame.distance_state.sample_timestamp)
        owner._current_forward_percent = 38
        owner._current_forward_allow_below_min = True
        return new_decision.actions, True
    owner._commit_depth_linear_decision = commit
    return owner, rt, driver, symbols, clock, frame, target, decision, commits


def refresh(c, **kw):
    o, _, _, _, _, frame, target, decision, _ = c
    args = dict(is_fresh_depth=True, target_steerable=True, low_quality_visible=False)
    args.update(kw)
    return o._refresh_visual_depth_linear_authority(frame, target, decision, **args)


def test_centered_fresh_target_exits_search_hold_and_reaches_forward_writer(case):
    o, rt, driver, s, _, frame, _, _, commits = case
    stop_count = len(driver.stops)
    o._depth30_linear_snapshot = ("forward", 99, 1, 99.8)  # may never be restored
    o.action_queue.put(s.rotate_left)
    assert refresh(case)
    assert not o._brake_hold_active
    assert o.action_queue.empty() and o.current_command is None
    assert len(commits) == 1
    assert o._depth30_linear_snapshot == ("forward", 38, 1, frame.distance_state.sample_timestamp)
    assert len(driver.stops) == stop_count  # producer release did no motor I/O
    rt.send_percent_drive(38)
    assert driver.pairs[-1] == (76, -76)


@pytest.mark.parametrize("kind", ["stale", "future", "pre_stop", "missing", "duplicate"])
def test_invalid_depth_cannot_unlock(case, kind):
    o, _, _, _, _, frame, _, _, commits = case
    stamp = {
        "stale": 99.8, "future": 100.6, "pre_stop": 99.89,
        "missing": None, "duplicate": 100.52,
    }[kind]
    frame = replace(frame, distance_state=replace(frame.distance_state, sample_timestamp=stamp))
    case = (*case[:5], frame, *case[6:])
    if kind == "duplicate":
        o._depth30_linear_sample_watermark = (1, 100.52)
    assert not refresh(case)
    assert o._brake_hold_active and not commits


@pytest.mark.parametrize("kind", ["wrong_uid", "weak", "no_detector", "duplicate_uid",
    "hazard", "front", "depth_brake", "depth_target", "safety_distance", "untrusted",
    "near", "no_demand", "explicit", "shutdown", "other_hold", "emergency",
    "searching", "new_brake_pending", "no_stop_proof", "wrong_episode_uid", "protected_queue"])
def test_other_holds_and_unqualified_evidence_remain_stopped(case, kind):
    o, rt, _, s, _, frame, target, decision, commits = case
    args = {}
    if kind == "wrong_uid": o._follow_controller.active_target_id = 2
    if kind == "weak": args["low_quality_visible"] = True
    if kind == "no_detector":
        target = replace(target, depth_observation=None)
    if kind == "duplicate_uid": frame.persons.append(target)
    if kind == "hazard": frame = replace(frame, hazard=replace(frame.hazard, active=True))
    if kind == "front": frame = replace(frame, obstacles=replace(frame.obstacles, front=True))
    if kind == "depth_brake": frame = replace(frame, distance_state=replace(frame.distance_state, brake_latched=True))
    if kind == "depth_target": frame = replace(frame, distance_state=replace(frame.distance_state, target_latched=True))
    if kind == "safety_distance": frame = replace(frame, distance_state=replace(frame.distance_state, safety_distance_m=.3))
    if kind == "untrusted": o._follow_controller._distance_longitudinally_untrusted = lambda f: True
    if kind == "near": frame = replace(frame, distance_m=1.2)
    if kind == "no_demand": decision = replace(decision, actions=[ControlAction.stop("hold")])
    if kind == "explicit": o._explicit_stop_requested = True
    if kind == "shutdown": o._runtime_shutdown_requested = True
    if kind == "other_hold": o._brake_hold_label = "safety_hold_front_ir"
    if kind == "emergency": rt.send_stop_with_brake_hold("hard_stop")
    if kind == "searching": o.search_state = "searching"
    if kind == "new_brake_pending": rt._search_reacquire_brake_request = rt._search_reacquire_brake_applied
    if kind == "no_stop_proof": rt._search_reacquire_settling = None
    if kind == "wrong_episode_uid": rt._search_reacquire_brake_uid = 2
    if kind == "protected_queue": o.action_queue.put(command(s, protected_stop=True))
    case = (*case[:5], frame, target, decision, commits)
    assert not refresh(case, **args)
    assert o._brake_hold_active and not commits


@pytest.mark.parametrize("kind", ["old_image", "pre_quiet_image", "stale_feedback", "moving",
                                  "hard_stop", "failed_safety"])
def test_rechecks_physical_release_conditions(case, kind):
    o, rt, _, _, clock, frame, target, _, commits = case
    if kind in {"old_image", "pre_quiet_image"}:
        stamp = 99.89 if kind == "old_image" else 100.46
        frame = replace(frame, capture_timestamp=stamp)
        target = replace(target, depth_observation=replace(target.depth_observation, capture_timestamp=stamp))
        frame = replace(frame, persons=[target])
        case = (*case[:5], frame, target, *case[7:])
    if kind == "stale_feedback": rt.get_steering_feedback = lambda: quiet(99.8)
    if kind == "moving": rt.get_steering_feedback = lambda: quiet(clock[0], left_forward_rpm=8)
    if kind == "hard_stop": rt.hard_stop_check = lambda a: True
    if kind == "failed_safety":
        def fail(a): raise RuntimeError("fake safety unavailable")
        rt.hard_stop_check = fail
    assert not refresh(case)
    assert o._brake_hold_active and not commits


def test_temporary_failed_measurement_then_fresh_depth_recovers_without_new_turn(case):
    assert not refresh(case, is_fresh_depth=False)
    assert case[0]._brake_hold_active
    assert refresh(case)
    assert not case[0]._brake_hold_active and len(case[-1]) == 1


def test_real_distance_controller_commit_and_motor_after_centered_search_stop(case, monkeypatch):
    from test_distance_tracking_response import setup as tracking_setup
    from test_distance_pi_controller import configured
    import request_0513_modular as runtime
    o, rt, driver, _, clock, frame, target, decision, _ = case
    _, controller, _ = configured(tracking_setup.__wrapped__(monkeypatch),
                                 distance_pi_launch_request_rpm=180.)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runtime, "FORWARD_MAX_RPM", 200)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT", 100)
    monkeypatch.setattr(runtime, "MOTOR_RS485_TARGET_MIN_INTERVAL_SEC", .05)
    o._follow_controller = controller
    del o._commit_depth_linear_decision  # use real runtime admission, not the spy
    frame = replace(frame, steering_feedback=quiet(clock[0]), distance_state=replace(
        frame.distance_state, source_detail="depth_multiregion", fusion_confidence=1.,
        filtered_distance_m=3.136, sample_age_sec=.03))
    assert o._refresh_visual_depth_linear_authority(frame, target, decision,
        is_fresh_depth=True, target_steerable=True, low_quality_visible=False)
    grant = o._depth30_linear_snapshot
    assert not o._brake_hold_active and grant[0] == "forward" and grant[1] > 0
    assert grant[3] == frame.distance_state.sample_timestamp
    assert o._depth30_linear_timing.depth_expires_at == pytest.approx(
        grant[3] + o._depth_linear_max_age_sec("forward"))
    rt.send_percent_drive(grant[1])
    assert driver.pairs[-1] == (grant[1]*2, -grant[1]*2)
    # Expiry still revokes forward; the recovery never stretches that lease.
    clock[0] = o._depth30_linear_timing.depth_expires_at + .01
    assert o._fresh_depth_linear_snapshot(1, now=clock[0]) is None
