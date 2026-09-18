"""CAP193: an old walking prior cannot cap NEW accepted range correction.

Real controller/PID/runtime methods; no camera, serial or motor constructors.
"""
from dataclasses import replace
import math

import pytest
import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction, ControlDecision
from car_control_modular.controllers import FollowSafetyController
from car_control_modular.longitudinal_feedforward import LongitudinalFeedforwardEvidence
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


def cap193(setup, owner, monkeypatch):
    clock, old, frame = setup
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock.now)
    for name, value in dict(
        FORWARD_MAX_RPM=200, DISTANCE_APPROACH_ENABLE=True,
        DISTANCE_APPROACH_NO_MATCHING_MAX_RPM=60,
        DISTANCE_MATCHING_BASE_MAX_RPM=80,
        ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT=100,
    ).items():
        monkeypatch.setattr(runtime, name, value)
    c = FollowSafetyController(replace(
        old.cfg, distance_approach_enable=True,
        distance_approach_no_matching_max_rpm=60,
        distance_matching_base_max_rpm=80, forward_max_rpm=200,
        distance_feedforward_wheel_circumference_m=.816814,
        distance_pid_output_rise_rpm_per_sec=240,
    ))
    c.active_target_id = 1
    c._has_seen_person = True
    stamp = clock.now-.020
    previous = stamp-.088519765
    c._distance_approach_sample_trusted = True
    c._distance_pid_sample_timestamp = previous
    c._braking_rate_source = "raw_depth_window"
    c._update_distance_pid(2.462568, now=clock.now-.09)
    c.accept_longitudinal_limit(previous, 24.)
    c._longitudinal_bridge.remember(LongitudinalFeedforwardEvidence(
        "ready", eligible=True, target_id=1, sample_timestamp=previous,
        target_rpm=40.), 2.4242)
    c._longitudinal_motion_stamp = previous
    current = frame(2.4242, rpm=90., stamp=stamp)
    current = replace(current, distance_state=replace(current.distance_state, raw_distance_m=2.3904))
    assert c._bridge_longitudinal_motion(current, 1, clock.now, stamp, 0., 0., "warming_up")
    c._distance_pid_sample_timestamp = stamp
    c._braking_range_rate = -.429227
    result = c._update_distance_pid(current.distance_m, now=clock.now)
    owner._follow_controller = c
    owner._depth30_linear_snapshot = ("forward", 12, 1, previous)
    return clock, c, frame, current, result


def commit(owner, frame, result, percent=None):
    if percent is None:
        percent = result.output_rpm // 2
    return owner._commit_depth_linear_decision(
        ControlDecision(actions=[ControlAction.forward(percent, "longitudinal_distance_pid")],
                        reason="longitudinal_distance_pid"),
        frame, 1, is_fresh_depth=True,
    )


def test_cap193_new_distance_can_accelerate_past_old_24_without_increasing_prior(setup, owner, monkeypatch, caplog):
    clock, c, _, f, r = cap193(setup, owner, monkeypatch)
    assert r.tracking_base_rpm == 24
    assert r.output_rpm == 45  # 24 + 240RPM/s * 88.5ms, NOT instantly 68RPM.
    assert r.output_rpm < r.unslewed_output_rpm
    assert r.output_rpm <= r.approach_cap_rpm
    before = c._distance_pid._last_ts, c._distance_pid._integral_m_s
    actions, accepted = commit(owner, f, r)
    assert accepted and actions[0].speed_percent == 22  # Quantized 44RPM, formerly24.
    assert (c._distance_pid._last_ts, c._distance_pid._integral_m_s) == before
    assert owner._depth30_linear_timing.depth_expires_at == pytest.approx(f.distance_state.sample_timestamp+.18)
    assert owner._depth30_linear_timing.feedforward_timestamp == c._longitudinal_bridge.origin.sample_timestamp
    assert "cap_scope=prior_only" in caplog.text


def test_next_fresh_range_can_rise_but_old_prior_cannot_grow_back(setup, owner, monkeypatch):
    clock, c, frame, f, r = cap193(setup, owner, monkeypatch)
    commit(owner, f, r)
    origin = c._longitudinal_bridge.origin
    clock.now += .05
    f = frame(2.42, rpm=80., stamp=clock.now-.02)
    assert c._bridge_longitudinal_motion(f, 1, clock.now, f.distance_state.sample_timestamp, 0., 0., "warming_up")
    assert c._longitudinal_motion_evidence.target_rpm <= 24  # Larger total isn't faster human evidence.
    assert c._longitudinal_bridge.origin is origin
    c._distance_pid_sample_timestamp = f.distance_state.sample_timestamp
    r2 = c._update_distance_pid(f.distance_m, now=clock.now)
    assert 45 < r2.output_rpm <= 44+240*.05+1
    actions, accepted = commit(owner, f, r2)
    assert accepted and actions[0].speed_percent > 22


def test_new_measured_human_speed_replaces_prior_component_bound(setup, owner, monkeypatch):
    clock, c, _, _, _ = cap193(setup, owner, monkeypatch)
    bridge = c._longitudinal_bridge
    assert bridge._last_bridge_rpm == 24
    bridge.remember(LongitudinalFeedforwardEvidence(
        "ready", eligible=True, target_id=1, sample_timestamp=clock.now,
        target_rpm=60.), 2.4)
    evidence = bridge.evaluate(
        now=clock.now+.05, stamp=clock.now+.04, uid=1, distance=2.4,
        yaw=0., bearing=0., previous_output=80., baseline=20.,
        fall_rate_rpm_per_sec=80., near_distance=1.7, prior_max_age_sec=.35,
    )
    assert evidence.target_rpm == pytest.approx(56.)
    assert bridge.origin.sample_timestamp == clock.now


def test_legacy_pid_keeps_its_original_no_acceleration_bridge(setup):
    clock, c, frame = setup
    c._longitudinal_bridge_output_cap = 24
    c._tracking_base_rpm = lambda distance, now: 24.
    c._distance_pid_sample_timestamp = clock.now
    result = c._update_distance_pid(2.4242, now=clock.now)
    assert result.approach_mode == "legacy_pid" and result.output_rpm <= 24
    assert c.fresh_bridge_forward_percent(frame(2.4242), clock.now, clock.now, 24.) is None


@pytest.mark.parametrize("case", ["hazard", "obstacle", "brake", "safety", "held", "jump", "uid", "pid_mismatch", "stale"])
def test_new_range_component_rejects_invalid_evidence(setup, owner, monkeypatch, case):
    clock, c, _, f, _ = cap193(setup, owner, monkeypatch)
    stamp = f.distance_state.sample_timestamp
    if case == "hazard": f = replace(f, hazard=replace(f.hazard, active=True))
    if case == "obstacle": f = replace(f, obstacles=replace(f.obstacles, front=True))
    if case == "brake": f = replace(f, distance_state=replace(f.distance_state, brake_latched=True))
    if case == "safety": f = replace(f, distance_state=replace(f.distance_state, safety_distance_m=.4))
    if case == "held": f = replace(f, distance_state=replace(f.distance_state, raw_distance_m=None))
    if case == "jump": f = replace(f, distance_state=replace(f.distance_state, source_detail="distance_jump_pending"))
    if case == "uid": c.active_target_id = 2
    if case == "pid_mismatch": c._distance_pid_last_sample_timestamp = stamp-.01
    if case == "stale": clock.now = stamp+.181
    assert c.fresh_bridge_forward_percent(f, stamp, clock.now, 24.) == 0


def test_expired_prior_at_commit_does_not_reapply_old_total_cap(setup, owner, monkeypatch):
    clock, c, _, f, r = cap193(setup, owner, monkeypatch)
    c._longitudinal_bridge.origin = replace(c._longitudinal_bridge.origin, sample_timestamp=clock.now-.351)
    # Simulate an ordinary distance cap below the independently calculated
    # 45RPM range budget: both initial cap and fallback must pass that budget.
    monkeypatch.setattr(runtime.PersonTracker, "_depth_longitudinal_cap_percent", staticmethod(lambda _: 12))
    actions, accepted = commit(owner, f, r)
    assert accepted and actions[0].speed_percent == 22
    assert owner._depth30_linear_timing.feedforward_timestamp is None


def test_live_bridge_caches_independent_fallback_not_ordinary_cap(setup, owner, monkeypatch):
    clock, c, _, f, r = cap193(setup, owner, monkeypatch)
    origin = f.distance_state.sample_timestamp-.31
    c._longitudinal_bridge.origin = replace(c._longitudinal_bridge.origin, sample_timestamp=origin)
    monkeypatch.setattr(runtime.PersonTracker, "_depth_longitudinal_cap_percent", staticmethod(lambda _: 12))
    actions, _ = commit(owner, f, r)
    assert actions[0].speed_percent == 22
    assert owner._depth30_linear_timing.distance_only_percent == 22
    clock.now = origin+.351  # FF expired, physical Depth not expired.
    assert owner._fresh_depth_linear_snapshot(1)[1] == 22
    clock.now = f.distance_state.sample_timestamp+.181
    assert owner._fresh_depth_linear_snapshot(1) is None


def test_small_remaining_prior_cannot_erase_independent_distance_budget(setup, owner, monkeypatch):
    cap193(setup, owner, monkeypatch)
    monkeypatch.setattr(runtime.PersonTracker, "_depth_longitudinal_cap_percent", staticmethod(lambda _: 12))
    actions = runtime.PersonTracker._cap_depth_longitudinal_actions(
        [ControlAction.forward(30, "distance_only_slew")], distance_m=2.4242,
        tracking_base_rpm=1., distance_control_percent=30,
    )
    assert actions[0].speed_percent == 30  # 60RPM budget survives 1RPM old FF.
    # A budget is not a speed floor; preserve a new explicit zero request.
    assert runtime.PersonTracker._cap_depth_longitudinal_actions(
        [ControlAction.forward(0, "stop")], distance_m=2.4242,
        tracking_base_rpm=1., distance_control_percent=30,
    )[0].speed_percent == 0


@pytest.mark.parametrize("old", ["expired", "revoked"])
def test_no_live_grant_uses_bounded_new_range_not_bridge(setup, owner, monkeypatch, old):
    clock, c, _, f, r = cap193(setup, owner, monkeypatch)
    owner._depth30_linear_snapshot = None if old == "revoked" else ("forward", 12, 1, clock.now-.181)
    f = replace(f, steering_feedback=replace(f.steering_feedback, left_forward_rpm=0., right_forward_rpm=0.))
    actions, accepted = commit(owner, f, r)
    assert accepted and 0 < actions[0].speed_percent <= 6  # <=12RPM measured restart, not45.
    assert owner._depth30_linear_timing.feedforward_timestamp is None
    assert owner._depth30_linear_timing.depth_expires_at == pytest.approx(f.distance_state.sample_timestamp+.18)


@pytest.mark.parametrize("case", ["stop", "search", "backward", "parking", "shutdown"])
def test_new_components_cannot_bypass_runtime_revocation(setup, owner, monkeypatch, case):
    _, _, _, f, r = cap193(setup, owner, monkeypatch)
    if case == "stop": owner._explicit_stop_requested = True
    if case == "search": owner.search_state = "searching"
    if case == "backward": owner._depth30_linear_snapshot = ("backward", 12, 1, f.distance_state.sample_timestamp-.03)
    if case == "parking": owner._brake_hold_active = True
    if case == "shutdown": owner._runtime_shutdown_requested = True
    actions, _ = commit(owner, f, r)
    assert all(a.speed_percent == 0 for a in actions)
    assert owner._fresh_depth_linear_snapshot(1) is None


def test_zero_and_same_or_older_samples_never_accelerate_or_renew(setup, owner, monkeypatch):
    _, _, _, f, r = cap193(setup, owner, monkeypatch)
    commit(owner, f, r)
    timing = owner._depth30_linear_timing
    assert commit(owner, f, r) == ([], False)
    older = replace(f, distance_state=replace(f.distance_state, sample_timestamp=f.distance_state.sample_timestamp-.01))
    assert commit(owner, older, r) == ([], False)
    assert owner._depth30_linear_timing is timing
    newer = replace(f, distance_state=replace(f.distance_state, sample_timestamp=f.distance_state.sample_timestamp+.01))
    actions, _ = commit(owner, newer, r, percent=0)
    assert actions[0].speed_percent == 0
    assert owner._fresh_depth_linear_snapshot(1) is None


def test_reduced_live_grant_withdraws_only_old_base_and_keeps_new_distance(setup, owner, monkeypatch):
    clock, c, frame, _, _ = cap193(setup, owner, monkeypatch)
    clock.now += .04
    f = frame(2., rpm=20., stamp=clock.now-.02)
    c._distance_pid_sample_timestamp = f.distance_state.sample_timestamp
    c._distance_pid._last_output_rpm = 100.
    c._distance_pid.last_result = None  # This test isolates component arithmetic, not a range jump.
    c._braking_range_rate = 0.
    c._longitudinal_motion_evidence = replace(c._longitudinal_motion_evidence, sample_timestamp=f.distance_state.sample_timestamp)
    r = c._update_distance_pid(2., now=clock.now)
    budget = c.fresh_bridge_forward_percent(f, f.distance_state.sample_timestamp, clock.now, 4.)
    assert r.tracking_base_rpm == 24
    assert budget*2 <= r.output_rpm-(24-4)+1
    assert budget*2 >= math.floor(r.distance_only_rpm)-1


def test_runtime_prior_reduction_cannot_be_revived_by_next_distance_approval(setup, owner, monkeypatch):
    clock, c, frame, f, r = cap193(setup, owner, monkeypatch)
    owner._depth30_linear_snapshot = ("forward", 2, 1, f.distance_state.sample_timestamp-.03)
    actions, accepted = commit(owner, f, r)
    assert accepted and actions[0].speed_percent == 22  # Fresh distance can still approve44.
    assert c._longitudinal_bridge._last_bridge_rpm == 4.
    assert c._tracking_base_rpm(f.distance_m, clock.now) == 4.
    clock.now += .05
    f = frame(2.42, rpm=80., stamp=clock.now-.02)
    assert c._bridge_longitudinal_motion(f, 1, clock.now, f.distance_state.sample_timestamp, 0., 0., "warming_up")
    assert c._longitudinal_motion_evidence.target_rpm <= 4.
    c._distance_pid_sample_timestamp = f.distance_state.sample_timestamp
    next_result = c._update_distance_pid(f.distance_m, now=clock.now)
    assert next_result.tracking_base_rpm <= 4.
