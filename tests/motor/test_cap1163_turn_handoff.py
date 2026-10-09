"""CAP1158–1163 execution continuity; fake clock/driver, never opens serial.

Recorded-prefix checks stop at the FIRST changed output. Recorded feedback
after that point is not the physical response to the new controller.
"""
import logging

import pytest

from car_control_modular.wheel_zero_cross import WheelZeroCrossGuard
from test_cap676_turn_continuity import runtime
from test_visible_wheel_continuity import feedback


DECEL = dict(allow_aligned_turn=True, aligned_deceleration_max_rpm=200)


def pending_guard(*, write_zero=True):
    guard = WheelZeroCrossGuard()
    assert guard.limit((-7, 7), feedback(10., 18, 28), 10., **DECEL)[0] == (0, 0)
    if write_zero:
        guard.note_sent((0, 0), 10.001)
    return guard


def first_aligned(guard, *, stamp=10.05):
    result = guard.limit((-5, 5), feedback(stamp, -2, 10), stamp + .01, **DECEL)
    assert result[0] == (0, 0)
    guard.note_sent((0, 0), stamp + .01)


@pytest.mark.parametrize("mirror", [False, True])
def test_recorded_prefix_releases_before_cap1163_without_depth(monkeypatch, caplog, mirror):
    rt, owner, driver, _, clock = runtime(monkeypatch)
    owner._vision_control_state = "target_visible_low_quality"
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
    assert rt._visible_wheel_control_active()

    def reflect(pair):
        return pair[::-1] if mirror else pair

    # Original output was zero at ALL five events. Later output/feedback is
    # deliberately not replayed: our fifth event is already counterfactual.
    trace = [
        (27050.954713845, 27050.921929920, (18, 28), 7),
        (27051.014209647, 27050.971768603, (11, 20), 7),
        (27051.067895603, 27051.022728130, (11, 11), 7),
        (27051.112918996, 27051.084124091, (-2, 10), 5),
        (27051.188360668, 27051.124156324, (-12, 9), 7),
    ]
    with caplog.at_level(logging.INFO):
        for index, (now, stamp, measured, yaw) in enumerate(trace):
            clock[0] = now
            rt.get_steering_feedback = lambda: feedback(stamp, *reflect(measured))
            left, right = reflect((-yaw, yaw))
            with owner.motor_io_lock:
                rt._send_follow_wheel_targets(left, -right, "FOLLOW20", visible_required=True)
            if index < len(trace) - 1:
                assert driver.pairs[-1] == (0, 0)
    left, right = reflect((-7, 7))
    assert driver.pairs == [(0, 0)] * 4 + [(left, -right)]
    assert "reason=cross_aligned_decelerating_turn_handoff" in caplog.text
    assert "depth_fresh=False" in caplog.text
    assert not driver.stops
    assert rt._visible_wheel_guard.pending_signs is None
    # This is a command-time improvement, NOT a measured vehicle-response claim.
    assert (27051.363115423 - trace[-1][0]) * 1000 == pytest.approx(174.754755)


@pytest.mark.parametrize("left", [12, 36])
@pytest.mark.parametrize("mirror", [False, True])
def test_opposing_cap1163_feedback_still_stops(monkeypatch, left, mirror):
    rt, owner, driver, _, clock = runtime(monkeypatch)
    owner._vision_control_state = "target_visible_low_quality"
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
    requested = (7, -7) if mirror else (-7, 7)
    measured = (1, left) if mirror else (left, 1)
    for step in range(4):
        clock[0] = 10. + step * .05
        rt.get_steering_feedback = lambda: feedback(clock[0] - .01, *measured)
        with owner.motor_io_lock:
            rt._send_follow_wheel_targets(requested[0], -requested[1], "FOLLOW20", visible_required=True)
        assert driver.pairs[-1] == (0, 0)


@pytest.mark.parametrize("case", [
    "duplicate", "out_of_order", "pre_zero", "gap", "stale", "future",
    "nan_stamp", "nan_speed", "untrusted", "left_error", "right_error",
    "opposing", "wheel_acceleration", "nonzero_base", "large_request",
    "large_feedback", "changed_direction", "opt_out", "aligned_opt_out",
])
def test_deceleration_release_requires_current_independent_safe_evidence(case):
    guard = pending_guard()
    first_aligned(guard)
    now, sample, request, options = 10.11, feedback(10.10, -12, 9), (-7, 7), dict(DECEL)
    if case == "duplicate": sample.timestamp = 10.05
    elif case == "out_of_order": sample.timestamp = 10.04
    elif case == "pre_zero": sample.timestamp = 9.999
    elif case == "gap": now, sample.timestamp = 10.31, 10.30
    elif case == "stale": sample.timestamp = 9.
    elif case == "future": sample.timestamp = 11.
    elif case == "nan_stamp": sample.timestamp = float("nan")
    elif case == "nan_speed": sample.left_forward_rpm = float("nan")
    elif case == "untrusted": sample.trustworthy = False
    elif case == "left_error": sample.left_error = 1
    elif case == "right_error": sample.right_error = 1
    elif case == "opposing": sample.left_forward_rpm = 12
    elif case == "wheel_acceleration": sample.left_forward_rpm = -6
    elif case == "nonzero_base": request = (-6, 8)
    elif case == "large_request": request = (-11, 11)
    elif case == "large_feedback": sample.left_forward_rpm = -201
    elif case == "changed_direction": request = (7, -7)
    elif case == "opt_out": options["aligned_deceleration_max_rpm"] = 0.
    elif case == "aligned_opt_out": options["allow_aligned_turn"] = False
    result = guard.limit(request, sample, now, **options)
    assert result[0] == (0, 0), (case, result)


def test_never_written_zero_cannot_release_decelerating_turn():
    guard = pending_guard(write_zero=False)
    # Do not note_sent: computation is not evidence of a physical zero write.
    for now, wheels in [(10.05, (-2, 10)), (10.10, (-12, 9)), (10.15, (-12, 9))]:
        assert guard.limit((-7, 7), feedback(now, *wheels), now, **DECEL)[0] == (0, 0)


def test_two_samples_before_the_actual_zero_do_not_count():
    guard = pending_guard(write_zero=False)
    guard.note_sent((0, 0), 10.10)
    for stamp in (10.05, 10.08):
        assert guard.limit((-7, 7), feedback(stamp, -12, 9), 10.12, **DECEL)[0] == (0, 0)
        assert guard.decelerating_count == 0


@pytest.mark.parametrize("source", ["commanded", "measured"])
def test_whole_car_reverse_provenance_cannot_use_decelerating_turn(source):
    guard = WheelZeroCrossGuard()
    if source == "commanded":
        guard.note_sent((-8, -8), 9.99)
    wheels = (-8, -8) if source == "measured" else (18, 28)
    assert guard.limit((-7, 7), feedback(10., *wheels), 10., **DECEL)[0] == (0, 0)
    guard.note_sent((0, 0), 10.001)
    first_aligned(guard)
    assert guard.limit((-7, 7), feedback(10.10, -12, 9), 10.11, **DECEL)[0] == (0, 0)


def test_upstream_zero_preserves_wait_provenance_not_release_permissions():
    guard = pending_guard()
    first_aligned(guard)
    original = (guard.pending_signs, guard.pending_wheels, guard.started,
                guard.zero_since, guard.feedback_stamp)
    guard.quiet_count = guard.aligned_count = guard.decelerating_count = 1
    guard.residual_count = guard.residual_turn_count = 1
    guard.resume_signs = (-1, 1)
    guard.residual_turn_signs = (-1, 1)
    guard.residual_turn_until = guard.residual_forward_until = 11.
    for now in (10.07, 10.09):
        assert guard.limit((0, 0), feedback(now, 0, 0), now,
                           preserve_wait_on_zero=True) == ((0, 0), "upstream_zero_wait")
        guard.note_sent((0, 0), now)
        assert (guard.pending_signs, guard.pending_wheels, guard.started,
                guard.zero_since, guard.feedback_stamp) == original
        assert guard.quiet_count == guard.aligned_count == guard.decelerating_count == 0
        assert guard.residual_count == guard.residual_turn_count == 0
        assert guard.resume_signs is guard.residual_turn_signs is None
        assert guard.residual_turn_until == guard.residual_forward_until == 0
    # The upstream zero did not count its encoder samples as confirmation.
    assert guard.limit((-7, 7), feedback(10.10, -12, 9), 10.11, **DECEL)[0] == (0, 0)
    assert guard.decelerating_count == 1


def test_explicit_zero_still_cancels_wait_and_cannot_replay_turn():
    guard = pending_guard()
    first_aligned(guard)
    assert guard.limit((0, 0), feedback(10.08), 10.08) == ((0, 0), "explicit_zero")
    assert guard.pending_signs is None and guard.zero_since is None
    assert guard.decelerating_count == 0


@pytest.mark.parametrize("loss", ["yaw", "uid", "revision", "search", "shutdown", "explicit_stop"])
def test_authority_lost_during_final_feedback_read_cannot_release(monkeypatch, loss):
    rt, owner, driver, _, clock = runtime(monkeypatch)
    owner._vision_control_state = "target_visible_low_quality"
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
    for now, wheels, yaw in [(10., (18, 28), 7), (10.05, (-2, 10), 5)]:
        clock[0] = now
        rt.get_steering_feedback = lambda: feedback(now, *wheels)
        with owner.motor_io_lock:
            rt._send_follow_wheel_targets(-yaw, -yaw, "FOLLOW20", visible_required=True)
    clock[0] = 10.10
    def read():
        if loss == "yaw": owner._has_fresh_lateral_yaw = lambda uid: False
        elif loss == "uid": owner._follow_controller.active_target_id = 2
        elif loss == "revision": owner._lateral_yaw_revision += 1
        elif loss == "search": owner.search_state = "searching"
        elif loss == "shutdown": owner._runtime_shutdown_requested = True
        elif loss == "explicit_stop": owner._explicit_stop_requested = True
        return feedback(clock[0], -12, 9)
    rt.get_steering_feedback = read
    with owner.motor_io_lock:
        rt._send_follow_wheel_targets(-7, -7, "FOLLOW20", visible_required=True)
    assert driver.pairs and all(pair == (0, 0) for pair in driver.pairs)


def test_danger_preempts_periodic_qualified_release(monkeypatch):
    from test_follow_wheel_periodic import setup_periodic
    rt, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    rt.config.follow_forward_loss_handoff_enable = True
    rt.config.follow_cross_brake_enable = True
    rt.config.motor_forward_max_target_rpm = 200
    owner._vision_control_state = "target_visible_low_quality"
    state[:] = [0., -7., 11., 11.]
    rt.get_steering_feedback = lambda: feedback(clock[0], 18, 28)
    rt._service_follow_wheels()
    clock[0] = 10.05
    state[1] = -5.
    rt.get_steering_feedback = lambda: feedback(clock[0], -2, 10)
    rt._service_follow_wheels()
    assert rt._visible_wheel_guard.decelerating_count == 1
    clock[0] = 10.10
    state[1] = -7.
    rt.get_steering_feedback = lambda: feedback(clock[0], -12, 9)
    reasons = []
    rt.hard_stop_check = lambda action: True
    rt.send_stop_with_brake_hold = reasons.append
    rt._service_follow_wheels()
    assert reasons == ["follow20_hard_stop"]
    assert all(pair == (0, 0) for pair in driver.pairs)
