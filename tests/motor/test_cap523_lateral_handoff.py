"""CAP523: no invented forward authority; do not stop the aligned outer wheel."""
import pytest

from car_control_modular.forward_loss_handoff import ForwardLossHandoff
from test_visible_wheel_continuity import feedback, visible_runtime
from test_follow_wheel_periodic import setup_periodic


def armed():
    handoff = ForwardLossHandoff()
    handoff.note_sent((84, 64), 9.9)
    assert handoff.limit((8, -8), feedback(10, 72, 52), 10,
                         allow_outer_deceleration=True)[0] == (0, 0)
    handoff.note_sent((0, 0), 10.01)
    return handoff


@pytest.mark.parametrize("sign", [1, -1])
def test_only_inner_needs_standstill_after_actual_zero(sign):
    h = armed()
    pair = (8*sign, -8*sign)
    measured = (40, 0) if sign > 0 else (0, 40)
    assert h.limit(pair, feedback(10.05, *measured), 10.05,
                   allow_outer_deceleration=True)[0] == (0, 0)
    # A cached encoder value cannot count as the second quiet sample.
    assert h.limit(pair, feedback(10.05, *measured), 10.06,
                   allow_outer_deceleration=True)[0] == (0, 0)
    measured = (32, 0) if sign > 0 else (0, 32)
    assert h.limit(pair, feedback(10.1, *measured), 10.1,
                   allow_outer_deceleration=True) == (pair, "forward_loss_inner_stopped_turn_ready")
    assert sum(pair) == 0


@pytest.mark.parametrize("case", ["opt_out", "no_zero", "pre_zero", "inner_fast",
    "both_reverse", "fault", "accelerating", "stale", "untrusted", "overspeed", "large_request"])
def test_shortcut_never_bypasses_reversal_feedback_requirements(case):
    h = armed()
    pair = (11, -11) if case == "large_request" else (8, -8)
    if case == "no_zero": h.zero_sent_at = None
    for index, now in enumerate((10.05, 10.1)):
        fb = feedback(now, 40, 0)
        if case == "pre_zero": fb.timestamp = 10.005
        if case == "inner_fast": fb.right_forward_rpm = 2
        if case == "both_reverse": fb.left_forward_rpm = fb.right_forward_rpm = -20
        if case == "fault": fb.right_error = 1
        if case == "accelerating": fb.left_forward_rpm = 30 + index*10
        if case == "stale": fb.timestamp = now-.2
        if case == "untrusted": fb.trustworthy = False
        if case == "overspeed": fb.left_forward_rpm = 201
        assert h.limit(pair, fb, now, allow_outer_deceleration=case != "opt_out")[0] == (0, 0)


@pytest.mark.parametrize("sign", [1, -1])
def test_real_writer_emits_symmetric_turn_without_forward_grant(monkeypatch, sign):
    r, owner, driver, _, clock = visible_runtime(monkeypatch)
    r.config.follow_forward_loss_handoff_enable = True
    r.config.follow_turn_residual_max_rpm = 0
    r._send_follow_wheel_targets(24, -24, "CAP519")
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
    clock[0] = 10.05
    r.get_steering_feedback = lambda: feedback(clock[0], 72, 52)
    r._send_follow_wheel_targets(8*sign, 8*sign, "CAP523")
    assert driver.pairs[-1] == (0, 0)
    assert r._forward_loss_handoff.zero_sent_at == clock[0]
    for now, outer in ((10.1, 40), (10.15, 32)):
        clock[0] = now
        measured = (outer, 0) if sign > 0 else (0, outer)
        r.get_steering_feedback = lambda: feedback(clock[0], *measured)
        r._send_follow_wheel_targets(8*sign, 8*sign, "CAP523_CONTINUE")
    # Serial right-wheel sign is inverted: physical +8/-8 (or -8/+8).
    assert driver.pairs[-1] == (8*sign, 8*sign)
    assert not driver.stops


@pytest.mark.parametrize("case", ["low_quality", "yaw_expired", "revision_changed"])
def test_real_writer_does_not_reuse_confirmation_after_authority_change(monkeypatch, case):
    r, owner, driver, _, clock = visible_runtime(monkeypatch)
    r.config.follow_forward_loss_handoff_enable = True
    r._send_follow_wheel_targets(24, -24, "CAP519")
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
    for now, outer, inner in ((10.05, 72, 52), (10.1, 40, 0)):
        clock[0] = now
        r.get_steering_feedback = lambda: feedback(clock[0], outer, inner)
        r._send_follow_wheel_targets(8, 8, "BRAKING")
    clock[0] = 10.15
    def read():
        if case == "revision_changed": owner._lateral_yaw_revision += 1
        return feedback(clock[0], 32, 0)
    r.get_steering_feedback = read
    if case == "low_quality": owner._vision_control_state = "target_visible_low_quality"
    if case == "yaw_expired": owner._has_fresh_lateral_yaw = lambda uid: False
    r._send_follow_wheel_targets(8, 8, "CURRENT")
    assert driver.pairs[-1] == (0, 0)


@pytest.mark.parametrize("expire_while_reading", [False, True])
def test_periodic_writer_revalidates_current_yaw_after_inner_confirmation(monkeypatch, expire_while_reading):
    r, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    r.config.follow_forward_loss_handoff_enable = True
    r.config.follow_turn_residual_max_rpm = 0
    state[:] = [24., 8., 10.01, 11.]
    r._service_follow_wheels()
    for now, outer, inner in ((10.02, 72, 52), (10.08, 40, 0)):
        clock[0] = now
        r.get_steering_feedback = lambda: feedback(clock[0], outer, inner)
        r._service_follow_wheels()
        assert driver.pairs[-1] == (0, 0)
    clock[0] = 10.14
    def read():
        if expire_while_reading:
            state[3] = clock[0]-.001
        return feedback(clock[0], 32, 0)
    r.get_steering_feedback = read
    r._service_follow_wheels()
    assert driver.pairs[-1] == ((0, 0) if expire_while_reading else (8, 8))
