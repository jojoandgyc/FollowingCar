"""CAP329: 91/109 -> lost Depth -> -4/+4 must not start a reversal."""
import math
from types import SimpleNamespace

import pytest

from car_control_modular.forward_loss_handoff import ForwardLossHandoff
from test_visible_wheel_continuity import feedback, visible_runtime
from test_follow_wheel_periodic import setup_periodic


@pytest.mark.parametrize("yaw", [-8, 8])
def test_periodic_depth_gap_never_creates_reverse_pending_and_new_forward_resumes(monkeypatch, yaw):
    r, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    r.config.follow_forward_loss_handoff_enable = True
    r.config.follow_forward_handoff_enable = True
    state[:] = [24., yaw, 10.01, 11.]
    r._service_follow_wheels()
    for t in (10.011, 10.06, 10.11):
        clock[0] = t
        r._service_follow_wheels()
        assert driver.pairs[-1] == (0, 0)
        assert r._visible_wheel_guard.pending_signs is None
        assert r._forward_loss_handoff.started == 10.011
    state[:] = [62., yaw, 11., 11.]
    clock[0] = 10.16
    r.get_steering_feedback = lambda: feedback(clock[0], 12, 11)
    r._service_follow_wheels()
    assert driver.pairs[-1] == (62+yaw, -(62-yaw))
    assert r._forward_loss_handoff.started is None
    assert not driver.stops


def test_two_distinct_quiet_samples_allow_current_turn_not_old_turn():
    h = ForwardLossHandoff()
    h.note_sent((91, 109))
    assert h.limit((-4, 4), feedback(10, 34, 32), 10)[0] == (0, 0)
    for _ in range(3):
        assert h.limit((-4, 4), feedback(10.05, 0, 0), 10.05)[0] == (0, 0)
    assert h.limit((8, -8), feedback(10.1, 0, 0), 10.1) == ((8, -8), "forward_loss_stopped_turn_ready")


@pytest.mark.parametrize("sample", [None, feedback(9), feedback(11), feedback(float("nan")),
    feedback(10, left=float("inf")), feedback(10, trustworthy=False), feedback(10, 20, 20)])
def test_feedback_cannot_be_replaced_by_timeout(sample):
    h = ForwardLossHandoff()
    h.note_sent((30, 40))
    assert h.limit((-4, 4), sample, 10)[0] == (0, 0)
    assert h.limit((-4, 4), sample, 12)[0] == (0, 0)
    assert h.armed


def test_zero_updates_do_not_reset_wait_and_no_forward_start_is_not_changed():
    h = ForwardLossHandoff()
    assert h.limit((8, -8), feedback(10), 10) == ((8, -8), None)
    h.note_sent((10, 20))
    assert h.limit((0, 0), feedback(10, 3, 3), 10)[0] == (0, 0)
    h.note_sent((0, 0))
    assert h.limit((0, 0), feedback(10.05), 10.05)[0] == (0, 0)
    assert h.limit((-4, 4), feedback(10.1), 10.1)[0] == (-4, 4)


@pytest.mark.parametrize("state", ["uid", "search"])
def test_identity_or_scope_change_does_not_inherit_forward_loss(monkeypatch, state):
    r, owner, driver, _, clock = visible_runtime(monkeypatch)
    r.config.follow_forward_loss_handoff_enable = True
    r._send_follow_wheel_targets(24, -24, "TEST")
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
    if state == "uid":
        owner._follow_controller.active_target_id = 2
    else:
        owner.search_state = "searching"
    r._send_follow_wheel_targets(4, 4, "TEST", visible_required=True)
    assert not r._forward_loss_handoff.armed


def test_real_backward_feedback_not_hidden_by_forward_resume(monkeypatch):
    r, owner, driver, _, clock = visible_runtime(monkeypatch)
    r.config.follow_forward_loss_handoff_enable = True
    r.config.follow_forward_handoff_enable = True
    r._send_follow_wheel_targets(24, -24, "TEST")
    r.get_steering_feedback = lambda: feedback(clock[0], -20, -20)
    r._send_follow_wheel_targets(24, -24, "TEST")
    assert driver.pairs[-1] == (0, 0)
    assert r._visible_wheel_guard.pending_full_reverse
