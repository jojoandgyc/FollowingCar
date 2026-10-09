"""CAP272: a new forward grant arriving after the early axes check wins.

No hardware. The packet must be rebuilt through ordinary safety/zero-cross
checks, not patched or replayed from a queued command.
"""
import pytest

from test_follow_wheel_periodic import setup_periodic
from test_visible_wheel_continuity import feedback


def setup_arrival(monkeypatch, cause=None):
    rt, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    state[:] = [0., 0., 11., 11.]
    rt.config.follow_forward_handoff_enable = True
    rt.get_steering_feedback = lambda: feedback(clock[0], 20, 20)
    stage = [False, False, 0]
    original_limit = rt._visible_wheel_guard.limit
    original_visible = rt._visible_wheel_control_active
    def limit(*args, **kwargs):
        value = original_limit(*args, **kwargs)
        stage[0] = True
        stage[2] += 1
        return value
    def visible():
        result = original_visible()
        if stage[0] and not stage[1]:
            # The early axes equality check has just passed. A concurrent
            # producer now commits new depth before the old zero is written.
            stage[1] = True
            state[:2] = [72., 10.]
            if cause == 'expired': state[2] = 9.
            if cause == 'identity': owner._follow_controller.active_target_id = 2
            if cause == 'shutdown': owner._runtime_shutdown_requested = True
            if cause == 'explicit': owner._explicit_stop_requested = True
            if cause == 'danger': rt.hard_stop_check = lambda _: True
            if cause == 'reverse': rt.get_steering_feedback = lambda: feedback(clock[0], -20, -20)
        return result
    rt._visible_wheel_guard.limit = limit
    rt._visible_wheel_control_active = visible
    return rt, owner, driver, stage


def test_new_authorized_forward_supersedes_unsent_zero(monkeypatch, caplog):
    rt, owner, driver, stage = setup_arrival(monkeypatch)
    with caplog.at_level('INFO'):
        rt._service_follow_wheels()
    assert stage[1] and stage[2] == 2
    assert driver.pairs == [(82, -62)]
    assert not driver.stops
    assert 'zero_plan_superseded_before_write' in caplog.text
    assert rt._follow_wheel_clock.last_axes[2:] == (72., 10.)


@pytest.mark.parametrize('cause', ['expired', 'identity', 'shutdown', 'explicit', 'danger', 'reverse'])
def test_new_plan_still_passes_all_existing_stop_checks(monkeypatch, cause):
    rt, owner, driver, stage = setup_arrival(monkeypatch, cause)
    rt._service_follow_wheels()
    assert stage[1] and stage[2] <= 2
    assert all(pair == (0, 0) for pair in driver.pairs)
    if cause == 'danger': assert driver.stops


def test_unchanged_zero_does_not_retry(monkeypatch):
    rt, _, driver, _, _, state = setup_periodic(monkeypatch)
    state[:2] = [0., 0.]
    rt._service_follow_wheels()
    assert driver.pairs == [(0, 0)]


def setup_forward_loss_late_arrival(monkeypatch, *, yaw=0., change="forward"):
    """Publish after the early zero-plan read, at the final write boundary."""
    rt, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    rt.config.follow_forward_loss_handoff_enable = True
    # CAP277 had moving wheels, but its last encoder report preceded the
    # zero plan. That labels even a requested (0, 0) as feedback_wait.
    rt.get_steering_feedback = lambda: feedback(clock[0] - .02, 20, 20)
    rt._service_follow_wheels()
    assert rt._forward_loss_handoff.armed
    clock[0] += .06
    state[:2] = [0., yaw]
    original_read = rt._read_follow_axes
    after_loss_reads = [0]

    def read(now):
        axes = original_read(now)
        if rt._forward_loss_handoff.started is not None:
            after_loss_reads[0] += 1
            if after_loss_reads[0] == 2:
                # The second read is the early zero-plan comparison. Return
                # its old axes, then let the producer change the live state.
                if change != "no_grant":
                    state[0] = 20.
                if change == "identity":
                    owner._follow_controller.active_target_id = 2
                elif change == "explicit_stop":
                    owner._explicit_stop_requested = True
                elif change == "brake_hold":
                    owner._brake_hold_active = True
                elif change == "parking":
                    owner._near_yaw_park_request = object()
                elif change == "reverse_feedback":
                    rt.get_steering_feedback = lambda: feedback(clock[0] - .02, -20, -20)
        return axes

    rt._read_follow_axes = read
    return rt, owner, driver, after_loss_reads


def test_requested_zero_feedback_wait_yields_to_new_forward_at_write(monkeypatch, caplog):
    rt, _, driver, reads = setup_forward_loss_late_arrival(monkeypatch)
    with caplog.at_level("INFO"):
        rt._service_follow_wheels()
    assert reads[0] > 2
    assert driver.pairs == [(24, -24), (20, -20)]
    assert "zero_superseded_at_write" in caplog.text
    assert "zero_plan_superseded_before_write" not in caplog.text
    assert not driver.stops


@pytest.mark.parametrize("change,yaw", [
    ("no_grant", 0.),
    ("forward", 4.),  # A blocked pivot is a safety zero, not requested zero.
    ("identity", 0.),
    ("explicit_stop", 0.),
    ("brake_hold", 0.),
    ("parking", 0.),
    ("reverse_feedback", 0.),
])
def test_feedback_wait_zero_does_not_bypass_safety_owner(monkeypatch, change, yaw):
    rt, _, driver, reads = setup_forward_loss_late_arrival(
        monkeypatch, yaw=yaw, change=change)
    rt._service_follow_wheels()
    assert reads[0] >= 2
    assert all(pair == (0, 0) for pair in driver.pairs[1:])
