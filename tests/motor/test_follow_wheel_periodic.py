"""Fake-clock and fake-driver coverage, no hardware execution."""
from test_visible_wheel_continuity import visible_runtime, feedback
import pytest


def setup_periodic(monkeypatch):
    rt, owner, driver, symbols, clock = visible_runtime(monkeypatch)
    rt.config.follow_wheel_period_sec = .05
    state = [24., 0., 10.18, 10.18]
    def axes(now):
        return (1, owner._lateral_yaw_revision, state[0] if now < state[2] else 0.,
                state[1] if now < state[3] else 0.)
    owner._follow_wheel_axes = axes
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: (
        ("forward", state[0]) if clock[0] < state[2] and state[0] else None)
    owner._has_fresh_lateral_yaw = lambda uid: clock[0] < state[3] and state[1] != 0
    rt.get_steering_feedback = lambda: feedback(clock[0], 24, 24)
    return rt, owner, driver, symbols, clock, state


def test_updates_coalesce_both_axes_and_do_not_emit_old_queue_actions(monkeypatch):
    rt, owner, driver, s, clock, state = setup_periodic(monkeypatch)
    rt._service_follow_wheels()
    assert driver.pairs == [(24, -24)]
    clock[0] += .01
    state[0], state[1] = 30, 5
    rt.send_robot_command(s.steer_left)  # stale name cannot reverse canonical right yaw
    rt._service_follow_wheels()
    clock[0] += .01
    state[0], state[1] = 40, 6
    rt.send_robot_command(s.forward)  # must not lose the yaw axis
    rt._service_follow_wheels()
    assert len(driver.pairs) == 1
    clock[0] = 10.05
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (46, -34)
    assert len(driver.pairs) == 2 and not driver.stops


def test_no_new_vision_still_refreshes_valid_axes(monkeypatch):
    rt, _, driver, _, clock, _ = setup_periodic(monkeypatch)
    for now in (10., 10.05, 10.10, 10.15):
        clock[0] = now
        rt._service_follow_wheels()
    assert driver.pairs == [(24, -24)] * 4


def test_depth_expires_inside_period_not_after_fifty_ms(monkeypatch):
    rt, _, driver, _, clock, state = setup_periodic(monkeypatch)
    state[2] = 10.01
    rt._service_follow_wheels()
    clock[0] = 10.011
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (0, 0)


def test_yaw_zero_immediate_preserves_base_and_ignores_stale_soft_stop(monkeypatch):
    rt, owner, driver, s, clock, state = setup_periodic(monkeypatch)
    state[1] = 5
    rt._service_follow_wheels()
    clock[0] += .01
    state[1] = 0
    owner._use_soft_stop_next = True
    rt.send_robot_command(s.stop)
    rt._service_follow_wheels()
    assert driver.pairs == [(29, -19), (24, -24)]


@pytest.mark.parametrize("change", ["explicit_stop", "search", "shutdown"])
def test_state_exit_revokes_periodic_pair_once(monkeypatch, change):
    rt, owner, driver, _, clock, _ = setup_periodic(monkeypatch)
    rt._service_follow_wheels()
    clock[0] += .01
    if change == "explicit_stop": owner._explicit_stop_requested = True
    elif change == "search": owner.search_state = "searching"
    else: owner._runtime_shutdown_requested = True
    rt._service_follow_wheels()
    rt._service_follow_wheels()
    assert driver.pairs == [(24, -24), (0, 0)]


def test_expiry_while_reading_feedback_cannot_write_stale_pair(monkeypatch):
    rt, _, driver, _, clock, state = setup_periodic(monkeypatch)
    state[2] = 10.01
    def read():
        clock[0] = 10.02
        return feedback(clock[0], 24, 24)
    rt.get_steering_feedback = read
    rt._service_follow_wheels()
    assert driver.pairs == [(0, 0)]


def test_stall_skips_ticks_instead_of_burst_replay(monkeypatch):
    rt, _, driver, _, clock, state = setup_periodic(monkeypatch)
    state[2:] = [11., 11.]
    rt._service_follow_wheels()
    clock[0] = 10.31
    rt._service_follow_wheels()
    rt._service_follow_wheels()
    assert len(driver.pairs) == 2


def test_hard_stop_is_checked_between_periods(monkeypatch):
    rt, _, driver, _, clock, _ = setup_periodic(monkeypatch)
    rt._service_follow_wheels()
    clock[0] += .005
    reasons = []
    rt.hard_stop_check = lambda action: True
    rt.send_stop_with_brake_hold = reasons.append
    rt._service_follow_wheels()
    assert reasons == ["follow20_hard_stop"]
    assert rt._follow_wheel_clock.last_axes is None
    assert driver.pairs == [(24, -24)]  # no second motion write before stop


def test_direct_old_packet_cannot_escape_periodic_writer(monkeypatch):
    rt, _, driver, _, _, _ = setup_periodic(monkeypatch)
    rt.send_percent_drive(60, allow_below_min=True)
    assert driver.pairs == []
    rt._service_follow_wheels()
    assert driver.pairs == [(24, -24)]
