"""Current forward/yaw axes survive obsolete handoffs; fake motor and clock only."""
from contextlib import contextmanager

import pytest

from test_follow_wheel_periodic import setup_periodic
from test_visible_wheel_continuity import feedback


def enable_handoff(runtime):
    runtime.config.follow_forward_handoff_enable = True
    runtime.config.follow_forward_loss_handoff_enable = True
    runtime.config.follow_turn_residual_max_rpm = 4


def replace_on_lock(owner, change):
    original_lock = owner.motor_io_lock
    fired = []

    @contextmanager
    def lock():
        with original_lock:
            if not fired:
                fired.append(True)
                change()
            yield

    class Lock:
        def __enter__(self):
            self.context = lock()
            return self.context.__enter__()

        def __exit__(self, *args):
            return self.context.__exit__(*args)

    owner.motor_io_lock = Lock()


@pytest.mark.parametrize("yaw", [-8., 0., 8.])
def test_restored_axes_supersede_revocation_while_waiting_for_motor_lock(monkeypatch, yaw):
    runtime, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    enable_handoff(runtime)
    runtime._service_follow_wheels()
    driver.pairs.clear()
    clock[0] = 10.06
    owner.search_state = "searching"

    def restore():
        owner.search_state = "none"
        state[:] = [52., yaw, 10.30, 10.30]

    replace_on_lock(owner, restore)
    runtime._service_follow_wheels()
    assert driver.pairs == [(52+yaw, -(52-yaw))]
    assert not driver.stops
    assert runtime._follow_wheel_clock.last_axes[2:] == (52., yaw)


@pytest.mark.parametrize("veto", ["expired", "identity", "explicit", "shutdown", "danger", "reverse"])
def test_lock_handoff_still_rechecks_all_authority_and_actual_reverse(monkeypatch, veto):
    runtime, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    enable_handoff(runtime)
    runtime._service_follow_wheels()
    driver.pairs.clear()
    clock[0] = 10.06
    owner.search_state = "searching"

    def restore():
        owner.search_state = "none"
        state[:] = [52., 0., 10.30, 10.30]
        if veto == "expired": state[2] = 9.
        if veto == "identity": owner._follow_controller.active_target_id = 2
        if veto == "explicit": owner._explicit_stop_requested = True
        if veto == "shutdown": owner._runtime_shutdown_requested = True
        if veto == "danger": runtime.hard_stop_check = lambda _: True
        if veto == "reverse": runtime.get_steering_feedback = lambda: feedback(clock[0], -20, -20)

    replace_on_lock(owner, restore)
    runtime._service_follow_wheels()
    assert all(pair == (0, 0) for pair in driver.pairs)
    assert driver.pairs or driver.stops
    if veto == "shutdown":
        # The newly latched shutdown won motor exclusion; no legacy zero
        # speed command may follow its emergency STOP.
        assert not driver.pairs and driver.stops == [1]


@pytest.mark.parametrize("new_yaw", [-8., 0., 8.])
def test_new_forward_during_pivot_current_release_rebuilds_without_reparking(monkeypatch, new_yaw):
    runtime, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    enable_handoff(runtime)
    state[:] = [0., -7., 10.25, 10.25]
    runtime.get_steering_feedback = lambda: feedback(clock[0])
    runtime.backend.normal_zero_hold = True
    original = runtime.backend.prepare_speed_mode
    calls = []

    def prepare():
        original()
        if not calls:
            calls.append(True)
            state[:2] = [52., new_yaw]
            owner._lateral_yaw_revision += 1

    runtime.backend.prepare_speed_mode = prepare
    runtime._service_follow_wheels()
    assert driver.pairs == [(52+new_yaw, -(52-new_yaw))]
    assert not driver.stops


@pytest.mark.parametrize("veto", ["expired", "identity", "explicit", "danger", "reverse"])
def test_current_release_rebuild_never_revives_expired_or_conflicting_authority(monkeypatch, veto):
    runtime, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    enable_handoff(runtime)
    state[:] = [0., -7., 10.25, 10.25]
    runtime.get_steering_feedback = lambda: feedback(clock[0])
    runtime.backend.normal_zero_hold = True
    original = runtime.backend.prepare_speed_mode
    calls = []

    def prepare():
        original()
        if not calls:
            calls.append(True)
            state[:2] = [52., 8.]
            owner._lateral_yaw_revision += 1
            if veto == "expired": state[2] = state[3] = 9.
            if veto == "identity": owner._follow_controller.active_target_id = 2
            if veto == "explicit": owner._explicit_stop_requested = True
            if veto == "danger": runtime.hard_stop_check = lambda _: True
            if veto == "reverse": runtime.get_steering_feedback = lambda: feedback(clock[0], -20, -20)

    runtime.backend.prepare_speed_mode = prepare
    runtime._service_follow_wheels()
    assert all(pair == (0, 0) for pair in driver.pairs)
    assert driver.pairs or driver.stops or runtime.backend.normal_zero_hold
    if veto == "reverse":
        assert runtime._visible_wheel_guard.pending_full_reverse


def test_current_release_rebuild_is_bounded_when_revision_keeps_changing(monkeypatch):
    runtime, owner, driver, _, _, state = setup_periodic(monkeypatch)
    state[:] = [0., -7., 10.25, 10.25]
    runtime.get_steering_feedback = lambda: feedback(10.)
    runtime.backend.normal_zero_hold = True
    original = runtime.backend.prepare_speed_mode
    calls = []

    def prepare():
        original()
        calls.append(True)
        state[:2] = [52., 8.]
        owner._lateral_yaw_revision += 1

    runtime.backend.prepare_speed_mode = prepare
    runtime._service_follow_wheels()
    assert len(calls) == 2
    assert not any(left or right for left, right in driver.pairs)
    assert runtime._follow_wheel_clock.last_axes is None


def test_legitimate_yaw_changes_do_not_drop_forward_base(monkeypatch):
    runtime, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    enable_handoff(runtime)
    for index, yaw in enumerate((8., 0., -8., 8.)):
        clock[0] = 10.0 + index*.06
        state[:] = [52., yaw, clock[0]+.25, clock[0]+.20]
        owner._lateral_yaw_revision += 1
        runtime._service_follow_wheels()
        assert driver.pairs[-1] == (52+yaw, -(52-yaw))
    assert not driver.stops


def test_depth_gap_then_new_grant_replaces_handoff_with_current_curve(monkeypatch):
    runtime, _, driver, _, clock, state = setup_periodic(monkeypatch)
    enable_handoff(runtime)
    state[:] = [52., 8., 10.07, 10.25]
    runtime._service_follow_wheels()
    clock[0] = 10.08
    runtime._service_follow_wheels()
    assert driver.pairs[-1] == (0, 0)  # No invented forward arc during depth loss.
    clock[0] = 10.14
    state[:] = [52., -8., 10.35, 10.35]
    runtime._service_follow_wheels()
    assert driver.pairs[-1] == (44, -60)
    assert runtime._forward_loss_handoff.started is None
    assert runtime._visible_wheel_guard.pending_signs is None
