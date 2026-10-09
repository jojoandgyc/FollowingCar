import pytest
from test_visible_wheel_continuity import visible_runtime, feedback


@pytest.mark.parametrize("reason", ["stop_signal", "queued_action_stop_signal"])
def test_interrupting_soft_stop_cannot_escalate_to_normal(monkeypatch, reason):
    r, owner, driver, _, _ = visible_runtime(monkeypatch)
    owner._soft_stop_active = True
    r.send_stop_with_brake_hold(reason, preserve_motion_params=True)
    assert driver.pairs == [(0, 0)]
    assert not driver.stops
    assert owner._soft_stop_active
    assert not getattr(owner, "_brake_hold_active", False)


@pytest.mark.parametrize("reason", ["hard_stop", "front_ir", "explicit_stop"])
def test_safety_and_explicit_stop_not_softened(monkeypatch, reason):
    r, owner, driver, _, _ = visible_runtime(monkeypatch)
    owner._soft_stop_active = True
    r.send_stop_with_brake_hold(reason)
    assert driver.stops


def test_low_quality_lateral_keeps_guard_but_cannot_use_forward(monkeypatch):
    r, owner, driver, _, clock = visible_runtime(monkeypatch)
    r.config.follow_forward_loss_handoff_enable = True
    owner._vision_control_state = "target_visible_low_quality"
    assert r._visible_wheel_control_active()
    # A stale upstream grant cannot leak forward through the lateral-only path.
    r.get_steering_feedback = lambda: feedback(clock[0], 0, 0)
    with owner.motor_io_lock:
        r._send_follow_wheel_targets(14, -34, "LOW_QUALITY")
    assert driver.pairs == [(-10, -10)]
    assert not driver.stops


def test_real_queue_empty_interrupt_keeps_soft_stop(monkeypatch):
    import queue
    import threading
    r, owner, driver, s, _ = visible_runtime(monkeypatch)
    owner.action_stop_event = threading.Event()
    owner._soft_stop_active = True
    owner.current_command = s.stop
    owner.stop_action_execution = True
    r.config.enable_motor_rpm_feedback = False
    r._service_follow_wheels = lambda: None
    r._service_yaw_pulses = lambda: None
    class EmptyOnce:
        def get_nowait(self):
            owner.action_stop_event.set()
            raise queue.Empty
    owner.action_queue = EmptyOnce()
    owner.action_queue_lock = threading.Lock()
    owner.command_lock = threading.Lock()
    r.run_loop()
    assert driver.pairs == [(0, 0)] and not driver.stops
    assert owner.current_command is None
    assert not owner.stop_action_execution


def test_periodic_soft_stop_keeps_provenance_for_later_interrupt(monkeypatch):
    from test_follow_wheel_periodic import setup_periodic
    r, owner, driver, s, _, state = setup_periodic(monkeypatch)
    state[0] = state[1] = 0
    owner._use_soft_stop_next = True
    r.send_robot_command(s.stop)
    assert owner._soft_stop_active
    r.send_stop_with_brake_hold("stop_signal")
    assert driver.pairs == [(0, 0)] and not driver.stops
    r.send_robot_command(s.rotate_left)
    assert not owner._soft_stop_active
