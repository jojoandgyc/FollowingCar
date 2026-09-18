"""Near-center parking exercises real dispatch with a fake serial driver only."""
import pytest
import queue
import threading

from car_control_modular.near_yaw_parking import NearYawParkRequest
from test_follow_wheel_periodic import setup_periodic


def request_park(owner, clock):
    request = NearYawParkRequest(1, 46, clock[0] - .10, clock[0], "predictive_brake_coast")
    owner._near_yaw_park_request = request
    return request


def test_explicit_park_preempts_periodic_tick_and_sends_normal_once(monkeypatch):
    rt, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    state[:2] = [0., 6.]
    rt._service_follow_wheels()
    request_park(owner, clock)
    rt._service_follow_wheels()
    count = len(driver.pairs)
    for _ in range(5):
        clock[0] += .01
        rt._service_follow_wheels()
    assert driver.stops == [0]  # NORMAL, not just a zero target
    assert len(driver.pairs) == count  # no FOLLOW20_REVOKED after parking
    assert owner._brake_hold_active
    assert owner._brake_hold_label == "near_yaw_park"
    assert owner._brake_hold_stop_mode == "normal"
    assert owner._follow_distance_hold is None
    assert rt._follow_wheel_clock.last_axes is None


@pytest.mark.parametrize("packet", [
    "forward_zero", "soft_stop", "turn", "pulse_zero", "transition_hold",
    "percent_zero", "reverse_zero", "direct_pair", "transition_zero", "transition_stop", "ordinary_stop",
])
def test_old_packets_cannot_unlock_park(monkeypatch, packet):
    rt, owner, driver, s, clock, _ = setup_periodic(monkeypatch)
    request = request_park(owner, clock)
    rt._service_follow_wheels()
    snapshot = (list(driver.pairs), list(driver.stops))
    owner._current_forward_percent = 0
    owner._use_soft_stop_next = True
    if packet == "forward_zero":
        rt.send_robot_command(s.forward)
        rt.send_percent_drive(0)
    elif packet == "soft_stop":
        rt.send_robot_command(s.stop)
    elif packet == "turn":
        rt.send_robot_command(s.rotate_right)
    elif packet == "pulse_zero":
        rt.send_rotate_pulse_zero_stop()
    elif packet == "transition_hold":
        rt.config.rotate_pulse_transition_rpm = 4
        rt.send_rotate_transition_hold(s.rotate_left)
    elif packet == "percent_zero":
        rt.send_percent_diff(0, 0, 0, 0, "STEER")
    elif packet == "reverse_zero":
        rt.send_percent_backward(0)
    elif packet == "direct_pair":
        with owner.motor_io_lock:
            rt._send_follow_wheel_targets(0, 0, "FOLLOW20")
            rt._send_follow_wheel_targets(7, 7, "TURN")
    elif packet == "transition_zero":
        rt.config.motor_rs485_transition_stop_mode = "zero"
        rt.send_transition_stop_sequence("old_rotate")
    elif packet == "transition_stop":
        rt.send_transition_stop_sequence("old_rotate")
    elif packet == "ordinary_stop":
        rt.send_stop_with_brake_hold("stop_signal")
    assert (driver.pairs, driver.stops) == snapshot
    assert owner._near_yaw_park_request is request
    assert owner._brake_hold_label == "near_yaw_park"
    assert not rt.can_release_brake_hold(s.rotate_left)
    assert not rt.can_release_brake_hold(s.forward)


def test_zero_without_explicit_parking_keeps_existing_semantics(monkeypatch):
    rt, owner, driver, _, _, state = setup_periodic(monkeypatch)
    state[:2] = [0., 0.]
    rt._service_follow_wheels()
    assert driver.pairs == [(0, 0)] and not driver.stops
    assert not owner._brake_hold_active


def test_zero_yaw_with_valid_forward_authority_still_drives_straight(monkeypatch):
    rt, _, driver, _, _, state = setup_periodic(monkeypatch)
    state[:2] = [24., 0.]
    rt._service_follow_wheels()
    assert driver.pairs == [(24, -24)] and not driver.stops


def test_zero_base_without_park_still_allows_recenter(monkeypatch):
    rt, _, driver, _, _, state = setup_periodic(monkeypatch)
    state[:2] = [0., 5.]
    rt.get_steering_feedback = lambda: None
    rt._service_follow_wheels()
    assert not driver.stops  # no distance-only global all-stop condition


def test_fresh_producer_release_restores_writer_not_old_queued_release(monkeypatch):
    rt, owner, driver, s, clock, state = setup_periodic(monkeypatch)
    request_park(owner, clock)
    rt._service_follow_wheels()
    assert not rt.can_release_brake_hold(s.rotate_right)
    # Only a validated producer transition may perform this atomic release.
    with owner.motor_io_lock:
        owner._near_yaw_park_request = None
        owner._brake_hold_active = False
        owner._brake_hold_stop_mode = None
        owner._brake_hold_label = "brake"
        owner._lateral_yaw_revision += 1
        state[:2] = [24., 0.]
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (24, -24)
    rt.send_percent_brake(mode="normal", label="near_yaw_park")
    assert driver.stops == [0]
    assert rt._near_yaw_park_applied is None


def test_hard_stop_overrides_normal_without_later_downgrade(monkeypatch):
    rt, owner, driver, _, clock, _ = setup_periodic(monkeypatch)
    request_park(owner, clock)
    rt._service_follow_wheels()
    rt.send_stop_with_brake_hold("hard_stop")
    rt._service_follow_wheels()
    rt.send_percent_brake(mode="normal", label="near_yaw_park")
    assert driver.stops == [0, 1]
    assert owner._brake_hold_label == "safety_hold_hard_stop"
    assert owner._brake_hold_stop_mode == "emergency"


def test_existing_emergency_is_not_downgraded_by_new_park(monkeypatch):
    rt, owner, driver, _, clock, _ = setup_periodic(monkeypatch)
    rt.send_stop_with_brake_hold("hard_stop")
    request_park(owner, clock)
    rt._service_follow_wheels()
    assert driver.stops == [1]
    assert owner._brake_hold_label == "safety_hold_hard_stop"


def test_pending_park_is_rechecked_after_feedback_before_motor_write(monkeypatch):
    rt, owner, driver, _, clock, _ = setup_periodic(monkeypatch)
    original = rt.get_steering_feedback
    def feedback():
        request_park(owner, clock)
        return original()
    rt.get_steering_feedback = feedback
    rt._service_follow_wheels()
    assert not driver.pairs
    rt._service_follow_wheels()
    assert driver.stops == [0]


def test_zero_packet_prepared_before_park_is_vetoed_under_motor_lock(monkeypatch):
    rt, owner, driver, _, clock, _ = setup_periodic(monkeypatch)
    lock = owner.motor_io_lock
    class PublishOnAcquire:
        def __enter__(self):
            lock.acquire()
            request_park(owner, clock)
        def __exit__(self, *exc):
            lock.release()
    owner.motor_io_lock = PublishOnAcquire()
    assert not rt.send_percent_drive(0)
    assert not driver.pairs
    owner.motor_io_lock = lock
    rt._service_follow_wheels()
    assert driver.stops == [0]


def test_cancel_aux_pulses_without_zero_before_normal(monkeypatch):
    rt, owner, driver, _, clock, _ = setup_periodic(monkeypatch)
    rt._visible_yaw_pulse_direction = 1
    rt._visible_yaw_pulse_kind = "tracking"
    rt._visible_yaw_pulse_started_monotonic = clock[0] - .10
    rt.get_steering_feedback = lambda: None
    request_park(owner, clock)
    rt._service_follow_wheels()
    assert not rt.yaw_aux_pulse_active()
    assert driver.stops == [0]
    assert driver.pairs == [(0, 0)]  # only NORMAL's own pre-zero


@pytest.mark.parametrize("action_name", ["stop", "rotate_left", "forward"])
def test_popped_action_cannot_cross_parking_generation(monkeypatch, action_name):
    rt, owner, _, s, clock, _ = setup_periodic(monkeypatch)
    action = getattr(s, action_name)
    assert rt._near_yaw_park_queue_action_allowed(action, 0)
    request_park(owner, clock)
    owner._near_yaw_park_generation = 1
    assert not rt._near_yaw_park_queue_action_allowed(action, 0)
    assert not rt._near_yaw_park_queue_action_allowed(action, 1)
    # An already-popped STOP must also be dropped after a fresh release.
    owner._near_yaw_park_request = None
    owner._near_yaw_park_generation = 2
    assert not rt._near_yaw_park_queue_action_allowed(action, 1)
    assert rt._near_yaw_park_queue_action_allowed(action, 2)


@pytest.mark.parametrize("safety", [False, True])
def test_real_loop_drops_action_popped_before_release_but_keeps_safety(monkeypatch, safety):
    rt, owner, driver, s, _, _ = setup_periodic(monkeypatch)
    owner.action_stop_event = threading.Event()
    owner.action_queue = queue.Queue()
    owner.action_queue.put(s.stop)
    owner._near_yaw_park_generation = 1
    owner.current_command = None
    rt.config.enable_motor_rpm_feedback = False
    rt.config.action_intent_stale_sec = .45
    rt._service_follow_wheels = lambda: None
    rt._service_yaw_pulses = lambda: None
    rt.hard_stop_check = lambda action: safety
    class ReleaseBeforeCommandLock:
        def __enter__(self):
            owner._near_yaw_park_generation = 2
            owner.action_stop_event.set()
        def __exit__(self, *exc):
            pass
    owner.command_lock = ReleaseBeforeCommandLock()
    rt.run_loop()
    assert driver.stops == ([1] if safety else [])
    assert owner.current_command is None
