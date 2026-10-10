"""Capture-time body heading uses only the immutable encoder publication cache."""
from dataclasses import replace
from collections import deque
from types import SimpleNamespace
import logging
import threading

import pytest

from car_control_modular.action_runtime import MotionActionRuntime
from car_control_modular.control_types import SteeringFeedback


def runtime():
    # No backend, serial object, hardware threads or motor initialization.
    rt = object.__new__(MotionActionRuntime)
    rt.owner = SimpleNamespace(motor_io_lock=threading.Lock())
    rt._steering_feedback_lock = threading.Lock()
    rt._steering_feedback = None
    rt._steering_heading_history = ()
    rt._observe_executed_speed_response = lambda sample: None
    return rt


def sample(stamp, heading, rate=20., **changes):
    return replace(SteeringFeedback(
        timestamp=stamp, integrated_yaw_right_deg=heading,
        yaw_rate_right_dps=rate, raw_yaw_rate_right_dps=rate,
        trustworthy=True, yaw_rate_confirmed=True,
    ), **changes)


@pytest.mark.parametrize("direction", [-1, 1])
def test_interpolates_real_capture_heading_in_both_turn_directions(direction):
    rt = runtime()
    first, second = sample(10., direction * 5.), sample(10.1, direction * 7.)
    rt._publish_steering_feedback(first)
    rt._publish_steering_feedback(second)
    assert rt.get_steering_heading_at(10.025) == pytest.approx(direction * 5.5)
    assert rt.get_steering_heading_at(10.05) == pytest.approx(direction * 6.)
    assert rt.get_steering_feedback() is second
    assert rt.get_recording_feedback() is second
    assert rt._steering_heading_history == (first, second)


def test_exact_endpoint_is_available_without_extrapolation():
    rt = runtime()
    assert rt.get_steering_heading_at(10.) is None
    rt._publish_steering_feedback(sample(10., 3.))
    assert rt.get_steering_heading_at(10.) == 3.
    assert rt.get_steering_heading_at(9.999999) is None
    assert rt.get_steering_heading_at(10.000001) is None
    rt._publish_steering_feedback(sample(10.15, 6.))
    assert rt.get_steering_heading_at(10.075) == pytest.approx(4.5)
    assert rt.get_steering_heading_at(10.15) == 6.


def test_large_sample_gap_is_not_filled_even_if_both_samples_are_good():
    rt = runtime()
    rt._publish_steering_feedback(sample(10., 3.))
    rt._publish_steering_feedback(sample(10.151, 6.))
    assert rt.get_steering_heading_at(10.075) is None
    assert rt.get_steering_heading_at(10.) == 3.
    assert rt.get_steering_heading_at(10.151) == 6.


@pytest.mark.parametrize("change", [
    {"trustworthy": False}, {"yaw_rate_confirmed": False},
    {"left_error": 1}, {"right_error": 2},
    {"timestamp": float("nan")}, {"timestamp": float("inf")},
    {"timestamp": 0}, {"timestamp": True},
    {"integrated_yaw_right_deg": float("nan")},
    {"integrated_yaw_right_deg": float("inf")},
    {"yaw_rate_right_dps": float("nan")},
    {"yaw_rate_right_dps": 91.},
])
def test_bad_feedback_breaks_chain_but_does_not_change_feedback_contract(change):
    rt = runtime()
    rt._publish_steering_feedback(sample(10., 0.))
    bad = replace(sample(10.05, 1.), **change)
    rt._publish_steering_feedback(bad)
    assert rt.get_steering_feedback() is bad
    assert rt.get_recording_feedback() is bad
    assert rt._steering_heading_history == ()
    rt._publish_steering_feedback(sample(10.1, 2.))
    assert rt.get_steering_heading_at(10.05) is None
    rt._publish_steering_feedback(sample(10.15, 3.))
    assert rt.get_steering_heading_at(10.125) == pytest.approx(2.5)


@pytest.mark.parametrize("stamp", [10., 9.9])
def test_duplicate_or_out_of_order_publication_breaks_chain(stamp):
    rt = runtime()
    rt._publish_steering_feedback(sample(10., 0.))
    rt._publish_steering_feedback(sample(stamp, 1.))
    assert rt._steering_heading_history == ()
    rt._publish_steering_feedback(sample(10.1, 2.))
    assert rt.get_steering_heading_at(10.05) is None


@pytest.mark.parametrize("heading", [100., -100., 359.])
def test_heading_reset_or_impossible_jump_is_not_interpolated(heading):
    rt = runtime()
    rt._publish_steering_feedback(sample(10., 0.))
    rt._publish_steering_feedback(sample(10.05, heading))
    assert rt.get_steering_heading_at(10.025) is None
    assert rt.get_steering_heading_at(10.05) is None


def test_unwrapped_heading_can_pass_through_360_without_a_reset():
    rt = runtime()
    rt._publish_steering_feedback(sample(10., 359.))
    rt._publish_steering_feedback(sample(10.1, 361.))
    assert rt.get_steering_heading_at(10.05) == pytest.approx(360.)


@pytest.mark.parametrize("stamp", [None, "invalid", False, float("nan"), float("inf"), 0., -1.])
def test_invalid_capture_returns_unavailable(stamp):
    rt = runtime()
    rt._publish_steering_feedback(sample(10., 3.))
    assert rt.get_steering_heading_at(stamp) is None


def test_read_does_not_acquire_motor_or_feedback_lock_or_change_any_state():
    rt = runtime()
    rt._publish_steering_feedback(sample(10., 0.))
    rt._publish_steering_feedback(sample(10.1, 2.))
    history = rt._steering_heading_history
    feedback = rt._steering_feedback

    class ForbiddenLock:
        def __enter__(self):
            raise AssertionError("capture geometry must not wait for serial or feedback")
        def __exit__(self, *args):
            pass

    rt.owner.motor_io_lock = ForbiddenLock()
    rt._steering_feedback_lock = ForbiddenLock()
    assert rt.get_steering_heading_at(10.05) == pytest.approx(1.)
    assert rt._steering_heading_history is history
    assert rt._steering_feedback is feedback


def test_copy_on_write_cache_stays_bounded_and_old_snapshot_is_immutable():
    rt = runtime()
    rt._publish_steering_feedback(sample(10., 0.))
    old = rt._steering_heading_history
    for index in range(1, 100):
        rt._publish_steering_feedback(sample(10. + index * .1, index * 2.))
    assert len(old) == 1
    assert len(rt._steering_heading_history) == 64
    assert rt.get_steering_heading_at(10.05) is None
    assert rt.get_steering_heading_at(19.85) == pytest.approx(197.)


def test_executed_response_observer_runs_outside_motor_and_feedback_locks():
    rt = runtime()
    observed = []
    def observer(feedback):
        assert rt.owner.motor_io_lock.acquire(blocking=False)
        rt.owner.motor_io_lock.release()
        assert rt._steering_feedback_lock.acquire(blocking=False)
        rt._steering_feedback_lock.release()
        observed.append(feedback)
    rt._observe_executed_speed_response = observer
    feedback = sample(10., 5.)
    rt._publish_steering_feedback(feedback)
    assert observed == [feedback]


def test_runtime_restart_retires_heading_origin_without_starting_hardware(monkeypatch):
    rt = runtime()
    rt._publish_steering_feedback(sample(10., 33.))
    rt.owner.action_stop_event = threading.Event()
    rt.config = SimpleNamespace(visible_steering_pid_enable=False)
    rt.logger = logging.getLogger(__name__)
    rt._steering_feedback_yaw_samples = deque()
    monkeypatch.setattr("car_control_modular.action_runtime.threading.Thread",
                        lambda **kwargs: SimpleNamespace(start=lambda: None))
    rt.start()
    assert rt._steering_heading_history == ()
    assert rt.get_steering_heading_at(10.) is None
    assert rt._steering_feedback_integrated_yaw_deg == 0.
    rt._publish_steering_feedback(sample(10.1, 0.))
    assert rt.get_steering_heading_at(10.05) is None
