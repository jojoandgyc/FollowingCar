"""Saved-run regressions: no serial, cameras, motor thread or live authority."""
from dataclasses import replace
import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import SteeringFeedback
from test_depth_authority_250 import writer
from test_relative_depth_continuation_runtime import (
    CIRCUMFERENCE, NOW, at_age, owner, pi_owner, publish, relative,
)


def test_cap32_previous_completed_command_cost_preserves_new_lower_request(relative):
    # Prior actual packet 43/33, not the PI's newly lowered 18 RPM request.
    calls = []
    def completed(uid, now):
        calls.append((uid, now))
        return 43.
    relative.owner._action_runtime.continuation_executed_speed_bound_rpm = completed
    stamp, original = publish(relative, distance=1.912, percent=9, rpm=20.,
                              stamp=NOW-.0508, target_speed=.229613, range_rate=-.04)
    timing = relative.owner._depth30_linear_timing
    assert calls and calls[0][0] == 1
    assert timing.continuation_speed_bound_m_s == pytest.approx(43*CIRCUMFERENCE/60.)
    value = at_age(relative, stamp, .0534, rpm=21.)
    assert value == original and value[1] == 9
    assert timing.depth_expires_at == pytest.approx(stamp+.25)
    # Old write is a cost frozen at admission, not a new reader permission.
    relative.owner._action_runtime.continuation_executed_speed_bound_rpm = lambda *_: 200.
    assert at_age(relative, stamp, .055, rpm=21.) == value
    assert at_age(relative, stamp, .056, rpm=44.) is None


def test_without_completed_history_physical_overspeed_still_vetoes(relative):
    stamp, _ = publish(relative, distance=1.912, percent=9, rpm=20.,
                       stamp=NOW-.0508, target_speed=.229613, range_rate=-.04)
    assert at_age(relative, stamp, .0534, rpm=21.) is None


def test_cap172_encoder_102ms_old_does_not_turn_64_into_zero(relative):
    stamp, original = publish(relative, distance=2.142, percent=32, rpm=0.,
                              stamp=NOW-.0759, target_speed=.6, range_rate=.6)
    relative.clock.now = stamp+.084
    relative.feedback = replace(relative.feedback, timestamp=relative.clock.now-.1023)
    value = relative.owner._fresh_depth_linear_snapshot(1)
    assert value is not None and value[1] > 0 and value[1] <= original[1]
    assert value[3] == stamp
    assert at_age(relative, stamp, .15, rpm=8.) is not None
    assert at_age(relative, stamp, .251, rpm=8.) is None


@pytest.mark.parametrize('pair', [(-2., 0.), (-1., -1.), (0., -3.), (-3., 2.)])
def test_cap145_near_zero_turn_tail_does_not_revoke_good_depth(relative, pair):
    stamp, original = publish(relative, distance=2.016, percent=8, rpm=0.,
                              stamp=NOW-.028, target_speed=.2, range_rate=.2)
    relative.clock.now = stamp+.065
    relative.feedback = SteeringFeedback(timestamp=relative.clock.now, trustworthy=True,
                                         left_forward_rpm=pair[0], right_forward_rpm=pair[1])
    assert relative.owner._fresh_depth_linear_snapshot(1) == original
    # It may be captured for braking budget, not certified as parked or given
    # permission to skip the independent final wheel zero-cross guard.
    frame = replace(relative.frame(2.016, stamp=stamp), steering_feedback=relative.feedback)
    distance, speed = relative.owner._depth_continuation_evidence(frame, 1, 8, relative.clock.now)
    assert distance == 2.016
    assert speed == pytest.approx(16*CIRCUMFERENCE/60.)


@pytest.mark.parametrize('pair', [(0., -37.), (-1., 20.), (-3.01, 0.)])
def test_real_reversing_wheel_not_excused_as_small_tail(relative, pair):
    stamp, _ = publish(relative)
    relative.feedback = replace(relative.feedback, left_forward_rpm=pair[0], right_forward_rpm=pair[1])
    assert relative.owner._fresh_depth_linear_snapshot(1) is None
    frame = replace(relative.frame(2.016, stamp=stamp), steering_feedback=relative.feedback)
    assert relative.owner._depth_continuation_evidence(frame, 1, 8, relative.clock.now) == (None, None)


def test_cap863_feedback_arriving_during_read_is_not_from_future(relative):
    stamp, original = publish(relative, distance=3.)
    relative.clock.now = stamp+.0385
    before = relative.clock.now
    def cache_read():
        relative.clock.now = before+.004
        return replace(relative.feedback, timestamp=relative.clock.now)
    relative.owner._action_runtime.get_steering_feedback = cache_read
    assert relative.owner._fresh_depth_linear_snapshot(1, now=before) == original


@pytest.mark.parametrize('expired', ['depth', 'visibility'])
def test_feedback_read_cannot_hide_deadline_crossing(relative, expired):
    stamp, _ = publish(relative, distance=3.)
    relative.clock.now = stamp+.248 if expired == 'depth' else stamp+.08
    relative.owner._last_vision_control_ts = (relative.clock.now-.01 if expired == 'depth'
                                             else relative.clock.now-.248)
    before = relative.clock.now
    def cache_read():
        relative.clock.now = before+.004
        return replace(relative.feedback, timestamp=relative.clock.now)
    relative.owner._action_runtime.get_steering_feedback = cache_read
    assert relative.owner._fresh_depth_linear_snapshot(1, now=before) is None


def test_actual_future_feedback_still_invalid_after_clock_refresh(relative):
    publish(relative)
    relative.feedback = replace(relative.feedback, timestamp=relative.clock.now+.02)
    assert relative.owner._fresh_depth_linear_snapshot(1) is None


def test_visibility_is_captured_before_final_validation_clock(relative, monkeypatch):
    stamp, original = publish(relative, distance=3.)
    before = relative.clock.now
    def read_clock():
        # A new visual result can arrive between the clock read and further
        # arithmetic. The gate must use its already captured visibility time.
        relative.owner._last_vision_control_ts = before+.004
        return before
    monkeypatch.setattr(runtime.time, 'monotonic', read_clock)
    assert relative.owner._fresh_depth_linear_snapshot(1, now=before) == original
    assert original[3] == stamp


@pytest.mark.parametrize('forward_handoff', [False, True])
def test_small_reverse_tail_does_not_certify_stopped_at_final_writer(relative, forward_handoff):
    stamp, original = publish(relative, distance=3., percent=10, rpm=0.)
    action, backend = writer(relative)
    action.config.follow_forward_handoff_enable = forward_handoff
    action.get_steering_feedback = lambda: relative.feedback
    relative.feedback = replace(relative.feedback, left_forward_rpm=-2., right_forward_rpm=-2.)
    assert relative.owner._fresh_depth_linear_snapshot(1) == original
    action._service_follow_wheels()
    assert backend.pairs and backend.pairs[-1][:2] == (0, 0)
    assert action._visible_wheel_guard.pending_full_reverse
    assert action._visible_wheel_guard.quiet_count == 0
    assert relative.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp+.25)


def test_invalid_motion_does_not_borrow_relative_tail_or_physical_only_budget(relative):
    stamp, _ = publish(relative, distance=3.)
    controller = relative.owner._follow_controller
    controller.last_distance_pid_result.pi_motion_window_used = False
    controller.last_distance_pid_result.approach_closing_m_s = 2.
    frame = relative.frame(3., rpm=16., stamp=stamp)
    _, budget = relative.owner._depth_continuation_evidence(frame, 1, 8, relative.clock.now)
    assert budget == 2.  # Legacy closure reservation is unchanged.
    frame = replace(frame, steering_feedback=replace(frame.steering_feedback,
                    left_forward_rpm=-2., right_forward_rpm=0.))
    assert relative.owner._depth_continuation_evidence(frame, 1, 8, relative.clock.now) == (None, None)


def test_relative_budget_does_not_count_approaching_target_twice(relative):
    publish(relative, distance=3., percent=20, rpm=40., target_speed=-.188,
            range_rate=-.7325, result_changes={'approach_closing_m_s': .7325})
    timing = relative.owner._depth30_linear_timing
    assert timing.continuation_speed_bound_m_s == pytest.approx(40*CIRCUMFERENCE/60.)
    assert timing.continuation_motion.target_speed_bound_m_s == -.188


def test_cap261_real_closing_momentum_still_requires_braking(relative):
    stamp, _ = publish(relative, distance=2.2, percent=27, rpm=53.,
                       target_speed=-.0183509, range_rate=-.733)
    linear = relative.owner._depth30_linear_snapshot
    timing = replace(relative.owner._depth30_linear_timing, continuation_distance_m=1.974)
    relative.owner._publish_depth_linear_pair(linear, timing)
    relative.clock.now = stamp+.08
    relative.feedback = replace(relative.feedback, timestamp=relative.clock.now)
    assert relative.owner._fresh_depth_linear_snapshot(1) is None
