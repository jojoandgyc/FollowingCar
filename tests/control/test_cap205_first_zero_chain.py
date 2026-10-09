"""Recorded first zero episode: braking, publication, then fresh PI recovery.

No closed-loop trajectory is invented: the negative/near-zero encoder tail is
the recorded consequence of the old run, not a forecast after fixing it.
"""
from dataclasses import replace

import pytest
import request_0513_modular as runtime

from car_control_modular.sample_braking import SampleBrakingAssessment
from test_depth_authority_250 import authority, advance, decide_commit, seed, writer
from test_distance_pi_controller import configured
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


def recorded_controller(a, setup):
    _, a.controller, _ = configured(
        setup, target_distance_m=1.4, forward_start_distance_m=1.48,
        forward_stop_distance_m=1.43, near_distance_rotate_only_distance_m=1.43,
        distance_pi_kp_per_sec=3., distance_pi_ki_per_sec2=.4,
        distance_pi_launch_request_rpm=180., distance_pi_launch_full_error_m=.5,
        distance_approach_deceleration_m_s2=1., distance_approach_response_delay_sec=.15,
        depth_longitudinal_sample_max_age_sec=.25,
        distance_pi_braking_stop_distance_m=1.1,
        distance_target_motion_control_enable=False,
        visible_steering_pid_max_correction_rpm=10.)
    a.owner._follow_controller = a.controller
    a.controller._live_longitudinal_authority_reader = a.owner._fresh_depth_linear_snapshot
    a.controller._braking_execution_bound_reader = lambda uid, now: 1.


def test_first_stop_precedes_cap205_and_is_a_current_momentum_veto():
    # CAP192 range; first 0/0 at 23:18:21.929, before CAP205 capture.
    # Original physical sample/assessment and the encoder at the read veto.
    assessment = SampleBrakingAssessment(
        uid=1, sample_timestamp=12855.158025714, checked_at=12855.224979272,
        distance_m=1.5714000000000001, travel_bound_rpm=44., outer_rpm=42.,
        feedback_timestamp=12855.223178277, target_speed_m_s=0., stop_distance_m=1.1,
        circumference_m=.816814, deceleration_m_s2=1., response_delay_sec=.15,
        max_rpm=200., outer_allowance_rpm=10.)
    budget = assessment.budget(12855.285114424, 47., authorized_rpm=36.,
                               execution_bound_rpm=46.)
    assert budget.reason == "shared_braking_momentum"
    assert budget.cap_rpm == 0
    assert budget.required_stop_m > budget.margin_m > 0


@pytest.mark.parametrize("wheel_pair", [(-1., 0.), (0., 0.), (1., 1.), (-4., 0.), (-3., 7.)])
def test_recorded_near_fresh_depth_restarts_with_quiet_tail_but_not_real_reverse(
        authority, setup, wheel_pair):
    """Do not invent a different encoder tail for the actual CAP205 frame.

    A <=3RPM all-wheel quiet tail can propose a NEW bounded request; the
    executor still owns actual reverse transitions. Genuine opposite wheel
    motion remains rejected. Near-range recovery must not wait for 1.9m.
    """
    a = authority
    recorded_controller(a, setup)
    stamp, _ = seed(a, distance=1.5332, rpm=24.)
    # CAP202's accumulated PI memory is an observed starting-state datum.
    a.controller._distance_pid._distance_pi.integral_m_s = .03890
    offset = 12855.806868377-12855.71038025
    age = .1539
    advance(a, stamp+offset+age)
    current = a.frame(1.5514, stamp=stamp+offset, rpm=.5*sum(wheel_pair),
                      capture_frame_id=205)
    current = replace(current, distance_state=replace(current.distance_state,
        raw_distance_m=1.587), steering_feedback=replace(current.steering_feedback,
            timestamp=a.clock.now-.0274, left_forward_rpm=wheel_pair[0],
            right_forward_rpm=wheel_pair[1], raw_yaw_rate_right_dps=-1.57))
    _, actions, _ = decide_commit(a, current)
    result = a.controller.last_distance_pid_result
    assert result.approach_cap_rpm > 38
    assert result.pi_demand_rpm > 26
    if min(wheel_pair) < 0 and max(map(abs, wheel_pair)) > 3:
        assert result.pi_final_limit_reason == "execution_recovery"
        assert not result.pi_fresh_grant_recovery_used
        assert not result.pi_depth_expiry_recovery_used
        # Positive mean rotation can remain the measured request anchor, but
        # it cannot acquire new recovery acceleration credit or bypass the
        # writer's real reversal guard.
        assert result.output_rpm <= max(0., sum(wheel_pair)/2)
    else:
        assert result.pi_final_limit_reason == "fresh_grant_recovery_step"
        assert result.pi_fresh_grant_recovery_used
        assert not result.pi_depth_expiry_recovery_used
        assert 0 < result.output_rpm <= max(0., sum(wheel_pair)/2)+12.
        assert any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
        assert a.owner._depth30_linear_snapshot[3] == stamp+offset


def test_next_independent_sample_can_leave_near_recovery_without_waiting_for_19m(
        authority, setup):
    a = authority
    recorded_controller(a, setup)
    stamp, _ = seed(a, distance=1.5332, rpm=24.)
    advance(a, stamp+.251)
    first = a.frame(1.5514, stamp=stamp+.09649, rpm=0., capture_frame_id=205)
    decide_commit(a, first)
    first_request = a.controller.last_distance_pid_result.output_rpm
    assert 0 < first_request <= 12
    # The very first fresh sample now progresses; the next sample continues
    # its real-time ramp without requiring a second recovery attempt.
    advance(a, stamp+.301)
    current = a.frame(1.58, stamp=stamp+.25, rpm=0., capture_frame_id=207)
    _, actions, accepted = decide_commit(a, current)
    assert accepted
    assert first_request < a.controller.last_distance_pid_result.output_rpm <= first_request+12
    assert any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    action, backend = writer(a)
    action.get_steering_feedback = lambda: a.feedback
    action._service_follow_wheels()
    assert backend.pairs[-1][0] > 0 and backend.pairs[-1][1] < 0
