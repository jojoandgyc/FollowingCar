"""CAP361/378: temporary missing relative motion is not proof of recentering.

These are pure PID tests, not evidence for extending motion/identity leases.
"""
from dataclasses import replace

import pytest

from car_control_modular.control_types import SteeringFeedback
from car_control_modular.steering_pid import VisualSteeringPid, VisualSteeringPidConfig


def config():
    return VisualSteeringPidConfig(enabled=True, image_error_only=True,
        image_brake_assist=True, image_capture_motion=True,
        execution_response_trial_sec=.35, image_slow_brake_continuity_sec=.5,
        camera_hfov_deg=66., deadband_deg=3., dynamic_large_error_deg=10.,
        max_correction_rpm=10., max_yaw_rate_dps=35.,
        predictive_brake_decel_dps2=60., predictive_brake_margin_deg=1.25,
        predictive_brake_response_sec=.05)


def feedback(now, left=20., right=42., raw=-34.485, filtered=-28.215):
    return SteeringFeedback(timestamp=now-.02, trustworthy=True,
        left_forward_rpm=left, right_forward_rpm=right,
        raw_yaw_rate_right_dps=raw, yaw_rate_right_dps=filtered)


def call(pid, now=10., age=.157, error=-15., rate=None, fb=None, **kwargs):
    return pid.update(.5+error/66., 20,
        feedback(now) if fb is None else fb, now=now,
        visual_age_sec=age, target_image_rate_dps=rate, **kwargs)


@pytest.mark.parametrize("sign", [-1, 1])
def test_cap378_missing_rate_reduces_measured_turn_instead_of_cancelling(sign):
    p=VisualSteeringPid(config())
    f=feedback(10, left=31+sign*11, right=31-sign*11,
        raw=sign*34.485, filtered=sign*28.215)
    r=call(p, error=sign*21.9045, fb=f)
    assert r.correction_rpm == sign*2
    assert abs(r.correction_rpm) <= abs(f.left_forward_rpm-f.right_forward_rpm)/8
    assert r.brake_continuity_reason == "fresh_position_deceleration"
    assert r.brake_continuity_remaining_sec == .15
    assert r.brake_reduction_rpm > 0
    assert not r.predictive_braking and not r.target_rate_valid
    assert r.base_rpm == 20


def test_cap361_old_rate_is_only_history_never_new_rate_or_authority():
    p=VisualSteeringPid(config())
    cap=29818.682327744
    first=call(p, now=cap+.15, age=.15, error=-7.7022, rate=-32.54,
        fb=feedback(cap+.15, 24, 23, 1.6272, 3.2544))
    assert first.target_rate_valid
    now=29818.965
    r=call(p, now=now, age=now-cap, error=-7.7022, rate=-32.54,
        fb=feedback(now, 33, 41, -12.54, -12.54))
    assert r.correction_rpm == -1
    assert r.stopping_distance_deg == pytest.approx(8.5607100902)
    assert r.brake_continuity_reason == "recent_outward_deceleration"
    assert not r.target_rate_valid
    # Giving an old rate directly to a fresh controller cannot manufacture
    # the earlier valid observation.
    fresh=call(VisualSteeringPid(config()), now=now, age=now-cap,
        error=-7.7022, rate=-32.54, fb=feedback(now,33,41,-12.54,-12.54))
    assert fresh.correction_rpm == 0
    expired=call(p,now=cap+.351,age=.351,error=-7.7022,rate=-32.54,
        fb=feedback(cap+.351,33,41,-12.54,-12.54))
    assert expired.correction_rpm == 0


@pytest.mark.parametrize("new_frames", [False, True])
def test_episode_does_not_restart_on_duplicates_or_new_unknown_frames(new_frames):
    p=VisualSteeringPid(config())
    assert call(p, age=.05).correction_rpm == -2
    for dt in [.05, .10, .149]:
        r=call(p, now=10+dt, age=.05 if new_frames else .05+dt)
        assert r.correction_rpm == -2
        assert r.brake_continuity_remaining_sec == pytest.approx(.15-dt)
    for dt in [.151, .2, .4]:
        r=call(p, now=10+dt, age=.05 if new_frames else .05+dt)
        assert r.correction_rpm == 0
    assert p._image_brake_observation_started == 10


def test_only_new_measured_outward_evidence_rearms_after_budget_spent():
    p=VisualSteeringPid(config())
    call(p, age=.05)
    assert call(p, now=10.16, age=.05).correction_rpm == 0
    observed=call(p, now=10.2, age=.05, rate=-25.)
    assert observed.target_rate_valid
    assert observed.correction_rpm <= -4
    resumed=call(p, now=10.25, age=.05)
    assert resumed.correction_rpm == -2
    assert resumed.brake_continuity_remaining_sec == .15


def test_duplicate_outward_rate_cannot_rearm_budget():
    p=VisualSteeringPid(config())
    call(p, now=10, age=.05, rate=-25)
    call(p, now=10.02, age=.07)
    assert call(p, now=10.18, age=.23).correction_rpm == 0
    # Same physical sample, now with a cached outward rate, isn't new proof.
    call(p, now=10.19, age=.24, rate=-25)
    assert call(p, now=10.2, age=.25).correction_rpm == 0


@pytest.mark.parametrize("rate", [5., 30., 80.])
def test_measured_inward_motion_immediately_overrides_continuity(rate):
    p=VisualSteeringPid(config())
    call(p, age=.05)
    r=call(p, now=10.02, age=.04, rate=rate)
    assert r.correction_rpm == 0
    assert r.predictive_braking
    assert call(p, now=10.04, age=.04).correction_rpm == 0


def test_cropped_turnaround_cue_is_not_treated_as_missing_motion():
    p=VisualSteeringPid(config())
    call(p)
    r=call(p,now=10.02,age=.05,braking_image_rate_dps=15.)
    assert r.correction_rpm == 0
    assert call(p,now=10.04,age=.05).correction_rpm == 0


@pytest.mark.parametrize("age,rate", [(.251,None),(.351,None),(-.01,None),
    (.1,float('nan')),(.1,130.)])
def test_invalid_or_expired_visual_data_cannot_start_observation(age,rate):
    r=call(VisualSteeringPid(config()),age=age,rate=rate)
    if age >= 0:
        assert r.correction_rpm == 0
    assert r.brake_continuity_reason == "none"


def test_center_high_yaw_and_policy_zero_still_cancel_immediately():
    p=VisualSteeringPid(config())
    call(p)
    assert call(p, now=10.01, age=.02, error=-2).correction_rpm == 0
    assert call(p, now=10.02, age=.02).correction_rpm == 0
    q=VisualSteeringPid(config())
    call(q)
    assert call(q, now=10.01, age=.02,
        fb=feedback(10.01, left=10,right=40,raw=-47,filtered=-45)).correction_rpm == 0
    assert call(q, now=10.02, age=.02).correction_rpm == 0
    assert call(VisualSteeringPid(config()),max_correction_override_rpm=0).correction_rpm == 0


def test_out_of_order_capture_and_reverse_clock_do_not_rearm():
    p=VisualSteeringPid(config())
    call(p, age=.05)
    r=call(p,now=10.01,age=.10)
    assert r.correction_rpm == 0
    assert call(p, now=9.99, age=.03).correction_rpm == 0


@pytest.mark.parametrize("changes", [dict(trustworthy=False),dict(timestamp=9.8),
    dict(left_forward_rpm=42,right_forward_rpm=20)])
def test_untrusted_stale_or_sign_inconsistent_feedback_cannot_enable_bridge(changes):
    r=call(VisualSteeringPid(config()),fb=replace(feedback(10),**changes))
    assert r.brake_continuity_reason == "none"
    assert r.brake_continuity_remaining_sec == 0


def test_reset_drops_historical_rate_and_episode():
    p=VisualSteeringPid(config())
    call(p,rate=-25)
    call(p)
    p.reset()
    assert p._image_outward_observation is None
    assert p._image_brake_observation_started is None
    assert p._image_brake_observation_spent
    assert call(p,now=10.02,age=.1).correction_rpm == 0
    # A new measured outward sample, not reset itself, permits another episode.
    call(p,now=10.04,age=.1,rate=-25)
    assert call(p,now=10.06,age=.1).correction_rpm == -2


def test_unused_budget_survives_initial_reset_but_used_budget_never_restarts():
    p=VisualSteeringPid(config())
    p.reset()
    assert call(p).correction_rpm == -2
    for i in range(1,5):
        p.reset()
        assert call(p,now=10+i*.1,age=.1).correction_rpm == 0


def test_trial_disabled_keeps_original_predictive_stop():
    p=VisualSteeringPid(replace(config(),execution_response_trial_sec=0))
    # Old model's stopping angle also exceeds this error.
    r=call(p,error=-12.)
    assert r.correction_rpm == 0
    assert r.brake_continuity_reason == "none"


def test_bridge_does_not_enable_buildup_or_add_forward_speed():
    from car_control_modular.turn_buildup import steering_buildup_allowed
    r=call(VisualSteeringPid(config()))
    assert r.correction_rpm == -2
    assert not steering_buildup_allowed(r)
    assert r.requested_base_rpm == r.base_rpm == 20


@pytest.mark.parametrize("base", [0, -20])
def test_no_new_permission_for_pivot_or_reverse(base):
    r=VisualSteeringPid(config()).update(.5-15/66,base,feedback(10),now=10,
        visual_age_sec=.157)
    assert r.correction_rpm == 0
    assert r.brake_continuity_reason == "none"


def test_controller_main_and_refresh_do_not_apply_old_minimum_floor(monkeypatch):
    from car_control_modular.controllers import FollowSafetyController, FollowPolicyConfig
    from test_cap443_capture_braking import sample
    c=FollowSafetyController(FollowPolicyConfig(visible_steering_pid_enable=True,
        visible_steering_pid_image_error_only=True,
        visible_steering_pid_image_brake_assist=True,
        visible_steering_pid_execution_response_trial_sec=.35,
        visible_steering_pid_image_capture_motion=True, forward_max_rpm=200))
    c.active_target_id=1
    c._visual_steering_pid.config=config()
    monkeypatch.setattr(c,"_visible_base_forward_percent",lambda *a,**kw:10)
    monkeypatch.setattr(c,"_capture_steering_observation",lambda *a,**kw:None)
    target,frame=sample(378,9.843,.1681)
    frame=replace(frame,steering_feedback=feedback(10))
    action=c._pid_action_for_visible_target(target,frame,10)
    assert action.kind == "steer_left" and action.steer_correction_rpm == 2
    assert action.speed_percent == 10
    r=c.refresh_visible_lateral_pid(x_ratio=.1681,base_rpm=20,
        feedback=feedback(10.05),now=10.05,visual_age_sec=.207)
    assert r.correction_rpm == -2
    assert r.brake_continuity_elapsed_sec == pytest.approx(.05)
    r=c.refresh_visible_lateral_pid(x_ratio=.1681,base_rpm=20,
        feedback=feedback(10.16),now=10.16,visual_age_sec=.317)
    assert r.correction_rpm == 0
