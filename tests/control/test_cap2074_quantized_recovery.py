"""CAP2074: 1 RPM is zero percent, not an issued positive grant.

Use the real controller and depth admission with fake clocks/feedback.
No camera, serial port, or wheel commands are used.
"""
from dataclasses import replace

import pytest

from test_depth_authority_250 import authority, advance, decide_commit
from test_distance_pi_controller import configured
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


@pytest.mark.parametrize("rpm", [0., .5, 1., 1.5])
@pytest.mark.parametrize("active_profile", [False, True])
def test_fresh_samples_can_ramp_from_quantized_zero_without_restart(authority, setup, rpm, active_profile):
    a = authority
    if active_profile:
        _, a.controller, _ = configured(
            setup, target_distance_m=1.4, distance_pi_kp_per_sec=3.,
            distance_pi_launch_request_rpm=180., distance_pi_launch_full_error_m=.5,
            depth_longitudinal_sample_max_age_sec=.25,
        )
        a.owner._follow_controller = a.controller
        a.controller._live_longitudinal_authority_reader = a.owner._fresh_depth_linear_snapshot
    first = a.frame(2.253, rpm=rpm, capture_frame_id=2074)
    _, actions, accepted = decide_commit(a, first)
    assert accepted and all(x.speed_percent == 0 for x in actions)
    assert a.owner._depth30_linear_snapshot is None
    previous = a.controller.last_distance_pid_result
    assert previous.output_rpm == int(rpm)
    assert a.controller._distance_pid._distance_pi._last_output_rpm == 0
    # A physically newer sample arrives 80ms later, while the prior sample
    # is still timely. The prior zero did not lose any positive motor grant.
    advance(a, a.clock.now + .08)
    current = a.frame(2.225, rpm=rpm, capture_frame_id=2076)
    _, actions, accepted = decide_commit(a, current)
    result = a.controller.last_distance_pid_result
    assert accepted and result.pi_status == "tracking"
    approved = max(x.speed_percent*2 for x in actions if x.kind == "forward")
    assert 0 < approved <= 240*.08
    assert result.pi_execution_anchor_rpm == 0
    assert a.owner._depth30_linear_snapshot[3] == current.distance_state.sample_timestamp
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(a.clock.now+.25)


@pytest.mark.parametrize("event", ["expired", "revoked", "rejected", "parked"])
def test_real_execution_reset_still_restarts_from_measured_speed(authority, event):
    a = authority
    first = a.frame(2.253, rpm=1.)
    decide_commit(a, first)
    advance(a, a.clock.now + (.251 if event == "expired" else .08))
    if event == "revoked":
        a.controller.suspend_longitudinal_authority(a.clock.now, "identity_lost")
    elif event == "rejected":
        a.controller.reject_longitudinal_sample(first.distance_state.sample_timestamp)
    elif event == "parked":
        a.controller.set_normal_parking(True, 1)
    _, actions, _ = decide_commit(a, a.frame(2.225, rpm=1.))
    assert a.controller.last_distance_pid_result.pi_status in {"recovering", "parked_preview"}
    assert all(x.speed_percent == 0 for x in actions)
    assert a.owner._depth30_linear_snapshot is None


@pytest.mark.parametrize("event", ["duplicate", "older", "stale_new", "near", "hazard"])
def test_quantized_zero_cannot_authorize_stale_or_unsafe_evidence(authority, event):
    a = authority
    first = a.frame(2.253, rpm=1.)
    decide_commit(a, first)
    old_stamp = first.distance_state.sample_timestamp
    advance(a, a.clock.now + (.20 if event == "stale_new" else .08))
    stamp = dict(duplicate=old_stamp, older=old_stamp-.01,
                 stale_new=old_stamp+.001).get(event, a.clock.now)
    current = a.frame(1.4 if event == "near" else 2.225, rpm=1., stamp=stamp)
    if event == "hazard":
        current = replace(current, hazard=replace(current.hazard, active=True))
    _, actions, _ = decide_commit(a, current)
    assert not any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    assert a.owner._depth30_linear_snapshot is None


@pytest.mark.parametrize("rpm", [2., 20.])
def test_executable_positive_request_without_live_grant_still_recovers(authority, rpm):
    a = authority
    first = a.frame(2.253, rpm=rpm)
    decide_commit(a, first)
    assert a.owner._depth30_linear_snapshot is not None
    # No execution receipt can prove the issued nonzero command survived.
    a.owner._depth30_continuation_veto = (1, first.distance_state.sample_timestamp)
    a.owner._depth30_linear_snapshot = None
    advance(a, a.clock.now+.08)
    _, actions, _ = decide_commit(a, a.frame(2.225, rpm=0.))
    assert a.controller.last_distance_pid_result.pi_status == "recovering"
    assert all(x.speed_percent == 0 for x in actions)
