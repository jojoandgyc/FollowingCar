"""Crossing a deadline inside a reader remains physical expiry, not bad physics."""
from dataclasses import replace

import pytest

from test_depth_authority_300 import authority300, setup, owner, seed300
from test_depth_authority_250 import advance, decide_commit


def test_continuation_boundary_has_physical_reason_and_new_sample_can_recover(authority300):
    a = authority300
    stamp, grant = seed300(a, distance=2.5, rpm=26.)
    timing = a.owner._depth30_linear_timing
    advance(a, stamp+.301)
    ok, reason = a.owner._depth_forward_continuation_safe(
        grant, timing, a.clock.now, feedback=a.feedback, shared_braking=True)
    assert not ok and reason == "physical_depth_expired"
    a.controller.suspend_longitudinal_authority(a.clock.now, "lateral_depth:"+reason)
    assert not a.controller._distance_pid._distance_pi._expiry_recovery_forbidden
    _, actions, accepted = decide_commit(a, a.frame(2.49, rpm=26., stamp=a.clock.now-.01))
    result = a.controller.last_distance_pid_result
    # Pure distance now uses the same fresh-sample ramp on either side of an
    # old deadline, not the separate short-expiry recovery branch.
    assert accepted and result.pi_fresh_grant_recovery_used
    assert not result.pi_depth_expiry_recovery_used
    assert 26 < result.output_rpm <= min(38., result.approach_cap_rpm)
    assert any(x.kind == "forward" and x.speed_percent*2 > 26 for x in actions)
    assert a.owner._depth30_linear_snapshot[3] != stamp
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(
        a.owner._depth30_linear_snapshot[3]+.30)


@pytest.mark.parametrize("earlier", ["identity_lost", "emergency_stop", "lateral_depth:invalid_braking_model"])
def test_physical_expiry_cannot_launder_prior_safety_withdrawal(authority300, earlier):
    a = authority300
    stamp, _ = seed300(a, distance=2.5, rpm=26.)
    a.controller.suspend_longitudinal_authority(stamp+.10, earlier)
    a.controller.suspend_longitudinal_authority(stamp+.301, "lateral_depth:physical_depth_expired")
    assert a.controller._distance_pid._distance_pi._expiry_recovery_forbidden


def test_expiry_label_before_deadline_is_not_recovery_credit(authority300):
    a = authority300
    stamp, _ = seed300(a)
    a.controller.suspend_longitudinal_authority(stamp+.20, "lateral_depth:physical_depth_expired")
    assert a.controller._distance_pid._distance_pi._expiry_recovery_forbidden
