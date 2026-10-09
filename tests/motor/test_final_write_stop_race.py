"""A late safety owner cannot be replaced by a speed or speed-zero write."""

import pytest
from types import SimpleNamespace

from test_follow_same_grant_speed_contraction import _live_depth_grant
from test_visible_wheel_continuity import feedback


@pytest.mark.parametrize("late_owner", ["explicit_stop", "shutdown", "motor_stop"])
def test_quiet_final_cap_cannot_write_after_late_stop_owner(monkeypatch, late_owner):
    runtime, owner, driver, _, _ = _live_depth_grant(monkeypatch)
    owner._depth_forward_continuation_required = lambda *_: True
    visited = [False]

    def final_cap(_linear, _timing, _now, *, feedback=None, quiet=False):
        assert quiet and feedback is not None
        if not visited[0]:
            visited[0] = True
            if late_owner == "explicit_stop":
                owner._explicit_stop_requested = True
            elif late_owner == "shutdown":
                owner._runtime_shutdown_requested = True
            else:
                # A completed STOP invalidates the old speed receipt even if
                # its state publisher has not yet acquired the control lock.
                runtime.backend.send_stop("concurrent_stop", mode="emergency")
        return 60, "same_grant_braking_margin"

    owner._depth_forward_continuation_limit = final_cap
    runtime._service_follow_wheels()

    assert visited[0]
    assert driver.pairs == []  # Neither old positive speed nor a speed-mode zero.
    assert driver.stops == ([1] if late_owner == "motor_stop" else [])


@pytest.mark.parametrize("late_change", ["grant", "uid", "timing", "ttl"])
def test_second_quiet_cap_cannot_outlive_authority(monkeypatch, late_change):
    runtime, owner, driver, clock, grant = _live_depth_grant(monkeypatch)
    owner._depth_forward_continuation_required = lambda *_: True
    if late_change == "timing":
        owner._depth30_linear_timing = SimpleNamespace(
            snapshot=owner._depth30_linear_snapshot,
            depth_expires_at=grant["stamp"] + .25,
            feedforward_expires_at=None,
            distance_only_percent=0)
    calls = [0]

    def final_cap(_linear, _timing, _now, *, feedback=None, quiet=False):
        assert quiet
        calls[0] += 1
        if calls[0] == 2:
            if late_change == "grant":
                owner._depth30_linear_snapshot = None
            elif late_change == "uid":
                owner._follow_controller.active_target_id = 2
            elif late_change == "timing":
                owner._depth30_linear_timing = SimpleNamespace(
                    snapshot=owner._depth30_linear_snapshot,
                    depth_expires_at=grant["stamp"] + .25,
                    feedforward_expires_at=None,
                    distance_only_percent=0)
            else:
                clock[0] = grant["stamp"] + .251
        return 60, "same_grant_braking_margin"

    owner._depth_forward_continuation_limit = final_cap
    current = owner._depth30_linear_snapshot
    assert not runtime._linear_packet_within_write_deadline(
        current, 1, 50., feedback=feedback(clock[0], 20, 20))
    assert calls[0] == 2
    assert driver.pairs == []
