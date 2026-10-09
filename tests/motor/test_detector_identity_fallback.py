"""Detector identity revocation must survive the legacy motor fallback."""

import pytest

from test_depth_drive_rpm import make_runtime
from test_follow_wheel_periodic import setup_periodic
from test_visible_wheel_continuity import visible_runtime


@pytest.mark.parametrize("lease, expected", [
    (False, []),
    (None, [(24, -24)]),
])
def test_queued_drive_after_visible_exit_does_not_escape_detector_lease(
    monkeypatch, lease, expected,
):
    runtime, owner, driver, symbols, _, _ = setup_periodic(monkeypatch)
    runtime._service_follow_wheels()
    owner._vision_control_state = "lost_confirming"
    owner._detector_identity_lease = lease
    runtime._service_follow_wheels()
    assert driver.pairs == [(24, -24), (0, 0)]

    # The old forward command can still be dispatched while the producer is
    # replacing its queue. It must not re-enter speed mode after the zero.
    driver.pairs.clear()
    runtime.send_robot_command(symbols.forward)
    assert driver.pairs == expected


@pytest.mark.parametrize("lease, allowed", [(False, False), (None, True)])
def test_raw_steer_fallback_requires_full_identity(monkeypatch, lease, allowed):
    runtime, owner, driver, symbols, _ = visible_runtime(monkeypatch)
    owner._vision_control_state = "lost_confirming"
    owner._detector_identity_lease = lease
    owner._current_steer_base_percent = 20
    owner._current_steer_correction_rpm = 5
    owner.current_command = symbols.steer_right

    runtime.send_robot_command(symbols.steer_right)
    assert bool(driver.pairs) is allowed


def test_percent_fallback_blocks_motion_but_preserves_zero_and_turn():
    runtime, owner, driver, symbols = make_runtime(raw_mode=False)
    owner._detector_identity_lease = False

    runtime.send_robot_command(symbols.forward)
    assert driver.pairs == []
    assert not runtime.send_percent_diff(20, 0x01, 10, 0x01, "STEER")
    assert driver.pairs == []

    assert runtime.send_percent_drive(0)
    assert driver.pairs == [(0, 0)]
    assert runtime.send_percent_diff(20, 0x01, 20, 0x02, "TURN")
    assert len(driver.pairs) == 2

    owner._detector_identity_lease = None
    runtime.send_robot_command(symbols.forward)
    assert driver.pairs[-1] == (24, -24)
