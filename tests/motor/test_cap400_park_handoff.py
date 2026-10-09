"""Production producer -> production executor -> fake motor, no serial I/O."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "control"))
from test_lateral_zero_runtime import owner, _intent, NOW
from test_depth_drive_rpm import make_runtime
from test_visible_wheel_continuity import feedback


@pytest.mark.parametrize("initial_percent", [0, 3])
def test_pivot_prediction_reaches_normal_writer_and_cannot_be_overwritten(owner, initial_percent):
    executor, _, driver, symbols = make_runtime(max_rpm=200)
    executor.owner = owner
    owner.motor_io_lock = executor.backend.io_lock
    owner._action_runtime = executor
    executor.get_steering_feedback = lambda: feedback(NOW-.04, -8, 8)
    if initial_percent:
        owner._depth30_linear_snapshot = ("forward", initial_percent, 1, NOW-.03)
    intent = _intent(owner, near_distance_mode=False, park_requested=True, hold_zero=True)
    owner._publish_lateral_zero(intent, "predictive_brake_coast")
    request = owner._near_yaw_park_request
    # The visual producer has not touched the motor; only this consumer writes.
    assert not driver.pairs and not driver.stops
    executor._service_follow_wheels()
    assert driver.stops == [1, 0]  # actual NORMAL call, not a pair of zero targets
    count = len(driver.pairs)
    for rpm in (6, 18, 22):
        owner._depth30_linear_snapshot = ("forward", rpm//2, 1, NOW-.01)
        owner._current_forward_percent = rpm//2
        executor.send_robot_command(symbols.forward)
        executor._service_follow_wheels()
        assert owner._near_yaw_park_request is request
        assert driver.stops == [1, 0]
        assert len(driver.pairs) == count
    assert not executor.near_yaw_park_release_ready(request, NOW-.01, NOW)
