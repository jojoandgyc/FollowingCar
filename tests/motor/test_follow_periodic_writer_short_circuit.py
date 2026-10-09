"""The periodic writer must not re-run a discarded Depth admission."""

from types import SimpleNamespace

import pytest

from car_control_modular.action_runtime import MotionActionRuntime


def test_periodic_write_skips_redundant_active_read_before_write_check():
    class ReachedWriteCheck(Exception):
        pass

    def redundant_active_read():
        raise AssertionError("active read would re-evaluate Depth under motor lock")

    def stop_at_write_check():
        raise ReachedWriteCheck

    runtime = SimpleNamespace(
        _near_yaw_park_blocks_write=lambda _label: False,
        _periodic_follow_writing=True,
        _periodic_follow_active=redundant_active_read,
        _visible_wheel_control_active=stop_at_write_check,
    )
    with pytest.raises(ReachedWriteCheck):
        MotionActionRuntime._plan_follow_wheel_targets(
            runtime, 20, 20, "FOLLOW20")
