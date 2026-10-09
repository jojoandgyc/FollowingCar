"""CAP366: a late same-UID yaw must not be erased by an old FOLLOW20 zero.

All motor and time sources are fakes; these tests never open a serial port.
"""

import pytest

from test_follow_wheel_periodic import setup_periodic
from test_visible_wheel_continuity import feedback


def late_yaw_at_write(monkeypatch, *, change="yaw"):
    rt, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    state[:] = [0., -4., 11., 11.]
    rt.get_steering_feedback = lambda: feedback(clock[0], 0, 0)
    rt._service_follow_wheels()
    assert driver.pairs == [(-4, -4)]

    clock[0] = 10.05
    state[1] = 0.  # An expired yaw initially makes this an ordinary zero plan.
    original_config = rt.config
    original_config.follow_cross_brake_enable = True
    injected = []
    danger = [False]
    rt.hard_stop_check = lambda _action=None: danger[0]

    class LateConfig:
        def __getattr__(self, name):
            if name == "follow_cross_brake_mode" and not injected:
                # This accessor is evaluated after the first zero-plan check,
                # just before the physical speed-target write.
                injected.append(True)
                state[1] = -4.
                owner._lateral_yaw_revision += 1
                if change == "identity":
                    owner._follow_controller.active_target_id = 2
                elif change == "stop":
                    owner._explicit_stop_requested = True
                elif change == "danger":
                    danger[0] = True
            return getattr(original_config, name)

    rt.config = LateConfig()
    return rt, owner, driver, injected


def test_late_same_uid_yaw_rebuilds_instead_of_writing_old_zero(monkeypatch, caplog):
    rt, _, driver, injected = late_yaw_at_write(monkeypatch)
    with caplog.at_level("INFO"):
        rt._service_follow_wheels()
    assert injected
    assert driver.pairs == [(-4, -4), (-4, -4)]
    assert not driver.stops
    assert "zero_superseded_at_write" in caplog.text


@pytest.mark.parametrize("change", ["identity", "stop"])
def test_late_yaw_cannot_cancel_lost_identity_or_explicit_stop(monkeypatch, change):
    rt, _, driver, injected = late_yaw_at_write(monkeypatch, change=change)
    rt._service_follow_wheels()
    assert injected
    assert driver.pairs[-1] == (0, 0)


def test_late_yaw_cannot_cancel_new_hard_danger(monkeypatch):
    rt, _, driver, injected = late_yaw_at_write(monkeypatch, change="danger")
    rt._service_follow_wheels()
    assert injected
    assert driver.stops  # Emergency STOP wins over ordinary zero and new yaw.
    assert driver.pairs == [(-4, -4)]
