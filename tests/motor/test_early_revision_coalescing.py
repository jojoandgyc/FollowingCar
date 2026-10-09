"""Metadata-only producer updates; fake serial driver and clock, no devices."""
import pytest

from test_follow_wheel_periodic import setup_periodic
from test_visible_wheel_continuity import feedback


def revision_updates(monkeypatch, *, change=None):
    runtime, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    state[:] = [32., 10., 10.25, 10.25]
    stamp = [10.]
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: (
        ("forward", state[0], uid, stamp[0]) if clock[0] < state[2] else None)
    runtime._service_follow_wheels()
    assert driver.pairs == [(42, -22)]
    driver.pairs.clear()
    clock[0] += .05
    calls = {"feedback": 0, "safety": 0}

    def read():
        calls["feedback"] += 1
        owner._lateral_yaw_revision += 1
        sample = feedback(clock[0], 20, 20)
        if calls["feedback"] == 2 and change:
            change(runtime, owner, clock, state, stamp, sample)
        return sample

    runtime.get_steering_feedback = read
    return runtime, owner, driver, clock, state, stamp, calls


def test_same_pair_second_revision_keeps_current_forward_grant(monkeypatch, caplog):
    runtime, owner, driver, clock, _, _, calls = revision_updates(monkeypatch)
    with caplog.at_level("INFO"):
        runtime._service_follow_wheels()
    assert driver.pairs == [(42, -22)]
    assert not driver.stops
    assert calls["feedback"] == 2
    assert runtime._follow_wheel_clock.last_axes == owner._follow_wheel_axes(clock[0])
    assert "follow_wheel_revision_coalesced" in caplog.text
    assert "FOLLOW20_AUTHORITY_CHANGED" not in caplog.text


@pytest.mark.parametrize("veto", ["new_sample", "base", "yaw", "expired_depth",
    "feedback_stale", "feedback_error", "feedback_reverse", "pending_reverse",
    "uid", "search", "explicit_stop", "shutdown", "handoff", "brake_hold"])
def test_metadata_adoption_cannot_bypass_changed_authority_or_guards(monkeypatch, veto):
    def change(runtime, owner, clock, state, stamp, sample):
        if veto == "new_sample": stamp[0] += .01
        elif veto == "base": state[0] += 1
        elif veto == "yaw": state[1] -= 1
        elif veto == "expired_depth": state[2] = clock[0] - .01
        elif veto == "feedback_stale": sample.timestamp -= .151
        elif veto == "feedback_error": sample.left_error = 1
        elif veto == "feedback_reverse": sample.left_forward_rpm = -20
        elif veto == "pending_reverse": runtime._visible_wheel_guard.pending_full_reverse = True
        elif veto == "uid": owner._follow_controller.active_target_id = 2
        elif veto == "search": owner.search_state = "searching"
        elif veto == "explicit_stop": owner._explicit_stop_requested = True
        elif veto == "shutdown": owner._runtime_shutdown_requested = True
        elif veto == "handoff": owner._search_handoff_uid = 1
        elif veto == "brake_hold": owner._brake_hold_active = True

    runtime, _, driver, _, _, _, calls = revision_updates(monkeypatch, change=change)
    runtime._service_follow_wheels()
    assert all(pair == (0, 0) for pair in driver.pairs)
    assert calls["feedback"] == 2


@pytest.mark.parametrize("when", ["feedback", "final_safety"])
def test_intervening_stop_receipt_and_mode_keep_ownership(monkeypatch, when):
    def stop(runtime, *_):
        runtime.backend.send_stop("concurrent_stop", mode="emergency")

    runtime, _, driver, _, _, _, calls = revision_updates(
        monkeypatch, change=stop if when == "feedback" else None)
    if when == "final_safety":
        def safety(_):
            calls["safety"] += 1
            if calls["safety"] == 3:
                stop(runtime)
            return False
        runtime.hard_stop_check = safety
    runtime._service_follow_wheels()
    assert driver.stops == [1]
    assert driver.pairs == []
    assert runtime.backend.last_speed_receipt is None


def test_third_revision_is_not_coalesced_again(monkeypatch):
    runtime, owner, driver, _, _, _, calls = revision_updates(monkeypatch)

    def safety(_):
        calls["safety"] += 1
        if calls["safety"] == 3:
            owner._lateral_yaw_revision += 1
        return False

    runtime.hard_stop_check = safety
    runtime._service_follow_wheels()
    assert driver.pairs == [(0, 0)]
    assert calls == {"feedback": 2, "safety": 3}


@pytest.mark.parametrize("change", ["new_sample", "expired_depth", "hazard"])
def test_final_safety_and_physical_grant_are_rechecked_after_adoption(monkeypatch, change):
    runtime, _, driver, clock, state, stamp, calls = revision_updates(monkeypatch)

    def safety(_):
        calls["safety"] += 1
        if calls["safety"] == 3:
            if change == "new_sample": stamp[0] += .01
            elif change == "expired_depth": state[2] = clock[0] - .01
            else: return True
        return False

    runtime.hard_stop_check = safety
    runtime._service_follow_wheels()
    if change == "hazard":
        assert driver.stops == [1]
        assert driver.pairs == []
    else:
        assert driver.pairs == [(0, 0)]


def test_no_completed_receipt_cannot_adopt_metadata(monkeypatch):
    runtime, _, driver, _, _, _, _ = revision_updates(monkeypatch)
    runtime.backend.last_speed_receipt = None
    runtime._service_follow_wheels()
    assert driver.pairs == [(0, 0)]


def test_coalescing_audit_is_logged_after_motor_write(monkeypatch):
    runtime, _, driver, _, _, _, _ = revision_updates(monkeypatch)
    original = runtime.logger.info

    def info(message, *args, **kwargs):
        if message.startswith("follow_wheel_revision_coalesced"):
            assert driver.pairs == [(42, -22)]
        original(message, *args, **kwargs)

    monkeypatch.setattr(runtime.logger, "info", info)
    runtime._service_follow_wheels()
