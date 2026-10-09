"""Two producer updates during one guarded write; fake motor and clock only."""
from types import SimpleNamespace

import pytest

from car_control_modular.final_yaw_coalescing import contract_forward_yaw
from car_control_modular.lateral_intent import LateralIntentStore
from test_follow_wheel_periodic import setup_periodic
from test_visible_wheel_continuity import feedback


def two_updates(monkeypatch, *, sign=1, new_yaw=7, last_check=None,
                second_update=None, initial_yaw=10, sample_factory=None):
    runtime, owner, driver, symbols, clock, state = setup_periodic(monkeypatch)
    state[:] = [32., sign*initial_yaw, 10.25, 10.25]
    calls = {"feedback": 0, "safety": 0}
    sample = sample_factory() if sample_factory else feedback(clock[0], 20, 20)

    def read_cache():
        calls["feedback"] += 1
        if calls["feedback"] == 1:
            state[1] = sign*(initial_yaw-1)
            owner._lateral_yaw_revision += 1
        return sample

    def check_safety(_):
        calls["safety"] += 1
        if calls["safety"] == 3:
            state[1] = sign*new_yaw
            owner._lateral_yaw_revision += 1
            if second_update:
                second_update(runtime, owner, clock, state, sample)
        if calls["safety"] == 4 and last_check:
            return bool(last_check(runtime, owner, clock, state, sample))
        return False

    runtime.get_steering_feedback = read_cache
    runtime.hard_stop_check = check_safety
    return runtime, owner, driver, clock, state, calls


@pytest.mark.parametrize("sign,new_yaw", [(1, 7), (-1, 7), (1, 0), (-1, 0), (1, 9), (-1, 9)])
def test_second_same_base_yaw_contraction_writes_current_pair_once(monkeypatch, caplog, sign, new_yaw):
    runtime, owner, driver, clock, state, calls = two_updates(
        monkeypatch, sign=sign, new_yaw=new_yaw)
    with caplog.at_level("INFO"):
        runtime._service_follow_wheels()
    expected = (32+sign*new_yaw, -(32-sign*new_yaw))
    assert driver.pairs == [expected]
    assert not driver.stops
    assert calls == {"feedback": 3, "safety": 4}
    assert runtime._follow_wheel_clock.last_axes == owner._follow_wheel_axes(clock[0])
    assert runtime._follow_wheel_clock.last_axes == (1, 3, 32., sign*new_yaw)
    assert runtime._visible_wheel_guard.last_output == (expected[0], -expected[1])
    assert "follow_wheel_yaw_coalesced" in caplog.text
    assert "revision=3 base_rpm=32.0 yaw_rpm=%.1f" % (sign*new_yaw) in caplog.text
    assert "FOLLOW_FINAL_AUTHORITY_CHANGED" not in caplog.text


def test_same_straight_pair_with_new_revision_does_not_insert_zero(monkeypatch):
    runtime, owner, driver, clock, _, calls = two_updates(monkeypatch, initial_yaw=1, new_yaw=0)
    runtime._service_follow_wheels()
    assert driver.pairs == [(32, -32)]
    assert not driver.stops
    assert calls == {"feedback": 3, "safety": 4}
    assert runtime._follow_wheel_clock.last_axes == owner._follow_wheel_axes(clock[0]) == (1, 3, 32., 0.)


@pytest.mark.parametrize("veto", ["depth", "yaw", "feedback_age", "feedback_untrusted",
    "feedback_error", "feedback_nan", "uid", "search", "explicit", "third_yaw",
    "third_revision", "new_handoff"])
def test_last_check_cannot_extend_evidence_or_adopt_a_third_update(monkeypatch, veto):
    def change(runtime, owner, clock, state, sample):
        if veto == "depth": state[2] = 9.
        elif veto == "yaw": state[3] = 9.
        elif veto == "feedback_age": clock[0] += .151
        elif veto == "feedback_untrusted": sample.trustworthy = False
        elif veto == "feedback_error": sample.left_error = 1
        elif veto == "feedback_nan": sample.right_forward_rpm = float("nan")
        elif veto == "uid": owner._follow_controller.active_target_id = 2
        elif veto == "search": owner.search_state = "searching"
        elif veto == "explicit": owner._explicit_stop_requested = True
        elif veto == "third_yaw": state[1] = 6.
        elif veto == "third_revision": owner._lateral_yaw_revision += 1
        elif veto == "new_handoff": owner._search_handoff_uid = 1

    runtime, _, driver, _, _, calls = two_updates(monkeypatch, last_check=change)
    runtime._service_follow_wheels()
    assert driver.pairs == [(0, 0)]
    assert calls == {"feedback": 3, "safety": 4}
    assert runtime._follow_wheel_clock.last_axes is None


def test_danger_in_contraction_check_gets_emergency_stop(monkeypatch):
    runtime, _, driver, _, _, calls = two_updates(monkeypatch, last_check=lambda *args: True)
    runtime._service_follow_wheels()
    assert driver.stops == [1]
    assert not driver.pairs
    assert calls == {"feedback": 2, "safety": 4}


@pytest.mark.parametrize("hold", ["near_park", "search_park", "ordinary", "shutdown", "fault"])
def test_new_stop_owner_is_never_overwritten_by_speed_zero(monkeypatch, hold):
    def stop(runtime, owner, *_):
        runtime.backend.send_stop("concurrent_owner", mode="emergency", preserve_zero=True)
        if hold == "near_park":
            owner._near_yaw_park_request = SimpleNamespace(capture_frame_id=255, uid=1, reason="park")
        elif hold == "search_park": runtime._search_reacquire_brake_request = object()
        elif hold == "ordinary": owner._brake_hold_active = True
        elif hold == "shutdown": owner._runtime_shutdown_requested = True
        else: runtime.backend._record_motion_write_fault("test", OSError("fault"))

    runtime, _, driver, _, _, calls = two_updates(monkeypatch, last_check=stop)
    runtime._service_follow_wheels()
    assert driver.stops == [1]
    assert not driver.pairs
    assert calls == {"feedback": 2, "safety": 4}


@pytest.mark.parametrize("base,yaw", [(31, 7), (33, 7), (32, -7), (32, 10)])
def test_second_change_cannot_change_base_reverse_or_enlarge_yaw(monkeypatch, base, yaw):
    def change(_runtime, _owner, _clock, state, _sample):
        state[:2] = [float(base), float(yaw)]
    runtime, _, driver, _, _, calls = two_updates(monkeypatch, second_update=change)
    runtime._service_follow_wheels()
    assert driver.pairs == [(0, 0)]
    assert calls == {"feedback": 2, "safety": 3}


@pytest.mark.parametrize("pending", ["pending_signs", "resume_signs", "pending_full_reverse",
    "commanded_reverse", "residual_turn_signs", "residual_forward_until"])
def test_pending_reversal_or_resume_cannot_use_contraction(monkeypatch, pending):
    def change(runtime, *_):
        value = ((1, 1) if pending.endswith("signs") else
                 10.1 if pending.endswith("until") else True)
        setattr(runtime._visible_wheel_guard, pending, value)
    runtime, _, driver, _, _, calls = two_updates(monkeypatch, second_update=change)
    runtime._service_follow_wheels()
    assert driver.pairs == [(0, 0)]
    assert calls == {"feedback": 2, "safety": 3}


def test_response_boost_cannot_be_relabelled_as_ordinary_contraction(monkeypatch):
    runtime, _, driver, _, _, calls = two_updates(monkeypatch)
    runtime.config.follow_turn_response_assist_enable = True
    runtime._turn_response_assist.adjust = lambda base, yaw, *args, **kwargs: (
        base, yaw+1, "response_boost")
    runtime._service_follow_wheels()
    assert driver.pairs == [(0, 0)]
    assert calls == {"feedback": 2, "safety": 3}


@pytest.mark.parametrize("change", ["intent", "pending", "resume", "new_reverse_feedback"])
def test_final_contraction_rechecks_intent_guard_and_latest_feedback(monkeypatch, change):
    def final_change(runtime, owner, _clock, _state, _sample):
        if change == "intent":
            current[0] = SimpleNamespace(park_requested=True)
        elif change == "pending": runtime._visible_wheel_guard.pending_signs = (1, -1)
        elif change == "resume": runtime._visible_wheel_guard.resume_signs = (1, 1)
        else: runtime.get_steering_feedback = lambda: feedback(10., -2, 20)

    runtime, owner, driver, _, _, _ = two_updates(monkeypatch, last_check=final_change)
    current = [SimpleNamespace(park_requested=False)]
    owner._lateral_intent_store = SimpleNamespace(snapshot=lambda: current[0])
    runtime._service_follow_wheels()
    assert driver.pairs == [(0, 0)]


@pytest.mark.parametrize("flag", ["park_requested", "forward_countersteer", "countersteer_rpm"])
def test_new_braking_intent_does_not_inherit_old_tracking_phase(monkeypatch, flag):
    runtime, owner, driver, _, _, calls = two_updates(monkeypatch)
    intent = SimpleNamespace(**{flag: 1})
    owner._lateral_intent_store = SimpleNamespace(snapshot=lambda: intent)
    runtime._service_follow_wheels()
    assert driver.pairs == [(0, 0)]
    assert calls["safety"] == 3


def test_stop_during_final_cache_read_is_not_overwritten(monkeypatch):
    runtime, owner, driver, _, _, _ = two_updates(monkeypatch)
    read = runtime.get_steering_feedback
    count = [0]
    def read_and_stop():
        count[0] += 1
        value = read()
        if count[0] == 3:
            runtime.backend.send_stop("cache_read_race", mode="emergency", preserve_zero=True)
            owner._brake_hold_active = True
        return value
    runtime.get_steering_feedback = read_and_stop
    runtime._service_follow_wheels()
    assert not driver.pairs
    assert driver.stops == [1]


def test_forward_brake_pulse_cannot_use_ordinary_contraction(monkeypatch):
    from test_cap382_response_execution import brake_intent, fb
    runtime, owner, driver, _, _, calls = two_updates(
        monkeypatch, initial_yaw=6, new_yaw=4, sample_factory=fb)
    owner._follow_controller.cfg = SimpleNamespace(
        visible_steering_pid_image_error_only=True,
        visible_steering_pid_execution_response_trial_sec=.35,
        visible_steering_pid_max_correction_rpm=10,
        near_distance_rotation_only_max_rpm=7)
    owner._lateral_intent_store = LateralIntentStore()
    owner._lateral_intent_store.publish(brake_intent())
    runtime._service_follow_wheels()
    assert driver.pairs == [(0, 0)]
    assert calls["safety"] == 3


def test_actual_reverse_feedback_keeps_zero_cross_guard(monkeypatch):
    runtime, _, driver, _, _, calls = two_updates(
        monkeypatch, sample_factory=lambda: feedback(10., -20, -20))
    runtime._service_follow_wheels()
    assert driver.pairs == [(0, 0)]
    assert runtime._visible_wheel_guard.pending_full_reverse
    assert calls["safety"] == 2  # No nonzero plan reached final coalescing.


@pytest.mark.parametrize("applied,planned,current", [
    ((42, 22), (1, 2, 32., 10.), (2, 3, 32., 7.)),
    ((42, -2), (1, 2, 20., 22.), (1, 3, 20., 7.)),
    ((42, 22), (1, 2, 32., 10.), (1, 3, 32., float("nan"))),
    ((42, 22), (1, 2, 32., 10.), (1, 3, 32., 7.5)),
    ((41, 24), (1, 2, 32.5, 8.5), (1, 3, 32.5, 7.)),
])
def test_contraction_rejects_identity_reversal_invalid_and_rounding_changes(applied, planned, current):
    assert contract_forward_yaw(applied, planned, current) is None


def test_coalescing_log_occurs_only_after_speed_write(monkeypatch):
    runtime, _, driver, _, _, _ = two_updates(monkeypatch)
    info = runtime.logger.info
    def log(message, *args, **kwargs):
        if message.startswith("follow_wheel_yaw_coalesced"):
            assert driver.pairs == [(39, -25)]
        info(message, *args, **kwargs)
    monkeypatch.setattr(runtime.logger, "info", log)
    runtime._service_follow_wheels()
    assert driver.pairs == [(39, -25)]
