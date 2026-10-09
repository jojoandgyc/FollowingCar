"""Final yaw/depth publication races use a new live grant, never an old lease."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.lateral_intent import LateralIntentStore
from car_control_modular.detector_identity_lease import DetectorIdentityLease
from test_cap331_intent_handoff import ordinary_intent
from test_follow_wheel_periodic import setup_periodic
from test_visible_wheel_continuity import feedback


def _writer(monkeypatch, case="zero_yaw", veto=None):
    runtime, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    state[:] = [32., -5., 11., 11.]
    raw = [("forward", 32., 1, 10.)]
    owner._depth30_linear_snapshot = raw[0]
    owner._depth_linear_max_age_sec = lambda kind: .25
    owner._last_vision_control_ts = 10.
    owner._follow_controller.cfg = SimpleNamespace(
        visible_steering_pid_image_error_only=True,
        visible_steering_pid_execution_response_trial_sec=.35,
        visible_steering_pid_max_correction_rpm=10.,
        near_distance_rotation_only_max_rpm=7.)
    store = owner._lateral_intent_store = LateralIntentStore()
    store.publish(ordinary_intent(sign=-1, cap=188, capture=9.95,
                                 published=10., initial_correction_rpm=-5))

    def depth(uid, now=None):
        at = clock[0] if now is None else now
        grant = raw[0]
        if uid != 1 or not grant[3] <= at < grant[3]+.25:
            return None
        return ("forward", min(grant[1], state[0]), uid, grant[3])

    owner._fresh_depth_linear_snapshot = depth
    runtime.get_steering_feedback = lambda: feedback(clock[0], 20, 20)
    runtime._service_follow_wheels()
    assert driver.pairs == [(27, -37)]
    driver.pairs.clear()
    clock[0] = 10.05
    calls = {"feedback": 0, "safety": 0}

    def read_feedback():
        calls["feedback"] += 1
        if calls["feedback"] == 1:
            raw[0] = owner._depth30_linear_snapshot = ("forward", 54., 1, 10.02)
            state[:2] = [54., -5.]
            owner._lateral_yaw_revision += 1
        return feedback(clock[0], 20, 20)

    def safety(_):
        calls["safety"] += 1
        if calls["safety"] == 3:
            raw[0] = owner._depth30_linear_snapshot = ("forward", 54., 1, 10.04)
            owner._last_vision_control_ts = clock[0]
            intent = ordinary_intent(sign=-1, cap=190, capture=10.01,
                                     published=clock[0], initial_correction_rpm=-5)
            if case == "zero_yaw":
                state[1] = 0.
                intent = replace(intent, initial_correction_rpm=0,
                                 forward_countersteer=True, park_requested=True)
                intent = store.publish(intent)
                owner._lateral_intent_zero_sequence = intent.sequence
            else:
                store.publish(intent)
                owner._has_fresh_lateral_yaw = lambda uid: False
            owner._lateral_yaw_revision += 1
            if veto == "stop":
                runtime.backend.send_stop("concurrent_stop", mode="emergency", preserve_zero=True)
            elif veto == "uid":
                owner._follow_controller.active_target_id = 2
            elif veto == "park":
                owner._near_yaw_park_request = SimpleNamespace(
                    uid=1, capture_frame_id=190, reason="real_park")
            elif veto == "ttl":
                clock[0] = 10.30
            elif veto == "missing_depth":
                owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
        return False

    runtime.get_steering_feedback = read_feedback
    runtime.hard_stop_check = safety
    return runtime, owner, driver, clock, state, raw, calls


@pytest.mark.parametrize("case", ["zero_yaw", "expired_yaw"])
def test_new_live_depth_and_removed_yaw_do_not_write_zero(monkeypatch, case, caplog):
    runtime, owner, driver, _, _, _, calls = _writer(monkeypatch, case)
    with caplog.at_level("INFO"):
        runtime._service_follow_wheels()
    assert calls["safety"] >= 3
    # Current guarded 49/59 is contracted to 49/49, not held at old 27/37,
    # and not allowed to reuse the old -5 yaw after its expiry/revocation.
    assert driver.pairs == [(49, -49)]
    assert not driver.stops
    assert "reason=final_live_forward_handoff" in caplog.text
    assert "FOLLOW_FINAL_AUTHORITY_CHANGED" not in caplog.text


@pytest.mark.parametrize("veto", ["stop", "uid", "park", "ttl", "missing_depth"])
def test_live_forward_handoff_preserves_real_revocations(monkeypatch, veto):
    runtime, _, driver, *_ = _writer(monkeypatch, veto=veto)
    runtime._service_follow_wheels()
    assert all(not (left > 0 and right < 0) for left, right in driver.pairs)
    if veto == "stop":
        assert driver.stops == [1]
        assert not driver.pairs


@pytest.mark.parametrize("late_change", ["stop", "stop_after_limit", "lower_cap", "grant", "intent"])
def test_live_forward_handoff_rechecks_terminal_proof(monkeypatch, late_change):
    runtime, owner, driver, clock, _, raw, _ = _writer(monkeypatch)
    original = runtime._final_live_forward_handoff
    candidates = []

    def after_candidate(*args, **kwargs):
        candidate = original(*args, **kwargs)
        if candidate is not None:
            candidates.append(candidate)
            if late_change == "stop":
                runtime.backend.send_stop("candidate_stop", mode="emergency", preserve_zero=True)
            elif late_change == "lower_cap":
                owner._fresh_depth_linear_snapshot = lambda uid, now=None: (
                    "forward", 30., uid, raw[0][3])
            elif late_change == "grant":
                raw[0] = owner._depth30_linear_snapshot = ("forward", 54., 1, 10.041)
            elif late_change == "intent":
                owner._lateral_intent_store.publish(ordinary_intent(
                    cap=191, capture=10.02, published=clock[0]))
        return candidate

    runtime._final_live_forward_handoff = after_candidate
    terminal = runtime._linear_packet_write_limit

    def after_limit(*args, **kwargs):
        decision = terminal(*args, **kwargs)
        if late_change == "stop_after_limit":
            runtime.backend.send_stop("terminal_stop", mode="emergency", preserve_zero=True)
        return decision

    runtime._linear_packet_write_limit = after_limit
    runtime._service_follow_wheels()
    assert candidates
    if late_change == "lower_cap":
        assert driver.pairs == [(30, -30)]
    elif late_change in {"grant", "intent"}:
        # These publications are genuinely fresh (10.041 < now=10.05),
        # same-UID and ordinary. They invalidate the UNSENT contraction,
        # not the right to replan current axes after releasing the I/O lock.
        assert driver.pairs == [(54, -54)]
        assert runtime._follow_wheel_clock.last_axes[2:] == (54., 0.)
        assert runtime._forward_execution_anchor.sample_timestamp == raw[0][3]
    else:
        assert all(not (left > 0 and right < 0) for left, right in driver.pairs)
    if late_change.startswith("stop"):
        assert driver.stops == [1]
        assert not driver.pairs


def test_stop_during_candidate_check_is_not_overwritten_by_fallback_zero(monkeypatch):
    runtime, _, driver, *_ = _writer(monkeypatch)
    original = runtime._final_live_forward_handoff
    entered = []

    def stop_during_feedback(*args, **kwargs):
        read = runtime.get_steering_feedback

        def feedback_and_stop():
            sample = read()
            entered.append(True)
            runtime.backend.send_stop("candidate_read_stop", mode="emergency", preserve_zero=True)
            return sample

        runtime.get_steering_feedback = feedback_and_stop
        try:
            return original(*args, **kwargs)
        finally:
            runtime.get_steering_feedback = read

    runtime._final_live_forward_handoff = stop_during_feedback
    runtime._service_follow_wheels()
    assert entered
    assert driver.stops == [1]
    assert not driver.pairs


@pytest.mark.parametrize("veto", ["held_zero", "uncleared_countersteer", "pending_reverse",
                                    "resume", "full_reverse", "commanded_reverse",
                                    "residual_turn", "residual_forward", "fault"])
def test_neutral_bridge_cannot_bypass_motion_owners(monkeypatch, veto):
    runtime, owner, driver, *_ = _writer(monkeypatch)
    original = runtime._final_live_forward_handoff

    def changed_candidate(*args, **kwargs):
        if veto == "held_zero":
            owner._lateral_intent_store.publish(replace(
                owner._lateral_intent_store.snapshot(), hold_zero=True))
        elif veto == "uncleared_countersteer":
            owner._lateral_intent_zero_sequence = -1
        elif veto == "pending_reverse":
            runtime._visible_wheel_guard.pending_signs = (-1, 1)
        elif veto == "resume":
            runtime._visible_wheel_guard.resume_signs = (1, 1)
        elif veto == "full_reverse":
            runtime._visible_wheel_guard.pending_full_reverse = True
        elif veto == "commanded_reverse":
            runtime._visible_wheel_guard.commanded_reverse = True
        elif veto == "residual_turn":
            runtime._visible_wheel_guard.residual_turn_signs = (-1, 1)
        elif veto == "residual_forward":
            runtime._visible_wheel_guard.residual_forward_until = 10.20
        else:
            runtime.backend.motion_write_fault = "serial_fault"
        return original(*args, **kwargs)

    runtime._final_live_forward_handoff = changed_candidate
    runtime._service_follow_wheels()
    assert all(not (left > 0 and right < 0) for left, right in driver.pairs)


def test_last_new_grant_yaw_sign_flip_can_only_contract_to_straight(monkeypatch, caplog):
    runtime, owner, driver, clock, state, _, _ = _writer(monkeypatch)
    original = runtime._final_live_forward_handoff
    candidates = []

    def flip_before_candidate(*args, **kwargs):
        # The old unsent curve was 49/59. A new right turn must NOT borrow
        # the guard result of that left turn, even with new valid Depth.
        state[1] = 6.
        owner._lateral_intent_store.publish(ordinary_intent(
            sign=1, cap=191, capture=10.02, published=clock[0],
            initial_correction_rpm=6))
        owner._lateral_yaw_revision += 1
        candidate = original(*args, **kwargs)
        candidates.append(candidate)
        return candidate

    runtime._final_live_forward_handoff = flip_before_candidate
    with caplog.at_level("INFO"):
        runtime._service_follow_wheels()
    assert candidates and candidates[0] is not None
    assert driver.pairs == [(49, -49)]
    assert "reason=final_live_forward_handoff" in caplog.text


@pytest.mark.parametrize("phase", ["candidate", "final_reader"])
@pytest.mark.parametrize("expiry", ["feedback", "identity"])
def test_handoff_does_not_refresh_feedback_or_detector_identity_deadline(
        monkeypatch, phase, expiry):
    runtime, owner, driver, clock, _, _, _ = _writer(monkeypatch)
    if expiry == "identity":
        owner._detector_identity_lease = DetectorIdentityLease(
            uid=1, track_id=1, verified_capture=180, verified_timestamp=9.90,
            observation_capture=188, observation_timestamp=10.00, expires_at=10.10)
    original = runtime._final_live_forward_handoff
    candidates = []
    delayed = []

    def install_delayed_reader():
        read = owner._fresh_depth_linear_snapshot

        def delayed_reader(uid, now=None):
            if not delayed:
                delayed.append(clock[0])
                if expiry == "feedback":
                    # No encoder publication occurred during this delay.
                    # A commit-time cache reread must see the SAME sample,
                    # not manufacture fresh feedback from the fake clock.
                    sample = feedback(clock[0], 20, 20)
                    runtime.get_steering_feedback = lambda: sample
                clock[0] = 10.201 if expiry == "feedback" else 10.11
            return read(uid, now=now)

        owner._fresh_depth_linear_snapshot = delayed_reader

    def delayed_candidate(*args, **kwargs):
        if phase == "candidate":
            install_delayed_reader()
        candidate = original(*args, **kwargs)
        candidates.append(candidate)
        if phase == "final_reader":
            assert candidate is not None
            install_delayed_reader()
        return candidate

    runtime._final_live_forward_handoff = delayed_candidate
    runtime._service_follow_wheels()
    assert delayed and candidates
    assert (candidates[0] is None) == (phase == "candidate")
    assert all(not (left > 0 and right < 0) for left, right in driver.pairs)
