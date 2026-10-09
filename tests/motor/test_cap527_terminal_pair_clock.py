"""Final pair arithmetic may not consume yaw/Depth lifetime or undo STOP.

Only fake clocks, feedback and motor drivers are used.
"""
from types import SimpleNamespace

import pytest

import car_control_modular.action_runtime as action_runtime
from car_control_modular.lateral_intent import LateralIntentStore
from test_cap1663_turn_response import intent
from test_follow_same_grant_speed_contraction import _live_depth_grant
from test_visible_wheel_continuity import feedback


def writer(monkeypatch, yaw=-3.):
    runtime, owner, driver, clock, grant = _live_depth_grant(monkeypatch)
    grant.update(initial_percent=62., first_cap=62., middle_cap=62.,
                 late_cap=62., terminal_cap=62., yaw=yaw)
    owner._depth30_linear_snapshot = ('forward', 62., 1, grant['stamp'])
    owner._depth_forward_continuation_required = lambda *args: True
    owner._depth_forward_continuation_limit = lambda *args, **kwargs: (
        60., 'same_grant_relative_braking_cap')
    return runtime, owner, driver, clock, grant


@pytest.mark.parametrize('expiry', ['absolute', 'nominal_continuation'])
def test_terminal_quiet_cap_cannot_reuse_expired_real_yaw_intent(monkeypatch, expiry):
    runtime, owner, driver, clock, _ = writer(monkeypatch)
    owner._lateral_intent_store = LateralIntentStore()
    payload = intent(10.04, capture_timestamp=9.95, initial_correction_rpm=-3,
                     valid_until=10.075 if expiry == 'absolute' else 10.24,
                     nominal_valid_until=0. if expiry == 'absolute' else 10.075,
                     x_ratio=.61, target_image_rate_dps=40.)
    published = owner._lateral_intent_store.publish(payload)
    sampled_feedback = feedback(10.05, 20, 20)
    sampled_feedback.yaw_rate_right_dps = 0.
    sampled_feedback.raw_yaw_rate_right_dps = 0.
    runtime.get_steering_feedback = lambda: sampled_feedback
    owner._has_fresh_lateral_yaw = lambda uid: (
        uid == 1 and published.valid(clock[0])
        and published.continuation_allowed(clock[0], sampled_feedback))
    calls = []

    def quiet_cap(*args, **kwargs):
        calls.append(clock[0])
        clock[0] = 10.09  # Depth and sampled wheel feedback remain fresh.
        return 60., 'same_grant_relative_braking_cap'

    owner._depth_forward_continuation_limit = quiet_cap
    runtime._service_follow_wheels()
    assert calls
    assert not owner._has_fresh_lateral_yaw(1)
    # Yaw expires independently of the fresh forward grant. Rebuild straight
    # under the same physical deadline, including in the legacy full planner.
    assert runtime._follow_write_veto_reason is None
    assert driver.pairs == [(60, -60)]
    assert owner._depth30_linear_snapshot[3] == 10.05
    assert not driver.stops


@pytest.mark.parametrize('change', ['depth_expiry', 'stop'])
def test_final_pair_contraction_precedes_terminal_clock_and_stop_check(monkeypatch, change):
    runtime, owner, driver, clock, grant = writer(monkeypatch, yaw=0.)
    real_contract = action_runtime.contract_forward_speed
    calls = []

    def delayed_contract(pair, cap):
        result = real_contract(pair, cap)
        calls.append((pair, result))
        if change == 'depth_expiry':
            clock[0] = grant['stamp']+.251
        else:
            runtime.backend.send_stop('concurrent_terminal_stop', mode='emergency',
                                      preserve_zero=True)
        return result

    monkeypatch.setattr(action_runtime, 'contract_forward_speed', delayed_contract)
    runtime._service_follow_wheels()
    assert calls and calls[-1][1] is not None
    if change == 'stop':
        assert driver.stops == [1]
        assert driver.pairs == []  # Even speed-mode zero would undo STOP ownership.
    else:
        assert driver.pairs == [(0, 0)]
        assert runtime._follow_write_veto_reason == 'terminal_authority_or_feedback_expired'


def test_intent_snapshot_lock_wait_precedes_terminal_physical_clock(monkeypatch):
    runtime, owner, driver, clock, grant = writer(monkeypatch, yaw=0.)
    cap_calls = []
    snapshot_blocked = []

    def quiet_cap(*args, **kwargs):
        cap_calls.append(clock[0])
        return 60., 'same_grant_relative_braking_cap'

    def snapshot():
        if len(cap_calls) >= 2:
            snapshot_blocked.append(clock[0])
            clock[0] = grant['stamp']+.251
        return None  # Same object/revision; only the lock wait consumed time.

    owner._lateral_intent_store = SimpleNamespace(snapshot=snapshot)
    owner._depth_forward_continuation_limit = quiet_cap
    runtime._service_follow_wheels()
    assert snapshot_blocked
    assert driver.pairs == [(0, 0)]
    assert runtime._follow_write_veto_reason == 'terminal_authority_or_feedback_expired'
