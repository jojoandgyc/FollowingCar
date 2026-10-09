"""CAP1163--1204: buildup must not turn an approved base into an inner-wheel cap.

Policy replays and the production writer with fake serial; no hardware access.
"""
from dataclasses import replace

import pytest

from functools import partial
from car_control_modular.turn_buildup import TurnBuildup as ProductionTurnBuildup

# Keep the historical timing comparator explicit, not a runtime setting.
TurnBuildup = partial(ProductionTurnBuildup, response_delay_sec=.1)
from test_cap837_turn_buildup import image_intent, setup
from test_visible_wheel_continuity import feedback


def replay(base, measured, sign=-1, *, legacy=False, yaw=10, limit=10):
    policy = TurnBuildup(legacy_common_accel_limit=legacy)
    policy.note_sent(1, (base+sign*yaw, base-sign*yaw), 9.86, 0.)
    for t, cap in ((10., 1186), (10.05, 1187)):
        result = policy.adjust(base, sign*yaw, image_intent(t, cap, sign),
                               feedback(t, measured, measured), t, 1, limit=limit)
    return policy, result


@pytest.mark.parametrize('base,measured,old_base', [(90, 30, 42), (96, 36, 48), (100, 46, 58)])
@pytest.mark.parametrize('sign', [-1, 1])
def test_actual_common_cut_cases_preserve_approved_base(base, measured, old_base, sign):
    policy, result = replay(base, measured, sign)
    assert result == (base, sign*10, 'buildup_yaw_only')
    assert policy.started == 10.05
    # The numerical historical comparator remains explicit and reproducible.
    _, old_result = replay(base, measured, sign, legacy=True)
    assert old_result == (old_base, sign*10, 'buildup_common_accel_limited')


@pytest.mark.parametrize('sign', [-1, 1])
def test_yaw_headroom_changes_difference_not_mean(sign):
    _, (base, yaw, phase) = replay(90, 30, sign, yaw=6)
    assert (base, yaw, phase) == (90, sign*8, 'buildup_yaw_only')
    pair = (base+yaw, base-yaw)
    assert sum(pair)/2 == 90 and abs(pair[0]-pair[1]) == 16
    assert min(pair) > 0


def test_legacy_comparator_survives_uid_reset_but_does_not_share_evidence():
    policy, _ = replay(90, 30, legacy=True)
    fresh = replace(image_intent(10.10, 1190, -1), target_id=2)
    result = policy.adjust(90, -10, fresh, feedback(10.10, 30, 30), 10.10, 2, limit=10)
    assert result == (90, -10, 'buildup_no_sent_reference')
    assert policy.legacy_common_accel_limit and not policy.history
    assert policy.started is None and policy.lag_count == 0


@pytest.mark.parametrize('value', [False, None, 'False', 1])
def test_legacy_common_cut_requires_explicit_boolean_opt_in(value):
    _, result = replay(90, 30, legacy=value)
    assert result == (90, -10, 'buildup_yaw_only')


@pytest.mark.parametrize('base,measured', [(90, 30), (96, 36), (100, 46)])
def test_real_writer_keeps_log_replay_mean_and_twenty_rpm_difference(monkeypatch, caplog, base, measured):
    runtime, owner, driver, _, clock = setup(monkeypatch)
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: ('forward', base//2)
    runtime.get_steering_feedback = lambda: feedback(clock[0], measured, measured)
    with caplog.at_level('INFO'):
        for t, cap in ((10., 1186), (10.11, 1187), (10.16, 1188)):
            clock[0] = t
            owner._lateral_intent_store.publish(image_intent(t, cap, -1))
            runtime._send_follow_wheel_targets(base-10, -(base+10), 'FOLLOW20',
                                               max_target_override=200)
    assert driver.pairs == [(base-10, -(base+10))]*3
    assert not driver.stops
    assert 'response_phase=buildup_yaw_only' in caplog.text
    assert 'common_accel_removed_rpm=0.0' in caplog.text
    assert 'execution_base_loss_rpm=0.0' in caplog.text


def test_buildup_cannot_override_current_wheel_deceleration():
    policy = TurnBuildup()
    policy.note_sent(1, (96, 84), 9.86, 0.)
    for t in (10., 10.05):
        result = policy.adjust(90, 6, image_intent(t, 1197),
                               feedback(t, 90, 90), t, 1, limit=10)
    assert result == (90, 6, 'buildup_deceleration_preserved')


def test_new_frame_cannot_extend_same_buildup_episode():
    policy, result = replay(90, 30, yaw=6)
    assert result[:2] == (90, -8)
    policy.note_sent(1, (82, 98), 10.05, 9.86)
    result = policy.adjust(90, -6, image_intent(10.41, 1204, -1),
                           feedback(10.41, 30, 30), 10.41, 1, limit=10)
    assert result == (90, -6, 'buildup_timeout')
    assert policy.started == 10.05
