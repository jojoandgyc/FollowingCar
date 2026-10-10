"""Paired -> crop-yaw handoff uses real ACK provenance, not a new reversal."""
from dataclasses import replace

import pytest

from car_control_modular.detector_identity_lease import (
    ValidatedVisualObservation, publish_visual_identity_evidence,
)
from car_control_modular.low_quality_lateral import LimitedYawEvidence
from car_control_modular.short_follow import (
    ShortFollowConfig, ShortFollowController, ShortFollowObservation,
)
from car_control_modular.wheel_zero_cross import WheelZeroCrossGuard
from test_cap1040_limited_yaw_owner import setup
from test_visible_wheel_continuity import feedback


@pytest.mark.parametrize('direction', [-1, 1])
def test_same_pivot_ack_does_not_restart_zero_crossing(direction):
    guard = WheelZeroCrossGuard()
    pair = (5*direction, -5*direction)
    measured = (4., 2.) if direction > 0 else (2., 4.)
    receipt = ((6*direction, -6*direction), 10.)
    result, reason = guard.limit(pair, feedback(10.03, *measured), 10.04,
                                 acknowledged_pivot=receipt)
    assert result == pair and reason == 'acknowledged_pivot_continuation'
    until = guard.pivot_handoff[1]
    guard.note_sent(pair, 10.04)
    for tick in (10.08, 10.12):
        result, _ = guard.limit(pair, feedback(tick, *measured), tick)
        assert result == pair and guard.pivot_handoff[1] == until
        guard.note_sent(pair, tick)
    # A sustained opposing wheel must not get endless credit from repeats.
    result, _ = guard.limit(pair, feedback(until+.001, *measured), until+.001,
                           acknowledged_pivot=(pair, until))
    assert result == (0, 0)


@pytest.mark.parametrize('case', ['forward', 'zero', 'opposite', 'large', 'old',
    'pre_ack', 'bad_feedback', 'large_reverse', 'whole_reverse'])
def test_non_pivot_or_unexplained_feedback_cannot_use_handoff(case):
    guard = WheelZeroCrossGuard()
    pair, stamp, measured = (6, -6), 10., (4, 2)
    if case == 'forward': pair = (6, 6)
    if case == 'zero': pair = (0, 0)
    if case == 'opposite': pair = (-6, 6)
    if case == 'large': pair = (20, -20)
    if case == 'old': stamp = 9.
    if case == 'large_reverse': measured = (4, 38)
    if case == 'whole_reverse': measured = (-4, -4)
    sample = feedback(9.99 if case == 'pre_ack' else 10.03, *measured,
                      trustworthy=case != 'bad_feedback')
    result, reason = guard.limit((5, -5), sample, 10.04,
                                 acknowledged_pivot=(pair, stamp))
    assert result == (0, 0)
    assert reason != 'acknowledged_pivot_continuation'


def paired_then_crop(monkeypatch, direction=-1):
    rt, owner, driver, symbols, clock, intent = setup(monkeypatch)
    if direction > 0:
        intent = owner._lateral_intent_store.publish(replace(intent,
            x_ratio=1-intent.x_ratio, initial_correction_rpm=7))
        owner._lateral_intent_last_correction_rpm = 7
    owner._short_follow = ShortFollowController(ShortFollowConfig(enabled=True))
    owner._short_follow.activate(1, clock[0]-.1)
    owner._vision_control_state = 'target_visible_depth_valid'
    proof = ValidatedVisualObservation(1, 16, 1038, clock[0]-.05,
                                      clock[0], clock[0]+.4, 'full')
    publish_visual_identity_evidence(owner, observation=proof, lease=None)
    owner._short_follow.update(ShortFollowObservation(1, 1038, clock[0]-.05,
        clock[0]-.03, 1.4, .2 if direction < 0 else .8), clock[0])
    rt.get_steering_feedback = lambda: feedback(clock[0], 0, 0)
    assert rt._service_short_follow()
    assert driver.pairs[-1][0] * direction > 0 and driver.pairs[-1][0] == driver.pairs[-1][1]
    assert not driver.stops
    clock[0] += .04
    publication = publish_visual_identity_evidence(owner, observation=False, lease=False)
    source = replace(owner._limited_yaw_evidence.source, identity_publication=publication)
    owner._limited_yaw_evidence = LimitedYawEvidence(source, intent, intent.valid_until)
    owner._vision_control_state = 'target_visible_low_quality'
    owner._short_follow.retire_for_lateral_handoff(clock[0])
    measured = (2, 4) if direction < 0 else (4, 2)
    rt.get_steering_feedback = lambda: feedback(clock[0], *measured)
    return rt, owner, driver, clock


def test_real_writer_pivot_to_limited_yaw_has_no_zero_or_stop(monkeypatch):
    rt, owner, driver, clock = paired_then_crop(monkeypatch)
    assert rt._service_short_follow()
    assert driver.pairs[-1] == (-7, -7)
    assert not driver.stops and all(p != (0, 0) for p in driver.pairs)
    clock[0] += .06
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (-7, -7)
    assert not driver.stops
    assert owner._validated_visual_observation is False  # no identity upgrade


@pytest.mark.parametrize('case', ['new_stop', 'different_receipt', 'unbound_plan',
    'other_uid', 'stale_feedback', 'forward_plan', 'expired_crop'])
def test_only_current_bound_paired_ack_can_transfer(monkeypatch, case):
    rt, owner, driver, clock = paired_then_crop(monkeypatch)
    receipt = rt.backend.last_speed_receipt
    if case == 'new_stop': rt.backend.send_stop('test', mode='emergency')
    if case == 'different_receipt': rt.backend.send_targets(-7, -7, 'other')
    if case == 'unbound_plan': owner._short_follow_last_applied_plan = None
    if case == 'other_uid':
        owner._short_follow_last_applied_plan = replace(owner._short_follow_last_applied_plan, uid=2)
    if case == 'forward_plan':
        owner._short_follow_last_applied_plan = replace(owner._short_follow_last_applied_plan,
                                                       left_rpm=7, right_rpm=7, base_rpm=7)
    sample = feedback(9. if case == 'stale_feedback' else clock[0], 2, 4)
    if case == 'expired_crop': clock[0] += .2
    assert rt._acknowledged_pivot_handoff(receipt, 1, (-7, 7), sample, clock[0]) is None


@pytest.mark.parametrize('direction', [-1, 1])
@pytest.mark.parametrize('change', ['same', 'smaller_tail', 'expired_window',
    'large_tail', 'opposite_response', 'new_rejection', 'stop_generation'])
def test_periodic_commit_rechecks_new_encoder_inside_original_pivot_window(
        monkeypatch, direction, change):
    rt, owner, driver, clock = paired_then_crop(monkeypatch, direction)
    assert rt._service_short_follow()
    assert driver.pairs[-1] == (7*direction, 7*direction)  # raw right sign inverted
    window = rt._visible_wheel_guard.pivot_handoff
    assert window is not None
    # A genuinely newer admitted crop keeps lateral authority current long
    # enough to isolate expiry of the earlier physical response window.
    clock[0] += .06
    previous = owner._limited_yaw_evidence
    publication = publish_visual_identity_evidence(owner, observation=False, lease=False)
    source = replace(previous.source, capture=1041, timestamp=clock[0]-.01,
                     identity_publication=publication)
    intent = owner._lateral_intent_store.publish(replace(previous.intent,
        capture_frame_id=source.capture, capture_timestamp=source.timestamp,
        decision_capture_frame_id=source.capture, published_at=clock[0],
        valid_until=clock[0]+.25))
    owner._limited_yaw_evidence = LimitedYawEvidence(source, intent, intent.valid_until)
    owner._lateral_yaw_revision += 1
    measured = (2, 4) if direction < 0 else (4, 2)
    guarded = feedback(clock[0], *measured)
    rt.get_steering_feedback = lambda: guarded
    writes = []
    original_send = rt.backend.send_targets
    def send(*args, **kwargs):
        writes.append(args[:3])
        return original_send(*args, **kwargs)
    monkeypatch.setattr(rt.backend, 'send_targets', send)
    injected = []
    def safety(_action):
        # The service-entry safety read precedes the ownership token/guard.
        # Inject only in the packet's post-guard safety read, before commit.
        if not injected and rt._periodic_follow_writing:
            injected.append(True)
            clock[0] = window[1] if change == 'expired_window' else clock[0]+.001
            inner, outer = (1.5, 4) if change == 'smaller_tail' else (2, 4)
            if change == 'large_tail': inner = 3
            if change == 'opposite_response': inner, outer = 4, -4
            newer = (inner, outer) if direction < 0 else (outer, inner)
            rt._steering_feedback = feedback(clock[0], *newer)
            if change == 'new_rejection':
                publish_visual_identity_evidence(owner, observation=False, lease=False)
            if change == 'stop_generation':
                rt.backend.send_stop('intervening_test_stop', mode='emergency')
        return False
    rt.hard_stop_check = safety
    before = len(driver.pairs)
    rt._service_follow_wheels()
    assert injected
    if change in {'same', 'smaller_tail'}:
        assert driver.pairs[before:] == [(7*direction, 7*direction)]
        assert not driver.stops
        assert rt._visible_wheel_guard.pivot_handoff == window
    else:
        assert all(pair == (0, 0) for pair in driver.pairs[before:])
        if change in {'expired_window', 'large_tail', 'opposite_response'}:
            assert (0, 0, 'FOLLOW_COMMIT_FEEDBACK_CHANGED') in writes
        if change == 'stop_generation':
            assert driver.stops
    assert owner._validated_visual_observation is False
    assert not owner._short_follow.snapshot().active


@pytest.mark.parametrize('direction', [-1, 1])
@pytest.mark.parametrize('change', ['none', 'expired', 'large_tail', 'opposite_request',
    'forward', 'stale', 'missing_window'])
def test_pivot_commit_predicate_never_changes_window(direction, change):
    guard = WheelZeroCrossGuard()
    pair = (5*direction, -5*direction)
    measured = (4, 2) if direction > 0 else (2, 4)
    guard.limit(pair, feedback(10.03, *measured), 10.04,
                acknowledged_pivot=((6*direction, -6*direction), 10.))
    now = guard.pivot_handoff[1] if change == 'expired' else 10.08
    if change == 'large_tail': measured = (4, 3) if direction > 0 else (3, 4)
    if change == 'opposite_request': pair = tuple(-v for v in pair)
    if change == 'forward': pair = (5, 5)
    if change == 'missing_window': guard.pivot_handoff = None
    sample = feedback(9. if change == 'stale' else now, *measured)
    state = vars(guard).copy()
    assert guard.accepts_acknowledged_pivot(pair, sample, now) == (change == 'none')
    assert vars(guard) == state
