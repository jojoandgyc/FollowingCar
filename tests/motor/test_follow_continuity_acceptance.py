"""Multi-period command-stream acceptance, using real execution and fake I/O.

These tests measure *commands*, not wheel response.  The shared fixture uses
the real runtime/backend, receipt ledger, wheel guard and braking assessment;
only the physical driver and sensor publication are fake.  Every observation
has a fixed sample timestamp and deadline.  Encoder refresh and producer
churn must never renew an old depth or identity lease.
"""
from dataclasses import replace

import pytest

from car_control_modular.detector_identity_lease import ValidatedVisualObservation
from test_unified_forward_snapshot import feedback, writer


def qualified_writer(monkeypatch, *, base=60., yaw=0.):
    rt, owner, driver, clock, publish_raw, _ = writer(monkeypatch, base=base, yaw=yaw)
    captures = [1]

    def publish(base, yaw, *, new_depth=True):
        captures[0] += 1
        stamp = clock[0]
        owner._validated_visual_observation = ValidatedVisualObservation(
            1, 3, captures[0], stamp, stamp, stamp+.30, "full")
        return publish_raw(base, yaw, new_depth=new_depth)

    publish(base, yaw)
    return rt, owner, driver, clock, publish


def positive(pair):
    # This fake backend uses an inverted right motor register for forward.
    return pair[0] > 0 and pair[1] < 0


@pytest.mark.parametrize("expire_yaw", [False, True])
@pytest.mark.parametrize("stale_soft_stop", [False, True])
def test_three_second_normal_stream_never_inserts_zero(
        monkeypatch, expire_yaw, stale_soft_stop):
    rt, owner, driver, clock, publish = qualified_writer(monkeypatch)
    sample_stamps = set()
    start = clock[0]
    base = 60.
    for tick in range(60):
        clock[0] = start + tick*.05
        new_depth = tick % 4 == 0  # Measured vision/depth at 5 Hz, writer at 20 Hz.
        if new_depth:
            base = (60., 54., 66., 48.)[(tick // 4) % 4]
            yaw = (-5., 0., 5., 0.)[(tick // 4) % 4]
            intent = publish(base, yaw)
            if expire_yaw and yaw:
                store = owner._lateral_intent_store
                intent = store.publish(replace(intent, valid_until=clock[0]+.01))
                owner._lateral_turn_response_policy = (intent.sequence, False)
                clock[0] += .02
        rt._steering_feedback = feedback(clock[0], 24., 24.)
        if stale_soft_stop and tick % 7 == 0:
            owner._use_soft_stop_next = True
            rt.send_robot_command(rt.symbols.stop)
        rt._service_follow_wheels()
        assert all(positive(pair) for pair in driver.pairs)
        assert not driver.stops
        sample = owner._depth30_linear_snapshot[3]
        sample_stamps.add(sample)
        assert owner._depth30_linear_timing.depth_expires_at == pytest.approx(sample+.25)
        assert not owner.motor_io_lock.locked()

    assert len(sample_stamps) == 15
    assert clock[0] - start == pytest.approx(2.95)
    # Continuity cannot be achieved by never updating the first command.
    assert len(driver.pairs) >= 30
    assert len(set(driver.pairs)) >= 3


@pytest.mark.parametrize("stage", ["feedback", "guard", "terminal"])
def test_twenty_periods_of_fresh_publication_churn_do_not_exhaust_into_zero(
        monkeypatch, stage):
    rt, owner, driver, clock, publish = qualified_writer(monkeypatch, base=60., yaw=-5.)
    changes = []

    def update():
        changes.append(clock[0])
        clock[0] += .001
        idx = len(changes)
        publish((58., 54., 62.)[idx % 3], (-4., 0., 4.)[idx % 3])

    if stage == "feedback":
        original = rt.get_steering_feedback

        def hooked():
            update()
            return original()

        monkeypatch.setattr(rt, "get_steering_feedback", hooked)
    elif stage == "guard":
        original = rt._visible_wheel_guard.limit

        def hooked(*args, **kwargs):
            result = original(*args, **kwargs)
            update()
            return result

        monkeypatch.setattr(rt._visible_wheel_guard, "limit", hooked)
    else:
        original = rt._linear_packet_write_limit

        def hooked(*args, **kwargs):
            update()
            return original(*args, **kwargs)

        monkeypatch.setattr(rt, "_linear_packet_write_limit", hooked)

    for _ in range(20):
        clock[0] += .05
        rt._steering_feedback = feedback(clock[0], 24., 24.)
        before = len(changes)
        rt._service_follow_wheels()
        assert 0 < len(changes)-before <= 20  # No unbounded replan loop.
        assert all(positive(pair) for pair in driver.pairs)
        assert not driver.stops
        assert not owner.motor_io_lock.locked()

    assert clock[0] > 11.
    # An adversarial producer can change on every reader call.  A still-legal
    # acknowledged packet may be retained, but a stable next publication must
    # be delivered on the next ordinary period, not held forever.
    if stage == "feedback":
        monkeypatch.setattr(rt, "get_steering_feedback", original)
    elif stage == "guard":
        monkeypatch.setattr(rt._visible_wheel_guard, "limit", original)
    else:
        monkeypatch.setattr(rt, "_linear_packet_write_limit", original)
    clock[0] += .05
    publish(62., 0.)
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (62, -62)
    assert len(driver.pairs) >= 2


@pytest.mark.parametrize("deadline", ["depth", "identity"])
def test_encoder_ticks_and_same_sample_republication_do_not_extend_deadlines(
        monkeypatch, deadline):
    rt, owner, driver, clock, publish = qualified_writer(monkeypatch)
    original_raw = owner._depth30_linear_snapshot
    original_timing = owner._depth30_linear_timing
    proof = owner._validated_visual_observation
    if deadline == "identity":
        owner._validated_visual_observation = replace(proof, expires_at=clock[0]+.12)
    expires = (original_timing.depth_expires_at if deadline == "depth"
               else owner._validated_visual_observation.expires_at)
    for fraction in (.2, .5, .8):
        clock[0] = proof.timestamp + fraction*(expires-proof.timestamp)
        rt._steering_feedback = feedback(clock[0], 24., 24.)
        rt._service_follow_wheels()
        assert all(positive(pair) for pair in driver.pairs)
    # Reassigning identical sensor evidence does not create a new sample.
    owner._depth30_linear_snapshot = tuple(original_raw)
    owner._depth30_linear_timing = original_timing
    clock[0] = expires + .001
    rt._steering_feedback = feedback(clock[0], 24., 24.)
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (0, 0) or driver.stops
    assert owner._depth30_linear_timing.depth_expires_at == original_timing.depth_expires_at


@pytest.mark.parametrize("fault", ["uid", "identity", "front_ir", "explicit_stop", "exit"])
def test_real_revocation_interrupts_a_previously_continuous_stream(monkeypatch, fault):
    rt, owner, driver, clock, publish = qualified_writer(monkeypatch)
    for tick in range(8):
        clock[0] += .05
        publish(54., (-4., 0., 4.)[tick % 3])
        rt._service_follow_wheels()
    assert all(positive(pair) for pair in driver.pairs) and not driver.stops
    before = len(driver.pairs)
    clock[0] += .001  # Safety must preempt even between ordinary 50 ms ticks.
    if fault == "uid":
        owner._follow_controller.active_target_id = 2
    elif fault == "identity":
        owner._validated_visual_observation = False
        owner._vision_control_state = "target_lost"
    elif fault == "front_ir":
        rt.hard_stop_check = lambda _action: True
    elif fault == "explicit_stop":
        owner._explicit_stop_requested = True
    else:
        owner._runtime_shutdown_requested = True
    rt._service_follow_wheels()
    assert not any(positive(pair) for pair in driver.pairs[before:])
    assert driver.stops or driver.pairs[before:] == [(0, 0)]
    assert not owner.motor_io_lock.locked()


def test_new_independent_depth_after_real_expiry_resumes_without_a_stop_loop(monkeypatch):
    rt, owner, driver, clock, publish = qualified_writer(monkeypatch, base=54.)
    expired_timing = owner._depth30_linear_timing
    clock[0] = expired_timing.depth_expires_at + .001
    rt._steering_feedback = feedback(clock[0], 24., 24.)
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (0, 0)
    zero_index = len(driver.pairs)-1
    clock[0] += .001
    publish(54., 0.)
    # A normal positive update can wait until the already scheduled 50 ms
    # period; the acceptance condition is no extra recovery-stop cycle.
    clock[0] += .05
    rt._steering_feedback = feedback(clock[0], 24., 24.)
    rt._service_follow_wheels()
    assert positive(driver.pairs[-1])
    for _ in range(8):
        clock[0] += .05
        publish(54., 0.)
        rt._service_follow_wheels()
    assert all(positive(pair) for pair in driver.pairs[zero_index+1:])
    assert not driver.stops
    assert owner._depth30_linear_timing.depth_expires_at > expired_timing.depth_expires_at


@pytest.mark.parametrize("cap_call", [1, 2])
@pytest.mark.parametrize("fault", ["untrustworthy", "missing", "newer_reverse"])
def test_latest_feedback_failure_after_guard_cannot_escape_final_commit(
        monkeypatch, cap_call, fault):
    rt, owner, driver, clock, publish = qualified_writer(monkeypatch)
    before = len(driver.pairs)
    original = owner._depth_forward_continuation_limit
    calls = []

    def changed_after_budget(*args, **kwargs):
        result = original(*args, **kwargs)
        if getattr(rt, "_follow_snapshot_planning", False):
            calls.append(clock[0])
            if len(calls) == cap_call:
                if fault == "untrustworthy":
                    rt._steering_feedback = replace(rt._steering_feedback, trustworthy=False)
                elif fault == "missing":
                    rt._steering_feedback = None
                else:
                    clock[0] += .001
                    rt._steering_feedback = feedback(clock[0], -20., -20.)
        return result

    monkeypatch.setattr(owner, "_depth_forward_continuation_limit", changed_after_budget)
    clock[0] += .05
    publish(54., 0.)
    rt._service_follow_wheels()
    assert len(calls) >= cap_call
    assert not any(positive(pair) for pair in driver.pairs[before:])
    assert not owner.motor_io_lock.locked()
