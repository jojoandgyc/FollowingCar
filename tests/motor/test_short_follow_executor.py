"""Adversarial handoffs for the real paired writer; no hardware is opened."""
from types import SimpleNamespace
import queue
import threading
from dataclasses import replace

import pytest

from test_depth_drive_rpm import make_runtime
from car_control_modular.short_follow import (
    ShortFollowConfig, ShortFollowController, ShortFollowObservation,
)
from car_control_modular.detector_identity_lease import (
    ValidatedVisualObservation, publish_visual_identity_evidence,
)
from car_control_modular.action_command import ActionCommandSnapshot


def short_runtime(monkeypatch):
    runtime, owner, driver, symbols = make_runtime(max_rpm=200)
    clock = [10.]
    monkeypatch.setattr("car_control_modular.short_follow_executor.time.monotonic", lambda: clock[0])
    owner.running = True
    owner.search_state = "none"
    owner._follow_controller = SimpleNamespace(active_target_id=1, search_state="none")
    owner._short_follow = ShortFollowController(ShortFollowConfig(enabled=True))
    owner._short_follow.activate(1, clock[0])
    runtime.get_steering_feedback = lambda: SimpleNamespace(timestamp=clock[0],
        left_forward_rpm=0., right_forward_rpm=0., trustworthy=True)
    publish(runtime, clock, 1)
    return runtime, owner, driver, symbols, clock


def publish(runtime, clock, capture, distance=2., x=.5):
    now = clock[0]
    runtime.owner._validated_visual_observation = ValidatedVisualObservation(
        1, 1, capture, now, now, now+.5, "full")
    return runtime.owner._short_follow.update(ShortFollowObservation(
        1, capture, now, now, distance, x), now)


def planned_pair(owner, limit=None):
    plan = owner._short_follow.snapshot().plan
    scale = min(1., limit / max(plan.left_rpm, plan.right_rpm)) if limit is not None else 1.
    return round(plan.left_rpm * scale), -round(plan.right_rpm * scale)


@pytest.mark.parametrize("check_number", [1, 2])
@pytest.mark.parametrize("new_pair", [False, True])
def test_identity_published_during_safety_read_uses_post_publication_clock(
        monkeypatch, check_number, new_pair):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    original_plan = owner._short_follow.snapshot().plan
    calls = []
    def safety(_action):
        calls.append(1)
        if len(calls) == check_number:
            clock[0] += .020
            if new_pair:
                publish(rt, clock, 2, distance=2.1, x=.1)
            proof = ValidatedVisualObservation(1, 1, 2, clock[0], clock[0], clock[0]+.5, "full")
            publish_visual_identity_evidence(owner, observation=proof, lease=None)
        return False
    rt.hard_stop_check = safety
    rt._service_short_follow()
    assert driver.pairs == [planned_pair(owner)]
    assert not driver.stops
    if not new_pair:
        assert owner._short_follow.snapshot().plan is original_plan
        assert original_plan.expires_at == pytest.approx(10.3)


@pytest.mark.parametrize("adverse", ["expiry", "identity_rejected", "wrong_uid", "hazard"])
def test_terminal_refresh_does_not_weaken_real_stop(monkeypatch, adverse):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    calls = []
    def safety(_action):
        calls.append(1)
        if len(calls) == 2:
            clock[0] += .31 if adverse == "expiry" else .02
            if adverse in {"identity_rejected", "wrong_uid"}:
                proof = False if adverse == "identity_rejected" else ValidatedVisualObservation(
                    2, 2, 2, clock[0], clock[0], clock[0]+.5, "full")
                publish_visual_identity_evidence(owner, observation=proof, lease=None)
            return adverse == "hazard"
        return False
    rt.hard_stop_check = safety
    rt._service_short_follow()
    assert not driver.pairs
    assert driver.stops


def set_feedback(rt, clock, left, right, *, stamp=None):
    sample = SimpleNamespace(timestamp=clock[0] if stamp is None else stamp,
        left_forward_rpm=left, right_forward_rpm=right, trustworthy=True)
    rt.get_steering_feedback = lambda: sample
    return sample


def test_cap216_small_single_wheel_reverse_recovers_without_stop(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt._service_short_follow()
    for offset, pair in ((.01, (1, -5)), (.073, (33, -3)), (.12, (1, 8))):
        clock[0] = 10.+offset
        set_feedback(rt, clock, *pair)
        rt._service_short_follow()
        if offset < .12:
            assert max(map(abs, driver.pairs[-1])) <= (40 if len(driver.pairs) > 1 else 200)
    assert not driver.stops
    assert owner._short_follow.snapshot().plan is not None
    assert rt._short_follow_executor._reverse_pending is None


@pytest.mark.parametrize("stale", [False, True])
def test_small_reverse_needs_new_good_feedback_and_has_finite_deadline(monkeypatch, stale):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt._service_short_follow()
    clock[0] = 10.01
    set_feedback(rt, clock, 20, -3)
    rt._service_short_follow()
    rt._service_short_follow()  # Same sample is not a second confirmation.
    assert not driver.stops
    clock[0] = 10.17 if stale else 10.14
    if not stale:
        set_feedback(rt, clock, 20, -3)
    rt._service_short_follow()
    assert driver.stops
    assert owner._short_follow.snapshot().plan is None


@pytest.mark.parametrize("pair", [(10, -6), (-3, -4), (0, -20)])
def test_clear_reverse_still_stops_immediately(monkeypatch, pair):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt._service_short_follow()
    clock[0] += .01
    set_feedback(rt, clock, *pair)
    rt._service_short_follow()
    assert driver.stops == [1]


def test_feedback_stop_keeps_quiet_samples_while_waiting_for_new_depth(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt._service_short_follow()
    clock[0] += .01
    set_feedback(rt, clock, 20, -20)
    rt._service_short_follow()
    for offset in (.06, .11):
        clock[0] = 10.+offset
        set_feedback(rt, clock, 1, 0)
        rt._service_short_follow()
        assert len(driver.pairs) == 1  # Quiet evidence alone cannot authorize motion.
    clock[0] = 10.12
    publish(rt, clock, 2)
    rt._service_short_follow()
    assert len(driver.pairs) == 2
    assert driver.stops == [1]


def test_entry_settling_does_not_restart_feedback_stop_episode(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt._service_short_follow()
    clock[0] += .01
    set_feedback(rt, clock, 20, -20)
    rt._service_short_follow()
    writer = rt._short_follow_executor
    first_ack = writer._entry_stop_at
    clock[0] += .04
    set_feedback(rt, clock, 0, 0)
    publish(rt, clock, 2)
    rt._service_short_follow()
    assert writer._entry_stop_at == first_ack
    assert writer._entry_quiet_count == 1
    clock[0] += .04
    set_feedback(rt, clock, 0, 0)
    rt._service_short_follow()
    assert len(driver.pairs) == 2


def test_new_reverse_after_settling_without_plan_needs_new_stop_and_quiet(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt._service_short_follow()
    clock[0] = 10.01
    set_feedback(rt, clock, -20, -20)
    rt._service_short_follow()
    for now in (10.06, 10.11):
        clock[0] = now
        set_feedback(rt, clock, 0, 0)
        rt._service_short_follow()
    assert rt._short_follow_executor._entry_stop_at is None
    assert owner._short_follow.snapshot().plan is None
    clock[0] = 10.12
    set_feedback(rt, clock, -20, -20)
    rt._service_short_follow()
    assert rt._short_follow_executor._entry_stop_at == 10.12
    assert driver.stops == [1, 1]
    clock[0] = 10.29  # Fault feedback has aged out but its STOP debt survives.
    publish(rt, clock, 2)
    rt._service_short_follow()
    assert len(driver.pairs) == 1
    for now in (10.31, 10.36):
        clock[0] = now
        set_feedback(rt, clock, 0, 0)
        rt._service_short_follow()
    assert len(driver.pairs) == 2


def test_out_of_order_quiet_feedback_is_not_second_stop_confirmation(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt.get_steering_feedback = lambda: None
    rt._service_short_follow()
    clock[0] = 10.1
    set_feedback(rt, clock, 0, 0)
    rt._service_short_follow()
    set_feedback(rt, clock, 0, 0, stamp=10.05)
    rt._service_short_follow()
    assert not driver.pairs
    clock[0] = 10.15
    set_feedback(rt, clock, 0, 0)
    rt._service_short_follow()
    assert len(driver.pairs) == 1


def test_short_service_does_not_enter_legacy_axes_braking_or_recovery(monkeypatch):
    rt, owner, driver, _, _ = short_runtime(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("legacy axis/braking path was entered")
    rt._service_distance_brake = rt._service_near_yaw_park = forbidden
    rt._read_follow_axes = rt._depth_forward_continuation_limit = forbidden
    rt._distance_brake_episode = object()
    owner._near_yaw_park_request = object()
    expected = planned_pair(owner)
    rt._service_follow_wheels()
    assert driver.pairs == [expected]
    assert not driver.stops


@pytest.mark.parametrize("old_writer", [
    lambda rt, s: rt.send_robot_command(s.forward),
    lambda rt, s: rt.send_robot_command(s.rotate_left),
    lambda rt, s: rt.send_robot_command(s.stop),
    lambda rt, s: rt.send_percent_drive(80),
    lambda rt, s: rt.send_yaw_only(-8),
    lambda rt, s: rt.send_transition_stop_sequence("normal_turn"),
    lambda rt, s: rt.send_rotate_pulse_zero_stop(),
    lambda rt, s: rt.send_percent_diff(10, 1, 10, 2, "STEER"),
    lambda rt, s: rt.send_stop_with_brake_hold("queued_action_stop_signal"),
    lambda rt, s: rt.send_percent_brake(label="near_yaw_park"),
    lambda rt, s: rt._follow_motor_call("send_targets", 0, 0, "FOLLOW20_OLD_ZERO"),
    lambda rt, s: rt._follow_motor_call("send_targets", 100, -90, "FOLLOW20_OLD"),
])
def test_old_axis_and_soft_stop_writers_cannot_compete(monkeypatch, old_writer):
    rt, owner, driver, s, clock = short_runtime(monkeypatch)
    expected = planned_pair(owner)
    rt._service_short_follow()
    old_writer(rt, s)
    assert driver.pairs == [expected]
    assert not driver.stops
    assert owner._short_follow.snapshot().plan is not None


def test_real_external_stop_preempts_and_requires_new_observation(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt._service_short_follow()
    rt.send_stop_with_brake_hold("hard_stop")
    assert driver.stops == [1]
    assert owner._short_follow.snapshot().plan is None
    clock[0] += .05
    rt._service_short_follow()
    assert len(driver.pairs) == 1
    assert owner._short_follow_completed_stop_epoch == owner._short_follow.snapshot().epoch
    publish(rt, clock, 2)
    rt._service_short_follow()
    assert len(driver.pairs) == 2


def test_same_epoch_plan_update_during_prepare_uses_complete_latest_pair(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    original = rt.backend.prepare_speed_mode
    def prepare():
        original()
        if owner._short_follow.snapshot().plan.capture_id == 1:
            clock[0] += .001
            publish(rt, clock, 2, 2.1, .1)
    rt.backend.prepare_speed_mode = prepare
    rt._service_short_follow()
    plan = owner._short_follow.snapshot().plan
    assert driver.pairs == [(plan.left_rpm, -plan.right_rpm)]
    assert plan.left_rpm < plan.right_rpm
    assert not driver.stops


def test_stop_during_prepare_cannot_resume_same_old_plan(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    original = rt.backend.prepare_speed_mode
    once = [True]
    def prepare():
        original()
        if once[0]:
            once[0] = False
            rt.backend.send_stop("external_emergency", mode="emergency")
    rt.backend.prepare_speed_mode = prepare
    rt._service_short_follow()
    assert owner._short_follow.snapshot().plan is None
    clock[0] += .05
    rt._service_short_follow()
    assert not driver.pairs


@pytest.mark.parametrize("kind", ["search_request", "search_hold", "safety_hold"])
def test_new_real_hold_during_prepare_cannot_be_overwritten(monkeypatch, kind):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    original = rt.backend.prepare_speed_mode
    marker = object()
    def prepare():
        original()
        if kind == "search_request":
            rt._search_reacquire_brake_request = marker
        else:
            owner._brake_hold_active = True
            owner._brake_hold_label = "search_reacquire_brake" if kind == "search_hold" else "safety_emergency"
            rt._search_reacquire_settling = marker
    rt.backend.prepare_speed_mode = prepare
    rt._service_short_follow()
    assert not driver.pairs
    assert owner._short_follow.snapshot().plan is None
    if kind == "search_request":
        assert rt._search_reacquire_brake_request is marker
    elif kind == "search_hold":
        assert rt._search_reacquire_settling is marker
    else:
        assert driver.stops


class BeforeLock:
    def __init__(self, lock, before):
        self.lock, self.before = lock, before

    def __enter__(self):
        if self.before is not None:
            before, self.before = self.before, None
            before()
        self.lock.acquire()
        return self

    def __exit__(self, *args):
        self.lock.release()


def test_old_zero_waiting_for_io_does_not_replace_new_positive_plan(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    first = planned_pair(owner)
    rt._service_short_follow()
    clock[0] += .1
    publish(rt, clock, 2, distance=1.49)
    def newer():
        clock[0] += .1
        publish(rt, clock, 3, distance=2.)
    owner.motor_io_lock = BeforeLock(owner.motor_io_lock, newer)
    rt._service_short_follow()
    assert driver.pairs == [first, planned_pair(owner)]
    assert not driver.stops


def test_exit_waiting_for_io_cannot_stop_reactivated_normal_owner(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    first = planned_pair(owner)
    rt._service_short_follow()
    owner._short_follow.deactivate("search", clock[0])
    def reactivate():
        clock[0] += .1
        owner._short_follow.activate(1, clock[0])
        publish(rt, clock, 2)
    owner.motor_io_lock = BeforeLock(owner.motor_io_lock, reactivate)
    rt._service_short_follow()
    assert not driver.stops
    rt._service_short_follow()
    assert driver.pairs == [first, planned_pair(owner)]


def test_deactivation_requires_physical_stop_before_legacy_writer(monkeypatch):
    rt, owner, driver, s, clock = short_runtime(monkeypatch)
    rt._service_short_follow()
    owner._short_follow.deactivate("search", clock[0])
    rt.send_percent_drive(24, allow_below_min=True)
    assert len(driver.pairs) == 1  # Still fenced while STOP has not completed.
    assert rt._service_short_follow()
    assert driver.stops == [1]
    assert not rt._service_short_follow()
    assert not rt._short_follow_blocks_legacy()


@pytest.mark.parametrize("bad", [
    {"trustworthy": False}, {"left_forward_rpm": float("nan")},
    {"left_error": 1}, {"left_forward_rpm": -20}, {"right_forward_rpm": 250},
])
def test_new_adverse_feedback_stops_and_does_not_resurrect_old_plan(monkeypatch, bad):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt._service_short_follow()
    sample = SimpleNamespace(timestamp=clock[0], left_forward_rpm=0., right_forward_rpm=0., trustworthy=True)
    for key, value in bad.items():
        setattr(sample, key, value)
    rt.get_steering_feedback = lambda: sample
    clock[0] += .001
    rt._service_short_follow()
    assert driver.stops == [1]
    assert owner._short_follow.snapshot().plan is None
    rt.get_steering_feedback = lambda: SimpleNamespace(timestamp=clock[0], left_forward_rpm=0., right_forward_rpm=0., trustworthy=True)
    clock[0] += .05
    rt._service_short_follow()
    assert len(driver.pairs) == 1


def test_unknown_entry_waits_for_two_new_quiet_encoder_samples(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt.get_steering_feedback = lambda: None
    rt._service_short_follow()
    assert driver.stops == [1]
    assert not driver.pairs
    for i in (1, 2):
        clock[0] += .05
        sample = SimpleNamespace(timestamp=clock[0], left_forward_rpm=0., right_forward_rpm=0., trustworthy=True)
        rt.get_steering_feedback = lambda: sample
        rt._service_short_follow()
        if i == 1:
            rt._service_short_follow()  # Repeated sample is not another confirmation.
            assert not driver.pairs
    assert driver.pairs == [planned_pair(owner)]


def test_reverse_feedback_cannot_age_out_and_resume_without_quiet_confirmation(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt._service_short_follow()
    sample = SimpleNamespace(timestamp=clock[0], left_forward_rpm=-20., right_forward_rpm=-20., trustworthy=True)
    rt.get_steering_feedback = lambda: sample
    rt._service_short_follow()
    clock[0] += .2
    publish(rt, clock, 2)
    rt._service_short_follow()
    assert len(driver.pairs) == 1
    for capture in (3, 4):
        clock[0] += .05
        sample = SimpleNamespace(timestamp=clock[0], left_forward_rpm=0., right_forward_rpm=0., trustworthy=True)
        publish(rt, clock, capture)
        rt._service_short_follow()
    assert len(driver.pairs) == 2


def test_depth_watchdog_checks_final_io_age_without_normal_feedback_ttl(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt._service_short_follow()
    rt.get_steering_feedback = lambda: None
    clock[0] += .2
    rt._service_short_follow()
    assert len(driver.pairs) == 2
    assert not driver.stops
    clock[0] += .05
    owner.motor_io_lock = BeforeLock(owner.motor_io_lock, lambda: clock.__setitem__(0, 10.301))
    rt._service_short_follow()
    assert len(driver.pairs) == 2
    assert driver.stops == [1]


def queue_stop(owner, symbols, *, protected):
    if not hasattr(owner, "action_queue"):
        owner.action_queue = queue.Queue()
        owner.action_queue_lock = threading.Lock()
    owner.action_queue.put_nowait(ActionCommandSnapshot(symbols.stop, 1, 10., 1, 1,
        10., "manual_emergency" if protected else "normal_yaw_zero", not protected,
        protected_stop=protected))


@pytest.mark.parametrize("when", ["before_tick", "during_prepare"])
def test_protected_queue_stop_is_consumed_even_without_explicit_flag(monkeypatch, when):
    rt, owner, driver, symbols, clock = short_runtime(monkeypatch)
    if when == "before_tick":
        queue_stop(owner, symbols, protected=True)
    else:
        original = rt.backend.prepare_speed_mode
        def prepare():
            original()
            queue_stop(owner, symbols, protected=True)
        rt.backend.prepare_speed_mode = prepare
    rt._service_short_follow()
    assert driver.stops == [1]
    assert not driver.pairs
    assert owner._explicit_stop_requested
    assert owner._short_follow.snapshot().plan is None
    assert owner.action_queue.empty()


def test_ordinary_queued_stop_cannot_pause_complete_pair(monkeypatch):
    rt, owner, driver, symbols, _ = short_runtime(monkeypatch)
    expected = planned_pair(owner)
    queue_stop(owner, symbols, protected=False)
    rt._service_short_follow()
    assert driver.pairs == [expected]
    assert not driver.stops


@pytest.mark.parametrize("source", ["pending_flag", "direct_safety"])
def test_real_stop_receipt_is_available_before_uid_activation(monkeypatch, source):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    owner._short_follow.deactivate("startup", clock[0])
    owner._follow_controller.active_target_id = None
    if source == "pending_flag":
        owner._explicit_stop_requested = True
        owner._last_explicit_stop_reason = "front_ir"
        owner._explicit_stop_provenance = ("sensor_safety", "front_ir")
        owner._short_follow.revoke("explicit:front_ir", clock[0])
        assert rt._service_short_follow()
    else:
        rt.send_percent_brake(mode="emergency", label="safety_front_ir")
    assert driver.stops == [1]
    assert not driver.pairs
    state = owner._short_follow.snapshot()
    assert not state.active
    assert owner._short_follow_completed_stop_epoch == state.epoch


def test_real_two_thread_stop_revokes_writer_waiting_for_motor_without_deadlock(monkeypatch):
    rt, owner, driver, _, _ = short_runtime(monkeypatch)
    entered, revoked = threading.Event(), threading.Event()
    original_lock = owner.motor_io_lock
    owner.motor_io_lock = BeforeLock(original_lock, entered.set)
    original_revoke = owner._short_follow.revoke
    def revoke(reason, now):
        result = original_revoke(reason, now)
        revoked.set()
        return result
    owner._short_follow.revoke = revoke
    errors = []
    def run(call):
        try:
            call()
        except BaseException as error:
            errors.append(error)
    original_lock.acquire()
    writer = threading.Thread(target=run, args=(rt._service_short_follow,), daemon=True)
    stopper = threading.Thread(target=run, args=(lambda: rt.send_stop_with_brake_hold("hard_stop"),), daemon=True)
    try:
        writer.start()
        assert entered.wait(1)
        stopper.start()
        assert revoked.wait(1), "revocation must not wait for the executor's serial wait"
        assert owner._short_follow.snapshot().plan is None
    finally:
        original_lock.release()
    writer.join(1)
    stopper.join(1)
    assert not writer.is_alive() and not stopper.is_alive()
    assert not errors
    assert not driver.pairs
    assert driver.stops


@pytest.mark.parametrize("maximum", [0, 10, 20, 40, 200])
def test_paired_mode_never_raises_existing_motor_hard_limit(monkeypatch, maximum):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt.backend.config = replace(rt.backend.config, max_target=maximum)
    clock[0] += .001
    publish(rt, clock, 2, distance=2.4, x=.1)
    rt._service_short_follow()
    if maximum == 0:
        assert driver.stops and not driver.pairs
    else:
        left, right = driver.pairs[-1]
        assert 0 < left < -right <= min(maximum, owner._short_follow.config.max_rpm)
        actual = owner._short_follow_last_applied_plan
        assert (actual.left_rpm, -actual.right_rpm) == (left, right)


def test_40ms_serial_writes_keep_50ms_start_cadence_not_90ms(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    starts = []
    original = driver.set_left_speed
    def delayed_write(value):
        starts.append(clock[0])
        original(value)
        clock[0] += .04
    driver.set_left_speed = delayed_write
    rt._service_short_follow()
    assert clock[0] == pytest.approx(10.04)
    clock[0] = 10.045
    rt._service_short_follow()
    assert len(driver.pairs) == 1
    for timestamp in (10.05, 10.10):
        clock[0] = timestamp
        rt._service_short_follow()
    assert starts == pytest.approx([10., 10.05, 10.10])
    assert owner._last_motor_dispatch_ts == pytest.approx(10.14)  # ACK clock is still actual completion.


def test_slow_serial_skips_missed_ticks_without_burst_and_safety_is_unthrottled(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    original = driver.set_left_speed
    def delayed_write(value):
        original(value)
        clock[0] += .13
    driver.set_left_speed = delayed_write
    rt._service_short_follow()
    assert clock[0] == pytest.approx(10.13)
    for _ in range(4):
        rt._service_short_follow()
    assert len(driver.pairs) == 1
    clock[0] = 10.14  # Still before the next scheduled 10.15 write.
    owner._explicit_stop_requested = True
    rt._service_short_follow()
    assert driver.stops == [1]
    assert len(driver.pairs) == 1


def test_fresh_feedback_allows_actual_pi_pair_above_old_40_rpm_limit(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    clock[0] += .001
    plan = publish(rt, clock, 2, distance=3., x=.1)
    assert max(plan.left_rpm, plan.right_rpm) > 40
    rt._service_short_follow()
    assert driver.pairs == [planned_pair(owner)]
    assert max(map(abs, driver.pairs[-1])) > 40
    assert not driver.stops


@pytest.mark.parametrize("missing", [True, False])
def test_feedback_gap_caps_pair_instead_of_zero_without_renewing_depth(monkeypatch, missing):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    plan = owner._short_follow.snapshot().plan
    rt._service_short_follow()
    assert max(map(abs, driver.pairs[-1])) > 40
    if missing:
        rt.get_steering_feedback = lambda: None
    else:
        rt.get_steering_feedback = lambda: SimpleNamespace(timestamp=10.,
            left_forward_rpm=90., right_forward_rpm=90., trustworthy=True)
    clock[0] += .16
    rt._service_short_follow()
    assert driver.pairs[-1] == planned_pair(owner, 40)
    assert not driver.stops
    assert owner._short_follow.snapshot().plan.expires_at == plan.expires_at
    clock[0] = plan.expires_at + .001
    rt._service_short_follow()
    assert driver.stops == [1]


def test_terminal_feedback_gap_scales_both_wheels_and_logs_reason(monkeypatch, caplog):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    clock[0] += .001
    publish(rt, clock, 2, distance=3., x=.1)
    original = rt.backend.prepare_speed_mode
    def prepare():
        original()
        rt.get_steering_feedback = lambda: None
    rt.backend.prepare_speed_mode = prepare
    with caplog.at_level("INFO"):
        rt._service_short_follow()
    assert driver.pairs == [planned_pair(owner, 40)]
    assert 0 < driver.pairs[-1][0] < -driver.pairs[-1][1] == 40
    assert "feedback_speed_cap=40" in caplog.text
    assert not driver.stops


def test_lower_current_request_does_not_reclassify_lawful_old_speed_as_fault(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    clock[0] += .001
    previous = publish(rt, clock, 2, distance=4.)
    rt._service_short_follow()
    prior_speed = max(previous.left_rpm, previous.right_rpm)
    clock[0] += .1
    rt.get_steering_feedback = lambda: SimpleNamespace(timestamp=clock[0],
        left_forward_rpm=prior_speed, right_forward_rpm=prior_speed, trustworthy=True)
    lower = publish(rt, clock, 3, distance=1.7)
    assert 0 < max(lower.left_rpm, lower.right_rpm) < prior_speed
    rt._service_short_follow()
    assert driver.pairs[-1] == planned_pair(owner)
    assert not driver.stops


def test_successful_backend_limit_reports_actual_base_not_encoder_lag(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt.backend.config = replace(rt.backend.config, max_target=30)
    reported = []
    original = owner._short_follow.acknowledge_output
    def acknowledge(plan, base):
        reported.append((plan, base))
        return original(plan, base)
    owner._short_follow.acknowledge_output = acknowledge
    rt._service_short_follow()
    assert len(reported) == 1
    assert reported[0][1] == 30  # The current encoder is zero, not the limit.
    assert driver.pairs[-1] == (30, -30)
    assert owner._short_follow_last_applied_plan.base_rpm == 30


def test_failed_dual_write_never_reports_an_applied_pi_output(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    reported = []
    owner._short_follow.acknowledge_output = lambda *args: reported.append(args)
    def fail(_value):
        raise OSError("left wheel did not acknowledge")
    driver.set_left_speed = fail
    with pytest.raises(Exception):
        rt._service_short_follow()
    assert not reported
    assert rt.backend.motion_write_fault


def test_takeover_from_known_legal_high_forward_receipt_needs_no_zero(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt.backend.send_targets(120, -120, "previous_legal_forward", history_uid=1)
    rt.get_steering_feedback = lambda: SimpleNamespace(timestamp=clock[0],
        left_forward_rpm=120., right_forward_rpm=120., trustworthy=True)
    rt._service_short_follow()
    assert driver.pairs[-1] == planned_pair(owner)
    assert len(driver.pairs) == 2
    assert not driver.stops


def test_absolute_overspeed_respects_lower_motor_hardware_limit(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt.backend.config = replace(rt.backend.config, max_target=80)
    rt._service_short_follow()
    clock[0] += .001
    rt.get_steering_feedback = lambda: SimpleNamespace(timestamp=clock[0],
        left_forward_rpm=100., right_forward_rpm=100., trustworthy=True)
    rt._service_short_follow()
    assert driver.stops == [1]
    assert owner._short_follow.snapshot().plan is None


def test_entry_settling_new_depth_cannot_accumulate_integral_during_completed_stop(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt.get_steering_feedback = lambda: None
    initial_epoch = owner._short_follow.snapshot().epoch
    for capture in range(2, 23):
        clock[0] += .1
        publish(rt, clock, capture, distance=1.6)
        rt._service_short_follow()
        assert owner._short_follow._integral_m_s == pytest.approx(0.)
    assert not driver.pairs
    assert 1 <= len(driver.stops) <= 3  # Most waits used STOP throttling.
    assert owner._short_follow.snapshot().epoch == initial_epoch
    for capture in (23, 24):
        clock[0] += .05
        rt.get_steering_feedback = lambda: SimpleNamespace(timestamp=clock[0],
            left_forward_rpm=0., right_forward_rpm=0., trustworthy=True)
        publish(rt, clock, capture, distance=1.6)
        rt._service_short_follow()
    assert len(driver.pairs) == 1
    # Only the final newly executable sample can add a small PI increment.
    assert owner._short_follow._integral_m_s <= .4 * .17 * .05 + 1e-9


def test_failed_stop_does_not_report_zero_pi_output(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt.get_steering_feedback = lambda: None
    reported = []
    owner._short_follow.acknowledge_output = lambda *args: reported.append(args)
    def fail(*args, **kwargs):
        raise OSError("STOP failed")
    rt.backend.send_stop = fail
    with pytest.raises(OSError, match="STOP failed"):
        rt._service_short_follow()
    assert not reported
