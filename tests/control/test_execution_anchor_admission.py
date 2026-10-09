"""PI execution evidence must remain current through depth admission.

Real PI, admission, and motor receipt bookkeeping; fake driver/clock only.
"""

from types import SimpleNamespace

import pytest

import request_0513_modular as runtime_module
from car_control_modular.longitudinal_execution import ForwardExecutionAnchor
from car_control_modular.action_runtime import MotionActionRuntime
from car_control_modular.mssd_motor import MssdMotorBackend, MssdMotorConfig
from test_depth_authority_250 import authority, advance, seed
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


class AdmissionDriver:
    def __init__(self):
        self.right = 0
        self.pairs = []
        self.stops = []

    def set_right_speed(self, value):
        self.right = int(value)

    def set_left_speed(self, value):
        self.pairs.append((int(value), self.right))

    def stop_all(self, mode=0):
        self.stops.append(int(mode))


@pytest.fixture
def pending_admission(authority):
    a = authority
    stamp, _ = seed(a, distance=2.5, rpm=40.)
    backend = MssdMotorBackend(MssdMotorConfig(
        port="unused-offline", slave_id=1, baudrate=115200, timeout=.1,
        lib_dir="unused-offline", max_target=200, percent_limit=100,
        left_sign=-1, right_sign=1, forward_target_sign=-1,
        m1_is_left_wheel=True, exit_parking_mode_on_arm=False,
        stop_mode="emergency", stop_zero_delay_sec=0., startup_parking_enabled=False,
    ))
    backend.driver = AdmissionDriver()  # No serial client import or open.
    a.owner.motor_io_lock = backend.io_lock
    advance(a, stamp + .130)
    backend.send_targets(40, -40, "FOLLOW20")
    receipt = backend.last_speed_receipt
    anchor = ForwardExecutionAnchor(1, stamp, 40., receipt.completed_at, receipt)

    def reader(uid, sample_timestamp, now):
        linear = a.owner._depth30_linear_snapshot
        if (uid == anchor.uid and sample_timestamp == anchor.sample_timestamp
                and backend.last_speed_receipt is anchor.receipt
                and linear is not None and linear[0] == "forward" and linear[1] > 0
                and linear[2:] == (uid, sample_timestamp)
                and 0 <= now - anchor.sent_at <= .1
                and 0 <= now - anchor.sample_timestamp <= .25):
            return anchor
        return None

    a.controller._longitudinal_execution_reader = reader
    advance(a, stamp + .181)
    a.owner._depth30_continuation_veto = (1, stamp)
    current = a.frame(2.5, rpm=0., stamp=stamp + .129)
    a.feedback = current.steering_feedback
    decision = a.controller.decide(10, current, longitudinal_only=True)
    assert a.controller.last_distance_pid_result.output_rpm == 52
    assert any(action.kind == "forward" and action.speed_percent > 0
               for action in decision.actions)
    assert reader(1, stamp, a.clock.now) is anchor
    return SimpleNamespace(a=a, backend=backend, anchor=anchor, old_stamp=stamp,
                           frame=current, decision=decision)


def commit(pending):
    return pending.a.owner._commit_depth_linear_decision(
        pending.decision, pending.frame, 1, is_fresh_depth=True,
    )


def assert_no_forward(actions):
    assert not any(action.kind == "forward" and action.speed_percent > 0 for action in actions)


def test_unchanged_completed_packet_permits_new_sample_admission(pending_admission):
    p = pending_admission
    actions, accepted = commit(p)
    assert accepted
    assert [(action.kind, action.speed_percent) for action in actions] == [("forward", 26)]
    assert p.a.owner._depth30_linear_snapshot == ("forward", 26, 1, p.frame.distance_state.sample_timestamp)
    assert p.backend.last_speed_receipt is p.anchor.receipt
    assert p.backend.driver.pairs == [(40, -40)]  # Admission itself does no I/O.


def test_successful_admission_consumes_proof_without_rejecting_its_own_new_snapshot(pending_admission):
    p = pending_admission
    _, accepted = commit(p)
    assert accepted
    controller = p.a.controller
    sample = p.frame.distance_state.sample_timestamp
    snapshot = p.a.owner._depth30_linear_snapshot
    timing = p.a.owner._depth30_linear_timing
    watermark = p.a.owner._depth30_linear_sample_watermark
    # The new grant owns a new sample, so the old execution reader correctly
    # stops returning that old proof; the admitted result must remain usable.
    assert controller._longitudinal_execution_reader(1, p.old_stamp, p.a.clock.now) is None
    assert controller.distance_only_forward_percent(p.frame, sample) == 26
    assert not controller._distance_pid._distance_pi._execution_suspended

    advance(p.a, p.a.clock.now + .01)
    assert controller.distance_only_forward_percent(p.frame, sample) == 26
    actions, accepted = commit(p)
    assert not accepted and not actions
    assert p.a.owner._depth30_linear_snapshot is snapshot
    assert p.a.owner._depth30_linear_timing is timing
    assert p.a.owner._depth30_linear_sample_watermark == watermark
    assert p.backend.driver.pairs == [(40, -40)]


@pytest.mark.parametrize("event", ["zero", "stop", "same_writer", "different_writer"])
def test_packet_changed_after_pi_before_commit_cannot_authorize_forward(pending_admission, event):
    p = pending_admission
    if event == "stop":
        p.backend.send_stop("AFTER_PI", mode="emergency")
    else:
        pair = {"zero": (0, 0), "same_writer": (40, -40), "different_writer": (30, -20)}[event]
        p.backend.send_targets(*pair, "OTHER_WRITER_AFTER_PI")
    assert p.backend.last_speed_receipt is not p.anchor.receipt
    writes = (list(p.backend.driver.pairs), list(p.backend.driver.stops))

    actions, accepted = commit(p)
    assert accepted  # Record the new sample's rejection/zero watermark.
    assert_no_forward(actions)
    assert p.a.owner._depth30_linear_snapshot is None
    assert p.a.controller._distance_pid._distance_pi._execution_suspended
    assert (p.backend.driver.pairs, p.backend.driver.stops) == writes

    # Both the already processed NEW physical sample and the old source sample
    # remain unusable; a rejected cached PI request cannot become a later grant.
    for stamp in (p.frame.distance_state.sample_timestamp, p.old_stamp):
        advance(p.a, p.a.clock.now + .002)
        repeated = p.a.frame(2.5, rpm=0., stamp=stamp)
        decision = p.a.controller.decide(10, repeated, longitudinal_only=True)
        actions, _ = p.a.owner._commit_depth_linear_decision(
            decision, repeated, 1, is_fresh_depth=True,
        )
        assert_no_forward(actions)
        assert p.a.owner._depth30_linear_snapshot is None
    assert (p.backend.driver.pairs, p.backend.driver.stops) == writes


def test_receipt_expiring_after_pi_before_commit_cannot_donate_ramp_budget(pending_admission):
    p = pending_admission
    advance(p.a, p.anchor.sent_at + .100001)
    assert p.a.clock.now - p.frame.distance_state.sample_timestamp < .18
    assert p.backend.last_speed_receipt is p.anchor.receipt
    actions, _ = commit(p)
    assert_no_forward(actions)
    assert p.a.owner._depth30_linear_snapshot is None
    assert p.a.controller._distance_pid._distance_pi._execution_suspended


def test_zero_during_final_approval_callback_is_rechecked_before_publish(pending_admission, monkeypatch):
    p = pending_admission
    original = p.a.controller.accept_longitudinal_limit
    calls = []

    def accept_then_zero(sample_timestamp, approved_rpm):
        calls.append((sample_timestamp, approved_rpm))
        original(sample_timestamp, approved_rpm)
        p.backend.send_targets(0, 0, "ZERO_DURING_ADMISSION")

    monkeypatch.setattr(p.a.controller, "accept_longitudinal_limit", accept_then_zero)
    actions, accepted = commit(p)
    assert calls == [(p.frame.distance_state.sample_timestamp, 52.)]
    assert accepted
    assert_no_forward(actions)
    assert p.a.owner._depth30_linear_snapshot is None
    assert p.a.controller._distance_pid._distance_pi._execution_suspended
    assert p.backend.driver.pairs == [(40, -40), (0, 0)]


@pytest.mark.parametrize("source", ["receipt", "anchor"])
def test_read_only_execution_reader_rechecks_source_identity_after_state_checks(
    pending_admission, source,
):
    p = pending_admission
    probe = SimpleNamespace(
        _forward_execution_anchor=p.anchor, backend=p.backend, owner=p.a.owner,
    )

    def state_check():
        # Deterministically model another writer publishing its state after
        # the reader's first identity check, without threads or motor I/O.
        if source == "receipt":
            p.backend.last_speed_receipt = None
        else:
            probe._forward_execution_anchor = None
        return True

    probe._visible_wheel_control_active = state_check
    before = (p.a.owner._depth30_linear_snapshot, p.a.owner._depth30_linear_timing)
    assert MotionActionRuntime.forward_execution_anchor(
        probe, 1, p.old_stamp, p.a.clock.now,
    ) is None
    assert (p.a.owner._depth30_linear_snapshot, p.a.owner._depth30_linear_timing) == before
    assert p.backend.driver.pairs == [(40, -40)]
    assert not p.backend.driver.stops


def test_final_proof_check_publish_and_consume_hold_the_motor_lock(pending_admission, monkeypatch):
    p = pending_admission
    controller = p.a.controller
    original_check = controller.longitudinal_execution_proof_valid
    original_consume = controller.consume_longitudinal_execution_proof
    checks, consumes = [], []

    def checked(sample_timestamp, now):
        checks.append(p.backend.io_lock.locked())
        assert p.a.owner._depth30_linear_snapshot[3] == p.old_stamp
        return original_check(sample_timestamp, now)

    def consumed(sample_timestamp):
        consumes.append(p.backend.io_lock.locked())
        assert p.backend.io_lock.locked()
        assert p.a.owner._depth30_linear_snapshot == ("forward", 26, 1, sample_timestamp)
        assert controller._distance_pi_execution_anchor_proof is not None
        original_consume(sample_timestamp)
        assert controller._distance_pi_execution_anchor_proof is None

    monkeypatch.setattr(controller, "longitudinal_execution_proof_valid", checked)
    monkeypatch.setattr(controller, "consume_longitudinal_execution_proof", consumed)
    with p.a.owner._control_update_lock:
        actions, accepted = commit(p)
    assert accepted and actions[0].speed_percent == 26
    assert checks == [False, True]  # Initial budget read, then atomic publication.
    assert consumes == [True]
    assert not p.backend.io_lock.locked()
    assert p.backend.driver.pairs == [(40, -40)]


@pytest.mark.parametrize("event", [
    "zero", "stop", "receipt_expired", "new_depth_expired", "dispatch_budget",
])
def test_waiting_for_motor_lock_rechecks_packet_and_physical_deadlines(
    pending_admission, monkeypatch, caplog, event,
):
    caplog.set_level("INFO")
    p = pending_admission
    entered = []
    sample = p.frame.distance_state.sample_timestamp
    if event == "new_depth_expired":
        # Isolate the NEW physical-sample gate: even a proof provider which
        # still returns its cached token cannot admit a >180ms measurement.
        monkeypatch.setattr(p.a.controller, "_longitudinal_execution_reader", lambda *args: p.anchor)
    if event == "dispatch_budget":
        monkeypatch.setattr(runtime_module, "MOTOR_RS485_TARGET_MIN_INTERVAL_SEC", .17)

    class OtherWriterBeforeAcquire:
        def __enter__(self):
            # Run the competing transaction to completion before this caller
            # obtains the shared lock. No scheduling, sleep, or real serial I/O.
            with p.backend.io_lock:
                entered.append(event)
                if event == "zero":
                    p.backend.send_targets(0, 0, "ZERO_WHILE_ADMISSION_WAITED")
                elif event == "stop":
                    p.backend.send_stop("STOP_WHILE_ADMISSION_WAITED", mode="emergency")
                elif event == "receipt_expired":
                    advance(p.a, p.anchor.sent_at + .100001)
                elif event == "new_depth_expired":
                    advance(p.a, sample + .180001)
                else:
                    advance(p.a, p.old_stamp + .22)
                    assert p.a.controller._longitudinal_execution_reader(1, p.old_stamp, p.a.clock.now) is p.anchor
                    assert p.a.clock.now - sample < .18
            p.backend.io_lock.acquire()
            return self

        def __exit__(self, *_args):
            p.backend.io_lock.release()

    p.a.owner.motor_io_lock = OtherWriterBeforeAcquire()
    with p.a.owner._control_update_lock:
        actions, accepted = commit(p)
    assert entered == [event]
    assert accepted  # The rejected sample advances the watermark with zero.
    assert_no_forward(actions)
    assert p.a.owner._depth30_linear_snapshot is None
    assert p.a.controller._distance_pid._distance_pi._execution_suspended
    assert p.a.controller._distance_pi_execution_anchor_proof is not None
    assert not p.backend.io_lock.locked()
    assert p.backend.driver.pairs == ([(40, -40), (0, 0)] if event == "zero" else [(40, -40)])
    assert p.backend.driver.stops == ([1] if event == "stop" else [])
    veto_logs = [record.getMessage() for record in caplog.records
                 if record.getMessage().startswith("depth_execution_admission_veto ")]
    assert len(veto_logs) == 1
    expected_reason = {
        "new_depth_expired": "physical_sample_changed_or_expired",
        "dispatch_budget": "dispatch_budget",
    }.get(event, "execution_receipt_invalid")
    assert f"reason={expected_reason} " in veto_logs[0]
    assert f"sample_age_ms={(p.a.clock.now-sample)*1000.:.1f} " in veto_logs[0]
    assert f"remaining_ms={(sample+.25-p.a.clock.now)*1000.:.1f} " in veto_logs[0]
    expected_budget = 170. if event == "dispatch_budget" else max(
        30., runtime_module.MOTOR_RS485_TARGET_MIN_INTERVAL_SEC*1000.)
    assert f"budget_ms={expected_budget:.1f} " in veto_logs[0]
    # Diagnostic labels must not change the original rejection/state-machine
    # contract, and physical/budget rejection must still short-circuit receipts.
    assert actions[0].reason == "execution_packet_changed_before_admission"
    receipt_veto = "distance_pi_execution_continuity_veto"
    assert (receipt_veto in caplog.text) == (event in {"zero", "stop", "receipt_expired"})
