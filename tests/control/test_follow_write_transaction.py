"""CAP87: production PI/grant/writer interleavings, fake serial and clock only."""
from dataclasses import replace

import pytest

from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner
from test_depth_authority_250 import authority, advance, decide_commit
from test_feedback_interval_execution import execution_case, admit
from test_execution_anchor_admission import AdmissionDriver
from car_control_modular.executed_speed_budget import record_completed_speed
from car_control_modular.mssd_motor import MssdMotorBackend
import car_control_modular.action_runtime as action_module


@pytest.mark.parametrize("phase", ["before_right", "after_right", "before_left", "after_left", "before_ledger"])
def test_normal_dual_write_never_revokes_covered_sample(execution_case, phase):
    a = execution_case()
    admit(a)
    old_receipt = a.backend.last_speed_receipt
    old_grant = a.owner._fresh_depth_linear_snapshot(1)
    seen = []

    def inspect():
        seen.append((a.backend.last_speed_receipt,
                     a.action.braking_interval_continuation_bound_rpm(1, a.stamp, a.clock.now),
                     a.owner._fresh_depth_linear_snapshot(1)))
        # Feedback retirement racing a write must not poison prior provenance.
        a.action._observe_executed_speed_response(a.feedback)

    for side in ("right", "left"):
        original = getattr(a.backend.driver, "set_"+side+"_speed")
        def hook(value, _side=side, _original=original):
            if phase == "before_"+_side:
                inspect()
            _original(value)
            if phase == "after_"+_side:
                inspect()
        setattr(a.backend.driver, "set_"+side+"_speed", hook)
    note = a.action._note_follow_packet
    def after_ack(*args, **kwargs):
        if phase == "before_ledger":
            inspect()
        return note(*args, **kwargs)
    a.action._note_follow_packet = after_ack
    a.action._service_follow_wheels()
    assert len(seen) == 1
    receipt, bound, grant = seen[0]
    assert bound == 76.
    assert grant == old_grant
    if phase != "before_ledger":
        assert receipt is None  # Never masquerade a pending write as an ACK.
    else:
        assert receipt is not old_receipt
    assert a.backend.driver.pairs[-1] == (76, -76)
    assert a.owner._fresh_depth_linear_snapshot(1) == old_grant
    assert a.owner._depth30_linear_timing is a.timing
    assert a.timing.depth_expires_at == pytest.approx(a.stamp+.30)
    assert a.action._continuation_executed_speed_history[-1].receipt is a.backend.last_speed_receipt


@pytest.mark.parametrize("event", ["uid", "ttl", "feedback_stale", "explicit_stop", "identity_revoke"])
def test_transaction_accounting_does_not_keep_invalid_motion_alive(execution_case, event):
    a = execution_case()
    admit(a)
    old = a.backend.driver.set_right_speed
    def during(value):
        old(value)
        if event == "uid":
            a.controller.active_target_id = 2
        elif event == "ttl":
            a.clock.now = a.stamp+.300001
        elif event == "feedback_stale":
            a.feedback = replace(a.feedback, timestamp=a.clock.now-.151)
        elif event == "explicit_stop":
            a.owner._explicit_stop_requested = True
        else:
            a.owner._revoke_depth_linear_authority("identity_rejected")
        assert a.owner._fresh_depth_linear_snapshot(1) is None
    a.backend.driver.set_right_speed = during
    a.action._service_follow_wheels()
    a.backend.driver.set_right_speed = old
    before = len(a.backend.driver.pairs)
    a.action._service_follow_wheels()
    assert not any(left > 0 and right < 0 for left, right in a.backend.driver.pairs[before:])


@pytest.mark.parametrize("side", ["right", "left"])
def test_partial_write_failure_discards_transaction_and_stays_stopped(execution_case, side):
    a = execution_case()
    admit(a)
    original = getattr(a.backend.driver, "set_"+side+"_speed")
    def fail_nonzero(value):
        if value:
            raise OSError("test speed ACK failure")
        original(value)
    setattr(a.backend.driver, "set_"+side+"_speed", fail_nonzero)
    # Runtime may propagate the fault; backend must latch it in either case.
    try:
        a.action._service_follow_wheels()
    except OSError:
        pass
    assert a.backend.motion_write_fault
    assert a.backend.last_speed_receipt is None
    assert a.backend.last_speed_write is None
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    assert a.backend.driver.stops


@pytest.mark.parametrize("entry", ["untracked", "stop", "uid", "missing_anchor"])
def test_pending_pair_cannot_fabricate_an_interval_chain(execution_case, entry):
    a = execution_case()
    admit(a)
    if entry == "stop":
        a.backend.send_stop("test", mode="emergency")
    elif entry == "missing_anchor":
        a.action._continuation_executed_speed_history = ()
    seen = []
    old = a.backend.driver.set_right_speed
    def during(value):
        old(value)
        seen.append(a.action.braking_interval_continuation_bound_rpm(1, a.stamp, a.clock.now))
    a.backend.driver.set_right_speed = during
    a.backend.send_targets(76, -76, "test", history_uid=None if entry == "untracked" else 2 if entry == "uid" else 1)
    assert seen == [None]
    assert a.action.braking_interval_continuation_bound_rpm(1, a.stamp, a.clock.now) is None


@pytest.mark.parametrize("phase", ["before_right", "after_right", "after_left", "before_ledger"])
def test_cap87_new_depth_covers_prior_legitimate_inflight_38rpm(execution_case, phase):
    a = execution_case()
    # Replace fixture's initial 76RPM, not just its ledger: the actual history
    # for this log trace is completed28 -> admitted38 -> concurrent new Depth.
    a.backend = MssdMotorBackend(a.backend.config)
    a.backend.driver = AdmissionDriver()
    a.action.backend = a.backend
    a.owner.motor_io_lock = a.backend.io_lock
    a.clock.now = 99.99
    a.backend.send_targets(28, -28, "PRIOR_COMPLETED")
    receipt = a.backend.last_speed_receipt
    a.action._continuation_executed_speed_history = record_completed_speed(
        (), uid=1, applied=(28, 28), signs=(1, -1), receipt=receipt,
        previous_receipt=None, now=a.clock.now, packet_written=True)
    a.clock.now = 100.09
    a.owner._last_vision_control_ts = 100.08
    decide_commit(a, a.frame(1.7747243, rpm=38., stamp=100.))
    assert a.owner._fresh_depth_linear_snapshot(1) == ("forward", 19, 1, 100.)
    seen = []
    def new_depth():
        a.clock.now = 100.224
        a.owner._last_vision_control_ts = 100.22
        decide_commit(a, a.frame(1.78600455, rpm=4.5, stamp=100.1344))
        seen.append(a.owner._fresh_depth_linear_snapshot(1))
        assert a.owner._depth30_linear_timing.braking_assessment.travel_bound_rpm == 38.
        assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(100.4344)
    originals = {}
    for side in ("right", "left"):
        original = getattr(a.backend.driver, "set_"+side+"_speed")
        originals[side] = original
        def hook(value, _side=side, _original=original):
            if phase == "before_"+_side:
                new_depth()
            _original(value)
            if phase == "after_"+_side:
                new_depth()
        setattr(a.backend.driver, "set_"+side+"_speed", hook)
    note = a.action._note_follow_packet
    def after_ack(*args, **kwargs):
        if phase == "before_ledger":
            new_depth()
        return note(*args, **kwargs)
    a.action._note_follow_packet = after_ack
    a.action._service_follow_wheels()
    assert a.backend.driver.pairs == [(28, -28), (38, -38)]
    assert len(seen) == 1 and seen[0] is not None and seen[0][1] > 2
    assert a.owner._fresh_depth_linear_snapshot(1) == seen[0]
    assert getattr(a.owner, "_depth30_read_veto", None) is None
    # Follow through to physical dispatch, not only nonzero metadata.
    a.action._note_follow_packet = note
    for side, original in originals.items():
        setattr(a.backend.driver, "set_"+side+"_speed", original)
    a.action._service_follow_wheels()
    assert a.backend.driver.pairs[-1][0] > 0 > a.backend.driver.pairs[-1][1]
    assert (0, 0) not in a.backend.driver.pairs


def test_feedback_only_retirement_during_read_keeps_conservative_interval(execution_case, monkeypatch):
    a = execution_case()
    admit(a)
    original = action_module.executed_interval_speed_bound_rpm
    calls = []
    def retiring(history, **kwargs):
        result = original(history, **kwargs)
        calls.append(result)
        # Immutable publication of response evidence, same actual receipt.
        a.action._continuation_executed_speed_history = tuple(replace(x) for x in history)
        return result
    monkeypatch.setattr(action_module, "executed_interval_speed_bound_rpm", retiring)
    assert a.action.braking_interval_continuation_bound_rpm(1, a.stamp, a.clock.now) == 76.
    assert a.owner._fresh_depth_linear_snapshot(1) is not None
    assert calls and all(x == 76. for x in calls)


@pytest.mark.parametrize("replacement", ["untracked", "stop", "fault"])
def test_changed_physical_write_during_interval_read_never_uses_old_snapshot(execution_case, monkeypatch, replacement):
    a = execution_case()
    admit(a)
    original = action_module.executed_interval_speed_bound_rpm
    seen = []
    def changed(history, **kwargs):
        result = original(history, **kwargs)
        if not seen:
            seen.append(True)
            if replacement == "stop":
                a.backend.send_stop("EXPLICIT", mode="emergency")
            elif replacement == "fault":
                a.backend._record_motion_write_fault("test", OSError("link failure"))
            else:
                a.backend.send_targets(110, -110, "OTHER_WRITER")
        return result
    monkeypatch.setattr(action_module, "executed_interval_speed_bound_rpm", changed)
    assert a.action.braking_interval_continuation_bound_rpm(1, a.stamp, a.clock.now) is None
    assert a.owner._fresh_depth_linear_snapshot(1) is None
