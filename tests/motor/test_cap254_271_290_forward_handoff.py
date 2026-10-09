"""Real writer, fake hardware: ordinary yaw/encoder publication is not STOP."""
from dataclasses import replace

import pytest

from test_unified_forward_snapshot import feedback, writer
from test_follow_refresh_audit import cross_yaw_deadline
from test_zero_publication_lifecycle import churn_at_three_terminal_checks


@pytest.mark.parametrize("state", ["resume", "obsolete_pivot", "negative_quantization"])
@pytest.mark.parametrize("cap_call", [1, 2])
def test_expired_yaw_preserves_new_forward_after_wheel_guard(monkeypatch, state, cap_call):
    rt, owner, driver, clock, _, _ = writer(monkeypatch, base=52., yaw=-7.)
    rt.config.follow_forward_handoff_enable = True
    original_depth = owner._depth30_linear_snapshot
    old_intent = owner._lateral_intent_store.snapshot()
    calls = cross_yaw_deadline(monkeypatch, rt, owner, clock, cap_call=cap_call)
    if state == "resume":
        rt._visible_wheel_guard.resume_signs = (1, 1)
    elif state == "obsolete_pivot":
        rt._visible_wheel_guard.pending_signs = (-1, 1)
        rt._visible_wheel_guard.started = 10.1
    else:
        rt._steering_feedback = feedback(clock[0], -1., 3.)
    rt._service_follow_wheels()
    assert len(calls) >= cap_call
    assert driver.pairs == [(45, -59), (52, -52)]
    assert not driver.stops and not old_intent.valid(clock[0])
    assert owner._depth30_linear_snapshot is original_depth
    assert rt._forward_execution_anchor.sample_timestamp == original_depth[3]
    assert not owner.motor_io_lock.locked()


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("changes", [1, 2])
def test_same_depth_yaw_changes_use_guard_not_stricter_handoff_feedback(monkeypatch, side, changes):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=0.)
    depth = owner._depth30_linear_snapshot
    reads = []
    sign = -1 if side == "left" else 1

    def cached_feedback():
        reads.append(True)
        if len(reads) <= changes:
            publish(60., sign*(7. if len(reads) == 1 else 10.), new_depth=False)
        values = (-1., 3.) if side == "left" else (3., -1.)
        rt._steering_feedback = feedback(clock[0], *values)
        return rt._steering_feedback

    monkeypatch.setattr(rt, "get_steering_feedback", cached_feedback)
    rt._service_follow_wheels()
    yaw = sign*(7 if changes == 1 else 10)
    assert driver.pairs == [(60, -60), (60+yaw, -(60-yaw))]
    assert owner._depth30_linear_snapshot is depth
    assert not driver.stops


@pytest.mark.parametrize("change", ["same", "newer_same", "slower", "faster", "reverse",
                                  "stale", "error", "older", "stop"])
def test_churn_receipt_requalifies_encoder_values_not_object_identity(monkeypatch, change):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=-7.)
    receipt = rt.backend.last_speed_receipt
    original = rt._current_receipt_survives_publication
    cap = owner._depth_forward_continuation_limit
    state = {"in_check": False, "updated": False}

    def check(*args, **kwargs):
        state["in_check"] = True
        try:
            return original(*args, **kwargs)
        finally:
            state["in_check"] = False

    def limit(*args, **kwargs):
        result = cap(*args, **kwargs)
        if state["in_check"] and not state["updated"]:
            state["updated"] = True
            current = rt._steering_feedback
            if change == "newer_same":
                clock[0] += .001
                current = replace(current, timestamp=clock[0])
            elif change == "slower":
                current = replace(current, left_forward_rpm=23., right_forward_rpm=22.)
            elif change == "faster":
                current = replace(current, left_forward_rpm=25.)
            elif change == "reverse":
                current = replace(current, left_forward_rpm=-2.)
            elif change == "stale":
                current = replace(current, timestamp=clock[0]-.151)
            elif change == "older":
                current = replace(current, timestamp=current.timestamp-.001)
            elif change == "error":
                current = replace(current, left_error=1)
            elif change == "stop":
                rt.backend.send_stop("new-stop", mode="emergency", preserve_zero=True)
            rt._steering_feedback = replace(current)
        return result

    monkeypatch.setattr(rt, "_current_receipt_survives_publication", check)
    monkeypatch.setattr(owner, "_depth_forward_continuation_limit", limit)
    churn_at_three_terminal_checks(monkeypatch, rt, clock, publish, base=80., yaw=-10.)
    rt._service_follow_wheels()
    assert state["updated"]
    if change in {"same", "newer_same", "slower"}:
        assert driver.pairs == [(53, -67)] and not driver.stops
        assert rt.backend.last_speed_receipt is receipt
    elif change == "stop":
        assert driver.pairs == [(53, -67)] and driver.stops == [1]
    else:
        assert driver.pairs == [(53, -67), (0, 0)]


@pytest.mark.parametrize("fault", ["reverse_motion", "reverse_provenance", "depth", "visual", "stop"])
def test_extended_snapshot_scope_does_not_bypass_real_reverse_or_revocation(monkeypatch, fault):
    rt, owner, driver, clock, _, _ = writer(monkeypatch, base=52., yaw=-7.)
    rt.config.follow_forward_handoff_enable = True
    rt._visible_wheel_guard.resume_signs = (1, 1)
    if fault == "reverse_motion":
        rt._steering_feedback = feedback(clock[0], -20., -20.)
        rt._visible_wheel_guard.resume_signs = None
    elif fault == "reverse_provenance":
        rt._visible_wheel_guard.commanded_reverse = True
        rt._visible_wheel_guard.pending_full_reverse = True
        rt._visible_wheel_guard.resume_signs = None
        rt._steering_feedback = feedback(clock[0], -2., -2.)
    else:
        cross_yaw_deadline(monkeypatch, rt, owner, clock, fault=fault)
    rt._service_follow_wheels()
    assert all(pair == (0, 0) for pair in driver.pairs[1:])
    assert driver.pairs[-1] == (0, 0) or driver.stops
    assert not owner.motor_io_lock.locked()
