"""Publication transitions are not STOPs; real authority losses still are.

Runs the real periodic writer and driver-receipt ledger with fake I/O only.
"""
import pytest
from dataclasses import replace
import threading

from test_unified_forward_snapshot import writer, feedback, inject


@pytest.mark.parametrize("phase", ["zero_intent", "clear_intent", "pending_yaw"])
@pytest.mark.parametrize("stage", ["before", "guard", "terminal"])
def test_lateral_policy_generation_gap_never_clears_qualified_forward(monkeypatch, phase, stage):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch)
    old_policy = owner._lateral_turn_response_policy

    def publication():
        publish(88., 5. if phase == "pending_yaw" else 0., new_depth=False)
        owner._lateral_turn_response_policy = old_policy
        if phase == "clear_intent":
            owner._lateral_intent_store.clear()
            owner._has_fresh_lateral_yaw = lambda uid: False

    if stage == "before":
        publication()
    else:
        inject(monkeypatch, rt, stage, publication)
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (88, -88)
    assert len(driver.pairs) == 2 and not driver.stops
    assert all(left > 0 and right < 0 for left, right in driver.pairs)
    assert owner._depth30_linear_snapshot[3] == 10.  # No fabricated fresh depth.


@pytest.mark.parametrize("phase", ["zero_intent", "clear_intent"])
@pytest.mark.parametrize("fault", ["depth_expired", "feedback_expired", "identity", "explicit_stop", "hard_stop"])
def test_missing_turn_policy_does_not_bypass_actual_stop(monkeypatch, phase, fault):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch)
    old_policy = owner._lateral_turn_response_policy
    publish(88., 0., new_depth=False)
    owner._lateral_turn_response_policy = old_policy
    if phase == "clear_intent":
        owner._lateral_intent_store.clear()
        owner._has_fresh_lateral_yaw = lambda uid: False
    if fault == "depth_expired":
        clock[0] = 10.251
        rt._steering_feedback = feedback(clock[0], 24., 24.)
    elif fault == "feedback_expired":
        rt._steering_feedback = feedback(clock[0]-.151, 24., 24.)
    elif fault == "identity":
        owner._follow_controller.active_target_id = 2
    elif fault == "explicit_stop":
        owner._explicit_stop_requested = True
    else:
        rt.hard_stop_check = lambda _: True
    rt._service_follow_wheels()
    assert all(pair == (0, 0) for pair in driver.pairs[1:])
    assert driver.pairs[-1] == (0, 0) or driver.stops


def churn_at_three_terminal_checks(monkeypatch, rt, clock, publish, *, base=80., yaw=0.,
                                   after_last=None):
    # Change evidence AFTER the first current-budget evaluation, not before
    # the admission function starts. A normal fresh publication at entry is
    # now adopted and is no longer a reason to exercise retry exhaustion.
    original = rt._linear_packet_write_limit
    original_budget = rt.owner._depth_forward_continuation_limit
    calls = []
    pending = [False]

    def change(*args, **kwargs):
        calls.append(clock[0])
        pending[0] = len(calls) <= 3
        return original(*args, **kwargs)

    def budget(*args, **kwargs):
        result = original_budget(*args, **kwargs)
        if pending[0]:
            pending[0] = False
            clock[0] += .005
            publish(base, yaw)
            if len(calls) == 3 and after_last is not None:
                after_last()
        return result

    monkeypatch.setattr(rt, "_linear_packet_write_limit", change)
    monkeypatch.setattr(rt.owner, "_depth_forward_continuation_limit", budget)
    return calls


@pytest.mark.parametrize("yaw", [0., -4., 4.])
def test_bounded_churn_may_defer_only_an_independently_permitted_actual_receipt(monkeypatch, yaw):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=yaw)
    receipt = rt.backend.last_speed_receipt
    execution = rt._forward_execution_anchor
    history = tuple(rt._continuation_executed_speed_history)
    last_axes = rt._follow_wheel_clock.last_axes
    calls = churn_at_three_terminal_checks(monkeypatch, rt, clock, publish, base=80., yaw=yaw)
    rt._service_follow_wheels()
    assert len(calls) == 4  # Three proposals and one check of the already sent pair.
    assert len(driver.pairs) == 1 and not driver.stops
    assert rt.backend.last_speed_receipt is receipt
    assert rt._forward_execution_anchor is execution
    assert tuple(rt._continuation_executed_speed_history) == history
    assert rt._follow_wheel_clock.last_axes == last_axes
    assert receipt.completed_at == 10.
    assert owner._depth30_linear_timing.depth_expires_at == pytest.approx(
        owner._depth30_linear_snapshot[3]+.25)
    # The original service deadline is not moved. With no next publication,
    # the next call may immediately commit the latest still-qualified request.
    rt._service_follow_wheels()
    assert len(driver.pairs) == 2 and all(driver.pairs[-1])
    assert rt.backend.last_speed_receipt is not receipt


@pytest.mark.parametrize("fault", ["cap", "turn_reversed", "expired", "feedback", "uid", "stop", "hazard"])
def test_churn_cannot_hold_old_packet_against_new_limit_or_safety(monkeypatch, fault):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=4.)
    receipt = rt.backend.last_speed_receipt

    def final_change():
        if fault == "expired":
            clock[0] += .251
            rt._steering_feedback = feedback(clock[0], 24., 24.)
        elif fault == "feedback":
            rt._steering_feedback = feedback(clock[0]-.151, 24., 24.)
        elif fault == "uid":
            owner._follow_controller.active_target_id = 2
        elif fault == "stop":
            rt.backend.send_stop("concurrent-stop", mode="emergency", preserve_zero=True)
        elif fault == "hazard":
            rt.hard_stop_check = lambda _: True

    churn_at_three_terminal_checks(monkeypatch, rt, clock, publish,
        base=59. if fault == "cap" else 80.,
        yaw=-4. if fault == "turn_reversed" else 4., after_last=final_change)
    rt._service_follow_wheels()
    if fault in {"cap", "turn_reversed"}:
        # The prior curved packet may not be held, but the latest independent
        # Depth grant can still authorize a freshly checked straight packet.
        expected = 59 if fault == "cap" else 80
        assert driver.pairs == [(64, -56), (expected, -expected)]
        assert not driver.stops
        return
    assert all(pair == (0, 0) for pair in driver.pairs[1:])
    assert driver.pairs[-1] == (0, 0) or driver.stops
    assert rt.backend.last_speed_receipt is not receipt or driver.stops
    if fault == "stop":
        assert len(driver.pairs) == 1  # No speed-zero packet undoes STOP mode.


@pytest.mark.parametrize("fault", ["receipt_not_owned", "last_axes_other_uid"])
def test_deferred_receipt_must_belong_to_same_follow_writer_and_uid(monkeypatch, fault):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=0.)
    results = []
    original = rt._current_receipt_survives_publication

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        results.append(result)
        return result

    def detach_owner():
        if fault == "receipt_not_owned":
            # Identical fields do not confer ownership of a receipt object.
            rt._follow_wheel_last_receipt = replace(rt.backend.last_speed_receipt)
        else:
            rt._follow_wheel_clock.last_axes = (2, *rt._follow_wheel_clock.last_axes[1:])

    monkeypatch.setattr(rt, "_current_receipt_survives_publication", observed)
    churn_at_three_terminal_checks(monkeypatch, rt, clock, publish, after_last=detach_owner)
    rt._service_follow_wheels()
    assert results == [False]
    assert driver.pairs == [(60, -60), (0, 0)]


@pytest.mark.parametrize("stage", ["commit_negative", "cap_new_negative", "cap_new_same_values"])
def test_deferred_receipt_rechecks_encoder_after_commit_and_final_cap(monkeypatch, stage):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=0.)
    state = {"deferral": False, "changed": False}
    results = []
    original = rt._current_receipt_survives_publication
    begin = rt._begin_follow_commit
    cap = owner._depth_forward_continuation_limit

    def checked(*args, **kwargs):
        state["deferral"] = True
        try:
            result = original(*args, **kwargs)
            results.append(result)
            return result
        finally:
            state["deferral"] = False

    def update_feedback():
        state["changed"] = True
        if stage == "cap_new_same_values":
            rt._steering_feedback = replace(rt._steering_feedback)
        else:
            clock[0] += .001
            rt._steering_feedback = feedback(clock[0], -1., 24.)

    def after_commit():
        result = begin()
        if state["deferral"] and not state["changed"] and stage == "commit_negative":
            update_feedback()
        return result

    def after_cap(*args, **kwargs):
        result = cap(*args, **kwargs)
        if state["deferral"] and not state["changed"] and stage.startswith("cap_"):
            update_feedback()
        return result

    monkeypatch.setattr(rt, "_current_receipt_survives_publication", checked)
    monkeypatch.setattr(rt, "_begin_follow_commit", after_commit)
    monkeypatch.setattr(owner, "_depth_forward_continuation_limit", after_cap)
    churn_at_three_terminal_checks(monkeypatch, rt, clock, publish)
    rt._service_follow_wheels()
    harmless_refresh = stage == "cap_new_same_values"
    assert state["changed"] and results == [harmless_refresh]
    assert driver.pairs == ([(60, -60)] if harmless_refresh else [(60, -60), (0, 0)])


@pytest.mark.parametrize("override,allowed", [(None, True), (64., True), (63.999, False), (0., False)])
def test_deferred_receipt_cannot_ignore_outer_wheel_override(monkeypatch, override, allowed):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=4.)
    publish(80., 4.)
    receipt = rt.backend.last_speed_receipt
    arm_planning_owner(monkeypatch, rt)
    with rt._follow_planning_attempt():
        result = rt._current_receipt_survives_publication(1, max_target_override=override)
    assert result is allowed
    assert driver.pairs == [(64, -56)]  # This validator never sends a new plan.
    assert rt.backend.last_speed_receipt is receipt


def arm_planning_owner(monkeypatch, rt):
    """Enter the same capability scope normally supplied by the service loop."""
    monkeypatch.setattr(rt, "_follow_planning_offlock", True)
    monkeypatch.setattr(rt, "_follow_planning_thread_id", threading.get_ident())
    monkeypatch.setattr(rt, "_periodic_follow_writing", True)
    rt._follow_plan_token = (rt.backend.last_speed_receipt, rt.backend.stop_write_generation)
    rt._follow_commit_lock_times = []


@pytest.mark.parametrize("override,allowed", [(None, True), (0., False)])
def test_explicit_zero_override_cannot_send_new_positive_snapshot(monkeypatch, override, allowed):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=0.)
    publish(80., 0.)
    arm_planning_owner(monkeypatch, rt)
    with rt._follow_planning_attempt():
        assert rt._send_ordinary_snapshot_forward(1, "ZERO_OVERRIDE", max_target_override=override) is allowed
    assert driver.pairs == [(60, -60), (80, -80) if allowed else (0, 0)]
