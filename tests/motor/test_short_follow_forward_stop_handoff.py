"""CAP761/773: real forward STOP, legacy zero ACKs, then fresh paired control.

Only fake-driver writes are used. Physical response history cannot create a
plan or lengthen identity/Depth authority, and never explains a large reverse.
"""
from dataclasses import replace

import pytest

from car_control_modular.detector_identity_lease import publish_visual_identity_evidence
from test_short_follow_executor import publish, set_feedback, short_runtime
from test_short_follow_response_history import normalized_pair, publish_live


def cap773_handoff(monkeypatch, *, peak=64):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    owner._short_follow.config = replace(owner._short_follow.config, max_rpm=peak)
    clock[0] += .01
    assert publish(rt, clock, 758, distance=2., x=.15)
    rt._service_short_follow()
    assert normalized_pair(driver) == (peak - 18, peak)
    clock[0] += .06
    set_feedback(rt, clock, min(47, peak), min(62, peak))
    publish_visual_identity_evidence(owner, observation=False, lease=None)
    rt._service_short_follow()
    assert driver.stops == [1]
    stop_at = clock[0]
    writer = rt._short_follow_executor
    assert writer._park_tail_started_at == stop_at
    assert writer._park_tail_until == pytest.approx(stop_at + .75)
    assert writer._park_tail_peak_rpm == peak

    owner._short_follow.deactivate("identity_or_search_handoff", clock[0])
    rt._service_short_follow()
    assert not writer._owned
    # The legacy lane asked for a pivot, but its zero-cross guard wrote only
    # zeros. These ACKs must not create pivot credit or renew the STOP clock.
    for elapsed, pair in ((.05, (42, 51)), (.17, (22, 33)), (.30, (9, 21)), (.43, (-5, 6))):
        clock[0] = stop_at + elapsed
        set_feedback(rt, clock, *(min(v, peak) for v in pair))
        rt.backend.send_targets(0, 0, "LEGACY_CROSS_WAIT_ZERO", history_uid=1)
        assert not rt._service_short_follow()
        assert writer._park_tail_until == pytest.approx(stop_at + .75)
        assert not writer._turn_responses
    return rt, owner, driver, clock, stop_at


@pytest.mark.parametrize("pair", [(-5, 6), (-3, 1)])
@pytest.mark.parametrize("peak", [32, 64])
def test_cap773_fresh_plan_recovers_after_legacy_zeros_with_fixed_bounded_speed(monkeypatch, pair, peak):
    rt, owner, driver, clock, stop_at = cap773_handoff(monkeypatch, peak=peak)
    clock[0] = stop_at + .58
    set_feedback(rt, clock, *pair)
    owner._short_follow.activate(1, clock[0])
    plan = publish_live(rt, clock, 773, distance=2.17, x=.15)
    before = len(driver.pairs)
    rt._service_short_follow()
    assert len(driver.pairs) == before + 1
    assert 0 < min(normalized_pair(driver)) <= max(normalized_pair(driver)) <= min(40, peak)
    assert driver.stops == [1]
    writer = rt._short_follow_executor
    assert writer._entry_stop_at is None and writer._live_feedback_reason is None
    assert writer._park_tail_until == pytest.approx(stop_at + .75)
    assert plan.expires_at == pytest.approx(plan.depth_timestamp + .3)


def test_tail_history_without_new_plan_never_restarts_motion(monkeypatch):
    rt, owner, driver, clock, stop_at = cap773_handoff(monkeypatch)
    clock[0] = stop_at + .58
    set_feedback(rt, clock, -3, 1)
    owner._short_follow.activate(1, clock[0])
    before = len(driver.pairs)
    rt._service_short_follow()
    assert owner._short_follow.snapshot().plan is None
    assert len(driver.pairs) == before
    assert rt._short_follow_executor._park_tail_until == pytest.approx(stop_at + .75)


@pytest.mark.parametrize("renewal", ["zeros", "stop", "epoch_and_stop", "new_forward_then_stop"])
def test_no_zero_or_repeated_stop_can_renew_the_original_response_clock(monkeypatch, renewal):
    rt, owner, driver, clock, stop_at = cap773_handoff(monkeypatch)
    writer = rt._short_follow_executor
    clock[0] = stop_at + .58
    set_feedback(rt, clock, -3, 1)
    if renewal == "zeros":
        rt.backend.send_targets(0, 0, "LEGACY_ZERO_AGAIN", history_uid=1)
        rt._service_short_follow()
    else:
        if renewal in {"epoch_and_stop", "new_forward_then_stop"}:
            owner._short_follow.activate(1, clock[0])
        if renewal == "new_forward_then_stop":
            assert publish_live(rt, clock, 773)
            rt._service_short_follow()
        clock[0] += .04
        writer._stop_locked("awaiting_observation", owner._short_follow.snapshot().epoch,
                            clock[0], force=True)
    assert writer._park_tail_until == pytest.approx(stop_at + .75)
    clock[0] = stop_at + .751
    set_feedback(rt, clock, -3, 1)
    owner._short_follow.activate(1, clock[0])
    assert publish_live(rt, clock, 779)
    before = len(driver.pairs)
    rt._service_short_follow()
    assert len(driver.pairs) == before
    assert writer._live_feedback_reason == "feedback_reverse"


def test_positive_coast_without_new_forward_ack_does_not_retire_stop_history(monkeypatch):
    rt, owner, driver, clock, stop_at = cap773_handoff(monkeypatch)
    clock[0] = stop_at + .48
    set_feedback(rt, clock, 9, 21)
    owner._short_follow.activate(1, clock[0])  # No new plan: only observe the coast.
    rt._service_short_follow()
    writer = rt._short_follow_executor
    assert writer._park_tail_until == pytest.approx(stop_at + .75)
    clock[0] = stop_at + .58
    set_feedback(rt, clock, -3, 1)
    assert publish_live(rt, clock, 773)
    before = len(driver.pairs)
    rt._service_short_follow()
    assert len(driver.pairs) == before + 1
    assert writer._entry_stop_at is None


def test_expired_unobserved_tail_cannot_renew_without_confirmed_forward_recovery(monkeypatch):
    rt, owner, _, _, clock = short_runtime(monkeypatch)
    rt._service_short_follow()
    clock[0] += .05
    set_feedback(rt, clock, 5, 7)
    writer = rt._short_follow_executor
    writer._stop_locked("awaiting_observation", owner._short_follow.snapshot().epoch,
                        clock[0], force=True)
    stop_at = writer._park_tail_started_at
    clock[0] = stop_at + .65
    set_feedback(rt, clock, 0, 0, stamp=clock[0] - .001)
    assert publish_live(rt, clock, 773)
    rt._service_short_follow()
    assert rt.backend.last_speed_receipt.completed_at == clock[0]
    # The latest healthy feedback still predates the new forward ACK. Let the
    # response interval expire without ever observing a negative wheel, then
    # send another ordinary STOP. Neither fact proves completed recovery.
    clock[0] = stop_at + .76
    assert not writer._park_tail_reverse_seen
    writer._stop_locked("awaiting_observation", owner._short_follow.snapshot().epoch,
                        clock[0], force=True)
    assert writer._park_tail_started_at == stop_at
    assert writer._park_tail_until == pytest.approx(stop_at + .75)


def test_new_positive_feedback_after_real_forward_ack_retires_credit(monkeypatch):
    rt, owner, driver, clock, stop_at = cap773_handoff(monkeypatch)
    clock[0] = stop_at + .55
    set_feedback(rt, clock, -3, 1)
    owner._short_follow.activate(1, clock[0])
    assert publish_live(rt, clock, 773)
    rt._service_short_follow()
    ack_at = rt.backend.last_speed_receipt.completed_at
    clock[0] += .05
    set_feedback(rt, clock, 5, 7, stamp=ack_at - .001)
    rt._service_short_follow()
    writer = rt._short_follow_executor
    assert writer._park_tail_until == pytest.approx(stop_at + .75)
    clock[0] += .05
    set_feedback(rt, clock, 12, 16)
    rt._service_short_follow()
    assert writer._park_tail_until == float("-inf")
    assert writer._park_tail_peak_rpm == 0


@pytest.mark.parametrize("fault", ["both_reverse", "large_reverse", "motor_error", "untrusted",
                                  "hard_stop", "external_stop", "identity_rejected", "expired_plan"])
def test_cap773_response_credit_never_waives_actual_veto(monkeypatch, fault):
    rt, owner, driver, clock, stop_at = cap773_handoff(monkeypatch)
    clock[0] = stop_at + .58
    pair = {"both_reverse": (-3, -4), "large_reverse": (-6, 1)}.get(fault, (-3, 1))
    sample = set_feedback(rt, clock, *pair)
    owner._short_follow.activate(1, clock[0])
    plan = publish_live(rt, clock, 773)
    if fault == "motor_error": sample.left_error = 1
    elif fault == "untrusted": sample.trustworthy = False
    elif fault == "hard_stop": rt.hard_stop_check = lambda _action: True
    elif fault == "external_stop": rt.backend.send_stop("independent_safety_stop", mode="emergency")
    elif fault == "identity_rejected": publish_visual_identity_evidence(owner, observation=False, lease=None)
    elif fault == "expired_plan":
        clock[0] = plan.expires_at + .001
        set_feedback(rt, clock, 0, 0)
    before = len(driver.pairs)
    rt._service_short_follow()
    assert len(driver.pairs) == before
    assert driver.stops
    if fault in {"hard_stop", "external_stop", "motor_error", "untrusted", "both_reverse", "large_reverse"}:
        assert rt._short_follow_executor._park_tail_until == float("-inf")


def test_zero_only_origin_never_creates_forward_response_credit(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    owner._short_follow.deactivate("search", clock[0])
    rt.backend.send_targets(0, 0, "ZERO_ONLY", history_uid=1)
    rt._service_short_follow()
    writer = rt._short_follow_executor
    writer._stop_locked("awaiting_observation", owner._short_follow.snapshot().epoch,
                        clock[0], force=True)
    assert writer._park_tail_until == float("-inf")
    clock[0] += .05
    set_feedback(rt, clock, -3, 1)
    owner._short_follow.activate(1, clock[0])
    assert publish_live(rt, clock, 773)
    before = len(driver.pairs)
    rt._service_short_follow()
    assert len(driver.pairs) == before
    assert writer._live_feedback_reason == "feedback_reverse"


@pytest.mark.parametrize("ack_age", [.501, 2.])
def test_old_forward_ack_cannot_start_a_new_stop_response_window(monkeypatch, ack_age):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt._service_short_follow()
    assert driver.pairs
    writer = rt._short_follow_executor
    clock[0] += ack_age
    set_feedback(rt, clock, -3, 1)
    writer._stop_locked("awaiting_observation", owner._short_follow.snapshot().epoch,
                        clock[0], force=True)
    assert writer._park_tail_until == float("-inf")
    assert writer._park_tail_peak_rpm == 0


def test_forward_stop_tail_logs_original_deadline_and_ceiling_once(monkeypatch, caplog):
    rt, owner, _, clock, stop_at = cap773_handoff(monkeypatch)
    owner._short_follow.activate(1, clock[0])
    with caplog.at_level("INFO"):
        for cap, elapsed in ((773, .55), (775, .60), (777, .65)):
            clock[0] = stop_at + elapsed
            set_feedback(rt, clock, -3, 1)
            assert publish_live(rt, clock, cap)
            rt._service_short_follow()
    records = [r.getMessage() for r in caplog.records
               if r.getMessage().startswith("short_follow_forward_stop_tail ")]
    assert len(records) == 1
    assert f"original_stop_ts={stop_at:.9f}" in records[0]
    assert f"deadline={stop_at + .75:.9f}" in records[0]
    assert "prior_peak_rpm=64" in records[0] and "capped_rpm=40" in records[0]
    assert "deadline_renewed=False motion_authorized=False" in records[0]
