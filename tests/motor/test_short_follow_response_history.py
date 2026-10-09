"""Observed wheel response is bounded history, never renewed motion authority.

Synthetic CAP354/356 and CAP483 regressions through the real controller,
executor and motor backend. FakeDriver records writes without opening hardware.
"""
import pytest

from car_control_modular.detector_identity_lease import (
    ValidatedVisualObservation, publish_visual_identity_evidence,
)
from test_short_follow_executor import publish, set_feedback, short_runtime


def normalized_pair(driver):
    left, right = driver.pairs[-1]
    return left, -right


def pivot_x(owner, rpm, direction):
    cfg = owner._short_follow.config
    offset = cfg.center_deadband_ratio + (
        cfg.yaw_full_error_ratio - cfg.center_deadband_ratio
    ) * rpm / cfg.pivot_limit_rpm
    return .5 + direction * offset


def shrinking_pivot(monkeypatch, direction):
    rt, owner, driver, symbols, clock = short_runtime(monkeypatch)
    start = clock[0]
    for cap, elapsed, rpm, feedback in (
        (346, .01, 8, (0, 0)),
        (348, .07, 7, (8, -8)),
        (351, .13, 5, (5, -6)),
        (354, .19, 2, (5, -6)),
    ):
        clock[0] = start + elapsed
        set_feedback(rt, clock, *(direction * v for v in feedback))
        assert publish(rt, clock, cap, distance=1.35, x=pivot_x(owner, rpm, direction))
        assert rt._service_short_follow()
        assert normalized_pair(driver) == (direction * rpm, -direction * rpm)
        assert not driver.stops
    return rt, owner, driver, symbols, clock


def awaiting_after_search_zero(monkeypatch, direction):
    rt, owner, driver, symbols, clock = short_runtime(monkeypatch)
    controller = owner._short_follow
    controller.deactivate("search", clock[0])
    clock[0] += .01
    rt.backend.send_targets(7 * direction, 7 * direction, "TURN", history_uid=1)
    assert not rt._service_short_follow()  # Observe the actual search ACK.
    clock[0] += .05
    set_feedback(rt, clock, 7 * direction, -7 * direction)
    rt.backend.send_targets(0, 0, "TURN_ZERO", history_uid=1)
    zero_ack_at = rt.backend.last_speed_receipt.completed_at
    assert not rt._service_short_follow()
    clock[0] += .06
    assert controller.activate(1, clock[0])
    set_feedback(rt, clock, 4 * direction, -4 * direction)
    assert rt._service_short_follow()
    assert controller.snapshot().plan is None
    assert driver.stops == [1]  # No observation still requires a real STOP.
    assert len(driver.pairs) == 2
    return rt, owner, driver, symbols, clock, zero_ack_at


@pytest.mark.parametrize("direction", [1, -1], ids=["right", "left"])
@pytest.mark.parametrize("sample_offset", [-.01, .01], ids=["before_new_ack", "after_new_ack"])
def test_cap356_same_direction_pivot_reduction_is_not_reverse_fault(
        monkeypatch, direction, sample_offset):
    rt, owner, driver, _, clock = shrinking_pivot(monkeypatch, direction)
    plan = owner._short_follow.snapshot().plan
    original_expiry = plan.expires_at
    reduced_ack = rt.backend.last_speed_receipt.completed_at
    clock[0] += .06
    set_feedback(rt, clock, 5 * direction, -6 * direction,
                 stamp=reduced_ack + sample_offset)
    rt._service_short_follow()
    assert normalized_pair(driver) == (2 * direction, -2 * direction)
    assert len(driver.pairs) == 5
    assert not driver.stops
    assert rt._short_follow_executor._entry_stop_at is None
    assert owner._short_follow.snapshot().plan is plan
    assert plan.expires_at == original_expiry  # Response credit never renews depth.


@pytest.mark.parametrize("direction", [1, -1], ids=["right", "left"])
def test_repeated_small_pivot_and_new_observations_cannot_renew_larger_response(monkeypatch, direction):
    rt, owner, driver, _, clock = shrinking_pivot(monkeypatch, direction)
    reduced_at = clock[0]
    for cap, elapsed in ((356, .06), (359, .12), (362, .18)):
        clock[0] = reduced_at + elapsed
        set_feedback(rt, clock, 5 * direction, -6 * direction)
        assert publish(rt, clock, cap, distance=1.35, x=pivot_x(owner, 2, direction))
        rt._service_short_follow()
        assert not driver.stops
        assert normalized_pair(driver) == (2 * direction, -2 * direction)
    written_before = len(driver.pairs)
    clock[0] = reduced_at + .36
    set_feedback(rt, clock, 5 * direction, -6 * direction)
    assert publish(rt, clock, 366, distance=1.35, x=pivot_x(owner, 2, direction))
    rt._service_short_follow()
    assert len(driver.pairs) == written_before
    assert driver.stops
    assert rt._short_follow_executor._live_feedback_reason == "feedback_reverse"


@pytest.mark.parametrize("direction", [1, -1], ids=["right", "left"])
def test_encoder_reaching_small_pivot_retires_larger_response_credit(monkeypatch, direction):
    rt, owner, driver, _, clock = shrinking_pivot(monkeypatch, direction)
    clock[0] += .06
    set_feedback(rt, clock, 2 * direction, -2 * direction)
    assert publish(rt, clock, 356, distance=1.35, x=pivot_x(owner, 2, direction))
    rt._service_short_follow()
    assert not driver.stops
    written_before = len(driver.pairs)
    clock[0] += .06
    set_feedback(rt, clock, 5 * direction, -6 * direction)
    assert publish(rt, clock, 359, distance=1.35, x=pivot_x(owner, 2, direction))
    rt._service_short_follow()
    assert len(driver.pairs) == written_before
    assert driver.stops


@pytest.mark.parametrize("direction", [1, -1], ids=["right", "left"])
def test_cap483_awaiting_stop_preserves_diagnostic_tail_until_new_plan(monkeypatch, direction):
    rt, owner, driver, _, clock, _ = awaiting_after_search_zero(monkeypatch, direction)
    clock[0] += .02
    set_feedback(rt, clock, 4 * direction, -4 * direction)
    rt._service_short_follow()
    assert driver.stops == [1]
    assert len(driver.pairs) == 2
    assert owner._short_follow.snapshot().plan is None
    assert rt._short_follow_executor._live_feedback_reason is None
    assert rt._short_follow_executor._entry_stop_at is None

    clock[0] += .04
    set_feedback(rt, clock, 4 * direction, -4 * direction)
    plan = publish(rt, clock, 486, distance=2.45, x=.5 + .2 * direction)
    assert plan is not None
    assert rt._service_short_follow()
    left, right = normalized_pair(driver)
    assert min(left, right) > 0
    assert direction * (left - right) > 0
    assert len(driver.pairs) == 3
    assert driver.stops == [1]
    assert rt._short_follow_executor._entry_stop_at is None
    assert plan.expires_at == pytest.approx(plan.depth_timestamp + .3)


@pytest.mark.parametrize("direction", [1, -1], ids=["right", "left"])
def test_repeated_waiting_stops_cannot_extend_original_search_response_deadline(monkeypatch, direction):
    rt, owner, driver, _, clock, zero_ack_at = awaiting_after_search_zero(monkeypatch, direction)
    executor = rt._short_follow_executor
    for elapsed in (.15, .25):
        clock[0] = zero_ack_at + elapsed
        set_feedback(rt, clock, 4 * direction, -4 * direction)
        # Exercise actual additional STOP ACKs, not only throttled repeats.
        executor._stop_locked("awaiting_observation", owner._short_follow.snapshot().epoch,
                              clock[0], force=True)
        rt._service_short_follow()
        assert executor._live_feedback_reason is None
        assert executor._entry_stop_at is None
        assert owner._short_follow.snapshot().plan is None
        assert len(driver.pairs) == 2
    written_before = len(driver.pairs)
    clock[0] = zero_ack_at + .36
    set_feedback(rt, clock, 4 * direction, -4 * direction)
    assert publish(rt, clock, 488, distance=2.45, x=.5 + .2 * direction)
    rt._service_short_follow()
    assert len(driver.pairs) == written_before
    assert executor._live_feedback_reason == "feedback_reverse"


@pytest.mark.parametrize("direction", [1, -1], ids=["right", "left"])
def test_observation_expiry_still_stops_despite_known_pivot_response(monkeypatch, direction):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    clock[0] += .01
    plan = publish(rt, clock, 346, distance=1.35, x=pivot_x(owner, 8, direction))
    rt._service_short_follow()
    assert normalized_pair(driver) == (8 * direction, -8 * direction)
    written_before = len(driver.pairs)
    clock[0] = plan.expires_at + .001
    set_feedback(rt, clock, 5 * direction, -6 * direction)
    rt._service_short_follow()
    assert driver.stops == [1]
    assert len(driver.pairs) == written_before
    clock[0] += .02
    set_feedback(rt, clock, 4 * direction, -4 * direction)
    rt._service_short_follow()
    assert len(driver.pairs) == written_before  # History cannot resurrect the old plan.
    assert driver.stops == [1]
    assert rt._short_follow_executor._live_feedback_reason is None
    assert rt._short_follow_executor._entry_stop_at is None


@pytest.mark.parametrize("adverse", ["unknown", "both_reverse", "large_reverse", "wrong_wheel", "fault"])
def test_response_history_does_not_waive_unexplained_or_faulted_reverse(monkeypatch, adverse):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    if adverse != "unknown":
        rt.backend.send_targets(8, 8, "TURN", history_uid=1)
    sample_pair = {
        "unknown": (5, -6), "both_reverse": (-5, -6),
        "large_reverse": (5, -39), "wrong_wheel": (-6, 5), "fault": (5, -6),
    }[adverse]
    clock[0] += .06
    sample = set_feedback(rt, clock, *sample_pair)
    if adverse == "fault":
        sample.right_error = 1
    assert publish(rt, clock, 354, distance=1.35, x=pivot_x(owner, 2, 1))
    written_before = len(driver.pairs)
    rt._service_short_follow()
    assert len(driver.pairs) == written_before
    assert driver.stops


def test_candidate_plan_without_ack_is_not_historical_wheel_execution(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    clock[0] += .06
    assert publish(rt, clock, 346, distance=1.35, x=pivot_x(owner, 8, 1))
    # The 8/-8 plan was never serviced or acknowledged; it cannot explain this.
    clock[0] += .06
    set_feedback(rt, clock, 5, -6)
    assert publish(rt, clock, 354, distance=1.35, x=pivot_x(owner, 2, 1))
    rt._service_short_follow()
    assert not driver.pairs
    assert driver.stops


def publish_live(rt, clock, capture, *, distance=2., x=.5):
    """Publish actual joint identity evidence after an earlier rejection."""
    plan = publish(rt, clock, capture, distance=distance, x=x)
    now = clock[0]
    publish_visual_identity_evidence(rt.owner, observation=ValidatedVisualObservation(
        1, 1, capture, now, now, now + .5, "full"), lease=None)
    return plan


def forward_ordinary_stop(monkeypatch, reason="identity"):
    rt, owner, driver, symbols, clock = short_runtime(monkeypatch)
    rt._service_short_follow()
    assert min(normalized_pair(driver)) > 0
    assert not driver.stops
    clock[0] += .06
    set_feedback(rt, clock, 30, 40)
    if reason == "identity":
        publish_visual_identity_evidence(owner, observation=False, lease=None)
    elif reason == "measurement":
        owner._short_follow.revoke("awaiting_observation", clock[0])
    elif reason == "ownership_exit":
        owner._short_follow.deactivate("identity_or_search_handoff", clock[0])
    else:
        raise AssertionError(reason)
    rt._service_short_follow()
    assert driver.stops == [1]
    assert rt.backend.last_speed_receipt is None
    writer = rt._short_follow_executor
    assert writer._park_tail_until == pytest.approx(clock[0] + .35)
    return rt, owner, driver, symbols, clock


@pytest.mark.parametrize("reason", ["identity", "measurement", "ownership_exit"])
@pytest.mark.parametrize("feedback", [(-3, 10), (10, -3)], ids=["left_tail", "right_tail"])
def test_cap563_known_forward_stop_tail_allows_fresh_plan_without_new_stop(
        monkeypatch, reason, feedback):
    rt, owner, driver, _, clock = forward_ordinary_stop(monkeypatch, reason)
    deadline = rt._short_follow_executor._park_tail_until
    clock[0] += .06
    set_feedback(rt, clock, *feedback)
    assert owner._short_follow.activate(1, clock[0])
    plan = publish_live(rt, clock, 572)
    assert plan is not None
    rt._service_short_follow()
    assert len(driver.pairs) == 2
    assert min(normalized_pair(driver)) > 0
    assert driver.stops == [1]
    assert rt._short_follow_executor._entry_stop_at is None
    assert rt._short_follow_executor._park_tail_until == deadline
    assert plan.expires_at == pytest.approx(plan.depth_timestamp + .3)


@pytest.mark.parametrize("reason", ["identity", "measurement"])
def test_known_forward_stop_tail_alone_cannot_restore_motion(monkeypatch, reason):
    rt, owner, driver, _, clock = forward_ordinary_stop(monkeypatch, reason)
    for elapsed in (.04, .08, .12):
        clock[0] = 10.06 + elapsed
        set_feedback(rt, clock, -3, 10)
        rt._service_short_follow()
        assert len(driver.pairs) == 1
        assert owner._short_follow.snapshot().plan is None
        assert rt._short_follow_executor._live_feedback_reason is None
        assert rt._short_follow_executor._entry_stop_at is None
    assert driver.stops == [1]


@pytest.mark.parametrize("fresh_forward_ack", [False, True], ids=["stop_refresh", "new_forward_then_stop"])
def test_repeated_ordinary_stops_do_not_restart_unresolved_forward_tail(
        monkeypatch, fresh_forward_ack):
    rt, owner, driver, _, clock = forward_ordinary_stop(monkeypatch, "measurement")
    writer = rt._short_follow_executor
    original_deadline = writer._park_tail_until
    clock[0] += .06
    set_feedback(rt, clock, -3, 10)
    if fresh_forward_ack:
        assert publish_live(rt, clock, 570)
        rt._service_short_follow()
        assert len(driver.pairs) == 2
    else:
        rt._service_short_follow()
        assert len(driver.pairs) == 1
    clock[0] += .10
    set_feedback(rt, clock, -3, 10)
    owner._short_follow.revoke("awaiting_observation", clock[0])
    # Exercise a real repeated STOP ACK, not just the stop-packet throttle.
    writer._stop_locked("awaiting_observation", owner._short_follow.snapshot().epoch,
                        clock[0], force=True)
    assert writer._park_tail_until == original_deadline
    before = len(driver.pairs)
    clock[0] = original_deadline + .001
    set_feedback(rt, clock, -3, 10)
    assert publish_live(rt, clock, 574)
    rt._service_short_follow()
    assert len(driver.pairs) == before
    assert writer._live_feedback_reason == "feedback_reverse"


@pytest.mark.parametrize("feedback_state", ["fresh", "aged", "missing"])
def test_forward_tail_timeout_cannot_restart_review_or_be_erased_by_stale_feedback(
        monkeypatch, feedback_state):
    rt, owner, driver, _, clock = forward_ordinary_stop(monkeypatch, "measurement")
    writer = rt._short_follow_executor
    deadline = writer._park_tail_until
    clock[0] += .06
    set_feedback(rt, clock, -3, 10)
    assert publish_live(rt, clock, 570)
    rt._service_short_follow()
    assert driver.stops == [1]
    assert len(driver.pairs) == 2
    before = len(driver.pairs)
    clock[0] = deadline + .001
    if feedback_state == "fresh":
        set_feedback(rt, clock, -3, 10)
    elif feedback_state == "missing":
        rt.get_steering_feedback = lambda: None
    assert publish_live(rt, clock, 574)
    rt._service_short_follow()
    assert len(driver.pairs) == before
    assert len(driver.stops) == 2
    assert writer._live_feedback_reason == "feedback_reverse"


def test_new_positive_feedback_retires_forward_stop_tail_credit(monkeypatch):
    rt, owner, driver, _, clock = forward_ordinary_stop(monkeypatch, "measurement")
    writer = rt._short_follow_executor
    clock[0] += .06
    set_feedback(rt, clock, -3, 10)
    assert publish_live(rt, clock, 570)
    rt._service_short_follow()
    clock[0] += .06
    set_feedback(rt, clock, 10, 20)
    assert publish_live(rt, clock, 572)
    rt._service_short_follow()
    assert writer._park_tail_until == float("-inf")

    clock[0] += .06
    set_feedback(rt, clock, -3, 10)
    assert publish_live(rt, clock, 574)
    rt._service_short_follow()
    # A NEW anomaly uses existing bounded forward review, not the old waiver.
    assert writer._reverse_pending is not None
    assert max(normalized_pair(driver)) <= 40
    clock[0] += .14
    set_feedback(rt, clock, -3, 10)
    assert publish_live(rt, clock, 576)
    before = len(driver.pairs)
    rt._service_short_follow()
    assert len(driver.pairs) == before
    assert writer._live_feedback_reason == "feedback_reverse"


@pytest.mark.parametrize("adverse", ["both_reverse", "large_reverse", "overspeed", "fault", "untrusted"])
def test_forward_stop_tail_never_waives_large_reverse_or_faults(monkeypatch, adverse):
    rt, owner, driver, _, clock = forward_ordinary_stop(monkeypatch, "measurement")
    clock[0] += .06
    pair = {"both_reverse": (-3, -4), "large_reverse": (-6, 10),
            "overspeed": (-3, 230)}.get(adverse, (-3, 10))
    sample = set_feedback(rt, clock, *pair)
    if adverse == "fault":
        sample.left_error = 1
    elif adverse == "untrusted":
        sample.trustworthy = False
    assert publish_live(rt, clock, 572)
    rt._service_short_follow()
    assert len(driver.pairs) == 1
    assert len(driver.stops) == 2


@pytest.mark.parametrize("hard_stop", ["runtime_safety", "external_stop"])
def test_hard_stop_clears_forward_response_history(monkeypatch, hard_stop):
    rt, owner, driver, _, clock = forward_ordinary_stop(monkeypatch, "measurement")
    clock[0] += .06
    set_feedback(rt, clock, -3, 10)
    assert publish_live(rt, clock, 570)
    rt._service_short_follow()
    assert len(driver.pairs) == 2
    clock[0] += .02
    if hard_stop == "runtime_safety":
        rt.hard_stop_check = lambda _action: True
    else:
        rt.backend.send_stop("independent_safety_stop", mode="emergency")
    rt._service_short_follow()
    assert rt._short_follow_executor._park_tail_until == float("-inf")
    before = len(driver.pairs)
    rt.hard_stop_check = lambda _action: False
    clock[0] += .06
    set_feedback(rt, clock, -3, 10)
    assert publish_live(rt, clock, 572)
    rt._service_short_follow()
    assert len(driver.pairs) == before
    assert rt._short_follow_executor._live_feedback_reason == "feedback_reverse"
