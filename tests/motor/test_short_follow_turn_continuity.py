"""CAP-inspired signed-turn acceptance through the real paired motor writer.

These are synthetic observation/encoder sequences, not a replay predicting
physical stopping distance. The backend uses FakeDriver and opens no hardware.
"""
from dataclasses import replace

import pytest

from car_control_modular.detector_identity_lease import publish_visual_identity_evidence
from test_short_follow_executor import publish, set_feedback, short_runtime


def normalized_pair(driver):
    """FakeDriver stores wire signs: the right wheel is inverted."""
    left, right = driver.pairs[-1]
    return left, -right


def mirrored_x(x, direction):
    return x if direction == 1 else 1. - x


def tick_observation(rt, clock, cap, distance, x, feedback):
    clock[0] += .06
    set_feedback(rt, clock, *feedback)
    plan = publish(rt, clock, cap, distance=distance, x=x)
    assert plan is not None
    assert rt._service_short_follow()
    return plan


def assert_pivot(owner, driver, direction):
    left, right = normalized_pair(driver)
    assert left == -right
    assert 0 < direction * left <= 8
    assert owner._short_follow_last_applied_plan.base_rpm == 0
    assert not owner.is_forwarding


def assert_arc(driver, direction):
    left, right = normalized_pair(driver)
    assert min(left, right) >= 0
    assert left + right > 0
    assert direction * (left - right) > 0


@pytest.mark.parametrize("direction", [1, -1], ids=["right", "left"])
def test_cap404_to544_distance_stop_preserves_tracking_turn(monkeypatch, direction):
    rt, owner, driver, symbols, clock = short_runtime(monkeypatch)
    initial_epoch = owner._short_follow.snapshot().epoch

    tick_observation(rt, clock, 404, 1.40, mirrored_x(.65, direction), (0, 0))
    assert_pivot(owner, driver, direction)
    assert owner.current_command == (symbols.rotate_right if direction == 1 else symbols.rotate_left)
    prior_pivot = normalized_pair(driver)

    # New distance permits forward travel while the acknowledged pivot still
    # appears in the encoder. It is a legal handoff, not unexplained reverse.
    tick_observation(rt, clock, 414, 1.626, mirrored_x(.715, direction), prior_pivot)
    assert_arc(driver, direction)
    assert owner.is_forwarding

    positive_feedback = (20, 10) if direction == 1 else (10, 20)
    tick_observation(rt, clock, 430, 1.74, mirrored_x(.802, direction), positive_feedback)
    assert_arc(driver, direction)

    # This sequence supplies low-speed feedback for the near-distance case;
    # it does not assume a fast moving wheel can physically reverse instantly.
    tick_observation(rt, clock, 512, 1.429, mirrored_x(.706, direction), (1, 1))
    assert_pivot(owner, driver, direction)
    tick_observation(rt, clock, 544, 1.26, mirrored_x(.65, direction), normalized_pair(driver))
    assert_pivot(owner, driver, direction)
    assert len(driver.pairs) == 5
    assert all(pair != (0, 0) for pair in driver.pairs)
    assert not driver.stops
    assert owner._short_follow.snapshot().epoch == initial_epoch

    # Centered, close target now legitimately stops BOTH wheels immediately.
    tick_observation(rt, clock, 546, 1.26, .5, (0, 0))
    assert len(driver.pairs) == 5
    assert driver.stops == [1]
    assert not owner.is_forwarding


@pytest.mark.parametrize("direction", [1, -1], ids=["right", "left"])
def test_acknowledged_near_pivot_feedback_is_not_reverse_fault(monkeypatch, direction):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    plan = tick_observation(rt, clock, 404, 1.40, mirrored_x(.8, direction), (0, 0))
    expiry = plan.expires_at
    for _ in range(3):
        clock[0] += .06
        set_feedback(rt, clock, *normalized_pair(driver))
        rt._service_short_follow()
        assert_pivot(owner, driver, direction)
        assert owner._short_follow.snapshot().plan is plan
        assert plan.expires_at == expiry
    assert not driver.stops

    # Repeated acknowledged pivot writes cannot renew depth or identity time.
    clock[0] = expiry + .001
    set_feedback(rt, clock, *normalized_pair(driver))
    rt._service_short_follow()
    assert driver.stops == [1]


@pytest.mark.parametrize("direction", [1, -1], ids=["right", "left"])
@pytest.mark.parametrize("label", ["TURN", "SEARCH"])
def test_acknowledged_search_pivot_enters_forward_arc_without_stop(monkeypatch, direction, label):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    # Actual dual-wheel backend write/ACK, not a forged historical object.
    rt.backend.send_targets(7 * direction, 7 * direction, label, history_uid=1)
    tick_observation(rt, clock, 448, 1.684, mirrored_x(.856, direction),
                     (7 * direction, -7 * direction))
    assert_arc(driver, direction)
    assert len(driver.pairs) == 2
    assert not driver.stops


@pytest.mark.parametrize("direction", [1, -1], ids=["right", "left"])
def test_pivot_to_arc_reverse_tail_has_fixed_finite_window(monkeypatch, direction):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt.backend.send_targets(7 * direction, 7 * direction, "TURN", history_uid=1)
    tail = (7 * direction, -7 * direction)
    tick_observation(rt, clock, 448, 1.684, mirrored_x(.856, direction), tail)
    started = clock[0]
    assert not driver.stops
    for cap, offset in ((450, .10), (453, .20), (456, .30)):
        clock[0] = started + offset
        set_feedback(rt, clock, *tail)
        assert publish(rt, clock, cap, distance=1.8, x=mirrored_x(.8, direction))
        rt._service_short_follow()
        assert not driver.stops
        assert_arc(driver, direction)

    # Repeated new distance/identity results cannot make reverse tail eternal.
    clock[0] = started + .36
    set_feedback(rt, clock, *tail)
    assert publish(rt, clock, 459, distance=1.8, x=mirrored_x(.8, direction))
    rt._service_short_follow()
    assert driver.stops


@pytest.mark.parametrize("direction", [1, -1], ids=["right", "left"])
def test_positive_feedback_ends_search_tail_credit(monkeypatch, direction):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt.backend.send_targets(7 * direction, 7 * direction, "TURN", history_uid=1)
    tick_observation(rt, clock, 448, 1.684, mirrored_x(.856, direction),
                     (7 * direction, -7 * direction))
    tick_observation(rt, clock, 450, 1.75, mirrored_x(.8, direction), (10, 10))
    assert not driver.stops
    written_before = len(driver.pairs)

    # New negative motion after both wheels became positive is not the old
    # pivot tail, even though its original 350 ms window has not elapsed.
    tick_observation(rt, clock, 453, 1.8, mirrored_x(.8, direction),
                     (7 * direction, -7 * direction))
    assert len(driver.pairs) == written_before
    assert driver.stops


@pytest.mark.parametrize("adverse", ["no_receipt", "opposite_feedback", "both_reverse", "oversize", "stale_receipt"])
def test_unexplained_reverse_cannot_borrow_new_pivot_plan(monkeypatch, adverse):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    feedback = (7, -7)
    if adverse != "no_receipt":
        rt.backend.send_targets(7, 7, "TURN", history_uid=1)
    if adverse == "opposite_feedback":
        feedback = (-7, 7)
    elif adverse == "both_reverse":
        feedback = (-7, -7)
    elif adverse == "oversize":
        feedback = (7, -20)
    elif adverse == "stale_receipt":
        clock[0] += .51
    written_before = len(driver.pairs)
    # In the opposite case the NEW plan agrees with the anomalous feedback,
    # but the last real ACK does not. A proposed command is not past evidence.
    x = .2 if adverse == "opposite_feedback" else .8
    tick_observation(rt, clock, 404, 1.40, x, feedback)
    assert len(driver.pairs) == written_before
    assert driver.stops


@pytest.mark.parametrize("adverse", ["hard_stop", "identity_rejected", "unsafe_distance"])
def test_pivot_keeps_independent_emergency_identity_and_distance_guards(monkeypatch, adverse):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    tick_observation(rt, clock, 404, 1.40, .8, (0, 0))
    assert_pivot(owner, driver, 1)
    written_before = len(driver.pairs)
    clock[0] += .06
    set_feedback(rt, clock, *normalized_pair(driver))
    if adverse == "hard_stop":
        rt.hard_stop_check = lambda _action: True
    elif adverse == "identity_rejected":
        publish_visual_identity_evidence(owner, observation=False, lease=None)
    else:
        assert publish(rt, clock, 405, distance=1.10, x=.8)
    rt._service_short_follow()
    assert len(driver.pairs) == written_before
    assert driver.stops


@pytest.mark.parametrize("direction", [1, -1], ids=["right", "left"])
def test_far_turn_uses_full_sixteen_rpm_difference(monkeypatch, direction):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    tick_observation(rt, clock, 430, 2., mirrored_x(.8, direction), (0, 0))
    left, right = normalized_pair(driver)
    assert direction * (left - right) == 16
    assert not driver.stops


def test_pivot_scaling_respects_small_hardware_limit(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt.backend.config = replace(rt.backend.config, max_target=4)
    tick_observation(rt, clock, 404, 1.40, .9, (0, 0))
    assert normalized_pair(driver) == (4, -4)
    assert not driver.stops


@pytest.mark.parametrize("direction", [1, -1], ids=["right_to_left", "left_to_right"])
def test_acked_pivot_direction_change_blends_without_zero(monkeypatch, direction):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    tick_observation(rt, clock, 404, 1.40, mirrored_x(.8, direction), (0, 0))
    old_feedback = normalized_pair(driver)
    tick_observation(rt, clock, 406, 1.40, mirrored_x(.8, -direction), old_feedback)
    assert_pivot(owner, driver, -direction)
    tick_observation(rt, clock, 408, 1.40, mirrored_x(.8, -direction), old_feedback)
    assert_pivot(owner, driver, -direction)
    tick_observation(rt, clock, 410, 1.40, mirrored_x(.8, -direction), normalized_pair(driver))
    assert_pivot(owner, driver, -direction)
    assert len(driver.pairs) == 4
    assert not driver.stops


@pytest.mark.parametrize("fault", ["external_stop", "write_fault", "encoder_fault", "untrusted"])
def test_search_tail_cannot_mask_real_fault_or_stop_generation(monkeypatch, fault):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt.backend.send_targets(7, 7, "TURN", history_uid=1)
    tick_observation(rt, clock, 448, 1.684, .856, (7, -7))
    assert not driver.stops
    written_before = len(driver.pairs)
    clock[0] += .06
    sample = set_feedback(rt, clock, 7, -7)
    assert publish(rt, clock, 450, distance=1.8, x=.8)
    if fault == "external_stop":
        rt.backend.send_stop("independent_safety_stop", mode="emergency")
    elif fault == "write_fault":
        rt.backend.motion_write_fault = "incomplete_dual_wheel_write"
    elif fault == "encoder_fault":
        sample.right_error = 1
    else:
        sample.trustworthy = False
    rt._service_short_follow()
    assert len(driver.pairs) == written_before
    assert driver.stops
    assert owner._short_follow.snapshot().plan is None


@pytest.mark.parametrize("cleared_first", [False, True], ids=["both_reverse", "reversed_after_recovery"])
def test_distance_stop_tail_cannot_authorize_new_unexplained_reverse(monkeypatch, cleared_first):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt._service_short_follow()
    tick_observation(rt, clock, 512, 1.429, .5, (5, 5))
    assert driver.stops == [1]
    if cleared_first:
        tick_observation(rt, clock, 514, 1.8, .7, (5, 5))
        assert_arc(driver, 1)
    written_before = len(driver.pairs)
    tick_observation(rt, clock, 516, 1.8, .7, (-4, -4))
    assert len(driver.pairs) == written_before
    assert len(driver.stops) == 2


def test_forward_recovery_clears_park_tail_before_small_single_wheel_review(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt._service_short_follow()
    tick_observation(rt, clock, 512, 1.429, .5, (5, 5))
    tick_observation(rt, clock, 514, 1.8, .7, (5, 5))
    tick_observation(rt, clock, 516, 1.8, .7, (4, -4))
    # Existing small single-wheel anomaly review may continue at <=40 RPM,
    # but an old distance STOP must not silently waive that review entirely.
    assert rt._short_follow_executor._reverse_pending is not None
    assert max(map(abs, normalized_pair(driver))) <= 40
    assert len(driver.stops) == 1


@pytest.mark.parametrize("missing", [False, True], ids=["stale_feedback", "missing_feedback"])
def test_observed_search_reverse_tail_cannot_clear_by_feedback_aging(monkeypatch, missing):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt.backend.send_targets(7, 7, "TURN", history_uid=1)
    tick_observation(rt, clock, 448, 1.684, .856, (7, -7))
    started = clock[0]
    clock[0] = started + .10
    set_feedback(rt, clock, 7, -7)
    assert publish(rt, clock, 450, distance=1.8, x=.8)
    rt._service_short_follow()
    assert not driver.stops
    written_before = len(driver.pairs)
    clock[0] = started + .36
    if missing:
        rt.get_steering_feedback = lambda: None
    assert publish(rt, clock, 459, distance=1.8, x=.8)
    rt._service_short_follow()
    assert len(driver.pairs) == written_before
    assert driver.stops


@pytest.mark.parametrize("first_feedback", [(7, -7), (5, 5)], ids=["tail_resolved", "no_reverse_tail_seen"])
def test_settled_search_turn_does_not_invent_a_later_reverse_fault(monkeypatch, first_feedback):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt.backend.send_targets(7, 7, "TURN", history_uid=1)
    tick_observation(rt, clock, 448, 1.684, .856, first_feedback)
    tick_observation(rt, clock, 450, 1.8, .8, (5, 5))
    clock[0] += .36
    assert publish(rt, clock, 459, distance=1.8, x=.8)
    rt._service_short_follow()
    assert not driver.stops
    assert 0 < max(map(abs, normalized_pair(driver))) <= 40


@pytest.mark.parametrize("direction", [1, -1], ids=["right", "left"])
@pytest.mark.parametrize("real_stop", [False, True], ids=["speed_zero", "stop_generation"])
def test_cap480_inactive_search_zero_handoff_preserves_only_real_ack_lineage(
        monkeypatch, direction, real_stop):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    controller = owner._short_follow
    controller.deactivate("search", clock[0])
    clock[0] += .01
    rt.backend.send_targets(7 * direction, 7 * direction, "TURN", history_uid=1)
    assert not rt._service_short_follow()  # Observe ACK while not the writer.
    assert not driver.stops

    clock[0] += .05
    rt.backend.send_targets(0, 0, "TURN_ZERO", history_uid=1)
    assert not rt._service_short_follow()  # This is a speed ACK, not STOP.
    if real_stop:
        rt.backend.send_stop("independent_search_safety_stop", mode="emergency")
        assert not rt._service_short_follow()
    written_before = len(driver.pairs)
    clock[0] += .06
    assert controller.activate(1, clock[0])
    feedback = (8, -5) if direction == 1 else (-5, 8)
    set_feedback(rt, clock, *feedback)
    assert publish(rt, clock, 480, distance=2.071, x=mirrored_x(.794, direction))
    rt._service_short_follow()
    if real_stop:
        assert len(driver.pairs) == written_before
        assert len(driver.stops) >= 2
    else:
        assert len(driver.pairs) == written_before + 1
        assert_arc(driver, direction)
        assert not driver.stops
