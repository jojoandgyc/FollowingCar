"""Real paired writer/backend with FakeDriver: STOP tails and raced exits."""
import pytest

from car_control_modular.detector_identity_lease import ValidatedVisualObservation
from car_control_modular.short_follow import ShortFollowConfig, ShortFollowController, ShortFollowObservation
from test_cap1040_limited_yaw_owner import setup as limited_runtime
from test_short_follow_executor import publish, set_feedback, short_runtime
from test_visible_wheel_continuity import feedback


def search_zero_entry(monkeypatch, *, lineage=True, stop=True):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    controller = owner._short_follow
    controller.deactivate("search", clock[0])
    if lineage:
        rt.backend.send_targets(-7, -7, "TURN", history_uid=1)
        assert not rt._service_short_follow()
    clock[0] += .05
    set_feedback(rt, clock, 13, 24)
    rt.backend.send_targets(0, 0, "OBSERVATION_ZERO", history_uid=1)
    assert not rt._service_short_follow()
    clock[0] += .06
    controller.activate(1, clock[0])
    plan = publish(rt, clock, 313, distance=1.35, x=.3)
    set_feedback(rt, clock, 13, 8)
    if stop:
        assert rt._service_short_follow()
        assert driver.stops == [1]
        assert rt._short_follow_executor._entry_stop_acknowledged
    return rt, owner, driver, clock, plan


def publish_delayed(rt, clock, capture, capture_stamp):
    now = clock[0]
    rt.owner._validated_visual_observation = ValidatedVisualObservation(
        1, 3, capture, capture_stamp, now, capture_stamp + .5, "full")
    return rt.owner._short_follow.update(ShortFollowObservation(
        1, capture, capture_stamp, now, 2., .4), now)


def test_cap313_small_two_wheel_tail_does_not_revoke_new_qualified_capture(monkeypatch):
    rt, owner, driver, clock, old_plan = search_zero_entry(monkeypatch)
    writer, controller = rt._short_follow_executor, owner._short_follow
    stopped_at = writer._entry_stop_at
    deadline, epoch, floor = writer._park_tail_until, controller.snapshot().epoch, controller._source_floor
    pair_count = len(driver.pairs)
    assert writer._park_tail_entry_stop_at == stopped_at
    for offset, pair in ((.06, (-3, -3)), (.09, (0, -3))):
        clock[0] = stopped_at + offset
        set_feedback(rt, clock, *pair)
        rt._service_short_follow()
        rt._service_short_follow()  # Repeated feedback is not quiet proof.
        assert writer._live_feedback_reason is None
        assert writer._entry_quiet_count == 0
        assert controller.snapshot().epoch == epoch
        assert controller._source_floor == floor
        assert controller.snapshot().plan is old_plan
        assert writer._park_tail_until == deadline
        assert driver.stops == [1] and len(driver.pairs) == pair_count
    # Exposure preceded the -3/-3 handling, not the genuine first STOP.
    clock[0] = stopped_at + .10
    plan = publish_delayed(rt, clock, 315, stopped_at + .03)
    assert plan is not None
    expiry = plan.expires_at
    for offset in (.12, .17):
        clock[0] = stopped_at + offset
        set_feedback(rt, clock, 0, 0)
        rt._service_short_follow()
    assert writer._entry_stop_at is None
    assert driver.stops == [1] and len(driver.pairs) == pair_count + 1
    assert min(driver.pairs[-1][0], -driver.pairs[-1][1]) > 0
    assert controller.snapshot().plan is plan and plan.expires_at == expiry


@pytest.mark.parametrize("feedback_state", ["fresh", "stale", "missing"])
def test_entry_tail_has_fixed_deadline_even_with_new_plans_and_stop_refresh(monkeypatch, feedback_state):
    rt, owner, driver, clock, _ = search_zero_entry(monkeypatch)
    writer, controller = rt._short_follow_executor, owner._short_follow
    stopped_at, deadline = writer._entry_stop_at, writer._park_tail_until
    clock[0] = stopped_at + .10
    set_feedback(rt, clock, -3, -3)
    rt._service_short_follow()
    clock[0] += .10
    assert publish(rt, clock, 315)
    writer._stop_locked("entry_settling", controller.snapshot().epoch, clock[0], force=True)
    assert writer._park_tail_until == deadline and writer._entry_stop_at == stopped_at
    before = len(driver.pairs)
    clock[0] = deadline + .001
    if feedback_state == "fresh":
        set_feedback(rt, clock, -3, -3)
    elif feedback_state == "missing":
        rt.get_steering_feedback = lambda: None
    rt._service_short_follow()
    assert writer._live_feedback_reason == "feedback_reverse"
    assert controller.snapshot().plan is None
    assert len(driver.pairs) == before


@pytest.mark.parametrize("adverse", ["minus38", "minus6", "motor_error", "untrusted",
    "before_stop", "external_stop", "explicit_stop", "hazard"])
def test_stop_tail_never_waives_strong_reverse_or_real_fault(monkeypatch, adverse):
    rt, owner, driver, clock, _ = search_zero_entry(monkeypatch)
    writer = rt._short_follow_executor
    stopped_at = writer._entry_stop_at
    clock[0] += .06
    sample = set_feedback(rt, clock, -3, -3)
    if adverse == "minus38": sample.left_forward_rpm, sample.right_forward_rpm = 0, -38
    elif adverse == "minus6": sample.right_forward_rpm = -6
    elif adverse == "motor_error": sample.right_error = 1
    elif adverse == "untrusted": sample.trustworthy = False
    elif adverse == "before_stop": sample.timestamp = stopped_at - .001
    elif adverse == "external_stop": rt.backend.send_stop("independent_safety", mode="emergency")
    elif adverse == "explicit_stop": owner._explicit_stop_requested = True
    elif adverse == "hazard": rt.hard_stop_check = lambda _action: True
    before = len(driver.pairs)
    rt._service_short_follow()
    assert len(driver.pairs) == before
    assert owner._short_follow.snapshot().plan is None
    assert len(driver.stops) >= 2
    assert writer._park_tail_entry_stop_at is None


def test_unknown_zero_entry_does_not_manufacture_search_response(monkeypatch):
    rt, owner, driver, clock, _ = search_zero_entry(monkeypatch, lineage=False)
    assert rt._short_follow_executor._park_tail_entry_stop_at is None
    clock[0] += .06
    set_feedback(rt, clock, -3, -3)
    rt._service_short_follow()
    assert rt._short_follow_executor._live_feedback_reason == "feedback_reverse"
    assert owner._short_follow.snapshot().plan is None and driver.stops == [1, 1]


def test_failed_entry_stop_cannot_create_tail_credit(monkeypatch):
    rt, _, driver, _, _ = search_zero_entry(monkeypatch, stop=False)
    def fail(*_args, **_kwargs):
        raise RuntimeError("fake failed STOP ACK")
    monkeypatch.setattr(driver, "stop_all", fail)
    with pytest.raises(Exception, match="fake failed STOP ACK"):
        rt._service_short_follow()
    writer = rt._short_follow_executor
    assert writer._park_tail_entry_stop_at is None
    assert writer._park_tail_until == float("-inf")
    assert not writer._entry_stop_acknowledged


def test_tail_quiet_samples_cannot_restore_expired_plan(monkeypatch):
    rt, owner, driver, clock, plan = search_zero_entry(monkeypatch)
    before = len(driver.pairs)
    clock[0] += .06
    set_feedback(rt, clock, -3, -3)
    rt._service_short_follow()
    clock[0] = plan.expires_at + .001
    for _ in range(2):
        set_feedback(rt, clock, 0, 0)
        rt._service_short_follow()
        clock[0] += .02
    assert len(driver.pairs) == before
    assert owner._short_follow.snapshot().plan is plan
    assert not plan.valid(clock[0])


def test_entry_tail_needs_two_distinct_ordered_quiet_samples(monkeypatch):
    rt, _, driver, clock, _ = search_zero_entry(monkeypatch)
    writer = rt._short_follow_executor
    before = len(driver.pairs)
    clock[0] += .06
    set_feedback(rt, clock, -3, -3)
    rt._service_short_follow()
    clock[0] += .04
    quiet_at = clock[0]
    set_feedback(rt, clock, 0, 0)
    rt._service_short_follow()
    for stamp in (quiet_at, quiet_at - .02):
        clock[0] += .01
        set_feedback(rt, clock, 0, 0, stamp=stamp)
        rt._service_short_follow()
        assert writer._entry_quiet_count == 1
        assert len(driver.pairs) == before and driver.stops == [1]
    clock[0] += .02
    set_feedback(rt, clock, 0, 0)
    rt._service_short_follow()
    assert len(driver.pairs) == before + 1 and driver.stops == [1]


def raced_exit(monkeypatch, phase, successor="pivot", adverse=None, *, soft_handoff=False):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt._service_short_follow()
    writer, controller = rt._short_follow_executor, owner._short_follow
    clock[0] += .06
    set_feedback(rt, clock, 10, 10)
    calls = []
    def handoff():
        assert not writer._owned
        calls.append(True)
        if successor == "pivot":
            rt.backend.send_targets(-5, -5, "LIMITED_YAW_HANDOFF", history_uid=1)
            return True
        if successor in {"guard_zero", "unowned_zero"}:
            rt.backend.send_targets(0, 0, "LIMITED_YAW_HANDOFF", history_uid=1)
            rt._visible_wheel_waiting = successor == "guard_zero"
        return False
    rt._write_limited_yaw_successor = handoff
    def retire():
        if soft_handoff:
            controller.retire_for_lateral_handoff(clock[0])
        else:
            controller.deactivate("identity_or_search_handoff", clock[0])
    if phase == "start":
        retire()
    elif phase == "safety_read":
        rt.hard_stop_check = lambda _action: (retire(), False)[1]
    elif phase == "stop_recheck":
        clock[0] = controller.snapshot().plan.expires_at + .001
        set_feedback(rt, clock, 10, 10)
        safety_calls = []
        def safety(_action):
            safety_calls.append(True)
            if len(safety_calls) == 2:
                retire()
            return False
        rt.hard_stop_check = safety
    elif phase == "prepare":
        original = rt.backend.prepare_speed_mode
        def prepare():
            original()
            retire()
        rt.backend.prepare_speed_mode = prepare
    else:
        raise AssertionError(phase)
    if adverse == "explicit": owner._explicit_stop_requested = True
    elif adverse == "hazard": rt.hard_stop_check = lambda _action: True
    elif adverse == "external_stop": rt.backend.send_stop("external", mode="emergency")
    elif adverse == "feedback_reverse": set_feedback(rt, clock, 0, -38)
    rt._service_short_follow()
    return rt, owner, driver, writer, calls


@pytest.mark.parametrize("phase", ["start", "safety_read", "stop_recheck", "prepare"])
@pytest.mark.parametrize("successor", ["pivot", "guard_zero"])
@pytest.mark.parametrize("soft_handoff", [False, True])
def test_mid_tick_deactivation_uses_same_guarded_owner_exit(monkeypatch, phase, successor, soft_handoff):
    rt, owner, driver, writer, calls = raced_exit(
        monkeypatch, phase, successor, soft_handoff=soft_handoff)
    assert calls == [True]
    assert not driver.stops
    assert driver.pairs[-1] == ((-5, -5) if successor == "pivot" else (0, 0))
    assert not writer._owned and not owner._short_follow.snapshot().active
    assert not rt._service_short_follow()  # Old owner cannot emit a later STOP.
    assert not driver.stops


@pytest.mark.parametrize("successor", ["none", "unowned_zero"])
def test_failed_or_unowned_successor_still_requires_stop_barrier(monkeypatch, successor):
    _, _, driver, writer, calls = raced_exit(monkeypatch, "prepare", successor)
    assert calls == [True] and driver.stops == [1]
    assert not writer._owned


@pytest.mark.parametrize("adverse", ["explicit", "hazard", "external_stop", "feedback_reverse"])
def test_raced_handoff_cannot_bypass_hard_safety(monkeypatch, adverse):
    _, _, driver, _, calls = raced_exit(monkeypatch, "prepare", adverse=adverse)
    assert not calls and driver.stops
    assert len(driver.pairs) == 1


@pytest.mark.parametrize("resume", [True, False], ids=["fresh_feedback", "expired_identity"])
def test_real_guard_zero_retains_its_own_resume_checks(monkeypatch, resume):
    rt, owner, driver, _, clock, _ = limited_runtime(monkeypatch)
    owner._short_follow = ShortFollowController(ShortFollowConfig(enabled=True))
    owner._short_follow.activate(1, 9.9)
    assert owner._short_follow.update(ShortFollowObservation(
        1, 1039, 9.95, 9.97, 1.4, .3), 10.)
    owner._short_follow.retire_for_lateral_handoff(10.)
    writer = rt._short_follow_executor_instance()
    writer._owned = True
    writer._controller = owner._short_follow
    rt.get_steering_feedback = lambda: feedback(9., -4, 4)
    assert rt._service_short_follow()
    assert driver.pairs == [(0, 0)] and not driver.stops
    assert not writer._owned and rt._visible_wheel_waiting
    clock[0] += .06 if resume else .151
    rt.get_steering_feedback = lambda: feedback(clock[0], 0, 0)
    rt._service_follow_wheels()
    assert not driver.stops
    if resume:
        assert driver.pairs[-1] == (-7, -7)
    else:
        assert all(pair == (0, 0) for pair in driver.pairs)
    assert not owner._short_follow.snapshot().active
