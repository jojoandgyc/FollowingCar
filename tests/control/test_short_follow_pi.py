"""Distance PI in the single paired-wheel path, without motor hardware.

These regressions separate controller state from authority: repeated writer
ticks and rejected measurements must not integrate error or renew evidence.
"""
from dataclasses import replace

import pytest

from car_control_modular.short_follow import (
    ShortFollowConfig,
    ShortFollowController,
    ShortFollowObservation,
)


def _controller(**overrides):
    cfg = ShortFollowConfig(enabled=True, **overrides)
    core = ShortFollowController(cfg)
    assert core.activate(1, now=99.0)
    return core


def _sample(stamp, *, capture=1, uid=1, distance=1.56, center=.5,
            depth_age=.02, capture_age=.04, raw=None):
    return ShortFollowObservation(
        uid=uid, capture_id=capture,
        capture_timestamp=stamp-capture_age,
        depth_timestamp=stamp-depth_age,
        distance_m=distance, center_x_ratio=center,
        raw_distance_m=raw,
    )


def _publish(core, stamp, **kwargs):
    plan = core.update(_sample(stamp, **kwargs), now=stamp)
    assert plan is not None
    return plan


def test_restored_pi_uses_metre_second_units_not_minimal_distance_table():
    core = _controller()
    plan = _publish(core, 100.)
    cfg = core.config
    effective_error = 1.56-cfg.target_distance_m-cfg.deadband_m
    expected_p = cfg.kp_per_sec*effective_error*60/cfg.wheel_circumference_m
    assert plan.p_rpm == pytest.approx(expected_p)
    assert plan.i_rpm == pytest.approx(0.)
    assert plan.integral_dt_sec == pytest.approx(0.)
    assert plan.base_request_rpm == pytest.approx(expected_p)
    assert plan.base_rpm > 16
    assert plan.left_rpm == plan.right_rpm == plan.base_rpm

    farther = _publish(core, 100.1, capture=2, distance=2.)
    assert farther.base_rpm > 40
    assert farther.base_rpm <= farther.speed_cap_rpm
    assert farther.base_rpm <= cfg.max_rpm


def test_persistent_positive_error_adds_integral_without_changing_authority_epoch():
    core = _controller()
    plans = [_publish(core, 100.+index*.1, capture=index+1)
             for index in range(8)]
    cfg = core.config
    per_second = cfg.ki_per_sec2*(1.56-cfg.target_distance_m-cfg.deadband_m)
    assert plans[-1].i_rpm == pytest.approx(
        per_second*.7*60/cfg.wheel_circumference_m)
    assert plans[-1].base_rpm > plans[0].base_rpm
    assert len({plan.epoch for plan in plans}) == 1
    assert len({plan.sequence for plan in plans}) == 8
    assert all(plan.moving for plan in plans)


def test_integral_time_is_depth_sample_interval_not_processing_interval():
    core = _controller()
    first = _publish(core, 100.)
    # New depth arrived 100 ms later, but this callback took 170 ms to run.
    sample = replace(_sample(100.17, capture=2),
                     depth_timestamp=first.depth_timestamp+.10)
    second = core.update(sample, now=100.17)
    assert second is not None
    assert second.integral_dt_sec == pytest.approx(.10)
    expected = core.config.ki_per_sec2*.13*.10*60/core.config.wheel_circumference_m
    assert second.i_rpm == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["duplicate", "new_capture_same_depth", "older_depth",
                                  "older_capture", "wrong_uid", "future", "expired"])
def test_rejected_observation_neither_integrates_nor_renews_plan(kind):
    core = _controller()
    _publish(core, 100.)
    accepted = _sample(100.1, capture=2)
    plan = core.update(accepted, now=100.1)
    candidate = {
        "duplicate": accepted,
        "new_capture_same_depth": replace(accepted, capture_id=3,
                                            capture_timestamp=100.12),
        "older_depth": replace(accepted, depth_timestamp=accepted.depth_timestamp-.01),
        "older_capture": replace(accepted, capture_id=1,
                                   depth_timestamp=accepted.depth_timestamp+.01),
        "wrong_uid": _sample(100.15, capture=3, uid=2),
        "future": replace(_sample(100.15, capture=3), depth_timestamp=100.16),
        "expired": _sample(100.15, capture=3, depth_age=.301),
    }[kind]
    assert core.update(candidate, now=100.15) is None
    assert core.snapshot().plan is plan
    next_plan = _publish(core, 100.2, capture=3)
    assert next_plan.integral_dt_sec == pytest.approx(.1)
    assert next_plan.i_rpm == pytest.approx(plan.i_rpm*2)
    assert next_plan.expires_at == pytest.approx(next_plan.depth_timestamp+.30)


def test_writer_snapshot_reads_are_not_pi_updates():
    core = _controller()
    _publish(core, 100.)
    plan = _publish(core, 100.1, capture=2)
    for _ in range(100):
        with core.write_snapshot() as snapshot:
            assert snapshot.plan is plan
    assert core.snapshot().plan.i_rpm == plan.i_rpm
    assert core.snapshot().plan.expires_at == plan.expires_at


def test_sample_gap_freezes_short_memory_without_low_speed_restart():
    core = _controller()
    _publish(core, 100.)
    previous = _publish(core, 100.1, capture=2)
    held = _publish(core, 100.42, capture=3)
    assert held.integral_dt_sec == pytest.approx(0.)
    assert held.i_rpm == pytest.approx(previous.i_rpm)
    assert held.base_rpm == previous.base_rpm
    resumed = _publish(core, 100.52, capture=4)
    assert resumed.integral_dt_sec == pytest.approx(.1)
    assert resumed.i_rpm > held.i_rpm


def test_long_sample_gap_clears_integral_without_backfilling_missing_time():
    core = _controller()
    _publish(core, 100.)
    previous = _publish(core, 100.1, capture=2)
    assert previous.i_rpm > 0
    resumed = _publish(core, 100.47, capture=3)
    assert resumed.integral_dt_sec == pytest.approx(0.)
    assert resumed.i_rpm == pytest.approx(0.)
    assert resumed.base_request_rpm == pytest.approx(resumed.p_rpm)
    assert resumed.moving


def test_old_plan_expiry_does_not_reset_current_fresh_sample_pi():
    core = _controller()
    first = _publish(core, 100.)
    previous = _publish(core, 100.1, capture=2)
    now = 100.39
    assert not previous.valid(now)
    # The depth-sample gap is only 200 ms; delayed processing is not a new
    # physical stop, nor permission to integrate the entire callback gap.
    sample = replace(_sample(now, capture=3),
                     depth_timestamp=previous.depth_timestamp+.2)
    resumed = core.update(sample, now=now)
    assert resumed is not None and resumed.moving
    assert resumed.integral_dt_sec == pytest.approx(.2)
    assert resumed.i_rpm > previous.i_rpm > first.i_rpm
    assert resumed.epoch == previous.epoch
    assert resumed.base_rpm >= first.base_rpm


def test_ordinary_turn_and_straight_updates_preserve_pi_memory_and_never_insert_zero():
    core = _controller()
    plans = [_publish(core, 100.+index*.1, capture=index+1, center=center)
             for index, center in enumerate((.2, .5, .8, .5, .2))]
    assert plans[0].left_rpm < plans[0].right_rpm
    assert plans[1].left_rpm == plans[1].right_rpm
    assert plans[2].left_rpm > plans[2].right_rpm
    assert all(plan.left_rpm > 0 and plan.right_rpm > 0 for plan in plans)
    assert all(later.i_rpm > prior.i_rpm for prior, later in zip(plans, plans[1:]))
    assert len({plan.epoch for plan in plans}) == 1


def test_continuous_braking_cap_limits_outer_wheel_without_integral_windup():
    core = _controller()
    plans = [_publish(core, 100.+index*.1, capture=index+1, distance=3., center=.8)
             for index in range(20)]
    assert all(plan.moving for plan in plans)
    assert all(max(plan.left_rpm, plan.right_rpm) <= plan.speed_cap_rpm for plan in plans)
    assert all(plan.base_request_rpm > plan.speed_cap_rpm for plan in plans)
    assert all(plan.i_rpm == pytest.approx(0.) for plan in plans)
    assert all(plan.base_rpm > 40 for plan in plans)


def test_integral_is_bounded_when_persistent_error_eventually_hits_output_cap():
    core = _controller(integral_max_m_s=.02)
    plans = [_publish(core, 100.+index*.1, capture=index+1)
             for index in range(100)]
    max_i = .02*60/core.config.wheel_circumference_m
    assert all(0 <= plan.i_rpm <= max_i+1e-9 for plan in plans)
    assert plans[-1].i_rpm == pytest.approx(max_i)


@pytest.mark.parametrize("event", ["near_distance", "raw_near_distance", "hard_stop", "uid_change"])
def test_real_stop_and_identity_change_clear_integral(event):
    core = _controller()
    _publish(core, 100.)
    previous = _publish(core, 100.1, capture=2)
    assert previous.i_rpm > 0
    uid = 1
    if event in {"near_distance", "raw_near_distance"}:
        stopped = _publish(core, 100.2, capture=3,
                           distance=1.49 if event == "near_distance" else 2.,
                           raw=1.49 if event == "raw_near_distance" else None)
        assert not stopped.moving and stopped.i_rpm == pytest.approx(0.)
    elif event == "hard_stop":
        core.revoke("manual_emergency", now=100.2)
    else:
        uid = 2
        assert core.activate(uid, now=100.2)
    resumed = _publish(core, 100.3, capture=4, uid=uid)
    assert resumed.moving
    assert resumed.i_rpm == pytest.approx(0.)


def test_lower_final_applied_budget_rolls_back_unacknowledged_integral_step():
    core = _controller()
    first = _publish(core, 100.)
    limited = _publish(core, 100.1, capture=2)
    assert limited.i_rpm > first.i_rpm
    core.acknowledge_output(limited, base_rpm=10.)
    # The immutable requested plan remains useful for diagnosing the cap.
    assert core.snapshot().plan is limited
    resumed = _publish(core, 100.2, capture=3)
    assert resumed.i_rpm == pytest.approx(limited.i_rpm)


@pytest.mark.parametrize("stale_kind", ["sequence", "epoch"])
def test_stale_write_ack_cannot_rollback_new_plan_integral(stale_kind):
    core = _controller()
    _publish(core, 100.)
    old = _publish(core, 100.1, capture=2)
    if stale_kind == "epoch":
        core.revoke("manual_emergency", now=100.2)
        _publish(core, 100.3, capture=3)
        current = _publish(core, 100.4, capture=4)
        next_stamp, next_capture = 100.5, 5
    else:
        current = _publish(core, 100.2, capture=3)
        next_stamp, next_capture = 100.3, 4
    core.acknowledge_output(old, base_rpm=0.)
    following = _publish(core, next_stamp, capture=next_capture)
    one_step = core.config.ki_per_sec2*.13*.1*60/core.config.wheel_circumference_m
    assert following.i_rpm == pytest.approx(current.i_rpm+one_step)


def test_equal_or_higher_applied_speed_does_not_remove_pi_compensation():
    core = _controller()
    _publish(core, 100.)
    current = _publish(core, 100.1, capture=2)
    core.acknowledge_output(current, base_rpm=current.base_rpm)
    following = _publish(core, 100.2, capture=3)
    assert following.i_rpm == pytest.approx(2*current.i_rpm)


def test_repeated_ack_of_one_plan_cannot_remove_older_integral_memory():
    core = _controller()
    _publish(core, 100.)
    executed = _publish(core, 100.1, capture=2)
    assert core.acknowledge_output(executed, base_rpm=executed.base_rpm)
    current = _publish(core, 100.2, capture=3)
    for _ in range(10):
        core.acknowledge_output(current, base_rpm=10.)
    following = _publish(core, 100.3, capture=4)
    assert following.i_rpm == pytest.approx(current.i_rpm)


@pytest.mark.parametrize("publishes_per_write", [2, 3])
def test_fast_producer_superseded_plans_cannot_leak_integral_past_writer_cap(publishes_per_write):
    core = _controller()
    first = _publish(core, 100., distance=1.6)
    assert first.base_rpm > 30
    assert core.acknowledge_output(first, base_rpm=30.)
    step_i = core.config.ki_per_sec2*.17*.03*60/core.config.wheel_circumference_m
    sequence = 1
    for _ in range(8):
        for step in range(publishes_per_write):
            sequence += 1
            plan = _publish(core, 100.+(sequence-1)*.03,
                            capture=sequence, distance=1.6)
            assert plan.i_rpm == pytest.approx(step_i*(step+1))
        assert plan.base_rpm > 30
        assert core.acknowledge_output(plan, base_rpm=30.)
        # The ACK covers every unexecuted integral increment represented by
        # this latest publication, not just its final 30 ms increment.
        assert core.snapshot().plan is plan


def test_full_ack_preserves_executed_integral_and_only_later_pending_work_rolls_back():
    core = _controller()
    _publish(core, 100., distance=1.6)
    for index in range(1, 4):
        executed = _publish(core, 100.+index*.03, capture=index+1, distance=1.6)
    assert executed.i_rpm > 0
    assert core.acknowledge_output(executed, base_rpm=executed.base_rpm)
    for index in range(4, 7):
        pending = _publish(core, 100.+index*.03, capture=index+1, distance=1.6)
    assert pending.i_rpm > executed.i_rpm
    assert core.acknowledge_output(pending, base_rpm=30.)
    next_plan = _publish(core, 100.21, capture=8, distance=1.6)
    step_i = core.config.ki_per_sec2*.17*.03*60/core.config.wheel_circumference_m
    assert next_plan.i_rpm == pytest.approx(executed.i_rpm+step_i)


@pytest.mark.parametrize("invalid_base", [None, -1., float("nan"), float("inf"), True])
def test_invalid_ack_does_not_commit_or_forget_unexecuted_integral(invalid_base):
    core = _controller()
    _publish(core, 100., distance=1.6)
    for index in range(1, 4):
        pending = _publish(core, 100.+index*.03, capture=index+1, distance=1.6)
    assert not core.acknowledge_output(pending, base_rpm=invalid_base)
    later = _publish(core, 100.12, capture=5, distance=1.6)
    assert later.i_rpm > pending.i_rpm
    assert core.acknowledge_output(later, base_rpm=30.)
    following = _publish(core, 100.15, capture=6, distance=1.6)
    step_i = core.config.ki_per_sec2*.17*.03*60/core.config.wheel_circumference_m
    assert following.i_rpm == pytest.approx(step_i)


@pytest.mark.parametrize("event", ["near_stop", "hard_stop", "uid_change"])
def test_stop_or_new_identity_discards_pending_ack_accounting_before_new_execution(event):
    core = _controller()
    _publish(core, 100., distance=1.6)
    for index in range(1, 4):
        old_pending = _publish(core, 100.+index*.03, capture=index+1, distance=1.6)
    assert old_pending.i_rpm > 0
    uid = 1
    if event == "near_stop":
        assert not _publish(core, 100.12, capture=5, distance=1.49).moving
    elif event == "hard_stop":
        core.revoke("manual_emergency", now=100.12)
    else:
        uid = 2
        assert core.activate(uid, now=100.12)
    first_new = _publish(core, 100.2, capture=6, uid=uid, distance=1.6)
    assert first_new.i_rpm == pytest.approx(0.)
    executed = _publish(core, 100.23, capture=7, uid=uid, distance=1.6)
    assert core.acknowledge_output(executed, base_rpm=executed.base_rpm)
    assert not core.acknowledge_output(old_pending, base_rpm=0.)
    pending = _publish(core, 100.26, capture=8, uid=uid, distance=1.6)
    assert core.acknowledge_output(pending, base_rpm=30.)
    following = _publish(core, 100.29, capture=9, uid=uid, distance=1.6)
    assert following.i_rpm == pytest.approx(2*executed.i_rpm)


def test_raw_near_evidence_caps_safety_but_filtered_distance_drives_pi_error():
    core = _controller()
    plan = _publish(core, 100., distance=2.2, raw=1.8)
    cfg = core.config
    expected_p = cfg.kp_per_sec*(2.2-cfg.target_distance_m-cfg.deadband_m)*60/cfg.wheel_circumference_m
    assert plan.p_rpm == pytest.approx(expected_p)
    independently_close = _publish(_controller(), 100., distance=1.8)
    assert plan.speed_cap_rpm == independently_close.speed_cap_rpm
    assert plan.base_rpm <= plan.speed_cap_rpm


def test_pi_has_no_forced_minimum_starting_rpm_or_180_rpm_launch_pulse():
    core = _controller(kp_per_sec=.1, ki_per_sec2=0.)
    plan = _publish(core, 100., distance=1.6)
    assert 0 < plan.base_rpm < 16
    assert plan.base_request_rpm == pytest.approx(plan.p_rpm)
    assert plan.i_rpm == pytest.approx(0.)


@pytest.mark.parametrize("gap,expected_dt,retained", [
    (.30, .30, True), (.35, 0., True), (.35001, 0., False),
])
def test_integral_gap_boundaries_use_source_time_without_float_reset_chatter(gap, expected_dt, retained):
    core = _controller()
    _publish(core, 100.)
    previous = _publish(core, 100.1, capture=2)
    plan = _publish(core, 100.1+gap, capture=3)
    assert plan.integral_dt_sec == pytest.approx(expected_dt)
    if retained:
        assert plan.i_rpm >= previous.i_rpm
    else:
        assert plan.i_rpm == pytest.approx(0.)


@pytest.mark.parametrize("capture,distance,raw,center,expected", [
    (404, 1.4664, 1.4008, .6512, (3, -3)),
    (409, 1.4593, 1.4833, .7090, (5, -5)),
    (512, 1.4298, 1.3936, .7065, (5, -5)),
])
def test_logged_near_distance_offset_stops_translation_not_centering(capture, distance, raw, center, expected):
    core = _controller()
    plan = _publish(core, 100., capture=capture, distance=distance, raw=raw, center=center)
    assert (plan.left_rpm, plan.right_rpm) == expected
    assert plan.moving and plan.pivot and not plan.forwarding
    assert plan.reason == "pivot_right"
    assert plan.base_rpm == 0 and plan.i_rpm == 0
    assert plan.limit_reason == "target_distance_reached"
    assert plan.expires_at == pytest.approx(100.-.02+.30)


@pytest.mark.parametrize("center,expected_delta", [(.60, 2), (.70, 10), (.80, 18), (.95, 18)])
def test_stronger_arc_steering_does_not_raise_the_longitudinal_pi_or_outer_wheel(center, expected_delta):
    straight = _publish(_controller(), 100., distance=2., center=.5)
    turn = _publish(_controller(), 100., distance=2., center=center)
    assert turn.base_request_rpm == straight.base_request_rpm
    assert turn.p_rpm == straight.p_rpm and turn.i_rpm == straight.i_rpm
    assert turn.base_rpm == straight.base_rpm
    assert turn.left_rpm == straight.left_rpm
    assert turn.left_rpm - turn.right_rpm == expected_delta
    assert turn.forwarding and not turn.pivot
    assert 0 <= turn.right_rpm <= turn.left_rpm <= turn.speed_cap_rpm


@pytest.mark.parametrize("center", [.05, .20, .80, .95])
def test_near_turn_is_bounded_signed_pair_and_never_becomes_translation(center):
    core = _controller()
    plan = _publish(core, 100., distance=1.45, center=center)
    assert plan.pivot and plan.moving and not plan.forwarding
    assert plan.left_rpm == -plan.right_rpm
    assert max(abs(plan.left_rpm), abs(plan.right_rpm)) <= 8
    assert plan.base_rpm == 0
    assert (plan.left_rpm < 0) == (center < .5)


@pytest.mark.parametrize("distance,raw,center", [
    (1.45, None, .5), (1.45, None, .42), (1.45, None, .58),
    (1.10, None, .95), (1.05, None, .05),
    (1.45, 1.10, .95), (2.5, 1.05, .95),
])
def test_centered_or_close_danger_still_requests_actual_stop(distance, raw, center):
    plan = _publish(_controller(), 100., distance=distance, raw=raw, center=center)
    assert plan.left_rpm == plan.right_rpm == 0
    assert not plan.moving and not plan.forwarding and not plan.pivot


def test_forward_arc_to_near_pivot_to_forward_arc_has_no_inserted_stop_or_epoch_change():
    core = _controller()
    distances = (1.70, 1.49, 1.54, 1.56, 1.70)
    plans = [_publish(core, 100.+n*.1, capture=n+1, distance=d, center=.8)
             for n, d in enumerate(distances)]
    assert [plan.pivot for plan in plans] == [False, True, True, False, False]
    assert all(plan.moving for plan in plans)
    assert len({plan.epoch for plan in plans}) == 1
    assert all(plan.base_rpm == 0 for plan in plans if plan.pivot)
    assert all(plan.base_rpm > 0 for plan in plans if plan.forwarding)


def test_pivot_ack_and_repeated_samples_cannot_accumulate_forward_integral():
    core = _controller()
    _publish(core, 100., distance=1.56)
    before = _publish(core, 100.1, capture=2, distance=1.56)
    assert before.i_rpm > 0
    # Establish a distance hold, then let its normal hysteresis permit yaw
    # only; a positive distance error in that band must not accumulate I.
    plan = _publish(core, 100.2, capture=3, distance=1.49, center=.8)
    assert plan.pivot and plan.i_rpm == 0
    for n in range(3, 8):
        plan = _publish(core, 100.+n*.1, capture=n+1, distance=1.54, center=.8)
        assert plan.pivot and plan.i_rpm == 0 and plan.integral_dt_sec == 0
        assert core.acknowledge_output(plan, base_rpm=0)
    resumed = _publish(core, 100.8, capture=9, distance=1.56, center=.8)
    assert resumed.forwarding and resumed.i_rpm == 0


@pytest.mark.parametrize("settings", [{"pivot_max_rpm": 0}, {"yaw_max_delta_rpm": 0},
                                      {"yaw_max_delta_rpm": 1}])
def test_disabled_or_subinteger_pivot_budget_is_not_rounded_up_to_motion(settings):
    cfg = _controller(**settings)
    plan = _publish(cfg, 100., distance=1.45, center=.95)
    assert not plan.moving


@pytest.mark.parametrize("distance,center,allowed", [
    (1.4, .8, True), (1.1, .8, False), (1.1001, .8, True),
    (1.4, .5, False), (1.4, .42, False), (1.4, .58, False),
    (1.4, 1.01, False), (float("nan"), .8, False), (1.4, float("nan"), False),
])
def test_pivot_eligibility_shared_with_settled_search_handoff(distance, center, allowed):
    cfg = ShortFollowConfig(enabled=True)
    assert cfg.pivot_allowed(distance, center) is allowed
