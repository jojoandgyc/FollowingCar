"""Visible understeer replay through the real writer, with fake serial only."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.short_follow import ShortFollowConfig, ShortFollowController, ShortFollowPlan
from car_control_modular.short_follow_yaw import ShortFollowYawResponse
from test_short_follow_executor import publish, set_feedback, short_runtime


def config(**kwargs):
    return ShortFollowConfig(enabled=True, yaw_response_exponent=.5,
        yaw_understeer_reduction_rpm=8, **kwargs)


def plan(now, *, cap=None, pair=(40, 56), x=.069, epoch=1, uid=1):
    cap = int(round(now*1000)) if cap is None else cap
    return ShortFollowPlan(uid, cap, epoch, *pair, "steer_left", cap, now-.04,
        now-.02, now+.28, 1.8, base_request_rpm=56., base_rpm=max(pair),
        speed_cap_rpm=56., longitudinal_reason="forward", yaw_capture_id=cap,
        yaw_capture_timestamp=now-.04, yaw_center_x_ratio=x, yaw_control_center_x_ratio=x)


def sample(stamp, pair):
    return SimpleNamespace(timestamp=stamp, trustworthy=True, left_error=0, right_error=0,
        left_forward_rpm=pair[0], right_forward_rpm=pair[1])


class ResponseCase:
    def __init__(self):
        self.response = ShortFollowYawResponse()
        self.receipt = SimpleNamespace(completed_at=10.)
        self.source = plan(10., cap=824)
        self.response.acknowledge(self.source, self.receipt, 40, 56)

    def tick(self, now, *, wheels=(50., 50.), source=None, feedback_stamp=None, acknowledge=True):
        self.source = source or plan(now)
        observed = sample(now if feedback_stamp is None else feedback_stamp, wheels)
        output = self.response.adjust(self.source, observed, self.receipt, now, config())
        if acknowledge:
            self.receipt = SimpleNamespace(completed_at=now)
            self.response.acknowledge(self.source, self.receipt, output.left_rpm, output.right_rpm)
        return output


def confirmed_case():
    a = ResponseCase()
    # All samples are after >=100 ms of completed command exposure; the
    # first two do not suffice, nor does an arbitrary number of rapid reads.
    for now in (10.10, 10.18, 10.26, 10.34, 10.42, 10.50):
        result = a.tick(now)
    assert result.yaw_common_reduction_rpm == 8
    return a


@pytest.mark.parametrize("x,expected", [(.59, 3), (.60, 5), (.62, 7), (.70, 12), (.90, 16)])
def test_low_offset_curve_is_smooth_bounded_and_opt_in(x, expected):
    new = ShortFollowController(config())
    old = ShortFollowController(ShortFollowConfig(enabled=True))
    left, right, _ = new._mix_yaw(60, 1.8, x, "forward")
    assert left-right == expected
    assert max(left, right) == 60 and 0 <= left-right <= 16
    if x == .59:
        assert old._mix_yaw(60, 1.8, x, "forward")[:2] == (60, 59)
    assert new._mix_yaw(60, 1.8, .58, "forward")[:2] == (60, 60)
    assert new._mix_yaw(0, 1.35, 1., "target_distance_reached")[:2] == (8, -8)


def test_cap833_wheel_response_replay_only_reduces_common_forward_speed():
    a = ResponseCase()
    outputs = []
    # CAP833 actual commanded pairs and low fresh wheel deltas (-6,-2,-1,0).
    # Exposure began at CAP824, not at the first low feedback sample.
    for now, pair, measured in [(10.10, (40, 56), (50, 56)),
                                (10.18, (42, 58), (50, 52)),
                                (10.26, (43, 59), (50, 51)),
                                (10.34, (43, 59), (50, 50))]:
        source = replace(plan(now, cap=833, pair=pair),
            capture_timestamp=10.06, yaw_capture_timestamp=10.06)
        output = a.tick(now, source=source, wheels=measured)
        outputs.append((output.left_rpm, output.right_rpm))
        assert output.left_rpm-output.right_rpm == -16
        assert output.capture_timestamp == source.capture_timestamp
        assert output.depth_timestamp == source.depth_timestamp
        assert output.expires_at == source.expires_at and output.epoch == source.epoch
        assert output.limit_reason == source.limit_reason and output.base_request_rpm == 56
    assert outputs == [(40, 56), (42, 58), (41, 57), (39, 55)]


def test_actual_cap824_to_833_timestamps_ack_and_feedback_replay():
    # run_20261010_202921_69780_74a8ce0e, 20:30:07.972..08.552.
    # Subtract 35740 from monotonic clocks. This is a command-only replay:
    # the recorded response is NOT a prediction of altered physical motion.
    rows = [
        # send, ACK, cap, capture, depth, expiry, yaw cap/stamp, control x,
        # normalized request, measured delta, feedback age (seconds)
        (.408926966, .425536949, 824, -.001735121, .162765549, .498264879,
         828, .195968892, .178273519, (30, 46), -9., .059158666),
        (.499308610, .511487709, 828, .195968892, .374503321, .695968892,
         830, .331047579, .131664041, (37, 53), -2., .002009531),
        (.572226875, .582751101, 830, .331047579, .474619961, .824619961,
         830, .331047579, .135503403, (37, 53), -1., .027966312),
        (.674096095, .683180400, 830, .331047579, .540167128, .831047579,
         833, .490865663, .094673733, (40, 56), -6., .006846405),
        (.749716582, .771376351, 833, .490865663, .640618301, .990618301,
         833, .490865663, .100956621, (42, 58), -2., .034382521),
        (.820825395, .830242483, 833, .490865663, .667984379, .990865663,
         833, .490865663, .094899835, (43, 59), -1., .015458805),
        (.886927297, .900945598, 833, .490865663, .667984379, .990865663,
         833, .490865663, .096399599, (43, 59), 0., .024158412),
    ]
    response = ShortFollowYawResponse()
    receipt = SimpleNamespace(completed_at=35740.321897651)
    response.acknowledge(plan(35740.32, cap=824, pair=(30, 46), epoch=15), receipt, 30, 46)
    pairs, reductions = [], []
    for send, ack, cap, capture, depth, expiry, yaw_cap, yaw_stamp, x, requested, delta, age in rows:
        now = 35740+send
        source = replace(plan(now, cap=cap, pair=requested, x=x, epoch=15),
            capture_timestamp=35740+capture, depth_timestamp=35740+depth,
            expires_at=35740+expiry, yaw_capture_id=yaw_cap,
            yaw_capture_timestamp=35740+yaw_stamp)
        measured = sample(now-age, (50+delta, 50))
        result = response.adjust(source, measured, receipt, now, config(depth_ttl_sec=.35))
        pairs.append((result.left_rpm, result.right_rpm))
        reductions.append(result.yaw_common_reduction_rpm)
        assert result.expires_at == source.expires_at
        receipt = SimpleNamespace(completed_at=35740+ack)
        response.acknowledge(source, receipt, result.left_rpm, result.right_rpm)
    assert reductions == [0, 0, 0, 2, 4, 6, 8]
    assert pairs == [(30, 46), (37, 53), (37, 53), (38, 54), (38, 54), (37, 53), (35, 51)]


def test_three_independent_samples_and_minimum_span_are_both_required():
    a = ResponseCase()
    for now in (10.10, 10.12, 10.14, 10.16):
        assert a.tick(now).yaw_common_reduction_rpm == 0
    assert a.tick(10.26).yaw_common_reduction_rpm == 2


def test_duplicate_feedback_cannot_confirm_or_deepen_response():
    a = ResponseCase()
    for now in (10.10, 10.14, 10.18, 10.22):
        assert a.tick(now, feedback_stamp=10.10).yaw_common_reduction_rpm == 0
    b = confirmed_case()
    # A duplicate remains within 150 ms but cannot make a new intervention.
    b.response._reduction = 2
    assert b.tick(10.56, feedback_stamp=10.50).yaw_common_reduction_rpm == 2


def test_one_good_sample_recovers_common_speed_in_bounded_steps():
    a = confirmed_case()
    result = a.tick(10.56, wheels=(40, 56))
    assert result.yaw_common_reduction_rpm == 6
    # Rapid writer calls and a duplicate good sample do not jump to full speed.
    assert a.tick(10.58, wheels=(40, 56), feedback_stamp=10.56).yaw_common_reduction_rpm == 6
    assert a.tick(10.62, wheels=(40, 56)).yaw_common_reduction_rpm == 4
    assert a.tick(10.68, wheels=(40, 56)).yaw_common_reduction_rpm == 2
    assert a.tick(10.74, wheels=(40, 56)).yaw_common_reduction_rpm == 0


@pytest.mark.parametrize("change", ["center", "reverse", "taper", "feedback_stale", "feedback_reverse"])
def test_old_understeer_evidence_cannot_trigger_in_new_or_unusable_context(change):
    a = confirmed_case()
    source, wheels, stamp = plan(10.56), (50, 50), 10.56
    if change == "center": source = plan(10.56, pair=(56, 56), x=.5)
    elif change == "reverse": source = plan(10.56, pair=(56, 40), x=.9)
    elif change == "taper": source = plan(10.56, pair=(53, 56), x=.56)
    elif change == "feedback_stale": stamp = 10.40
    elif change == "feedback_reverse": wheels = (7, -19)
    result = a.tick(10.56, source=source, wheels=wheels, feedback_stamp=stamp)
    assert result.yaw_common_reduction_rpm == 6
    assert result.yaw_response_sample_count == 0
    assert result.left_rpm-result.right_rpm == source.left_rpm-source.right_rpm
    assert result.yaw_response_reason == "response_recovery"


@pytest.mark.parametrize("change", ["expiry", "uid", "epoch", "unknown_ack", "pivot"])
def test_no_old_response_state_crosses_authority_or_actual_write_boundary(change):
    a = confirmed_case()
    source = plan(10.56)
    if change == "expiry": source = replace(source, expires_at=10.55)
    elif change == "uid": source = replace(source, uid=2)
    elif change == "epoch": source = replace(source, epoch=2)
    elif change == "unknown_ack": a.receipt = SimpleNamespace(completed_at=10.54)
    elif change == "pivot": source = plan(10.56, pair=(-8, 8))
    result = a.tick(10.56, source=source, acknowledge=False)
    assert result is source
    assert a.response._count == 0 and a.response._reduction == 0


def test_feedback_gap_cannot_complete_old_confirmation_window():
    a = ResponseCase()
    assert a.tick(10.10).yaw_response_sample_count == 1
    assert a.tick(10.18).yaw_response_sample_count == 2
    assert a.tick(10.34, feedback_stamp=10.18).yaw_response_sample_count == 0
    assert a.tick(10.42).yaw_response_sample_count == 1


def test_inner_wheel_remains_positive_and_differential_does_not_grow():
    a = confirmed_case()
    source = plan(10.56, pair=(3, 19))
    result = a.tick(10.56, source=source)
    assert (result.left_rpm, result.right_rpm) == (1, 17)
    assert result.moving and result.forwarding and not result.pivot


def test_compatible_defaults_and_environment(monkeypatch):
    baseline = ShortFollowConfig()
    assert baseline.yaw_response_exponent == 1 and baseline.yaw_understeer_reduction_rpm == 0
    monkeypatch.setenv("SHORT_FOLLOW_YAW_RESPONSE_EXPONENT", ".5")
    monkeypatch.setenv("SHORT_FOLLOW_YAW_UNDERSTEER_REDUCTION_RPM", "8")
    current = ShortFollowConfig.from_env()
    assert current.yaw_response_exponent == .5 and current.yaw_understeer_reduction_rpm == 8


@pytest.mark.parametrize("bad", [{"yaw_response_exponent": .49}, {"yaw_response_exponent": 1.01},
    {"yaw_understeer_reduction_rpm": -1}, {"yaw_understeer_reduction_rpm": 9},
    {"yaw_understeer_reduction_rpm": 1.5}])
def test_invalid_tuning_is_rejected(bad):
    with pytest.raises(ValueError):
        ShortFollowConfig(**bad)


def test_real_writer_fresh_cap_understeer_reaches_fake_packets_without_stop(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    owner._short_follow = ShortFollowController(config())
    owner._short_follow.activate(1, clock[0])
    results = []
    for index, now in enumerate((10.01, 10.11, 10.19, 10.27, 10.35, 10.43, 10.51)):
        clock[0] = now
        source = publish(rt, clock, 824+index, distance=1.8, x=.069)
        set_feedback(rt, clock, 0 if index == 0 else 50, 0 if index == 0 else 50)
        rt._service_short_follow()
        actual = owner._short_follow_last_applied_plan
        results.append(actual.yaw_common_reduction_rpm)
        assert actual.expires_at == source.expires_at
        assert actual.epoch == source.epoch
        assert owner._short_follow.snapshot().plan is source
        assert driver.pairs[-1] == (actual.left_rpm, -actual.right_rpm)
        assert 0 < actual.left_rpm < actual.right_rpm <= source.right_rpm
        assert actual.right_rpm-actual.left_rpm == 16
    assert results == [0, 0, 0, 2, 4, 6, 8]
    assert not driver.stops and rt._short_follow_executor._entry_stop_at is None
    assert not owner._brake_hold_active
    # No feedback state can keep the last plan alive without a new Depth.
    clock[0] = source.expires_at+.001
    set_feedback(rt, clock, 50, 50)
    before = len(driver.pairs)
    rt._service_short_follow()
    assert len(driver.pairs) == before and driver.stops == [1]


@pytest.mark.parametrize("change", ["center", "crossed_heading", "new_direction", "bad_identity", "reverse_feedback"])
def test_opt_in_response_preserves_writer_taper_direction_and_safety(monkeypatch, change):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    owner._short_follow = ShortFollowController(config())
    owner._short_follow.activate(1, clock[0])
    for index, now in enumerate((10.01, 10.11, 10.19, 10.27, 10.35, 10.43, 10.51)):
        clock[0] = now
        publish(rt, clock, 824+index, distance=1.8, x=.069)
        set_feedback(rt, clock, 0 if index == 0 else 50, 0 if index == 0 else 50)
        rt._service_short_follow()
    assert owner._short_follow_last_applied_plan.yaw_common_reduction_rpm == 8
    clock[0] = 10.57
    source = publish(rt, clock, 840, distance=1.8,
        x=.5 if change == "center" else .90 if change == "new_direction" else .069)
    observed = set_feedback(rt, clock, 50, 50)
    if change == "crossed_heading":
        source = replace(source, yaw_capture_yaw_deg=0.)
        owner._short_follow._state = replace(owner._short_follow.snapshot(), plan=source)
        observed.integrated_yaw_right_deg = -30.
        observed.yaw_rate_right_dps = 0.
        observed.yaw_rate_confirmed = True
    elif change == "bad_identity": owner._validated_visual_observation = False
    elif change == "reverse_feedback": observed.right_forward_rpm = -19.
    before = len(driver.pairs)
    rt._service_short_follow()
    if change in {"bad_identity", "reverse_feedback"}:
        assert len(driver.pairs) == before and driver.stops == [1]
        assert rt._short_follow_executor._yaw_response._count == 0
    else:
        assert not driver.stops
        actual = owner._short_follow_last_applied_plan
        assert actual.yaw_common_reduction_rpm == 6
        assert actual.yaw_response_sample_count == 0
        if change == "new_direction": assert actual.left_rpm > actual.right_rpm > 0
        else: assert actual.left_rpm == actual.right_rpm > 0
        assert actual.expires_at == source.expires_at and actual.epoch == source.epoch


@pytest.mark.parametrize("limit,expected,maximum_reduction", [(8, (1, 8), 5), (1, (0, 1), 0)])
def test_real_writer_low_limit_cannot_turn_response_correction_into_new_zero(
        monkeypatch, limit, expected, maximum_reduction):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    rt.backend.config = replace(rt.backend.config, max_target=limit)
    owner._short_follow = ShortFollowController(config())
    owner._short_follow.activate(1, clock[0])
    corrections = []
    for index, now in enumerate((10.01, 10.11, 10.19, 10.27, 10.35, 10.43, 10.51)):
        clock[0] = now
        source = publish(rt, clock, 824+index, distance=1.8, x=.069)
        # Exercise an already lawful, unscaled pair near the rounding edge.
        # The same production final limiter and fake dual-wheel ACKs apply.
        source = replace(source, left_rpm=7, right_rpm=23, base_rpm=23.)
        owner._short_follow._state = replace(owner._short_follow.snapshot(), plan=source)
        set_feedback(rt, clock, 0 if index == 0 else 1, 0 if index == 0 else 1)
        rt._service_short_follow()
        left, wire_right = driver.pairs[-1]
        corrections.append(owner._short_follow_last_applied_plan.yaw_common_reduction_rpm)
        assert max(abs(left), abs(wire_right)) <= limit
        if limit == 8:
            assert left >= 1 and -wire_right >= 1
    assert (left, -wire_right) == expected
    assert max(corrections) == maximum_reduction
    assert not driver.stops


def test_zero_hardware_limit_retains_the_original_stop_instead_of_raising_a_wheel(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    owner._short_follow = ShortFollowController(config())
    owner._short_follow.activate(1, clock[0])
    rt.backend.config = replace(rt.backend.config, max_target=0)
    clock[0] += .01
    publish(rt, clock, 833, distance=1.8, x=.069)
    set_feedback(rt, clock, 0, 0)
    rt._service_short_follow()
    assert driver.pairs == [] and driver.stops == [1]
    assert rt._short_follow_executor._stop_key[1] == "motor_limit_zero"


@pytest.mark.parametrize("limit", [0, 1])
def test_active_reduction_cannot_override_a_new_zero_or_one_rpm_limit(limit):
    a = confirmed_case()
    source = plan(10.56, pair=(7, 23))
    output = a.response.adjust(source, sample(10.56, (1, 1)), a.receipt, 10.56,
                               config(), wheel_limit=limit)
    assert output.yaw_common_reduction_rpm == 0
    scale = min(1., limit/max(output.left_rpm, output.right_rpm))
    actual = round(output.left_rpm*scale), round(output.right_rpm*scale)
    baseline = round(source.left_rpm*scale), round(source.right_rpm*scale)
    assert actual == baseline and max(actual) <= limit
