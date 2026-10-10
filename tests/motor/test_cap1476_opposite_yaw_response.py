"""Opposite wheel DIFFERENTIAL under forward drive, with no real devices.

Recorded feedback is replayed unchanged: corrected commands here are not a
prediction of the physical chassis response to those different commands.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.short_follow import (
    ShortFollowConfig, ShortFollowController, ShortFollowPlan,
)
from car_control_modular.short_follow_yaw import ShortFollowYawResponse
from test_short_follow_executor import publish, set_feedback, short_runtime


def config():
    return ShortFollowConfig(enabled=True, depth_ttl_sec=.35,
        yaw_understeer_reduction_rpm=8, yaw_response_exponent=.5)


def plan(now, *, cap=None, pair=(100, 84), x=.9, epoch=42):
    cap = int(round(now * 1000)) if cap is None else cap
    return ShortFollowPlan(1, cap, epoch, *pair, "steer_right", cap,
        now-.04, now-.02, now+.28, 2.74, base_request_rpm=289.52,
        base_rpm=max(pair), speed_cap_rpm=100.65, longitudinal_reason="forward",
        yaw_capture_id=cap, yaw_capture_timestamp=now-.04,
        yaw_center_x_ratio=x, yaw_control_center_x_ratio=x)


def sample(now, delta):
    return SimpleNamespace(timestamp=now, trustworthy=True, left_error=0,
        right_error=0, left_forward_rpm=84.+delta, right_forward_rpm=84.)


class ResponseCase:
    def __init__(self):
        self.response = ShortFollowYawResponse()
        self.receipt = SimpleNamespace(completed_at=10.)
        self.response.acknowledge(plan(10.), self.receipt, 100, 84)

    def tick(self, now, delta=-4., *, source=None, stamp=None, feedback=None,
             wheel_limit=200):
        source = source or plan(now)
        observed = feedback or sample(now if stamp is None else stamp, delta)
        result = self.response.adjust(source, observed, self.receipt, now,
                                      config(), wheel_limit=wheel_limit)
        assert result.expires_at == source.expires_at
        assert result.depth_timestamp == source.depth_timestamp
        assert result.capture_timestamp == source.capture_timestamp
        assert result.epoch == source.epoch
        assert result.left_rpm-result.right_rpm == source.left_rpm-source.right_rpm
        self.receipt = SimpleNamespace(completed_at=now)
        self.response.acknowledge(source, self.receipt, result.left_rpm, result.right_rpm)
        return result


@pytest.mark.parametrize("delta,reason,reduction", [
    (-4., "opposite_response_confirmed", 2),
    (2., "understeer_confirmed", 2),
    (16., "response_recovered", 0),
])
def test_fresh_positive_wheels_distinguish_opposite_weak_and_recovered(delta, reason, reduction):
    case = ResponseCase()
    outputs = [case.tick(now, delta) for now in (10.10, 10.18, 10.26)]
    assert [o.yaw_response_sample_count for o in outputs] == ([1, 2, 3] if reduction else [0, 0, 0])
    assert outputs[-1].yaw_response_reason == reason
    assert outputs[-1].yaw_common_reduction_rpm == reduction
    assert (outputs[-1].left_rpm, outputs[-1].right_rpm) == (100-reduction, 84-reduction)


def test_not_enough_command_exposure_is_not_reported_as_recovery():
    case = ResponseCase()
    for now in (10.02, 10.06, 10.09):
        output = case.tick(now)
        assert output.yaw_response_reason == "command_exposure_pending"
        assert output.yaw_response_sample_count == output.yaw_common_reduction_rpm == 0
    assert case.tick(10.10).yaw_response_sample_count == 1


def test_negative_differential_does_not_erase_two_weak_response_samples():
    case = ResponseCase()
    assert case.tick(10.10, 2).yaw_response_sample_count == 1
    assert case.tick(10.18, 2).yaw_response_sample_count == 2
    output = case.tick(10.26, -4)
    assert output.yaw_response_sample_count == 3
    assert output.yaw_response_reason == "opposite_response_confirmed"
    assert output.yaw_common_reduction_rpm == 2


def test_duplicate_opposite_feedback_never_confirms_or_deepens_reduction():
    case = ResponseCase()
    for now in (10.10, 10.14, 10.18, 10.22):
        result = case.tick(now, stamp=10.10)
        assert result.yaw_response_sample_count == 1
        assert result.yaw_common_reduction_rpm == 0
    case = ResponseCase()
    for now in (10.10, 10.18, 10.26):
        result = case.tick(now)
    repeated = case.tick(10.30, stamp=10.26)
    assert repeated.yaw_response_sample_count == 3
    assert repeated.yaw_common_reduction_rpm == 2


def test_three_rapid_opposite_samples_cannot_skip_minimum_confirmation_span():
    case = ResponseCase()
    for now in (10.10, 10.12, 10.14):
        assert case.tick(now).yaw_common_reduction_rpm == 0
    assert case.tick(10.26).yaw_common_reduction_rpm == 2


def test_real_recovery_unwinds_without_step_to_original_speed():
    case = ResponseCase()
    outputs = [case.tick(now) for now in (10.10, 10.18, 10.26, 10.34, 10.42, 10.50)]
    assert [o.yaw_common_reduction_rpm for o in outputs] == [0, 0, 2, 4, 6, 8]
    for now, expected in ((10.56, 6), (10.62, 4), (10.68, 2), (10.74, 0)):
        result = case.tick(now, 16)
        assert result.yaw_common_reduction_rpm == expected
        assert result.yaw_response_sample_count == 0
        assert result.yaw_response_reason == ("response_recovery" if expected else "response_recovered")


@pytest.mark.parametrize("fault", ["new_direction", "old_capture", "old_feedback", "old_epoch",
    "wrong_uid", "unknown_ack", "wheel_reverse", "wheel_error", "expired"])
def test_opposite_response_cannot_cross_old_or_unsafe_context(fault):
    case = ResponseCase()
    case.tick(10.10)
    case.tick(10.18)
    source, observed = plan(10.26), sample(10.26, -4)
    if fault == "new_direction":
        source = plan(10.26, pair=(84, 100), x=.1)
    elif fault == "old_capture":
        source = replace(source, yaw_capture_id=10100, yaw_capture_timestamp=10.06)
    elif fault == "old_feedback":
        observed.timestamp = 10.10
    elif fault == "old_epoch":
        source = replace(source, epoch=41)
    elif fault == "wrong_uid":
        source = replace(source, uid=2)
    elif fault == "unknown_ack":
        case.receipt = SimpleNamespace(completed_at=10.18)
    elif fault == "wheel_reverse":
        observed.left_forward_rpm = -4.
    elif fault == "wheel_error":
        observed.left_error = 1
    elif fault == "expired":
        source = replace(source, expires_at=10.25)
    result = case.tick(10.26, source=source, feedback=observed)
    assert result.yaw_response_sample_count == 0
    assert result.yaw_common_reduction_rpm == 0
    assert (result.left_rpm, result.right_rpm) == (source.left_rpm, source.right_rpm)


def test_changed_direction_must_build_new_exposure_and_confirmation():
    case = ResponseCase()
    case.tick(10.10)
    case.tick(10.18)
    # The old right command cannot count as exposure for a new left command.
    left_plan = lambda now: plan(now, pair=(84, 100), x=.1)
    switched = case.tick(10.26, 4, source=left_plan(10.26))
    assert switched.yaw_response_sample_count == 0
    pending = case.tick(10.34, 4, source=left_plan(10.34))
    assert pending.yaw_response_reason == "command_exposure_pending"
    for now, count in ((10.42, 1), (10.50, 2), (10.58, 3)):
        result = case.tick(now, 4, source=left_plan(now))
        assert result.yaw_response_sample_count == count
    assert result.yaw_common_reduction_rpm == 2
    assert result.left_rpm-result.right_rpm == -16


def test_actual_cap1476_to_1494_feedback_and_ack_timing_replay():
    # run_20261010_215423_80340_d3574af3, log 20736..21036.
    # Times relative to 40876: now reconstructed from CSV capture clock +
    # logged yaw age; depth/expiry from logged age (0.1ms precision). Feedback
    # and ACK spacing are retained, including the >150ms gap resetting count.
    # now, ACK, feedback, cap, yaw_cap, capture, depth, expiry, yaw_capture, x, L, R, delta
    rows = [
        (.527423016,.535991534,.500859554,1476,1476,.351105,.506123016,.851123016,.351105,.820242023,95,79,0),
        (.622044321,.632766161,.562422142,1476,1476,.351105,.534644321,.851144321,.351105,.820242023,97,81,0),
        (.739662513,.755079650,.661410883,1479,1479,.486410,.616962513,.966962513,.486410,.831616211,98,82,-4),
        (.876349759,.890035990,.844735340,1481,1481,.586207,.752949759,1.086249759,.586207,.841612577,99,83,-3),
        (.994454507,1.008008712,.941947651,1481,1485,.586207,.752954507,1.086254507,.787724,.866072893,99,83,-3),
        (1.068867900,1.091448088,1.060664105,1485,1485,.787724,1.015767900,1.287767900,.787724,.866072893,100,84,-4),
        (1.169005768,1.181340137,1.158670525,1488,1488,.952696,1.084605768,1.434605768,.952696,.890123892,100,84,-3),
        (1.229309555,1.244633432,1.210179907,1488,1488,.952696,1.118709555,1.452709555,.952696,.890123892,100,84,-5),
        (1.335774582,1.349951345,1.323120356,1488,1491,.952696,1.118674582,1.452674582,1.120329,.908681631,100,84,2),
        (1.400562340,1.411634973,1.371526190,1491,1491,1.120329,1.287562340,1.620362340,1.120329,.903257631,100,84,7),
        (1.505665503,1.526166406,1.446615268,1491,1494,1.120329,1.344265503,1.620365503,1.285437,.897364362,100,84,15),
        (1.612700415,1.627250883,1.554210167,1494,1494,1.285437,1.457100415,1.785400415,1.285437,.827234475,96,80,16),
    ]
    response, receipt = ShortFollowYawResponse(), None
    results = []
    for now, ack, stamp, cap, yaw_cap, capture, depth, expiry, yaw_capture, x, left, right, delta in rows:
        source = replace(plan(40876+now, cap=cap, pair=(left, right), x=x),
            capture_timestamp=40876+capture, depth_timestamp=40876+depth,
            expires_at=40876+expiry, yaw_capture_id=yaw_cap,
            yaw_capture_timestamp=40876+yaw_capture)
        result = response.adjust(source, sample(40876+stamp, delta), receipt,
                                 40876+now, config())
        results.append(result)
        assert (result.capture_timestamp, result.depth_timestamp, result.expires_at,
                result.epoch) == (source.capture_timestamp, source.depth_timestamp,
                                 source.expires_at, source.epoch)
        assert result.left_rpm-result.right_rpm == 16
        assert 0 < result.right_rpm <= source.right_rpm
        receipt = SimpleNamespace(completed_at=40876+ack)
        response.acknowledge(source, receipt, result.left_rpm, result.right_rpm)
    assert [r.yaw_common_reduction_rpm for r in results] == [0,0,0,0,0,2,4,6,8,6,4,2]
    assert [r.yaw_response_sample_count for r in results] == [0,0,1,1,2,3,4,5,6,0,0,0]
    assert results[1].yaw_response_reason == "command_exposure_pending"
    assert results[5].yaw_response_reason == "opposite_response_confirmed"


def test_real_writer_opposite_differential_reaches_fake_packets_without_stop(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    owner._short_follow = ShortFollowController(config())
    owner._short_follow.activate(1, clock[0])
    reductions = []
    for index, now in enumerate((10.01,10.11,10.19,10.27,10.35,10.43,10.51)):
        clock[0] = now
        source = publish(rt, clock, 1476+index, distance=2.74, x=.9)
        set_feedback(rt, clock, 0 if index == 0 else 80, 0 if index == 0 else 84)
        rt._service_short_follow()
        actual = owner._short_follow_last_applied_plan
        reductions.append(actual.yaw_common_reduction_rpm)
        assert actual.expires_at == source.expires_at and actual.epoch == source.epoch
        assert owner._short_follow.snapshot().plan is source
        assert actual.left_rpm-actual.right_rpm == 16
        assert 0 < actual.right_rpm < actual.left_rpm <= source.left_rpm
        assert driver.pairs[-1] == (actual.left_rpm, -actual.right_rpm)
    assert reductions == [0,0,0,2,4,6,8]
    assert not driver.stops and not owner._brake_hold_active
    # New response evidence never extends the original watchdog.
    clock[0] = source.expires_at+.001
    set_feedback(rt, clock, 80, 84)
    before = len(driver.pairs)
    rt._service_short_follow()
    assert len(driver.pairs) == before and driver.stops == [1]


@pytest.mark.parametrize("limit,expected_max", [(200,8), (8,5), (1,0), (0,0)])
def test_opposite_response_respects_common_reduction_and_final_rounding_bounds(limit, expected_max):
    case = ResponseCase()
    reductions = []
    for now in (10.10,10.18,10.26,10.34,10.42,10.50):
        # Keep the existing unscaled pair's differential; apply exactly the
        # writer's final hardware scale/round and never create a new zero.
        source = plan(now, pair=(23,7))
        output = case.tick(now, source=source, wheel_limit=limit)
        reductions.append(output.yaw_common_reduction_rpm)
        scale = min(1., limit/max(output.left_rpm, output.right_rpm))
        actual = round(output.left_rpm*scale), round(output.right_rpm*scale)
        assert max(actual) <= limit
        if limit >= 8:
            assert min(actual) >= 1
        else:
            baseline_scale = min(1., limit/max(source.left_rpm, source.right_rpm))
            assert actual == (round(source.left_rpm*baseline_scale), round(source.right_rpm*baseline_scale))
    # The 7RPM inner wheel permits only six RPM before final scaling.
    assert max(reductions) == min(expected_max, 6)
