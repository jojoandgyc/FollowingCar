"""Independent matching budget; no motor, camera or serial use."""
from dataclasses import replace

import pytest

from car_control_modular.longitudinal_feedforward import LongitudinalFeedforwardConfig, LongitudinalFeedforwardEstimator
from test_longitudinal_feedforward import update
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner, NOW
from test_longitudinal_authority_runtime import _commit


def estimator(**changes):
    return LongitudinalFeedforwardEstimator(LongitudinalFeedforwardConfig(
        max_tracking_base_rpm=80, target_speed_filter_alpha=1, **changes))


def warmed():
    e = estimator()
    for i in range(9):
        r = update(e, now=100+i*.1, distance=2, ego=80)
    assert r.target_rpm == pytest.approx(80)
    return e


def test_match_budget_is_independent_and_requires_three_samples_before_exceeding_40():
    e = estimator()
    values=[]
    for i in range(10):
        r=update(e, now=100+i*.1, distance=2, ego=80)
        values.append(r.target_rpm)
        assert r.target_rpm <= 80
        if i >= 2: assert r.target_rpm-values[-2] <= 8.00001
    assert values[:2] == [0,40]
    assert values[2] == pytest.approx(48)
    assert values[-1] == pytest.approx(80)
    assert r.unbounded_target_rpm == pytest.approx(80)
    assert r.matching_cap_rpm == 80


def test_high_estimate_is_still_capped_and_not_double_added():
    e = estimator()
    for i in range(12):
        r = update(e, now=100+i*.05, distance=2+i*.025, ego=60)
    assert r.target_speed_m_s == pytest.approx(1.1)
    assert r.target_rpm == 80
    assert r.status == 'ready_capped'
    # extra_ff remains diagnostic, never added again by the PID.
    assert r.feedforward_rpm == 60


def test_far_still_positive_nonclosing_match_drop_is_rate_limited():
    e = warmed()
    r = update(e, now=100.9, distance=2, ego=10)
    assert r.unbounded_target_rpm == pytest.approx(45)
    assert r.target_rpm == pytest.approx(72)
    assert r.matching_rate_limited
    r = update(e, now=101, distance=2, ego=10)
    assert r.target_rpm == pytest.approx(64)


@pytest.mark.parametrize('reason', ['near','stopped','hazard','yaw','expired','uid','jump'])
def test_smoothing_never_holds_positive_speed_against_safety_or_rejected_evidence(reason):
    e = warmed()
    args=dict(now=100.9,distance=2,ego=10)
    if reason=='near':
        # Seed nearby geometry independently (a 30cm jump is itself rejected).
        e=estimator()
        for i in range(9): update(e,now=100+i*.1,distance=1.65,ego=80)
        args['distance']=1.65
    elif reason=='stopped': args.update(distance=1.95,ego=0)
    elif reason=='hazard': args['trusted']=False
    elif reason=='yaw': args['yaw_rate_dps']=20
    elif reason=='expired': args['sample_timestamp']=100.5
    elif reason=='uid': args['target_id']=2
    elif reason=='jump': args['distance']=2.5
    r=update(e,**args)
    assert not r.matching_rate_limited
    if reason=='near': assert r.target_rpm < 50
    else: assert not r.eligible and r.target_rpm == 0


def test_reset_rebuild_and_duplicate_cannot_ramp_matching_speed():
    e=warmed()
    e.reset()
    update(e,now=101,distance=2,ego=80)
    r=update(e,now=101.1,distance=2,ego=80)
    assert r.target_rpm==40
    for _ in range(10):
        r=update(e,now=101.1,distance=2,ego=80)
        assert not r.eligible
    r=update(e,now=101.2,distance=2,ego=80)
    assert r.target_rpm==pytest.approx(48)


@pytest.mark.parametrize('limit',[20,60,80,200])
def test_matching_ceiling_cannot_exceed_encoder_validation_budget(limit):
    e=LongitudinalFeedforwardEstimator(LongitudinalFeedforwardConfig(max_tracking_base_rpm=limit))
    for i in range(20): r=update(e,now=100+i*.1,distance=2,ego=100)
    assert r.target_rpm <= min(limit,100)


@pytest.mark.parametrize('changes', [{'max_tracking_base_rpm':-1}, {'max_tracking_base_rpm':float('nan')},
    {'tracking_rise_rpm_per_sec':0}, {'tracking_fall_rpm_per_sec':-1}])
def test_invalid_new_configuration_fails_closed(changes):
    e=LongitudinalFeedforwardEstimator(replace(LongitudinalFeedforwardConfig(), **changes))
    assert update(e).status == 'invalid_config'


def test_controller_wires_independent_budget_without_exceeding_total(setup):
    from car_control_modular.controllers import FollowSafetyController
    _,c,_=setup
    c=FollowSafetyController(replace(c.cfg,distance_matching_base_max_rpm=80,forward_max_rpm=60))
    assert c._longitudinal_feedforward.config.max_tracking_base_rpm == 60


@pytest.mark.parametrize('distance,base,requested,expected', [
    (1.5, 80, 60, 40),    # match 80 RPM, no catch-up inside deadband
    (1.9, 80, 60, 50),    # match 80 + bounded correction 20 RPM
    (1.9, 48, 60, 34),    # warming/ramped estimate, not full configured budget
    (1.5, None, 60, 10),  # no qualified estimate: ordinary near limit unchanged
    (1.5, 80, 0, 0),      # explicit stop is never raised to matching speed
])
def test_runtime_does_not_recap_matching_base_at_legacy_40(monkeypatch, distance, base, requested, expected):
    import request_0513_modular as runtime
    from car_control_modular.control_types import ControlAction
    for name,value in dict(FORWARD_MAX_RPM=200, DISTANCE_MATCHING_BASE_MAX_RPM=80,
        ASTRA_DEPTH_LONGITUDINAL_MAX_FORWARD_PERCENT=10,
        ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT=100,
        ASTRA_DEPTH_NEAR_GUARD_DISTANCE_M=1.8,
        ASTRA_DEPTH_LONGITUDINAL_FAR_DISTANCE_M=2.2,
        TARGET_DISTANCE=1.5, DISTANCE_PID_DEADBAND_M=.03).items():
        monkeypatch.setattr(runtime,name,value)
    result=runtime.PersonTracker._cap_depth_longitudinal_actions(
        [ControlAction.forward(requested,'test')],distance,base)
    assert result[0].speed_percent == expected
    reverse=runtime.PersonTracker._cap_depth_longitudinal_actions(
        [ControlAction.backward(60,'test')],distance,base)
    without=runtime.PersonTracker._cap_depth_longitudinal_actions(
        [ControlAction.backward(60,'test')],distance,None)
    assert reverse[0].speed_percent == without[0].speed_percent


@pytest.mark.parametrize('same_sample',[True,False])
def test_pid_limit_audit_does_not_attribute_old_pid_to_new_depth(owner,caplog,same_sample):
    from types import SimpleNamespace
    owner._follow_controller.last_distance_pid_result=SimpleNamespace(output_rpm=90)
    owner._follow_controller._distance_pid_last_sample_timestamp=NOW-(.04 if same_sample else .1)
    _commit(owner,stamp=NOW-.04,percent=17)
    assert ('pid_requested_rpm=90.0' if same_sample else 'pid_requested_rpm=None') in caplog.text
