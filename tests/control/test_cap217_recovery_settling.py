"""CAP210--220: an execution recovery ramp is not a braking command.

CPU-only controller calls; no motor, serial or sensor access.
"""
from dataclasses import replace

import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.longitudinal_approach import RawDepthMotionEvidence


def controller():
    return DistancePiController(DistancePiConfig(
        kp_per_sec=3., physical_ttl_sec=.25, deceleration_m_s2=.7,
        launch_request_rpm=180., launch_full_error_m=.5,
        stationary_stop_preview_enabled=True))


def observe(c, stamp, age, ego, *, distance=2.421, target=.5):
    rate = None if ego is None else target-ego*.816814/60.
    return c.update(
        distance, 1.4, sample_timestamp=stamp, execution_now=stamp+age,
        deadband_m=.03, max_output_rpm=200., rise_rpm_per_sec=240.,
        ego_forward_rpm=ego, preview_outer_forward_rpm=ego,
        preview_feedback_timestamp=stamp+age-.02,
        raw_distance_m=distance, range_rate_m_s=rate,
        raw_closure_valid=ego is not None,
        raw_motion_evidence=(RawDepthMotionEvidence(stamp, rate, target, .1, 2)
                             if ego is not None else None))


@pytest.mark.parametrize('quantize', [False, True])
def test_feedback_gap_then_ramp_below_existing_wheel_speed_does_not_latch(quantize):
    c = controller()
    first = observe(c, 100., .02, None)
    assert first.output_rpm == 0
    recovery = observe(c, 100.1, .015, 52.)
    assert recovery.output_rpm == 22
    assert recovery.cap_rpm > 52.
    assert recovery.final_limit_reason == 'software_slew'
    if quantize:
        c.accept_output_limit(recovery.sample_timestamp, 21., quantization_rpm=2.)
    requests = []
    for stamp, age, ego, distance in [
        (100.2, .074, 41., 2.421),
        (100.3, .018, 32.5, 2.390),
        (100.4, .03, 29., 2.377),
    ]:
        r = observe(c, stamp, age, ego, distance=distance)
        assert not r.brake_settling_limited
        assert r.output_rpm <= r.cap_rpm
        assert r.output_rpm <= r.execution_anchor_rpm+240*r.execution_ramp_dt_sec
        requests.append(r.output_rpm)
    assert min(requests) > 22


def test_material_downstream_cut_still_requires_braking_to_settle():
    c = controller()
    observe(c, 100., .02, None)
    recovering = observe(c, 100.1, .015, 52.)
    assert recovering.output_rpm == 22
    # A real external cut is distinct from that 22RPM calculation itself.
    assert c.accept_output_limit(recovering.sample_timestamp, 10.)
    r = observe(c, 100.2, .074, 41., distance=2.40)
    assert r.brake_settling_limited
    assert r.output_rpm == 10


@pytest.mark.parametrize('distance,ego,target', [
    (1.39, 41., 0.), (1.50, 80., 0.), (1.8, 65., -.5),
])
def test_new_close_or_fast_approach_still_brakes(distance, ego, target):
    c = controller()
    observe(c, 100., .02, None)
    observe(c, 100.1, .015, 52.)
    r = observe(c, 100.2, .074, ego, distance=distance, target=target)
    assert r.output_rpm == 0


def test_default_removes_extra_headroom_not_physical_stop_budget():
    current = controller()
    extra = DistancePiController(replace(
        current.config, stationary_stop_preview_headroom_rpm=15.))
    plain = observe(current, 100., .06, 49., distance=2.42, target=.8)
    padded = observe(extra, 100., .06, 49., distance=2.42, target=.8)
    assert plain.stationary_preview_cap_rpm > padded.stationary_preview_cap_rpm
    assert plain.stationary_preview_cap_rpm-padded.stationary_preview_cap_rpm == pytest.approx(
        min(15., .2*plain.stationary_preview_cap_rpm))
    assert plain.output_rpm <= plain.stationary_preview_cap_rpm
    near = observe(current, 100.1, .03, 80., distance=1.7, target=.8)
    assert near.stationary_preview_status == 'momentum_brake'
    assert near.output_rpm == 0
