"""CAP825--957: tapered distance demand must survive an ordinary park preview."""
import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController


def controller():
    return DistancePiController(DistancePiConfig(
        kp_per_sec=3., launch_request_rpm=180., launch_full_error_m=.5,
        physical_ttl_sec=.25, motion_memory_sec=.35,
    ))


def update(c, stamp, *, distance=2.1858, now=None, rate=.2):
    return c.update(
        distance, 1.4, sample_timestamp=stamp,
        execution_now=stamp + .02 if now is None else now,
        deadband_m=.03, max_output_rpm=200., rise_rpm_per_sec=180.,
        ego_forward_rpm=0., range_rate_m_s=rate, raw_closure_valid=True,
        raw_distance_m=distance,
    )


def test_tapered_park_preview_has_brake_bounded_demand_without_execution_credit():
    c = controller()
    c.set_normal_parking(True)
    for index in range(161):
        stamp = 100. + index * .05
        c.suspend(stamp, "no_live_grant_before_pi", reset_execution=True)
        result = update(c, stamp)
        assert result.status == "parked_preview"
        assert result.output_rpm == 0
        assert 0 < result.unslewed_output_rpm <= result.cap_rpm
        assert result.unslewed_output_rpm <= result.demand_rpm
        assert result.sample_dt_sec == 0
        assert result.integral_m_s == 0
        assert not result.software_rise_bypassed
        assert c._last_execution_ts is None
        assert c._approved_rpm == 0


def test_release_rejects_preview_replay_and_restarts_ramp_on_new_samples():
    c = controller()
    c.set_normal_parking(True)
    preview = update(c, 100.)
    assert preview.unslewed_output_rpm > 0
    c.set_normal_parking(False)

    replay = update(c, 100., now=100.1)
    assert replay.status == "suspended_duplicate"
    assert replay.output_rpm == replay.unslewed_output_rpm == 0
    assert c._last_sample_ts == 100.
    assert c._last_execution_ts is None

    # Eight seconds of parking are not a forward acceleration budget.
    outputs = []
    for index in range(3):
        result = update(c, 108. + index * .05)
        outputs.append(result.output_rpm)
        assert result.output_rpm <= result.cap_rpm
        assert not result.software_rise_bypassed
        if index == 0:
            assert result.sample_dt_sec == 0
            assert result.integral_m_s == 0
    assert outputs == [0, 9, 18]


@pytest.mark.parametrize("release", [False, True])
@pytest.mark.parametrize("distance", [1.2, 1.4, 1.429])
def test_near_distance_cannot_unlock_forward_demand(release, distance):
    c = controller()
    c.set_normal_parking(True)
    update(c, 100.)
    if release:
        c.set_normal_parking(False)
    result = update(c, 100.05, distance=distance)
    assert result.output_rpm == result.unslewed_output_rpm == 0
    assert result.demand_rpm == 0


@pytest.mark.parametrize("release", [False, True])
@pytest.mark.parametrize("age,status", [(.181, "continuation_only"), (.251, "stale_sample")])
def test_late_depth_cannot_unlock_or_resume_forward_demand(release, age, status):
    c = controller()
    c.set_normal_parking(True)
    update(c, 100.)
    if release:
        c.set_normal_parking(False)
    result = update(c, 100.05, now=100.05 + age)
    assert result.status == status
    assert result.output_rpm == result.unslewed_output_rpm == 0
    assert c._last_sample_ts == 100.
    assert c._last_execution_ts is None
