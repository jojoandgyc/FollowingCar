"""Only taper steering; do not add yaw drive, timing authority or identity."""
from dataclasses import replace
import pytest
from test_cap443_capture_braking import observe, pid_config
from car_control_modular.visual_steering_evidence import CaptureSteeringEvidence
from car_control_modular.control_types import SteeringFeedback
from car_control_modular.steering_pid import VisualSteeringPid


def test_cap170_172_175_uses_last_consistent_interval_not_old_average():
    e=CaptureSteeringEvidence()
    observe(e,170,31546.372244,.7166)
    observe(e,172,31546.471340,.7042)
    r=observe(e,175,31546.635339,.6603)
    latest=(.6603-.7042)*60/(31546.635339-31546.471340)
    assert r.rate_dps == pytest.approx(latest)
    assert r.rate_dps < -16


@pytest.mark.parametrize("side",[-1,1])
def test_accelerating_yaw_uses_fresh_interval_for_braking_only(side):
    cfg=replace(pid_config(),predictive_brake_decel_dps2=60,
                predictive_brake_response_sec=.05)
    fb=SteeringFeedback(timestamp=10.,trustworthy=True,yaw_rate_right_dps=side*9.405,
                        raw_yaw_rate_right_dps=side*17.2425)
    # CAP209-like filtered 9.4, raw 17.2. Previously the lower filter won.
    result=VisualSteeringPid(cfg).update(.5+side*.25,0,fb,now=10.,visual_age_sec=.1289)
    expected=17.2425*(.1289+.05)+17.2425**2/(2*60)
    assert result.stopping_distance_deg == pytest.approx(expected)
    assert result.correction_rpm*side >= 0
    assert abs(result.correction_rpm) <= 10


@pytest.mark.parametrize("raw",[35,-9,float("nan")])
def test_disagreeing_or_invalid_feedback_does_not_gain_braking_authority(raw):
    cfg=replace(pid_config(),predictive_brake_decel_dps2=60)
    fb=SteeringFeedback(timestamp=10.,trustworthy=True,yaw_rate_right_dps=9,
                        raw_yaw_rate_right_dps=raw)
    r=VisualSteeringPid(cfg).update(.75,0,fb,now=10.,visual_age_sec=.12)
    assert not r.feedback_used and r.correction_rpm == 10
