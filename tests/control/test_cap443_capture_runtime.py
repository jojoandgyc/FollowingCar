"""Actual intent producer and parking diagnostics, mocked I/O only."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from test_lateral_zero_runtime import owner, _target, NOW
from car_control_modular.control_types import ControlAction
from car_control_modular.visual_steering_evidence import SteeringObservation
from car_control_modular.action_runtime import MotionActionRuntime
from car_control_modular.near_yaw_parking import NearYawParkRequest, ParkSettlingEvidence


@pytest.mark.parametrize("same_capture", [True,False])
def test_actual_publisher_keeps_corrected_geometry_and_provenance(owner,same_capture):
    owner._follow_controller.last_capture_steering_observation = SteeringObservation(
        1,576 if same_capture else 575,NOW-.09,.734375,.68,.68,-20.,"capture_rate_valid",.2)
    owner._follow_controller.last_steering_pid_result.target_image_rate_dps=-20.
    assert owner._publish_lateral_intent_from_decision(width=640,target=_target(),
        runtime_actions=[ControlAction.rotate_right("test")], control_source="vision",
        target_steerable=True,low_quality_visible=False)
    intent=owner._lateral_intent_store.snapshot()
    assert intent.x_ratio == (.68 if same_capture else .734375)
    assert intent.target_image_rate_dps == -20.
    assert intent.capture_frame_id == 576
    assert intent.capture_timestamp == NOW-.09


def test_stop_response_audit_does_not_release_or_refresh_motor_authority():
    e=ParkSettlingEvidence(NearYawParkRequest(1,447,9.9,9.98,"predictive"),10.)
    rt=SimpleNamespace(logger=Mock())
    log=MotionActionRuntime._log_near_yaw_stop_response
    log(rt,e)
    assert not rt.logger.info.called
    def fb(t): return SimpleNamespace(timestamp=t,trustworthy=True,left_forward_rpm=0,right_forward_rpm=0)
    e.observe(fb(10.02),10.02); e.observe(fb(10.08),10.08)
    log(rt,e); log(rt,e)
    assert rt.logger.info.call_count == 1
    assert rt.logger.info.call_args.args[-1] == pytest.approx(80.)
    assert e.sent_at == 10.
    assert not e.release_ready(9.99,fb(10.08),10.1)
