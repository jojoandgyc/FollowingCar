"""NORMAL yaw parking allows passive latest ranging, not old motor grants."""
from car_control_modular.near_yaw_parking import NearYawParkRequest
from test_depth_priority_admission import visible, eligible, context_owner, NOW


def test_typed_yaw_parking_can_measure_latest_depth_without_motion_grant(visible):
    visible._near_yaw_park_request = NearYawParkRequest(1, 1000, NOW-.15, NOW-.10, "center_hold")
    visible._brake_hold_active = True
    visible._brake_hold_label = "near_yaw_park"
    visible._brake_hold_stop_mode = "normal"
    assert eligible(visible)
    assert visible._brake_hold_active
    assert visible._fresh_depth_linear_snapshot(1, now=NOW) is None


def test_safety_hold_cannot_borrow_yaw_parking_measurement_exception(visible):
    visible._near_yaw_park_request = NearYawParkRequest(1, 1000, NOW-.15, NOW-.10, "center_hold")
    visible._brake_hold_active = True
    visible._brake_hold_label = "safety_hold_front_ir"
    visible._brake_hold_stop_mode = "emergency"
    assert not eligible(visible)
