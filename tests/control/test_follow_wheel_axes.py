from types import SimpleNamespace
import pytest

import request_0513_modular as runtime
from car_control_modular.lateral_intent import LateralControlIntent, LateralIntentStore


def owner(monkeypatch):
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 10.)
    monkeypatch.setattr(runtime, "MOTOR_FORWARD_MAX_TARGET_RPM", 100)
    o = object.__new__(runtime.PersonTracker)
    o._follow_controller = SimpleNamespace(active_target_id=1)
    o._lateral_yaw_revision = 1
    o._fresh_depth_linear_snapshot = lambda uid, now=None: ("forward", 40, uid, 9.9)
    o._lateral_intent_store = LateralIntentStore()
    o._lateral_intent_last_sequence = -1
    o._lateral_intent_last_correction_rpm = 0
    o._lateral_intent_zero_sequence = -1
    o.search_state = "none"
    o._vision_control_state = "target_visible_depth_valid"
    o._explicit_stop_requested = o._runtime_shutdown_requested = False
    o.running = True
    return o


def publish(o, **changes):
    args = dict(sequence=0, target_id=1, frame_index=1, published_at=9.9, valid_until=10.1,
                x_ratio=.6, motion_dx_ratio=0, target_image_rate_dps=0, mode="forward",
                base_percent=20, base_rpm=20, initial_correction_rpm=5, correction_limit_rpm=10,
                confidence=.9, bbox_quality="reliable", reason="test")
    return o._lateral_intent_store.publish(LateralControlIntent(**{**args, **changes}))


def test_real_reader_uses_current_depth_not_visual_base_and_signed_latest_yaw(monkeypatch):
    o = owner(monkeypatch)
    intent = publish(o)
    assert o._follow_wheel_axes(10.) == (1, 1, 40., 5.)
    o._lateral_intent_last_sequence = intent.sequence
    o._lateral_intent_last_correction_rpm = -6
    assert o._follow_wheel_axes(10.) == (1, 1, 40., -6.)
    o._fresh_depth_linear_snapshot = lambda uid, now=None: None
    assert o._follow_wheel_axes(10.) == (1, 1, 0., -6.)


@pytest.mark.parametrize("change", [{"hold_zero":True}, {"target_id":2}, {"valid_until":9.99}])
def test_zero_stale_or_other_uid_yaw_does_not_clear_depth(monkeypatch, change):
    o = owner(monkeypatch)
    publish(o, **change)
    assert o._follow_wheel_axes(10.) == (1, 1, 40., 0.)


def test_reverse_does_not_enter_normal_follow_scheduler(monkeypatch):
    o = owner(monkeypatch)
    o._fresh_depth_linear_snapshot = lambda uid, now=None: ("backward", 30, uid, 9.9)
    assert o._follow_wheel_axes(10.) is None
