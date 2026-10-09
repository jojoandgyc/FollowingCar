"""CAP353: stale forward intent must not override the 7 RPM pivot policy."""
from types import SimpleNamespace

import pytest

from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.steering_limits import effective_correction_limit
from test_lateral_zero_runtime import owner, _intent, NOW, runtime


def config():
    return FollowPolicyConfig(visible_steering_pid_enable=True,
        visible_steering_pid_image_error_only=True,
        visible_steering_pid_max_correction_rpm=10,
        near_distance_rotation_only_max_rpm=7)


@pytest.mark.parametrize("x", [.17, .83])
def test_camera_and_refresh_share_actual_mode_ceiling(x):
    c = FollowSafetyController(config())
    for base, near, expected in [(50, False, 10), (0, False, 7), (17, True, 7)]:
        first = c._update_lateral_pid(c._visual_steering_pid, x_ratio=x,
            base_rpm=base, feedback=None, now=10, near_distance_mode=near)
        assert abs(first.correction_rpm) <= expected
        assert first.correction_policy_limit_rpm == expected
    assert effective_correction_limit(c.cfg, 0, policy_limit=5) == 5


@pytest.mark.parametrize("sign", [-1, 1])
def test_stale_forward_intent_clamped_on_first_and_refreshed_tick(owner, monkeypatch, sign):
    owner._follow_controller.cfg = config()
    _intent(owner, mode="forward", base_rpm=17, base_percent=9,
            initial_correction_rpm=sign*15, correction_limit_rpm=15,
            near_distance_mode=False, x_ratio=.5+sign*.3)
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: None)
    owner._lateral_intent_last_correction_rpm = sign*15
    calls = []
    result = owner._follow_controller.last_steering_pid_result
    def parked(**kwargs):
        calls.append(kwargs)
        result.correction_rpm = sign*7
        return result
    owner._follow_controller.refresh_parked_lateral_pid = parked
    owner._follow_controller.refresh_visible_lateral_pid = lambda **kw: pytest.fail("stale forward mode used")
    owner._service_lateral_intent(NOW)
    assert owner._last_vision_correction_rpm == sign*7
    owner._service_lateral_intent(NOW+.04)
    assert calls[-1]["base_rpm"] == 0
    assert calls[-1]["max_correction_rpm"] == 7
    assert calls[-1]["near_distance_mode"] is True
    assert owner._last_vision_correction_rpm == sign*7
    # Fresh translation may resume; the old frame is never renewed.
    expiry = owner._lateral_intent_store.snapshot().valid_until
    monkeypatch.setattr(runtime.PersonTracker, "_fresh_depth_linear_snapshot",
                        lambda self, uid, now=None: ("forward", 20, uid, NOW))
    def forward(**kwargs):
        calls.append(kwargs)
        result.correction_rpm = sign*10
        return result
    owner._follow_controller.refresh_visible_lateral_pid = forward
    owner._service_lateral_intent(NOW+.08)
    assert calls[-1]["max_correction_rpm"] == 10
    assert 0 < sign*owner._last_vision_correction_rpm <= 10
    assert owner._lateral_intent_store.snapshot().valid_until == expiry
