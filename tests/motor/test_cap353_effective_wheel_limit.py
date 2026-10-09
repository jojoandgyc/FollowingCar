"""Final fake motor writer clamps stale producers and temporary boost alike."""
from types import SimpleNamespace
from dataclasses import replace

import pytest

from test_cap1663_turn_response import prepare
from test_visible_wheel_continuity import feedback


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("base,near,policy,expected", [
    (42, False, 15, 10), (0, False, 15, 7),
    (42, True, 15, 7), (42, False, 5, 5),
])
def test_all_producers_and_boost_obey_final_cap(monkeypatch, sign, base, near, policy, expected):
    r, o, d, _, clock = prepare(monkeypatch)
    o._follow_controller.cfg = SimpleNamespace(
        visible_steering_pid_image_error_only=True,
        visible_steering_pid_max_correction_rpm=10,
        near_distance_rotation_only_max_rpm=7)
    original = o._lateral_intent_store.snapshot()
    o._lateral_intent_store.publish(replace(original,
        x_ratio=.5+sign*.15, target_image_rate_dps=sign*40., visual_error_deg=sign*9.9,
        near_distance_mode=near, correction_limit_rpm=policy))
    o._fresh_depth_linear_snapshot = lambda uid, now=None: ("forward", 21) if base else None
    r.get_steering_feedback = lambda: feedback(clock[0], 0, 0)
    for i in range(3):
        clock[0] = 10 + i*.05
        r._send_follow_wheel_targets(base+sign*15, -(base-sign*15), "FOLLOW20")
        left, raw_right = d.pairs[-1]
        assert left+raw_right == sign*2*expected
        assert (left-raw_right)/2 == base
    assert not d.stops
    # Cancelling yaw must not be undone by a cap or boost; no live Depth
    # means it cannot manufacture forward travel either.
    o._has_fresh_lateral_yaw = lambda uid: False
    o._fresh_depth_linear_snapshot = lambda uid, now=None: None
    r._send_follow_wheel_targets(base+sign*15, -(base-sign*15), "FOLLOW20")
    assert d.pairs[-1] == (0, 0)
