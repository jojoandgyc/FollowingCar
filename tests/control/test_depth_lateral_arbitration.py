#!/usr/bin/env python3
"""Regression tests for the split Depth30/visible-yaw authority."""

from __future__ import annotations

import threading
import time
import sys
from pathlib import Path
from types import SimpleNamespace

# Keep direct ``pytest tests/control/...`` invocation independent of the
# caller's PYTHONPATH; production imports are unchanged.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction, PersonTarget


def _tracker_stub(*, current_command=None, pending=()):
    tracker = object.__new__(runtime.PersonTracker)
    tracker.command_lock = threading.Lock()
    tracker.action_queue_lock = threading.Lock()
    tracker.action_queue = SimpleNamespace(queue=list(pending))
    tracker.current_command = current_command
    tracker.search_state = "none"
    tracker._vision_control_state = "target_visible_depth_valid"
    tracker._last_vision_control_ts = time.monotonic()
    tracker._last_depth30_translation_ts = time.monotonic()
    tracker._last_depth30_translation_kind = "forward"
    tracker._last_vision_correction_target_id = 1
    tracker._follow_controller = SimpleNamespace(
        cfg=SimpleNamespace(center_left_ratio=0.45, center_right_ratio=0.55),
        last_steering_pid_result=SimpleNamespace(correction_rpm=5),
    )
    return tracker


def test_recent_depth_translation_survives_short_visual_loss(monkeypatch):
    monkeypatch.setattr(runtime, "MODULE_ASTRA_DEPTH_ENABLE", True)
    tracker = _tracker_stub()
    assert tracker._should_preserve_depth30_translation_for_visual_loss(
        "lost_history_hold_left"
    )
    tracker.search_state = "searching"
    assert not tracker._should_preserve_depth30_translation_for_visual_loss(
        "lost_history_hold_left"
    )


def test_expired_depth_translation_does_not_block_visual_search(monkeypatch):
    monkeypatch.setattr(runtime, "MODULE_ASTRA_DEPTH_ENABLE", True)
    tracker = _tracker_stub()
    tracker._last_depth30_translation_ts = time.monotonic() - 1.0
    assert not tracker._should_preserve_depth30_translation_for_visual_loss(
        "lost_history_hold_left"
    )


def test_zero_depth_hold_preserves_active_visible_rotation(monkeypatch):
    monkeypatch.setattr(runtime, "MODULE_ASTRA_DEPTH_ENABLE", True)
    tracker = _tracker_stub(current_command=runtime.ACTION_ROTATE_RIGHT)
    target = PersonTarget((360.0, 20.0, 500.0, 460.0), 1, 0.9, 61600.0)
    assert tracker._depth30_should_preserve_lateral_motion(
        target,
        [ControlAction.forward(0, "longitudinal_distance_untrusted_hold")],
        width=640,
    )


def test_zero_depth_hold_preserves_pending_visible_rotation(monkeypatch):
    monkeypatch.setattr(runtime, "MODULE_ASTRA_DEPTH_ENABLE", True)
    tracker = _tracker_stub(pending=(runtime.ACTION_ROTATE_LEFT,))
    target = PersonTarget((120.0, 20.0, 260.0, 460.0), 1, 0.9, 61600.0)
    assert tracker._depth30_should_preserve_lateral_motion(
        target,
        [ControlAction.forward(0, "longitudinal_distance_untrusted_hold")],
        width=640,
    )


def test_positive_depth_follow_is_not_suppressed(monkeypatch):
    monkeypatch.setattr(runtime, "MODULE_ASTRA_DEPTH_ENABLE", True)
    tracker = _tracker_stub()
    target = PersonTarget((360.0, 20.0, 500.0, 460.0), 1, 0.9, 61600.0)
    assert not tracker._depth30_should_preserve_lateral_motion(
        target,
        [ControlAction.forward(25, "longitudinal_distance_pid")],
        width=640,
    )


def test_stale_visual_yaw_is_not_preserved_by_depth_hold(monkeypatch):
    monkeypatch.setattr(runtime, "MODULE_ASTRA_DEPTH_ENABLE", True)
    tracker = _tracker_stub(current_command=runtime.ACTION_ROTATE_RIGHT)
    tracker._last_vision_control_ts = time.monotonic() - 1.0
    target = PersonTarget((360.0, 20.0, 500.0, 460.0), 1, 0.9, 61600.0)
    assert not tracker._depth30_should_preserve_lateral_motion(
        target,
        [ControlAction.forward(0, "longitudinal_distance_untrusted_hold")],
        width=640,
    )


def test_settle_releases_on_consistent_outward_motion():
    from car_control_modular.control_types import SensorFrame, SteeringFeedback
    from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController

    controller = FollowSafetyController(
        FollowPolicyConfig(
            center_left_ratio=0.45,
            center_right_ratio=0.55,
            near_distance_settle_release_margin_ratio=0.03,
        )
    )
    controller._near_settle_target_id = 1
    controller._near_settle_until = time.monotonic() + 1.0
    target = PersonTarget((300.0, 20.0, 380.0, 460.0), 1, 0.9, 35200.0)
    moving_right = PersonTarget((340.0, 20.0, 420.0, 460.0), 1, 0.9, 35200.0)
    frame = SensorFrame(
        width=640,
        height=480,
        persons=[moving_right],
        steering_feedback=SteeringFeedback(
            timestamp=time.monotonic(), yaw_rate_right_dps=0.0, trustworthy=True
        ),
    )
    assert not controller._near_settle_hold_active(
        moving_right,
        frame,
        time.monotonic(),
        motion_dx_ratio=0.0625,
        target_image_rate_dps=8.0,
    )
