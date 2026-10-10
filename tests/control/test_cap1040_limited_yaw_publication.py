"""Carry existing crop admission, not generic rejected UID, into yaw output."""
from dataclasses import replace

import pytest
from car_control_modular.control_types import ControlAction, PersonTarget
from car_control_modular.detector_identity_lease import publish_visual_identity_evidence
from car_control_modular.low_quality_lateral import LimitedYawSource, limited_yaw_identity_live
from test_lateral_zero_runtime import owner, NOW


def prepare(owner):
    owner._vision_control_state = "target_visible_low_quality"
    owner._last_control_decision_reason = "target_visible_low_quality_yaw"
    owner._follow_controller.search_state = "none"
    owner._active_capture_timestamp = owner._last_command_capture_timestamp
    bbox = (0., 0., 198., 479.)
    publication = publish_visual_identity_evidence(owner, observation=False, lease=False)
    source = LimitedYawSource(1, 16, 576, owner._active_capture_timestamp, bbox, publication)
    owner._limited_yaw_source = source
    target = PersonTarget(bbox, 1, .885, 94842.)
    return target, source


def publish(owner, target):
    return owner._publish_lateral_intent_from_decision(
        width=640, target=target, runtime_actions=[ControlAction.rotate_left("crop")],
        control_source="vision", target_steerable=False, low_quality_visible=True)


def test_real_lateral_publisher_transfers_same_frame_crop_admission(owner):
    target, source = prepare(owner)
    assert publish(owner, target)
    intent = owner._lateral_intent_store.snapshot()
    assert limited_yaw_identity_live(owner, 1, NOW, intent)
    assert owner._has_fresh_lateral_yaw(1)
    assert owner._limited_yaw_evidence.source is source
    assert owner._detector_identity_lease is False  # no forward/identity upgrade


@pytest.mark.parametrize("mismatch", ["no_source", "capture", "timestamp", "bbox", "uid", "publication"])
def test_rejected_frame_cannot_borrow_another_crop_admission(owner, mismatch):
    target, source = prepare(owner)
    if mismatch == "no_source":
        owner._limited_yaw_source = None
    elif mismatch == "publication":
        publish_visual_identity_evidence(owner, observation=False, lease=False)
    else:
        values = dict(capture=575, timestamp=NOW-.11, bbox=(1., 0., 199., 479.), uid=2)
        owner._limited_yaw_source = replace(source, **{mismatch: values[mismatch]})
    assert publish(owner, target)
    assert not limited_yaw_identity_live(owner, 1, NOW)
    assert not owner._has_fresh_lateral_yaw(1)


def test_new_rejection_invalidates_even_same_false_lease_value(owner):
    target, _ = prepare(owner)
    assert publish(owner, target)
    assert owner._has_fresh_lateral_yaw(1)
    publish_visual_identity_evidence(owner, observation=False, lease=False)
    assert not owner._has_fresh_lateral_yaw(1)
