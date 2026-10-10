"""Actual perception adapter and fake motor: expiry is not identity rejection."""
from dataclasses import replace

import pytest

from car_control_modular.control_types import SensorFrame, DistanceState, SteeringFeedback
from car_control_modular.detector_identity_lease import (
    ValidatedVisualObservation, publish_visual_identity_evidence,
)
from test_cap609_reacquire_depth_transfer import handoff, capture, queue_capture
from test_short_follow_adapter import paired, owner


def start(h):
    h.obj.search_state = h.obj._follow_controller.search_state = "none"
    capture(h, 1245)
    assert queue_capture(h)
    h.motor._service_short_follow()
    assert h.driver.pairs and not h.driver.stops
    return h.obj._short_follow.snapshot().plan


def missing(h, cap=1252, **changes):
    frame = SensorFrame(width=640, height=480, capture_frame_id=cap,
        capture_timestamp=h.clock[0]-.02)
    kw = dict(is_fresh_depth=False, control_source="vision", target_steerable=True,
              low_quality_visible=False, now=h.clock[0])
    kw.update(changes)
    return h.obj._short_follow_adapter.handle(frame, None, **kw)


def current_range(h, cap, exposure, *, stamp=None):
    now = h.clock[0]
    stamp = now-.005 if stamp is None else stamp
    target = replace(h.a.target, depth_observation=replace(h.a.target.depth_observation,
        capture_frame_id=cap, capture_timestamp=exposure))
    publish_visual_identity_evidence(h.obj, observation=ValidatedVisualObservation(
        1, 3, cap, exposure, now, exposure+.5, "full"), lease=None)
    frame = SensorFrame(width=640, height=480, persons=[target], distance_m=2.645,
        capture_frame_id=cap, capture_timestamp=exposure,
        distance_state=DistanceState(source="vision_depth", source_detail="depth_multi_region",
            raw_distance_m=2.645, used_distance_m=2.645, sample_timestamp=stamp))
    return h.obj._short_follow_adapter.handle(frame, target, is_fresh_depth=True,
        control_source="depth30", target_steerable=True, low_quality_visible=False, now=now)


def test_cap1254_pre_expiry_exposure_with_new_range_resumes_on_first_frame(handoff):
    h = handoff
    old = start(h)
    h.clock[0] = old.expires_at+.002
    retired_at = h.clock[0]
    exposure = retired_at-.006  # CAP1254 exposed just before CAP1252 was processed.
    floor = h.obj._short_follow._source_floor
    assert not missing(h)
    assert h.obj._short_follow.snapshot().reason == "observation_expired"
    assert h.obj._short_follow._source_floor == floor
    h.motor._service_short_follow()
    assert len(h.driver.stops) == 1  # Actual expiry still stops the old packet.
    h.clock[0] += .06
    assert current_range(h, 1254, exposure)
    plan = h.obj._short_follow.snapshot().plan
    assert plan is not None and plan.forwarding and plan.capture_timestamp < retired_at
    assert plan.expires_at == pytest.approx(min(plan.depth_timestamp+.3, exposure+.5))
    h.motor._service_short_follow()
    assert h.driver.pairs[-1] == (plan.left_rpm, -plan.right_rpm)
    assert len(h.driver.stops) == 1


@pytest.mark.parametrize("change", ["identity_rejected", "identity_expired", "search",
    "low_quality", "not_steerable", "different_uid"])
def test_real_rejection_does_not_borrow_ordinary_expiry_recovery(handoff, change):
    h = handoff
    old = start(h)
    h.clock[0] = old.expires_at+.002
    before = h.clock[0]-.006
    kw = {}
    if change == "identity_rejected":
        publish_visual_identity_evidence(h.obj, observation=False, lease=None)
    elif change == "identity_expired":
        h.clock[0] += .3
    elif change == "search": h.obj._follow_controller.search_state = "searching"
    elif change == "low_quality": kw["low_quality_visible"] = True
    elif change == "not_steerable": kw["target_steerable"] = False
    elif change == "different_uid": h.obj._follow_controller.active_target_id = 2
    assert not missing(h, **kw)
    assert h.obj._short_follow.snapshot().reason != "observation_expired"
    assert h.obj._short_follow._source_floor >= h.clock[0]
    h.obj._follow_controller.search_state = "none"
    h.obj._follow_controller.active_target_id = 1
    h.clock[0] += .01
    current_range(h, 1254, before)
    assert h.obj._short_follow.snapshot().plan is None


def test_new_capture_cannot_reuse_expired_depth(handoff):
    h = handoff
    old = start(h)
    h.clock[0] = old.expires_at+.002
    assert not missing(h)
    h.clock[0] += .05
    current_range(h, 1254, h.clock[0]-.01, stamp=old.depth_timestamp)
    assert h.obj._short_follow.snapshot().plan is None


def test_new_plan_already_published_wins_over_missing_frame(handoff):
    h = handoff
    old = start(h)
    h.clock[0] = old.expires_at+.002
    assert current_range(h, 1254, h.clock[0]-.03)
    current = h.obj._short_follow.snapshot()
    assert missing(h)
    assert h.obj._short_follow.snapshot() is current
    h.motor._service_short_follow()
    assert not h.driver.stops


def test_normal_expiry_does_not_erase_new_emergency(handoff):
    h = handoff
    old = start(h)
    h.clock[0] = old.expires_at+.002
    assert not missing(h)
    h.obj._short_follow.revoke("emergency", h.clock[0]+.01)
    h.clock[0] += .02
    current_range(h, 1254, h.clock[0]-.025)
    assert h.obj._short_follow.snapshot().plan is None
