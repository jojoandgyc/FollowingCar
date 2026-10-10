"""CAP1457 geometry through search confirmation, real depth and paired writer.

Image/range and serial are in-memory fixtures. No physical trajectory claimed.
"""
import pytest

from car_control_modular.control_types import SteeringFeedback
from car_control_modular.detector_identity_lease import publish_visual_identity_evidence
from test_cap1542_reacquire_handoff import completed_candidate, prepare_consumer
from test_cap609_reacquire_depth_transfer import handoff, capture, queue_capture
from test_short_follow_adapter import paired, owner
from test_search_observation_arbitration import _record


@pytest.mark.parametrize("mirror", [False, True])
def test_confirmed_cross_center_reaches_paired_yaw_then_forward_without_park(handoff, monkeypatch, mirror):
    h = handoff
    bbox = (346.6, 4.4, 486.4, 473.1)
    if mirror:
        bbox = (640-bbox[2], bbox[1], 640-bbox[0], bbox[3])
    candidate = completed_candidate(h, bbox=bbox)
    direction = 1 if mirror else -1
    side = "right" if mirror else "left"
    h.obj.search_direction = h.obj._follow_controller.search_direction = side
    # Use a supported first range (not a far-distance three-frame jump) to
    # isolate the parking handoff. Invalid/unconfirmed depth remains separate.
    h.camera._latest_depth[:] = 1800
    h.motor.backend.send_targets(7*direction, 7*direction, "TURN", history_uid=1)
    h.motor.get_steering_feedback = lambda: SteeringFeedback(
        timestamp=h.clock[0], trustworthy=True, left_forward_rpm=4*direction,
        right_forward_rpm=-6*direction, raw_yaw_rate_right_dps=15*direction,
        yaw_rate_right_dps=15*direction)
    assert not h.obj._hold_for_confirmed_search_reacquire([candidate], width=640, height=480)
    assert not getattr(h.motor, "_search_reacquire_brake_request", None)
    assert not getattr(h.obj, "_brake_hold_active", False)
    assert h.obj.search_state == h.obj._follow_controller.search_state == "none"
    assert h.obj._reacquire_depth_pending
    assert queue_capture(h)
    first = h.obj._short_follow.snapshot().plan
    assert first is not None and first.moving and not first.forwarding
    assert first.left_rpm * direction < 0  # Turn toward current target, not old search.
    h.motor._service_short_follow()
    assert not h.driver.stops and (0, 0) not in h.driver.pairs
    assert h.driver.pairs[-1] == (first.left_rpm, -first.right_rpm)

    h.clock[0] += .07
    capture(h, 1544)
    publish_visual_identity_evidence(h.obj, observation=h.obj._validated_visual_observation, lease=None)
    # The next confirmed visual frame clears the real one-frame longitudinal
    # veto before delivering its new range. No manual search/permission reset.
    prepare_consumer(h, dict(uid=1, mapped_uid=1, reason="mapped_similar_follow",
        bbox_quality_ok=True, reacquire_geometry_ok=True), monkeypatch)
    h.obj._consume_track_records([_record(track=3, uid=1, bbox=bbox, score=.90)],
        640, 480, "paired_takeover_test")
    second = h.obj._short_follow.snapshot().plan
    assert second is not None and second.forwarding
    h.motor.get_steering_feedback = lambda: SteeringFeedback(
        timestamp=h.clock[0], trustworthy=True, left_forward_rpm=1, right_forward_rpm=1)
    h.motor._service_short_follow()
    assert not h.driver.stops and (0, 0) not in h.driver.pairs
    assert h.driver.pairs[-1] == (second.left_rpm, -second.right_rpm)
    assert min(second.left_rpm, second.right_rpm) > 0
