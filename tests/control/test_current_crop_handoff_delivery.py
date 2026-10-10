"""Current mapped crop is a lateral handoff, not a new hazard watermark."""
from dataclasses import replace

import pytest

from car_control_modular.control_types import SensorFrame, DistanceState
from car_control_modular.detector_identity_lease import (
    ValidatedVisualObservation, publish_visual_identity_evidence,
)
from car_control_modular.low_quality_lateral import LimitedYawSource
from test_short_follow_adapter import paired, owner, process
from test_lateral_zero_runtime import NOW


def crop(a):
    process(a)
    before = a.owner._short_follow.snapshot().plan
    publication = publish_visual_identity_evidence(a.owner, observation=False, lease=False)
    a.owner._limited_yaw_source = LimitedYawSource(
        1, 1, 577, NOW-.03, a.target.bbox, publication)
    frame = SensorFrame(width=640, height=480, persons=[a.target],
        capture_frame_id=577, capture_timestamp=NOW-.03)
    return before, frame


def handle(a, frame):
    return a.owner._short_follow_adapter.handle(frame, a.target, is_fresh_depth=False,
        control_source='vision', target_steerable=False, low_quality_visible=True, now=NOW)


def test_current_crop_retires_forward_without_invalidating_next_exposure(paired):
    a = paired
    before, frame = crop(a)
    floor = a.owner._short_follow._source_floor
    assert not handle(a, frame)
    state = a.owner._short_follow.snapshot()
    assert not state.active and state.plan is None and state.reason == 'lateral_handoff'
    assert a.owner._short_follow._source_floor == floor
    assert a.owner._short_follow._last_depth == before.depth_timestamp
    assert a.owner._validated_visual_observation is False

    # CAP118-equivalent exposed before the previous crop was processed, but
    # after all actual safety events. It must not wait for a third exposure.
    exposure, depth = NOW-.01, NOW-.005
    publication = ValidatedVisualObservation(1, 1, 578, exposure, NOW, exposure+.5, 'full')
    publish_visual_identity_evidence(a.owner, observation=publication, lease=None)
    new_frame = replace(frame, capture_frame_id=578, capture_timestamp=exposure,
        distance_m=2., distance_state=DistanceState(source='vision_depth',
            raw_distance_m=2., used_distance_m=2., sample_timestamp=depth))
    assert a.owner._short_follow_adapter.handle(new_frame, a.target, is_fresh_depth=True,
        control_source='vision', target_steerable=True, low_quality_visible=False, now=NOW)
    plan = a.owner._short_follow.snapshot().plan
    assert plan is not None and plan.forwarding
    assert plan.capture_timestamp == exposure < NOW
    assert plan.depth_timestamp == depth


@pytest.mark.parametrize('change', ['missing', 'uid', 'capture', 'timestamp', 'bbox',
    'publication', 'stale', 'search', 'stop', 'brake', 'hazard'])
def test_unqualified_or_hazard_crop_keeps_hard_revoke(paired, change):
    a = paired
    _, frame = crop(a)
    source = a.owner._limited_yaw_source
    if change == 'missing': a.owner._limited_yaw_source = None
    if change == 'uid': a.owner._limited_yaw_source = replace(source, uid=2)
    if change == 'capture': a.owner._limited_yaw_source = replace(source, capture=576)
    if change == 'timestamp': a.owner._limited_yaw_source = replace(source, timestamp=NOW-.02)
    if change == 'bbox': a.owner._limited_yaw_source = replace(source, bbox=(0., 0., 1., 1.))
    if change == 'publication': publish_visual_identity_evidence(a.owner, observation=False, lease=False)
    if change == 'stale':
        a.owner._limited_yaw_source = replace(source, timestamp=NOW-.6)
        frame = replace(frame, capture_timestamp=NOW-.6)
    if change == 'search': a.owner.search_state = 'searching'
    if change == 'stop': a.owner._explicit_stop_requested = True
    if change == 'brake': a.owner._brake_hold_active = True
    if change == 'hazard': frame = replace(frame, hazard=replace(frame.hazard, active=True))
    handled = handle(a, frame)
    if change == 'stop':
        assert handled  # paired explicit-stop owner must consume this frame
    else:
        assert not handled
    assert a.owner._short_follow.snapshot().reason != 'lateral_handoff'
    assert a.owner._short_follow._source_floor >= NOW


@pytest.mark.parametrize('event', ['revoke', 'deactivate', 'new_uid', 'new_plan'])
def test_crop_qualification_cannot_overwrite_intervening_state(paired, monkeypatch, event):
    a = paired
    _, frame = crop(a)
    adapter, controller = a.owner._short_follow_adapter, a.owner._short_follow
    changed = []
    original = adapter._current_lateral_handoff
    def qualify(*args):
        assert original(*args)
        if event == 'revoke': controller.revoke('identity_rejected', NOW)
        elif event == 'deactivate': controller.deactivate('identity_rejected', NOW)
        elif event == 'new_uid': controller.activate(2, NOW)
        else:
            state = controller.snapshot()
            controller._state = replace(state, plan=replace(state.plan, sequence=state.plan.sequence+1))
        changed.append(controller.snapshot())
        return True
    monkeypatch.setattr(adapter, '_current_lateral_handoff', qualify)
    assert handle(a, frame)  # consume the stale frame; no legacy yaw publication
    assert controller.snapshot() is changed[0]
    assert a.owner._limited_yaw_source is None
    assert a.owner._short_follow_handled_frame is True
