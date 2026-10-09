"""CAP754..769 bank -> identity publication -> depth ROI -> paired writer.

Recorded crop geometry, times and full-body distances drive the real bank;
embeddings and fresh depth values are synthetic, not an RKNN/video replay.
The real adapter/controller/executor end at a fake serial driver. No camera,
serial device, or motor is opened by this regression.
"""
from copy import deepcopy
from pathlib import Path
from types import MethodType, SimpleNamespace
import sys

import pytest

_TESTS = Path(__file__).resolve().parents[1]
for _part in ("motor", "vision"):
    sys.path.insert(0, str(_TESTS / _part))

import request_0513_modular as main
from car_control_modular.control_types import DistanceState, PersonTarget, SensorFrame
from car_control_modular.depth_target_geometry import resolve_depth_target_observation
from car_control_modular.detector_identity_lease import ValidatedVisualObservation
from car_control_modular.short_follow import ShortFollowConfig, ShortFollowController
from car_control_modular.short_follow_adapter import ShortFollowAdapter
from rk_vision.tracker import TrackRecord
from test_cap761_crop_continuity import (
    ANCHOR_BOX, ANCHOR_CAP, ANCHOR_TS, CROP_SAMPLES, meta, seeded_bank,
    submit_crop,
)
from test_short_follow_executor import short_runtime, set_feedback
from test_search_observation_arbitration import owner as classification_owner


def _record(uid, bbox):
    x1, y1, x2, y2 = bbox
    return TrackRecord(track_id=3, reid_uid=uid, x1=x1, y1=y1, x2=x2, y2=y2,
        class_id=0, score=.94, cx=(x1+x2)/2, cy=(y1+y2)/2,
        area=(x2-x1)*(y2-y1), angle_deg=0., tracker_state=2, time_since_update=0)


def _chain(monkeypatch):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    clock[0] = ANCHOR_TS+.18
    bank = seeded_bank()
    owner._short_follow = ShortFollowController(ShortFollowConfig(
        enabled=True, depth_ttl_sec=.35, yaw_max_delta_rpm=18))
    owner._short_follow.activate(1, ANCHOR_TS-.1)
    owner._short_follow_adapter = ShortFollowAdapter(owner, owner._short_follow, main.logger)
    owner._action_runtime = rt
    owner._rknn_pipeline = SimpleNamespace(tracker=SimpleNamespace(identity_bank=bank),
        last_frame_width=640, last_identity_processing={})
    owner._identity_assignment_debug_for_track = lambda track: dict(bank.last_assignments.get(track, {}))
    owner._update_detector_identity_lease = MethodType(main.PersonTracker._update_detector_identity_lease, owner)
    owner._validated_visual_observation = None
    owner._detector_identity_lease = None
    owner._publish_follow_recording = lambda *_: None
    chain = SimpleNamespace(rt=rt, owner=owner, driver=driver, clock=clock, bank=bank,
        adapter=owner._short_follow_adapter, last_frame=None, last_target=None)
    metadata = meta(ANCHOR_CAP, ANCHOR_TS, 332, ANCHOR_BOX, track_id=3)
    _deliver(chain, 1, metadata)
    assert driver.pairs and not driver.stops
    return chain


def _advance(chain, until):
    """Service cached plans between captures without renewing their clocks."""
    while chain.clock[0]+.05 < until-1e-9:
        chain.clock[0] += .05
        left, raw_right = chain.driver.pairs[-1]
        set_feedback(chain.rt, chain.clock, left, -raw_right)
        assert chain.rt._service_short_follow()
    chain.clock[0] = until
    left, raw_right = chain.driver.pairs[-1]
    set_feedback(chain.rt, chain.clock, left, -raw_right)


def _deliver(chain, uid, metadata, *, fresh_depth=True, prior_depth=None):
    owner, now = chain.owner, chain.clock[0]
    cap, stamp = metadata["capture_frame_id"], metadata["capture_timestamp"]
    box = tuple(metadata["detector_bbox"])
    owner.frame_index = metadata["frame_index"]
    owner._active_capture_frame_id, owner._active_capture_timestamp = cap, stamp
    owner._rknn_pipeline.last_identity_processing = dict(mode="full", full_features_current=True,
        capture_frame_id=cap, capture_timestamp=stamp)
    record = _record(uid, box)
    assert owner._update_detector_identity_lease([record], cap, stamp, now=now, stale=False)
    if uid == 0:
        assert owner._validated_visual_observation is False
        chain.rt._service_short_follow()
        return None
    proof = owner._validated_visual_observation
    assert isinstance(proof, ValidatedVisualObservation) and proof.capture == cap
    assert proof.continuation_sample_timestamp is None
    # Resolve the production UID/detector association. A mapped-only UID0
    # cannot manufacture a normal depth target through this resolver.
    metadata = dict(metadata, source_detection_index=0)
    row = dict(raw_track_id=3, uid=uid, detector_bbox=box, display_bbox=box,
        sample_metadata=metadata, assignment=chain.bank.last_assignments[3])
    geometry = resolve_depth_target_observation(target_id=1, display_bbox=box,
        capture_frame_id=cap, capture_timestamp=stamp, observations=[row],
        width=640, height=480)
    assert geometry is not None and geometry.source == "yolo_detector"
    target = PersonTarget(box, uid, .94, record.area, depth_observation=geometry)
    depth_stamp = stamp+.10 if prior_depth is None else prior_depth
    state = DistanceState(source="vision_depth", source_detail="depth_multiregion",
        raw_distance_m=1.95, used_distance_m=1.95, sample_timestamp=depth_stamp,
        temporal_status="new_sample" if fresh_depth else "duplicate")
    frame = SensorFrame(width=640, height=480, persons=[target], distance_m=1.95,
        distance_state=state, capture_frame_id=cap, capture_timestamp=stamp)
    normal_quality = main.PersonTracker._bound_reid_bbox_allowed_for_control(
        chain.bank.last_assignments[3])
    assert normal_quality
    assert chain.adapter.handle(frame, target, is_fresh_depth=fresh_depth,
        control_source="depth30", target_steerable=normal_quality,
        low_quality_visible=not normal_quality, now=now)
    chain.last_frame, chain.last_target = frame, target
    assert chain.rt._service_short_follow()
    return owner._short_follow.snapshot().plan


def _run_crops(chain):
    for frame, sample in enumerate(CROP_SAMPLES, 333):
        _advance(chain, sample[1]+.18)
        uid, metadata = submit_crop(chain.bank, sample, frame)
        assert uid == 1
        _deliver(chain, uid, metadata)


def test_real_cap761_identity_to_writer_chain_never_inserts_zero(monkeypatch):
    chain = _chain(monkeypatch)
    entry = chain.bank.identities[1]
    anchor = deepcopy(entry.last_strong_observation)
    templates = deepcopy(entry.feature_metadata)
    original_epoch = chain.owner._short_follow.snapshot().epoch
    _run_crops(chain)
    assert chain.owner._short_follow.snapshot().epoch == original_epoch
    assert chain.adapter.owned and chain.rt._short_follow_executor.blocks_legacy()
    assert not chain.driver.stops
    assert len(chain.driver.pairs) > len(CROP_SAMPLES)
    assert all(0 < left < -right and 0 < (-right-left) <= 18
               for left, right in chain.driver.pairs)
    assert entry.last_strong_observation == anchor
    assert entry.feature_metadata == templates
    assert chain.owner._short_follow.snapshot().plan.capture_id == 769


def test_real_main_record_consumer_keeps_accepted_crop_on_normal_path(classification_owner, monkeypatch):
    """Exercise the actual normal/low-quality branch before the adapter.

    Raw metadata is still weak/three-edge; only the bank's newly checked
    mapped identity qualifies it. Do not fake steerability from a UID alone.
    """
    owner = classification_owner
    bank = seeded_bank()
    owner._follow_controller.active_target_id = 1
    owner._follow_controller.search_state = owner.search_state = "none"
    owner._vision_control_state = "target_visible_depth_valid"
    owner._rknn_pipeline = SimpleNamespace(tracker=SimpleNamespace(identity_bank=bank))
    for frame, sample in enumerate(CROP_SAMPLES, 333):
        uid, metadata = submit_crop(bank, sample, frame)
        assert uid == 1 and metadata["quality_bbox_ok"] is False
        assert metadata["detector_edge_touch_count"] == 3
        owner.frame_index = frame
        owner._active_capture_frame_id, owner._active_capture_timestamp = sample[:2]
        monkeypatch.setattr(main.time, "monotonic", lambda stamp=sample[1]: stamp+.18)
        owner._assignments = bank.last_assignments
        owner._events.clear()
        owner._consume_track_records([_record(uid, sample[2])], 640, 480, "offline_crop")
        assert len(owner._events) == 1
        kind, persons, options = owner._events[0]
        assert kind == "normal" and len(persons) == 1 and persons[0][1] == 1
        assert options["low_quality_visible"] is False
        assert options.get("target_steerable", True) is True


def test_crop_window_does_not_extend_motor_depth_deadline(monkeypatch):
    chain = _chain(monkeypatch)
    _run_crops(chain)
    plan = chain.owner._short_follow.snapshot().plan
    assert plan.expires_at == pytest.approx(plan.depth_timestamp+.35)
    chain.clock[0] = plan.expires_at+.001
    set_feedback(chain.rt, chain.clock, 20, 30)
    proof = chain.owner._validated_visual_observation
    assert proof.live(1, chain.clock[0])  # Identity still live; depth is not.
    count = len(chain.driver.pairs)
    assert chain.rt._service_short_follow()
    assert len(chain.driver.pairs) == count and chain.driver.stops == [1]
    assert chain.rt._short_follow_executor._stop_key[1] == "observation_expired"


def test_new_accepted_crop_with_duplicate_depth_cannot_renew_pair(monkeypatch):
    chain = _chain(monkeypatch)
    _run_crops(chain)
    plan = chain.owner._short_follow.snapshot().plan
    sample = (770, ANCHOR_TS+.90, CROP_SAMPLES[-1][2], .10)
    _advance(chain, sample[1]+.18)
    uid, metadata = submit_crop(chain.bank, sample, 338)
    assert uid == 1
    assert _deliver(chain, uid, metadata, fresh_depth=False,
                    prior_depth=plan.depth_timestamp) is plan
    assert chain.owner._validated_visual_observation.capture == 770
    assert chain.owner._short_follow.snapshot().plan.expires_at == plan.expires_at
    chain.clock[0] = plan.expires_at+.001
    set_feedback(chain.rt, chain.clock, 20, 30)
    chain.rt._service_short_follow()
    assert chain.driver.stops == [1]


@pytest.mark.parametrize("veto", ["appearance", "geometry", "expired_origin"])
def test_real_bank_rejection_still_cancels_latest_paired_motion(monkeypatch, veto):
    chain = _chain(monkeypatch)
    _run_crops(chain)
    changes = {}
    sample = (770, ANCHOR_TS+.95, CROP_SAMPLES[-1][2], .10)
    if veto == "appearance":
        sample = (*sample[:3], .201)
    elif veto == "geometry":
        changes["quality_bbox_reason"] = "identity_center_jump>0.4"
    else:
        sample = (770, ANCHOR_TS+1.001, sample[2], .10)
    # The previous plan must still be live when the new identity verdict
    # arrives, otherwise a TTL stop could falsely satisfy this test.
    _advance(chain, sample[1]+.03)
    assert chain.owner._short_follow.snapshot().plan.valid(chain.clock[0])
    uid, metadata = submit_crop(chain.bank, sample, 338, **changes)
    assert uid == 0
    before = len(chain.driver.pairs)
    _deliver(chain, uid, metadata)
    assert len(chain.driver.pairs) == before and chain.driver.stops == [1]
    assert chain.owner._short_follow.snapshot().plan is None
    assert chain.rt._short_follow_executor._stop_key[1] == "identity_not_live"
