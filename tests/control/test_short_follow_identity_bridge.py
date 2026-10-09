"""Keep only a pre-existing paired plan during qualified UID0 rechecks.

The real identity-lease updater and record consumer run here; no hardware or
new identity/depth authorization is provided by a grey candidate.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.detector_identity_lease import ValidatedVisualObservation
from car_control_modular.short_follow import (
    ShortFollowConfig, ShortFollowController, ShortFollowObservation,
)
from car_control_modular.short_follow_adapter import ShortFollowAdapter
from test_identity_recheck_execution_bridge import confirmed_then_pending
from test_search_observation_arbitration import NOW, _record, owner


def paired_pending(owner, monkeypatch, reason="verified_continuation_recheck", distance=2.0):
    anchor, _legacy = confirmed_then_pending(owner, monkeypatch, reason)
    short = ShortFollowController(ShortFollowConfig(enabled=True))
    short.activate(7, NOW-1.0)
    original = short.update(ShortFollowObservation(
        uid=7, capture_id=231, capture_timestamp=NOW-.08,
        depth_timestamp=NOW-.04, distance_m=distance, center_x_ratio=.50,
    ), now=NOW)
    owner._short_follow = short
    owner._short_follow_adapter = ShortFollowAdapter(owner, short, runtime.logger)
    owner._depth30_linear_snapshot = None
    owner._depth30_linear_timing = None
    owner._detector_identity_lease = None
    owner._validated_visual_observation = ValidatedVisualObservation(
        uid=7, track_id=3, capture=231, timestamp=NOW-.08,
        validated_at=NOW-.06, expires_at=NOW+.42, kind="full",
    )
    owner._rknn_pipeline.last_frame_width = 640
    owner._rknn_pipeline.last_identity_processing = {
        "mode": "full", "full_features_current": True,
        "capture_frame_id": 233, "capture_timestamp": NOW-.01,
    }
    return anchor, original


def publish_recheck(owner, records, *, now=NOW):
    return owner._update_detector_identity_lease(
        records, owner._active_capture_frame_id, owner._active_capture_timestamp,
        now=now, stale=False,
    )


@pytest.mark.parametrize("reason", ["verified_continuation_recheck", "partial_boundary_recheck"])
def test_real_identity_publication_and_record_consumer_preserve_original_paired_plan(owner, monkeypatch, reason):
    anchor, plan = paired_pending(owner, monkeypatch, reason)
    before = owner._short_follow.snapshot()
    records = [_record(uid=0)]
    assert publish_recheck(owner, records)
    proof = owner._validated_visual_observation
    assert isinstance(proof, ValidatedVisualObservation)
    assert proof.capture == 231
    assert proof.timestamp == NOW-.08
    assert proof.continuation_sample_timestamp == plan.depth_timestamp
    assert proof.expires_at == pytest.approx(min(NOW+.20, plan.expires_at))
    assert proof.permits_depth(7, plan.depth_timestamp, NOW)
    assert not proof.permits_depth(7, NOW, NOW)

    owner._consume_track_records(records, 640, 480, "test")
    assert owner._events == []
    assert owner._short_follow.snapshot() is before
    assert owner._short_follow.snapshot().plan is plan
    assert owner._confirmed_identity_execution_anchor is anchor
    assert owner._depth30_linear_snapshot is None
    assert owner._longitudinal_context is None


def test_grey_proof_cannot_publish_a_new_depth_or_update_either_wheel(owner, monkeypatch):
    _anchor, plan = paired_pending(owner, monkeypatch)
    assert publish_recheck(owner, [_record(uid=0)])
    owner._publish_follow_recording = lambda *_: None
    now = NOW+.05
    frame = SimpleNamespace(
        width=640, capture_frame_id=235, capture_timestamp=NOW+.03,
        distance_m=3.0,
        distance_state=SimpleNamespace(sample_timestamp=NOW+.04,
                                       raw_distance_m=3.0, safety_distance_m=None),
        hazard=SimpleNamespace(active=False),
        obstacles=SimpleNamespace(front=False, left=False, right=False),
    )
    target = SimpleNamespace(track_id=7, center=(500, 240))
    assert owner._short_follow_adapter.handle(
        frame, target, is_fresh_depth=True, control_source="depth30",
        target_steerable=True, low_quality_visible=False, now=now)
    assert owner._short_follow.snapshot().plan is plan
    assert owner._short_follow.snapshot().plan.expires_at == plan.expires_at


@pytest.mark.parametrize("case", [
    "competition_failed", "competing_candidate", "geometry_jump", "bank_conflict",
    "bank_revoked", "bank_suspect", "unbound", "wrong_reason", "explicit_stop",
    "different_uid", "search", "epoch_mismatch", "stationary", "expired_plan",
])
def test_paired_motion_does_not_bypass_any_identity_or_epoch_protection(owner, monkeypatch, case):
    _anchor, plan = paired_pending(owner, monkeypatch, distance=1.45 if case == "stationary" else 2.0)
    assignment = owner._assignments[3]
    bank = owner._rknn_pipeline.tracker.identity_bank
    record = _record(uid=0)
    now = NOW
    if case == "competition_failed":
        assignment["identity_competition"]["passed"] = False
    elif case == "competing_candidate":
        assignment["identity_competition"]["candidate_count"] = 2
    elif case == "geometry_jump":
        record = _record(uid=0, bbox=(430., 20., 610., 460.))
    elif case == "bank_conflict":
        bank._mapped_geometry_conflicts[3] = {"uid": 7}
    elif case == "bank_revoked":
        bank._geometry_revoked_uids[7] = 1
    elif case == "bank_suspect":
        bank._reacquire_control_suspects[7] = {"reason": "identity_conflict"}
    elif case == "unbound":
        bank.track_to_uid.clear()
    elif case == "wrong_reason":
        assignment["reason"] = "recent_partial_conflict"
    elif case == "explicit_stop":
        owner._explicit_stop_requested = True
    elif case == "different_uid":
        assignment["mapped_uid"] = 8
    elif case == "search":
        owner._follow_controller.search_state = "searching"
    elif case == "epoch_mismatch":
        state = owner._short_follow.snapshot()
        owner._short_follow._state = replace(state, epoch=state.epoch+1)
    elif case == "expired_plan":
        now = plan.expires_at+.001
        assignment["identity_recheck_deadline"] = NOW+.40
    assert owner._retain_visual_proof_for_identity_recheck(
        [record], owner._active_capture_frame_id, owner._active_capture_timestamp, now) is None


def test_repeated_grey_frames_cannot_roll_the_original_plan_or_proof_deadline(owner, monkeypatch):
    anchor, plan = paired_pending(owner, monkeypatch)
    for index, offset in enumerate((0.0, .05, .10)):
        now = NOW+offset
        cap = 233+index*2
        stamp = now-.01
        monkeypatch.setattr(runtime.time, "monotonic", lambda now=now: now)
        owner.frame_index = 101+index
        owner._active_capture_frame_id, owner._active_capture_timestamp = cap, stamp
        owner._assignments[3]["identity_recheck_capture"] = cap
        owner._assignments[3]["identity_competition"]["frame_index"] = owner.frame_index
        owner._rknn_pipeline.last_identity_processing.update(
            capture_frame_id=cap, capture_timestamp=stamp)
        assert publish_recheck(owner, [_record(uid=0)], now=now)
        owner._consume_track_records([_record(uid=0)], 640, 480, "test")
        assert owner._events == []
        assert owner._short_follow.snapshot().plan is plan
        assert owner._confirmed_identity_execution_anchor is anchor
        proof = owner._validated_visual_observation
        assert proof.continuation_sample_timestamp == NOW-.04
        assert proof.expires_at == pytest.approx(NOW+.20)
    assert not owner._validated_visual_observation.live(7, NOW+.201)
    assert owner._short_follow.snapshot().plan is plan


@pytest.mark.parametrize("race", ["new_plan", "revoke"])
def test_motion_evidence_changed_during_bridge_verification_cannot_be_retained(owner, monkeypatch, race):
    paired_pending(owner, monkeypatch)
    original_check = runtime.PersonTracker._existing_drive_grant_during_identity_recheck

    def change_after_check(self, *args):
        result = original_check(self, *args)
        assert result
        if race == "revoke":
            self._short_follow.revoke("concurrent_stop", NOW)
        else:
            replacement = self._short_follow.update(ShortFollowObservation(
                uid=7, capture_id=232, capture_timestamp=NOW-.06,
                depth_timestamp=NOW-.02, distance_m=2.2, center_x_ratio=.7,
            ), now=NOW)
            assert replacement is not None
        return result

    monkeypatch.setattr(runtime.PersonTracker, "_existing_drive_grant_during_identity_recheck", change_after_check)
    assert owner._retain_visual_proof_for_identity_recheck(
        [_record(uid=0)], 233, NOW-.01, NOW) is None


def test_active_paired_owner_never_uses_retired_legacy_grant_as_fallback(owner, monkeypatch):
    paired_pending(owner, monkeypatch)
    owner._short_follow.revoke("hard_stop", NOW)
    owner._depth30_linear_snapshot = ("forward", 27, 7, NOW-.01)
    assert owner._identity_recheck_motion_evidence(7, NOW) is None
    assert not owner._existing_drive_grant_during_identity_recheck(
        _record(uid=0), owner._assignments[3], NOW, 640)
