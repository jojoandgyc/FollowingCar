"""A grey identity frame may only finish an already approved physical grant."""
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from test_cap1307_boundary_recheck import prepare
from test_search_observation_arbitration import NOW, _record, owner


def confirmed_then_pending(owner, monkeypatch, reason="verified_continuation_recheck"):
    prepare(owner)
    monkeypatch.setattr(runtime, "MODULE_ASTRA_DEPTH_ENABLE", True)
    owner._depth30_linear_snapshot = ("forward", 27, 7, NOW-.04)
    owner._active_capture_timestamp = NOW-.08
    owner._assignments[3] = dict(uid=7, bbox_quality_ok=True,
                                 bbox_quality_tier="strong", bank_updated=False)
    owner._consume_track_records([_record(uid=7)], 640, 480, "test")
    anchor = owner._confirmed_identity_execution_anchor
    assert anchor["uid"] == 7 and anchor["track_id"] == 3
    assert anchor["capture_frame_id"] == 231
    owner._events.clear()
    owner._context_events.clear()
    owner.frame_index += 1
    owner._active_capture_frame_id = 233
    owner._active_capture_timestamp = NOW-.01
    owner._rknn_pipeline = SimpleNamespace(tracker=SimpleNamespace(
        identity_bank=SimpleNamespace(track_to_uid={3: 7},
            _mapped_geometry_conflicts={}, _geometry_revoked_uids={},
            _reacquire_control_suspects={})))
    owner._assignments[3] = dict(uid=0, mapped_uid=7, reason=reason,
        identity_control_rejected=True, identity_recheck_pending=True,
        identity_recheck_capture=233, identity_recheck_deadline=NOW+.20,
        bbox_quality_ok=False, bbox_quality_tier="reject", bank_updated=False,
        identity_competition=dict(uid=7, frame_index=owner.frame_index,
                                  candidate_count=1, source_detection_index=0,
                                  passed=True))
    if reason == "verified_continuation_recheck":
        owner._assignments[3]["identity_continuation"] = dict(
            status="hold", source="partial", reference_cap=231, deadline=NOW+.20)
    return anchor, owner._depth30_linear_snapshot


@pytest.mark.parametrize("reason", ["verified_continuation_recheck", "partial_boundary_recheck"])
def test_single_same_track_recheck_keeps_only_existing_grant(owner, monkeypatch, reason):
    anchor, original_grant = confirmed_then_pending(owner, monkeypatch, reason)
    owner._consume_track_records([_record(uid=0)], 640, 480, "test")
    assert owner._events == []
    assert owner._context_events == []
    assert owner._longitudinal_context is None
    assert owner._depth30_linear_snapshot is original_grant
    assert owner._confirmed_identity_execution_anchor is anchor
    assert owner._follow_controller.active_target_id == 7
    assert owner._follow_controller.search_state == "none"
    assert not owner._deferred_timeout


def test_repeated_grey_frames_cannot_roll_anchor_or_physical_deadline(owner, monkeypatch):
    anchor, original_grant = confirmed_then_pending(owner, monkeypatch)
    for offset, cap in [(0., 233), (.05, 235), (.10, 237)]:
        monkeypatch.setattr(runtime.time, "monotonic", lambda offset=offset: NOW+offset)
        owner.frame_index = 101 + (cap-233)//2
        owner._active_capture_frame_id = cap
        owner._active_capture_timestamp = NOW+offset-.01
        owner._assignments[3]["identity_recheck_capture"] = cap
        owner._assignments[3]["identity_competition"]["frame_index"] = owner.frame_index
        owner._consume_track_records([_record(uid=0)], 640, 480, "test")
        assert owner._events == []
        assert owner._depth30_linear_snapshot is original_grant
        assert owner._confirmed_identity_execution_anchor is anchor
        assert owner._assignments[3]["identity_recheck_deadline"] == NOW+.20
    # The bank's recheck has time left, but the *original* Depth sample does not.
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.15)
    owner.frame_index = 104
    owner._active_capture_frame_id = 239
    owner._active_capture_timestamp = NOW+.14
    owner._assignments[3]["identity_recheck_capture"] = 239
    owner._assignments[3]["identity_competition"]["frame_index"] = owner.frame_index
    owner._consume_track_records([_record(uid=0)], 640, 480, "test")
    assert owner._events == [("queue", [runtime.ACTION_STOP], "identity_boundary_recheck")]
    assert owner._depth30_linear_snapshot is None
    assert owner._confirmed_identity_execution_anchor is None


@pytest.mark.parametrize("case", ["different_track", "different_uid", "crowd",
    "conflict", "competition_failed", "competition_missing", "competition_stale",
    "competition_other_uid", "competing_candidate", "geometry_jump", "search",
    "explicit_stop", "stale_source", "wrong_proof", "reverse_grant", "unbound",
    "bank_conflict", "bank_revoked", "bank_suspect", "assignment_conflict",
    "resolution_change"])
def test_uncertain_or_unsafe_recheck_never_inherits_motion(owner, monkeypatch, case):
    confirmed_then_pending(owner, monkeypatch)
    records = [_record(uid=0)]
    width = 640
    assignment = owner._assignments[3]
    if case == "different_track":
        owner._assignments[4] = dict(assignment)
        records = [_record(track=4, uid=0)]
    elif case == "different_uid":
        assignment["mapped_uid"] = 8
    elif case == "crowd":
        records.append(_record(track=4, uid=0))
    elif case == "conflict":
        assignment["reason"] = "recent_partial_conflict"
    elif case == "competition_failed":
        assignment["identity_competition"] = dict(uid=7, passed=False)
    elif case == "competition_missing":
        assignment.pop("identity_competition")
    elif case == "competition_stale":
        assignment["identity_competition"]["frame_index"] -= 1
    elif case == "competition_other_uid":
        assignment["identity_competition"]["uid"] = 8
    elif case == "competing_candidate":
        assignment["identity_competition"]["candidate_count"] = 2
    elif case == "geometry_jump":
        records = [_record(uid=0, bbox=(430., 20., 610., 460.))]
    elif case == "search":
        owner._follow_controller.search_state = "searching"
    elif case == "explicit_stop":
        owner._explicit_stop_requested = True
    elif case == "stale_source":
        owner._confirmed_identity_execution_anchor["capture_timestamp"] = NOW-.36
    elif case == "wrong_proof":
        assignment["identity_continuation"]["status"] = "reject"
    elif case == "reverse_grant":
        owner._depth30_linear_snapshot = ("backward", 27, 7, NOW-.04)
    elif case == "unbound":
        owner._rknn_pipeline.tracker.identity_bank.track_to_uid.clear()
    elif case == "bank_conflict":
        owner._rknn_pipeline.tracker.identity_bank._mapped_geometry_conflicts[3] = {"uid": 7}
    elif case == "bank_revoked":
        owner._rknn_pipeline.tracker.identity_bank._geometry_revoked_uids[7] = 100
    elif case == "bank_suspect":
        owner._rknn_pipeline.tracker.identity_bank._reacquire_control_suspects[7] = {"reason": "identity_conflict"}
    elif case == "assignment_conflict":
        assignment["search_contradiction_retained"] = True
    elif case == "resolution_change":
        width = 800
    owner._consume_track_records(records, width, 480, "test")
    assert owner._events and owner._events[0][0] in {"queue", "normal"}
    assert owner._depth30_linear_snapshot is None
    assert owner._confirmed_identity_execution_anchor is None


def test_new_confirmed_frame_recovers_normal_control_after_recheck(owner, monkeypatch):
    confirmed_then_pending(owner, monkeypatch)
    owner._consume_track_records([_record(uid=0)], 640, 480, "test")
    owner._assignments[3] = dict(uid=7, bbox_quality_ok=True,
                                 bbox_quality_tier="strong", bank_updated=False)
    owner.frame_index += 1
    owner._active_capture_frame_id = 235
    owner._active_capture_timestamp = NOW-.005
    owner._consume_track_records([_record(uid=7)], 640, 480, "test")
    assert [event[0] for event in owner._events] == ["normal"]
    assert owner._confirmed_identity_execution_anchor["capture_frame_id"] == 235


def test_depth_revoke_during_recheck_diagnostic_does_not_crash_vision(owner, monkeypatch):
    confirmed_then_pending(owner, monkeypatch)

    def revoke_after_check(_self, _rec, _assignment, _now, _width):
        owner._depth30_linear_snapshot = None
        return True

    monkeypatch.setattr(runtime.PersonTracker,
                        "_existing_drive_grant_during_identity_recheck",
                        revoke_after_check)
    owner._consume_track_records([_record(uid=0)], 640, 480, "test")
    assert owner._depth30_linear_snapshot is None
    assert owner._events == []
