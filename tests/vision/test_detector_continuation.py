"""Real tracker, synthetic detections/features: no models, camera or motors."""
from dataclasses import FrozenInstanceError
import pickle

import numpy as np
import pytest

from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.detector_continuation import FULL_PROOF_TTL_SEC
from rk_vision.yolo11 import Detection


BOX = (230., 70., 350., 400.)
COLOR = np.ones(16)
FEATURE = np.array([1., 0., 0.], dtype=np.float32)


def context(cap, stamp, **extra):
    return dict(capture_frame_id=cap, capture_timestamp=stamp, integrated_yaw_deg=0., **extra)


def full(tracker, cap, stamp, *, bbox=BOX, uid=1, now=None):
    detections = [Detection(bbox, .95, 0)]
    ctx = context(cap, stamp)
    records = tracker.update(detections, [FEATURE], image_width=640, image_height=480,
                             frame_context=ctx, color_features=[COLOR])
    tracker.note_full_identity_verification(records, detections=detections,
        color_features=[COLOR], frame_context=ctx, image_width=640, image_height=480,
        now=stamp+.04 if now is None else now, active_uid=uid)
    return records


def ready_tracker():
    tracker = DeepSortTracker(DeepSortTrackerConfig(n_init=1,
        identity_new_confirm_frames=1, identity_update_interval=1))
    for cap in range(1, 5):
        full(tracker, cap, 10.+cap*.05)
    assert tracker.last_detector_continuation_reason == "full_verified"
    return tracker


def plan(tracker, *, cap=5, stamp=10.25, now=None, bbox=BOX, color=None, uid=1,
         raw_count=1, ctx=None):
    return tracker.plan_detected_continuation([Detection(bbox, .95, 0)],
        image_width=640, image_height=480, frame_context=ctx or context(cap, stamp),
        active_uid=uid, now=stamp+.04 if now is None else now,
        color_features=[COLOR] if color is None else color, raw_candidate_count=raw_count)


def test_two_full_verifications_required_and_color_is_lazy():
    tracker = DeepSortTracker(DeepSortTrackerConfig(n_init=1, identity_new_confirm_frames=1))
    full(tracker, 1, 10.05)
    full(tracker, 2, 10.10)
    full(tracker, 3, 10.15)
    calls = []
    assert plan(tracker, cap=4, stamp=10.20, color=lambda: calls.append(1)) is None
    assert not calls
    assert tracker.last_detector_continuation_reason == "full_verification_streak"
    full(tracker, 4, 10.20)
    p = plan(tracker, color=lambda: calls.append(1) or [COLOR])
    assert p is not None and calls == [1]
    with pytest.raises(FrozenInstanceError):
        p.observation.capture = 55


def test_fast_measurements_preserve_every_identity_state_and_metric(monkeypatch):
    tracker = ready_tracker()
    before = pickle.dumps(tracker.identity_bank)
    metric = pickle.dumps(tracker.deepsort.tracker.metric)
    def forbidden(*args, **kwargs):
        raise AssertionError("fast observation must not assign or train")
    monkeypatch.setattr(type(tracker.identity_bank), "assign", forbidden)
    monkeypatch.setattr(type(tracker.deepsort.tracker.metric), "partial_fit", forbidden)
    original_anchor = tracker.identity_bank.identities[1].last_strong_observation
    for cap, stamp, bbox in ((5, 10.25, (233., 70., 353., 400.)),
                             (6, 10.30, (236., 70., 356., 400.))):
        rows = tracker.commit_detected_continuation(plan(tracker, cap=cap, stamp=stamp, bbox=bbox), now=stamp+.045)
        assert len(rows) == 1 and rows[0].reid_uid == 1
        assert rows[0].time_since_update == 0
        assert (rows[0].x1, rows[0].y1, rows[0].x2, rows[0].y2) == bbox
        assignment = tracker.control_assignment_for_track(1)
        assert assignment["identity_evidence_kind"] == "detector_continuation"
        assert assignment["identity_verified_capture"] == 4
        assert assignment["identity_verified_timestamp"] == pytest.approx(10.2)
        assert assignment["identity_valid_until"] == pytest.approx(10.2 + FULL_PROOF_TTL_SEC)
        assert not assignment["initial_identity_confirmed"] and not assignment["bank_updated"]
        assert tracker.last_identity_observations[0]["sample_metadata"]["capture_frame_id"] == cap
        assert tracker.deepsort.tracker.tracks[0].last_feature is None
        assert tracker.identity_bank.identities[1].last_strong_observation is original_anchor
    monkeypatch.undo()
    assert pickle.dumps(tracker.identity_bank) == before
    assert pickle.dumps(tracker.deepsort.tracker.metric) == metric
    calls = []
    assert plan(tracker, cap=7, stamp=10.35, color=lambda: calls.append(1)) is None
    assert not calls and tracker.last_detector_continuation_reason == "fast_budget_exhausted"
    assert full(tracker, 7, 10.35)[0].track_id == 1
    assert tracker.control_assignment_for_track(1) is None
    assert tracker._detector_proof.verified.capture == 7


@pytest.mark.parametrize("case", ["expired", "stale", "future", "duplicate", "reordered", "gap",
    "raw_competitor", "false_count", "uid", "color", "geometry", "scale", "bad_box", "yaw_missing"])
def test_unsafe_detection_cannot_continue_or_renew_proof(case):
    tracker = ready_tracker()
    kw = {}
    if case == "expired": kw["now"] = 10.2 + FULL_PROOF_TTL_SEC + .01
    elif case == "stale": kw["now"] = 10.45
    elif case == "future": kw["now"] = 10.24
    elif case == "duplicate": kw.update(cap=4, stamp=10.2)
    elif case == "reordered": kw.update(cap=3, stamp=10.21)
    elif case == "gap": kw.update(cap=8, stamp=10.411)
    elif case == "raw_competitor": kw["raw_count"] = 2
    elif case == "false_count": kw["raw_count"] = True
    elif case == "uid": kw["uid"] = 2
    elif case == "color": kw["color"] = [np.eye(16)[0]]
    elif case == "geometry": kw["bbox"] = (400., 70., 520., 400.)
    elif case == "scale": kw["bbox"] = (240., 80., 310., 260.)
    elif case == "bad_box": kw["bbox"] = (-1., 70., 100., 400.)
    else: kw["ctx"] = dict(capture_frame_id=5, capture_timestamp=10.25)
    assert plan(tracker, **kw) is None
    assert tracker._detector_proof is None


@pytest.mark.parametrize("case", ["search", "mapping", "conflict", "revoked", "suspect", "pending",
    "quarantine", "recheck", "quality", "predicted", "extra_track", "exclusion"])
def test_commit_rechecks_identity_negative_evidence(case, monkeypatch):
    tracker = ready_tracker()
    p = plan(tracker)
    assert p is not None
    bank = tracker.identity_bank
    if case == "search": tracker.set_search_reacquire_context(active_uid=1, searching=True, direction="left")
    elif case == "mapping": bank.track_to_uid[1] = 2
    elif case == "conflict": bank._mapped_geometry_conflicts[1] = {"uid": 1}
    elif case == "revoked": bank._geometry_revoked_uids[1] = 1
    elif case == "suspect": bank._reacquire_control_suspects[1] = {}
    elif case == "pending": bank.pending_handoffs[1] = object()
    elif case == "quarantine": monkeypatch.setattr(bank._reacquire_quarantine, "is_held", lambda uid: True)
    elif case == "recheck": bank.last_assignments[1]["identity_recheck_pending"] = True
    elif case == "quality": bank.last_assignments[1]["bbox_quality_ok"] = False
    elif case == "predicted": tracker.deepsort.tracker.tracks[0].time_since_update = 1
    elif case == "extra_track": tracker.deepsort.tracker.tracks.append(tracker.deepsort.tracker.tracks[0])
    elif case == "exclusion": monkeypatch.setattr(bank, "search_exclusion_for", lambda *a, **kw: {"excluded": True})
    before = tracker._frame_index
    assert tracker.commit_detected_continuation(p, now=10.30) is None
    assert tracker._frame_index == before
    assert tracker.control_assignment_for_track(1) is None


def test_periodic_full_is_due_by_capture_time_and_cannot_slide():
    tracker = ready_tracker()
    p = plan(tracker, stamp=10.35)
    assert tracker.commit_detected_continuation(p, now=10.395)
    assert plan(tracker, cap=6, stamp=10.4) is None
    assert tracker.last_detector_continuation_reason == "full_recheck_due"
    assert tracker._detector_proof.verified.timestamp == pytest.approx(10.2)
    # Late completion is not a new capture-time verification lease.
    full(tracker, 7, 10.45, now=10.81)
    assert tracker._detector_proof is None


def test_plan_is_single_use_and_commit_clock_and_full_replacement_are_checked():
    for case in ("repeat", "clock", "expired", "new_full", "anchor", "anchor_mutated"):
        tracker = ready_tracker()
        p = plan(tracker)
        if case == "repeat": assert tracker.commit_detected_continuation(p, now=10.30)
        elif case == "new_full": full(tracker, 6, 10.30)
        elif case == "anchor": tracker.identity_bank.identities[1].last_strong_observation = {}
        elif case == "anchor_mutated": tracker.identity_bank.identities[1].last_strong_observation["capture_frame_id"] = 100
        assert tracker.commit_detected_continuation(p, now=(10.28 if case == "clock" else
            10.2 + FULL_PROOF_TTL_SEC + .01 if case == "expired" else 10.31)) is None


def test_expired_full_proof_cannot_borrow_fast_streak_to_requalify():
    tracker = ready_tracker()
    for cap, stamp in ((5, 10.25), (6, 10.30)):
        assert tracker.commit_detected_continuation(plan(tracker, cap=cap, stamp=stamp),
                                                    now=stamp+.04)
    expired_at = tracker._detector_proof.deadline
    late_capture = expired_at + .05
    assert full(tracker, 7, late_capture)[0].reid_uid == 1
    assert tracker._detector_proof.full_count == 1
    assert plan(tracker, cap=8, stamp=late_capture+.05) is None


def test_fast_proof_survives_bounded_full_check_at_recorded_frame_pace():
    tracker = ready_tracker()
    # The October 4 recording had ~179 ms capture spacing and ~152 ms result
    # age. The next full result must arrive before the fixed proof expires.
    fast_stamp = 10.37
    assert tracker.commit_detected_continuation(
        plan(tracker, cap=5, stamp=fast_stamp, now=10.46), now=10.47)
    old_deadline = tracker._detector_proof.deadline
    assert old_deadline == pytest.approx(10.8)
    assert full(tracker, 6, 10.54, now=10.71)[0].reid_uid == 1
    assert 10.71 < old_deadline
    assert tracker._detector_proof.full_count == 2


@pytest.mark.parametrize("guard,limit,bbox,reason", [
    ("identity_max_single_frame_area_shrink_ratio", .85,
     (240., 95., 340., 375.), "detector_area_shrink"),
    ("identity_max_center_jump_ratio", .05,
     (280., 70., 400., 400.), "detector_center_jump"),
])
def test_fast_frame_obeys_configured_dynamic_box_guards(guard, limit, bbox, reason):
    from dataclasses import replace
    tracker = ready_tracker()
    tracker.config = replace(tracker.config, **{guard: limit})
    assert plan(tracker, bbox=bbox) is None
    assert tracker.last_detector_continuation_reason == reason
    assert tracker._frame_index == 4  # full processing can still use this frame


def test_fast_measurements_advance_local_quality_clocks_for_next_full():
    tracker = ready_tracker()
    for cap, stamp in ((5, 10.25), (6, 10.30)):
        assert tracker.commit_detected_continuation(plan(tracker, cap=cap, stamp=stamp),
                                                    now=stamp+.04)
    assert tracker._last_quality_area_by_track_id[1][0] == 6
    assert tracker._last_identity_center_frame_by_track_id[1] == 6


@pytest.mark.parametrize("gap", [.180001, .190, .194916, .199999])
def test_bounded_capture_gap_can_use_fast_path_without_refreshing_identity(gap):
    tracker = ready_tracker()
    proof = tracker._detector_proof
    stamp = proof.previous.timestamp + gap
    bank_before = pickle.dumps(tracker.identity_bank)
    p = plan(tracker, stamp=stamp)
    assert p is not None
    rows = tracker.commit_detected_continuation(p, now=stamp+.045)
    assert rows[0].reid_uid == 1
    assert tracker._detector_proof.verified is proof.verified
    assert tracker._detector_proof.deadline == proof.deadline
    assert pickle.dumps(tracker.identity_bank) == bank_before


@pytest.mark.parametrize("gap", [.20, .201739, .21])
def test_bounded_gap_still_requires_periodic_full_but_keeps_streak(gap):
    tracker = ready_tracker()
    proof = tracker._detector_proof
    stamp = proof.previous.timestamp + gap
    assert plan(tracker, stamp=stamp) is None
    assert tracker.last_detector_continuation_reason == "full_recheck_due"
    assert tracker._detector_proof is proof
    assert full(tracker, 5, stamp)[0].reid_uid == 1
    assert tracker._detector_proof.full_count == 2
    assert plan(tracker, cap=6, stamp=stamp+.05) is not None


@pytest.mark.parametrize("gap", [.194916, .201739, .21])
def test_two_full_checks_can_bootstrap_at_recorded_capture_cadence(gap):
    tracker = DeepSortTracker(DeepSortTrackerConfig(n_init=1,
        identity_new_confirm_frames=1, identity_update_interval=1))
    for cap in range(1, 4):
        full(tracker, cap, 10.+cap*.05)
    proof = tracker._detector_proof
    assert proof.full_count == 1
    stamp = proof.previous.timestamp + gap
    assert plan(tracker, cap=4, stamp=stamp) is None
    assert tracker.last_detector_continuation_reason == "full_verification_streak"
    assert tracker._detector_proof is proof
    assert full(tracker, 4, stamp)[0].reid_uid == 1
    assert tracker._detector_proof.full_count == 2
    assert plan(tracker, cap=5, stamp=stamp+.05) is not None


@pytest.mark.parametrize("gap", [.210001, .23, .30])
def test_over_210ms_gap_requires_new_full_streak(gap):
    tracker = ready_tracker()
    stamp = tracker._detector_proof.previous.timestamp + gap
    assert plan(tracker, stamp=stamp) is None
    assert tracker.last_detector_continuation_reason == "detection_gap"
    assert tracker._detector_proof is None
    assert full(tracker, 5, stamp)[0].reid_uid == 1
    assert tracker._detector_proof.full_count == 1


@pytest.mark.parametrize("age", [.18, .180001, .20, .21])
def test_capture_gap_extension_does_not_extend_current_detection_age(age):
    tracker = ready_tracker()
    stamp = tracker._detector_proof.previous.timestamp + .194916
    assert plan(tracker, stamp=stamp, now=stamp+age) is None
    assert tracker.last_detector_continuation_reason == "detection_stale"
    assert tracker._detector_proof is None


def test_commit_cannot_use_gap_budget_as_extra_processing_time():
    tracker = ready_tracker()
    # Keep enough full-check headroom so this isolates the detection-age gate.
    stamp = tracker._detector_proof.previous.timestamp + .10
    p = plan(tracker, stamp=stamp, now=stamp+.179999)
    assert p is not None
    assert tracker.commit_detected_continuation(p, now=stamp+.18) is None
    assert tracker._frame_index == 4


def test_detector_time_limits_are_independent():
    from rk_vision.detector_continuation import (
        MAX_DETECTION_GAP_SEC, MAX_DETECTION_AGE_SEC,
        MAX_FULL_RESULT_AGE_SEC, FULL_RECHECK_INTERVAL_SEC, MAX_FAST_FRAMES,
    )
    assert MAX_DETECTION_GAP_SEC == .21
    assert MAX_DETECTION_AGE_SEC == .18
    assert MAX_FULL_RESULT_AGE_SEC == .35
    assert FULL_RECHECK_INTERVAL_SEC == .20
    assert FULL_PROOF_TTL_SEC == .60
    assert MAX_FAST_FRAMES == 2


def test_reset_search_and_active_target_switch_require_two_new_full_checks():
    for change in ("reset", "search", "uid"):
        tracker = ready_tracker()
        if change == "reset": tracker.reset()
        elif change == "search":
            tracker.set_search_reacquire_context(active_uid=1, searching=True, direction="left")
            tracker.set_search_reacquire_context(active_uid=1, searching=False, direction=None)
        else:
            assert plan(tracker, uid=2) is None
        assert tracker._detector_proof is None
        assert plan(tracker) is None


def test_current_optional_context_is_frozen_and_never_borrowed_from_full_frame():
    tracker = ready_tracker()
    ctx = context(5, 10.25, control_frame_id=55, yaw_rate_dps=2.)
    p = plan(tracker, ctx=ctx)
    ctx.update(control_frame_id=999, yaw_rate_dps=999.)
    assert tracker.commit_detected_continuation(p, now=10.30)
    meta = tracker.last_identity_observations[0]["sample_metadata"]
    assert meta["control_frame_id"] == 55 and meta["yaw_rate_dps"] == 2.
    assert tracker.commit_detected_continuation(plan(tracker, cap=6, stamp=10.30), now=10.35)
    meta = tracker.last_identity_observations[0]["sample_metadata"]
    assert "control_frame_id" not in meta and "yaw_rate_dps" not in meta


def test_favorable_yaw_sign_cannot_adopt_nonoverlapping_person():
    tracker = ready_tracker()
    ctx = context(5, 10.25)
    ctx["integrated_yaw_deg"] = 14.
    assert plan(tracker, ctx=ctx, bbox=(345., 70., 465., 400.)) is None
    assert tracker.last_detector_continuation_reason == "adjacent_geometry"


def test_control_context_revocation_invalidates_an_already_planned_frame():
    tracker = ready_tracker()
    p = plan(tracker)
    tracker.set_detector_continuation_context(active_uid=1, allowed=False)
    tracker.set_detector_continuation_context(active_uid=1, allowed=True)
    assert tracker.commit_detected_continuation(p, now=10.30) is None
    assert tracker._detector_proof is None
