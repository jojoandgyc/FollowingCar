"""CAP133-style stable target + excluded background: real tracker, no RKNN."""
import pickle

import numpy as np
import pytest

from rk_vision.yolo11 import Detection
from test_detector_continuation import BOX, COLOR, FEATURE, context, ready_tracker
from test_detector_continuation_pipeline import fast_pipeline, seed


BACKGROUND = (20., 140., 90., 390.)
BACKGROUND_FEATURE = np.array([.8, .6, 0.], dtype=np.float32)


def detections(*, target=BOX, background=BACKGROUND, reverse=False):
    rows = [Detection(target, .95, 0), Detection(background, .90, 0)]
    return rows[::-1] if reverse else rows


def full_pair(tracker, cap, *, background=BACKGROUND, co_visible=True):
    stamp = 10.+cap*.05
    rows = detections(background=background)
    ctx = context(cap, stamp)
    records = tracker.update(rows, [FEATURE, BACKGROUND_FEATURE], image_width=640,
        image_height=480, frame_context=ctx, color_features=[COLOR, COLOR])
    if not co_visible and 2 in tracker.identity_bank._identity_exclusion._observations:
        tracker.identity_bank._identity_exclusion._observations[2].exclusions.clear()
    tracker.note_full_identity_verification(records, detections=rows,
        color_features=[COLOR, COLOR], frame_context=ctx, image_width=640,
        image_height=480, now=stamp+.04, active_uid=1, raw_candidate_count=2)
    return records


def ready_pair():
    tracker = ready_tracker()
    for cap in (5, 6, 7):
        full_pair(tracker, cap)
    assert tracker.last_detector_continuation_reason == "full_verified"
    assert len(tracker._detector_proof.backgrounds) == 1
    return tracker


def pair_plan(tracker, cap=8, *, rows=None, color=None, now=None, **kwargs):
    stamp = 10.+cap*.05
    return tracker.plan_detected_continuation(rows or detections(**kwargs),
        frame_context=context(cap, stamp), active_uid=1,
        image_width=640, image_height=480, raw_candidate_count=len(rows) if rows else 2,
        now=stamp+.04 if now is None else now,
        color_features=[COLOR, COLOR] if color is None else color)


def test_known_background_is_not_a_new_competitor_and_bank_remains_immutable():
    tracker = ready_pair()
    bank = pickle.dumps(tracker.identity_bank)
    metric = pickle.dumps(tracker.deepsort.tracker.metric)
    deadline = tracker._detector_proof.deadline
    for cap, reverse in ((8, True), (9, False)):
        proposed = pair_plan(tracker, cap, reverse=reverse)
        assert proposed is not None
        records = tracker.commit_detected_continuation(proposed, now=10.+cap*.05+.045)
        assert [(r.track_id, r.reid_uid) for r in records] == [(1, 1), (2, 0)]
        assert tracker._detector_proof.deadline == deadline
        assert tracker._detector_proof.verified.capture == 7
        obs = tracker.last_identity_observations
        assert obs[0]["sample_metadata"]["source_detection_index"] == int(reverse)
        assert obs[0]["sample_metadata"]["candidate_count"] == 2
        assert obs[1]["assignment"]["reason"] == "detector_excluded_background"
        assert obs[1]["assignment"]["exclusion_verified_capture"] == 7
        assert tracker.control_assignment_for_track(2)["uid"] == 0
        assert all(t.time_since_update == 0 and t.last_feature is None
                   for t in tracker.deepsort.tracker.tracks)
    assert pickle.dumps(tracker.identity_bank) == bank
    assert pickle.dumps(tracker.deepsort.tracker.metric) == metric
    assert pair_plan(tracker, 10) is None
    assert tracker.last_detector_continuation_reason == "fast_budget_exhausted"
    assert full_pair(tracker, 10)[0].reid_uid == 1
    assert tracker._detector_proof.verified.capture == 10


@pytest.mark.parametrize("fault", ["unknown", "missing", "target_missing", "overlap",
    "background_jump", "background_grows", "target_jump", "wrong_target_color",
    "expired", "duplicate"])
def test_unknown_or_changed_people_cannot_borrow_background_exclusion(fault):
    tracker = ready_pair()
    kwargs = {}
    if fault == "unknown": kwargs["rows"] = detections()+[Detection((480., 70., 600., 400.), .96, 0)]
    if fault == "missing": kwargs["rows"] = detections()[:1]
    if fault == "target_missing": kwargs["rows"] = detections()[1:]
    if fault == "overlap": kwargs["background"] = (235., 140., 305., 390.)
    if fault == "background_jump": kwargs["background"] = (450., 140., 520., 390.)
    if fault == "background_grows": kwargs["background"] = (0., 20., 205., 460.)
    if fault == "target_jump": kwargs["target"] = (350., 70., 470., 400.)
    if fault == "wrong_target_color": kwargs["color"] = [np.eye(16)[0], COLOR]
    if fault == "expired": kwargs["now"] = tracker._detector_proof.deadline+.001
    if fault == "duplicate": kwargs["cap"] = 7
    before = tracker._frame_index
    assert pair_plan(tracker, **kwargs) is None
    assert tracker._frame_index == before
    assert tracker._detector_proof is None


@pytest.mark.parametrize("fault", ["exclusion_revoked", "wrong_witness", "became_target",
    "background_conflict", "target_conflict", "predicted_background"])
def test_commit_rechecks_negative_evidence_for_all_matched_tracks(fault):
    tracker = ready_pair()
    proposed = pair_plan(tracker)
    assert proposed is not None
    bank = tracker.identity_bank
    if fault == "exclusion_revoked":
        bank._identity_exclusion._observations[2].exclusions.clear()
    if fault == "wrong_witness":
        bank._identity_exclusion._observations[2].exclusions[1]["reference_track_id"] = 99
    if fault == "became_target": bank.track_to_uid[2] = 1
    if fault == "background_conflict": bank._mapped_geometry_conflicts[2] = {"uid": 1}
    if fault == "target_conflict": bank._geometry_revoked_uids[1] = 1
    if fault == "predicted_background": tracker.deepsort.tracker.tracks[1].time_since_update = 1
    assert tracker.commit_detected_continuation(proposed, now=10.445) is None
    assert tracker._frame_index == 7


def test_reid_loser_without_geometry_exclusion_cannot_seed_multi_person_proof():
    tracker = ready_pair()
    bank = tracker.identity_bank
    bank._identity_exclusion._observations[2].exclusions.clear()
    assert bank.last_assignments[2]["identity_competition"]["reason"] != "ineligible_uid_competitor"
    rows = detections()
    records = [type("Record", (), {"reid_uid": 1, "track_id": 1})()]
    tracker.note_full_identity_verification(records, detections=rows,
        color_features=[COLOR, COLOR], frame_context=context(7, 10.35),
        image_width=640, image_height=480, now=10.39, active_uid=1)
    assert tracker._detector_proof is None
    assert tracker.last_detector_continuation_reason == "full_background_unverified"


def test_actual_pipeline_skips_embeddings_with_confirmed_excluded_background(fast_pipeline):
    s = fast_pipeline; p = s.pipeline
    seed(s)
    original = p.reid.extract
    def extract(packet, persons, fmt):
        original(packet, persons, fmt)
        return [FEATURE if d.bbox == BOX else BACKGROUND_FEATURE for d in persons]
    p.reid.extract = extract
    p.detector.detections = detections()
    for cap in (5, 6, 7): s.step(cap)
    assert p.tracker.last_detector_continuation_reason == "full_verified"
    before = p.reid.calls
    p.detector.detections = detections(reverse=True)
    rows = s.step(8)
    assert p.last_identity_processing["mode"] == "detector_continuation"
    assert p.reid.calls == before
    assert p.detector.calls == 8
    assert [(r.track_id, r.reid_uid) for r in rows] == [(1, 1), (2, 0)]
    p.detector.detections.append(Detection((450., 70., 590., 400.), .94, 0))
    s.step(9)
    assert p.last_identity_processing["mode"] == "full"
    assert p.reid.calls == before+1


def test_background_certificate_does_not_extend_periodic_full_or_absolute_lease():
    tracker = ready_pair()
    deadline = tracker._detector_proof.deadline
    assert pair_plan(tracker, cap=12, now=10.64) is None
    assert tracker.last_detector_continuation_reason == "detection_gap"
    assert deadline == pytest.approx(10.95)


def test_explicit_competition_ineligibility_can_seed_but_not_clear_new_conflict():
    tracker = ready_tracker()
    small = (20., 300., 55., 390.)
    # Exercise independent scale/position ineligibility without the stronger
    # co-visible memory. This does not manufacture an appearance score/veto.
    for cap in (5, 6, 7):
        full_pair(tracker, cap, background=small, co_visible=False)
    assert tracker._detector_proof.backgrounds[0].eligibility[0] == "ineligible_uid_competitor"
    proposed = pair_plan(tracker, 8, background=small)
    assert proposed is not None
    tracker.identity_bank.last_assignments[2]["identity_competition"]["competition_eligible"] = True
    assert tracker.commit_detected_continuation(proposed, now=10.445) is None


def test_target_competition_failure_prevents_proof_even_with_known_background():
    tracker = ready_pair()
    tracker.identity_bank.last_assignments[1]["identity_competition"]["passed"] = False
    assert pair_plan(tracker) is None
    assert tracker.last_detector_continuation_reason == "identity_assignment_rejected"
