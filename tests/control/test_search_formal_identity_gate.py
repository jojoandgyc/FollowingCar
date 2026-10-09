"""CAP2284: mapped UID / detector confirmation must not redirect the search."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.search_identity_evidence import confirmed_search_candidate_uid


CAP = 2284
STAMP = 8900.990770376
BBOX = (165.0466766357422, 7.852706, 344.605071067, 475.3)


def evidence(uid=1):
    record = SimpleNamespace(track_id=4, reid_uid=uid, class_id=0, time_since_update=0)
    observation = dict(
        raw_track_id=4, uid=uid, detector_bbox=BBOX,
        sample_metadata=dict(capture_frame_id=CAP, capture_timestamp=STAMP, is_fresh=True),
        assignment=dict(uid=uid, mapped_uid=1, bbox_quality_ok=True,
                        reacquire_geometry_ok=True, reason="mapped"),
    )
    return dict(active_uid=1, raw_track_id=4, capture_frame_id=CAP,
                capture_timestamp=STAMP, records=[record], observations=[observation])


def controller():
    item = FollowSafetyController(FollowPolicyConfig(center_left_ratio=.45, center_right_ratio=.55))
    item.active_target_id = 1
    item._has_seen_person = True
    item.search_state = "searching"
    item.search_direction = "right"
    item._lost_exit_direction = "right"
    item._lost_hint_source = "latest_reliable_capture_side"
    item._target_stop_latched = True
    item._search_rotation_started_at = 8900.90
    return item


def note(item, **kwargs):
    args = dict(frame_width=640, confirmed=True, source="formal",
                candidate_score=.8891248106956482, candidate_tracked=True,
                candidate_identity_match=False, capture_frame_id=CAP)
    args.update(kwargs)
    return item.note_search_candidate_evidence(BBOX, **args)


def test_cap2284_mapped_uid_retained_but_formal_uid_zero_cannot_switch(caplog):
    data = evidence(uid=0)
    data["observations"][0]["assignment"].update(
        reason="preferred_search_mapped_distance_reject", distance=.204,
        best_uid=1, mapped_uid=1,
    )
    uid = confirmed_search_candidate_uid(**data)
    assert uid == 0
    item = controller()
    with caplog.at_level("INFO"):
        assert note(item, candidate_confirmed_uid=uid) is False
    assert item.search_direction == item._lost_exit_direction == "right"
    assert item._search_rotation_started_at == 8900.90
    assert item._lost_hint_source == "latest_reliable_capture_side"
    assert item._target_stop_latched is True
    assert "capture_frame_id=2284" in caplog.text
    assert "reason=identity_not_confirmed" in caplog.text


def test_current_confirmed_uid_may_update_direction_without_authorizing_motion():
    item = controller()
    uid = confirmed_search_candidate_uid(**evidence())
    assert uid == 1
    assert note(item, candidate_confirmed_uid=uid) is False
    assert item.search_direction == item._lost_exit_direction == "left"
    assert item.active_target_id == 1
    assert item._target_stop_latched is True
    assert item.last_selected_target is None


@pytest.mark.parametrize("tracked", [False, True])
@pytest.mark.parametrize("hint", [False, True])
@pytest.mark.parametrize("completed", [False, True])
def test_raw_track_or_strong_reid_hint_never_substitutes_for_formal_uid(tracked, hint, completed):
    item = controller()
    note(item, candidate_tracked=tracked, candidate_identity_match=hint,
         confirmed=completed, source="blocked")
    assert item.search_direction == item._lost_exit_direction == "right"
    assert item._search_rotation_started_at == 8900.90


@pytest.mark.parametrize("uid", [0, None, -1, 2, "bad", float("nan")])
def test_wrong_or_invalid_formal_uid_does_not_redirect(uid):
    item = controller()
    note(item, candidate_confirmed_uid=uid)
    assert item.search_direction == "right"


def test_same_side_unconfirmed_observation_does_not_reset_trusted_search_budget():
    item = controller()
    item.search_direction = item._lost_exit_direction = "left"
    assert note(item) is False
    assert item.search_direction == "left"
    assert item._lost_hint_source == "latest_reliable_capture_side"
    assert item._search_rotation_started_at == 8900.90


def test_unconfirmed_candidate_during_centering_preserves_existing_pause_and_direction():
    item = controller()
    item._stale_direction_recovery_active = True
    item._stale_direction_recovery_stage = "candidate_centering"
    item._stale_candidate_side = "right"
    item._stale_candidate_center_ratio = .7
    assert note(item) is True
    assert item.search_direction == "right"
    assert item._stale_candidate_side == "right"
    assert item._stale_candidate_center_ratio == .7


def test_stale_gap_cannot_seed_new_search_direction_from_detector_only():
    item = controller()
    item.search_state = "none"
    item.search_direction = None
    item._stale_direction_recovery_active = True
    item._stale_direction_recovery_stage = "observe"
    item.note_stale_candidate_evidence(BBOX, frame_width=640, confirmed=True, source="formal")
    assert item.search_state == "none"
    assert item.search_direction is None


@pytest.mark.parametrize("field,value", [
    ("reid_uid", 0), ("reid_uid", 2), ("time_since_update", 1), ("class_id", 1),
])
def test_record_must_be_current_formal_person(field, value):
    data = evidence()
    setattr(data["records"][0], field, value)
    assert confirmed_search_candidate_uid(**data) == 0


@pytest.mark.parametrize("field,value", [
    ("uid", 0), ("uid", 2), ("bbox_quality_ok", False),
    ("reacquire_geometry_ok", False), ("identity_control_rejected", True),
    ("search_excluded", True), ("search_contradiction_retained", True),
    ("reacquire_geometry", {"ok": False}),
    ("reacquire_geometry", {"search_cross_edge_conflict": True}),
    ("reacquire_geometry", {"short_handoff_identity_conflict": True}),
    ("identity_competition", {"passed": False}),
])
def test_current_negative_identity_evidence_dominates_formal_uid(field, value):
    data = evidence()
    data["observations"][0]["assignment"][field] = value
    assert confirmed_search_candidate_uid(**data) == 0


@pytest.mark.parametrize("field,value", [
    ("capture_frame_id", CAP - 1), ("capture_timestamp", STAMP - .01),
    ("is_fresh", False), ("identity_competition", {"passed": False}),
])
def test_observation_must_have_current_capture_provenance(field, value):
    data = evidence()
    data["observations"][0]["sample_metadata"][field] = value
    assert confirmed_search_candidate_uid(**data) == 0


@pytest.mark.parametrize("key", ["records", "observations"])
@pytest.mark.parametrize("count", [0, 2])
def test_missing_or_ambiguous_identity_proof_rejected(key, count):
    data = evidence()
    data[key] *= count
    assert confirmed_search_candidate_uid(**data) == 0


def test_excluded_candidate_cannot_gain_direction_rights():
    assert confirmed_search_candidate_uid(**evidence(), excluded=True) == 0


def test_confirmed_detector_probe_uses_same_proof_despite_negative_raw_track_id():
    data = evidence()
    data["raw_track_id"] = data["records"][0].track_id = -1
    data["observations"][0]["raw_track_id"] = -1
    assert confirmed_search_candidate_uid(**data) == 1


def bound_runtime_candidate(data):
    """Run the production local helper and note-call arguments without RKNN/I/O."""
    tree = ast.parse(Path(runtime.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PersonTracker")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process_external_frame")
    helper = next(n for n in method.body if isinstance(n, ast.FunctionDef)
                  and n.name == "candidate_confirmed_identity_uid")
    binding = dict(raw_track_id=4, bbox=BBOX, exclusion=None)
    ns = dict(vars(runtime))
    ns.update(self=SimpleNamespace(_rknn_pipeline=SimpleNamespace(tracker=SimpleNamespace(
        last_identity_observations=data["observations"]))), active_candidate_uid=1,
        current_capture_id=CAP, frame_received_ts=STAMP, candidate_records=data["records"],
        exclusion_bindings=(binding,))
    exec(compile(ast.Module(body=[helper], type_ignores=[]), runtime.__file__, "exec"), ns)
    lateral_assignment = next(n for n in method.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "current_candidate_matches_target" for t in n.targets))
    ns["current_candidate_decision"] = SimpleNamespace(bbox=BBOX)
    exec(compile(ast.Module(body=[lateral_assignment], type_ignores=[]), runtime.__file__, "exec"), ns)
    call = next(n for n in ast.walk(method) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "note_candidate")
    uid_arg = next(k.value for k in call.keywords if k.arg == "candidate_confirmed_uid")
    ns["gate_decision"] = SimpleNamespace(bbox=BBOX)
    uid = eval(compile(ast.Expression(uid_arg), runtime.__file__, "eval"), ns)
    return uid, ns["current_candidate_matches_target"]


@pytest.mark.parametrize("uid", [0, 1])
def test_runtime_note_and_lateral_candidate_share_same_formal_uid_proof(uid):
    confirmed_uid, lateral_match = bound_runtime_candidate(evidence(uid))
    assert confirmed_uid == uid
    assert lateral_match is bool(uid)
    item = controller()
    note(item, candidate_confirmed_uid=confirmed_uid)
    assert item.search_direction == ("left" if uid else "right")


def test_runtime_rejected_mapping_cannot_be_promoted_by_previous_best_uid():
    data = evidence(uid=0)
    data["observations"][0]["assignment"].update(
        mapped_uid=1, best_uid=1, reason="preferred_search_mapped_distance_reject")
    assert bound_runtime_candidate(data) == (0, False)


def test_runtime_ambiguous_detector_binding_cannot_gain_identity():
    data = evidence()
    extra = copy.deepcopy(data["observations"][0])
    extra["raw_track_id"] = 5
    data["observations"].append(extra)
    # The production association itself requires a unique high-overlap match.
    assert runtime.PersonTracker._search_binding_for_bbox(BBOX, (
        dict(raw_track_id=4, bbox=BBOX, exclusion=None),
        dict(raw_track_id=5, bbox=BBOX, exclusion=None),
    )) is None
