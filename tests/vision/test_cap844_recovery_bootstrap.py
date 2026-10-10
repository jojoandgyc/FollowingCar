"""Recorded CAP844–888 metadata, synthetic vectors at the logged distances.

Exercises the public assignment path; not an embedding/model or motor replay.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from test_cap874_identity_reacquire import feature, metadata

DATA = json.loads((Path(__file__).parent / "fixtures/cap844_recovery.json").read_text())
ROWS = {r["cap"]: r for r in DATA["rows"]}


def meta(row, **changes):
    m = metadata(row["cap"], row["ts"], row["bbox"], row["yaw"])
    m.update(partial_feature_source="osnet_torso", partial_observation=True,
             detector_edge_touch_count=row["edges"], edge_touch_count=row["display_edges"],
             quality_bbox_ok=row["quality"] == "strong", bbox_quality_tier=row["quality"])
    m.update(changes)
    return m


def send(b, cap, **changes):
    row = dict(ROWS[cap])
    row.update(changes.pop("row_changes", {}))
    m = meta(row, **changes)
    box = row["bbox"]
    return b.assign(track_id=3, feature=feature(row["full"]),
        partial_feature=feature(row["recent_partial"]), confidence=row["confidence"],
        area=(box[2]-box[0])*(box[3]-box[1]), frame_index=row["frame"],
        candidate_count=m.get("candidate_count", 1),
        bbox_quality_ok=m["quality_bbox_ok"], bbox_quality_tier=m["bbox_quality_tier"],
        sample_metadata=m, preferred_uid=1, preferred_candidate_ok=True)


def setup():
    b = IdentityBank(IdentityBankConfig(new_identity_confirm_frames=1,
        controlled_handoff_enable=True, template_memory_enable=True,
        template_crosscheck_enable=True, handoff_geometry_max_gap_frames=15,
        preferred_search_reacquire_threshold=.20, preferred_search_reacquire_max_age_sec=.35))
    ref = DATA["reference"]
    # A trusted synthetic gallery at a valid torso crop. Actual protected
    # geometry is retained separately, including its original CAP and clock.
    m = metadata(708, ref["capture_timestamp"]-2., (180., 10., 400., 470.), 0., False)
    m["partial_feature_source"] = "osnet_torso"
    assert b.assign(track_id=1, feature=feature(0), partial_feature=feature(0),
        confidence=.95, area=100000, frame_index=340, sample_metadata=m) == 1
    b.identities[1].last_strong_observation = deepcopy(ref)
    previous = DATA["previous"]
    b._bind_reacquired_identity(1, 3, previous["frame"], meta(previous))
    b._remember_strong_observation(1, 3, meta(previous), previous["frame"])
    assert send(b, 833) == 0
    assert b.last_assignments[3]["reason"] == "recent_partial_conflict"
    return b


def test_logged_good_segment_recovers_and_does_not_immediately_fall_back():
    b = setup()
    protected = deepcopy(b._reacquire_search_anchors[1])
    outputs = {}
    for cap in ROWS:
        if cap == 833:
            continue
        outputs[cap] = send(b, cap)
        if b._reacquire_quarantine.is_held(1):
            assert not b.last_assignments[3]["bank_updated"]
            assert b._reacquire_search_anchors[1] == protected
        if outputs[cap] == 0:
            assert not b.last_assignments[3]["bank_updated"]
    assert outputs[844] == 0  # Missing auxiliary evidence is not permission.
    assert outputs[846] == 0  # Seed only, never a one-frame confirmation.
    assert outputs[848] == 1
    assert outputs[864] == 0  # New seed after genuine torso conflict.
    assert all(outputs[c] == 1 for c in ROWS if 866 <= c <= 882)
    assert all(outputs[c] == 0 for c in (850, 852, 855, 861, 862, 883, 886, 888))


def test_seed_does_not_refresh_identity_anchor_gallery_or_quarantine():
    b = setup()
    before = deepcopy(b.identities[1].last_strong_observation)
    seen = b.identities[1].last_seen_frame
    learning = deepcopy(b.identities[1].template_memory.last_learning)
    assert send(b, 866) == 0
    a = b.last_assignments[3]
    assert a["reacquire_control_seeded"]
    assert a["reacquire_control_geometry_source"] == "local_seed"
    assert a["reacquire_geometry_reason"] == "stale_reference"
    assert b.identities[1].last_strong_observation == before
    assert b.identities[1].last_seen_frame == seen
    assert b.identities[1].template_memory.last_learning == learning
    assert b._reacquire_quarantine.is_held(1)
    assert send(b, 867) == 1
    assert b.last_assignments[3]["reacquire_control_local_reference_cap"] == 866


@pytest.mark.parametrize("changes", [
    {"is_fresh": False}, {"capture_timestamp": None},
    {"identity_competition": {"passed": False}},
    {"candidate_count": 2, "candidate_score_gap": 0.},
    {"candidate_count": 2, "candidate_score_gap": .9},
    {"row_changes": {"full": .301}},
    {"row_changes": {"recent_partial": .38}},
    {"row_changes": {"quality": "weak", "edges": 3}},
])
def test_unqualified_frame_cannot_seed(changes):
    b = setup()
    assert send(b, 866, **changes) == 0
    assert not b.last_assignments[3].get("reacquire_control_seeded")
    assert not b._reacquire_control_suspects[1].get("local_observation")
    assert not b.last_assignments[3]["bank_updated"]


@pytest.mark.parametrize("changes", [
    {"is_fresh": False},
    {"row_changes": {"cap": 866, "ts": ROWS[866]["ts"]}},
    {"row_changes": {"cap": 865, "ts": ROWS[866]["ts"]-.01}},
    {"row_changes": {"ts": ROWS[866]["ts"]+.351}},
    {"row_changes": {"bbox": (10., 100., 70., 220.), "yaw": ROWS[866]["yaw"]}},
    {"identity_competition": {"passed": False}},
    {"row_changes": {"recent_partial": .38}},
])
def test_second_frame_must_be_new_timely_and_consistent(changes):
    b = setup()
    assert send(b, 866) == 0
    assert send(b, 867, **changes) == 0
    assert not b.last_assignments[3]["bank_updated"]


def test_unknown_torso_preserves_bounded_proof_but_never_confirms():
    b = setup()
    assert send(b, 866) == 0
    proof = deepcopy(b._reacquire_control_suspects[1]["local_observation"])
    assert proof is not None
    assert send(b, 867, row_changes={"quality": "weak", "edges": 3}) == 0
    assert b.last_assignments[3]["reason"] == "secondary_evidence_unavailable"
    assert b._reacquire_control_suspects[1]["local_observation"] == proof
    assert send(b, 869) == 1


def test_persisted_identity_conflict_still_blocks_low_distance_candidate():
    b = setup()
    b._mapped_geometry_conflicts[3] = {
        "uid": 1, "search_contradiction": True, "rejected_frame": 410,
        "rejected_capture": 833, "reference": deepcopy(DATA["reference"])}
    for cap in (866, 867, 869, 871):
        assert send(b, cap) == 0
        assert b.last_assignments[3]["reason"] == "mapped_geometry_reject"
        assert not b.last_assignments[3]["bank_updated"]


def test_disabled_late_candidate_policy_cannot_bootstrap():
    b = setup()
    b.config = replace(b.config, preferred_search_reacquire_late_candidate_enable=False)
    for cap in (866, 867):
        assert send(b, cap) == 0
        assert not b.last_assignments[3]["reacquire_control_seeded"]


def test_multi_candidate_needs_current_identity_competition_to_seed():
    b = setup()
    proof = {"uid": 1, "frame_index": ROWS[866]["frame"], "passed": True,
             "source_detection_index": 0, "candidate_count": 2}
    assert send(b, 866, candidate_count=2, source_detection_index=0,
                identity_competition=proof) == 0
    assert b.last_assignments[3]["reacquire_control_seeded"]
    proof["frame_index"] = ROWS[867]["frame"]
    assert send(b, 867, candidate_count=2, source_detection_index=0,
                identity_competition=proof) == 1


@pytest.mark.parametrize("stage", ["seed", "confirm"])
@pytest.mark.parametrize("invalid_proof", [
    "missing", "empty", "previous_frame", "future_frame", "wrong_uid", "no_pass",
])
def test_multi_candidate_requires_same_frame_uid_proof_at_both_stages(stage, invalid_proof):
    b = setup()
    cap = 866
    if stage == "confirm":
        assert send(b, 866) == 0
        assert b.last_assignments[3]["reacquire_control_seeded"]
        cap = 867
    proof = {"uid": 1, "frame_index": ROWS[cap]["frame"], "passed": True,
             "source_detection_index": 0, "candidate_count": 2}
    if invalid_proof == "missing":
        proof = None
    elif invalid_proof == "empty":
        proof = {}
    elif invalid_proof == "previous_frame":
        proof["frame_index"] -= 1
    elif invalid_proof == "future_frame":
        proof["frame_index"] += 1
    elif invalid_proof == "wrong_uid":
        proof["uid"] = 2
    else:
        proof.pop("passed")
    # A large DETECTOR confidence gap must not replace identity competition.
    assert send(b, cap, candidate_count=2, candidate_score_gap=.4, source_detection_index=0,
                identity_competition=proof) == 0
    a = b.last_assignments[3]
    assert not a.get("reacquire_control_seeded")
    assert not a.get("reacquire_control_recovered")
    assert not a["bank_updated"]
    assert b._reacquire_control_suspects[1]["streak"] == 0
    assert b._reacquire_control_suspects[1].get("local_observation") is None
    # The rejected frame must not leave half a proof for later confirmation.
    assert send(b, 869) == 0
    assert b.last_assignments[3]["reacquire_control_seeded"]
    assert send(b, 871) == 1


def test_unknown_samples_do_not_extend_the_local_proof_deadline():
    b = setup()
    assert send(b, 866) == 0
    original = deepcopy(b._reacquire_control_suspects[1]["local_observation"])
    for cap in (867, 869, 871, 873):
        assert send(b, cap, row_changes={"quality": "weak", "edges": 3}) == 0
        assert b._reacquire_control_suspects[1]["local_observation"] == original
    assert send(b, 875) == 0
    assert b.last_assignments[3]["reacquire_control_seeded"]
    assert b.last_assignments[3]["reacquire_control_local_candidate_cap"] == 875
    assert send(b, 877) == 1


def test_future_protected_anchor_is_not_treated_as_old():
    b = setup()
    b._reacquire_search_anchors[1]["frame_index"] = 1000
    b._reacquire_search_anchors[1]["capture_timestamp"] = ROWS[888]["ts"]+1
    assert send(b, 866) == 0
    assert not b.last_assignments[3]["reacquire_control_seeded"]


def test_missing_geometry_cannot_seed():
    b = setup()
    assert send(b, 866, detector_bbox=None) == 0
    assert not b.last_assignments[3].get("reacquire_control_seeded")


def test_rejected_capture_cannot_be_reprocessed_as_a_new_good_seed():
    b = setup()
    assert send(b, 866, row_changes={"recent_partial": .38}) == 0
    assert send(b, 866) == 0
    assert not b.last_assignments[3]["reacquire_control_seeded"]
    assert send(b, 867) == 0
    assert b.last_assignments[3]["reacquire_control_seeded"]
    assert send(b, 869) == 1


def test_out_of_order_conflict_does_not_move_capture_watermark_backwards():
    b = setup()
    assert send(b, 866) == 0
    assert send(b, 862) == 0  # Repeated old torso conflict may not rewind time.
    assert send(b, 864) == 0
    assert not b.last_assignments[3]["reacquire_control_seeded"]
    assert send(b, 867) == 0
    assert b.last_assignments[3]["reacquire_control_seeded"]
