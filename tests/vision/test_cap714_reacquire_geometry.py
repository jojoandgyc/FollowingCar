"""Recorded CAP694/714–755 geometry with synthetic, logged-distance features.

This is a deterministic policy regression, not a claim of model accuracy.
"""
from copy import deepcopy

import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from test_cap874_identity_reacquire import feature, metadata


ANCHOR_BOX = (568.375244, 118.910919, 640., 475.628296)
ANCHOR_TIME = 24774.417047365
ANCHOR_YAW = -49.7643804867
ROWS = [
    (301,714,24775.480386519,(0.,184.917114,119.735565,403.857727),-39.884674,.191143,.271204),
    (303,718,24775.680266378,(0.,183.706268,103.977585,403.820343),-42.978936,.249159,.300107),
    (304,722,24775.881010133,(0.,186.682678,101.385490,407.241028),-41.725519,.232175,.285548),
    (305,723,24775.944373670,(.733070,191.694153,99.409225,407.338409),-40.746827,.225465,.286606),
    (306,726,24776.112235186,(0.,197.167206,126.834198,472.317932),-40.222870,.255854,.263188),
    (322,746,24777.144570096,(0.,188.982971,146.825577,407.454346),-46.564929,.184739,.260570),
    (324,750,24777.344437706,(.727631,189.998322,172.353928,473.663757),-47.928399,.227554,.265688),
    (325,753,24777.544060322,(12.242607,192.027954,166.484192,473.888245),-49.243373,.193720,.284380),
    (326,755,24777.644695859,(34.714966,195.939911,187.475449,402.305237),-50.841614,.202930,.273584),
]


def meta(cap, stamp, box, yaw=ANCHOR_YAW, search=True, opposite=True):
    return dict(metadata(cap, stamp, box, yaw, search, opposite),
                partial_observation=True, partial_feature_source="osnet_torso")


def seeded_bank():
    b = IdentityBank(IdentityBankConfig(camera_hfov_deg=60., min_area=0,
        new_identity_confirm_frames=1, update_interval=1, controlled_handoff_enable=True,
        template_memory_enable=True, template_crosscheck_enable=True))
    assert b.assign(track_id=3, feature=feature(0), partial_feature=feature(0),
        confidence=.95, area=25549, frame_index=286,
        sample_metadata=meta(694, ANCHOR_TIME, ANCHOR_BOX, search=False)) == 1
    return b


def send(b, row, track=3, *, search=True, opposite=True):
    frame, cap, stamp, box, yaw, full, partial = row
    return b.assign(track_id=track, feature=feature(full), partial_feature=feature(partial),
        confidence=.95, area=(box[2]-box[0])*(box[3]-box[1]), frame_index=frame,
        sample_metadata=meta(cap, stamp, box, yaw, search, opposite),
        bbox_quality_ok=True, bbox_quality_tier="strong",
        preferred_uid=1 if search else None, preferred_candidate_ok=search and not opposite)


@pytest.mark.parametrize("full,partial,allowed", [(.225,.261,False),(.23,.01,False),(.10,.261,True)])
@pytest.mark.parametrize("caller_ok", [False,True])
def test_opposite_partial_never_promotes_weak_full_evidence(full, partial, allowed, caller_ok):
    b = seeded_bank(); m = meta(723, ROWS[3][2], ROWS[3][3])
    result = b._preferred_search_reacquire_candidate(feature=feature(full),
        partial_feature=feature(partial), preferred_uid=1, candidate_ok=caller_ok,
        candidate_count=1, sample_metadata=m)
    assert (result is not None) == allowed
    if allowed:
        assert result[-1] == "strong"
        assert result[1] == pytest.approx(full)
    assert m["reacquire_distance_limit"] == .20


def test_direction_compatible_partial_path_is_preserved():
    b=seeded_bank();m=meta(723,ROWS[3][2],ROWS[3][3],opposite=False)
    result=b._preferred_search_reacquire_candidate(feature=feature(.4),
        partial_feature=feature(.12),preferred_uid=1,candidate_ok=True,sample_metadata=m)
    assert result[-1]=="partial" and result[1]==pytest.approx(.12)


@pytest.mark.parametrize("first", [0,1,2])
def test_recorded_wrong_track_never_reclaims_uid_or_rewrites_anchor(first):
    b=seeded_bank();anchor=deepcopy(b.identities[1].last_strong_observation)
    for row in ROWS[first:]:
        assert send(b,row)==0
        a=b.last_assignments[3]
        assert a["identity_control_rejected"] and not a["bank_updated"]
        assert b.identities[1].last_strong_observation==anchor
    assert not b.pending_late_handoffs
    assert len(b.identities[1].features)==1


def test_clipped_equal_areas_still_preserve_large_cross_edge_conflict():
    b=seeded_bank();r=ROWS[0];m=meta(r[1],r[2],r[3],r[4])
    g=b._handoff_geometry(1,m,r[0])
    assert g["area_similarity"]>.97 and g["yaw_compensated_center_jump_ratio"]>.68
    assert b._search_geometry_contradiction(g,m)
    assert g["search_cross_edge_conflict"]


def test_stale_frame_count_keeps_residuals_but_never_proves_continuity():
    b=seeded_bank();r=ROWS[2];m=meta(r[1],r[2],r[3],r[4])
    g=b._handoff_geometry(1,m,r[0])
    assert g["reason"]=="stale_reference" and g["ok"] is None
    assert g["yaw_compensated_center_jump_ratio"]>.73
    assert b._search_geometry_contradiction(g,m)


@pytest.mark.parametrize("dt,yaw,search",[(7.45,ANCHOR_YAW,True),(1.46,ANCHOR_YAW+35,True),
    (1.46,None,True),(-1.,ANCHOR_YAW,True),(1.46,ANCHOR_YAW,False)])
def test_cross_edge_rule_requires_bounded_time_turn_and_search(dt,yaw,search):
    b=seeded_bank();m=meta(722,ANCHOR_TIME+dt,ROWS[2][3],yaw,search)
    g=b._handoff_geometry(1,m,304)
    assert not b._search_geometry_contradiction(g,m)


def test_yaw_explained_displacement_is_not_cross_edge_conflict():
    b=seeded_bank();m=meta(722,ANCHOR_TIME+1.,ROWS[2][3],ANCHOR_YAW+50.)
    g=b._handoff_geometry(1,m,304)
    assert g["yaw_compensated_center_jump_ratio"]<.1
    assert not b._search_geometry_contradiction(g,m)


def test_new_raw_id_inherits_established_negative_tracklet():
    b=seeded_bank()
    assert send(b,ROWS[2])==0
    assert send(b,ROWS[3],track=9)==0
    assert b._mapped_geometry_conflicts[9]["search_contradiction"]


def test_association_geometry_review_is_read_only_until_assignment():
    b=seeded_bank();r=ROWS[0];m=meta(r[1],r[2],r[3],r[4])
    before=deepcopy(b.identities[1].last_strong_observation)
    assert b.review_mapped_geometry(3,m,r[0])["mapped_geometry_blocked"]
    assert not b._mapped_geometry_conflicts and b.track_to_uid[3]==1
    assert b.identities[1].last_strong_observation==before
    assert send(b,r)==0 and 3 in b._mapped_geometry_conflicts


def test_strong_opposite_match_after_large_turn_still_uses_two_frames():
    b=seeded_bank()
    for i,expected in [(0,0),(1,1)]:
        row=(304+i,722+i,ANCHOR_TIME+1.5+i*.1,ROWS[2][3],ANCHOR_YAW+50.,.10,.10)
        assert send(b,row,track=9)==expected
    assert b.last_assignments[9]["authorization_match_source"]=="strong"
    assert b.last_assignments[9]["protected_search_anchor_cap"]==694


def test_correct_other_candidate_can_reacquire_after_wrong_track_rejected():
    b=seeded_bank();assert send(b,ROWS[0])==0
    for i,expected in [(0,0),(1,1)]:
        row=(304+i,722+i,ANCHOR_TIME+1.5+i*.1,ANCHOR_BOX,ANCHOR_YAW,.10,.10)
        assert send(b,row,track=9,opposite=False)==expected


@pytest.mark.parametrize("source",["partial","soft_partial","soft_strong"])
def test_late_observer_cannot_bypass_opposite_full_gate(source):
    b=seeded_bank();r=ROWS[2];m=meta(r[1],r[2],r[3],r[4])
    g=b._handoff_geometry(1,m,r[0]);g["reason"]="search_reacquire_time_window"
    assert b._observe_late_search_candidate(track_id=9,candidate_uid=1,distance=.12,
        partial_feature=feature(.12),match_source=source,candidate_count=1,
        bbox_quality_ok=True,sample_metadata=m,frame_index=r[0],geometry=g) is None
    assert not b.pending_late_handoffs


def test_weak_observer_cannot_bypass_opposite_direction_with_caller_ok():
    b=seeded_bank();r=ROWS[2];m=meta(r[1],r[2],r[3],r[4])
    b._assign_weak_search_candidate(track_id=9,feature=feature(.01),partial_feature=feature(.01),
        frame_index=r[0],candidate_count=1,quality_reason="edge_touch>2",sample_metadata=m,
        preferred_uid=1,preferred_candidate_ok=True,diagnostics={})
    assert not b.pending_weak_handoffs


def reacquired_bank():
    b=seeded_bank();box=(40.,100.,200.,460.)
    for i,expected in [(0,0),(1,1)]:
        row=(400+i,900+i,ANCHOR_TIME+8+i*.1,box,ANCHOR_YAW,.1,.1)
        assert send(b,row,track=9,opposite=False)==expected
    return b,box


def test_probationary_geometry_cannot_become_next_search_reference():
    b,box=reacquired_bank()
    assert b.identities[1].last_strong_observation["capture_frame_id"]==901
    m=meta(903,ANCHOR_TIME+8.3,box,opposite=False)
    assert b._handoff_geometry(1,m,403)["reference"]["capture_frame_id"]==694
    m["search_reacquire_context_active"]=False
    assert b._handoff_geometry(1,m,403)["reference"]["capture_frame_id"]==901
    # A second handoff must not overwrite the saved trusted anchor.
    b._bind_reacquired_identity(1,10,403,m)
    assert b._reacquire_search_anchors[1]["capture_frame_id"]==694
    b.reset();assert not b._reacquire_search_anchors


def test_verified_probation_release_promotes_current_reference():
    b,box=reacquired_bank()
    for i in range(1,16):
        row=(401+i,901+i,ANCHOR_TIME+8.1+i*.1,box,ANCHOR_YAW,.1,.1)
        assert send(b,row,track=9,search=False,opposite=False)==1
    assert not b._reacquire_quarantine.is_held(1)
    assert not b._reacquire_search_anchors
    m=meta(920,ANCHOR_TIME+10.,box,opposite=False)
    assert b._handoff_geometry(1,m,420)["reference"]["capture_frame_id"]==916
