"""Deterministic identity-policy regression; synthetic vectors reproduce logged d.

These tests validate gates, not the embedding model or simulated car dynamics.
"""
import math
from copy import deepcopy

import numpy as np
import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig


ANCHOR = (0., 19.055359, 76.937637, 380.134094)
ROWS = [
    (371,874,10389.681619,(474.653503,231.494019,536.510864,418.849792),-72.669758,.171927),
    (372,877,10389.817220,(468.258911,181.695618,535.432922,421.545013),-74.108066,.164920),
    (373,881,10390.020750,(457.602783,178.220703,524.930481,429.395966),-75.381763,.290842),
    (374,884,10390.222865,(393.151337,177.489868,525.738220,424.327087),-75.712680,.372018),
    (380,898,10390.948878,(297.084900,169.673508,401.007263,425.482208),-76.141692,.402270),
    (386,912,10391.6,(116.7,156.4,257.5,450.9),-77.0,.149466),
    (390,926,10392.412784,(0.,146.923599,109.241539,461.896912),-77.753405,.259389),
]


def feature(d):
    return np.asarray([1-d, math.sqrt(max(0.,1-(1-d)**2)),0.], dtype=np.float32)


def metadata(cap, ts, bbox, yaw=0., search=True, opposite=False):
    area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1])/(640*480)
    return dict(capture_frame_id=cap,capture_timestamp=ts,is_fresh=True,
        bbox=list(bbox),detector_bbox=list(bbox),detector_area_ratio=area,area_ratio=area,
        center_x_ratio=(bbox[0]+bbox[2])/1280,detector_center_x_ratio=(bbox[0]+bbox[2])/1280,
        integrated_yaw_deg=yaw,edge_touch_count=0,detector_edge_touch_count=0,
        quality_bbox_ok=True,bbox_quality_tier="strong",candidate_count=1,
        candidate_score_gap=.95,detector_confidence=.95,
        search_reacquire_context_active=search,search_direction_compatible=not opposite)


def assign(bank, row, *, track=5, opposite=True, search=True, partial=None, extra=None):
    frame,cap,ts,bbox,yaw,d=row
    meta=metadata(cap,ts,bbox,yaw,search,opposite)
    meta.update(extra or {})
    return bank.assign(track_id=track,feature=feature(d),partial_feature=partial,
        confidence=.95,area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1]),frame_index=frame,
        bbox_quality_ok=True,bbox_quality_tier="strong",sample_metadata=meta,
        preferred_uid=1 if search else None,preferred_candidate_ok=search and not opposite)


def bank_at_857():
    bank=IdentityBank(IdentityBankConfig(new_identity_confirm_frames=1,min_area=0,
        controlled_handoff_enable=True,controlled_handoff_threshold=.30,
        controlled_handoff_confirm_frames=2,preferred_search_reacquire_confirm_frames=2,
        handoff_geometry_max_gap_frames=12,handoff_geometry_max_center_jump_ratio=.30,
        handoff_geometry_min_area_similarity=.40,preferred_search_reacquire_max_age_sec=.35,
        preferred_search_reacquire_threshold=.20,preferred_search_soft_candidate_threshold=.30,
        mapped_verify_threshold=.45,partial_appearance_enable=True))
    assert assign(bank,(359,857,10388.785306,ANCHOR,-66.329599,0),search=False)==1
    return bank


def test_cap874_926_wrong_track_never_reclaims_uid_even_when_it_returns_left():
    bank=bank_at_857()
    old=deepcopy(bank.identities[1].last_strong_observation)
    for row in ROWS:
        assert assign(bank,row)==0
        result=bank.last_assignments[5]
        assert result["identity_control_rejected"]
        assert result["bank_updated"] is False
        assert bank.identities[1].last_strong_observation==old
    assert not bank.pending_late_handoffs
    assert len(bank.identities[1].features)==1


def test_wrong_candidate_cannot_escape_by_getting_new_raw_track_numbers():
    bank=bank_at_857()
    for row,track in zip(ROWS[:4],(5,6,7,8)):
        assert assign(bank,row,track=track)==0
        assert bank.last_assignments[track]["identity_control_rejected"]
    assert set(bank._mapped_geometry_conflicts)=={5,6,7,8}


def test_correct_independent_track_can_reacquire_while_wrong_track_is_excluded():
    bank=bank_at_857()
    assert assign(bank,ROWS[0])==0
    for frame,cap,ts,expected in [(372,877,10389.82,0),(373,881,10390.02,1)]:
        assert assign(bank,(frame,cap,ts,ANCHOR,-66.329599,.08),track=6,opposite=False)==expected
    assert bank.track_to_uid[6]==1 and 5 not in bank.track_to_uid


def test_stale_anchor_without_measured_contradiction_still_allows_two_frame_observation():
    bank=bank_at_857()
    # No previously measured conflict: old geometry alone cannot reject a late person.
    a=list(ROWS[1]);a[0]=390
    b=list(ROWS[2]);b[0]=391;b[-1]=.10
    assert assign(bank,a,track=9)==0
    assert assign(bank,b,track=9)==1


@pytest.mark.parametrize("d,accepted",[(.15,True),(.199,True),(.201,False),(.291,False)])
def test_opposite_side_no_automatic_distance_relaxation(d,accepted):
    bank=bank_at_857()
    r=ROWS[1]
    meta=metadata(r[1],r[2],r[3],r[4],opposite=True)
    result=bank._preferred_search_reacquire_candidate(feature=feature(d),partial_feature=None,
        preferred_uid=1,candidate_ok=False,candidate_count=1,sample_metadata=meta)
    assert (result is not None)==accepted
    assert meta["reacquire_distance_limit"]==.20


def test_opposite_soft_fallback_does_not_reintroduce_point_three_threshold():
    bank=bank_at_857()
    row=list(ROWS[2]);row[0]=390
    assert assign(bank,row,track=9)==0
    assert bank.last_assignments[9]["reason"]=="preferred_search_reacquire_rejected"


def probation_bank():
    bank=bank_at_857()
    for frame,cap,ts in [(372,877,10389.82),(373,881,10390.02)]:
        assign(bank,(frame,cap,ts,ANCHOR,-66.329599,.10),track=6,opposite=False)
    assert bank.track_to_uid[6]==1 and bank._reacquire_quarantine.is_held(1)
    return bank


def probation_sample(bank,frame,d,*,cap=None,ts=None,extra=None,partial=None):
    return assign(bank,(frame,cap or frame*3,ts or (10390.02+.1*(frame-373)),
                       ANCHOR,-66.329599,d),track=6,search=False,opposite=False,
                       extra=extra,partial=partial)


def test_new_handoff_cannot_keep_uid_at_point_37_or_point_40():
    bank=probation_bank()
    old=deepcopy(bank.identities[1].last_strong_observation)
    for frame,d in [(374,.372),(375,.402),(376,.259),(377,.149)]:
        assert probation_sample(bank,frame,d)==0
        assert bank.last_assignments[6]["identity_control_rejected"]
        assert bank.identities[1].last_strong_observation==old
        assert not bank.last_assignments[6]["bank_updated"]
    assert probation_sample(bank,378,.12)==1
    assert bank.last_assignments[6]["reacquire_control_recovered"]


def test_repeated_capture_cannot_clear_reacquire_suspicion():
    bank=probation_bank()
    assert probation_sample(bank,374,.38)==0
    for frame in (375,376,377):
        assert probation_sample(bank,frame,.10,cap=1125,ts=10390.22)==0
    assert bank.last_assignments[6]["reacquire_control_recovery_streak"]==1
    assert probation_sample(bank,378,.10,cap=1128,ts=10390.32)==1


def test_valid_partial_descriptor_remains_separate_from_full_body_gate():
    bank=probation_bank()
    bank.identities[1].partial_features=[feature(0)]
    assert probation_sample(bank,374,.38,partial=feature(.05),extra={"partial_observation":True})==1
    # A good torso vector on a full-body observation is not permission to bypass.
    assert probation_sample(bank,375,.38,partial=feature(.05))==0


def test_stable_original_track_keeps_ordinary_mapped_verifier():
    bank=bank_at_857()
    assert assign(bank,(360,860,10388.9,ANCHOR,-66.329599,.38),search=False)==1
    assert not bank._reacquire_quarantine.is_held(1)


def test_camera_motion_explains_large_pixel_displacement_without_identity_hold():
    bank=bank_at_857()
    row=list(ROWS[0])
    shift=((row[3][0]+row[3][2])-(ANCHOR[0]+ANCHOR[2]))/1280
    row[4]=-66.329599-shift*bank.config.camera_hfov_deg
    assign(bank,row)
    assert not bank._mapped_geometry_conflicts


def test_bank_reset_clears_negative_and_probation_state():
    bank=probation_bank()
    probation_sample(bank,374,.38)
    assert bank._reacquire_control_suspects
    bank.reset()
    assert not bank._reacquire_control_suspects and not bank._mapped_geometry_conflicts


def test_match_diagnostics_report_top_three_without_changing_minimum_score():
    bank=bank_at_857()
    entry=bank.identities[1]
    entry.features=[feature(.1719),feature(.2059),feature(.2157)]
    evidence=entry.match_evidence(feature(0),.1)
    assert evidence["strong_template_count"]==3
    assert evidence["strong_distance"]==pytest.approx(.1719,abs=1e-6)
    assert evidence["strong_top3_mean_distance"]==pytest.approx((.1719+.2059+.2157)/3,abs=1e-6)
    assert entry.distance(feature(0))==pytest.approx(.1719,abs=1e-6)


def test_suspicion_cannot_be_cleared_by_widely_spaced_strong_frames():
    bank=probation_bank()
    assert probation_sample(bank,374,.38)==0
    assert probation_sample(bank,375,.10)==0
    assert probation_sample(bank,376,.10,ts=10391.)==0
    assert bank.last_assignments[6]["reacquire_control_recovery_streak"]==1


def test_competing_identity_cannot_clear_post_handoff_suspicion():
    bank=probation_bank()
    assert probation_sample(bank,374,.38)==0
    assert probation_sample(bank,375,.10)==0
    assert probation_sample(bank,376,.10,extra={"identity_competition":{"passed":False}})==0
    assert bank.last_assignments[6]["reacquire_control_recovery_streak"]==0
