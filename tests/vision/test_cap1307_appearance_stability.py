"""Recorded CAP1304/1307/1309 geometry, synthetic cosine distances.

This tests policy, not an RKNN embedding rerun or a motor trajectory.
"""
from copy import deepcopy
import math

import numpy as np
import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from rk_vision.template_memory import TemplateMemory
from test_cap874_identity_reacquire import feature, metadata

BOX1141 = (342.7001648, 29.4058228, 553.8286133, 476.2197266)
ROWS = {
    1304: (645, 9831.855375772, (242.3813782,93.2716370,339.3453369,476.1178589),214.2159383),
    1307: (646, 9831.996028954, (250.7648621,90.7164001,340.1360168,475.0222778),217.4110874),
    1309: (647, 9832.092883216, (249.4526825,92.6660156,339.4591980,477.1339722),219.9303521),
    1311: (648, 9832.223017669, (249.5129852,93.7824860,340.5636902,477.9057617),218.8507342),
}


def meta(cap):
    frame, stamp, box, yaw = ROWS[cap]
    return dict(metadata(cap, stamp, box, yaw, search=False), frame_index=frame,
        track_id=5, image_width=640, image_height=480, partial_feature_source='osnet_torso',
        partial_observation=True, detector_edge_touch_count=1)


def setup(two_templates=False):
    b = IdentityBank(IdentityBankConfig(template_memory_enable=True,
        template_crosscheck_enable=True, appearance_region_safety_enable=True,
        mapped_verify_threshold=.45, partial_match_threshold=.45,
        controlled_handoff_enable=True, camera_hfov_deg=60.))
    m = dict(meta(1304), capture_frame_id=961, capture_timestamp=9813.951710226)
    b._create_identity(feature(0), 466, m, feature(0))
    if two_templates:
        t = dict(m, capture_frame_id=1141, capture_timestamp=9823.371275443,
                 detector_bbox=list(BOX1141))
        b.identities[1].template_memory.remember(np.array([.8,.6,0.]), t, 'partial')
    b._bind_reacquired_identity(1, 5, 644, dict(meta(1304), capture_frame_id=1303,
        capture_timestamp=9831.75))
    return b


def torso_pair(d961, d1141):
    x = 1-d961; y = (1-d1141-.8*x)/.6
    return np.asarray([x,y,math.sqrt(1-x*x-y*y)], dtype=np.float32)


def send(b, cap, part=.25, full=.15, count=1, **changes):
    m = dict(meta(cap), **changes)
    box = m['detector_bbox']
    return b.assign(track_id=m['track_id'], feature=feature(full),
        partial_feature=feature(part) if isinstance(part, (int,float)) else part,
        confidence=.86, area=(box[2]-box[0])*(box[3]-box[1]), frame_index=m['frame_index'],
        candidate_count=count, bbox_quality_ok=True, bbox_quality_tier='strong', sample_metadata=m)


def test_cap1307_template_set_no_longer_flaps_at_shape_boundary():
    b = setup(True)
    assert send(b,1304,torso_pair(.32,.2528619)) == 1
    before = deepcopy(b.identities[1].template_memory.last_learning)
    # .27 to CAP1141 is synthetic: actual post-filter distance wasn't logged.
    assert send(b,1307,torso_pair(.3449727,.27),full=.1542648) == 1
    a = b.last_assignments[5]
    ev = a['reacquire_recent_partial_evidence']
    assert ev['count'] == 2 and ev['shape_hysteresis_caps'] == [1141]
    assert ev['winner_cap'] == 1141 and ev['distance'] == pytest.approx(.27)
    assert not ev['winner_crop_shape_consistent']  # strict diagnostic stays honest
    assert not a['bank_updated']
    assert b.identities[1].template_memory.last_learning == before
    assert send(b,1309,torso_pair(.336122,.28)) == 1
    assert send(b,1311,torso_pair(.32,.2794402)) == 1
    assert b.last_assignments[5]['reacquire_recent_partial_evidence']['shape_hysteresis_caps'] == []


def test_small_torso_excursion_is_one_stationary_recheck_not_identity_permission():
    b = setup(); assert send(b,1304) == 1
    old = deepcopy(b.identities[1].last_strong_observation)
    learning = deepcopy(b.identities[1].template_memory.last_learning)
    assert send(b,1307,.3449727) == 0
    a = b.last_assignments[5]
    assert a['identity_recheck_pending'] and a['identity_control_rejected']
    assert a['reason'] == 'partial_boundary_recheck' and not a['bank_updated']
    assert a['identity_recheck_deadline'] == ROWS[1307][1]+.25
    assert b.identities[1].last_strong_observation == old
    assert b.identities[1].template_memory.last_learning == learning
    assert b._reacquire_quarantine.is_held(1)
    assert send(b,1309,.336122) == 1
    assert not b.last_assignments[5].get('identity_recheck_pending')


def test_persistent_borderline_cannot_repeatedly_pause_loss_or_renew_reference():
    b = setup(); send(b,1304)
    assert send(b,1307,.345) == 0
    assert send(b,1309,.345) == 0
    assert b.last_assignments[5]['reason'] == 'partial_evidence_tentative'
    assert not b.last_assignments[5].get('identity_recheck_pending')
    assert send(b,1311,.345) == 0
    assert not b.last_assignments[5].get('identity_recheck_pending')


@pytest.mark.parametrize('case', ['duplicate','older','late','search','other_track',
    'suspect','competition','unknown','conflict','large_tentative','weak_full','geometry','no_yaw','crowd'])
def test_unsafe_or_noncontinuous_evidence_cannot_request_boundary_recheck(case):
    b = setup(); send(b,1304)
    changes = {}; part=.345; full=.15
    if case == 'duplicate': changes.update(capture_frame_id=1304, capture_timestamp=ROWS[1304][1])
    if case == 'older': changes['capture_timestamp']=ROWS[1304][1]-.01
    if case == 'late': changes['capture_timestamp']=ROWS[1304][1]+.251
    if case == 'search': changes['search_reacquire_context_active']=True
    if case == 'other_track': changes['track_id']=6
    if case == 'suspect': b._reacquire_control_suspects[1]={'track_id':5,'reason':'partial_conflict','streak':0}
    if case == 'competition': changes['identity_competition']={'uid':1,'frame_index':646,'passed':False}
    if case == 'unknown': part=None
    if case == 'conflict': part=.46
    if case == 'large_tentative': part=.37
    if case == 'weak_full': full=.25
    if case == 'geometry': changes.update(detector_bbox=[0,90,89,475],detector_center_x_ratio=.07)
    if case == 'no_yaw': changes['integrated_yaw_deg']=None
    if case == 'crowd': changes['count']=2  # metadata's stale count=1 cannot override tracker count
    assert send(b,1307,part,full,**changes) == 0
    assert not b.last_assignments[changes.get('track_id',5)].get('identity_recheck_pending')


def test_duplicate_pending_frame_cannot_extend_recheck_deadline():
    b=setup(); send(b,1304); send(b,1307,.345)
    assert b.last_assignments[5]['identity_recheck_pending']
    send(b,1307,.345)
    assert not b.last_assignments[5].get('identity_recheck_pending')


@pytest.mark.parametrize('case', ['no_reference','other_coverage','too_narrow','too_old','unadmitted','other_track'])
def test_template_hysteresis_never_admits_unqualified_template(case):
    b=setup(True); send(b,1304,torso_pair(.32,.25))
    m=meta(1307); ref=deepcopy(b._appearance_verified[1]); memory=b.identities[1].template_memory
    if case=='no_reference': ref=None
    if case=='other_coverage': m['detector_bbox'][1]=0
    if case=='too_narrow': m['detector_bbox'][2]=m['detector_bbox'][0]+80
    if case=='too_old': m['capture_timestamp']+=1
    if case=='unadmitted': ref['comparable_caps']=[]
    if case=='other_track': m['track_id']=9
    result=memory.evidence(torso_pair(.345,.27),m,'partial',reliable_only=True,
                           comparable_only=True,shape_reference=ref)
    assert 1141 not in result['comparable_caps']


def test_queries_do_not_renew_template_or_positive_observation():
    b=setup(True); send(b,1304,torso_pair(.32,.25))
    memory=b.identities[1].template_memory
    prior=deepcopy(b._appearance_verified[1]); learning=deepcopy(memory.last_learning)
    for _ in range(3):
        memory.evidence(torso_pair(.345,.27),meta(1307),'partial',reliable_only=True,
                        comparable_only=True,shape_reference=b._appearance_verified[1])
    assert b._appearance_verified[1] == prior and memory.last_learning == learning


def test_reset_drops_hysteresis_and_pending_budget():
    b=setup(); send(b,1304); send(b,1307,.345)
    assert b._appearance_verified
    b.reset()
    assert not b._appearance_verified


def test_missing_capture_cannot_crash_or_refresh_positive_reference():
    b=setup(); send(b,1304)
    old=deepcopy(b._appearance_verified)
    send(b,1307,.25,capture_frame_id=None)
    assert not b._appearance_verified or b._appearance_verified == old
