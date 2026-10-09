"""CAP1167 policy regressions: recorded boxes, synthetic cosine descriptors.

No RKNN/model replay, camera, serial port, or motor commands.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from rk_vision.template_memory import TemplateMemory
from test_cap874_identity_reacquire import feature, metadata

TEMPLATE = [104.7378845, 10.9239655, 362.1352539, 475.9024048]
BOXES = {
    1160: [223.1473389,55.0072937,365.3406677,476.5067139],
    1161: [249.7251434,53.7355042,389.3731689,474.7618408],
    1164: [287.0202332,58.6654358,433.6796265,473.5271606],
    1166: [319.8871765,56.6999817,463.6335449,474.9732666],
    1167: [341.5568237,58.1633911,470.3696899,473.5674438],
    1169: [363.8190002,54.9396820,471.9653625,475.0385742],
    1170: [365.9447937,55.5282898,477.9597473,472.7634277],
    1171: [371.6730042,52.2943115,497.3327942,475.1187744],
}


def meta(cap, stamp, box, direction='left', **extra):
    m = dict(metadata(cap, stamp, box, search=True), frame_index=cap, track_id=9,
             image_width=640, image_height=480, partial_feature_source='osnet_torso',
             partial_observation=True, detector_edge_touch_count=1,
             search_direction=direction,
             search_direction_compatible=((box[0]+box[2])/1280 <= .55 if direction=='left'
                                          else (box[0]+box[2])/1280 >= .45))
    m.update(extra)
    return m


def bank():
    b = IdentityBank(IdentityBankConfig(template_memory_enable=True, template_crosscheck_enable=True,
        appearance_region_safety_enable=True, partial_match_threshold=.45,
        partial_confirm_threshold=.40, controlled_handoff_enable=True,
        mapped_verify_threshold=.45, camera_hfov_deg=60.))
    b._create_identity(feature(0),702,meta(702,1.,TEMPLATE),feature(0))
    return b


def send(b,cap,part=.30,full=.23,direction='left',**extra):
    m=meta(cap,25.+(cap-1160)*.05,BOXES.get(cap,BOXES[1171]),direction,**extra)
    return b.assign(track_id=m['track_id'],feature=feature(full),partial_feature=feature(part),
        confidence=.92,area=60000,frame_index=cap,bbox_quality_ok=True,bbox_quality_tier='strong',
        sample_metadata=m,preferred_uid=1,preferred_candidate_ok=True)


def confirmed():
    b=bank()
    assert send(b,1160)==0
    assert send(b,1161)==1
    assert b._candidate_observations.rows[(1,9)]['late_confirmed_source']=='partial'
    return b


def test_direction_change_preserves_confirmed_partial_source_not_crossing():
    b=confirmed(); old=deepcopy(b.identities[1].template_memory.last_learning)
    assert send(b,1164)==1
    assert send(b,1166,direction='right')==1
    a=b.last_assignments[9]
    assert a['reason']=='mapped_late_continuation' and a['match_source']=='partial'
    row=b._candidate_observations.rows[(1,9)]
    assert row['count']==1 and not row['crossed']
    assert b.identities[1].template_memory.last_learning==old
    assert b._reacquire_quarantine.is_held(1)


def test_recorded_side_narrowing_keeps_only_previously_verified_template():
    b=confirmed(); old=deepcopy(b.identities[1].template_memory.last_learning)
    for cap in (1164,1166,1167,1169,1170,1171):
        assert send(b,cap,direction='right' if cap>=1166 else 'left')==1
        a=b.last_assignments[9]
        if cap in (1169,1170):
            ev=a['reacquire_recent_partial_evidence']
            assert ev['shape_hysteresis_caps']==[702]
            assert ev['winner_crop_shape_consistent'] is False
        assert not a['bank_updated']
    assert b.identities[1].template_memory.last_learning==old


@pytest.mark.parametrize('part',[.353,.365,.386,.366,.399])
def test_requested_threshold_accepts_current_torso_without_learning(part):
    b=confirmed()
    assert send(b,1164,part=part)==1
    assert b.last_assignments[9]['reacquire_partial_confirm_limit']==.40
    assert not b.last_assignments[9]['bank_updated']


@pytest.mark.parametrize('part,reason',[(.401,'partial_evidence_tentative'),(.46,'recent_partial_conflict')])
def test_tentative_and_conflicting_torso_still_revoke_identity(part,reason):
    b=confirmed()
    assert send(b,1164,part=part)==0
    assert b.last_assignments[9]['reason']==reason
    assert not b.last_assignments[9]['bank_updated']


@pytest.mark.parametrize('kind',['duplicate','stale','gap','new_track','competition','jump','conflict','weak_full'])
def test_partial_proof_does_not_bypass_invalid_or_conflicting_evidence(kind):
    b=confirmed(); kw={}; cap=1164
    if kind=='duplicate':kw.update(capture_frame_id=1161,capture_timestamp=25.05)
    if kind=='stale':kw['is_fresh']=False
    if kind=='gap':kw['capture_timestamp']=26.
    if kind=='new_track':kw['track_id']=10
    if kind=='competition':kw['identity_competition']=dict(uid=1,frame_index=cap,passed=False)
    if kind=='jump':kw['detector_bbox']=[550.,230.,610.,410.]
    if kind=='conflict':
        b._mapped_geometry_conflicts[9]=dict(uid=1,search_contradiction=True,
            reference=deepcopy(b.identities[1].last_strong_observation))
    if kind=='weak_full':kw['full']=.42
    assert send(b,cap,direction='right',**kw)==0
    assert not b.last_assignments[kw.get('track_id',9)]['bank_updated']


def test_new_side_view_without_confirmed_partial_proof_cannot_use_shape_retention():
    b=bank()
    for cap in (1169,1170):
        assert send(b,cap,direction='right')==0
        assert b.last_assignments[9]['reason']=='secondary_evidence_unavailable'


def test_runtime_config_wires_point_four_to_identity_bank(monkeypatch):
    import os
    from car_control_modular.config_loader import load_config_to_env
    from rk_vision.pipeline import RKNNVisionConfig
    from rk_vision.tracker import DeepSortTrackerConfig, DeepSortTracker
    monkeypatch.setattr(os,'environ',os.environ.copy())
    load_config_to_env(str(Path(__file__).resolve().parents[2]/'car_control_modular/config/reid_runtime.ini'))
    cfg=RKNNVisionConfig.from_env()
    assert cfg.identity_partial_confirm_threshold==.40
    tracker=DeepSortTracker(DeepSortTrackerConfig(identity_partial_confirm_threshold=cfg.identity_partial_confirm_threshold,
        identity_partial_match_threshold=cfg.identity_partial_match_threshold))
    assert tracker.identity_bank.config.partial_confirm_threshold==.40
    assert tracker.identity_bank.config.partial_match_threshold==.45


@pytest.mark.parametrize('scene',['cap874','cap714','cap2394'])
def test_recorded_wrong_person_geometry_stays_rejected_at_new_threshold(scene):
    if scene=='cap874':
        from test_cap874_identity_reacquire import bank_at_857,assign,ROWS
        b=bank_at_857();run=lambda row:assign(b,row)
    elif scene=='cap714':
        from test_cap714_reacquire_geometry import seeded_bank,send as submit,ROWS
        b=seeded_bank();run=lambda row:submit(b,row)
    else:
        from test_cap2394_handoff_conflict import bank as setup,send as submit,ROWS
        b=setup();run=lambda row:submit(b,row)
    b.config=replace(b.config,partial_match_threshold=.45,partial_confirm_threshold=.40)
    for row in ROWS:
        assert run(row)==0
        assert all(not a.get('bank_updated') for a in b.last_assignments.values() if a.get('uid')==0)


@pytest.mark.parametrize('change', ['abrupt_shape','too_thin','side_crop','height','new_template','expired'])
def test_side_retention_is_not_blanket_comparability(change):
    m=TemplateMemory();t=meta(702,1.,TEMPLATE)
    m.remember(feature(0),t,'partial')
    p=meta(1167,25.35,BOXES[1167]);q=meta(1169,25.45,BOXES[1169])
    ref={'metadata':p,'comparable_caps':[702],'partial_continuation':True}
    if change=='abrupt_shape':p['detector_bbox']=[300.,58.,470.,474.]
    if change=='too_thin':q['detector_bbox']=[364.,55.,466.,475.]
    if change=='side_crop':q['detector_bbox']=[0.,55.,108.,475.]
    if change=='height':q['detector_bbox']=[364.,130.,455.,475.]
    if change=='new_template':ref['comparable_caps']=[]
    if change=='expired':q['capture_timestamp']=31.1
    assert not m.evidence(feature(.30),q,'partial',comparable_only=True,shape_reference=ref)['count']
