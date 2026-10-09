"""Policy regression with synthetic distances, not a model-accuracy test."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from rk_vision.template_memory import TemplateMemory
from test_reacquire_crosscheck import bank, send, meta


def trial():
    b = bank()
    b.config = replace(b.config, partial_match_threshold=.45)
    return b


def observe(b, cap, *, x=.6, part=.39, track=4, **extra):
    box = (x*640-75, 40., x*640+75, 460.)
    data = dict(partial_observation=True, search_direction="right",
                search_direction_compatible=x >= .45, detector_bbox=box,
                detector_center_x_ratio=x, detector_area_ratio=150*420/(640*480))
    data.update(extra)
    return send(b, cap, 8+(cap-20)*.1, full=.30, part=part, track=track,
                search=True, extra=data)


def test_grey_zone_observes_without_uid_or_learning_then_crossing_can_confirm():
    b = trial(); old = deepcopy(b.identities[1].last_strong_observation)
    for cap, x in [(20,.65),(21,.5),(22,.4)]:
        assert observe(b, cap, x=x) == 0
        a = b.last_assignments[4]
        assert a['reason'] == 'partial_evidence_tentative'
        assert not a['bank_updated']
        assert b.identities[1].last_strong_observation == old
    assert observe(b,23,x=.35,part=.30)==0
    assert observe(b,24,x=.30,part=.29)==1
    assert not b.last_assignments[4]['bank_updated']
    assert b._reacquire_quarantine.is_held(1)


def test_same_candidate_continues_after_partial_reacquire_without_learning():
    b=trial()
    assert observe(b,20)==0
    assert observe(b,21,x=.4,part=.30)==0
    assert observe(b,22,x=.3,part=.29)==1
    assert observe(b,23,x=.25,part=.29)==1
    assert not b.last_assignments[4]['bank_updated']


@pytest.mark.parametrize('part',[.35,.373,.396,.421,.449])
def test_cap1161_pattern_cannot_confirm_by_repeating_grey_zone(part):
    b=trial()
    for cap in range(20,30):
        assert send(b,cap,8+.1*(cap-20),full=.144,part=part,track=4,
                    search=True,extra={'partial_observation':True,'search_direction':'right'})==0
        assert b.last_assignments[4]['reason']=='partial_evidence_tentative'
        assert not b.last_assignments[4]['bank_updated']


def test_point_46_remains_conflict():
    b=trial(); assert observe(b,20,part=.46)==0
    assert b.last_assignments[4]['reason']=='recent_partial_conflict'


@pytest.mark.parametrize('changes',[
    {'is_fresh':False}, {'integrated_yaw_deg':None},
    {'candidate_count':2,'candidate_score_gap':0.},
    {'identity_competition':{'passed':False}},
    {'search_direction':'left'}, {'capture_timestamp':9.},
])
def test_invalid_or_interrupted_tracklet_cannot_carry_cross_side_permission(changes):
    b=trial(); assert observe(b,20)==0
    assert observe(b,21,x=.4,**changes)==0
    assert observe(b,22,x=.3,part=.29)==0
    assert observe(b,23,x=.25,part=.29)==0


def test_new_raw_id_does_not_inherit_crossing():
    b=trial();observe(b,20)
    for cap in (21,22,23):
        assert observe(b,cap,x=.3,part=.29,track=5)==0


def test_initial_opposite_side_candidate_never_gets_partial_exception():
    b=trial()
    for cap in (20,21,22):
        assert observe(b,cap,x=.3,part=.26)==0


def test_duplicate_cannot_create_observation_or_confirmation():
    b=trial();observe(b,20)
    assert observe(b,21,x=.4,part=.29)==0
    assert observe(b,21,x=.4,part=.29)==0
    assert b.last_assignments[4].get('candidate_observation') is None
    assert not b.last_assignments[4]['bank_updated']


def test_reset_discards_untrusted_tracklets():
    b=trial();observe(b,20)
    assert b._candidate_observations.rows
    b.reset();assert not b._candidate_observations.rows


def test_diverse_coverage_survives_redundant_recent_views_but_expires():
    m=TemplateMemory(capacity=3)
    def info(cap, top):
        return dict(meta(cap,float(cap)),image_width=640,image_height=480,
                    detector_bbox=[200.,top,400.,470.])
    m.remember(np.eye(5)[0],info(1,0.),'partial')
    for cap in range(2,9):
        m.remember(np.eye(5)[1+(cap%4)],info(cap,40.),'partial')
    assert len(m.recent['partial'])==3
    assert any(row[1]['capture_frame_id']==1 for row in m.recent['partial'])
    before=deepcopy(m.last_learning)
    m.evidence(np.eye(5)[0],info(10,0.),'partial')
    assert m.last_learning==before
    m.advance(info(40,0.))
    assert not m.recent['partial']


def test_coverage_is_diagnostic_not_anatomical_identity_or_conflict_override():
    m=TemplateMemory()
    a=dict(meta(1,1),image_width=640,image_height=480,detector_bbox=[200.,0.,400.,479.])
    q=dict(meta(2,2),image_width=640,image_height=480,detector_bbox=[200.,50.,400.,479.])
    m.remember(np.array([1.,0.]),a,'partial')
    e=m.evidence(np.array([0.,1.]),q,'partial',reliable_only=True)
    # A broad crop with only a vertical-border difference is now comparable;
    # this does not turn its actual conflicting descriptor into a match.
    assert e['comparable_count']==1 and e['distance']==1.
    assert e['comparison_mode']=='vertical_border_bridge'
    assert e['query_coverage']!=e['winner_coverage']


def test_runtime_threshold_is_045(monkeypatch):
    import os
    from pathlib import Path
    from car_control_modular.config_loader import load_config_to_env
    from rk_vision.pipeline import RKNNVisionConfig
    monkeypatch.setattr(os,'environ',os.environ.copy())
    load_config_to_env(str(Path(__file__).resolve().parents[2]/'car_control_modular/config/reid_runtime.ini'))
    assert RKNNVisionConfig.from_env().identity_partial_match_threshold==.45


@pytest.mark.parametrize('scene',['cap874','cap714','cap2394'])
def test_previous_wrong_people_remain_rejected_with_trial_045(scene):
    if scene=='cap874':
        from test_cap874_identity_reacquire import bank_at_857, assign, ROWS
        b=bank_at_857(); run=lambda row:assign(b,row)
    elif scene=='cap714':
        from test_cap714_reacquire_geometry import seeded_bank, send as submit, ROWS
        b=seeded_bank();run=lambda row:submit(b,row)
    else:
        from test_cap2394_handoff_conflict import bank as setup, send as submit, ROWS
        b=setup();run=lambda row:submit(b,row)
    b.config=replace(b.config,partial_match_threshold=.45)
    for row in ROWS:
        assert run(row)==0
        assert all(not a.get('bank_updated') for a in b.last_assignments.values()
                   if a.get('uid')==0)


def test_crossing_observation_has_total_lifetime_not_just_rolling_timeout():
    b=trial();observe(b,20)
    for cap in range(21,51):
        assert observe(b,cap,x=.4)==0
    assert observe(b,51,x=.35,part=.29)==0
    assert observe(b,52,x=.3,part=.29)==0


def test_same_raw_id_large_jump_cannot_inherit_crossing():
    b=trial();observe(b,20,x=.8)
    assert observe(b,21,x=.1,part=.29)==0
    assert observe(b,22,x=.15,part=.29)==0


def test_continuation_cannot_ignore_new_torso_conflict():
    b=trial();observe(b,20)
    assert observe(b,21,x=.4,part=.30)==0
    assert observe(b,22,x=.3,part=.29)==1
    assert observe(b,23,x=.25,part=.46)==0
    assert not b.last_assignments[4]['bank_updated']
