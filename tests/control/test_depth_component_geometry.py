"""Seed-filled torso regressions missed by the earlier small-object tests."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from car_control_modular.depth_track_shadow import CameraPose,DepthTrackShadow,Intrinsics

K=Intrinsics(160,120,140,140,80,60)


def body(width=60,shift=0):
    d=np.full((120,160),5000,dtype=np.uint16)
    d[15:105,80-width//2+shift:80+width//2+shift]=2000
    return d


def seed(t,d,roi=(68,40,92,80)):
    return t.seed(d,timestamp=10.,pose=CameraPose(10.,0,0,0),uid=1,visual_timestamp=10.,
                  torso_bbox=roi,trusted_distance_m=2.,now=10.,identity_confirmed=True)


def update(t,d,dt):
    return t.update(d,timestamp=10+dt,pose=CameraPose(10+dt,0,0,0),now=10+dt,active_uid=1)


@pytest.mark.parametrize('width',[36,48,60,72])
def test_static_body_larger_than_seed_roi_does_not_grow_or_drift(width):
    d=body(width); before=d.copy();t=DepthTrackShadow(K);first=seed(t,d)
    assert first.status=='seeded'
    for i in range(1,11):
        r=update(t,d,i/30)
        assert r.status=='tracked'
        assert r.camera_xyz==pytest.approx(first.camera_xyz)
        c=r.diagnostics['candidates'][0]
        assert c['area_ratio']==pytest.approx(1.)
        assert c['component_area_px']>c['measurement_support_px']
        assert not r.control_allowed
    assert np.array_equal(before,d)
    assert t.visual_ts==10.


@pytest.mark.parametrize('dt',[1/30,.05,.10,.17])
def test_search_window_size_is_not_measurement_size(dt):
    t=DepthTrackShadow(K);first=seed(t,body());r=update(t,body(),dt)
    assert r.status=='tracked'
    assert r.diagnostics['selector_roi']!=first.diagnostics['selector_roi']
    assert r.diagnostics['candidates'][0]['measurement_roi']==first.diagnostics['candidates'][0]['measurement_roi']
    assert r.camera_xyz==pytest.approx(first.camera_xyz)


@pytest.mark.parametrize('direction',[-1,1])
def test_core_translates_with_person_not_locked_to_old_pixels(direction):
    t=DepthTrackShadow(K);a=seed(t,body())
    for i in range(1,8):
        r=update(t,body(shift=direction*2*i),i/30)
        assert r.status=='tracked'
        assert r.camera_xyz[0]-a.camera_xyz[0]==pytest.approx(direction*2*i*2/140)


def test_leg_depth_changes_do_not_move_torso_measurement():
    d=body();t=DepthTrackShadow(K);a=seed(t,d)
    d[85:105,50:110]=2200
    r=update(t,d,1/30)
    assert r.status=='tracked' and r.camera_xyz==pytest.approx(a.camera_xyz)
    assert r.diagnostics['candidates'][0]['depth_spread_m']==0


def test_seed_connected_to_frame_wide_surface_is_rejected():
    d=body();d[100:103,:]=2000
    t=DepthTrackShadow(K);r=seed(t,d)
    assert r.status=='seed_ambiguous_or_empty' and t.position is None
    assert r.diagnostics['candidates'][0]['reject_reasons']==['seed_background_extent']


def test_wall_replacing_body_still_rejected_without_relaxing_growth_limit():
    t=DepthTrackShadow(K);seed(t,body(width=36))
    r=update(t,np.full((120,160),2000,dtype=np.uint16),1/30)
    assert r.status=='association_ambiguous_or_lost'
    assert 'area_growth' in r.diagnostics['candidates'][0]['reject_reasons']
    assert r.diagnostics['area_growth_limit']==2.5


def test_gradual_expansion_cannot_ratchet_seed_area_allowance():
    t=DepthTrackShadow(K);seed(t,body(width=20),roi=(60,15,100,105))
    for i,w in enumerate([30,40,50],1):assert update(t,body(width=w),i*.06).status=='tracked'
    r=update(t,body(width=60),.24)
    assert r.status=='association_ambiguous_or_lost'
    c=r.diagnostics['candidates'][0]
    assert c['area_ratio']<2.5 and c['seed_area_ratio']>2.5
    assert 'seed_area_growth' in c['reject_reasons']


def test_two_bodies_in_selector_are_not_collapsed_to_one_measurement():
    t=DepthTrackShadow(K);seed(t,body())
    d=body();d[:,79:81]=5000
    r=update(t,d,.1)
    assert r.status=='association_ambiguous_or_lost' and r.candidate_count==2
    assert len(r.diagnostics['candidates'])==2


def test_large_unrelated_component_outside_selector_not_selected():
    d=body(width=20);d[20:105,140:160]=2000
    t=DepthTrackShadow(K);r=seed(t,d)
    assert r.status=='seeded'
    assert r.diagnostics['eligible_components']==1
    assert r.diagnostics['candidates'][0]['component_bbox']==[70,15,90,105]


def test_fragment_budget_is_bounded_and_diagnostic():
    d=np.full((120,160),5000,dtype=np.uint16)
    for y in [20,32,44]:
        for x in [20,32,44]:d[y:y+8,x:x+8]=2000
    t=DepthTrackShadow(K)
    assert t._candidates(d,(15,15,60,60),2.,CameraPose(10.,0,0,0),20)==[]
    assert t._diagnostics['scan_rejection']=='component_budget_exceeded'


def test_core_spread_and_world_step_rejections_have_separate_reasons():
    t=DepthTrackShadow(K);seed(t,body())
    d=body();d[40:60,50:110]=1760;d[60:80,50:110]=2240
    r=update(t,d,.033)
    assert r.status=='association_ambiguous_or_lost'
    assert r.diagnostics['candidates'][0]['reject_reasons']==['depth_spread']
    t=DepthTrackShadow(K);seed(t,body())
    r=update(t,body(shift=20),.033)
    assert r.status=='association_ambiguous_or_lost'
    assert 'world_step' in r.diagnostics['candidates'][0]['reject_reasons']


def test_diagnostic_snapshot_not_mutated_by_later_update():
    t=DepthTrackShadow(K);r=seed(t,body());saved=json.dumps(r.diagnostics,allow_nan=False)
    update(t,body(shift=1),.033)
    assert json.dumps(r.diagnostics,allow_nan=False)==saved


def test_spatial_check_explicitly_uses_synthetic_time_and_does_not_claim_replay(tmp_path):
    from tools.depth_shadow_spatial_check import report
    metadata=dict(rgb_size=[160,120],anchor_bbox_rgb=[47,20,113,107],
                  baseline_distance_m=2.,capture_frame_id=194)
    np.savez(tmp_path/'sample_000.npz',depth_mm=body(),metadata=json.dumps(metadata))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(hfov_deg=60)))
    r=report(tmp_path)
    assert r['mode']=='same_image_synthetic_time_zero_motion' and r['not_a_temporal_replay']
    assert r['passed']==1 and not r['results'][0]['motor_authority_created']


def test_online_rows_and_metrics_expose_specific_rejection(tmp_path):
    from tools.depth_track_online_metrics import report
    t=DepthTrackShadow(K);d=body();d[100:103,:]=2000;r=seed(t,d)
    row=dict(phase='seed',status=r.status,uid=1,association_diagnostics=r.diagnostics)
    (tmp_path/'observations.jsonl').write_text(json.dumps(row)+'\n')
    data=report(tmp_path)
    assert data['candidate_rejection_counts']=={'seed_background_extent':1}
    assert data['distinct_tracked_samples']==0


def test_online_engine_preserves_new_geometry_audit():
    from test_depth_track_online import session,anchor
    s=session();rows=s.process(anchor(),(1,10.,2.),10.14)
    assert rows and all(r['association_diagnostics']['component_stats_space']=='full_depth_grid' for r in rows)
    assert all(not r['control_allowed'] and not r['geometry_verified'] for r in rows)
