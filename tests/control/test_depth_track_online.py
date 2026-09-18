"""Online shadow tests use arrays and frozen cache records, never hardware."""
from dataclasses import replace
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from car_control_modular.control_types import DepthTargetObservation, SteeringFeedback
from car_control_modular.depth_track_online import (
    Anchor, AssociationSession, DepthTrackOnlineObserver, EncoderPoseHistory, MODEL,
)


def feedback(t, **kw):
    return replace(SteeringFeedback(timestamp=t, trustworthy=True), **kw)


def image(x=80, z=2000):
    d = np.full((120, 160), 5000, dtype=np.uint16)
    d[40:80, x-10:x+10] = z
    return d


def anchor(t=10., uid=1, cap=1, generation=0):
    return Anchor(generation, DepthTargetObservation((40.,20.,120.,100.), uid, 1, cap, t), 160,120)


def session(end=10.14):
    s = AssociationSession(60, .816814)
    for t in np.arange(9.98, end+.02, .02):
        s.poses.add(feedback(float(t)), float(t))
    s.add_frames([(10+i/30, image()) for i in range(5)])
    return s


def test_one_visual_anchor_tracks_several_new_depths_without_yolo():
    s = session()
    records = s.process(anchor(), (1,10.,2.),10.14)
    assert records[0]['status']=='seeded'
    updates = [r for r in records if r['phase']=='update']
    assert len(updates)==4 and all(r['status']=='tracked' for r in updates)
    assert all(not r['control_allowed'] and not r['motor_authority_created'] for r in records)
    assert all(not r['geometry_verified'] and r['geometry_model']==MODEL for r in records)
    assert s.tracker.visual_ts==10.
    assert s.process(anchor(), (1,10.,2.),10.14)==[]


def test_historical_replay_is_not_counted_as_live_gap_filling():
    s=session(10.30)
    records=s.process(anchor(),(1,10.,2.),10.30)
    replay=[r for r in records if r['phase']=='replay']
    assert len(replay)==3
    assert all(not r['potential_range_gap_fill'] for r in replay)
    live=[r for r in records if r['phase']=='update']
    assert len(live)==1 and live[0]['potential_range_gap_fill']


def test_depth_does_not_renew_visual_clock():
    s=session()
    s.process(anchor(),(1,10.,2.),10.14)
    rows=s.process(anchor(),(1,10.3,2.),10.351)
    assert rows[0]['status']=='visual_lease_expired'
    assert s.tracker is None


@pytest.mark.parametrize('evidence,status', [
    (None,'seed_wait_range'), ((2,10.,2.),'seed_range_identity_or_value'),
    ((1,10.,float('nan')),'seed_range_identity_or_value'),
    ((1,9.7,2.),'seed_alignment_or_age'),
    ((1,10.4,2.),'seed_alignment_or_age'),
])
def test_missing_unrelated_or_stale_seed_range_rejected(evidence,status):
    s=session()
    assert s.process(anchor(),evidence,10.14)[0]['status']==status
    assert s.tracker is None


def test_pose_is_not_assumed_stationary_when_missing():
    s=session()
    s.poses.samples.clear()
    assert s.process(anchor(),(1,10.,2.),10.14)[0]['status']=='seed_pose_unbracketed'


def test_grid_mismatch_rejected():
    s=session()
    assert s.process(replace(anchor(),width=640),(1,10.,2.),10.14)[0]['status']=='seed_grid_mismatch'


def test_ambiguous_seed_does_not_pick_the_nearest_person():
    s=session()
    s.frames.clear()
    d=image(); d[:,79:81]=5000
    s.add_frames([(10.,d)])
    assert s.process(anchor(),(1,10.,2.),10.14)[0]['status']=='seed_ambiguous_or_empty'
    assert s.tracker.position is None


def test_explicit_revocation_clears_tracking():
    s=session(); s.process(anchor(),(1,10.,2.),10.14)
    assert s.process(None,(1,10.,2.),10.14)==[]
    assert s.tracker is None and s.anchor_key is None


def test_new_feedback_segment_cannot_reuse_old_world_coordinates():
    s=session(); s.process(anchor(),(1,10.,2.),10.14)
    s.poses.add(feedback(10.30),10.30)
    assert s.process(anchor(),(1,10.,2.),10.30)[0]['status']=='seed_pose_unbracketed'
    assert s.tracker is None


def test_feedback_bracket_interpolation_and_wheel_geometry():
    p=EncoderPoseHistory(.816814)
    p.add(feedback(10.,left_forward_rpm=60,right_forward_rpm=60),10.)
    p.add(feedback(10.1,left_forward_rpm=60,right_forward_rpm=60),10.1)
    assert p.at(10.05).z_m==pytest.approx(.0408407)
    assert p.at(10.101) is None and p.at(9.999) is None
    n=len(p.samples); p.add(feedback(10.1),10.1)
    assert len(p.samples)==n


@pytest.mark.parametrize('change', [dict(trustworthy=False),dict(left_forward_rpm=201),
    dict(yaw_rate_right_dps=46), dict(raw_yaw_rate_right_dps=46),
    dict(raw_yaw_rate_right_dps=float('nan')), dict(timestamp=9.7),
    dict(integrated_yaw_right_deg=float('nan'))])
def test_bad_feedback_clears_pose_instead_of_inventing_zero(change):
    p=EncoderPoseHistory(.816814)
    p.add(feedback(10.),10.)
    p.add(replace(feedback(10.01),**change),10.01)
    assert not p.samples


class DepthSource:
    _depth_orientation=SimpleNamespace(coordinate_space='external_uvc_unmirrored')
    _shadow_range_evidence=(1,10.,2.)
    def copy_depth_history(self, *, after_timestamp, max_frames, nonblocking):
        assert nonblocking is True and max_frames<=4
        return ()


def observer(tmp_path):
    return DepthTrackOnlineObserver(DepthSource(), lambda: feedback(10.14), tmp_path/'shadow',
        logging.getLogger(__name__),hfov_deg=60,circumference=.816814,
        clock=lambda:10.14,autostart=False)


def test_publish_revoke_and_old_frame_cannot_renew_current_anchor(tmp_path):
    o=observer(tmp_path); a=anchor()
    o.publish(a.observation,160,120); state=o._state
    o.publish(replace(a.observation,capture_timestamp=9.9),160,120)
    assert o._state==state
    o.revoke(); assert o._state[0]==1 and o._state[1] is None
    o.publish(anchor(uid=2).observation,160,120)
    assert o._state[1].observation.target_id==2
    o.close(); o.publish(anchor().observation,160,120)
    assert o._state[1] is None


def test_revocation_during_computation_discards_all_results(tmp_path):
    o=observer(tmp_path); o.publish(anchor().observation,160,120)
    def process(*args):
        o.revoke()
        return [{'status':'tracked'}]
    o.session.process=process
    assert o.tick()==[]
    assert o.counts['superseded_during_processing']==1


def test_no_orientation_no_association(tmp_path):
    o=observer(tmp_path); o.depth=DepthSource(); o.depth._depth_orientation=None
    o.publish(anchor().observation,160,120)
    o.session.process=lambda *args: pytest.fail('must not process unknown orientation')
    assert o.tick()==[]


def test_seed_wait_logging_is_rate_limited(tmp_path):
    o=observer(tmp_path); o.publish(anchor().observation,160,120)
    assert len(o.tick())==1
    assert o.tick()==[]


def test_worker_exception_disables_only_observer(tmp_path,caplog):
    o=observer(tmp_path)
    o.tick=lambda: (_ for _ in ()).throw(RuntimeError('synthetic failure'))
    o._run()
    assert o._stop.is_set()
    assert 'control unchanged' in caplog.text


def test_existing_output_is_not_overwritten(tmp_path,caplog):
    o=observer(tmp_path); o.output.mkdir()
    marker=o.output/'manifest.json'; marker.write_text('old')
    o._run()
    assert marker.read_text()=='old' and o._stop.is_set()


def test_worker_output_and_bounded_snapshot_are_observation_only(tmp_path):
    o=observer(tmp_path); o.session=session(); o.publish(anchor().observation,160,120)
    original=o.tick
    def once():
        rows=original(); o._stop.set(); return rows
    o.tick=once
    o._run()
    rows=[json.loads(line) for line in (o.output/'observations.jsonl').read_text().splitlines()]
    assert rows and all(not r['control_allowed'] for r in rows)
    assert len(list(o.output.glob('*.npz')))==1
    assert json.loads((o.output/'summary.json').read_text())['snapshots']==1


def test_depth_array_copy_does_not_mutate_source_and_is_bounded():
    s=session(); d=np.full((480,640),2000,dtype=np.uint16)
    s.add_frames([(11.,d)])
    assert s.frames[-1][1].shape==(120,160)
    s.frames[-1][1][:]=0
    assert np.all(d==2000)


def test_nonblocking_history_does_not_wait_or_change_old_api():
    from car_control_modular.astra_depth import AstraDepthConfig,AstraDepthRuntime
    d=AstraDepthRuntime(AstraDepthConfig())
    d._depth_history.append((10.,image()))
    with d._depth_lock:
        assert d.copy_depth_history(after_timestamp=0,nonblocking=True)==()
    rows=d.copy_depth_history(after_timestamp=0)
    assert len(rows)==1 and not rows[0][1].flags.writeable


def test_metrics_do_not_count_seed_replay_or_duplicate_as_new_frames(tmp_path):
    from tools.depth_track_online_metrics import report
    o=observer(tmp_path); o.session=session(); o.publish(anchor().observation,160,120)
    rows=o.tick()
    replay=dict(rows[-1],phase='replay')
    (tmp_path/'observations.jsonl').write_text('\n'.join(json.dumps(r) for r in rows+[rows[-1],replay]))
    data=report(tmp_path)
    assert data['distinct_tracked_samples']==4
    assert data['tracked_hz_over_success_span']==pytest.approx(30)
    assert data['potential_range_gap_fill_frames']==0
    assert not data['geometry_verified'] and not data['motor_authority_created']


def test_actual_runtime_hooks_publish_raw_geometry_and_revoke_without_control_changes(monkeypatch,tmp_path):
    from test_depth_target_snapshot import owner as owner_fixture, persons, RAW
    # Invoke the existing no-hardware fixture factory, not PersonTracker.__init__.
    main=owner_fixture.__wrapped__(monkeypatch)
    o=observer(tmp_path); main._depth_track_observer=o
    main._publish_longitudinal_context(640,480,persons())
    assert o._state[1].observation.bbox==RAW
    context=main._longitudinal_context
    # Stale YOLO is not proof of lost identity: the fixed 350ms shadow lease
    # may continue. Actual original movement revocation remains unchanged.
    main._clear_longitudinal_context(revoke_translation=False,reason='stale_vision_result')
    assert main._longitudinal_context is None and o._state[1] is not None
    main._clear_longitudinal_context(revoke_translation=False,reason='visual_target_missing_or_ambiguous')
    assert o._state[1] is None
    assert context['person_targets'][0].depth_observation.bbox==RAW


@pytest.mark.parametrize('search,steerable',[('searching',True),('none',False)])
def test_runtime_search_or_unsteerable_target_cannot_seed_shadow(monkeypatch,tmp_path,search,steerable):
    from test_depth_target_snapshot import owner as owner_fixture, persons
    main=owner_fixture.__wrapped__(monkeypatch)
    main._depth_track_observer=observer(tmp_path)
    main.search_state=search
    main._publish_longitudinal_context(640,480,persons(),target_steerable=steerable)
    assert main._depth_track_observer._state[1] is None
