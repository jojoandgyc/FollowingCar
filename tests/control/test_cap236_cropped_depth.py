"""Real recorded identity geometry; depth samples are synthetic, no hardware."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from car_control_modular.cropped_depth_observation import resolve_cropped_depth_observation
from car_control_modular.depth_target_geometry import resolve_depth_target_observation
from car_control_modular.control_types import DistanceState, PersonTarget, SensorFrame
from car_control_modular.controllers import FollowSafetyController
from car_control_modular.distance_runtime import DistanceRuntime
from car_control_modular.astra_depth import AstraDepthMeasurement
from test_depth_target_snapshot import owner as context_owner

DATA = json.loads((Path(__file__).parent/'fixtures/cap236_cropped_depth.json').read_text())
ROWS = {r['sample_metadata']['capture_frame_id']: r for r in DATA['rows']}


def setup(cap=236):
    r = deepcopy(ROWS[cap]); m = r['sample_metadata']
    anchor = ROWS[231]; am = anchor['sample_metadata']
    ref = dict(track_id=1, geometry_source='detector', bbox=anchor['detector_bbox'],
               capture_frame_id=231, capture_timestamp=am['capture_timestamp'],
               integrated_yaw_deg=am['integrated_yaw_deg'])
    bank = NS(track_to_uid={1: 1}, identities={1: NS(last_strong_observation=ref)},
        _reacquire_control_suspects={}, _geometry_revoked_uids={}, _mapped_geometry_conflicts={},
        _reacquire_quarantine=NS(is_held=lambda uid: False))
    return dict(bank=bank, observations=[r], uid=1, raw_track_id=1,
        display_bbox=r['display_bbox'], capture_id=cap, capture_timestamp=m['capture_timestamp'],
        width=640, height=480, now=m['capture_timestamp']+.16)


@pytest.mark.parametrize('cap', [236,239,240,241])
def test_logged_edge_crops_can_measure_but_not_become_normal_targets(cap):
    args = setup(cap); original = deepcopy(args['observations'])
    out, reason = resolve_cropped_depth_observation(**args)
    assert reason == 'accepted' and out.source == 'yolo_cropped_observation'
    assert out.bbox == tuple(ROWS[cap]['detector_bbox'])
    assert out.capture_frame_id == cap
    assert args['observations'] == original and original[0]['uid'] == 0
    assert resolve_depth_target_observation(target_id=1, display_bbox=args['display_bbox'],
        capture_frame_id=cap, capture_timestamp=args['capture_timestamp'],
        observations=args['observations'], width=640, height=480) is None


@pytest.mark.parametrize('change', [
    'unmapped','conflict','suspect','quarantine','revoked','old_ref','old_image',
    'future_image','wrong_cap','raw_change','crowd','competition','stale_proof',
    'area_reason','aspect_reason','unknown_recent','weak_recent','low_score',
    'search','partial_conflict','near_full','small_fragment','turn','nan_score','missing_frame',
])
def test_unqualified_frames_remain_closed(change):
    args = setup(); b = args['bank']; r = args['observations'][0]
    m, a = r['sample_metadata'], r['assignment']
    if change == 'unmapped': b.track_to_uid.clear()
    if change == 'conflict': b._mapped_geometry_conflicts[1] = {'uid':1}
    if change == 'suspect': b._reacquire_control_suspects[1] = {}
    if change == 'quarantine': b._reacquire_quarantine.is_held = lambda uid: True
    if change == 'revoked': b._geometry_revoked_uids[1] = 1
    if change == 'old_ref': b.identities[1].last_strong_observation['capture_timestamp'] -= 1
    if change == 'old_image': args['now'] += 1
    if change == 'future_image': args['now'] = args['capture_timestamp']-.01
    if change == 'wrong_cap': args['capture_id'] += 1
    if change == 'raw_change': args['raw_track_id'] = 2
    if change == 'crowd': m['candidate_count'] = 2
    if change == 'competition': m['identity_competition']['passed'] = False
    if change == 'stale_proof': m['identity_competition']['frame_index'] -= 1
    if change == 'area_reason': a['bbox_quality_reason'] += ',area<900'
    if change == 'aspect_reason': a['bbox_quality_reason'] += ',aspect<0.1'
    if change == 'unknown_recent': a['template_recent_evidence'] = {}
    if change == 'weak_recent': a['template_recent_evidence']['distance'] = .3
    if change == 'low_score': m['detector_confidence'] = .5
    if change == 'search': m['search_reacquire_context_active'] = True
    if change == 'partial_conflict': a['reacquire_partial_state'] = 'mismatch'
    if change == 'near_full': r['detector_bbox'] = [0,0,639,479]
    if change == 'small_fragment': r['detector_bbox'] = [0,0,50,100]
    if change == 'turn': m['integrated_yaw_deg'] -= 30
    if change == 'nan_score': m['detector_confidence'] = float('nan')
    if change == 'missing_frame': r.pop('frame_index')
    assert resolve_cropped_depth_observation(**args)[0] is None


def runtime(args, monkeypatch, **measurement_changes):
    now = args['now']; calls = []
    monkeypatch.setattr('car_control_modular.distance_runtime.time.monotonic', lambda: now)
    sample = AstraDepthMeasurement(distance_m=1.6, raw_distance_m=1.62, sample_age_sec=.1,
        valid_pixels=1000, detail='depth_multiregion', sample_timestamp=now-.1,
        observation_sample_timestamp=now-.1, temporal_status='new_sample')
    sample = replace(sample, **measurement_changes)
    dr = DistanceRuntime.__new__(DistanceRuntime)
    dr.config = NS(module_astra_depth_enable=True, vision_depth_require_detector_bbox=True)
    dr.sensor_runtime = NS(get_astra_target_distance=lambda *a, **k: (calls.append((a,k)), sample)[1])
    dr._last_vision_depth_target = object(); dr._vision_depth_fusion = object()
    dr.last_distance_state = DistanceState(source='vision_depth', used_distance_m=1.8)
    return dr, calls


def test_ranging_uses_capture_time_without_touching_normal_fusion_or_roi(monkeypatch):
    args = setup(); obs, _ = resolve_cropped_depth_observation(**args)
    dr, calls = runtime(args, monkeypatch)
    old = (dr._last_vision_depth_target, dr._vision_depth_fusion, dr.last_distance_state)
    state = dr.get_cropped_observation_state(640,480,obs)
    assert state.used_distance_m == 1.6 and state.sample_timestamp == args['now']-.1
    assert state.source == 'cropped_depth_observation'
    assert (dr._last_vision_depth_target, dr._vision_depth_fusion, dr.last_distance_state) == old
    assert calls[0][0][0] == obs.bbox
    assert calls[0][1]['reference_timestamp'] == obs.capture_timestamp
    assert calls[0][1]['evidence_capture_frame_id'] == 236
    assert not calls[0][1].get('use_latest_depth')
    assert not FollowSafetyController._is_fresh_depth_state(SensorFrame(distance_state=state))
    assert FollowSafetyController._distance_longitudinally_untrusted(
        FollowSafetyController.__new__(FollowSafetyController), SensorFrame(distance_state=state))
    target = PersonTarget(bbox=obs.bbox, track_id=1, confidence=.9, area=10000,depth_observation=obs)
    assert dr._depth_target_for_ranging(target,640,480,now=args['now'],
        use_latest_depth=True,capture_timestamp=None)[0] is None


@pytest.mark.parametrize('changes', [
    {'sample_timestamp':None}, {'sample_timestamp':float('nan')},
    {'sample_timestamp':0}, {'raw_distance_m':None}, {'distance_m':float('nan')},
    {'temporal_status':'duplicate'}, {'temporal_status':'older_than_anchor'},
    {'detail':'depth_distance_jump_pending'}, {'detail':'depth_multiregion_hold'},
])
def test_invalid_depth_never_becomes_fresh_distance(monkeypatch, changes):
    args = setup(); obs, _ = resolve_cropped_depth_observation(**args)
    dr, _ = runtime(args, monkeypatch, **changes)
    state = dr.get_cropped_observation_state(640,480,obs)
    assert state.used_distance_m is None and state.sample_count == 0


def test_real_main_binding_deduplicates_and_preserves_identity(monkeypatch):
    from request_0513_modular import PersonTracker
    args = setup(); dr, calls = runtime(args, monkeypatch)
    instance = NS(_distance_runtime=dr, _follow_controller=NS(active_target_id=1),
        _rknn_pipeline=NS(tracker=NS(identity_bank=args['bank'],last_identity_observations=args['observations'])))
    persons = [(args['display_bbox'],1,.94,120000)]
    state = PersonTracker._observe_cropped_target_depth(instance,persons,640,480,236,args['capture_timestamp'],None)
    assert state.used_distance_m == 1.6 and len(calls) == 1
    again = PersonTracker._observe_cropped_target_depth(instance,persons,640,480,236,args['capture_timestamp'],None)
    assert again.used_distance_m is None and len(calls) == 1
    assert args['observations'][0]['uid'] == 0
    assert args['bank'].identities[1].last_strong_observation['capture_frame_id'] == 231


def test_real_low_quality_process_receives_distance_without_normal_depth_roi(context_owner, monkeypatch):
    import request_0513_modular as main
    from car_control_modular.control_types import HazardState
    obj = context_owner; args = setup(); dr, calls = runtime(args, monkeypatch)
    obj._active_capture_frame_id = 236
    obj._active_capture_timestamp = args['capture_timestamp']
    obj._rknn_pipeline.tracker.identity_bank = args['bank']
    obj._rknn_pipeline.tracker.last_identity_observations = args['observations']
    obj._last_dispatched_action = main.ACTION_STOP
    obj._follow_controller.set_last_dispatched = lambda _: None
    obj._get_obstacle_status = lambda: {}
    obj._action_runtime = NS(get_steering_feedback=lambda: None)
    obj._current_hazard_state_for_controller = lambda: HazardState()
    obj._depth_longitudinal_authority_enabled = lambda: True
    obj._publish_longitudinal_context = lambda *a, **k: pytest.fail('measurement promoted to motion ROI')
    dr.select_target = lambda ts: ts[0]
    obj._distance_runtime = dr
    class DecisionReached(Exception): pass
    def decide(_, frame, **kw):
        assert frame.distance_m == 1.6
        assert frame.distance_state.source == 'cropped_depth_observation'
        assert frame.persons[0].depth_observation is None
        assert kw['low_quality_visible'] and not kw['target_steerable']
        raise DecisionReached()
    obj._follow_controller.decide = decide
    with pytest.raises(DecisionReached):
        obj._process_detections_modular(640,480,[(args['display_bbox'],1,.94,120000)],
                                        low_quality_visible=True,target_steerable=False)
    assert len(calls) == 1 and obj._longitudinal_context is None
