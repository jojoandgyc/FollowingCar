"""Real bank boundaries with logged CAP1202 crop; synthetic appearance."""
from copy import deepcopy
import numpy as np
import pytest

from test_startup_enrollment_20260924 import bank, assign, meta, V
from rk_vision.crop_continuity import mapped_crop_continuous

ANCHOR = (350., 10., 620., 475.)
CROP = (361.8, 4.3, 631.1, 476.)


def seeded():
    b = bank()
    assert assign(b, meta(1197, 100., 1, ANCHOR)) == 0
    assert assign(b, meta(1200, 100.1, 2, ANCHOR)) == 1
    return b


def send(b, cap=1202, stamp=100.2, frame=3, box=CROP, vector=V, **changes):
    m = meta(cap, stamp, frame, box, quality_bbox_ok=False, bbox_quality_tier='weak',
             detector_edge_touch_count=3)
    m.update(changes)
    return b.assign(track_id=1, feature=vector, partial_feature=V, confidence=.94,
        area=(box[2]-box[0])*(box[3]-box[1]), frame_index=frame,
        candidate_count=m['candidate_count'], bbox_quality_ok=False,
        bbox_quality_tier='weak', bbox_quality_reason='edge_touch>2', sample_metadata=m)


def test_crop_continuation_keeps_uid_without_gallery_or_anchor_updates():
    b = seeded(); e = b.identities[1]
    anchor = deepcopy(e.last_strong_observation)
    features = deepcopy(e.feature_metadata)
    learning = deepcopy(e.template_memory.last_learning)
    assert send(b) == 1
    a = b.last_assignments[1]
    assert a['reason'] == 'mapped_crop_continuation'
    assert a['bbox_quality_ok'] and not a['bank_updated']
    assert a['crop_continuation_origin_cap'] == 1200
    assert e.last_strong_observation == anchor
    assert e.feature_metadata == features and e.template_memory.last_learning == learning
    assert send(b, 1214, 100.59, 4) == 1
    assert send(b, 1216, 100.61, 5) == 0  # Does not roll its half-second origin.


@pytest.mark.parametrize('case', ['search','multiple','stale','wrong_person','sliver','jump','quarantine','duplicate'])
def test_crop_bridge_cannot_reacquire_or_hide_conflicts(case):
    b=seeded(); kwargs={}
    if case=='search': kwargs['search_reacquire_context_active']=True
    if case=='multiple': kwargs['candidate_count']=2
    if case=='stale': kwargs['is_fresh']=False
    if case=='wrong_person': kwargs['vector']=np.array([0.,1.,0.])
    if case=='sliver': kwargs['box']=(555.,3.,640.,476.)
    if case=='jump': kwargs['box']=(0.,3.,180.,476.)
    if case=='duplicate': kwargs.update(cap=1200,stamp=100.1)
    if case=='quarantine':
        b._reacquire_quarantine.arm(1,1,capture_frame_id=1200,capture_timestamp=100.1,frame_index=2)
    assert send(b, **kwargs)==0
    assert not b.last_assignments[1]['bank_updated']


def test_geometry_proxy_does_not_treat_four_edges_as_broad_body():
    m=meta(2,100.1,2,(0.,0.,640.,480.),track_id=1)
    assert not mapped_crop_continuous(m, dict(track_id=1,capture_frame_id=1,capture_timestamp=100.))


def test_original_startup_still_cannot_enroll_side_crop():
    b=bank()
    assert send(b)==0 and send(b,1203,100.25,4)==0
    assert not b.identities


def test_real_tracker_continuation_reaches_detector_depth_resolver():
    from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
    from rk_vision.yolo11 import Detection
    from car_control_modular.depth_target_geometry import resolve_depth_target_observation
    tracker=DeepSortTracker(DeepSortTrackerConfig(n_init=1,
        identity_template_memory_enable=True, identity_template_crosscheck_enable=True,
        identity_appearance_region_safety_enable=True))
    for i,box in enumerate([ANCHOR,ANCHOR,ANCHOR,ANCHOR,CROP]):
        records=tracker.update([Detection(box,.94,0)],[V],partial_features=[V],
            partial_feature_sources=['osnet_torso'],image_width=640,image_height=480,
            frame_context=dict(control_frame_id=1+i,capture_frame_id=1198+i,
                               capture_timestamp=100.+i*.05))
    assert records[0].reid_uid==1
    a=tracker.identity_bank.last_assignments[1]
    assert a['reason']=='mapped_crop_continuation' and not a['bank_updated']
    row=tracker.last_identity_observations[0]
    geometry=resolve_depth_target_observation(target_id=1,display_bbox=row['display_bbox'],
        capture_frame_id=1202,capture_timestamp=100.2,
        observations=tracker.last_identity_observations,width=640,height=480)
    assert geometry is not None and geometry.bbox==CROP and geometry.source=='yolo_detector'
