"""Real tracker/bank FULL -> fast -> FULL with CAP309..336 metadata.

Boxes, capture clocks/yaw and gallery distances come from run_20261010_234438.
Embeddings/color, gallery bootstrap and the old raw2 Kalman object are synthetic;
the missed raw2 lifetime is the recorded condition, not a serialized tracker.
Fast completion is a controlled healthy .10 s (not a replay of CPU scheduling).
"""
from copy import deepcopy
import pickle

import numpy as np
import pytest

from rk_vision.deepsort.track import TrackState
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection
from test_detector_continuation_pipeline import fast_pipeline, seed
from test_reacquire_crosscheck import feature
from test_similar_follow_bank import make_bank
from tools.replay_cap334_recovery import gallery_snapshot


# time, yaw, score, gallery distance, detector box
ROWS = {
    309: (47430.248491087, 27.9947850781491, .8928923010826111, .11677050590515137,
          (46.88380432128906, 1.5037841796875, 239.02169799804688, 477.7928466796875)),
    311: (47430.35164723, 27.188149120180462, .9117296934127808, .10803139209747314,
          (74.40615844726562, .408050537109375, 266.8892822265625, 477.982666015625)),
    313: (47430.449978743, 26.297607755489736, .9154971837997437, .10027706623077393,
          (95.17707061767578, 2.3878936767578125, 285.93438720703125, 476.6253662109375)),
    315: (47430.579883846, 25.793867196542884, .8815898895263672, .14142680168151855,
          (113.69733428955078, .716705322265625, 290.3266296386719, 476.04351806640625)),
    318: (47430.715709051, 25.967538952252895, .8891248106956482, .18009328842163086,
          (128.7095947265625, .478515625, 282.49688720703125, 477.64825439453125)),
    322: (47430.919195449, 25.967538952252895, .8815898895263672, .15587615966796875,
          (160.61456298828125, 1.5647430419921875, 285.3486022949219, 479.5396728515625)),
    326: (47431.116551458, 29.918222292492864, .8288451433181763, .1523447036743164,
          (184.02838134765625, 2.373382568359375, 329.1534729003906, 466.86993408203125)),
    329: (47431.286771474, 32.08560633356581, .8815898895263672, .1118084192276001,
          (189.705810546875, 1.2034912109375, 353.66229248046875, 456.2919921875)),
    332: (47431.452840303, 32.08560633356581, .8665199279785156, .1920253038406372,
          (207.27395629882812, 1.31451416015625, 356.074462890625, 456.7969970703125)),
    336: (47431.651681149, 31.330159351830247, .8665199279785156, .22401630878448486,
          (226.87362670898438, 0., 354.40533447265625, 460.14410400390625)),
}
COLOR = np.ones(16)


def sample(cap):
    stamp, yaw, score, _, box = ROWS[cap]
    return [Detection(box, score, 0)], dict(capture_frame_id=cap,
        capture_timestamp=stamp, integrated_yaw_deg=yaw)


def full(tracker, cap):
    detections, ctx = sample(cap)
    records = tracker.update(detections, [feature(ROWS[cap][3])], image_width=640,
        image_height=480, frame_context=ctx, color_features=[COLOR])
    if any(record.reid_uid == 1 for record in records):
        tracker.set_search_reacquire_context(active_uid=1, searching=False, direction=None)
    tracker.note_full_identity_verification(records, detections=detections,
        color_features=[COLOR], frame_context=ctx, image_width=640, image_height=480,
        now=ctx['capture_timestamp']+.14, active_uid=1, raw_candidate_count=1)
    return records


def add_missed_track(tracker, *, raw_id=2, missed=5, max_age=30):
    old = deepcopy(tracker.deepsort.tracker.tracks[0])
    old.track_id, old.time_since_update, old._max_age = raw_id, missed, max_age
    old.age += missed
    old.mean[0] = 96.  # old raw2's left-edge location, not current raw3
    tracker.deepsort.tracker.tracks.append(old)
    return old


def before_332():
    tracker = DeepSortTracker(DeepSortTrackerConfig(n_init=1, bbox_expand_scale=1.,
        feature_update_interval=1, identity_similar_follow_enable=True, hfov_deg=60.,
        identity_template_memory_enable=True, identity_template_crosscheck_enable=True,
        identity_template_learning_guard_enable=True))
    tracker.identity_bank = make_bank()
    tracker.deepsort.tracker._next_id = 3
    tracker._frame_index = 104
    tracker.set_search_reacquire_context(active_uid=1, searching=True, direction='left')
    for cap in (309, 311, 313, 315, 318, 322, 326, 329):
        records = full(tracker, cap)
        if cap == 313:
            assert [(row.track_id, row.reid_uid) for row in records] == [(3, 1)]
            old = add_missed_track(tracker)
        if cap >= 315:
            assert tracker.last_detector_continuation_reason == 'full_verified'
    assert old.time_since_update == 10
    assert tracker._detector_proof.permission == 'similar_follow'
    return tracker, old


def propose(tracker, cap=332, *, raw_count=1, detections=None):
    observed, ctx = sample(cap)
    return tracker.plan_detected_continuation(observed if detections is None else detections,
        image_width=640, image_height=480, frame_context=ctx, active_uid=1,
        now=ctx['capture_timestamp']+.10, color_features=[COLOR], raw_candidate_count=raw_count)


def test_recorded_current_identity_can_skip_reid_while_old_raw_normally_ages():
    tracker, old = before_332()
    gallery, bank = gallery_snapshot(tracker.identity_bank), pickle.dumps(tracker.identity_bank)
    proof, age, misses = tracker._detector_proof, old.age, old.time_since_update
    proposed = propose(tracker)
    assert proposed is not None
    rows = tracker.commit_detected_continuation(proposed, now=ROWS[332][0]+.11)
    assert [(r.track_id, r.reid_uid, r.time_since_update) for r in rows] == [(3, 1, 0)]
    assert tuple((rows[0].x1, rows[0].y1, rows[0].x2, rows[0].y2)) == ROWS[332][4]
    assert old.age == age+1 and old.time_since_update == misses+1
    assert tracker._detector_proof.deadline == proof.deadline
    assert tracker._detector_proof.verified.capture == 329
    assert tracker.control_assignment_for_track(3)['identity_permission'] == 'similar_follow'
    assert tracker.control_assignment_for_track(2) is None
    assert pickle.dumps(tracker.identity_bank) == bank
    assert gallery_snapshot(tracker.identity_bank) == gallery
    # The next recorded capture needs normal FULL again; a missed raw track
    # cannot suppress the periodic fresh appearance/competition checks.
    assert propose(tracker, 336) is None
    assert tracker.last_detector_continuation_reason == 'full_recheck_due'
    assert [(r.track_id, r.reid_uid) for r in full(tracker, 336)] == [(3, 1)]
    assert tracker.last_detector_continuation_reason == 'full_verified'
    assert old.time_since_update == misses+2
    assert not tracker.identity_bank.last_assignments[3]['bank_updated']


def test_missed_raw_is_deleted_at_normal_max_age_even_on_fast_frame():
    tracker, old = before_332()
    old._max_age = old.time_since_update
    assert tracker.commit_detected_continuation(propose(tracker), now=ROWS[332][0]+.11)
    assert old.is_deleted()
    assert [t.track_id for t in tracker.deepsort.tracker.tracks] == [3]


@pytest.mark.parametrize('fault', ['current_old', 'current_tentative', 'duplicate', 'missing_target',
    'extra_detection', 'raw_count', 'competition', 'missing_competition', 'uid_competition',
    'count_competition', 'geometry', 'revoked', 'same_side_person'])
def test_old_track_exception_does_not_hide_current_ambiguity_or_conflict(fault):
    tracker, old = before_332()
    assignment = tracker.identity_bank.last_assignments[3]
    options = {}
    if fault == 'current_old': old.time_since_update = 0
    elif fault == 'current_tentative': old.time_since_update, old.state = 0, TrackState.TENTATIVE
    elif fault == 'duplicate': old.track_id = 3
    elif fault == 'missing_target': tracker.deepsort.tracker.tracks[0].time_since_update = 1
    elif fault == 'extra_detection': options.update(raw_count=2,
        detections=sample(332)[0]+[Detection((20., 50., 150., 450.), .9, 0)])
    elif fault == 'raw_count': options['raw_count'] = 2
    elif fault == 'competition': assignment['identity_competition']['passed'] = False
    elif fault == 'missing_competition': assignment.pop('identity_competition')
    elif fault == 'uid_competition': assignment['identity_competition']['uid'] = 2
    elif fault == 'count_competition': assignment['identity_competition']['candidate_count'] = 2
    elif fault == 'geometry': tracker.identity_bank._mapped_geometry_conflicts[3] = dict(uid=1)
    elif fault == 'revoked': tracker.identity_bank._geometry_revoked_uids[1] = {}
    else: options['detections'] = [Detection((25., 50., 150., 450.), .9, 0)]
    before = tracker._frame_index
    assert propose(tracker, **options) is None
    assert tracker._frame_index == before


@pytest.mark.parametrize('field', ['frame_index', 'source_detection_index'])
def test_full_cannot_ignore_old_track_with_misbound_competition(field):
    tracker, _ = before_332()
    d, ctx = sample(332)
    rows = tracker.update(d, [feature(ROWS[332][3])], image_width=640, image_height=480,
        frame_context=ctx, color_features=[COLOR])
    tracker.identity_bank.last_assignments[3]['identity_competition'][field] += 1
    tracker.note_full_identity_verification(rows, detections=d, color_features=[COLOR],
        frame_context=ctx, image_width=640, image_height=480, now=ROWS[332][0]+.14,
        active_uid=1, raw_candidate_count=1)
    assert tracker._detector_proof is None
    assert tracker.last_detector_continuation_reason == 'full_identity_unverified'


def test_commit_cannot_ignore_old_track_that_became_current_after_plan():
    tracker, old = before_332()
    proposed = propose(tracker)
    old.time_since_update = 0
    before = tracker._frame_index
    assert tracker.commit_detected_continuation(proposed, now=ROWS[332][0]+.11) is None
    assert tracker._frame_index == before


def test_actual_pipeline_skips_extractor_but_not_detector_with_missed_raw(fast_pipeline):
    s = fast_pipeline
    seed(s)
    t = s.pipeline.tracker
    old = add_missed_track(t, raw_id=2, missed=1, max_age=2)
    bank = pickle.dumps(t.identity_bank)
    for cap in (5, 6):
        rows = s.step(cap)
        assert [(r.track_id, r.reid_uid) for r in rows] == [(1, 1)]
        assert s.pipeline.last_identity_processing['mode'] == 'detector_continuation'
    assert old.time_since_update == 3 and old.is_deleted()
    assert s.pipeline.detector.calls == 6 and s.pipeline.reid.calls == 4
    assert pickle.dumps(t.identity_bank) == bank
