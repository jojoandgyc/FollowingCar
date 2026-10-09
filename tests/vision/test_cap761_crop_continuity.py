"""CAP754..769 recorded crop/time/full distances; synthetic embeddings only.

No hardware or inference. Each accepted frame rechecks the bank; none may
learn a template, move the original strong anchor, or grant motor time.
"""
from copy import deepcopy

import numpy as np
import pytest

from test_startup_enrollment_20260924 import bank, assign, meta, V
from rk_vision.crop_continuity import mapped_crop_continuous


ANCHOR_CAP = 754
ANCHOR_TS = 39514.254843146
ANCHOR_BOX = (14.006210327148438, .8696136474609375, 188.44754028320312, 476.95928955078125)
# (capture, timestamp, detector bbox, current full-body cosine distance)
CROP_SAMPLES = (
    (758, 39514.455090392, (0., 2.1056060791015625, 179.92544555664062, 473.21240234375), .147203266620636),
    (761, 39514.622908025, (.00012969970703125, 1.7281951904296875, 175.58297729492188, 475.3670654296875), .1757291555404663),
    (764, 39514.787874687, (0., 2.48406982421875, 147.97079467773438, 473.0726318359375), .10517311096191406),
    (767, 39514.922457727, (0., 2.1331634521484375, 182.35702514648438, 474.72198486328125), .10748565196990967),
    (769, 39515.022552039, (.01805877685546875, 1.413726806640625, 201.7787628173828, 472.3018798828125), .10001957416534424),
)


def vector_at_distance(distance):
    cosine = 1.-distance
    return np.array([cosine, np.sqrt(1.-cosine*cosine), 0.], dtype=np.float64)


def seeded_bank():
    b = bank()
    assert assign(b, meta(752, ANCHOR_TS-.1, 331, ANCHOR_BOX), track=3) == 0
    assert assign(b, meta(754, ANCHOR_TS, 332, ANCHOR_BOX), track=3) == 1
    return b


def crop_metadata(sample, frame, **changes):
    cap, stamp, bbox, _ = sample
    m = meta(cap, stamp, frame, bbox, track_id=3, quality_bbox_ok=False,
             bbox_quality_tier='weak', detector_edge_touch_count=3)
    m.update(changes)
    if 'detector_bbox' in changes:
        x1, y1, x2, y2 = changes['detector_bbox']
        m.update(bbox=changes['detector_bbox'], detector_center_x_ratio=(x1+x2)/1280.,
                 detector_area_ratio=(x2-x1)*(y2-y1)/(640*480))
    return m


def submit_crop(b, sample, frame, **changes):
    m = crop_metadata(sample, frame, **changes)
    box = m['detector_bbox']
    uid = b.assign(track_id=3, feature=vector_at_distance(sample[3]), partial_feature=V,
        confidence=.94, area=(box[2]-box[0])*(box[3]-box[1]), frame_index=frame,
        candidate_count=m['candidate_count'], bbox_quality_ok=False,
        bbox_quality_tier='weak', bbox_quality_reason='edge_touch>2', sample_metadata=m)
    return uid, m


def test_logged_crop_sequence_rechecks_each_frame_without_learning_or_rolling_origin(caplog):
    b = seeded_bank()
    entry = b.identities[1]
    anchor = deepcopy(entry.last_strong_observation)
    templates = deepcopy(entry.feature_metadata)
    learning = deepcopy(entry.template_memory.last_learning)
    with caplog.at_level('INFO', logger='PersonTracker'):
        for frame, sample in enumerate(CROP_SAMPLES, 333):
            assert submit_crop(b, sample, frame)[0] == 1
            a = b.last_assignments[3]
            assert a['reason'] == 'mapped_crop_continuation' and not a['bank_updated']
            assert a['crop_continuation_origin_cap'] == ANCHOR_CAP
            assert a['crop_continuation_origin_timestamp'] == ANCHOR_TS
            assert a['crop_continuation_window_ms'] == 1000
            assert a['crop_continuation_full_limit'] == .20
            assert a['crop_continuation_reason'] == 'accepted'
            assert a['crop_continuation_remaining_ms'] == pytest.approx((1.-(sample[1]-ANCHOR_TS))*1000)
            assert entry.last_strong_observation == anchor
            assert entry.feature_metadata == templates
            assert entry.template_memory.last_learning == learning
    assert 'anchor_cap=754' in caplog.text and 'motion_deadline_renewed=False' in caplog.text


@pytest.mark.parametrize('offset,accepted', [(1., True), (1.00001, False), (1.2, False)])
def test_window_is_fixed_to_normal_anchor_not_the_last_crop(offset, accepted):
    b = seeded_bank()
    for frame, sample in enumerate(CROP_SAMPLES, 333):
        assert submit_crop(b, sample, frame)[0] == 1
    sample = (770, ANCHOR_TS+offset, CROP_SAMPLES[-1][2], .10)
    assert (submit_crop(b, sample, 338)[0] == 1) is accepted
    assert b.identities[1].last_strong_observation['capture_frame_id'] == 754


@pytest.mark.parametrize('case', ['search', 'multiple', 'stale', 'sliver', 'opposite_side',
    'quarantine', 'suspect', 'competition', 'archive_only', 'recent_mismatch',
    'explicit_conflict', 'wrong_appearance', 'duplicate_anchor'])
def test_extended_crop_window_does_not_bypass_identity_or_quality_vetoes(case):
    b = seeded_bank()
    changes = {}
    sample = CROP_SAMPLES[1]
    if case == 'search': changes['search_reacquire_context_active'] = True
    elif case == 'multiple': changes['candidate_count'] = 2
    elif case == 'stale': changes['is_fresh'] = False
    elif case == 'sliver': changes['detector_bbox'] = (0., 2., 60., 475.)
    elif case == 'opposite_side': changes['detector_bbox'] = (470., 2., 640., 475.)
    elif case == 'quarantine':
        b._reacquire_quarantine.arm(1, 3, capture_frame_id=754,
            capture_timestamp=ANCHOR_TS, frame_index=332)
    elif case == 'suspect':
        b._reacquire_quarantine.arm(1, 3, capture_frame_id=754,
            capture_timestamp=ANCHOR_TS, frame_index=332)
        b._reacquire_control_suspects[1] = dict(track_id=3, reason='partial_conflict',
            streak=0, capture=754, timestamp=ANCHOR_TS)
    elif case == 'competition':
        changes['source_detection_index'] = 0
        changes['identity_competition'] = dict(uid=1, frame_index=333, candidate_count=1,
            source_detection_index=0, passed=False, reason='identity_margin')
    elif case in ('archive_only', 'recent_mismatch'):
        memory = b.identities[1].template_memory
        if case == 'archive_only':
            for rows in memory.recent.values():
                for _, info in rows:
                    info['capture_timestamp'] = ANCHOR_TS-31.
        else:
            memory.recent['strong'] = [(np.array([0., 0., 1.]), info)
                                       for _, info in memory.recent['strong']]
    elif case == 'explicit_conflict': changes['quality_bbox_reason'] = 'identity_center_jump>0.4'
    elif case == 'wrong_appearance': sample = (*sample[:3], .201)
    elif case == 'duplicate_anchor': sample = (754, ANCHOR_TS, sample[2], sample[3])
    anchor = deepcopy(b.identities[1].last_strong_observation)
    assert submit_crop(b, sample, 333, **changes)[0] == 0
    assert not b.last_assignments[3]['bank_updated']
    assert b.identities[1].last_strong_observation == anchor


def test_crop_shape_window_is_not_a_physical_depth_lease():
    m = crop_metadata(CROP_SAMPLES[-1], 337)
    assert mapped_crop_continuous(m, dict(track_id=3, capture_frame_id=754, capture_timestamp=ANCHOR_TS))
    # Identity metadata has no motor authorization or executable wheel plan.
    assert not {'depth_valid_until', 'motion_valid_until', 'wheel_targets'} & m.keys()
