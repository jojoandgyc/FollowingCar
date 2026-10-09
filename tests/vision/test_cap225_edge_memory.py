"""CAP221/225 recorded crops; synthetic descriptors, no model/hardware replay."""
from copy import deepcopy
import math

import numpy as np
import pytest

from rk_vision.template_memory import TemplateMemory


BOXES = {
    63: [300.6915283203125, 26.992019653320312, 515.7787475585938, 474.44915771484375],
    221: [229.5108642578125, 121.13343811035156, 323.7209777832031, 474.82879638671875],
    225: [229.54322814941406, 125.28028869628906, 319.2991943359375, 468.17431640625],
}
STAMPS = {63: 19963.52081526, 221: 19971.741906026, 225: 19971.940085638}


def meta(cap, **changes):
    value = dict(capture_frame_id=cap, capture_timestamp=STAMPS[cap], track_id=1,
                 detector_bbox=list(BOXES[cap]), image_width=640, image_height=480,
                 detector_confidence=.87, partial_feature_source='osnet_torso',
                 quality_bbox_ok=True, bbox_quality_tier='strong', is_fresh=True)
    value.update(changes)
    return value


def feature(distance=0.):
    cosine = 1.-distance
    return np.array([cosine, math.sqrt(1.-cosine*cosine)], dtype='float32')


def memory():
    result = TemplateMemory()
    assert result.remember(feature(), meta(63), 'partial')
    return result


def reference(**changes):
    result = dict(metadata=meta(221), comparable_caps=[63], pose_caps=[63],
                  pose_continuation=True, pose_started=STAMPS[221])
    result.update(changes)
    return result


def evidence(mem, query=None, proof=None, distance=.2936334):
    return mem.evidence(feature(distance), query or meta(225), 'partial', reliable_only=True,
                        comparable_only=True, shape_reference=proof or reference())


def test_recorded_vertical_edge_change_retains_only_the_already_admitted_template():
    mem = memory()
    prior = mem.evidence(feature(.3444943), meta(221), 'partial', comparable_only=True)
    assert prior['comparable_caps'] == [63]
    assert TemplateMemory.coverage_key(meta(221)) == 'top0_bottom1_side0'
    assert TemplateMemory.coverage_key(meta(225)) == 'top0_bottom0_side0'
    assert TemplateMemory.pose_sequence_usable(meta(225), meta(221), entering=True)
    assert not TemplateMemory.scale_crop_usable(meta(225))
    assert not TemplateMemory.vertical_border_comparable(meta(225), meta(63))
    out = evidence(mem)
    assert out['distance'] == pytest.approx(.2936334, abs=1e-6)
    assert out['winner_cap'] == 63
    assert out['comparison_mode'] == 'verified_pose_continuation'
    assert out['pose_bridge_caps'] == [63]
    assert out['scale_bridge_caps'] == []
    assert out['pose_retention_remaining_ms'] == pytest.approx(1801.820388, abs=.001)


@pytest.mark.parametrize('proof', [None, {}, reference(pose_continuation=False),
                                 reference(pose_caps=[]), reference(pose_caps=[62]),
                                 reference(pose_started=None)])
def test_unverified_candidate_or_new_template_cannot_bootstrap_retention(proof):
    out = memory().evidence(feature(), meta(225), 'partial', comparable_only=True,
                            shape_reference=proof)
    assert out['count'] == 0
    assert out['pose_bridge_caps'] == []


@pytest.mark.parametrize('changes', [
    {'track_id': 2}, {'capture_frame_id': 221}, {'capture_frame_id': None},
    {'capture_timestamp': STAMPS[221]}, {'capture_timestamp': STAMPS[221]-.01},
    {'capture_timestamp': STAMPS[221]+.251}, {'is_fresh': False},
    {'quality_bbox_ok': False}, {'bbox_quality_tier': 'weak'},
    {'identity_control_rejected': True}, {'detector_confidence': .79},
    {'detector_confidence': float('nan')}, {'detector_confidence': float('inf')},
    {'image_width': 1280}, {'partial_feature_source': 'fallback'},
    {'detector_bbox': [0, 125, 90, 468]},
    {'detector_bbox': [230, 125, 293, 365]},
    {'detector_bbox': [230, 125, 270, 280]},
    {'detector_bbox': [230, 125, 310, 468]},
    {'detector_bbox': [230, 125, 320, 430]},
])
def test_bad_capture_geometry_or_quality_cannot_use_pose_retention(changes):
    out = evidence(memory(), meta(225, **changes))
    assert out['count'] == 0
    assert out['pose_bridge_caps'] == []


def test_entry_requires_one_nearby_vertical_change_but_continuation_may_keep_labels():
    q, p = meta(225), meta(221)
    assert TemplateMemory.pose_sequence_usable(q, p, entering=True)
    assert not TemplateMemory.pose_sequence_usable(q, q, entering=True)
    assert TemplateMemory.pose_sequence_usable(q, q)
    assert TemplateMemory.pose_sequence_usable(p, q, entering=True)
    p['detector_bbox'] = [200, 8, 350, 472]
    q['detector_bbox'] = [201, 12, 349, 468]
    assert not TemplateMemory.pose_sequence_usable(q, p, entering=True)
    assert not TemplateMemory.pose_sequence_usable(q, p)


def test_template_shape_and_height_limits_still_apply_to_admitted_caps():
    assert TemplateMemory.pose_template_usable(meta(225), meta(63))
    # Current crop remains usable, but the old template is too broad.
    wide = meta(63, detector_bbox=[250, 27, 500, 475])
    assert TemplateMemory.pose_crop_usable(wide)
    assert not TemplateMemory.pose_template_usable(meta(225), wide)
    # Same shape without enough shared scale/coverage cannot be retained.
    short = meta(225, detector_bbox=[230, 170, 308, 470])
    assert TemplateMemory.pose_crop_usable(short)
    assert not TemplateMemory.pose_template_usable(short, meta(63))


def test_exact_region_negative_evidence_takes_precedence():
    mem = memory()
    assert mem.remember(feature(.49), meta(225, capture_frame_id=224,
                        capture_timestamp=STAMPS[225]-.01), 'partial')
    out = evidence(mem, distance=0.)
    assert out['winner_cap'] == 224
    assert out['distance'] == pytest.approx(.49)
    assert out['comparison_mode'] == 'exact_coverage'
    assert out['pose_bridge_caps'] == []


def test_normal_border_bridge_negative_evidence_also_takes_precedence():
    mem = memory()
    assert mem.remember(feature(.49), meta(63, capture_frame_id=64,
                        capture_timestamp=STAMPS[63]+.1,
                        detector_bbox=[230, 147, 334, 476]), 'partial')
    q = meta(225, detector_bbox=[230, 145, 334, 467])
    out = evidence(mem, q, distance=0.)
    assert out['winner_cap'] == 64
    assert out['distance'] == pytest.approx(.49)
    assert out['comparison_mode'] == 'vertical_border_bridge'
    assert out['pose_bridge_caps'] == []


def test_retention_exposes_current_torso_mismatch_instead_of_rewriting_it():
    out = evidence(memory(), distance=.46)
    assert out['pose_bridge_caps'] == [63]
    assert out['distance'] == pytest.approx(.46)


def test_fixed_budget_cannot_roll_with_successful_queries_or_refresh_templates():
    mem = memory()
    before = deepcopy(mem)
    proof = reference()
    for i in range(1, 11):
        q = meta(225, capture_frame_id=225+i, capture_timestamp=STAMPS[221]+.19*i)
        out = evidence(mem, q, proof)
        assert out['pose_bridge_caps'] == [63]
        assert out['pose_retention_remaining_ms'] == pytest.approx(2000.-190*i)
        proof['metadata'] = q
    q = meta(225, capture_frame_id=240, capture_timestamp=STAMPS[221]+2.09)
    assert evidence(mem, q, proof)['count'] == 0
    assert mem.last_learning == before.last_learning
    assert mem.watermark == before.watermark
    for tier in ('strong', 'partial'):
        for storage in ('recent', 'representatives'):
            old_rows, new_rows = getattr(before, storage)[tier], getattr(mem, storage)[tier]
            assert len(old_rows) == len(new_rows)
            for (old_f, old_m), (new_f, new_m) in zip(old_rows, new_rows):
                np.testing.assert_array_equal(old_f, new_f)
                assert old_m == new_m


def test_missing_or_future_fixed_start_and_recent_expiry_fail_closed():
    mem = memory()
    assert evidence(mem, proof=reference(pose_started=STAMPS[225]))['count'] == 0
    mem.recent['partial'][0][1]['capture_timestamp'] = STAMPS[225]-30.01
    assert evidence(mem)['count'] == 0


def test_retention_does_not_relax_paired_proof_threshold_or_same_capture_requirement():
    mem = memory()
    assert mem.remember(feature(), meta(63), 'strong')
    assert not mem.paired_recent_evidence(feature(.25), feature(.27), meta(225),
                                          shape_reference=reference())['qualified']
    assert mem.paired_recent_evidence(feature(.25), feature(.25), meta(225),
                                     shape_reference=reference())['qualified']
    mem.recent['strong'][0][1]['capture_frame_id'] = 62
    assert not mem.paired_recent_evidence(feature(.25), feature(.25), meta(225),
                                          shape_reference=reference())['qualified']
