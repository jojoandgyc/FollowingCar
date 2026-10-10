"""Recorded CAP459/467/469/474/488 geometry; synthetic appearance vectors.

This checks geometry policy and real assignment wiring, not camera inference.
"""
from copy import deepcopy

import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
from test_cap874_identity_reacquire import feature, metadata


ROWS = {
    459: (157, 33182.070224320, -28.996369203789623,
          (173.54855346679688, 5.240081787109375, 366.8357238769531, 473.139892578125)),
    467: (159, 33182.466902089, -28.119350084524903,
          (90.45924377441406, 1.6136322021484375, 257.02313232421875, 477.03375244140625)),
    469: (160, 33182.598234369, -27.83070219277553,
          (45.46519470214844, 0., 222.04458618164062, 476.0106201171875)),
    474: (161, 33182.861917985, -26.967675589756965,
          (11.624855041503906, .99261474609375, 179.63796997070312, 476.5081787109375)),
    488: (167, 33183.592120901, -29.14152062706922,
          (.6617040634155273, 2.611663818359375, 63.37952423095703, 472.99517822265625)),
}


def meta(cap, *, raw=7, mirror=False, **extra):
    frame, stamp, yaw, box = ROWS[cap]
    if mirror:
        box, yaw = (640-box[2], box[1], 640-box[0], box[3]), -yaw
    m = metadata(cap, stamp, box, yaw, search=cap >= 474)
    m.update(track_id=raw, frame_index=frame, image_width=640, image_height=480,
             source_detection_index=0, identity_competition=dict(
                 uid=1, frame_index=frame, source_detection_index=0, candidate_count=1,
                 passed=True, reason='single_candidate'))
    if cap >= 474:
        m.update(quality_bbox_ok=False, bbox_quality_tier='weak',
                 quality_bbox_reason='edge_touch>2')
    m.update(extra)
    return m


def send(bank, cap, *, mirror=False, **extra):
    m = meta(cap, mirror=mirror, **extra)
    distance = {467: .2141703367, 469: .2754719853, 474: .2411606908, 488: .1299114227}[cap]
    return bank.assign(track_id=m['track_id'], feature=feature(distance), confidence=.9,
        area=m['detector_area_ratio']*640*480, frame_index=m['frame_index'],
        bbox_quality_ok=m['quality_bbox_ok'], bbox_quality_tier=m['bbox_quality_tier'],
        bbox_quality_reason=m.get('quality_bbox_reason',''), sample_metadata=m,
        preferred_uid=1 if cap >= 474 else None, preferred_candidate_ok=cap >= 474)


def ready(*, mirror=False):
    b = IdentityBank(IdentityBankConfig(new_identity_confirm_frames=1, min_area=0,
        controlled_handoff_enable=True, template_crosscheck_enable=True,
        camera_hfov_deg=60., mapped_bad_quality_returns_unassigned=True))
    m = meta(459, raw=4, mirror=mirror)
    assert b.assign(track_id=4, feature=feature(0), confidence=.95,
        area=m['detector_area_ratio']*640*480, frame_index=157, sample_metadata=m) == 1
    b._bind_reacquired_identity(1, 7, 159, meta(467, mirror=mirror))
    assert send(b, 467, mirror=mirror) == 1
    assert send(b, 469, mirror=mirror) == 1
    assert b._mapped_position_observations[7]['last']['capture_frame_id'] == 469
    return b


@pytest.mark.parametrize('mirror', [False, True])
def test_recorded_outward_crop_is_unknown_without_revoking_uid_or_renewing_position(mirror):
    b = ready(mirror=mirror)
    old = deepcopy(b.identities[1].last_strong_observation)
    samples = len(b.identities[1].features)
    assert send(b, 474, mirror=mirror) == 0
    assert b._mapped_position_observations[7]['last']['capture_frame_id'] == 469
    g = b.review_mapped_geometry(7, meta(488, mirror=mirror), 167)
    assert g['reason'] == 'center_jump,area_change' and g['ok'] is False
    assert g['yaw_compensated_center_jump_ratio'] == pytest.approx(.37456219746)
    assert g['area_similarity'] == pytest.approx(.32620239862)
    assert g['mapped_geometry_observation_only'] and not g['mapped_geometry_blocked']
    assert g['edge_exit_observation']['reference_capture_frame_id'] == 469
    assert g['edge_exit_observation']['age_sec'] == pytest.approx(.993886532)
    assert not b._mapped_geometry_conflicts and not b._geometry_revoked_uids
    assert send(b, 488, mirror=mirror) == 0
    assert b.last_assignments[7]['reason'] == 'mapped_edge_observation_only'
    assert not b.last_assignments[7]['bank_updated']
    assert b.track_to_uid[7] == 1
    assert not b._mapped_geometry_conflicts and not b._geometry_revoked_uids
    assert b.identities[1].last_strong_observation == old
    assert len(b.identities[1].features) == samples
    assert b._mapped_position_observations[7]['last']['capture_frame_id'] == 469


@pytest.mark.parametrize('fault', ['expired', 'wrong_raw', 'changed_entry', 'no_earlier',
    'reverse_motion', 'missing_yaw', 'competition', 'already_conflicted', 'short_body',
    'opposite_edge', 'not_edge', 'stale', 'swap', 'out_of_order', 'appearance', 'metadata_raw'])
def test_crop_cannot_cancel_missing_local_proof_or_independent_contradictions(fault):
    b = ready(); m = meta(488)
    row = b._mapped_position_observations[7]
    if fault == 'expired': m['capture_timestamp'] = ROWS[469][1]+1.001
    elif fault == 'wrong_raw': row['uid'] = 2
    elif fault == 'changed_entry': row['entry'] = deepcopy(row['entry'])
    elif fault == 'no_earlier': row['earlier'] = None
    elif fault == 'reverse_motion': row['earlier']['bbox'] = (0.,0.,100.,476.)
    elif fault == 'missing_yaw': m.pop('integrated_yaw_deg')
    elif fault == 'competition': m['identity_competition']['passed'] = False
    elif fault == 'appearance': m['identity_competition']['distance'] = .8
    elif fault == 'metadata_raw': m['track_id'] = 8
    elif fault == 'already_conflicted':
        b._mapped_geometry_conflicts[7] = dict(uid=1, search_contradiction=True,
            rejected_frame=161, reference=deepcopy(b._reacquire_search_anchors[1]))
    elif fault == 'short_body':
        m.update(detector_bbox=(0.,200.,63.,400.), detector_area_ratio=63*200/(640*480))
    elif fault == 'opposite_edge':
        m.update(detector_bbox=(577.,2.,640.,473.), detector_center_x_ratio=1217/1280)
    elif fault == 'not_edge': m['detector_bbox'] = (20.,2.,83.,473.)
    elif fault == 'stale': m['is_fresh'] = False
    elif fault == 'swap': m['quality_bbox_reason'] = 'identity_swap_competing_track'
    elif fault == 'out_of_order': m['capture_frame_id'] = 469
    g = b.review_mapped_geometry(7, m, 167, commit=True)
    assert not g.get('mapped_geometry_observation_only')
    if fault not in ('stale',):
        assert g['mapped_geometry_blocked']


def test_pure_association_review_uses_same_crop_policy_without_mutating_bank():
    b = ready()
    previous = b._mapped_position_observations[7]
    for _ in range(3):
        assert b.review_mapped_geometry(7, meta(488), 167)['mapped_geometry_observation_only']
    assert b._mapped_position_observations[7] is previous
    assert b.track_to_uid[7] == 1 and not b._geometry_revoked_uids
    b.reset()
    assert not b._mapped_position_observations


@pytest.mark.parametrize('swap', [False, True])
def test_whole_frame_geometry_review_keeps_dimensions_and_explicit_swap_priority(swap):
    b = ready(); m = meta(488)
    # The run retained CAP459 as the trusted reference during quarantine;
    # make that saved reference explicit for this pre-assignment entry point.
    b.identities[1].last_strong_observation = deepcopy(b._reacquire_search_anchors[1])
    # The real tracker sends width/height separately, before assign() runs.
    observation = dict(raw_track_id=7, mapped_uid=1, detector_bbox=m['detector_bbox'],
        capture_frame_id=488, capture_timestamp=m['capture_timestamp'],
        integrated_yaw_deg=m['integrated_yaw_deg'], feature=feature(.13),
        confidence=.746, is_fresh=True, identity_swap=swap,
        identity_competition=m['identity_competition'])
    b.observe_frame_evidence(frame_index=167, observations=[observation], width=640, height=480)
    assert bool(b._geometry_revoked_uids) is swap
    if not swap:
        assert b.track_to_uid[7] == 1
        assert b._mapped_position_observations[7]['edge_exit']['last']['capture_frame_id'] == 488
        assert send(b, 488) == 0
        assert b.last_assignments[7]['reason'] == 'mapped_edge_observation_only'


def test_continued_crop_stays_unknown_without_renewing_accepted_position():
    b = ready()
    assert send(b, 488) == 0
    for offset in range(1, 10):
        assert send(b, 488, capture_frame_id=488+offset,
            capture_timestamp=ROWS[488][1]+.10*offset, frame_index=167+offset) == 0
        g = b.last_assignments[7]['reacquire_geometry']
        if g.get('reason') == 'center_jump,area_change':
            assert g['mapped_geometry_observation_only']
            assert g['edge_exit_observation']['observation_chain']
        assert b.track_to_uid[7] == 1 and not b._geometry_revoked_uids
        assert b._mapped_position_observations[7]['last']['capture_frame_id'] == 469


def test_lost_crop_chain_cannot_cancel_a_later_geometry_contradiction():
    b = ready()
    assert send(b, 488) == 0
    m = meta(488, capture_frame_id=498, capture_timestamp=ROWS[488][1]+.351)
    g = b.review_mapped_geometry(7, m, 168, commit=True)
    assert not g.get('mapped_geometry_observation_only')
    assert g['mapped_geometry_blocked'] and b._geometry_revoked_uids


def test_continuous_edge_sliver_never_becomes_identity_conflict_as_it_disappears():
    b = ready()
    assert send(b, 488) == 0
    for offset, width in enumerate((30., 8., 1.), 1):
        box = (.5, 2., .5+width, 473.)
        m = meta(488, capture_frame_id=488+offset,
            capture_timestamp=ROWS[488][1]+.1*offset, frame_index=167+offset,
            detector_bbox=box, detector_center_x_ratio=(1.+width)/1280,
            detector_area_ratio=width*471/(640*480))
        g = b.review_mapped_geometry(7, m, 167+offset, commit=True)
        assert g['mapped_geometry_observation_only']
        assert not b._geometry_revoked_uids and not b._mapped_geometry_conflicts
        assert b._mapped_position_observations[7]['last']['capture_frame_id'] == 469
