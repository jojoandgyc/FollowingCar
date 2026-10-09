"""CAP555 precursor: real capture geometry/times, synthetic cosine vectors.

Source: run_20260928_223853_35353_20b3f2ee/reid_diagnostics/events.jsonl.
This starts at the recorded, already-confirmed partial handoff checkpoint;
it tests the policy and learning lifecycle, not OSNet or motor behaviour.
"""
from copy import deepcopy
import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig, _geometry_observation
from test_cap874_identity_reacquire import feature, metadata


# CAP: (control frame, capture time, detector box, integrated yaw,
#       authorization full distance including recent floor, torso distance).
# Torso uses logged comparable-recent distance through CAP436, then the logged
# archive distance at scheduled updates to exercise the unchanged .30 gate.
ROWS = {
    308: (115, 5554.647182475, [326.9185, 75.5336, 495.6356, 476.2596], 63.84304, 0., 0.),
    390: (163, 5558.905694164, [43.7493, 95.2580, 180.7082, 476.0384], 34.19299, .181009, .386055),
    419: (174, 5560.406986593, [234.7871, 139.0424, 341.4836, 457.4405], 22.21175, .185051, .346792),
    423: (175, 5560.639610184, [220.2993, 137.7290, 331.3986, 448.5391], 22.37508, .142282, .328788),
    426: (176, 5560.837792322, [215.8114, 140.3796, 323.8444, 442.7995], 21.97344, .183369, .394636),
    428: (177, 5560.938922527, [214.7878, 142.5462, 323.0290, 445.8233], 21.72826, .165960, .377763),
    432: (178, 5561.137533405, [211.2430, 135.9375, 321.5258, 442.4886], 21.93884, .172006, .341544),
    434: (179, 5561.242336182, [215.9986, 135.4059, 322.9750, 448.3584], 21.70564, .157222, .333664),
    436: (180, 5561.370880310, [220.2673, 133.0318, 330.9436, 449.6433], 20.93489, .195753, .352819),
    450: (185, 5562.101747072, [235.6445, 105.1118, 368.3962, 475.5146], 16.77225, .170159, .312723),
    465: (190, 5562.869702969, [285.7465, 74.2393, 461.1716, 476.8297], 14.40133, .227027, .328397),
    480: (195, 5563.634746940, [322.9517, 26.6248, 561.8357, 476.2325], 20.46362, .228758, .279972),
}


def meta(cap, **changes):
    frame, stamp, box, yaw, _, _ = ROWS[cap]
    m = dict(metadata(cap, stamp, box, yaw, search=cap == 390),
             frame_index=frame, control_frame_id=frame,
             track_id=1 if cap == 308 else 2, image_width=640, image_height=480,
             partial_feature_source='osnet_torso', partial_observation=cap in (308, 390),
             source_detection_index=0, search_direction='left' if cap == 390 else None)
    m.update(changes)
    return m


def checkpoint():
    b = IdentityBank(IdentityBankConfig(
        template_memory_enable=True, template_crosscheck_enable=True,
        appearance_region_safety_enable=True, partial_match_threshold=.45,
        partial_confirm_threshold=.40, controlled_handoff_enable=True,
        mapped_verify_threshold=.45, update_interval=5, camera_hfov_deg=60.))
    b._create_identity(feature(0), ROWS[308][0], meta(308), feature(0))
    b._bind_reacquired_identity(1, 2, ROWS[390][0], meta(390))
    previous = _geometry_observation(meta(419), ROWS[419][0])
    b.identities[1].last_strong_observation = deepcopy(previous)
    b._remember_track_seen(2, 1, ROWS[419][0])
    row = b._candidate_observations.observe(1, 2, meta(419), previous, True, lambda _: True)
    row.update(confirmed=deepcopy(previous), late_confirmed_source='partial')
    return b


def send(b, cap, *, full=None, partial=None, **changes):
    frame, _, box, _, fd, pd = ROWS[cap]
    m = meta(cap, **changes)
    return b.assign(
        track_id=2, frame_index=frame, feature=feature(fd if full is None else full),
        partial_feature=feature(pd if partial is None else partial), confidence=.90,
        area=(box[2]-box[0])*(box[3]-box[1]), sample_metadata=m,
        bbox_quality_ok=m['quality_bbox_ok'], bbox_quality_tier=m['bbox_quality_tier'])


def test_cap423_426_428_strong_proof_exits_partial_return_before_any_learning():
    b = checkpoint()
    old = deepcopy(b.identities[1].template_memory.last_learning)
    for cap, streak in ((423, 1), (426, 2), (428, 3)):
        assert send(b, cap) == 1
        a = b.last_assignments[2]
        if cap == 423:
            assert a.get('partial_continuation_regular_verification') is True
            assert a['partial_continuation_verification_source'] == 'strong'
        assert a['quarantine_region_pair']['qualified'] is False
        assert a['template_quarantine_streak'] == streak
        assert a['template_update_quarantined'] is (cap != 428)
        assert not a['bank_updated']  # Next scheduled update is control frame 180.
        assert b.identities[1].template_memory.last_learning == old
    assert b.last_assignments[2]['template_quarantine_reason'] == 'released'
    for cap in (432, 434, 436):
        assert send(b, cap) == 1
    assert b.last_assignments[2]['bank_updated']
    assert b.identities[1].template_memory.last_learning['strong'][1] == 436
    # The torso learning threshold stays .30: releasing isolation does not
    # authorize learning the .353 crop. A later qualifying crop can update it.
    assert b.identities[1].template_memory.last_learning['partial'][1] == 308
    for cap in (450, 465, 480):
        assert send(b, cap) == 1
    assert b.identities[1].template_memory.last_learning['partial'][1] == 480


def test_archive_best_match_cannot_bypass_recent_full_distance_floor():
    b = checkpoint()
    e = b.identities[1]
    # Old archive appears perfect; every frozen recent full sample is .25 away.
    e.template_memory.recent['strong'] = [(feature(.25), meta(308))]
    old = deepcopy(e.template_memory.last_learning)
    for cap in (423, 426, 428):
        assert send(b, cap, full=0., partial=.35) == 1
        a = b.last_assignments[2]
        assert a['authorization_full_distance_floor'] == pytest.approx(.25)
        assert not a.get('partial_continuation_regular_verification')
        assert a['template_quarantine_streak'] == 0
        assert a['template_update_quarantined']
        assert not a['bank_updated']
    assert e.template_memory.last_learning == old


@pytest.mark.parametrize('kind', ['tentative', 'conflict', 'competition', 'geometry_conflict',
                                  'weak', 'stale', 'duplicate', 'gap', 'search', 'full_distance'])
def test_partial_confirmation_cannot_launder_invalid_strong_proof(kind):
    b = checkpoint()
    old = deepcopy(b.identities[1].template_memory.last_learning)
    for cap in (423, 426, 428):
        changes = {}
        if kind == 'tentative':
            changes['partial'] = .401
        elif kind == 'conflict':
            changes['partial'] = .46
        elif kind == 'competition':
            changes['identity_competition'] = dict(uid=1, frame_index=ROWS[cap][0], passed=False)
        elif kind == 'geometry_conflict':
            b._mapped_geometry_conflicts[2] = dict(uid=1, search_contradiction=True,
                reference=deepcopy(b.identities[1].last_strong_observation))
        elif kind == 'weak':
            changes.update(quality_bbox_ok=False, bbox_quality_tier='weak')
        elif kind == 'stale':
            changes['is_fresh'] = False
        elif kind == 'duplicate':
            changes.update(capture_frame_id=423, capture_timestamp=ROWS[423][1])
        elif kind == 'gap':
            changes['capture_timestamp'] = ROWS[423][1]+.4*(ROWS[cap][0]-175)
        elif kind == 'search':
            changes.update(search_reacquire_context_active=True, search_direction='left')
        elif kind == 'full_distance':
            changes['full'] = .201
        send(b, cap, **changes)
        assert b.last_assignments[2]['template_update_quarantined']
        assert not b.last_assignments[2]['bank_updated']
    assert b.identities[1].template_memory.last_learning == old


def test_stricter_configured_quarantine_threshold_is_respected():
    b = checkpoint()
    b._reacquire_quarantine.max_strong_distance = .10
    for cap in (423, 426, 428):
        send(b, cap)
        assert not b.last_assignments[2].get('partial_continuation_regular_verification')
    assert b._reacquire_quarantine.is_held(1)


def test_unconfirmed_candidate_never_uses_confirmed_partial_exit():
    b = checkpoint()
    b._candidate_observations.rows[(1, 2)].pop('confirmed')
    assert send(b, 423) == 1  # Normal independent mapped verification can still pass.
    assert not b.last_assignments[2].get('partial_continuation_regular_verification')
