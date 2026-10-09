"""Saved geometry/time, synthetic descriptors: fixed-origin pose/scale policy.

The recorded runtime starts search at CAP1171. These continuous-follow tests
explicitly keep search false; they are not predictions of closed-loop motion.
"""
from copy import deepcopy

import pytest

from rk_vision.template_memory import TemplateMemory
from test_cap1171_pose_observation_clock import checkpoint, meta, send, ROWS
from test_cap225_pose_continuity import gallery
from test_cap874_identity_reacquire import feature


SCALE_ROWS = {
    1170: (423, 29147.675965326, (317.759552, 124.4112244, 425.6681213, 466.6065674), 145.2734243),
    1171: (424, 29147.739537454, (314.9462585, 127.8544312, 423.5535889, 466.6363525), 146.0759268),
    1173: (425, 29147.840391238, (311.3014526, 128.3272705, 419.6635437, 463.0999146), 146.3343614),
    1175: (426, 29147.943311722, (304.2506714, 127.9286652, 413.8768616, 451.9230957), 146.7891845),
    1176: (427, 29148.003967839, (303.0201416, 128.1686096, 411.7766724, 436.8835449), 147.0691621),
    1178: (428, 29148.10965092, (301.6011353, 131.903656, 410.402771, 434.0678711), 147.0691621),
    1180: (429, 29148.206071773, (300.135376, 133.943634, 410.3542786, 431.7516479), 147.0691621),
    1182: (430, 29148.306864018, (298.9121094, 137.7090607, 409.715332, 433.9124146), 147.0691621),
    1184: (431, 29148.406288088, (301.756897, 141.0680847, 409.7220764, 434.6097412), 147.1349652),
    1186: (432, 29148.505194171, (303.7980957, 143.9761047, 408.5760803, 426.7346802), 147.5634234),
    1187: (433, 29148.575002269, (304.2073364, 143.955658, 408.019928, 424.75177), 148.3729238),
    1190: (434, 29148.735623776, (288.7178345, 144.7391357, 387.6772156, 409.098877), 149.0849793),
    1191: (435, 29148.771538425, (282.2803955, 145.464325, 380.473053, 406.7945862), 150.1575182),
    1194: (436, 29148.935518682, (246.2769165, 146.9421692, 335.191803, 404.7262573), 152.6926018),
    1196: (437, 29149.036407466, (215.8422241, 148.3467255, 299.0018311, 392.1112366), 154.8612317),
    1198: (438, 29149.135534919, (172.8042755, 151.5147095, 261.1343994, 398.997345), 157.1037805),
    1200: (439, 29149.237654797, (122.8725433, 147.6573944, 219.7236786, 402.8320312), 159.8989323),
}


def send_scale(bank, cap, **changes):
    frame, stamp, box, yaw = SCALE_ROWS[cap]
    values = dict(capture_frame_id=cap, frame_index=frame, capture_timestamp=stamp,
                  detector_bbox=box, integrated_yaw_deg=yaw)
    values.update(changes)
    return send(bank, **values)


def active_before_scale():
    bank = checkpoint()
    assert send(bank, 1166) == 1
    assert send(bank, 1168) == 1
    for cap in (1170, 1171, 1173, 1175):
        assert send_scale(bank, cap) == 1
    return bank


def test_same_source_pose_to_scale_preserves_fixed_gallery_origin_and_deadline():
    bank = active_before_scale()
    frozen = gallery(bank)
    origin = deepcopy(bank._appearance_verified[1]['pose_origin'])
    for cap in SCALE_ROWS:
        if cap < 1176 or cap == 1200:
            continue
        assert send_scale(bank, cap) == 1
        state = bank._appearance_verified[1]
        assert state['pose_origin'] == origin
        assert state['pose_started'] == ROWS[1161][1]
        assert state['pose_caps'] == [944]
        assert not bank.last_assignments[2]['bank_updated']
        assert bank._reacquire_quarantine.is_held(1)
        assert gallery(bank) == frozen
    assert send_scale(bank, 1200) == 0
    assert bank.last_assignments[2]['reason'] == 'secondary_evidence_unavailable'
    assert gallery(bank) == frozen


def test_pose_entry_template_height_gate_is_not_globally_relaxed():
    query = meta(1166, detector_bbox=SCALE_ROWS[1176][2])
    template = meta(944)
    assert not TemplateMemory.pose_template_usable(query, template)
    assert TemplateMemory.pose_template_usable(query, template, allow_scale=True)
    bank = checkpoint()
    assert send_scale(bank, 1176, capture_timestamp=ROWS[1161][1]+.1) == 0


def test_normal_vertical_bridge_cannot_evict_original_retained_pose_template():
    bank = active_before_scale()
    memory = bank.identities[1].template_memory
    template = meta(944, capture_frame_id=914,
        detector_bbox=[412.2820435, 60.8995056, 616.1331177, 476.5831909])
    # Deliberately bad other approved view. A normal vertical-border match
    # must not discard the still-valid original reference solely by routing.
    # Restore an already approved historical row; remember() correctly refuses
    # to insert a timestamp older than the live observation watermark.
    memory.recent['partial'].append((feature(2.), template))
    assert send_scale(bank, 1176) == 1
    assert send_scale(bank, 1178) == 1
    frozen = gallery(bank)
    assert send_scale(bank, 1180) == 1
    proof = bank.last_assignments[2]['reacquire_recent_partial_evidence']
    assert proof['winner_cap'] == 944
    assert 914 in proof['comparable_caps']
    assert 914 not in proof['pose_bridge_caps']
    assert bank._appearance_verified[1]['pose_caps'] == [944]
    assert gallery(bank) == frozen


def test_exact_region_conflict_still_wins_over_bounded_pose_scale_evidence():
    bank = active_before_scale()
    current = meta(1166, capture_frame_id=1169,
        capture_timestamp=ROWS[1168][1]+.01, detector_bbox=SCALE_ROWS[1176][2])
    bank.identities[1].template_memory.recent['partial'].append((feature(2.), current))
    assert send_scale(bank, 1176) == 0
    assert bank.last_assignments[2]['reason'] == 'recent_partial_conflict'


@pytest.mark.parametrize('case', ['no_source', 'no_epoch', 'pending', 'search', 'side',
    'tiny', 'large_step', 'large_total_scale', 'large_total_shape', 'bad_full', 'bad_partial',
    'competition', 'conflict', 'revoked', 'stale', 'duplicate', 'expired', 'new_track'])
def test_scale_retention_still_requires_current_evidence_and_original_source(case):
    bank = active_before_scale()
    frozen = gallery(bank)
    changes = {}
    if case == 'no_source': bank._appearance_verified[1].pop('continuation_source')
    if case == 'no_epoch': bank._appearance_verified[1].pop('pose_started')
    if case == 'pending': bank._appearance_verified[1]['pending_continuation_deadline'] = SCALE_ROWS[1176][1]+.1
    if case == 'search': changes['search_reacquire_context_active'] = True
    if case == 'side': changes['detector_bbox'] = [0,128,108,436]
    if case == 'tiny': changes['detector_bbox'] = [310,128,350,220]
    if case == 'large_step': changes['detector_bbox'] = [320,160,400,400]
    if case == 'large_total_scale':
        # Previous frame is close enough: only the frozen origin can reject.
        bank._appearance_verified[1]['metadata']['detector_bbox'] = [320,160,405,405]
        changes['detector_bbox'] = [320,165,400,400]
    if case == 'large_total_shape':
        bank._appearance_verified[1]['metadata']['detector_bbox'] = [275,128,443,436]
        changes['detector_bbox'] = [275,128,443,436]
    if case == 'bad_full': changes['full'] = .31
    if case == 'bad_partial': changes['part'] = .46
    if case == 'competition': changes['candidate_count'] = 2
    if case == 'conflict': bank._mapped_geometry_conflicts[2] = dict(uid=1,
        search_contradiction=True, reference=deepcopy(bank.identities[1].last_strong_observation))
    if case == 'revoked': bank._geometry_revoked_uids[1] = {}
    if case == 'stale': changes['is_fresh'] = False
    if case == 'duplicate': changes['capture_frame_id'] = 1175
    if case == 'expired': changes['capture_timestamp'] = ROWS[1161][1]+2.01
    if case == 'new_track': changes['track_id'] = 3
    assert send_scale(bank, 1176, **changes) == 0
    assert not bank.last_assignments[changes.get('track_id',2)]['bank_updated']
    assert gallery(bank) == frozen


@pytest.mark.parametrize('origin', [None, {}, {'image_width':640,'image_height':480}])
def test_incomplete_original_epoch_cannot_restart_cumulative_scale_or_raise(origin):
    bank = active_before_scale()
    if origin is None:
        bank._appearance_verified[1].pop('pose_origin')
    else:
        bank._appearance_verified[1]['pose_origin'] = origin
    assert send_scale(bank, 1176) == 0
    assert not bank.last_assignments[2]['bank_updated']
