"""Candidate memory can cross a short raw-ID break; it cannot become a gallery."""
from copy import deepcopy

import pytest

from test_similar_follow_bank import make_bank, activate
from test_reacquire_crosscheck import send, meta
from test_cap874_identity_reacquire import metadata
from tools.replay_cap334_recovery import gallery_snapshot


def test_unique_recent_raw_handoff_retains_follow_across_search_side():
    bank = make_bank()
    activate(bank)
    before = gallery_snapshot(bank)
    origin = deepcopy(bank._similar_follow_states[(1, 3)])
    assert send(bank, 22, 3.364, full=.35, part=None, track=6,
                extra={'search_direction_compatible': False}) == 1
    state = bank._similar_follow_states[(1, 6)]
    assert (1, 3) not in bank._similar_follow_states
    assert bank.track_to_uid == {6: 1}
    assert state['origin_cap'] == origin['origin_cap']
    assert origin['observation']['track_id'] == 3
    assert state['observation']['track_id'] == 6
    assert bank.last_assignments[6]['similar_follow']['handoff_from_track_id'] == 3
    assert bank.last_assignments[6]['similar_follow']['handoff_reference_cap'] == 21
    assert not bank.last_assignments[6]['bank_updated']
    assert 1 in bank._similar_learning_fences
    assert gallery_snapshot(bank) == before


@pytest.mark.parametrize('kind', ['gap', 'multiple', 'competition', 'appearance',
                                  'geometry', 'source_conflict', 'exclusion'])
def test_raw_handoff_requires_current_evidence(monkeypatch, kind):
    bank = make_bank()
    activate(bank)
    extra = {}
    ts, full = 3.3, .35
    if kind == 'gap':
        ts = 3.451
    elif kind == 'multiple':
        extra['candidate_count'] = 2
    elif kind == 'competition':
        extra['identity_competition'] = dict(uid=1, frame_index=22,
            candidate_count=1, passed=False, reason='ambiguous')
    elif kind == 'appearance':
        full = .51
    elif kind == 'geometry':
        extra = metadata(22, ts, (470., 40., 630., 460.), search=False)
    elif kind == 'source_conflict':
        bank._mapped_geometry_conflicts[3] = {'uid': 1}
    else:
        monkeypatch.setattr(bank, 'search_exclusion_for', lambda *a, **k:
                            {'reason': 'previously_covisisible_different_person'})
    before = gallery_snapshot(bank)
    assert send(bank, 22, ts, full=full, part=None, track=6, extra=extra) != 1
    assert bank.track_to_uid.get(6) != 1
    assert gallery_snapshot(bank)['1'] == before['1']


def crop_meta(cap, ts, box=(470., 0., 639., 479.), weak=True):
    result = metadata(cap, ts, box, search=True)
    result.update(image_width=640, image_height=480)
    if weak:
        result.update(quality_bbox_ok=False, bbox_quality_tier='weak',
                      quality_bbox_reason='edge_touch>2',
                      detector_edge_touch_count=3, edge_touch_count=3)
    return result


def cropped_bank():
    bank = make_bank()
    for cap, ts, expected in ((20, 3., 0), (21, 3.1, 1)):
        assert send(bank, cap, ts, full=.44, part=None, track=3, search=True,
                    extra=crop_meta(cap, ts, (450., 0., 620., 479.), weak=False)) == expected
    return bank


def test_three_edge_crop_retains_follow_without_learning_or_rolling_crop_budget():
    bank = cropped_bank()
    before = gallery_snapshot(bank)
    for cap, ts in ((22, 3.3), (23, 3.5), (24, 3.7)):
        assert send(bank, cap, ts, full=.44, part=None, track=3, search=True,
                    extra=crop_meta(cap, ts)) == 1
        detail = bank.last_assignments[3]
        assert detail['similar_follow']['crop_continuation']
        assert not detail['bank_updated']
        assert bank._similar_follow_states[(1, 3)]['last_strong_timestamp'] == 3.1
        assert gallery_snapshot(bank) == before
    assert send(bank, 25, 3.86, full=.44, part=None, track=3, search=True,
                extra=crop_meta(25, 3.86)) == 0
    assert gallery_snapshot(bank) == before


@pytest.mark.parametrize('change', [
    {'quality_bbox_reason': 'aspect<0.18'}, {'quality_bbox_reason': 'edge_touch>2;aspect'},
    {'is_fresh': False}, {'detector_bbox': [0., 0., 640., 480.]},
    {'detector_bbox': [615., 0., 639., 479.]},
    {'identity_competition': dict(uid=1, frame_index=22, candidate_count=1,
                                  passed=False, reason='ambiguous')},
])
def test_crop_exception_is_not_a_general_quality_override(change):
    bank = cropped_bank()
    extra = {**crop_meta(22, 3.3), **change}
    assert send(bank, 22, 3.3, full=.44, part=None, track=3, search=True, extra=extra) == 0


def test_duplicate_and_predicted_frames_do_not_renew_candidate_memory():
    bank = make_bank()
    activate(bank)
    before = deepcopy(bank._similar_follow_states[(1, 3)])
    assert send(bank, 21, 3.2, full=.44, part=None, track=3) == 0
    assert bank._similar_follow_states[(1, 3)] == before
    bank.assign(track_id=3, feature=None, confidence=.95, area=84000,
        frame_index=22, sample_metadata={**meta(22, 3.3), 'is_fresh': False})
    assert bank._similar_follow_states[(1, 3)] == before
    assert send(bank, 23, 3.4, full=.44, part=None, track=3) == 1
