from copy import deepcopy

import pytest

from rk_vision.similar_follow import evaluate_similar_follow


def sample(cap=334, timestamp=10., **changes):
    result = dict(capture_frame_id=cap, capture_timestamp=timestamp, track_id=3,
                  is_fresh=True, quality_bbox_ok=True, bbox_quality_tier='strong',
                  detector_bbox=[100., 20., 220., 400.], integrated_yaw_deg=2.,
                  partial_distance=None, nested={'values': [1]})
    result.update(changes)
    return result


def evaluate(current=None, **changes):
    kwargs = dict(uid=1, track_id=3, current=sample() if current is None else current,
                  geometry=dict(ok=True, yaw_compensated_center_jump_ratio=.03,
                                area_similarity=.90),
                  competition_ok=True, blocked=False, full_distance=.463,
                  direction_compatible=True)
    kwargs.update(changes)
    return evaluate_similar_follow(**kwargs)


def active():
    first = evaluate()
    return evaluate(sample(336, 10.1), state=first.state, full_distance=.441)


def test_two_new_observations_follow_without_torso_or_template_approval():
    first = evaluate()
    second = evaluate(sample(336, 10.1), state=first.state, full_distance=.441)
    assert first.status == 'observe'
    assert second.status == 'follow'
    assert not first.learning_allowed and not second.learning_allowed
    assert second.state['origin_cap'] == 334
    assert second.state['last_cap'] == 336


def test_active_retention_accepts_cap_range_above_entry_limit():
    result = evaluate(sample(338, 10.2), state=active().state, full_distance=.513)
    assert result.status == 'follow'
    assert not result.learning_allowed


def test_pending_cannot_use_retention_threshold():
    result = evaluate(sample(336, 10.1), state=evaluate().state, full_distance=.513)
    assert result.status == 'reject'
    assert result.state is None


def test_fresh_evidence_can_continue_beyond_two_seconds():
    state = active().state
    for index in range(1, 31):
        result = evaluate(sample(336 + index, 10.1 + .1 * index),
                          state=state, full_distance=.513, direction_compatible=False)
        assert result.status == 'follow'
        assert not result.learning_allowed
        state = result.state
    assert state['origin_timestamp'] == 10.


@pytest.mark.parametrize('direction', [False, None, 1, 'true'])
def test_new_candidate_cannot_start_on_unknown_or_incompatible_side(direction):
    assert evaluate(direction_compatible=direction).reason == 'entry_direction_unverified'


def test_continuous_pending_candidate_can_cross_center():
    result = evaluate(sample(336, 10.1), state=evaluate().state,
                      direction_compatible=False)
    assert result.status == 'follow'


def test_incompatible_candidate_cannot_reuse_crossing_after_gap():
    result = evaluate(sample(350, 11.), state=active().state, direction_compatible=False)
    assert result.reason == 'observation_gap'
    assert result.state is None
    assert evaluate(sample(352, 11.1), state=result.state,
                    direction_compatible=False).status == 'reject'


@pytest.mark.parametrize('cap,timestamp', [(336, 10.1), (335, 10.2), (337, 10.1), (337, 10.)])
def test_duplicate_or_out_of_order_never_grants_follow_or_refreshes(cap, timestamp):
    state = active().state
    result = evaluate(sample(cap, timestamp), state=state)
    assert result.status == 'ignore'
    assert result.state == state
    assert result.state is not state
    assert not result.learning_allowed


@pytest.mark.parametrize('changes,reason', [
    ({'blocked': True}, 'identity_blocked'),
    ({'blocked': None}, 'identity_blocked'),
    ({'competition_ok': False}, 'competition_unverified'),
    ({'competition_ok': None}, 'competition_unverified'),
    ({'partial_conflict': True}, 'reliable_partial_conflict'),
    ({'partial_conflict': None}, 'reliable_partial_conflict'),
    ({'full_distance': .551}, 'full_distance_conflict'),
])
def test_current_conflicts_clear_state(changes, reason):
    result = evaluate(sample(338, 10.2), state=active().state, **changes)
    assert result.status == 'reject'
    assert result.reason == reason
    assert result.state is None


@pytest.mark.parametrize('changes', [
    {'is_fresh': False}, {'is_fresh': 1}, {'quality_bbox_ok': False},
    {'bbox_quality_tier': 'weak'}, {'search_observation_only': True},
    {'observation_only': True}, {'preferred_search_low_confidence': True},
])
def test_quality_or_staleness_clears_state(changes):
    result = evaluate(sample(338, 10.2, **changes), state=active().state)
    assert result.reason == 'observation_unverified'
    assert result.state is None


@pytest.mark.parametrize('geometry', [None, {}, {'ok': False},
    {'ok': True, 'yaw_compensated_center_jump_ratio': .201, 'area_similarity': .9},
    {'ok': True, 'yaw_compensated_center_jump_ratio': .1, 'area_similarity': .549},
    {'ok': True, 'yaw_compensated_center_jump_ratio': float('nan'), 'area_similarity': .9},
    {'ok': True, 'yaw_compensated_center_jump_ratio': .1, 'area_similarity': float('inf')},
    {'ok': True, 'yaw_compensated_center_jump_ratio': True, 'area_similarity': .9},
])
def test_missing_or_discontinuous_local_geometry_rejects(geometry):
    result = evaluate(sample(338, 10.2), state=active().state, geometry=geometry)
    assert result.status == 'reject' and result.state is None


def test_local_geometry_and_time_at_bounds_accept():
    result = evaluate(sample(338, 10.6), state=active().state,
                      geometry=dict(ok=True, yaw_compensated_center_jump_ratio=.20,
                                    area_similarity=.55))
    assert result.status == 'follow'


@pytest.mark.parametrize('changes', [
    {'uid': None}, {'uid': True}, {'track_id': 0},
    {'full_distance': float('nan')}, {'full_distance': '0.1'}, {'full_distance': -.1},
    {'entry_limit': float('inf')}, {'max_gap': 0}, {'max_gap': None},
    {'full_distance': 10 ** 1000},
    {'entry_limit': .5, 'retain_limit': .4},
])
def test_bad_parameters_fail_closed(changes):
    assert evaluate(**changes).status == 'reject'


@pytest.mark.parametrize('changes', [
    {'capture_frame_id': None}, {'capture_frame_id': True},
    {'capture_timestamp': None}, {'capture_timestamp': float('nan')},
    {'capture_timestamp': -1}, {'track_id': 4},
])
def test_invalid_capture_rejected(changes):
    assert evaluate(sample(**changes)).status == 'reject'


def test_loose_caller_limits_are_capped():
    assert evaluate(full_distance=.501, entry_limit=1., retain_limit=1.).status == 'reject'
    assert evaluate(sample(338, 10.2), state=active().state, full_distance=.551,
                    entry_limit=1., retain_limit=1.).status == 'reject'
    assert evaluate(sample(338, 10.7), state=active().state, max_gap=10.).reason == 'observation_gap'


def test_stricter_configurable_limits_respected():
    assert evaluate(full_distance=.46, entry_limit=.45).status == 'reject'
    assert evaluate(sample(338, 10.2), state=active().state,
                    full_distance=.51, retain_limit=.50).status == 'reject'


@pytest.mark.parametrize('changes', [
    {'uid': 2}, {'track_id': 4}, {'entry_direction_compatible': False},
    {'count': 1, 'active': True}, {'last_timestamp': 50}, {'origin_cap': 999},
    {'active': 1}, {'observation': None},
])
def test_inconsistent_state_cannot_grant_continuation(changes):
    state = active().state
    state.update(changes)
    result = evaluate(sample(338, 10.2), state=state)
    assert result.reason == 'invalid_state'
    assert result.state is None


def test_inputs_and_saved_nested_metadata_are_not_mutated_or_aliased():
    current = sample()
    snapshot = deepcopy(current)
    first = evaluate(current)
    assert current == snapshot
    current['nested']['values'].append(2)
    assert first.state['observation']['nested']['values'] == [1]
    state_before = deepcopy(first.state)
    next_sample = sample(336, 10.1)
    next_before = deepcopy(next_sample)
    second = evaluate(next_sample, state=first.state)
    assert first.state == state_before
    assert next_sample == next_before
    second.state['observation']['nested']['values'].append(3)
    assert next_sample == next_before


def test_none_or_missing_torso_is_not_an_identity_conflict():
    for partial in (None, float('nan'), .41):
        result = evaluate(sample(338, 10.2, partial_distance=partial), state=active().state)
        assert result.status == 'follow'
        assert not result.learning_allowed
