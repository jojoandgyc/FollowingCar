"""Pure temporal/geometry replay: no camera, driver or identity bank writes."""
import pytest

from car_control_modular.search_observation_retry import SearchObservationRetry


BOX = (1., 2., 365., 471.)


def tick(gate, t=10., cap=643, **changes):
    args = dict(now=t+.09, session=(1, 1), capture_id=cap,
                capture_timestamp=t, eligible=True, bbox=BOX, score=.84,
                blocked=True, zero_sent_at=None)
    args.update(changes)
    return gate.update(**args)


def entered():
    gate = SearchObservationRetry()
    assert tick(gate) is None
    decision = tick(gate, 10.1, 646, bbox=(1., 2., 439., 471.))
    assert decision.entered and decision.pause_rotation and not decision.completed
    return gate


def test_capture643_646_can_retry_once_after_already_observed():
    gate = entered()
    decision = tick(gate, 10.2, 648, zero_sent_at=10.21)
    assert decision.pause_rotation and not decision.completed  # captured before zero
    decision = tick(gate, 10.30, 651, zero_sent_at=10.21)
    assert decision.completed and not decision.pause_rotation
    assert decision.reason == "search_retry_post_zero_capture"
    for i in range(30):
        assert tick(gate, 10.4+i*.1, 652+i) is None
    assert gate.spent


@pytest.mark.parametrize("zero", [None, 9., 20., float('nan')])
def test_only_a_successful_zero_in_this_window_can_complete(zero):
    gate = entered()
    decision = tick(gate, 10.25, 648, zero_sent_at=zero)
    assert decision.pause_rotation and not decision.completed
    decision = tick(gate, 10.45, 651, zero_sent_at=zero)
    assert decision.completed and decision.reason == "search_retry_timeout"


def test_duplicate_old_and_stale_captures_never_build_entry_chain():
    gate = SearchObservationRetry()
    tick(gate)
    assert tick(gate) is None
    assert tick(gate, 9.9, 642) is None
    assert tick(gate, 10.1, 645, now=10.4) is None
    assert not gate.spent


@pytest.mark.parametrize("second", [
    dict(bbox=(450., 2., 639., 471.)),
    dict(bbox=(1., 2., 50., 80.)),
    dict(eligible=False), dict(bbox=None), dict(blocked=False),
])
def test_no_retry_for_jump_fragment_missing_or_nonblocked_candidate(second):
    gate = SearchObservationRetry()
    tick(gate)
    assert tick(gate, 10.1, 646, **second) is None
    assert not gate.active


def test_long_gap_cannot_join_two_observations():
    gate = SearchObservationRetry()
    tick(gate)
    assert tick(gate, 10.4, 646) is None


@pytest.mark.parametrize("changes", [dict(eligible=False), dict(bbox=(450., 2., 639., 471.))])
def test_lost_or_changed_candidate_releases_immediately_without_renewing_budget(changes):
    gate = entered()
    assert tick(gate, 10.2, 648, **changes).completed
    for i in range(10):
        assert tick(gate, 10.3+i*.1, 650+i) is None


def test_zero_refresh_does_not_extend_absolute_deadline():
    gate = entered()
    deadline = gate.deadline
    for i in range(2):
        tick(gate, 10.2+i*.1, 648+i, zero_sent_at=10.28+i*.1)
        assert gate.deadline == deadline
    assert tick(gate, 10.41, 651).completed


def test_new_search_epoch_or_uid_gets_new_budget_not_raw_track_drift():
    gate = entered()
    gate.release()
    assert tick(gate, 10.2, 648) is None
    assert tick(gate, 10.3, 650, session=(2, 1)) is None
    assert tick(gate, 10.4, 652, session=(2, 1)).entered
    assert tick(gate, 10.5, 654, session=None) is None
    assert not gate.spent and not gate.active


def test_pre_zero_capture_after_zero_receipt_still_waits_and_no_identity_grant():
    gate = entered()
    result = tick(gate, 10.20, 648, now=10.30, zero_sent_at=10.25)
    assert result.pause_rotation and not result.preferred_target_match
    assert result.source == "credible_retry"
