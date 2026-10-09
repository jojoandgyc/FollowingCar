"""A newly bound candidate cannot replace the finite search's direction proof."""
from copy import deepcopy
from queue import Queue

import pytest

from car_control_modular.search_brake_observation import provisional_reacquire_direction
from test_brake_hold_direction_observation import owner, sample, consume


def cap589_assignment(**changes):
    # Exact relevant fields from the 20261009_214211 event. Initial enrollment
    # was confirmed, but no post-binding continuation/recovery had passed.
    assignment = dict(
        uid=1, best_uid=1, reason="preferred_search_late_reacquire",
        template_update_quarantined=True, template_quarantine_reason="armed",
        template_quarantine_streak=0, initial_identity_confirmed=True,
        bbox_quality_ok=True, reacquire_geometry_ok=True,
        identity_competition=dict(uid=1, passed=True, reason="single_candidate"),
    )
    assignment.update(changes)
    return assignment


def test_cap589_is_provisional_despite_formal_uid_and_initial_confirmation():
    assignment = cap589_assignment()
    before = deepcopy(assignment)
    assert provisional_reacquire_direction(assignment)
    assert assignment == before


@pytest.mark.parametrize("reason", [
    "preferred_search_late_reacquire", "preferred_search_mapped_late_reacquire",
    "preferred_search_soft_reacquire",
])
def test_only_new_late_or_soft_binding_is_provisional(reason):
    assert provisional_reacquire_direction(cap589_assignment(reason=reason))


@pytest.mark.parametrize("changes", [
    {"reason": "mapped_verified_continuation"},
    {"reason": "mapped_observed_partial"},
    {"reason": "preferred_search_reacquire"},
    {"reason": "preferred_search_late_candidate_wait", "uid": 0},
    {"template_update_quarantined": False},
    {"template_quarantine_reason": "confirming", "template_quarantine_streak": 1},
    {"template_quarantine_reason": "minimum_duration", "template_quarantine_streak": 3},
    {"template_quarantine_reason": "region_pair_confirming", "template_quarantine_streak": 2},
    {"template_quarantine_reason": "duplicate_capture"},
    {"template_quarantine_reason": "released", "template_update_quarantined": False},
    {"template_quarantine_streak": 1},
])
def test_existing_verification_and_gallery_wait_are_not_new_direction_barriers(changes):
    assert not provisional_reacquire_direction(cap589_assignment(**changes))


@pytest.mark.parametrize("assignment", [
    None, {}, {"reason": "mapped"},
    {"reason": "preferred_search_late_reacquire"},
    {"reason": "preferred_search_late_reacquire", "template_update_quarantined": True},
])
def test_legacy_missing_fields_do_not_add_new_identity_gate(assignment):
    assert not provisional_reacquire_direction(assignment)


@pytest.mark.parametrize("changes", [
    {"identity_continuation": {"status": "accept", "source": "partial",
                               "reference_cap": 587, "pair_cap": 342}},
    {"reacquire_control_recovered": True},
])
def test_current_verified_evidence_does_not_wait_for_gallery_release(changes):
    assert not provisional_reacquire_direction(cap589_assignment(**changes))


@pytest.mark.parametrize("changes", [
    {"identity_continuation": {"status": "hold"}},
    {"identity_continuation": {"status": "reject"}},
    {"identity_continuation": None},
    {"identity_continuation": "accept"},
    {"reacquire_control_recovered": False},
    {"reacquire_control_recovered": "true"},
    {"reacquire_control_recovered": 1},
])
def test_tentative_or_malformed_evidence_is_not_verification(changes):
    assert provisional_reacquire_direction(cap589_assignment(**changes))


@pytest.mark.parametrize("veto", [
    {"identity_control_rejected": True},
    {"search_excluded": True},
    {"reacquire_geometry_ok": False},
    {"identity_competition": {"passed": False}},
    {"identity_continuation": {"status": "reject"}},
])
def test_positive_flag_cannot_override_current_contradiction(veto):
    assignment = cap589_assignment(
        reacquire_control_recovered=True,
        identity_continuation={"status": "accept", "source": "partial"},
    )
    assignment.update(veto)
    assert provisional_reacquire_direction(assignment)


def test_real_observation_entry_keeps_cap589_style_candidate_out_of_history(owner, caplog):
    # The shared fixture already has a trusted CAP651. Rebase this CAP589
    # assignment to CAP654 so it is genuinely newer, not rejected as old data.
    t, c = owner, owner._follow_controller
    t.action_queue = Queue()
    queued = object()
    t.action_queue.put(queued)
    history = list(c._target_direction_history.entries)
    hint = dict(direction="right", loss_capture_frame_id=652, last_capture_frame_id=651)
    c._historical_direction_hint = hint
    r = sample(t, 654, .10)
    obs = t._rknn_pipeline.tracker.last_identity_observations[0]
    obs["assignment"] = cap589_assignment()
    with caplog.at_level("INFO"):
        consume(t, [r])

    assert "observation_reason=provisional_reacquire_observation" in caplog.text
    assert list(c._target_direction_history.entries) == history
    assert c._direction_latest_visible_capture_id == 651
    assert c._historical_direction_hint is hint
    assert (c.search_state, c.search_direction) == ("searching", "right")
    assert t._search_brake_latest_observation is None
    # The production consume entry is held; fixture sentinels also fail if it
    # tries normal control/depth processing or to queue a person action.
    assert t._current_forward_percent == 0
    assert t._longitudinal_valid_until == 99.8
    assert t.action_queue.get_nowait() is queued
    assert t.action_queue.empty()
    assert obs["uid"] == 1 and obs["assignment"]["template_update_quarantined"] is True


@pytest.mark.parametrize("verification", [
    {"identity_continuation": {"status": "accept", "source": "partial",
                               "reference_cap": 654, "pair_cap": 342}},
    {"reacquire_control_recovered": True},
])
def test_next_verified_frame_publishes_direction_without_waiting_for_gallery(owner, verification, caplog):
    t, c = owner, owner._follow_controller
    t.action_queue = Queue()
    first = sample(t, 654, .12)
    t._rknn_pipeline.tracker.last_identity_observations[0]["assignment"] = cap589_assignment()
    consume(t, [first])
    assert c._direction_latest_visible_capture_id == 651
    first_stamp = t._active_capture_timestamp

    second = sample(t, 655, .10)
    obs = t._rknn_pipeline.tracker.last_identity_observations[0]
    obs["assignment"] = cap589_assignment(**verification)
    with caplog.at_level("INFO"):
        consume(t, [second])

    assert t._active_capture_timestamp - first_stamp == pytest.approx(.054)
    assert obs["assignment"]["template_update_quarantined"] is True
    assert "observation_reason=trusted_observation_recorded" in caplog.text
    assert c._direction_latest_visible_capture_id == 655
    assert c._target_direction_history.latest_visible_evidence().capture_frame_id == 655
    assert c.last_person_center_x == pytest.approx(.10 * 640)
    assert t._search_brake_latest_observation == (1, 655)
    # Publishing a new direction observation neither changes the active yaw
    # action nor releases the hold/refreshes longitudinal authority.
    assert (c.search_state, c.search_direction) == ("searching", "right")
    assert t.pending is True
    assert t._current_forward_percent == 0
    assert t._longitudinal_valid_until == 99.8
    assert t.action_queue.empty()


@pytest.mark.parametrize("veto", [
    {"identity_control_rejected": True},
    {"identity_competition": {"passed": False}},
    {"reacquire_geometry_ok": False},
    {"identity_continuation": {"status": "reject"}},
])
def test_real_entry_does_not_publish_contradictory_recovered_frame(owner, veto):
    t, c = owner, owner._follow_controller
    history = list(c._target_direction_history.entries)
    r = sample(t, 654, .10)
    assignment = cap589_assignment(reacquire_control_recovered=True)
    assignment.update(veto)
    t._rknn_pipeline.tracker.last_identity_observations[0]["assignment"] = assignment
    consume(t, [r])
    assert list(c._target_direction_history.entries) == history
    assert c._direction_latest_visible_capture_id == 651
    assert t._search_brake_latest_observation is None
    assert t._longitudinal_valid_until == 99.8
