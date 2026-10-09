"""Startup is a stopped, single-candidate transaction, not lost-target search."""
import pytest

from car_control_modular.control_types import (
    HazardState, LateralCandidateEvidence, ObstacleState, PersonTarget, SensorFrame,
)
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController


@pytest.fixture
def clock(monkeypatch):
    value = [100.0]
    monkeypatch.setattr("car_control_modular.controllers.time.monotonic", lambda: value[0])
    return value


def person(uid=1, *, proof=False, right=False, area=40000):
    return PersonTarget(
        (500, 40, 630, 450) if right else (250, 80, 390, 430),
        track_id=uid, confidence=.94, area=area, initial_identity_confirmed=proof,
    )


def frame(*persons, cap=10, stamp=99.95, **kwargs):
    return SensorFrame(width=640, height=480, persons=list(persons),
                       capture_frame_id=cap, capture_timestamp=stamp,
                       distance_m=2.0, **kwargs)


def controller(**kwargs):
    return FollowSafetyController(FollowPolicyConfig(
        initial_target_confirm_frames=2, release_target_on_lost=False, **kwargs))


def assert_wait(c, decision):
    assert c.active_target_id is None and not c._has_seen_person
    assert c.search_state == "none" and c.search_direction is None
    assert decision.clear_action_queue and decision.stop_action_execution
    assert decision.explicit_stop_requested or decision.soft_stop_requested
    assert all(action.kind == "stop" for action in decision.actions)
    assert not decision.is_forwarding


def test_empty_startup_never_scans_even_with_legacy_search_switch(clock):
    c = controller(search_before_first_seen=True, startup_search_delay_sec=0)
    c.search_state, c.search_direction = "searching", "right"
    d = c.decide(1, frame(lateral_candidate=LateralCandidateEvidence(
        10, (500, 10, 630, 470), .95)))
    assert_wait(c, d)
    assert d.reason == "wait_first_person"


def test_edge_candidate_stays_stopped_until_new_capture_confirmed(clock):
    c = controller(visible_steering_pid_enable=True)
    target = person(right=True)
    assert_wait(c, c.decide(1, frame(target)))
    assert c._initial_candidate_id == 1
    assert_wait(c, c.decide(2, frame(target)))
    c.decide(3, frame(target, cap=11, stamp=99.99))
    assert c.active_target_id == 1 and c._has_seen_person


def test_missing_or_other_larger_person_does_not_replace_first_candidate(clock):
    c = controller()
    assert_wait(c, c.decide(1, frame(person())))
    assert_wait(c, c.decide(2, frame(cap=11, stamp=99.97)))
    assert_wait(c, c.decide(3, frame(person(2, proof=True, area=100000), cap=12, stamp=99.98)))
    assert c._initial_candidate_id == 1
    c.decide(4, frame(person(), person(2, area=100000), cap=13, stamp=99.99))
    assert c.active_target_id == 1


def test_ambiguous_first_frame_waits_for_unique_candidate(clock):
    c = controller()
    assert_wait(c, c.decide(1, frame(person(), person(2, area=100000))))
    assert c._initial_candidate_id is None
    assert_wait(c, c.decide(2, frame(person(2), cap=11, stamp=99.99)))
    assert c._initial_candidate_id == 2


def test_enrollment_proof_skips_duplicate_confirmation_even_with_later_bystander(clock):
    c = controller()
    c.decide(1, frame(person(proof=True), person(2, area=100000)))
    assert c.active_target_id == 1 and c._has_seen_person


@pytest.mark.parametrize("cap,stamp", [(10, 99.0), (10, 100.01), (0, 99.95),
                                      (10, 0), (10, float("nan")), (10, float("inf"))])
def test_stale_future_or_malformed_proof_never_locks(clock, cap, stamp):
    c = controller()
    assert_wait(c, c.decide(1, frame(person(proof=True), cap=cap, stamp=stamp)))
    assert c._initial_candidate_id is None


def test_proof_without_capture_provenance_never_locks(clock):
    c = controller()
    for n in range(5):
        assert_wait(c, c.decide(n, frame(person(proof=True), cap=0, stamp=0)))


@pytest.mark.parametrize("cap,stamp", [(10, 99.99), (9, 99.99), (11, 99.95), (11, 99.94)])
def test_replayed_or_out_of_order_observation_does_not_confirm(clock, cap, stamp):
    c = controller()
    assert_wait(c, c.decide(1, frame(person())))
    assert_wait(c, c.decide(2, frame(person(), cap=cap, stamp=stamp)))
    assert c._initial_candidate_frames == 1


def test_legacy_no_capture_requires_distinct_control_frames(clock):
    c = controller()
    f = frame(person(), cap=0, stamp=0)
    assert_wait(c, c.decide(1, f))
    assert_wait(c, c.decide(1, f))
    c.decide(2, f)
    assert c.active_target_id == 1


def test_long_gap_restarts_confirmation_but_keeps_reserved_person(clock):
    c = controller()
    assert_wait(c, c.decide(1, frame(person())))
    clock[0] = 101
    assert_wait(c, c.decide(2, frame(person(2, proof=True), cap=12, stamp=100.95)))
    assert_wait(c, c.decide(3, frame(person(), cap=13, stamp=100.96)))
    assert c._initial_candidate_frames == 1 and c._initial_candidate_id == 1
    c.decide(4, frame(person(), cap=14, stamp=100.98))
    assert c.active_target_id == 1


@pytest.mark.parametrize("kwargs", [dict(low_quality_visible=True), dict(longitudinal_only=True)])
def test_unqualified_or_non_visual_path_cannot_create_startup_lock(clock, kwargs):
    c = controller()
    assert_wait(c, c.decide(1, frame(person(proof=True, right=True)), **kwargs))
    assert c._initial_candidate_id is None


def test_unsteerable_without_enrollment_proof_cannot_lock(clock):
    c = controller()
    assert_wait(c, c.decide(1, frame(person()), target_steerable=False))
    assert c._initial_candidate_id is None


def test_proven_identity_can_lock_without_authorizing_unsuitable_geometry(clock):
    c = controller()
    d = c.decide(1, frame(person(proof=True, right=True)), target_steerable=False)
    assert c.active_target_id == 1 and c._has_seen_person
    assert d.explicit_stop_requested and not d.actions
    assert c.last_steering_pid_result is None
    assert d.reason == "target_visible_unsteerable_hold"


def test_rotation_only_trial_still_confirms_identity_before_motion(clock):
    c = controller()
    assert_wait(c, c.decide(1, frame(person(right=True)), rotation_only=True))
    c.decide(2, frame(person(right=True), cap=11, stamp=99.99), rotation_only=True)
    assert c.active_target_id == 1 and c._has_seen_person


@pytest.mark.parametrize("uid", [0, -2])
def test_unknown_and_geometry_only_targets_never_lock(clock, uid):
    c = controller()
    assert_wait(c, c.decide(1, frame(person(uid, proof=True))))


@pytest.mark.parametrize("kwargs,reason", [(dict(hazard=HazardState(True, "hazard")), "hazard"),
                                         (dict(obstacles=ObstacleState(front=True)), "front_ir")])
def test_emergency_gate_precedes_startup_lock(clock, kwargs, reason):
    c = controller()
    d = c.decide(1, frame(person(proof=True), **kwargs))
    assert_wait(c, d)
    assert d.reason == reason and d.explicit_stop_requested


def test_explicit_reset_releases_candidate_reservation(clock):
    c = controller()
    c.decide(1, frame(person()))
    c.clear_active_target("operator_reset")
    c.decide(2, frame(person(2, proof=True), cap=11, stamp=99.99))
    assert c.active_target_id == 2


def test_after_lock_loss_keeps_identity_instead_of_selecting_bystander(clock):
    c = controller(lost_confirm_frames=1, search_timeout_sec=30)
    c.decide(1, frame(person(proof=True, right=True)))
    d = c.decide(2, frame(person(2, proof=True), cap=11, stamp=99.99))
    assert c.active_target_id == 1 and c._has_seen_person
    assert c.last_selected_target is None
    assert not d.is_forwarding


def test_locked_identity_survives_temporary_unsteerable_observation(clock):
    c = controller()
    c.decide(1, frame(person(proof=True)))
    d = c.decide(2, frame(person(), cap=11, stamp=99.99), target_steerable=False)
    assert c.active_target_id == 1 and c._has_seen_person
    assert d.explicit_stop_requested and d.reason == "target_visible_unsteerable_hold"
