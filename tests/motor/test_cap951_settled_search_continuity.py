"""CAP951-991: settled search hold must not resample quiet at vision rate.

Production executor/producer methods, fake clock, cached feedback and driver.
No camera, serial device or motor threads are started.
"""
from dataclasses import replace

import pytest

from test_cap357_search_depth_resume import case, owner, refresh
from test_search_handoff_execution import quiet


def fresh_case(case, capture_timestamp=None):
    o, rt, driver, symbols, clock, frame, target, decision, commits = case
    stamp = clock[0] - .10 if capture_timestamp is None else capture_timestamp
    target = replace(target, depth_observation=replace(
        target.depth_observation, capture_timestamp=stamp))
    frame = replace(frame, capture_timestamp=stamp, persons=[target],
                    distance_state=replace(frame.distance_state,
                                           sample_timestamp=clock[0]-.08))
    return o, rt, driver, symbols, clock, frame, target, decision, commits


def test_slow_visual_decision_keeps_original_quiet_boundary_and_recovers(case):
    o, rt, driver, _, clock, *_ = case
    evidence = rt._search_reacquire_settling
    ready_at = evidence.ready_at
    applied = rt._search_reacquire_brake_applied
    writes = (list(driver.pairs), list(driver.stops))
    assert rt._search_reacquire_brake_request is None and o._brake_hold_active

    # The actual log had 149-178 ms gaps between positive vision decisions,
    # while the feedback/action loops continued at their ordinary fast rate.
    for now in (100.60, 100.65, 100.70, 100.75, 100.80, 100.85, 100.90):
        clock[0] = now
        rt._service_follow_wheels()
        assert o._brake_hold_active
        assert rt._search_reacquire_brake_applied is applied
        assert evidence.ready_at == ready_at
        assert evidence.last_sample == now
        assert o._depth30_linear_snapshot is None
    assert (driver.pairs, driver.stops) == writes
    assert refresh(fresh_case(case))
    assert not o._brake_hold_active and len(case[-1]) == 1
    assert (driver.pairs, driver.stops) == writes  # release/admission is no I/O


@pytest.mark.parametrize("fault", ["moving", "stale", "untrusted", "future", "out_of_order"])
def test_bad_feedback_erases_quiet_and_needs_new_post_quiet_image(case, fault):
    o, rt, _, _, clock, *_ = case
    evidence = rt._search_reacquire_settling
    clock[0] = 100.60
    rt._service_follow_wheels()
    clock[0] = 100.65
    bad = {
        "moving": quiet(100.65, left_forward_rpm=3),
        "stale": quiet(100.40),
        "untrusted": replace(quiet(100.65), trustworthy=False),
        "future": quiet(100.66),
        "out_of_order": quiet(100.59),
    }[fault]
    rt.get_steering_feedback = lambda: bad
    rt._service_follow_wheels()
    assert evidence.ready_at is None
    assert o._brake_hold_active
    rt.get_steering_feedback = lambda: quiet(clock[0])
    clock[0] = 100.70
    rt._service_follow_wheels()
    assert evidence.ready_at is None
    clock[0] = 100.75
    rt._service_follow_wheels()
    assert evidence.ready_at == pytest.approx(100.75)
    assert not refresh(fresh_case(case, capture_timestamp=100.72))
    assert evidence.reason == "image_before_quiet_confirmation"
    clock[0] = 100.80
    rt._service_follow_wheels()
    assert refresh(fresh_case(case, capture_timestamp=100.78))


def test_real_feedback_gap_still_requires_two_new_quiet_samples(case):
    o, rt, _, _, clock, *_ = case
    evidence = rt._search_reacquire_settling
    clock[0] = 100.90  # no intervening action/encoder observations
    rt._service_follow_wheels()
    assert evidence.ready_at is None and evidence.quiet_count == 1
    assert not refresh(fresh_case(case))
    # Reusing one cache entry never completes confirmation.
    rt._service_follow_wheels()
    assert evidence.quiet_count == 1
    clock[0] = 100.95
    rt._service_follow_wheels()
    assert evidence.ready_at == pytest.approx(100.95)
    clock[0] = 101.
    rt._service_follow_wheels()
    assert refresh(fresh_case(case, capture_timestamp=100.98))


@pytest.mark.parametrize("change", ["uid", "episode", "evidence", "pending", "current_not_released",
    "hold_released", "hold_label", "hold_mode", "near_park", "explicit", "shutdown", "not_running"])
def test_observer_cannot_follow_a_replaced_hold(case, change):
    o, rt, _, _, clock, *_ = case
    evidence = rt._search_reacquire_settling
    old_last = evidence.last_sample
    if change == "uid": o._follow_controller.active_target_id = 2
    elif change == "episode": rt._search_reacquire_brake_applied = object()
    elif change == "evidence": rt._search_reacquire_settling = None
    elif change == "pending": rt._search_reacquire_brake_request = evidence.request
    elif change == "current_not_released": evidence.current_released_at = None
    elif change == "hold_released": o._brake_hold_active = False
    elif change == "hold_label": o._brake_hold_label = "safety_hold_front"
    elif change == "hold_mode": o._brake_hold_stop_mode = "free"
    elif change == "near_park": o._near_yaw_park_request = object()
    elif change == "explicit": o._explicit_stop_requested = True
    elif change == "shutdown": o._runtime_shutdown_requested = True
    elif change == "not_running": o.running = False
    clock[0] = 100.60
    rt._observe_settled_search_brake()
    assert evidence.last_sample == old_last


def test_settled_observation_never_releases_by_itself(case):
    o, rt, driver, _, clock, *_ = case
    writes = (list(driver.pairs), list(driver.stops))
    for step in range(1, 41):
        clock[0] = 100.55 + .05*step
        rt._service_follow_wheels()
    assert o._brake_hold_active
    assert o._depth30_linear_snapshot is None
    assert (driver.pairs, driver.stops) == writes
    assert not case[-1]
