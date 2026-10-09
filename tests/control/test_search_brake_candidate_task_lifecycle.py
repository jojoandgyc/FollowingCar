"""CAP242/247/260: candidate evidence must not orphan an interrupted scan.

Production candidate publisher, controller, brake consumer and fake motor
runtime are connected here. No camera, serial port or hardware thread starts.
"""
import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import request_0513_modular as application

from car_control_modular.control_types import PersonTarget, SensorFrame, SteeringFeedback
from car_control_modular.target_direction_history import TargetDirectionHistory
from test_failed_search_brake_handoff import (
    NormalControlEntry, next_missing_decision, observation, release_with_capture,
    request_and_stop, settle, setup_search,
)


def candidate(tracker, clock, x, cap=590, uid=1):
    return tracker._publish_search_candidate_task_evidence(
        (640*x-50, 2, 640*x+50, 475), frame_width=640,
        confirmed=True, source="formal", candidate_score=.92,
        candidate_tracked=True, candidate_confirmed_uid=uid,
        capture_frame_id=cap, now=clock[0])


def budget(ctl):
    return (ctl._search_resume_token, ctl._search_rotation_started_at,
            ctl._lost_started_at, ctl._search_rotation_origin_integrated_yaw_deg,
            ctl._search_heading_min_deg, ctl._search_heading_max_deg,
            ctl._search_rotation_accumulated_deg)


def pipeline_candidate_call(tracker, clock, bbox, cap):
    """Execute the actual process_external_frame publication block, no RKNN."""
    tree = ast.parse(Path(application.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PersonTracker")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "process_external_frame")
    i = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "note_candidate" for t in n.targets))
    ns = dict(vars(application))
    ns.update(self=tracker, gate_decision=SimpleNamespace(bbox=bbox, source="formal", score=.92),
              stale_result_discarded=False, search_settling_pending=False,
              candidate_min_score=.2, width=640, current_capture_id=cap,
              gate_observation_confirmed=True, exclusion_bindings=(),
              candidate_has_active_reid_evidence=lambda _: True,
              candidate_matches_active_target=lambda _: True,
              candidate_confirmed_identity_uid=lambda _: 1)
    exec(compile(ast.Module(body=method.body[i:i+2], type_ignores=[]),
                 application.__file__, "exec"), ns)


def test_cap242_247_258_real_publisher_and_verified_partial_observation(monkeypatch):
    tracker, rt, driver, _, clock, _ = setup_search(monkeypatch, "left")
    ctl = tracker._follow_controller
    ctl.cfg = replace(ctl.cfg, center_left_ratio=.47, center_right_ratio=.53)
    ctl._target_direction_history = TargetDirectionHistory()
    ctl._direction_latest_visible_capture_id = 0
    ctl._record_target_direction_evidence(SensorFrame(width=640, height=480,
        capture_frame_id=188, capture_timestamp=99.7),
        PersonTarget((0., 0., 180., 479.), 1, .94, 180.*479.), reliable=True)

    def verified(cap, x):
        rec = observation(tracker, clock, cap, 1, x)
        item = tracker._rknn_pipeline.tracker.last_identity_observations[0]
        item["assignment"].update(uid=1, reason="mapped_verified_continuation",
            template_update_quarantined=True, template_quarantine_reason="not_high_quality_strong",
            template_quarantine_streak=0, identity_continuation=dict(
                status="accept", reason="verified_pair", source="partial",
                reference_cap=cap-2, pair_cap=41, deadline=clock[0]+.25))
        return rec

    rec = verified(242, .4127)
    assert rt.request_search_reacquire_brake(242, tracker._active_capture_timestamp, "candidate_center")
    tracker._consume_track_records([rec], 640, 480, "test")
    rt._service_follow_wheels()
    original = budget(ctl)
    clock[0] = 100.2
    bbox = (258.6977844238281, 2.067901611328125, 447.45733642578125, 476.08294677734375)
    pipeline_candidate_call(tracker, clock, bbox, 247)
    assert ctl.search_direction == "left" and budget(ctl) == original
    rec = verified(247, .5516836881637573)
    tracker._consume_track_records([rec], 640, 480, "test")
    assert tracker._search_brake_latest_observation == (1, 247)
    assert ctl.search_brake_resume_context_current(rt._search_reacquire_resume_context)
    for i in range(5, 20):
        clock[0] = 100. + i*.05
        rt._service_follow_wheels()
    clock[0] = 100.99
    rec = verified(258, .7517027616500854)
    with pytest.raises(NormalControlEntry):
        tracker._consume_track_records([rec], 640, 480, "test")
    assert ctl.search_state == "none" and ctl.search_direction is None
    assert ctl._target_direction_history.latest_visible_evidence().capture_frame_id == 258
    assert ctl._target_direction_history.latest_reliable_side().direction == "right"
    assert not rt.search_reacquire_brake_pending()
    assert all(pair == (0, 0) for pair in driver.pairs)


@pytest.mark.parametrize("x", [.2, .5, .8], ids=["same_side", "center", "opposite_side"])
def test_hold_keeps_task_but_records_trusted_current_location(monkeypatch, x):
    tracker, rt, driver, _, clock, _ = setup_search(monkeypatch, "left")
    request_and_stop(tracker, rt, clock, candidate_x=.3)
    ctl = tracker._follow_controller
    original = budget(ctl)
    clock[0] += .20
    assert not candidate(tracker, clock, x)
    assert budget(ctl) == original
    assert ctl.search_direction == "left"
    rec = observation(tracker, clock, 590, 1, x)
    tracker._consume_track_records([rec], 640, 480, "test")
    assert ctl._target_direction_history.latest_visible_evidence().capture_frame_id == 590
    assert tracker._search_brake_latest_observation == (1, 590)
    assert budget(ctl) == original and ctl.search_direction == "left"
    assert all(pair == (0, 0) for pair in driver.pairs)
    settle(rt, clock)
    release_with_capture(tracker, rt, clock, uid=1, x=.8)
    assert ctl.search_state == "none" and ctl.search_direction is None
    assert ctl._target_direction_history.latest_reliable_side().direction == "right"
    assert ctl._target_direction_history.latest_visible_evidence().capture_frame_id == 606


def test_candidate_missing_during_hold_cannot_mutate_centering_task(monkeypatch):
    tracker, rt, _, _, clock, _ = setup_search(monkeypatch, "left")
    request_and_stop(tracker, rt, clock, candidate_x=.3)
    ctl = tracker._follow_controller
    ctl._stale_direction_recovery_active = True
    ctl._stale_direction_recovery_stage = "candidate_centering"
    ctl._stale_candidate_side = "left"
    ctl._candidate_center_missing_frames = 0
    before = budget(ctl)
    for _ in range(10):
        assert not tracker._publish_search_candidate_task_evidence()
    assert ctl._candidate_center_missing_frames == 0
    assert ctl.search_direction == "left" and budget(ctl) == before


@pytest.mark.parametrize("x,direction", [(.2, "left"), (.5, "left"), (.8, "right")])
def test_normal_candidate_updates_keep_scan_origin_time_and_coverage(monkeypatch, x, direction):
    tracker, _, _, _, clock, _ = setup_search(monkeypatch, "left")
    ctl = tracker._follow_controller
    before = budget(ctl)
    candidate(tracker, clock, x)
    assert ctl.search_state == "searching" and ctl.search_direction == direction
    assert budget(ctl) == before
    assert ctl.search_status(clock[0]).progress_deg == pytest.approx(144.)
    assert ctl.capture_search_brake_resume() is not None


@pytest.mark.parametrize("uid", [0, 2])
def test_unconfirmed_or_other_uid_cannot_redirect_finite_task(monkeypatch, uid):
    tracker, _, _, _, clock, _ = setup_search(monkeypatch, "left")
    before = budget(tracker._follow_controller)
    candidate(tracker, clock, .8, uid=uid)
    assert tracker._follow_controller.search_direction == "left"
    assert budget(tracker._follow_controller) == before


def test_two_holds_keep_new_trusted_right_search_after_second_candidate_lost(monkeypatch):
    tracker, rt, driver, _, clock, heading = setup_search(monkeypatch, "left")
    request_and_stop(tracker, rt, clock, candidate_x=.3)
    clock[0] += .2
    candidate(tracker, clock, .8)
    settle(rt, clock)
    release_with_capture(tracker, rt, clock, uid=1, x=.8)
    ctl = tracker._follow_controller
    # Release uses current right-side evidence, not the old left command.
    for cap in (611, 613, 615):
        decision = next_missing_decision(tracker, clock, heading, cap)
    assert ctl.search_direction == "right"
    assert [a.kind for a in decision.actions] == ["rotate_right"]
    before = budget(ctl)
    rec = observation(tracker, clock, 620, 1, .9)
    assert rt.request_search_reacquire_brake(620, tracker._active_capture_timestamp,
                                           "candidate_predictive_stop")
    assert rt._search_reacquire_resume_context is not None
    tracker._consume_track_records([rec], 640, 480, "test")
    rt._service_follow_wheels()
    started = clock[0]
    for i in range(1, 20):
        clock[0] = started + .05*i
        rt._service_follow_wheels()
    clock[0] = started + .99
    rec = observation(tracker, clock, 640, 0, .9)
    with pytest.raises(NormalControlEntry):
        tracker._consume_track_records([rec], 640, 480, "test")
    assert ctl.search_state == "searching" and ctl.search_direction == "right"
    assert budget(ctl) == before
    assert ctl._target_direction_history.latest_visible_evidence() is None
    decision = next_missing_decision(tracker, clock, heading, 645)
    assert [a.kind for a in decision.actions] == ["rotate_right"]
    assert all(pair == (0, 0) for pair in driver.pairs)


def prepare_depth_wait_publisher(tracker):
    queued, stopped = [], []
    tracker._clear_lateral_intent = lambda _why: None
    tracker._clear_longitudinal_context = lambda **_kw: None
    tracker._replace_action_queue = lambda actions, reason: queued.append((actions, reason))
    tracker._publish_observation_soft_zero = stopped.append
    return queued, stopped


def test_direct_depth_wait_publisher_starts_one_finite_scan_without_decide(monkeypatch):
    tracker, rt, _, symbols, clock, _ = setup_search(monkeypatch, "left")
    ctl = tracker._follow_controller
    ctl._reset_search_timeout()
    old_loss = ctl._lost_started_at
    queued, stopped = prepare_depth_wait_publisher(tracker)
    assert tracker._publish_search_reacquire_direction_hold("confirmed_search_reacquire_depth_wait")
    started = ctl._search_rotation_started_at
    context = ctl.capture_search_brake_resume()
    assert started == clock[0] and context is not None
    assert ctl._lost_started_at == old_loss
    assert queued[-1][0] == [symbols.rotate_left] and not stopped
    original = budget(ctl)
    clock[0] += .2
    assert tracker._publish_search_reacquire_direction_hold("confirmed_search_reacquire_depth_wait")
    assert budget(ctl) == original
    assert ctl.capture_search_brake_resume().rotation_started_at == started


def test_depth_wait_publisher_reads_feedback_before_sampling_budget_clock(monkeypatch):
    tracker, rt, _, _, clock, _ = setup_search(monkeypatch, "left")
    ctl = tracker._follow_controller
    ctl.cfg = replace(ctl.cfg, search_timeout_sec=0.)
    ctl._reset_search_timeout()

    def newly_published_feedback():
        clock[0] += .003
        return SteeringFeedback(timestamp=clock[0], trustworthy=True,
                                integrated_yaw_right_deg=-144.)

    rt.get_steering_feedback = newly_published_feedback
    queued, stopped = prepare_depth_wait_publisher(tracker)
    assert tracker._publish_search_reacquire_direction_hold("confirmed_search_reacquire_depth_wait")
    assert queued and not stopped
    assert ctl._search_rotation_origin_integrated_yaw_deg == -144.


@pytest.mark.parametrize("expired", ["timeout", "coverage"])
def test_direct_depth_wait_publisher_checks_existing_budget_before_queueing(monkeypatch, expired):
    tracker, rt, _, _, clock, _ = setup_search(monkeypatch, "left")
    ctl = tracker._follow_controller
    candidate(tracker, clock, .8)
    before_start = ctl._search_rotation_started_at
    if expired == "timeout":
        ctl.cfg = replace(ctl.cfg, search_timeout_sec=1.)
        rt.get_steering_feedback = lambda: None
        ctl._search_rotation_feedback_last_ts = clock[0] - 1.
    else:
        for angle in (-216., -288., -360.):
            ctl._update_search_rotation_progress(SensorFrame(width=640, height=480,
                steering_feedback=SteeringFeedback(timestamp=clock[0], trustworthy=True,
                    integrated_yaw_right_deg=angle)))
    queued, stopped = prepare_depth_wait_publisher(tracker)
    assert not tracker._publish_search_reacquire_direction_hold("confirmed_search_reacquire_depth_wait")
    assert not queued and stopped
    assert ctl.search_state == "timed_out"
    assert ctl._search_rotation_started_at == before_start


def test_unmeasured_no_timeout_search_never_gains_unbounded_direct_yaw(monkeypatch):
    tracker, rt, _, _, clock, _ = setup_search(monkeypatch, "left")
    ctl = tracker._follow_controller
    ctl.cfg = replace(ctl.cfg, search_timeout_sec=0.)
    ctl._reset_search_timeout()
    rt.get_steering_feedback = lambda: None
    queued, stopped = prepare_depth_wait_publisher(tracker)
    assert not tracker._publish_search_reacquire_direction_hold("confirmed_search_reacquire_depth_wait")
    assert not queued and stopped
    first = ctl._search_rotation_started_at
    clock[0] += .2
    rt.get_steering_feedback = lambda: SteeringFeedback(
        timestamp=clock[0], trustworthy=True, integrated_yaw_right_deg=10.)
    assert tracker._publish_search_reacquire_direction_hold("confirmed_search_reacquire_depth_wait")
    assert ctl._search_rotation_started_at == first
    assert ctl._search_rotation_origin_integrated_yaw_deg == 10.
    assert ctl.capture_search_brake_resume() is not None
