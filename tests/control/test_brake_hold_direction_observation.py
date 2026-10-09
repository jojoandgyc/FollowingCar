"""CAP651 right -> held CAP654..666 left -> renewed search must be left.

No cameras, serial ports or motor threads; exercise the production consume
entry, identity metadata adapter and real controller, not permissive mocks.
"""
from types import SimpleNamespace

import pytest
import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction, PersonTarget, SensorFrame
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController


@pytest.fixture
def owner(monkeypatch):
    t = object.__new__(runtime.PersonTracker)
    t.clock = 100.
    monkeypatch.setattr(runtime.time, "monotonic", lambda: t.clock)
    t._follow_controller = FollowSafetyController(FollowPolicyConfig(
        direction_history_enable=True, lost_confirm_frames=3,
        lost_confirm_sec=0., initial_target_confirm_frames=1, search_timeout_sec=60.))
    c = t._follow_controller
    c.active_target_id = 1
    c._has_seen_person = True
    c.search_state, c.search_direction = "searching", "right"
    c._lost_exit_direction = "right"
    c._lost_hint_source = "search_candidate_last_right"
    c._last_visible_steer_action = ControlAction.rotate_right("old")
    c.last_steering_pid_result = object()
    t.search_state, t.search_direction = "searching", "right"
    t._rknn_pipeline = SimpleNamespace(tracker=SimpleNamespace(last_identity_observations=[]))
    t.pending = True
    t._action_runtime = SimpleNamespace(
        search_reacquire_brake_pending=lambda: t.pending,
        _search_reacquire_brake_request=SimpleNamespace(capture_frame_id=654))
    t._current_forward_percent = 0
    t._longitudinal_valid_until = 99.8
    def forbidden(*args, **kwargs):
        raise AssertionError("observation path must not update control or depth")
    t._clear_longitudinal_context = forbidden
    t._queue_actions_for_persons = forbidden
    t._clear_lateral_intent = forbidden
    target = PersonTarget((460, 40, 610, 450), 1, .95, 61500)
    c._record_target_direction_evidence(SensorFrame(width=640, height=480,
        capture_frame_id=651, capture_timestamp=99.7), target, reliable=True)
    return t


def sample(t, cap, x, uid=1):
    t.clock = 100. + (cap-654)*.054
    t._active_capture_frame_id = cap
    t._active_capture_timestamp = t.clock-.10
    bbox = (max(0., x*640-60), 30., min(640., x*640+60), 450.)
    r = SimpleNamespace(track_id=4, reid_uid=uid, class_id=0, time_since_update=0,
                        score=.95, x1=300., y1=0., x2=600., y2=479.)
    t._rknn_pipeline.tracker.last_identity_observations = [dict(
        raw_track_id=4, uid=uid, detector_bbox=bbox,
        sample_metadata=dict(capture_frame_id=cap, capture_timestamp=t._active_capture_timestamp,
                             is_fresh=True, integrated_yaw_deg=20.),
        assignment=dict(bbox_quality_ok=True, reacquire_geometry_ok=True,
                        distance=.2199, reason="skip_update_reacquire_quarantine"))]
    return r


def consume(t, records):
    t._consume_track_records(records, 640, 480, "test")


def test_cap_replay_updates_history_without_releasing_stop(owner, caplog):
    t, c = owner, owner._follow_controller
    with caplog.at_level("INFO"):
        for cap, x in [(654,.684), (656,.552), (659,.290), (662,.172), (666,.099)]:
            consume(t, [sample(t, cap, x)])
            assert c._direction_latest_visible_capture_id == cap
            assert c.search_direction == "right"  # no premature motor reversal
            assert t._current_forward_percent == 0
            assert t._longitudinal_valid_until == 99.8
        assert c.last_steering_pid_result is None
        assert c._last_visible_steer_action is None
        assert c.last_person_center_x == pytest.approx(.099*640)
        assert "latest_observation_capture_frame_id=666 action_evidence_capture_frame_id=654" in caplog.text
    t.pending = False
    t._active_capture_frame_id = 671
    t._finish_search_brake_observation_hold()
    assert c.search_direction is None and c._lost_exit_direction is None
    assert t._longitudinal_valid_until == 99.8
    for cap in (671,673,675):
        t.clock += .10
        frame = SensorFrame(width=640, height=480, capture_frame_id=cap,
                            capture_timestamp=t.clock)
        decision = c.decide(cap, frame)
        assert not any(a.kind in ("rotate_right", "steer_right", "forward") for a in decision.actions)
    assert c.search_direction == "left"
    assert any(a.kind == "rotate_left" for a in decision.actions)


@pytest.mark.parametrize("case", ["uid0", "other_uid", "predicted", "non_person",
    "low_score", "duplicate_uid", "stale", "future", "wrong_cap", "wrong_stamp",
    "quality", "geometry", "competition", "excluded", "not_fresh", "no_provenance",
    "invalid_bbox", "no_detector_bbox"])
def test_untrusted_frames_cannot_change_history(owner, case):
    t = owner
    r = sample(t, 654, .1)
    obs = t._rknn_pipeline.tracker.last_identity_observations[0]
    records = [r]
    if case == "uid0": r.reid_uid = 0
    elif case == "other_uid": r.reid_uid = 2
    elif case == "predicted": r.time_since_update = 1
    elif case == "non_person": r.class_id = 58
    elif case == "low_score": r.score = 0
    elif case == "duplicate_uid": records.append(r)
    elif case == "stale": t.clock += 1.
    elif case == "future": t.clock -= 1.
    elif case == "wrong_cap": obs["sample_metadata"]["capture_frame_id"] = 651
    elif case == "wrong_stamp": obs["sample_metadata"]["capture_timestamp"] -= .1
    elif case == "quality": obs["assignment"]["bbox_quality_ok"] = False
    elif case == "geometry": obs["assignment"]["reacquire_geometry_ok"] = False
    elif case == "competition": obs["sample_metadata"]["identity_competition"] = dict(passed=False)
    elif case == "excluded": obs["assignment"]["search_excluded"] = True
    elif case == "not_fresh": obs["sample_metadata"]["is_fresh"] = False
    elif case == "no_provenance": t._rknn_pipeline.tracker.last_identity_observations = []
    elif case == "invalid_bbox": obs["detector_bbox"] = (0,0,float("nan"),400)
    elif case == "no_detector_bbox": obs["detector_bbox"] = None
    consume(t, records)
    assert t._follow_controller._direction_latest_visible_capture_id == 651
    t._finish_search_brake_observation_hold()
    assert t._follow_controller.search_direction == "right"


def test_uid0_gap_does_not_discard_last_trusted_observation(owner):
    consume(owner, [sample(owner, 662, .172)])
    consume(owner, [sample(owner, 664, .134, uid=0)])
    assert owner._follow_controller._direction_latest_visible_capture_id == 662
    consume(owner, [sample(owner, 666, .099)])
    assert owner._follow_controller._direction_latest_visible_capture_id == 666


def test_old_or_duplicate_capture_does_not_rewrite_history(owner):
    consume(owner, [sample(owner, 666, .1)])
    before = list(owner._follow_controller._target_direction_history.entries)
    for cap in (666,662):
        consume(owner, [sample(owner, cap, .9)])
    assert list(owner._follow_controller._target_direction_history.entries) == before


def test_identity_changed_during_hold_cannot_release_new_search(owner):
    consume(owner, [sample(owner, 666, .1)])
    owner._follow_controller.active_target_id = 2
    owner._finish_search_brake_observation_hold()
    assert owner._follow_controller.search_direction == "right"


def test_other_person_does_not_block_unique_assigned_uid(owner):
    r = sample(owner, 666, .1)
    other = SimpleNamespace(**vars(r))
    other.reid_uid, other.track_id = 0, 5
    consume(owner, [r, other])
    assert owner._follow_controller._direction_latest_visible_capture_id == 666


def test_release_is_one_shot_and_does_not_clear_next_decision(owner):
    consume(owner, [sample(owner, 666, .1)])
    owner._finish_search_brake_observation_hold()
    owner._follow_controller.search_direction = "left"
    owner._finish_search_brake_observation_hold()
    assert owner._follow_controller.search_direction == "left"


def test_timestamp_regression_cannot_replace_newer_observation(owner):
    consume(owner, [sample(owner, 666, .1)])
    r = sample(owner, 668, .9)
    owner._active_capture_timestamp -= .15
    owner.clock -= .15
    owner._rknn_pipeline.tracker.last_identity_observations[0]["sample_metadata"]["capture_timestamp"] = owner._active_capture_timestamp
    consume(owner, [r])
    assert owner._follow_controller._direction_latest_visible_capture_id == 666


def test_consume_release_resets_direction_before_normal_control(owner):
    consume(owner, [sample(owner, 666, .1)])
    owner.pending = False
    def normal_entry(**kwargs):
        assert owner._follow_controller.search_direction is None
        assert owner._follow_controller._lost_exit_direction is None
        raise RuntimeError("normal control reached after direction reset")
    owner._clear_longitudinal_context = normal_entry
    with pytest.raises(RuntimeError, match="normal control reached"):
        consume(owner, [])


def test_live_exclusion_blocks_observation_even_when_assignment_snapshot_does_not(owner):
    r = sample(owner, 666, .1)
    owner._rknn_pipeline.tracker.identity_bank = SimpleNamespace(
        search_exclusion_for=lambda *args, **kwargs: dict(reason="co_visible_distinct_person"))
    consume(owner, [r])
    assert owner._follow_controller._direction_latest_visible_capture_id == 651


def test_real_logged_detector_boxes_choose_left_without_hardware(owner):
    # Original detector boxes from run_20260921_134517, not expanded track boxes.
    rows = [
        (654,5321.810702398,(322.776672,32.164169,552.516785,473.235962)),
        (656,5321.910727184,(239.185120,49.750565,467.699066,472.143921)),
        (659,5322.106402452,(70.051788,21.190414,301.125580,473.865662)),
        (662,5322.253254858,(.873077,10.312057,219.769623,477.126099)),
        (666,5322.450456670,(.756977,80.008743,125.929436,472.483704)),
    ]
    for cap, stamp, bbox in rows:
        r = sample(owner, cap, .5)
        owner.clock, owner._active_capture_timestamp = stamp+.1, stamp
        obs = owner._rknn_pipeline.tracker.last_identity_observations[0]
        obs["detector_bbox"] = bbox
        obs["sample_metadata"]["capture_timestamp"] = stamp
        consume(owner, [r])
    owner._finish_search_brake_observation_hold()
    c = owner._follow_controller
    for cap in (668,670,671):
        c._record_target_direction_evidence(SensorFrame(width=640,
            capture_frame_id=cap, capture_timestamp=5322.6+(cap-668)*.054), None, reliable=True)
    frame = SensorFrame(width=640, capture_frame_id=673, capture_timestamp=5322.9)
    c._capture_lost_exit_direction(frame, search_entry=True)
    assert c._lost_exit_direction == "left"
    assert c._target_direction_history.latest_reliable_side().last_visible_capture_frame_id == 666


@pytest.mark.parametrize("age", [.2132, .2239, .30, .50])
def test_delayed_verified_geometry_is_history_only(owner, monkeypatch, age, caplog):
    monkeypatch.setattr(runtime, "VISION_CONTROL_MAX_RESULT_AGE_SEC", .210)
    r = sample(owner, 666, .83)
    owner.clock = owner._active_capture_timestamp + age
    with caplog.at_level("INFO"):
        consume(owner, [r])
    c = owner._follow_controller
    assert c._direction_latest_visible_capture_id == 666
    assert c._target_direction_history.latest_visible_evidence().timestamp == owner._active_capture_timestamp
    assert c.search_direction == "right"  # actuator state not changed by observation
    assert owner._current_forward_percent == 0
    assert owner._longitudinal_valid_until == 99.8
    assert "motion_authorized=False" in caplog.text
    assert "history_max_age_ms=500.0 motion_max_age_ms=210.0" in caplog.text


@pytest.mark.parametrize("age", [.501, 1., -0.01, float("nan"), float("inf")])
def test_history_window_does_not_accept_unbounded_or_invalid_age(owner, age):
    r = sample(owner, 666, .83)
    owner.clock = owner._active_capture_timestamp + age
    consume(owner, [r])
    assert owner._follow_controller._direction_latest_visible_capture_id == 651
    assert owner._longitudinal_valid_until == 99.8


@pytest.mark.parametrize("reject", ["uid0", "geometry", "competition", "identity", "quality", "predicted"])
def test_late_window_does_not_bypass_identity_gates(owner, reject):
    r = sample(owner, 666, .83)
    owner.clock = owner._active_capture_timestamp + .30
    obs = owner._rknn_pipeline.tracker.last_identity_observations[0]
    if reject == "uid0": r.reid_uid = 0
    elif reject == "geometry": obs["assignment"]["reacquire_geometry_ok"] = False
    elif reject == "competition": obs["sample_metadata"]["identity_competition"] = dict(passed=False)
    elif reject == "identity": obs["assignment"]["identity_control_rejected"] = True
    elif reject == "quality": obs["assignment"]["bbox_quality_ok"] = False
    elif reject == "predicted": r.time_since_update = 1
    consume(owner, [r])
    assert owner._follow_controller._direction_latest_visible_capture_id == 651


def test_cap1226_to1240_replay_searches_right_after_hold(owner, monkeypatch):
    monkeypatch.setattr(runtime, "VISION_CONTROL_MAX_RESULT_AGE_SEC", .210)
    t, c = owner, owner._follow_controller
    # The controller's pre-brake selected tracking box was still on the left.
    c.note_brake_hold_observation(SensorFrame(width=640, height=480,
        capture_frame_id=1226, capture_timestamp=29182.192740054),
        PersonTarget((10, 20, 235.4, 459), 1, .904, 98950.6))
    c.search_direction = c._lost_exit_direction = "left"
    rows = [
        (1228, 29182.293986404, .2409, (65.6242,27.7799,284.9793,474.5524)),
        (1237, 29182.786996651, .2239, (353.2532,38.2220,565.9906,475.7416)),
        (1240, 29182.957708978, .2132, (426.8488,32.7195,639.2535,474.2030)),
    ]
    for cap, stamp, age, bbox in rows:
        r = sample(t, cap, .5)
        t.clock, t._active_capture_timestamp = stamp + age, stamp
        obs = t._rknn_pipeline.tracker.last_identity_observations[0]
        obs["detector_bbox"] = bbox
        obs["sample_metadata"]["capture_timestamp"] = stamp
        consume(t, [r])
        assert c.search_direction == "left"  # hold still owns all motion
        assert t._longitudinal_valid_until == 99.8
    assert c._direction_latest_visible_capture_id == 1240
    consume(t, [sample(t, 1244, .917, uid=0)])
    t.pending = False
    t._finish_search_brake_observation_hold()
    assert c.search_direction is None and c._lost_exit_direction is None
    for cap in (1252, 1254, 1257):
        frame = SensorFrame(width=640, height=480, capture_frame_id=cap,
                            capture_timestamp=29183.5 + (cap-1252)*.054)
        c._record_target_direction_evidence(frame, None, reliable=True)
    c._capture_lost_exit_direction(frame, search_entry=True)
    assert c._lost_exit_direction == "right"
    assert c._target_direction_history.latest_reliable_side().last_visible_capture_frame_id == 1240
    assert t._current_forward_percent == 0


@pytest.mark.parametrize("pending", [True, False])
def test_stale_pipeline_branch_keeps_safety_and_only_records_during_hold(owner, pending):
    # Execute the production branch itself; no camera, inference or motor thread.
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(runtime.PersonTracker.process_external_frame)))
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name) and n.test.id == "stale_result_discarded"
        and any(isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
                and x.func.attr == "_handle_stale_vision_result" for x in ast.walk(n)))
    r = sample(owner, 666, .83)
    owner.clock = owner._active_capture_timestamp + .30
    owner.pending = pending
    owner.frame_index = 529
    calls = []
    def invalidate(**kwargs):
        calls.append(kwargs)
        # Same ordering as the real safety handler; recording before this call
        # would lose the new visible slot again.
        owner._follow_controller.note_stale_visual_result(frame_width=640,
            now=owner.clock, capture_frame_id=666,
            capture_timestamp=owner._active_capture_timestamp)
    owner._handle_stale_vision_result = invalidate
    env = dict(vars(runtime), self=owner, width=640, height=480, records=[r],
               stale_result_discarded=True, vision_result_age_sec=.25)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[])),
                 "production_stale_branch", "exec"), env)
    assert len(calls) == 1
    assert owner._follow_controller._direction_latest_visible_capture_id == (666 if pending else 651)
    assert owner._longitudinal_valid_until == 99.8
    if pending:
        owner._finish_search_brake_observation_hold()
        assert not owner._follow_controller.stale_direction_recovery_active
        assert owner._follow_controller.search_direction is None


def test_brake_history_timing_is_current_capture_only(owner, caplog):
    r = sample(owner, 666, .83)
    owner._brake_observation_pipeline_timing = dict(capture_frame_id=666,
        capture_timestamp=owner._active_capture_timestamp,
        capture_to_pipeline_ms=66.4, vision_processing_ms=146.8)
    with caplog.at_level("INFO"):
        consume(owner, [r])
    assert "capture_to_pipeline_ms=66.4 vision_processing_ms=146.8" in caplog.text
    caplog.clear()
    with caplog.at_level("INFO"):
        consume(owner, [sample(owner, 668, .85)])
    assert "capture_to_pipeline_ms=None vision_processing_ms=None" in caplog.text
