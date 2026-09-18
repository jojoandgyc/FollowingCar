"""CAP834 latest-first ranging and deadline-aware motion admission, no hardware."""
from dataclasses import replace
from types import SimpleNamespace
import threading

import pytest
import request_0513_modular as runtime
from car_control_modular.control_types import DistanceState, HazardState
from test_depth_target_snapshot import owner as context_owner, persons, NOW
from test_lateral_zero_runtime import owner
from test_longitudinal_authority_runtime import _commit
from test_search_observation_arbitration import owner as scene, _record


@pytest.fixture
def visible(context_owner):
    obj = context_owner
    obj._vision_control_state = "target_visible_depth_valid"
    obj._depth_longitudinal_authority_enabled = lambda: True
    obj._follow_controller.last_selected_target = obj._persons_to_targets(persons(), width=640, height=480)[0]
    obj._reacquire_depth_pending = False
    obj._explicit_stop_requested = False
    obj._brake_hold_active = False
    obj._longitudinal_wake_event = threading.Event()
    return obj


def eligible(obj, targets=None, **kwargs):
    targets = targets if targets is not None else obj._persons_to_targets(persons(), width=640, height=480)
    return obj._visible_latest_depth_eligible(
        targets, targets[0], obj._active_capture_frame_id, obj._active_capture_timestamp,
        control_source=kwargs.get("source", "vision"),
        target_steerable=kwargs.get("steerable", True),
        low_quality_visible=kwargs.get("weak", False),
    )


def test_normal_visible_current_detector_is_eligible(visible):
    assert eligible(visible)


def test_typed_distance_parking_allows_latest_measurement_but_not_wheel_authority(visible):
    from car_control_modular.follow_distance_hold import FollowDistanceHold
    visible._brake_hold_active = True
    visible._brake_hold_label = 'follow_distance_hold'
    visible._brake_hold_stop_mode = None
    visible._follow_distance_hold = FollowDistanceHold(1, NOW-.1)
    assert eligible(visible)
    assert visible._fresh_depth_linear_snapshot(1, now=NOW) is None


def test_parked_depth_thread_measures_without_running_pid_or_writing_actions(visible):
    from car_control_modular.follow_distance_hold import FollowDistanceHold
    from car_control_modular.control_types import SteeringFeedback
    obj = visible
    obj._brake_hold_active = True
    obj._brake_hold_label = 'follow_distance_hold'
    obj._brake_hold_stop_mode = None
    obj._follow_distance_hold = FollowDistanceHold(1, NOW-.15)
    obj._last_dispatched_action = runtime.ACTION_STOP
    obj._follow_controller.set_last_dispatched = lambda _: None
    obj._follow_controller._is_fresh_depth_state = lambda frame: True
    obj._follow_controller.update = lambda *a, **kw: pytest.fail('parked PID must not run')
    obj._get_obstacle_status = lambda: {}
    obj._action_runtime = SimpleNamespace(get_steering_feedback=lambda: SteeringFeedback(
        timestamp=NOW-.01, trustworthy=True))
    obj._current_hazard_state_for_controller = lambda: HazardState()
    measured = []
    def measure(*args, **kwargs):
        measured.append(kwargs)
        return DistanceState(source='vision_depth', sample_timestamp=NOW-.02,
                             raw_distance_m=1.7, used_distance_m=1.7)
    obj._distance_runtime = SimpleNamespace(select_target=lambda ts: ts[0],get_frame_distance_state=measure)
    assert obj._process_detections_modular(640,480,persons(),control_source='depth30',depth_use_latest=True)==[]
    assert len(measured)==1 and obj._brake_hold_active
    assert obj._follow_distance_hold.count==1


def test_parked_fresh_depth_context_reaches_sampling_dispatch(visible):
    from car_control_modular.follow_distance_hold import FollowDistanceHold
    visible._brake_hold_active = True
    visible._brake_hold_label = 'follow_distance_hold'
    visible._brake_hold_stop_mode = None
    visible._follow_distance_hold = FollowDistanceHold(1,NOW-.15)
    visible._publish_longitudinal_context(640,480,persons())
    context = visible._longitudinal_context
    calls=[]
    visible._queue_actions_for_persons_locked=lambda *a,**kw:calls.append(kw)
    visible._queue_actions_for_persons(640,480,persons(),control_source='depth30',expected_target_id=1,
        depth_target_snapshot=context['person_targets'],evidence_capture_frame_id=1015,
        evidence_capture_timestamp=NOW-.1)
    assert len(calls)==1 and visible._brake_hold_active


@pytest.mark.parametrize("case", ["search", "controller_search", "lost", "weak_state", "stop",
    "brake", "shutdown", "stopped", "pending", "other_uid", "initial", "ambiguous",
    "missing_detector", "wrong_capture", "wrong_uid", "wrong_source", "expired", "future"])
def test_priority_never_bypasses_existing_identity_or_safety_gates(visible, case):
    targets = visible._persons_to_targets(persons(), width=640, height=480)
    if case == "search": visible.search_state = "searching"
    elif case == "controller_search": visible._follow_controller.search_state = "searching"
    elif case == "lost": visible._vision_control_state = "lost_confirming"
    elif case == "weak_state": visible._vision_control_state = "target_visible_low_quality"
    elif case == "stop": visible._explicit_stop_requested = True
    elif case == "brake": visible._brake_hold_active = True
    elif case == "shutdown": visible._runtime_shutdown_requested = True
    elif case == "stopped": visible.running = False
    elif case == "pending": visible._reacquire_depth_pending = True
    elif case == "other_uid": visible._follow_controller.active_target_id = 2
    elif case == "initial": visible._follow_controller.last_selected_target = None
    elif case == "ambiguous": targets *= 2
    elif case == "missing_detector": targets = [replace(targets[0], depth_observation=None)]
    else:
        obs = targets[0].depth_observation
        if case == "wrong_capture": obs = replace(obs, capture_frame_id=9)
        elif case == "wrong_uid": obs = replace(obs, target_id=2)
        elif case == "wrong_source": obs = replace(obs, source="tracker_bbox")
        else:
            stamp = NOW - .181 if case == "expired" else NOW + .01
            visible._active_capture_timestamp = stamp
            obs = replace(obs, capture_timestamp=stamp)
        targets = [replace(targets[0], depth_observation=obs)]
    assert not eligible(visible, targets)


@pytest.mark.parametrize("kwargs", [dict(source="depth30"), dict(weak=True), dict(steerable=False)])
def test_priority_only_applies_to_regular_visual_path(visible, kwargs):
    assert not eligible(visible, **kwargs)


@pytest.mark.parametrize("search", [False, True])
@pytest.mark.parametrize("capture_age", [.1, .175])
def test_real_visual_process_samples_latest_before_decision_and_keeps_capture_provenance(visible, search, capture_age):
    obj = visible
    obj._active_capture_timestamp = NOW-capture_age
    obj._rknn_pipeline.tracker.last_identity_observations[0]["sample_metadata"]["capture_timestamp"] = NOW-capture_age
    obj._last_dispatched_action = runtime.ACTION_STOP
    obj._follow_controller.set_last_dispatched = lambda _: None
    obj._get_obstacle_status = lambda: {}
    obj._action_runtime = SimpleNamespace(get_steering_feedback=lambda: None)
    obj._current_hazard_state_for_controller = lambda: HazardState()
    if search:
        obj.search_state = obj._follow_controller.search_state = "searching"
    observations = []
    def measure(*args, **kwargs):
        observations.append(kwargs)
        if not search:
            # ROI is published before sampling, rather than after decision/logs.
            assert obj._longitudinal_wake_event.is_set()
            assert obj._longitudinal_context["capture_timestamp"] == NOW-capture_age
            assert obj._longitudinal_context["person_targets"][0].depth_observation.bbox == args[1].depth_observation.bbox
        return DistanceState(source="vision_depth", sample_timestamp=NOW-.02,
                             raw_distance_m=2.906, used_distance_m=2.906)
    obj._distance_runtime = SimpleNamespace(select_target=lambda ts: ts[0], get_frame_distance_state=measure)
    class DecisionReached(Exception): pass
    def decide(_, frame, **kwargs):
        assert frame.capture_timestamp == NOW-capture_age  # never stamp old RGB as now
        assert frame.distance_state.sample_timestamp == NOW-.02
        raise DecisionReached()
    obj._follow_controller.decide = decide
    with pytest.raises(DecisionReached):
        obj._process_detections_modular(640, 480, persons())
    assert observations[0]["depth_use_latest"] is not search
    assert observations[0]["capture_timestamp"] == (NOW-capture_age if search else None)
    if search:
        assert obj._longitudinal_context is None


@pytest.mark.parametrize("flag", ["_explicit_stop_requested", "_brake_hold_active"])
def test_early_roi_cannot_outlive_later_stop_in_same_visual_decision(visible, flag):
    visible._publish_longitudinal_context(640, 480, persons())
    context = visible._longitudinal_context
    setattr(visible, flag, True)
    calls = []
    visible._queue_actions_for_persons_locked = lambda *a, **kw: calls.append(kw)
    visible._queue_actions_for_persons(
        640, 480, persons(), control_source="depth30", expected_target_id=1,
        depth_target_snapshot=context["person_targets"], evidence_capture_frame_id=1015,
        evidence_capture_timestamp=NOW-.1,
    )
    assert not calls


@pytest.mark.parametrize("kind", ["forward", "backward"])
def test_cap834_four_ms_lease_cannot_start_motion(owner, kind, caplog):
    actions, accepted = _commit(owner, stamp=NOW-.1756, percent=55, kind=kind)
    assert (actions, accepted) == ([], False)
    assert owner._depth30_linear_snapshot is None
    assert "action=defer_positive" in caplog.text
    assert "remaining_ms=4.4" in caplog.text


def test_latest_sample_replaces_deferred_history_without_waiting(owner, monkeypatch):
    assert not _commit(owner, stamp=NOW-.1756, percent=55)[1]
    assert _commit(owner, stamp=NOW-.02, percent=55)[1]
    assert owner._depth30_linear_snapshot == ("forward", 55, 1, NOW-.02)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.024)
    assert owner._fresh_depth_linear_snapshot(1) == ("forward", 55, 1, NOW-.02)


def test_almost_expired_sample_can_stop_immediately(owner):
    _commit(owner, stamp=NOW-.176, percent=55)  # not admitted
    assert _commit(owner, stamp=NOW-.174, percent=0)[1]
    assert owner._depth30_linear_snapshot is None


def test_short_lease_may_reduce_but_never_renew_existing_authority(owner, monkeypatch):
    _commit(owner, stamp=NOW-.04, percent=40)
    old = owner._depth30_linear_snapshot
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.105)
    # A newer physical sample, only 35ms left on arrival (budget=50ms).
    actions, committed = _commit(owner, stamp=NOW-.039, percent=10)
    assert committed and actions[0].speed_percent == 10
    assert owner._depth30_linear_snapshot == ("forward", 10, 1, old[3])


def test_short_lease_cannot_accelerate_or_refresh_old_deadline(owner, monkeypatch):
    _commit(owner, stamp=NOW-.04, percent=17)
    old = owner._depth30_linear_snapshot
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.105)
    assert _commit(owner, stamp=NOW-.039, percent=40) == ([], False)
    assert owner._depth30_linear_snapshot == old
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.141)
    assert owner._fresh_depth_linear_snapshot(1) is None


def test_deferred_positive_reports_actual_limit_to_pid(owner):
    calls = []
    owner._follow_controller.accept_longitudinal_limit = lambda *args: calls.append(args)
    _commit(owner, stamp=NOW-.1756, percent=55)
    assert calls == [(NOW-.1756, 0)]


def test_zero_is_not_filtered_by_dispatch_budget(owner):
    _commit(owner, stamp=NOW-.17, percent=0)
    assert owner._depth30_linear_sample_watermark == (1, NOW-.17)
    assert _commit(owner, stamp=NOW-.175, percent=55) == ([], False)


@pytest.mark.parametrize("old,new", [("forward", "backward"), ("backward", "forward")])
def test_late_direction_change_stops_instead_of_preserving_opposite_motion(owner, monkeypatch, old, new):
    _commit(owner, stamp=NOW-.04, percent=17, kind=old)
    revision = owner._lateral_yaw_revision
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.105)
    actions, accepted = _commit(owner, stamp=NOW-.039, percent=12, kind=new)
    assert accepted and actions[0].speed_percent == 0
    assert owner._depth30_linear_snapshot is None
    assert owner._lateral_yaw_revision > revision


@pytest.mark.parametrize("early", [False, True])
def test_visual_tail_does_not_replace_early_roi_snapshot(scene, early):
    scene.search_state = scene._follow_controller.search_state = "none"
    scene._vision_control_state = "target_visible_depth_valid"
    scene._assignments[3] = {"bbox_quality_ok": True}
    scene._longitudinal_context = None
    scene._longitudinal_context_lock = threading.Lock()
    marker = {"frame_index": scene.frame_index,
              "capture_frame_id": scene._active_capture_frame_id, "target_id": 7}
    def process(*args, **kwargs):
        if early:
            scene._longitudinal_context = marker
    scene._queue_actions_for_persons = process
    scene._consume_track_records([_record()], 640, 480, "test")
    assert scene._context_events.count("publish") == int(not early)
    if early:
        assert scene._longitudinal_context is marker
