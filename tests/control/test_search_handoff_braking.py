"""Search handoff stop permission stays separate from identity/Depth authority."""
from dataclasses import replace
from types import SimpleNamespace
import queue
import threading

import pytest
import request_0513_modular as runtime
from car_control_modular.action_command import ActionCommandSnapshot
from car_control_modular.controllers import FollowPolicyConfig
from car_control_modular.control_types import SteeringFeedback
from car_control_modular.search_reacquire_braking import search_brake_reason


def policy():
    return FollowPolicyConfig(center_left_ratio=.47, center_right_ratio=.53,
        visible_steering_pid_camera_hfov_deg=60.,
        visible_steering_pid_camera_latency_sec=.13,
        visible_steering_pid_predictive_brake_decel_dps2=100.,
        visible_steering_pid_predictive_brake_margin_deg=1.,
        visible_steering_pid_predictive_brake_response_sec=.05)


def reason(x, **changes):
    args = dict(bbox=(640*x-60, 5, 640*x+60, 470), width=640,
        direction="left", eligible=True, now=10., capture_timestamp=9.9,
        max_age=.19, feedback=SteeringFeedback(timestamp=9.98, trustworthy=True,
            yaw_rate_right_dps=-18., raw_yaw_rate_right_dps=-18.), policy=policy())
    args.update(changes)
    return search_brake_reason(**args)


@pytest.mark.parametrize("x,expected", [(.08,None), (.285,None),
    (.406,"candidate_predictive_stop"), (.463,"candidate_predictive_stop"),
    (.50,"candidate_center"), (.575,"candidate_crossed_center")])
def test_cap_1969_to_2011_geometry(x, expected):
    assert reason(x) == expected


@pytest.mark.parametrize("changes", [dict(eligible=False), dict(direction=None),
    dict(capture_timestamp=9.7), dict(capture_timestamp=10.1),
    dict(capture_timestamp=float("nan")), dict(bbox=(100,5,100,470)), dict(width=0)])
def test_bad_or_stale_candidate_cannot_request_stop(changes):
    assert reason(.5, **changes) is None


def test_wide_box_crossing_line_alone_is_not_stop_permission():
    assert reason(.25, bbox=(0, 5, 400, 470), feedback=None) is None


def test_predictive_stop_requires_fresh_rotation_feedback():
    assert reason(.406, feedback=None) is None
    assert reason(.406, feedback=SteeringFeedback(timestamp=8., trustworthy=True, yaw_rate_right_dps=-18)) is None
    assert reason(.406, feedback=SteeringFeedback(timestamp=9.98, trustworthy=True, yaw_rate_right_dps=18)) is None


def tracker(monkeypatch):
    clock = [10.]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])
    t = object.__new__(runtime.PersonTracker)
    t.search_state, t.search_direction = "searching", "left"
    t._active_capture_timestamp, t._active_capture_frame_id = 9.9, 2008
    t._follow_controller = SimpleNamespace(cfg=policy(), search_state="searching", search_direction="left")
    t.clears, t.requests, t.soft = [], [], []
    t._clear_lateral_intent = lambda why: t.clears.append(why)
    t._clear_longitudinal_context = lambda **kw: t.clears.append(kw)
    t._publish_observation_soft_zero = lambda *a, **kw: t.soft.append(a)
    t._action_runtime = SimpleNamespace(search_reacquire_brake_pending=lambda: False,
        get_steering_feedback=lambda: SteeringFeedback(timestamp=clock[0], trustworthy=True, yaw_rate_right_dps=-18),
        request_search_reacquire_brake=lambda *args: t.requests.append(args))
    return t, clock


def test_center_candidate_brakes_without_uid_or_forward_permission(monkeypatch):
    t, _ = tracker(monkeypatch)
    assert t._hold_search_reacquire_brake(bbox=(207,5,392,479), width=640, eligible=True)
    assert len(t.requests) == 1 and t._current_forward_percent == 0
    assert t.search_direction == "left" and t.search_state == "searching"


def test_settled_confirmed_target_does_not_repeat_normal_stop(monkeypatch):
    t, clock = tracker(monkeypatch)
    args = dict(bbox=(207,5,392,479), width=640, eligible=True, confirmed=True)
    assert t._hold_search_reacquire_brake(**args)
    clock[0] += .10
    t._active_capture_timestamp = clock[0]-.05
    assert not t._hold_search_reacquire_brake(**args)
    assert len(t.requests) == 1


def test_unconfirmed_center_observation_is_bounded(monkeypatch):
    t, clock = tracker(monkeypatch)
    args = dict(bbox=(207,5,392,479), width=640, eligible=True)
    assert t._hold_search_reacquire_brake(**args)
    clock[0] += runtime.SEARCH_EVIDENCE_MAX_HOLD_SEC+.01
    t._active_capture_timestamp = clock[0]-.05
    assert not t._hold_search_reacquire_brake(**args)
    assert len(t.requests) == 1


@pytest.mark.parametrize("kind", ["preferred_search_late_reacquire", "preferred_search_soft_reacquire"])
def test_completed_identity_chain_reused_but_depth_stays_pending(monkeypatch, kind):
    t, _ = tracker(monkeypatch)
    t.frame_index = 771
    t._reset_confirmed_search_reacquire()
    released = []
    t._follow_controller.active_target_id = 1
    t._follow_controller.release_search_on_confirmed_target = released.append
    t._follow_controller.set_search_observation_hold = lambda _: None
    t._start_visual_reacquire_hold = lambda *a, **kw: None
    t._distance_runtime = object()
    t._observe_search_reacquire_depth = lambda **kw: (False, None)
    monkeypatch.setattr(runtime, "MODULE_ASTRA_DEPTH_ENABLE", True)
    monkeypatch.setattr(runtime, "SEARCH_REACQUIRE_DEPTH_GATE_ENABLE", True)
    # Far left, not yet within predictive stop; no extra visual confirmation.
    candidate = dict(stable_id=1, bbox=(30,5,120,470), score=.9, area=40000,
        rec=SimpleNamespace(track_id=3, time_since_update=0),
        debug={"assignment": dict(uid=1, reason=kind, distance=.218,
            reacquire_geometry_ok=True, bbox_quality_ok=True, late_candidate_streak=2)})
    assert not t._hold_for_confirmed_search_reacquire([candidate], width=640)
    assert released == ["visual_reacquire_depth_pending"]
    assert t._reacquire_depth_pending and t.search_state == "none"


def queue_owner():
    t = object.__new__(runtime.PersonTracker)
    t.action_queue = queue.Queue()
    t.action_queue_lock = threading.Lock()
    t.motor_io_lock = threading.Lock()
    t.frame_index = 798
    t.current_command = None
    t.search_state, t.search_direction = "none", None
    t._action_queue_seq = t._last_action_queue_seq = 0
    t._last_tracker_action_kind = None
    t._last_action_queue_replace_ts = t._last_tracker_action_change_ts = 0.
    t._last_tracker_action_change_frame = -1
    t._last_dispatched_action = None
    t._actions_signature = lambda actions: tuple(actions)
    return t


def test_real_publisher_replaces_ordinary_stop_and_keeps_provenance():
    t = queue_owner()
    t._use_soft_stop_next = True
    t._last_command_capture_frame = 2074
    t._replace_action_queue([runtime.ACTION_STOP], "lateral_zero:expired")
    old = t.action_queue.queue[0]
    t._last_command_capture_frame = 2077
    t._use_soft_stop_next = False
    t._replace_action_queue([runtime.ACTION_ROTATE_RIGHT], "visual_pid_right_encoder")
    assert t._action_queue_snapshot_locked() == [runtime.ACTION_ROTATE_RIGHT]
    new = t.action_queue.queue[0]
    assert isinstance(new, ActionCommandSnapshot) and new.revision > old.revision
    assert old.capture_frame_id == 2074 and old.soft_stop
    assert new.capture_frame_id == 2077 and not new.soft_stop


def test_real_publisher_never_discards_protected_stop():
    t = queue_owner()
    t._replace_action_queue([runtime.ACTION_STOP], "user_stop")
    old = t.action_queue.queue[0]
    t._replace_action_queue([runtime.ACTION_ROTATE_RIGHT], "visual_pid_right_encoder")
    assert t.action_queue.queue[0] is old and old.protected_stop
    assert t._action_queue_snapshot_locked() == [runtime.ACTION_STOP, runtime.ACTION_ROTATE_RIGHT]


def test_early_identity_release_keeps_one_center_brake_obligation(monkeypatch):
    t, clock = tracker(monkeypatch)
    t._follow_controller.search_state = "none"
    t._follow_controller.active_target_id = 1
    t.search_state, t.search_direction = "none", None
    t._search_handoff_uid, t._search_handoff_direction = 1, "left"
    args = dict(bbox=(310,5,426,470), width=640, eligible=True, confirmed=True)
    assert t._hold_search_reacquire_brake(**args)
    assert t._search_handoff_uid is None
    clock[0] += .1
    t._active_capture_timestamp = clock[0]-.05
    assert not t._hold_search_reacquire_brake(**args)
    assert len(t.requests) == 1
