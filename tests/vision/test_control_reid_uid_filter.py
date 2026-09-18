#!/usr/bin/env python3
from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import bunker_hazard_detector  # noqa: F401
except ModuleNotFoundError:
    # 开发机没有板端危险检测扩展；本测试只验证视觉候选筛选。
    hazard_stub = types.ModuleType("bunker_hazard_detector")
    hazard_stub.BunkerHazardMonitor = object
    hazard_stub.RKNNBunkerHazardDetector = object
    hazard_stub.HazardState = types.SimpleNamespace
    hazard_stub.check_hazard_from_dets = lambda *_args, **_kwargs: None
    sys.modules["bunker_hazard_detector"] = hazard_stub

try:
    import track_first_person2  # noqa: F401
except ModuleNotFoundError:
    tracker_stub = types.ModuleType("track_first_person2")
    tracker_stub.FirstPersonTracker = object
    sys.modules["track_first_person2"] = tracker_stub

import request_0513_modular as mod
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig, TrackRecord


def test_detector_crop_edge_reason_keeps_partial_identity_tier() -> None:
    tracker = DeepSortTracker(DeepSortTrackerConfig())
    edge_crop_tier = tracker._bbox_identity_tier(
        bbox_quality_ok=False,
        bbox_quality_reason="edge_touch>2,detector_crop:edge_touch>2",
        confidence=0.9,
        bbox=(0.0, 0.0, 639.0, 479.0),
    )
    assert edge_crop_tier == "weak"

    mixed_risk_tier = tracker._bbox_identity_tier(
        bbox_quality_ok=False,
        bbox_quality_reason="area_ratio>0.75,detector_crop:edge_touch>2",
        confidence=0.9,
        bbox=(0.0, 0.0, 639.0, 479.0),
    )
    assert mixed_risk_tier == "reject"


def _track(
    track_id: int,
    reid_uid: int,
    bbox=(10.0, 20.0, 110.0, 220.0),
) -> TrackRecord:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return TrackRecord(
        track_id=track_id,
        reid_uid=reid_uid,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        class_id=mod.PERSON_CLASS_ID,
        score=0.9,
        cx=(x1 + x2) / 2.0,
        cy=(y1 + y2) / 2.0,
        area=max(0.0, x2 - x1) * max(0.0, y2 - y1),
        angle_deg=0.0,
        tracker_state=2,
        time_since_update=0,
    )


class _DummyTracker:
    def __init__(self, assignment=None) -> None:
        self.frame_index = 1
        self.queued_persons = None
        self.queue_calls = []
        self.queue_options = []
        self.published_contexts = []
        self._last_person_reid_debug_by_stable_id = {}
        self._bunker_runtime = SimpleNamespace(check_merged_dets=lambda *_args: None)
        self._follow_controller = SimpleNamespace(
            active_target_id=None,
            search_state="none",
            search_direction=None,
            defer_search_timeout=lambda _seconds: None,
            search_status=lambda _now=None: SimpleNamespace(state="none"),
        )
        self._single_person_geometry_bbox = None
        self._single_person_geometry_frame = -1
        self._single_person_geometry_track_id = None
        self._single_person_geometry_streak = 0
        self._single_person_geometry_anchor_uid = None
        self._single_person_geometry_unsteerable_uid = None
        self._visible_unsteerable_uid = None
        self._visible_unsteerable_track_id = None
        self._visible_unsteerable_last_ts = None
        self._visible_unsteerable_bbox = None
        self._visible_unsteerable_recovery_frames = 0
        self._search_geometry_reacquire_track_id = None
        self._search_geometry_reacquire_last_frame = -1
        self._search_geometry_reacquire_frames = 0
        self._search_geometry_reacquire_bbox = None
        self._confirmed_search_reacquire_uid = None
        self._confirmed_search_reacquire_track_id = None
        self._confirmed_search_reacquire_bbox = None
        self._confirmed_search_reacquire_last_frame = -1
        self._confirmed_search_reacquire_streak = 0
        self.search_state = "none"
        self.search_direction = None
        self.assignment = dict(assignment or {})

    _stable_id_from_track_record = staticmethod(mod.PersonTracker._stable_id_from_track_record)
    _bbox_iou_xyxy = staticmethod(mod.PersonTracker._bbox_iou_xyxy)
    _bound_reid_bbox_allowed_for_control = staticmethod(
        mod.PersonTracker._bound_reid_bbox_allowed_for_control
    )
    _bbox_quality_is_fragment = staticmethod(mod.PersonTracker._bbox_quality_is_fragment)
    _single_person_geometry_fallback_id = mod.PersonTracker._single_person_geometry_fallback_id
    _reset_search_geometry_reacquire = mod.PersonTracker._reset_search_geometry_reacquire
    _search_geometry_reacquire_id = mod.PersonTracker._search_geometry_reacquire_id
    _reset_confirmed_search_reacquire = mod.PersonTracker._reset_confirmed_search_reacquire
    _hold_for_confirmed_search_reacquire = (
        mod.PersonTracker._hold_for_confirmed_search_reacquire
    )

    def _identity_assignment_debug_for_track(self, _track_id: int):
        return dict(self.assignment)

    def _handle_hazard_safety_state(self, _state) -> bool:
        return False

    def _hold_for_visible_unsteerable_target(self, **_kwargs) -> bool:
        return False

    def _clear_longitudinal_context(self, **_kwargs) -> None:
        return None

    def _publish_longitudinal_context(
        self,
        _width: int,
        _height: int,
        persons,
        *,
        target_steerable: bool = True,
    ) -> None:
        self.published_contexts.append((list(persons), bool(target_steerable)))

    def _queue_actions_for_persons(
        self,
        _width: int,
        _height: int,
        persons,
        *,
        target_steerable: bool = True,
        **_kwargs,
    ) -> None:
        self.queued_persons = list(persons)
        self.queue_calls.append((list(persons), bool(target_steerable)))
        self.queue_options.append(dict(_kwargs))


def _consume(
    records,
    *,
    reid_enable: bool,
    assignment=None,
    active_target_id=None,
):
    old_reid_enable = mod.VISION_REID_ENABLE
    mod.VISION_REID_ENABLE = reid_enable
    try:
        dummy = _DummyTracker(assignment)
        dummy._follow_controller.active_target_id = active_target_id
        mod.PersonTracker._consume_track_records(dummy, records, 640, 480, "test")
        return dummy.queued_persons
    finally:
        mod.VISION_REID_ENABLE = old_reid_enable


def _assert_active_target_track_uses_established_mapping_only() -> None:
    tracker = object.__new__(mod.PersonTracker)
    tracker._rknn_pipeline = SimpleNamespace(
        tracker=SimpleNamespace(
            identity_bank=SimpleNamespace(
                last_assignments={
                    3: {
                        "uid": 0,
                        "mapped_uid": 7,
                        "best_uid": 7,
                        "reason": "mapped_low_quality",
                        "bbox_quality_ok": False,
                    },
                    9: {
                        "uid": 0,
                        "best_uid": 7,
                        "reason": "uid_claimed_recently",
                        "bbox_quality_ok": True,
                    },
                }
            )
        )
    )
    correct = _track(3, 0, bbox=(0.0, 20.0, 180.0, 470.0))
    wrong = _track(9, 0, bbox=(300.0, 80.0, 500.0, 420.0))
    record, assignment = mod.PersonTracker._active_target_track_record(
        tracker,
        [wrong, correct],
        7,
    )
    if record is not correct or int(assignment.get("mapped_uid", 0)) != 7:
        raise AssertionError(
            f"established mapped UID did not beat best_uid hypothesis: {record}, {assignment}"
        )
    tracker._rknn_pipeline.tracker.identity_bank.last_assignments = {
        4: {
            "uid": 0,
            "mapped_uid": 7,
            "best_uid": 7,
            "reason": "mapped_verify_reject",
            "bbox_quality_ok": True,
        }
    }
    rejected, _assignment = mod.PersonTracker._active_target_track_record(
        tracker,
        [_track(4, 0)],
        7,
    )
    if rejected is not None:
        raise AssertionError("a rejected mapped UID must not regain target priority")


def _assert_search_timeout_stops_before_runtime_exit() -> None:
    events = []

    class _ActionRuntime:
        def send_stop_with_brake_hold(self, reason: str) -> None:
            events.append(("stop", reason))

    dummy = SimpleNamespace(
        search_state="timed_out",
        frame_index=99,
        _runtime_shutdown_requested=True,
        _explicit_stop_requested=True,
        _last_explicit_stop_reason="search_timeout_exit",
        _follow_controller=SimpleNamespace(last_selected_target=None),
        _last_control_decision_reason="target_lost_exit",
        _current_rotate_raw_target=0,
        _current_rotate_raw_source="test",
        _current_steer_base_percent=0,
        _current_steer_correction_rpm=0,
        _last_target_loss_trace_frame=-1,
        _action_runtime=_ActionRuntime(),
        _control_update_lock=threading.RLock(),
        running=True,
        action_stop_event=threading.Event(),
        _process_detections_modular=lambda *_args, **_kwargs: [],
        _clear_action_queue=lambda reason: events.append(("clear", reason)),
        _prepare_direct_stop=lambda reason: events.append(("prepare", reason)),
        _mark_direct_stop_sent=lambda reason: events.append(("marked", reason)),
        _should_skip_redundant_direct_stop=lambda _reason: True,
    )
    dummy._queue_actions_for_persons_locked = types.MethodType(
        mod.PersonTracker._queue_actions_for_persons_locked,
        dummy,
    )
    mod.PersonTracker._queue_actions_for_persons(dummy, 640, 480, [])
    if ("stop", "search_timeout_exit") not in events:
        raise AssertionError(f"timeout exit must send a direct STOP even after a recent stop: {events}")
    if dummy.running or not dummy.action_stop_event.is_set():
        raise AssertionError("timeout exit must end the main loop only after issuing STOP")


def _assert_near_camera_occlusion_gate() -> None:
    gate = mod.PersonTracker._bbox_quality_is_near_camera_occlusion
    if not gate(
        "area_ratio>0.75,width_ratio>0.85,edge_touch>2",
        (0.0, 0.0, 639.0, 479.0),
        640,
        480,
    ):
        raise AssertionError("near-full-frame locked person must enter occlusion hold")
    if gate(
        "area_ratio>0.75,edge_touch>2",
        (110.0, 0.0, 585.0, 479.0),
        640,
        480,
    ):
        raise AssertionError("ordinary clipped large box must keep the coarse path")
    if gate(
        "area_ratio>0.75,width_ratio>0.85,edge_touch>2,aspect<0.18",
        (0.0, 0.0, 639.0, 80.0),
        640,
        480,
    ):
        raise AssertionError("wide fragment must not be treated as camera occlusion")


def _assert_stale_result_revokes_published_yaw_without_brake() -> None:
    events = []
    controller = SimpleNamespace(
        search_state="none",
        search_direction="right",
        lost_confirm_frames=0,
        stale_direction_recovery_active=True,
    )

    def note_stale_visual_result(
        *,
        frame_width: int,
        now: float,
        capture_frame_id: int,
        capture_timestamp: float,
    ) -> bool:
        assert frame_width == 640
        assert now > 0.0
        assert capture_frame_id == 0
        assert capture_timestamp == 0.0
        controller.search_state = "direction_probe"
        controller.search_direction = None
        return True

    controller.note_stale_visual_result = note_stale_visual_result
    dummy = SimpleNamespace(
        frame_index=902,
        _follow_controller=controller,
        _control_update_lock=threading.RLock(),
        command_lock=threading.RLock(),
        action_queue_lock=threading.RLock(),
        current_command=mod.ACTION_ROTATE_RIGHT,
        _last_dispatched_action=mod.ACTION_ROTATE_RIGHT,
        _brake_hold_active=False,
        _explicit_stop_requested=False,
        _use_soft_stop_next=False,
        _vision_control_state="target_visible_depth_valid",
        _current_forward_percent=30,
        _current_steer_base_percent=30,
        _current_steer_correction_rpm=8,
        _current_steer_limit_reason="visible",
        _current_rotate_raw_target=8,
        _current_rotate_raw_source="visible",
        _current_rotate_pulse_enabled=False,
        is_forwarding=False,
        search_state="none",
        search_direction="right",
        lost_confirm_frames=0,
        _clear_lateral_intent=lambda reason: events.append(("clear_lateral", reason)),
        _clear_longitudinal_context=lambda **kwargs: events.append(("clear_longitudinal",)),
        _action_queue_snapshot_locked=lambda: [],
        _action_names_for_log=lambda actions: [
            mod.ACTION_NAMES.get(action, str(action)) for action in actions
        ],
        _replace_action_queue=lambda actions, reason: events.append(
            ("replace", list(actions), reason)
        ),
    )
    handled = mod.PersonTracker._handle_stale_vision_result(
        dummy,
        width=640,
        result_age_sec=0.1825,
    )

    if not handled:
        raise AssertionError("stale result must enter direction recovery")
    if dummy.search_state != "direction_probe" or dummy.search_direction is not None:
        raise AssertionError(
            f"stale result must invalidate old direction: {dummy.search_state}/{dummy.search_direction}"
        )
    if dummy._explicit_stop_requested:
        raise AssertionError("stale vision gap must not request brake hold")
    if not dummy._use_soft_stop_next:
        raise AssertionError("active yaw must be replaced by one soft zero command")
    if ("replace", [mod.ACTION_STOP], "stale_vision_zero_yaw") not in events:
        raise AssertionError(f"soft zero was not published: {events}")
    if dummy._current_rotate_raw_target != 0 or dummy._current_steer_correction_rpm != 0:
        raise AssertionError("stale yaw parameters were not cleared")


def _assert_rotation_only_action_filter() -> None:
    cases = (
        (mod.ControlAction.forward(80, "far"), "stop", 0, 0),
        (mod.ControlAction.backward(60, "near"), "stop", 0, 0),
        (
            mod.ControlAction.backward(60, "near_right", correction_rpm=7),
            "steer_right",
            0,
            7,
        ),
        (
            mod.ControlAction.backward(60, "near_left", correction_rpm=-5),
            "steer_left",
            0,
            5,
        ),
        (
            mod.ControlAction.steer_left(75, 80, 120, "off_center", correction_rpm=9),
            "steer_left",
            0,
            9,
        ),
    )
    for source, expected_kind, expected_speed, expected_correction in cases:
        filtered = mod.PersonTracker._rotation_only_action(source)
        actual = (
            filtered.kind,
            int(filtered.speed_percent),
            int(filtered.steer_correction_rpm),
        )
        expected = (expected_kind, expected_speed, expected_correction)
        if actual != expected:
            raise AssertionError(
                f"rotation-only filter mismatch for {source}: expected={expected} actual={actual}"
            )

    rotate = mod.ControlAction.rotate_left("search_left")
    if mod.PersonTracker._rotation_only_action(rotate) != rotate:
        raise AssertionError("rotation-only mode must preserve lost-target rotation")


def _assert_confirmed_search_reacquire_respects_configured_stability() -> None:
    stops = []
    holds = []

    class _ReacquireTracker:
        _bbox_iou_xyxy = staticmethod(mod.PersonTracker._bbox_iou_xyxy)
        _reset_confirmed_search_reacquire = (
            mod.PersonTracker._reset_confirmed_search_reacquire
        )
        _hold_for_confirmed_search_reacquire = (
            mod.PersonTracker._hold_for_confirmed_search_reacquire
        )
        _publish_observation_soft_zero = (
            mod.PersonTracker._publish_observation_soft_zero
        )

        def __init__(self) -> None:
            self.frame_index = 200
            self.search_state = "searching"
            self._confirmed_search_reacquire_uid = None
            self._confirmed_search_reacquire_track_id = None
            self._confirmed_search_reacquire_bbox = None
            self._confirmed_search_reacquire_last_frame = -1
            self._confirmed_search_reacquire_streak = 0
            self._last_control_decision_reason = "search_right"
            self.is_forwarding = True
            self.soft_actions = []
            self._follow_controller = SimpleNamespace(
                active_target_id=7,
                search_status=lambda _now=None: SimpleNamespace(
                    state="searching",
                    active_target_id=7,
                ),
                set_search_observation_hold=lambda enabled: holds.append(bool(enabled)),
            )
            self._action_runtime = SimpleNamespace(
                send_stop_with_brake_hold=lambda reason: stops.append(str(reason))
            )

        def _clear_action_queue(self, _reason: str) -> None:
            return None

        def _clear_lateral_intent(self, _reason: str) -> None:
            return None

        def _clear_longitudinal_context(self, **_kwargs) -> None:
            return None

        def _replace_action_queue(self, actions, reason: str) -> None:
            self.soft_actions.append((list(actions), str(reason)))

        def _should_skip_redundant_direct_stop(self, _reason: str) -> bool:
            return False

        def _prepare_direct_stop(self, _reason: str) -> None:
            return None

        def _mark_direct_stop_sent(self, _reason: str) -> None:
            return None

    tracker = _ReacquireTracker()
    impostor_record = _track(19, 8, bbox=(120.0, 40.0, 340.0, 450.0))
    impostor = [{
        "stable_id": 8,
        "geometry_fallback": False,
        "bbox": (120.0, 40.0, 340.0, 450.0),
        "rec": impostor_record,
    }]
    if tracker._hold_for_confirmed_search_reacquire(impostor, width=640):
        raise AssertionError("a different confirmed UID must not enter target reacquire")
    first_record = _track(20, 7, bbox=(120.0, 40.0, 340.0, 450.0))
    first = [{
        "stable_id": 7,
        "geometry_fallback": False,
        "bbox": (120.0, 40.0, 340.0, 450.0),
        "rec": first_record,
    }]
    first_held = tracker._hold_for_confirmed_search_reacquire(first, width=640)
    required = max(1, int(mod.SEARCH_CONFIRMED_REACQUIRE_FRAMES))
    if required == 1:
        if first_held:
            raise AssertionError("single-frame reacquire policy must resume immediately")
        if stops or holds or tracker.soft_actions:
            raise AssertionError(
                "single-frame reacquire must not publish an observation stop: "
                f"stops={stops} holds={holds} soft={tracker.soft_actions}"
            )
        return
    if not first_held:
        raise AssertionError("multi-frame reacquire policy must hold the first frame")
    tracker.frame_index = 201
    second_record = _track(21, 7, bbox=(125.0, 40.0, 345.0, 450.0))
    second = [{
        "stable_id": 7,
        "geometry_fallback": False,
        "bbox": (125.0, 40.0, 345.0, 450.0),
        "rec": second_record,
    }]
    second_held = tracker._hold_for_confirmed_search_reacquire(second, width=640)
    if required > 2 and not second_held:
        raise AssertionError("the second stable UID frame must still be held")
    if required <= 2:
        if second_held:
            raise AssertionError("two stable UID frames must release normal follow control")
    else:
        tracker.frame_index = 202
        third_record = _track(22, 7, bbox=(130.0, 40.0, 350.0, 450.0))
        third = [{
            "stable_id": 7,
            "geometry_fallback": False,
            "bbox": (130.0, 40.0, 350.0, 450.0),
            "rec": third_record,
        }]
        if tracker._hold_for_confirmed_search_reacquire(third, width=640):
            raise AssertionError("three stable UID frames must release normal follow control")
        expected_soft = [
            ([mod.ACTION_FORWARD], "confirmed_search_reacquire_wait"),
            ([mod.ACTION_FORWARD], "confirmed_search_reacquire_wait"),
        ]
        if stops or holds != [True, True] or tracker.soft_actions != expected_soft:
            raise AssertionError(
                "reacquire observation must use publisher zero without brake hold: "
                f"stops={stops} holds={holds} soft={tracker.soft_actions}"
            )

    class _DepthWaitTracker(_ReacquireTracker):
        _publish_search_reacquire_direction_hold = (
            mod.PersonTracker._publish_search_reacquire_direction_hold
        )

        def __init__(self) -> None:
            super().__init__()
            self.search_direction = "left"
            self._distance_runtime = object()
            self._follow_controller.search_status = lambda _now=None: SimpleNamespace(
                state="searching",
                direction="left",
                active_target_id=7,
            )

        def _observe_search_reacquire_depth(self, **_kwargs):
            return False, SimpleNamespace(
                used_distance_m=None,
                sample_count=0,
                source_detail="insufficient_depth_pixels",
                sample_age_sec=None,
            )

    depth_tracker = _DepthWaitTracker()
    depth_record = _track(30, 7, bbox=(120.0, 40.0, 340.0, 450.0))
    depth_candidate = [{
        "stable_id": 7,
        "geometry_fallback": False,
        "bbox": (120.0, 40.0, 340.0, 450.0),
        "rec": depth_record,
    }]
    previous_depth_enable = mod.MODULE_ASTRA_DEPTH_ENABLE
    previous_depth_gate = mod.SEARCH_REACQUIRE_DEPTH_GATE_ENABLE
    mod.MODULE_ASTRA_DEPTH_ENABLE = True
    mod.SEARCH_REACQUIRE_DEPTH_GATE_ENABLE = True
    try:
        if not depth_tracker._hold_for_confirmed_search_reacquire(
            depth_candidate, width=640
        ):
            raise AssertionError("invalid depth must keep the reacquire hold active")
    finally:
        mod.MODULE_ASTRA_DEPTH_ENABLE = previous_depth_enable
        mod.SEARCH_REACQUIRE_DEPTH_GATE_ENABLE = previous_depth_gate
    expected_depth_hold = [
        ([mod.ACTION_ROTATE_LEFT], "confirmed_search_reacquire_depth_wait")
    ]
    if depth_tracker.soft_actions != expected_depth_hold:
        raise AssertionError(
            "invalid depth must preserve the frozen search direction: "
            f"soft={depth_tracker.soft_actions}"
        )

    # IdentityBank's late-candidate path has already completed its own 2/2
    # visual confirmation.  An invalid Depth sample must suppress forward
    # context, but must not keep the old frozen search direction active.
    late_tracker = _DepthWaitTracker()
    released = []
    late_tracker._follow_controller.release_search_on_confirmed_target = (
        lambda reason: released.append(str(reason))
    )
    late_tracker.assignment = {
        "uid": 7,
        "reason": "preferred_search_late_reacquire",
        "distance": 0.12,
        "match_source": "strong",
        "reacquire_geometry_ok": True,
        "bbox_quality_ok": True,
    }
    late_candidate = [{
        "stable_id": 7,
        "geometry_fallback": False,
        "bbox": (120.0, 40.0, 340.0, 450.0),
        "rec": _track(31, 7, bbox=(120.0, 40.0, 340.0, 450.0)),
        "debug": {"assignment": dict(late_tracker.assignment)},
    }]
    previous_depth_enable = mod.MODULE_ASTRA_DEPTH_ENABLE
    previous_depth_gate = mod.SEARCH_REACQUIRE_DEPTH_GATE_ENABLE
    mod.MODULE_ASTRA_DEPTH_ENABLE = True
    mod.SEARCH_REACQUIRE_DEPTH_GATE_ENABLE = True
    try:
        if late_tracker._hold_for_confirmed_search_reacquire(
            late_candidate, width=640
        ):
            raise AssertionError(
                "2/2 late visual confirmation must release search even with invalid depth"
            )
    finally:
        mod.MODULE_ASTRA_DEPTH_ENABLE = previous_depth_enable
        mod.SEARCH_REACQUIRE_DEPTH_GATE_ENABLE = previous_depth_gate
    if late_tracker.search_state != "none" or late_tracker.search_direction is not None:
        raise AssertionError(
            "late visual confirmation must clear the frozen search direction"
        )
    if late_tracker._reacquire_depth_pending is not True:
        raise AssertionError(
            "invalid depth must remain an explicit longitudinal pending state: "
            f"pending={getattr(late_tracker, '_reacquire_depth_pending', 'missing')} "
            f"released={released} soft={late_tracker.soft_actions}"
        )
    if released != ["visual_reacquire_depth_pending"]:
        raise AssertionError(f"unexpected visual release reason: {released}")
    if late_tracker.soft_actions:
        raise AssertionError(
            "released late visual confirmation must not publish a search rotate action"
        )


def _assert_search_candidate_observation_uses_soft_stop() -> None:
    queued = []
    hard_stops = []

    class _EvidenceTracker:
        _publish_observation_soft_zero = mod.PersonTracker._publish_observation_soft_zero
        _apply_search_candidate_gate_decision = (
            mod.PersonTracker._apply_search_candidate_gate_decision
        )

        def __init__(self) -> None:
            self.frame_index = 176
            self._last_control_decision_reason = "stale_probe_primary_right"
            self._search_evidence_observation_active = False
            self._search_evidence_observation_source = "none"
            self._search_evidence_observation_deadline = 0.0
            self._follow_controller = SimpleNamespace(
                defer_search_timeout=lambda _seconds: None,
            )
            self._action_runtime = SimpleNamespace(
                send_stop_with_brake_hold=lambda reason: hard_stops.append(str(reason)),
            )
            self.is_forwarding = True

        def _clear_lateral_intent(self, reason: str) -> None:
            queued.append(("clear_lateral", str(reason)))

        def _clear_longitudinal_context(self, **_kwargs) -> None:
            queued.append(("clear_longitudinal",))

        def _replace_action_queue(self, actions, reason: str) -> None:
            queued.append(("replace", list(actions), str(reason)))

    tracker = _EvidenceTracker()
    tracker._apply_search_candidate_gate_decision(
        mod.SearchCandidateGateDecision(
            pause_rotation=True,
            entered=True,
            source="formal",
            reason="formal_candidate_observe_start",
            score=0.81,
            bbox=(500.0, 20.0, 639.0, 470.0),
            hold_frame=1,
            hold_frames=3,
        )
    )
    expected = (
        "replace",
        [mod.ACTION_STOP],
        "search_candidate_evidence_observe",
    )
    if hard_stops or expected not in queued:
        raise AssertionError(
            "candidate observation must publish soft STOP without brake hold: "
            f"hard={hard_stops} queued={queued}"
        )
    if not tracker._search_evidence_observation_active:
        raise AssertionError("candidate observation window was not armed")
    if not tracker._use_soft_stop_next:
        raise AssertionError("candidate observation must mark the queued STOP as soft")


def _assert_visible_unsteerable_target_holds_without_search() -> None:
    deferred = []
    integration = _DummyTracker()
    integration._follow_controller = SimpleNamespace(
        active_target_id=7,
        defer_search_timeout=lambda seconds: deferred.append(float(seconds)),
    )
    integration._hold_for_visible_unsteerable_target = types.MethodType(
        mod.PersonTracker._hold_for_visible_unsteerable_target,
        integration,
    )
    integration._finish_visible_unsteerable_hold = types.MethodType(
        mod.PersonTracker._finish_visible_unsteerable_hold,
        integration,
    )
    old_reid_enable = mod.VISION_REID_ENABLE
    mod.VISION_REID_ENABLE = True
    try:
        # Frame 1 establishes the geometry cache with the real UID.
        mod.PersonTracker._consume_track_records(
            integration,
            [_track(3, 7)],
            640,
            480,
            "test",
        )
        # Two detector gaps may create a new DeepSORT track with UID0. The
        # continuous oversized box must preserve UID7 only as unsteerable.
        integration.frame_index = 4
        integration.assignment = {
            "bbox_quality_ok": False,
            "bbox_quality_reason": "area_ratio>0.75,width_ratio>0.85",
        }
        mod.PersonTracker._consume_track_records(
            integration,
            [_track(5, 0)],
            640,
            480,
            "test",
        )
        integration.frame_index = 5
        integration.assignment = {}
        mod.PersonTracker._consume_track_records(
            integration,
            [_track(5, 7)],
            640,
            480,
            "test",
        )
        integration.frame_index = 6
        mod.PersonTracker._consume_track_records(
            integration,
            [_track(5, 7)],
            640,
            480,
            "test",
        )
    finally:
        mod.VISION_REID_ENABLE = old_reid_enable

    steerable_flags = [flag for _persons, flag in integration.queue_calls]
    if steerable_flags != [True, True, True, True]:
        raise AssertionError(
            "quality-rejected UID0 frames must be passed as an empty control set; "
            f"got flags={steerable_flags} calls={integration.queue_calls}"
        )
    if integration.queue_calls[1][0]:
        raise AssertionError(
            "quality-rejected frame must never inject a synthetic UID into control: "
            f"{integration.queue_calls}"
        )
    if integration._visible_unsteerable_uid is not None:
        raise AssertionError("geometry-only low-quality frames must not establish a hold identity")
    if [flag for _persons, flag in integration.published_contexts] != [True, True, True]:
        raise AssertionError(
            "30Hz Depth context must exclude low-quality holds and resume only on a complete box: "
            f"{integration.published_contexts}"
        )
    limited_caps = [
        options.get("target_steering_limit_rpm")
        for options in integration.queue_options[1:]
    ]
    if any(cap is not None for cap in limited_caps):
        raise AssertionError(f"quality-rejected targets must not receive a yaw cap: {limited_caps}")


def _assert_mapped_close_crop_uses_separate_hold_path() -> None:
    tracker = _DummyTracker(
        {
            "bbox_quality_ok": False,
            "bbox_quality_reason": "edge_touch>2",
            "mapped_uid": 7,
        }
    )
    tracker._follow_controller.active_target_id = 7
    tracker._hold_for_visible_unsteerable_target = types.MethodType(
        mod.PersonTracker._hold_for_visible_unsteerable_target,
        tracker,
    )
    tracker._finish_visible_unsteerable_hold = types.MethodType(
        mod.PersonTracker._finish_visible_unsteerable_hold,
        tracker,
    )
    old_reid_enable = mod.VISION_REID_ENABLE
    mod.VISION_REID_ENABLE = True
    try:
        mod.PersonTracker._consume_track_records(
            tracker,
            [_track(3, 0, bbox=(0.0, 0.0, 430.0, 479.0))],
            640,
            480,
            "test",
        )
    finally:
        mod.VISION_REID_ENABLE = old_reid_enable

    if len(tracker.queue_calls) != 1:
        raise AssertionError(f"mapped crop must make exactly one control decision: {tracker.queue_calls}")
    persons, steerable = tracker.queue_calls[0]
    if len(persons) != 1 or int(persons[0][1]) != 7 or steerable:
        raise AssertionError(f"mapped crop must use unsteerable UID hold: {tracker.queue_calls}")
    options = tracker.queue_options[0]
    if not options.get("low_quality_visible"):
        raise AssertionError(f"mapped crop must reset loss through low-quality state: {options}")
    yaw_limit = options.get("target_steering_limit_rpm")
    if yaw_limit is None or not (0.0 < float(yaw_limit) <= mod.VISIBLE_LOW_QUALITY_STEER_EDGE_MAX_CORRECTION_RPM):
        raise AssertionError(f"mapped edge crop must receive a bounded yaw limit: {options}")
    if tracker.published_contexts:
        raise AssertionError("mapped crop must never publish Depth longitudinal context")

    fragment = _DummyTracker(
        {
            "bbox_quality_ok": False,
            "bbox_quality_reason": "area<900,area_shrink<0.30",
            "mapped_uid": 7,
        }
    )
    fragment._follow_controller.active_target_id = 7
    mod.VISION_REID_ENABLE = True
    try:
        mod.PersonTracker._consume_track_records(
            fragment,
            [_track(4, 0, bbox=(600.0, 20.0, 620.0, 42.0))],
            640,
            480,
            "test",
        )
    finally:
        mod.VISION_REID_ENABLE = old_reid_enable
    if fragment.queued_persons:
        raise AssertionError(f"mapped fragment must stay outside all control: {fragment.queue_calls}")

def _assert_search_geometry_reacquires_unique_target() -> None:
    if not mod.SINGLE_PERSON_GEOMETRY_FALLBACK_ENABLE:
        tracker = _DummyTracker()
        tracker.search_state = "searching"
        tracker._follow_controller.active_target_id = 7
        tracker._follow_controller.search_state = "searching"
        result = tracker._search_geometry_reacquire_id(
            _track(9, 0, bbox=(190.0, 0.0, 639.0, 479.0)),
            {"bbox_quality_ok": True},
            person_count=1,
            width=640,
        )
        if result is not None:
            raise AssertionError(f"disabled geometry reacquire returned UID: {result}")
        return
    tracker = _DummyTracker(
        {
            "bbox_quality_ok": False,
            "bbox_quality_reason": "edge_touch>2",
            "mapped_uid": 0,
        }
    )
    tracker.search_state = "searching"
    tracker.search_direction = "right"
    tracker._follow_controller.active_target_id = 7
    tracker._follow_controller.search_state = "searching"
    tracker._follow_controller.search_direction = "right"
    candidate = _track(9, 0, bbox=(190.0, 0.0, 639.0, 479.0))
    first = tracker._search_geometry_reacquire_id(
        candidate,
        tracker.assignment,
        person_count=1,
        width=640,
    )
    tracker.frame_index = 2
    tracker.assignment = {
        "bbox_quality_ok": False,
        "bbox_quality_reason": "area_ratio>0.75,edge_touch>2",
        "mapped_uid": 0,
    }
    second = tracker._search_geometry_reacquire_id(
        _track(9, 0, bbox=(210.0, 0.0, 639.0, 479.0)),
        tracker.assignment,
        person_count=1,
        width=640,
    )
    if first is not None or second != 7:
        raise AssertionError(
            f"two direction-consistent search frames must recover UID7: {first}, {second}"
        )

    # The controller leaves search as soon as it consumes the recovered target.
    # The same DeepSORT track must keep the recovered UID on following frames.
    tracker.search_state = "none"
    tracker._follow_controller.search_state = "none"
    tracker.frame_index = 3
    held = tracker._search_geometry_reacquire_id(
        _track(9, 0, bbox=(220.0, 0.0, 639.0, 479.0)),
        tracker.assignment,
        person_count=1,
        width=640,
    )
    if held != 7:
        raise AssertionError(f"confirmed search track must retain UID7 after search ends: {held}")

    tracker.frame_index = 4
    fragment = tracker._search_geometry_reacquire_id(
        _track(10, 0, bbox=(600.0, 20.0, 620.0, 42.0)),
        {
            "bbox_quality_ok": False,
            "bbox_quality_reason": "area<900,area_shrink<0.30",
        },
        person_count=1,
        width=640,
    )
    if fragment is not None or tracker._search_geometry_reacquire_frames != 0:
        raise AssertionError("fragment boxes must never inherit the locked search UID")

    tracker.frame_index = 5
    thin_fragment = tracker._search_geometry_reacquire_id(
        _track(11, 0, bbox=(590.0, 20.0, 639.0, 300.0)),
        {
            "bbox_quality_ok": False,
            "bbox_quality_reason": "aspect<0.18",
        },
        person_count=1,
        width=640,
    )
    if thin_fragment is not None or tracker._search_geometry_reacquire_frames != 0:
        raise AssertionError("aspect-rejected boxes must never inherit the locked search UID")


def _assert_pid_zero_guard_brakes_unsettled_yaw() -> None:
    result = SimpleNamespace(
        correction_rpm=0,
        feedback_used=True,
        measured_yaw_rate_dps=27.7,
        desired_yaw_rate_dps=16.4,
        visual_error_deg=18.2,
        correction_limit_rpm=10.0,
        overspeed_brake_rpm=2.8,
        rate_p_rpm=-1.8,
        same_direction_overspeed_braking=False,
        output_floor_reason="none",
    )
    correction = mod.FollowSafetyController._pid_zero_guard_correction(
        result,
        current_x_ratio=0.803,
        center_left_ratio=0.40,
        center_right_ratio=0.60,
        min_correction_rpm=2,
    )
    if correction >= 0:
        raise AssertionError(
            f"right-yaw overspeed outside center must produce left braking, got {correction}"
        )

    settled = SimpleNamespace(
        correction_rpm=0,
        feedback_used=True,
        measured_yaw_rate_dps=0.5,
        desired_yaw_rate_dps=0.0,
        visual_error_deg=0.0,
        correction_limit_rpm=6.0,
        overspeed_brake_rpm=0.0,
        rate_p_rpm=0.0,
        same_direction_overspeed_braking=False,
        output_floor_reason="center_hold",
    )
    correction = mod.FollowSafetyController._pid_zero_guard_correction(
        settled,
        current_x_ratio=0.50,
        center_left_ratio=0.40,
        center_right_ratio=0.60,
        min_correction_rpm=2,
    )
    if correction != 0:
        raise AssertionError(f"settled centered target must remain zero, got {correction}")


def _assert_long_low_quality_never_recovers_uid() -> None:
    tracker = _DummyTracker()
    tracker._follow_controller.active_target_id = 7
    tracker._hold_for_visible_unsteerable_target = types.MethodType(
        mod.PersonTracker._hold_for_visible_unsteerable_target,
        tracker,
    )
    tracker._finish_visible_unsteerable_hold = types.MethodType(
        mod.PersonTracker._finish_visible_unsteerable_hold,
        tracker,
    )
    old_reid_enable = mod.VISION_REID_ENABLE
    mod.VISION_REID_ENABLE = True
    try:
        mod.PersonTracker._consume_track_records(
            tracker,
            [_track(3, 7, bbox=(110.0, 0.0, 585.0, 479.0))],
            640,
            480,
            "test",
        )
        for frame_index in range(2, 10):
            tracker.frame_index = frame_index
            tracker.assignment = {
                "bbox_quality_ok": False,
                "bbox_quality_reason": "area_ratio>0.75",
                "mapped_uid": 7 if frame_index < 6 else 0,
            }
            mod.PersonTracker._consume_track_records(
                tracker,
                [_track(3, 0, bbox=(120.0 + frame_index * 5.0, 0.0, 639.0, 479.0))],
                640,
                480,
                "test",
            )

        tracker.frame_index = 10
        tracker.assignment = {
            "bbox_quality_ok": False,
            "bbox_quality_reason": "edge_touch>2",
            "mapped_uid": 0,
        }
        mod.PersonTracker._consume_track_records(
            tracker,
            [_track(3, 0, bbox=(205.0, 0.0, 639.0, 479.0))],
            640,
            480,
            "test",
        )
        first_recovery = tracker.queue_calls[-1]
        tracker.frame_index = 11
        mod.PersonTracker._consume_track_records(
            tracker,
            [_track(3, 0, bbox=(220.0, 0.0, 639.0, 479.0))],
            640,
            480,
            "test",
        )
        second_recovery = tracker.queue_calls[-1]
    finally:
        mod.VISION_REID_ENABLE = old_reid_enable

    if first_recovery[0]:
        raise AssertionError(
            f"a low-quality UID0 frame must not recover UID7: {first_recovery}"
        )
    if second_recovery[0]:
        raise AssertionError(
            "continued low-quality frames must remain outside control until a complete box returns: "
            f"{second_recovery}"
        )


def main() -> int:
    initial_temporary = _consume([_track(2, 0)], reid_enable=True)
    if initial_temporary:
        raise AssertionError(
            "an initial UID0 box must wait for formal ReID confirmation and never steer: "
            f"{initial_temporary}"
        )

    observed_candidate = _consume(
        [_track(2, 0)],
        reid_enable=True,
        active_target_id=7,
    )
    if observed_candidate:
        raise AssertionError(
            "unconfirmed ReID observation must not publish a control target: "
            f"{observed_candidate}"
        )

    confirmed = _consume([_track(2, 7)], reid_enable=True)
    if len(confirmed) != 1 or int(confirmed[0][1]) != 7:
        raise AssertionError(f"confirmed ReID uid should be accepted, got {confirmed}")

    coarse_large_box = _consume(
        [_track(2, 7)],
        reid_enable=True,
        assignment={
            "bbox_quality_ok": False,
            "bbox_quality_reason": "area_ratio>0.65,width_ratio>0.80",
        },
    )
    if coarse_large_box:
        raise AssertionError(
            "a low-quality large/cropped box must not enter control: "
            f"{coarse_large_box}"
        )

    edge_only_box = _consume(
        [_track(2, 7)],
        reid_enable=True,
        assignment={
            "bbox_quality_ok": False,
            "bbox_quality_reason": "edge_touch>2",
        },
    )
    if edge_only_box:
        raise AssertionError(
            "edge-only boxes must not enter control: "
            f"{edge_only_box}"
        )

    fallback = _consume([_track(2, 0)], reid_enable=False)
    if len(fallback) != 1 or int(fallback[0][1]) != -3:
        raise AssertionError(f"non-ReID mode should keep raw-track fallback id, got {fallback}")

    # 已锁定 UID=7 后，DeepSORT 换 raw track 且 ReID 正在受控交接时，
    # 唯一且几何连续的人体框应继续交给 UID=7，不能让控制层看到人数=0。
    old_reid_enable = mod.VISION_REID_ENABLE
    mod.VISION_REID_ENABLE = True
    try:
        dummy = _DummyTracker()
        dummy._follow_controller.active_target_id = 7
        mod.PersonTracker._consume_track_records(dummy, [_track(2, 7)], 640, 480, "test")
        dummy.frame_index = 2
        dummy.assignment = {
            "bbox_quality_ok": False,
            "bbox_quality_reason": "edge_touch>2",
        }
        mod.PersonTracker._consume_track_records(dummy, [_track(3, 0)], 640, 480, "test")
        handoff_fallback = dummy.queued_persons
    finally:
        mod.VISION_REID_ENABLE = old_reid_enable
    if handoff_fallback:
        raise AssertionError(
            "edge-only handoff must stay outside normal UID control until a complete box returns: "
            f"{handoff_fallback}"
        )

    old_reid_enable = mod.VISION_REID_ENABLE
    mod.VISION_REID_ENABLE = True
    try:
        bad_dummy = _DummyTracker()
        bad_dummy._follow_controller.active_target_id = 7
        mod.PersonTracker._consume_track_records(bad_dummy, [_track(2, 7)], 640, 480, "test")
        bad_dummy.frame_index = 2
        bad_dummy.assignment = {
            "bbox_quality_ok": False,
            "bbox_quality_reason": "area_ratio>0.65,width_ratio>0.80",
        }
        mod.PersonTracker._consume_track_records(bad_dummy, [_track(3, 0)], 640, 480, "test")
        bad_handoff_fallback = bad_dummy.queued_persons
    finally:
        mod.VISION_REID_ENABLE = old_reid_enable
    if bad_handoff_fallback != []:
        raise AssertionError(f"distorted uid=0 bbox must never enter steering: {bad_handoff_fallback}")

    # A discontinuous UID0 box and any multi-person scene must not claim the
    # active UID, even as an unsteerable target.
    old_reid_enable = mod.VISION_REID_ENABLE
    mod.VISION_REID_ENABLE = True
    try:
        discontinuous = _DummyTracker()
        discontinuous._follow_controller.active_target_id = 7
        mod.PersonTracker._consume_track_records(
            discontinuous,
            [_track(2, 7)],
            640,
            480,
            "test",
        )
        discontinuous.frame_index = 2
        discontinuous.assignment = {
            "bbox_quality_ok": False,
            "bbox_quality_reason": "area_ratio>0.75,width_ratio>0.85",
        }
        mod.PersonTracker._consume_track_records(
            discontinuous,
            [_track(3, 0, bbox=(500.0, 20.0, 630.0, 220.0))],
            640,
            480,
            "test",
        )
        discontinuous.frame_index = 3
        mod.PersonTracker._consume_track_records(
            discontinuous,
            [_track(3, 0, bbox=(501.0, 20.0, 631.0, 220.0))],
            640,
            480,
            "test",
        )
        multi = _DummyTracker(discontinuous.assignment)
        multi._follow_controller.active_target_id = 7
        mod.PersonTracker._consume_track_records(
            multi,
            [_track(4, 0), _track(5, 0, bbox=(400.0, 20.0, 520.0, 220.0))],
            640,
            480,
            "test",
        )
    finally:
        mod.VISION_REID_ENABLE = old_reid_enable
    if discontinuous._single_person_geometry_unsteerable_uid is not None:
        raise AssertionError("discontinuous UID0 box must not claim the active UID")
    if multi._single_person_geometry_unsteerable_uid is not None:
        raise AssertionError("multi-person UID0 boxes must not claim the active UID")
    _assert_confirmed_search_reacquire_respects_configured_stability()
    _assert_search_candidate_observation_uses_soft_stop()
    _assert_search_geometry_reacquires_unique_target()
    _assert_pid_zero_guard_brakes_unsettled_yaw()
    _assert_long_low_quality_never_recovers_uid()
    _assert_visible_unsteerable_target_holds_without_search()
    _assert_mapped_close_crop_uses_separate_hold_path()
    _assert_near_camera_occlusion_gate()
    _assert_search_timeout_stops_before_runtime_exit()
    _assert_stale_result_revokes_published_yaw_without_brake()
    _assert_rotation_only_action_filter()
    _assert_active_target_track_uses_established_mapping_only()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
