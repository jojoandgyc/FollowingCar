"""CAP609/650: accepted search range reaches the paired writer without rescan.

Real search gate, Astra filters/fusion, runtime queue, paired controller and
motor writer; only acquisition and the serial driver are in memory.
"""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import queue
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import request_0513_modular as runtime
from car_control_modular.astra_depth import AstraDepthConfig, AstraDepthRuntime
from car_control_modular.control_types import DepthTargetObservation, SteeringFeedback
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.depth_async_scheduler import DepthAsyncScheduler
from car_control_modular.detector_identity_lease import ValidatedVisualObservation
from car_control_modular.sensor_modules import SensorRuntime
from test_depth_raw_geometry_runtime import make_runtime as make_distance
from test_short_follow_adapter import paired, owner, NOW

sys.path.append(str(Path(__file__).resolve().parents[1] / "motor"))
from test_depth_drive_rpm import make_runtime as make_motor


@pytest.fixture
def handoff(paired, monkeypatch):
    a = paired
    obj = a.owner
    clock = [NOW]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runtime, "SEARCH_REACQUIRE_DEPTH_GATE_ENABLE", True)
    monkeypatch.setattr(runtime, "SEARCH_CONFIRMED_REACQUIRE_FRAMES", 2)
    monkeypatch.setattr(runtime, "SEARCH_REACQUIRE_DEPTH_CONFIRM_FRAMES", 2)
    obj._follow_controller = FollowSafetyController(FollowPolicyConfig())
    obj._follow_controller.active_target_id = 1
    obj.search_state = obj._follow_controller.search_state = "searching"
    obj.search_direction = obj._follow_controller.search_direction = "right"
    obj._follow_controller.decide = lambda *_a, **_kw: pytest.fail("legacy normal controller")
    obj._reset_confirmed_search_reacquire()
    obj._reacquire_depth_pending = False
    obj._longitudinal_context_lock = threading.Lock()
    obj._longitudinal_context = None
    obj._longitudinal_wake_event = threading.Event()
    obj._depth_async_scheduler = DepthAsyncScheduler()
    obj._longitudinal_thread = SimpleNamespace(is_alive=lambda: False)
    obj._publish_search_reacquire_direction_hold = lambda *_: True
    obj._start_visual_reacquire_hold = lambda *_a, **_kw: None
    obj.action_queue = queue.Queue()

    motor, _, driver, _ = make_motor(max_rpm=200)
    motor.owner = obj
    obj.motor_io_lock = motor.backend.io_lock
    obj._action_runtime = motor
    motor.get_steering_feedback = lambda: SteeringFeedback(timestamp=clock[0], trustworthy=True)
    camera = AstraDepthRuntime(AstraDepthConfig())
    camera._np = np
    camera._latest_depth = np.full((480, 640), 1800, dtype=np.uint16)
    sensors = object.__new__(SensorRuntime)
    sensors.config = SimpleNamespace(astra_depth_enable=True)
    sensors.astra_depth = camera
    distance, _ = make_distance(vision_depth_detector_bbox_max_age_sec=.25)
    distance.owner, distance.sensor_runtime = obj, sensors
    obj._distance_runtime = distance
    a.target = replace(a.target, bbox=(390., 40., 550., 440.))
    obj._persons_to_targets = lambda *_a, **_kw: [a.target]
    monkeypatch.setattr(runtime, "resolve_depth_target_observation",
                        lambda **_kw: a.target.depth_observation)
    return SimpleNamespace(a=a, obj=obj, motor=motor, driver=driver,
                           camera=camera, distance=distance, clock=clock)


def capture(h, cap):
    obj = h.obj
    obj.frame_index += 1
    obj._active_capture_frame_id = cap
    obj._active_capture_timestamp = h.clock[0] - .04
    obs = DepthTargetObservation(h.a.target.bbox, 1, 3, cap, obj._active_capture_timestamp)
    h.a.target = replace(h.a.target, depth_observation=obs)
    obj._validated_visual_observation = ValidatedVisualObservation(
        1, 3, cap, obs.capture_timestamp, h.clock[0]-.01,
        obs.capture_timestamp+.5, "full")
    h.camera._latest_depth_ts = obs.capture_timestamp
    h.camera._depth_history.append((obs.capture_timestamp, h.camera._latest_depth))
    return dict(stable_id=1, bbox=h.a.target.bbox, score=.9, area=h.a.target.area,
        rec=SimpleNamespace(track_id=3, time_since_update=0),
        debug={"assignment": dict(uid=1, reason="strong", distance=.2,
                                  reacquire_geometry_ok=True, bbox_quality_ok=True)})


def accept_two(h, cap=609):
    first = capture(h, cap-1)
    assert h.obj._hold_for_confirmed_search_reacquire([first], width=640, height=480)
    assert h.obj._confirmed_search_reacquire_depth_streak == 1
    assert not h.obj._short_follow.snapshot().active
    h.clock[0] += .07
    second = capture(h, cap)
    assert not h.obj._hold_for_confirmed_search_reacquire([second], width=640, height=480)
    assert not h.obj._reacquire_depth_pending
    assert h.obj.search_state == h.obj._follow_controller.search_state == "none"
    return h.obj._search_reacquire_depth_transfer


def queue_capture(h):
    target = h.a.target
    return h.obj._queue_actions_for_persons(640, 480,
        [(target.bbox, target.track_id, target.confidence, target.area)])


@pytest.mark.parametrize("cap", [609, 650])
def test_accepted_two_of_two_becomes_first_pair_without_intermediate_stop(handoff, monkeypatch, caplog, cap):
    h = handoff
    caplog.set_level("INFO", logger=runtime.logger.name)
    transfer = accept_two(h, cap)
    assert transfer.distance_state.sample_timestamp == h.camera._last_accepted_ts
    with pytest.raises(FrozenInstanceError):
        transfer.distance_state.sample_timestamp = h.clock[0]
    monkeypatch.setattr(h.distance, "get_frame_distance_state",
                        lambda *_a, **_kw: pytest.fail("accepted search sample reread"))
    monkeypatch.setattr(h.obj._depth_async_scheduler, "begin_fallback",
                        lambda **_kw: pytest.fail("handoff reserved a second scan"))
    assert queue_capture(h)
    plan = h.obj._short_follow.snapshot().plan
    assert plan is not None and plan.moving
    assert plan.capture_id == cap and plan.uid == 1
    assert plan.capture_timestamp == transfer.capture_timestamp
    assert plan.depth_timestamp == transfer.distance_state.sample_timestamp
    assert plan.expires_at == pytest.approx(min(plan.depth_timestamp+.3, plan.capture_timestamp+.5))
    assert h.obj._last_frame_distance_state is transfer.distance_state
    assert h.obj._search_reacquire_depth_transfer is None
    h.motor._service_short_follow()
    assert h.driver.pairs == [(plan.left_rpm, -plan.right_rpm)]
    assert not h.driver.stops and not h.obj._queued_calls
    assert "streak=2/2" in caplog.text and "result=resume_lateral" in caplog.text
    assert "consumed_once=True" in caplog.text


@pytest.mark.parametrize("change", ["expired_depth", "uid", "identity_rejected", "epoch",
                                   "capture", "raw_track", "geometry", "pending", "unsteerable", "low_quality",
                                   "no_proof", "invalid_proof", "wrong_proof", "no_stamp", "nan_stamp"])
def test_transfer_is_not_admitted_after_its_evidence_changes(handoff, change):
    h = handoff
    transfer = accept_two(h)
    if change == "expired_depth": h.clock[0] = transfer.distance_state.sample_timestamp+.301
    elif change == "uid": h.obj._follow_controller.active_target_id = 2
    elif change == "identity_rejected": h.obj._validated_visual_observation = False
    elif change == "epoch": h.obj._depth_async_scheduler.revoke("identity_reset")
    elif change == "capture": h.obj._active_capture_frame_id += 1
    elif change == "raw_track":
        h.a.target = replace(h.a.target, depth_observation=replace(h.a.target.depth_observation, raw_track_id=4))
    elif change == "geometry": h.a.target = replace(h.a.target, bbox=(380., 40., 550., 440.))
    elif change == "pending": h.obj._reacquire_depth_pending = True
    elif change == "no_proof": h.obj._validated_visual_observation = None
    elif change == "invalid_proof": h.obj._validated_visual_observation = object()
    elif change == "wrong_proof":
        h.obj._validated_visual_observation = replace(h.obj._validated_visual_observation, uid=2)
    elif change in {"no_stamp", "nan_stamp"}:
        h.obj._search_reacquire_depth_transfer = replace(transfer, distance_state=replace(
            transfer.distance_state, sample_timestamp=None if change == "no_stamp" else float("nan")))
    assert h.obj._take_search_reacquire_depth(640, 480, [h.a.target],
        h.obj._active_capture_frame_id, h.obj._active_capture_timestamp,
        target_steerable=change != "unsteerable", low_quality_visible=change == "low_quality") is None
    assert h.obj._search_reacquire_depth_transfer is None
    assert not h.obj._short_follow.snapshot().active
    assert not h.driver.pairs


def test_same_physical_sample_after_handoff_preserves_plan_and_integral(handoff):
    h = handoff
    # Leave headroom for a real, positive integral increment on the next
    # independent sample, so duplicate rejection is not merely preserving zero.
    h.obj._short_follow.config = replace(h.obj._short_follow.config, kp_per_sec=.5)
    accept_two(h)
    assert queue_capture(h)
    h.clock[0] += .05
    capture(h, 610)
    assert queue_capture(h)
    before = h.obj._short_follow.snapshot()
    before_integral = h.obj._short_follow._integral_m_s
    assert before_integral > 0 and before.plan.integral_dt_sec == pytest.approx(.05)
    # A later normal attempt really reaches Astra and returns duplicate, with
    # no fresh sample_timestamp. It cannot integrate or redate the first pair.
    h.clock[0] += .05
    assert queue_capture(h)
    duplicate = h.obj._last_frame_distance_state
    assert duplicate.temporal_status == "duplicate" and duplicate.sample_timestamp is None
    assert duplicate.observation_timestamp == before.plan.depth_timestamp
    assert h.obj._short_follow.snapshot() is before
    assert h.obj._short_follow._integral_m_s == before_integral
    h.motor._service_short_follow()
    assert h.driver.pairs and not h.driver.stops


def test_writer_cannot_observe_empty_mailbox_during_valid_activation(handoff, monkeypatch):
    h = handoff
    accept_two(h)
    activating, release, writer_entered, writer_done = (threading.Event() for _ in range(4))
    failures = []
    original_activate = h.obj._short_follow.activate

    def activate(*args):
        result = original_activate(*args)
        activating.set()
        assert release.wait(2)
        return result

    def produce():
        try:
            assert queue_capture(h)
        except BaseException as exc:
            failures.append(exc)

    def write():
        try:
            writer_entered.set()
            h.motor._service_short_follow()
        except BaseException as exc:
            failures.append(exc)
        finally:
            writer_done.set()

    monkeypatch.setattr(h.obj._short_follow, "activate", activate)
    producer, writer = threading.Thread(target=produce), threading.Thread(target=write)
    producer.start()
    try:
        assert activating.wait(2)
        writer.start()
        assert writer_entered.wait(2)
        assert not writer_done.wait(.05)
        assert not h.driver.stops
    finally:
        release.set()
        producer.join(2)
        if writer.ident is not None:
            writer.join(2)
    assert not producer.is_alive() and not writer.is_alive() and not failures
    assert h.driver.pairs and not h.driver.stops


def test_queue_retirement_does_not_hold_paired_mailbox(handoff, monkeypatch):
    h = handoff
    accept_two(h)
    retiring = threading.Event()
    failures = []
    original = h.obj._short_follow_adapter._retire_legacy_normal

    def retire():
        retiring.set()
        original()

    def produce():
        try:
            assert queue_capture(h)
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(h.obj._short_follow_adapter, "_retire_legacy_normal", retire)
    producer = threading.Thread(target=produce)
    h.obj.action_queue_lock.acquire()
    producer.start()
    try:
        assert retiring.wait(2)
        # The queue owner must be able to inspect the paired mailbox before
        # releasing its queue lock; publication cannot invert those locks.
        acquired = h.obj._short_follow._lock.acquire(timeout=.1)
        assert acquired
        h.obj._short_follow._lock.release()
    finally:
        h.obj.action_queue_lock.release()
        producer.join(2)
    assert not producer.is_alive() and not failures
    h.motor._service_short_follow()
    assert h.driver.pairs and not h.driver.stops
