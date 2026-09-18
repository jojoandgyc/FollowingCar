"""CAP965 regression: a 50ms duplicate must not cut a 28ms-old DRIVE packet."""
from dataclasses import replace
from types import SimpleNamespace
import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import HazardState, ObstacleState
from test_lateral_zero_runtime import owner, NOW, _intent
from test_longitudinal_authority_runtime import _commit, _frame
from test_distance_tracking_response import setup, decide


def replay(frame, stamp):
    return replace(frame, distance_state=replace(
        frame.distance_state, raw_distance_m=None, sample_timestamp=None,
        observation_timestamp=stamp, temporal_status="duplicate",
        source_detail="depth_sample_observation_discarded_fused_radar_hold_hold",
    ))


@pytest.mark.parametrize("kind", ["forward", "backward"])
@pytest.mark.parametrize("older", [0.0, .02])
def test_replay_preserves_exact_authority_without_queue_refresh(owner, kind, older, caplog):
    _commit(owner, kind=kind)
    _intent(owner)
    snapshot = owner._depth30_linear_snapshot
    revision = owner._lateral_yaw_revision
    frame = replay(_frame(), NOW-.04-older)
    for _ in range(5):
        assert owner._preserve_depth_replay(frame, 1)
    assert owner._depth30_linear_snapshot == snapshot
    assert owner._lateral_yaw_revision == revision
    assert owner._queued_calls == []
    assert "deadline_renewed=False pid_updated=False" in caplog.text


@pytest.mark.parametrize("failure", [
    "expired", "uid", "missing", "search", "hazard", "obstacle", "brake",
    "near_candidate", "pending", "newer", "unknown_time", "shutdown", "stop",
    "low_quality", "new_rejection",
])
def test_replay_never_bypasses_safety(owner, monkeypatch, failure):
    _commit(owner)
    frame = replay(_frame(), NOW-.04)
    uid = 1
    if failure == "expired":
        monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.15)
    elif failure == "uid":
        uid = 2
    elif failure == "missing":
        frame = replace(frame, persons=[])
    elif failure == "search":
        owner.search_state = "searching"
    elif failure == "hazard":
        frame = replace(frame, hazard=HazardState(active=True))
    elif failure == "obstacle":
        frame = replace(frame, obstacles=ObstacleState(front=True))
    elif failure == "shutdown":
        owner._runtime_shutdown_requested = True
    elif failure == "stop":
        owner._explicit_stop_requested = True
    elif failure == "low_quality":
        owner._vision_control_state = "target_visible_low_quality"
    else:
        changes = {
            "brake": {"brake_latched": True},
            "near_candidate": {"safety_distance_m": .3},
            "pending": {"temporal_status": "older_than_pending"},
            "newer": {"observation_timestamp": NOW-.01},
            "unknown_time": {"observation_timestamp": None},
            "new_rejection": {"temporal_status": "new_sample"},
        }[failure]
        frame = replace(frame, distance_state=replace(frame.distance_state, **changes))
    assert not owner._preserve_depth_replay(frame, uid)


def test_duplicate_does_not_reset_velocity_chain_or_authorize_hold_acceleration(setup):
    clock, controller, frame = setup
    decide(controller, frame(1.90, rpm=30))
    clock.now += .1
    decide(controller, frame(1.90, rpm=30))
    evidence = controller._longitudinal_motion_evidence
    stamp = clock.now
    clock.now += .03
    held = replay(frame(1.90, rpm=30), stamp)
    controller._observe_longitudinal_motion(held, held.persons[0])
    assert controller._longitudinal_motion_evidence is evidence
    assert controller._tracking_base_rpm(1.90, clock.now) is None
    clock.now += .05
    decide(controller, frame(1.90, rpm=30))
    assert controller._longitudinal_motion_evidence.sample_count >= 3
    assert controller.last_distance_pid_result.tracking_base_rpm == pytest.approx(30)


@pytest.mark.parametrize("unsafe", ["yaw", "expired", "obstacle"])
def test_replay_cannot_preserve_unsafe_velocity_evidence(setup, unsafe):
    clock, controller, frame = setup
    decide(controller, frame(1.90, rpm=30))
    stamp = clock.now
    clock.now += .2 if unsafe == "expired" else .03
    held = replay(frame(1.90, rpm=30, yaw=20 if unsafe == "yaw" else 0), stamp)
    if unsafe == "obstacle":
        held = replace(held, obstacles=ObstacleState(front=True))
    controller._observe_longitudinal_motion(held, held.persons[0])
    assert controller._longitudinal_motion_evidence is None


def test_real_depth_dispatch_skips_decide_and_queue_for_replay(owner):
    _commit(owner)
    original = owner._depth30_linear_snapshot
    target = _frame().persons[0]
    owner._get_obstacle_status = lambda: dict(front=False, left=False, right=False)
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: None)
    owner._distance_runtime = SimpleNamespace(
        select_target=lambda targets: targets[0],
        get_frame_distance_state=lambda *a, **k: replay(_frame(), NOW-.04).distance_state,
    )
    owner._current_hazard_state_for_controller = lambda: HazardState()
    owner._follow_controller.set_last_dispatched = lambda command: None
    owner._follow_controller.decide = lambda *a, **k: pytest.fail("duplicate reached PID")
    result = owner._process_detections_modular(
        640, 480, [(target.bbox, target.track_id, target.confidence, target.area)],
        control_source="depth30", depth_use_latest=True,
        depth_target_snapshot=(target,), evidence_capture_frame_id=576,
        evidence_capture_timestamp=NOW-.09,
    )
    assert result == []
    assert owner._depth30_linear_snapshot == original
    assert owner._queued_calls == []
