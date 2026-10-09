"""Production recovery callback: actual constructor binding, no hardware init."""
import ast
import inspect
import textwrap
from dataclasses import replace

import pytest

import request_0513_modular as runtime
from car_control_modular.detector_identity_lease import DetectorIdentityLease
from test_lateral_zero_runtime import NOW, owner
from test_visual_depth_publication import full_result


def bind_restart_reader(tracker):
    tree = ast.parse(textwrap.dedent(inspect.getsource(runtime.PersonTracker.__init__)))
    bindings = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Attribute)
                        and target.attr == "_longitudinal_restart_identity_reader"
                        for target in node.targets)]
    assert len(bindings) == 1
    exec(compile(ast.Module(body=bindings, type_ignores=[]),
                 inspect.getfile(runtime.PersonTracker), "exec"),
         {"self": tracker, "PersonTracker": runtime.PersonTracker})


@pytest.fixture
def ready(owner):
    full_result(owner)
    owner._follow_controller.search_state = "none"
    owner._depth_roi_safety_clear = True
    # A withdrawn old grant must not be a prerequisite for a NEW restart.
    owner._depth30_linear_snapshot = None
    owner._fresh_depth_linear_snapshot = lambda *args, **kwargs: pytest.fail("old grant read")
    owner._get_obstacle_status = lambda: pytest.fail("no hardware/counter reads")
    bind_restart_reader(owner)
    return owner


def test_production_callback_allows_current_identity_not_control_completion(ready):
    o = ready
    o._last_vision_control_ts = NOW-10.
    o._vision_control_state = "direction_uncertain"
    before = o._validated_visual_observation
    assert o._follow_controller._longitudinal_restart_identity_reader(
        1, NOW-.119, now=NOW) is True
    assert o._validated_visual_observation is before
    assert o._depth30_linear_snapshot is None and not o._queued_calls


@pytest.mark.parametrize("fault", [
    "missing", "rejected", "proof_uid", "active_uid", "expired", "not_yet_valid",
    "grey_same_sample", "grey_other_sample", "fast_invalid", "fast_expired",
    "search", "controller_search", "not_running", "safety_unknown", "safety_blocked",
    "stop", "shutdown", "brake", "reacquire", "park", "controller_park",
    "sample_future", "sample_nan", "sample_bool", "sample_zero", "now_nan",
    "now_bool", "uid_bool",
])
def test_production_callback_fails_closed_and_does_not_mutate(ready, fault):
    o, uid, sample, now = ready, 1, NOW-.119, NOW
    proof = o._validated_visual_observation
    if fault == "missing": o._validated_visual_observation = None
    elif fault == "rejected": o._validated_visual_observation = False
    elif fault == "proof_uid": o._validated_visual_observation = replace(proof, uid=2)
    elif fault == "active_uid": o._follow_controller.active_target_id = 2
    elif fault == "expired": o._validated_visual_observation = replace(proof, expires_at=now)
    elif fault == "not_yet_valid": o._validated_visual_observation = replace(proof, validated_at=now+.01)
    elif fault.startswith("grey_"):
        o._validated_visual_observation = replace(proof,
            continuation_sample_timestamp=sample if fault == "grey_same_sample" else sample-.01)
    elif fault == "fast_invalid": o._detector_identity_lease = False
    elif fault == "fast_expired":
        o._detector_identity_lease = DetectorIdentityLease(1, 1, 200, now-.6, 256, now-.16, now)
    elif fault == "search": o.search_state = "searching"
    elif fault == "controller_search": o._follow_controller.search_state = "searching"
    elif fault == "not_running": o.running = False
    elif fault == "safety_unknown": del o._depth_roi_safety_clear
    elif fault == "safety_blocked": o._depth_roi_safety_clear = False
    elif fault in {"stop", "shutdown", "brake", "reacquire"}:
        setattr(o, {"stop": "_explicit_stop_requested", "shutdown": "_runtime_shutdown_requested",
                    "brake": "_brake_hold_active", "reacquire": "_reacquire_depth_pending"}[fault], True)
    elif fault == "park": o._near_yaw_park_request = object()
    elif fault == "controller_park": o._follow_controller._normal_parking_uid = 1
    elif fault == "sample_future": sample = now+.001
    elif fault == "sample_nan": sample = float("nan")
    elif fault == "sample_bool": sample = True
    elif fault == "sample_zero": sample = 0.
    elif fault == "now_nan": now = float("nan")
    elif fault == "now_bool": now = True
    elif fault == "uid_bool": uid = True
    before = o._validated_visual_observation
    assert o._follow_controller._longitudinal_restart_identity_reader(uid, sample, now) is False
    assert o._validated_visual_observation is before
    assert o._depth30_linear_snapshot is None and not o._queued_calls


def test_recheck_cannot_mint_restart_but_next_full_identity_can(ready):
    o = ready
    o._validated_visual_observation = False
    assert not o._follow_controller._longitudinal_restart_identity_reader(1, NOW-.10, NOW)
    full_result(o, cap=257, stamp=NOW-.14)
    assert o._follow_controller._longitudinal_restart_identity_reader(1, NOW-.10, NOW)
    deadline = o._validated_visual_observation.expires_at
    assert not o._follow_controller._longitudinal_restart_identity_reader(1, deadline-.01, deadline)


@pytest.mark.parametrize("reason", ["stale_vision_result", "lateral_depth:continuation_feedback_invalid"])
def test_real_binding_and_new_distance_can_restart_after_visual_withdrawal(
        authority, setup, monkeypatch, reason):
    from test_fresh_distance_restart_progress import prepare
    from test_depth_authority_250 import decide_commit
    a = authority
    frame, old = prepare(a, setup, monkeypatch, reason=reason,
                         distance=1.9207109, pair=(0., 0.), gap=.319, age=.119)
    full_result(a.owner, cap=222, stamp=a.clock.now-.12, now=a.clock.now)
    a.owner._depth_roi_safety_clear = True
    bind_restart_reader(a.owner)
    _, actions, accepted = decide_commit(a, frame)
    assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
    result = a.controller.last_distance_pid_result
    assert result.pi_fresh_grant_recovery_used and 0 < result.output_rpm <= 12.
    assert a.owner._depth30_linear_snapshot[3] != old[3]
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(
        frame.distance_state.sample_timestamp+.30)


# Import fixtures explicitly: tests/ uses no global hardware-owning conftest.
from test_depth_authority_250 import authority  # noqa: E402,F401
from test_distance_tracking_response import setup  # noqa: E402,F401
