"""Production constructor binding with real PI/snapshot code, no hardware init."""
import ast
import inspect
import textwrap

import pytest

import request_0513_modular as runtime
from test_depth_authority_250 import authority, advance, seed, decide_commit
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


def bind_production_reader(tracker):
    # Execute only the actual assignment, not PersonTracker.__init__ (which
    # opens hardware). Do not recreate its lambda with a different signature.
    tree = ast.parse(textwrap.dedent(inspect.getsource(runtime.PersonTracker.__init__)))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Attribute)
                           and target.attr == "_live_longitudinal_authority_reader"
                           for target in node.targets)]
    assert len(assignments) == 1
    binding = ast.Module(body=assignments, type_ignores=[])
    exec(compile(binding, inspect.getfile(runtime.PersonTracker), "exec"),
         {"self": tracker, "PersonTracker": runtime.PersonTracker})


@pytest.mark.parametrize("visual", [False, True], ids=["depth_fast_loop", "visual_loop"])
@pytest.mark.parametrize("state", ["live", "integration_gap", "expired", "revoked"])
def test_real_binding_handles_next_positive_pi_sample(authority, state, visual):
    a = authority
    bind_production_reader(a.owner)
    stamp, original = seed(a, distance=2.5, rpm=40.)
    deadline = a.owner._depth30_linear_timing.depth_expires_at
    gap = {"live": .1, "integration_gap": .186, "expired": .26, "revoked": .1}[state]
    advance(a, stamp + gap)
    if state == "revoked":
        a.owner._revoke_depth_linear_authority("test_binding_revoke")
    live = a.controller._live_longitudinal_authority_reader(1)
    assert (live is not None) == (state in {"live", "integration_gap"})
    a.controller.decide(11, a.frame(2.5, rpm=27.5), longitudinal_only=not visual)
    result = a.controller.last_distance_pid_result
    assert result.output_rpm > 0  # Not an exception hidden as zero control.
    if state in {"expired", "revoked"}:
        assert result.pi_status == "recovering"
        assert result.output_rpm <= 27.5
    else:
        assert result.pi_status == ("tracking" if state == "live" else "tracking_gap_no_integral")
        assert result.output_rpm >= 40
    # Computing a new request alone must not renew the old motor grant.
    if state != "revoked":
        assert a.owner._depth30_linear_snapshot == original
        assert a.owner._depth30_linear_timing.depth_expires_at == deadline


def test_real_binding_cold_start_reaches_positive_runtime_grant(authority):
    a = authority
    bind_production_reader(a.owner)
    initial = a.clock.now
    approved = []
    for index in range(4):
        advance(a, initial + index * .05)
        _, actions, _ = decide_commit(a, a.frame(1.8, rpm=0.))
        approved.append(max((x.speed_percent for x in actions if x.kind == "forward"), default=0))
    assert approved[0] == 0
    assert all(speed > 0 for speed in approved[1:])
    grant = a.controller._live_longitudinal_authority_reader(1)
    assert grant is not None and grant[0] == "forward" and grant[1] > 0
    assert grant[3] == pytest.approx(a.clock.now)
