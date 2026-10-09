"""A fresh measurement must not restart PI merely to test an older lease.

CPU-only controller/admission tests. Successful packet evidence is not depth
authority; real STOP/revocation and fresh-sample braking retain priority.
"""
import ast
import inspect
import textwrap
from dataclasses import replace
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.longitudinal_execution import ForwardExecutionAnchor
from test_depth_authority_250 import authority, advance, seed, decide_commit
from test_distance_pi_controller import configured
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


def bind_execution_reader(tracker):
    tree = ast.parse(textwrap.dedent(inspect.getsource(runtime.PersonTracker.__init__)))
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Attribute) and
                           t.attr == '_longitudinal_execution_reader' for t in n.targets)]
    assert len(assignments) == 1
    exec(compile(ast.Module(body=assignments, type_ignores=[]), __file__, 'exec'),
         {'self': tracker, 'PersonTracker': runtime.PersonTracker})


@pytest.mark.parametrize('available', [False, True])
def test_old_continuation_denied_new_fresh_depth_uses_completed_command(authority, available):
    a = authority
    stamp, grant = seed(a, distance=2.5, rpm=40.)
    deadline = a.owner._depth30_linear_timing.depth_expires_at
    advance(a, stamp+.181)
    # Simulate the real read-time continuation veto. No zero was sent and
    # the canonical UID/sample has not been revoked. Encoder is still lagging.
    a.owner._depth30_continuation_veto = (1, stamp)
    anchor = ForwardExecutionAnchor(1, stamp, 40., stamp+.130, object())
    a.owner._action_runtime.forward_execution_anchor = (
        lambda uid, sample, now: anchor if available else None)
    bind_execution_reader(a.owner)
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    current = a.frame(2.5, rpm=0., stamp=stamp+.129)
    a.controller.decide(10, current, longitudinal_only=True)
    result = a.controller.last_distance_pid_result
    if available:
        assert result.pi_status == 'tracking'
        assert 40 <= result.output_rpm <= 40+240*.051
    else:
        assert result.pi_status == 'recovering' and result.output_rpm == 0
    # The computation never revives the old lease or extends its timestamp.
    assert a.owner._depth30_linear_snapshot == grant
    assert a.owner._depth30_linear_timing.depth_expires_at == deadline
    assert a.owner._fresh_depth_linear_snapshot(1) is None


@pytest.mark.parametrize('event', ['expired', 'revoked', 'stale_new', 'duplicate', 'near'])
def test_completed_packet_cannot_override_real_recovery_or_new_braking(authority, event):
    a = authority
    stamp, _ = seed(a, distance=2.5, rpm=40.)
    advance(a, stamp+(.261 if event == 'expired' else .181))
    anchor = ForwardExecutionAnchor(1, stamp, 40., a.clock.now-.05, object())
    a.controller._longitudinal_execution_reader = lambda *args: anchor
    a.owner._depth30_continuation_veto = (1, stamp)
    if event == 'revoked': a.owner._revoke_depth_linear_authority('identity_lost')
    incoming = stamp if event in {'stale_new', 'duplicate'} else a.clock.now-.02
    current = a.frame(1.4 if event == 'near' else 2.5, rpm=0., stamp=incoming)
    decision, actions, accepted = decide_commit(a, current)
    assert not any(x.kind == 'forward' and x.speed_percent > 0 for x in actions)
    if event in {'expired', 'revoked'}:
        assert a.controller.last_distance_pid_result.pi_status == 'recovering'


def test_zero_receipt_arriving_during_pi_rejects_old_execution_proof(authority, monkeypatch):
    a = authority
    stamp, _ = seed(a, distance=2.5, rpm=40.)
    advance(a, stamp+.181)
    a.owner._depth30_continuation_veto = (1, stamp)
    evidence = [ForwardExecutionAnchor(1, stamp, 40., stamp+.13, object())]
    a.controller._longitudinal_execution_reader = lambda *args: evidence[0]
    original = a.controller._distance_pid.update
    def update(*args, **kwargs):
        result = original(*args, **kwargs)
        evidence[0] = None  # Executor has now sent zero/STOP.
        return result
    monkeypatch.setattr(a.controller._distance_pid, 'update', update)
    _, actions, _ = decide_commit(a, a.frame(2.5, rpm=0., stamp=stamp+.129))
    assert a.controller.last_distance_pid_result.output_rpm == 0
    assert not any(x.kind == 'forward' and x.speed_percent > 0 for x in actions)
    assert a.controller._distance_pid._distance_pi._execution_suspended


def test_packet_anchor_uses_send_time_not_unsent_ramp_budget():
    pi = DistancePiController(DistancePiConfig(physical_ttl_sec=.25))
    params = dict(target_distance_m=1.4, deadband_m=.03, max_output_rpm=200.,
                  rise_rpm_per_sec=240., ego_forward_rpm=60., range_rate_m_s=.5,
                  raw_closure_valid=True)
    pi.update(2.5, sample_timestamp=100., execution_now=100., **params)
    assert pi.retain_execution_anchor(100., 40., 100.15, 100.181)
    result = pi.update(2.5, sample_timestamp=100.13, execution_now=100.181,
                       **dict(params, ego_forward_rpm=0.))
    assert result.status == 'tracking'
    assert 40 <= result.output_rpm <= 40+240*.031


@pytest.mark.parametrize('change', ['suspended', 'parked', 'wrong_sample', 'old_write',
                                  'old_depth', 'future', 'zero', 'nan'])
def test_invalid_execution_evidence_never_restores_clock(change):
    pi = DistancePiController(DistancePiConfig(physical_ttl_sec=.25))
    pi.update(2.5, 1.4, sample_timestamp=100., execution_now=100., deadband_m=.03,
              max_output_rpm=200., ego_forward_rpm=40.)
    args = [100., 40., 100.15, 100.181]
    if change == 'suspended': pi.suspend(100.1, 'stop', reset_execution=True)
    if change == 'parked': pi.set_normal_parking(True)
    if change == 'wrong_sample': args[0] = 99.99
    if change == 'old_write': args[2] = 100.01
    if change == 'old_depth': args[3] = 100.251
    if change == 'future': args[2] = 100.2
    if change == 'zero': args[1] = 0.
    if change == 'nan': args[1] = float('nan')
    before = (pi._last_sample_ts, pi._last_execution_ts, pi.integral_m_s)
    assert not pi.retain_execution_anchor(*args)
    assert (pi._last_sample_ts, pi._last_execution_ts, pi.integral_m_s) == before
