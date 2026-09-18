"""CAP316/322/335/345: accepted depth, FF origin and motor TTL are distinct."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import request_0513_modular as runtime
from test_lateral_zero_runtime import owner, NOW, _intent
from test_longitudinal_authority_runtime import _commit, _frame
from test_depth_replay_continuity import replay
from test_distance_tracking_response import setup, decide


def bridge(owner, percent=41, fallback=25, origin=NOW-.14):
    owner._depth30_linear_snapshot = ("forward", percent, 1, origin)
    owner._follow_controller._longitudinal_motion_evidence = SimpleNamespace(status="transient_bridge")
    owner._follow_controller._longitudinal_bridge = SimpleNamespace(origin=SimpleNamespace(sample_timestamp=origin))
    owner._follow_controller.distance_only_forward_percent = lambda frame, stamp: fallback


def test_350ms_estimate_memory_is_not_350ms_depth_permission(owner, monkeypatch):
    bridge(owner, origin=NOW-.24)
    owner._follow_controller.longitudinal_prior_max_age_sec = .35
    owner._depth30_linear_snapshot = ('forward', 41, 1, NOW-.05)
    actions, accepted = _commit(owner, stamp=NOW-.01, percent=35)
    assert accepted and actions[0].speed_percent == 35
    timing=owner._depth30_linear_timing
    assert timing.feedforward_expires_at == pytest.approx(NOW+.11)
    assert timing.depth_expires_at == pytest.approx(NOW+.17)
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: NOW+.12)
    assert owner._fresh_depth_linear_snapshot(1)[1] == 25
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: NOW+.171)
    assert owner._fresh_depth_linear_snapshot(1) is None


def test_retained_prior_cannot_restart_expired_motor_grant(owner):
    bridge(owner, origin=NOW-.24)
    owner._follow_controller.longitudinal_prior_max_age_sec = .35
    actions, _ = _commit(owner, stamp=NOW-.01, percent=35)
    assert not any(a.kind=='forward' and a.speed_percent>0 for a in actions)
    assert owner._depth30_linear_snapshot is None


@pytest.mark.parametrize("origin_age,new_age", [(.14,.02), (.11,.007), (.16,.021), (.12,.031)])
def test_four_logged_cap_shapes_preserve_duplicate_without_renewal(owner, origin_age, new_age):
    bridge(owner, origin=NOW-origin_age)
    actions, accepted = _commit(owner, stamp=NOW-new_age, percent=29)
    assert accepted and actions[0].speed_percent == 29
    snapshot, timing, revision = owner._depth30_linear_snapshot, owner._depth30_linear_timing, owner._lateral_yaw_revision
    held = replay(_frame(), NOW-new_age)
    for _ in range(5):
        assert owner._preserve_depth_replay(held, 1)
    assert owner._depth30_linear_snapshot is snapshot
    assert owner._depth30_linear_timing is timing
    assert owner._lateral_yaw_revision == revision
    assert timing.accepted_depth_timestamp == NOW-new_age
    assert timing.feedforward_timestamp == NOW-origin_age
    assert timing.depth_expires_at == NOW-new_age+.18


def test_expired_ff_downgrades_same_depth_without_pid_update_or_acceleration(owner, monkeypatch, caplog):
    bridge(owner)
    _commit(owner, stamp=NOW-.02, percent=40)
    timing = owner._depth30_linear_timing
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.05)
    assert owner._fresh_depth_linear_snapshot(1)[1] == 25
    assert owner._preserve_depth_replay(replay(_frame(), NOW-.02), 1)
    assert owner._depth30_linear_timing is timing
    assert "depth_ff_fallback" in caplog.text
    assert "pid_updated=False deadline_renewed=False" in caplog.text


def test_ff_expiring_between_decision_and_commit_uses_new_depth_fallback(owner, caplog):
    bridge(owner, origin=NOW-.19)
    # Existing translation is backed by a newer Depth, although FF is old.
    owner._depth30_linear_snapshot = ("forward", 40, 1, NOW-.05)
    actions, accepted = _commit(owner, stamp=NOW-.02, percent=35)
    assert accepted and actions[0].speed_percent == 25
    assert owner._fresh_depth_linear_snapshot(1)[1] == 25
    assert owner._depth30_linear_timing.feedforward_timestamp is None
    assert "depth_ff_admission_fallback" in caplog.text


@pytest.mark.parametrize("case", ["stop", "uid", "search", "depth_expired", "zero", "revoked", "near"])
def test_fallback_never_bypasses_revocation_or_missing_authority(owner, monkeypatch, case):
    bridge(owner, fallback=0 if case == "near" else 25)
    _commit(owner, stamp=NOW-.02, percent=40)
    if case == "stop": owner._explicit_stop_requested = True
    if case == "uid": owner._follow_controller.active_target_id = 2
    if case == "search": owner.search_state = "searching"
    if case == "zero": _commit(owner, stamp=NOW-.01, percent=0)
    if case == "revoked": owner._revoke_depth_linear_authority("test")
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+(.17 if case == "depth_expired" else .05))
    assert owner._fresh_depth_linear_snapshot(1) is None
    assert not owner._preserve_depth_replay(replay(_frame(), NOW-.02), 1)


def test_watermark_for_deferred_sample_is_not_accepted_provenance(owner):
    _commit(owner, stamp=NOW-.17, percent=20)  # rejected for insufficient write budget
    assert owner._depth30_linear_snapshot is None
    assert not owner._preserve_depth_replay(replay(_frame(), NOW-.17), 1)


def test_actual_short_depth_lease_retains_older_deadline_but_records_accepted_sample(owner, monkeypatch):
    _commit(owner, stamp=NOW-.04, percent=40)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.105)
    assert _commit(owner, stamp=NOW-.039, percent=10)[1]
    assert owner._depth30_linear_snapshot[3] == NOW-.04
    assert owner._depth30_linear_timing.accepted_depth_timestamp == NOW-.039
    assert owner._preserve_depth_replay(replay(_frame(), NOW-.039), 1)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.141)
    assert owner._fresh_depth_linear_snapshot(1) is None


def test_fallback_bound_to_exact_snapshot_cannot_leak_to_newer_normal_control(owner, monkeypatch):
    bridge(owner)
    _commit(owner, stamp=NOW-.02, percent=40)
    owner._follow_controller._longitudinal_motion_evidence = None
    _commit(owner, stamp=NOW-.01, percent=35)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.05)
    assert owner._fresh_depth_linear_snapshot(1)[1] == 35
    assert owner._depth30_linear_timing.feedforward_timestamp is None


def test_nonfresh_reduction_carries_clocks_without_extending_either(owner, monkeypatch):
    bridge(owner, fallback=25)
    _commit(owner, stamp=NOW-.02, percent=40)
    original = owner._depth30_linear_timing
    assert _commit(owner, percent=30, fresh=False)[1]
    current = owner._depth30_linear_timing
    assert current.depth_expires_at == original.depth_expires_at
    assert current.feedforward_expires_at == original.feedforward_expires_at
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.05)
    assert owner._fresh_depth_linear_snapshot(1)[1] == 25


def test_distance_only_component_does_not_integrate_again_and_obeys_start_band(setup):
    clock, controller, frame = setup
    decide(controller, frame(1.9, rpm=30))
    clock.now += .1
    current = frame(1.9, rpm=30)
    decide(controller, current)
    before = controller._distance_pid._integral_m_s, controller.last_distance_pid_result
    values = [controller.distance_only_forward_percent(current, clock.now) for _ in range(5)]
    assert 0 < values[0] <= controller.last_distance_pid_result.output_rpm
    assert len(set(values)) == 1
    assert (controller._distance_pid._integral_m_s, controller.last_distance_pid_result) == before
    assert controller.distance_only_forward_percent(frame(1.5), clock.now) == 0
    assert controller.distance_only_forward_percent(current, clock.now-.02) == 0


@pytest.mark.parametrize("changes", [dict(safety_distance_m=.3), dict(brake_latched=True),
                                    dict(source_detail="distance_jump_pending")])
def test_fallback_not_cached_for_unsafe_distance(setup, changes):
    clock, controller, frame = setup
    current = frame(1.9)
    decide(controller, current)
    current = replace(current, distance_state=replace(current.distance_state, **changes))
    assert controller.distance_only_forward_percent(current, clock.now) == 0


def test_motor_write_rechecks_ff_deadline_and_preserves_yaw(owner, monkeypatch):
    from test_yaw_zero_regression import make_runtime
    bridge(owner)
    _commit(owner, stamp=NOW-.02, percent=40)
    _intent(owner)
    motor, _, backend, _ = make_runtime()
    motor.owner = owner
    motor._visible_wheel_control_active = lambda: True
    owner._has_fresh_lateral_yaw = lambda uid: True
    # Isolate clock validation from physical zero-cross dynamics.
    motor._visible_wheel_guard = SimpleNamespace(reset=lambda: None,
        limit=lambda request, feedback, now, *, allow_forward_handoff=False: (request, "continuous"),
        note_sent=lambda *args: None)
    motor.get_steering_feedback = lambda: None
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.05)
    sent = []
    backend.send_targets = lambda left, right, label, **kw: sent.append((left, right))
    ls = backend.wheel_raw_state_to_target("left", 1, 0x01)
    rs = backend.wheel_raw_state_to_target("right", 1, 0x01)
    motor._send_follow_wheel_targets(45*ls, 35*rs, "STEER", visible_required=True)
    normalized = sent[-1][0]*ls, sent[-1][1]*rs
    # Expired FF must not write the old base40; depth-only base25 retains yaw5.
    assert normalized == (30, 20)
