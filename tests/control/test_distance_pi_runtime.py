"""PI is authorized by physical depth, never a diagnostic feedforward lease."""
from types import SimpleNamespace
from dataclasses import replace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction, ControlDecision, SteeringFeedback
from test_lateral_zero_runtime import NOW, owner
from test_longitudinal_authority_runtime import _frame


@pytest.fixture
def pi_owner(owner, monkeypatch):
    monkeypatch.setattr(runtime, "FORWARD_MAX_RPM", 200)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT", 100)
    monkeypatch.setattr(runtime, "MOTOR_RS485_TARGET_MIN_INTERVAL_SEC", .05)
    monkeypatch.setattr(runtime.PersonTracker, "_depth_longitudinal_cap_percent", staticmethod(lambda _: 12))
    c = owner._follow_controller
    c.distance_pi_enabled = True
    c._longitudinal_motion_evidence = SimpleNamespace(status="transient_bridge")
    c._longitudinal_bridge = SimpleNamespace(origin=SimpleNamespace(sample_timestamp=NOW-10))
    c.distance_only_forward_percent = lambda frame, stamp: 40
    c._tracking_base_rpm = lambda *args: pytest.fail("PI must not consult FF matching speed")
    c.fresh_bridge_forward_percent = lambda *args: pytest.fail("PI must not consult FF bridge authority")
    c.fresh_distance_recovery_percent = lambda *args: pytest.fail("PI must not re-enter legacy recovery")
    owner.approvals = []
    owner.rejections = []
    owner.suspensions = []
    c.accept_longitudinal_limit = lambda stamp, rpm: owner.approvals.append((stamp, rpm))
    c.reject_longitudinal_sample = lambda stamp, reason: owner.rejections.append((stamp, reason))
    c.suspend_longitudinal_authority = lambda now, reason: owner.suspensions.append((now, reason))
    return owner


def commit(owner, *, stamp=NOW-.02, percent=40, distance=1.5, fresh=True, kind="forward"):
    c = owner._follow_controller
    c._distance_pid_last_sample_timestamp = stamp
    c.last_distance_pid_result = SimpleNamespace(approach_mode="distance_pi", output_rpm=percent*2)
    return owner._commit_depth_linear_decision(
        ControlDecision(actions=[getattr(ControlAction, kind)(percent, "distance_pi")], reason="distance_pi"),
        _frame(stamp=stamp, distance=distance), 1, is_fresh_depth=fresh,
    )


@pytest.mark.parametrize("prior", [None, ("forward", 12, 1, NOW-.19), ("forward", 12, 1, NOW-.04)])
def test_pi_uses_current_distance_budget_not_old_24rpm_or_ff_lease(pi_owner, prior):
    pi_owner._depth30_linear_snapshot = prior
    actions, accepted = commit(pi_owner)
    assert accepted and actions[0].speed_percent == 40
    timing = pi_owner._depth30_linear_timing
    assert timing.feedforward_timestamp is None and timing.feedforward_expires_at is None
    assert timing.depth_expires_at == pytest.approx(NOW-.02+.18)
    assert pi_owner.approvals == [(NOW-.02, 80.)]
    assert not pi_owner.rejections


def test_pi_target_band_can_keep_integral_speed_with_final_total_cap(pi_owner, monkeypatch):
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT", 30)
    actions, accepted = commit(pi_owner, distance=runtime.TARGET_DISTANCE)
    assert accepted and actions[0].speed_percent == 30
    assert pi_owner.approvals == [(NOW-.02, 60.)]


def test_pi_expiration_and_replay_cannot_extend_physical_180ms(pi_owner, monkeypatch):
    commit(pi_owner)
    before = pi_owner._depth30_linear_timing
    assert commit(pi_owner) == ([], False)
    assert pi_owner._depth30_linear_timing is before
    assert len(pi_owner.approvals) == 1 and not pi_owner.rejections
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.161)
    assert pi_owner._fresh_depth_linear_snapshot(1) is None
    assert commit(pi_owner) == ([], False)
    assert len(pi_owner.approvals) == 1
    assert not pi_owner.rejections  # Expired replay cannot undo accepted I.


def test_pi_nonfresh_hold_can_reduce_but_not_accelerate_or_renew(pi_owner):
    commit(pi_owner)
    timing = pi_owner._depth30_linear_timing
    actions, accepted = commit(pi_owner, percent=70, fresh=False, distance=None)
    assert accepted and actions[0].speed_percent == 40
    assert pi_owner._depth30_linear_timing.depth_expires_at == timing.depth_expires_at
    actions, accepted = commit(pi_owner, percent=25, fresh=False, distance=None)
    assert accepted and actions[0].speed_percent == 25
    assert pi_owner._depth30_linear_timing.depth_expires_at == timing.depth_expires_at
    assert len(pi_owner.approvals) == 1 and not pi_owner.rejections


@pytest.mark.parametrize("case", ["stop", "shutdown", "search", "lost", "quality", "uid", "rotation", "brake"])
def test_pi_new_sample_cannot_bypass_runtime_safety(pi_owner, monkeypatch, case):
    if case == "stop": pi_owner._explicit_stop_requested = True
    if case == "shutdown": pi_owner._runtime_shutdown_requested = True
    if case == "search": pi_owner.search_state = "searching"
    if case == "lost": pi_owner._vision_control_state = "target_lost"
    if case == "quality": pi_owner._vision_control_state = "target_visible_low_quality"
    if case == "uid": pi_owner._follow_controller.active_target_id = 2
    if case == "rotation": monkeypatch.setattr(runtime, "FOLLOW_ROTATION_ONLY", True)
    if case == "brake": pi_owner._brake_hold_active = True
    actions, _ = commit(pi_owner)
    assert actions[0].speed_percent == 0
    assert pi_owner._fresh_depth_linear_snapshot(1) is None
    assert pi_owner.approvals == []
    assert pi_owner.rejections == [(NOW-.02, "depth_admission_unqualified")]


def test_pi_invalid_evidence_rejects_sample_without_zero_antiwindup(pi_owner):
    pi_owner._follow_controller.distance_only_forward_percent = lambda frame, stamp: None
    actions, _ = commit(pi_owner)
    assert actions[0].speed_percent == 0
    assert not pi_owner.approvals
    assert pi_owner.rejections == [(NOW-.02, "depth_admission_unqualified")]


def test_pi_valid_braking_zero_is_written_back(pi_owner):
    pi_owner._follow_controller.distance_only_forward_percent = lambda frame, stamp: 0
    actions, _ = commit(pi_owner, percent=0)
    assert actions[0].speed_percent == 0
    assert pi_owner.approvals == [(NOW-.02, 0)]
    assert not pi_owner.rejections


def test_pi_dispatch_budget_rejection_does_not_backcalculate_unauthorized_zero(pi_owner):
    assert commit(pi_owner, stamp=NOW-.14) == ([], False)
    assert pi_owner._depth30_linear_snapshot is None
    assert not pi_owner.approvals
    assert pi_owner.rejections == [(NOW-.14, "depth_dispatch_budget")]


def test_pi_dispatch_budget_safety_reduction_keeps_old_deadline(pi_owner):
    commit(pi_owner, stamp=NOW-.16, percent=40)  # Too late for a new grant.
    pi_owner._depth30_linear_snapshot = ("forward", 40, 1, NOW-.16)
    actions, accepted = commit(pi_owner, stamp=NOW-.14, percent=20)
    assert accepted and actions[0].speed_percent == 20
    assert pi_owner._depth30_linear_snapshot == ("forward", 20, 1, NOW-.16)
    assert pi_owner.approvals == [(NOW-.14, 40)]
    assert pi_owner._depth30_linear_timing.depth_expires_at == pytest.approx(NOW+.02)


def test_pi_late_direction_change_revokes_without_zero_antiwindup(pi_owner):
    pi_owner._depth30_linear_snapshot = ("forward", 40, 1, NOW-.16)
    actions, accepted = commit(pi_owner, stamp=NOW-.14, kind="backward", percent=10, distance=.8)
    assert accepted and actions[0].kind == "forward" and actions[0].speed_percent == 0
    assert pi_owner._depth30_linear_snapshot is None
    assert not pi_owner.approvals
    assert not pi_owner.rejections  # Legacy reverse is not a PI integration.


@pytest.mark.parametrize("stamp", [NOW-.181, NOW+.01, float("nan")])
def test_pi_invalid_physical_sample_never_grants_authority(pi_owner, stamp):
    assert commit(pi_owner, stamp=stamp) == ([], False)
    assert not pi_owner.approvals
    assert len(pi_owner.rejections) == 1


def test_pi_repeated_authority_revoke_does_not_erase_integral(pi_owner):
    commit(pi_owner)
    before = list(pi_owner.approvals)
    pi_owner._revoke_depth_linear_authority("expired")
    pi_owner._revoke_depth_linear_authority("expired")
    assert pi_owner.approvals == before and not pi_owner.rejections
    assert pi_owner._depth30_linear_snapshot is None
    assert pi_owner.suspensions == [(NOW, "expired")]


def test_pi_revoke_without_existing_motor_grant_does_not_suspend_execution(pi_owner):
    pi_owner._revoke_depth_linear_authority("no_existing_grant")
    assert not pi_owner.suspensions
    assert not pi_owner.approvals and not pi_owner.rejections


def test_pi_early_revoke_preserves_memory_but_resumes_from_measured_speed(pi_owner, monkeypatch):
    from car_control_modular.controllers import FollowSafetyController
    from car_control_modular.distance_pi import DistancePiConfig, DistancePiController

    pi = DistancePiController(DistancePiConfig())
    pi.integral_m_s = .3
    limits = dict(target_distance_m=1.5, deadband_m=.03, max_output_rpm=200.,
                  rise_rpm_per_sec=240., fall_rpm_per_sec=300.,
                  range_rate_m_s=0., raw_closure_valid=True)
    initial = pi.update(2., sample_timestamp=NOW-.02, execution_now=NOW,
                        ego_forward_rpm=60., **limits)
    assert initial.output_rpm > 0
    before_integral = pi.integral_m_s
    pi_owner._follow_controller.distance_only_forward_percent = lambda frame, stamp: int(pi.last_result.output_rpm/2)
    pi_owner._follow_controller._distance_pid = pi
    pi_owner._follow_controller.suspend_longitudinal_authority = (
        lambda now, reason: FollowSafetyController.suspend_longitudinal_authority(
            pi_owner._follow_controller, now, reason)
    )
    commit(pi_owner, percent=int(initial.output_rpm/2), distance=2.)
    watermark = pi_owner._depth30_linear_sample_watermark
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.01)
    pi_owner._revoke_depth_linear_authority("stale_vision_result")
    assert pi.integral_m_s == before_integral
    assert pi._last_sample_ts == NOW-.02
    assert pi_owner._depth30_linear_sample_watermark == watermark
    assert pi_owner._depth30_linear_snapshot is None
    # The old physical deadline is still live, but its grant was explicitly
    # withdrawn. A new trusted sample must not resume the former 60RPM ramp.
    resumed = pi.update(2., sample_timestamp=NOW+.02, execution_now=NOW+.03,
                        ego_forward_rpm=0., **limits)
    assert resumed.status == "recovering" and resumed.sample_dt_sec == 0
    assert resumed.output_rpm == 0
    assert resumed.integral_m_s == before_integral


def test_pi_does_not_expand_reverse_ceiling(pi_owner):
    actions, accepted = commit(pi_owner, kind="backward", percent=80, distance=.8)
    expected = runtime.PersonTracker._cap_depth_longitudinal_actions(
        [ControlAction.backward(80, "old")], distance_m=.8,
    )
    assert accepted and actions[0].speed_percent == expected[0].speed_percent == 12
    assert not pi_owner.approvals


def test_pi_mode_keeps_legacy_reverse_over_consecutive_samples(pi_owner, monkeypatch):
    pi_owner._follow_controller.distance_only_forward_percent = lambda *args: pytest.fail("Reverse is not forward PI")
    for offset in (0., .05, .1):
        monkeypatch.setattr(runtime.time, "monotonic", lambda offset=offset: NOW+offset)
        stamp = NOW+offset-.02
        actions, accepted = commit(pi_owner, stamp=stamp, kind="backward", percent=30, distance=.8)
        assert accepted and actions[0].kind == "backward" and actions[0].speed_percent == 12
        assert pi_owner._fresh_depth_linear_snapshot(1) == ("backward", 12, 1, stamp)
        assert pi_owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp+.18)
        assert pi_owner._depth30_linear_timing.feedforward_expires_at is None
    assert not pi_owner.approvals and not pi_owner.rejections


@pytest.fixture
def extended_pi_owner(pi_owner, monkeypatch):
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC", .25)
    monkeypatch.setattr(runtime, "VISION_MMWAVE_FUSION_ENCODER_WHEEL_CIRCUMFERENCE_M", .816814)
    c = pi_owner._follow_controller
    c.distance_only_forward_percent = lambda frame, stamp: 20
    stamp = NOW-.02
    c._distance_pid_last_sample_timestamp = stamp
    c.last_distance_pid_result = SimpleNamespace(approach_mode="distance_pi", output_rpm=40)
    frame = replace(_frame(stamp, distance=3.), steering_feedback=SteeringFeedback(
        timestamp=NOW, trustworthy=True, left_forward_rpm=40., right_forward_rpm=40., yaw_rate_right_dps=0.))
    actions, accepted = pi_owner._commit_depth_linear_decision(
        ControlDecision(actions=[ControlAction.forward(20, "distance_pi")], reason="distance_pi"),
        frame, 1, is_fresh_depth=True)
    assert accepted and actions[0].speed_percent == 20
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.19)
    pi_owner._last_vision_control_ts = NOW+.18
    return pi_owner


@pytest.mark.parametrize("requested,expected", [(None, 20), (40, 20), (10, 10)])
def test_late_depth_can_only_hold_or_reduce_original_grant(extended_pi_owner, requested, expected):
    tracker = extended_pi_owner
    before = tracker._depth30_linear_timing
    watermark = tracker._depth30_linear_sample_watermark
    approval = list(tracker.approvals)
    decision = ControlDecision(actions=[] if requested is None else [ControlAction.forward(requested, "late")])
    actions, accepted = tracker._commit_depth_linear_decision(
        decision, _frame(NOW-.005, distance=2.9), 1, is_fresh_depth=True)
    assert accepted and actions[0].speed_percent == expected
    assert tracker._depth30_linear_snapshot == ("forward", expected, 1, NOW-.02)
    assert tracker._depth30_linear_timing.depth_expires_at == before.depth_expires_at
    assert tracker._depth30_linear_timing.accepted_depth_timestamp == before.accepted_depth_timestamp
    assert tracker._depth30_linear_sample_watermark == watermark
    assert tracker.approvals == approval and not tracker.rejections


def test_late_depth_does_not_restart_revoked_grant(extended_pi_owner):
    tracker = extended_pi_owner
    watermark = tracker._depth30_linear_sample_watermark
    tracker._revoke_depth_linear_authority("identity_conflict")
    assert tracker._commit_depth_linear_decision(
        ControlDecision(actions=[ControlAction.forward(40, "late")]),
        _frame(NOW-.005, distance=3.), 1, is_fresh_depth=True) == ([], False)
    assert tracker._depth30_linear_sample_watermark == watermark
    assert tracker._depth30_linear_snapshot is None


def test_late_new_close_depth_withdraws_original_grant(extended_pi_owner):
    tracker = extended_pi_owner
    actions, accepted = tracker._commit_depth_linear_decision(
        ControlDecision(), _frame(NOW-.005, distance=1.52), 1, is_fresh_depth=True)
    assert accepted and actions[0].speed_percent == 0
    assert tracker._depth30_linear_snapshot is None


@pytest.mark.parametrize("stop", ["explicit", "soft", "shutdown", "action"])
def test_late_depth_cannot_turn_a_stop_decision_into_continuation(extended_pi_owner, stop):
    tracker = extended_pi_owner
    decision = ControlDecision(
        explicit_stop_requested=stop == "explicit", soft_stop_requested=stop == "soft",
        shutdown_requested=stop == "shutdown",
        actions=[ControlAction.stop("stop")] if stop == "action" else [],
    )
    actions, accepted = tracker._commit_depth_linear_decision(
        decision, _frame(NOW-.005, distance=3.), 1, is_fresh_depth=True)
    assert accepted and actions[0].speed_percent == 0
    assert tracker._depth30_linear_snapshot is None


@pytest.mark.parametrize("field,value", [("_explicit_stop_requested", True),
                                        ("search_state", "searching"),
                                        ("_vision_control_state", "target_lost")])
def test_late_depth_owner_veto_permanently_revokes_old_grant(extended_pi_owner, field, value):
    tracker = extended_pi_owner
    original = getattr(tracker, field)
    setattr(tracker, field, value)
    assert tracker._commit_depth_linear_decision(
        ControlDecision(), _frame(NOW-.005, distance=3.), 1, is_fresh_depth=True) == ([], False)
    assert tracker._depth30_linear_snapshot is None
    setattr(tracker, field, original)
    assert tracker._fresh_depth_linear_snapshot(1) is None


def test_250ms_read_requires_cached_braking_margin_and_current_visibility(extended_pi_owner):
    tracker = extended_pi_owner
    assert tracker._fresh_depth_linear_snapshot(1)[1] == 20
    timing = tracker._depth30_linear_timing
    tracker._depth30_linear_timing = replace(timing, continuation_distance_m=1.7)
    assert tracker._fresh_depth_linear_snapshot(1) is None
    tracker._depth30_linear_timing = timing
    tracker._last_vision_control_ts = NOW-.1
    assert tracker._fresh_depth_linear_snapshot(1) is None


def test_denied_continuation_cannot_revive_after_rgb_refresh_without_new_depth(extended_pi_owner):
    tracker = extended_pi_owner
    snapshot = tracker._depth30_linear_snapshot
    tracker._last_vision_control_ts = NOW-.1
    assert tracker._fresh_depth_linear_snapshot(1) is None
    assert tracker._depth30_linear_snapshot is snapshot  # Read thread does not clear control state.
    tracker._last_vision_control_ts = NOW+.19
    assert tracker._fresh_depth_linear_snapshot(1) is None
    assert tracker._depth30_continuation_veto == (1, NOW-.02)
