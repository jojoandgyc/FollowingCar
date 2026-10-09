"""Only transient lateral Depth reads may retain an unexpired forward lease."""

from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import request_0513_modular as runtime
from test_depth_authority_250 import advance, authority, decide_commit, seed
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import _intent, owner


def test_expired_yaw_keeps_canonical_grant_on_transient_read_gap(authority, monkeypatch):
    a = authority
    stamp, grant = seed(a, distance=2.5, rpm=40.)
    _intent(a.owner, valid_until=stamp+.14)
    advance(a, stamp+.21)
    assert a.owner._fresh_depth_linear_snapshot(1) is not None
    # A one-off metadata/read race is not an explicit braking veto. It may
    # cancel yaw without changing the hardware stop mode or renewing Depth.
    with monkeypatch.context() as patcher:
        patcher.setattr(runtime.PersonTracker, "_fresh_depth_linear_snapshot",
                          lambda *_args, **_kwargs: None)
        a.owner._service_lateral_intent(a.clock.now)
    assert a.owner._depth30_linear_snapshot == grant
    assert getattr(a.owner, "_depth30_continuation_veto", None) != (1, stamp)
    assert a.owner._queued_calls[-1][0] == (runtime.ACTION_STEER_RIGHT,)
    assert a.owner._current_forward_percent == 0
    assert a.owner._current_steer_base_percent == 0
    assert a.owner._current_steer_correction_rpm == 0
    assert a.owner._fresh_depth_linear_snapshot(1) is not None
    assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp+.25)
    advance(a, stamp+.23)
    current = a.frame(2.55, rpm=40., stamp=stamp+.22)
    _decision, actions, accepted = decide_commit(a, current)
    assert accepted and any(action.kind == "forward" and action.speed_percent > 0
                            for action in actions)
    assert a.owner._depth30_linear_snapshot[3] == current.distance_state.sample_timestamp


def test_zero_braking_cap_revokes_grant_and_uses_stop(authority, monkeypatch):
    a = authority
    stamp, _grant = seed(a, distance=2.5, rpm=40.)
    _intent(a.owner, valid_until=stamp+.14)
    advance(a, stamp+.21)
    monkeypatch.setattr(runtime.PersonTracker, "_depth_forward_continuation_limit",
                        lambda *_args, **_kwargs: (0, "braking_margin"))
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    a.owner._service_lateral_intent(a.clock.now)
    assert a.owner._depth30_linear_snapshot is None
    assert a.owner._depth30_continuation_veto == (1, stamp)
    assert a.owner._queued_calls[-1][0] == (runtime.ACTION_STOP,)
    assert a.owner._current_forward_percent == 0
    assert a.owner._current_steer_base_percent == 0
    assert a.owner._fresh_depth_linear_snapshot(1) is None


def test_active_yaw_transient_read_gap_cancels_yaw_without_revoking_depth(authority, monkeypatch):
    a = authority
    stamp, grant = seed(a, distance=2.5, rpm=40.)
    _intent(a.owner, mode="forward", valid_until=stamp+.24)
    advance(a, stamp+.21)
    assert a.owner._fresh_depth_linear_snapshot(1) is not None
    with monkeypatch.context() as patcher:
        patcher.setattr(runtime.PersonTracker, "_fresh_depth_linear_snapshot",
                          lambda *_args, **_kwargs: None)
        a.owner._service_lateral_intent(a.clock.now)
    assert a.owner._depth30_linear_snapshot == grant
    assert a.owner._queued_calls[-1][0] == (runtime.ACTION_STEER_RIGHT,)
    assert a.owner._current_forward_percent == 0
    assert a.owner._current_steer_correction_rpm == 0


def test_active_yaw_with_zero_cap_does_not_enter_pending_steer(authority, monkeypatch):
    a = authority
    stamp, _grant = seed(a, distance=2.5, rpm=40.)
    _intent(a.owner, mode="forward", valid_until=stamp+.24)
    advance(a, stamp+.21)
    monkeypatch.setattr(runtime.PersonTracker, "_depth_forward_continuation_limit",
                        lambda *_args, **_kwargs: (0, "braking_margin"))
    a.owner._service_lateral_intent(a.clock.now)
    assert a.owner._depth30_linear_snapshot is None
    assert a.owner._depth30_continuation_veto == (1, stamp)
    # This is the pre-pending active-yaw path; it must not become a STEER hold.
    assert a.owner._queued_calls[-1][0] == (runtime.ACTION_ROTATE_RIGHT,)


def test_existing_same_grant_veto_keeps_stop_on_expired_yaw(authority):
    a = authority
    stamp, _grant = seed(a, distance=2.5, rpm=40.)
    _intent(a.owner, valid_until=stamp+.14)
    advance(a, stamp+.21)
    a.owner._depth30_continuation_veto = (1, stamp)
    a.owner._service_lateral_intent(a.clock.now)
    assert a.owner._depth30_linear_snapshot is None
    assert a.owner._queued_calls[-1][0] == (runtime.ACTION_STOP,)


def test_expired_feedforward_with_zero_distance_cap_keeps_stop(authority):
    a = authority
    stamp, _grant = seed(a, distance=2.5, rpm=40.)
    a.owner._depth30_linear_timing = replace(
        a.owner._depth30_linear_timing,
        feedforward_expires_at=stamp+.19, distance_only_percent=0,
    )
    _intent(a.owner, valid_until=stamp+.14)
    advance(a, stamp+.21)
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    assert getattr(a.owner, "_depth30_continuation_veto", None) != (1, stamp)
    assert a.owner._pending_forward_depth_lease(1, a.clock.now) is None
    a.owner._service_lateral_intent(a.clock.now)
    assert a.owner._depth30_linear_snapshot is None
    assert a.owner._queued_calls[-1][0] == (runtime.ACTION_STOP,)


@pytest.mark.parametrize("audit_kind", ["cap", "continuation"])
def test_same_grant_zero_cap_audit_keeps_stop_during_read_race(
    authority, monkeypatch, audit_kind,
):
    a = authority
    stamp, grant = seed(a, distance=2.5, rpm=40.)
    _intent(a.owner, valid_until=stamp+.14)
    advance(a, stamp+.21)
    if audit_kind == "cap":
        a.owner._last_depth_continuation_cap_audit = (grant, "braking_margin", 0)
    else:
        a.owner._last_depth_continuation_audit = (grant, False, "braking_margin", 0)
    with monkeypatch.context() as patcher:
        patcher.setattr(runtime.PersonTracker, "_fresh_depth_linear_snapshot",
                          lambda *_args, **_kwargs: None)
        a.owner._service_lateral_intent(a.clock.now)
    assert a.owner._depth30_linear_snapshot is None
    assert a.owner._queued_calls[-1][0] == (runtime.ACTION_STOP,)


@pytest.mark.parametrize("cause", ["expired", "uid", "brake", "near_park"])
def test_true_expiry_and_safety_still_revoke_canonical_grant(authority, cause):
    a = authority
    stamp, _grant = seed(a, distance=2.5, rpm=40.)
    intent = _intent(a.owner)
    advance(a, stamp + (.251 if cause == "expired" else .21))
    a.owner._depth30_continuation_veto = (1, stamp)
    if cause == "uid":
        a.controller.active_target_id = 2
    elif cause == "brake":
        a.owner._brake_hold_active = True
    elif cause == "near_park":
        a.owner._near_yaw_park_request = object()
    a.owner._publish_lateral_zero(intent, "expired")
    if cause == "uid":
        # Cancellation from the former target cannot publish STOP into the
        # replacement UID's generation. The motor's actual grant reader still
        # denies this old tuple; no stale translation is preserved for use.
        assert a.owner._fresh_depth_linear_snapshot(1, quiet=True) is None
        assert not a.owner._queued_calls
        return
    assert a.owner._depth30_linear_snapshot is None
    assert a.owner._queued_calls[-1][0] == (runtime.ACTION_STOP,)
