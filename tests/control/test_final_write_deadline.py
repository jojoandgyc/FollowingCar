"""The final authority reader may itself outlive a physical motor lease.

Real tracker/commit/writer, fake clock and backend only; no hardware startup.
"""
from dataclasses import replace

import pytest

import request_0513_modular as runtime
from car_control_modular.detector_identity_lease import DetectorIdentityLease
from test_depth_authority_250 import writer
from test_relative_depth_continuation_runtime import (
    at_age, owner, pi_owner, publish, relative,
)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("delay,allowed", [(0., True), (.003, False)])
def test_log_in_last_grant_read_cannot_extend_physical_deadline(
        relative, monkeypatch, periodic, delay, allowed):
    stamp, _ = publish(relative)
    action, backend = writer(relative)
    action.config.follow_wheel_period_sec = .05 if periodic else 0.
    at_age(relative, stamp, .190)
    armed = False
    hits = []

    def hard_stop(_):
        nonlocal armed
        relative.clock.now = stamp+.249
        armed = True
        return False

    original_log = runtime.logger.info

    def delayed_log(fmt, *args, **kwargs):
        nonlocal armed
        if armed and fmt.startswith("depth_relative_continuation_cap"):
            relative.clock.now += delay
            armed = False
            hits.append(relative.clock.now)
        original_log(fmt, *args, **kwargs)

    monkeypatch.setattr(runtime.logger, "info", delayed_log)
    action.hard_stop_check = hard_stop
    if periodic:
        action._service_follow_wheels()
    else:
        with relative.owner.motor_io_lock:
            action._send_follow_wheel_targets(60, -60, "DEADLINE_TEST", visible_required=True)
    assert hits
    assert backend.pairs
    assert any(left > 0 and right < 0 for left, right, _ in backend.pairs) is allowed
    if not allowed:
        assert all((left, right) == (0, 0) for left, right, _ in backend.pairs)
    assert relative.owner._depth30_linear_timing.depth_expires_at == pytest.approx(stamp+.25)


@pytest.mark.parametrize("fault", ["depth", "shorter_timing", "feedforward", "revoke", "replace", "uid", "veto", "future"])
def test_pure_write_check_rejects_expired_or_replaced_grant(relative, fault):
    stamp, _ = publish(relative)
    action, _ = writer(relative)
    grant = at_age(relative, stamp, .10)
    timing = relative.owner._depth30_linear_timing
    if fault == "depth":
        relative.clock.now = stamp+.251
    elif fault == "shorter_timing":
        relative.owner._depth30_linear_timing = replace(timing, depth_expires_at=stamp+.09)
    elif fault == "feedforward":
        relative.owner._depth30_linear_timing = replace(
            timing, feedforward_expires_at=stamp+.09, distance_only_percent=5)
    elif fault == "revoke":
        relative.owner._depth30_linear_snapshot = None
    elif fault == "replace":
        relative.owner._depth30_linear_snapshot = ("forward", grant[1], 1, stamp+.01)
    elif fault == "uid":
        relative.owner._follow_controller.active_target_id = 2
    elif fault == "veto":
        relative.owner._depth30_continuation_veto = (1, stamp)
    else:
        relative.clock.now = stamp-.01
    assert not action._linear_packet_within_write_deadline(
        grant, 1, 60., feedback=relative.feedback)


def test_pure_write_check_allows_current_reduced_grant_and_zero(relative):
    stamp, _ = publish(relative)
    action, _ = writer(relative)
    grant = at_age(relative, stamp, .249)
    assert grant is not None
    assert action._linear_packet_within_write_deadline(
        grant, 1, 60., feedback=relative.feedback)
    relative.clock.now = stamp+.251
    assert action._linear_packet_within_write_deadline(None, 1, 0.)


def test_backward_keeps_shorter_physical_deadline(relative):
    stamp, _ = publish(relative)
    action, _ = writer(relative)
    grant = ("backward", 30, 1, stamp)
    relative.owner._depth30_linear_snapshot = grant
    relative.owner._depth30_linear_timing = None
    relative.clock.now = stamp+.179
    assert action._linear_packet_within_write_deadline(grant, 1, -60.)
    relative.clock.now = stamp+.181
    assert not action._linear_packet_within_write_deadline(grant, 1, -60.)


def test_final_forward_write_rechecks_feedback_after_callbacks(relative):
    stamp, _ = publish(relative)
    action, _ = writer(relative)
    grant = at_age(relative, stamp, .12)
    # This packet is younger than the 180ms Depth continuation boundary, so
    # the age-dependent cap is not what vetoes it. The earlier guard may have
    # seen the encoder at 149ms; the final physical write must reject it once
    # other callbacks have consumed another 2ms.
    timing = relative.owner._depth30_linear_timing
    relative.owner._depth30_linear_timing = replace(timing, continuation_motion=None)
    relative.feedback = replace(relative.feedback,
                                timestamp=relative.clock.now-.149)
    assert action._linear_packet_within_write_deadline(
        grant, 1, 60., feedback=relative.feedback)
    relative.feedback = replace(relative.feedback,
                                timestamp=relative.clock.now-.151)
    assert not action._linear_packet_within_write_deadline(
        grant, 1, 60., feedback=relative.feedback)


def test_final_write_rechecks_identity_after_cap_callback(relative):
    stamp, _ = publish(relative)
    action, _ = writer(relative)
    grant = at_age(relative, stamp, .19)
    relative.owner._detector_identity_lease = DetectorIdentityLease(
        1, 1, 1, stamp-.1, 2, stamp-.01, stamp+.195)
    original = relative.owner._depth_forward_continuation_limit

    def revoke(*args, **kwargs):
        result = original(*args, **kwargs)
        relative.owner._detector_identity_lease = False
        return result

    relative.owner._depth_forward_continuation_limit = revoke
    assert not action._linear_packet_within_write_deadline(
        grant, 1, 60., feedback=relative.feedback)
    assert action._follow_write_veto_reason == "terminal_authority_or_feedback_expired"
