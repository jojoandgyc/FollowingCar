"""Longer measured visibility may support new Depth, never stale yaw or ROI."""
from dataclasses import replace

import pytest

import request_0513_modular as runtime
from car_control_modular.detector_identity_lease import (
    DetectorIdentityLease, MAX_VALIDATED_VISIBILITY_AGE_SEC,
    validated_visual_observation,
)
from car_control_modular.visual_steering_evidence import MAX_CAPTURE_AGE_SEC
from test_distance_only_runtime_authority import distance_only, publish
from test_distance_pi_runtime import pi_owner
from test_lateral_zero_runtime import owner, NOW, _intent
from test_visual_depth_publication import full_result, recheck_result


def proof(*, age=.1, window=.5, ceiling=.5, lease=None):
    return validated_visual_observation(
        uid=1, track_id=1, capture=256, timestamp=NOW-age, now=NOW,
        visibility_window=window, capture_max_age_sec=ceiling,
        identity_lease=lease)


@pytest.mark.parametrize("age", [.001, .10, .18, .34])
def test_500_visibility_uses_capture_deadline_not_validation_completion(age):
    p = proof(age=age)
    assert p.expires_at == pytest.approx(p.timestamp+.5)
    assert p.live(1, p.timestamp+.499)
    assert not p.live(1, p.timestamp+.501)
    assert p.permits_depth(1, p.timestamp+.48, p.timestamp+.499)
    assert not p.live(2, p.timestamp+.499)


@pytest.mark.parametrize("ceiling", [.18, .25, .35, .50])
def test_explicit_shorter_capture_and_visibility_budget_is_never_extended(ceiling):
    p = proof(age=.01, window=ceiling, ceiling=ceiling)
    assert p.expires_at == pytest.approx(p.timestamp+ceiling)
    assert not p.live(1, p.timestamp+ceiling+.001)


@pytest.mark.parametrize("age", [.350001, .40, .499, .501])
def test_new_late_result_cannot_mint_500ms_proof(age):
    assert proof(age=age) is None


def test_default_factory_preserves_legacy_350ms_capture_ceiling():
    p = validated_visual_observation(uid=1, track_id=1, capture=256,
        timestamp=NOW-.1, now=NOW, visibility_window=.5)
    assert p.expires_at == pytest.approx(p.timestamp+.35)
    assert MAX_VALIDATED_VISIBILITY_AGE_SEC == .5
    assert proof(ceiling=50., window=50.).expires_at == pytest.approx(NOW+.4)


@pytest.mark.parametrize("value", [0., -.1, float("nan"), float("inf"), True, None])
def test_invalid_visibility_configuration_does_not_authorize(value):
    assert proof(ceiling=value) is None
    assert proof(window=value) is None


def test_shorter_expired_config_cannot_return_dead_record():
    assert proof(age=.2, ceiling=.18) is None


def test_detector_visibility_never_outlives_its_original_identity_proof():
    lease = DetectorIdentityLease(1, 1, 250, NOW-.25, 256, NOW-.1, NOW+.05)
    p = proof(lease=lease)
    assert p.expires_at == lease.expires_at
    assert p.live(1, NOW+.049)
    assert not p.live(1, NOW+.051)


def test_production_adapter_publishes_500ms_measured_visibility(owner):
    full_result(owner, stamp=NOW-.1)
    p = owner._validated_visual_observation
    assert p.expires_at == pytest.approx(p.timestamp+.5)
    assert p.live(1, p.timestamp+.499)
    assert not p.live(1, p.timestamp+.501)


@pytest.mark.parametrize("ceiling", [.18, .25, .35, .50])
def test_production_config_preserves_shorter_visibility_deadline(owner, monkeypatch, ceiling):
    monkeypatch.setattr(runtime, "VISUAL_DEPTH_VISIBILITY_MAX_AGE_SEC", ceiling)
    full_result(owner, stamp=NOW-.1)
    p = owner._validated_visual_observation
    assert p.expires_at == pytest.approx(p.timestamp+ceiling)
    assert not p.live(1, p.timestamp+ceiling+.001)


@pytest.mark.parametrize("fault", ["duplicate", "old_capture", "old_timestamp"])
def test_500ms_adapter_replay_cannot_renew_capture_deadline(owner, fault):
    data, records = full_result(owner, stamp=NOW-.1)
    p = owner._validated_visual_observation
    cap = 256 if fault == "duplicate" else 255 if fault == "old_capture" else 257
    stamp = p.timestamp+.01 if fault == "old_capture" else p.timestamp
    owner._rknn_pipeline.last_identity_processing.update(
        capture_frame_id=cap, capture_timestamp=stamp)
    assert not owner._update_detector_identity_lease(records, cap, stamp, now=NOW+.2, stale=False)
    assert owner._validated_visual_observation is p
    assert not p.live(1, p.timestamp+.501)


@pytest.mark.parametrize("fault", ["rejected", "pending", "uid0", "excluded"])
def test_new_identity_rejection_clears_500ms_visibility_immediately(owner, fault):
    data, records = full_result(owner, stamp=NOW-.1)
    p = owner._validated_visual_observation
    assert p.live(1, NOW+.01)
    if fault == "rejected": data["identity_control_rejected"] = True
    if fault == "pending": data["identity_recheck_pending"] = True
    if fault == "uid0": records[0].reid_uid = 0
    if fault == "excluded": data["search_excluded"] = True
    owner._rknn_pipeline.last_identity_processing.update(
        capture_frame_id=257, capture_timestamp=NOW-.01)
    owner._update_detector_identity_lease(records, 257, NOW-.01, now=NOW+.01, stale=False)
    assert owner._validated_visual_observation is False


def test_real_geometry_conflict_still_revokes_grey_bridge(distance_only):
    recheck_result(distance_only, conflict=True)
    assert distance_only.owner._validated_visual_observation is False
    assert distance_only.owner._fresh_depth_linear_snapshot(1, quiet=True) is None


@pytest.mark.parametrize("visual_age,allowed", [(.349, True), (.499, True), (.501, False)])
def test_500ms_visibility_supports_independently_fresh_depth(distance_only, visual_age, allowed):
    a, o = distance_only, distance_only.owner
    full_result(o, stamp=NOW-.1)
    p = o._validated_visual_observation
    a.clock.now = p.timestamp+visual_age
    a.stamp = a.clock.now-.02
    a.feedback[0] = replace(a.feedback[0], timestamp=a.clock.now)
    a.shared = replace(a.shared, sample_timestamp=a.stamp,
                       checked_at=a.clock.now, feedback_timestamp=a.clock.now)
    a.frame = replace(a.frame, steering_feedback=a.feedback[0],
        distance_state=replace(a.frame.distance_state, sample_timestamp=a.stamp,
                               observation_timestamp=a.stamp))
    publish(a)
    assert (o._fresh_depth_linear_snapshot(1, quiet=True) is not None) is allowed


def test_extended_visibility_does_not_extend_yaw_or_capture_steering(owner, monkeypatch):
    full_result(owner, stamp=NOW-.1)
    p = owner._validated_visual_observation
    _intent(owner, capture_timestamp=p.timestamp, valid_until=p.timestamp+.25)
    assert MAX_CAPTURE_AGE_SEC == .25
    at = p.timestamp+.499
    assert p.live(1, at)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: at)
    assert not owner._has_fresh_lateral_yaw(1)
