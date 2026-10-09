"""A fast detector result never renews its source identity or motor lease."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import request_0513_modular as runtime
from car_control_modular.detector_identity_lease import (
    DetectorIdentityLease, from_assignment, motion_identity_live, detector_execution_supported,
)
from car_control_modular.control_types import SteeringFeedback
from test_distance_pi_runtime import pi_owner
from test_lateral_zero_runtime import owner, NOW, _intent
from test_relative_depth_continuation_runtime import relative, publish


def assignment(cap=4, stamp=99.9):
    return dict(uid=1, identity_evidence_kind="detector_continuation",
                identity_verified_capture=2, identity_verified_timestamp=99.7,
                identity_valid_until=100.05, capture_frame_id=cap,
                capture_timestamp=stamp, bbox_quality_ok=True,
                bank_updated=False, initial_identity_confirmed=False)


def test_fixed_proof_does_not_renew_with_new_detection():
    first = from_assignment(assignment(), uid=1, track_id=1, capture=4, timestamp=99.9, now=100.)
    second = from_assignment(assignment(5, 100.01), uid=1, track_id=1,
                             capture=5, timestamp=100.01, now=100.02)
    assert first.expires_at == second.expires_at == 100.05
    assert first.live(1, 100.049) and not first.live(1, 100.05)
    assert not second.live(2, 100.02)


@pytest.mark.parametrize("key,value", [
    ("identity_valid_until", 100.5), ("identity_verified_timestamp", float("nan")),
    ("identity_valid_until", 100.301),
    ("identity_verified_capture", 5), ("identity_verified_capture", True),
    ("bank_updated", True), ("initial_identity_confirmed", True),
    ("identity_control_rejected", True), ("bbox_quality_ok", False),
    ("capture_timestamp", 99.8), ("capture_frame_id", 3),
    ("uid", 2), ("mapped_uid", 2),
])
def test_bad_or_borrowed_proof_never_authorizes(key, value):
    data = assignment(); data[key] = value
    assert from_assignment(data, uid=1, track_id=1, capture=4, timestamp=99.9, now=100.) is None


@pytest.mark.parametrize("quiet", [False, True])
def test_motor_reader_expiry_even_when_depth_and_feedback_are_fresh(relative, quiet):
    stamp, original = publish(relative)
    o = relative.owner
    o._detector_identity_lease = from_assignment(assignment(), uid=1, track_id=1,
        capture=4, timestamp=99.9, now=100.)
    assert o._fresh_depth_linear_snapshot(1, quiet=quiet) is not None
    relative.clock.now = 100.06
    relative.feedback = replace(relative.feedback, timestamp=100.06)
    assert 100.06-stamp < .25
    assert o._fresh_depth_linear_snapshot(1, quiet=quiet) is None
    assert o._depth30_linear_snapshot is original


def test_feedback_wait_cannot_cross_fixed_identity_deadline(relative):
    publish(relative)
    o = relative.owner
    o._detector_identity_lease = from_assignment(assignment(), uid=1, track_id=1,
        capture=4, timestamp=99.9, now=100.)
    def read():
        relative.clock.now = 100.06
        return replace(relative.feedback, timestamp=100.06)
    o._action_runtime.get_steering_feedback = read
    assert o._fresh_depth_linear_snapshot(1, now=100., quiet=True) is None


def test_identity_deadline_also_closes_live_yaw_and_combined_axes(owner, monkeypatch):
    _intent(owner)
    owner._lateral_yaw_revision = 1
    owner._detector_identity_lease = from_assignment(assignment(), uid=1, track_id=1,
        capture=4, timestamp=99.9, now=100.)
    # Even a separately live longitudinal request cannot escape final binding.
    owner._fresh_depth_linear_snapshot = lambda *a, **k: ("forward", 20, 1, 99.98)
    assert owner._has_fresh_lateral_yaw(1)
    assert owner._follow_wheel_axes(100.)[2:] == (.2 * runtime.MOTOR_FORWARD_MAX_TARGET_RPM, 5.)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 100.06)
    assert not owner._has_fresh_lateral_yaw(1)
    assert owner._follow_wheel_axes(100.06)[2:] == (0., 0.)


def fixture_pipeline(owner):
    data = assignment()
    tracker = SimpleNamespace(control_assignment_for_track=lambda _: dict(data),
                              identity_bank=SimpleNamespace(last_assignments={}))
    owner._rknn_pipeline = SimpleNamespace(tracker=tracker,
        last_identity_processing={"mode": "detector_continuation"})
    return data, [SimpleNamespace(reid_uid=1, track_id=1, time_since_update=0)]


def test_full_validation_alone_can_release_fast_lease(owner):
    data, records = fixture_pipeline(owner)
    assert owner._update_detector_identity_lease(records, 4, 99.9, now=100., stale=False)
    source = owner._detector_identity_lease
    # Duplicate or reordered full output cannot release the original deadline.
    owner._rknn_pipeline.last_identity_processing = dict(mode="full", full_features_current=True,
        capture_frame_id=5, capture_timestamp=100.01)
    assert not owner._update_detector_identity_lease(records, 4, 99.9, now=100.01, stale=False)
    assert owner._detector_identity_lease is source
    data.clear(); data.update(uid=1, bbox_quality_ok=True, reason="mapped")
    assert owner._update_detector_identity_lease(records, 5, 100.01, now=100.02, stale=False)
    assert owner._detector_identity_lease is None


@pytest.mark.parametrize("fault", ["stale", "missing_features", "uid0", "pending", "rejected",
                                  "wrong_capture", "wrong_timestamp", "capture_expired"])
def test_full_failure_cannot_release_fast_deadline(owner, fault):
    data, records = fixture_pipeline(owner)
    owner._update_detector_identity_lease(records, 4, 99.9, now=100., stale=False)
    owner._rknn_pipeline.last_identity_processing = {
        "mode": "full", "full_features_current": fault != "missing_features",
        "capture_frame_id": 4 if fault == "wrong_capture" else 5,
        "capture_timestamp": 99.9 if fault == "wrong_timestamp" else 100.01}
    data.clear(); data.update(uid=1, bbox_quality_ok=True)
    if fault == "uid0": records[0].reid_uid = 0
    if fault == "pending": data["identity_recheck_pending"] = True
    if fault == "rejected": data["identity_control_rejected"] = True
    owner._update_detector_identity_lease(records, 5, 100.01,
        now=100.5 if fault == "capture_expired" else 100.02, stale=fault == "stale")
    assert not motion_identity_live(owner, 1, 100.02)


def test_full_verified_control_is_not_blocked_by_gallery_only_quarantine(owner):
    data, records = fixture_pipeline(owner)
    owner._update_detector_identity_lease(records, 4, 99.9, now=100., stale=False)
    owner._rknn_pipeline.last_identity_processing = dict(mode="full", full_features_current=True,
        capture_frame_id=5, capture_timestamp=100.01)
    data.clear()
    data.update(uid=1, bbox_quality_ok=True, template_update_quarantined=True,
                identity_control_rejected=False, reason="mapped")
    assert owner._update_detector_identity_lease(records, 5, 100.01, now=100.02, stale=False)
    assert owner._detector_identity_lease is None


def test_old_full_only_adapters_keep_existing_contract(owner):
    owner._rknn_pipeline = SimpleNamespace()
    assert owner._update_detector_identity_lease([], 4, 99.9, now=100., stale=False)
    assert motion_identity_live(owner, 1, 100.)


@pytest.mark.parametrize("period,scope,expected", [
    (.05, True, True), (.1, True, True), (0., True, False),
    (.05, False, False), (float("nan"), True, False), (None, True, False),
])
def test_fast_mode_only_uses_deadline_aware_execution(period, scope, expected):
    executor = SimpleNamespace(config=SimpleNamespace(follow_wheel_period_sec=period),
                               _visible_wheel_control_active=lambda: scope)
    assert detector_execution_supported(executor) is expected
    assert not detector_execution_supported(None)
