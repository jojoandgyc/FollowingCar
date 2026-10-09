"""Actual identity publisher + paired adapter race regressions, no hardware.

The mailbox is committed as a whole. Control time is taken after observing
that mailbox, never borrowed from the beginning of a slower perception task.
"""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import SensorFrame
from car_control_modular.detector_identity_lease import (
    DetectorIdentityLease, ValidatedVisualObservation, VisualIdentityEvidence,
    publish_visual_identity_evidence, read_visual_identity_evidence,
)
from test_detector_identity_lease import fixture_pipeline
from test_lateral_zero_runtime import NOW, owner
from test_short_follow_adapter import paired, process


class MirroredWriteObserver:
    """Force a consumer read between the two real publisher mirror writes."""
    def __init__(self, owner, reader):
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "reader", reader)

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def __setattr__(self, name, value):
        setattr(self.owner, name, value)
        if name in ("_validated_visual_observation", "_detector_identity_lease"):
            self.reader(self.owner)


def test_immutable_publication_and_read_then_clock_order():
    initial = ValidatedVisualObservation(1, 1, 10, 99.9, 100., 100.4, "full")
    later = replace(initial, capture=11, timestamp=100.01, validated_at=100.02,
                    expires_at=100.51)
    owner = SimpleNamespace()
    original = publish_visual_identity_evidence(owner, observation=initial, lease=None)

    def clock():
        publish_visual_identity_evidence(owner, observation=later, lease=None)
        return 100.02

    observed, now = read_visual_identity_evidence(owner, clock=clock)
    assert observed is original
    assert observed.live(1, now)
    assert owner._visual_identity_evidence.observation is later
    with pytest.raises(FrozenInstanceError):
        observed.lease = False


@pytest.mark.parametrize("transition", ["full_to_fast", "fast_to_full", "fast_to_reject"])
def test_real_publisher_cannot_expose_torn_full_fast_or_rejected_identity(owner, transition):
    data, records = fixture_pipeline(owner)
    first = ValidatedVisualObservation(1, 1, 2, 99.7, 99.71, 100.2, "full")
    publish_visual_identity_evidence(owner, observation=first, lease=None)
    if transition != "full_to_fast":
        assert owner._update_detector_identity_lease(records, 4, 99.9, now=100., stale=False)
        owner._rknn_pipeline.last_identity_processing = dict(
            mode="full", full_features_current=True, capture_frame_id=5,
            capture_timestamp=100.01)
        data.clear()
        data.update(uid=1, bbox_quality_ok=True, reason="mapped")
        if transition == "fast_to_reject":
            data["identity_control_rejected"] = True
    observations = []

    def observe(owner):
        evidence, now = read_visual_identity_evidence(owner, clock=lambda: 100.02)
        observations.append(evidence)
        if transition == "full_to_fast":
            assert evidence.observation.kind == "detector_continuation"
            assert isinstance(evidence.lease, DetectorIdentityLease)
            assert evidence.observation.capture == evidence.lease.observation_capture == 4
            assert evidence.live(1, now)
        elif transition == "fast_to_full":
            assert evidence.observation.capture == 5
            assert evidence.observation.kind == "full"
            assert evidence.lease is None
            assert evidence.live(1, now)
        else:
            assert evidence.observation is False and evidence.lease is False
            assert not evidence.live(1, now)

    proxy = MirroredWriteObserver(owner, observe)
    cap, stamp = (4, 99.9) if transition == "full_to_fast" else (5, 100.01)
    assert runtime.PersonTracker._update_detector_identity_lease(
        proxy, records, cap, stamp, now=100.02, stale=False)
    assert len(observations) == 2
    assert observations[0] is observations[1] is owner._visual_identity_evidence


def test_legacy_owner_without_mailbox_keeps_explicit_rejection_and_uid_checks():
    proof = ValidatedVisualObservation(1, 1, 1, 99.9, 100., 100.4, "full")
    owner = SimpleNamespace(_validated_visual_observation=proof,
                            _detector_identity_lease=None)
    evidence, now = read_visual_identity_evidence(owner, clock=lambda: 100.1)
    assert evidence.live(1, now) and not evidence.live(2, now)
    owner._detector_identity_lease = False
    evidence, now = read_visual_identity_evidence(owner, clock=lambda: 100.1)
    assert not evidence.live(1, now)
    owner._detector_identity_lease = None
    owner._validated_visual_observation = False
    evidence, now = read_visual_identity_evidence(owner, clock=lambda: 100.1)
    assert not evidence.live(1, now)


def adapter_frame(a, *, capture=577, timestamp=NOW-.03, depth_timestamp=NOW-.01):
    return SensorFrame(width=640, height=480, persons=[a.target], distance_m=2.,
        distance_state=replace(a.owner._distance_runtime.get_frame_distance_state(),
                               sample_timestamp=depth_timestamp),
        capture_frame_id=capture, capture_timestamp=timestamp)


@pytest.mark.parametrize("kind", ["full", "detector_continuation"])
def test_paired_adapter_accepts_new_proof_even_if_control_entry_clock_predates_it(paired, monkeypatch, kind):
    a = paired
    process(a)
    prior = a.owner._short_follow.snapshot().plan
    data, records = fixture_pipeline(a.owner)
    a.owner._rknn_pipeline.last_identity_processing = dict(
        mode=kind, full_features_current=True, capture_frame_id=577,
        capture_timestamp=NOW-.03)
    if kind == "full":
        data.clear()
        data.update(uid=1, bbox_quality_ok=True, reason="mapped")
    else:
        data.update(capture_frame_id=577, capture_timestamp=NOW-.03)
    assert a.owner._update_detector_identity_lease(
        records, 577, NOW-.03, now=NOW+.01, stale=False)
    assert a.owner._visual_identity_evidence.observation.validated_at == NOW+.01
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.02)
    assert a.owner._short_follow_adapter.handle(
        adapter_frame(a), a.target, is_fresh_depth=True, control_source="depth30",
        target_steerable=True, low_quality_visible=False, now=NOW)
    plan = a.owner._short_follow.snapshot().plan
    assert plan is not prior and plan.moving
    assert plan.depth_timestamp == NOW-.01
    assert plan.expires_at == pytest.approx(NOW-.01+a.owner._short_follow.config.depth_ttl_sec)


def test_paired_adapter_old_clock_cannot_extend_expired_depth(paired, monkeypatch):
    a = paired
    process(a)
    prior = a.owner._short_follow.snapshot().plan
    proof = ValidatedVisualObservation(1, 1, 577, NOW+.25, NOW+.30, NOW+.75, "full")
    publish_visual_identity_evidence(a.owner, observation=proof, lease=None)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.31)
    assert a.owner._short_follow_adapter.handle(
        adapter_frame(a, timestamp=NOW+.25, depth_timestamp=NOW),
        a.target, is_fresh_depth=True, control_source="depth30",
        target_steerable=True, low_quality_visible=False, now=NOW+.10)
    state = a.owner._short_follow.snapshot()
    assert state.plan is prior  # the stale new sample cannot refresh the mailbox
    assert not state.plan.valid(NOW+.31)


@pytest.mark.parametrize("kind", ["uid_conflict", "rejected", "expired", "future"])
def test_paired_adapter_refresh_is_not_permission_to_ignore_negative_evidence(paired, monkeypatch, kind):
    a = paired
    process(a)
    prior = a.owner._short_follow.snapshot().plan
    proof = ValidatedVisualObservation(1, 1, 577, NOW-.03, NOW+.01, NOW+.47, "full")
    if kind == "uid_conflict":
        proof = replace(proof, uid=2)
    elif kind == "rejected":
        proof = False
    elif kind == "expired":
        proof = replace(proof, expires_at=NOW+.015)
    else:
        proof = replace(proof, validated_at=NOW+.03)
    publish_visual_identity_evidence(a.owner, observation=proof, lease=None)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW+.02)
    handled = a.owner._short_follow_adapter.handle(
        adapter_frame(a), a.target, is_fresh_depth=True, control_source="depth30",
        target_steerable=True, low_quality_visible=False, now=NOW)
    state = a.owner._short_follow.snapshot()
    if kind in ("uid_conflict", "rejected"):
        assert not handled and not state.active and state.plan is None
    else:
        assert state.plan is prior  # no new plan; executor separately gates expiry
