"""Fixed identity deadline for measured detector-only continuation frames.

The deadline is independent of Depth acquisition and of command publication.
It can only be removed by a newer completed full verification, not by another
detector sample or by the periodic motor writer.
"""
from dataclasses import dataclass
import math
import time
from rk_vision.detector_continuation import FULL_PROOF_TTL_SEC, MAX_FULL_RESULT_AGE_SEC


# Visibility supporting independently fresh Depth is not a steering/ROI lease.
# Keep the admission age of a NEW full result separate: a proof validated on
# time may remain usable after that entry gate, but a late result cannot mint it.
MAX_VALIDATED_VISIBILITY_AGE_SEC = .50


def _number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


@dataclass(frozen=True)
class DetectorIdentityLease:
    uid: int
    track_id: int
    verified_capture: int
    verified_timestamp: float
    observation_capture: int
    observation_timestamp: float
    expires_at: float

    def live(self, uid, now):
        return (_number(now) and uid == self.uid
                and self.observation_timestamp <= now < self.expires_at)


@dataclass(frozen=True)
class ValidatedVisualObservation:
    """Current measured-person evidence, independent of yaw/depth publication.

    Full validation completion is not a new camera exposure. Both the real
    capture visibility ceiling and the configured visibility window bound it;
    detector-only continuation additionally keeps its original identity TTL.
    """
    uid: int
    track_id: int
    capture: int
    timestamp: float
    validated_at: float
    expires_at: float
    kind: str
    # Existing narrow grey-frame bridge may finish one already issued Depth
    # grant. It never grants a new measurement or refreshes this visual TTL.
    continuation_sample_timestamp: float | None = None

    def live(self, uid, now):
        return bool(_number(now) and uid == self.uid
                    and self.validated_at <= now < self.expires_at)

    def permits_depth(self, uid, sample_timestamp, now):
        return bool(self.live(uid, now) and (
            self.continuation_sample_timestamp is None
            or self.continuation_sample_timestamp == sample_timestamp))


@dataclass(frozen=True)
class VisualIdentityEvidence:
    """One completed identity publication, not two independently read fields.

    ``None`` preserves startup/full-only compatibility; ``False`` is an
    explicit rejection. A detector lease and its corresponding observation
    must change together, including when full verification retires that lease.
    """
    observation: ValidatedVisualObservation | bool | None
    lease: DetectorIdentityLease | bool | None

    def motion_identity_live(self, uid, now):
        return self.lease is None or (
            isinstance(self.lease, DetectorIdentityLease) and self.lease.live(uid, now))

    def live(self, uid, now):
        return bool(isinstance(self.observation, ValidatedVisualObservation)
                    and self.observation.live(uid, now)
                    and self.motion_identity_live(uid, now))

    def permits_depth(self, uid, sample_timestamp, now):
        return bool(self.live(uid, now)
                    and self.observation.permits_depth(uid, sample_timestamp, now))


def publish_visual_identity_evidence(owner, *, observation, lease):
    """Commit a complete result before mirroring it to legacy readers.

    The normal paired control path reads only the immutable publication.
    Legacy fields remain available for search and legacy control, but cannot
    expose an intermediate full/fast transition to the paired motor writer.
    """
    evidence = VisualIdentityEvidence(observation, lease)
    owner._visual_identity_evidence = evidence
    owner._validated_visual_observation = observation
    owner._detector_identity_lease = lease
    return evidence


def read_visual_identity_evidence(owner, *, clock=None):
    """Read the proof FIRST, then the clock used to assess that proof.

    Sampling time before a concurrent publication can label a genuinely new
    proof as being from the future. Never repair that race by extending its
    expiry or by accepting a future timestamp. Old full-only adapters/fakes
    without this mailbox retain their two-field contract; runtime publishers
    all use the atomic mailbox.
    """
    evidence = getattr(owner, "_visual_identity_evidence", None)
    if not isinstance(evidence, VisualIdentityEvidence):
        evidence = VisualIdentityEvidence(
            getattr(owner, "_validated_visual_observation", None),
            getattr(owner, "_detector_identity_lease", None))
    return evidence, (time.monotonic() if clock is None else clock())


def validated_visual_observation(*, uid, track_id, capture, timestamp, now,
                                 visibility_window, identity_lease=None,
                                 capture_max_age_sec=MAX_FULL_RESULT_AGE_SEC):
    if (any(type(v) is not int or v <= 0 for v in (uid, track_id, capture))
            or any(not _number(v) for v in (timestamp, now, visibility_window, capture_max_age_sec))
            or not 0 < timestamp <= now < timestamp + MAX_FULL_RESULT_AGE_SEC
            or visibility_window <= 0 or capture_max_age_sec <= 0):
        return None
    until = min(timestamp + min(MAX_VALIDATED_VISIBILITY_AGE_SEC, capture_max_age_sec),
                now + min(MAX_VALIDATED_VISIBILITY_AGE_SEC, visibility_window))
    if now >= until:
        return None
    kind = "full"
    if identity_lease is not None:
        if (not isinstance(identity_lease, DetectorIdentityLease)
                or not identity_lease.live(uid, now)
                or identity_lease.track_id != track_id
                or identity_lease.observation_capture != capture
                or identity_lease.observation_timestamp != timestamp):
            return None
        until = min(until, identity_lease.expires_at)
        kind = "detector_continuation"
    return ValidatedVisualObservation(uid, track_id, capture, timestamp, now, until, kind)


def from_assignment(assignment, *, uid, track_id, capture, timestamp, now):
    if not isinstance(assignment, dict):
        return None
    verified = assignment.get("identity_verified_capture")
    stamp = assignment.get("identity_verified_timestamp")
    until = assignment.get("identity_valid_until")
    if (assignment.get("identity_evidence_kind") != "detector_continuation"
            or assignment.get("uid") != uid
            or assignment.get("mapped_uid", uid) != uid
            or any(type(v) is not int or v <= 0 for v in (uid, track_id, capture, verified))
            or any(not _number(v) for v in (stamp, until, timestamp, now))
            or not 0 < stamp < timestamp <= now < until
            or not verified < capture or until > stamp + FULL_PROOF_TTL_SEC + 1e-9
            or assignment.get("capture_frame_id") != capture
            or assignment.get("capture_timestamp") != timestamp
            or assignment.get("identity_control_rejected")
            or assignment.get("bbox_quality_ok") is not True
            or assignment.get("bank_updated") or assignment.get("initial_identity_confirmed")):
        return None
    return DetectorIdentityLease(uid, track_id, verified, stamp, capture, timestamp, until)


def motion_identity_live(owner, uid, now):
    lease = getattr(owner, "_detector_identity_lease", None)
    # None is the unchanged full-verification policy; invalid fast evidence is
    # represented by False, never mistaken for the absence of a lease.
    return lease is None or (isinstance(lease, DetectorIdentityLease) and lease.live(uid, now))


def detector_execution_supported(executor):
    """Only the canonical periodic writer has the fast identity deadline gate.

    Legacy raw/percent, pulse and rotation-only modes retain full ReID; they
    must not inherit a fast proof through a writer that cannot check it.
    """
    period = getattr(getattr(executor, "config", None), "follow_wheel_period_sec", None)
    scope = getattr(executor, "_visible_wheel_control_active", None)
    return bool(_number(period) and period >= .05 and callable(scope) and scope())
