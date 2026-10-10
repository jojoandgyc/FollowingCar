"""Capture-bounded descriptors for association, never identity/template learning."""
from dataclasses import dataclass, field
import math
from numbers import Integral, Real


def capture_clock(context):
    cap, stamp = context.get("capture_frame_id"), context.get("capture_timestamp")
    if (isinstance(cap, bool) or not isinstance(cap, Integral) or cap <= 0
            or isinstance(stamp, bool) or not isinstance(stamp, Real)):
        return None
    stamp = float(stamp)
    return (int(cap), stamp) if math.isfinite(stamp) and stamp >= 0 else None


@dataclass
class AssociationObservation:
    track: object
    uid: int
    identity: object
    capture: int
    timestamp: float
    samples: list = field(default_factory=list)
    reason: str = "accepted_current_follow"


class ProvisionalAssociationCache:
    """At most three observations per live raw-track object and identity object.

    An empty entry is a tombstone: IoU must not resurrect an expired or revoked
    provisional raw track. Ordinary independent verification may release it.
    """
    def __init__(self):
        self.entries = {}

    def invalidate(self, raw, reason):
        entry = self.entries.get(raw)
        if entry is not None:
            entry.samples.clear()
            entry.reason = reason

    def prepare(self, tracks, uid_by_track, identities, context, *, max_gap, conflicts, revoked,
                detector_position_bridge=None):
        live = {int(track.track_id): track for track in tracks}
        self.entries = {raw: entry for raw, entry in self.entries.items()
                        if live.get(raw) is entry.track}
        clock = capture_clock(context)
        result = {}
        limit = min(.5, max(0., float(max_gap)))
        for raw, entry in self.entries.items():
            bridge = detector_position_bridge or {}
            proof = bridge.get("proof")
            # Only the exactly-bound latest FULL descriptor may be carried
            # across an already measured fast position, once, inside its
            # original <=600ms deadline. Its timestamp is never rewritten.
            carried = bool(proof is not None and clock is not None
                and bridge.get("identity") is entry.identity
                and proof.permission == "similar_follow" and proof.fast_count > 0
                and proof.uid == entry.uid and proof.track_id == raw
                and (proof.verified.capture, proof.verified.timestamp) == (entry.capture, entry.timestamp)
                and (bridge.get("capture_frame_id"), bridge.get("capture_timestamp")) == clock
                and entry.capture < proof.previous.capture < clock[0]
                and entry.timestamp < proof.previous.timestamp < clock[1] < proof.deadline
                and proof.deadline <= entry.timestamp + .60
                and clock[1]-proof.previous.timestamp <= limit)
            if uid_by_track.get(raw) != entry.uid or identities.get(entry.uid) is not entry.identity:
                self.invalidate(raw, "identity_binding_changed")
            elif raw in conflicts or entry.uid in revoked:
                self.invalidate(raw, "identity_conflict")
            elif clock is not None and clock[1] >= entry.timestamp:
                had_samples = bool(entry.samples)
                entry.samples[:] = [sample for sample in entry.samples
                                    if 0 <= clock[1] - sample[1] <= limit
                                    or (carried and sample[:2] == (entry.capture, entry.timestamp))]
                if had_samples and not entry.samples:
                    entry.reason = "capture_expired"
            # Old/invalid captures cannot consume cache evidence or refresh it.
            fresh = clock is not None and clock[0] > entry.capture and clock[1] > entry.timestamp
            result[raw] = [row[2] for row in entry.samples] if fresh else []
        return result

    def remember(self, track, uid, identity, feature, context):
        import numpy as np
        clock = capture_clock(context)
        if clock is None or feature is None:
            return False
        vector = np.asarray(feature, dtype="float32")
        norm = float(np.linalg.norm(vector))
        if (vector.ndim != 1 or not vector.size or not np.all(np.isfinite(vector))
                or not math.isfinite(norm) or norm <= 0):
            return False
        raw = int(track.track_id)
        entry = self.entries.get(raw)
        if entry is not None:
            if (entry.track is not track or entry.uid != uid or entry.identity is not identity
                    or clock[0] <= entry.capture or clock[1] <= entry.timestamp):
                return False
        else:
            entry = AssociationObservation(track, uid, identity, *clock)
            self.entries[raw] = entry
        entry.capture, entry.timestamp = clock
        entry.samples.append((clock[0], clock[1], vector.copy()))
        del entry.samples[:-3]
        entry.reason = "accepted_current_follow"
        return True

    def diagnostics(self, raw):
        entry = self.entries.get(raw)
        if entry is None:
            return dict(samples=0, reason="no_provisional_evidence")
        return dict(uid=entry.uid, capture_frame_id=entry.capture,
                    capture_timestamp=entry.timestamp, samples=len(entry.samples),
                    reason=entry.reason, trusted_gallery_updated=False)
