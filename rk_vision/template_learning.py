"""Bounded, learning-only admission of paired identity templates.

This gate has no identity, motion, clock-renewal or motor authority.  Its owner
supplies *already approved* full/torso pairs for initial bootstrapping, then only
commits a current pair when ``observe().allow`` is true.  A pending observation
is never exposed as a matching template.  Neither a query hit nor elapsed time
can promote an unapproved observation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable

import numpy as np


def _number(value: Any):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _identifier(value: Any):
    result = _number(value)
    return int(result) if result is not None and result > 0 and result.is_integer() else None


def _vector(value: Any):
    try:
        result = np.asarray(value, dtype=np.float32).reshape(-1).copy()
    except (TypeError, ValueError, OverflowError):
        return None
    if not result.size or not np.isfinite(result).all():
        return None
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    result /= norm
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class TemplatePair:
    capture_frame_id: int
    capture_timestamp: float
    full_feature: Any
    partial_feature: Any


@dataclass(frozen=True)
class LearningDecision:
    allow: bool
    reason: str
    confirmations: int = 0
    parent_caps: tuple[int, ...] = ()
    revoked_caps: tuple[int, ...] = ()
    risk_active: bool = False
    full_distance: float | None = None
    partial_distance: float | None = None
    pending_cap: int | None = None
    # Latest discarded sequence, retained across later observations.
    pending_reset_reason: str | None = None
    pending_reset_cap: int | None = None
    pending_reset_count: int = 0
    pending_full_distance: float | None = None
    pending_partial_distance: float | None = None


@dataclass
class _Pending:
    capture: int
    timestamp: float
    started: float
    parent_caps: set[int]
    full_feature: Any
    partial_feature: Any
    count: int = 1


@dataclass
class _State:
    track: int | None = None
    last_capture: int = 0
    last_timestamp: float = -1.
    risk_capture: int = 0
    risk_timestamp: float = -1.
    risk_active: bool = False
    risk_reasons: tuple[str, ...] = ()
    bootstrapped: bool = False
    # Bootstrap roots and later admissions contain only provenance, not crops.
    approved: dict[int, tuple[float, frozenset[int]]] = field(default_factory=dict)
    frozen: tuple[TemplatePair, ...] = ()
    pending: _Pending | None = None
    revoked: set[int] = field(default_factory=set)
    pending_reset_reason: str | None = None
    pending_reset_cap: int | None = None
    pending_reset_count: int = 0


class TemplateLearningGuard:
    """Confirm learning against an immutable, independent parent snapshot.

    One pending observation and at most ``max_parents`` descriptor pairs are
    retained per UID. ``prune_sources`` keeps exact provenance for the owner's
    live gallery, rather than accumulating every historical approval. Shallow
    independent parents are preferred; an overlong branch cannot expand, but
    other branches can still learn. ``prune`` removes explicitly retired UIDs.
    None of those learning holds changes the owner's current tracking result.
    """

    def __init__(self, *, normal_full_limit=.30, normal_partial_limit=.30,
                 risk_full_limit=.20, risk_partial_limit=.20, min_frames=2,
                 normal_min_span_sec=.08, risk_clear_sec=.20, max_gap_sec=1.5,
                 max_pending_distance=.12,
                 maturity_sec=2., max_parents=8, max_lineage=256,
                 max_ancestry=64, max_uids=64):
        for name, value in (("normal_full_limit", normal_full_limit),
                            ("normal_partial_limit", normal_partial_limit),
                            ("risk_full_limit", risk_full_limit),
                            ("risk_partial_limit", risk_partial_limit),
                            ("normal_min_span_sec", normal_min_span_sec),
                            ("risk_clear_sec", risk_clear_sec),
                            ("max_pending_distance", max_pending_distance),
                            ("max_gap_sec", max_gap_sec), ("maturity_sec", maturity_sec)):
            number = _number(value)
            if number is None or number <= 0:
                raise ValueError(name + " must be finite and positive")
            setattr(self, name, number)
        if self.risk_full_limit > self.normal_full_limit or self.risk_partial_limit > self.normal_partial_limit:
            raise ValueError("risk recovery must not weaken the normal pair limits")
        self.min_frames = max(2, int(min_frames))
        self.max_parents = max(1, int(max_parents))
        self.max_lineage = max(self.max_parents, int(max_lineage))
        self.max_ancestry = max(1, min(int(max_ancestry), self.max_lineage - 1))
        self.max_uids = max(1, int(max_uids))
        self._states: dict[int, _State] = {}

    def _state(self, uid):
        identity = _identifier(uid)
        if identity is None:
            return None
        if identity not in self._states:
            if len(self._states) >= self.max_uids:
                return None
            self._states[identity] = _State()
        return self._states[identity]

    def _snapshot(self, state, sources, stamp):
        pairs = []
        seen = set()
        for row in sources:
            if not isinstance(row, TemplatePair):
                continue
            cap = _identifier(row.capture_frame_id)
            captured = _number(row.capture_timestamp)
            if (cap is None or cap in seen or cap in state.revoked or captured is None
                    or captured < 0 or captured >= stamp):
                continue
            if state.bootstrapped:
                admission = state.approved.get(cap)
                if (admission is None or captured != admission[0]
                        or (admission[1] and stamp - admission[0] < self.maturity_sec)):
                    continue
            full, partial = _vector(row.full_feature), _vector(row.partial_feature)
            if full is None or partial is None:
                continue
            pairs.append(TemplatePair(cap, captured, full, partial))
            seen.add(cap)
        # Do not let the latest child fill every snapshot slot merely because
        # the caller happens to enumerate newest-first.  Old accepted roots
        # remain independent learning evidence while the owner retains them.
        pairs.sort(key=lambda pair: (
            len(state.approved.get(pair.capture_frame_id, (0., frozenset()))[1]),
            pair.capture_timestamp, pair.capture_frame_id))
        pairs = pairs[:self.max_parents]
        if not state.bootstrapped and pairs:
            # Initial seed features already passed the owner's startup gate.
            for pair in pairs:
                state.approved[pair.capture_frame_id] = (pair.capture_timestamp, frozenset())
            state.bootstrapped = True
        return tuple(pairs)

    def note_risk(self, *, uid, track_id, capture_frame_id, capture_timestamp,
                  mature_pairs: Iterable[TemplatePair] = (), risk_reasons=()):
        """Freeze learning on a new observation; never authorize anything.

        Empty reasons do not renew a clock or clear a previous risk.  Repeated
        and out-of-order risk captures cannot change an existing snapshot.
        """
        state = self._state(uid)
        cap, stamp, track = (_identifier(capture_frame_id), _number(capture_timestamp),
                             _identifier(track_id))
        if state is None or cap is None or stamp is None or stamp < 0 or track is None:
            return LearningDecision(False, "invalid_risk_observation")
        reasons = tuple(str(reason) for reason in risk_reasons if reason)
        if state.track is not None and state.track != track:
            reasons += ("track_changed",)
        if not reasons:
            return LearningDecision(False, "no_new_risk", risk_active=state.risk_active)
        if cap <= max(state.last_capture, state.risk_capture) or stamp <= max(state.last_timestamp, state.risk_timestamp):
            return LearningDecision(False, "risk_duplicate_or_out_of_order", risk_active=state.risk_active)
        if not state.risk_active:
            state.frozen = self._snapshot(state, mature_pairs, stamp)
        state.track = track
        self._reset_pending(state, "risk_observation", cap)
        state.risk_active = True
        state.risk_capture, state.risk_timestamp = cap, stamp
        state.risk_reasons = tuple(dict.fromkeys(reasons))[:8]
        return self._decision(state, "risk_frozen")

    @staticmethod
    def _reset_pending(state, reason, cap):
        if state.pending is not None:
            state.pending_reset_reason = reason
            state.pending_reset_cap = cap
            state.pending_reset_count += 1
            state.pending = None

    @staticmethod
    def _decision(state, reason, *, allow=False, **kwargs):
        pending = state.pending
        details = dict(confirmations=pending.count if pending else 0,
                       pending_cap=pending.capture if pending else None,
                       risk_active=state.risk_active,
                       pending_reset_reason=state.pending_reset_reason,
                       pending_reset_cap=state.pending_reset_cap,
                       pending_reset_count=state.pending_reset_count)
        details.update(kwargs)
        return LearningDecision(allow, reason, **details)

    def observe(self, *, uid, track_id, capture_frame_id, capture_timestamp,
                full_feature, partial_feature, mature_pairs: Iterable[TemplatePair] = (),
                eligible=True, is_fresh=True, risk_reasons=(), commit_allowed=True):
        """Observe every qualified capture; authorize only a scheduled CURRENT pair.

        ``commit_allowed`` limits writes without skipping evidence between
        writes. A throttled confirmation stays pending and cannot be supplied
        as an approved parent. Adjacent observations must match one common,
        frozen approved parent as well as each other; pending appearance alone
        can never support a new pose or clear a risk.
        """
        state = self._state(uid)
        cap, stamp, track = (_identifier(capture_frame_id), _number(capture_timestamp),
                             _identifier(track_id))
        if state is None or cap is None or stamp is None or stamp < 0 or track is None:
            return LearningDecision(False, "invalid_observation")
        sources = tuple(mature_pairs)
        if cap <= state.last_capture or stamp <= state.last_timestamp:
            return self._decision(state, "duplicate_or_out_of_order")
        risk = tuple(risk_reasons)
        if risk or (state.track is not None and track != state.track):
            self.note_risk(uid=uid, track_id=track, capture_frame_id=cap,
                           capture_timestamp=stamp, mature_pairs=sources, risk_reasons=risk)
        state.track = track
        state.last_capture, state.last_timestamp = cap, stamp
        if risk or cap <= state.risk_capture or stamp <= state.risk_timestamp:
            self._reset_pending(state, "risk_observation", cap)
            return self._decision(state, "risk_observation")
        if eligible is not True or is_fresh is not True:
            self._reset_pending(state, "ineligible_learning_observation", cap)
            return self._decision(state, "ineligible_learning_observation")
        full, partial = _vector(full_feature), _vector(partial_feature)
        if full is None or partial is None:
            self._reset_pending(state, "paired_features_unavailable", cap)
            return self._decision(state, "paired_features_unavailable")
        if state.pending is not None and stamp - state.pending.timestamp > self.max_gap_sec:
            self._reset_pending(state, "observation_gap", cap)
        if state.pending is None and not state.risk_active:
            state.frozen = self._snapshot(state, sources, stamp)
        full_limit = self.risk_full_limit if state.risk_active else self.normal_full_limit
        partial_limit = self.risk_partial_limit if state.risk_active else self.normal_partial_limit
        matched = []
        for pair in state.frozen:
            if (pair.capture_frame_id in state.revoked or pair.full_feature.size != full.size
                    or pair.partial_feature.size != partial.size
                    or len(state.approved.get(pair.capture_frame_id, (0., frozenset()))[1]) >= self.max_ancestry):
                continue
            fd, pd = float(1. - pair.full_feature.dot(full)), float(1. - pair.partial_feature.dot(partial))
            if fd <= full_limit and pd <= partial_limit:
                matched.append((pair.capture_frame_id, fd, pd))
        if not matched:
            reason = "frozen_pair_mismatch" if state.frozen else "mature_pair_unavailable"
            self._reset_pending(state, reason, cap)
            return self._decision(state, reason)
        caps = {row[0] for row in matched}
        pending = state.pending
        same_shape = (pending is not None and pending.full_feature.size == full.size
                      and pending.partial_feature.size == partial.size)
        pending_fd = float(1. - pending.full_feature.dot(full)) if same_shape else None
        pending_pd = float(1. - pending.partial_feature.dot(partial)) if same_shape else None
        distances = dict(pending_full_distance=pending_fd, pending_partial_distance=pending_pd)
        pending_matches = (same_shape and pending_fd <= self.max_pending_distance
                           and pending_pd <= self.max_pending_distance)
        if pending is None or not caps.intersection(pending.parent_caps) or not pending_matches:
            if pending is not None:
                if not caps.intersection(pending.parent_caps):
                    reset_reason = "parent_changed"
                elif not same_shape:
                    reset_reason = "pending_feature_shape_changed"
                elif pending_fd > self.max_pending_distance and pending_pd > self.max_pending_distance:
                    reset_reason = "pending_pair_distance"
                elif pending_fd > self.max_pending_distance:
                    reset_reason = "pending_full_distance"
                else:
                    reset_reason = "pending_partial_distance"
                self._reset_pending(state, reset_reason, cap)
            state.pending = _Pending(cap, stamp, stamp, caps, full, partial)
            return self._decision(state, "pending_confirmation", **distances)
        state.pending.count += 1
        state.pending.capture, state.pending.timestamp = cap, stamp
        state.pending.parent_caps.intersection_update(caps)
        # Compare the next observation to this independently verified capture,
        # so natural pose changes are not amplified by the write interval.
        state.pending.full_feature, state.pending.partial_feature = full, partial
        minimum_span = self.risk_clear_sec if state.risk_active else self.normal_min_span_sec
        if (state.pending.count < self.min_frames
                or stamp - state.pending.started + 1e-12 < minimum_span
                or (state.risk_active and stamp - state.risk_timestamp + 1e-12 < self.risk_clear_sec)):
            return self._decision(state, "pending_confirmation", **distances)
        if commit_allowed is not True:
            return self._decision(state, "commit_throttled",
                                  parent_caps=tuple(sorted(state.pending.parent_caps)), **distances)
        # One independent, same-source full/torso pair suffices.  Recording the
        # union of EVERY matching child would manufacture ancestry growth even
        # when every new sample still matches the same clean original parent.
        parent = min(state.pending.parent_caps, key=lambda parent_cap: (
            len(state.approved[parent_cap][1]), state.approved[parent_cap][0], parent_cap))
        parent_caps = (parent,)
        ancestors = set(parent_caps)
        for parent in parent_caps:
            ancestors.update(state.approved.get(parent, (0., frozenset()))[1])
        if len(state.approved) >= self.max_lineage or len(ancestors) >= self.max_lineage:
            self._reset_pending(state, "lineage_capacity", cap)
            return self._decision(state, "lineage_capacity")
        state.approved[cap] = (stamp, frozenset(ancestors))
        best = min((row for row in matched if row[0] in parent_caps), key=lambda row: row[1] + row[2])
        confirmations = state.pending.count
        state.pending = None
        state.risk_active = False
        state.risk_reasons = ()
        return self._decision(state, "paired_learning_approved", allow=True,
                              confirmations=confirmations, parent_caps=parent_caps,
                              full_distance=best[1], partial_distance=best[2], **distances)

    def revoke_sources(self, uid, capture_frame_ids):
        """Return specified sources and every tracked descendant for isolation."""
        state = self._states.get(_identifier(uid))
        caps = {cap for value in capture_frame_ids if (cap := _identifier(value)) is not None}
        if state is None:
            return caps
        affected = caps | {cap for cap, (_, ancestors) in state.approved.items() if caps.intersection(ancestors)}
        # Every stored ancestor remains in the bounded registry.  Retaining
        # those revoked records prevents a caller from reintroducing them.
        state.revoked.update(cap for cap in affected if cap in state.approved)
        state.frozen = tuple(pair for pair in state.frozen if pair.capture_frame_id not in affected)
        self._reset_pending(state, "sources_revoked", state.last_capture or None)
        return affected

    def prune_sources(self, uid, live_caps):
        """Retain exact ancestors of current archive/recent/representative CAPs.

        The owner must include every feature tier, even temporarily isolated
        samples still physically retained. A pending or risk-held parent
        snapshot also stays pinned until it is no longer used. Sources not in
        this registry cannot re-enter through ``mature_pairs`` after pruning;
        bootstrapping is deliberately not re-enabled for an empty registry.

        Returns retired provenance CAPs, not a request to delete gallery data.
        """
        state = self._states.get(_identifier(uid))
        if state is None:
            return set()
        live = {cap for value in live_caps if (cap := _identifier(value)) is not None}
        if state.pending is not None or state.risk_active:
            live.update(pair.capture_frame_id for pair in state.frozen)
        keep = live.intersection(state.approved)
        for cap in tuple(keep):
            keep.update(state.approved[cap][1])
        retired = set(state.approved).difference(keep)
        state.approved = {cap: node for cap, node in state.approved.items() if cap in keep}
        state.revoked.intersection_update(keep)
        state.frozen = tuple(pair for pair in state.frozen if pair.capture_frame_id in keep)
        return retired

    def prune(self, live_uids):
        keep = {_identifier(uid) for uid in live_uids}
        self._states = {uid: state for uid, state in self._states.items() if uid in keep}

    def diagnostics(self, uid):
        state = self._states.get(_identifier(uid))
        if state is None:
            return {}
        return dict(risk_active=state.risk_active, risk_reasons=state.risk_reasons,
                    frozen_parent_caps=tuple(pair.capture_frame_id for pair in state.frozen),
                    pending_cap=state.pending.capture if state.pending else None,
                    confirmations=state.pending.count if state.pending else 0,
                    pending_started_timestamp=state.pending.started if state.pending else None,
                    pending_last_timestamp=state.pending.timestamp if state.pending else None,
                    pending_span_sec=(state.pending.timestamp - state.pending.started) if state.pending else 0.,
                    pending_reset_reason=state.pending_reset_reason,
                    pending_reset_cap=state.pending_reset_cap,
                    pending_reset_count=state.pending_reset_count,
                    last_capture=state.last_capture,
                    lineage_count=len(state.approved), revoked_caps=tuple(sorted(state.revoked)))
