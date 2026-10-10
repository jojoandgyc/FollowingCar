"""Shared detector/retry observation budget; never identity authority."""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

from .search_candidate_gate import BBox, SearchCandidateGate, SearchCandidateGateDecision


DEFERRED_EDGE_COVERAGE = "candidate_observation_deferred_edge_coverage"


def _same_search_edge(bbox, width, search_direction):
    """Current detector geometry, not a predicted/display box or identity."""
    try:
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in (*bbox, width)):
            return False
        x1, y1, x2, y2 = bbox
        if not width > 0 or not 0 <= x1 < x2 <= width or not 0 <= y1 < y2:
            return False
        center = (x1+x2)/(2*width)
        return ((search_direction == "left" and x1 <= width*.02 and center <= .20)
                or (search_direction == "right" and x2 >= width*.98 and center >= .80))
    except (TypeError, ValueError, OverflowError):
        return False


def detector_only_side_search_observation(bbox, score, width, height,
                                          search_direction, *, min_score, confidence_limit):
    """A low-score edge fragment alone need not interrupt an existing sweep.

    Called only for a fresh, unique formal detection from a completed tracker
    update with NO identity observations. Never a fallback for rejected or
    ambiguous identity evidence. This neither identifies the person nor changes
    search direction, speed, deadline, or the eligibility of a central look.
    """
    try:
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in (score, height, min_score, confidence_limit)):
            return False
        if not _same_search_edge(bbox, width, search_direction):
            return False
        x1, y1, x2, y2 = bbox
        return bool(0 < confidence_limit <= 1
                    and 0 < min_score <= score < min(.50, confidence_limit)
                    and height > 0 and y2 <= height
                    and .12 <= (x2-x1)/(y2-y1) < .50)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False


def _low_score_observation_only(assignment, metadata):
    """Current low-score association with an explicit non-conflict outcome.

    The bank's generic low-score rejection used to hide both missing anchors
    and real contradictions. Only its explicit current diagnostic can separate
    those cases; an absent flag must not be interpreted as no conflict.
    """
    a, m = assignment, metadata
    if (a.get("reason") not in {"low_score_observation_rejected", "low_score_observation_only",
                                "similar_follow_observe"}
            or a.get("low_score_observation_blocked") is not False
            or m.get("low_score_continuation") is not True
            or m.get("association_reason") != "low_score_existing_track"):
        return False
    try:
        cap, previous_cap = m.get("capture_frame_id"), m.get("association_previous_capture_frame_id")
        stamp, previous_stamp = m.get("capture_timestamp"), m.get("association_previous_capture_timestamp")
        score, limit = m.get("detector_confidence"), m.get("association_confidence_limit")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in (stamp, previous_stamp, score, limit)):
            return False
        return bool(type(cap) is int and type(previous_cap) is int and cap > previous_cap >= 0
                    and 0 < stamp-previous_stamp <= .50
                    and 0 < limit <= 1 and 0 < score < min(.50, limit))
    except (TypeError, ValueError, OverflowError):
        return False


def incomparable_edge_observation(assignment, metadata, output_uid, uid, bbox, width):
    """Defer STOP, not identity: stopping cannot restore a side-cut body crop.

    Only explicit CURRENT evidence qualifies. Missing metadata must keep the
    existing observation opportunity. The caller binds the detection, CAP and
    timestamp and checks uniqueness; no old anchor or low ReID score is used.
    """
    a, m = assignment or {}, metadata or {}
    try:
        proof = a.get("identity_competition") or m.get("identity_competition") or {}
        full = a.get("template_recent_evidence") or {}
        partial = a.get("template_recent_partial_evidence") or {}
        x1, y1, x2, y2 = map(float, bbox)
        width = float(width)
        center = (x1 + x2) / (2 * width)
        return bool(
            uid is not None and output_uid != uid
            and a.get("reason") == "secondary_evidence_unavailable"
            and a.get("reacquire_partial_comparable") is False
            and a.get("reacquire_partial_state") == "unknown"
            and m.get("is_fresh") is True
            and proof.get("passed") is True and proof.get("uid") == uid
            and proof.get("candidate_count") == 1
            and proof.get("frame_index") is not None
            and proof.get("frame_index") == m.get("control_frame_id", m.get("frame_index"))
            and m.get("source_detection_index") is not None
            and proof.get("source_detection_index") == m.get("source_detection_index")
            and full.get("count", 0) > 0 and partial.get("count", 0) > 0
            and full.get("comparable_count") == partial.get("comparable_count") == 0
            and str(full.get("query_coverage", "")).endswith("_side1")
            and partial.get("query_coverage") == full.get("query_coverage")
            and all(math.isfinite(v) for v in (x1, y1, x2, y2, width))
            and width > 0 and 0 <= x1 < x2 <= width and y1 < y2
            and ((x1 <= 2 and center <= .15) or (x2 >= width - 2 and center >= .85))
        )
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


def side_crop_search_observation(assignment, metadata, output_uid, uid, bbox, width,
                                 search_direction):
    """Skip an unsuitable stationary look, never authorize target movement.

    A fresh unique same-side crop with gallery support may still be UID0
    because its secondary region is unavailable or its detector score dipped.
    Neither quality-label change alone requests a stationary look. Leave the
    existing bounded search in charge, without assigning this candidate,
    extending search/identity clocks, or changing its learning permission.

    The caller binds this observation to the exact current CAP/detection,
    checks all visible candidates and hard safety, and requires active search.
    No candidate/self-reference descriptor can substitute for gallery proof.
    """
    if not isinstance(assignment, dict) or not isinstance(metadata, dict):
        return False
    a, m = assignment, metadata
    geometry = a.get("reacquire_geometry") or {}
    similar = a.get("similar_follow") or {}
    competition = a.get("identity_competition") or m.get("identity_competition") or {}
    match = a.get("match_evidence") or {}
    if not all(isinstance(value, dict) for value in (geometry, similar, competition, match)):
        return False
    conflict_flags = ("search_excluded", "candidate_geometry_conflict", "identity_recheck_pending",
        "mapped_geometry_blocked", "search_contradiction_retained", "search_cross_edge_conflict",
        "short_handoff_identity_conflict")
    if (any(container.get(key) for container in (a, m, geometry) for key in conflict_flags)
            or m.get("search_direction_compatible") is False
            or (geometry.get("ok") is False and geometry.get("reason") not in {
                "stale_reference", "search_reacquire_time_window", "not_evaluated"})
            or (a.get("reacquire_geometry_ok") is False and a.get("reacquire_geometry_reason") not in {
                "stale_reference", "search_reacquire_time_window", "not_evaluated"})
            or a.get("reacquire_partial_state") in {"conflict", "mismatch"}
            or a.get("low_score_observation_blocked") is True
            or any(a.get(key) for key in ("bank_updated", "recent_bank_updated", "learning_written_tiers"))
            or a.get("learning_allowed") is True or similar.get("learning_allowed") is True):
        return False
    try:
        # Real rejected assignments retain this read-only match_evidence via
        # IdentityBank.assign's diagnostics merge; their top-level source may
        # be absent after the secondary gate rebuilt the UID0 assignment.
        source = match.get("match_source", a.get("match_source"))
        matched_uid = match.get("matched_uid", a.get("best_uid", a.get("mapped_uid")))
        distance = match.get("strong_distance", match.get("distance",
            a.get("strong_distance", a.get("distance"))))
        values = (*bbox, width, distance)
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in values):
            return False
        x1, y1, x2, y2 = bbox
        if not width > 0 or not 0 <= x1 < x2 <= width or not 0 <= y1 < y2:
            return False
        reasons = {item.strip() for item in str(m.get("quality_bbox_reason")
                   or m.get("bbox_quality_reason") or "").split(",") if item.strip()}
        same_edge = _same_search_edge(bbox, width, search_direction)
        crop_quality = (m.get("quality_bbox_ok") is False and m.get("bbox_quality_tier") == "weak"
                        and bool(reasons) and reasons <= {"edge_touch>2", "aspect<0.18"})
        clean_quality = (m.get("quality_bbox_ok") is True and m.get("bbox_quality_tier") == "strong"
                         and not reasons)
        unknown_secondary = (a.get("reason") == "secondary_evidence_unavailable"
                             and a.get("reacquire_partial_comparable") is False
                             and a.get("reacquire_partial_state") == "unknown")
        observation_only = unknown_secondary or _low_score_observation_only(a, m)
        frame = m.get("control_frame_id", m.get("frame_index"))
        source_index = m.get("source_detection_index")
        return bool(type(uid) is int and uid > 0 and type(output_uid) is int and output_uid == 0
            and a.get("uid", output_uid) == 0 and matched_uid == uid
            and a.get("mapped_uid", uid) in (None, 0, uid)
            and a.get("best_uid", uid) in (None, 0, uid)
            and observation_only
            and source == "strong" and 0 <= distance <= .30
            and m.get("is_fresh") is True and (crop_quality or clean_quality) and same_edge
            and competition.get("passed") is True and competition.get("uid") == uid
            and type(competition.get("candidate_count")) is int and competition["candidate_count"] == 1
            and type(frame) is int and frame > 0 and competition.get("frame_index") == frame
            and type(source_index) is int and source_index >= 0
            and competition.get("source_detection_index") == source_index)
    except (TypeError, ValueError, OverflowError):
        return False


def confirmed_observation_for_release(assignment, metadata, output_uid, uid):
    """Consume current bank confirmation; never infer UID from similarity."""
    a, m = assignment or {}, metadata or {}
    geometry = a.get("reacquire_geometry") or {}
    proof = a.get("identity_competition") or m.get("identity_competition") or {}
    return bool(uid is not None and output_uid == uid and a.get("uid") == uid
        and m.get("is_fresh") is True and m.get("quality_bbox_ok") is True
        and a.get("bbox_quality_ok") is True
        and a.get("reacquire_geometry_ok") is True
        and not a.get("identity_control_rejected") and not a.get("search_excluded")
        and not a.get("candidate_geometry_conflict")
        and a.get("reason") not in {"recent_partial_conflict", "mapped_geometry_reject",
            "identity_center_jump_reject", "search_candidate_identity_ambiguous",
            "search_candidate_excluded"}
        and not any(geometry.get(k) for k in ("mapped_geometry_blocked",
            "search_contradiction_retained", "search_cross_edge_conflict",
            "short_handoff_identity_conflict"))
        and geometry.get("ok") is not False
        and a.get("reacquire_partial_state") not in {"conflict", "mismatch"}
        and proof.get("passed") is True and proof.get("uid") == uid
        and m.get("source_detection_index") is not None
        and proof.get("source_detection_index") == m.get("source_detection_index")
        and proof.get("frame_index") is not None
        and proof.get("frame_index") == m.get("control_frame_id", m.get("frame_index")))


def retry_evidence_source(assignment, metadata, uid):
    """Qualify a bounded STOP observation, never a UID or movement grant.

    A partial match needs current reliable recent evidence and competition.
    Merely carrying a small partial_distance (possibly from an old template)
    is insufficient. Raw tracker IDs are deliberately not continuity keys.
    """
    a, m = assignment or {}, metadata or {}
    geometry = a.get("reacquire_geometry") or {}
    if (a.get("search_excluded") is True or a.get("candidate_geometry_conflict") is True
            or any(geometry.get(k) for k in (
                "mapped_geometry_blocked", "search_contradiction_retained",
                "search_cross_edge_conflict", "short_handoff_identity_conflict"))
            or a.get("reason") in {
                "recent_partial_conflict", "mapped_geometry_reject",
                "identity_center_jump_reject", "search_candidate_identity_ambiguous",
                "search_candidate_excluded"}):
        return None
    try:
        matched_uid = a.get("best_uid") or a.get("mapped_uid") or a.get("uid")
        if int(matched_uid or 0) != int(uid):
            return None
        # Unknown/non-comparable appearance can request STOP, never identity.
        # Continuity, uniqueness and the one-shot budget are checked by owner.
        if (a.get("reason") == "secondary_evidence_unavailable"
                and a.get("reacquire_partial_comparable") is False
                and m.get("quality_bbox_ok") is True
                and m.get("bbox_quality_tier") == "strong"
                and m.get("is_fresh") is True):
            proof = a.get("identity_competition") or m.get("identity_competition") or {}
            full = (a.get("template_recent_evidence") or {}).get("distance")
            if (proof.get("passed") is True and proof.get("candidate_count") == 1
                    and proof.get("uid") == uid
                    and proof.get("frame_index") == m.get("control_frame_id", m.get("frame_index"))
                    and m.get("source_detection_index") is not None
                    and proof.get("source_detection_index") == m.get("source_detection_index")
                    and full is not None and math.isfinite(float(full)) and 0 <= float(full) <= .55
                    and not (geometry.get("ok") is False and geometry.get("reason") not in
                             {"stale_reference", "search_reacquire_time_window"})):
                return "unverified_stop_only"
        if a.get("match_source") == "strong":
            distance = float(a.get("strong_distance", a.get("distance")))
            return "strong" if math.isfinite(distance) and 0 <= distance <= .30 else None
        if (a.get("match_source") != "partial"
                or a.get("identity_control_rejected") is True
                or a.get("reacquire_partial_state") != "match"
                or m.get("quality_bbox_ok") is not True
                or m.get("bbox_quality_tier") != "strong"
                or m.get("partial_feature_source") != "osnet_torso"
                or m.get("is_fresh") is not True):
            return None
        recent = a.get("reacquire_recent_partial_evidence") or {}
        distance = float(recent["distance"])
        limit = min(.34, float(a["reacquire_partial_confirm_limit"]))
        if not (recent.get("count", 0) > 0 and math.isfinite(distance)
                and math.isfinite(limit) and 0 <= distance <= limit):
            return None
        proof = a.get("identity_competition") or m.get("identity_competition") or {}
        frame = m.get("control_frame_id", m.get("frame_index"))
        if (frame is None or proof.get("frame_index") != frame
                or proof.get("uid") != uid or proof.get("passed") is not True
                or proof.get("source_detection_index") != m.get("source_detection_index")):
            return None
        if geometry.get("ok") is False and geometry.get("reason") != "late_candidate_observation":
            return None
        return "partial"
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


class SettledObservation:
    """Two distinct trustworthy quiet wheel samples, then a newer RGB capture."""

    def __init__(self):
        self.last_stamp = None
        self.quiet_count = 0
        self.ready_at = None

    def update(self, now, capture_timestamp, zero_sent_at, started_at, feedback):
        try:
            stamp = float(feedback.timestamp)
            left, right = float(feedback.left_forward_rpm), float(feedback.right_forward_rpm)
            zero = float(zero_sent_at)
            valid = (feedback.trustworthy is True and
                     all(math.isfinite(v) for v in (stamp,left,right,zero,now,capture_timestamp))
                     and started_at <= zero <= stamp <= now and now-stamp <= .15
                     and abs(left) <= 1.0 and abs(right) <= 1.0
                     and not getattr(feedback, 'left_error', 0)
                     and not getattr(feedback, 'right_error', 0))
        except (AttributeError, TypeError, ValueError):
            valid = False
        if not valid:
            self.quiet_count, self.ready_at = 0, None
            return False
        if self.last_stamp is not None and stamp < self.last_stamp:
            self.quiet_count, self.ready_at = 0, None
            return False
        if self.last_stamp is None or stamp > self.last_stamp:
            if self.last_stamp is not None and stamp-self.last_stamp > .15:
                self.quiet_count, self.ready_at = 0, None
            self.last_stamp = stamp
            self.quiet_count += 1
            if self.quiet_count >= 2 and self.ready_at is None:
                self.ready_at = stamp
        return bool(self.ready_at is not None
                    and capture_timestamp > self.ready_at
                    and capture_timestamp >= zero + .08)


class DetectorObservationSettlement:
    """Do not finish the ordinary two-frame pause on an image taken in motion."""

    def __init__(self):
        self.active = False
        self.settled = SettledObservation()

    def update(self, decision, *, now, capture_timestamp, capture_id, search_active,
               zero_sent_at, feedback, max_hold_sec, stale=False):
        if not search_active:
            self.active = False
            return decision
        if decision.reason == DEFERRED_EDGE_COVERAGE and not stale:
            # Release only this detector observation. This is not a settled
            # image or successful identity confirmation, and never touches
            # any action-runtime braking/safety latch.
            was_active = self.active
            self.active = False
            return replace(decision, entered=False, completed=was_active or decision.completed,
                           pause_rotation=False, preferred_target_match=False)
        if decision.entered and decision.pause_rotation and not self.active and not stale:
            self.active = True
            self.started_at = now
            self.deadline = now + min(.30, max(.10, max_hold_sec))
            self.seed = decision
            self.last_capture = (capture_id, capture_timestamp)
            self.settled = SettledObservation()
            return replace(decision, completed=False)
        if not self.active:
            return decision
        replay = decision.reason == "candidate_observation_duplicate_or_old"
        if not stale and not replay and (decision.bbox is None or not SearchObservationRetry.continuous(
                self.seed.bbox, decision.bbox)):
            self.active = False
            return replace(decision, entered=False, completed=True, pause_rotation=False,
                           bbox=None, preferred_target_match=False,
                           reason='observation_candidate_lost_or_changed')
        fresh = (not stale and not replay and 0 <= now-capture_timestamp <= .19 and capture_id > self.last_capture[0]
                 and capture_timestamp > self.last_capture[1])
        ready = self.settled.update(now, capture_timestamp, zero_sent_at, self.started_at, feedback)
        if fresh:
            self.last_capture = (capture_id, capture_timestamp)
        complete = now >= self.deadline or (fresh and ready)
        self.active = not complete
        reason = ('observation_timeout_unsettled' if now >= self.deadline else
                  'observation_settled_capture' if complete else 'observation_await_settled_capture')
        result = replace(self.seed, entered=False, completed=complete,
                         pause_rotation=not complete, reason=reason,
                         bbox=self.seed.bbox if replay else decision.bbox,
                         score=self.seed.score if replay else decision.score,
                         preferred_target_match=False)
        if fresh:
            self.seed = result
        return result


class SearchObservationRetry:
    """Observe a continuous credible candidate that exhausted the detector gate.

    Entry requires two distinct capture timestamps with local box continuity.
    The budget is spent once per (search epoch, UID), even if a track changes or
    a candidate vanishes. Completion needs a capture after a successful motor
    zero write and two quiet wheel samples; timeout never claims a target.
    """

    def __init__(self, max_hold_sec: float = 0.30) -> None:
        self.max_hold_sec = max(.10, min(.30, float(max_hold_sec)))
        self.reset()

    def reset(self) -> None:
        self.session = None
        self.spent = False
        self.active = False
        self.started_at = 0.0
        self.deadline = 0.0
        self.previous = None
        self.last_capture = None
        self.bbox = None
        self.score = 0.0
        self.reason = "inactive"
        self.settled = SettledObservation()

    def release(self) -> None:
        self.active = False
        self.previous = None

    def spend_detector_budget(self, session):
        """The ordinary detector look already spent this search's pause.

        A credible retry is a fallback when no ordinary look ran, not a
        second independent 300ms stop for the same search/UID.
        """
        if session is None:
            return
        if session != self.session:
            self.reset()
            self.session = session
        self.spent = True

    def confirmed_release(self, session, bbox, score):
        self.spend_detector_budget(session)
        self.active = False
        self.previous = None
        self.bbox, self.score = bbox, score
        return self._decision("identity_confirmed", completed=True)

    @staticmethod
    def continuous(first: BBox, second: BBox) -> bool:
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in (first, second)]
        return bool(min(areas) > 0 and min(areas) / max(areas) >= .55
                    and SearchCandidateGate._iou(first, second) >= .35)

    def _decision(self, reason: str, *, entered=False, completed=False):
        self.reason = reason
        return SearchCandidateGateDecision(
            pause_rotation=not completed, entered=entered, completed=completed,
            source="credible_retry", reason="search_retry_" + reason,
            bbox=self.bbox, score=self.score,
            hold_frame=2 if completed else 1, hold_frames=2,
        )

    def update(self, *, now: float, session, capture_id: int, capture_timestamp: float,
               eligible: bool, bbox: Optional[BBox], score: float,
               blocked: bool, zero_sent_at: Optional[float] = None, feedback=None):
        if session is None:
            self.reset()
            return None
        if session != self.session:
            self.reset()
            self.session = session
        if self.active and now >= self.deadline:
            self.active = False
            return self._decision("timeout", completed=True)
        fresh = bool(math.isfinite(capture_timestamp) and 0 <= now - capture_timestamp <= .19
                     and (self.last_capture is None or (
                         capture_id > self.last_capture[0] and capture_timestamp > self.last_capture[1])))
        if not fresh:
            return self._decision("await_new_capture") if self.active else None
        self.last_capture = (capture_id, capture_timestamp)
        if not eligible or bbox is None:
            self.previous = None
            if self.active:
                self.active = False
                return self._decision("candidate_lost_or_ambiguous", completed=True)
            self.reason = "no_credible_candidate"
            return None
        if self.active:
            if not self.continuous(self.bbox, bbox):
                self.active = False
                return self._decision("candidate_changed", completed=True)
            self.bbox, self.score = bbox, score
            post_zero = self.settled.update(now, capture_timestamp, zero_sent_at,
                                           self.started_at, feedback)
            if post_zero:
                self.active = False
                return self._decision("post_settled_capture", completed=True)
            return self._decision("await_settled_capture")
        continuous = bool(self.previous is not None
                          and 0 < capture_timestamp - self.previous[0] <= .25
                          and self.continuous(self.previous[1], bbox))
        self.previous = (capture_timestamp, bbox)
        if self.spent or not blocked or not continuous:
            self.reason = "budget_spent" if self.spent else "await_continuity"
            return None
        self.spent = self.active = True
        self.bbox, self.score = bbox, score
        self.started_at, self.deadline = now, now + self.max_hold_sec
        return self._decision("start", entered=True)
