"""One immutable physical sample's brake assessment, never motion authority.

Past travel comes only from measured/completed motion. A proposed speed may
reserve future response time, but cannot be charged as travel before it exists.
The same assessment serves fresh PI and the admitted command's final writer.
"""
from dataclasses import dataclass
import math

from .depth_authority_timing import MAX_FORWARD_DEPTH_TTL_SEC


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sample_feedback_time_valid(sample_timestamp, feedback_timestamp, now, *,
                               current_feedback_independent=False):
    """Validate two physical clocks, not a synthetic paired measurement.

    Pure-distance braking uses CURRENT own-wheel speed, not an ego endpoint
    for estimating target motion. A newer, independently fresh encoder report
    must not fail merely because Depth took >150ms to process. Older encoder
    reports retain the 150ms pre-Depth limit. Past travel is still charged
    from Depth capture using the assessment's immutable execution bound.
    """
    return bool(type(current_feedback_independent) is bool
        and all(map(_finite, (sample_timestamp, feedback_timestamp, now)))
        and 0 <= now-sample_timestamp <= .18
        and 0 <= now-feedback_timestamp <= .15
        and (sample_timestamp-feedback_timestamp <= .15
             if current_feedback_independent else
             abs(feedback_timestamp-sample_timestamp) <= .15))


@dataclass(frozen=True)
class SampleBrakeBudget:
    cap_rpm: float
    margin_m: float
    required_stop_m: float
    reason: str


@dataclass(frozen=True)
class SampleBrakingAssessment:
    uid: int
    sample_timestamp: float
    checked_at: float
    distance_m: float
    travel_bound_rpm: float
    outer_rpm: float
    feedback_timestamp: float
    target_speed_m_s: float
    stop_distance_m: float
    circumference_m: float
    deceleration_m_s2: float
    response_delay_sec: float
    max_rpm: float
    feedback_reserve_sec: float = .15
    outer_allowance_rpm: float = 0.
    # Explicit trial: use the admission feedback's real age (floor one 50ms
    # execution period), frozen for this physical sample. Later feedback can
    # veto but cannot refund this reserve or increase an admitted command.
    observed_feedback_reserve: bool = False
    # Opt-in only with a completed-command history covering sample->checked.
    # The caller freezes that interval's bound in travel_bound_rpm, including
    # high commands retired AFTER this sample. This is not motor authority or
    # a claim that commanded RPM bounds physical speed on every surface.
    feedback_interval_covered: bool = False
    # Pure-distance mode only. This is not target-motion synchronization,
    # extended feedback lifetime, or permission to refund earlier travel.
    current_feedback_independent: bool = False

    def __post_init__(self):
        values = (self.sample_timestamp, self.checked_at, self.distance_m,
                  self.travel_bound_rpm, self.outer_rpm, self.feedback_timestamp,
                  self.target_speed_m_s, self.stop_distance_m, self.circumference_m,
                  self.deceleration_m_s2, self.response_delay_sec, self.max_rpm,
                  self.feedback_reserve_sec, self.outer_allowance_rpm)
        if (type(self.uid) is not int or self.uid <= 0 or not all(map(_finite, values))
                or min(self.sample_timestamp, self.distance_m, self.stop_distance_m,
                       self.circumference_m, self.deceleration_m_s2, self.max_rpm) <= 0
                or not sample_feedback_time_valid(
                    self.sample_timestamp, self.feedback_timestamp, self.checked_at,
                    current_feedback_independent=self.current_feedback_independent)
                or not 0 <= self.outer_rpm <= self.travel_bound_rpm <= self.max_rpm+5.
                or self.target_speed_m_s > 0 or self.response_delay_sec < 0
                or not 0 <= self.outer_allowance_rpm <= self.max_rpm
                or self.feedback_reserve_sec != .15
                or type(self.observed_feedback_reserve) is not bool
                or type(self.feedback_interval_covered) is not bool
                or (self.current_feedback_independent and self.target_speed_m_s != 0.)
                or (self.feedback_interval_covered and (
                    not self.observed_feedback_reserve or self.target_speed_m_s != 0.))):
            raise ValueError("invalid sample braking assessment")

    @property
    def effective_feedback_reserve_sec(self):
        if self.feedback_interval_covered:
            # sample->checked is already charged using the proven interval
            # bound. A feedback sample captured near Depth must not charge
            # that same processing delay a second time. Keep the pre-Depth
            # lag, a full execution-period floor, and the separate response
            # delay. This policy is fixed at admission, never switched on by
            # later low feedback or by changing an old grant's timestamp.
            return max(.05, self.sample_timestamp-self.feedback_timestamp)
        return (max(.05, self.checked_at-self.feedback_timestamp)
                if self.observed_feedback_reserve else self.feedback_reserve_sec)

    def valid_for(self, uid, stamp):
        return type(uid) is int and uid == self.uid and stamp == self.sample_timestamp

    def _root(self, space, delay):
        a = self.deceleration_m_s2
        return a*(math.sqrt(delay*delay+2*max(0., space)/a)-delay)

    def budget(self, now, outer_rpm, *, authorized_rpm=None, completed_rpm=None,
               execution_bound_rpm=None, tightened_distance_m=None):
        """Tighten one sample without renewing it or gaining acceleration credit.

        ``authorized_rpm=None`` computes a fresh candidate ceiling. Once a
        command exists, ``authorized_rpm`` is its *current* command ceiling.
        ``execution_bound_rpm`` carries the immutable originally admitted outer
        wheel/travel bound. Lowering the command cannot make an earlier legal
        write illegal, or refund its accumulated travel. Omitting the execution
        bound retains the legacy single-ceiling behavior.

        A closer late distance may tighten space, but cannot replace this
        sample's identity, timestamp, or original admission calculation.
        """
        if (not all(map(_finite, (now, outer_rpm)))
                or not self.checked_at <= now <= self.sample_timestamp+MAX_FORWARD_DEPTH_TTL_SEC
                or not 0 <= outer_rpm <= self.max_rpm+5.
                or (completed_rpm is not None and (not _finite(completed_rpm)
                    or not 0 <= completed_rpm <= self.max_rpm+5.))
                or (authorized_rpm is not None and (not _finite(authorized_rpm)
                    or not 0 <= authorized_rpm <= self.max_rpm))
                or (execution_bound_rpm is not None and (
                    authorized_rpm is None or not _finite(execution_bound_rpm)
                    or execution_bound_rpm < 0))
                or (tightened_distance_m is not None and (
                    not _finite(tightened_distance_m)
                    or not 0 < tightened_distance_m <= self.distance_m))):
            return SampleBrakeBudget(0., 0., 0., "invalid_shared_braking_evidence")
        scale = self.circumference_m/60.
        target = self.target_speed_m_s
        original = self.travel_bound_rpm*scale
        initial_age = self.checked_at-self.sample_timestamp
        reserve = self.effective_feedback_reserve_sec
        past_travel = max(0., original-target)*initial_age
        initial_margin = (self.distance_m-self.stop_distance_m-past_travel
                          -original*reserve-.02)
        # First solve under the already-existing speed bound. If a candidate
        # exceeds it, reserve its extra feedback uncertainty as FUTURE delay,
        # rather than pretending it travelled since Depth capture.
        candidate_speed = max(0., target+self._root(initial_margin, self.response_delay_sec))
        if candidate_speed > original:
            candidate_speed = max(0., target+self._root(
                initial_margin+reserve*(original-target),
                self.response_delay_sec+reserve))
        # The longitudinal proposal is the MEAN wheel speed. Reserve its
        # allowed outer-wheel steering correction only for FUTURE motion;
        # hypothetical yaw must not be charged to already travelled space.
        initial_cap = max(0., min(self.max_rpm,
                                  candidate_speed/scale-self.outer_allowance_rpm))
        # Admission is checked against the original sample, not against a later
        # reduction of either distance or command. The latter must yield a
        # smaller command rather than retroactively invalidating legal history.
        distance_reduction = (0. if tightened_distance_m is None else
                              self.distance_m-tightened_distance_m)
        if authorized_rpm is None:
            # The fresh call uses the exact snapshot clock. Any later caller
            # must carry the admitted command, not invent a new candidate.
            if now != self.checked_at or (completed_rpm is not None
                    and completed_rpm > self.travel_bound_rpm+1e-9):
                return SampleBrakeBudget(0., initial_margin, 0., "shared_braking_plan_changed")
            fresh_margin = initial_margin-distance_reduction
            if distance_reduction:
                candidate_speed = max(0., target+self._root(
                    fresh_margin, self.response_delay_sec))
                if candidate_speed > original:
                    candidate_speed = max(0., target+self._root(
                        fresh_margin+reserve*(original-target),
                        self.response_delay_sec+reserve))
                initial_cap = max(0., min(self.max_rpm,
                    candidate_speed/scale-self.outer_allowance_rpm))
            future_bound = max(original, (initial_cap+self.outer_allowance_rpm)*scale)
            margin = fresh_margin-(future_bound-original)*reserve
            cap = initial_cap
        else:
            if authorized_rpm > initial_cap+1e-9:
                return SampleBrakeBudget(0., initial_margin, 0., "shared_braking_request_exceeds_plan")
            future_bound = max(original, (authorized_rpm+self.outer_allowance_rpm)*scale)
            if execution_bound_rpm is not None:
                admitted_limit = max(self.travel_bound_rpm,
                                     initial_cap+self.outer_allowance_rpm)
                if (execution_bound_rpm*scale < future_bound-1e-9
                        or execution_bound_rpm > admitted_limit+1e-9):
                    return SampleBrakeBudget(0., initial_margin, 0.,
                                             "invalid_shared_braking_execution_bound")
                future_bound = execution_bound_rpm*scale
            if completed_rpm is not None and completed_rpm*scale > future_bound+1e-9:
                return SampleBrakeBudget(0., initial_margin, 0., "shared_braking_new_higher_write")
            margin = (initial_margin-distance_reduction
                      -(future_bound-original)*reserve
                      -max(0., future_bound-target)*(now-self.checked_at))
            cap = min(authorized_rpm, initial_cap, max(0.,
                (target+self._root(margin, self.response_delay_sec))/scale-self.outer_allowance_rpm))
        closing = max(0., outer_rpm*scale-target)
        required = closing*self.response_delay_sec+closing*closing/(2*self.deceleration_m_s2)
        if margin <= 0 or required > margin+1e-9:
            return SampleBrakeBudget(0., margin, required, "shared_braking_momentum")
        return SampleBrakeBudget(cap, margin, required, "shared_braking_cap")
