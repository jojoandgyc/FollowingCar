"""Sample-driven forward PI with a separate relative-motion braking bound.

This module owns neither sensors nor motor authorization. Keeping controller
memory never grants motion: the caller must still enforce the physical Depth
deadline and pass an ego speed only when its encoder evidence is synchronized
and fresh. Deceleration is an experimental model, not a guaranteed stop bound.
"""
from dataclasses import dataclass, replace
import math
from typing import Optional
from .depth_continuation import relative_braking_budget
from .depth_authority_timing import MAX_FORWARD_DEPTH_TTL_SEC
from .longitudinal_approach import RawDepthMotionEvidence
from .longitudinal_execution import ForwardExecutionAnchor, ForwardRecoveryAnchor
from .sample_braking import SampleBrakingAssessment


def execution_continuity_reason_allowed(reason):
    """Withdrawals that can retain a verified, still-current motor receipt.

    This is narrower than permission for a fresh low-speed restart. A new
    depth/feedback assessment and an unchanged completed write are mandatory;
    missing feedback never extends the withdrawn grant itself.
    """
    return reason in {
        "physical_depth_expired", "no_live_grant_before_pi",
        "lateral_depth:continuation_feedback_stale",
        "lateral_depth:continuation_feedback_unavailable",
    }


def fresh_grant_recovery_reason_allowed(reason):
    """Continuity withdrawals only; a new sample cannot relabel a safety stop."""
    return reason in {
        "no_live_grant_before_pi", "depth_dispatch_budget",
        "late_depth_no_live_authority", "late_depth_continuation_feedback_invalid",
        # These are exact read-time reasons, bound to the withdrawn physical
        # sample by the producer. They permit ONLY a newly qualified sample's
        # existing bounded restart, never continuation of the stopped grant.
        "lateral_depth:visibility_expired",
        "lateral_depth:continuation_feedback_stale",
        "lateral_depth:continuation_feedback_unavailable",
        "lateral_zero_no_qualified_depth:revoke:expired_waiting_control_lock",
        "lateral_zero_no_qualified_depth:revoke:expired",
        "lateral_zero_no_qualified_depth:visual_pid_center_hold",
        "lateral_zero_no_qualified_depth:visual_pid_image_yaw_zero",
        *{"lateral_zero_no_qualified_depth:pid_zero:visual_pid_"+side+"_"+source
          for side in ("left", "right") for source in ("encoder", "camera")},
    }


def fresh_identity_restart_reason(reason):
    """Observation loss requiring NEW identity and quiet-wheel evidence.

    These reasons do not qualify receipt continuity or revive a prior lease.
    The owning controller must obtain a current identity/control proof before
    supplying even a bounded fresh-grant ramp step. In particular, feedback
    invalidity can describe real rotation, not just a delayed encoder sample.
    """
    return reason in {"stale_vision_result", "lateral_depth:continuation_feedback_invalid"}


@dataclass(frozen=True)
class DistancePiConfig:
    kp_per_sec: float = 1.0
    ki_per_sec2: float = .4
    integral_max_m_s: float = .8
    wheel_circumference_m: float = .816814
    deceleration_m_s2: float = .4
    response_delay_sec: float = .2
    # The longer physical lease is only for the caller's bounded continuation.
    # It must never silently extend the age permitted for a new PI update.
    physical_ttl_sec: float = .18
    fresh_update_max_age_sec: float = .18
    max_integration_gap_sec: float = .18
    retain_integral_sec: float = .35
    stationary_speed_m_s: float = .03
    stationary_confirm_samples: int = 3
    # Opt-in motor-target experiment; never a floor on the braking cap.
    launch_request_rpm: float = 0.
    # 0 preserves the original constant-request experiment. Positive values
    # taper the extra far-chase demand with effective distance error, and use
    # the normal rise budget. This is NOT a new floor on approved motor RPM.
    launch_full_error_m: float = 0.
    # Opt-in memory of RELATIVE motion, not motor authorization or a human
    # moving/stopped classifier. Never refreshed by degraded/duplicate samples.
    motion_memory_sec: float = 0.
    # Optional new-sample preview. Assume the person can stop immediately:
    # measured walking speed must not loosen this stopping envelope. Optional
    # extra headroom is separate from that physical stopping constraint.
    stationary_stop_preview_enabled: bool = False
    stationary_stop_preview_distance_m: float = 1.2
    stationary_stop_preview_headroom_rpm: float = 0.
    # Compatibility default. The production pure-distance mode uses only
    # current distance and actual wheel response for its physical brake bound.
    use_target_motion: bool = True
    observed_feedback_reserve: bool = False
    feedback_interval_deduplication: bool = False

    def __post_init__(self):
        if type(self.use_target_motion) is not bool:
            raise ValueError("invalid target motion control switch")
        if type(self.observed_feedback_reserve) is not bool:
            raise ValueError("invalid feedback reserve switch")
        if type(self.feedback_interval_deduplication) is not bool:
            raise ValueError("invalid feedback interval switch")
        if (isinstance(self.launch_full_error_m, bool)
                or not math.isfinite(self.launch_full_error_m)
                or not 0 <= self.launch_full_error_m <= 2.):
            raise ValueError("invalid distance PI launch_full_error_m")
        if (isinstance(self.motion_memory_sec, bool) or not math.isfinite(self.motion_memory_sec)
                or not 0 <= self.motion_memory_sec <= .35):
            raise ValueError("invalid distance PI motion_memory_sec")
        if type(self.stationary_stop_preview_enabled) is not bool:
            raise ValueError("invalid distance PI stationary stop preview switch")
        if (isinstance(self.stationary_stop_preview_distance_m, bool)
                or not math.isfinite(self.stationary_stop_preview_distance_m)
                or self.stationary_stop_preview_distance_m <= 0):
            raise ValueError("invalid distance PI stationary stop preview distance")
        if (isinstance(self.stationary_stop_preview_headroom_rpm, bool)
                or not math.isfinite(self.stationary_stop_preview_headroom_rpm)
                or not 0 <= self.stationary_stop_preview_headroom_rpm <= 25.):
            raise ValueError("invalid distance PI preview headroom")
        if (isinstance(self.launch_request_rpm, bool)
                or not math.isfinite(self.launch_request_rpm)
                or not 0 <= self.launch_request_rpm <= 200.):
            raise ValueError("invalid distance PI launch_request_rpm")
        for name, ceiling in (("kp_per_sec", 3.), ("ki_per_sec2", 2.),
                              ("integral_max_m_s", 1.5), ("wheel_circumference_m", 3.),
                              ("deceleration_m_s2", 1.), ("response_delay_sec", 1.),
                              ("physical_ttl_sec", MAX_FORWARD_DEPTH_TTL_SEC), ("fresh_update_max_age_sec", .18),
                              ("max_integration_gap_sec", .18),
                              ("retain_integral_sec", 1.), ("stationary_speed_m_s", .1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= ceiling:
                raise ValueError(f"invalid distance PI {name}: {value}")
        if self.retain_integral_sec < self.max_integration_gap_sec:
            raise ValueError("PI retention must cover the maximum integration interval")
        if (isinstance(self.stationary_confirm_samples, bool)
                or not isinstance(self.stationary_confirm_samples, int)
                or not 2 <= self.stationary_confirm_samples <= 10):
            raise ValueError("invalid PI stationary confirmation count")


@dataclass(frozen=True)
class DistancePiResult:
    sample_timestamp: float
    error_m: float
    p_m_s: float = 0.
    integral_m_s: float = 0.
    output_rpm: float = 0.
    unslewed_output_rpm: float = 0.
    cap_rpm: float = 0.
    closing_m_s: float = 0.
    braking_distance_m: float = 0.
    delay_sec: float = 0.
    sample_dt_sec: float = 0.
    status: str = "uninitialized"
    brake_source: str = "stationary_fallback"
    slew_limited: bool = False
    integral_frozen: bool = True
    pi_demand_rpm: float = 0.
    launch_floor_rpm: float = 0.
    demand_rpm: float = 0.
    software_rise_bypassed: bool = False
    demand_limit_reason: str = "none"
    motion_origin_ts: Optional[float] = None
    motion_uncertainty_m_s: float = 0.
    brake_recovery_limited: bool = False
    brake_recovery_anchor_rpm: float = 0.
    memory_time_penalty_m_s: float = 0.
    memory_rotation_penalty_m_s: float = 0.
    memory_retained_rate_m_s: Optional[float] = None
    memory_endpoint_rate_m_s: Optional[float] = None
    effective_range_rate_m_s: float = 0.
    target_velocity_bound_m_s: float = 0.
    braking_distance_input_m: float = 0.
    brake_settling_limited: bool = False
    brake_settling_anchor_rpm: float = 0.
    # Request-stage provenance, not a new speed limit or motor permission.
    # demand_limit_reason is retained for older log parsers; the binding
    # reason below distinguishes a tighter recovery ramp from the envelope.
    final_limit_reason: str = "not_evaluated"
    pre_settling_cap_rpm: float = 0.
    execution_anchor_rpm: float = 0.
    execution_ramp_dt_sec: float = 0.
    ramp_output_rpm: float = 0.
    brake_recovery_cap_rpm: Optional[float] = None
    pre_quantization_rpm: float = 0.
    motion_window_used: bool = False
    motion_window_target_speed_m_s: Optional[float] = None
    motion_window_span_sec: float = 0.
    motion_window_range_rate_m_s: Optional[float] = None
    depth_expiry_recovery_step_sec: float = 0.
    depth_expiry_recovery_used: bool = False
    depth_expiry_completed_anchor_used: bool = False
    depth_expiry_completed_anchor_rpm: float = 0.
    brake_settling_uncertainty_released: bool = False
    memory_endpoint_fallback: bool = False
    memory_endpoint_cap_rpm: float = 0.
    fresh_grant_recovery_step_sec: float = 0.
    fresh_grant_recovery_used: bool = False
    stationary_preview_status: str = "disabled"
    stationary_preview_cap_rpm: Optional[float] = None
    stationary_preview_loss_rpm: float = 0.
    stationary_preview_margin_m: Optional[float] = None
    stationary_preview_required_stop_m: Optional[float] = None
    brake_settling_preview_released: bool = False
    braking_assessment: Optional[SampleBrakingAssessment] = None
    execution_recovery_anchor_used: bool = False


class DistancePiController:
    def __init__(self, config: DistancePiConfig):
        self.config = config
        self.reset()

    def reset(self):
        # Parking is an external execution state, not estimator history.
        self._normal_parking = getattr(self, "_normal_parking", False)
        self.integral_m_s = 0.
        self._last_sample_ts = None
        self._last_execution_ts = None
        self._last_target = None
        self._last_output_rpm = 0.
        self._integral_before_update = 0.
        self._approved_rpm = 0.
        self._sample_requested_rpm = 0.
        self._sample_rejected = False
        self._execution_suspended = False
        self._stationary_samples = 0
        self.last_result = None
        self._motion_memory = None
        self._brake_recovery_pending = False
        self._last_raw_distance_m = None
        self._expiry_recovery_forbidden = self._normal_parking
        self._execution_continuity_forbidden = self._normal_parking
        self._uncertain_zero_brake_anchor = False
        self._preview_zero_brake_anchor = False
        self._fresh_grant_recovery_forbidden = self._normal_parking
        self._last_ego_forward_rpm = 0.

    def set_normal_parking(self, active: bool):
        """Preview fresh distance while held; no ramp credit or old grant.

        The runtime still owns parking release and depth admission. On release,
        a new request uses the current brake envelope and measured wheel motion,
        not the unexecuted preview's recovery ramp.
        """
        if self._normal_parking == bool(active):
            return
        self._normal_parking = bool(active)
        self._brake_recovery_pending = False
        self._motion_memory = None
        self._execution_suspended = True
        self._last_output_rpm = self._approved_rpm = 0.
        self._stationary_samples = 0
        self._expiry_recovery_forbidden = True
        self._execution_continuity_forbidden = True
        self._uncertain_zero_brake_anchor = False
        self._preview_zero_brake_anchor = False
        self._fresh_grant_recovery_forbidden = True

    def invalidate_motion_memory(self):
        if not self.config.use_target_motion:
            self._motion_memory = None
            return  # A diagnostic estimator reset is not an execution event.
        if (not self._normal_parking and self.config.motion_memory_sec > 0 and self.last_result is not None
                and self.last_result.brake_source in {
                    "raw_relative_motion", "relative_motion_memory", "raw_endpoint_distance_bound"}):
            self._brake_recovery_pending = True
        self._motion_memory = None
        self._uncertain_zero_brake_anchor = False
        self._preview_zero_brake_anchor = False

    def suspend(self, now: float, reason: str, *, retain: bool = True,
                reset_execution: bool = False):
        """Pause without updating any evidence timestamp or motor deadline."""
        if not math.isfinite(now):
            raise ValueError("PI suspension time must be finite")
        if (not retain or (self._last_sample_ts is not None
                           and now-self._last_sample_ts > self.config.retain_integral_sec)):
            self.integral_m_s = 0.
            self._stationary_samples = 0
        if reset_execution or not retain:
            self._execution_suspended = True
            self._stationary_samples = 0
            self._preview_zero_brake_anchor = False
            if reason not in {"physical_depth_expired", "no_live_grant_before_pi"}:
                self._expiry_recovery_forbidden = True
                self._uncertain_zero_brake_anchor = False
            if (reason != "physical_depth_expired"
                    and not fresh_grant_recovery_reason_allowed(reason)
                    and not (not self.config.use_target_motion and fresh_identity_restart_reason(reason))):
                self._fresh_grant_recovery_forbidden = True
            if not retain or not execution_continuity_reason_allowed(reason):
                self._execution_continuity_forbidden = True
        if not retain:
            self.invalidate_motion_memory()

    def accept_output_limit(self, sample_timestamp: float, approved_rpm: float,
                            *, quantization_rpm: float = 0.) -> bool:
        """Apply successively tighter final limits once, including quantization.

        Each approval refers to this sample's pre-integration state. Repeated
        or less restrictive approvals cannot accumulate a second correction or
        revive a previously withdrawn command.
        """
        r = self.last_result
        if (r is None or sample_timestamp != self._last_sample_ts
                or not math.isfinite(approved_rpm) or approved_rpm < 0
                or not math.isfinite(quantization_rpm) or quantization_rpm < 0
                or approved_rpm >= self._approved_rpm-1e-9):
            return False
        self._approved_rpm = approved_rpm
        self._last_output_rpm = approved_rpm
        if approved_rpm == 0:
            # An external withdrawal is not an estimator uncertainty stop.
            self._uncertain_zero_brake_anchor = False
        approved_speed = approved_rpm*self.config.wheel_circumference_m/60.
        # The extra launch/chase request can be above P+I. Limiting only that
        # extra demand does not saturate the PI: retain its new compensation
        # while the approved command still covers P+I. Use the current memory
        # because an earlier slew/approval may already have rolled it back.
        pi_rpm = max(0., r.p_m_s+self.integral_m_s)*60./self.config.wheel_circumference_m
        material_limit = (self._sample_requested_rpm-approved_rpm > quantization_rpm+1e-9
                          and pi_rpm > approved_rpm+1e-9)
        if material_limit:
            self.integral_m_s = min(self.integral_m_s, self._integral_before_update,
                                   max(0., approved_speed))
        self.last_result = replace(r, output_rpm=approved_rpm,
                                   integral_m_s=self.integral_m_s,
                                   integral_frozen=r.integral_frozen or material_limit)
        return True

    def retain_execution_anchor(self, sample_timestamp, rpm, sent_at, now):
        """Bound the next fresh calculation by a recently completed packet.

        Does not undo suspension, change a sample timestamp, or grant motion.
        Unsent requested speed and time before that packet are not ramp credit.
        """
        if (self._execution_suspended or self._normal_parking
                or self._last_execution_ts is None
                or sample_timestamp != self._last_sample_ts
                or not all(math.isfinite(v) for v in (sample_timestamp, rpm, sent_at, now))
                or rpm <= 0 or not sample_timestamp <= sent_at <= now
                or not 0 <= now-sent_at <= .10
                or not 0 <= now-sample_timestamp <= self.config.physical_ttl_sec):
            return False
        self.accept_output_limit(sample_timestamp, rpm)
        self._last_execution_ts = max(self._last_execution_ts, sent_at)
        return True

    def reject_output(self, sample_timestamp: float) -> bool:
        """Reject this request as an execution anchor, once per sample.

        Roll back only a positive integral increment. Even when no increment
        exists, an unissued request cannot become the next acceleration origin.
        A caller may still approve a smaller braking command on the old grant.
        """
        if (self.last_result is None or sample_timestamp != self._last_sample_ts
                or self._sample_rejected):
            return False
        self._sample_rejected = True
        self._execution_suspended = True
        self._expiry_recovery_forbidden = True
        self._execution_continuity_forbidden = True
        self._uncertain_zero_brake_anchor = False
        self._preview_zero_brake_anchor = False
        self._stationary_samples = 0
        self.integral_m_s = min(self.integral_m_s, self._integral_before_update)
        self.last_result = replace(self.last_result, integral_m_s=self.integral_m_s,
                                   integral_frozen=True)
        return True

    def update(self, actual_distance_m: float, target_distance_m: float, *,
               sample_timestamp: float, execution_now: float, deadband_m: float,
               max_output_rpm: float, rise_rpm_per_sec: float = 0.,
               fall_rpm_per_sec: float = 0., ego_forward_rpm: Optional[float] = None,
               preview_outer_forward_rpm: Optional[float] = None,
               preview_feedback_timestamp: Optional[float] = None,
               preview_completed_rpm: Optional[float] = None,
               braking_assessment: Optional[SampleBrakingAssessment] = None,
               range_rate_m_s: Optional[float] = None, raw_closure_valid: bool = False,
               raw_motion_evidence: Optional[RawDepthMotionEvidence] = None,
               allow_motion_memory: bool = False,
               allow_motion_memory_endpoint_fallback: bool = False,
               motion_memory_rotation_bound: float = .25,
               raw_distance_m: Optional[float] = None,
               measurement_jump_clamped: bool = False,
               depth_expiry_recovery_step_sec: float = 0.,
               depth_expiry_execution_anchor: Optional[ForwardExecutionAnchor] = None,
               depth_expiry_expected_uid: Optional[int] = None,
               fresh_grant_recovery_step_sec: float = 0.,
               execution_recovery_proof: Optional[ForwardRecoveryAnchor] = None,
               execution_recovery_uid: Optional[int] = None,
               ) -> DistancePiResult:
        c = self.config
        if not c.use_target_motion:
            # Do this before validating optional estimator inputs. None, NaN,
            # wrong-window or oscillating estimates are diagnostics only and
            # must not alter demand, integration, settling or recovery.
            range_rate_m_s = raw_motion_evidence = None
            raw_closure_valid = allow_motion_memory = False
            allow_motion_memory_endpoint_fallback = False
            motion_memory_rotation_bound = .25
            self._motion_memory = None
            self._brake_recovery_pending = False
        if (isinstance(depth_expiry_recovery_step_sec, bool)
                or not math.isfinite(depth_expiry_recovery_step_sec)
                or not 0 <= depth_expiry_recovery_step_sec <= .15):
            raise ValueError('invalid depth expiry recovery step')
        if (isinstance(fresh_grant_recovery_step_sec, bool)
                or not math.isfinite(fresh_grant_recovery_step_sec)
                or not 0 <= fresh_grant_recovery_step_sec <= .05):
            raise ValueError('invalid fresh grant recovery step')
        if (not math.isfinite(motion_memory_rotation_bound)
                or not .25 <= motion_memory_rotation_bound <= .35):
            raise ValueError('invalid retained-motion rotation bound')
        values = (actual_distance_m, target_distance_m, sample_timestamp, execution_now,
                  deadband_m, max_output_rpm, rise_rpm_per_sec, fall_rpm_per_sec)
        if (not all(math.isfinite(v) for v in values) or actual_distance_m <= 0
                or target_distance_m <= 0 or min(values[4:]) < 0):
            raise ValueError("invalid distance PI sample or limits")
        error = actual_distance_m-target_distance_m
        age = execution_now-sample_timestamp
        if age < -1e-9 or age > c.physical_ttl_sec+1e-9:
            self.suspend(execution_now, "stale_sample", reset_execution=True)
            return DistancePiResult(sample_timestamp, error, integral_m_s=self.integral_m_s,
                                    status="stale_sample")
        if age > c.fresh_update_max_age_sec+1e-9:
            # Zero here means SKIP, never a replacement motor grant. Only the
            # outer runtime may continue a still-valid original grant within
            # its unchanged deadline. A late sample cannot start the car,
            # advance the physical/PI timeline, or donate acceleration time.
            return DistancePiResult(sample_timestamp, error, integral_m_s=self.integral_m_s,
                                    status="continuation_only")
        if self._last_sample_ts is not None and sample_timestamp <= self._last_sample_ts:
            if sample_timestamp == self._last_sample_ts and not self._execution_suspended:
                return replace(self.last_result, status="duplicate", sample_dt_sec=0.,
                               integral_frozen=True)
            return DistancePiResult(sample_timestamp, error, integral_m_s=self.integral_m_s,
                                    status="out_of_order" if sample_timestamp < self._last_sample_ts
                                    else "suspended_duplicate")
        if self._last_execution_ts is not None and execution_now < self._last_execution_ts:
            return DistancePiResult(sample_timestamp, error, integral_m_s=self.integral_m_s,
                                    status="execution_out_of_order")
        if measurement_jump_clamped:
            # Rejection is not a new distance anchor: a repeated background
            # return must not become accepted merely by appearing twice here.
            self.suspend(execution_now, "measurement_jump", reset_execution=True)
            self.invalidate_motion_memory()
            return DistancePiResult(sample_timestamp, error, integral_m_s=self.integral_m_s,
                                    status="measurement_jump")
        if self._last_target is not None and target_distance_m != self._last_target:
            self.reset()
        gap = None if self._last_sample_ts is None else sample_timestamp-self._last_sample_ts
        # Samples can be close in capture time while execution has already
        # crossed the previous physical grant's deadline. That elapsed time
        # is neither an integration interval nor an acceleration budget.
        authority_expired = (self._last_sample_ts is not None
                             and execution_now-self._last_sample_ts > c.physical_ttl_sec+1e-9)
        long_gap = gap is not None and execution_now-self._last_sample_ts > c.retain_integral_sec
        if authority_expired:
            self._stationary_samples = 0
        if long_gap:
            self.integral_m_s = 0.
            self._stationary_samples = 0
        dt = (gap if gap is not None and gap <= c.max_integration_gap_sec+1e-9
              and not self._normal_parking and not self._execution_suspended
              and not authority_expired and not long_gap else 0.)
        scale = 60./c.wheel_circumference_m
        # A signed, fresh encoder observation is valid even during residual
        # rotation. Its sign is motion evidence, not permission to calculate a
        # forward request. Actual reverse transitions belong to the executor.
        ego_valid = (ego_forward_rpm is not None and math.isfinite(ego_forward_rpm)
                     and abs(ego_forward_rpm) <= max_output_rpm+5.)
        ego = ego_forward_rpm/scale if ego_valid else 0.
        relative_valid = (raw_closure_valid and ego_valid and range_rate_m_s is not None
                          and math.isfinite(range_rate_m_s) and abs(range_rate_m_s) <= 3.
                          and not measurement_jump_clamped)
        window_used = False
        window_target = window_rate = None
        window_span = 0.
        if raw_motion_evidence is not None:
            # A supplied but mismatched/stale window is unavailable evidence,
            # never permission to fall back to an unpaired target-speed sum.
            window_used = bool(relative_valid
                and isinstance(raw_motion_evidence, RawDepthMotionEvidence)
                and raw_motion_evidence.valid_for(sample_timestamp, range_rate_m_s))
            if window_used:
                window_target = raw_motion_evidence.target_speed_bound_m_s
                window_rate = raw_motion_evidence.range_rate_m_s
                window_span = raw_motion_evidence.span_sec
                # Transport the same-window target bound to CURRENT measured
                # car motion. During braking this removes fictitious target
                # approach; during acceleration it prevents fictitious target
                # retreat. It is not optional human-speed feedforward.
                range_rate_m_s = window_target-ego
                relative_valid = abs(range_rate_m_s) <= 3.
            else:
                relative_valid = False
                # Preserve measured rapid approach as a conservative veto.
                # Unknown motion must not start using an invalid target bound.
                range_rate_m_s = (min(-ego, range_rate_m_s)
                                 if range_rate_m_s is not None and math.isfinite(range_rate_m_s)
                                 else -ego)
        memory_used = False
        motion_origin = None
        uncertainty = 0.
        time_penalty = rotation_penalty = 0.
        retained_rate = endpoint_rate = None
        memory = self._motion_memory
        if relative_valid:
            self._motion_memory = (sample_timestamp, float(range_rate_m_s), ego, raw_distance_m)
            motion_origin = sample_timestamp
        elif (c.motion_memory_sec > 0 and allow_motion_memory and not raw_closure_valid and ego_valid
              and memory is not None and error > deadband_m
              and raw_distance_m is not None and math.isfinite(raw_distance_m) and raw_distance_m > 0
              and raw_distance_m-target_distance_m > deadband_m
              and memory[3] is not None and math.isfinite(memory[3]) and memory[3] > 0
              and 0 < execution_now-memory[0] <= c.motion_memory_sec):
            # Correct the old relative rate for measured car-speed changes.
            # A2m/s² uncertainty growth is an explicit trial assumption, not
            # a guarantee about human acceleration. It can only reduce output.
            time_penalty = 2.0*(execution_now-memory[0])
            rotation_penalty = motion_memory_rotation_bound-.25
            uncertainty = time_penalty + rotation_penalty
            retained_rate = memory[1] + memory[2] - ego - uncertainty
            range_rate_m_s = retained_rate
            # New raw depth contradicting the old trend must tighten the
            # bound immediately, even before a regression window is ready.
            # .25m/s covers the caller's allowed rotation uncertainty; endpoint
            # evidence is only a veto here, never permission to accelerate.
            endpoint_rate = ((raw_distance_m-memory[3])/(sample_timestamp-memory[0])
                             - motion_memory_rotation_bound)
            range_rate_m_s = min(range_rate_m_s, endpoint_rate)
            motion_origin = memory[0]
            memory_used = True
        else:
            self._motion_memory = None
        if (not self._normal_parking and not relative_valid and c.motion_memory_sec > 0 and self.last_result is not None
                and self.last_result.brake_source in {
                    "raw_relative_motion", "relative_motion_memory", "raw_endpoint_distance_bound"}):
            self._brake_recovery_pending = True
        finite_rate = range_rate_m_s is not None and math.isfinite(range_rate_m_s)
        rate = (range_rate_m_s if relative_valid or memory_used else
                min(-ego, range_rate_m_s) if finite_rate else -ego)
        closing = max(0., -rate)
        motion_available = ego_valid or finite_rate
        # A person walking toward the car has negative ground speed; clipping
        # that to zero would incorrectly authorize extra forward closure.
        target_velocity = ego+rate if relative_valid or memory_used else 0.
        # PI uses filtered error for comfort. Braking cannot spend distance
        # which a NEW trusted raw sample says has already been travelled.
        # A farther raw return cannot enlarge the filtered-distance budget.
        raw_valid = (raw_distance_m is not None and math.isfinite(raw_distance_m)
                     and raw_distance_m > 0)
        brake_distance = min(actual_distance_m, raw_distance_m) if raw_valid else actual_distance_m
        brake_error = brake_distance-target_distance_m
        remaining = max(0., brake_error-deadband_m)
        delay = c.response_delay_sec+max(0., age)
        a = c.deceleration_m_s2
        root = math.sqrt((a*delay)**2+2*a*remaining)
        command_closure = 2*a*remaining/(root+a*delay)
        observed_closure = math.sqrt(2*a*max(0., remaining-closing*delay))
        cap = max(0., min(max_output_rpm,
                          (target_velocity+min(command_closure, observed_closure))*scale))
        memory_uncertainty_stop = bool(
            memory_used and cap == 0 and time_penalty > 0
            and retained_rate is not None and endpoint_rate is not None
            and retained_rate < endpoint_rate)
        memory_endpoint_fallback = False
        memory_endpoint_cap = 0.
        previous = self.last_result
        previous_ego_rpm = (max(0., (previous.target_velocity_bound_m_s
                                   - previous.effective_range_rate_m_s)*scale)
                            if previous is not None else 0.)
        if not c.use_target_motion:
            previous_ego_rpm = self._last_ego_forward_rpm
        previous_measured_braking = bool(previous is not None and (
            previous.brake_settling_limited
            or previous.cap_rpm+5. < previous_ego_rpm))
        # CAP222: .34s of memory uncertainty can invent .68m/s of target
        # approach despite a NEW, farther raw endpoint. Only the caller can
        # certify that this endpoint actually entered the geometry/encoder-
        # checked warming window (not a skipped or uncompensated-turn sample).
        # In that narrow case use the MORE conservative of stationary-target
        # and raw-endpoint closure, retaining the existing rotation allowance.
        # This is a new distance request, NOT a new target-speed measurement.
        if (allow_motion_memory_endpoint_fallback is True
                and memory_uncertainty_stop and not previous_measured_braking
                and not self._normal_parking and not self._sample_rejected
                and motion_memory_rotation_bound == .25
                and ego_valid and ego_forward_rpm > 0 and self._approved_rpm > 0
                and memory is not None and memory[1]+memory[2] >= -c.stationary_speed_m_s
                and sample_timestamp > memory[0]
                and (raw_distance_m-memory[3])/(sample_timestamp-memory[0])
                    >= -min(max(0., memory[2]), max(0., ego))-c.stationary_speed_m_s):
            endpoint_bound = min(-ego, endpoint_rate)
            endpoint_closing = max(0., -endpoint_bound)
            endpoint_observed = math.sqrt(2*a*max(0., remaining-endpoint_closing*delay))
            endpoint_cap = max(0., min(max_output_rpm,
                (ego+endpoint_bound+min(command_closure, endpoint_observed))*scale))
            # Do not reinterpret an ongoing real brake as a measurement gap.
            # Supporting CURRENT wheel motion is required; no old high RPM,
            # startup floor, or expiry step may enlarge this request.
            if endpoint_cap >= ego_forward_rpm:
                memory_endpoint_fallback = True
                memory_endpoint_cap = endpoint_cap
                rate, closing = endpoint_bound, endpoint_closing
                target_velocity = ego+endpoint_bound
                cap = min(endpoint_cap, ego_forward_rpm, self._approved_rpm)
                memory_uncertainty_stop = False
        if memory_used:
            # Unknown motion may not accelerate, relaunch from zero, or revive
            # an old positive grant after its expiry/revocation. A NEW reliable
            # measurement can resume at no more than current measured speed.
            resume_cap = (max(0., ego_forward_rpm) if authority_expired or self._execution_suspended
                          else self._approved_rpm)
            cap = min(cap, resume_cap, self._approved_rpm)
        if brake_error < -deadband_m or measurement_jump_clamped or not motion_available:
            cap = 0.
        if not c.use_target_motion:
            # No relative-speed envelope to the setpoint. At zero distance
            # error learned PI speed may remain positive; the one physical
            # wheel/distance assessment below still has final veto authority.
            rate = target_velocity = 0.
            closing = max(0., ego)
            cap = (max_output_rpm if ego_valid and not isinstance(ego_forward_rpm, bool)
                   and raw_valid and brake_error >= -deadband_m
                   and not measurement_jump_clamped else 0.)
        preview_status = "disabled" if not c.stationary_stop_preview_enabled else "unavailable"
        preview_cap = preview_margin = preview_required = None
        preview_loss = 0.
        pre_preview_cap = cap
        evaluated_assessment = braking_assessment
        if not c.use_target_motion and evaluated_assessment is None:
            # Standalone/legacy callers have no shared runtime snapshot. Apply
            # the same physical model, without inventing target motion or a
            # publishable identity/lease. Strict timestamp validation remains.
            completed = preview_completed_rpm if preview_completed_rpm is not None else 0.
            try:
                if (isinstance(completed, bool) or not isinstance(completed, (int, float))
                        or not math.isfinite(completed) or not 0 <= completed <= max_output_rpm+5.):
                    raise ValueError("invalid completed physical speed")
                evaluated_assessment = SampleBrakingAssessment(
                    1, sample_timestamp, execution_now, brake_distance,
                    max(preview_outer_forward_rpm, completed), preview_outer_forward_rpm,
                    preview_feedback_timestamp, 0., c.stationary_stop_preview_distance_m,
                    c.wheel_circumference_m, c.deceleration_m_s2, c.response_delay_sec,
                    max_output_rpm, observed_feedback_reserve=c.observed_feedback_reserve,
                    current_feedback_independent=True)
            except (TypeError, ValueError):
                preview_status, preview_cap, cap = "invalid_feedback", 0., 0.
        if ((c.stationary_stop_preview_enabled or not c.use_target_motion)
                and evaluated_assessment is not None):
            if (isinstance(evaluated_assessment, SampleBrakingAssessment)
                    and evaluated_assessment.sample_timestamp == sample_timestamp
                    and evaluated_assessment.checked_at == execution_now
                    and evaluated_assessment.distance_m == brake_distance
                    and evaluated_assessment.outer_rpm == preview_outer_forward_rpm
                    and evaluated_assessment.feedback_timestamp == preview_feedback_timestamp
                    and (not evaluated_assessment.current_feedback_independent or not c.use_target_motion)
                    and (c.use_target_motion or (ego_valid
                        and evaluated_assessment.target_speed_m_s == 0.
                        and evaluated_assessment.stop_distance_m == c.stationary_stop_preview_distance_m
                        and evaluated_assessment.circumference_m == c.wheel_circumference_m
                        and evaluated_assessment.deceleration_m_s2 == c.deceleration_m_s2
                        and evaluated_assessment.response_delay_sec == c.response_delay_sec
                        and evaluated_assessment.observed_feedback_reserve == c.observed_feedback_reserve
                        and (not evaluated_assessment.feedback_interval_covered
                             or c.feedback_interval_deduplication)
                        and evaluated_assessment.max_rpm == max_output_rpm))):
                shared = evaluated_assessment.budget(execution_now, preview_outer_forward_rpm)
                preview_cap, preview_margin = shared.cap_rpm, shared.margin_m
                preview_required = shared.required_stop_m
                preview_status = ("momentum_brake" if shared.reason == "shared_braking_momentum"
                                  else "bounded" if shared.cap_rpm > 0 else "invalid_budget")
                cap = min(cap, preview_cap)
            else:
                preview_status, preview_cap, cap = "invalid_shared_assessment", 0., 0.
        elif c.stationary_stop_preview_enabled and c.use_target_motion:
            feedback_valid = (
                ego_valid and raw_valid and not measurement_jump_clamped
                and isinstance(preview_outer_forward_rpm, (int, float))
                and not isinstance(preview_outer_forward_rpm, bool)
                and math.isfinite(preview_outer_forward_rpm)
                and 0 <= preview_outer_forward_rpm <= max_output_rpm+5.
                and isinstance(preview_feedback_timestamp, (int, float))
                and not isinstance(preview_feedback_timestamp, bool)
                and math.isfinite(preview_feedback_timestamp)
                and 0 <= execution_now-preview_feedback_timestamp <= .15
                and abs(preview_feedback_timestamp-sample_timestamp) <= .15)
            if feedback_valid:
                # Candidate demand has NOT been issued yet: it cannot spend
                # distance since capture. Only fresh measured wheel speed or a
                # recently completed positive packet can bound PAST travel.
                completed_rpm = (preview_completed_rpm
                    if isinstance(preview_completed_rpm, (int, float))
                    and not isinstance(preview_completed_rpm, bool)
                    and math.isfinite(preview_completed_rpm)
                    and 0 <= preview_completed_rpm <= max_output_rpm+5. else 0.)
                bound_rpm = max(preview_outer_forward_rpm, completed_rpm)
                # Even reliable receding-target evidence cannot be counted as
                # future braking space: the person may stop immediately.
                budget = relative_braking_budget(
                    distance_m=brake_distance,
                    stop_distance_m=c.stationary_stop_preview_distance_m,
                    speed_bound_m_s=bound_rpm/scale, age_sec=age,
                    outer_speed_m_s=preview_outer_forward_rpm/scale,
                    target_speed_m_s=0., deceleration_m_s2=a,
                    response_delay_sec=c.response_delay_sec,
                    wheel_circumference_m=c.wheel_circumference_m,
                    max_rpm=max_output_rpm,
                    same_grant_no_acceleration=False)
                if budget is not None:
                    provisional_cap = budget.max_allowed_rpm
                    # Do not add an unconditional 15RPM/20% chase penalty on
                    # top of the stopping envelope. The default is zero; an
                    # explicitly requested extra margin can only tighten it.
                    # Measured momentum and the stop-distance budget remain
                    # binding even with no extra command headroom.
                    preview_cap = max(0., provisional_cap-min(
                        c.stationary_stop_preview_headroom_rpm,
                        .2*provisional_cap))
                    preview_margin = budget.margin_m
                    preview_required = budget.required_stop_m
                    preview_status = (
                        "momentum_brake" if budget.margin_m <= 0
                        or budget.required_stop_m > budget.margin_m
                        else "bounded")
                    cap = min(cap, preview_cap)
                else:
                    preview_status = "invalid_budget"
        # CAP230->233: a new positive envelope is not proof that the previous
        # braking command has taken effect. Keep the lower APPROVED request
        # while actual wheels are still faster and new raw distance/relative
        # motion still proves closure. A flat distance during speed matching
        # must not lock a far-away target to an initial fallback command.
        # This does not grant motion, require a fixed wait, or
        # carry a stale command over an expired/revoked lease. Five RPM covers
        # the existing two-RPM approval quantization and small feedback noise.
        clearly_away = bool(raw_valid and self._last_raw_distance_m is not None
            and relative_valid and rate > .05
            and raw_distance_m > self._last_raw_distance_m+.005)
        closure_observed = bool(raw_valid and self._last_raw_distance_m is not None
            and (raw_distance_m < self._last_raw_distance_m-.005
                 or (relative_valid and rate < -.05)))
        if not c.use_target_motion:
            # No finite difference of target distance here. Only an actual
            # prior brake plus uncompleted wheel response can retain settling;
            # a new physical budget supporting the wheels releases it.
            clearly_away = False
            closure_observed = bool(ego_valid and cap < max(0., ego_forward_rpm)+5.)
        unsettled = bool(raw_valid and self._last_raw_distance_m is not None
            and self.last_result is not None and ego_valid
            # A recovery ramp can sit below LATER accelerating wheels without
            # ever having asked the measured wheels to brake (CAP270). Keep
            # settling only after a real prior braking envelope/approval.
            # CAP217: the recovery calculation itself was 22RPM while wheels
            # were still at 52RPM. Approval of that same request is NOT a
            # separate braking cut. Only a downstream reduction beyond the
            # two-RPM approval quantization can establish that provenance.
            # A tighter current envelope always wins independently below.
            and (previous_measured_braking
                 or (self._approved_rpm+5. < previous_ego_rpm
                     and self._approved_rpm+2. < self._sample_requested_rpm))
            and not self._normal_parking and not self._execution_suspended
            and not authority_expired and not long_gap
            and ego_forward_rpm > self._approved_rpm+5.
            and closure_observed and not clearly_away)
        # CAP222/224: an uncertainty-only memory zero is not proof that the
        # car still needs to finish a measured braking maneuver. Once a NEW
        # synchronized raw window independently permits the current wheel
        # speed, remove only that zero's settling latch. The current brake
        # envelope and the ordinary execution/evidence-recovery ramps remain
        # in force; old/invalid windows cannot release it. Actual prior
        # braking, new insufficient stopping space, and external zero commands
        # retain their normal protection.
        uncertainty_settling_released = bool(
            unsettled and self._uncertain_zero_brake_anchor
            and window_used and relative_valid and ego_valid
            and cap > max(0., ego_forward_rpm)+5.)
        preview_settling_released = bool(
            unsettled and self._preview_zero_brake_anchor
            and window_used and relative_valid and ego_valid
            and preview_cap is not None
            and cap > max(0., ego_forward_rpm)+5.)
        # CAP119->122: the old zero came from THIS controller's momentum
        # budget, not a STOP/parking command. A NEW matching physical budget
        # can already support the current wheels while remaining inside the
        # extra 5RPM settling tolerance (63.8 allowed vs 60.5 measured). That
        # tolerance must not keep a resolved model stop latched at zero.
        # No target-motion estimate or old authority is restored. The current
        # sample must independently pass the shared model; this transition is
        # capped at measured speed as well as the normal ramp, never a launch.
        distance_settling_released = bool(
            unsettled and not c.use_target_motion
            and self._preview_zero_brake_anchor
            and self._approved_rpm == self._sample_requested_rpm == 0.
            and not self._sample_rejected
            and preview_status == "bounded"
            and isinstance(evaluated_assessment, SampleBrakingAssessment)
            and brake_distance > target_distance_m+deadband_m
            and 0. < ego_forward_rpm <= min(cap, evaluated_assessment.outer_rpm))
        preview_settling_released |= distance_settling_released
        if uncertainty_settling_released or preview_settling_released:
            unsettled = False
            self._preview_zero_brake_anchor = False
        settling_anchor = self._approved_rpm if unsettled else 0.
        before_settling_cap = cap
        if unsettled:
            cap = min(cap, settling_anchor)
        elif distance_settling_released:
            cap = min(cap, ego_forward_rpm)
        # Both samples and actual standstill are required; range error alone
        # must never erase the speed learned while following a walking person.
        stationary = (relative_valid and error <= deadband_m
                      and abs(ego) <= c.stationary_speed_m_s
                      and (c.motion_memory_sec > 0 or abs(ego+rate) <= c.stationary_speed_m_s)
                      and abs(rate) <= c.stationary_speed_m_s)
        if not c.use_target_motion:
            stationary = bool(ego_valid and abs(error) <= deadband_m
                              and abs(ego) <= c.stationary_speed_m_s)
        self._stationary_samples = self._stationary_samples+1 if stationary else 0
        if self._stationary_samples >= c.stationary_confirm_samples:
            self.integral_m_s = 0.
        # A real closing constraint can discharge old compensation even when
        # the outer runtime accepts this request unchanged. Do not erase it
        # merely because P hits its bound or motion evidence becomes missing.
        if (relative_valid or not c.use_target_motion) and self.integral_m_s > cap/scale:
            self.integral_m_s = max(cap/scale, self.integral_m_s-a*dt)
        self._integral_before_update = self.integral_m_s
        effective_error = math.copysign(max(0., abs(error)-deadband_m), error)
        p = c.kp_per_sec*effective_error
        increment = c.ki_per_sec2*effective_error*dt if not measurement_jump_clamped else 0.
        candidate_i = max(0., min(c.integral_max_m_s, self.integral_m_s+increment))
        # Freeze positive windup under the independent brake or speed ceiling;
        # a negative error can still unwind the forward speed compensation.
        frozen = dt == 0. or measurement_jump_clamped
        if candidate_i > self.integral_m_s and (p+candidate_i)*scale > cap+1e-9:
            candidate_i = self.integral_m_s
            frozen = True
        self.integral_m_s = candidate_i
        pi_demand = max(0., (p+self.integral_m_s)*scale)
        launch_active = bool(c.launch_request_rpm > 0 and ego_valid and error > deadband_m)
        launch_floor = c.launch_request_rpm if launch_active else 0.
        if c.launch_full_error_m > 0:
            # Continuous, memoryless taper: small error is PI-led; far chase
            # retains the configured request. A short expiry/recovery cannot
            # restart a fixed boost pulse. Squaring also removes the floor's
            # finite step at the deadband; the brake cap is unchanged.
            fraction = min(1., max(0., effective_error)/c.launch_full_error_m)
            launch_floor *= fraction*fraction
        bypass_rise = launch_active and c.launch_full_error_m == 0
        demand = max(pi_demand, launch_floor)
        request = min(cap, demand)
        if preview_cap is not None:
            # Same post-antiwindup demand on both sides isolates the request
            # cut due to preview, not a separate recovery or software ramp.
            preview_loss = max(0., min(pre_preview_cap, demand)
                                - min(pre_preview_cap, preview_cap, demand))
        settling_limited = unsettled and min(demand, before_settling_cap) > cap+1e-9
        # A long integration interval is not an execution revocation. A NEW
        # fresh measurement can continue the live ramp without integrating
        # unseen time. Runtime must report actual early withdrawals separately.
        integration_gap = gap is not None and gap > c.max_integration_gap_sec+1e-9
        # A previously accepted ZERO request never owned positive motor
        # authority. Expiring that sample cannot keep resetting its zero-origin
        # ramp: 200ms samples processed 70ms later otherwise produce 0 forever.
        # Only this NEW independently checked, near-stationary distance request
        # may take a bounded step. Real suspension/parking/rejection still wins;
        # neither old integral time nor an expired positive command is reused.
        zero_origin_progress = bool(
            authority_expired and not c.use_target_motion
            and self._last_execution_ts is not None
            and self._last_output_rpm == self._approved_rpm == self._sample_requested_rpm == 0.
            and not self._execution_suspended and not self._normal_parking
            and not self._sample_rejected and not self._fresh_grant_recovery_forbidden
            and ego_valid and abs(ego_forward_rpm) <= 5.
            and raw_valid and brake_distance > target_distance_m+deadband_m
            and preview_status == "bounded"
            and isinstance(evaluated_assessment, SampleBrakingAssessment)
            and evaluated_assessment.outer_rpm <= 5.)
        recovering = (self._last_execution_ts is None or self._execution_suspended
                      or authority_expired)
        if recovering:
            # Existing measured wheel motion is not a new 24RPM launch. No
            # synthetic 100ms or blind-gap acceleration budget is awarded.
            anchor = max(0., ego_forward_rpm) if ego_valid else 0.
            slew_dt = 0.
            if zero_origin_progress:
                anchor = 0.
                slew_dt = min(.05, max(0., execution_now-self._last_execution_ts))
        else:
            anchor = self._last_output_rpm
            slew_dt = max(0., execution_now-self._last_execution_ts)
        measured_anchor = anchor
        # A brief lease gap is not proof that the actuator stopped. A NEW
        # independently qualified range/feedback assessment can continue the
        # still-current, actually completed positive command's normal ramp.
        # The old lease stays expired, no blind interval is integrated, and a
        # zero/STOP/other receipt must invalidate this proof at the caller.
        execution_continuity = bool(
            recovering and not c.use_target_motion and not long_gap
            and not self._normal_parking and not self._sample_rejected
            and not self._execution_continuity_forbidden
            and not self._fresh_grant_recovery_forbidden
            and ego_valid and ego_forward_rpm >= 0
            and raw_valid and brake_distance > target_distance_m+deadband_m
            and preview_status == "bounded"
            and isinstance(braking_assessment, SampleBrakingAssessment)
            and isinstance(execution_recovery_proof, ForwardRecoveryAnchor)
            and execution_recovery_proof.valid_for(
                execution_recovery_uid, self._last_sample_ts, sample_timestamp,
                execution_now, min(.35, c.retain_integral_sec))
            and braking_assessment.uid == execution_recovery_uid
            and execution_recovery_proof.executed.rpm <= max_output_rpm
            and self._approved_rpm > 0)
        if execution_continuity:
            executed = execution_recovery_proof.executed
            anchor = min(self._approved_rpm, executed.rpm)
            # Use actual completed-write time, never the old depth's age or a
            # synthetic recovery step. The memory window bounds this interval.
            slew_dt = execution_now-executed.sent_at
        output = request
        # The caller can explicitly qualify one short physical-depth expiry
        # with a NEW far sample and fresh, same-direction wheel evidence. The
        # Caller-qualified ramp credit is a cap on this NEW request, never a
        # continuation of the expired motor lease or elapsed blind-gap credit.
        # A longer credit requires an independently verified, short-expiry
        # receding-depth window; the current brake envelope remains binding.
        expiry_step = (depth_expiry_recovery_step_sec if recovering and authority_expired
                       and not execution_continuity and not zero_origin_progress
                       and not self._normal_parking and not self._expiry_recovery_forbidden
                       and ego_valid and ego_forward_rpm >= 0 and rise_rpm_per_sec > 0
                       else 0.)
        # Separate new authorization from old-lease expiry. The caller must
        # qualify the current identity, raw distance, withdrawal and BOTH
        # wheels. This request never renews the old grant or skips reversal
        # guards. The independent fresh brake envelope remains binding.
        pure_restart_speed = bool(
            not c.use_target_motion and ego_valid
            and isinstance(evaluated_assessment, SampleBrakingAssessment)
            and (ego_forward_rpm >= 0 or (abs(ego_forward_rpm) <= 3.
                                         and evaluated_assessment.outer_rpm <= 3.)))
        # A caller may qualify a NEW confirmed range after pending geometry
        # reset all old PI state. Require its actual shared assessment and
        # quiet forward wheels; no previous request/lease is reconstructed.
        first_confirmed_step = bool(
            not c.use_target_motion and self._last_sample_ts is None
            and self._last_execution_ts is None and ego_valid
            and 0 <= ego_forward_rpm <= 3.
            and isinstance(braking_assessment, SampleBrakingAssessment)
            and braking_assessment.valid_for(depth_expiry_expected_uid, sample_timestamp)
            and braking_assessment.outer_rpm <= 3.)
        fresh_step = (fresh_grant_recovery_step_sec if recovering and not execution_continuity
                      and not zero_origin_progress
                      and (self._last_sample_ts is not None or first_confirmed_step)
                      and not self._normal_parking
                      and not self._sample_rejected
                      and not self._fresh_grant_recovery_forbidden
                      and ego_valid and (pure_restart_speed if not c.use_target_motion
                                        else abs(ego_forward_rpm) <= 5.)
                      and raw_valid and brake_distance > target_distance_m+(
                          deadband_m if not c.use_target_motion else .5)
                      and (not c.use_target_motion or (self._last_raw_distance_m is not None
                           and raw_distance_m >= self._last_raw_distance_m))
                      and not memory_used and rise_rpm_per_sec > 0 else 0.)
        if fresh_step > 0 and not c.use_target_motion:
            # The caller qualifies BOTH physical wheels and the withdrawal
            # reason. Recheck the independent current brake evidence here;
            # a requested recovery tick is not permission to bypass a stop.
            # Do not spend a fixed 50ms on samples arriving faster than that,
            # or borrow any of the blind interval beyond one normal tick.
            if (preview_status != "bounded"
                    or not isinstance(evaluated_assessment, SampleBrakingAssessment)):
                fresh_step = 0.
            elif not first_confirmed_step:
                fresh_step = (min(fresh_step, max(0., gap or 0.),
                                  max(0., execution_now-self._last_execution_ts))
                              if self._last_execution_ts is not None else 0.)
        recovery_step = max(expiry_step, fresh_step)
        fresh_rise_budget = min(240., rise_rpm_per_sec)*fresh_step
        completed_anchor_used = False
        completed_anchor_rpm = 0.
        completed_ramp_cap = 0.
        proof = depth_expiry_execution_anchor
        if (expiry_step > 0 and isinstance(proof, ForwardExecutionAnchor)
                and type(depth_expiry_expected_uid) is int
                and depth_expiry_expected_uid > 0
                and type(proof.uid) is int
                and proof.uid == depth_expiry_expected_uid
                and self._last_sample_ts is not None
                and proof.sample_timestamp == self._last_sample_ts
                and proof.receipt is not None
                and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                        for v in (proof.rpm, proof.sent_at))
                and all(math.isfinite(v) for v in (proof.rpm, proof.sent_at))
                and proof.rpm > 0
                and self._last_sample_ts <= proof.sent_at
                <= self._last_sample_ts+c.physical_ttl_sec
                and 0 <= execution_now-proof.sent_at <= .10
                and brake_distance > target_distance_m+.35
                and request > measured_anchor):
            # A completed positive packet is evidence of the actuator's
            # current target, not a renewal of the old depth lease. Its rise
            # credit is proportional to time since the actual write, capped
            # at one additional recovery step above measured-wheel recovery.
            # A delayed/stalled wheel cannot inherit an arbitrary old request.
            # A completed old packet may add at most the historical 50ms
            # proof credit. The stronger NEW-depth recession step must never
            # multiply that old-packet allowance as a side effect.
            proof_step = min(expiry_step, .05)
            measured_two_step_cap = measured_anchor+2*rise_rpm_per_sec*proof_step
            completed_anchor_rpm = min(proof.rpm, measured_two_step_cap)
            completed_ramp_cap = min(
                proof.rpm+rise_rpm_per_sec*min(proof_step, execution_now-proof.sent_at),
                measured_two_step_cap)
            if expiry_step > .05:
                completed_ramp_cap = min(completed_ramp_cap,
                                         measured_anchor+rise_rpm_per_sec*expiry_step)
        if rise_rpm_per_sec > 0 and output > anchor and (
                not bypass_rise or recovery_step > 0 or zero_origin_progress):
            rise_budget = min(240., rise_rpm_per_sec) if zero_origin_progress else rise_rpm_per_sec
            output = min(output, max(anchor+rise_budget*slew_dt
                                     + max(rise_rpm_per_sec*expiry_step, fresh_rise_budget),
                                     completed_ramp_cap))
        elif fall_rpm_per_sec > 0 and output < anchor:
            output = max(output, anchor-fall_rpm_per_sec*slew_dt)
        # Comfort deceleration cannot override braking, true stop or saturation.
        output = max(0., min(output, cap)) if request > 0 else 0.
        ramp_output = output
        # Restored evidence must not turn launch into a one-sample low->high
        # rebound. Only this transition uses the normal rise budget, not
        # initial launch/continuous valid tracking. A tighter brake wins NOW.
        # No fixed wait, 5RPM restart, or clock renewed by missing evidence.
        brake_recovery_limited = False
        recovery_anchor = 0.
        recovery_cap = None
        if relative_valid and self._brake_recovery_pending and not self._normal_parking:
            recovery_anchor = min(self._approved_rpm, anchor) if recovering else self._approved_rpm
            rise_budget = rise_rpm_per_sec if rise_rpm_per_sec > 0 else 240.
            recovery_cap = recovery_anchor + rise_budget*min(
                slew_dt, c.fresh_update_max_age_sec)+fresh_rise_budget
            brake_recovery_limited = output > recovery_cap+1e-9
            output = min(output, recovery_cap)
            # Repeated dropouts preserve progress, rather than restart a
            # fixed low-speed stage. Downstream limits feed the next anchor.
            self._brake_recovery_pending = brake_recovery_limited
        slew_limited = abs(output-request) > 1e-9
        limit_reason = ("brake_evidence_recovery" if brake_recovery_limited else
                        "brake_settling" if settling_limited else
                        "braking_envelope" if demand > cap+1e-9 and cap < max_output_rpm-1e-9
                        else "total_rpm_cap" if demand > cap+1e-9
                        else "software_slew" if slew_limited else "none")
        # CAP5240: a 79RPM envelope plus a 30RPM measured recovery anchor
        # produced 30RPM, previously reported only as braking_envelope. Keep
        # the entire numerical path unchanged and expose the binding stage.
        if brake_recovery_limited:
            final_limit_reason = "brake_evidence_recovery"
        elif output < request-1e-9:
            final_limit_reason = ("execution_continuity_slew" if execution_continuity else
                                  "zero_origin_restart" if zero_origin_progress else
                                  "fresh_grant_recovery_step" if fresh_step > 0 else
                                  "depth_expiry_recovery_step" if expiry_step > 0 else
                                  "execution_recovery" if recovering else "software_slew")
        elif output > request+1e-9:
            final_limit_reason = "software_deceleration"
        else:
            final_limit_reason = limit_reason
        # A ramp limiting the extra launch/chase demand can still execute all
        # of P+I. Freeze only when the PI itself exceeds the bounded output.
        if (output < min(request, pi_demand)-1e-9
                and self.integral_m_s > self._integral_before_update):
            self.integral_m_s = self._integral_before_update
            frozen = True
        pre_quantization_output = output
        output = math.floor(output+1e-9)
        uncertainty_zero = bool(
            memory_uncertainty_stop and output == 0
            and not previous_measured_braking and not self._normal_parking)
        if uncertainty_zero:
            self._uncertain_zero_brake_anchor = True
        elif not memory_used and not relative_valid:
            # A fresh fallback/fault must not inherit the earlier stop's
            # interpretation, especially after a new close raw return.
            self._uncertain_zero_brake_anchor = False
        elif relative_valid and (not window_used or cap <= max(0., ego_forward_rpm)+5.
                                 or output+5. >= max(0., ego_forward_rpm)):
            self._uncertain_zero_brake_anchor = False
        if (preview_cap == 0 and preview_loss > 0 and output == 0
                and preview_status == "momentum_brake"):
            self._preview_zero_brake_anchor = True
        elif output > 0 or self._normal_parking:
            self._preview_zero_brake_anchor = False
        # Report an actual difference in the quantized request. A valid
        # packet that is masked by a tighter new brake or request is unused.
        measured_only_rpm = math.floor(min(request, cap,
            measured_anchor+rise_rpm_per_sec*expiry_step)+1e-9)
        completed_anchor_used = bool(completed_anchor_rpm > measured_anchor+1e-9
                                     and output > measured_only_rpm)
        status = ("parked_preview" if self._normal_parking else
                  "stationary" if self._stationary_samples >= c.stationary_confirm_samples else
                  "long_gap_reset" if long_gap else
                  "execution_continuity" if execution_continuity else "recovering" if recovering else
                  "tracking_gap_no_integral" if integration_gap else "tracking")
        result = DistancePiResult(sample_timestamp, error, p, self.integral_m_s, output,
                                  request, cap, closing, closing*delay+closing**2/(2*a),
                                  delay, dt, status,
                                  "ego_distance" if not c.use_target_motion else
                                  "raw_relative_motion" if relative_valid else
                                  "raw_endpoint_distance_bound" if memory_endpoint_fallback else
                                  "relative_motion_memory" if memory_used else
                                  "motion_unknown_bound" if motion_available and c.motion_memory_sec > 0 else
                                  "stationary_fallback" if motion_available else "missing_motion_evidence",
                                  slew_limited, frozen, pi_demand, launch_floor, demand,
                                  bypass_rise and not brake_recovery_limited, limit_reason, motion_origin, uncertainty,
                                  brake_recovery_limited, recovery_anchor,
                                  time_penalty, rotation_penalty, retained_rate, endpoint_rate,
                                  rate, target_velocity, brake_distance,
                                  settling_limited, settling_anchor,
                                  final_limit_reason, before_settling_cap,
                                  anchor, slew_dt, ramp_output, recovery_cap,
                                  pre_quantization_output)
        result = replace(result, motion_window_used=window_used and relative_valid,
                         motion_window_target_speed_m_s=window_target,
                         motion_window_span_sec=window_span,
                         motion_window_range_rate_m_s=window_rate,
                         depth_expiry_recovery_step_sec=expiry_step,
                         depth_expiry_recovery_used=expiry_step > 0 and output > measured_anchor+1e-9,
                         depth_expiry_completed_anchor_used=completed_anchor_used,
                         depth_expiry_completed_anchor_rpm=completed_anchor_rpm,
                         brake_settling_uncertainty_released=uncertainty_settling_released,
                         memory_endpoint_fallback=memory_endpoint_fallback,
                         memory_endpoint_cap_rpm=memory_endpoint_cap,
                         fresh_grant_recovery_step_sec=fresh_step,
                         fresh_grant_recovery_used=fresh_step > 0 and output > measured_anchor+1e-9,
                         stationary_preview_status=preview_status,
                         stationary_preview_cap_rpm=preview_cap,
                         stationary_preview_loss_rpm=preview_loss,
                         stationary_preview_margin_m=preview_margin,
                         stationary_preview_required_stop_m=preview_required,
                         brake_settling_preview_released=preview_settling_released,
                         braking_assessment=braking_assessment,
                         execution_recovery_anchor_used=execution_continuity)
        self._last_sample_ts, self._last_execution_ts = sample_timestamp, execution_now
        self._last_ego_forward_rpm = max(0., ego_forward_rpm) if ego_valid else 0.
        self._last_target = target_distance_m
        self._last_raw_distance_m = float(raw_distance_m) if raw_valid else None
        self._last_output_rpm = self._approved_rpm = self._sample_requested_rpm = output
        self._execution_suspended = self._normal_parking
        if self._normal_parking:
            self._last_output_rpm = self._approved_rpm = 0.
            self._last_execution_ts = None
            self._brake_recovery_pending = False
        self._sample_rejected = False
        self._expiry_recovery_forbidden = self._normal_parking
        self._execution_continuity_forbidden = self._normal_parking
        self._fresh_grant_recovery_forbidden = self._normal_parking
        self.last_result = result
        return result
