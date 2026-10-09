"""Sample-driven forward PI with a separate relative-motion braking bound.

This module owns neither sensors nor motor authorization. Keeping controller
memory never grants motion: the caller must still enforce the physical Depth
deadline and pass an ego speed only when its encoder evidence is synchronized
and fresh. Deceleration is an experimental model, not a guaranteed stop bound.
"""
from dataclasses import dataclass, replace
import math
from typing import Optional


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

    def __post_init__(self):
        if (isinstance(self.launch_full_error_m, bool)
                or not math.isfinite(self.launch_full_error_m)
                or not 0 <= self.launch_full_error_m <= 2.):
            raise ValueError("invalid distance PI launch_full_error_m")
        if (isinstance(self.motion_memory_sec, bool) or not math.isfinite(self.motion_memory_sec)
                or not 0 <= self.motion_memory_sec <= .35):
            raise ValueError("invalid distance PI motion_memory_sec")
        if (isinstance(self.launch_request_rpm, bool)
                or not math.isfinite(self.launch_request_rpm)
                or not 0 <= self.launch_request_rpm <= 200.):
            raise ValueError("invalid distance PI launch_request_rpm")
        for name, ceiling in (("kp_per_sec", 3.), ("ki_per_sec2", 2.),
                              ("integral_max_m_s", 1.5), ("wheel_circumference_m", 3.),
                              ("deceleration_m_s2", 1.), ("response_delay_sec", 1.),
                              ("physical_ttl_sec", .25), ("fresh_update_max_age_sec", .18),
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

    def invalidate_motion_memory(self):
        if (not self._normal_parking and self.config.motion_memory_sec > 0 and self.last_result is not None
                and self.last_result.brake_source in {"raw_relative_motion", "relative_motion_memory"}):
            self._brake_recovery_pending = True
        self._motion_memory = None

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
        approved_speed = approved_rpm*self.config.wheel_circumference_m/60.
        material_limit = self._sample_requested_rpm-approved_rpm > quantization_rpm+1e-9
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
        self._stationary_samples = 0
        self.integral_m_s = min(self.integral_m_s, self._integral_before_update)
        self.last_result = replace(self.last_result, integral_m_s=self.integral_m_s,
                                   integral_frozen=True)
        return True

    def update(self, actual_distance_m: float, target_distance_m: float, *,
               sample_timestamp: float, execution_now: float, deadband_m: float,
               max_output_rpm: float, rise_rpm_per_sec: float = 0.,
               fall_rpm_per_sec: float = 0., ego_forward_rpm: Optional[float] = None,
               range_rate_m_s: Optional[float] = None, raw_closure_valid: bool = False,
               allow_motion_memory: bool = False,
               motion_memory_rotation_bound: float = .25,
               raw_distance_m: Optional[float] = None,
               measurement_jump_clamped: bool = False) -> DistancePiResult:
        c = self.config
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
                and self.last_result.brake_source in {"raw_relative_motion", "relative_motion_memory"}):
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
        if memory_used:
            # Unknown motion may not accelerate, relaunch from zero, or revive
            # an old positive grant after its expiry/revocation. A NEW reliable
            # measurement can resume at no more than current measured speed.
            resume_cap = (max(0., ego_forward_rpm) if authority_expired or self._execution_suspended
                          else self._approved_rpm)
            cap = min(cap, resume_cap, self._approved_rpm)
        if brake_error < -deadband_m or measurement_jump_clamped or not motion_available:
            cap = 0.
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
        unsettled = bool(raw_valid and self._last_raw_distance_m is not None
            and self.last_result is not None and ego_valid
            and not self._normal_parking and not self._execution_suspended
            and not authority_expired and not long_gap
            and ego_forward_rpm > self._approved_rpm+5.
            and closure_observed and not clearly_away)
        settling_anchor = self._approved_rpm if unsettled else 0.
        before_settling_cap = cap
        if unsettled:
            cap = min(cap, settling_anchor)
        # Both samples and actual standstill are required; range error alone
        # must never erase the speed learned while following a walking person.
        stationary = (relative_valid and error <= deadband_m
                      and abs(ego) <= c.stationary_speed_m_s
                      and (c.motion_memory_sec > 0 or abs(ego+rate) <= c.stationary_speed_m_s)
                      and abs(rate) <= c.stationary_speed_m_s)
        self._stationary_samples = self._stationary_samples+1 if stationary else 0
        if self._stationary_samples >= c.stationary_confirm_samples:
            self.integral_m_s = 0.
        # A real closing constraint can discharge old compensation even when
        # the outer runtime accepts this request unchanged. Do not erase it
        # merely because P hits its bound or motion evidence becomes missing.
        if relative_valid and self.integral_m_s > cap/scale:
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
        settling_limited = unsettled and min(demand, before_settling_cap) > cap+1e-9
        # A long integration interval is not an execution revocation. A NEW
        # fresh measurement can continue the live ramp without integrating
        # unseen time. Runtime must report actual early withdrawals separately.
        integration_gap = gap is not None and gap > c.max_integration_gap_sec+1e-9
        recovering = (self._last_execution_ts is None or self._execution_suspended
                      or authority_expired)
        if recovering:
            # Existing measured wheel motion is not a new 24RPM launch. No
            # synthetic 100ms or blind-gap acceleration budget is awarded.
            anchor = max(0., ego_forward_rpm) if ego_valid else 0.
            slew_dt = 0.
        else:
            anchor = self._last_output_rpm
            slew_dt = max(0., execution_now-self._last_execution_ts)
        output = request
        if rise_rpm_per_sec > 0 and output > anchor and not bypass_rise:
            output = min(output, anchor+rise_rpm_per_sec*slew_dt)
        elif fall_rpm_per_sec > 0 and output < anchor:
            output = max(output, anchor-fall_rpm_per_sec*slew_dt)
        # Comfort deceleration cannot override braking, true stop or saturation.
        output = max(0., min(output, cap)) if request > 0 else 0.
        # Restored evidence must not turn launch into a one-sample low->high
        # rebound. Only this transition uses the normal rise budget, not
        # initial launch/continuous valid tracking. A tighter brake wins NOW.
        # No fixed wait, 5RPM restart, or clock renewed by missing evidence.
        brake_recovery_limited = False
        recovery_anchor = 0.
        if relative_valid and self._brake_recovery_pending and not self._normal_parking:
            recovery_anchor = min(self._approved_rpm, anchor) if recovering else self._approved_rpm
            rise_budget = rise_rpm_per_sec if rise_rpm_per_sec > 0 else 240.
            recovery_cap = recovery_anchor + rise_budget*min(slew_dt, c.fresh_update_max_age_sec)
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
        if output < request-1e-9 and self.integral_m_s > self._integral_before_update:
            self.integral_m_s = self._integral_before_update
            frozen = True
        output = math.floor(output+1e-9)
        status = ("parked_preview" if self._normal_parking else
                  "stationary" if self._stationary_samples >= c.stationary_confirm_samples else
                  "long_gap_reset" if long_gap else "recovering" if recovering else
                  "tracking_gap_no_integral" if integration_gap else "tracking")
        result = DistancePiResult(sample_timestamp, error, p, self.integral_m_s, output,
                                  request, cap, closing, closing*delay+closing**2/(2*a),
                                  delay, dt, status,
                                  "raw_relative_motion" if relative_valid else
                                  "relative_motion_memory" if memory_used else
                                  "motion_unknown_bound" if motion_available and c.motion_memory_sec > 0 else
                                  "stationary_fallback" if motion_available else "missing_motion_evidence",
                                  slew_limited, frozen, pi_demand, launch_floor, demand,
                                  bypass_rise and not brake_recovery_limited, limit_reason, motion_origin, uncertainty,
                                  brake_recovery_limited, recovery_anchor,
                                  time_penalty, rotation_penalty, retained_rate, endpoint_rate,
                                  rate, target_velocity, brake_distance,
                                  settling_limited, settling_anchor)
        self._last_sample_ts, self._last_execution_ts = sample_timestamp, execution_now
        self._last_target = target_distance_m
        self._last_raw_distance_m = float(raw_distance_m) if raw_valid else None
        self._last_output_rpm = self._approved_rpm = self._sample_requested_rpm = output
        self._execution_suspended = self._normal_parking
        if self._normal_parking:
            self._last_output_rpm = self._approved_rpm = 0.
            self._last_execution_ts = None
            self._brake_recovery_pending = False
        self._sample_rejected = False
        self.last_result = result
        return result
