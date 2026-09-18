"""Bounded target-speed evidence from trusted Depth and encoder observations.

Near the camera axis at low yaw, target Z speed is approximately the measured
range rate plus ego forward speed. This estimator owns no motor, PID, identity,
or safety authority. Its absolute ``target_rpm`` replaces a PID launch baseline;
the diagnostic ``feedforward_rpm`` must not be added to that baseline again.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Optional


@dataclass(frozen=True)
class LongitudinalFeedforwardConfig:
    wheel_circumference_m: float = 0.60
    baseline_rpm: float = 20.0
    max_feedforward_rpm: float = 20.0
    # Zero preserves the legacy baseline+extra budget for old configurations.
    max_tracking_base_rpm: float = 0.0
    tracking_rise_rpm_per_sec: float = 80.0
    tracking_fall_rpm_per_sec: float = 80.0
    min_distance_m: float = 1.47
    max_sample_age_sec: float = 0.25
    max_feedback_age_sec: float = 0.15
    max_feedback_depth_skew_sec: float = 0.15
    min_sample_interval_sec: float = 0.02
    max_sample_gap_sec: float = 0.25
    max_distance_jump_m: float = 0.30
    max_abs_range_rate_m_s: float = 1.50
    max_abs_ego_rpm: float = 100.0
    max_abs_target_speed_m_s: float = 1.50
    max_abs_yaw_rate_dps: float = 5.0
    max_abs_target_bearing_deg: float = 10.0
    target_speed_deadband_m_s: float = 0.03
    target_speed_filter_alpha: float = 0.35
    compensated_max_yaw_rate_dps: float = 15.0


@dataclass(frozen=True)
class LongitudinalFeedforwardEvidence:
    status: str
    eligible: bool = False
    range_rate_m_s: Optional[float] = None
    target_speed_m_s: Optional[float] = None
    target_rpm: float = 0.0
    feedforward_rpm: float = 0.0
    target_id: Optional[int] = None
    sample_timestamp: Optional[float] = None
    sample_count: int = 0
    uncompensated_range_rate_m_s: Optional[float] = None
    rotation_rate_m_s: Optional[float] = None
    unbounded_target_rpm: float = 0.0
    matching_cap_rpm: float = 0.0
    matching_rate_limited: bool = False
    chain_reset_reason: Optional[str] = None
    instantaneous_target_speed_m_s: Optional[float] = None
    window_target_speed_m_s: Optional[float] = None
    speed_window_sec: float = 0.0
    speed_window_samples: int = 0
    decline_policy: str = "none"


@dataclass(frozen=True)
class _Sample:
    timestamp: float
    distance_m: float
    ego_forward_m_s: float
    feedback_timestamp: float
    rotation_rate_m_s: Optional[float] = None
    yaw_rate_dps: float = 0.0


def depth_rotation_rate(*, depth, bbox, width, hfov_deg, capture_stamp, depth_stamp, yaw,
                        low_yaw_max_age_sec=.18):
    """Bounded +omega*X contribution to camera Z-rate for right-positive yaw.

    Project the fresh raw detector bearing to depth time. This is a short,
    near-axis approximation, not world odometry or identity association.
    """
    values = [_number(v) for v in (depth, width, hfov_deg, capture_stamp, depth_stamp, yaw)]
    if any(v is None for v in values):
        return None
    z, w, hfov, rgb_ts, dep_ts, omega_dps = values
    low_age = _number(low_yaw_max_age_sec)
    if low_age is None or low_age <= 0:
        return None
    # This is detector-to-depth alignment, NOT depth freshness or a lease.
    max_alignment = min(.25, low_age) if abs(omega_dps) <= 5 else .18
    if not (z > 0 and w > 0 and 0 < hfov < 160 and rgb_ts > 0
            and -.02 <= dep_ts-rgb_ts <= max_alignment + 1e-9 and abs(omega_dps) <= 15):
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or not (x2>x1 and y2>y1):
        return None
    fx = .5 * w / math.tan(math.radians(hfov) * .5)
    angle = math.atan(((x1+x2)*.5 - w*.5) / fx) - math.radians(omega_dps) * (dep_ts-rgb_ts)
    bearing = math.degrees(angle)
    rate = math.radians(omega_dps) * z * math.tan(angle)
    if abs(bearing) > 10 or abs(rate) > .25:
        return None
    return rate, bearing


def bounded_disagreeing_yaw_rotation(*, raw_yaw, filtered_yaw, **geometry):
    """Conservative Z-rate correction for a small near-axis yaw ambiguity.

    The median of three wheel samples can have the opposite sign to the
    latest sample. Do not choose whichever sign makes the target faster.
    Bound BOTH rotation directions, including bearing projection to Depth
    time, and subtract the positive bound from the speed estimate. This
    slightly underestimates target speed if actual yaw stays inside the
    measured magnitude envelope. It is not a calibrated motion model.

    Caller must require fresh aligned encoder/Depth, raw UID-bound geometry,
    and a far target. No clock, identity or motor authority is created here.
    """
    raw, filtered = _number(raw_yaw), _number(filtered_yaw)
    if raw is None or filtered is None or raw * filtered >= 0:
        return None
    magnitude = max(abs(raw), abs(filtered))
    if magnitude > 15:
        return None
    ends = [depth_rotation_rate(**geometry, yaw=sign*magnitude) for sign in (-1, 1)]
    if any(value is None for value in ends):
        return None
    bearing = max((value[1] for value in ends), key=abs)
    bound = math.radians(magnitude) * float(geometry['depth']) * abs(math.tan(math.radians(bearing)))
    # <=4 RPM equivalent with the current 0.60m circumference. Larger
    # uncertainty still rebuilds the estimator through the existing gate.
    if bound > .04:
        return None
    return bound, bearing


def far_closure_consistent(*, distance, previous_distance, interval, ego_speed, near):
    """Allow modest far-target closure consistent with measured forward travel.

    Not a depth trust check or permission to move. Caller validates identity,
    sample/feedback timestamps, hazards and positive wheel speeds first.
    """
    values = (distance, previous_distance, interval, ego_speed, near)
    if any(_number(v) is None for v in values):
        return False
    closure = previous_distance - distance
    if not (0 < interval <= .35 and min(distance, previous_distance) > near
            and 0 < closure <= .15 and ego_speed > 0):
        return False
    closing_speed = closure / interval
    return (closing_speed <= 1.0 and closure <= ego_speed * interval + .03
            and (distance - near) / closing_speed >= 1.0)


class LongitudinalFeedforwardBridge:
    """Decay a *previously measured* speed, never estimate speed while turning.

    The original evidence deadline is immutable. A fresh accepted Depth and
    fresh encoder feedback are still required by the caller on every use.
    This is not a motor lease and cannot start or accelerate the vehicle.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.origin = None
        self.distance = None
        self._last_bridge_rpm = None

    def remember(self, evidence, distance):
        if evidence.eligible and evidence.status in {"ready", "ready_capped"}:
            self.origin, self.distance = evidence, distance
            self._last_bridge_rpm = None
        else:
            self.reset()

    def limit_prior_rpm(self, rpm):
        """Remember a reduced contribution for this origin, not a motor cap."""
        value = _number(rpm)
        value = max(0., value) if value is not None else 0.
        if self._last_bridge_rpm is not None:
            value = min(value, self._last_bridge_rpm)
        self._last_bridge_rpm = value
        return value

    def evaluate(self, *, now, stamp, uid, distance, yaw, bearing, previous_output, baseline,
                 fall_rate_rpm_per_sec=None, near_distance=None, prior_max_age_sec=.18,
                 ego_speed=None):
        origin = self.origin
        values = [now, stamp, distance, yaw, bearing, previous_output, baseline, self.distance]
        if origin is None or any(_number(v) is None for v in values):
            return None
        if _number(prior_max_age_sec) is None or not .18 <= prior_max_age_sec <= .35:
            return None
        age = now - origin.sample_timestamp
        safe_closing = far_closure_consistent(
            distance=distance, previous_distance=self.distance,
            interval=stamp-origin.sample_timestamp, ego_speed=ego_speed, near=near_distance,
        )
        if not (
            origin.target_id == uid and 0.0 <= age < prior_max_age_sec
            and origin.sample_timestamp < stamp <= now and now - stamp <= 0.18
            and abs(yaw) <= 15.0 and abs(bearing) <= 10.0
            and previous_output > 0.0
            # Closing / jumping observations must not carry a walking prior.
            and (-0.03 <= distance - self.distance <= 0.30 or safe_closing)
        ):
            return None
        floor = min(origin.target_rpm, max(0.0, baseline))
        rpm = min(previous_output, floor + (origin.target_rpm - floor) * (1.0 - age / prior_max_age_sec))
        if fall_rate_rpm_per_sec is not None:
            fall = _number(fall_rate_rpm_per_sec)
            near = _number(near_distance)
            interval = stamp - origin.sample_timestamp
            if (fall is None or fall <= 0 or near is None or distance <= near
                    or ((distance-self.distance)/interval < -.03 and not safe_closing)):
                return None
            # Absolute origin prevents repeated ticks renewing the prior.
            # Only a far, non-closing prior can use the bounded fall path.
            rpm = min(previous_output, max(0.0, origin.target_rpm - fall * age))
        # Fresh range correction may raise the TOTAL output while the prior
        # is retained. That is not new evidence of faster human motion and
        # must not let a previously reduced matching contribution grow back.
        rpm = self.limit_prior_rpm(rpm)
        return replace(origin, status="transient_bridge", sample_timestamp=stamp,
                       target_rpm=rpm, feedforward_rpm=max(0.0, rpm - baseline))


def _number(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


class LongitudinalFeedforwardEstimator:
    """Two-physical-sample speed estimator with fail-closed output.

    ``trusted`` must mean a fresh, accepted, identity-bound depth observation,
    not a held/reused/pending distance. The caller supplies forward-normalized
    *measured* mean wheel RPM, never a requested motor command. The same target
    UID and current raw-box provenance remain the caller's responsibility.

    Duplicate/older observations return zero evidence without changing a newer
    chain. A new rejected observation clears the chain; retrying its timestamp
    cannot restore it. ``reset`` also clears the target and timestamp watermark.
    """

    def __init__(self, config: Optional[LongitudinalFeedforwardConfig] = None, *, shared_window=None) -> None:
        self.config = config if config is not None else LongitudinalFeedforwardConfig()
        self.shared_window = shared_window
        self.reset()

    def reset(self) -> None:
        if self.shared_window is not None:
            self.shared_window.reset()
        self._gap_yaw_bounds = None
        self._target_id: Optional[int] = None
        self._sample_watermark: Optional[float] = None
        self._previous: Optional[_Sample] = None
        self._sample_count = 0
        self._filtered_target_speed: Optional[float] = None
        self._last_matching_rpm: Optional[float] = None
        self._turn_gap_pending = False
        self._speed_samples = []
        self._stop_suspect_pending = False
        self._positive_after_suspect = 0
        self.last_result = LongitudinalFeedforwardEvidence("reset")

    def _clear_chain(self) -> None:
        if self.shared_window is not None:
            self.shared_window.reset()
        self._gap_yaw_bounds = None
        self._previous = None
        self._sample_count = 0
        self._filtered_target_speed = None
        self._last_matching_rpm = None
        self._turn_gap_pending = False
        self._speed_samples = []
        self._stop_suspect_pending = False
        self._positive_after_suspect = 0

    def _window_speed(self, sample):
        """Fit target displacement, integrating ego/rotation over the SAME interval.

        History is bounded to 180ms; it is never an authorization or a new
        timestamp. Adjacent-sample outlier/safety checks run before this method.
        """
        self._speed_samples = [s for s in self._speed_samples
                               if 0 < sample.timestamp-s.timestamp <= .18 + 1e-9][-15:]
        self._speed_samples.append(sample)
        samples = self._speed_samples
        span = samples[-1].timestamp-samples[0].timestamp
        if len(samples) < 3 or span < .06:
            return None, span, len(samples)
        displacement = 0.0
        points = [(0.0, samples[0].distance_m)]
        for old, new in zip(samples, samples[1:]):
            dt = new.timestamp-old.timestamp
            rotation = .5*((old.rotation_rate_m_s or 0.)+(new.rotation_rate_m_s or 0.))
            displacement += (.5*(old.ego_forward_m_s+new.ego_forward_m_s)-rotation)*dt
            points.append((new.timestamp-samples[0].timestamp, new.distance_m+displacement))
        mt = sum(t for t, _ in points)/len(points)
        mz = sum(z for _, z in points)/len(points)
        speed = sum((t-mt)*(z-mz) for t, z in points)/sum((t-mt)**2 for t, _ in points)
        return speed, span, len(samples)

    def preserve_compensated_gap(self, *, now, yaw) -> bool:
        """No new sample: keep a baseline only through a short stable turn.

        Caller has already checked UID, no hazard, fresh encoder and no new
        rejected depth. Next physical sample must independently pass all gates.
        """
        old = self._previous
        if self.shared_window is not None:
            # Bookkeeping only. No missing-depth tick may add a sample, move
            # timestamps, or grant FF. Verify uncertainty at the next endpoint.
            if (old is None or _number(now) is None or _number(yaw) is None
                    or not 0 <= now-old.timestamp <= .18 or abs(yaw) > 15):
                return False
            low, high = self._gap_yaw_bounds or (old.yaw_rate_dps, old.yaw_rate_dps)
            self._gap_yaw_bounds = (min(low,yaw),max(high,yaw))
            self._turn_gap_pending = True
            return True
        if (old is None or old.rotation_rate_m_s is None
                or _number(now) is None or _number(yaw) is None
                or not 0 <= now-old.timestamp <= .18
                or abs(yaw) > 15 or abs(yaw-old.yaw_rate_dps) > 3):
            return False
        self._turn_gap_pending = True
        return True

    @property
    def has_compensated_baseline(self) -> bool:
        return self._previous is not None and self._previous.rotation_rate_m_s is not None

    def _result(self, status: str, *, stamp=None, clear=False, **fields):
        if clear:
            self._clear_chain()
        self.last_result = LongitudinalFeedforwardEvidence(
            status=status, target_id=self._target_id, sample_timestamp=stamp,
            sample_count=self._sample_count, **fields,
        )
        return self.last_result

    def _valid_config(self) -> bool:
        c = self.config
        if not isinstance(c, LongitudinalFeedforwardConfig):
            return False
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               for value in vars(c).values()):
            return False
        numeric = {name: _number(value) for name, value in vars(c).items()}
        if any(value is None for value in numeric.values()):
            return False
        nonnegative = {"baseline_rpm", "max_feedforward_rpm", "target_speed_deadband_m_s", "max_tracking_base_rpm"}
        if any(value < 0.0 if name in nonnegative else value <= 0.0
               for name, value in numeric.items()):
            return False
        return bool(
            c.min_sample_interval_sec <= min(0.25, c.max_sample_gap_sec)
            and c.target_speed_filter_alpha <= 1.0
            and c.target_speed_deadband_m_s < c.max_abs_target_speed_m_s
            and c.min_distance_m > 0.50
        )

    def update(
        self, *, now, sample_timestamp, distance_m, target_id,
        feedback_timestamp, ego_forward_rpm, yaw_rate_dps, trusted,
        target_bearing_deg=0.0, rotation_rate_m_s=None,
    ) -> LongitudinalFeedforwardEvidence:
        c = self.config
        if not self._valid_config():
            return self._result("invalid_config", clear=True)
        identity = _number(target_id)
        if identity is None or identity <= 0.0 or int(identity) != identity:
            self.reset()
            return self._result("invalid_target")
        identity = int(identity)
        if identity != self._target_id:
            self.reset()
            self._target_id = identity
        current, stamp = _number(now), _number(sample_timestamp)
        if current is None or stamp is None or current <= 0.0 or stamp <= 0.0:
            return self._result("invalid_timestamp", clear=True)
        if stamp > current:
            # A corrupt future timestamp must not poison the valid watermark.
            return self._result("future_depth", stamp=stamp, clear=True)
        if self._sample_watermark is not None and stamp <= self._sample_watermark + 1e-9:
            reason = "duplicate_depth" if abs(stamp - self._sample_watermark) <= 1e-9 else "out_of_order_depth"
            return self._result(reason, stamp=stamp)
        self._sample_watermark = stamp
        if current - stamp > min(0.25, c.max_sample_age_sec) + 1e-9:
            return self._result("stale_depth", stamp=stamp, clear=True)
        if trusted is not True:
            return self._result("untrusted_depth", stamp=stamp, clear=True)
        distance = _number(distance_m)
        if distance is None or distance <= 0.0:
            return self._result("invalid_distance", stamp=stamp, clear=True)
        if distance < c.min_distance_m:
            return self._result("too_close", stamp=stamp, clear=True)
        feedback_stamp = _number(feedback_timestamp)
        ego_rpm, yaw, bearing = (_number(value) for value in (
            ego_forward_rpm, yaw_rate_dps, target_bearing_deg,
        ))
        if feedback_stamp is None or feedback_stamp <= 0.0 or ego_rpm is None or yaw is None or bearing is None:
            return self._result("invalid_feedback", stamp=stamp, clear=True)
        if feedback_stamp > current:
            return self._result("future_feedback", stamp=stamp, clear=True)
        if current - feedback_stamp > c.max_feedback_age_sec + 1e-9:
            return self._result("stale_feedback", stamp=stamp, clear=True)
        if abs(feedback_stamp - stamp) > c.max_feedback_depth_skew_sec + 1e-9:
            return self._result("feedback_depth_misaligned", stamp=stamp, clear=True)
        if abs(ego_rpm) > c.max_abs_ego_rpm:
            return self._result("ego_speed_out_of_bounds", stamp=stamp, clear=True)
        rotation = _number(rotation_rate_m_s)
        if rotation_rate_m_s is not None and (rotation is None or abs(rotation) > .25):
            return self._result("invalid_rotation_compensation", stamp=stamp, clear=True)
        yaw_limit = (min(15.0, c.compensated_max_yaw_rate_dps) if rotation is not None
                     else min(5.0, c.max_abs_yaw_rate_dps))
        if abs(yaw) > yaw_limit:
            return self._result("turning", stamp=stamp, clear=True)
        if abs(bearing) > min(10.0, c.max_abs_target_bearing_deg):
            return self._result("off_axis", stamp=stamp, clear=True)
        sample = _Sample(stamp, distance, ego_rpm * c.wheel_circumference_m / 60.0, feedback_stamp, rotation, yaw)
        if self.shared_window is not None:
            return self._update_shared_sample(sample, bearing)
        previous = self._previous
        reset_reason = None
        if self._turn_gap_pending:
            if (previous is None or rotation is None or stamp-previous.timestamp > .18
                    or abs(yaw-previous.yaw_rate_dps) > 3):
                self._clear_chain()
                previous = None
                reset_reason = "compensated_gap_unverified"
            self._turn_gap_pending = False
        if previous is not None and ((previous.rotation_rate_m_s is None) != (rotation is None)):
            self._clear_chain()
            previous = None  # do not differentiate across compensation mode changes
            reset_reason = "compensation_mode_change"
        if previous is None:
            self._previous, self._sample_count = sample, 1
            self._speed_samples = [sample]
            return self._result("warming_up", stamp=stamp, chain_reset_reason=reset_reason)
        if feedback_stamp < previous.feedback_timestamp - 1e-9:
            return self._result("out_of_order_feedback", stamp=stamp, clear=True)
        delta_time = stamp - previous.timestamp
        if delta_time > min(0.25, c.max_sample_gap_sec) + 1e-9:
            self._clear_chain()
            self._previous, self._sample_count = sample, 1
            self._speed_samples = [sample]
            return self._result("gap_reset", stamp=stamp)
        if delta_time < c.min_sample_interval_sec - 1e-9:
            # Keep the earlier baseline until enough physical time has passed;
            # do not turn near-duplicate timestamps into a noisy derivative.
            return self._result("interval_too_short", stamp=stamp)
        distance_delta = distance - previous.distance_m
        if abs(distance_delta) > c.max_distance_jump_m + 1e-9:
            return self._result("depth_jump", stamp=stamp, clear=True)
        range_rate = distance_delta / delta_time
        if abs(range_rate) > c.max_abs_range_rate_m_s + 1e-9:
            return self._result("range_rate_out_of_bounds", stamp=stamp, clear=True)
        uncompensated = range_rate
        interval_rotation = (None if rotation is None else .5 * (previous.rotation_rate_m_s + rotation))
        if interval_rotation is not None:
            range_rate -= interval_rotation
            if abs(range_rate) > c.max_abs_range_rate_m_s + 1e-9:
                return self._result("compensated_rate_out_of_bounds", stamp=stamp, clear=True)
        # The distance difference covers an interval. Using only its final ego
        # speed would invent target motion when the vehicle accelerates.
        interval_ego_speed = 0.5 * (previous.ego_forward_m_s + sample.ego_forward_m_s)
        target_speed = range_rate + interval_ego_speed
        if abs(target_speed) > c.max_abs_target_speed_m_s + 1e-9:
            return self._result("target_speed_out_of_bounds", stamp=stamp, clear=True)
        self._previous = sample
        self._sample_count += 1
        instant_speed = target_speed
        window_speed, window_span, window_count = self._window_speed(sample)
        near_distance = c.min_distance_m + .23
        urgent_decline = (distance <= near_distance or
                          (range_rate < 0 and (distance-near_distance)/-range_rate <= .4))
        use_window = c.max_tracking_base_rpm > 0 and not urgent_decline and window_speed is not None
        if use_window and (not math.isfinite(window_speed) or abs(window_speed) > c.max_abs_target_speed_m_s):
            return self._result("window_speed_out_of_bounds", stamp=stamp, clear=True)
        if use_window:
            target_speed = window_speed
        diagnostics = dict(
            instantaneous_target_speed_m_s=instant_speed,
            window_target_speed_m_s=window_speed if use_window else None,
            speed_window_sec=window_span if use_window else 0.,
            speed_window_samples=window_count if use_window else 2,
        )
        if min(instant_speed, target_speed) <= c.target_speed_deadband_m_s:
            # One small instantaneous contradiction is not a confirmed stop.
            # Use ONLY an already established, positive short window, and
            # strictly decrease output on this new physical sample. A second
            # weak sample stops; two positive samples are needed to re-arm.
            if (use_window and not urgent_decline and distance > near_distance + .10
                    and window_count >= 4 and window_span >= .10
                    and window_speed >= .10 and -.05 <= instant_speed <= c.target_speed_deadband_m_s
                    and self._last_matching_rpm is not None and self._last_matching_rpm > 0
                    and not self._stop_suspect_pending and delta_time <= .10):
                self._stop_suspect_pending = True
                self._positive_after_suspect = 0
                old = self._last_matching_rpm
                rpm = min(c.max_tracking_base_rpm, max(0., old-c.tracking_fall_rpm_per_sec*delta_time))
                self._last_matching_rpm = rpm
                self._filtered_target_speed = rpm*c.wheel_circumference_m/60.
                return self._result(
                    "ready", stamp=stamp, eligible=rpm > 0, range_rate_m_s=range_rate,
                    target_speed_m_s=self._filtered_target_speed, target_rpm=rpm,
                    feedforward_rpm=max(0., rpm-c.baseline_rpm),
                    uncompensated_range_rate_m_s=uncompensated, rotation_rate_m_s=interval_rotation,
                    unbounded_target_rpm=rpm, matching_cap_rpm=c.max_tracking_base_rpm,
                    matching_rate_limited=True, decline_policy="stop_suspect_bounded", **diagnostics,
                )
            # Near/rapidly-closing, strongly negative, repeated weak or
            # unsupported evidence still removes walking history immediately.
            self._filtered_target_speed = 0.0
            self._last_matching_rpm = None
            self._speed_samples = [sample]  # no old walking window after a stop
            self._stop_suspect_pending = False
            self._positive_after_suspect = 0
            return self._result(
                "no_forward_motion", stamp=stamp,
                range_rate_m_s=range_rate, target_speed_m_s=min(instant_speed, target_speed),
                uncompensated_range_rate_m_s=uncompensated, rotation_rate_m_s=interval_rotation,
                decline_policy="stop_evidence", **diagnostics,
            )
        if self._stop_suspect_pending:
            self._positive_after_suspect += 1
            if self._positive_after_suspect >= 2:
                self._stop_suspect_pending = False
                self._positive_after_suspect = 0
        if self._filtered_target_speed is None:
            self._filtered_target_speed = target_speed
        else:
            alpha = c.target_speed_filter_alpha
            self._filtered_target_speed += alpha * (target_speed - self._filtered_target_speed)
        if c.max_tracking_base_rpm > 0 and urgent_decline:
            self._filtered_target_speed = min(self._filtered_target_speed, target_speed)
        rpm_unbounded = max(0.0, self._filtered_target_speed * 60.0 / c.wheel_circumference_m)
        legacy_limit = c.baseline_rpm + min(20.0, c.max_feedforward_rpm)
        rpm_limit = legacy_limit
        if c.max_tracking_base_rpm > 0:
            rpm_limit = min(100.0, c.max_tracking_base_rpm)
            # Two frames may start a match, but must not immediately launch
            # beyond the former budget from a single range derivative.
            if self._sample_count < 3:
                rpm_limit = min(rpm_limit, legacy_limit)
        target_rpm = min(rpm_unbounded, rpm_limit)
        rate_limited = False
        if c.max_tracking_base_rpm > 0 and self._last_matching_rpm is not None:
            old = self._last_matching_rpm
            if target_rpm > old:
                target_rpm = min(target_rpm, old + c.tracking_rise_rpm_per_sec * delta_time)
            elif not urgent_decline:
                # Far positive target motion may include normal catch-up.
                # A negative relative range rate alone is not a stopped person.
                target_rpm = min(rpm_limit, max(target_rpm, old - c.tracking_fall_rpm_per_sec * delta_time))
            rate_limited = abs(target_rpm-min(rpm_unbounded, rpm_limit)) > 1e-9
        self._last_matching_rpm = target_rpm
        return self._result(
            "ready_capped" if rpm_unbounded > rpm_limit else "ready", stamp=stamp,
            eligible=target_rpm > 0.0, range_rate_m_s=range_rate,
            target_speed_m_s=self._filtered_target_speed, target_rpm=target_rpm,
            feedforward_rpm=max(0.0, target_rpm - c.baseline_rpm),
            uncompensated_range_rate_m_s=uncompensated, rotation_rate_m_s=interval_rotation,
            unbounded_target_rpm=rpm_unbounded, matching_cap_rpm=rpm_limit,
            matching_rate_limited=rate_limited,
            decline_policy="immediate_near_or_ttc" if urgent_decline else "bounded_far_positive",
            **diagnostics,
        )

    def _update_shared_sample(self, sample, bearing):
        """Profile mode: ONE accepted raw/encoder timeline for closure and FF.

        All original current-sample identity/time/encoder/yaw gates ran above.
        The legacy estimator and its policy remain unchanged when not opted in.
        """
        c, w, stamp = self.config, self.shared_window, sample.timestamp
        previous = self._previous
        reset_reason = None
        # Low-yaw geometry may intermittently omit rotation. A bounded omitted
        # contribution <=.02m/s can share the numerical zero convention, rather
        # than resetting solely because None became an explicit 0.0.
        if sample.rotation_rate_m_s is None:
            uncertainty = abs(math.radians(sample.yaw_rate_dps)*sample.distance_m*
                              math.tan(math.radians(bearing)))
            if uncertainty > .02:
                return self._result("rotation_uncertainty", stamp=stamp, clear=True)
            sample = replace(sample, rotation_rate_m_s=0.)
        if previous is not None:
            dt = stamp-previous.timestamp
            if sample.feedback_timestamp < previous.feedback_timestamp-1e-9:
                return self._result("out_of_order_feedback", stamp=stamp, clear=True)
            if dt > .18+1e-9:
                reset_reason = "shared_physical_gap"
            elif self._turn_gap_pending:
                low, high = self._gap_yaw_bounds or (previous.yaw_rate_dps,previous.yaw_rate_dps)
                spread = max(high,sample.yaw_rate_dps)-min(low,sample.yaw_rate_dps)
                # Conservative near-axis bound over the full allowed +/-10deg
                # bearing band. Larger/non-bracketed rotations rebuild as before.
                uncertainty = max(previous.distance_m,sample.distance_m)*math.tan(math.radians(10))*math.radians(spread)
                if uncertainty > .04:
                    reset_reason = "shared_gap_rotation_uncertainty"
            if reset_reason is not None:
                self._clear_chain()
                previous = None
            elif dt >= c.min_sample_interval_sec:
                delta = sample.distance_m-previous.distance_m
                instant = delta/dt-.5*(previous.rotation_rate_m_s+sample.rotation_rate_m_s)
                if abs(delta) > c.max_distance_jump_m or abs(instant) > c.max_abs_range_rate_m_s:
                    return self._result("range_rate_out_of_bounds", stamp=stamp, clear=True)
            else:
                # Keep the earlier baseline; do not fracture the shared window
                # with sub-resolution samples or make them fresh FF evidence.
                return self._result("interval_too_short", stamp=stamp)
        self._turn_gap_pending = False
        self._gap_yaw_bounds = None
        self._previous = sample
        rate = w.update(uid=self._target_id, stamp=stamp, raw=sample.distance_m,
                        rotation=sample.rotation_rate_m_s, ego_speed=sample.ego_forward_m_s,
                        feedback_stamp=sample.feedback_timestamp)
        self._sample_count = len(w.samples)
        if rate is None or w.target_speed is None:
            return self._result("warming_up", stamp=stamp, chain_reset_reason=reset_reason)
        speed, instant_speed = w.target_speed, w.instant_target_speed
        if (abs(rate) > c.max_abs_range_rate_m_s
                or abs(speed) > c.max_abs_target_speed_m_s
                or abs(instant_speed) > c.max_abs_target_speed_m_s):
            return self._result("target_speed_out_of_bounds", stamp=stamp, clear=True)
        near = c.min_distance_m+.23
        urgent = (sample.distance_m <= near or
                  (rate < 0 and (sample.distance_m-near)/-rate <= .4))
        # Unknown/window rebuild is not a stop. A valid near/rapid-closure
        # sample remains able to remove matching immediately, including a new
        # strong approach before the regression fully catches up.
        if urgent:
            speed = min(speed, instant_speed)
        elif (w.instant_rate < 0 and
              (sample.distance_m-c.min_distance_m)/-w.instant_rate <= .25):
            urgent = True
            speed = min(speed,instant_speed)
        diagnostics = dict(
            range_rate_m_s=rate, target_speed_m_s=speed,
            rotation_rate_m_s=sample.rotation_rate_m_s,
            instantaneous_target_speed_m_s=instant_speed,
            window_target_speed_m_s=w.target_speed, speed_window_sec=w.span,
            speed_window_samples=len(w.samples),
        )
        if speed <= c.target_speed_deadband_m_s:
            self._last_matching_rpm = None
            self._filtered_target_speed = 0.
            return self._result("no_forward_motion", stamp=stamp,
                                decline_policy="shared_stop_evidence", **diagnostics)
        unbounded = speed*60/c.wheel_circumference_m
        cap = min(100.,c.max_tracking_base_rpm) if c.max_tracking_base_rpm > 0 else c.baseline_rpm+min(20.,c.max_feedforward_rpm)
        if len(w.samples) < 3:
            cap = min(cap,c.baseline_rpm+min(20.,c.max_feedforward_rpm))
        rpm = min(unbounded,cap)
        old = self._last_matching_rpm
        if old is not None and previous is not None:
            dt = stamp-previous.timestamp
            if rpm > old:
                rpm = min(rpm,old+c.tracking_rise_rpm_per_sec*dt)
            elif not urgent:
                rpm = min(cap,max(rpm,old-c.tracking_fall_rpm_per_sec*dt))
        self._last_matching_rpm = rpm
        self._filtered_target_speed = speed
        return self._result(
            "ready_capped" if unbounded > cap else "ready", stamp=stamp, eligible=rpm>0,
            target_rpm=rpm, feedforward_rpm=max(0.,rpm-c.baseline_rpm),
            unbounded_target_rpm=unbounded, matching_cap_rpm=cap,
            matching_rate_limited=abs(rpm-min(unbounded,cap))>1e-9,
            decline_policy="shared_near_or_ttc" if urgent else "shared_window",
            **diagnostics,
        )
