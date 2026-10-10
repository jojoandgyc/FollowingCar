"""Sample-driven distance PI: one qualified observation, one wheel pair.

No hardware, target-speed estimation, recovery ramp, queue, or axis leases.
The adapter owns identity/measurement qualification; the sole executor owns
current hardware safety and the physical write. Publication never renews the
source clocks, and ordinary updates do not change the revocation epoch.
"""
from dataclasses import dataclass, replace
from contextlib import contextmanager
import math
import os
import threading
from typing import Optional

from .short_follow_yaw import ShortFollowYawObservation, tapered_center


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class ShortFollowConfig:
    enabled: bool = False
    target_distance_m: float = 1.4
    stop_margin_m: float = .10
    restart_margin_m: float = .15
    max_rpm: int = 200
    kp_per_sec: float = 3.
    ki_per_sec2: float = .4
    integral_max_m_s: float = .8
    deadband_m: float = .03
    wheel_circumference_m: float = .816814
    max_integral_gap_sec: float = .30
    memory_sec: float = .35
    braking_stop_distance_m: float = 1.1
    deceleration_m_s2: float = 1.
    response_delay_sec: float = .15
    braking_margin_m: float = .02
    yaw_max_delta_rpm: int = 16
    yaw_full_error_ratio: float = .30
    pivot_max_rpm: int = 8
    center_deadband_ratio: float = .08
    yaw_camera_hfov_deg: float = 60.
    yaw_damping_sec: float = .10
    depth_ttl_sec: float = .30
    visual_ttl_sec: float = .50
    write_period_sec: float = .05
    stop_refresh_sec: float = 1.

    def __post_init__(self):
        values = tuple(value for key, value in self.__dict__.items() if key != "enabled")
        if (not all(_finite(value) for value in values)
                or not 1.1 <= self.target_distance_m <= 3.
                or not 0 <= self.stop_margin_m < self.restart_margin_m <= .30
                or not 1 <= self.max_rpm <= 200
                or not 0 < self.kp_per_sec <= 10
                or not 0 <= self.ki_per_sec2 <= 5
                or not 0 <= self.integral_max_m_s <= 2
                or not 0 <= self.deadband_m <= .10
                or not .1 <= self.wheel_circumference_m <= 3.
                or not .05 <= self.max_integral_gap_sec <= self.depth_ttl_sec
                or not self.max_integral_gap_sec <= self.memory_sec <= 1.
                or not 1.1 <= self.braking_stop_distance_m <= self.target_distance_m
                or not .1 <= self.deceleration_m_s2 <= 1.
                or not .15 <= self.response_delay_sec <= 1.
                or not .02 <= self.braking_margin_m <= .20
                or not 0 <= self.yaw_max_delta_rpm <= 24
                or not 0 <= self.center_deadband_ratio < .5
                or not self.center_deadband_ratio < self.yaw_full_error_ratio <= .5
                or not 0 <= self.pivot_max_rpm <= 8
                or not 30 <= self.yaw_camera_hfov_deg <= 120
                or not 0 <= self.yaw_damping_sec <= .20
                # A finite 350 ms delivery-window trial is opt-in. The
                # compatibility default remains 300 ms; _speed_cap budgets
                # the entire selected watchdog before every plan is issued.
                or not .05 <= self.depth_ttl_sec <= .35
                or not .05 <= self.visual_ttl_sec <= .50
                or not .02 <= self.write_period_sec <= .10
                or not .10 <= self.stop_refresh_sec <= 1.):
            raise ValueError("invalid paired PI config; requires bounded PI, braking and finite clocks")
        if any(int(v) != v for v in (self.max_rpm, self.yaw_max_delta_rpm, self.pivot_max_rpm)):
            raise ValueError("short-follow wheel limits must be integer RPM")

    @property
    def pivot_limit_rpm(self):
        return int(min(self.pivot_max_rpm, self.max_rpm, self.yaw_max_delta_rpm // 2))

    def pivot_allowed(self, distance_m, center_x_ratio):
        """Geometric permission only; identity/source clocks remain external.

        Shared by the planner and settled-search handoff. This never grants
        translation or allows a pending/uncertain identity to authorize yaw.
        The planner separately requires its longitudinal distance-hold state.
        """
        return bool(_finite(distance_m) and _finite(center_x_ratio)
                    and 0 <= center_x_ratio <= 1
                    and distance_m > self.braking_stop_distance_m + 1e-9
                    and abs(center_x_ratio - .5) > self.center_deadband_ratio + 1e-9
                    and self.pivot_limit_rpm > 0)

    @classmethod
    def from_env(cls):
        mode = os.environ.get("FOLLOW_NORMAL_MODE", "legacy").strip().lower()
        if mode not in {"legacy", "paired"}:
            raise ValueError("FOLLOW_NORMAL_MODE must be legacy or paired")
        kwargs = {"enabled": mode == "paired",
                  "target_distance_m": float(os.environ.get("TARGET_DISTANCE", "1.4"))}
        # Reuse the production PI tuning, not the legacy RPM-domain PID or
        # the miniverify P-only formula. There is only one set of PI knobs.
        shared = {
            "kp_per_sec": "DISTANCE_PI_KP_PER_SEC",
            "ki_per_sec2": "DISTANCE_PI_KI_PER_SEC2",
            "integral_max_m_s": "DISTANCE_PI_INTEGRAL_MAX_M_S",
            "memory_sec": "DISTANCE_PI_MEMORY_SEC",
            "deadband_m": "DISTANCE_PID_DEADBAND_M",
            "wheel_circumference_m": "VISION_MMWAVE_FUSION_ENCODER_WHEEL_CIRCUMFERENCE_M",
            "braking_stop_distance_m": "DISTANCE_PI_BRAKING_STOP_DISTANCE_M",
            "deceleration_m_s2": "DISTANCE_APPROACH_DECELERATION_M_S2",
            "response_delay_sec": "DISTANCE_APPROACH_RESPONSE_DELAY_SEC",
            "yaw_camera_hfov_deg": "VISION_HFOV_DEG",
        }
        for name, env in shared.items():
            if env in os.environ:
                kwargs[name] = float(os.environ[env])
        kwargs["max_rpm"] = min(float(os.environ.get(name, "200"))
                                for name in ("FORWARD_MAX_RPM", "MOTOR_FORWARD_MAX_TARGET_RPM"))
        for name in cls.__dataclass_fields__:
            if name in {"enabled", "target_distance_m", "max_rpm"} or name in shared:
                continue
            value = os.environ.get("SHORT_FOLLOW_" + name.upper())
            if value is not None:
                kwargs[name] = float(value)
        # A shorter watchdog never permits integrating across a longer gap.
        kwargs["max_integral_gap_sec"] = min(kwargs.get("max_integral_gap_sec", .30),
                                             kwargs.get("depth_ttl_sec", .30))
        return cls(**kwargs)


@dataclass(frozen=True)
class ShortFollowObservation:
    uid: int
    capture_id: int
    capture_timestamp: float
    depth_timestamp: float
    distance_m: float
    center_x_ratio: float
    raw_distance_m: Optional[float] = None


@dataclass(frozen=True)
class ShortFollowPlan:
    uid: int
    sequence: int
    epoch: int
    left_rpm: int
    right_rpm: int
    reason: str
    capture_id: int
    capture_timestamp: float
    depth_timestamp: float
    expires_at: float
    distance_m: float
    base_request_rpm: float = 0.
    base_rpm: float = 0.
    speed_cap_rpm: float = 0.
    p_rpm: float = 0.
    i_rpm: float = 0.
    integral_dt_sec: float = 0.
    limit_reason: str = "none"
    # Yaw can be refreshed independently; the original depth/visual clocks
    # above are never renewed by such a refresh.
    longitudinal_reason: str = ""
    yaw_capture_id: int = 0
    yaw_capture_timestamp: float = 0.
    yaw_center_x_ratio: float = .5
    yaw_capture_yaw_deg: Optional[float] = None
    yaw_control_center_x_ratio: float = .5
    yaw_adjustment_reason: str = "none"

    @property
    def moving(self):
        return self.left_rpm != 0 or self.right_rpm != 0

    @property
    def forwarding(self):
        return self.left_rpm + self.right_rpm > 0

    @property
    def pivot(self):
        return bool(self.left_rpm == -self.right_rpm and (
            (self.reason == "pivot_left" and self.left_rpm < 0 < self.right_rpm)
            or (self.reason == "pivot_right" and self.right_rpm < 0 < self.left_rpm)))

    def valid(self, now):
        return bool(_finite(now) and max(self.capture_timestamp, self.depth_timestamp) <= now < self.expires_at)


@dataclass(frozen=True)
class ShortFollowSnapshot:
    active: bool
    uid: Optional[int]
    epoch: int
    plan: Optional[ShortFollowPlan]
    reason: str


class ShortFollowController:
    """Latest immutable pair with a separate epoch for actual revocations.

    Producers lock only local scalar arithmetic/publication. The executor
    holds a short commit guard across the final checks and dual-wheel write,
    ordering revocations against physical I/O. A failed update leaves the
    prior complete plan untouched.
    """
    def __init__(self, config):
        self.config = config
        self._lock = threading.RLock()
        self._state = ShortFollowSnapshot(False, None, 0, None, "inactive")
        self._sequence = 0
        self._source_floor = 0.
        self._last_depth = 0.
        self._last_capture = 0.
        self._last_capture_id = 0
        self._parked = True
        self._integral_m_s = 0.
        self._unacknowledged_integral = 0.
        self._integral_stamp = None
        self._latest_yaw = None
        self._latest_yaw_center = None

    def _reset_integral(self):
        self._integral_m_s = 0.
        self._unacknowledged_integral = 0.
        self._integral_stamp = None

    def _speed_cap(self, distance):
        """One continuous distance bound, not a latched braking/recovery state.

        The outer wheel must leave the configured test stopping space even
        if no next sample arrives before its original watchdog deadline.
        This uses the existing experimental deceleration/response settings;
        it is not a measured guarantee. Target velocity never raises the cap.
        Ordinary tightening returns a smaller positive command, not STOP.
        """
        cfg = self.config
        space = max(0., distance - cfg.braking_stop_distance_m - cfg.braking_margin_m)
        horizon = cfg.depth_ttl_sec + cfg.response_delay_sec
        a = cfg.deceleration_m_s2
        speed = (2. * space) / (math.sqrt((a * horizon) ** 2 + 2. * a * space) + a * horizon)
        return min(float(cfg.max_rpm), speed * 60. / cfg.wheel_circumference_m)

    def acknowledge_output(self, plan, base_rpm):
        """Anti-windup from the written pair, never from measured wheel lag.

        A late ACK cannot touch a newer sample or a revoked identity epoch.
        If the executor had a lower ceiling, undo positive increments since
        the last output ACK, including intermediate plans replaced before
        they could be written. Repeated writes cannot undo them twice;
        ordinary motor quantization is not saturation.
        """
        if not isinstance(plan, ShortFollowPlan) or not _finite(base_rpm) or base_rpm < 0:
            return False
        with self._lock:
            current = self._state.plan
            if (current is None or not self._state.active
                    or (current.uid, current.epoch, current.sequence) != (plan.uid, plan.epoch, plan.sequence)):
                return False
            if base_rpm < current.base_rpm - 1e-9 and self._unacknowledged_integral > 0:
                self._integral_m_s = max(0., self._integral_m_s - self._unacknowledged_integral)
            self._unacknowledged_integral = 0.
            return True

    def snapshot(self):
        with self._lock:
            return self._state

    @contextmanager
    def write_snapshot(self):
        """Serialize the last epoch check and wheel write against revocation.

        Executor acquires motor I/O first, performs any parking-mode work,
        then holds this only for cache checks and the bounded dual-wheel write.
        Producers never acquire motor I/O while holding this mailbox lock.
        """
        with self._lock:
            yield self._state

    def activate(self, uid, now):
        if not self.config.enabled or not isinstance(uid, int) or isinstance(uid, bool) or uid <= 0 or not _finite(now) or now <= 0:
            return False
        with self._lock:
            old = self._state
            if old.active and old.uid == uid:
                return True
            if old.uid is not None and old.uid != uid:
                self._source_floor = max(self._source_floor, now)
            self._state = ShortFollowSnapshot(True, uid, old.epoch + 1, None, "awaiting_observation")
            self._parked = True
            self._reset_integral()
            self._latest_yaw = None
            self._latest_yaw_center = None
            return True

    def _revoke(self, reason, now, *, deactivate):
        if not _finite(now) or now <= 0:
            raise ValueError("revocation requires the actual monotonic clock")
        with self._lock:
            old = self._state
            # Repeated watchdog ticks must not move the recovery watermark.
            # A new distinct hard event still creates a new revocation epoch.
            active = old.active and not deactivate
            if old.plan is None and old.active == active and old.reason == str(reason):
                return old
            self._source_floor = max(self._source_floor, now)
            self._parked = True
            self._reset_integral()
            self._latest_yaw = None
            self._latest_yaw_center = None
            self._state = ShortFollowSnapshot(active, old.uid, old.epoch + 1, None, str(reason))
            return self._state

    def revoke(self, reason, now):
        return self._revoke(reason, now, deactivate=False)

    def wait_for_existing_brake(self, now):
        """Suspend the pair without inventing a new observation rejection time.

        The adapter uses this only while an already requested search STOP is
        settling. Its release contract checks the physical STOP, fresh quiet
        feedback and post-STOP image/depth. Waiting for that contract is not a
        new hazard or identity verdict: processing time must not invalidate an
        otherwise eligible image captured just before this call (CAP507).

        Retire any queued plan by epoch, but preserve the previous hard source
        floor and sample watermarks. Release still needs a new qualified
        observation; no old plan, authorization or deadline is restored.
        """
        if not _finite(now) or now <= 0:
            raise ValueError("brake waiting requires the actual monotonic clock")
        with self._lock:
            old = self._state
            reason = "await_existing_brake_completion"
            if not old.active or (old.plan is None and old.reason == reason):
                return old
            self._parked = True
            self._reset_integral()
            self._latest_yaw = None
            self._latest_yaw_center = None
            self._state = ShortFollowSnapshot(True, old.uid, old.epoch + 1, None, reason)
            return self._state

    def deactivate(self, reason, now):
        return self._revoke(reason, now, deactivate=True)

    def _valid_yaw(self, obs, now):
        return bool(isinstance(obs, ShortFollowYawObservation)
            and _finite(now) and isinstance(obs.uid, int) and not isinstance(obs.uid, bool) and obs.uid > 0
            and isinstance(obs.capture_id, int) and not isinstance(obs.capture_id, bool) and obs.capture_id > 0
            and _finite(obs.capture_timestamp) and 0 < obs.capture_timestamp <= now
            and now < obs.capture_timestamp + self.config.visual_ttl_sec
            and _finite(obs.center_x_ratio) and 0 <= obs.center_x_ratio <= 1
            and (obs.capture_yaw_deg is None or _finite(obs.capture_yaw_deg)))

    @staticmethod
    def _yaw_newer(candidate, prior):
        return bool(prior is None or (
            candidate.capture_id > prior.capture_id
            and candidate.capture_timestamp > prior.capture_timestamp))

    def _mix_yaw(self, base, distance, center, longitudinal_reason):
        cfg = self.config
        left = right = int(base)
        reason = longitudinal_reason
        lateral = center - .5
        if (abs(lateral) > cfg.center_deadband_ratio + 1e-9
                and cfg.yaw_max_delta_rpm > 0):
            fraction = min(1., (abs(lateral) - cfg.center_deadband_ratio)
                           / (cfg.yaw_full_error_ratio - cfg.center_deadband_ratio))
            if base > 0:
                delta = min(int(base), max(1, int(round(fraction * cfg.yaw_max_delta_rpm))))
                if lateral < 0:
                    left, reason = int(base) - delta, "steer_left"
                else:
                    right, reason = int(base) - delta, "steer_right"
            elif (reason in {"target_distance_reached", "distance_restart_hysteresis", "reacquire_depth_pending"}
                    and cfg.pivot_allowed(distance, center)):
                pivot = max(1, int(round(fraction * cfg.pivot_limit_rpm)))
                left, right = (-pivot, pivot) if lateral < 0 else (pivot, -pivot)
                reason = "pivot_left" if lateral < 0 else "pivot_right"
        return left, right, reason

    def _with_yaw(self, plan, obs, *, current_yaw_deg=None, yaw_rate_deg_s=None,
                  center_limit=None):
        center, adjustment = tapered_center(obs, current_yaw_deg=current_yaw_deg,
            yaw_rate_deg_s=yaw_rate_deg_s, camera_hfov_deg=self.config.yaw_camera_hfov_deg,
            damping_sec=self.config.yaw_damping_sec)
        if center_limit is not None and abs(center - .5) > abs(center_limit - .5):
            center, adjustment = center_limit, "retained_yaw_taper"
        longitudinal_reason = plan.longitudinal_reason or (
            "forward" if plan.base_rpm > 0 else plan.limit_reason)
        left, right, reason = self._mix_yaw(plan.base_rpm, plan.distance_m, center, longitudinal_reason)
        return replace(plan, left_rpm=left, right_rpm=right, reason=reason,
            longitudinal_reason=longitudinal_reason, yaw_capture_id=obs.capture_id,
            yaw_capture_timestamp=obs.capture_timestamp, yaw_center_x_ratio=obs.center_x_ratio,
            yaw_capture_yaw_deg=obs.capture_yaw_deg, yaw_control_center_x_ratio=center,
            yaw_adjustment_reason=adjustment)

    def update_lateral(self, observation, now, *, current_yaw_deg=None, yaw_rate_deg_s=None,
                       publication_guard=None):
        """Publish a new qualified visual yaw without waiting for depth.

        Invalid/repeated samples do nothing. A first visual sample may be
        remembered while awaiting depth, but can never create a motion plan.
        Existing longitudinal values and *all* original deadlines are kept.
        """
        if not self._valid_yaw(observation, now):
            return None
        with self._lock:
            old = self._state
            if (not self.config.enabled or not old.active or old.uid != observation.uid
                    or observation.capture_timestamp <= self._source_floor
                    or not self._yaw_newer(observation, self._latest_yaw)):
                return None
            if publication_guard is not None:
                try:
                    if not publication_guard():
                        return None
                except Exception:
                    return None
            self._latest_yaw = observation
            self._latest_yaw_center, _ = tapered_center(observation,
                current_yaw_deg=current_yaw_deg, yaw_rate_deg_s=yaw_rate_deg_s,
                camera_hfov_deg=self.config.yaw_camera_hfov_deg,
                damping_sec=self.config.yaw_damping_sec)
            prior = old.plan
            if prior is None or not prior.valid(now):
                return None
            plan = self._with_yaw(prior, observation, current_yaw_deg=current_yaw_deg,
                                  yaw_rate_deg_s=yaw_rate_deg_s)
            self._sequence += 1
            plan = replace(plan, sequence=self._sequence)
            self._state = ShortFollowSnapshot(True, old.uid, old.epoch, plan, plan.reason)
            return plan

    def execution_plan(self, plan, now, *, current_yaw_deg=None, yaw_rate_deg_s=None):
        """Use fresh cached feedback to reduce an existing turn at write time.

        No publication, integration, new observation or authorization occurs.
        Even when feedback changes, a previously tapered request cannot grow
        back or change direction without a new qualified visual observation.
        """
        if (not isinstance(plan, ShortFollowPlan) or not plan.valid(now)
                or plan.yaw_capture_id <= 0):
            return plan
        obs = ShortFollowYawObservation(plan.uid, plan.yaw_capture_id,
            plan.yaw_capture_timestamp, plan.yaw_center_x_ratio, plan.yaw_capture_yaw_deg)
        proposed = self._with_yaw(plan, obs, current_yaw_deg=current_yaw_deg,
                                  yaw_rate_deg_s=yaw_rate_deg_s)
        old_delta = plan.left_rpm - plan.right_rpm
        new_delta = proposed.left_rpm - proposed.right_rpm
        if (abs(new_delta) > abs(old_delta)
                or old_delta * new_delta < 0):
            return plan
        return proposed

    def update(self, observation, now, *, current_yaw_deg=None, yaw_rate_deg_s=None,
               longitudinal_allowed=True):
        """Return a newly published plan, or None without changing a live plan."""
        if (not isinstance(observation, ShortFollowObservation) or not _finite(now)
                or type(longitudinal_allowed) is not bool):
            return None
        obs, cfg = observation, self.config
        numbers = (obs.capture_timestamp, obs.depth_timestamp, obs.distance_m, obs.center_x_ratio)
        if (not all(_finite(v) for v in numbers)
                or not isinstance(obs.capture_id, int) or isinstance(obs.capture_id, bool) or obs.capture_id <= 0
                or not isinstance(obs.uid, int) or isinstance(obs.uid, bool) or obs.uid <= 0
                or not 0 < obs.capture_timestamp <= now or not 0 < obs.depth_timestamp <= now
                or not 0 <= obs.center_x_ratio <= 1 or obs.distance_m <= 0
                or (obs.raw_distance_m is not None and (not _finite(obs.raw_distance_m) or obs.raw_distance_m <= 0))):
            return None
        expires = min(obs.depth_timestamp + cfg.depth_ttl_sec,
                      obs.capture_timestamp + cfg.visual_ttl_sec)
        if now >= expires:
            return None
        with self._lock:
            old = self._state
            if (not cfg.enabled or not old.active or old.uid != obs.uid
                    or min(obs.capture_timestamp, obs.depth_timestamp) <= self._source_floor
                    or obs.depth_timestamp <= self._last_depth
                    or obs.capture_timestamp < self._last_capture or obs.capture_id < self._last_capture_id):
                return None
            # Raw near evidence always wins over a lagging filtered distance.
            distance = min(obs.distance_m, obs.raw_distance_m) if obs.raw_distance_m is not None else obs.distance_m
            stop_at = cfg.target_distance_m + cfg.stop_margin_m
            start_at = cfg.target_distance_m + cfg.restart_margin_m
            effective_error = math.copysign(max(0., abs(obs.distance_m - cfg.target_distance_m) - cfg.deadband_m),
                                            obs.distance_m - cfg.target_distance_m)
            p_rpm = cfg.kp_per_sec * effective_error * 60. / cfg.wheel_circumference_m
            cap = self._speed_cap(distance)
            dt = 0.
            if self._integral_stamp is not None:
                gap = obs.depth_timestamp - self._integral_stamp
                if gap > cfg.memory_sec + 1e-9:
                    self._reset_integral()
                elif gap <= cfg.max_integral_gap_sec + 1e-9:
                    dt = gap
                # A bounded gap freezes memory; don't integrate missing time
                # or force a low-speed restart because a former plan expired.
            self._integral_stamp = obs.depth_timestamp
            base = 0
            demand = max(0., p_rpm + self._integral_m_s * 60. / cfg.wheel_circumference_m)
            limit_reason = "none"
            if not longitudinal_allowed:
                # A confirmed identity and one accepted physical range can
                # recenter while translation is still awaiting confirmation.
                # No synthetic distance, new lease or separate motor writer:
                # the same bounded complete pair carries zero forward speed.
                self._parked = True
                self._reset_integral()
                demand, dt = 0., 0.
                left = right = 0
                reason = limit_reason = "reacquire_depth_pending"
            elif distance <= stop_at + 1e-9:
                self._parked = True
                self._reset_integral()
                demand, dt = max(0., p_rpm), 0.
                left = right = 0
                reason = "target_distance_reached"
                limit_reason = reason
            elif self._parked and distance < start_at - 1e-9:
                self._reset_integral()
                demand, dt = max(0., p_rpm), 0.
                left = right = 0
                reason = "distance_restart_hysteresis"
                limit_reason = reason
            else:
                self._parked = False
                candidate_i = max(0., min(cfg.integral_max_m_s,
                    self._integral_m_s + cfg.ki_per_sec2 * effective_error * dt))
                candidate_demand = max(0., p_rpm + candidate_i * 60. / cfg.wheel_circumference_m)
                # Conditional integration: no accumulation further into an
                # output ceiling, while negative error can always unwind I.
                if candidate_i <= self._integral_m_s or candidate_demand <= cap:
                    self._unacknowledged_integral += max(0., candidate_i - self._integral_m_s)
                    self._integral_m_s = candidate_i
                demand = max(0., p_rpm + self._integral_m_s * 60. / cfg.wheel_circumference_m)
                base = max(0, min(int(math.floor(cap)), int(round(demand))))
                if demand > cap:
                    limit_reason = "max_rpm" if cap >= cfg.max_rpm else "distance_braking_cap"
                left = right = base
                reason = "forward"
            # A depth result may finish after a newer visual sample. It may
            # update PI and its own lease, but not roll yaw back to the old box.
            yaw = ShortFollowYawObservation(obs.uid, obs.capture_id,
                                           obs.capture_timestamp, obs.center_x_ratio)
            latest = self._latest_yaw
            if latest is not None and not self._yaw_newer(yaw, latest):
                yaw = latest
            else:
                self._latest_yaw = yaw
                self._latest_yaw_center = None
            self._sequence += 1
            plan = ShortFollowPlan(obs.uid, self._sequence, old.epoch, left, right, reason,
                obs.capture_id, obs.capture_timestamp, obs.depth_timestamp, expires, distance,
                base_request_rpm=demand, base_rpm=base, speed_cap_rpm=cap,
                p_rpm=p_rpm, i_rpm=self._integral_m_s * 60. / cfg.wheel_circumference_m,
                integral_dt_sec=dt, limit_reason=limit_reason,
                longitudinal_reason=reason)
            plan = self._with_yaw(plan, yaw, current_yaw_deg=current_yaw_deg,
                yaw_rate_deg_s=yaw_rate_deg_s, center_limit=self._latest_yaw_center)
            # Preserve an already planned taper even if this new Depth result
            # beats the first physical write and its feedback is unavailable.
            # Only a new visual observation releases this same-frame bound.
            self._latest_yaw_center = plan.yaw_control_center_x_ratio
            self._last_depth, self._last_capture, self._last_capture_id = obs.depth_timestamp, obs.capture_timestamp, obs.capture_id
            self._state = ShortFollowSnapshot(True, obs.uid, old.epoch, plan, plan.reason)
            return plan
