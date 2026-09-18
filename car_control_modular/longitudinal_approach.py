"""Forward speed reference, not an identity decision or a motor permission.

All inputs must come from an already accepted physical range observation.
The braking model is a commissioning assumption, NOT a certified stop bound.
No timers, hardware reads, retained target speeds or independent authorizations.
"""
from dataclasses import dataclass
import math
from typing import Optional


def closure_rotation_bound(*, depth, bearing_deg, yaw_dps, geometry_age=0.):
    """Upper bound on rotation's Z-rate in the existing <=15deg/s domain.

    Broader bearing is permitted for a conservative CLOSURE bound, not for
    target-speed matching. Includes bearing uncertainty since RGB capture.
    Near-axis pinhole/short-interval assumption; not a calibrated safety model.
    """
    if not all(math.isfinite(v) for v in (depth, bearing_deg, yaw_dps, geometry_age)):
        return None
    if depth <= 0 or abs(yaw_dps) > 15 or not -.02 <= geometry_age <= (.25 if abs(yaw_dps) <= 5 else .18):
        return None
    angle = abs(bearing_deg) + abs(yaw_dps) * abs(geometry_age)
    if angle > 45:
        return None
    bound = abs(math.radians(yaw_dps)) * depth * math.tan(math.radians(angle))
    return bound if bound <= .25 else None


def bounded_encoder_fallback(*, now, stamp, left, right, trustworthy,
                             max_rpm, feedback_limit, last_request, rise_rpm_s):
    """Brief stale-feedback bound, never a replacement fresh measurement.

    Acceleration is a commissioning assumption, not a certified physical bound.
    Include the last issued request so a large command cannot hide behind an
    older low wheel speed. No timestamps or motor permissions are extended.
    """
    if (not trustworthy or not all(math.isfinite(v) for v in (now, stamp, left, right))
            or not 0 <= now-stamp <= .30
            or max(abs(left),abs(right)) > feedback_limit):
        return max_rpm, 'encoder_unavailable_max_bound'
    measured = max(0., .5*(left+right))
    if now-stamp <= .15:
        return measured, 'encoder_fallback'
    rise = max(240., rise_rpm_s)
    request = max(0., last_request) if math.isfinite(last_request) else max_rpm
    return min(max_rpm,max(request,measured+rise*(now-stamp))), 'encoder_age_bound'


@dataclass(frozen=True)
class ApproachConfig:
    gain_per_sec: float = 1.0
    max_catchup_m_s: float = 0.60
    deceleration_m_s2: float = 0.40
    response_delay_sec: float = 0.20
    wheel_circumference_m: float = 0.816814
    no_matching_max_rpm: float = 0.0

    def __post_init__(self):
        for name, maximum in (("gain_per_sec", 2.0), ("max_catchup_m_s", .8),
                              ("deceleration_m_s2", 1.0), ("response_delay_sec", 1.0),
                              ("wheel_circumference_m", 3.0)):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= maximum:
                raise ValueError(f"invalid approach {name}: {value}")
        if not math.isfinite(self.no_matching_max_rpm) or not 0 <= self.no_matching_max_rpm <= 60:
            raise ValueError("no-matching reference must be within 0..60 RPM")

    @property
    def correction_max_rpm(self):
        return self.max_catchup_m_s * 60.0 / self.wheel_circumference_m


@dataclass(frozen=True)
class ApproachResult:
    output_rpm: float
    correction_rpm: float
    cap_rpm: float
    closing_speed_m_s: float
    braking_distance_m: float
    mode: str


def approach_reference(config: ApproachConfig, *, error_m: float, deadband_m: float,
                       tracking_base_rpm: Optional[float], range_rate_m_s: float,
                       max_output_rpm: float, measurement_age_sec: float = 0.,
                       raw_closure_valid: bool = False) -> ApproachResult:
    values = (error_m, deadband_m, range_rate_m_s, max_output_rpm, measurement_age_sec)
    if (not all(math.isfinite(v) for v in values) or deadband_m < 0
            or max_output_rpm < 0 or measurement_age_sec < 0):
        raise ValueError("approach reference requires finite range/rate and nonnegative limits")
    if tracking_base_rpm is not None and (
            not math.isfinite(tracking_base_rpm) or tracking_base_rpm < 0):
        raise ValueError("invalid approach matching base")
    # None means unknown speed, NOT proof that the person has stopped. No
    # old target speed or 20RPM launch bias is invented in that case.
    base = 0.0 if tracking_base_rpm is None else tracking_base_rpm
    closing = max(0.0, -range_rate_m_s)
    a, delay = config.deceleration_m_s2, config.response_delay_sec + measurement_age_sec
    remaining = max(0.0, error_m - deadband_m)
    braking_distance = closing * delay + closing * closing / (2.0 * a)
    # Both the commanded closing speed and the observed closing speed must
    # leave room for the response delay. Stable form of the quadratic root.
    root = math.sqrt((a * delay) ** 2 + 2.0 * a * remaining)
    command_bound = 2.0 * a * remaining / (root + a * delay)
    measured_bound = math.sqrt(2.0 * a * max(0.0, remaining - closing * delay))
    bound = min(command_bound, measured_bound)
    chase_limit = config.max_catchup_m_s
    if tracking_base_rpm is None and raw_closure_valid and config.no_matching_max_rpm > 0:
        # Distance-only request, never an invented human velocity. The same
        # distance ramp, braking envelope and caller's slew/lease still apply.
        chase_limit = config.no_matching_max_rpm * config.wheel_circumference_m / 60.
    demand = min(chase_limit, config.gain_per_sec * remaining)
    correction = min(demand, bound)
    scale = 60.0 / config.wheel_circumference_m
    mode = "catchup"
    if remaining == 0:
        mode = "matching" if base > 0 else "stop"
    elif correction + 1e-9 < demand or braking_distance >= remaining:
        mode = "decelerating"
    elif tracking_base_rpm is None:
        mode = "distance_only"
    # Below the target band, even a positive estimated target speed cannot
    # create forward motion here. The caller owns any protected reverse.
    if error_m < -deadband_m:
        return ApproachResult(0., 0., 0., closing, braking_distance, "too_close")
    cap = min(max_output_rpm, base + bound * scale)
    return ApproachResult(min(cap, base + correction * scale), correction * scale,
                          cap, closing, braking_distance, mode)


class RawDepthClosingWindow:
    """Shared short physical-depth regression for closure and target velocity.

    Caller verifies identity, range acceptance and encoder alignment. Rotation
    contributes to camera Z but not translation closure; integrate it over the
    identical depth interval before fitting. Optional measured ego velocities
    are integrated over those same intervals for target velocity. Zero target
    speed can coexist with valid closure. No motor authority is kept here.
    """
    def __init__(self, *, max_gap_sec=.18):
        if not math.isfinite(max_gap_sec) or not .18 <= max_gap_sec <= .30:
            raise ValueError("invalid raw closure max gap")
        self.max_gap_sec = max_gap_sec
        self.reset()

    def reset(self):
        self.samples = []
        self.uid = None
        self.rate = None
        self.status = "warming_up"
        self.span = 0.
        self.target_speed = None
        self.instant_rate = None
        self.instant_target_speed = None

    def update(self, *, uid, stamp, raw, rotation=None, ego_speed=None, feedback_stamp=None):
        if (uid is None or any(v is None or not math.isfinite(v) for v in (stamp, raw))
                or raw <= 0 or (rotation is not None and (
                    not math.isfinite(rotation) or abs(rotation) > .25))):
            self.reset()
            self.status = "invalid_raw_geometry"
            return None
        if ((ego_speed is None) != (feedback_stamp is None)
                or (ego_speed is not None and (
                    not math.isfinite(ego_speed) or not math.isfinite(feedback_stamp)
                    or abs(feedback_stamp-stamp) > .15))):
            self.reset()
            self.status = "invalid_encoder_alignment"
            return None
        if uid != self.uid:
            self.reset()
            self.uid = uid
        if self.samples and stamp <= self.samples[-1][0]:
            return self.rate  # no repeated sample can advance the window
        if (self.samples and feedback_stamp is not None and self.samples[-1][4] is not None
                and feedback_stamp < self.samples[-1][4]):
            self.reset()
            self.status = "out_of_order_encoder"
            return None
        if self.samples and (stamp-self.samples[-1][0] > self.max_gap_sec
                             or (rotation is None) != (self.samples[-1][2] is None)):
            self.reset()
            self.uid = uid
        recent = [s for s in self.samples if stamp-s[0] <= .18][-15:]
        # Keep the usual180ms fit on normal samples. Only bridge a short gap
        # with its last physical endpoint; never retimestamp or extend a grant.
        self.samples = recent or self.samples[-1:]
        self.samples.append((stamp, raw, rotation, ego_speed, feedback_stamp))
        self.span = stamp-self.samples[0][0]
        if len(self.samples) < 2 or self.span < .025:
            self.rate = None
            self.target_speed = None
            self.instant_rate = None
            self.instant_target_speed = None
            self.status = "warming_up"
            return None
        correction = 0.
        points = [(0., self.samples[0][1])]
        for old, new in zip(self.samples, self.samples[1:]):
            dt = new[0]-old[0]
            correction += .5*((old[2] or 0.)+(new[2] or 0.))*dt
            points.append((new[0]-self.samples[0][0], new[1]-correction))
        mt = sum(t for t, _ in points)/len(points)
        mz = sum(z for _, z in points)/len(points)
        rate = sum((t-mt)*(z-mz) for t, z in points)/sum((t-mt)**2 for t, _ in points)
        # Do not replace an implausible raw jump with zero approach speed.
        # Caller uses the explicit conservative fallback, never the 60RPM path.
        if abs(rate) > 3.:
            self.reset()
            self.uid = uid
            self.status = "raw_rate_out_of_bounds"
            return None
        self.rate, self.status = rate, "raw_depth_window"
        old, new = self.samples[-2:]
        self.instant_rate = ((new[1]-old[1])/(new[0]-old[0])
                             - .5*((old[2] or 0.)+(new[2] or 0.)))
        self.target_speed = self.instant_target_speed = None
        if all(s[3] is not None for s in self.samples):
            # Integrate MEASURED ego speed over exactly the same intervals as
            # the rotation-corrected depth regression, not just final RPM.
            travel = 0.
            target_points = [(points[0][0], points[0][1])]
            for i, (old, new) in enumerate(zip(self.samples, self.samples[1:]), 1):
                travel += .5*(old[3]+new[3])*(new[0]-old[0])
                target_points.append((points[i][0], points[i][1]+travel))
            mz = sum(z for _, z in target_points)/len(target_points)
            self.target_speed = sum((t-mt)*(z-mz) for t,z in target_points)/sum((t-mt)**2 for t,_ in target_points)
            self.instant_target_speed = self.instant_rate+.5*(self.samples[-2][3]+self.samples[-1][3])
        return rate
