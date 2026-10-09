"""Capture-bound small yaw lead, parked or following.

Never grants translation or extends a lease.
"""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class OutwardTrajectoryLead:
    uid: int
    capture_id: int
    capture_timestamp: float
    first_timestamp: float
    x: float
    rate_dps: float
    correction_rpm: int

    def matches(self, uid, cap, stamp, now):
        return bool(self.uid == uid and self.capture_id == cap
                    and self.capture_timestamp == stamp
                    and 0 <= now - stamp <= .25
                    and 0 < self.first_timestamp < stamp
                    and abs(self.correction_rpm) == 4)


def make_outward_lead(obs, now, *, hfov, deadband, release_margin, enabled):
    if (not enabled or obs is None or obs.reason != "capture_rate_valid"
            or not obs.outward_consistent or obs.rate_dps is None):
        return None
    values = (obs.control_x, obs.rate_dps, obs.capture_timestamp,
              obs.first_timestamp, hfov, deadband, release_margin, now)
    if not all(math.isfinite(float(v)) for v in values):
        return None
    error = (obs.control_x - .5) * hfov
    age = now - obs.capture_timestamp
    # Three independent captures, both increments outward. Stay in the
    # small-error corridor; ordinary steering owns larger displacements.
    if not (0 <= age <= .25 and .08 <= obs.span_sec <= .35
            and 1 <= abs(error) <= 6 and 3 <= abs(obs.rate_dps) <= 60
            and error * obs.rate_dps > 0):
        return None
    predicted = abs(error) + abs(obs.rate_dps) * min(.25, age + .15)
    boundary = max(0., deadband) + max(0., release_margin)
    if predicted <= boundary:
        return None
    rpm = 4  # CAP191: 2 RPM failed to establish useful early motion.
    return OutwardTrajectoryLead(obs.target_id, obs.capture_frame_id,
        obs.capture_timestamp, obs.first_timestamp, obs.control_x,
        obs.rate_dps, rpm if error > 0 else -rpm)


def apply_outward_lead(result, lead, x, now):
    if (lead is None or not lead.matches(lead.uid, lead.capture_id, lead.capture_timestamp, now)
            or abs(x-lead.x) > 1e-6 or result.predictive_braking
            or result.output_floor_reason == "predictive_countersteer"):
        return result
    limit = min(4, max(0, int(result.correction_limit_rpm)))
    if limit < 4:
        return result  # Never bypass a quality/safety ceiling.
    correction = min(limit, abs(lead.correction_rpm)) * (1 if lead.correction_rpm > 0 else -1)
    return replace(result, correction_rpm=correction, unsaturated_rpm=float(correction),
        correction_limit_rpm=float(limit), correction_policy_limit_rpm=float(limit),
        output_floor_reason="outward_trajectory_lead", output_floor_rpm=0.,
        startup_kick_active=False, target_rate_valid=True,
        target_image_rate_dps=lead.rate_dps, outward_lead=lead)
