"""Capture-time steering observations. No identity, depth or motor authority.

Use the detector box already associated with this UID, never another detection.
The tracker box remains authoritative elsewhere (including search and ReID).
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math


@dataclass(frozen=True)
class SteeringObservation:
    target_id: int
    capture_frame_id: int
    capture_timestamp: float
    tracker_x: float
    detector_x: float | None
    control_x: float
    rate_dps: float | None
    reason: str
    span_sec: float = 0.0
    first_timestamp: float = 0.0
    outward_consistent: bool = False
    inward_turnaround_rate_dps: float | None = None
    # Separate from ordinary velocity: can preserve at most 2 RPM of an
    # existing same-side yaw demand, never seed lead/forward/identity authority.
    outward_continuity_rate_dps: float | None = None


class CaptureSteeringEvidence:
    def __init__(self):
        self.samples = deque(maxlen=3)
        self.key = None
        self.last = None
        # Inward-only history survives quality-mode changes. It cannot seed
        # position correction, outward lead or motion authorization.
        self.brake_samples = deque(maxlen=3)
        self.brake_key = None
        self.reliable_anchor = None

    def observe(self, target, frame, now, hfov):
        uid, cap, stamp = int(target.track_id), int(frame.capture_frame_id), float(frame.capture_timestamp)
        x = target.center[0] / max(1, frame.width)
        fallback = SteeringObservation(uid, cap, stamp, x, None, x, None, "unqualified")
        if self.last is not None and uid == self.last.target_id:
            if cap == self.last.capture_frame_id and stamp == self.last.capture_timestamp:
                # Reading a frame again is not a new velocity sample.
                if 0 <= now-stamp <= .25:
                    return self.last
                return fallback
            if cap <= self.last.capture_frame_id or stamp <= self.last.capture_timestamp:
                return fallback
        obs = target.depth_observation
        braking_only = obs is None and target.braking_observation is not None
        if braking_only:
            obs = target.braking_observation
        valid = bool(uid > 0 and cap > 0 and frame.width > 0 and frame.height > 0
            and math.isfinite(stamp) and 0 < stamp <= now and now-stamp <= .25
            and target.confidence >= .5 and obs is not None
            and obs.target_id == uid and obs.raw_track_id > 0
            and obs.capture_frame_id == cap and abs(obs.capture_timestamp-stamp) < 1e-6
            and obs.source == ("yolo_braking_only" if braking_only else "yolo_detector"))
        detector_x = area = None
        if valid:
            a, b, c, d = map(float, obs.bbox)
            ta, tb, tc, td = map(float, target.bbox)
            valid = all(math.isfinite(v) for v in (a,b,c,d,ta,tb,tc,td))
            area = (c-a)*(d-b)
            tracker_area = (tc-ta)*(td-tb)
            overlap = max(0., min(c,tc)-max(a,ta))*max(0., min(d,td)-max(b,tb))
            detector_x = (a+c)/2/frame.width
            valid = bool(valid and 0 <= a < c <= frame.width and 0 <= b < d <= frame.height
                and c-a >= 24 and d-b >= 48 and tracker_area > 0
                and .4 <= area/tracker_area <= 2.5 and overlap/max(1.,area) >= .5
                and abs(detector_x-x) <= .10)
        if not valid:
            self.samples.clear()
            self.brake_samples.clear()
            self.brake_key = None
            self.key = None
            self.last = fallback
            self.reliable_anchor = None
            return fallback
        brake_key = (uid, int(obs.raw_track_id))
        if not braking_only:
            self.reliable_anchor = (brake_key, stamp)
        if brake_key != self.brake_key:
            self.brake_samples.clear()
        self.brake_key = brake_key
        if self.brake_samples:
            t, px, pa = self.brake_samples[-1]
            if not (.025 <= stamp-t <= .25 and abs(detector_x-px) <= .20
                    and .6 <= area/pa <= 1.67):
                self.brake_samples.clear()
        self.brake_samples.append((stamp, detector_x, area))
        # Braking-only edge geometry must not become two of the three samples
        # granting outward lead on the first recovered normal-quality frame.
        key = (uid, int(obs.raw_track_id), braking_only)
        if key != self.key:
            self.samples.clear()
        self.key = key
        if self.samples:
            t, px, pa = self.samples[-1]
            dt = stamp-t
            if not (.025 <= dt <= .25 and abs(detector_x-px) <= .20 and .6 <= area/pa <= 1.67):
                self.samples.clear()
        self.samples.append((stamp, detector_x, area))
        corrected, rate, span, reason = x, None, 0., "warming"
        first_timestamp, outward_consistent = 0., False
        if len(self.samples) >= 2:
            # Bounded lag correction, with no single-frame direction reversal.
            corrected = x + max(-.06, min(.06, detector_x-x))
            if (corrected-.5)*(x-.5) < 0:
                corrected = .5
            reason = "position_corrected"
        if len(self.samples) == 3:
            values = list(self.samples)
            slopes = [(q[1]-p[1])*hfov/(q[0]-p[0]) for p,q in zip(values,values[1:])]
            span = values[-1][0]-values[0][0]
            first_timestamp = values[0][0]
            outward_consistent = bool(slopes[0]*slopes[1] > 0
                and min(map(abs, slopes)) >= 3
                and min(map(abs, slopes)) >= .25 * max(map(abs, slopes)))
            # Two consistent increments, not one jittering bbox. The rate is
            # relative image motion, NOT target ground velocity or body yaw.
            if (span <= .35 and slopes[0]*slopes[1] >= 0
                    and max(map(abs,slopes)) <= 120 and abs(slopes[0]-slopes[1]) <= 40):
                # Both increments must agree, but braking needs the newest
                # interval, not their mean. Averaging the older slower interval
                # delayed braking while accelerating through CAP172..179.
                rate, reason = slopes[-1], "capture_rate_valid"
            else:
                reason = "rate_inconsistent"
        if len(self.samples) < 3 and len(self.brake_samples) == 3:
            values = list(self.brake_samples)
            slopes = [(q[1]-p[1])*hfov/(q[0]-p[0]) for p,q in zip(values, values[1:])]
            if (values[-1][0]-values[0][0] <= .35 and slopes[0]*slopes[1] > 0
                    and max(map(abs, slopes)) <= 120 and abs(slopes[0]-slopes[1]) <= 40
                    and slopes[-1]*(x-.5) < 0):
                corrected, rate = x, slopes[-1]
                span = values[-1][0]-values[0][0]
                first_timestamp = values[0][0]
                reason = "braking_continuity:capture_rate_valid"
                outward_consistent = False
        turnaround_rate = None
        if rate is None and len(self.brake_samples) == 3:
            values = list(self.brake_samples)
            p, q, r = values
            a, b = (q[1]-p[1])*hfov/(q[0]-p[0]), (r[1]-q[1])*hfov/(r[0]-q[0])
            # First inward interval after outward motion is NOT ordinary
            # velocity/lead evidence. A separate brake-only cue additionally
            # needs fresh agreeing physical yaw at the PID consumer.
            if (r[0]-p[0] <= .35 and .05 <= r[0]-q[0] <= .25
                    and abs(r[1]-q[1]) >= .004 and a*(x-.5) > 0 and b*(x-.5) < 0
                    and all((v[1]-.5)*(x-.5) > 0 for v in values)
                    and 3 <= abs(b) <= 60 and abs(a) <= 60 and abs(a-b) <= 40):
                turnaround_rate = b
        outward_rate = None
        if braking_only:
            anchor = self.reliable_anchor
            if (anchor is not None and anchor[0] == brake_key
                    and 0 <= stamp-anchor[1] <= .8
                    and (a <= 2 or c >= frame.width-2)
                    and rate is not None
                    and .08 <= span <= .35 and 1 <= abs(rate) <= 60
                    and min(map(abs, slopes)) >= 1
                    and abs(x-.5)*hfov > 6 and rate*(x-.5) > 0
                    and all((v[1]-.5)*(x-.5) > 0 for v in self.samples)):
                outward_rate = rate
            # Keep the existing position/direction. Geometry may only supply
            # inward velocity or bounded outward continuity, never outward lead.
            corrected = x
            if rate is not None and rate * (x-.5) >= 0:
                rate = None
            reason = "braking_only:" + reason
        self.last = SteeringObservation(uid, cap, stamp, x, detector_x, corrected, rate, reason,
                                        span, first_timestamp, outward_consistent, turnaround_rate,
                                        outward_rate)
        return self.last


def matching_control_x(observation, target_id, capture_id, capture_timestamp, fallback):
    """Only propagate this exact observation to the lateral fast loop."""
    if (observation is not None and observation.target_id == target_id
            and observation.capture_frame_id == capture_id
            and observation.capture_timestamp == capture_timestamp):
        return observation.control_x
    return fallback
