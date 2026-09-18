"""Read-only recording diagnostics. Never feeds values back into control."""
from dataclasses import dataclass
import math
from typing import Optional


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def age_ms(now, stamp):
    stamp = number(stamp)
    return None if stamp is None or stamp <= 0 else (now-stamp)*1000.


@dataclass(frozen=True)
class FollowRecordingSnapshot:
    published_at: float
    capture_frame_id: int
    uid: Optional[int]
    source: str
    reason: str
    set_distance_m: float
    target_x: Optional[float]
    raw_distance_m: Optional[float]
    used_distance_m: Optional[float]
    depth_timestamp: Optional[float]
    depth_detail: str
    speed_timestamp: Optional[float]
    target_speed_m_s: Optional[float]
    relative_speed_m_s: Optional[float]
    speed_status: str
    pid_timestamp: Optional[float]
    pid_rpm: Optional[float]
    matching_base_rpm: Optional[float]
    forward_scale_rpm: float


def build_follow_snapshot(controller, frame, decision, source, now):
    """Called inside the existing serialized decision path, only if recording.

    Copy scalar diagnostics from the already-computed result; no estimation,
    sensor access, authority getters or extra locking.
    """
    uid = controller.active_target_id
    target = next((p for p in frame.persons if p.track_id == uid), None)
    visible = target is not None and controller.search_state == 'none'
    evidence = getattr(controller, '_longitudinal_motion_evidence', None)
    if not visible or getattr(evidence, 'target_id', None) != uid:
        evidence = None
    speed_stamp = getattr(evidence, 'sample_timestamp', None)
    if getattr(evidence, 'status', None) == 'transient_bridge':
        # Bridge target_speed is old evidence, NOT a new speed measurement.
        origin = getattr(getattr(controller, '_longitudinal_bridge', None), 'origin', None)
        speed_stamp = getattr(origin, 'sample_timestamp', None)
    pid = getattr(controller, 'last_distance_pid_result', None) if visible else None
    state = frame.distance_state
    return FollowRecordingSnapshot(
        published_at=now, capture_frame_id=frame.capture_frame_id, uid=uid,
        source=source, reason=decision.reason or 'none',
        set_distance_m=controller.cfg.target_distance_m,
        target_x=None if target is None or frame.width <= 0 else target.center[0]/frame.width,
        raw_distance_m=state.raw_distance_m, used_distance_m=frame.distance_m,
        depth_timestamp=state.sample_timestamp, depth_detail=state.source_detail,
        speed_timestamp=speed_stamp,
        target_speed_m_s=getattr(evidence, 'target_speed_m_s', None),
        relative_speed_m_s=getattr(evidence, 'range_rate_m_s', None),
        speed_status=getattr(evidence, 'status', 'unavailable'),
        pid_timestamp=getattr(controller, '_distance_pid_last_sample_timestamp', None) if pid else None,
        pid_rpm=getattr(pid, 'output_rpm', None),
        matching_base_rpm=getattr(pid, 'tracking_base_rpm', None),
        forward_scale_rpm=controller.cfg.forward_max_rpm,
    )


def recording_authority(owner):
    """Snapshot immutable cached timing, without the mutating authority getter."""
    timing = getattr(owner, '_depth30_linear_timing', None)
    if (timing is None or timing.snapshot != getattr(owner, '_depth30_linear_snapshot', None)
            or getattr(owner, '_explicit_stop_requested', False)
            or getattr(owner, '_brake_hold_active', False)
            or getattr(owner, 'search_state', 'none') != 'none'):
        return None
    return timing


@dataclass(frozen=True)
class FollowRecordingView:
    observation_cap: Optional[int] = None
    uid: Optional[int] = None
    snapshot_age_ms: Optional[float] = None
    status: str = 'missing'
    source: str = 'none'
    reason: str = 'none'
    target_x: Optional[float] = None
    set_distance_m: Optional[float] = None
    raw_distance_m: Optional[float] = None
    used_distance_m: Optional[float] = None
    distance_error_m: Optional[float] = None
    depth_age_ms: Optional[float] = None
    depth_status: str = 'missing'
    depth_detail: str = ''
    target_speed_m_s: Optional[float] = None
    relative_speed_m_s: Optional[float] = None
    speed_age_ms: Optional[float] = None
    speed_status: str = 'missing'
    estimator_status: str = 'unavailable'
    pid_rpm: Optional[float] = None
    matching_base_rpm: Optional[float] = None
    pid_age_ms: Optional[float] = None
    authorized_rpm: Optional[float] = None
    authority_remaining_ms: Optional[float] = None
    authority_status: str = 'missing'

    @classmethod
    def from_snapshot(cls, snapshot, timing, now):
        if snapshot is None:
            return cls()
        s = snapshot
        age = age_ms(now, s.published_at)
        if age is None or age < 0:
            return cls(status='future' if age is not None else 'invalid')
        depth_age, speed_age, pid_age = (
            age_ms(now, stamp) for stamp in (s.depth_timestamp, s.speed_timestamp, s.pid_timestamp))
        def freshness(sample_age):
            return ('missing' if sample_age is None else 'future' if sample_age < 0 else
                    'fresh' if sample_age <= 180. else 'stale')
        status = 'current' if age <= 180. else 'stale'
        speed_status = freshness(speed_age)
        valid_speed = speed_status == 'fresh' and status == 'current'
        if s.speed_status == 'transient_bridge' and valid_speed:
            speed_status = 'bridge'
        if valid_speed and number(s.target_speed_m_s) is None and number(s.relative_speed_m_s) is None:
            speed_status = 'unavailable'
        used, target = number(s.used_distance_m), number(s.set_distance_m)
        authority_status, rpm, remaining = 'none', None, None
        if timing is not None:
            kind, percent, uid, stamp = timing.snapshot
            remaining = (timing.depth_expires_at-now)*1000.
            if uid != s.uid:
                authority_status = 'uid_mismatch'
            elif stamp > now:
                authority_status = 'future'
            elif remaining <= 0:
                authority_status = 'expired'
                rpm = 0.
            else:
                authority_status = 'active'
                if timing.feedforward_expires_at is not None and now >= timing.feedforward_expires_at:
                    percent = min(percent, timing.distance_only_percent)
                    authority_status = 'distance_only'
                rpm = (percent*s.forward_scale_rpm/100. if kind == 'forward' else
                       -percent*s.forward_scale_rpm/100. if kind == 'backward' else None)
        return cls(
            observation_cap=s.capture_frame_id, uid=s.uid, snapshot_age_ms=age,
            status=status, source=s.source, reason=s.reason, target_x=number(s.target_x),
            set_distance_m=target, raw_distance_m=number(s.raw_distance_m), used_distance_m=used,
            distance_error_m=None if used is None or target is None else used-target,
            depth_age_ms=depth_age, depth_status=freshness(depth_age), depth_detail=s.depth_detail,
            target_speed_m_s=number(s.target_speed_m_s) if valid_speed else None,
            relative_speed_m_s=number(s.relative_speed_m_s) if valid_speed else None,
            speed_age_ms=speed_age, speed_status=speed_status, estimator_status=s.speed_status,
            pid_rpm=number(s.pid_rpm) if freshness(pid_age) == 'fresh' else None,
            matching_base_rpm=number(s.matching_base_rpm) if freshness(pid_age) == 'fresh' else None,
            pid_age_ms=pid_age, authorized_rpm=rpm, authority_remaining_ms=remaining,
            authority_status=authority_status,
        )

    def labels(self):
        def fmt(v, signed=False):
            return 'n/a' if v is None else (f'{v:+.2f}' if signed else f'{v:.2f}')
        def age(v):
            return 'n/a' if v is None else f'{v:.0f}ms'
        return (
            f'EST Vt {fmt(self.target_speed_m_s, True)} Vrel {fmt(self.relative_speed_m_s, True)} m/s '
            f'{self.speed_status.upper()} {age(self.speed_age_ms)}',
            f'RANGE raw {fmt(self.raw_distance_m)} used {fmt(self.used_distance_m)} '
            f'SET {fmt(self.set_distance_m)} ERR {fmt(self.distance_error_m, True)} m',
            f'PID {fmt(self.pid_rpm)} BASE {fmt(self.matching_base_rpm)} '
            f'AUTH {fmt(self.authorized_rpm)} RPM {self.authority_status.upper()}',
            f'OBS U{self.uid} CAP {self.observation_cap} X {fmt(self.target_x)} '
            f'DEPTH {age(self.depth_age_ms)} {self.depth_status.upper()} {self.source}',
            f'WHY {self.reason} / EST {self.estimator_status}',
        )

    @classmethod
    def csv_columns(cls):
        return tuple('follow_'+name for name in cls.__dataclass_fields__)

    def csv_values(self):
        return tuple('' if (value := getattr(self, name)) is None else
                     f'{value:.4f}' if isinstance(value, float) else value
                     for name in self.__dataclass_fields__)
