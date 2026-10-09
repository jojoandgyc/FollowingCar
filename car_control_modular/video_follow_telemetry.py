"""Read-only recording diagnostics. Never feeds values back into control."""
from dataclasses import dataclass
import math
from typing import Optional

from .detector_identity_lease import DetectorIdentityLease, ValidatedVisualObservation


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def age_ms(now, stamp):
    stamp = number(stamp)
    return None if stamp is None or stamp <= 0 else (now-stamp)*1000.


def _paired_motion_kind(left, right):
    """Describe cached signed requests only; never infer physical movement."""
    left, right = number(left), number(right)
    if left is None or right is None:
        return ''
    if left == right == 0:
        return 'stop'
    if left + right == 0:
        return 'pivot_right' if left > 0 else 'pivot_left'
    if min(left, right) < 0:
        return 'invalid_pair'
    return 'forward' if left == right else 'steer_right' if left > right else 'steer_left'


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
    control_mode: str = 'legacy'
    paired_left_rpm: Optional[float] = None
    paired_right_rpm: Optional[float] = None
    paired_sequence: Optional[int] = None
    paired_epoch: Optional[int] = None
    paired_capture_frame_id: Optional[int] = None
    paired_expires_at: Optional[float] = None
    paired_pi_request_rpm: Optional[float] = None
    paired_base_rpm: Optional[float] = None
    paired_speed_cap_rpm: Optional[float] = None
    paired_p_rpm: Optional[float] = None
    paired_i_rpm: Optional[float] = None
    paired_integral_dt_sec: Optional[float] = None
    paired_limit_reason: str = ''
    paired_motion_kind: str = ''


def build_follow_snapshot(controller, frame, decision, source, now, *, paired_snapshot=None):
    """Called inside the existing serialized decision path, only if recording.

    Copy scalar diagnostics from the already-computed result; no estimation,
    sensor access, authority getters or extra locking.
    """
    uid = controller.active_target_id
    target = next((p for p in frame.persons if p.track_id == uid), None)
    visible = target is not None and controller.search_state == 'none'
    paired = paired_snapshot is not None and paired_snapshot.active
    plan = paired_snapshot.plan if paired else None
    if plan is not None and (plan.uid != uid or plan.epoch != paired_snapshot.epoch):
        plan = None
    speed_enabled = not paired and getattr(controller.cfg, 'distance_target_motion_control_enable', True)
    evidence = getattr(controller, '_longitudinal_motion_evidence', None)
    if not speed_enabled or not visible or getattr(evidence, 'target_id', None) != uid:
        evidence = None
    speed_stamp = getattr(evidence, 'sample_timestamp', None)
    if getattr(evidence, 'status', None) == 'transient_bridge':
        # Bridge target_speed is old evidence, NOT a new speed measurement.
        origin = getattr(getattr(controller, '_longitudinal_bridge', None), 'origin', None)
        speed_stamp = getattr(origin, 'sample_timestamp', None)
    pid = getattr(controller, 'last_distance_pid_result', None) if visible and not paired else None
    state = frame.distance_state
    return FollowRecordingSnapshot(
        published_at=now, capture_frame_id=frame.capture_frame_id, uid=uid,
        source=source, reason=decision.reason or 'none',
        set_distance_m=controller.cfg.target_distance_m,
        target_x=None if target is None or frame.width <= 0 else target.center[0]/frame.width,
        raw_distance_m=state.raw_distance_m, used_distance_m=frame.distance_m,
        depth_timestamp=plan.depth_timestamp if plan is not None else state.sample_timestamp,
        depth_detail=state.source_detail,
        speed_timestamp=speed_stamp,
        target_speed_m_s=getattr(evidence, 'target_speed_m_s', None),
        relative_speed_m_s=getattr(evidence, 'range_rate_m_s', None),
        speed_status=getattr(evidence, 'status', 'unavailable') if speed_enabled else 'disabled',
        pid_timestamp=getattr(controller, '_distance_pid_last_sample_timestamp', None) if pid else None,
        pid_rpm=getattr(pid, 'output_rpm', None),
        matching_base_rpm=getattr(pid, 'tracking_base_rpm', None),
        forward_scale_rpm=controller.cfg.forward_max_rpm,
        control_mode='paired' if paired else 'legacy',
        paired_left_rpm=None if plan is None else plan.left_rpm,
        paired_right_rpm=None if plan is None else plan.right_rpm,
        paired_sequence=None if plan is None else plan.sequence,
        paired_epoch=None if not paired else paired_snapshot.epoch,
        paired_capture_frame_id=None if plan is None else plan.capture_id,
        paired_expires_at=None if plan is None else plan.expires_at,
        paired_pi_request_rpm=None if plan is None else plan.base_request_rpm,
        paired_base_rpm=None if plan is None else plan.base_rpm,
        paired_speed_cap_rpm=None if plan is None else plan.speed_cap_rpm,
        paired_p_rpm=None if plan is None else plan.p_rpm,
        paired_i_rpm=None if plan is None else plan.i_rpm,
        paired_integral_dt_sec=None if plan is None else plan.integral_dt_sec,
        paired_limit_reason='' if plan is None else plan.limit_reason,
        paired_motion_kind='' if plan is None else _paired_motion_kind(plan.left_rpm, plan.right_rpm),
    )


@dataclass(frozen=True)
class PairedRecordingAuthority:
    """Cached plan permission, NOT a physical write receipt or measured RPM."""
    uid: Optional[int]
    epoch: int
    plan: object
    expires_at: Optional[float]
    blocked_reason: str = ''
    valid_from: Optional[float] = None


def recording_authority(owner):
    """Snapshot immutable cached timing, without the mutating authority getter."""
    short = getattr(owner, '_short_follow', None)
    if short is not None and short.config.enabled:
        state = short.snapshot()
        if state.active:
            plan = state.plan
            reason = ''
            if getattr(owner, '_runtime_shutdown_requested', False):
                reason = 'shutdown'
            elif (getattr(getattr(owner, '_motor_backend', None), 'motion_write_fault', None)
                    or getattr(getattr(owner, '_motor_backend', None), 'parking_release_fault', None)):
                reason = 'motor_fault'
            elif getattr(owner, '_explicit_stop_requested', False):
                reason = 'explicit_stop'
            elif getattr(owner, '_brake_hold_active', False):
                reason = 'brake_hold'
            elif (getattr(owner, 'search_state', 'none') != 'none'
                    or getattr(getattr(owner, '_follow_controller', None), 'search_state', 'none') != 'none'):
                reason = 'search'
            elif plan is None:
                reason = state.reason or 'waiting_observation'
            elif plan.uid != state.uid or plan.epoch != state.epoch:
                reason = 'epoch_or_uid_mismatch'
            elif getattr(getattr(owner, '_follow_controller', None), 'active_target_id', state.uid) != state.uid:
                reason = 'uid_mismatch'
            proof = getattr(owner, '_validated_visual_observation', None)
            deadline = None if plan is None else plan.expires_at
            valid_from = None if plan is None else max(plan.capture_timestamp, plan.depth_timestamp)
            if plan is not None:
                if (not isinstance(proof, ValidatedVisualObservation) or proof.uid != plan.uid
                        or (proof.continuation_sample_timestamp is not None
                            and proof.continuation_sample_timestamp != plan.depth_timestamp)):
                    reason = reason or 'identity_unavailable'
                else:
                    deadline = min(deadline, proof.expires_at)
                    valid_from = max(valid_from, proof.validated_at)
                identity = getattr(owner, '_detector_identity_lease', None)
                if identity is not None:
                    if not isinstance(identity, DetectorIdentityLease) or identity.uid != plan.uid:
                        reason = reason or 'identity_unavailable'
                    else:
                        deadline = min(deadline, identity.expires_at)
                        valid_from = max(valid_from, identity.observation_timestamp)
            return PairedRecordingAuthority(state.uid, state.epoch, plan, deadline, reason, valid_from)
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
    control_mode: str = 'legacy'
    paired_left_rpm: Optional[float] = None
    paired_right_rpm: Optional[float] = None
    paired_sequence: Optional[int] = None
    paired_epoch: Optional[int] = None
    paired_capture_frame_id: Optional[int] = None
    authorized_left_rpm: Optional[float] = None
    authorized_right_rpm: Optional[float] = None
    authority_sequence: Optional[int] = None
    authority_capture_frame_id: Optional[int] = None
    paired_pi_request_rpm: Optional[float] = None
    paired_base_rpm: Optional[float] = None
    paired_speed_cap_rpm: Optional[float] = None
    paired_p_rpm: Optional[float] = None
    paired_i_rpm: Optional[float] = None
    paired_integral_dt_sec: Optional[float] = None
    paired_limit_reason: str = ''
    paired_motion_kind: str = ''
    authority_motion_kind: str = ''

    @classmethod
    def from_snapshot(cls, snapshot, timing, now):
        if snapshot is None:
            return cls()
        s = snapshot
        paired_view = s.control_mode == 'paired' or isinstance(timing, PairedRecordingAuthority)
        age = age_ms(now, s.published_at)
        if age is None or age < 0:
            return cls(status='future' if age is not None else 'invalid',
                       control_mode='paired' if paired_view else 'legacy')
        depth_age, speed_age, pid_age = (
            age_ms(now, stamp) for stamp in (s.depth_timestamp, s.speed_timestamp, s.pid_timestamp))
        def freshness(sample_age):
            return ('missing' if sample_age is None else 'future' if sample_age < 0 else
                    'fresh' if sample_age <= 180. else 'stale')
        status = 'current' if age <= 180. else 'stale'
        depth_status = freshness(depth_age)
        if s.control_mode == 'paired' and s.paired_expires_at is not None:
            status = 'current' if now < s.paired_expires_at else 'stale'
            if depth_age is not None and depth_age >= 0:
                depth_status = 'fresh' if now < s.paired_expires_at else 'stale'
        speed_status = 'disabled' if paired_view or s.speed_status == 'disabled' else freshness(speed_age)
        valid_speed = not paired_view and speed_status == 'fresh' and status == 'current'
        if s.speed_status == 'transient_bridge' and valid_speed:
            speed_status = 'bridge'
        if valid_speed and number(s.target_speed_m_s) is None and number(s.relative_speed_m_s) is None:
            speed_status = 'unavailable'
        used, target = number(s.used_distance_m), number(s.set_distance_m)
        authority_status, rpm, remaining = 'none', None, None
        authorized_left = authorized_right = authority_sequence = authority_cap = None
        authority_motion = ''
        if isinstance(timing, PairedRecordingAuthority):
            plan = timing.plan
            remaining = None if timing.expires_at is None else (timing.expires_at-now)*1000.
            if timing.uid != s.uid:
                authority_status = 'uid_mismatch'
            elif timing.blocked_reason:
                authority_status = timing.blocked_reason
            elif plan is None:
                authority_status = 'waiting_observation'
            elif timing.valid_from is not None and timing.valid_from > now:
                authority_status = 'future'
            elif remaining is None or remaining <= 0:
                authority_status = 'expired'
                rpm = authorized_left = authorized_right = 0.
                authority_motion = 'stop'
            else:
                authority_status = 'active' if plan.moving else 'distance_stop'
                authorized_left, authorized_right = float(plan.left_rpm), float(plan.right_rpm)
                rpm = (authorized_left+authorized_right)/2.
                authority_motion = _paired_motion_kind(authorized_left, authorized_right)
                authority_sequence, authority_cap = plan.sequence, plan.capture_id
        elif timing is not None and s.control_mode != 'paired':
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
            depth_age_ms=depth_age, depth_status=depth_status, depth_detail=s.depth_detail,
            target_speed_m_s=number(s.target_speed_m_s) if valid_speed else None,
            relative_speed_m_s=number(s.relative_speed_m_s) if valid_speed else None,
            speed_age_ms=None if paired_view else speed_age, speed_status=speed_status,
            estimator_status='disabled' if paired_view else s.speed_status,
            pid_rpm=number(s.pid_rpm) if not paired_view and freshness(pid_age) == 'fresh' else None,
            matching_base_rpm=number(s.matching_base_rpm) if not paired_view and freshness(pid_age) == 'fresh' else None,
            pid_age_ms=None if paired_view else pid_age, authorized_rpm=rpm, authority_remaining_ms=remaining,
            authority_status=authority_status,
            control_mode='paired' if paired_view else 'legacy',
            paired_left_rpm=number(s.paired_left_rpm), paired_right_rpm=number(s.paired_right_rpm),
            paired_sequence=s.paired_sequence, paired_epoch=s.paired_epoch,
            paired_capture_frame_id=s.paired_capture_frame_id,
            authorized_left_rpm=authorized_left, authorized_right_rpm=authorized_right,
            authority_sequence=authority_sequence, authority_capture_frame_id=authority_cap,
            paired_pi_request_rpm=number(s.paired_pi_request_rpm),
            paired_base_rpm=number(s.paired_base_rpm),
            paired_speed_cap_rpm=number(s.paired_speed_cap_rpm),
            paired_p_rpm=number(s.paired_p_rpm), paired_i_rpm=number(s.paired_i_rpm),
            paired_integral_dt_sec=number(s.paired_integral_dt_sec),
            paired_limit_reason=s.paired_limit_reason,
            paired_motion_kind=s.paired_motion_kind,
            authority_motion_kind=authority_motion,
        )

    def labels(self):
        def fmt(v, signed=False):
            return 'n/a' if v is None else (f'{v:+.2f}' if signed else f'{v:.2f}')
        def age(v):
            return 'n/a' if v is None else f'{v:.0f}ms'
        if self.control_mode == 'paired':
            return (
                f'PAIR REQ {self.paired_motion_kind.upper()} L {fmt(self.paired_left_rpm)} R {fmt(self.paired_right_rpm)} RPM '
                f'CAP {self.paired_capture_frame_id} SEQ {self.paired_sequence}',
                f'RANGE raw {fmt(self.raw_distance_m)} used {fmt(self.used_distance_m)} '
                f'SET {fmt(self.set_distance_m)} ERR {fmt(self.distance_error_m, True)} m',
                f'AUTH PLAN {self.authority_motion_kind.upper()} L {fmt(self.authorized_left_rpm)} R {fmt(self.authorized_right_rpm)} '
                f'{self.authority_status.upper()} {age(self.authority_remaining_ms)} '
                f'SEQ {self.authority_sequence} NOT ACK',
                f'OBS U{self.uid} CAP {self.observation_cap} X {fmt(self.target_x)} '
                f'DEPTH {age(self.depth_age_ms)} {self.depth_status.upper()} {self.source}',
                f'WHY {self.reason} / PAIRED / TARGET SPEED OFF',
                f'PI REQ {fmt(self.paired_pi_request_rpm)} P {fmt(self.paired_p_rpm)} '
                f'I {fmt(self.paired_i_rpm)} BASE {fmt(self.paired_base_rpm)} '
                f'CAP {fmt(self.paired_speed_cap_rpm)} RPM {self.paired_limit_reason}',
            )
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
