#!/usr/bin/env python3
"""Read-only, versioned longitudinal metrics for one run and optional CAP range.

Example: python3 tools/follow_metrics.py RUN_DIR --cap-start 938 --cap-end 1088
No hardware, runtime imports, log writes, interpolation of missing distances,
or inference of physical vehicle stops from commanded zero RPM.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
from statistics import mean, median, pstdev
from zoneinfo import ZoneInfo


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def field(line, key):
    match = re.search(r"\b" + re.escape(key) + r"=([^\s]+)", line)
    return match.group(1) if match else None


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {"count": len(values), "mean": mean(values), "median": median(values),
            "p95": values[max(0, math.ceil(.95 * len(values)) - 1)], "max": values[-1]}


def ratio(n, d):
    return 100.0 * n / d if d else None


def short_speed_drops(records, duration):
    """Distinct ascending physical samples: >10RPM drop within 200ms.

    Classified by logged source, not inferred causality; zero includes safety
    stops and must not be minimized blindly. Older logs may lack UID evidence.
    """
    previous = {}
    reasons, magnitudes = Counter(), []
    for wall, sample, uid, rpm, source in records:
        old = previous.get(uid)
        if old is not None and sample <= old[1]:
            continue
        if old is not None and 0 <= wall-old[0] <= .2 and old[2]-rpm > 10:
            reasons['zero' if rpm == 0 else old[3]+'->'+source] += 1
            magnitudes.append(old[2]-rpm)
        previous[uid] = (wall, sample, rpm, source)
    return dict(count=len(magnitudes), per_10sec=10*len(magnitudes)/duration if duration>0 else None,
                transitions=dict(reasons), drop_rpm=distribution(magnitudes))


def analyze(run, cap_start=None, cap_end=None, target_distance=None, tolerance=.15, tail_sec=.18,
            timezone="Asia/Shanghai"):
    run = Path(run)
    log = run / "request_0513_modular.log"
    csv_path = run / "camera_raw.frames.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [r for r in all_rows if number(r.get("capture_frame_id")) is not None
            and (cap_start is None or int(r["capture_frame_id"]) >= cap_start)
            and (cap_end is None or int(r["capture_frame_id"]) <= cap_end)
            and number(r.get("capture_unix_sec")) is not None]
    if not rows:
        raise ValueError("No captured rows in requested CAP range")
    rows.sort(key=lambda r: float(r["capture_unix_sec"]))
    start, end_capture = float(rows[0]["capture_unix_sec"]), float(rows[-1]["capture_unix_sec"])
    end = end_capture + tail_sec
    # CSV contains unprocessed capture frames. Do not count them as failed
    # visual/range observations, or duplicate a recorded control frame.
    seen, selected = set(), []
    for row in rows:
        ctrl = row.get("control_frame_id")
        if not ctrl or ctrl in seen:
            continue
        seen.add(ctrl)
        if number(row.get("selected_target_id")) not in (None, 0, -1):
            selected.append(row)
    distances = [(float(r["capture_unix_sec"]), number(r.get("target_distance_m"))) for r in selected]
    valid = [(t, d) for t, d in distances if d is not None]
    missing_reasons = Counter(r.get("distance_detail", "") or "unspecified"
                              for r in selected if number(r.get("target_distance_m")) is None)
    missing_segments, in_missing = 0, False
    for _t, d in distances:
        if d is None and not in_missing:
            missing_segments += 1
        in_missing = d is None

    pid, base, physical, remaining, feedback = [], [], set(), [], []
    resets, revokes, bridge_reasons, gaps = Counter(), Counter(), Counter(), Counter()
    matching_sources = Counter()
    no_matching_reasons = Counter()
    no_matching_closing_sources = Counter()
    distance_control_mode = None
    pi_config, pi_samples = {}, {}
    pi_limits, pi_limit_records = {}, 0
    pi_pause_reasons, pi_pause_memory = Counter(), Counter()
    pi_pause_origins, pi_rejected_samples = set(), set()
    pi_suspend_reasons, pi_suspend_origins = Counter(), set()
    depth_cache_records, depth_cache_origins = Counter(), set()
    pi_reject_reasons = Counter()
    depth_continuation_records, depth_continuation_reasons = Counter(), Counter()
    depth_continuation_origins, depth_late_origins = set(), set()
    fallback_bounds = {}
    normal_live_continuations = set()
    motion_status_by_sample, far_fresh_low_rpm = {}, {}
    far_closing_resumes = set()
    detached_prior_samples, near_no_matching_stops = {}, set()
    integral_rpm, integral_caps = [], []
    approach_samples = {}
    shared_motion_samples = {}
    integral_at_cap = 0
    continuity = Counter()
    recovery_starts, recovery_limited = 0, 0
    approved, pid_targets = [], set()
    approved_rpm = []
    follow_ticks, follow_tick_intervals, periodic_tick_intervals = Counter(), [], []
    rotation_compensated_records = measured_recovery_records = 0
    parking_events, parking_release_ms = Counter(), []
    roi_extension_events, authorized_depth = Counter(), set()
    skipped_observations, reset_details = Counter(), Counter()
    skipped_derivative_resets = 0
    recovery_losses, restored_samples = {}, set()
    scheduling_recovery_samples = {}
    scheduling_recovery_rejections = Counter()
    scheduling_recovery_pauses = set()
    scheduling_recovery_discards = Counter()
    scheduling_recovery_completions = set()
    scheduling_recovery_above_initial = set()
    direct_stop_records = 0
    forward_scale_rpm = None
    matching_fresh = matching_capped = matching_rate_limited = 0
    matching_two_sample = matching_warmup_capped = 0
    chain_reset_reasons, bridge_decay_policies = Counter(), Counter()
    compensated_gap_samples, derivative_reset_samples = set(), set()
    matching_unbounded, matching_caps = [], []
    matching_window_audited = matching_window_records = 0
    matching_window_ms, matching_instant_speed = [], []
    decline_policies = Counter()
    far_closing_recoveries = set()
    replay_retained_origins, replay_derivative_retained_origins = set(), set()
    replay_hold_resets = 0
    feedback_audits = {}
    estimator_statuses = Counter()
    feedback_config = {}
    bias_samples = {}
    pid_speed_records, approved_speed_records, pid_losses = [], [], {}
    stale_live_revokes = 0
    wheel_dispatch_reasons = Counter()
    wheel_dispatch_losses = []
    wheel_positive_to_zero = 0
    motor_zero_reasons = {}
    last_motor = None
    boundary_continuity_samples = set()
    writes, inherited = [], None
    inherited_motor_stamp = None
    zone = ZoneInfo(timezone)
    motor = re.compile(r"LZ30EMA 电机命令:.*左轮=(" + NUMBER + r")转/分 右轮=(" + NUMBER + r")转/分")
    encoder = re.compile(r"前进归一转速=(" + NUMBER + r")/(" + NUMBER + r")RPM")
    with log.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                stamp = datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=zone).timestamp()
            except ValueError:
                continue
            if "depth_clock_config " in line:
                scale = number(field(line, "forward_max_rpm"))
                forward_scale_rpm = scale if scale is not None and scale > 0 else None
            if "longitudinal_feedback_config " in line and stamp <= end:
                feedback_config = {name: number(field(line, name)) for name in (
                    "kp_rpm_per_m", "ego_limit_rpm", "yaw_uncertainty_max_m_s",
                    "disagreement_skew_ms", "depth_ttl_ms", "bias_trial_rpm")}
            if "distance_control_mode " in line and stamp <= end:
                explicit_mode = field(line, "mode")
                distance_control_mode = (explicit_mode if explicit_mode in {"distance_pi", "legacy"}
                                         else 'distance_only' if field(line, 'matching_enabled') == 'False'
                                         else 'optional')
                if distance_control_mode == "distance_pi":
                    pi_config = {key: number(field(line, key)) for key in (
                        "kp_per_sec", "ki_per_sec2", "integral_max_m_s", "memory_ms", "depth_ttl_ms", "launch_request_rpm")}
            match = motor.search(line)
            direct_stop = "LZ30EMA 停车命令:" in line
            if match or direct_stop:
                # LZ30EMA raw right-wheel sign is opposite forward-normalized.
                left, right = (float(match[1]), -float(match[2])) if match else (0.0, 0.0)
                last_motor = (stamp, left, right)
                if direct_stop: motor_zero_reasons[stamp] = "direct_stop"
                if stamp < start:
                    inherited = (start, left, right)
                    inherited_motor_stamp = stamp
                elif stamp <= end:
                    writes.append((stamp, left, right))
            if "visible_wheel_dispatch " in line:
                requested = re.search(r"requested_forward_rpm=\(("+NUMBER+r"),\s*("+NUMBER+r")\)",line)
                applied = re.search(r"applied_forward_rpm=\(("+NUMBER+r"),\s*("+NUMBER+r")\)",line)
                if requested and applied:
                    req = (float(requested[1])+float(requested[2]))*.5
                    app = (float(applied[1])+float(applied[2]))*.5
                    reason = field(line,"reason") or "unknown"
                    pair = (float(applied[1]),float(applied[2]))
                    # Attribute actual zero-command dwell only when adjacent
                    # dispatch and motor log agree. Unknown remains unknown.
                    if (last_motor is not None and 0 <= stamp-last_motor[0] <= .05
                            and pair == last_motor[1:] and pair == (0.,0.)):
                        motor_zero_reasons[last_motor[0]] = reason
                    if start <= stamp <= end:
                        wheel_dispatch_reasons[reason] += 1
                        if req > 0:
                            wheel_dispatch_losses.append(max(0.,req-app))
                            wheel_positive_to_zero += pair == (0.,0.)
            if not start <= stamp <= end:
                continue
            if "depth_forward_continuation " in line:
                allowed = field(line, "allowed") or "unknown"
                depth_continuation_records[allowed] += 1
                depth_continuation_reasons[field(line, "reason") or "unknown"] += 1
                sample = number(field(line, "sample_ts"))
                if sample is not None:
                    depth_continuation_origins.add((field(line, "uid"), sample, allowed))
            if "depth_late_sample_continuation " in line:
                sample = number(field(line, "original_sample_ts"))
                if sample is not None:
                    depth_late_origins.add((field(line, "uid"), sample))
            if "depth_cache_retained " in line:
                reason = field(line, "reason") or "unknown"
                depth_cache_records[reason] += 1
                stamp = number(field(line, "anchor_ts"))
                if stamp is not None:
                    depth_cache_origins.add((field(line, "uid"), stamp))
            if "distance_pi_authority_suspended " in line:
                reason = field(line, "reason") or "unknown"
                pi_suspend_reasons[reason] += 1
                stamp = number(field(line, "sample_ts"))
                if stamp is not None:
                    pi_suspend_origins.add((field(line, "uid"), stamp, reason))
            if "distance_pi uid=" in line:
                pi_stamp = number(field(line, "sample_ts"))
                if pi_stamp is not None:
                    if distance_control_mode is None:
                        distance_control_mode = "distance_pi"
                    pi_samples[(field(line, "uid"), pi_stamp)] = {
                        "status": field(line, "status") or "unknown",
                        "brake_source": field(line, "brake_source") or "unknown",
                        "closure_source": field(line, "closure_source") or "unknown",
                        "integral_m_s": number(field(line, "integral_m_s")),
                        "i_rpm": number(field(line, "i_rpm")),
                        "sample_dt_sec": number(field(line, "sample_dt_sec")),
                        "integral_frozen": field(line, "integral_frozen"),
                        "launch_floor_rpm": number(field(line, "launch_floor_rpm")),
                        "total_demand_rpm": number(field(line, "total_demand_rpm")),
                        "envelope_rpm": number(field(line, "envelope_rpm")),
                        "ego_rpm": number(field(line, "ego_rpm")),
                        "feedback_age_ms": number(field(line, "feedback_age_ms")),
                        "software_rise_bypassed": field(line, "software_rise_bypassed"),
                        "demand_limit_reason": field(line, "demand_limit_reason"),
                        "motion_origin_ts": number(field(line, "motion_origin_ts")),
                        "motion_uncertainty_m_s": number(field(line, "motion_uncertainty_m_s")),
                    }
            if "distance_pi_pause " in line:
                pi_pause_reasons[field(line, "reason") or "unknown"] += 1
                pi_pause_memory[field(line, "memory_retained") or "unknown"] += 1
                pi_stamp = number(field(line, "sample_ts"))
                if pi_stamp is not None:
                    pi_pause_origins.add((field(line, "uid"), pi_stamp))
            if "distance_pi_admission_rejected " in line:
                pi_reject_reasons[field(line, "reason") or "unknown"] += 1
                pi_stamp = number(field(line, "sample_ts"))
                if pi_stamp is not None:
                    pi_rejected_samples.add((field(line, "uid"), pi_stamp))
            if "distance_pi_limit " in line:
                pi_limit_records += 1
                pi_stamp = number(field(line, "sample_ts"))
                requested = number(field(line, "requested_rpm"))
                limited = number(field(line, "approved_rpm"))
                before_i = number(field(line, "integral_before_m_s"))
                after_i = number(field(line, "integral_after_m_s"))
                quantum = number(field(line, "quantization_rpm"))
                if pi_stamp is not None and None not in (requested, limited, before_i, after_i):
                    key = (field(line, "uid"), pi_stamp)
                    previous_limit = pi_limits.get(key)
                    pi_limits[key] = {
                        "requested_rpm": max(requested, previous_limit["requested_rpm"]) if previous_limit else requested,
                        "approved_rpm": min(limited, previous_limit["approved_rpm"]) if previous_limit else limited,
                        "before_i": max(before_i, previous_limit["before_i"]) if previous_limit else before_i,
                        "after_i": min(after_i, previous_limit["after_i"]) if previous_limit else after_i,
                        "quantization_rpm": quantum if quantum is not None else (
                            previous_limit["quantization_rpm"] if previous_limit else None),
                    }
            if "depth_boundary_continuity " in line:
                sample = number(field(line,"sample_ts"))
                if sample is not None: boundary_continuity_samples.add((field(line,"uid"),sample))
            direct_stop_records += int(direct_stop)
            if "longitudinal_bias_trial " in line:
                sample = number(field(line, "sample_ts"))
                bias = number(field(line, "applied_bias_rpm"))
                if sample is not None and bias is not None:
                    key = (field(line, "uid"), sample)
                    bias_samples[key] = max(bias_samples.get(key, 0.), bias)
            if "longitudinal_feedback_audit " in line:
                sample = number(field(line, "sample_ts"))
                if sample is not None:
                    feedback_audits[(field(line, "uid"), sample)] = field(line, "compensation") or "unknown"
            if "longitudinal_motion target=" in line:
                estimator_statuses[field(line, "status") or "unknown"] += 1
                sample = number(field(line, "sample_ts"))
                if sample is not None:
                    # Motion log prints 6 decimals; authority prints full precision.
                    # Align diagnostics to microseconds, NEVER control timestamps.
                    motion_status_by_sample[(field(line, "target"), round(sample, 6))] = field(line, "status") or "unknown"
            if "longitudinal_motion_bridge " in line:
                sample = number(field(line, "sample_ts"))
                if sample is not None:
                    motion_status_by_sample[(field(line, "uid"), round(sample, 6))] = "transient_bridge"
            if "longitudinal_motion_reset " in line:
                sample = number(field(line, "sample_ts"))
                if sample is not None:
                    motion_status_by_sample[(field(line, "uid"), round(sample, 6))] = "reset:"+(field(line,"reason") or "unknown")
            if "depth_ff_detached " in line:
                sample = number(field(line, "sample_ts"))
                detached_approved = number(field(line, "approved_percent"))
                if sample is not None and detached_approved is not None:
                    detached_prior_samples[(field(line, "uid"), sample)] = detached_approved
            if "near_no_matching_stop " in line:
                sample = number(field(line, "sample_ts"))
                if sample is not None:
                    near_no_matching_stops.add((field(line, "uid"), sample))
            if "depth_far_closing_recovery " in line:
                sample = number(field(line, "sample_ts"))
                if sample is not None:
                    far_closing_resumes.add((field(line, "uid"), sample))
            if "longitudinal_motion target=" in line and field(line, "status") in {"ready", "ready_capped"}:
                matching_fresh += 1
                window_ms = number(field(line, "speed_window_ms"))
                if window_ms is not None:
                    matching_window_audited += 1
                    matching_window_records += window_ms > 0
                    matching_window_ms.append(window_ms)
                instant_speed = number(field(line, "instant_target_speed"))
                if instant_speed is not None: matching_instant_speed.append(instant_speed)
                matching_capped += field(line, "status") == "ready_capped"
                two_sample = number(field(line, "samples")) == 2
                matching_two_sample += two_sample
                matching_warmup_capped += two_sample and field(line, "status") == "ready_capped"
                matching_rate_limited += field(line, "matching_rate_limited") == "True"
                value = number(field(line, "unbounded_target_rpm"))
                if value is not None: matching_unbounded.append(value)
                value = number(field(line, "matching_cap_rpm"))
                if value is not None: matching_caps.append(value)
            if "stale_vision_depth_audit " in line and (number(field(line, "remaining_ms")) or 0)>0:
                stale_live_revokes += 1
            if "longitudinal_motion target=" in line or "longitudinal_motion_bridge " in line:
                reset_reason = field(line, "chain_reset_reason")
                if reset_reason not in (None, "None"):
                    chain_reset_reasons[reset_reason] += 1
            if "longitudinal_motion target=" in line and field(line, "decline_policy") is not None:
                decline_policies[field(line, "decline_policy")] += 1
            if "longitudinal_motion_bridge " in line:
                bridge_decay_policies[field(line, "decay_policy") or "unknown"] += 1
            if "longitudinal_motion_observation_skipped " in line:
                skipped_observations[field(line, "reason") or "unknown"] += 1
                skipped_derivative_resets += field(line, "derivative_reset") == "True"
                original = number(field(line, "original_sample_ts"))
                if original is not None:
                    key = (field(line, "uid"), original)
                    if field(line, "derivative_reset") == "True": derivative_reset_samples.add(key)
                    if field(line, "compensated_gap_preserved") == "True": compensated_gap_samples.add(key)
            if "longitudinal_replay_retained " in line:
                original = number(field(line, "origin_ts"))
                if original is not None:
                    key = (field(line, "uid"), original)
                    replay_retained_origins.add(key)
                    if field(line, "derivative_reset") == "False": replay_derivative_retained_origins.add(key)
            if "longitudinal_motion_reset " in line:
                replay_hold_resets += field(line, "depth_detail") == "depth_sample_observation_discarded_fused_radar_hold_hold"
                reset_details[field(line, "depth_detail") or "unknown"] += 1
            if "depth_recovery_continuity_restored " in line:
                if field(line, "normal_live_continuation") == "True":
                    normal_live_continuations.add((field(line, "uid"), field(line, "sample_ts")))
                sample = number(field(line, "sample_ts"))
                if sample is not None:
                    restored_samples.add((field(line, "uid"), sample))
            if "depth_scheduling_recovery " in line:
                sample = number(field(line, "sample_ts"))
                if sample is not None:
                    key = (field(line, "uid"), sample)
                    scheduling_recovery_samples[key] = field(line, "event")
                    if field(line, "ramp_completed") == "True":
                        scheduling_recovery_completions.add(key)
                    cap = number(field(line, "cap_rpm"))
                    initial = number(field(line, "previous_approved_rpm"))
                    if cap is not None and initial is not None and cap > initial:
                        scheduling_recovery_above_initial.add(key)
            if "depth_scheduling_recovery_gap " in line:
                sample = number(field(line, "original_sample_ts"))
                if field(line, "action") == "pause" and sample is not None:
                    scheduling_recovery_pauses.add((field(line, "uid"), sample))
                elif field(line, "action") == "discard":
                    scheduling_recovery_discards[field(line, "reason")] += 1
            if "depth_scheduling_recovery_rejected " in line:
                scheduling_recovery_rejections[field(line, "reason")] += 1
            if "depth_speed_recovery_limit " in line:
                sample, loss = number(field(line, "sample_ts")), number(field(line, "lost_rpm"))
                # Older logs can be compared only when an explicit startup
                # scale is present; never assume one percent equals one RPM.
                if loss is None and forward_scale_rpm is not None:
                    requested = number(field(line, "requested_percent"))
                    limited = number(field(line, "approved_percent"))
                    if requested is not None and limited is not None:
                        loss = (requested-limited)*forward_scale_rpm/100
                if sample is not None and loss is not None and loss >= 0:
                    key = (field(line, "uid"), sample)
                    recovery_losses[key] = max(recovery_losses.get(key, 0), loss)
            if "depth_roi_extension " in line:
                roi_extension_events[field(line, "status") or "unknown"] += 1
            if "follow_wheel_tick " in line:
                reason = field(line, "reason") or "unknown"
                follow_ticks[reason] += 1
                interval = number(field(line, "interval_ms"))
                if interval is not None:
                    follow_tick_intervals.append(interval)
                    if reason == "periodic":
                        periodic_tick_intervals.append(interval)
            if "longitudinal_motion target=" in line and field(line, "compensated") == "True":
                rotation_compensated_records += 1
            if "depth_measured_recovery " in line:
                measured_recovery_records += 1
                physical_stamp = number(field(line, "sample_ts"))
                if field(line, "distance_policy") == "far_closing_measured" and physical_stamp is not None:
                    far_closing_recoveries.add((field(line, "uid"), physical_stamp))
            if "follow_distance_hold_started " in line:
                parking_events["started"] += 1
            if "follow_distance_hold_observe " in line:
                parking_events["observe_" + (field(line, "status") or "unknown")] += 1
            if "follow_distance_hold_released " in line:
                parking_events["released"] += 1
                held_ms = number(field(line, "held_ms"))
                if held_ms is not None:
                    parking_release_ms.append(held_ms)
            if "distance_brake_fallback " in line:
                sample = number(field(line, 'sample_ts'))
                bound = number(field(line, 'bound_rpm'))
                if sample is not None and bound is not None:
                    fallback_bounds[(field(line,'uid'), sample)] = bound
            if "longitudinal_shared_window " in line:
                stamp = number(field(line, 'sample_ts'))
                if stamp is not None:
                    shared_motion_samples[(field(line, 'uid'), stamp)] = {
                        'status': field(line, 'status'),
                        'reset_reason': field(line, 'reset_reason'),
                        'span_ms': number(field(line, 'span_ms')),
                        'range_rate': number(field(line, 'range_rate')),
                        'target_speed': number(field(line, 'window_target_speed')),
                    }
            if "longitudinal_approach uid=" in line:
                stamp = number(field(line, 'sample_ts'))
                if stamp is not None:
                    approach_samples[(field(line, 'uid'), stamp)] = {
                        'mode': field(line, 'mode'),
                        'correction_rpm': number(field(line, 'correction_rpm')),
                        'braking_distance_m': number(field(line, 'braking_distance_m')),
                        'request_rpm': number(field(line, 'request_rpm')),
                        'closing_source': field(line, 'closing_source') or 'legacy_unknown',
                        'closing_window_ms': number(field(line, 'closing_window_ms')),
                        'no_matching_reason': field(line, 'no_matching_reason'),
                    }
            if "distance_pid actual=" in line:
                integral = re.search(r"\bi=(" + NUMBER + r")rpm", line)
                if integral:
                    integral_rpm.append(float(integral[1]))
                cap = number(field(line, "integral_cap_rpm"))
                if cap is not None:
                    integral_caps.append(cap)
                    integral_at_cap += field(line, "integral_at_cap") == "True"
                value = re.search(r"\boutput=(" + NUMBER + r")rpm", line)
                ff = re.search(r"\btracking_base=(" + NUMBER + r")rpm", line)
                if value and ff:
                    pid.append(float(value[1]))
                    base.append(float(ff[1]))
                    physical_stamp = number(field(line, "sample_ts"))
                    if float(ff[1]) <= 0:
                        key = (field(line, 'uid'), physical_stamp)
                        reason = approach_samples.get(key, {}).get('no_matching_reason')
                        if reason is None:
                            reason = motion_status_by_sample.get(
                                (key[0], round(physical_stamp, 6)), 'unknown') if physical_stamp is not None else 'unknown'
                        no_matching_reasons[reason] += 1
                        no_matching_closing_sources[approach_samples.get(key, {}).get('closing_source','unknown')] += 1
                    if physical_stamp is not None:
                        pid_speed_records.append((stamp, physical_stamp, field(line, "uid"),
                                                  float(value[1]), field(line, "matching_source") or "unknown"))
                    matching_sources[field(line, "matching_source") or
                                     ("legacy_matching" if float(ff[1]) > 0 else "none")] += 1
                target = re.search(r"\btarget=(" + NUMBER + r")m", line)
                if target:
                    pid_targets.add(float(target[1]))
            if "Astra depth timeline:" in line and field(line, "temporal") == "new_sample":
                physical_stamp = number(field(line, "sample_ts"))
                if physical_stamp is not None and number(field(line, "raw")) is not None and number(field(line, "distance")) is not None:
                    physical.add(physical_stamp)
            if "depth_linear_limit " in line:
                scale = number(field(line, "forward_scale_rpm"))
                if scale is not None and scale > 0:
                    forward_scale_rpm = scale
                rpm = number(field(line, "approved_forward_rpm"))
                if rpm is not None:
                    approved_rpm.append(rpm)
                    physical_stamp = number(field(line, "sample_ts"))
                    if physical_stamp is not None:
                        approved_speed_records.append((stamp, physical_stamp, field(line,"uid"), rpm,
                                                       "fresh" if field(line,"fresh")=="True" else "held"))
                        distance = number(field(line, "distance"))
                        if field(line, "fresh") == "True" and distance is not None and distance > 2 and rpm < 30:
                            key = (field(line, "uid"), physical_stamp)
                            far_fresh_low_rpm[key] = motion_status_by_sample.get((key[0], round(physical_stamp, 6)), "unknown")
                        loss = number(field(line, "pid_to_approved_loss_rpm"))
                        if loss is not None and loss >= 0:
                            key=(field(line,"uid"), physical_stamp)
                            pid_losses[key]=max(pid_losses.get(key,0), loss)
                value = number(field(line, "remaining_ms"))
                if value is not None:
                    remaining.append(value)
                match = re.search(r"approved=\[\('forward', (\d+)\)\]", line)
                if match:
                    approved.append(int(match[1]))
                    physical_stamp = number(field(line, "sample_ts"))
                    if field(line, "fresh") == "True" and int(match[1]) > 0 and physical_stamp is not None:
                        authorized_depth.add(physical_stamp)
            for marker, counter, key in (
                ("longitudinal_motion_reset ", resets, "reason"),
                ("depth_linear_revoked ", revokes, "reason"),
                ("longitudinal_motion_bridge ", bridge_reasons, "reason"),
                ("depth_recovery_gap ", gaps, "action"),
            ):
                if marker in line:
                    counter[field(line, key) or "unknown"] += 1
            if "depth_speed_recovery_started " in line:
                recovery_starts += 1
            if "depth_speed_recovery_limit " in line:
                recovery_limited += 1
            for marker in ("depth_ff_fallback", "depth_ff_admission_fallback", "depth30_replay_preserve"):
                if marker + " " in line:
                    continuity[marker] += 1
            if ("depth_linear_authority " in line
                    and field(line, "temporal_status") in {"duplicate", "older_than_anchor"}
                    and "snapshot=None" in line
                    and (number(field(line, "previous_remaining_ms")) or 0) > 0):
                # Diagnostic, not proof of a bug: safety evidence can still
                # legitimately revoke a live lease on a duplicate sample.
                continuity["duplicate_zero_with_live_lease"] += 1
            match = encoder.search(line)
            if match and "可信=True" in line:
                feedback.append((float(match[1]) + float(match[2])) / 2)

    events = ([inherited] if inherited else []) + writes
    dwell = Counter()
    zero_reason_dwell = Counter()
    rpm_integral = 0.0
    for index, (t, left, right) in enumerate(events):
        duration = max(0.0, (events[index + 1][0] if index + 1 < len(events) else end) - t)
        forward = (left + right) / 2
        kind = "zero" if left == right == 0 else "forward" if forward > 0 else "reverse" if forward < 0 else "yaw_only"
        dwell[kind] += duration
        if kind == "zero":
            # The inherited entry has window start substituted for its time.
            origin = t
            if index == 0 and inherited is not None:
                origin = inherited_motor_stamp
            zero_reason_dwell[motor_zero_reasons.get(origin,"unknown")] += duration
        rpm_integral += duration * forward
    covered = sum(dwell.values())
    ordered = sorted(physical)
    sample_gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    authorized = sorted(authorized_depth)
    authorized_gaps = [b-a for a, b in zip(authorized, authorized[1:])]
    target = target_distance if target_distance is not None else (next(iter(pid_targets)) if len(pid_targets) == 1 else None)
    errors = [d - target for _, d in valid] if target is not None else []
    valid_span = valid[-1][0] - valid[0][0] if len(valid) > 1 else 0
    selected_uids = sorted({r["selected_target_id"] for r in selected})
    warnings = []
    pi_losses = [loss for key, loss in pid_losses.items() if key in pi_samples and loss > 0]
    pi_integral_reductions = [max(0., v["before_i"] - v["after_i"]) for v in pi_limits.values()]
    pi_limit_classes = Counter(
        "unknown" if v["quantization_rpm"] is None else
        "quantization_compatible" if v["requested_rpm"]-v["approved_rpm"] <= v["quantization_rpm"]+1e-9
        else "material" for v in pi_limits.values())
    quantization_step = forward_scale_rpm / 100.0 if forward_scale_rpm is not None else None
    pi_frozen = [v["integral_frozen"] == "True" for v in pi_samples.values()
                 if v["integral_frozen"] in {"True", "False"}]
    if len(selected_uids) > 1:
        warnings.append("Multiple UIDs: split by target before interpreting distance drift")
    if target is None:
        warnings.append("No unique logged target distance; supply --target-distance for error metrics")
    if covered < end - start - .001:
        warnings.append("Motor commands do not cover the entire window; missing time is not zero")
    return {
        "schema_version": 2, "run": str(run.resolve()),
        "window": {"cap_start": int(rows[0]["capture_frame_id"]), "cap_end": int(rows[-1]["capture_frame_id"]),
                   "capture_duration_sec": end_capture - start, "log_tail_sec": tail_sec,
                   "log_duration_sec": end - start, "timezone": timezone},
        "distance": {"target_m": target, "tolerance_m": tolerance, "selected_control_frames": len(selected),
                     "uids": selected_uids, "missing_frames": len(selected) - len(valid),
                     "missing_percent": ratio(len(selected) - len(valid), len(selected)),
                     "missing_segments": missing_segments, "missing_reasons": dict(missing_reasons),
                     "start_m": valid[0][1] if valid else None, "end_m": valid[-1][1] if valid else None,
                     "drift_m_s": (valid[-1][1] - valid[0][1]) / valid_span if valid_span else None,
                     "absolute_error_m": distribution([abs(e) for e in errors]),
                     "signed_error_m": distribution(errors),
                     "actual_distance_m": distribution([d for _, d in valid]),
                     "distance_std_m": pstdev([d for _, d in valid]) if valid else None,
                     "within_tolerance_percent": ratio(sum(abs(e) <= tolerance for e in errors), len(errors))},
        "depth": {"accepted_distinct_samples": len(ordered),
                  "boundary_continuity_distinct_samples": len(boundary_continuity_samples),
                  "roi_extension_events": dict(roi_extension_events),
                  "positive_authorized_samples": len(authorized),
                  "positive_authorized_hz": (len(authorized)-1)/(authorized[-1]-authorized[0]) if len(authorized)>1 else None,
                  "positive_authorized_gap_ms": distribution([x*1000 for x in authorized_gaps]),
                  "positive_authorized_gaps_over_180ms": sum(x>.18 for x in authorized_gaps),
                  "positive_authorized_gaps_over_180ms_per_10sec": 10*sum(x>.18 for x in authorized_gaps)/(end-start) if end>start else None,
                  "positive_authorized_gaps_180_to_250ms": sum(.18 < x <= .25 for x in authorized_gaps),
                  "positive_authorized_gaps_over_250ms": sum(x > .25 for x in authorized_gaps),
                  "positive_authorized_gaps_over_250ms_per_10sec": 10*sum(x>.25 for x in authorized_gaps)/(end-start) if end>start else None,
                  "effective_hz": (len(ordered) - 1) / (ordered[-1] - ordered[0]) if len(ordered) > 1 else None,
                  "sample_gap_ms": distribution([v * 1000 for v in sample_gaps]),
                  "gaps_over_180ms": sum(v > .18 for v in sample_gaps),
                  "gaps_180_to_250ms": sum(.18 < v <= .25 for v in sample_gaps),
                  "gaps_over_250ms": sum(v > .25 for v in sample_gaps),
                  "continuation_180_to_250": {
                      "check_records": dict(depth_continuation_records),
                      "check_reasons": dict(depth_continuation_reasons),
                      "allowed_distinct_grants": sum(v[2] == "True" for v in depth_continuation_origins),
                      "rejected_distinct_grants": sum(v[2] == "False" for v in depth_continuation_origins),
                      "late_sample_preserved_distinct_grants": len(depth_late_origins),
                      "interpretation": "Eligibility checks, not measured motor runtime or proof of prevented stops",
                  },
                  "cache_retained_records_by_reason": dict(depth_cache_records),
                  "cache_retained_distinct_anchors": len(depth_cache_origins),
                  "approved_remaining_ms": distribution(remaining)},
        "control": {"pid_records": len(pid), "no_matching_base_records": sum(v <= 0 for v in base),
                    "matching_fresh_records": matching_fresh,
                    "matching_window_audited_records": matching_window_audited,
                    "matching_window_percent": ratio(matching_window_records, matching_window_audited),
                    "matching_window_ms": distribution(matching_window_ms),
                    "matching_instant_target_speed_m_s": distribution(matching_instant_speed),
                    "decline_policies": dict(decline_policies),
                    "far_closing_recovery_distinct_samples": len(far_closing_recoveries),
                    "replay_retained_distinct_origins": len(replay_retained_origins),
                    "replay_derivative_retained_distinct_origins": len(replay_derivative_retained_origins),
                    "replay_hold_reset_records": replay_hold_resets,
                    "feedback_config": feedback_config,
                    "feedback_audited_distinct_samples": len(feedback_audits),
                    "feedback_compensation_distinct_samples": dict(Counter(feedback_audits.values())),
                    "estimator_status_records": dict(estimator_statuses),
                    "stop_suspect_bounded_records": decline_policies.get("stop_suspect_bounded", 0),
                    "bias_trial_audited_distinct_samples": len(bias_samples),
                    "bias_trial_positive_distinct_samples": sum(v > 0 for v in bias_samples.values()),
                    "bias_trial_max_requested_rpm_distinct_samples": distribution(list(bias_samples.values())),
                    "matching_two_sample_percent": ratio(matching_two_sample, matching_fresh),
                    "matching_warmup_capped_records": matching_warmup_capped,
                    "compensation_chain_reset_reasons": dict(chain_reset_reasons),
                    "bridge_decay_policies": dict(bridge_decay_policies),
                    "compensated_gap_preserved_distinct_origins": len(compensated_gap_samples),
                    "skipped_derivative_reset_distinct_origins": len(derivative_reset_samples),
                    "matching_capped_records": matching_capped,
                    "matching_capped_percent": ratio(matching_capped, matching_fresh),
                    "matching_rate_limited_records": matching_rate_limited,
                    "matching_unbounded_rpm": distribution(matching_unbounded),
                    "matching_caps_rpm": distribution(matching_caps),
                    "pid_short_speed_drops": short_speed_drops(pid_speed_records, end-start),
                    "approved_short_speed_drops": short_speed_drops(approved_speed_records, end-start),
                    "pid_to_approved_loss_rpm_distinct_samples": distribution(list(pid_losses.values())),
                    "stale_vision_live_depth_revocations": stale_live_revokes,
                    "no_matching_base_percent": ratio(sum(v <= 0 for v in base), len(base)),
                    "no_matching_base_interpretation": (
                        "diagnostic_only_by_design" if distance_control_mode == "distance_pi"
                        else "matching_availability"),
                    "far_fresh_below_30rpm_distinct_samples": len(far_fresh_low_rpm),
                    "far_fresh_below_30rpm_motion_status": dict(Counter(far_fresh_low_rpm.values())),
                    "far_closing_measured_ramp_distinct_samples": len(far_closing_resumes),
                    "detached_prior_distinct_samples": len(detached_prior_samples),
                    "detached_prior_positive_recovery_samples": sum(v > 0 for v in detached_prior_samples.values()),
                    "near_no_matching_stop_distinct_samples": len(near_no_matching_stops),
                    "matching_source_records": dict(matching_sources),
                    "no_matching_reason_records": dict(no_matching_reasons),
                    "distance_control_mode": distance_control_mode,
                    "distance_pi": {
                        "config": pi_config,
                        "distinct_samples": len(pi_samples),
                        "statuses": dict(Counter(v["status"] for v in pi_samples.values())),
                        "launch_audited_samples": sum(v["launch_floor_rpm"] is not None for v in pi_samples.values()),
                        "launch_requested_samples": sum((v["launch_floor_rpm"] or 0) > 0 for v in pi_samples.values()),
                        "launch_software_rise_bypassed_samples": sum(v["software_rise_bypassed"] == "True" for v in pi_samples.values()),
                        "launch_demand_limit_reasons": dict(Counter(v["demand_limit_reason"] for v in pi_samples.values()
                                                                    if (v["launch_floor_rpm"] or 0) > 0)),
                        "launch_total_demand_rpm": distribution([v["total_demand_rpm"] for v in pi_samples.values()
                                                                 if (v["launch_floor_rpm"] or 0) > 0 and v["total_demand_rpm"] is not None]),
                        "launch_brake_cap_rpm": distribution([v["envelope_rpm"] for v in pi_samples.values()
                                                              if (v["launch_floor_rpm"] or 0) > 0 and v["envelope_rpm"] is not None]),
                        "launch_feedback_rpm": distribution([v["ego_rpm"] for v in pi_samples.values()
                                                             if (v["launch_floor_rpm"] or 0) > 0 and v["ego_rpm"] is not None]),
                        "launch_feedback_age_ms": distribution([v["feedback_age_ms"] for v in pi_samples.values()
                                                                if (v["launch_floor_rpm"] or 0) > 0 and v["feedback_age_ms"] is not None]),
                        "brake_sources": dict(Counter(v["brake_source"] for v in pi_samples.values())),
                        "motion_memory_samples": sum(v["brake_source"] == "relative_motion_memory" for v in pi_samples.values()),
                        "motion_memory_distinct_origins": len({(key[0], v["motion_origin_ts"])
                            for key, v in pi_samples.items() if v["brake_source"] == "relative_motion_memory"
                            and v["motion_origin_ts"] is not None}),
                        "motion_memory_uncertainty_m_s": distribution([v["motion_uncertainty_m_s"]
                            for v in pi_samples.values() if v["brake_source"] == "relative_motion_memory"
                            and v["motion_uncertainty_m_s"] is not None]),
                        "closure_sources": dict(Counter(v["closure_source"] for v in pi_samples.values())),
                        "integral_m_s": distribution([v["integral_m_s"] for v in pi_samples.values()
                                                      if v["integral_m_s"] is not None]),
                        "integral_rpm": distribution([v["i_rpm"] for v in pi_samples.values()
                                                     if v["i_rpm"] is not None]),
                        "sample_dt_sec": distribution([v["sample_dt_sec"] for v in pi_samples.values()
                                                      if v["sample_dt_sec"] is not None]),
                        "integral_frozen_percent": ratio(sum(pi_frozen), len(pi_frozen)),
                        "pause_records": sum(pi_pause_reasons.values()),
                        "pause_reasons": dict(pi_pause_reasons),
                        "pause_memory_retained_records": pi_pause_memory["True"],
                        "pause_memory_cleared_records": pi_pause_memory["False"],
                        "pause_distinct_origins": len(pi_pause_origins),
                        "authority_suspend_records_by_reason": dict(pi_suspend_reasons),
                        "authority_suspend_distinct_origin_reasons": len(pi_suspend_origins),
                        "admission_rejected_records": sum(pi_reject_reasons.values()),
                        "admission_rejected_distinct_samples": len(pi_rejected_samples),
                        "admission_reject_reasons": dict(pi_reject_reasons),
                        "approval_limit_records": pi_limit_records,
                        "approval_limited_distinct_samples": len(pi_limits),
                        "approval_limit_classes": dict(pi_limit_classes),
                        "post_limit_integral_m_s": distribution([v["after_i"] for v in pi_limits.values()]),
                        "approval_integral_reduction_m_s": distribution(pi_integral_reductions),
                        "approval_integral_reduced_distinct_samples": sum(v > 1e-9 for v in pi_integral_reductions),
                        "positive_final_loss_rpm": distribution(pi_losses),
                        "quantization_step_rpm": quantization_step,
                        "quantization_compatible_limited_samples": (
                            sum(loss <= quantization_step + 1e-9 for loss in pi_losses)
                            if quantization_step is not None else None),
                        "material_limited_samples": (
                            sum(loss > quantization_step + 1e-9 for loss in pi_losses)
                            if quantization_step is not None else None),
                    },
                    "no_matching_closing_sources": dict(no_matching_closing_sources),
                    "stale_encoder_bound_rpm": distribution(list(fallback_bounds.values())),
                    "normal_live_continuation_samples": len(normal_live_continuations),
                    "shared_motion_window": {
                        "distinct_samples": len(shared_motion_samples),
                        "statuses": dict(Counter(v['status'] for v in shared_motion_samples.values())),
                        "reset_reasons": dict(Counter(v['reset_reason'] for v in shared_motion_samples.values()
                                                     if v['reset_reason'] not in (None, 'None'))),
                        "both_rates_valid_samples": sum(v['range_rate'] is not None and v['target_speed'] is not None
                                                       for v in shared_motion_samples.values()),
                        "span_ms": distribution([v['span_ms'] for v in shared_motion_samples.values()
                                                 if v['span_ms'] is not None]),
                    },
                    "rotation_compensated_records": rotation_compensated_records,
                    "measured_recovery_records": measured_recovery_records,
                    "distance_parking_events": dict(parking_events),
                    "distance_parking_release_ms": distribution(parking_release_ms),
                    "pid_output_rpm": distribution(pid), "motion_reset_reasons": dict(resets),
                    "approach_profile": {
                        "distinct_samples": len(approach_samples),
                        "modes": dict(Counter(v['mode'] for v in approach_samples.values())),
                        "closing_sources": dict(Counter(v['closing_source'] for v in approach_samples.values())),
                        "closing_window_ms": distribution([v['closing_window_ms'] for v in approach_samples.values()
                                                           if v['closing_window_ms'] is not None]),
                        "correction_rpm": distribution([v['correction_rpm'] for v in approach_samples.values()
                                                        if v['correction_rpm'] is not None]),
                        "braking_distance_m": distribution([v['braking_distance_m'] for v in approach_samples.values()
                                                              if v['braking_distance_m'] is not None]),
                    },
                    "integral_rpm": distribution(integral_rpm),
                    "integral_cap_rpm": distribution(integral_caps),
                    "integral_at_cap_percent": ratio(integral_at_cap, len(integral_caps)),
                    "bridge_reasons": dict(bridge_reasons), "recovery_starts": recovery_starts,
                    "recovery_limited_records": recovery_limited,
                    "skipped_observation_reasons": dict(skipped_observations),
                    "skipped_observation_derivative_resets": skipped_derivative_resets,
                    "motion_reset_depth_details": dict(reset_details),
                    "recovery_continuity_restored_samples": len(restored_samples),
                    "scheduling_recovery_distinct_samples": len(scheduling_recovery_samples),
                    "scheduling_recovery_starts": sum(v == "start" for v in scheduling_recovery_samples.values()),
                    "scheduling_recovery_rejections": dict(scheduling_recovery_rejections),
                    "scheduling_recovery_paused_distinct_samples": len(scheduling_recovery_pauses),
                    "scheduling_recovery_discard_reasons": dict(scheduling_recovery_discards),
                    "scheduling_recovery_completed_distinct_samples": len(scheduling_recovery_completions),
                    "scheduling_recovery_above_initial_approval_samples": len(scheduling_recovery_above_initial),
                    "all_recovery_starts_per_10sec": 10 * (recovery_starts + sum(
                        v == "start" for v in scheduling_recovery_samples.values())) / (end-start) if end > start else None,
                    "recovery_lost_rpm_distinct_samples": distribution(list(recovery_losses.values())),
                    "continuity_events": dict(continuity),
                    "motion_resets_per_10sec": 10 * sum(resets.values()) / (end-start) if end > start else None,
                    "recovery_starts_per_10sec": 10 * recovery_starts / (end-start) if end > start else None,
                    "recovery_gap_actions": dict(gaps), "authority_revoke_reasons": dict(revokes),
                    "approved_forward_percent": distribution(approved),
                    "approved_forward_rpm": distribution(approved_rpm)},
        "execution": {"command_covered_sec": covered, "command_dwell_sec": dict(dwell),
                      "zero_command_sec_by_reason": dict(zero_reason_dwell),
                      "cross_wait_zero_percent": ratio(zero_reason_dwell['cross_wait_zero']+
                                                       zero_reason_dwell['cross_timeout_zero'], covered),
                      "wheel_dispatch_reasons": dict(wheel_dispatch_reasons),
                      "positive_requested_but_zero_dispatch_records": wheel_positive_to_zero,
                      "requested_to_dispatched_loss_rpm": distribution(wheel_dispatch_losses),
                      "direct_stop_records": direct_stop_records,
                      "follow_tick_reasons": dict(follow_ticks),
                      "follow_tick_interval_ms": distribution(follow_tick_intervals),
                      "periodic_tick_interval_ms": distribution(periodic_tick_intervals),
                      "zero_command_percent": ratio(dwell["zero"], covered),
                      "time_weighted_command_base_rpm": rpm_integral / covered if covered else None,
                      "encoder_base_rpm_sparse_samples": distribution(feedback)},
        "limitations": ["Distances are sensor estimates, not tape-measured ground truth",
                        "Zero-command dwell is not physical standstill time",
                        "Encoder mean is sparse sample arithmetic mean, not integrated odometry",
                        "PID counts are update counts, not elapsed-time fractions",
                        "Recovery lost RPM is per distinct limited sample, not lost travel distance",
                        "Different walks/scenes are not a controlled before/after experiment",
                        "PI integral distributions describe logged requests before final approval feedback",
                        "Small PI approval losses are quantization-compatible, not proof of their cause"] + (
                            ["distance_pi intentionally has no matching-speed base; its absence is diagnostic, not degraded control"]
                            if distance_control_mode == "distance_pi" else []),
        "warnings": warnings,
    }


def comparison(current, previous):
    """Positive/negative deltas are raw changes, not an automatic quality grade."""
    paths = (
        ("distance", "drift_m_s"), ("distance", "missing_percent"),
        ("distance", "within_tolerance_percent"),
        ("distance", "absolute_error_m", "mean"), ("distance", "absolute_error_m", "p95"),
        ("distance", "actual_distance_m", "max"),
        ("depth", "effective_hz"), ("depth", "gaps_over_180ms"),
        ("depth", "positive_authorized_hz"),
        ("depth", "positive_authorized_gaps_over_180ms_per_10sec"),
        ("depth", "positive_authorized_gaps_over_250ms_per_10sec"),
        ("depth", "positive_authorized_gap_ms", "max"),
        ("control", "no_matching_base_percent"), ("control", "recovery_starts"),
        ("control", "motion_resets_per_10sec"), ("control", "recovery_starts_per_10sec"),
        ("control", "matching_capped_percent"),
        ("control", "matching_two_sample_percent"),
        ("control", "skipped_derivative_reset_distinct_origins"),
        ("control", "pid_short_speed_drops", "per_10sec"),
        ("control", "approved_short_speed_drops", "per_10sec"),
        ("control", "pid_to_approved_loss_rpm_distinct_samples", "mean"),
        ("execution", "zero_command_percent"), ("execution", "time_weighted_command_base_rpm"),
        ("execution", "cross_wait_zero_percent"),
        ("execution", "requested_to_dispatched_loss_rpm", "mean"),
    )
    result = {}
    for path in paths:
        new, old = current, previous
        for key in path:
            new, old = new[key], old[key]
        result[".".join(path)] = {"previous": old, "current": new,
                                 "delta": new - old if new is not None and old is not None else None}
        if (path == ("control", "no_matching_base_percent")
                and "distance_pi" in {current["control"].get("distance_control_mode"),
                                      previous["control"].get("distance_control_mode")}):
            result[".".join(path)].update(delta=None, diagnostic_only=True,
                interpretation="distance_pi intentionally disables matching-speed output")
    return {"previous_run": previous["run"], "previous_window": previous["window"],
            "metrics": result, "caution": "Compare matching target distance, walk speed, scene and duration; counts are not rates"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--cap-start", type=int)
    parser.add_argument("--cap-end", type=int)
    parser.add_argument("--target-distance", type=float)
    parser.add_argument("--tolerance", type=float, default=.15)
    parser.add_argument("--tail-sec", type=float, default=.18)
    parser.add_argument("--compare-run", type=Path)
    parser.add_argument("--compare-cap-start", type=int)
    parser.add_argument("--compare-cap-end", type=int)
    args = parser.parse_args()
    if args.tail_sec < 0 or args.tolerance <= 0 or (args.target_distance is not None and args.target_distance <= 0):
        parser.error("Invalid duration, tolerance or distance")
    try:
        result = analyze(args.run, args.cap_start, args.cap_end, args.target_distance, args.tolerance, args.tail_sec)
        if args.compare_run:
            previous = analyze(args.compare_run, args.compare_cap_start, args.compare_cap_end,
                               args.target_distance, args.tolerance, args.tail_sec)
            result["comparison"] = comparison(result, previous)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
