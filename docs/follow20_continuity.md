# Normal-follow wheel continuity — 2026-09-16

Scope: keep the existing Depth authorization, lateral PID, identity/search
gates and wheel reversal guard. Do not add an uninterruptible motor lock,
increase speed limits, or extend physical Depth TTL.

## Changes

1. `[lateral_intent] follow_wheel_period_sec=0.05` enables the action thread's
   sole normal-follow wheel writer. Each tick reads canonical current Depth
   speed and signed lateral correction, not the last queued action name.
   Ordinary DRIVE/STEER/TURN requests no longer emit independent packets while
   this mode owns the wheels. No saved wheel packet is replayed. Reverse,
   search, low-quality control and rotation-only retain their dedicated paths.
2. Base reductions and yaw-zero revocation may preempt the period. Hard-stop
   checks and axis expiry remain serviced by the action loop, including between
   normal ticks. Immediately before writing, recheck axes, UID/revision and
   mode after reading wheel feedback. A changed authority cannot send the old
   pair. The existing wheel-zero-cross guard still requires qualified feedback.
3. No new visual frame does NOT by itself imply zero wheel speed: valid
   canonical axes continue. Expired axes cannot be renewed by ticks. Missing
   Depth zeroes forward speed while valid yaw may remain; zero yaw preserves
   approved forward speed. A period overrun skips missed ticks, never bursts.
4. `[distance_pid] turn_compensation_enable=true` enables bounded camera Z-rate
   compensation using raw detector geometry and right-positive encoder yaw.
   The camera-rotation contribution is `omega * X`; subtract its interval mean
   from measured Z-rate before adding mean measured forward speed. Raw bearing
   is projected from RGB capture time to Depth time using yaw, not processing
   time. Requirements: correct UID/raw YOLO provenance; RGB-to-Depth interval
   -20..180ms, projected bearing within 10 degrees, yaw within 15 deg/s,
   rotation contribution <=0.25m/s, existing fresh feedback/depth/time-skew and
   speed/jump guards. Missing qualification retains the old <=5deg/s path.
   Switching compensated/uncompensated measurement chains requires warm-up.
5. `[distance_pid] measured_recovery_enable=true`: a pure ROI/replay scheduling
   interruption can retain a recovery reference for at most 350ms. This is
   NOT motion authorization: original Depth still expires at 180ms. Only new
   trusted same-UID Depth with consistent distance and fresh forward wheel
   feedback can resume above the old 25RPM stage. Resume cap is bounded by
   both prior approved PID output and measured mean wheel speed. Existing
   requested output, overall limits and recovery stages still apply. Jumps,
   hazards, search/UID changes, invalid feedback and long gaps keep old recovery.

Normal configuration enables these changes. Restore event-driven dispatch
with `follow_wheel_period_sec=0`, and disable the two new boolean settings to
restore legacy velocity/recovery behavior. Model/controller defaults remain
opt-in. The separate rotation-only INI was not changed.

## Limitations

The 50ms cadence is best-effort, not real-time: Python scheduling, driver writes
and other action-thread work can overrun. Safety preemption means zero/reduced
output is not restricted to 20Hz. It is NOT guaranteed that every packet lasts
50ms. Source TTL is never restarted by sending a packet.

Rotation compensation is a bounded near-axis approximation with analytic
offline tests; it is not calibrated world odometry. Encoder sign, wheel scale,
exposure timing, raw depth association and lateral person motion still need
hardware validation. This change does not fix all ROI/visual processing gaps
or guarantee an effective 30Hz target-distance stream.

## Baselines and next-run metrics

Run `run_20260916_160404_13910_e742aae9`:

| Metric | CAP111–264 | CAP608–720 |
| --- | ---: | ---: |
| Capture duration | 8.193s | 6.052s |
| Distance endpoints | 1.794→2.414m | 1.815→3.035m |
| Mean signed error to 1.5m | +0.559m | +0.698m |
| Distance drift | +0.076m/s | +0.202m/s |
| No matching base / PID records | 43/97 (44.3%) | 31/66 (47.0%) |
| Effective accepted Depth | 11.8Hz | 10.5Hz |
| Zero-command dwell | 12.3% | 14.9% |
| Time-weighted commanded base | 36.8RPM | 32.4RPM |
| Yaw-limit resets | 9 | 6 |
| Recovery starts | 1 | 2 |

Use `tools/follow_metrics.py RUN_DIR --cap-start N --cap-end M`. Execution
window includes the existing 180ms tail. Do not infer physical standstill from
zero-command dwell or compare different walks as a controlled A/B test.

New logs/metric fields:

- `follow_wheel_config`: verify new writer is enabled, normal-follow scope.
- `follow_wheel_tick`: UID, revision, base/yaw RPM, period, interval and reason.
  `execution.follow_tick_reasons`, `follow_tick_interval_ms` and
  `periodic_tick_interval_ms` separate ordinary cadence from safety reductions.
  Intervals are measured relative to previous completed send; they are not
  camera latency or hard real-time guarantees.
- `visible_wheel_dispatch`: requested/applied signed wheel pair and zero-cross
  reason; backend `LZ30EMA` command lines remain the authoritative write record.
- `longitudinal_motion`: `raw_range_rate`, `rotation_rate`, `compensated`;
  `control.rotation_compensated_records` counts computations, not time coverage.
- `depth_measured_recovery`: previous approved RPM, actual measured speed,
  recovery cap, gap and current depth age; `control.measured_recovery_records`.

Acceptance requires lower yaw-reset/no-base frequency and zero-command dwell,
smaller distance drift, while stopping/approaching targets remain safe. Verify
no motion survives expired Depth, no stale queued direction returns, and
search/reverse still work. Do not raise the 60% limit during this comparison.

Offline tests use fake driver/clock and analytic rotating-camera trajectories.
No camera/motor run was performed during implementation.

Final regression: 1467 passed, 2 existing search-direction failures remain
(`test_current_left_candidate_overrides_right_history_on_search_entry` and
`test_edge_target_crossing_aimline_still_turns_toward_bbox_center`). Runtime
configuration checks and Python compilation passed. These results do not
establish hardware safety or distance convergence.
