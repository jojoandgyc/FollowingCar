# CAP339–842: capture-time steering / first-pass settling

## Scope

Normal visible forward and parked recentering only. Direction still follows
image position. Search, ReID thresholds, depth filtering/authorization, distance
PI, wheel limits and NORMAL parking release conditions are unchanged.

`image_capture_motion=true` is enabled in the runtime INI. Set it to `false`
to roll back this change while retaining the previous encoder brake assist.

## What changes

- Use the UID-associated original detector geometry already carried by
  `DepthTargetObservation`. Do not modify this object or the tracking/identity
  box. Require matching capture ID/timestamp, confidence, overlap and size.
- After two locally continuous observations, correct tracking lag by at most
  0.06 image width. A correction crossing the aim line stops at the aim line;
  it cannot by itself command a reverse turn.
- Three distinct physical captures supply two consistent position slopes.
  Derive relative image speed from capture times, not processing/publication
  times. UID/raw-track changes, gaps, jumps and bad association reset evidence.
  Repeated/old frames do not accumulate evidence or renew timestamps.
- For a target approaching the center, predict relative image travel over
  observation age + `image_motion_response_sec`. Initial response horizon is
  0.18 seconds (bounded to 0.3), NOT a calibrated physical stop time.
  Compare against room before the center band/margin to taper same-side RPM.
  Combine with existing qualified encoder braking by taking the stricter
  taper, not by adding the two motions. Neither estimate can reverse yaw or
  grant forward motion. Unqualified image velocity falls back to the previous
  position/encoder behavior, without waiting for more images before steering.
- Carry the same corrected position/rate into the fast loop with exact
  UID/capture provenance. Do not project x again (would count delay twice).
  Intent expiry, predictive-zero latch and actual-post-stop image/quiet-wheel
  release requirements remain intact. Reverse motion is not newly opted in.

## Diagnostics and real response calibration

`steering_capture_observation`: UID, CAP, capture time/age, tracking center,
detector center, control center, relative image rate, time span, rejection reason.
Existing PID logs expose predicted stopping angle/latency and the final RPM.
Startup records the enabled flag and response horizon.

`near_yaw_stop_response`: actual STOP completion -> first qualified quiet
wheel feedback interval, once per park request. This is **encoder quiet**, not
proof of a settled camera. No serial read or sensor thread is added.
Combine with existing `near_yaw_park_applied`, `near_yaw_park_release_check`,
capture-stamped detector boxes, video and IMU records for residual image motion.
Do not equate a tilted IMU axis or its vector norm with planar yaw, or learn a
body brake constant from a moving person's image motion alone.

Next calibration: stationary person, several left/right starts, fixed scene
features to distinguish chassis rotation from subject movement. Compare actual
stop write, wheel quiet, and final visual position; only then adjust the response
horizon. This implementation does not claim to calibrate hardware offline.

## Acceptance on next run

Keep the current distance setting (1.4m), RPM limits and test route unchanged.
Label periods when the person is stationary versus moving. Segment each turn
from first correction to stable centering; use capture time for images and
write-completion time for motor actions.

1. First-pass success: no opposite correction needed before settling; trial
   target >=80% of comparable stationary-target turns (not yet achieved).
2. Maximum first crossing: report normalized-center overshoot and pixels;
   trial target <=0.08 width beyond x=0.5. Previous CAP443–469 reached roughly
   x=0.247 after right correction, an excursion of about0.253.
3. First center entry to stable hold: within x=0.45–0.55 for >=0.5s with fresh
   quiet feedback; aim <=1s. Failed episodes must remain in the denominator.
4. Report STOP-write -> quiet P50/P95 and visual residual motion separately.
5. Count pre-stop-image releases (must remain zero), opposite turn count,
   early braking while still far from center, visual-rate qualification ratio,
   and processing latency. Do not improve settling by indefinitely holding
   off-center, nor interpret normal centered parking as execution failure.

Synthetic tests establish branch/provenance correctness, not real closed-loop
trajectories. No hardware operation is part of this patch.

Validation: 60 new tests pass. `python3 -m pytest -q tests --tb=short`:
3653 passed, 3 pre-existing failures (two search-direction expectations and a
distance-setting assertion expecting 1.5m while the runtime INI is 1.4m).
Those expectations and the current distance setting were not changed here.
