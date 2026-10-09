# CAP1996 search-to-follow overshoot mitigation

Reference: run_20260922_131724_8284_72fcb157, CAP1996..2017.
Correct UID1 was reacquired at CAP1996. Search 7 RPM became a normal-follow
15 RPM yaw correction; NORMAL STOP triggered at CAP2001. Capture-to-stop
write took ~263ms; STOP-to-quiet confirmation took ~997ms. The latter is
encoder-based quiet confirmation, not an independent chassis stopping-time
measurement. No new turn write was logged during that stop/settle interval.

## Changes

- At confirmed search release, carry a ceiling equal to the previous search
  raw RPM, at most 7 RPM (fallback 7 if unavailable), into the existing
  first-center-brake obligation. This caps both directions, never creates a
  command, never raises a smaller correction, and preserves immediate zero.
- Cap the initial lateral intent, its refresh ceiling and the final wheel
  writer. The final check prevents a cached normal-follow result from
  reintroducing 15 RPM. Approved forward base is unchanged by this yaw cap.
- No time-based boost/release: the existing handoff UID lifetime owns the
  restriction. A completed first-center stop retires the obligation; new
  normal tracking is unchanged. Identity loss/UID replacement cannot reuse
  this as permission to turn.
- Search brake prediction includes actual image age (already present) plus
  an additional 50ms pending dispatch allowance. Following a stop, its last
  request-to-write delay sets a bounded 50..150ms allowance for the next
  episode. This does NOT model/calibrate the ~1s physical stop response.
- Install the motor-write veto before clearing old yaw/forward axes, avoiding
  an intermediate forward-only packet during brake-request construction.
- Log capture-to-stop time and post-stop encoder samples at most 10Hz:
  `search_handoff_yaw_armed`, `search_handoff_yaw_limited`,
  `search_brake_response_sample`, and extended
  `search_reacquire_brake_applied`. Existing quiet confirmation stays intact.

No changes to ReID thresholds, longitudinal PID, normal yaw ceiling, normal
parking mode/current, depth TTL, quiet threshold, or safe-stop priorities.
No live motor test performed. Existing dirty workspace edits preserved.

Verification: 15 new regression cases; focused handoff/periodic suites 71
passed. Full suite: 3936 passed, 3 failed. The remaining failures are two
existing search-direction expectations and the old approach config test
expecting 1.5m although the user selected 1.4m. Compilation and diff checks
passed. No test assertion or user distance was changed to hide these failures.

## Next run

Keep the same settings and label first-correct-CAP / center-CAP / stopped-CAP.
Prefer a repeatable search-reacquire with a stationary person in a clear area.

- Hard software check: during handoff, final requested yaw must not exceed
  its captured ceiling; safety/zero still take priority. Normal following
  outside that episode must not inherit the ceiling.
- Record maximum encoder yaw before STOP and target x at STOP; compare with
  the prior ~31.35 deg/s sampled peak and subsequent x=.873 at CAP2017.
- Report capture-to-stop, request-to-stop and stop-to-quiet separately, not
  one combined delay. Compare the ~263ms / ~33ms / ~997ms reference figures.
- Count speed writes after a pending NORMAL stop: must stay zero (until
  explicitly released), including writes of 0 RPM that would change mode.
- Measure center overshoot and time to stable centering. Do not claim a
  mechanical stopping improvement merely because software holds longer.

If stop-to-quiet remains near 1s despite lower approach yaw, independently
measure NORMAL stopping response before tuning the braking model or mode.
