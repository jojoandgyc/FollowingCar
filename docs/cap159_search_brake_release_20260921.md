# CAP112–255: search braking and first-pass overshoot

## Observed failure

Run `run_20260921_210248_60638_fc340c78`. User confirms the person stopped
at CAP147. CAP156 requested predictive search braking; NORMAL completed at
21:02:59.299. At .460 the old search-settle gate accepted -2/0 RPM, then CAP159
(detector x=.548) generated a NEW left command using CAP156's pre-stop geometry.
It was actually sent at .485. Retiring old queued commands alone was insufficient.
Subsequent right/left corrections also overshot, independently of this restart.

## Changes and boundaries

* Search NORMAL now uses `ParkSettlingEvidence`, as center parking does:
  fresh post-write feedback, both forward-normalized wheels <=1 RPM, at least
  two distinct samples spanning 40ms, then a fresh capture acquired after
  quiet confirmation. Timer expiry alone is never a movement permission.
* Quiet evidence is sampled from the existing cached encoder feedback on the
  existing motor tick, not through new serial reads or extra vision waits.
  Polling without an image cannot release the brake. Safety holds supersede it.
* The current visual result is recorded before releasing. On release, retire
  pre-STOP direction history, cached search side and lateral outputs. Delayed
  inference and historical backfill cannot reinsert pre-STOP evidence.
  Post-STOP identity-verified history remains available. If no reliable side
  remains, the existing direction-unresolved policy holds rather than inventing
  or replaying a direction. Fresh qualified evidence can resume control.
* A unique, same-track, locally continuous unconfirmed candidate around the
  center can veto release for the existing bounded observation interval
  (normally 300ms from quiet confirmation). It cannot assign identity,
  update templates, change direction or grant forward motion. The interval
  does not rearm on each image. Confirmed targets do not acquire this extra wait.
* For braking, two consistent visual increments now yield the newest interval's
  rate instead of their average. Fresh raw and filtered encoder yaw must still
  agree; use the fresh raw interval instead of the smaller lagging value.
  These rates only reduce same-side steering. No RPM increase, opposite drive,
  PI tuning, depth TTL change or removal of real wheel-reversal protection.

This does not calibrate the physical stop model. The previous 0.9s-to-quiet
observations include residual movement and quiet confirmation, and are NOT
installed as a blind 0.9s prediction/wait parameter. Image age and the existing
actuation-response margin remain explicit in the brake model.

## Logs and next-run validation

New `search_brake_release_check`: image timestamp, qualification/rejection
reason, fresh-image flag, candidate veto, quiet timestamp. `search_brake_stop_response`
records actual write-to-first-qualified-quiet interval. The release log also
includes STOP, quiet and image timestamps. `search_brake_direction_retired`
reports the latest surviving post-STOP capture, or none.

Keep RPM limits and distance setting unchanged; label when the person stops.
Use capture clocks for image positions and completion clocks for motor writes.

1. Pre-stop-image releases and pre-stop-history restarts after search braking:
   must both be zero (CAP156->159 is the regression baseline).
2. Verified new off-center targets resume without an added fixed observation
   wait; unconfirmed center observation cannot endlessly renew its window.
3. Report first-pass success with no reverse correction, maximum excursion
   past center, and first center entry -> stable hold (x=.45..55 for >=.5s).
   Baseline later excursions: x=.217 and x=.806, about .283/.306 from center.
   Trial targets: excursion <=.08, first-pass success >=80% on comparable
   stationary-target trials. These are NOT established real-world results.
4. Report stop-to-quiet latency separately from camera residual motion and
   time spent direction-unresolved. Do not claim an improvement merely by
   parking indefinitely with an off-center/unconfirmed target.

Validation uses fake motors, physical capture timestamps, production runtime
and tracker adapters. No real motor, camera or launch script is run by this change.

Validation result: 32 new regression cases pass. Full suite:
`python3 -m pytest -q tests --tb=short` — 3718 passed, 3 existing failures:
two search-direction expectations in `test_search_direction_switch.py` and
`test_longitudinal_approach.py` expecting a 1.5m setting while the INI is 1.4m.
Those tests/settings were not altered to hide the failures. Syntax compilation
and `git diff --check` also pass. Real-world first-pass settling remains unverified.
