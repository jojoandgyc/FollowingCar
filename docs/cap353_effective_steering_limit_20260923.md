# CAP353: consistent 20 RPM differential and effective pivot limits

User-approved scope: lower peak visible-follow differential and remove the
14/30 RPM producer conflict. Preserve CAP347 directional handoff, parking,
Depth expiry, genuine reversal guards and predictive deceleration.

## Policy

In the deployed image-position control profile, `max_correction_rpm=10`
means at most 20 RPM total wheel differential. The existing near/pivot cap
remains 7 RPM per wheel (14 differential), with any lower explicit limit
still taking precedence. These are upper bounds, never minimum outputs.

`steering_limits.effective_correction_limit` is shared by controller PID
updates, the lateral fast loop and final visible wheel execution. Other
legacy controller profiles and unrelated search behavior are not retuned.

- Main PID updates apply the global ceiling and lower near/zero-base ceiling.
- The fast loop checks live Depth authority before selecting the PID refresh
  path. An old `mode=forward` intent with no current translation uses the
  parked PID with zero base, instead of refreshing the forward PID at 15 RPM.
- First-tick reused results and slew output are also clamped immediately;
  lowering a limit does not wait for a slew ramp to decay from an old peak.
- The wheel writer checks the actual authorized base and current applicable
  intent ceiling after temporary boost. Old producers cannot bypass the cap.
- Current low-quality limits and search-reacquire limits may only reduce yaw.
  No stale frame is renewed and no Depth grant is manufactured by this policy.

`effective_mode` and `effective_limit_rpm` were added to lateral tick logs.
`effective_yaw_limit` records final-writer clamping. Logs still retain the
original intent mode and limit to expose disagreement rather than hide it.

## Tests and live follow-up

Offline tests cover both directions, the first stale forward tick, subsequent
parked refresh, new authorized forward recovery, lower policy limits, stale
15 RPM producer input, temporary boost, cancellation, and the updated INI
20 RPM differential. The CAP347 handoff tests remain in the full regression.

No motor/camera operation was performed. Validate on the next run that:

1. Visible-follow requested/applied differential never exceeds 20 RPM.
2. No-forward pivot follows the 14 RPM ceiling without 14/30 alternation.
3. Genuine depth/mode changes may still change the cap; this is not a fixed
   500ms command hold and safety/center braking still interrupts immediately.
4. Unnecessary zero duration and first-center overshoot improve together.

Final validation: 55 targeted tests passed; full suite 4111 passed, with the
same three pre-existing failures (1.5m assertion vs deployed 1.4m, and two
search-direction expectations). `git diff --check` passed. No hardware run,
commit or push was performed.
