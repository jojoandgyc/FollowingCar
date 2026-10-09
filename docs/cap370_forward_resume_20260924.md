# CAP370–392: ordinary parking resume and geometry-evidence continuity

Scope: longitudinal resume only; no hardware run, no P/RPM/setpoint changes.

- A fresh post-STOP image of the same qualified UID, fresh trustworthy depth
  outside the configured distance deadband, a positive raw range trend and
  a positive forward decision can request early ordinary current release.
  The hint expires at min(image+190ms, depth+180ms). It cannot grant wheel
  authority. The motor worker still clears/reads back both currents and sends
  FREE STOP; a later qualified release and actual-write checks are required.
- The 500ms default remains for unqualified/no-motion parking, search settling
  and recentering. Safety/explicit stops cannot use the forward hint. Qualified
  translation needs an image after the original STOP, not another image after
  the later FREE transaction. Feedback is still fresh; reversal guards remain.
- If a new positive same-UID grant supersedes an old periodic packet during
  current-release I/O, discard that packet and rebuild on the next worker tick.
  Do not reapply parking current solely for this replacement. Real lost
  authority, hazards and identity changes retain stopping behavior.
- For PI only, an otherwise acceptable 250–300ms geometry reference may skip
  a closure update while retaining the old window for at most 180ms from its
  last accepted physical sample. No skipped sample enters the window or
  refreshes its origin. Existing PI motion-memory uncertainty and new raw
  distance braking vetoes apply, output cannot exceed prior approval, and
  depth authorization still has its original deadline. Excess yaw, invalid
  feedback, identity conflicts, large gaps and raw jumps keep hard rejection.

Diagnostics: `near_yaw_forward_resume_requested`, `follow_wheel_retry` with
`parking_exit_new_forward_grant`, and `distance_closure_skipped` with retained
physical sample timestamp. Never interpret these as new motor authorization.

The active configuration currently targets 1.4m (release threshold 1.5m).
This patch follows that setting rather than silently changing it. Launch demand
remains 180RPM and Kp remains 3/s. CAP370's launch demand was already above the
braking envelope, so increasing P alone would not remove this bottleneck.

Validate on a controlled physical test separately: time from first eligible
distance evidence to first actual write; no repark on benign grant replacement;
no 102→30 collapse solely from marginal geometry expiry; stopping overshoot;
separate motor/encoder acceleration response. Offline tests do not prove physical
stopping distance or eliminate the observed motor response lag.

## Integration with the completed CAP837 change

Retain `TurnBuildup`: it can reduce the common forward command for a bounded
350ms episode after independently sampled post-write feedback proves that the
requested differential has not developed. This is an intentional exception to
unconditional forward-base preservation, not an extension of the parked hold.
It must not retain a low base after its deadline, restart on every image, restore
expired forward authority, or reapply parking current after a normal resume.

`tests/motor/test_cap370_cap837_integration.py` exercises early ordinary current
release followed by real writer turn buildup, deadline recovery in both
directions, reduced forward grants, expired depth, identity loss and danger.
The tests use fake hardware only. The earlier five CAP837 failures no longer
occur with the other session's completed changes. Physical assessment must
separately count `buildup_common_accel_limited` and longitudinal braking reasons;
combined code does not promise zero temporary forward reductions.
