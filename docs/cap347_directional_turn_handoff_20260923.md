# CAP347–354: remove redundant all-wheel standstill before turn handoff

Scope: latest run `run_20260922_202411_30124_49ca3e02`, CAP335–376.
No hardware operation; no change to PID, peak differential, Depth leases,
parking current, or emergency stops.

## Fault and decision ownership

At 20:24:31.701, Depth continuation failed `braking_margin` at age 182.8ms.
The valid positive base was removed. That safety decision remains unchanged.
At .709 a yaw-only -10/+10 request became 0/0 in ForwardLossHandoff:

- Fresh feedback 3/9 RPM was 18ms before the start of the handoff, so the
  separate post-handoff sample condition rejected it entirely.
- Subsequent 3/11 and 4/11 samples could not leave that state, since its
  residual check required BOTH wheels to be at most 4 RPM in magnitude.
- Once that condition finally passed, WheelZeroCrossGuard started another
  confirmation. The first nonzero command was at 32.103: 393.880ms later.

The right wheel already moved in the desired direction. Waiting for it to
stop destroyed useful turn motion without improving left-wheel reversal
qualification.

## Change

Share a directional residual predicate between handoff and wheel guard:

- Opposing residual is still capped by the configured 4 RPM maximum, further
  capped by requested wheel magnitude. This tolerance was already enabled.
- An aligned wheel need not be stationary; it may move up to its current
  requested magnitude plus that small tolerance. Larger overspeed is excluded.
- Whole-car backward motion outside the original small residual envelope is
  not newly admitted.
- A fresh pre-handoff sample may transfer decision ownership to the wheel
  guard, but is NOT a post-zero confirmation or a motion authorization.
- The guard requires two distinct, consecutive fresh samples after an actual
  zero write for bounded-residual reversal release. The first-zero timestamp
  remains stable across repeated zero writes.
- Once delegated, the old handoff is disarmed, so it cannot restart an
  all-wheel wait while the guard is already evaluating this turn.

No forward arc or positive mean speed is manufactured after Depth loss.
Current target/yaw authority and final-write checks remain in force. Explicit
zero remains zero. There is no forced 500ms motion interval, and stronger
opposing feedback still interrupts motion.

## Regression and limitations

`tests/motor/test_cap347_turn_handoff.py` covers mirrored turns, duplicate and
pre-zero samples, stale/future/invalid feedback, direction changes, strong
reversal, excessive outer-wheel speed, yaw expiry, and continuous qualified
turning beyond 500ms with new evidence.

The recorded unchanged prefix now permits -9/+9 at 20:24:31.800 after two
post-zero samples: 91.413ms instead of waiting until 32.103. This is a software
replay result, NOT a prediction of physical settling or later feedback. The
recorded feedback after a changed command cannot be treated as the resulting
real-world response; stronger opposing motion would still be guarded.

Live validation must measure first useful wheel differential, unintended
zero duration, and overshoot together. Peak-limit consistency (14 vs 30 RPM)
is a separate issue and is not changed in this focused patch.

Validation: focused handoff/continuity suite 83 passed before adding the
recorded-prefix regression; final new test file 17 passed. Full suite run:
4098 passed, 3 failed (the existing 1.5m-vs-1.4m configuration assertion and
two search-direction assertions). The final recorded-prefix test was run
separately after full-suite collection. `git diff --check` passed.
