# CAP2394 false reacquisition: preserve handoff contradictions

## Recorded failure

In run `20260922_153217_16547_d0bf177d`, CAP2389/2390 raw track16
was rejected against correct track13 at CAP2382. Compensated displacement
was about 0.424/0.423 of image width, area similarity 0.382/0.395, elapsed
time 0.367/0.432 s and yaw change 3.85 degrees. This was not persisted as a
contradiction. Search subsequently converted it to a time-window rejection,
and CAP2392/2394 local continuity granted UID1. CAP2437 released quarantine
and learned the wrong person; CAP2449 matched that new sample at d=0.043.

## Changes and limits

- Ordinary handoff and search now share an additional **bounded** negative
  evidence rule: fresh strong-quality original detector geometry, elapsed
  capture time in (0, 0.75] s, measured yaw change <=10 degrees, compensated
  center jump >max(0.40, 1.25*existing jump limit), and area similarity <0.50.
  These are conservative policy starting points, not measured physical limits.
- This is not a permanent ban on every geometric rejection. Missing yaw,
  low-quality geometry, normal scale-only changes, small displacement and long
  gaps do not establish this new contradiction. Existing cross-edge rules stay.
- Preserve the contradiction across search entry, reference expiry and a
  uniquely continuous raw-track change. Local positive continuity cannot erase
  it. Existing independent trusted-track revalidation/reset remains available.
- Preserve rejection CAP/reason in the held record and output geometry.
  Time-window diagnostics additionally retain `spatial_reason` and
  `reference_time_expired`; a time-window label is not identity proof.
- Binding cannot silently delete an unresolved contradiction. Quarantine
  cannot release for its conflicted track even via a direct observation call;
  the common gallery-update helper also rejects that origin.
- If samples already exist from the contradicted raw track **after the trusted
  reference frame**, isolate only those metadata-attributed samples from full,
  weak, torso and recent galleries. Keep up to 128 vectors/provenance records
  per UID for inspection and a cumulative count. Do not clear unrelated samples,
  guess origins of legacy rows, roll back time watermarks, or renew old samples.
  Logs report `identity_templates_isolated` with UID/track/cutoff/count.

No RPM, ReID threshold, motor authorization or depth-age settings changed.
No log/image files were deleted, no running process restarted and no hardware
test performed. Historical recordings remain evidence; isolation concerns
in-memory matching galleries when a contradiction is established.

## Verification / next recording

The new test module `tests/vision/test_cap2394_handoff_conflict.py` uses recorded
geometry and synthetic vectors at the logged distances. The CAP2389 rejection
must survive CAP2394, continued very-low distances and a raw-track change.
It also checks correct independent reacquisition, non-contradiction cases,
quarantine origin protection and source-specific sample isolation. Existing
CAP1348 positive recovery and CAP714/CAP874 negative regressions remain required.

Next-run criteria:

1. This candidate pattern must produce zero wrong UID confirmations and zero
   wrong-template updates (not just delayed confirmation).
2. Logs must retain the original rejection CAP across the state transition.
3. Correct target recovery success/time must be compared too: rejecting both
   correct and wrong people is not an improvement.
4. This fixes a policy hole; it does not establish that the embedding model can
   distinguish all lookalikes. Model/crop evaluation remains a separate task.

Validation: 13 new tests passed, including direct isolation/update entry points;
the CAP1348 positive-recovery regressions also passed. Full `tests/` run before
the final two test additions: 4019 passed, 3 existing failures (two search
direction cases and the target-distance configuration assertion). No unrelated
control settings or those failing expectations were changed.
