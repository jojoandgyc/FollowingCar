# CAP1348: keep recovery evidence across search entry

## Scope

Fix the quarantined mapped-identity recovery path, without changing ReID
thresholds, search direction policy, depth freshness, wheel commands or PID.
The original protected identity anchor is not replaced by a pending candidate.

The observed sequence was CAP1341/1344 recent-torso conflict, CAP1346 strict
recovery 1/2, then CAP1348 stale protected CAP1300 reference after search entry.
The last failure discarded the already established short candidate trajectory.

## Behaviour

- First recovery evidence must still pass strict full appearance, quality,
  competition and the existing geometry check.
- Keep its raw-track-scoped local geometry during quarantine. Only when the
  selected protected reference is `stale_reference` (not an explicit conflict),
  compare the next new capture against this local observation, within the
  existing 0.35-second maximum interval.
- The protected contradiction/exclusion checks still run first. Two locally
  continuous frames do not clear a recorded identity contradiction.
- Reuse the successful per-frame geometry result in mapped-track verification;
  do not independently reject it again against the stale reference.
- Continue checking strict appearance and local geometry on subsequent fresh
  search observations during quarantine. A mismatch resets local proof;
  repeated/old captures do not confirm or renew the trajectory.
- No gallery update is authorized by this recovery alone. Existing template
  quarantine and downstream motion freshness checks remain independent.

## Crop comparability limitation

The torso feature still comes from a fixed-fraction detector crop, not shoulder
and hip landmarks. Added `winner_crop_shape_consistent` to partial-template
evidence for diagnosing >2x aspect-ratio changes. It is diagnostic only: using
this heuristic as a hard gate failed existing valid clipped-person regressions.
It neither cancels a torso conflict nor grants identity permission. Reliable
anatomical comparability remains future work, not claimed solved here.

## New diagnostic fields

- `reacquire_control_geometry_source`: identity_anchor / local_recovery.
- `reacquire_control_protected_reference_cap`: retained old reference.
- `reacquire_control_local_reference_cap`: actual recovery comparison frame.
- Existing `reacquire_control_recovery_streak` and `reacquire_control_recovered`.

## Verification

`tests/vision/test_cap1348_recovery_continuity.py` exercises the full assignment
path with logged box/distance patterns and synthetic embeddings/timestamps.
It covers search transition, subsequent-frame continuity, duplicate/late data,
raw-track changes, competition, new torso conflict, geometry contradiction,
gallery quarantine and unchanged protected anchors. Existing CAP714/CAP874/
CAP1161 regressions are required alongside this positive recovery case.

Next recording: verify a qualified 1/2 observation survives search entry and
the next qualified frame confirms, while genuine opposite-person contradictions
remain rejected. Compare correct reacquisition success/latency AND false
reacquisition count. Offline policy tests do not measure real model accuracy or
authorize operating the car.

Results: 13 new regressions passed; vision suite 517 passed. The complete
`tests/` run reported 3849 passed and 3 existing failures: two search-direction
expectations, plus a longitudinal configuration assertion expecting 1.5 m while
the current workspace configuration is 1.4 m. Those unrelated settings/tests
were not changed. Run pytest against `tests/`, not the repository root (which
also contains duplicate backup tests and hardware utility scripts).
