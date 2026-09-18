# Similar-person reacquisition safety (2026-09-16)

## Baseline and scope

Run `run_20260916_152553_11669_37745d9c`: the user-labelled distractor at
CAP1006/1024/1073/1075/1080/1082/1085 has all seven full-body distances below
0.20; one is below 0.15. These seven recorded assignments all output UID0;
this segment demonstrates risk, not seven actual wrong reacquisitions.
At CAP1024 correct/distractor distances are 0.100519/0.149289 (gap 0.048770).
At CAP1085 they are 0.177677/0.189031 (gap 0.011354).
The strong template winners CAP805 and CAP528 depict the correct target.
Raw track IDs 6 and 7 exchange people; the correct person later uses ID8.

Plan: preserve co-visible negative evidence across uniquely associated local
ID changes, and veto ambiguous same-frame person-to-active-UID matches.
No global appearance threshold, motor speed, longitudinal authorization,
search direction, or minimum confirmation count changes.

## Negative evidence

`IdentityExclusionMemory` now compares the complete current detector geometry
snapshot against the previous snapshot, including IDs still present. A unique
one-to-one strict geometry match transfers negative evidence, not identity.
Existing short transfer limits remain: two control frames, 350ms, center
residual <=0.025 image width after available yaw compensation, size consistency
and IoU >=0.5. This is conservative local association, not a new long-term ReID.
Ambiguous matches do not transfer. A swap-flagged crop may carry an existing
negative label, but cannot create an exclusion or act as a trusted witness.
Broad same-ID continuity is retained only when not contradicted by a different
strict local association or a swap flag. Expiry/reset rules remain unchanged.

Transfer diagnostics are included in existing `search_exclusion` evidence:
`association_reason=unique_geometry_track_transfer`,
`transferred_from_track_id`, `transfer_capture_frame_id`, plus original source
CAP, reference box and current raw box. Repeated reports of one transfer must
be deduplicated by source/transfer CAP and track ID when counting events.

## Same-frame competition

During active search, the tracker computes every current C0 detector feature's
distance to the same active UID **before any per-track assignment/template
update**. It uses existing features; no extra RKNN inference. Suppressed
duplicate boxes do not compete; a tentative person without a formal DeepSORT
track still competes. Missing full-body features in a multi-person frame do
not establish an unambiguous winner. Single-person partial ReID is unchanged.

For each candidate, gap = closest other person's distance - own distance.
`IdentityBankConfig.preferred_search_candidate_min_margin` defaults to 0.05.
Only a candidate with gap >=0.05 passes this extra veto. This is NOT the existing
gallery `second_distance`, which compares different UIDs rather than people.
Both historical pairs above would fail this extra veto during search.
Existing quality/geometry/YOLO competition/confirmation requirements still
apply after a pass; passing alone grants nothing.

Failure returns UID0 with `reason=search_candidate_identity_ambiguous`, clears
that candidate's pending confirmation chains, and does not update templates.
An existing claim to the searched UID is detached on this path. Co-visible
exclusion retains priority. Gallery quarantine and downstream motion checks
remain intact. Stale/cross-UID competition evidence is not applied.

`reid_match_evidence.identity_competition` logs:
frame index, UID, detector source index, candidate count, own distance,
competitor index/distance, signed gap, required margin, passed and reason.
The same evidence is attached to runtime `last_identity_observations` metadata.

## Validation and next-run metrics

Regression coverage: simultaneous 6/7-style swaps, ID replacement, ambiguous
geometry, no false exclusion of the correct person, CAP1024/1085 distance
pairs, missing competitor feature, tentative competitor, detector-only probes,
frame/order isolation, clearing pending proof, no gallery writes on veto,
unchanged single-person/normal tracking and quality rejection after a pass.

On the next labelled run compare:

- wrong UID acceptance count, especially d<=0.15 candidates;
- same-frame distance gap distribution and ambiguous veto count;
- negative-evidence transfer events, lost labels, and false exclusions;
- correct-target reacquisition delay from first qualified detection;
- ambiguous waits with missing features versus genuinely similar distances.

A 0.05 margin can delay the correct person when both people look similar. The
fix intentionally prefers waiting over a weakly distinguished choice; do not
claim improved false-match rate without new labelled hardware footage.
Long occlusions/overlapping people are not solved by local geometry transfer.

Offline verification: 21 added regression cases pass; complete pytest suite
1425 passed, 2 pre-existing search-direction assertions failed
(`test_current_left_candidate_overrides_right_history_on_search_entry` and
`test_edge_target_crossing_aimline_still_turns_toward_bbox_center`). The standalone
DeepSORT tracker script passes. The standalone identity-bank script has a
`last_seen_frame` assertion failure, reproduced after removing this change's
competition gate in memory; it was not repaired as part of this scoped task.
Python compilation checks pass. No hardware execution performed.
