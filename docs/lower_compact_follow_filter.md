# Lower compact follow-crop filter (2026-09-22)

Run: `run_20260922_131724_8284_72fcb157`, detector coordinates 640x480.
The user identified CAP1945, CAP1953 and CAP1958 as wrong targets. Saved
ReID crops show chair/legs at CAP1945 and a seated bystander at 1953/1958.

| CAP | Original detector width x height | Center x | UID admission |
| --- | --- | --- | --- |
| 1945 | 133.83 x 134.76 | .106 | partial late reacquire after CAP1944 |
| 1953 | 166.87 x 194.27 | .251 | soft full-feature reacquire after CAP1951 |
| 1958 | 160.19 x 190.70 | .356 | retained track 13, template quarantine |

1945 full-gallery distance .302, recent partial .283 and aggregate partial
.264 admitted the local two-frame chain. Its area was only .100 of the
protected CAP1835 anchor, but that anchor was 5.72s old and local continuity
was used. 1953 matched the original CAP19 template at .176; its recent full
distance was .290, within the soft observation route. 1958 matched CAP19 at
.171. None of these three observations updated the identity gallery.
Local continuity does not establish identity; this patch does not claim to
repair the general late-reacquire policy or model confusion.

## Scoped deployment rule

Reject a raw detection from follow/identity/search-observation eligibility
only when ALL conditions hold:

- top >= 40% of image height;
- box height <= 42% of image height;
- bottom <= 92% of image height;
- width / height >= .80.

No left/right exclusion and no global increase of minimum height. Ratios
scale with resolution. Tall close crops, narrow standing people and
bottom-clipped crops are not rejected by this added rule. This is a policy
for the requested upright following scene, not proof that such a box can
never be the true person. A real target sitting or crouching into this same
geometry will also be excluded; supporting that use case needs a separate
identity-preserving policy.

`rk_vision/follow_bbox_policy.py` centralizes the rule. Pipeline applies it
to formal detections and probes before ReID/association/observation, then
again after probe merging. Tracker quality and detector-only identity
fallback use the same rule; an expanded Kalman box cannot rescue the raw
crop. Raw detector diagnostics remain visible. Rejection reason:
`lower_compact_bbox` in `follow_bbox_size_rejected`.

Offline application to saved identity observations rejects 34 records in
this run, including all three requested CAPs. This is not an accuracy score:
other records have not all been manually labelled. Tests cover the three
real boxes, doubled resolution, ordinary standing/close/edge controls,
formal/probe filtering, expanded-track bypass, and repeated full-pipeline
inputs. No live motor operation or global ReID threshold changes.
