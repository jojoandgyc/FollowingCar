# CAP437–463: forward handoff and braking evidence

## Scope

Run: `run_20260921_134517_12929_aaba7c57`. Capture window437–463 is
1.437s; last-dispatched zero-wheel time is0.495s (34.43%). This is a
command-duration metric, not measured standstill. Current setpoint is1.4m.

Two separate ceilings were observed:

* CAP437/440: total launch demand180RPM became16/26RPM under
  `motion_unknown_bound`. Accepted distance existed, but closure evidence was
  rejected by the independent15deg/s yaw gate.
* At13:45:44.245, approved98RPM became9RPM at the writer. A whole-car
  reverse observation had triggered zero-cross confirmation, followed by
  a second software launch from5RPM at80RPM/s.

## Changes

1. With the existing `follow_forward_handoff_enable` enabled and a fresh
   forward grant, TWO new aligned feedback samples now release the CURRENT
   approved forward wheel pair with `cross_confirmed_forward_handoff`.
   No5RPM restart/80RPM/s second ramp. The motor's own ramp is untouched.
   Opposing wheels still wait. Duplicate/stale/untrusted feedback cannot
   confirm. Mixed-sign rotation, backward requests and the opt-out retain
   their old behavior. Actual-write depth/identity/safety checks remain.
2. Only distance-PI closure accepts up to35deg/s when detector geometry is
   still fresh and the conservative rotation contribution is <=0.25m/s.
   Both raw and filtered yaw magnitudes participate through their maximum;
   their sign disagreement is not silently treated as zero rotation.
   Projected bearing<=45deg and high-yaw geometry age<=180ms remain.
   Two physical raw samples are still required. The positive rotation bound
   is subtracted from the measured range rate, reducing rather than adding
   permitted chase speed. Legacy approach/matching gates are unchanged.
3. Added `execution_base_loss_rpm`, `forward_confirmed_handoff`, closure
   geometry bounds and deduplicated `distance_closure_rejected` reasons.

No changes to P/I,180RPM launch demand,200RPM total budget,1.4m setpoint,
250ms physical-depth authorization, ReID or search logic. Unknown motion
without qualified geometry still uses the conservative braking fallback;
there is no unconditional180RPM command or fabricated human velocity.

The extended closure domain is a commissioning assumption, not a certified
braking bound. First-sample uncertainty and true reversal waiting remain.
This does not diagnose or repair the hardware cause of negative feedback
after zero speed, nor prove real-world tracking convergence.

## Offline verification

33 added cases pass, plus updated forward-handoff expectation. Tests cover
the real fake-driver writer, current reduced caps, expiry before confirmation
and during feedback/write, reverse/rotation retention, legacy behavior,
geometry/UID/age/feedback rejection, and fast approach braking.

Full baseline:3009passed,3failed. After change:3042passed, same3failed:
two existing search-direction cases and a test expecting1.5m where current
user configuration is1.4m. No motors/cameras were operated.

Controlled two-sample calculation with CAP-like raw/filtered ranges and
near-axis geometry: old closure gate16->26RPM, new16->103RPM. This is a
branch comparison with fixed inputs, NOT a replayed physical trajectory or
a guarantee of actual speed. The first sample correctly remains conservative.

## Next run acceptance

Use the same walking path and annotate moving/stationary CAP boundaries.

* Qualified normal-forward aligned release: zero5RPM re-launches; zero
  `cross_resume_ramp` events from this release path.
* After release, `execution_base_loss_rpm` should be0 unless a new explicit
  safety/authorization/reversal restriction intervenes. Record each cause.
* Same-scene command-zero share: below15% vs34.43% in this short baseline;
  separately count wait before confirmation and no-authority zeros.
* Track unknown-motion fallback count, rejection reasons, commanded/actual
  RPM gap, moving mean distance error and maximum distance. Do not compare
  different walking speeds as controlled trials.
* At stationarity: minimum distance, stopping time and unintended reverse
  wheel motion must not worsen. Start supervised with clear space and an
  emergency stop; stop the test if wheel oscillation persists.
