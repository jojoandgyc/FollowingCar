# CAP660–697: reduce avoidable zero-output handoffs

Baseline: `run_20260921_232120_74135_363a4e6a`, capture interval CAP660–697,
1.901s. Use actual completed motor writes aligned to capture timestamps, not
the decision produced later from the same CAP. Normalize both wheels forward
positive; left differential is right minus left.

Baseline command time: zero pair 0.877s (46.1%), equal forward 0.559s (29.4%),
left differential 0.465s (24.5%). Some early equal-wheel time is legitimate
predictive braking. Full 30 RPM differential lasted only 24ms and 38ms in two
handoffs. Visual geometry age averaged 143ms, maximum 218ms.

## Scope

- Correct configured HFOV to user-supplied 66 degrees (existing linear mapping:
  center to edge 33 degrees). This is not a new camera intrinsics calibration.
- For fresh capture-time visual motion and qualified *slow* yaw (<=10deg/s),
  keep at most 2 RPM same-side correction if a 0.5s lookahead still puts the
  target outside the center band plus 0.5 degrees. Center crossing, rapid
  approach, missing visual rate, old evidence, explicit zero policy remain
  brake-authoritative. Config `image_slow_brake_continuity_sec=0.50`.
- Live authorized forward motion with still-forward wheel feedback: clip yaw
  to the authorized base before creating an unnecessary reversing wheel.
  CAP676 base8/yaw-15 changes (-7,23) to (0,16), not a larger base. This is a
  smaller but executable wheel differential, not extra forward authority.
- Forward-loss handoff can delegate <=4 RPM residual motion to the wheel
  guard. Two distinct new feedback samples after zero still qualify the
  mixed-sign request. After handoff, a fixed 0.5s response interval tolerates
  bounded residual opposing speed without restarting the same wait. It uses
  the CURRENT request, never replays a previous command, and never renews
  depth/visual leases. Larger opposing motion, stale feedback, new direction,
  explicit zero and state/UID change still interrupt it.
- Only actually written forward motion resets forward-loss waiting progress;
  blocked forward requests no longer do so.
- Ordinary visible lateral queue interrupts go to the canonical wheel writer
  instead of inserting NORMAL parking. Explicit/safety/unknown stops and
  explicit near-center parking requests retain their previous protection.
  This change does NOT globally disable parking current or all NORMAL stops.

No changes to ReID thresholds, depth filters, depth TTL, search speed, total
RPM ceiling or maximum 30 RPM normal wheel differential. No unconditional
minimum-on timer. A fresh safe target stream may produce >0.5s continuous
motion; losing its authority must still stop it immediately.

## Next-run verification (experimental targets, not measured results)

- Near-identical outward-motion segment: zero-command time fraction aim <20%
  vs46.1%; separately label intentional center/safety stops.
- Count `forward_curve_no_reverse`: requests which used to demand a reversing
  inner wheel should now produce nonzero non-reversing arcs when allowed.
- `cross_bounded_residual_turn_handoff` and `_continue`: no repeated zero for
  the same bounded residual episode during the 0.5s response interval.
- `visible_turn_interrupt_handoff`: ordinary lateral transitions must not
  produce `stop_signal` NORMAL parking. Safety stops must still interrupt.
- Count avoidable <=50ms turn bursts (baseline two 30RPM bursts 24/38ms), and
  measure continuous commanded-turn intervals >=0.5s only while authorized.
- Measure image-to-first-write and first-write-to-measured-differential
  separately. Do not interpret cached RPM as response to a just-written pair.
- Check first crossing overshoot, direction switches, nearest distance and
  stop response; do not trade more overshoot for fewer zero commands.

Tests: `tests/control/test_cap660_turn_continuity.py`,
`tests/motor/test_cap676_turn_continuity.py`; fake clocks and fake motor drivers.
Historical logs can test branching but cannot establish closed-loop results.

Validation completed: 33 new regression cases pass. Full `pytest -q tests`
result: 3887 passed, 3 failed. The remaining failures are the two pre-existing
search-direction expectations and the existing configuration check expecting
1.5m while the runtime target is 1.4m. No hardware run or motor command issued.
