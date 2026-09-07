# Follow-Car Project Handoff

## Hardware and Runtime

- Board: `root@172.16.16.103`
- Project: `/home/topeet/Desktop/rk_car_runtime_module`
- Main launcher: `./run_request_0428_modular.sh`
- Main runtime: `request_0513_modular.py`
- Normal follow config: `car_control_modular/config/reid_runtime.ini`
- Rotation-only config: `car_control_modular/config/reid_runtime_rotation_only.ini`
- Camera: Astra Pro RGB + Depth, normally `/dev/video3` RGB and Depth stream through the Astra module.
- Current normal run enables Astra Depth and disables mmWave at runtime. IMU may be initialized by the normal config, but it must not be used in the steering PID unless explicitly requested.

## Core Control Requirements

1. Keep the selected person's image center near the camera center.
2. Keep the person distance near 1.5 m.
3. Front/side IR safety has highest priority.
4. Hard brake is 0.5 m. Normal distance parking/release logic must not override IR safety.
5. Steering uses camera horizontal error as the outer loop and encoder-derived body yaw rate as the inner feedback loop. Do not add IMU feedback without a new explicit request.
6. Horizontal steering must remain available when Depth is missing. Distance confidence may limit longitudinal speed, but it must not reduce yaw authority to an ineffective 1-6 RPM.

## Distance Control Ranges and State Machine

The distance controller is centered on a nominal target of `1.50 m`. The target is not
treated as an exact single-point stop command; it uses separate enter/exit thresholds so
Depth noise and motor inertia do not make the vehicle alternate rapidly between forward
and reverse.

### Configured thresholds

| Range or threshold | Meaning | Longitudinal behavior |
|---|---|---|
| `> 1.80 m` | Forward start threshold | Start or resume forward following when the target is visible and the distance is fresh/credible. |
| `1.65-1.80 m` | Forward hysteresis band | Keep the current forward/hold state; do not repeatedly start and stop at a single noisy boundary. |
| `1.50 m` | Nominal following distance | The distance PID setpoint. |
| `1.35-1.65 m` | PID deadband / practical hold band | Normally command zero longitudinal speed (while horizontal steering remains active). This is the intended steady-state range around 1.5 m. |
| `< 1.30 m` | Reverse start threshold | Require fresh stable Depth and the configured confirmation count, then reverse away from the target. A fresh near reading may enter this state immediately for safety. |
| `1.30-1.45 m` | Reverse recovery band | Continue reversing until the distance reaches at least `1.45 m`; reverse speed is controlled by the distance PID plus approach-speed feed-forward. |
| `>= 1.45 m` while reversing | Reverse release threshold | Stop the reverse component and return to the hold state. The 0.10 m separation from the reverse start threshold is deliberate hysteresis. |
| `<= 0.50 m` | Hard distance safety reference | Treat as an emergency-distance condition and stop longitudinal motion. This does not replace the physical IR safety path. |

The exact values currently deployed in `reid_runtime.ini` are:

```ini
[distance]
target_distance_m = 1.5
target_distance_release_m = 1.5
target_distance_release_hold_sec = 0.50
target_distance_release_confirm_frames = 3
brake_distance_m = 0.5
forward_start_distance_m = 1.80
forward_stop_distance_m = 1.65
reverse_start_distance_m = 1.30
reverse_immediate_distance_m = 1.30
reverse_stop_distance_m = 1.45
reverse_full_speed_distance_m = 1.00
```

`parking_enable = false` in the normal configuration. Therefore `1.50 m` is the PID
setpoint and the `1.35-1.65 m` deadband is the practical hold region; it is not a
permanent parking latch. The physical front/side IR trigger always has higher priority
than every Depth or PID decision and must issue an immediate hard stop.

### Direction and speed rules

- Distance greater than `1.80 m` permits forward motion. The forward command is reduced
  as the target approaches `1.65 m` and is held at zero inside the deadband.
- Distance below `1.30 m` permits reverse motion. Reverse must not wait for a slow PID
  ramp when a fresh near reading confirms that the target is approaching; the current
  test configuration provides a `30 RPM` feed-forward floor, with a runtime cap of
  `60 RPM` and an absolute configured maximum of `100 RPM`.
- Reversing stops at `1.45 m`, not at exactly `1.50 m`, so small Depth fluctuations do
  not cause immediate forward/reverse chatter.
- A target distance change must not disable horizontal camera/encoder steering. The
  vehicle can command zero longitudinal speed and a non-zero differential wheel speed
  to re-center the target.
- When changing from forward to reverse, the transition stop must not clear the newly
  calculated reverse RPM. The first effective reverse command must contain the computed
  speed instead of an unintended zero.

### Depth confidence and stale-data behavior

Distance thresholds only apply to a fresh, credible measurement. Depth is handled in
three confidence levels:

1. **High confidence (`0-200 ms` fresh):** use the multi-region foreground depth and
   run the normal distance PID.
2. **Medium confidence (`200-600 ms` since the last valid sample):** estimate distance
   from the last measurement, encoder displacement, and target-box scale change. Limit
   longitudinal speed to `20 RPM`; keep horizontal steering active.
3. **Low confidence (`>600 ms` without a valid sample):** do not accelerate toward an
   old or background distance. Stop longitudinal motion or use only a tightly limited
   recovery command; keep camera-based horizontal correction and target identity alive.

A sudden near measurement can be accepted for braking/reverse, while a sudden far
measurement must be confirmed over consecutive fresh frames. A stale near distance must
not reject a new far anchor forever: the old anchor is strict for less than `0.60 s`,
down-weighted from `0.60-1.50 s`, and no longer used as a jump baseline after `1.50 s`.

### Safety priority

The priority order is fixed:

1. Front/side infrared trigger: immediate hard stop.
2. Distance at or below `0.50 m`: stop longitudinal motion.
3. Missing or low-confidence distance: never issue an old-distance forward command;
   preserve horizontal steering and apply the confidence speed limit.
4. Fresh valid distance: apply the forward/hold/reverse thresholds and PID above.

This table describes the intended stable behavior: the car should settle near `1.50 m`,
remain within roughly `1.35-1.65 m` during normal measurement noise, move forward only
when it is genuinely farther than the forward-start region, and reverse promptly when a
fresh target distance enters the near-distance region.

### Distance PID parameters

The range thresholds decide which direction is allowed; the PID determines how much
longitudinal speed is requested inside an allowed direction. The current deployed values
are:

```ini
[distance_pid]
enable = true
kp_rpm_per_m = 22.0
ki_rpm_per_m_s = 1.5
kd_rpm_s_per_m = 6.0
integral_limit_m_s = 1.5
deadband_m = 0.15
derivative_filter_alpha = 0.25
```

The signed error is `distance - 1.50 m`: positive error means the person is too far
away and permits forward speed; negative error means the person is too close and permits
reverse speed only after the reverse-start rules above are satisfied. Inside the
`+/-0.15 m` deadband the longitudinal request is zero, but the lateral camera/encoder
PID continues to run.

### Typical examples

| Measured distance | Expected interpretation |
|---|---|
| `2.10 m` | Clearly too far: start/continue forward, with speed limited as it approaches `1.65 m`. |
| `1.70 m` | Between forward stop and start: preserve the current state and avoid a new burst of motion. |
| `1.52 m` | Inside the hold band: no forward or reverse request; keep correcting horizontal position. |
| `1.38 m` | Still inside the hold band: do not reverse merely because the value is below `1.50 m`. |
| `1.24 m` | Below reverse start: confirm fresh Depth, then reverse promptly. |
| `1.42 m` while reversing | Continue reverse until the release threshold is reached. |
| `1.47 m` while reversing | Release reverse and return to hold; do not immediately start forward. |
| `0.48 m` | Safety distance reached: stop longitudinal motion and rely on IR as the physical hard-stop authority. |

### State transition order

For each fresh target/depth update, the controller follows this order:

```text
IR trigger or hard-distance condition
    -> immediate longitudinal STOP

fresh distance < 1.30 m
    -> reverse (after confirmation, unless an immediate near-safety reading applies)

fresh distance in 1.30-1.45 m while reversing
    -> keep reversing with bounded speed

fresh distance >= 1.45 m while reversing
    -> release reverse and hold

fresh distance in 1.35-1.65 m
    -> distance hold (zero longitudinal speed, lateral steering still enabled)

fresh distance > 1.80 m
    -> forward follow

stale/invalid distance
    -> confidence-limited estimate or longitudinal hold; never use an old value to
       accelerate blindly, while lateral steering remains active
```

The state machine deliberately has asymmetric protection: accepting a sudden near
reading quickly helps prevent a collision, while accepting a sudden far reading requires
consecutive confirmation so a background wall cannot make the car accelerate. The
distance state also never owns the horizontal wheel difference; distance hold must not
freeze the camera-centering controller.

## Depth Measurement

- Astra Depth is approximately 30 FPS.
- Current measurement uses five torso regions: chest center, abdomen center, left torso, right torso, and lower abdomen.
- Each region uses valid foreground depth clusters and chooses a history-continuous cluster.
- Dynamic valid-pixel threshold is `max(20, visible_roi_area * 3%)`.
- Old distance anchors become weak after 0.6 s and invalid after 1.5 s.
- Sudden near readings are accepted quickly for braking/reverse; sudden far readings require confirmation.
- A recent log showed stable multi-region depth around 1.55 m while the target was centered.

## Steering PID

- Normal config currently logs:
  - center band: `0.40..0.60`
  - camera HFOV: `60 deg`
  - outer PID max yaw: `46 deg/s`
  - correction max: `16 RPM`
  - encoder feedback interval: `100 ms`
  - feedback median window: `3`
- Rotation-only testing previously used a 30 RPM startup kick, released when encoder yaw reached 4 deg/s. That startup behavior is only for `reid_runtime_rotation_only.ini` and must not be assumed to be active in normal follow mode.
- Do not treat the fixed 30 RPM kick as the complete PID. It is only mechanical startup compensation.

### Rotation decision conditions

Rotation is decided independently from the distance controller. The distance loop chooses
the common forward/reverse component; the camera/encoder loop chooses the differential
wheel component. A distance-hold decision therefore does **not** mean that both wheels
must be stopped: the car may rotate in place or use a small differential to keep the
person centered.

The horizontal target coordinate is normalized to `x in [0, 1]`, with `0.5` as the image
center. The outer-loop error is:

```text
e_x = target_center_x - 0.50
```

After applying the configured motor sign mapping:

| Target position | Rotation decision |
|---|---|
| `0.40 <= x <= 0.60` | Target is inside the center band. Desired yaw is zero and wheel difference is zero, unless the encoder feedback still shows residual body yaw that needs braking. |
| `x < 0.40` | Target is left of center. Request left corrective yaw; the farther left and faster it moves, the larger the requested yaw rate. |
| `x > 0.60` | Target is right of center. Request right corrective yaw using the same rule. |
| `x < 0.20` or `x > 0.80` | Target is near an image edge. Reduce longitudinal speed if needed, but preserve strong differential steering so the target is not allowed to leave the frame. |
| No high-quality target for the loss-confirmation window | Do not steer from an untrusted small box. Keep the last reliable exit direction, stop stale translation, and enter search only after the configured consecutive-miss count. |

The center band is a control dead zone, not a detector gate. A person at `x=0.45` with
valid distance may remain stationary longitudinally while the car continues tracking
distance; a person at `x=0.30` must receive a steering command even if the distance is
already inside `1.35-1.65 m`.

### Encoder feedback loop

The camera position is the outer loop and the wheel encoder is the only yaw feedback used
in normal follow mode. IMU data is intentionally excluded from this loop. The per-cycle
decision is:

```text
camera x error
    -> desired yaw rate (outer PID)
desired yaw rate - encoder measured yaw rate
    -> differential wheel correction (inner PI/PID)
common distance speed +/- differential correction
    -> left and right wheel targets
```

The encoder yaw estimate comes from the left/right wheel displacement difference over the
feedback interval. The feedback is filtered with the configured median window before it
is used for correction. A correction in the opposite direction of the measured yaw is
active braking, not a new target direction; it must be bounded so the vehicle does not
oscillate across the center band.

The final wheel command must be formed by mixing the two independent components:

```text
left_wheel  = longitudinal_speed - yaw_correction
right_wheel = longitudinal_speed + yaw_correction
```

The yaw correction must not be clamped to the longitudinal speed. In particular, when
`longitudinal_speed = 0` inside the distance hold band, a non-zero left/right difference
is still valid and is required for in-place centering. A STOP may be sent for IR,
hard-distance, or confirmed target loss, but must not be inserted after every ordinary
steering update or after a same-direction PID update.

### Rotation safety and target-quality gates

- Only a high-quality, geometrically plausible target updates the steering error,
  `last_valid_bbox`, and exit-direction history.
- A tiny, highly elongated, or sudden area-collapse detection may be retained as a weak
  ReID sample, but cannot provide a new steering command, change the search direction, or
  reset the target-loss counter by itself.
- Depth validity controls longitudinal confidence only. `far_background_guard` or a
  short Depth gap must not reduce a valid target's horizontal correction to an ineffective
  1-6 RPM.
- A fresh valid target at the image edge must be turned toward immediately; waiting for a
  second frame before starting the correction is likely to let the target leave the frame.
- When target distance is safe and valid, a near-distance/reverse overlay must not replace
  a current horizontal PID command with a stale cached direction.
- When the target is lost, search uses the last reliable left/right trajectory. Search
  rotation pauses when a credible candidate is seen and resumes only after the candidate
  fails the confirmation check. The mechanical settle time is based on encoder angular
  speed, not a fixed short delay.

### Rotation log fields

Every steering decision should be diagnosable from one log record containing:

```text
target_center_x, e_x, center_band, target_quality
desired_yaw_rate, measured_encoder_yaw_rate, feedback_age_ms
yaw_error, yaw_correction_rpm
longitudinal_speed_rpm, left_target_rpm, right_target_rpm
depth_source, depth_age_ms, action_kind, stop_reason
```

The key acceptance condition is: while a reliable person remains visible, the car must
continue producing a timely differential correction whenever `x` is outside `0.40-0.60`,
even if the distance loop is holding at `1.50 m` or Depth is temporarily unavailable.

## Known Recent Log Behavior

The latest normal run was started with `./run_request_0428_modular.sh` and used `reid_runtime.ini`:

- Astra Depth enabled, mmWave disabled, `rotation_only=False`.
- Distance was initially stable and close to target.
- When the person moved laterally, the vehicle began to oscillate and look sluggish.
- Last log: `/home/topeet/Desktop/rk_car_runtime_module/run_request_0428_modular_logs/request_0513_modular.log`
- Recent log duration was about 30 seconds, ending at frame 384.
- Latest frame had target center ratio about `0.452`, Depth `1.55 m`, but the controller reason was `target_distance_hold` and it issued a forward action with zero speed before shutdown.
- The log contains many `visual_pid_left`, `visual_pid_right`, and `reverse_visual_pid` decisions. This must be separated into stable-target and lateral-motion intervals before changing gains.
- Some normal runs have shown repeated STOP followed within a few milliseconds by a steer action. This creates visible jolts and must be eliminated for the same-direction candidate/target state.
- Search settle waits can take roughly 0.6-2.2 s because the mechanical chassis takes time to stop; this matters only after target loss, not during normal visible tracking.

## Diagnostic Procedure

When a new log is available, read it directly on the board:

```bash
cd /home/topeet/Desktop/rk_car_runtime_module
rg -n "pipeline_timing|控制决策:|视觉转向PID:|reverse_visual_pid|Astra depth regions|编码器转向反馈|电机下发时序|进入刹车保持状态|target_distance_hold" \
  run_request_0428_modular_logs/request_0513_modular.log
```

Compare these fields during lateral motion:

- target center ratio and its frame-to-frame trend;
- visual error, filtered error, desired yaw rate;
- measured encoder yaw rate and feedback age;
- PID output wheel difference and any opposite/overspeed brake flag;
- final left/right wheel targets and action kind;
- whether a STOP is sent before a same-direction steering update;
- Depth source, sample age, and whether `reverse_visual_pid` is active;
- pipeline timing and stale-result discard status.

Do not change gains until the log identifies whether the oscillation is caused by:

1. camera/encoder latency and stale visual results;
2. excessive opposite-yaw braking or derivative noise;
3. normal forward/reverse overlay repeatedly replacing the yaw command;
4. action queue STOP insertion;
5. target/ReID box switching;
6. Depth hold/reverse state overriding a valid horizontal PID.

## Backups

- Rotation-only deployment backup on board: `/home/topeet/Desktop/rk_car_runtime_module/.codex_backup_rotation30_20260821_2155`

## Important Constraint

This document is a handoff summary, not a migration of the original Codex transcript. The VS Code ChatGPT plugin can use this file as project context, but it will not automatically know every prior conversational turn.
