# Follow-Car Modular Runtime

This package contains the pieces that `request_0513_modular.py` calls while
running the RK3588 follow-car program.  The refactor is intentionally staged:
the active behavior stays in place while bulky motor, distance, and controller
logic is moved out of the 0513 entrypoint.

## Active Entry Point

Use `request_0513_modular.py` for the current car runtime.

```bash
./run_request_0428_modular.sh --config car_control_modular/config/reid_runtime.ini
python3 -u request_0513_modular.py --config car_control_modular/config/reid_runtime.ini
```

`request_0428_modular.py` and `request_0512_modular.py` are older/reference
entrypoints.  They can still be useful for comparison, but new control work
should target `request_0513_modular.py`.

`request_0513_modular.py` is RKNN-only.  The old `sample_reid`/CVIMODEL path is
kept only in the older reference entrypoints, so a ReID config for 0513 means
`engine = rknn` plus `reid_enable = true`.

## Active Runtime Path

The current request flow is:

1. `request_0513_modular.py` reads config, starts camera/model/sensors, builds
   `PersonTarget` and `SensorFrame`, then hands work to the runtime modules.
2. `BunkerHazardRuntime` starts split/merged bunker and pond detection and
   reports active visual hazards.
3. `DistanceRuntime` chooses the distance source and, when configured, matches
   a vision person box with mmwave targets.
4. `FollowSafetyController` decides whether to stop, search, rotate, steer, or
   move forward.
5. `MotionActionRuntime` consumes queued actions and applies transition stop,
   rotate pulse, brake hold, and repeated command timing.
6. `MssdMotorBackend` maps percent/raw commands to signed left and right wheel
   RPM values and calls the LZ-30EMA RS485 library.

## File Ownership

| File | Current status | Responsibility |
| --- | --- | --- |
| `control_types.py` | Active | Shared dataclasses such as `PersonTarget`, `SensorFrame`, `ControlAction`, and `ControlDecision`. |
| `controllers.py` | Active, with legacy candidates | Current policy is `FollowSafetyController` plus `FollowPolicyConfig`. The earlier `SafetyController`, `FollowController`, and `DecisionPipeline` are staged/legacy candidates and are not the 0513 runtime path. |
| `distance_runtime.py` | Active | Distance source selection, raw distance reads, and vision-mmwave target matching/cache. |
| `hazard_runtime.py` | Active | Starts bunker/pond hazard detectors, confirms consecutive active hazard frames, logs hazard triggers, and returns hazard states. |
| `action_runtime.py` | Active | Action thread, action queue, forward/rotate/stop execution, rotate pulse timing, brake hold, transition stops, and safety stop dispatch. |
| `mssd_motor.py` | Active | Compatibility adapter for the LZ-30EMA_2EC_N RS485 controller. It owns wheel RPM mapping and stop-mode dispatch. |
| `sensor_modules.py` | Active | Sensor lifecycle and lightweight reads for IR, ultrasonic, mmwave, and IMU. |
| `config_loader.py` | Active | Loads INI values into environment variables before runtime constants are evaluated. |

## Motor Driver Source

`mssd_motor.py` keeps its historical filename and class names so the control
runtime does not need a broad migration. Its implementation imports the
`lz30ema_rs485` package from `MOTOR_RS485_LIB_DIR` at runtime.

The board config points at the Lianzhan library root; the adapter automatically
resolves its `src/` package directory:

```ini
[motor]
backend = rs485_lz30ema
rs485_lib_dir = /home/topeet/lianzhan
rs485_max_target = 100  # maximum wheel RPM for percent mode
```

The old vendored `mssd_60ehb_rs485.py` file is retained only as historical
reference and is no longer imported by the active 0513 runtime.

## Cleanup Boundaries

Safe cleanup candidates, in order:

1. Review the legacy controller skeleton in `controllers.py` after tests confirm
   only `FollowSafetyController` is needed by active entrypoints.
2. Keep pruning request-level helpers once the same behavior is owned by an
   active submodule.

## Quick Checks

```bash
python3 -m py_compile request_0513_modular.py car_control_modular/*.py
python3 tests/motor/test_mssd_mapping.py
python3 tests/control/test_controller_logic.py
python3 tests/control/test_hazard_runtime.py
python3 tests/control/test_sensor_runtime.py
python3 tests/config/test_rk3588_runtime_config.py
```
