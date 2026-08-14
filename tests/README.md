# Runtime Module Tests

These tests are executable scripts.  They are intentionally split by module so
board-side checks can be run one at a time.

## Safe local checks

```bash
python3 tests/config/test_rk3588_runtime_config.py car_control_modular/config/reid_runtime.ini
python3 tests/control/test_controller_logic.py
python3 tests/vision/test_deepsort_tracker.py
python3 tests/vision/test_identity_bank.py
python3 tests/motor/test_mssd_mapping.py --config car_control_modular/config/reid_runtime.ini
python3 tests/sensors/test_ir_iio.py --fake
python3 tests/sensors/test_ultrasonic_iio.py --fake
```

On Windows without local `numpy`/`requests`, run the dependency-light checks via
`uv`, and run the vision checks in the RKNN Docker image:

```powershell
uv run --with requests python tests\motor\test_mssd_mapping.py --config car_control_modular\config\reid_runtime.ini
uv run --with requests python tests\sensors\test_ir_iio.py --fake
uv run python tests\sensors\test_ultrasonic_iio.py --fake
docker run --rm -v "${PWD}:/work" -w /work rk-car-rknn-toolkit2:2.3.2 python3 tests/vision/test_deepsort_tracker.py
docker run --rm -v "${PWD}:/work" -w /work rk-car-rknn-toolkit2:2.3.2 python3 tests/vision/test_identity_bank.py
```

## Board checks

IR IIO live read:

```bash
python3 tests/sensors/test_ir_iio.py --config car_control_modular/config/reid_runtime.ini --live
```

SR04 ultrasonic IIO live read:

```bash
python3 tests/sensors/test_ultrasonic_iio.py --config car_control_modular/config/reid_runtime.ini --live
```

MSSD motor dry-run mapping:

```bash
python3 tests/motor/test_mssd_mapping.py --config car_control_modular/config/reid_runtime.ini
```

MSSD motor live smoke test.  This sends a short command, then stop.  Keep
`--percent` at or below 20 during the current test phase.

```bash
python3 tests/motor/test_mssd_live.py --config car_control_modular/config/reid_runtime.ini --direction forward --percent 10 --seconds 0.2 --execute
```

## Tracker segment evaluation

Run the fixed local ONNXRuntime segment suite in Docker:

```bash
scripts/run_tracker_segment_suite_local.sh
```

Run the same segment suite on the RK3588 board over SSH.  By default this uses
the `rk` SSH host alias and `/home/topeet/Desktop/rk_car_runtime`:

```bash
scripts/run_tracker_segment_suite_board.sh
```

Compare a stricter detector threshold without changing config files:

```bash
CONFIDENCE_THRESHOLD=0.7 OUT_DIR=.test_outputs/tracker_segment_smokes_conf07 \
  scripts/run_tracker_segment_suite_local.sh

CONFIDENCE_THRESHOLD=0.7 REMOTE_OUT=smoke_frames/board_segment_smokes_conf07 \
  LOCAL_OUT=.test_outputs/board_segment_smokes_conf07 \
  scripts/run_tracker_segment_suite_board.sh
```

Verify predicted-only tracker output with an extra ReID crop against the
identity bank:

```bash
VERIFY_PREDICTED_REID=1 PREDICTED_REID_VERIFY_THRESHOLD=0.30 \
  OUT_DIR=.test_outputs/tracker_segment_smokes_verify \
  scripts/run_tracker_segment_suite_local.sh
```

Summarize existing JSONL/log outputs without rerunning inference:

```bash
python3 tools/tracker_segment_eval.py \
  --input-dir .test_outputs/board_segment_smokes \
  --suite-name board-lite2 \
  --output-md .test_outputs/tracker_reports/board_lite2_segment_report.md \
  --output-json .test_outputs/tracker_reports/board_lite2_segment_report.json
```
