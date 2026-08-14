# RK3588 Build And Test Commands

This runtime no longer builds the old HD05075A/CVI `sample_personv8_track`
program for the main person-follow path.

## Required runtime packages on RK3588

- Python 3
- `numpy`
- `opencv-python` or a board OpenCV package
- Rockchip `rknn-toolkit-lite2`
- RKNN runtime library from the board image

## Model files

Place converted RKNN models here:

```text
models/yolo11s.rknn
models/osnet_x0_5_msmt17_combineall_b1.rknn
```

Convert the default YOLO11s model with the RKNN-Toolkit2 2.3.2 Docker image:

```powershell
.\scripts\convert_yolo11_rknn.ps1 -Variant s -DType fp
```

## Syntax and logic checks

```bash
python3 -m compileall rk_vision request_0428_modular.py car_control_modular tools tests
python3 tests/config/test_rk3588_runtime_config.py car_control_modular/config/reid_runtime.ini
python3 tests/control/test_controller_logic.py
python3 tests/vision/test_deepsort_tracker.py
python3 tests/vision/test_identity_bank.py
python3 tests/motor/test_mssd_mapping.py --config car_control_modular/config/reid_runtime.ini
python3 tests/sensors/test_ir_iio.py --fake
python3 tests/sensors/test_ultrasonic_iio.py --fake
```

## Single-image model smoke test

```bash
python3 tools/rknn_image_smoke.py test.jpg \
  --yolo-model models/yolo11s.rknn \
  --reid-model models/osnet_x0_5_msmt17_combineall_b1.rknn \
  --reid-input-width 128 --reid-input-height 256 --reid-input-format RGB
```

## External frame integration shape

```python
import cv2
from request_0428_modular import PersonTracker

tracker = PersonTracker()
frame = cv2.imread("test.jpg")  # BGR HxWx3 uint8
tracker.process_external_frame(frame, frame_format="BGR")
```
