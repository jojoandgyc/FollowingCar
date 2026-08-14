# RKNN Toolkit2 Docker

Build this image from the Rockchip Toolkit2 package downloaded outside the repo.

Default local package on macOS/Linux:

```text
/Users/dylon/Downloads/Filez/WebTool/2.3.2/release/rknn-toolkit2-v2.3.2-2025-04-09.tgz
```

Build the Ubuntu 22.04 / Python 3.10 image:

```bash
scripts/build_rknn_toolkit2_image.sh
```

On Apple Silicon Macs the script defaults to `--platform linux/amd64` because
the RKNN-Toolkit2 conversion wheel is Linux x86_64.

The script can also build from the downloaded zip:

```bash
scripts/build_rknn_toolkit2_image.sh --package /Users/dylon/Downloads/rknn-toolkit2-2.3.2.zip
```

If Docker is not available yet, stage and verify the matching wheel only:

```bash
scripts/build_rknn_toolkit2_image.sh --stage-only
```

Default local package on Windows:

```text
C:\Users\Dylon\Downloads\rk\2.3.2\release\rknn-toolkit2-v2.3.2-2025-04-09.tgz
```

Build the same image from PowerShell:

```powershell
.\scripts\build_rknn_toolkit2_image.ps1
```

Use the local proxy for network access inside Docker build steps:

```powershell
.\scripts\build_rknn_toolkit2_image.ps1 -Proxy http://host.docker.internal:10808
```

If Docker Hub is unavailable but a local base image is present, pass it
explicitly with the matching wheel tag:

```powershell
.\scripts\build_rknn_toolkit2_image.ps1 -BaseImage cv18xx:v1 -PythonTag cp312 -Proxy http://host.docker.internal:10808
```

The `-Proxy` option helps commands running inside the build container. If Docker
cannot pull the base image from Docker Hub, configure the Docker Desktop daemon
proxy to `http://127.0.0.1:10808` first, then rerun the build.

Verify:

```bash
docker run --rm rk-car-rknn-toolkit2:2.3.2
```

The script extracts the matching Python wheel into `.cache/`, then builds a
Linux x86_64 image for model conversion.

The Dockerfile pins `onnx==1.16.1` after installing RKNN-Toolkit2. Newer ONNX
releases removed APIs still used by RKNN-Toolkit2 2.3.2 during model loading.
It also pre-installs CPU-only Torch before RKNN-Toolkit2 so the development
image does not pull CUDA packages.

Run repo checks inside the image:

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace rk-car-rknn-toolkit2:2.3.2 \
  python3 -m compileall rk_vision request_0428_modular.py car_control_modular tools tests

docker run --rm -v "$PWD:/workspace" -w /workspace rk-car-rknn-toolkit2:2.3.2 \
  python3 tests/vision/test_identity_bank.py

docker run --rm -v "$PWD:/workspace" -w /workspace rk-car-rknn-toolkit2:2.3.2 \
  python3 tests/vision/test_deepsort_tracker.py
```

Run the 1000s tracker replay from the container:

```bash
mkdir -p .test_outputs/tracker_1000s_identity_bank
docker run --rm -v "$PWD:/workspace" -w /workspace rk-car-rknn-toolkit2:2.3.2 \
  python3 tools/rknn_video_smoke.py .test_car/test_video/camera_record_1000s_20260511_132259.mp4 \
    --backend toolkit2 \
    --yolo-model models/yolo11s.rknn \
    --reid-model models/osnet_x0_5_msmt17_combineall_b1.rknn \
    --reid-input-width 128 \
    --reid-input-height 256 \
    --reid-input-format RGB \
    --reid-input-dtype float32 \
    --reid-input-layout NCHW \
    --reid-normalize imagenet \
    --feature-update-interval 3 \
    --max-output-age 5 \
    --max-frames 1707 \
    --jsonl .test_outputs/tracker_1000s_identity_bank/yolo11_osnet_x05_identity_bank.jsonl \
    --save-video .test_outputs/tracker_1000s_identity_bank/yolo11_osnet_x05_identity_bank.mp4 \
    --debug-tracker-state
```
