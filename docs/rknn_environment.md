# RKNN 2.3.2 environment

The RK3588 board at `192.168.0.64` currently reports:

- OS: Ubuntu 22.04.4 LTS, aarch64
- Python: 3.10.12
- Python runtime: `rknnlite` is installed, `rknn` is not installed
- Runtime library: `/usr/lib/librknnrt.so` version `2.3.2`

Use the local `2.3.2` release package as the source of truth:

```text
C:\Users\Dylon\Downloads\rk\2.3.2\release\rknn-toolkit2-v2.3.2-2025-04-09.tgz
```

The downloaded repository snapshot below overlaps with the release package. Keep
it as offline docs/examples, but do not use it as the default script source:

```text
C:\Users\Dylon\Downloads\rk\rknn-toolkit2-2.3.2.zip
```

Avoid the local `2.3.2\develop` directory for this project. It contains `2.3.3`
alpha/beta packages, which do not match the board runtime.

Host conversion uses RKNN-Toolkit2 in Docker:

```powershell
.\scripts\build_rknn_toolkit2_image.ps1
```

If using the local `cv18xx:v1` base image, use its Python 3.12 wheel:

```powershell
.\scripts\build_rknn_toolkit2_image.ps1 -BaseImage cv18xx:v1 -PythonTag cp312 -Proxy http://host.docker.internal:10808
```

Board-side Python inference uses RKNN-Toolkit-Lite2:

```powershell
.\scripts\stage_rknn_lite2_wheel.ps1
```

RKNPU2 runtime files can be staged with:

```powershell
.\scripts\extract_rknpu2_runtime.ps1
```

Keep Toolkit2, Toolkit-Lite2, `librknnrt.so`, and `rknn_server` on the same
major package version unless we intentionally upgrade the whole board stack.
