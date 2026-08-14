# RKNPU2 runtime

This directory is populated from the local Rockchip Toolkit2 package instead of
committing binary runtime files.

Extract the runtime package:

```powershell
.\scripts\extract_rknpu2_runtime.ps1
```

Default source package:

```text
C:\Users\Dylon\Downloads\rk\2.3.2\release\rknn-toolkit2-v2.3.2-2025-04-09.tgz
```

After extraction, useful paths are:

```text
third_party/rknpu2/runtime/Linux/librknn_api/include
third_party/rknpu2/runtime/Linux/librknn_api/aarch64
third_party/rknpu2/runtime/Linux/rknn_server/aarch64/usr/bin
```

Stage the board-side Python wheel for RK3588 Ubuntu 22.04 / Python 3.10:

```powershell
.\scripts\stage_rknn_lite2_wheel.ps1
```

This copies the `rknn_toolkit_lite2` aarch64 `cp310` wheel into
`third_party/rknpu2/packages/` for later transfer to the board.

Keep the board-side `rknn_server`, `librknnrt.so`, and conversion Toolkit2
versions matched.
