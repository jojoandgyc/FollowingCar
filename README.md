# FollowingCar

RK3588 跟随小车运行项目，集成人员检测与 ReID、摄像头采集、红外避障、
AT2410 毫米波测距、超声波、IMU 和 LZ30EMA 双轮电机控制。

The vehicle control, IR, mmwave, ultrasonic and motor HTTP/UART logic are kept
from the original project.  The vision path is changed to a decoupled RKNN
model pipeline:

```text
external camera frame -> RKNN YOLO11 -> RKNN ReID -> tracker -> control layer
```

主程序可直接打开开发板摄像头，也支持由外部代码传入 BGR/RGB 画面：

```python
tracker.process_external_frame(frame, frame_format="BGR")
```

## Models

Put RK3588 `.rknn` models under `models/`:

- `models/yolo11s.rknn`
- `models/deepsort.rknn`

YOLO11 can be converted with Rockchip's official `rknn_model_zoo/examples/yolo11`
flow.  The default ReID model is the board-provided DeepSORT embedding RKNN
model from the working RK3588 reference project.

## 在 RK3588 开发板运行

项目部署在 `/home/topeet/Desktop/rk_car_runtime_module` 时执行：

```bash
cd /home/topeet/Desktop/rk_car_runtime_module
sudo -v
./run_request_0428_modular.sh --config car_control_modular/config/reid_runtime.ini
```

按 `Ctrl+C` 会触发安全停车并退出。不要使用 `sudo ./run_request_0428_modular.sh`
启动整个项目，否则运行日志可能被创建为 `root` 权限。

RKNN 模型和板端运行时二进制未纳入 Git，请按 `models/README.md` 和
`docs/rknn_environment.md` 准备运行环境。

## Smoke Test

Once the RKNN runtime and models are available on the RK3588 board:

```bash
python3 tools/rknn_image_smoke.py test.jpg \
  --yolo-model models/yolo11s.rknn \
  --reid-model models/deepsort.rknn
```
