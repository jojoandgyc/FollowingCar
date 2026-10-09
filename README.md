# FollowingCar

多人开发的当前目录架构、线程/授权边界、活动配置和调试入口见
[项目架构与开发交接](docs/PROJECT_ARCHITECTURE_HANDOFF.md)。

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

The default RK3588 configuration uses the model files included under `models/`:

- `models/yolo11n_int8_person_val2017.rknn` (person detection)
- `models/osnet_x0_25_msmt17_b1.rknn` (person ReID)

`models/README.md` records the optional model conversion experiments.

## 在 RK3588 开发板运行

在 RK3588 开发板上进入克隆目录后执行：

```bash
cd /path/to/FollowingCar
sudo -v
./run_request_0428_modular.sh --config car_control_modular/config/reid_runtime.ini
```

按 `Ctrl+C` 会触发安全停车并退出。不要使用 `sudo ./run_request_0428_modular.sh`
启动整个项目，否则运行日志可能被创建为 `root` 权限。

现用 RKNN 模型、`librknnrt.so`、Astra OpenNI 运行文件和 LZ30EMA 电机驱动源码
已随此分支提供。默认配置从 `runtime_dependencies/` 读取后两项。
Python 包、设备驱动与硬件接口仍需在 RK3588 开发板安装；版本和路径见
[`runtime_dependencies/README.md`](runtime_dependencies/README.md)。

## 运行记录

每次启动在 `run_request_0428_modular_logs/run_<时间戳>_<唯一编号>/` 中保存
日志、`camera_raw.avi`、`camera_raw.frames.csv` 和搜索快照，保留最近三次运行。
启动第四次时删除最旧的一次受管记录，不能恢复；请将需要长期保留的记录移到此目录外。
第一次升级启动会把原来平铺的已知日志和视频归档为一次历史记录，其他文件不动。
终端输出的 `log_dir` 是本次运行的具体目录。

视频文字默认不透明度为 62%，右侧 `SHARP` 显示当前原始画面的清晰度，
同一数值写入 CSV 的 `sharpness` 列。计算方法为灰度图缩小至宽度不超过 320 像素后，
取 Laplacian 方差；它不是 0~1 分数，也不是搜索日志中的低角速度基线比例。
测量、文字叠加和编码均在录像后台线程执行，队列满时丢弃录像帧，不等待主控。

## 骨骼辅助测距观察实验

已加入默认关闭的骨骼旁路观察，只记录区域、距离统计和耗时，不改变跟随控制。
融合方案、相机专用采样与开启方式见 [方案说明](docs/pose_shadow_integration.md)，
两轮相机实测及局限见 [测试报告](docs/pose_shadow_test_report.md)。

## Smoke Test

Once the board-side Python dependencies are installed:

```bash
python3 tools/rknn_image_smoke.py test.jpg \
  --yolo-model models/yolo11n_int8_person_val2017.rknn \
  --reid-model models/osnet_x0_25_msmt17_b1.rknn
```
