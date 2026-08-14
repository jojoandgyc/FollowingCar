# 跟随车项目 5 分钟实车测试报告

## 1. 结论

测试时间：2026-08-12 12:08:59 至 12:13:59（约 300 秒）。

项目能够持续运行，摄像头、RKNN 检测、ReID、DeepSORT/IdentityBank、ICM20600 IMU 和 LZ30EMA 电机控制器均成功启动，5 分钟内没有摄像头读取失败、Python Traceback 或 NPU 崩溃。

但是，当前版本还不能判定为“跟随功能正常”，不建议在没有人工急停保护的情况下继续自动跟随测试。主要原因有两个：

1. 毫米波雷达 5 分钟内没有产生任何有效目标数据，但雷达距离缺失时小车仍会按 55% 的最低前进速度行驶。
2. 原目标丢失后，即使画面中重新出现稳定的人体检测框，系统也无法重新锁定，车辆会持续向右原地搜索。

另外，当前启动脚本对退出信号的传递不可靠，超时结束时 Python 没有执行正常的停车和资源清理流程。本次测试结束后已人工终止残留进程，并通过 `/home/topeet/lianzhan` 电机库发送双轮零速和 NORMAL STOP，确认输出 `explicit_stop=ok`。目前没有项目残留进程，相关设备节点均已释放。

## 2. 测试环境与命令

- 开发板：`192.168.0.64`
- 项目目录：`/home/topeet/Desktop/rk_car_runtime_module`
- 配置文件：`car_control_modular/config/reid_runtime.ini`
- Python 主程序：`request_0513_modular.py`
- 电机控制器：LZ30EMA，串口 `/dev/ttyS0`，115200 baud
- 摄像头：`/dev/video1`，GStreamer MJPEG，1920x1080@30 FPS
- 毫米波配置：`/dev/ttyS3`，9600 baud

正常启动命令：

```bash
cd /home/topeet/Desktop/rk_car_runtime_module
./run_request_0428_modular.sh \
  --config car_control_modular/config/reid_runtime.ini
```

本次测试在开发板后台运行，并由外层 `timeout` 在 300 秒时结束。外层返回码为 `124`，表示确实到达了 5 分钟超时时间。

## 3. 运行统计

| 项目 | 结果 |
| --- | ---: |
| 运行时长 | 约 300 秒 |
| 处理到的摄像头帧 | 约 2092 帧 |
| 控制决策 | 1433 次 |
| IMU 样本 | 520 次 |
| 电机命令 | 3368 次 |
| TURN 命令 | 3204 次 |
| TURN_ZERO 命令 | 163 次 |
| DRIVE 命令 | 1 次 |
| 毫米波有效帧 | 0 |
| `no_radar_targets` 控制记录 | 1432 次 |
| 摄像头读取失败 | 0 |
| Python Traceback | 0 |
| 电机发送平均耗时 | 约 7.43 ms |
| 电机发送最大耗时 | 约 179.20 ms |
| 最高温度 | 约 60 摄氏度 |
| Python RSS | 约 255 MB 增长到 340 MB |
| 持续搜索时 CPU | 约 3.6 个核心 |
| 完整日志大小 | 5,285,573 bytes |

RSS 在 5 分钟内有约 85 MB 增长，可能包含模型预热、缓存和队列稳定过程。仅凭本次测试不能判定内存泄漏，建议完成高优先级问题修复后再做一次 30 分钟稳定性测试。

## 4. 已确认正常或已解决

### 4.1 ICM20600 IMU 权限问题已解决

IMU 初始化成功，使用以下设备：

- 加速度 event：`/dev/input/event3`
- 陀螺仪 event：`/dev/input/event5`
- 加速度 misc：`/dev/mma8452_daemon`
- 陀螺仪 misc：`/dev/gyrosensor`

5 分钟内连续取得 520 条加速度和陀螺仪样本，没有权限错误或读取错误，说明之前的普通用户设备权限修复已经生效。

### 4.2 新 LZ30EMA 电机控制器可用

项目成功打开 `/dev/ttyS0`，能够下发 DRIVE、TURN、TURN_ZERO 和 STOP 类命令。测试中没有出现串口打开失败、Modbus 异常或电机控制线程崩溃，说明用 `/home/topeet/lianzhan` 替换原电机控制器后的基本通信链路已经工作。

### 4.3 视觉算法链路稳定

- GStreamer 成功打开 `/dev/video1`。
- YOLO RKNN 检测模型和 OSNet ReID RKNN 模型均成功加载。
- DeepSORT/IdentityBank 正常产生 track 和 ReID 状态。
- 运行到约 2092 帧，没有摄像头掉线、推理崩溃或 Python 异常。

## 5. 发现的问题

### P0：毫米波无数据时车辆仍会前进，存在安全风险

当前配置为：

```ini
[mmwave]
port = /dev/ttyS3
baudrate = 9600

[distance]
source = vision_mmwave
fallback_forward_percent = 35

[motion]
min_forward_percent = 55
```

雷达模块虽然打印了“初始化成功”，但这只说明串口能够打开。整个 5 分钟测试中毫米波有效帧为 0，控制日志持续显示：

```text
detail=no_radar_targets
```

更严重的是，在目标首次确认后，系统仍执行了：

```text
reason=distance_missing_front_clear speed=55
LZ30EMA command DRIVE left=33RPM right=-33RPM
```

配置注释将 `fallback_forward_percent=35` 描述为距离缺失时的低速回退，但它最终被 `min_forward_percent=55` 抬升到 55%，因此实际行为与配置意图不一致。

此前 AT2410 曾在 USB 四串口的 `/dev/ttyACM1` 观察到 `5A 0D 0A ...` 帧，而近期对 `/dev/ttyS3` 和 `/dev/ttyACM1` 的原始监听均为 0 字节。需要重新核对雷达供电、接线、持续输出模式以及实际设备节点，不能仅依据“串口打开成功”判断雷达可用。

建议：

1. 在毫米波链路修复前，将“距离缺失”改为停车，而不是继续前进。
2. 确认 AT2410 实际节点，并先用原始十六进制监听确认持续收到完整帧，再修改 `port`。
3. 修正速度计算逻辑，使 fallback 不会被普通前进最低速度强制抬升；最好为 fallback 单独设置上限。
4. 增加雷达数据超时健康状态，连续无有效帧时进入故障停车并明确报警。

### P1：丢失目标后不能重新锁定

系统锁定目标 `reid_uid=2` 后很快丢失，并在等待 15 秒后进入 `search_right`。之后车辆几乎一直原地向右搜索：1433 次控制决策中有 1408 次为 `search_right`。

在约 137 至 226 秒之后，摄像头中已经多次出现稳定人体，例如测试末尾：

```text
track_id=79 reid_uid=0 state=STABLE
skipped_unconfirmed_reid=1
candidate=none
persons=0
reason=search_right
```

当前配置为：

```ini
release_target_on_lost = false
```

控制层一直保留旧目标 `uid=2`，同时严格排除没有确认到原 ReID 的新人体，因此“视觉看见人”和“控制层认为有人”出现分离。现场表现就是人已经重新走到车前，车辆仍持续向右旋转，无法恢复跟随。

建议增加一个受控的重新捕获流程：旧目标超时后，先停车；对画面中心、连续多帧稳定且 ReID 分数达到阈值的人重新确认。不要简单地立即跟随任意新目标，否则多人场景容易跟错人。

### P1：启动脚本没有可靠传递退出信号

5 分钟测试使用 `timeout --signal=INT --kill-after=20s 300s` 结束。测试日志最后仍然是 TURN 命令，没有出现正常退出时应有的“人员跟踪已停止”和最终 LZ30EMA STOP，外层返回 `124`。

后续直接向 Python 进程发送 `SIGINT`，进程仍未退出；发送 `SIGTERM` 后才退出。这说明退出路径不能依赖当前的外层 shell/tee 结构和 Ctrl+C 传播。

风险是程序被超时、服务管理器或脚本停止时，Python 的 `finally` 可能没有执行，最后一个电机目标值可能继续有效。

建议：

1. 在启动脚本中保存 Python PID，并用 `trap` 将 `SIGINT`/`SIGTERM` 转发给 Python。
2. 或调整 shell 结构，使主执行进程能够正确接收信号，同时保留日志功能。
3. Python 同时处理 `SIGINT` 和 `SIGTERM`，统一进入 `finally`，先发送双轮零速和 STOP，再关闭串口与摄像头。
4. 增加硬件或驱动侧命令看门狗，超过一定时间未刷新命令时自动停车。

### P2：搜索阶段电机命令与日志刷新频率过高

当前配置：

```ini
rotate_duration = 1.2
rotate_pulse_brake_enable = true
rotate_pulse_pause_sec = 0.01
target_min_interval_sec = 0.05
```

持续搜索时，TURN 约每 58 ms 下发一次，约 17 次/秒；每 1.2 秒还有一次 TURN_ZERO。5 分钟共发送 3204 次 TURN，日志增长到约 5.3 MB。功能上符合当前配置，但会增加 RS485 总线压力、CPU 占用和日志写入量。

建议降低相同转向目标的刷新频率，仅在目标值变化或保活周期到达时发送；将逐帧调试日志降为 DEBUG，并为日志增加轮转上限。

### P3：RKNN 静态模型警告

两个 RKNN 模型启动时均出现：

```text
Query dynamic range failed
RKNN_ERR_MODEL_INVALID
```

运行库随后明确提示模型为 static shape，该警告可以忽略；本次推理持续正常。因此这是低优先级日志噪声，不是当前功能故障。

## 6. 建议修复顺序与复测标准

1. 先修复毫米波设备节点/数据接收，并实现“距离无效立即停车”的失效保护。
2. 再修复目标丢失后的安全重新捕获策略。
3. 修复启动脚本和 Python 的信号处理，验证 `SIGINT`、`SIGTERM` 都会留下明确 STOP 日志。
4. 降低搜索时的重复电机命令和日志频率。
5. 最后进行 30 分钟实车测试，覆盖接近、远离、横向走动、短暂遮挡、完全离开和重新出现。

复测至少应满足：毫米波持续有有效帧；拔掉或停止雷达后小车立即停车；目标短暂丢失后能恢复同一人；不能确认身份时停车而不是无限旋转；任何正常停止方式都能可靠发送双轮零速和 STOP；30 分钟内无设备掉线、异常退出或持续内存增长。

## 7. 日志位置

- 本次完整 5 分钟原始日志：`/tmp/codex-follow-5min-20260812-120859.log`
- 返回码：`/tmp/codex-follow-5min-20260812-120859.rc`
- 项目滚动日志：`/home/topeet/Desktop/rk_car_runtime_module/run_request_0428_modular_logs/request_0513_modular.log`

注意：`/tmp/codex-follow-5min-20260812-120546.log` 是第一次因 SSH 输出管道断开而提前阻塞的无效测试，不计入上述 5 分钟统计。
