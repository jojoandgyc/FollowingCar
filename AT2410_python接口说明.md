# AT2410 Python 接口说明

已根据以下文档整理出可直接复用的 Python 协议模块：

- `24G多目标风扇跟随应用协议文档(2).pdf`
- `MS24-4415J12M4-LDO-B-4J-NLS-1T2R-S7136H产品手册v2.0_20250707(1).pdf`

对应代码文件：

- `E:\Download\ATObjectDetection_v0.2.1(AT2410)\at2410_protocol.py`

## 当前已封装内容

1. UART 参数
   - 开发板当前波特率：`9600`
   - 数据位：`8`
   - 停止位：`1`
   - 校验位：`None`

2. 协议能力
   - 构造通用串口帧 `build_frame(...)`
   - 校验帧 `verify_checksum(...)`
   - 解析主动上报的多目标信息帧 `parse_frame(...)`
   - 流式拼包 `AT2410StreamParser`
   - 串口读帧客户端 `AT2410Radar`

3. 目标字段
   - `distance_cm`
   - `angle_deg`
   - `speed_cm_s`
   - `target_id`

## 直接使用

```python
from at2410_protocol import AT2410Radar

with AT2410Radar("COM3") as radar:
    frame = radar.read_frame()
    for target in frame.targets:
        print(target.target_id, target.distance_cm, target.angle_deg, target.speed_cm_s)
```

## 如果你已有串口框架

只想要解析函数时，可以直接：

```python
from at2410_protocol import AT2410StreamParser

parser = AT2410StreamParser()

while True:
    chunk = ser.read(ser.in_waiting or 1)
    for frame in parser.feed(chunk):
        print(frame)
```

## 文档里已确认的协议细节

- 帧头：`0x5A`
- 命令：多目标主动上报为 `0x0A`
- `LEN` 为“命令字 + 数据区”的总长度
- 校验为前面所有字节求和后取低 `8` 位
- 每个目标占 `8` 字节
- 最多 `3` 个目标
- 1 个目标固定前缀：`5A 0D 0A 01`，完整帧长 `16` 字节
- 2 个目标固定前缀：`5A 15 0A 02`，完整帧长 `24` 字节
- 3 个目标固定前缀：`5A 1D 0A 03`，完整帧长 `32` 字节
- 串口可能出现半包、粘包或其他设备的噪声；流解析器只按上述三种包头同步，并在校验失败后继续搜索下一帧

## 开发板持续监听

默认一直运行，按 `Ctrl+C` 结束：

```bash
python3 /home/topeet/devtest/golfcatdevtest/at2410/at2410_uart_test.py \
  --verbose \
  --raw
```

无参数时优先使用环境变量 `MMWAVE_AT2410_PORT`；未设置时使用 SIPEED UARTx4 的固定 `if00` 链接（对应当前 `/dev/ttyACM0`）和 `9600` 波特率。GPS 已确认连接在 `if02`（当前 `/dev/ttyACM1`）。

跟随车项目也使用同一个环境变量。INI 中的 `port` 只是默认值；启动前可统一覆盖：

```bash
export MMWAVE_AT2410_PORT=/dev/serial/by-id/usb-SIPEED_UARTx4_HS_FactoryAIOT_Prog_Serial-if00
```

先离线验证协议解析器：

```bash
python3 /home/topeet/devtest/golfcatdevtest/at2410/at2410_uart_test.py --self-test
```

脚本解析每帧后会输出实际目标数量，并逐个打印目标 ID、距离、角度和径向速度。无串口字节、收到非 AT2410 数据、校验失败和成功解析目标会显示为不同状态。

## 当前保守处理

这次只把文档里能明确确认的“主动上报多目标数据”做成稳定接口。  
如果你后面还有完整的“参数配置/下发指令”表，我可以继续把设置距离、灵敏度、开关日志之类的命令也一起补成完整 SDK。
