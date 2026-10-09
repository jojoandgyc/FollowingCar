# LZ-30EMA_2EC_N RS485 Python 库

这个库根据工作区中的两份 PDF 编写，覆盖驱动器 RS485 Modbus-RTU 协议中的 0x03、0x06、0x10 三类命令，并内置寄存器表、CRC 校验、异常响应解析和常用电机控制 API。

## 快速使用

不接硬件时可以直接运行测试：

```bash
# Linux
bash scripts/run_tests_linux.sh
```

```powershell
# Windows PowerShell
.\scripts\run_tests_windows.ps1
```

接 USB-485 转换器时安装可选串口依赖：

```bash
python -m pip install -e ".[serial]"
```

```python
from lz30ema_rs485 import LZ30EMAClient, StopMode, format_realtime_snapshot

client = LZ30EMAClient.from_serial("COM3", slave=1, baudrate=9600)
client.set_right_speed(1000)
client.set_left_speed(-500)
client.stop("right", StopMode.NORMAL)
status = client.read_motor_status("right")
print(status.speed_rpm, status.phase_current_a)

snapshot = client.read_realtime_snapshot()
print(format_realtime_snapshot(snapshot))
```

Linux 串口通常类似 `/dev/ttyUSB0`：

```python
client = LZ30EMAClient.from_serial("/dev/ttyUSB0", slave=1)
```

也可以直接用命令行打印实时信息：

```bash
python -m lz30ema_rs485.cli --port /dev/ttyUSB0 --slave 1
```

```powershell
python -m lz30ema_rs485.cli --port COM3 --slave 1
```

持续打印左右电机速度：

```bat
scripts\watch_speed_windows.bat --port COM7 --interval 0.5
```

或者用模块入口：

```bat
python -m lz30ema_rs485.cli --port COM7 --watch --speed-only --interval 0.5
```

打开图形界面：

```bat
scripts\motor_gui_windows.bat --port COM7
```

开发板 Ubuntu 本机屏幕运行 GUI：

```bash
cd /home/topeet/lianzhan
./scripts/motor_gui_board_linux.sh
```

板端启动器默认使用 `/dev/ttyS0`、`115200`、地址 `1`，并自动连接 XFCE 的 `:0` 桌面。启动前需要先退出跟随主程序、键盘遥控或其他占用 RS485 的进程，避免多个程序同时控制电机。

键盘控制小车：

```bash
# 开发板 Linux 本机键盘使用。默认读取 /dev/input/event*，能获得真正按下/松开事件。
cd /home/topeet/lianzhan
./scripts/keyboard_drive_auto_115200_linux.sh
```

```bash
# 也可以明确指定当前实测能通讯的 ttyS0。
cd /home/topeet/lianzhan
./scripts/keyboard_drive_ttys0_115200_linux.sh
```

```powershell
# Windows，有窗口焦点时可直接捕获按下/松开。
.\scripts\keyboard_drive_windows.ps1 --port COM7 --baudrate 9600 --slave 1 --input gui
```

默认键位：`W` 前进，`S` 后退，`A` 左转，`D` 右转，`1` 速度 +10RPM，`2` 速度 -10RPM，`Q/Esc` 退出。基础速度关系为：前进 `left=50,right=-50`，后退 `left=-50,right=50`，左转 `left=50,right=50`，右转 `left=-50,right=-50`。当前开发板启动脚本已默认加 `--swap-turns` 修正实车 A/D 左右相反的问题；如需临时取消，可在命令后加 `--no-swap-turns`。如果实际接线方向还需要调整，可以加 `--swap-sides`、`--invert-left`、`--invert-right`。

开发板本机键盘推荐运行：

```bash
cd /home/topeet/lianzhan
./scripts/keyboard_drive_auto_115200_linux.sh
```

这个脚本会先只读扫描 `/dev/ttyS0,/dev/ttyS3`，自动选择能通讯的端口，然后使用 event-only 键盘模式。如果当前用户没有 `/dev/input/event*` 权限，会自动提示并切到 `sudo`。event-only 能收到真实按下/松开，响应最快。

如果这个脚本完全没有按键反应，先运行调试脚本：

```bash
cd /home/topeet/lianzhan
./scripts/keyboard_drive_debug_keys_115200_linux.sh
```

按 `W/A/S/D` 后应该看到 `KEY dev=... code=... value=... mapped=...`。只有在不得不用普通终端输入时，才运行 `keyboard_drive_auto_115200_mixed_linux.sh` 或 `keyboard_drive_ttys0_115200_terminal_linux.sh`；这类模式收不到真正松手事件，会有推断延迟。

如果通过 SSH 终端控制，程序无法收到真正的松手事件，只能根据键盘重复输入推断，松手会有一点延迟。要让 Windows 键盘松手立即传到开发板，推荐使用远程键盘模式：

```bash
# 开发板上先启动 RS485 控制服务
cd /home/topeet/lianzhan
./scripts/drive_server_ttys3_115200_linux.sh
```

```bat
:: Windows 上再启动键盘客户端
F:
cd F:\lianzhankzq
scripts\keyboard_drive_board_windows.bat
```

GUI 顶部的“波特率”是电脑当前串口连接速率；参数区的“控制器485波特率”会写入驱动器 `0x00C9`。写入控制器波特率后，如果通讯断开，请在顶部选择同样的新波特率并点击“重连”，必要时给驱动器重新上电。

查看截图字段哪些能从协议读取：

```bash
python -m lz30ema_rs485.cli --coverage
```

## 覆盖范围

- Modbus RTU 帧：读保持寄存器 0x03、写单寄存器 0x06、写多寄存器 0x10。
- 校验：标准 Modbus CRC16，低字节在前。
- 响应解析：正常响应、CRC 错误、功能码不匹配、异常码。
- 寄存器：设备信息、实时状态、电机控制、电机参数、系统参数、应用参数、通信参数、PID 参数。
- 高层命令：读设备信息/状态、设置左右电机速度、差速速度、停止、抱闸、恢复出厂、相序学习、系统模式、RS485/TTL/CAN 通讯参数和 PID 系数。

## 说明

配置类参数在说明书中提示：用 0x10 写多寄存器时断电保存，用 0x06 写单寄存器断电后会恢复。库里的 `write_register(..., persist=True)` 会用 0x10 写单个寄存器，常用配置方法也提供 `persist` 参数。

PDF 第 14 页右电机 1000RPM 启动示例中，帧 `01 10 00 41 00 02 04 00 00 03 E8` 后印的是 `D7 67`；按标准 Modbus CRC16 和说明书附录算法计算应为 `36 ED`。本库按标准 CRC 实现，同页左电机 -500RPM 示例 `77 E1` 与本库计算一致。
