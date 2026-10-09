# 普通驻车退出：0A 确认后下发一次 FREE STOP

按用户本轮要求，把普通驻车退出改为：

1. 原有 5A NORMAL 驻车，保持至少 500ms（从实际 NORMAL 写完计时）。
2. 清左右轮驻车电流至 0A，并分别读回确认。
3. 再核对停车所有权和危险状态，向双轮发送一次 FREE STOP。
   后端映射为停止寄存器 `0x0040`、`0x0044` 写 `2`，不是速度目标 0RPM。
4. FREE STOP 实际完成后开始收集新停稳反馈与新图像。继续保留软件运动锁，
   不恢复旧速度指令，不把“FREE 已发出”视作物理停稳。

实现共用于普通原地驻车和搜索接回驻车。周期服务、普通驻车刷新及旧零速
调用不会重复 FREE 或补写速度，也不会重新设置 5A。新的有效运动通过原有
授权检查后才接续速度输出；下一次真正驻车仍恢复 5A。

FREE STOP 的 `preserve_zero=True` 仅抑制后续零速写入；不增加运动资格。
内部历史变量 `normal_zero_hold` 也用于这项 FREE 保持，不表示硬件仍处于 NORMAL。
普通停车请求的所有权标记继续存在，由 `current_released_at` 区分退出完成阶段；
它现在记录 0A 确认及 FREE STOP 都完成后的时间，不提前接受切换期间的反馈。

## 异常与边界

- 清电流失败或读回不一致：不发送 FREE，不放行运动，沿用故障保护。
- FREE STOP 写入失败（包括一轮成功、另一轮确认失败）：记录 `free_stop_failed`，
  不建立完成时间、不恢复运动，沿用紧急停车与故障锁存。
- 电流 I/O 或 FREE I/O 期间出现危险/显式停止/所有权改变：安全停车抢占，
  FREE 不会在后续周期重放覆盖它。
- 危险急停、显式停止、500ms 参数，以及退出后 2 秒未停稳的故障保护未删除。
- NORMAL 之前原有的前置零速和运行中的零速过渡不在本次修改范围内。

新日志应出现：`标签=ordinary_park_release_free 模式=free stop_value=2`
和 `pre_zero=False post_zero=False final_motor_command=stop`；随后为
`ordinary_park_current_released ... zero_rpm=False speed_write=False ... exit_stop_mode=free`。

## 验证

电机测试 709 项通过，覆盖实际调用顺序、单次 FREE、零速度包抑制、新指令恢复、
重新 5A 驻车、FREE I/O 延迟后的证据边界、FREE 失败及危险抢占；均为假驱动测试。
`git diff --check` 通过。未操作实车，FREE 退出对实际晃动的影响仍需安全实测。

完整回归 `python3 -m pytest -q tests --tb=no`：4500 passed、3 failed。
失败仍是已有的目标距离 1.5m/1.4m 配置断言和两项搜索方向断言，本轮未修改。

本文更新上一版 `cap1583_current_only_release_20260923.md` 中“仅清电流，不追加
任何 STOP”的普通退出流程；仍然不追加 0RPM。
