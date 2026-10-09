# 普通停车改用 0A EMERGENCY

按用户要求将主配置 `motor.stop_mode` 改为 `emergency`，驻车电流仍为 0A。
近距离停车、搜索接回、反打结束及轮速交接停车使用配置选择的实际 STOP。
本轮直接发送一次双轮 EMERGENCY，不再执行前置 EMERGENCY 后接 NORMAL，
也不为 EMERGENCY 增加零速写入或固定等待。旧 NORMAL 后端保留兼容性，
但本轮主程序普通停车不选择它。

500ms 从实际 EMERGENCY 下发完成起算，随后保留 FREE 退出及新证据检查。
恢复资格按停车来源区分：普通停车即便使用 EMERGENCY 也可走原有安全恢复；
危险、显式停止、关机、通信故障不能被普通停车的定时释放覆盖。
普通保持刷新不重复急停，不会在 FREE 后重新进入 NORMAL 或 EMERGENCY。
已删除的 2 秒停稳超时锁存不恢复。

停稳时间检查的一些历史字段仍使用 normal 名称，它们不是电机模式命令。
实际模式以 `near_yaw_park_applied`、`search_reacquire_brake_applied` 和
后端停车命令日志的 `mode` / `stop_value` 为准，本轮应为 emergency / 1。

无硬件回归覆盖近距离/搜索停车的命令顺序、500ms 退出、旧刷新拦截、
普通 EMERGENCY 搜索停稳后新深度恢复前进、反打结束，以及危险停车优先级。
本轮未操作实车。重新启动跟随程序后生效，不能据此保证实际停车时间缩短。

验证：电机测试 793 passed；全量 4638 passed、3 项既有失败
（目标距离配置断言、两项搜索方向测试）；`git diff --check` 通过。
