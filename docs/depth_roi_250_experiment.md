# 250ms框窗口与200RPM前进上限

## 目的与边界

用户本轮同时要求延长已确认目标的框窗口、将正常前进上限提高到200RPM。
这是两个变量，不再是单因素实验。分别考察深度接续和速度限幅，不能把全部改善
归因于250ms。不改ReID/搜索判据、PID增益、前馈40RPM预算、50ms执行器或停车能力。
独立深度跟踪仍未在线接入；本轮也没有修前馈有/无切换的偏置突降。

## 已实现

- `astra_depth.longitudinal_bbox_max_age_sec=0.25`：0~180ms沿用原路径；
  180~250ms必须可信编码器反馈<=150ms，原始和滤波角速度绝对值均<=5deg/s。
  持续跟随的同UID、原始检测框与搜索/危险/质量等现有门控仍保留。
  主视觉最新Depth选择和最终测距入口复核；没有反馈或转动较快不获延期。
- `longitudinal_sample_max_age_sec=0.18`：独立的物理Depth运动期限，按采集时间算。
  提交、重复样本复用、剩余预算、动作读取、最终电机写入均不能借ROI参数续期。
  环境变量同名映射最大钳到180ms；旧样本不变新样本。
- 正常INI的`motion.forward_max_rpm`与`motor.forward_max_target_rpm`均为200。
  `longitudinal_far_forward_percent=100`，远处最高200RPM，但只执行PID实际请求。
  近距`longitudinal_max_forward_percent=10`仍为20RPM；`min_forward_percent=20`
  保持兼容起步下限40RPM的物理含义（可信距离PID仍可低于此下限）。
  底层`rs485_max_target`原本已是200；每轮不超过200，接近上限时保留差速并降低基速。
  倒车独立60RPM上限保留，额外限制不允许错误远距倒车动作借前进上限放大。
  搜索原地转向的固定RPM、rotation-only配置、近距离停车和恢复限速参数不变。

旧框延长不等于定位预测：当前仍用原检测ROI，低角速度不是没有人体横移。
保留人体深度簇/背景/跳变保护，需实车检查额外70ms是否采到正确人体。
现有速度前馈中的旋转补偿不是采样框重定位，其原180ms适用范围没有放宽。
所以新Depth增加也不保证前馈同幅增加；超过100RPM时原有速度估计/恢复证据
边界也未全局放宽。200只是最大许可，不是已验证的跟随速度或制动能力。

## 日志与指标

启动：`depth_clock_config roi_max_ms=250 ... physical_depth_ttl_ms=180 ... forward_max_rpm=200`。
测距延期：`depth_roi_extension capture_frame_id uid age_ms status`。
`extended`仅表示允许一次采样；仍可能是重复帧或被后续深度保护拒绝，不能当成成功授权。
限幅：`depth_linear_limit`新增`forward_scale_rpm`、`approved_forward_rpm`，避免把
200RPM比例下的30%误解为30RPM。后续比较用RPM和实际轮速，不跨比例比较百分数。

`tools/follow_metrics.py`新增：

- `depth.roi_extension_events`：尝试状态计数，不是唯一物理帧数。
- `depth.positive_authorized_samples/hz/gap_ms/gaps_over_180ms`：按新鲜正向限幅
  记录去重的物理样本统计；之后仍可能被安全撤销，不能等同于电机实际执行。
- `control.approved_forward_rpm`：旧日志未记录时未知，不把旧百分数冒充RPM。

CAP837~984旧基线（175849运行）：采集7.945s，电机窗口加180ms；距离扩大0.231m/s，
零指令22.52%，加权命令41.30RPM。新鲜正向样本83个，10.54Hz，>180ms间隙9次，
最长457ms。原`depth.effective_hz=10.91`含测出但已过期的样本，保留字段用于兼容。

## 验证与回退

新增26项回归通过；完整`python3 -m pytest -q tests --tb=short`为1562通过、2项原有
`test_search_direction_switch.py`失败（候选搜索入口与中心线行为，本轮未修改）。
RK3588配置检查、修改模块编译及`git diff --check`通过。

软件模拟驱动不连接串口。检查旧Depth在200ms仍拒绝、220ms框低yaw能取新Depth、
高yaw/失效反馈拒绝延期、近距20RPM、正向上限200/差速单轮封顶、倒车与搜索RPM不翻倍。
实车尚未运行。控制器支持200RPM不代表该速度下能在1.5m范围安全制动；先验证低速
停止和急停，不应直接以200RPM接近人或障碍。测试应有足够空间、人工急停和低风险路线。

要隔离ROI效果，可保持200RPM配置不动，仅在相同低速路线对比ROI=0.18/0.25。
观察因过期清零时间、正向样本间隙、距离斜率、错距和近距离过冲，不能仅看extended次数。
ROI回退只需改`longitudinal_bbox_max_age_sec=0.18`。
整体退回旧速度比例需要一起恢复：`motion.forward_max_rpm=100`、
`motor.forward_max_target_rpm=100`、`motion.min_forward_percent=40`、
`astra_depth.longitudinal_max_forward_percent=20`、`longitudinal_far_forward_percent=60`。
物理Depth仍为180ms，不能随上述回退改变。
