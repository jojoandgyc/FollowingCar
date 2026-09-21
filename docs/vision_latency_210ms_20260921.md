# 210ms视觉结果入口与耗时诊断

## 范围

- 正常配置和仅旋转配置的 `vision.control_max_result_age_sec` 从0.19变为0.21。
- 不修改深度授权、转向续发、停车保护、身份门控和模型参数。
- 保留工作区原有修改（包括当前1.4m目标距离）。重新启动原脚本生效；显式环境变量仍可覆盖配置。

## 已知证据与边界

分析的是 `run_20260921_130904_10027_a0487d63`。CAP903的视觉入口耗时203.07ms：
YOLO53.04ms、ReID55.32ms、轨迹/身份93.89ms。相邻CAP901/907的轨迹阶段为25.67/21.49ms。
旧日志没有子阶段CPU/墙钟计时，不能断言93.89ms来自身份库、Kalman求解或调度等待中的哪一项。
无硬件合成轨迹的profile曾显示矩阵求解占大头，但不带profile的重复实验波动明显；没有据此替换求解器或改变追踪数学。

本轮335条视觉记录中：大于190ms有39条，大于210ms有25条。14条落在(190,210]，这是旧耗时反事实统计，不是实车改善结果。
入口result_age目前计量函数处理耗时，不等于完整曝光到执行延迟。

## 等价优化

ReID HSV颜色特征使用uint8整数分箱+bincount，替代三次通用histogram。
保留相同ROI、HSV转换、分箱边界、饱和度筛选及归一化；非uint8保留原路径。
没有缩图、抽样或改变特征维度。随机图、灰度分箱边界、全部色相/饱和度组合均与原实现逐元素相同。
同机、CAP903近似裁剪尺寸476×271的随机图离线测试，完整颜色特征中位耗时两轮分别约11.5→2.5ms、9.1→2.3ms。
这不是完整ReID或实车视觉总耗时的测试，不能用它宣称93ms问题已解决。

## 新日志

沿用每帧pipeline_timing，不新增高频独立日志行：

- `reid_color_ms`：颜色特征及融合耗时。
- `reid_postprocess_exclusive_ms`：旧postprocess减去包含在内的躯干推理；旧字段保留兼容。
- `tracker_stages_wall_cpu_ms`：每项为墙钟/当前线程CPU毫秒。
- `tracker_association`：DeepSORT封装整体（含输入整理、预测、关联与输出）。
- `deepsort_match/kalman_update/metric_update`：上述阶段内部的关联、状态更新和特征库更新。
- `tracker_geometry/competition/evidence/records/probe`：去重几何、候选竞争、帧证据、输出记录与UID分配、探针。
- `identity_match_evidence/decision/quarantine/logging`：身份分配内部的模板诊断、身份决策、隔离诊断和JSON日志输出；同帧多人累计。

这些是嵌套计时，不能把association、deepsort子阶段和identity子阶段全部相加。
墙钟减当前线程CPU并非纯锁等待时间，也可能包含调度、阻塞和原生库工作线程。
没有新增串口读取、运动控制锁、模型推理或图像写盘。

## 下轮对照

保持同配置路线，对比：视觉总耗时P50/P95、tracker P95、超过210ms比例、stale_vision_result撤销次数。
另外观察中心过冲、停车响应。CAP924～928的NORMAL驻车仍运动问题独立，不能用放宽视觉窗口替代驻车诊断。
