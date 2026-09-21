# 身份模板分层与老化（2026-09-21）

## 问题与范围

CAP874 命中 CAP231（34.74秒前，d=0.1719），CAP877 命中 CAP285
（31.93秒前，d=0.1649）。这些模板本身是正确目标；不能将年龄视为错误身份的证明。
此前全身库按多样性保留20条模板，没有时间淘汰，重复的新近可靠视角可能不入库。
本次只修改身份模板及其接回资格，不修改 YOLO/ReID 模型、全局阈值、PID、轮速、
深度授权或搜索转向规则。既有几何矛盾、竞争门控、模板隔离仍优先。

## 实现

- `template_memory.py` 保存独立近期全身/躯干样本，各最多8条，窗口30秒。
  只从原有已批准的模板更新入口学习；相似新样本替换近期重复项，保存新特征和真实
  CAP/采集时间，不给旧特征伪造时间。不同姿态在容量内保留，满后淘汰最早近期样本。
- 全身长期库最多6条（包括初始锚点），继续采用原有多样性替换；非锚点超过120秒
  淘汰。弱库过期项也淘汰；躯干档案保留独立锚点，不与全身向量混合。
- 使用真实采集时间，不用帧数推算年龄；查询只推进淘汰水位，不续期、不复活旧样本。
  重复、乱序、无采集时间、低质量、观察级候选不能写近期库。
- 全身匹配允许比较近期及长期样本；接回（包括普通新轨迹、搜索、接回隔离阶段）
  额外要求近期支持。全身使用现有软观察上限0.30；显式局部人体可由近期躯干支持，
  仍使用现有局部阈值。这只是必要条件，不能绕过原有严格阈值、连续帧、几何及竞争条件。
- 仅长期匹配输出 UID0、`identity_control_rejected=true`。不能通过mapped_uid、
  临时几何兜底恢复跟随，不能积累能自行变成近期模板的观察链。
- 已持续验证的正常绑定目标仍可按原有更新规则学习近期外观，不因单个旧模板过期中断。
  接回隔离期间禁止模板学习；几何冲突不因模板淘汰或近期匹配而清除。

## 明确限制

30秒内没有任何可安全写入的近期样本后，长期库只用于保留身份参考和记录匹配，
不能独自完成自动接回；此时需重新初始化身份。不要靠被拒候选续期解锁。
这项时间值是试验配置，不是已通过实车验证的最优值。近期样本也可能发生模型误匹配，
因此本功能不能保证消除全部相似人物误认，尤其不能替代几何矛盾保护。

## 配置与回退

默认运行配置 `car_control_modular/config/reid_runtime.ini` 的 `[identity_bank]`：

```ini
template_memory_enable = true
template_recent_sec = 30.0
template_archive_sec = 120.0
```

经 config_loader → RKNNVisionConfig → DeepSortTrackerConfig → IdentityBankConfig 接入。
独立直接构造配置的默认行为保持旧模式。设 `template_memory_enable=false` 并重启可回退
本次分层策略，不回退此前的几何矛盾/接回复核修复。运行中不改配置、不操作电机。

## 日志与下轮验证

- `template_recent_evidence` / `template_recent_partial_evidence`：数量、距离、命中CAP、年龄。
- `template_recent_supported` / `template_recent_required`：是否有近期支持、是否必须检查。
- `matched_template_age_sec`、命中 metadata 的 `template_role`：近期或长期来源。
- `recent_bank_updated`：真正更新近期库，避免与长期重复样本被跳过混淆。
- `archive_only_reacquire_observe`、`recent_template_mismatch`、`recent_template_unavailable`：
  无近期支持的明确原因；`template_archive_pruned`：实际淘汰数量。

下轮必须同时标注正确/错误人物，统计：

1. 明确错误人物获得活动UID次数（目标0）；已确认几何矛盾后重新接回次数（目标0）。
2. 只命中长期模板却获得UID次数（目标0），被拒候选更新近期库次数（目标0）。
3. 正确目标接回成功率、从首次可见到确认耗时；分别统计侧身、躯干、短时遮挡，
   检查是否因近期库不足明显恶化。不能只报告误接回减少而隐藏误拒绝。
4. `recent_template_unavailable` 与 `recent_template_mismatch` 次数、对应人物真实性，
   用于判断30秒/8条是否合适。没有标注不能将所有拒绝都算作改善。
5. 身份处理耗时P95及原控制周期，确认无新增相机/串口访问或磁盘图片写入。

## 测试说明

新增 `tests/vision/test_template_memory.py` 28项，覆盖时间淘汰、容量、重复/乱序帧、
质量条件、配置传递、隔离、正常接回、局部支持、非preferred旁路和CAP874几何矛盾。
与此前CAP874及控制兜底回归共49项。使用合成向量/历史框验证决策，
不代表重新执行了历史模型推理，也不等同实车效果验证。

全套回归3184项通过，3项原有失败：两项搜索方向测试，以及
`test_config_wires_profile_with_shared_grant_ttl_and_unchanged_target`（当前配置1.4m，
测试仍期望1.5m）。本轮没有调整这些无关参数或测试。编译及diff检查通过。
