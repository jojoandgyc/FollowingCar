# CAP338 身份错接修复（2026-09-18）

## 范围与原因

以 `run_20260918_130838_10995_f522445b` 的 CAP334～346 为依据。
CAP338 的 raw track 1 从中央正确人体跳到左侧小人体；代码已检测到
`center_jump,area_change`，但普通同轨迹路径仍输出 UID1。CAP341 又使用
该错误绑定作为见证人，将中央真正目标 raw track 3 标记为
`co_visible_distinct_person`，即使 ReID 距离为 0.0506/0.0475 仍无法接回。

这次只修改 DeepSORT 关联与身份资格传递，不修改 ReID 阈值、旋转参数、
纵向 PID、深度期限、搜索方向控制或电机接口；没有执行实车动作。

## 实现

1. `kalman_filter.gating_distance` 使用残差与 `S^-1 * 残差` 的内积，
   不再把完整求解结果再次平方。共用求解器和 Kalman 更新语义保持不变。
2. `DeepSort.update(match_validator=...)` 将纯 pair 校验传至外观与 IoU
   两条匹配路径，在分配之前屏蔽不合理关联。不会让错误最佳候选先占据
   匹配，再事后拒绝；被拒绝 pair 不更新原轨迹的 Kalman 状态和特征库。
   检测本身仍可作为独立新轨迹存在。
3. `review_mapped_geometry` 使用原始检测框和最后可信锚点，包括已有车体
   转动补偿。位置和面积同时矛盾时，撤销普通 raw-ID 对 UID 的占用并返回
   UID0；单纯面积变化、正常裁切仍保留原处理路径。
4. 已记录的严重矛盾不会因锚点超过原15帧窗口自动洗白。该 raw ID 缺少
   几何证据时仍不可信；与可信参考重新连续后，才能进入既有身份确认路径。
5. 整帧先复核所有旧映射，再生成排除证据。见证人必须通过几何检查，且
   不能是同帧 ReID 竞争失败者或存在多重 UID 主张者。
6. `invalidate_witness(uid, track_id, after_frame=...)` 只撤销该见证人在最后
   可信帧之后生成的负证据，覆盖证据跨轨迹转移的情况；保留更早真实共现
   排除以及其他见证人、其他 UID 的记录。
7. 错误占用被撤销后，唯一占优、几何连续的强匹配候选可两帧接续，无须
   再等待旧占用的5帧窗口。竞争失败不能回落到普通低距离单帧确认。
   完成接回后，旧错误轨迹再次出现也不能重启已结束的恢复事件。

## 回归与限制

新增58项无硬件测试：

- `tests/vision/test_kalman_gating.py`：10项；解析协方差、位置限定、正常更新、
  CAP338固定重建夹具和实际代价矩阵门控。错误框门控约40.125，大于9.4877。
- `tests/vision/test_deepsort_match_guard.py`：15项；外观/IoU、特征库隔离、
  合法替代候选、NMS后原检测索引及默认兼容。
- `tests/vision/test_exclusion_witness_revoke.py`：11项；定向撤销、证据转移、
  可信边界以前的证据保留以及来源记录。
- `tests/vision/test_mapped_identity_geometry.py`：22项；错误同轨迹UID拒绝、
  中央目标两帧接回、弱/裁切与转动补偿、竞争失败和重复旧冲突。

CAP几何测试使用实际检测框及合成指定余弦距离的向量，不是重新运行YOLO/
OSNet的录像推理，也不证明实车误认率已经下降。控制授权与视觉时效检查
保持原状；身份接回不等于自动绕过深度/危险保护。

最终完整 pytest：2835项通过，仍有此前的两项 `test_search_direction_switch.py` 失败，未借本次
身份修复改写它们。另直接运行旧 `test_identity_bank.py` 独立 main，会遇到
已有模板隔离与旧 `last_frame` 断言不一致；单独调用其中反方向候选 helper
也有旧原因码断言失败（实际 UID0、`preferred_search_reacquire_wait`，旧断言
要求 `preferred_search_reacquire_rejected`，并非误输出可信UID）。禁用本次
新增 review 后均仍复现。该文件不被 pytest
按 main 自动运行，不能将全套 pytest 通过数理解为这些旧断言已通过。

## 下一轮日志核对

- `deepsort_identity_pair_rejected`：原始检测索引、补偿后位移、面积相似度、
  参考 CAP；只是某个 pair 被拒绝，不代表整条原轨迹失效。
- `mapped_identity_geometry_reject`：实际映射撤销、UID、参考 CAP、撤销负证据数。
- 新排除记录的 `witness_geometry_valid`、`witness_reference_frame_index`、
  `witness_reference_capture_frame_id`。
- 检查多人经过时 UID 是否始终属于原人；不能出现严重几何拒绝仍授权跟随，
  或错误见证人排除中央正确人的情况。
- 比较原目标保持率、误切换次数、重新确认耗时与视觉处理P95延迟；同时
  验证转弯、短遮挡、近距离裁切没有明显增加掉轨，不靠提高速度或放宽
  ReID 阈值掩盖关联问题。
