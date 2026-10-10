# 最小跟随 ReID v2

该实现是独立于电机控制的身份链路。电机控制线程只消费已确认的目标框，不等待 OSNet 推理；OSNet 由一个最新优先的后台工作线程执行。因此特征提取变慢时，控制环不会排队积压。

## 数据流

```text
YOLO 人体框
  -> ByteTrack（高分匹配，再用低分框续轨）
  -> ReID 身份状态机
       INIT / ENROLLING：锁定首条轨迹，建立固定目标模板库
       LOCKED：只接收该 ByteTrack ID 的框
       SEARCHING：轮询画面中的人，异步 OSNet + 多模板余弦确认
  -> 已确认框 -> 深度测距 -> 跟随控制
```

模板库以 `front / left / right / back` 四个槽位保存全身和躯干 OSNet 向量。样本只能在初始采集期写入；达到请求的视角数后冻结。去重只在同一视角槽内进行，所以侧面和背面不会因为与正面相似而被错误丢弃。多人重叠、轨迹不连续或低质量裁剪不会写入模板库。

`LOCKED` 状态不会按面积或位置切换到另一人：当前轨迹 ID 消失时，系统会先进入 `SEARCHING`，只有 ReID 连续命中后才将新轨迹 ID 重新绑定为目标。

## 运行时证据与性能

每次 OSNet 请求会在运行日志目录的 `reid_v2/` 写出一张裁剪图和同名 JSON。JSON 包含 `track_id`、全身/躯干向量、裁剪框、质量和推理耗时。主日志中的 `minimal_timing` 同时记录 `reid_target_track_id`、模板数量、各视角数量和后台推理耗时。

当前模型为 OSNet 全身特征加躯干特征降级。KPR 尚未接入运行路径：仓库中没有可部署的 KPR 权重和 RKNN/ONNX 验证结果；在具备该模型前，不会把它误标为已启用。

## 关键配置

- `MINIMAL_BYTETRACK_HIGH_CONFIDENCE` / `MINIMAL_BYTETRACK_LOW_CONFIDENCE`：高、低分检测阈值。
- `MINIMAL_BYTETRACK_MATCH_IOU`：轨迹与检测框的 IoU 匹配阈值。
- `MINIMAL_BYTETRACK_MAX_LOST_FRAMES`：轨迹保留的最大丢帧数。
- `MINIMAL_REID_BOOTSTRAP_REQUIRED_VIEWS`：采集到几个独立视角后冻结模板库。
- `MINIMAL_REID_VIEW_CHANGE_THRESHOLD`：和当前视角特征相似度低于该值时，进入下一个视角槽。
- `MINIMAL_REID_ENROLL_DUPLICATE_SIMILARITY`：同一视角内的近重复样本阈值。
- `MINIMAL_REID_FULL_THRESHOLD` / `MINIMAL_REID_TORSO_THRESHOLD`：重识别命中阈值。
- `MINIMAL_REID_CONFIRM_HITS` / `MINIMAL_REID_CONFIRM_WINDOW`：重识别连续确认要求。
- `MINIMAL_REID_BACKEND`：独立指定 ReID 后端。随附 `.rknn` 模型应使用 `auto`/RKNN；只有替换为 ONNX OSNet 后才可设为 `onnxruntime`。
