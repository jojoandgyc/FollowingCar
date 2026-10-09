# Kalman 耗时长尾：单线程默认试验与等价求解器对照

## 范围与默认行为

正常入口仍为 `./run_request_0428_modular.sh`。
脚本在任何 Python 调用前，默认将 OPENBLAS / OMP / MKL / BLIS 线程数设为 1。
专用选择器 `FOLLOW_NUMERIC_THREADS` 优先于父进程已有线程环境，避免父环境的 8 线程悄悄覆盖试验。
允许值：1、2、4、8、inherit；非法值在硬件初始化、日志轮换前退出。

首轮默认仍是原 NumPy 求解器。只将数值线程配置作为主要试验变量，不同时改变身份策略。
直接运行 Python 不会自动强制线程数；必须在导入 NumPy 前设置环境。

```sh
# 首轮：单线程，原求解器
./run_request_0428_modular.sh

# 固定 8 线程对照（其余配置保持不变）
FOLLOW_NUMERIC_THREADS=8 ./run_request_0428_modular.sh

# 保留父环境，恢复原启动线程行为
FOLLOW_NUMERIC_THREADS=inherit ./run_request_0428_modular.sh

# 第二阶段：已确定线程配置后，单独比较三角求解器
FOLLOW_NUMERIC_THREADS=1 FOLLOW_KALMAN_SOLVER=triangular ./run_request_0428_modular.sh
```

`triangular` 需要 SciPy；选择该模式时在 Kalman 初始化阶段导入，缺少依赖会明确报错，
不在某个运行帧中突然切换到另一算法。默认 `numpy` 不增加该依赖。

## 数学行为保持

保留相同的 Kalman 模型、噪声参数、协方差更新、马氏距离公式、5 次 jitter 尝试和最终通用求解回退。
可选 `triangular` 仅替换已分解矩阵的两次求解：
`L y = b`、`L.T x = y`，仍返回完整逆乘积，而不是白化残差。
不修改 ReID 阈值、身份门控、转向、PID、深度期限和停车保护。

单线程 300 次离线更新测试：原 NumPy 求解器 P95 约 0.087ms，三角求解器约 0.098ms。
因此本轮没有证据支持将三角求解器设为默认。离线单轨迹微基准不是实车整帧性能。

## 新日志

`numeric_runtime`：视觉初始化、首帧及首个实际轨迹匹配各记录一次。
包括 NumPy 路径/版本、环境请求、进程实际加载数值库路径及可读到的实际线程数。
首个匹配日志用于覆盖 SciPy 延迟加载另一套 BLAS 的情况。
不安装 threadpoolctl、不修改运行中线程池；不支持的库返回未知而非假称已设为 1。
审计只读且仅启动初期执行，不在每帧遍历进程映射。

原 `pipeline_timing` 增加以下 wall/thread-CPU 阶段（单位 ms）：

- `deepsort_kf_predict`：所有轨迹预测。
- `deepsort_kf_project`：协方差投影，包含匹配门控和更新的调用。
- `deepsort_kf_cholesky`：分解。
- `deepsort_kf_solve_lower` / `deepsort_kf_solve_upper`：两次求解。
- `deepsort_kf_fallback`：所有分解尝试失败后的原回退求解。

`kalman_counts` 记录当前轨迹数、检测数、匹配数、新建数，以及分解/求解/预测/投影调用数、重试数。
计数和时间在每帧预测前重置，空检测帧不会重复上一帧求解记录。
`kalman_solver` 标明求解模式。细分耗时与原阶段有包含关系，不能直接全部相加。
wall 与 thread CPU 的差不是纯锁等待；它还包含调度和其他原生线程执行等因素。

## 离线工具与验证

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tools/benchmark_kalman.py --iterations 300
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 python3 tools/benchmark_kalman.py --iterations 300 --solver numpy
```

工具不读取摄像头、不初始化串口或电机，不改运行配置，只向 stdout 输出 JSON。
33 项新增测试覆盖随机 SPD 矩阵、float32/64、多种 RHS 形状、200 步轨迹、门控判定、
jitter/回退、逐帧计数、启动脚本环境策略、实际库线程查询。与原门控测试合计 43 项通过。
全套在单线程环境下 3686 项通过，3 项既有失败：目标距离配置断言和两项搜索方向断言。
Shell 语法、Python 编译、diff 空白检查通过。本轮未操作实车。

## 下轮验收

修改前日志：`run_20260921_202227_55971_696a215f`，575 个处理帧。
跟踪总耗时中位数 19.41ms、P95 62.29ms；Kalman 更新阶段 P95 37.98ms。
全部帧最高 246.92ms；CAP50 后跟踪最高仍为 144.31ms。
整帧处理超时 3/575（0.52%），去掉 CAP50 及之前为 1/562（0.18%），
所以本轮重点不能仅看过期比例；处理前等待/真实采集年龄也须单独核对。

同路线、相近候选数下，试验目标（未实现承诺）：

- 稳态跟踪 P95 争取低于 20ms，Kalman 更新 P95 低于 5ms。
- 单独统计超过 50ms 的跟踪帧数与原因，不用启动热身尖峰替代稳态指标。
- 比较 YOLO、ReID、采集等待、控制周期是否退化；单线程可能影响其他 CPU 数值工作负载。
- 正确目标接回率、误接回次数、方向依据和运动保护不应恶化。

若尖峰仍在，按细分求解耗时、线程 CPU、轨迹数和重试次数定位，再安排后台录像/日志隔离对照。
不要同时修改 RPM、身份阈值或时效期限，否则无法归因。
