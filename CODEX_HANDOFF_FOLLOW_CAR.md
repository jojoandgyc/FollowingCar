# Follow-Car Project Handoff

## 2026-09-18：追赶请求调整180RPM；近距60RPM暂未提高

主INI pi_launch_request_rpm=180，200总上限/原制动/时效/反馈要求保持；没有硬件操作。
用户提出1.8m静止目标的cap改60，未强制实施：剩余27cm、delay .2s时需要
约3.13m/s²减速度（现.4、合法最大1.0），有采样龄要求更高，尚无实测依据。
不能通过制动后加速度下限绕过。详见docs/distance_pi_launch_120.md顶部更新。
主配置/真实入口/0回退测试已同步180；新增3项纯控制回归验证180需求和近距制动不被抬高。
聚焦73项通过，全套2628通过、原有2项搜索方向失败；diff检查通过。

## 2026-09-18：120RPM起步/追赶请求试验

完整方案见docs/distance_pi_launch_120.md。主INI新增pi_launch_request_rpm=120；0回退，
老配置默认0。前进PI、error>deadband且轮速反馈有效时取max(PI请求,120)，
先过原制动/总上限，再旁路普通软件升速坡；实际批准可低于120或0。
不改电机固件加速寄存器，不改倒车/搜索/过零保护/身份/深度期限；P2.0、Ki0.4保留。
冷启动和真正失权后的合格新测量亦可按该策略恢复，但旧/过期帧不能续期或重新起步。
主日志新增pi_demand_rpm/launch_floor_rpm/total_demand_rpm/software_rise_bypassed/
demand_limit_reason/feedback_age_ms；follow_metrics新增去重launch统计，不加串口查询。
已通过模拟电机下发120、近距限幅、反馈缺失/失效、截止时间与换向保护等回归。
120RPM约1.63m/s；实际硬件加减速尚未测定，近距制动cap仍会压低请求。未操作实车。
新增49项回归通过，全套2625通过、原有2项搜索方向失败；编译/启动脚本语法/diff检查通过。

## 2026-09-18：修复PI授权回调接口异常

12:35:26及12:35:41两轮运行在正PI请求后的授权读取抛出TypeError：
上一轮新增调用传now关键字，主程序__init__实际lambda只接收uid，快环报错、主循环退出。
本轮统一为原单UID回调接口；真实snapshot自行读取当前monotonic检查期限，不跳过授权检查。
未调整P、深度期限、速度上限或搜索；未启动任何硬件。
新增test_live_authority_binding.py直接提取执行生产__init__中的绑定语句（不运行初始化），
接真实PI和snapshot函数，覆盖视觉/Depth两路、live/186ms间隔/过期/撤销及零轮速冷启动。
修改前8项均复现原TypeError；修改后9项通过。旧测试替身同步改为单UID，防止掩盖接口错误。
全套先跑2575通过、原有2项搜索方向失败；随后追加冷启动测试单独通过。

## 2026-09-18：CAP606～674缓存接续与距离PI响应

方案见 `docs/distance_pi_continuity_response.md`。主INI的距离PI Kp=2.0/s（原1.0），
Ki=0.4不变；正常启动脚本直接加载。5000ms旧框测距未采纳，ROI/授权/安全边界未延长。
同UID物理锚点仍在原200ms保持期时，stale ROI不再清融合缓存；返回保持值不带新测量时间。
180ms以上积分间隔只暂停积分，原实际授权仍有效则继续原加速斜坡；新鲜样本进PI前
先查询运行时实际授权，真实失权仍按实测轮速恢复，不能只相信PI内部250ms计时。
190ms视觉迟到撤权、180～250ms制动余量检查仍保留；本轮不宣称解决这两条剩余中断。
新增22项控制回归和1项指标回归；新增统计depth_cache_retained、
distance_pi_authority_suspended及tracking_gap_no_integral。未启动硬件。
全套2567通过，原有2项搜索方向测试仍失败；编译、脚本语法、diff检查通过。
理想模型P提高后移动峰距下降，但突停仍约16cm过冲；不等于实车结果或安全保证。
回退P对照只改主INI pi_kp_per_sec=1.0，保留本轮缓存/时间解耦修复。

## 2026-09-18：前进授权最大250ms，延续与新控制更新分离

详见 `docs/depth_authority_250.md`。用户同意将180ms延长到250ms；主INI
`[astra_depth] longitudinal_sample_max_age_sec=0.25`，原启动命令重启生效。
180～250ms只允许已有同UID前进授权维持/减速，必须仍满足视觉、身份、安全、保守制动余量检查；
不能据此开始前进、加速、更新PI、延长原始采样时间或accepted watermark。
新控制样本与PI最大积分间隔仍180ms，倒车仍180ms，条件式ROI250ms不变。
延续拒绝绑定原UID/采样时间，同一旧授权不能因RGB刷新复活；新鲜Depth才允许重新评估。
正常周期和非周期直接轮速下发都在反馈读取之后再次核对授权，避免等待期间跨过截止时间。
前进250ms是上限而非停车保证：距离不足、危险、明确停止、失目标可提前撤销。
PI增益、RPM上限、深度过滤、ReID和搜索参数未改；制动模型仍待实测标定。

新增日志depth_forward_continuation/depth_late_sample_continuation；指标新增180～250ms及>250ms
接续间隙、延续资格去重允许/拒绝数（不是实际运动时间或避免停车次数）。
本轮新增60项回归通过；全套2544通过，仍只有原有2项search_direction_switch失败，
3条旧P试验警告为预期。未启动相机/串口/电机，实际连续性、距离误差和近距过冲待下一轮验证。
回退：将同一INI改回0.18并重启。不要把旧PI初版段落的180ms当作当前前进授权配置。

## 2026-09-18：距离 PI 主控、独立制动与统一恢复

详见 `docs/distance_pi_control.md`。用户批准方案后完成实现，主INI默认
`[distance_pid] control_mode=distance_pi`，仍运行 `./run_request_0428_modular.sh`。
单轮回退用 `FOLLOW_DISTANCE_CONTROL_MODE=approach`；未改现有总上限200RPM、深度180ms授权、
条件式ROI250ms、身份/搜索/横向阈值，独立深度shadow仍无电机权限，未操作任何硬件。

前进为SI距离PI：Kp=1.0/s、Ki=0.4/s²、积分速度上限0.8m/s，轮周长0.816814m。
死区冻结而非清积分；人速仅诊断，不叠加前馈/旧20RPM偏置/旧60RPM无前馈总限幅。
制动使用可信原始距离短窗、同步轮速和新命令预测；缺相对运动证据保守退化。
减速度/延迟仍为待实测模型，不能宣称物理刹停保证。

正常PI只走一条加速斜坡，不重启旧25/45RPM阶段；内部记忆短暂保留不延长授权。
物理重复/乱序不积分，断档不补积；处理时上一物理授权已过期也从实测轮速重新接续。
拒发的新请求或提前撤销的旧授权不能成为下一次加速锚；拒发只回退本样本正积分增量，
不能用失权零指令反复洗掉旧I。最终批准向下量化，不穿越制动界限，正常2RPM量化不持续抑制积分。
保持旧保护倒车；修复新PI资格前置对不可转向可信近距退让、倒车释放后latch的衔接，
不因此授权不可转向目标前进。

新增distance_pi/pause/limit/admission_rejected/authority_suspended日志和指标。
`tools/follow_metrics.py`现在将PI无匹配基速标记为诊断，不再用于性能退化比较。
基线000408、CAP71～419：移动MAE0.534m、最大2.480m、速度短降9.14次/10s、零指令18.94%。
首轮目标≤0.40m/≤2.20m/≤5次每10s/零指令不恶化；站定单看收敛、最低距离、反复起步。
新增138项回归通过；全套2484通过，仍只有原有2项search_direction_switch失败；
3条旧P试验不影响新PI参数的警告为预期。编译、bash -n、git diff --check通过。
所有效果为待验证：历史回放和理想闭环模型不能证明实车下一轮距离轨迹。

## 2026-09-17：恢复反馈范围统一，桥接不再封顶新距离追赶

详见 `docs/recovery_bridge_distance_independence.md`。基线233228，移动CAP59～304。
恢复三条路径的旧100RPM资格门槛统一至已有估计器范围，当前profile约129.074RPM；
不提高指令预算。CAP188同输入84RPM请求由旧24恢复到84，仍检查实测轮速/时效/危险。
profile桥接只限制旧人速分量，controller和runtime都不再拿旧总批准封顶新距离；
无旧授权仍走新距离+实测轮速有界恢复，先验与Depth分开到期，180ms不变。
桥接旧基速只降不升，不能借距差提速重新增长；新鲜人速证据到达才重建。
CAP193原24RPM→45RPM正常斜率请求→44RPM量化批准；不是立即发68RPM。
增加组件/恢复反馈审计日志。未改INI、P、速度上限、横向/ReID/搜索或执行线程，未操作实车。
新增76项回归通过；全套2346通过，仍为原有2项search_direction_switch失败，无新增失败。
编译与git diff --check通过。下一轮机制验收、指标和基线见上述文档。

## 2026-09-17：独立距离/制动资格与可关闭的人速前馈

详见`docs/independent_distance_control.md`。基线230002、CAP67～452移动。
原始制动窗口与匹配人速窗口分开：人速资格拒绝不清掉距离制动，独立转动上界仍保守。
无前馈且独立制动合格时60RPM预算可用，非最低速度；180ms/危险/身份保护不变。
短暂反馈150～300ms旧时按旧轮速、年龄、加速上界和最后请求估算，不直接假定200RPM；
更旧/异常仍保守上界。上述模型假设未实车标定，不是安全保证。
默认可选前馈；`FOLLOW_MATCHING_MODE=distance_only ./run_request_0428_modular.sh`只关闭贡献，
仍记录人速估计，关闭桥接使用；`optional`恢复。新增模式、独立制动与反馈上界日志/指标。
本次未操作硬件；不得把纯距离模式100%无前馈记作估计退化，须对照误差/降速/近距过冲。
新增32项回归通过；全套2270通过、原有2项搜索方向失败，无新增失败。
编译、shell语法和空白检查通过；异常raw重复读取不得清掉已拒绝标记。

## 2026-09-17：匹配人速与接近制动统一物理时间线

详见 `docs/shared_motion_window.md`；基线223601、移动CAP78～338。
仅approach启用时，FF与制动共用RawDepthClosingWindow，原始深度减转动贡献，
人速再加同区间梯形积分的实测轮速。低不确定性补偿模式切换/普通短断档不无谓重建；
真实>180ms间隔、角速度超限、身份/危险/异常深度仍清空或保护，不延长电机授权。
修正同样本重复读取把已验证制动速度退回encoder_fallback的问题，不续期或重复加速。
新增longitudinal_shared_window与指标汇总。原PID模式、ReID/搜索/横向/限速配置未改。
新增27项测试；全套2227通过，原有2项search_direction_switch失败，无新增失败。
无硬件动作；下轮须实测预热/角速度拒绝各自占比、速度突降与移动误差，不宣称已消除17次yaw_limit。

## 2026-09-17：原始深度短窗制动、60RPM无匹配预算、有效授权正常接续

详见 `docs/raw_closing_recovery.md`；对应移动CAP81～261、220356日志。
制动使用同UID物理raw Depth的<=180ms短窗与现有转动补偿，不再用中位距离阶跃差分。
无合格短窗时按编码器保守参考，不开启60RPM预算；异常raw跳变仍保守制动。
主INI新增approach_no_matching_max_rpm=60，仅独立距离追赶，不伪造人速/不设置最低转速。
动态制动、正常斜率、180ms期限、危险停车不变。关闭预算设0；关闭profile可回退legacy PID。
普通测量尝试失败后，只有runtime实际旧授权仍有效且无危险/停转/活动恢复时，正常有界加速，
不因轮速尚未达到旧请求90%又启动恢复。真实过期或撤销继续原恢复路径。
新增closing_source/window/status、no_matching_reason、normal_live_continuation及对应指标。
28/125无匹配中19次预热、4次转动限制（其中3次需手工核查旧日志）、3次no_forward_motion、
1次间隔短、1次速度异常；没有把60RPM fallback计作匹配，避免虚假改善。
新增25项回归通过；标准tests目录全套2200通过、原有2项search_direction_switch失败，无新增失败。
直接pytest仓库根会收集tmp_sync历史副本引发37项collection errors，应使用pytest tests。
编译和git diff --check通过；无电机/相机/串口操作。下轮目标及风险见上述文档。

## 2026-09-17：正常纵向前进改用匹配＋追赶＋动态减速参考

完整方案/计算/风险/回退/下轮指标见 `docs/longitudinal_approach_profile.md`。
主INI默认approach_enable=true：正常前进替代旧P/I/D和20RPM偏置，原倒车不变；
1.0/s距差追赶，最多0.60m/s（44.1RPM），按当前接近速度、样本年龄、0.20s执行预算
及假设0.40m/s²提前减速。制动假设未经实车标定，非停车安全保证。
下游matching+20RPM预算与新曲线接通；反馈接受上限按base80+chase44.1+5容差≈129.1RPM，
防止新参考合法速度被旧105RPM门限拒绝；异常反馈/目标速度/转动保护不取消。
可信Depth180ms、ReID/搜索、差速执行和恢复逻辑保留；shadow仍无电机权限。
新模式不叠加FOLLOW_DISTANCE_P_TRIAL或FOLLOW_MATCHING_BIAS_TRIAL；关闭approach可回退。
日志新增longitudinal_approach及按UID/物理时间戳去重指标。
新增67项测试通过；全套2175通过，仍是原有两项search_direction_switch失败，无新增失败。
编译/空白检查通过，旧真实日志指标兼容。未操作实车，仿真仅理想模型，不证明实际效果。

## 2026-09-17：独立Depth连通域与躯干核心统一口径

见 `docs/depth_component_geometry.md`。seed/update在完整低分辨率网格测量连通域面积，
搜索ROI只筛选；距离/中心使用随组件移动的固定相对躯干核心，不混入新增腿部/地板。
2.5倍增长/速度/深度/身份期限未放宽，增加固定种子面积预算和背景连通种子拒绝。
新增细分候选审计与同图空间检查工具。12张真实快照：9张连续10次成功，3张明确拒绝
跨全图疑似背景连通；不是实车连续跟踪成功率。旁路仍不控制电机，原运控参数未改。
新增22项回归通过，全套2108通过、原有两项搜索方向失败，无新增失败；未操作实车。

## 2026-09-17：独立Depth逐帧跟踪在线观察

见 `docs/depth_track_online.md`。默认正常脚本+日志目录启用后台旁路，可用
FOLLOW_DEPTH_TRACK_SHADOW_ENABLE=0关闭。仅复用只读深度历史/编码器缓存/已绑定原始框，
不读串口、不改滤波/UID/搜索/PID或电机授权。视觉固定350ms、深度180ms为旁路预算。
缺少已核验内外参，明确标记HFOV+轮轴相机近似几何UNVERIFIED，绝不能给电机使用。
新增JSONL、限量小深度快照及离线指标工具，区分种子/重放/新帧/潜在测距断档补齐。
新增34项回归通过，全套2086通过，原有两项搜索方向失败；编译检查通过。
未操作实车；下轮先验证关联、处理耗时、错绑与原视觉/控制负载，不期待旁路直接改善运动。

## 2026-09-17：新深度独立恢复与近距禁止重复起步

见 `docs/fresh_distance_fallback_near_stop.md`。修复CAP730形状：旧FF授权失效时，
新鲜同UID远距可剔除FF、读取缓存距离PID，按实测轮速+一个50ms加速步受限恢复；
不重算PID、不复活旧授权、不延长Depth180ms。近距无有效人速匹配且低于启动阈值，
停止而非重新注入20RPM；有效人速匹配仍优先。未修改配置/P/搜索/身份或操作实车。
新增独立恢复与近距抑制日志、统计；39项新测试通过，全套2052通过、原有2项搜索失败。
下一轮按移动/站定分段检查独立恢复后实际下发、速度突降、误差与最低距离，见文档。

## 2026-09-17：26cm轮径与远距离速度有界接续

见 `docs/wheel_geometry_velocity_degradation.md`。默认reid_runtime轮周长0.600→0.816814m，
使用同一参数供深度预测/融合/纵向FF，不动电机RPM和旋转标定。P24/200RPM不变。
匹配模式速度先验记忆固定350ms，新鲜Depth才可衰减使用，原来源不续期；导数不能跨
未补偿转动。桥接不能重启失效前进，Depth180ms期限仍独立生效。远距恢复增加按实测
车速/闭合量/到近距带时间的受限路径，不再仅因缩短>3cm回到24RPM。
新增far低速与闭合恢复诊断；35项新回归通过，全套2013通过，原有2项搜索方向失败。
未操作实车。下一轮仍按移动/站定分段，详见文档指标；不要同时叠加P36或新速度上限。

## 2026-09-17：取消深度填色，每采样框显示独立米数；修复显示缓存淘汰

见 `docs/video_wheel_feedback.md` 最新节。每ROI白字黑底显示原始簇中位数（S为实际
选中簇、C为该区域最大候选簇），无簇n/a，不用全局融合距离填各框。Astra只追加
已有selected region evidence的标量距离与depth_size；测距/滤波/授权逻辑不改。
去掉色带、填色与像素处理，保留dt/UID/ROI CAP/拒绝原因，CSV追加regions_json。
19:33:58录像308帧、272 NO_SAMPLE，其中246帧离线有80ms内对应记录；录像排队
60帧时旧24条可视化缓存被淘汰。现在缓存512条纯统计（不持有Depth数组），保留
80ms显示配对限制及既有运动授权期限；可处理原始帧已出8帧环但统计仍可用的情形。
数值显示专用28项测试通过，原录像非阻塞自测通过。已有快照离线绘字中位0.68ms，
未启动相机/查看器/电机，旧录像不覆盖。全套1978通过，原有两项搜索方向失败。

## 2026-09-17：深度稀疏彩点升级半透明ROI色块

仅修改 `video_depth_overlay.py` 的录像后台绘制和相关测试/文档；不改测距/授权/PID。
40%原始深度色+60%RGB，固定0.4~5m红绿蓝色标；最近邻映射不插值、不填0/65535孔洞。
重叠ROI只叠一次，保留选中区域框、ROI CAP、dt、拒绝原因。最多5个ROI，每ROI投影
65536像素以内，超大区域跳过并提示SKIP。原时间匹配与缓存路径不变，旧录像不覆盖。
新增8项回归通过，完整1976通过、原有两项搜索方向失败，录像非阻塞自测通过。
用已保存真实快照离线检查画面，绘制中位2.66ms/P95 2.93ms（不含编码/并发负载）。
未开启相机或操作电机，下次原脚本运行生效。

## 2026-09-17：录像测距ROI原始深度彩点

见 `docs/video_wheel_feedback.md`。新增 `video_depth_overlay.py`，DepthDiagnostics
保留最多24条实际扫描元数据与既有Depth数组引用。录像线程try-lock读取，按Depth
真实时间匹配RGB，80ms内绘制原始ROI深度稀疏点，另标ROI来源CAP；超出不画点。
最多720点，固定0.4~5m近红远蓝；青色ROI表示候选簇选中区域，不代表控制已放行。
明确输出拒绝原因、时间差及CSV；未测量帧显示NO_SAMPLE。原RGB/Depth不变，不增加
传感器读取/推理/测距锁/电机操作。未修改PID、恢复或授权策略；旧录像不覆盖。
新增18项测试通过，全套1968通过，原有两项搜索方向测试仍失败。未操作实车。

## 2026-09-17：每帧目标速度与跟随诊断

见 `docs/video_wheel_feedback.md`。新增只读`video_follow_telemetry.py`，在既有决策
路径发布不可变诊断，采集时快照，录像后台叠字/CSV追加。Vt为纵向目标速度估计，
不是二维速度；Vrel为相对距离变化率。显示raw/used/SET/ERR、PID/BASE/AUTH、
来源CAP、UID、X、Depth年龄、决策/估计状态，以及同期编码器YAW。
速度过期不显示旧数值；桥接使用原速度原点。授权与实测轮速明确分开，前馈/Depth
截止时间独立显示，不调用会改变授权的getter。无新增传感器读取/控制等待/推理。
原脚本直接生效，旧录像不重编码；无PID/运动策略调整。实车效果和并发开销待观察。
新增22项测试通过；全套1950通过，原有两项搜索方向测试仍失败。录像非阻塞自测、
编译和diff检查通过。合成图叠字额外约3.75ms/帧（后台离线测试，不代表实车延迟）。

## 2026-09-17：反复短断档不再重启低速恢复

见 `docs/repeated_scheduling_recovery.md`。安全调度缺失保留动态恢复状态与原结束点，
物理样本时间不因失败尝试更新，参考500ms与运动授权180ms分开；ROI仍250ms。
最初批准速度仅用于恢复起点，不再作为固定上限；新测量按实测轮速/原加速度恢复。
连续性快捷分支不能绕开活动恢复；时间到但实测轮速不支持时不放行完整PID。
增加暂停/丢弃/完成/突破初始限幅日志及指标，重复样本只能减速不能加速。
模拟多次120ms断档输出24→48→60RPM，只启动一次；实车效果尚待验证。
原脚本直接生效，不改PID/RPM/身份/搜索；未实现1.5秒旧命令或新的独立Depth接管。
新增25项回归通过，全套1928通过、原有两项搜索方向失败，无新增失败。
编译及diff检查通过；未操作实车。下轮指标和边界见上述文档。

## 2026-09-17：每个录像CAP增加左右轮编码器RPM

见 `docs/video_wheel_feedback.md`。采集时读取MotionActionRuntime已发布的冻结反馈对象，
无串口读取、无电机/反馈锁等待；不改变运控。快照随录像帧入队，后台绘制WHEEL L/R、
反馈状态及采集时年龄。正号统一车体前进；过期显式STALE，缺失n/a而非0。
未经过YOLO的帧同样有轮速快照，CSV追加5列，旧录像不重编码。
直接原脚本启动即可生效，本次没有修改PID/恢复/搜索/身份策略。
新增13项测试通过；全套1903通过，原有两项搜索方向测试仍失败，无新增失败。
原录像慢消费者自测、编译和diff检查通过；未启动摄像头或操作实车。

## 2026-09-17：明确调度断档的新鲜样本恢复 + 独立 P36

见 `docs/scheduling_recovery_p36.md`。最新移动CAP90~392：MAE0.809m、最大3.380m、
零指令11.02%、批准短降9.04次/10秒、恢复受限样本平均损失14.11RPM。
同UID明确调度失败的恢复参考可保留500ms，但Depth授权仍180ms，过期仍停车。
新鲜可信远距、距离无明显闭合/跳变、实测双轮正向0..100RPM、yaw<=15°/s，
采用实测基速+有界加速恢复，不再仅因未达到旧批准值90%就回到固定25/45RPM两档。
参考改为实际经过质量限速的批准速度；旧请求不能充当批准值。
日志与指标将动态恢复损失计入原统计，并同时报告两类恢复总频率。
增加显式FOLLOW_DISTANCE_P_TRIAL=36（仍支持24/27，默认24），+50%的P独立试验。
保持上轮+5补偿做新代码P24/P36对照；不改I/D、轮周、总预算、身份/搜索/横向。
站定原始深度/编码器仍有1.363比例差，未独立实测，不能自动改轮周。
新增38项回归；全套1890通过、原有两项搜索方向失败，无新增失败。
编译/Shell/diff检查通过。未操作实车，恢复效果与P36收益待同场景验证。

## 2026-09-16：一次小幅停止矛盾降速与默认关闭常量试验

见 `docs/stop_suspect_bias_trial.md`。基线23:25:50移动CAP159~410：MAE0.469m、
最大2.276m、短掉速10.92次/10秒、无基速23.01%、零指令6.04%。
新鲜可靠远距、>=4点100ms正向窗口遇一次[-.05,.03]m/s瞬时值，只降速不清空；
重复弱证据清空，两次新鲜正向样本才重置容错资格。大负值/近距/TTC/过期等仍保护。
FOLLOW_MATCHING_BIAS_TRIAL=0|5|10默认0，只改变合格控制请求，不改估计/桥接原值；
距离目标+.2到+.5m淡出，疑似停止/预热/桥接不加。P24、匹配80、总200、Depth180ms不改。
新审计与指标区分估计/请求偏置/批准轮速。先同代码P24+0，再P24+5做对照。
新增只读encoder_scale_audit，固定表面与独立实测位移才可验证换算；覆盖不足拒绝外推，
不自动标定或控制硬件。本轮不因现有Depth/编码器比例1.435就修改0.60m轮周。
新增51项回归通过，全套1852通过、原有两项搜索方向失败；编译/Shell/diff检查通过。
未操作实车，效果未验证；真实停止的小负值可能多一个有界降速样本，需检查近距过冲。

## 2026-09-16：小角速度分歧有界补偿、105RPM反馈容差与P试验

见 `docs/feedback_matching_boundary.md`。基线23:06:08移动CAP92~322：MAE0.554m、
最大2.431m、短掉速10.32次/10秒、无基速20.19%、零指令7.18%。
异号raw/filtered yaw仅在新鲜可靠远距、反馈-Depth<=50ms、双向投影有效且
旋转影响上界<=0.04m/s时，用正上界扣除速度估计（保守低估），不放宽15°/s门槛。
运行估计器容差最多105RPM，不把电机200RPM上限当作可信反馈；默认独立估计器100。
新增反馈审计和物理样本去重指标。专用FOLLOW_DISTANCE_P_TRIAL=24|27只覆盖纵向P；
默认P24、匹配80、总200、Depth180ms、身份/搜索/横向/执行器不改。
先新代码P24再同场景P27，不要将逻辑和P同时变化的改善全归因于P。
新增44项回归通过；全套1801通过，原有两项搜索方向失败；未操作实车。

## 2026-09-16：重复深度的低置信度保持不再单独清空估计

见 `docs/replay_velocity_retention.md`。基线22:50:44移动CAP77~288：误差0.774m、
最大2.787m、批准短掉速14.69次/10秒、零指令6.81%；8条discarded/radar_hold重置。
仅已知0.45~0.50置信度、同一估计时间的明确duplicate可绕过保留参考时的置信度项。
PID/运动资格默认检查完全保留；原期限、身份、危险、背景/跳变、轮速/转动仍检查。
不推进任何证据时间，不允许旧帧授权/加速；转动未验证仍清求导链，超期仍清空。
新增按原证据去重的保留指标与hold重置计数。P24、匹配80、总200RPM均未修改。
新增30项回归通过，全套1757通过、原有2项搜索方向失败；编译/diff检查通过。
未操作实车。后续若连续性达标但仍偏远，可单变量评估纵向P24->27，不要同时加上限。

## 2026-09-16：远距离追近速度下降与恢复

见 `docs/far_catchup_velocity.md`。最新22:36:02移动CAP123~374作为修改前基线：
平均绝对误差0.883m、最大3.099m、批准短掉速15.37次/10秒、零指令10.07%。
独立匹配预算下新增<=180ms历史窗口，对深度加同区间编码器/旋转积分后做直线拟合；
仍先过相邻样本质量/跳变检查，瞬时停止/近距/预计400ms进入近距带立即减速。
远处正向目标的小幅追近也按80RPM/s下降，不因range_rate<-0.03就撤掉平滑。
调度断档新鲜恢复，仅同UID、远处3~12cm闭合、<=1m/s、双轮正向新鲜、
物理间隔<=180ms且满足实测轮速与旧输出连续性，才允许不重启25RPM。
危险/搜索/身份/Depth180ms/PID/总预算未放宽；原桥接和yaw模式重建仍保留。
日志及指标增加窗口跨度、瞬时/窗口速度、下降策略、远追近恢复计数。
未操作实车；效果待下一轮日志，不声称已解决全部跟随误差。
新增31项回归通过（替换原1项过严追近策略断言）；全套1727通过、原有2项搜索方向失败。
编译与diff空白检查通过。

## 2026-09-16：轮速同向恢复与2.5m边界连续性

见 `docs/execution_resume_boundary.md`。移动基线22:01:19 CAP195~514：误差0.864m、
零指令20.21%，其中过零等待1.014秒。正常双轮前进的pending过零状态，允许两次
新鲜同向反馈（150ms内）低速恢复：最大轮5RPM启动、按已下发轮速80RPM/s爬升。
真反向/旋转/倒车仍保留静止确认，超时不强行放行；Depth过期/状态变化仍可立即撤销。
深度仅对阈值±10cm、前后>=3躯干区域、同UID新鲜锚点、<=8cm/1.5m/s和框连续的
跨边界观测免去重复三次确认。初始/弱区域/旧锚点/真实跳变保持原保护。
新增实际请求->下发损失、正请求下发0次数、零命令按来源驻留与边界连续计数。
未调PID、速度预算、身份或搜索；未操作实车。下轮指标和验证命令见文档。
新增27项回归通过，全套1697通过、原有2项搜索方向失败；编译及diff检查通过。

## 2026-09-16：补偿窗口与短空档估计连续性

见 `docs/compensated_matching_continuity.md`。基线21:34:34移动CAP48~317，平均距离
误差0.899m、two_sample占比69.12%、无匹配41.71%、批准短掉速7.51次/10秒。
低yaw<=5的原始RGB到Depth补偿窗口扩展到250ms，Depth当前年龄仍180ms。
无观测时仅保留短时稳定、已有补偿基线（<=180ms，yaw<=15且变化<=3）；新端点
必须再次补偿/时间/反馈验证；不更新计数、时钟或运动授权。转动骤停也重建。
独立匹配预算开启时桥接按80RPM/s下降，只适用于远处不靠近；原证据截止不延长。
近距/危险/身份/搜索/三样本预热/恢复限速/PID与速度预算保持原边界。
新增补偿重建原因、桥接模式和不同基线重置/保留指标；未运行实车，需下一轮验证。
新增35项回归通过，全套1670通过、原有两项搜索方向测试失败；编译和diff检查通过。

## 2026-09-16：独立匹配基速预算和短时掉速指标

见 `docs/matching_speed_budget.md`。CAP160~410基线平均绝对误差0.967m，匹配触顶69.33%，
零指令10.13%；本轮尚无实车效果数据。正常INI新增matching_base_max_rpm=80，默认0兼容；
估计器和下发层同步拆掉旧额外前馈20RPM对匹配基速的重复限制。至少3个合格物理样本后
才能超过40RPM；rise80RPM/s，fall80RPM/s仅远处、未靠近且正速度时使用，真实减速绕过。
总200RPM、Depth180ms、ROI250ms、PID/身份/搜索/倒车保护未放宽。
stale_vision_depth_audit仅新增审计，视觉过期撤销和方向恢复尚未解耦；不要宣称已解决。
指标新增触顶率、200ms内>10RPM请求/批准降速（按来源分类而非推断因果）、同样本限幅损失。
短转动桥接截止仍180ms，本轮不解决全部桥接退出。下轮以距离误差、指令连续性、站定过冲
联合验收；指标目标和命令见文档。未操作电机。
新增30项回归通过，全套1635通过、原有2项search_direction_switch失败仍在；
配置、编译、diff检查通过。

## 2026-09-16：测量空档保留证据与有界连续恢复

方案和下轮基线见 `docs/scheduling_gap_continuity.md`。只处理CAP192~546暴露的
无观测→估计清空→恢复限速链；配置/PID/ReID/搜索/电机执行入口未改。
无raw/sample的明确ROI失败或回放，同UID、原证据仍有效且反馈安全时保留下一帧参考；
不提供PID采样时间，不续期。低yaw保留微分链，5~15度/秒仍重建微分并单独计数。
恢复参考新增原斜坡起点与降级标记；新鲜Depth、物理间隔<=180ms、处理间隔<=350ms、
实测基速>=原批准90%且其他原实测恢复门控通过，才延续旧斜坡。确实停下/异常仍保守恢复。
`follow_metrics.py` schema=2补齐独立停车日志，新增限幅损失RPM（UID/样本去重）、
证据跳过/微分重置/连续恢复指标及每10秒间隙数。旧移动基线零速16.42%，静止61.25%；
旧移动限幅损失均值14.73RPM（63个样本），不是行程损失。旧报表需用新脚本重算。
新增43项回归通过；全套1605通过、原有2项search_direction_switch失败仍在。
编译、diff检查通过，未操作实车。下一轮才能判断距离误差是否降低，不可把计数变少当作效果证明。

## 2026-09-16：250ms框窗口与200RPM前进许可

见 `docs/depth_roi_250_experiment.md`。用户同时授权两项：低转动连续跟随框窗口
250ms，正常远距前进上限200RPM；不是单因素实验。物理Depth期限独立为180ms，
不能借框延期续期。框180~250ms仅在新鲜可信编码器原始/滤波yaw均<=5deg/s时采样，
同UID/质量/搜索/危险/背景/跳变检查仍保留，不是独立深度跟踪或ROI运动预测。
200RPM比例下近距仍20RPM、倒车仍60RPM、固定搜索RPM不变；PID与40RPM前馈未调。
新增时钟配置、ROI延期状态、实际RPM限幅日志，以及正向新鲜深度间隙指标。
26项新增回归通过；全套1562通过、原有2项搜索方向测试失败。配置、编译、diff检查
通过。未操作实车；控制器支持200RPM不代表车体制动与近人安全性已验证。

## 2026-09-16：普通距离停车恢复入口

实施方案和边界见 `docs/follow_parking_recovery.md`。针对CAP746普通停车后直到
CAP822转向才解锁的问题，新增严格来源标记 `follow_distance_hold`；保持期间允许
合格原始框的Depth继续测量，但不发布运动。停车后同UID两张新鲜、连续且已远离
保持区的Depth，通过实时安全复核后仅解除本次普通保持，丢弃旧授权，由新样本
重新经过PID与限速。安全、故障、搜索、显式停止不适用此恢复；180ms期限不变。
新增started/observe/released日志及 `follow_metrics.py` 的停车计数、保持耗时分布。
44项新增回归通过；`python3 -m pytest -q tests --tb=short` 为1536通过，原有两项
search_direction_switch失败仍在。本轮编译、diff检查通过，未操作实车。
不要直接在根目录无参数收集pytest：会混入backups/tmp_sync及缺模块的串口脚本。
独立Depth在线shadow尚未接入：现有转向标定不代替深度网格内参及逐帧位姿对齐。
本轮不调PID、50ms周期或速度上限；实车是否减少81.3%的零指令占比需新日志验证。

## 2026-09-16：独立深度跟踪方案，阶段A仅离线

`docs/depth_independent_tracking.md` 给出A离线关联、B在线shadow、C受限控制方案。
新增 `depth_track_shadow.py` 和 `tools/depth_track_shadow_eval.py`；输入必须有
已确认UID、历史对齐的躯干ROI/距离、显式深度内参和逐帧相机位姿。
可以在无新视觉框时更新深度轨迹，但不续期350ms实验视觉租约，物理Depth仍180ms。
所有输出永久 `control_allowed=false`，没有接入正常运行线程或更改INI/电机逻辑。
`AstraDepthRuntime.copy_depth_history` 提供有界独立只读副本，不修改测距状态。
下一次正常运行不会自动生成新关联日志；阶段B/C尚未实现。旧稀疏异常快照不能
冒充30Hz连续回放。新增25项通过；完整1492通过、原有2项搜索方向失败仍在。
未操作实车。不要把测试通过解释为实际1.5m距离收敛或实车错绑率已验证。

## 2026-09-16：正常跟随50ms轮速执行与转动连续性

`docs/follow20_continuity.md` 记录方案、边界、回退方式和CAP111～264/608～720基线。
正常INI开启 `follow_wheel_period_sec=0.05`、`turn_compensation_enable=true`、
`measured_recovery_enable=true`。正常可见跟随采样最新Depth基速+有效横向轮差，
普通旧动作不再直接下发；安全撤销/零横向/降速可抢占，180ms深度期限不变。
转动补偿只用于新鲜原始检测框和可信编码器的近轴<=15deg/s场景；短暂纯调度
失效保留<=350ms恢复参考，但它不授权运动，必须新Depth恢复后参考实测轮速。
不改ReID、搜索、倒车、rotation-only、PID增益或60%总限速。未操作实车。
下轮先检查 `follow_wheel_config`，结合零指令、无基速、yaw重置、轮速周期与
距离扩大速度分析，不能仅凭新日志出现宣称改善。

## 2026-09-16：CAP1006～1085 相似人物、轨迹编号交换

先出方案后修复：排除证据按全帧唯一几何对应跨现存轨迹编号转移；swap 标记允许
携带已存在的排除，但不允许创造新排除/充当可信身份见证。搜索接回前统一计算
同帧 C0 对活动 UID 的距离，差距须至少0.05；这是额外拒绝门，不是确认授权。
包括尚无正式轨迹的检测框，不改变单人局部 ReID、运动参数及正常跟随路径。
日志新增 `identity_competition` 和排除转移来源；基线、代价、指标详见
`docs/reid_candidate_competition.md`。新增21项回归通过，全套1425通过、2项原有
搜索方向失败；独立身份库脚本 last_seen_frame 断言在去除本轮竞争门控后仍失败。
没有启动摄像头/电机，不能据此宣称实车误认率已下降。

## 2026-09-16：CAP217～366 约+0.36m持续偏差

当前物理样本普通距离PID获得显式有界追赶预算，避免1.8m要求29RPM被截20RPM；
仅正常跟随增加前进积分容量7.5m*s（Ki不变，I最多6RPM），倒车/目标带仍旧限额。
总上限60%、深度180ms、FF估计条件、ReID与搜索不变。新鲜样本与最终安全仲裁
仍是必要条件。基线、指标字段和测试边界见 `docs/follow_metrics.md` 最新章节。
未实车验证；全套1404通过，两项已有搜索方向失败未在本轮处理。

## 2026-09-16：CAP834 小框身份误接回修复

原始 YOLO 框在正式 ReID/跟踪、低分聚类及搜索观察入口统一做最小尺寸过滤。
正常/rotation-only 配置均为宽32、高80、面积3072（原始RGB像素，三项都满足）。
原始框保留录像/诊断；事件 `follow_bbox_size_rejected`。不改 PID/搜索方向规则。
范围、离线基线与回归测试见 `docs/follow_bbox_size.md`；未进行实车验证。

## 用户要求：后续每轮分析必须结合指标（2026-09-16）

每轮先用 `tools/follow_metrics.py` 统计用户指定CAP区间，明确运行目录、时间窗口、
样本分母及缺失值口径；报告距离误差/扩大速度、有效深度频率、前馈来源、零指令
占比和实测轮速，并与上一轮基线对比。不同长度的事件次数用每10秒频次辅助比较。
不要仅凭“日志出现了新功能”宣称改善，也不要把不同路径/走速当作严格A/B。
当前问题、指标基线及修复边界见 `docs/follow_metrics.md`；无实车新数据时只报告
离线验证结果，不把它当作距离保持已改善。分析请求不授权修改或运行电机。

## Hardware and Runtime

- Board: `root@172.16.16.103`
- Project: `/home/topeet/Desktop/rk_car_runtime_module`
- Main launcher: `./run_request_0428_modular.sh`
- Main runtime: `request_0513_modular.py`
- Normal follow config: `car_control_modular/config/reid_runtime.ini`
- Rotation-only config: `car_control_modular/config/reid_runtime_rotation_only.ini`
- Camera: Astra Pro RGB + Depth, with RGB selected through the stable Astra UVC by-id link `/dev/v4l/by-id/usb-Astra_Pro_HD_Camera_Astra_Pro_HD_Camera-video-index0` and Depth through the Astra module.
- Current normal run enables Astra Depth and disables mmWave at runtime. IMU may be initialized by the normal config, but it must not be used in the steering PID unless explicitly requested.

## Core Control Requirements

1. Keep the selected person's image center near the camera center.
2. Keep the person distance near 1.5 m.
3. Front/side IR safety has highest priority.
4. Hard brake is 0.5 m. Normal distance parking/release logic must not override IR safety.
5. Steering uses camera horizontal error as the outer loop and encoder-derived body yaw rate as the inner feedback loop. Do not add IMU feedback without a new explicit request.
6. Horizontal steering must remain available when Depth is missing. Distance confidence may limit longitudinal speed, but it must not reduce yaw authority to an ineffective 1-6 RPM.

## Distance Control Ranges and State Machine

The distance controller is centered on a nominal target of `1.50 m`. The target is not
treated as an exact single-point stop command; it uses separate enter/exit thresholds so
Depth noise and motor inertia do not make the vehicle alternate rapidly between forward
and reverse.

### Configured thresholds

| Range or threshold | Meaning | Longitudinal behavior |
|---|---|---|
| `> 1.80 m` | Forward start threshold | Start or resume forward following when the target is visible and the distance is fresh/credible. |
| `1.65-1.80 m` | Forward hysteresis band | Keep the current forward/hold state; do not repeatedly start and stop at a single noisy boundary. |
| `1.50 m` | Nominal following distance | The distance PID setpoint. |
| `1.35-1.65 m` | PID deadband / practical hold band | Normally command zero longitudinal speed (while horizontal steering remains active). This is the intended steady-state range around 1.5 m. |
| `< 1.30 m` | Reverse start threshold | Require fresh stable Depth and the configured confirmation count, then reverse away from the target. A fresh near reading may enter this state immediately for safety. |
| `1.30-1.45 m` | Reverse recovery band | Continue reversing until the distance reaches at least `1.45 m`; reverse speed is controlled by the distance PID plus approach-speed feed-forward. |
| `>= 1.45 m` while reversing | Reverse release threshold | Stop the reverse component and return to the hold state. The 0.10 m separation from the reverse start threshold is deliberate hysteresis. |
| `<= 0.50 m` | Hard distance safety reference | Treat as an emergency-distance condition and stop longitudinal motion. This does not replace the physical IR safety path. |

The exact values currently deployed in `reid_runtime.ini` are:

```ini
[distance]
target_distance_m = 1.5
target_distance_release_m = 1.5
target_distance_release_hold_sec = 0.50
target_distance_release_confirm_frames = 3
brake_distance_m = 0.5
forward_start_distance_m = 1.80
forward_stop_distance_m = 1.65
reverse_start_distance_m = 1.30
reverse_immediate_distance_m = 1.30
reverse_stop_distance_m = 1.45
reverse_full_speed_distance_m = 1.00
```

`parking_enable = false` in the normal configuration. Therefore `1.50 m` is the PID
setpoint and the `1.35-1.65 m` deadband is the practical hold region; it is not a
permanent parking latch. The physical front/side IR trigger always has higher priority
than every Depth or PID decision and must issue an immediate hard stop.

### Direction and speed rules

- Distance greater than `1.80 m` permits forward motion. The forward command is reduced
  as the target approaches `1.65 m` and is held at zero inside the deadband.
- Distance below `1.30 m` permits reverse motion. Reverse must not wait for a slow PID
  ramp when a fresh near reading confirms that the target is approaching; the current
  test configuration provides a `30 RPM` feed-forward floor, with a runtime cap of
  `60 RPM` and an absolute configured maximum of `100 RPM`.
- Reversing stops at `1.45 m`, not at exactly `1.50 m`, so small Depth fluctuations do
  not cause immediate forward/reverse chatter.
- A target distance change must not disable horizontal camera/encoder steering. The
  vehicle can command zero longitudinal speed and a non-zero differential wheel speed
  to re-center the target.
- When changing from forward to reverse, the transition stop must not clear the newly
  calculated reverse RPM. The first effective reverse command must contain the computed
  speed instead of an unintended zero.

### Depth confidence and stale-data behavior

Distance thresholds only apply to a fresh, credible measurement. Depth is handled in
three confidence levels:

1. **High confidence (`0-200 ms` fresh):** use the multi-region foreground depth and
   run the normal distance PID.
2. **Medium confidence (`200-600 ms` since the last valid sample):** estimate distance
   from the last measurement, encoder displacement, and target-box scale change. Limit
   longitudinal speed to `20 RPM`; keep horizontal steering active.
3. **Low confidence (`>600 ms` without a valid sample):** do not accelerate toward an
   old or background distance. Stop longitudinal motion or use only a tightly limited
   recovery command; keep camera-based horizontal correction and target identity alive.

A sudden near measurement can be accepted for braking/reverse, while a sudden far
measurement must be confirmed over consecutive fresh frames. A stale near distance must
not reject a new far anchor forever: the old anchor is strict for less than `0.60 s`,
down-weighted from `0.60-1.50 s`, and no longer used as a jump baseline after `1.50 s`.

### Safety priority

The priority order is fixed:

1. Front/side infrared trigger: immediate hard stop.
2. Distance at or below `0.50 m`: stop longitudinal motion.
3. Missing or low-confidence distance: never issue an old-distance forward command;
   preserve horizontal steering and apply the confidence speed limit.
4. Fresh valid distance: apply the forward/hold/reverse thresholds and PID above.

This table describes the intended stable behavior: the car should settle near `1.50 m`,
remain within roughly `1.35-1.65 m` during normal measurement noise, move forward only
when it is genuinely farther than the forward-start region, and reverse promptly when a
fresh target distance enters the near-distance region.

### Distance PID parameters

The range thresholds decide which direction is allowed; the PID determines how much
longitudinal speed is requested inside an allowed direction. The current deployed values
are:

```ini
[distance_pid]
enable = true
kp_rpm_per_m = 22.0
ki_rpm_per_m_s = 1.5
kd_rpm_s_per_m = 6.0
integral_limit_m_s = 1.5
deadband_m = 0.15
derivative_filter_alpha = 0.25
```

The signed error is `distance - 1.50 m`: positive error means the person is too far
away and permits forward speed; negative error means the person is too close and permits
reverse speed only after the reverse-start rules above are satisfied. Inside the
`+/-0.15 m` deadband the longitudinal request is zero, but the lateral camera/encoder
PID continues to run.

### Typical examples

| Measured distance | Expected interpretation |
|---|---|
| `2.10 m` | Clearly too far: start/continue forward, with speed limited as it approaches `1.65 m`. |
| `1.70 m` | Between forward stop and start: preserve the current state and avoid a new burst of motion. |
| `1.52 m` | Inside the hold band: no forward or reverse request; keep correcting horizontal position. |
| `1.38 m` | Still inside the hold band: do not reverse merely because the value is below `1.50 m`. |
| `1.24 m` | Below reverse start: confirm fresh Depth, then reverse promptly. |
| `1.42 m` while reversing | Continue reverse until the release threshold is reached. |
| `1.47 m` while reversing | Release reverse and return to hold; do not immediately start forward. |
| `0.48 m` | Safety distance reached: stop longitudinal motion and rely on IR as the physical hard-stop authority. |

### State transition order

For each fresh target/depth update, the controller follows this order:

```text
IR trigger or hard-distance condition
    -> immediate longitudinal STOP

fresh distance < 1.30 m
    -> reverse (after confirmation, unless an immediate near-safety reading applies)

fresh distance in 1.30-1.45 m while reversing
    -> keep reversing with bounded speed

fresh distance >= 1.45 m while reversing
    -> release reverse and hold

fresh distance in 1.35-1.65 m
    -> distance hold (zero longitudinal speed, lateral steering still enabled)

fresh distance > 1.80 m
    -> forward follow

stale/invalid distance
    -> confidence-limited estimate or longitudinal hold; never use an old value to
       accelerate blindly, while lateral steering remains active
```

The state machine deliberately has asymmetric protection: accepting a sudden near
reading quickly helps prevent a collision, while accepting a sudden far reading requires
consecutive confirmation so a background wall cannot make the car accelerate. The
distance state also never owns the horizontal wheel difference; distance hold must not
freeze the camera-centering controller.

## Depth Measurement

- Astra Depth is approximately 30 FPS.
- Current measurement uses five torso regions: chest center, abdomen center, left torso, right torso, and lower abdomen.
- Each region uses valid foreground depth clusters and chooses a history-continuous cluster.
- Dynamic valid-pixel threshold is `max(20, visible_roi_area * 3%)`.
- Old distance anchors become weak after 0.6 s and invalid after 1.5 s.
- Sudden near readings are accepted quickly for braking/reverse; sudden far readings require confirmation.
- A recent log showed stable multi-region depth around 1.55 m while the target was centered.

## Steering PID

- Normal config currently logs:
  - center band: `0.40..0.60`
  - camera HFOV: `60 deg`
  - outer PID max yaw: `46 deg/s`
  - correction max: `16 RPM`
  - encoder feedback interval: `100 ms`
  - feedback median window: `3`
- Rotation-only testing previously used a 30 RPM startup kick, released when encoder yaw reached 4 deg/s. That startup behavior is only for `reid_runtime_rotation_only.ini` and must not be assumed to be active in normal follow mode.
- Do not treat the fixed 30 RPM kick as the complete PID. It is only mechanical startup compensation.

### Rotation decision conditions

Rotation is decided independently from the distance controller. The distance loop chooses
the common forward/reverse component; the camera/encoder loop chooses the differential
wheel component. A distance-hold decision therefore does **not** mean that both wheels
must be stopped: the car may rotate in place or use a small differential to keep the
person centered.

The horizontal target coordinate is normalized to `x in [0, 1]`, with `0.5` as the image
center. The outer-loop error is:

```text
e_x = target_center_x - 0.50
```

After applying the configured motor sign mapping:

| Target position | Rotation decision |
|---|---|
| `0.40 <= x <= 0.60` | Target is inside the center band. Desired yaw is zero and wheel difference is zero, unless the encoder feedback still shows residual body yaw that needs braking. |
| `x < 0.40` | Target is left of center. Request left corrective yaw; the farther left and faster it moves, the larger the requested yaw rate. |
| `x > 0.60` | Target is right of center. Request right corrective yaw using the same rule. |
| `x < 0.20` or `x > 0.80` | Target is near an image edge. Reduce longitudinal speed if needed, but preserve strong differential steering so the target is not allowed to leave the frame. |
| No high-quality target for the loss-confirmation window | Do not steer from an untrusted small box. Keep the last reliable exit direction, stop stale translation, and enter search only after the configured consecutive-miss count. |

The center band is a control dead zone, not a detector gate. A person at `x=0.45` with
valid distance may remain stationary longitudinally while the car continues tracking
distance; a person at `x=0.30` must receive a steering command even if the distance is
already inside `1.35-1.65 m`.

### Encoder feedback loop

The camera position is the outer loop and the wheel encoder is the only yaw feedback used
in normal follow mode. IMU data is intentionally excluded from this loop. The per-cycle
decision is:

```text
camera x error
    -> desired yaw rate (outer PID)
desired yaw rate - encoder measured yaw rate
    -> differential wheel correction (inner PI/PID)
common distance speed +/- differential correction
    -> left and right wheel targets
```

The encoder yaw estimate comes from the left/right wheel displacement difference over the
feedback interval. The feedback is filtered with the configured median window before it
is used for correction. A correction in the opposite direction of the measured yaw is
active braking, not a new target direction; it must be bounded so the vehicle does not
oscillate across the center band.

The final wheel command must be formed by mixing the two independent components:

```text
left_wheel  = longitudinal_speed - yaw_correction
right_wheel = longitudinal_speed + yaw_correction
```

The yaw correction must not be clamped to the longitudinal speed. In particular, when
`longitudinal_speed = 0` inside the distance hold band, a non-zero left/right difference
is still valid and is required for in-place centering. A STOP may be sent for IR,
hard-distance, or confirmed target loss, but must not be inserted after every ordinary
steering update or after a same-direction PID update.

### Rotation safety and target-quality gates

- Only a high-quality, geometrically plausible target updates the steering error,
  `last_valid_bbox`, and exit-direction history.
- A tiny, highly elongated, or sudden area-collapse detection may be retained as a weak
  ReID sample, but cannot provide a new steering command, change the search direction, or
  reset the target-loss counter by itself.
- Depth validity controls longitudinal confidence only. `far_background_guard` or a
  short Depth gap must not reduce a valid target's horizontal correction to an ineffective
  1-6 RPM.
- A fresh valid target at the image edge must be turned toward immediately; waiting for a
  second frame before starting the correction is likely to let the target leave the frame.
- When target distance is safe and valid, a near-distance/reverse overlay must not replace
  a current horizontal PID command with a stale cached direction.
- When the target is lost, search uses the last reliable left/right trajectory. Search
  rotation pauses when a credible candidate is seen and resumes only after the candidate
  fails the confirmation check. The mechanical settle time is based on encoder angular
  speed, not a fixed short delay.

### Rotation log fields

Every steering decision should be diagnosable from one log record containing:

```text
target_center_x, e_x, center_band, target_quality
desired_yaw_rate, measured_encoder_yaw_rate, feedback_age_ms
yaw_error, yaw_correction_rpm
longitudinal_speed_rpm, left_target_rpm, right_target_rpm
depth_source, depth_age_ms, action_kind, stop_reason
```

The key acceptance condition is: while a reliable person remains visible, the car must
continue producing a timely differential correction whenever `x` is outside `0.40-0.60`,
even if the distance loop is holding at `1.50 m` or Depth is temporarily unavailable.

## Known Recent Log Behavior

The latest normal run was started with `./run_request_0428_modular.sh` and used `reid_runtime.ini`:

- Astra Depth enabled, mmWave disabled, `rotation_only=False`.
- Distance was initially stable and close to target.
- When the person moved laterally, the vehicle began to oscillate and look sluggish.
- Last log: `/home/topeet/Desktop/rk_car_runtime_module/run_request_0428_modular_logs/request_0513_modular.log`
- Recent log duration was about 30 seconds, ending at frame 384.
- Latest frame had target center ratio about `0.452`, Depth `1.55 m`, but the controller reason was `target_distance_hold` and it issued a forward action with zero speed before shutdown.
- The log contains many `visual_pid_left`, `visual_pid_right`, and `reverse_visual_pid` decisions. This must be separated into stable-target and lateral-motion intervals before changing gains.
- Some normal runs have shown repeated STOP followed within a few milliseconds by a steer action. This creates visible jolts and must be eliminated for the same-direction candidate/target state.
- Search settle waits can take roughly 0.6-2.2 s because the mechanical chassis takes time to stop; this matters only after target loss, not during normal visible tracking.

## Diagnostic Procedure

When a new log is available, read it directly on the board:

```bash
cd /home/topeet/Desktop/rk_car_runtime_module
rg -n "pipeline_timing|控制决策:|视觉转向PID:|reverse_visual_pid|Astra depth regions|编码器转向反馈|电机下发时序|进入刹车保持状态|target_distance_hold" \
  run_request_0428_modular_logs/request_0513_modular.log
```

Compare these fields during lateral motion:

- target center ratio and its frame-to-frame trend;
- visual error, filtered error, desired yaw rate;
- measured encoder yaw rate and feedback age;
- PID output wheel difference and any opposite/overspeed brake flag;
- final left/right wheel targets and action kind;
- whether a STOP is sent before a same-direction steering update;
- Depth source, sample age, and whether `reverse_visual_pid` is active;
- pipeline timing and stale-result discard status.

Do not change gains until the log identifies whether the oscillation is caused by:

1. camera/encoder latency and stale visual results;
2. excessive opposite-yaw braking or derivative noise;
3. normal forward/reverse overlay repeatedly replacing the yaw command;
4. action queue STOP insertion;
5. target/ReID box switching;
6. Depth hold/reverse state overriding a valid horizontal PID.

## Backups

- Rotation-only deployment backup on board: `/home/topeet/Desktop/rk_car_runtime_module/.codex_backup_rotation30_20260821_2155`

## Important Constraint

This document is a handoff summary, not a migration of the original Codex transcript. The VS Code ChatGPT plugin can use this file as project context, but it will not automatically know every prior conversational turn.
