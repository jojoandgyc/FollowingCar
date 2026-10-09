#!/usr/bin/env python3
"""Offline engineering diagrams. No application import, device I/O or runtime edits.

Run: python3 docs/follow_workflow_20261009/render_diagrams.py
Outputs: four PNG/SVG pairs, a PDF, and source provenance/labels in JSON.
Coordinates and wording below are the editable diagram source.
"""
from pathlib import Path
from html import escape
import hashlib
import json
import math
import zipfile
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
REPO = OUT.parent.parent
W = 2000
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
C = dict(bg="#F7F9FC", ink="#172B42", muted="#4F647A", line="#70859A",
         border="#CED8E3", white="#FFFFFF", blue="#155AB0", bluebg="#EAF2FF",
         green="#116E62", greenbg="#E8F5F1", amber="#996000", amberbg="#FFF3D9",
         red="#A63040", redbg="#FBEDEF", graybg="#EDF0F4")


@lru_cache(None)
def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size=size, index=2)


class Page:
    def __init__(self, number, name, title, subtitle, height=2000):
        self.number, self.name, self.height = number, name, height
        self.im = Image.new("RGB", (W, height), C["bg"])
        self.d = ImageDraw.Draw(self.im)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" role="img">',
                    f'<title>{escape(title)}</title><desc>{escape(subtitle)}</desc>',
                    f'<rect width="{W}" height="{height}" fill="{C["bg"]}"/>']
        self.labels = []
        self.text(60, 35, f"0{number} / {title}", 48, bold=True)
        self.text(60, 105, subtitle, 25, color=C["muted"])
        self.line([(60, 153), (1940, 153)], C["border"], 2)

    def text(self, x, y, text, size=27, color=None, bold=False, maxwidth=None):
        color = color or C["ink"]
        length = self.d.textlength(text, font=font(size, bold))
        if maxwidth is not None and length > maxwidth + 1:
            raise ValueError(f"Text too wide ({length:.0f}>{maxwidth}): {text}")
        if x < 0 or x + length > W or y < 0 or y + size > self.height:
            raise ValueError(f"Out of page: {text}")
        self.d.text((x, y), text, font=font(size, bold), fill=color, anchor="lt")
        # SVG and Pillow share a top-origin label; dominant baseline preserves editability.
        self.svg.append(f'<text x="{x}" y="{y}" font-family="Noto Sans CJK SC, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" dominant-baseline="text-before-edge" fill="{color}">{escape(text)}</text>')
        self.labels.append(text)
        return length

    def wrap(self, text, width, size=26, bold=False):
        lines = []
        for paragraph in text.split("\n"):
            line = ""
            for char in paragraph:
                if line and self.d.textlength(line + char, font=font(size, bold)) > width:
                    lines.append(line)
                    line = char
                else:
                    line += char
            lines.append(line)
        return lines

    def paragraph(self, x, y, text, width, size=26, color=None, lineheight=38, bold=False):
        lines = self.wrap(text, width, size, bold)
        for i, line in enumerate(lines):
            self.text(x, y + i * lineheight, line, size, color, bold, width)
        return y + len(lines) * lineheight

    def rect(self, x, y, w, h, fill=None, border=None, radius=16):
        fill, border = fill or C["white"], border or C["border"]
        self.d.rounded_rectangle((x, y, x+w, y+h), radius, fill, border, 2)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{border}" stroke-width="2"/>')

    def box(self, x, y, w, h, title, body="", kind="white", risk=None, size=26):
        self.rect(x, y, w, h, C[kind])
        reserve = 100 if risk else 0
        title_size = 31
        while self.d.textlength(title, font=font(title_size, True)) > w-44-reserve and title_size > 25:
            title_size -= 1
        self.text(x+22, y+16, title, title_size, bold=True, maxwidth=w-44-reserve)
        if risk:
            self.badge(x+w-102, y+17, risk)
        lineheight = size+8
        end = self.paragraph(x+22, y+58, body, w-44, size, lineheight=lineheight)
        if body and end-lineheight+size > y+h-12:
            raise ValueError(f"Box too short: {title}, end={end}, bottom={y+h}")

    def badge(self, x, y, text, kind="amber"):
        width = max(76, int(self.d.textlength(text, font=font(22, True)))+22)
        self.rect(x, y, width, 36, C[kind+"bg"], C[kind], 8)
        self.text(x+11, y+5, text, 22, color=C[kind], bold=True)

    def line(self, points, color=None, width=3, dashed=False):
        color = color or C["line"]
        if dashed:
            for a, b in zip(points, points[1:]):
                dx, dy = b[0]-a[0], b[1]-a[1]
                dist = math.hypot(dx, dy)
                for start in range(0, int(dist), 16):
                    stop = min(start+9, dist)
                    self.d.line([(a[0]+dx*start/dist, a[1]+dy*start/dist),
                                 (a[0]+dx*stop/dist, a[1]+dy*stop/dist)], fill=color, width=width)
        else:
            self.d.line(points, fill=color, width=width, joint="curve")
        dash = ' stroke-dasharray="9 7"' if dashed else ''
        self.svg.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="{width}"{dash}/>')

    def arrow(self, points, label=None, at=None, color=None, dashed=False):
        color = color or C["line"]
        self.line(points, color, 3, dashed)
        (px, py), (x, y) = points[-2:]
        angle = math.atan2(y-py, x-px)
        wing = [(x, y), (x-14*math.cos(angle-.45), y-14*math.sin(angle-.45)),
                (x-14*math.cos(angle+.45), y-14*math.sin(angle+.45))]
        self.d.polygon(wing, fill=color)
        self.svg.append(f'<polygon points="{" ".join(f"{a},{b}" for a,b in wing)}" fill="{color}"/>')
        if label:
            tx, ty = at
            width = self.d.textlength(label, font=font(23))
            self.rect(tx-6, ty-3, width+12, 32, C["bg"], C["bg"], 3)
            self.text(tx, ty, label, 23, color=color)

    def section(self, y, title, note=None):
        self.text(60, y, title, 33, bold=True)
        if note:
            self.text(60, y+48, note, 25, C["muted"])

    def footer(self, sources):
        self.line([(60, self.height-100), (1940, self.height-100)], C["border"], 2)
        self.text(60, self.height-82, sources, 20, C["muted"], maxwidth=1880)
        self.text(60, self.height-48, "2026-10-09 当前工作区 / 默认启动配置；环境覆盖另计。示意图，不是实测性能或停车距离保证。", 21, C["muted"])
        self.text(1840, self.height-48, f"{self.number} / 4", 23, C["muted"])

    def save(self):
        self.svg.append("</svg>")
        self.im.save(OUT/f"{self.name}.png", optimize=True)
        (OUT/f"{self.name}.svg").write_text("\n".join(self.svg), encoding="utf-8")
        return self


def architecture():
    p = Page(1, "01-system-flow", "从看见目标到车轮转动", "三条异步链汇合；①–⑤ 同号为跨栏接口。一个 CAP 不代表各线程同步完成；F1–F8 见图 4。", 1930)
    startup = [
        (60, "启动脚本 / 配置", "读取 INI 与环境覆盖\n自动派生实际距离阈值"),
        (540, "电机初始化 / CPU", "串口唯一占用、先零速停车\n0x005F=600，读回验证"),
        (1020, "首次锁定 / F1", "第一位唯一合格候选\n连续新帧建库，不盲目搜人"),
        (1500, "进入跟随", "身份、距离与执行条件通过\n才允许正常前进")]
    for x, title, body in startup:
        p.box(x, 180, 440, 154, title, body, "bluebg", size=24)
    for x in (500, 980, 1460):
        p.arrow([(x, 257), (x+40, 257)])
    for x, title in ((60, "A  视觉 / 身份 / 横向"), (720, "B  独立深度 / 纵向"), (1380, "C  执行 / 实际反馈")):
        p.text(x, 380, title, 31, bold=True)
    # Columns are aligned, and named ports avoid ambiguous crossing arrows.
    p.box(60, 440, 560, 130, "RGB 采集 / CPU", "Astra 640×480，标称 30fps\n记录采集时间和 CAP")
    p.box(60, 620, 560, 135, "YOLO11n / NPU", "输出 C0 人体框、检测置信度\nC0 只说明像人，不证明身份")
    p.arrow([(340,570),(340,620)])
    p.box(60, 805, 560, 177, "身份跟踪 / CPU + NPU", "快通道：当前框、颜色、几何\n全通道：OSNet、DeepSORT、身份库\n输出本帧可信 UID，而非只看轨迹号", risk="F1", size=25)
    p.arrow([(340,755),(340,805)])
    p.box(60, 1032, 560, 165, "提前发布可信观测 / CPU", "【①】UID、框、采集时间 → 深度\n【②】目标横坐标 → 横向控制\n普通新框不撤销同 UID 的在算任务", risk="F2", size=25)
    p.arrow([(340,982),(340,1032)])
    p.box(60, 1250, 560, 165, "横向控制 / CPU · 标称 30Hz", "像面偏差 → 转向修正 yaw\n发布独立时限的横向意图\n【③】yaw / 时间 / 版本 → 执行端", size=25)
    p.arrow([(340,1197),(340,1250)], "②", (350,1210))
    p.box(720, 440, 560, 130, "Depth 采集 / CPU", "保存物理采样时间与短历史\n标称 30fps ≠ 30Hz 合格距离")
    p.box(720, 620, 560, 135, "接收【①】可信目标框", "一个在算任务 + 一个最新待处理框\n启动 / 故障时保留受控同步回退", risk="F3", size=25)
    p.arrow([(1000,570),(1000,620)])
    p.box(720, 805, 560, 177, "目标区域测距 / CPU", "时间对齐、多区域有效深度、跳变确认\n去重、校验年龄、提交时复核 UID\n重复 / 旧观测不能续期", risk="F4", size=25)
    p.arrow([(1000,755),(1000,805)])
    p.box(720, 1032, 560, 200, "距离 PI + 运动授权 / CPU", "d−1.40m → PI 追赶请求\n共享制动、轮速、身份、时效检查\n【④】批准 base / 上限 / 截止时间\n人速估计当前不参与加减速", "greenbg", "F5", size=25)
    p.arrow([(1000,982),(1000,1032)])
    p.box(720, 1280, 560, 135, "独立深度目标跟踪旁路", "只观察、记录关联效果\n不能单独续身份或授权电机", "graybg", size=25)
    p.box(1380, 440, 560, 170, "编码器反馈 / CPU · 约 20Hz", "真实左右轮速、反馈时间、角速度\n【⑤】反馈 → 转向 / 制动 / 恢复\n不是上一条请求 RPM", risk="F8", size=25)
    p.box(1380, 680, 560, 185, "合成【③】+【④】/ CPU", "正常跟随统一轮速流 · 20Hz\nL = base + yaw；R = base − yaw\n普通转向更新不应清掉合法前进", "greenbg", "F6", size=25)
    p.arrow([(1660,610),(1660,680)], "⑤", (1670,630))
    p.box(1380, 925, 560, 170, "写入前最终核验 / CPU", "最新版本、实际轮速、STOP 所有权\n期限、换向保护、最终速度上限\n成功写入后回传实际执行预算", risk="F7", size=25)
    p.arrow([(1660,865),(1660,925)])
    p.box(1380, 1150, 560, 130, "RS485 / 驱动器", "右轮 ACK + 左轮 ACK → 完整回执\n写入失败 / 不确定状态 → 停车故障", size=25)
    p.arrow([(1660,1095),(1660,1150)])
    p.box(1380, 1330, 560, 120, "电机内部闭环 → 实际运动", "回执成功 ≠ 已达到目标轮速", "greenbg", size=25)
    p.arrow([(1660,1280),(1660,1330)])
    p.text(1410, 1478, "实际反馈回到【⑤】，构成闭环", 26, C["green"], bold=True)
    p.box(60, 1495, 1220, 170, "所有状态都受安全 / 所有权约束", "身份撤销、真实危险、驱动故障与程序退出可以抢占正常跟随。\n红外按方向保护：前方阻止前进；侧方确认后限制转向；搜索旋转有独立规则。\n搜索、近距旋转、倒车是独立状态路径，仍共用电机后端与串口所有权。", "redbg", size=26)
    p.box(1380, 1560, 560, 145, "异步诊断旁路", "录像、身份事件、零包审计、计时\n不产生授权，但资源竞争会增大延迟", "graybg", size=25)
    p.text(60, 1730, "关键边界：能显示距离 ≠ 有新测量；UID 缓存存在 ≠ 本帧已确认；PI 有请求 ≠ 指令已发送。", 29, bold=True, maxwidth=1880)
    p.footer("源码：request_0513_modular.py · rk_vision/pipeline.py · controllers.py · action_runtime.py · mssd_motor.py")
    return p.save()


def identity():
    p = Page(2, "02-identity-search", "身份确认、转身丢失与重新接回", "不要把“检测到同一轨迹”当成“确认还是同一个人”；身份输出、模板学习、电机授权是三件事。", 2110)
    p.box(60, 190, 560, 185, "首次启动：静止等待", "第一位唯一、质量合格的人体候选\n连续 2 张真实新图，外观 / 位置连续\n建立 UID 后仍须通过距离与电机条件", "bluebg", "F1", size=25)
    p.box(720, 190, 560, 185, "每次新图：YOLO / 跟踪", "过滤检测质量，区分强候选与弱观察\nraw track 连续不保证 UID 可信\n已排除背景不必强迫全画面只有一人", size=25)
    p.box(1380, 190, 560, 185, "当前检测可复用完整证明？", "已有 2 次完整验证，且新框连续\n没有新竞争者、交叉或证明过期\n满足才走快通道，否则完整核验", size=25)
    p.arrow([(620,280),(720,280)])
    p.arrow([(1280,280),(1380,280)])
    p.box(60, 455, 900, 188, "完整身份核验 / CPU + NPU", "全身与可比躯干 OSNet、颜色、几何、近期 / 长期模板、同帧身份竞争。\n低 ReID 距离只是相似证据，不能压过明确位置矛盾。\n躯干当前确认阈值 0.40；区域不可比 ≠ 已证明换人。", "bluebg", "F1", size=26)
    p.box(1040, 455, 900, 188, "有界快通道 / CPU", "当前框 + 颜色 + 几何接续；跳过本轮 OSNet，不写模板。\n最多连续 2 帧；距完整验证图像采集达 200ms 回全通道。\n完整证明自原图采集最长 600ms；快帧不能给它无限续期。", "greenbg", size=26)
    p.arrow([(1500,375),(1500,415),(510,415),(510,455)], "否 / 周期复核", (610,397))
    p.arrow([(1710,375),(1710,455)], "是", (1722,398))
    p.arrow([(510,643),(510,695),(1000,695),(1000,740)])
    p.arrow([(1490,643),(1490,695),(1000,695)])
    p.box(720, 740, 560, 130, "本帧身份决策", "可信通过 / 证据不足 / 明确身份冲突\n三种状态不可混为同一种“没检测到”", size=25)
    p.box(60, 940, 560, 183, "通过：本帧可信 UID", "发布测距框、更新可信方向历史\n进入图 3 的控制资格检查\n刚接回可跟随，但模板仍处隔离期", "greenbg", size=25)
    p.box(720, 940, 560, 183, "不足：待核验 / 弱观察", "通常本帧 UID0，内部仍可保留候选\n只保留有限观察，不给候选写模板\n特定已验证裁切接续另有严格入口", "amberbg", "F1", size=25)
    p.box(1380, 940, 560, 183, "冲突：撤销身份资格", "明确几何 / 外观 / 可信竞争矛盾\n负面证据关联候选，跨搜索保留\n两帧自身连续不能洗掉既有矛盾", "redbg", size=25)
    p.arrow([(810,870),(810,903),(340,903),(340,940)], "通过", (400,883))
    p.arrow([(1000,870),(1000,940)], "不足", (1012,890))
    p.arrow([(1190,870),(1190,903),(1660,903),(1660,940)], "冲突", (1440,883))
    p.box(60, 1200, 560, 220, "模板隔离 / 独立状态", "接回不立即给图库写入资格\n近期可靠库与长期代表分开老化\n稳定新证据、竞争 / 几何 / 区域验证\n达到隔离解除条件后才更新模板", "graybg", size=25)
    p.arrow([(340,1123),(340,1200)])
    p.box(720, 1200, 1220, 220, "无法持续确认 → 丢失 / 搜索原 UID", "撤销不合法前进；丢失确认后，按最新可信位置与转动补偿决定左右搜索。\n停车补看、候选居中、转向与搜索期限独立管理；停车期间仍可更新可信方向。\n发现候选 → 有界减速 / 停车补看 → 完整核验，不因“只有一个人”直接接回。\n当前默认不因丢失就换 UID；全周搜索失败或兜底超时则停车退出。", "bluebg", "F2", size=26)
    p.arrow([(1000,1123),(1000,1200)])
    p.arrow([(1660,1123),(1660,1200)])
    p.box(60, 1500, 900, 238, "保留旧保护锚点", "用途：检查已经成立的跨人 / 跨侧矛盾。\n参考过期只表示无法精确预测，不自动等于身份冲突。\n不能要求过期锚点先证明连续，才允许新候选开始观察。\n历史 CAP844 类“旧锚点自锁”已有专门恢复入口。", "amberbg", "F1", size=26)
    p.box(1040, 1500, 900, 238, "独立候选轨迹与接回确认", "保存局部连续性；重新检查身份、竞争、区域与矛盾。\n晚到 / 弱候选首帧只观察；严格强匹配另有单帧入口。\n核验通过 → 本帧 UID → 模板隔离 → 控制重新核验。\n候选穿过画面中心不应直接改写原可信搜索方向。", "greenbg", size=26)
    p.arrow([(1330,1420),(1330,1450),(510,1450),(510,1500)])
    p.arrow([(1640,1420),(1640,1500)])
    p.text(60, 1810, "最容易卡的场景：转身 / 背身、近景裁切变远景、相似衣着、多人交叉、隔离期证据更新不足。", 29, bold=True, maxwidth=1880)
    p.paragraph(60, 1865, "已有针对性保护与恢复代码 ≠ 所有场景已验证。验收必须同时看“正确人有限时间接回”和“错误人不接回”，不能只统计 ReID 是否变小。", 1880, 27)
    p.footer("源码：rk_vision/{initial_enrollment,pipeline,identity_bank,template_memory,reacquire_quarantine}.py · controllers.py")
    return p.save()


def motion():
    p = Page(3, "03-motion-decisions", "每一条前进、转向与零速如何产生", "按当前默认主路径绘制；特别区分请求、批准、写入与反馈。零速不是单一开关。", 2170)
    p.box(60, 190, 900, 175, "① 新距离进入纵向控制", "当前 UID 合法 + 真实新 Depth + 距离质量通过 + 样本年龄≤180ms。\n重复 / 乱序只可保留显示状态，不增加积分、不刷新授权。\n普通尝试失败不等于当前授权立即失效；仍检查原截止时间。", "bluebg", "F2", size=26)
    p.box(1040, 190, 900, 175, "② 当前横向意图", "可信目标的横向偏差 → PID / 转向不足反馈 → yaw。\n普通意图 150ms；受限桥接另检发布 / 采集时限和反馈。\n过期 yaw 应撤掉，但不能据此把独立合法的 base 一并清零。", "bluebg", size=26)
    p.box(60, 435, 900, 234, "距离 PI 产生“请求”", "误差 e=d−1.40m，扣 3cm 死区；P + 有界积分。\nKp=3/s，Ki=0.4/s²；不使用目标人速前馈来调节速度。\n起步追赶请求最高 180RPM，随误差渐变，并非恒定 180。\n软件变化率、抗积分饱和、恢复规则仍可能限制请求。", "greenbg", "F7", size=26)
    p.arrow([(510,365),(510,435)])
    p.box(1040, 435, 900, 234, "安全与特殊状态先决条件", "未完成首次锁定、明确危险、驱动故障、退出 → 阻止运动。\n近距优先仅旋转；当前 PI 入口附加 d<1.37m 等条件。\n倒车须独立深度确认 / 期限 / 换向保护，不是近了必倒。\n真实反转、停车所有权、红外方向限制不能由普通交接绕过。", "redbg", size=26)
    p.box(60, 739, 900, 233, "共享制动与时效 → 产生“批准”", "使用可信原始距离、实际轮速、已执行行程、响应与减速预算。\n1.10m 是模型试验边界，不是任何速度下的实际最小距离保证。\n批准 base ≤ 请求与当前制动上限；深度原采样+300ms 截止。\n180–300ms 仅旧授权受限接续；身份 / 轮速可提前撤销。", "amberbg", "F5", size=26)
    p.arrow([(510,669),(510,739)])
    p.box(1040, 739, 900, 233, "状态变化后怎么处理", "新鲜证据通过 → 当前请求独立再准入，不继承旧失败结论。\n短时失权不应每次都把新请求压回 0 或固定极低速。\n无新可信证据 → 不能无限保持旧前进；必须遵守原期限。\n只丢纵向时，不能把剩余差速直接变成未检查的反向原地转。", "bluebg", "F7", size=26)
    p.arrow([(1490,669),(1490,739)])
    p.box(60, 1050, 1880, 184, "③ 正常跟随合成轮包 / 20Hz", "归一化前进轮速：左轮 L=base+yaw，右轮 R=base−yaw；yaw>0 表示右转。\n例如 L10 / R20 → base 增加 10 → L20 / R30：保留同一转向差，不需要先停车。\n当前普通单轮修正上限 10RPM，双轮差上限 20RPM；串口数值再做安装方向符号转换。", "greenbg", "F6", size=28)
    p.arrow([(510,972),(510,1050)], "base", (523,996))
    p.arrow([(1490,972),(1490,1050)], "yaw / 状态 / 反馈", (1500,996))
    p.box(60, 1310, 900, 242, "④ 最终写入前：统一当前快照", "同 UID 的正常更新 → 重新组合并限速；不默认发送 0/0。\n旧零计划不能覆盖新授权；旧 yaw 不可用时独立审查直行。\n有界重算耗尽后，仅一次新鲜直行再准入，不借旧授权续命。\n检查反馈≤150ms、深度 / 身份期限、实际反转、STOP 与故障。", "bluebg", "F6", size=26)
    p.box(1040, 1310, 900, 242, "⑤ 串口写入与执行确认", "通过 → 发送左右轮，双 ACK 完成才记录完整执行回执。\n普通状态更新尽量合并；安全停车不等待 20Hz 周期。\n失败 / 不确定写入 → 撤销或故障停车，不能伪记为执行成功。\n0x005F=600RPM/s 是驱动设置；实际响应必须看轮速反馈。", "greenbg", "F8", size=26)
    p.arrow([(510,1234),(510,1310)])
    p.arrow([(960,1430),(1040,1430)])
    p.section(1625, "四种“零”必须分开记录，否则会反复误修")
    items = [(60,"控制请求 = 0","PI / 距离分支当前不需要前进"),
             (540,"实际写出 0/0","失效、核验失败或正常减速结果"),
             (1020,"STOP / 驻车","独立所有权与释放条件"),
             (1500,"轮速反馈 ≈ 0","车轮真正停稳，需要新鲜反馈")]
    for x,title,body in items:
        p.box(x, 1690, 440, 135, title, body, "graybg", size=24)
    p.paragraph(60, 1880, "普通驻车常有至少 500ms 保持及释放核验；特定新鲜前进路径可提前解除。每个 0/0 并不都触发驻车，故障 STOP 也不会等 500ms 自动解除。", 1880, 28)
    p.paragraph(60, 1970, "自动派生后 start/stop 变量约为 1.48/1.43m，但距离 PI 绕过旧前进滞回入口；不能把这两个变量画成所有 PI 指令的统一开关。", 1880, 26, C["muted"])
    p.footer("源码：distance_pi.py · sample_braking.py · controllers.py · lateral_intent.py · action_runtime.py · mssd_motor.py")
    return p.save()


def risks():
    p = Page(4, "04-stalls-and-ownership", "卡顿定位图与多人协作边界", "从第一条异常实际零包，沿 UID / CAP / 采样时间向上追。下列“负责”为建议分工；历史故障不等于当前已复现。", 2250)
    p.text(60, 184, "风险卡片：蓝色＝已有针对性修复，须实车复验；琥珀色＝持续敏感边界。必要安全停车仍须保留。", 28, bold=True)
    # Compact failure cards: symptom, mechanism, logs, owner, status.
    cards = [
        (60, 255, "F1  身份链", "转身后 UID0 / 重新搜索 / 距离消失", "裁剪不可比、模板不足、弱竞争、隔离期或旧锚点切换。", "查：身份 reason、full/partial、geometry、quarantine。", "负责：视觉身份组 / rk_vision；已有恢复修复，场景仍须回归。", "amberbg"),
        (1040, 255, "F2  时钟与交付", "有框 / 有距离，却不能授权", "采集时钟与处理完成混用；500ms 框窗不能替代新深度资格。", "查：capture_result_age、ROI age、sample age、截止时间。", "负责：感知调度组；早发布已接入，上游迟到仍可能发生。", "amberbg"),
        (60, 505, "F3  计算与提交", "独立深度标称 30Hz，合格输出却稀疏", "选帧 / 像素处理耗时与调度长尾，算完时样本已失效。", "查：depth transaction 的 wall / CPU、commit wait、context。", "负责：深度组；单任务+最新邮箱、锁外测距、计时已补。", "bluebg"),
        (1040, 505, "F4  确认等待", "深度出现几次，但新距离迟迟不被接受", "大裁切 / 跳变进入多次确认；重复物理样本不能增加计数。", "查：pending 1/3→2/3→accepted、duplicate、样本 ID。", "负责：深度组；去重是必要保护，确认重启与间隔仍须追查。", "amberbg"),
        (60, 755, "F5  制动 / 驻车", "距离仍远却限速为 0，之后持续等待", "制动预算、实际动量、反馈余量、驻车所有权叠加。", "查：remaining / required、brake veto、park release。", "负责：控制+实测组；已有余量去重，不等于完成制动标定。", "amberbg"),
        (1040, 755, "F6  发布 / 写入交接", "PI 有前进请求，实际却写出 0/0", "旧零计划、新授权、转向版本并发；不能保留旧包就误清零。", "查：publication、terminal、motor_zero_audit、packet_written。", "负责：执行组；已补统一再核验/独立直行，实车效果待复验。", "bluebg"),
        (60, 1005, "F7  恢复放大", "零包之后只能个位数，刚提速又掉下去", "低实测轮速、旧失权状态、固定恢复步长和量化相互放大。", "查：PI requested / approved、recovery reason、feedback。", "负责：控制+执行组；已补新证据接续，真实反转仍必须拦截。", "bluebg"),
        (1040, 1005, "F8  通信 / 实际响应", "已发高 RPM，但车轮迟到或反馈过期", "串口争用、长事务、后台竞争；驱动加速不等于指令阶跃。", "查：motor lock wait / max hold、ACK、反馈年龄和实测 RPM。", "负责：驱动组；独占/在途预算已补，勿同时开第二控制程序。", "amberbg"),
    ]
    for x,y,title,symptom,cause,logs,owner,kind in cards:
        p.rect(x,y,900,225,C[kind])
        p.text(x+20,y+16,title,31,bold=True)
        yy=p.paragraph(x+20,y+62,symptom,860,26,bold=True,lineheight=35)
        yy=p.paragraph(x+20,yy+7,cause,860,24,lineheight=32)
        yy=p.paragraph(x+20,yy+3,logs,860,23,lineheight=31)
        yy=p.paragraph(x+20,yy+3,owner,860,22,C["muted"],lineheight=30)
        if yy>y+225:
            raise ValueError(f"Risk overflow: {title}")
    p.section(1285, "不同期限不能互相替代", "这是资格上限，不是保证获准时长。新鲜轮速、普通新框或重发命令都不能替旧深度续命。")
    rows=[
        ("新 Depth / 旧前进授权", "180ms / 最长 300ms", "按原物理采样时间；180–300ms 仅受限旧授权接续"),
        ("测距框调度 / 可见证据", "各自最长 500ms", "框>250ms 仅历史选帧；采样关联与新鲜度仍各限 180ms"),
        ("普通横向 / 受限桥接", "150ms / 双时钟约束", "桥接另限发布 220ms、采集 350ms，且要求更鲜轮速"),
        ("最终轮速反馈", "150ms", "不是驱动响应时间；反馈恢复不自动恢复旧授权"),
        ("PI 状态记忆 / 完成执行历史", "350ms / 650ms", "只用于状态接续或行程预算，不是运动授权"),
    ]
    widths=[520,430,930]
    for i,(a,b,c) in enumerate(rows):
        y=1400+i*69
        if i%2==0:
            p.rect(60,y-8,1880,63,C["graybg"],C["graybg"],4)
        p.text(75,y,a,25,bold=True,maxwidth=500)
        p.text(595,y,b,25,maxwidth=410)
        p.text(1025,y,c,23,maxwidth=900)
    p.section(1780, "团队共同验收：跨模块追踪一条指令，而不是各自只测函数")
    chain=["可信 UID / CAP","物理深度采样","PI 请求 / 批准","最终零包原因","双 ACK / 实测"]
    for i,title in enumerate(chain):
        x=60+i*382
        p.box(x,1845,350,93,title,"", "white")
        if i<4:
            p.arrow([(x+350,1891),(x+382,1891)])
    p.paragraph(60,1975,"必须同时覆盖：直行↔左右转、旧零包遇新授权、反馈迟到后恢复、深度确认完成、目标转身接回，以及真正近距 / 危险 / 反转停车。",1880,27)
    p.paragraph(60,2059,"证据示例来自 2026-10-09 13:34 旧运行 CAP125–367：深度扫描约 152–159ms 后过期、重复样本确认等待、最终写入交接。后续代码已有修改，不能当作新版本仍原样复现。",1880,24,C["muted"],lineheight=34)
    p.footer("源码索引与生成时 SHA256 见 sources.json；F 编号是排查分类，不表示每一个零包都错误，也不保证问题已全部消除。")
    return p.save()


SOURCES = {
    "startup": [("request_0513_modular.py",15492),("car_control_modular/config/reid_runtime.ini",1272)],
    "identity": [("rk_vision/pipeline.py",1034),("rk_vision/identity_bank.py",995),("rk_vision/identity_bank.py",1180),
                 ("rk_vision/initial_enrollment.py",179),("rk_vision/detector_continuation.py",160),
                 ("rk_vision/reacquire_quarantine.py",1),("rk_vision/template_memory.py",1)],
    "depth": [("car_control_modular/depth_roi_policy.py",45),("car_control_modular/depth_async_scheduler.py",154),
              ("car_control_modular/depth_measurement_transaction.py",112),("request_0513_modular.py",6774),
              ("car_control_modular/astra_depth.py",1307)],
    "control": [("car_control_modular/distance_pi.py",399),("car_control_modular/distance_pi.py",835),
                ("car_control_modular/controllers.py",3351),("car_control_modular/controllers.py",6976),
                ("car_control_modular/sample_braking.py",17),("car_control_modular/lateral_intent.py",150),
                ("car_control_modular/detector_identity_lease.py",67)],
    "execution": [("car_control_modular/action_runtime.py",4557),("car_control_modular/action_runtime.py",6884),
                  ("car_control_modular/mssd_motor.py",511),("car_control_modular/near_yaw_parking.py",22),
                  ("car_control_modular/motor_ramp.py",1)],
    "historical_evidence": [("run_request_0428_modular_logs/run_20261009_133438_15872_3da03a39/request_0513_modular.log",2800),
                            ("run_request_0428_modular_logs/run_20261009_133438_15872_3da03a39/request_0513_modular.log",4397)],
}


def main():
    pages=[architecture(),identity(),motion(),risks()]
    pages[0].im.save(OUT/"follow-workflow-review.pdf", "PDF", resolution=120,
                     save_all=True, append_images=[p.im for p in pages[1:]],
                     title="目标跟随流程与卡顿排查 / 2026-10-09", author="Project workflow review")
    provenance={"snapshot_date":"2026-10-09", "scope":"Current working tree and default launcher configuration; no hardware operated.",
                "limitations":"High-level decision graph with key exceptions, not an exhaustive list of every conditional. Historical evidence predates latest repairs.",
                "renderer":"render_diagrams.py; Pillow raster and editable text/vector SVG", "sources":{},
                "pages":[{"file":p.name,"width":W,"height":p.height,"labels":p.labels} for p in pages]}
    for group,refs in SOURCES.items():
        provenance["sources"][group]=[]
        for relative,line in refs:
            path=REPO/relative
            if not path.is_file():
                raise FileNotFoundError(path)
            provenance["sources"][group].append({"file":relative,"line":line,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT/"sources.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    artifacts=[OUT/f"{p.name}.{ext}" for p in pages for ext in ("png","svg")]
    artifacts += [OUT/"follow-workflow-review.pdf",OUT/"sources.json",Path(__file__).resolve()]
    with zipfile.ZipFile(OUT/"follow-workflow-team-pack.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for artifact in artifacts:
            archive.write(artifact,arcname=f"follow_workflow_20261009/{artifact.name}")
    print(f"Generated {len(pages)} PNG/SVG diagrams, PDF and provenance in {OUT}")


if __name__ == "__main__":
    main()
