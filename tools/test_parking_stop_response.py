#!/usr/bin/env python3
"""Measured 60/100 RPM -> 5/10 A NORMAL for 300/500/700/900ms -> 0A/zero-speed.

Default is hardware-free planning. Every powered trial requires confirmation.
Encoder stability is not proof of zero ground slip or a stationary camera.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.test_forward_step_response import (
    ForwardSession, acquire_test_lock, _json_safe,
)
from tools.imu_turn_calibrate import DEFAULT_CONFIG, _ensure_follow_runtime_stopped, _unwrap_i32_delta

SIDES = ("left", "right")
PERIOD = .02
MAX_GAP = .15
QUIET_RPM = 1
QUIET_WINDOW = .10


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--execute", action="store_true")
    p.add_argument("--motion", choices=("forward", "left", "right"),
                   help="明确选择直行或原地旋转；RPM指每个轮子的绝对转速")
    p.add_argument("--setup", choices=("bench", "ground"), help="架空固定台架或平整地面")
    p.add_argument("--rpms", default="60,100")
    p.add_argument("--currents", default="5,10")
    p.add_argument("--holds-ms", default="300,500", help="300/500/700/900，逗号分隔；默认仍为300,500")
    p.add_argument("--rated-current-a", type=float,
                   help="现场核实的电机和控制器安全电流上限，不能仅根据软件寄存器范围填写")
    p.add_argument("--drive-timeout", type=float, default=2., help="未达到实测目标则终止，秒")
    p.add_argument("--max-travel", type=float, default=2., help="每次各轮累计编码器路程上限，米")
    p.add_argument("--wheel-diameter", type=float, default=.26)
    p.add_argument("--output-dir", default=str(ROOT / "calibration" / "parking_stop"))
    return p


def validate(args):
    combinations = []
    values = []
    for field, allowed in (("rpms", (60, 100)), ("currents", (5, 10)), ("holds_ms", (300, 500, 700, 900))):
        parsed = [int(x.strip()) for x in getattr(args, field).split(",")]
        if not parsed or len(set(parsed)) != len(parsed) or not set(parsed) <= set(allowed):
            raise ValueError(f"{field} 仅允许 {allowed}，不得重复")
        values.append(sorted(parsed))
    for field, lo, hi in (("drive_timeout", .5, 3.), ("max_travel", .2, 3.),
                          ("wheel_diameter", .1, .5)):
        v = getattr(args, field)
        if not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError(f"{field} 必须在 {lo}..{hi}")
    rating = args.rated_current_a
    if rating is not None and (not math.isfinite(rating) or not max(values[1]) <= rating <= 30):
        raise ValueError("安全电流上限必须覆盖测试电流且不超过30A；10A并非默认已验证安全")
    if args.execute and (args.motion is None or args.setup is None or rating is None):
        raise ValueError("实测必须指定 --motion、--setup 和已核实的 --rated-current-a")
    for rpm in values[0]:
        for current in values[1]:
            for hold in values[2]:
                combinations.append((rpm, current, hold))
    return combinations


def targets(rpm, motion):
    if motion == "forward": return {"left": rpm, "right": rpm}
    if motion == "left": return {"left": -rpm, "right": rpm}
    if motion == "right": return {"left": rpm, "right": -rpm}
    raise ValueError("未知运动方向")


class ParkingSession(ForwardSession):
    """Reuse checked controller mode, sign mapping, port setup and cleanup."""

    def read_wheel(self, side):
        start = time.monotonic()
        s = self.backend.driver.read_motor_status(side)
        return dict(side=side, read_started=start, timestamp=time.monotonic(),
            raw_rpm=s.speed_rpm, forward_rpm=self.forward_signs[side]*s.speed_rpm,
            position_deg=s.position_degree, current_a=s.phase_current_a,
            temperature_c=s.board_temperature_c, error_code=s.error_code)

    def event(self, trial, kind, side, value, operation):
        e = dict(kind=kind, side=side, value=value, started=time.monotonic(), completed=None)
        trial["events"].append(e)
        operation()
        e["completed"] = time.monotonic()
        if e["completed"]-e["started"] > MAX_GAP:
            raise RuntimeError(f"{kind}/{side} 通讯过慢，终止")

    def current(self, value, trial):
        for side in ("right", "left"):
            name = side+"_parking_current"
            self.event(trial, "current", side, value,
                lambda n=name: self.backend.driver.write_register(n, value, persist=False))
        for side in ("right", "left"):
            actual = float(self.backend.driver.read_register(side+"_parking_current"))
            trial["events"].append(dict(kind="current_readback", side=side,
                                        value=actual, completed=time.monotonic()))
            if not math.isfinite(actual) or abs(actual-value) > .01:
                raise RuntimeError("驻车电流回读不匹配")

    def speed(self, desired, trial):
        for side in ("right", "left"):
            raw = int(desired[side]*self.forward_signs[side])
            if abs(raw) > 100:
                raise ValueError("测试转速上限100RPM")
            self.event(trial, "speed", side, raw,
                lambda s=side, r=raw: getattr(self.backend.driver, "set_"+s+"_speed")(r))

    def normal(self, trial):
        # Deliberately NO zero-speed before/after NORMAL: measure this stop,
        # not software coast deceleration or accidental parking cancellation.
        for side in ("right", "left"):
            self.event(trial, "normal", side, 0, lambda s=side: self.backend.driver.stop(s, 0))


def window_stable(rows, start, end):
    """Three-valued: missing/coarse evidence is unknown, never a pass."""
    rows = [r for r in rows if r.get("trusted") and r["read_started"] >= start
            and r["timestamp"] <= end]
    if (len(rows) < 3 or rows[-1]["timestamp"]-rows[0]["timestamp"] < .05
            or end-rows[-1]["timestamp"] > .06
            or rows[0]["read_started"]-start > .06
            or any(b["timestamp"]-a["timestamp"] > .075 for a,b in zip(rows, rows[1:]))):
        return None
    displacement = [0]
    for a,b in zip(rows, rows[1:]):
        displacement.append(displacement[-1]+_unwrap_i32_delta(b["position_deg"], a["position_deg"]))
    return (all(abs(r["forward_rpm"]) <= QUIET_RPM for r in rows)
            and max(displacement)-min(displacement) <= 2
            and sum(abs(b-a) for a,b in zip(displacement, displacement[1:])) <= 2)


def analyze(trial):
    result = {}
    end = trial.get("hold_deadline")
    release = trial.get("release_completed")
    for side in SIDES:
        rows = [r for r in trial["samples"] if r["side"] == side]
        hold = [r for r in rows if r["phase"] == "hold"]
        post = [r for r in rows if r["phase"] == "released"]
        held = window_stable(hold, end-QUIET_WINDOW, end) if end else None
        lag = trial.get("release_started", float("inf"))-(end or 0)
        if lag < 0 or lag > .05: held = None
        post_end = trial.get("post_deadline")
        # Require stability throughout the measured post-release interval,
        # not just its last zero-speed sample.
        released = window_stable(post, release, post_end) if release and post_end else None
        quiet_at = None
        for row in hold:
            if window_stable(hold, row["timestamp"]-QUIET_WINDOW, row["timestamp"]) is True:
                quiet_at = (row["timestamp"]-trial["normal_completed"])*1000
                break
        result[side] = dict(stable_by_hold_deadline=held, stable_after_release=released,
            first_confirmed_quiet_ms=quiet_at,
            last_hold_rpm=hold[-1]["forward_rpm"] if hold else None,
            peak_hold_current_a=max((r["current_a"] for r in hold), default=None))
        normal = next((e for e in trial["events"] if e["kind"] == "normal"
                       and e["side"] == side and e.get("completed") is not None), None)
        clear = next((e for e in trial["events"] if e["kind"] == "current"
                      and e["side"] == side and e["value"] == 0
                      and e.get("started", 0) >= trial.get("release_started", float("inf"))), None)
        if normal and clear and clear.get("completed") is not None:
            result[side]["actual_hold_interval_ms"] = [
                (clear["started"]-normal["completed"])*1000,
                (clear["completed"]-normal["started"])*1000]
    vals = [v[k] for v in result.values() for k in ("stable_by_hold_deadline", "stable_after_release")]
    verdict = "not_stable" if False in vals else "inconclusive" if None in vals else "encoder_stable"
    if trial.get("status") != "complete": verdict = "aborted"
    return dict(verdict=verdict, wheels=result,
                note="编码器稳定不等于车身/摄像头完全静止；无地面滑移测量")


def run_trial(session, args, trial, clock=time.monotonic, sleep=time.sleep):
    desired = targets(trial["rpm"], args.motion)
    last, travel = {}, {s: 0. for s in SIDES}
    started = clock()

    def sample(phase):
        for side in SIDES:
            row = session.read_wheel(side)
            row.update(phase=phase, trusted=False)
            trial["samples"].append(row)  # retain fault sample
            prev = last.get(side)
            if (not all(math.isfinite(float(row[k])) for k in ("read_started", "timestamp", "forward_rpm",
                    "position_deg", "current_a", "temperature_c", "error_code"))
                    or not 0 <= clock()-row["timestamp"] <= MAX_GAP
                    or not 0 <= row["timestamp"]-row["read_started"] <= MAX_GAP):
                raise RuntimeError("无效/过期反馈")
            if row["error_code"] or row["temperature_c"] >= 60 or abs(row["current_a"]) > args.rated_current_a:
                raise RuntimeError("电机故障、温度达到60℃或电流超限")
            if abs(row["forward_rpm"]) > trial["rpm"]+10:
                raise RuntimeError("实测轮速超限")
            if phase == "drive" and row["forward_rpm"]*desired[side] < -2*trial["rpm"]:
                raise RuntimeError("运动方向错误")
            if prev:
                dt = row["timestamp"]-prev["timestamp"]
                # Setting+reading current and the two stop writes form an
                # explicitly timed I/O gap. No such gap may establish quiet.
                max_gap = .35 if phase != prev["phase"] else MAX_GAP
                if not 0 < dt <= max_gap: raise RuntimeError("反馈间隔过长")
                delta = _unwrap_i32_delta(row["position_deg"], prev["position_deg"])
                if abs(delta) > 110*6*dt+5: raise RuntimeError("编码器突跳")
                travel[side] += abs(delta)/360*math.pi*args.wheel_diameter
                if travel[side] >= args.max_travel: raise RuntimeError("达到单次路程上限")
            row["path_m"] = travel[side]
            row["trusted"] = True
            last[side] = row
        if clock()-started > 8: raise RuntimeError("单次测试超过8秒")

    def observe_until(deadline, phase):
        while clock() < deadline:
            sample(phase)
            sleep(min(PERIOD, max(0, deadline-clock())))

    session.current(0, trial)
    session.speed({s: 0 for s in SIDES}, trial)
    observe_until(clock()+.35, "baseline")
    if any(window_stable([r for r in trial["samples"] if r["side"] == s], clock()-.15, clock()) is not True
           for s in SIDES): raise RuntimeError("启动前未确认双轮稳定")
    session.speed(desired, trial)
    drive_deadline = clock()+args.drive_timeout
    at_speed_since = None
    at_speed_samples = None
    while clock() < drive_deadline:
        sample("drive")
        good = all(abs(last[s]["forward_rpm"]-desired[s]) <= max(3, trial["rpm"]*.05) for s in SIDES)
        if not good:
            at_speed_since = at_speed_samples = None
        elif at_speed_since is None:
            at_speed_since, at_speed_samples = clock(), last.copy()
        elif clock()-at_speed_since >= .2:
            for side in SIDES:
                previous = at_speed_samples[side]
                dt = last[side]["timestamp"]-previous["timestamp"]
                delta = _unwrap_i32_delta(last[side]["position_deg"], previous["position_deg"])
                normalized = delta*session.forward_signs[side]*(1 if desired[side] > 0 else -1)
                if not .5*trial["rpm"]*6*dt <= normalized <= 1.5*trial["rpm"]*6*dt:
                    raise RuntimeError("转速反馈与编码器位移不一致，不能确认达到实际RPM")
            break
        sleep(PERIOD)
    else:
        raise RuntimeError("实测双轮未持续200ms达到目标转速；不能当作该RPM停车试验")
    trial["achieved_rpm"] = {s: last[s]["forward_rpm"] for s in SIDES}
    trial["current_prepare_started"] = clock()
    session.current(trial["current_a"], trial)
    # Register preparation may have changed speed. Validate again before NORMAL.
    sample("armed")
    if any(abs(last[s]["forward_rpm"]-desired[s]) > max(3, trial["rpm"]*.05) for s in SIDES):
        raise RuntimeError("设置电流后实际转速不再满足测试档位")
    trial["pre_stop_rpm"] = {s: last[s]["forward_rpm"] for s in SIDES}
    trial["normal_started"] = clock()
    session.normal(trial)
    trial["normal_completed"] = clock()
    trial["hold_deadline"] = clock()+trial["hold_ms"]/1000
    observe_until(trial["hold_deadline"], "hold")
    trial["release_started"] = clock()
    session.current(0, trial)
    session.speed({s: 0 for s in SIDES}, trial)
    trial["release_completed"] = clock()
    trial["post_deadline"] = clock()+.5
    observe_until(trial["post_deadline"], "released")
    trial["status"] = "complete"


def save(directory, payload):
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / (datetime.now().strftime("parking_%Y%m%d_%H%M%S_%f_")+str(os.getpid()))
    for trial in payload["trials"]: trial["analysis"] = analyze(trial)
    with stem.with_suffix(".json").open("x", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2, allow_nan=False)
    fields = ["trial", "target_rpm", "parking_current_a", "hold_ms", "phase", "side", "read_started", "timestamp", "raw_rpm", "forward_rpm",
              "position_deg", "current_a", "temperature_c", "error_code", "path_m", "trusted"]
    with stem.with_suffix(".csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, trial in enumerate(payload["trials"], 1):
            writer.writerows(dict(trial=i, target_rpm=trial["rpm"], parking_current_a=trial["current_a"],
                                  hold_ms=trial["hold_ms"], **r) for r in trial["samples"])
    return stem.with_suffix(".json")


def main(argv=None):
    args = parser().parse_args(argv)
    try: plan = validate(args)
    except ValueError as exc: print(str(exc), file=sys.stderr); return 2
    print("停车测试计划（每轮RPM / 驻车电流A / 保持ms）：", plan)
    if not args.execute:
        print("仅预览，未打开串口。实测需 --execute --motion ... --setup ... --rated-current-a ...")
        return 0
    from car_control_modular.config_loader import load_config_to_env
    load_config_to_env(args.config)
    _ensure_follow_runtime_stopped()
    lock = acquire_test_lock(os.environ.get("MOTOR_RS485_PORT", "/dev/ttyS0"))
    session = ParkingSession()
    payload = dict(parameters=vars(args), trials=[], status="running")
    old_handlers = {}
    def interrupted(sig, frame): raise KeyboardInterrupt(f"signal {sig}")
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, interrupted)
        print("警告：会驱动车轮；100RPM原地旋转很快。必须固定台架或清空平整场地，准备物理急停。")
        print("本脚本不运行避障。额定电流须按电机/控制器实物核实；软件无法保证失联后停车。")
        if input("确认现场安全及电流额定限制，输入 PARK：").strip() != "PARK":
            payload["status"] = "cancelled"
            return 2
        _ensure_follow_runtime_stopped()
        session.open()
        payload["controller"] = session.controller_diagnostics
        for rpm, current, hold in plan:
            if input(f"{args.motion}: {rpm}RPM / {current}A / {hold}ms；输入 RUN 执行本次，其余结束：").strip() != "RUN":
                payload["status"] = "cancelled"
                break
            trial = dict(rpm=rpm, current_a=current, hold_ms=hold, status="running", events=[], samples=[])
            payload["trials"].append(trial)
            try: run_trial(session, args, trial)
            except BaseException:
                trial["status"] = "aborted"
                raise
            finally: session.stop("parking_trial_end")
            # Clear current while emergency-stopped before any operator wait.
            session.current(0, trial)
            trial["analysis"] = analyze(trial)
            print(json.dumps(trial["analysis"], ensure_ascii=False))
            print("已保存：", save(Path(args.output_dir), payload))
        else: payload["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        payload.update(status="aborted", error=str(exc))
        print(f"中止：{exc}；尝试急停和清0A", file=sys.stderr)
    finally:
        # A second Ctrl-C must not skip cleanup; physical emergency remains vital.
        for sig in old_handlers: signal.signal(sig, signal.SIG_IGN)
        try:
            try: session.close(ensure_stop=True)
            except Exception as exc:
                payload.update(status="cleanup_failed", cleanup_error=str(exc))
                print(f"清理失败，必须现场物理急停：{exc}", file=sys.stderr)
            print("最终结果：", save(Path(args.output_dir), payload))
        finally:
            for sig, handler in old_handlers.items(): signal.signal(sig, handler)
            lock.close()
    if payload["status"] != "complete": return 1
    return 0 if all(t["analysis"]["verdict"] == "encoder_stable" for t in payload["trials"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
