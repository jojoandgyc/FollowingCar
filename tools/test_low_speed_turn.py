#!/usr/bin/env python3
"""4/6 RPM fixed-duration pivot, observed by the motor-free camera recorder.

Default: plan only, no hardware. --execute also needs a fresh vision JSONL
and interactive TURN confirmation. No follow PID, speed ramp, or auto recenter.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.test_parking_stop_response import ParkingSession, targets, window_stable
from tools.test_forward_step_response import acquire_test_lock, _json_safe
from tools.imu_turn_calibrate import DEFAULT_CONFIG, _ensure_follow_runtime_stopped, _unwrap_i32_delta


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--rpms", default="4,6", help="每轮绝对RPM；原地轮差为其两倍")
    p.add_argument("--directions", default="left,right")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--hold-ms", type=int, choices=(500,3000), default=3000,
                   help="5A NORMAL时长；每轮总停车观察至少3秒")
    p.add_argument("--vision-jsonl", help="正在运行的 rknn_camera_track_record 输出")
    p.add_argument("--rated-current-a", type=float, help="现场已确认的安全相电流上限")
    p.add_argument("--hfov", type=float, default=66.)
    p.add_argument("--output-dir", default=str(ROOT/"calibration"/"low_speed_turn"))
    return p


def plan(args):
    rpms = [int(v) for v in args.rpms.split(",")]
    directions = args.directions.split(",")
    if not rpms or not set(rpms) <= {4,6} or len(set(rpms)) != len(rpms):
        raise ValueError("只允许4、6RPM，不可重复")
    if not directions or not set(directions) <= {"left","right"} or len(set(directions)) != len(directions):
        raise ValueError("方向只允许left,right，不可重复")
    if not 1 <= args.repeats <= 3 or not math.isfinite(args.hfov) or not 40 <= args.hfov <= 90:
        raise ValueError("重复次数1～3，视场40～90度")
    if args.execute and (not args.vision_jsonl or args.rated_current_a is None):
        raise ValueError("实测需指定实时视觉JSONL和已核实的安全相电流上限")
    if args.rated_current_a is not None and not (math.isfinite(args.rated_current_a)
                                                and 5 <= args.rated_current_a <= 30):
        raise ValueError("安全电流上限必须覆盖5A且不超过30A")
    return [(r,d) for _ in range(args.repeats) for r in rpms for d in directions]


def bearing(x, hfov=66.):
    """Rectilinear, centered principal point; approximate without calibration."""
    return math.degrees(math.atan((2*x-1)*math.tan(math.radians(hfov/2))))


def person_sample(item, now):
    stamp = float(item["capture_monotonic"])
    width = int(item["frame_width"])
    people = [d for d in item["detections"] if d["class_id"] == 0 and d["score"] >= .5]
    if not math.isfinite(stamp) or not 0 <= now-stamp <= .35 or width <= 0 or len(people) != 1:
        raise RuntimeError("视觉过期/未检测到唯一可信人体，停止本次测试")
    a,b,c,d = map(float, people[0]["bbox"])
    if not all(math.isfinite(v) for v in (a,b,c,d)) or not 0 <= a < c <= width or d <= b:
        raise RuntimeError("人体框无效")
    x=(a+c)/2/width
    if not .08 <= x <= .92:
        raise RuntimeError("目标接近画面边缘，终止而不是继续盲转")
    return dict(timestamp=stamp, cap=int(item["capture_index"]), x=x,
                area=(c-a)*(d-b), bbox=people[0]["bbox"], score=people[0]["score"])


class VisionReader:
    def __init__(self, path):
        self.handle = open(path, encoding="utf-8")
        self.pending = ""
        self.last_item = None

    def sample(self, now):
        # Only consume completed lines; do not discard a concurrent partial write.
        self.pending += self.handle.read()
        lines = self.pending.split("\n")
        self.pending = lines.pop()
        for line in lines:
            if line.strip(): self.last_item = json.loads(line)
        if self.last_item is None: raise RuntimeError("视觉日志尚无数据")
        return person_sample(self.last_item, now)

    def close(self): self.handle.close()


def run_trial(session, vision, args, trial, clock=time.monotonic, sleep=time.sleep):
    last, paths = {}, {s:0. for s in ("left","right")}
    start=clock()
    desired=targets(trial["rpm"], trial["direction"])

    def sample(phase):
        v=vision.sample(clock())
        if (trial["vision"] and v["cap"] == trial["vision"][-1]["cap"]
                and v["timestamp"] != trial["vision"][-1]["timestamp"]):
            raise RuntimeError("同一CAP时间戳改变，拒绝刷新旧图像")
        if not trial["vision"] or v["cap"] != trial["vision"][-1]["cap"]:
            if trial["vision"]:
                prev=trial["vision"][-1]
                if (v["timestamp"] <= prev["timestamp"] or v["cap"] <= prev["cap"]
                        or abs(v["x"]-prev["x"]) > .15
                        or not .6 <= v["area"]/prev["area"] <= 1.67):
                    raise RuntimeError("视觉关联跳变，拒绝混合不同目标")
            trial["vision"].append(dict(v, phase=phase))
        for side in ("left","right"):
            r=session.read_wheel(side)
            r.update(phase=phase, trusted=False)
            trial["samples"].append(r)
            if (not all(math.isfinite(float(r[k])) for k in ("timestamp","read_started","position_deg",
                    "forward_rpm","current_a","temperature_c","error_code"))
                    or not 0 <= clock()-r["timestamp"] <= .15
                    or not 0 <= r["timestamp"]-r["read_started"] <= .15):
                raise RuntimeError("电机反馈无效/超时")
            if r["error_code"] or r["temperature_c"] >= 60 or abs(r["current_a"]) > args.rated_current_a:
                raise RuntimeError("电机故障/过温/过流")
            if abs(r["forward_rpm"]) > 16:
                raise RuntimeError("低速测试实际轮速超过16RPM")
            if phase == "drive" and r["forward_rpm"]*desired[side] < -2*trial["rpm"]:
                raise RuntimeError("实测运动方向反向")
            if side in last:
                prev=last[side]; dt=r["timestamp"]-prev["timestamp"]
                delta=_unwrap_i32_delta(r["position_deg"],prev["position_deg"])
                if not 0 < dt <= (.35 if prev["phase"] != phase else .15) or abs(delta) > 16*6*dt+5:
                    raise RuntimeError("反馈断续/编码器突跳")
                paths[side] += abs(delta)/360*math.pi*.26
                if paths[side] > .30: raise RuntimeError("单轮累计编码器行程超过0.30m")
            r["trusted"]=True
            last[side]=r
        if clock()-start > 7: raise RuntimeError("单轮测试超时")

    def observe(end, phase):
        while clock() < end:
            sample(phase)
            sleep(min(.02,max(0,end-clock())))

    def require_quiet():
        now=clock()
        if any(window_stable([r for r in trial["samples"] if r["side"] == s],now-.15,now) is not True
               for s in ("left","right")):
            raise RuntimeError("双轮未稳定，禁止开始/自动接续下一轮")

    session.current(0,trial)
    session.speed(dict(left=0,right=0),trial)
    observe(clock()+.35,"baseline")
    require_quiet()
    if not .15 <= trial["vision"][-1]["x"] <= .85:
        raise RuntimeError("开始位置太偏，请重新站到画面中部")
    trial["drive_started"]=clock()
    session.speed(desired,trial)
    trial["drive_completed"]=clock()
    trial["drive_deadline"]=clock()+1.
    observe(trial["drive_deadline"],"drive")
    trial["brake_prepare_started"]=clock()
    session.current(5,trial)
    trial["normal_started"]=clock()
    session.normal(trial)
    trial["normal_completed"]=clock()
    if trial["normal_completed"]-trial["drive_deadline"] > .15:
        raise RuntimeError("停车下发超过计划150ms，终止并保留实际时长")
    observe(trial["normal_completed"]+args.hold_ms/1000,"park")
    trial["release_started"]=clock()
    session.current(0,trial)
    session.speed(dict(left=0,right=0),trial)
    trial["release_completed"]=clock()
    # Always observe the release; a zero snapshot must not authorize the next turn.
    observe(max(trial["normal_completed"]+3.,clock()+.35),"released")
    require_quiet()
    # Inference delivers an older capture. Freeze the evaluation interval,
    # then collect its pending results with motors already at zero. Sliding
    # the window with wall time would always discard the newest 100-200ms.
    trial["visual_window_end"]=clock()
    drain_deadline=clock()+.5
    while trial["vision"][-1]["timestamp"] < trial["visual_window_end"]:
        if clock() >= drain_deadline:
            raise RuntimeError("停车后视觉结果未收齐，禁止接续下一轮")
        sample("released")
        sleep(.02)
    require_quiet()
    trial["finished"]=clock()
    trial["status"]="complete"
    report=analyze(trial,args.hfov)
    if report["verdict"] != "visual_estimate" or not report["visually_stable"]:
        trial["status"]="inconclusive"
        raise RuntimeError("画面证据不足或停车后仍明显移动；保留记录，不继续下一轮")


def analyze(trial, hfov):
    if trial.get("status") not in {"complete","inconclusive"}: return dict(verdict="aborted")
    points=trial["vision"]
    def at(t):
        a=[v for v in points if v["timestamp"] <= t]
        b=[v for v in points if v["timestamp"] >= t]
        if not a or not b or b[0]["timestamp"]-a[-1]["timestamp"] > .20: return None
        lo,hi=a[-1],b[0]
        fraction=(t-lo["timestamp"])/max(1e-9,hi["timestamp"]-lo["timestamp"])
        return bearing(lo["x"],hfov)+(bearing(hi["x"],hfov)-bearing(lo["x"],hfov))*fraction
    t0=trial["drive_started"]; t1=trial["drive_completed"]+1
    begin, one_sec, stopped=at(t0),at(t1),at(trial["normal_completed"])
    visual_end=trial.get("visual_window_end",trial["finished"])
    tail=[bearing(v["x"],hfov) for v in points if visual_end-.3 <= v["timestamp"] <= visual_end]
    if None in (begin,one_sec,stopped) or len(tail) < 3:
        return dict(verdict="insufficient_visual_samples")
    settled=statistics.median(tail)
    # Two successive delivered frames beyond a 1-degree displacement are an
    # observable onset, not the instant the wheel/chassis first moved.
    onset=None
    moving=[v for v in points if t0 <= v["timestamp"] <= trial["normal_completed"]]
    for a,b in zip(moving,moving[1:]):
        da,db=begin-bearing(a["x"],hfov),begin-bearing(b["x"],hfov)
        if abs(da) >= 1 and abs(db) >= 1 and da*db > 0 and b["timestamp"]-a["timestamp"] <= .15:
            onset=(a["timestamp"]-t0)*1000
            break
    return dict(verdict="visual_estimate", positive="right",
        angle_to_1s_deadline_deg=begin-one_sec,
        average_to_1s_deadline_dps=(begin-one_sec)/(t1-t0),
        residual_after_normal_deg=stopped-settled,
        total_until_settled_deg=begin-settled,
        final_visual_span_deg=max(tail)-min(tail),
        visually_stable=max(tail)-min(tail) <= 1.,
        visual_window_end=visual_end,
        observed_1deg_onset_ms=onset,
        speed_write_ms=(trial["drive_completed"]-t0)*1000,
        stop_overrun_ms=(trial["normal_completed"]-t1)*1000,
        note="66度视场针孔近似；静止目标、近似纯旋转；含相机位置偏移和框抖动误差，不是稳态角速度")


def main(argv=None):
    args=parser().parse_args(argv)
    try: trials=plan(args)
    except ValueError as e: print(str(e),file=sys.stderr); return 2
    print(f"计划 {trials}：每次1000ms，5A驻车{args.hold_ms}ms，总停车观察至少3000ms；RPM为单轮值")
    if not args.execute: print("仅预览，未打开摄像头或串口"); return 0
    from car_control_modular.config_loader import load_config_to_env
    load_config_to_env(args.config)
    _ensure_follow_runtime_stopped()
    print("测试不运行避障！请在平整空地站定，单人入镜，旁人准备物理急停。整组会自动接续。")
    if input("确认就绪，输入 TURN：").strip() != "TURN": return 2
    lock=acquire_test_lock(os.environ.get("MOTOR_RS485_PORT","/dev/ttyS0"))
    session=ParkingSession(); vision=None
    payload=dict(parameters=vars(args),trials=[],status="running")
    directory=Path(args.output_dir)
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/(datetime.now().strftime("turn_%Y%m%d_%H%M%S_%f_")+str(os.getpid())+".json")
    def save():
        for trial in payload["trials"]: trial["analysis"]=analyze(trial,args.hfov)
        path.write_text(json.dumps(_json_safe(payload),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    def interrupt(sig,frame): raise KeyboardInterrupt(f"signal {sig}")
    handlers={sig:signal.signal(sig,interrupt) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        vision=VisionReader(args.vision_jsonl)
        vision.sample(time.monotonic())
        _ensure_follow_runtime_stopped()
        session.open()
        payload["controller"]=session.controller_diagnostics
        for rpm,direction in trials:
            trial=dict(rpm=rpm,direction=direction,status="running",events=[],samples=[],vision=[])
            payload["trials"].append(trial)
            print(f"开始 {direction} ±{rpm}RPM / 1000ms",flush=True)
            run_trial(session,vision,args,trial)
            save()
            print(json.dumps(trial["analysis"],ensure_ascii=False),flush=True)
        payload["status"]="complete"
    except (Exception,KeyboardInterrupt) as e:
        payload.update(status="aborted",error=str(e))
        if payload["trials"] and payload["trials"][-1]["status"] == "running":
            payload["trials"][-1]["status"]="aborted"
        print(f"中止：{e}",file=sys.stderr)
    finally:
        for sig in handlers: signal.signal(sig,signal.SIG_IGN)
        try:
            try: session.close(ensure_stop=True)
            except Exception as e:
                payload.update(status="cleanup_failed",cleanup_error=str(e))
                print(f"清理失败，必须物理急停：{e}",file=sys.stderr)
            if vision is not None: vision.close()
            save()
            print(f"结果：{path}",flush=True)
        finally:
            lock.close()
            for sig,old in handlers.items(): signal.signal(sig,old)
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__": raise SystemExit(main())
