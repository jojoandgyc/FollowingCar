#!/usr/bin/env python3
"""Bounded, attended STOP diagnostics. Default is a hardware-free plan.

Read-only opens/closes the client directly (no backend startup/cleanup writes).
One case per invocation; no retries, no controller-mode changes, no persistent
writes. Software safety cannot replace an independent motor-power cutoff.
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

from tools.imu_turn_calibrate import _ensure_follow_runtime_stopped, _unwrap_i32_delta
from tools.test_forward_step_response import acquire_test_lock, validate_controller_mode
from tools.cap114_stop_replay import DEFAULT_SOURCE, extract_plan, extract_zero_entry

SIDES = ("right", "left")


def trial_limits(profile, case, rpm, motion, drive_sec, confirmed_current_limit_a=None):
    """Higher envelope is only for the explicitly requested single forward case."""
    if profile == "cap114-80":
        if (case, rpm, motion, drive_sec) != ("cap114-replay", 80, "forward", 2.):
            raise ValueError("cap114-80仅允许CAP114回放、80RPM前进2秒")
        if (confirmed_current_limit_a is None or not math.isfinite(confirmed_current_limit_a)
                or not 5 <= confirmed_current_limit_a <= 30):
            raise ValueError("80RPM试验需显式填写已核实的运行相电流上限 --confirmed-current-limit-a (5..30A)；不默认采用历史30A")
        return dict(command_rpm=80, feedback_rpm=95, travel_deg=1350,
                    current_a=float(confirmed_current_limit_a))
    if confirmed_current_limit_a is not None:
        raise ValueError("显式电流上限目前仅用于cap114-80，其他试验保护不变")
    if profile == "cap114-60":
        if (case, rpm, motion, drive_sec) != ("cap114-replay", 60, "forward", 1.8):
            raise ValueError("cap114-60仅允许CAP114回放、60RPM前进1.8秒")
        return dict(command_rpm=60, feedback_rpm=75, travel_deg=900, current_a=6)
    if profile != "low-speed":
        raise ValueError("unknown test profile")
    if rpm != 0 and not 4 <= rpm <= 12:
        raise ValueError("低速RPM只允许0或4..12；60RPM必须显式选择cap114-60")
    if not math.isfinite(drive_sec) or not .2 <= drive_sec <= .8:
        raise ValueError("drive duration outside 0.2..0.8 seconds")
    return dict(command_rpm=12, feedback_rpm=20, travel_deg=65, current_a=6)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--read-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    p.add_argument("--attended-power-cutoff", action="store_true",
                   help="现场已确认有人看护且能独立切断电机电源")
    p.add_argument("--case", choices=("normal-free", "free", "emergency", "5a-free",
                                     "5a-emergency", "cap114-replay"), default="normal-free")
    p.add_argument("--replay-log-dir", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--replay-entry", choices=("immediate", "repeated-zero"), default="immediate",
                   help="仅CAP114回放：repeated-zero复原0/54/87/219ms零速，约325ms后NORMAL")
    p.add_argument("--profile", choices=("low-speed", "cap114-60", "cap114-80"), default="low-speed",
                   help="cap114-60:60RPM/1.8秒/3米净空；cap114-80:80RPM/2秒/5米净空，限直行回放")
    p.add_argument("--confirmed-current-limit-a", type=float, default=None,
                   help="仅cap114-80：现场核实的运行相电流保护上限5..30A，不是驻车电流；无默认值")
    p.add_argument("--rpm", type=int, default=0, help="默认0或4..12；专用回放档分别要求60或80")
    p.add_argument("--motion", choices=("forward", "left", "right"), default="forward")
    p.add_argument("--drive-sec", type=float, default=.5, help="默认0.2..0.8秒；专用60/80档要求1.8/2秒")
    p.add_argument("--observe-sec", type=float, default=1., help="停车后观察时间，1..8秒")
    p.add_argument("--countdown", type=int, default=0, help="实际试验前倒计时，0..5秒")
    p.add_argument("--output-dir", type=Path, default=ROOT / "calibration" / "stop_modes")
    return p


class Diagnostic:
    def __init__(self, driver, signs, result, clock=time.monotonic, sleep=time.sleep):
        self.driver, self.signs, self.result = driver, signs, result
        self.clock, self.sleep = clock, sleep
        self.modified = False
        self.previous = {}
        self.travel = dict.fromkeys(SIDES, 0.0)
        self.opposite = dict.fromkeys(SIDES, 0)
        self.expected = dict.fromkeys(SIDES, 0)
        self.limits = trial_limits("low-speed", "normal-free", 0, "forward", .5)

    def event(self, kind, side, value, operation):
        e = dict(kind=kind, side=side, value=value, started=self.clock(), acknowledged=False)
        self.result["events"].append(e)
        self.modified = True  # Includes an ambiguous/partially successful write.
        try:
            operation()
            e["acknowledged"] = True
        except BaseException as exc:
            e["error"] = str(exc)
            raise
        finally:
            e["completed"] = self.clock()
        if e["completed"] - e["started"] > .15:
            raise RuntimeError("write latency >150ms")

    def current(self, amps):
        for side in SIDES:
            name = side + "_parking_current"
            self.event("parking_setting", side, amps,
                       lambda: self.driver.write_register(name, amps, persist=False))
        actual = {side: self.driver.read_register(side + "_parking_current") for side in SIDES}
        self.result["events"].append(dict(kind="parking_readback", value=actual, completed=self.clock()))
        if any(not math.isfinite(float(x)) or abs(x - amps) > .01 for x in actual.values()):
            raise RuntimeError("parking current setting readback mismatch")

    def speed(self, values):
        for side in SIDES:
            value = values[side] * self.signs[side]
            if abs(value) > self.limits["command_rpm"]:
                raise ValueError("speed command exceeds selected profile limit")
            self.event("speed", side, value, lambda: self.driver.set_speed(side, value))

    def stop(self, mode):
        for side in SIDES:
            self.event("stop", side, mode, lambda: self.driver.stop(side, mode))
        return self.clock()

    def sample(self, phase, enforce=True):
        for side in SIDES:
            started = self.clock()
            status = self.driver.read_motor_status(side)
            now = self.clock()
            row = dict(phase=phase, side=side, read_started=started, timestamp=now,
                       forward_rpm=status.speed_rpm * self.signs[side], raw_rpm=status.speed_rpm,
                       position_deg=status.position_degree, current_a=status.phase_current_a,
                       pwm_percent=status.pwm_percent, temperature_c=status.board_temperature_c,
                       error_code=status.error_code)
            self.result["samples"].append(row)
            last = self.previous.get(side)
            if last:
                self.travel[side] += abs(_unwrap_i32_delta(row["position_deg"], last["position_deg"]))
            self.previous[side] = row
            if not enforce:
                continue
            if now - started > .15 or (last and now - last["timestamp"] > .20):
                raise RuntimeError("stale/slow feedback")
            if (status.error_code or status.board_temperature_c >= 55 or
                    status.phase_current_a > self.limits["current_a"]):
                raise RuntimeError(f"motor fault/temperature/current limit: {row}")
            # Low-speed: ~0.15m; 60RPM: ~2.04m; 80RPM: ~3.06m for a 26cm wheel.
            # Cumulative per-wheel displacement includes reversals and stopping.
            if (abs(status.speed_rpm) > self.limits["feedback_rpm"] or
                    self.travel[side] > self.limits["travel_deg"]):
                raise RuntimeError(f"speed/travel limit: {row}")
            wrong = self.expected[side] * row["forward_rpm"] < -3
            self.opposite[side] = self.opposite[side] + 1 if wrong else 0
            if self.opposite[side] >= 2:
                raise RuntimeError(f"unexpected reversal on {side}; cut power if motion persists")

    def observe(self, phase, seconds, enforce=True):
        start = self.clock()
        while self.clock() - start < seconds:
            self.sample(phase, enforce)
            self.sleep(.02)
        return start, self.clock()

    def quiet(self, phase, window=.25):
        rows = [r for r in self.result["samples"] if r["phase"] == phase]
        for side in SIDES:
            wheel = [r for r in rows if r["side"] == side]
            if not wheel or self.clock() - wheel[-1]["timestamp"] > .15:
                return False
            end = wheel[-1]["timestamp"]
            wheel = [r for r in wheel if r["timestamp"] >= end - window]
            if len(wheel) < 4 or wheel[-1]["timestamp"] - wheel[0]["timestamp"] < window * .8:
                return False
            if any(abs(r["forward_rpm"]) > 1 or r["error_code"] for r in wheel):
                return False
            if any(b["timestamp"] - a["timestamp"] > .10 for a, b in zip(wheel, wheel[1:])):
                return False
            if sum(abs(_unwrap_i32_delta(b["position_deg"], a["position_deg"]))
                   for a, b in zip(wheel, wheel[1:])) > 2:
                return False
        return True

    def wait_replay_deadline(self, deadline, phase):
        while self.clock() < deadline:
            if deadline-self.clock() > .02:
                self.sample(phase)
            self.sleep(max(0, min(.01, deadline-self.clock())))
        if self.clock()-deadline > .15:
            raise RuntimeError("replay deadline missed; do not burst-replay old commands")

    def repeated_zero_entry(self, entry):
        self.speed(dict.fromkeys(SIDES, 0))
        origin = self.clock()
        self.result['zero_entry_origin'] = origin
        dispatches = [dict(source_line=entry['events'][0]['line'], planned_offset=0.,
                           actual_offset=0., completed_offset=0.)]
        self.result['zero_entry_dispatches'] = dispatches
        for e in entry['events'][1:]:
            self.wait_replay_deadline(origin+e['offset'], 'zero_entry')
            record = dict(source_line=e['line'], planned_offset=e['offset'],
                          actual_offset=self.clock()-origin)
            dispatches.append(record)
            self.speed(dict.fromkeys(SIDES, 0))
            record['completed_offset'] = self.clock()-origin
        self.wait_replay_deadline(origin+entry['normal_offset']-entry['normal_prepare_budget_sec'],
                                  'zero_entry')
        self.result['normal_preparation_started'] = self.clock()
        # Runtime NORMAL itself has a pre-zero write, in addition to the four groups.
        self.speed(dict.fromkeys(SIDES, 0))

    def replay(self, plan):
        if 'zero_entry' in plan:
            self.repeated_zero_entry(plan['zero_entry'])
        else:
            self.speed(dict.fromkeys(SIDES, 0))
        self.current(5)
        self.result["normal_completed"] = self.stop(0)
        if 'zero_entry' in plan:
            self.result['first_zero_to_normal_ms'] = (
                self.result['normal_completed']-self.result['zero_entry_origin'])*1000
        origin = self.clock() + plan["normal_to_cap_sec"]
        self.result["replay_origin"] = origin
        self.result["replay_dispatches"] = []
        for entry in plan["events"]:
            deadline = origin + entry["offset"]
            self.wait_replay_deadline(deadline, 'replay')
            lateness = self.clock()-deadline
            if lateness > .15:
                raise RuntimeError("replay deadline missed; do not burst-replay old commands")
            event = dict(source_line=entry["line"], kind=entry["kind"], value=entry["value"],
                         planned_offset=entry["offset"], actual_offset=self.clock()-origin,
                         lateness_ms=lateness*1000)
            self.result["replay_dispatches"].append(event)
            if entry["kind"] == "current":
                self.current(entry["value"])
            else:
                self.stop(entry["value"])
            event["completed_offset"] = self.clock()-origin
        self.result["release_completed"] = self.clock()

    def run(self, case, rpm, motion, plan=None, drive_sec=.5, observe_sec=1., profile="low-speed",
            confirmed_current_limit_a=None):
        limits = trial_limits(profile, case, rpm, motion, drive_sec, confirmed_current_limit_a)
        if not math.isfinite(observe_sec) or not 1 <= observe_sec <= 8:
            raise ValueError("observation outside 1..8 seconds")
        if case == "cap114-replay" and plan is None:
            raise ValueError("replay requires a validated source plan")
        self.observe("baseline", .5)
        if not self.quiet("baseline"):
            raise RuntimeError("baseline not quiet; no test commands sent")
        self.current(0)
        self.speed(dict.fromkeys(SIDES, 0))
        self.observe("zero_baseline", .35)
        if not self.quiet("zero_baseline"):
            raise RuntimeError("zero speed itself did not remain quiet")
        self.limits = limits
        self.result["safety_limits"] = dict(limits)
        if rpm:
            values = dict(left=rpm, right=rpm)
            if motion == "left": values["left"] = -rpm
            if motion == "right": values["right"] = -rpm
            self.expected = {side: 1 if value > 0 else -1 for side, value in values.items()}
            print(f"开始运动：{motion} {rpm}RPM，{drive_sec:.1f}秒", flush=True)
            self.speed(values)
            self.observe("drive", drive_sec)
            self.result["pre_stop_rpm"] = {s: self.previous[s]["forward_rpm"] for s in SIDES}
        print(f"执行停车：{case}，随后观察{observe_sec:.1f}秒", flush=True)
        if case == "cap114-replay":
            self.result["replay_plan"] = plan
            self.replay(plan)
        elif case.startswith("5a-"):
            # Only the parking-current setting changes; no pre-zero speed or NORMAL.
            self.current(5)
            self.result["release_completed"] = self.stop(2 if case == "5a-free" else 1)
            self.observe("stop_5a", .5)
            self.result["quiet_at_500ms"] = self.quiet("stop_5a")
            self.current(0)
            self.result["current_cleared"] = self.clock()
        elif case == "normal-free":
            # Matches runtime NORMAL: zero targets, 5A dual readback, NORMAL.
            self.speed(dict.fromkeys(SIDES, 0))
            self.current(5)
            self.result["normal_completed"] = self.stop(0)
            self.observe("normal_5a", .5)
            self.result["quiet_at_500ms"] = self.quiet("normal_5a")
            self.current(0)
            self.result["release_completed"] = self.stop(2)
        else:
            # 0A already verified; no zero-target writes before/after these STOPs.
            self.result["release_completed"] = self.stop(2 if case == "free" else 1)
        self.observe("post_stop", observe_sec)
        self.result["post_stop_quiet"] = self.quiet("post_stop")
        if not self.result["post_stop_quiet"]:
            raise RuntimeError("not quiet after STOP; abort remaining experiments")

    def cleanup(self):
        """Attempt BOTH wheels even on partial failure; no speed commands."""
        if not self.modified:
            return
        errors = []
        for side in SIDES:
            try:
                self.event("cleanup_emergency", side, 1, lambda: self.driver.stop(side, 1))
            except BaseException as exc:
                errors.append(str(exc))
        for side in SIDES:
            try:
                self.event("cleanup_current", side, 0,
                           lambda: self.driver.write_register(side + "_parking_current", 0, persist=False))
            except BaseException as exc:
                errors.append(str(exc))
        try:
            self.result["final_parking_setting"] = {
                side: self.driver.read_register(side + "_parking_current") for side in SIDES}
            self.observe("cleanup_observe", .5, enforce=False)
            self.result["cleanup_quiet"] = self.quiet("cleanup_observe")
        except BaseException as exc:
            errors.append(str(exc))
        self.result["cleanup_errors"] = errors
        if errors or not self.result.get("cleanup_quiet") or any(
                x != 0 for x in self.result.get("final_parking_setting", {}).values()):
            self.result["status"] = "unsafe_or_unknown"
            print("警告：未确认停稳/清0A，请现场立即切断电机电源！", flush=True)


def save(result, directory):
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / (datetime.now().strftime("stop_%Y%m%d_%H%M%S_%f") + f"_{os.getpid()}")
    base.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if result["samples"]:
        with base.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result["samples"][0]))
            writer.writeheader()
            writer.writerows(result["samples"])
    print(f"结果：{base.with_suffix('.json')}", flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        limits = trial_limits(args.profile, args.case, args.rpm, args.motion, args.drive_sec,
                              args.confirmed_current_limit_a)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if (not math.isfinite(args.observe_sec) or not 1 <= args.observe_sec <= 8 or
            not 0 <= args.countdown <= 5):
        raise SystemExit("observe-sec须1..8；countdown须0..5")
    if args.execute and not args.attended_power_cutoff:
        raise SystemExit("必须确认现场有人能独立切断电机电源")
    if args.replay_entry != 'immediate' and args.case != 'cap114-replay':
        raise SystemExit('repeated-zero仅适用于CAP114回放')
    plan = extract_plan(args.replay_log_dir) if args.case == "cap114-replay" else None
    if plan and args.replay_entry == 'repeated-zero':
        plan['zero_entry'] = extract_zero_entry(plan)
    if not args.execute and not args.read_only:
        detail = ("5A停车500ms后清0A" if args.case.startswith("5a-") else
                  "CAP114停车序列，含5A NORMAL起始状态；不复原此前运动历史" if plan else
                  "NORMAL为5A/500ms后0A+FREE；普通FREE/EMERGENCY为0A")
        print(f"仅计划：{args.case}, {args.motion}, {args.rpm}RPM；静止检查→运动{args.drive_sec}秒(可选)→"
              f"STOP观察。{detail}。不会打开串口。")
        print(f"profile={args.profile} limits={limits}")
        if plan:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    signs = [int(os.environ.get(k, v)) for k, v in (("MOTOR_LEFT_SIGN", "-1"),
             ("MOTOR_RIGHT_SIGN", "1"), ("MOTOR_FORWARD_TARGET_SIGN", "-1"))]
    if any(v not in (-1, 1) for v in signs):
        raise SystemExit("invalid motor signs")
    port = os.environ.get("MOTOR_RS485_PORT", "/dev/ttyS0")
    result = dict(status="started", case=args.case, rpm=args.rpm, motion=args.motion,
                  read_only=args.read_only, port=port, events=[], samples=[],
                  drive_sec=args.drive_sec, observe_sec=args.observe_sec, countdown=args.countdown,
                  profile=args.profile, planned_limits=limits,
                  confirmed_current_limit_a=args.confirmed_current_limit_a, replay_entry=args.replay_entry)
    driver = session = lock = None
    old_handlers = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    try:
        lock = acquire_test_lock(port)
        _ensure_follow_runtime_stopped()
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, interrupted)
        if args.execute:
            for remaining in range(args.countdown, 0, -1):
                print(f"{args.case} 试验倒计时：{remaining}秒，请现场看护", flush=True)
                time.sleep(1)
            _ensure_follow_runtime_stopped()
        sys.path.insert(0, str(Path(os.environ.get("MOTOR_RS485_LIB_DIR", "/home/topeet/lianzhan")) / "src"))
        from lz30ema_rs485.client import LZ30EMAClient
        driver = LZ30EMAClient.from_serial(port, slave=int(os.environ.get("MOTOR_RS485_SLAVE_ID", "1")),
                    baudrate=int(os.environ.get("MOTOR_RS485_BAUDRATE", "115200")), timeout=.06)
        session = Diagnostic(driver, dict(left=signs[0]*signs[2], right=signs[1]*signs[2]), result)
        result["signs"] = session.signs
        result["bus"] = driver.read_bus_status()
        result["configured_mode"] = driver.read_register("system_mode")
        result["mode_policy"] = validate_controller_mode(result["bus"], result["configured_mode"])
        result["registers"] = {name: driver.read_register(name) for name in
                  ("foc_loop_mode", "closed_loop_acceleration", "closed_loop_deceleration",
                   "left_parking_current", "right_parking_current")}
        print(json.dumps({k: result[k] for k in ("bus", "registers", "mode_policy")}, ensure_ascii=False), flush=True)
        if args.read_only:
            session.observe("read_only", 1.0)
            result["quiet"] = session.quiet("read_only")
        else:
            session.run(args.case, args.rpm, args.motion, plan, args.drive_sec, args.observe_sec,
                        args.profile, args.confirmed_current_limit_a)
        result["status"] = "complete"
    except BaseException as exc:
        result.update(status="aborted", error=f"{type(exc).__name__}: {exc}")
        print(f"测试中止：{result['error']}", flush=True)
    finally:
        if session is not None:
            try:
                session.cleanup()
            except BaseException as exc:
                result.update(status="unsafe_or_unknown", cleanup_exception=str(exc))
                print("清理异常，请现场立即切断电机电源！", flush=True)
        if driver is not None:
            try:
                driver.close()  # transport close ONLY; no hidden zero-speed/STOP writes.
            except BaseException as exc:
                result.update(status="unsafe_or_unknown", close_error=str(exc))
        if lock is not None:
            lock.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        save(result, args.output_dir)
    print(f"status={result['status']} post_stop_quiet={result.get('post_stop_quiet')} "
          f"cleanup_quiet={result.get('cleanup_quiet')} read_only_quiet={result.get('quiet')}", flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
