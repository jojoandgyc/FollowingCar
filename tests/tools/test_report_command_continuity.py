import importlib.util
import json
from pathlib import Path
import sys

import pytest


PATH = Path(__file__).resolve().parents[2] / "tools" / "report_command_continuity.py"
SPEC = importlib.util.spec_from_file_location("report_command_continuity", PATH)
report = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = report
SPEC.loader.exec_module(report)


def speed(stamp, left=40, right=-40):
    return (f"LZ30EMA 电机命令: 标签=test 左轮={left}转/分 右轮={right}转/分 "
            f"history_uid=1 write_started_at={stamp-.001} write_completed_at={stamp}")


def stop(stamp, outcome="acknowledged"):
    return "motor_zero_audit " + json.dumps(dict(
        event_kind="stop", phase="stop_registers", outcome=outcome,
        completed_at=stamp, writes_complete=outcome == "acknowledged",
        acknowledged_sides=["left", "right"]))


def test_speed_changes_do_not_split_and_stop_register_does():
    result = report.summarize(report.read_events([
        stop(.5), speed(1.), speed(1.2, 50, -30), stop(1.6),
        stop(1.8), speed(2.), speed(2.2), speed(2.4, 0, 0), stop(3.)]))
    assert result["window_sec"] == pytest.approx(1.4)
    assert result["forward_sec"] == pytest.approx(1.)
    assert result["state_seconds"]["stopped"] == pytest.approx(.4)
    assert result["completed_forward_segments"] == 2
    assert result["mean_forward_ms"] == pytest.approx(500.)


def test_skipped_stop_does_not_interrupt_failed_stop_is_unknown():
    result = report.summarize(report.read_events([
        speed(1.), stop(1.1, "skipped"), speed(1.2),
        stop(1.3, "failed"), speed(1.5), stop(1.8)]))
    assert result["completed_forward_segments"] == 2
    assert result["state_seconds"]["unknown"] == pytest.approx(.2)
    assert result["forward_sec"] == pytest.approx(.6)


def test_rotation_is_not_forward_or_stop():
    result = report.summarize(report.read_events([
        speed(1.), speed(1.2, 10, 10), speed(1.5), stop(2.)]))
    assert result["state_seconds"]["rotation"] == pytest.approx(.3)
    assert result["state_seconds"]["stopped"] == 0.
    assert result["forward_sec"] == pytest.approx(.7)


def test_unfinished_forward_is_censored_not_extrapolated():
    result = report.summarize(report.read_events([speed(1.), speed(1.2)]))
    assert result["right_censored"]
    assert result["completed_forward_segments"] == 0
    assert result["forward_segments"][0]["duration_ms"] == pytest.approx(200.)
    assert result["mean_forward_ms"] is None


def test_log_order_is_not_monotonic_clock_order():
    result = report.summarize(report.read_events([speed(1.2), speed(1.), stop(1.5)]))
    assert result["forward_sec"] == pytest.approx(.5)


def test_raw_sign_mapping_is_explicit():
    result = report.summarize(report.read_events(
        [speed(1., -30, 30), stop(1.5)], left_forward_sign=-1, right_forward_sign=1))
    assert result["forward_sec"] == pytest.approx(.5)


def test_empty_legacy_and_malformed_lines_do_not_invent_receipts():
    events = report.read_events(["LZ30EMA 电机命令: 标签=old 左轮=40转/分 右轮=-40转/分",
                                 "motor_zero_audit {broken", "motor_zero_audit {}"])
    assert events == []
    assert report.summarize(events)["window_sec"] == 0.
