"""The offline report reads explicit ACK records only; never imports hardware."""
import json

import pytest

from tools.report_zero_commands import format_report, main, read_audit_events


def event(**updates):
    value = dict(event_id=1, event_kind="zero_speed", source="FOLLOW_SNAPSHOT_REVOKED",
                 phase="speed_pair", outcome="acknowledged", writes_complete=True,
                 generated_at=10.5, requested_raw=[0, 0], previous_acknowledged_raw=[30, -20],
                 previous_forward_yaw=[25., 5.], acknowledged_sides=["right", "left"],
                 failed_sides=[], receipt_after=2, stop_generation=0,
                 context=dict(uid=1, command_uid=1, command_cap=201, control_cap=205,
                              observation_cap=207, decision_stage="snapshot_admission",
                              zero_reason="terminal_feedback_invalid", depth_age_ms=75.,
                              feedback_age_ms=160., planned_axes=[1, 2, 24., 0.],
                              command_reason="visible_follow", control_reason="distance_pi",
                              depth_read_veto=[1, 10.425, "continuation_feedback_stale"]))
    value.update(updates)
    return value


def write_log(tmp_path, values):
    path = tmp_path / "input.log"
    path.write_text("\n".join("2026-10-07 23:18:08.123 INFO motor_zero_audit " + json.dumps(value)
                             for value in values), encoding="utf-8")
    return path


def test_summary_separates_protocol_zero_stop_failure_and_skip(tmp_path):
    path = write_log(tmp_path, [event(), event(event_kind="stop", phase="stop_registers"),
                               event(outcome="failed", writes_complete=False),
                               event(outcome="skipped", writes_complete=False)])
    values, malformed = read_audit_events(path)
    report = format_report(values, malformed)
    assert "审计事件 4 条" in report
    assert "2/1/1" not in report  # STOP cannot be merged into zero-speed ACK count.
    assert "1/1/1 zero_speed" in report and "1/0/0 stop" in report
    assert "stage=snapshot_admission" in report
    assert "reason=terminal_feedback_invalid" in report
    assert "不代表车轮已停稳" in report and "不能由这些记录单独计算停车时长" in report


def test_details_keep_three_different_caps_and_real_old_wheel_pair(tmp_path):
    values, malformed = read_audit_events(write_log(tmp_path, [event()]))
    report = format_report(values, malformed, details=True)
    assert "CAP(cmd/ctrl/obs)=201/205/207" in report
    assert "previous_raw=30/-20 previous_forward/yaw=25/5 requested_raw=0/0" in report
    assert "depth_age_ms=75 feedback_age_ms=160" in report
    assert "2026-10-07 23:18:08.123" in report
    assert "UID=1 command_UID=1" in report
    assert "command_reason=visible_follow control_reason=distance_pi" in report
    assert "depth_read_veto=1/10.425/continuation_feedback_stale" in report
    assert "generated_mono=10.5" in report


def test_legacy_logs_do_not_invent_zero_cause(tmp_path):
    path = tmp_path / "old.log"
    path.write_text("CAP205 深度=1.53\nLZ30EMA 电机命令 标签=FOLLOW20 左轮=0 右轮=0\n",
                    encoding="utf-8")
    values, malformed = read_audit_events(path)
    assert not values and malformed == 0
    report = format_report(values, malformed)
    assert "旧日志不能可靠追溯" in report
    assert "incomplete_lateral_publication" not in report


@pytest.mark.parametrize("bad", [None, [], {}, event(outcome="failed", writes_complete=True),
                                event(context=[]), event(event_kind="unknown")])
def test_malformed_audit_is_counted_not_treated_as_written_zero(tmp_path, bad):
    values, malformed = read_audit_events(write_log(tmp_path, [bad, event()]))
    assert len(values) == 1 and malformed == 1


def test_unparseable_json_does_not_stop_later_valid_records(tmp_path):
    path = write_log(tmp_path, [event()])
    path.write_text("motor_zero_audit {invalid}\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    values, malformed = read_audit_events(path)
    assert len(values) == 1 and malformed == 1 and values[0]["line_number"] == 2


def test_cli_is_read_only_and_compact_by_default(tmp_path, capsys):
    path = write_log(tmp_path, [event()])
    original = path.read_bytes()
    assert main([str(path)]) == 0
    compact = capsys.readouterr().out
    assert "逐条记录" not in compact
    assert main([str(path), "--details"]) == 0
    assert "逐条记录" in capsys.readouterr().out
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_cli_missing_file_reports_error_without_creating_it(tmp_path, capsys):
    path = tmp_path / "missing.log"
    assert main([str(path)]) == 2
    assert "无法读取日志" in capsys.readouterr().err
    assert not path.exists()
