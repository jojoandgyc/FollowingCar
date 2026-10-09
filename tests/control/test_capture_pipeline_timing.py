"""Capture waiting, inference, and decision cost remain distinct diagnostics."""
import pytest

from car_control_modular.capture_pipeline_timing import capture_pipeline_timing


def timeline(**changes):
    clocks = dict(capture_timestamp=100.0, pipeline_started_at=100.060,
                  vision_finished_at=100.142, decision_finished_at=100.154)
    clocks.update(changes)
    return capture_pipeline_timing(**clocks)


def test_cap2074_capture_wait_is_present_in_capture_ages():
    result = timeline()
    assert result["capture_clock_valid"] is True
    assert result["capture_to_pipeline_ms"] == pytest.approx(60.0)
    assert result["capture_result_age_ms"] == pytest.approx(142.0)
    assert result["capture_decision_age_ms"] == pytest.approx(154.0)
    assert result["capture_result_age_ms"] - result["capture_to_pipeline_ms"] == pytest.approx(82.0)


def test_synchronous_capture_has_zero_queue_age_and_preserves_old_processing_measure():
    result = timeline(pipeline_started_at=100.0)
    assert result["capture_clock_valid"] is True
    assert result["capture_to_pipeline_ms"] == 0.0
    assert result["capture_result_age_ms"] == pytest.approx(142.0)


@pytest.mark.parametrize("changes", [
    {"capture_timestamp": None}, {"capture_timestamp": True},
    {"capture_timestamp": "100"}, {"capture_timestamp": 0.0},
    {"capture_timestamp": float("nan")}, {"capture_timestamp": float("inf")},
    {"capture_timestamp": 100.061}, {"pipeline_started_at": float("inf")},
    {"vision_finished_at": 100.059}, {"decision_finished_at": 100.141},
    {"decision_finished_at": float("nan")},
])
def test_invalid_or_reversed_clocks_have_no_apparently_fresh_age(changes):
    result = timeline(**changes)
    assert result["capture_clock_valid"] is False
    assert all(result[key] is None for key in result if key != "capture_clock_valid")


def test_long_capture_age_is_reported_without_any_freshness_policy():
    result = timeline(decision_finished_at=104.0)
    assert result["capture_clock_valid"] is True
    assert result["capture_decision_age_ms"] == 4000.0
    assert set(result) == {
        "capture_clock_valid", "capture_to_pipeline_ms",
        "capture_result_age_ms", "capture_decision_age_ms",
    }


def test_real_pipeline_log_keeps_legacy_durations_and_adds_capture_clock_fields():
    # Run the actual call site without constructing RKNN or any devices.
    import ast
    import inspect
    import json
    import os
    import textwrap
    from types import SimpleNamespace
    import request_0513_modular as runtime

    tree = ast.parse(textwrap.dedent(inspect.getsource(runtime.PersonTracker.process_external_frame)))
    assignment, = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "capture_timing"
                           for target in node.targets)]
    log, = [node for node in ast.walk(tree) if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call) and node.value.args
            and isinstance(node.value.args[0], ast.Constant)
            and isinstance(node.value.args[0].value, str)
            and node.value.args[0].value.startswith("pipeline_timing:")]
    output = []
    context = dict(capture_pipeline_timing=capture_pipeline_timing,
                   frame_received_ts=100., pipeline_started_mono=100.060,
                   vision_finished_mono=100.142, control_finished_mono=100.154,
                   pipeline_started=200.060, vision_finished=200.142,
                   control_started=200.142, control_finished=200.154,
                   vision_result_age_sec=.082, stale_result_discarded=False,
                   camera_read_ms=0., timing={}, json=json, os=os,
                   self=SimpleNamespace(frame_index=862, _active_capture_frame_id=2074),
                   logger=SimpleNamespace(info=lambda message, *args: output.append(message % args)))
    exec(compile(ast.Module(body=[assignment, log], type_ignores=[]), __file__, "exec"), context)
    line, = output
    assert "result_age_ms=82.00 stale_result_discarded=False" in line
    assert "end_to_end_ms=94.00" in line
    assert "capture_clock_valid=True capture_to_pipeline_ms=60.0" in line
    assert "capture_result_age_ms=142.0 capture_decision_age_ms=154.0" in line
    assert "legacy_result_age_scope=processing_duration" in line
