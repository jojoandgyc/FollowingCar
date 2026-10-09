"""IIO safety diagnostics use existing samples, never a motor or real sysfs."""
import importlib.util
import io
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def hal(monkeypatch, caplog):
    spec = importlib.util.spec_from_file_location(
        "isolated_ir_iio", Path(__file__).resolve().parents[2] / "ir_hal.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.IR_BACKEND = "iio"
    module.IR_TRIGGER_VALUE = 0
    module.IR_RAW_LOG_ENABLE = True
    module.IR_RAW_LOG_EVERY_SEC = .5
    module._IDX_TO_IIO_PATH = {0: "fake/right", 1: "fake/front", 2: "fake/left"}
    module.clock = SimpleNamespace(now=10.)
    module.time = SimpleNamespace(monotonic=lambda: module.clock.now)
    module.values = {"fake/right": "1", "fake/front": "1", "fake/left": "1"}

    def read(path, *args, **kwargs):
        value = module.values[path]
        if isinstance(value, Exception):
            raise value
        return io.StringIO(value)

    module.open = Mock(side_effect=read)
    caplog.set_level(logging.INFO, logger="PersonTracker")
    return module


@pytest.mark.parametrize("idx,side", [(0, "right"), (1, "front"), (2, "left")])
@pytest.mark.parametrize("value,trigger", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_channel_and_polarity_are_logged_without_extra_reads(hal, caplog, idx, side, value, trigger):
    hal.IR_TRIGGER_VALUE = trigger
    hal.values[hal._IDX_TO_IIO_PATH[idx]] = str(value)
    assert hal.IR.is_triggered(idx) is (value == trigger)
    assert hal.open.call_count == 1
    assert f"side={side}" in caplog.text
    assert f"path=fake/{side}" in caplog.text
    assert f"raw={value} trigger_value={trigger}" in caplog.text
    assert "fail_closed=False" in caplog.text


def test_held_block_and_clear_are_visible_and_not_cached(hal, caplog):
    hal.values["fake/right"] = "0"
    assert hal.IR.is_triggered(0)
    hal.clock.now += .1
    assert hal.IR.is_triggered(0)
    assert len(caplog.records) == 1
    hal.clock.now += .5
    assert hal.IR.is_triggered(0)
    assert "event=held blocked_observed_ms=600.0" in caplog.records[-1].message
    hal.values["fake/right"] = "1"
    hal.clock.now += .01
    assert not hal.IR.is_triggered(0)  # No wait for a diagnostic interval.
    assert "event=changed blocked_observed_ms=0.0" in caplog.records[-1].message
    assert "previous_blocked_observed_ms=610.0" in caplog.records[-1].message
    assert hal.open.call_count == 4


def test_other_channels_do_not_reset_block_duration_or_throttle(hal, caplog):
    hal.values["fake/right"] = "0"
    assert hal.IR.is_triggered(0)
    assert not hal.IR.is_triggered(1)
    assert not hal.IR.is_triggered(2)
    hal.clock.now += .6
    assert hal.IR.is_triggered(0)
    assert "side=right" in caplog.records[-1].message
    assert "blocked_observed_ms=600.0" in caplog.records[-1].message


@pytest.mark.parametrize("bad", ["2", "-1", "65535", "", "garbled", OSError("read failed")])
def test_invalid_or_failed_read_is_not_misrepresented_as_clear(hal, caplog, bad):
    hal.values["fake/right"] = bad
    assert hal.IR.is_triggered(0)
    assert "fail_closed=True" in caplog.text
    assert "side=right" in caplog.text
    assert "error=None" not in caplog.text
    hal.values["fake/right"] = "1"
    assert not hal.IR.is_triggered(0)
    assert "fail_closed=False" in caplog.records[-1].message


def test_read_errors_are_throttled_but_new_states_are_not(hal, caplog):
    hal.values["fake/right"] = OSError("missing")
    for _ in range(5):
        assert hal.IR.is_triggered(0)
    assert len(caplog.records) == 1
    assert hal.open.call_count == 5
    hal.values["fake/right"] = "0"
    assert hal.IR.is_triggered(0)
    assert len(caplog.records) == 2
    assert "fail_closed=False" in caplog.records[-1].message


@pytest.mark.parametrize("idx", [999, "bad", None])
def test_invalid_channel_fails_closed_without_secondary_exception(hal, idx):
    assert hal.IR.is_triggered(idx)
    hal.open.assert_not_called()


def test_disabled_logging_does_not_change_safety_or_read_rate(hal, caplog):
    hal.IR_RAW_LOG_ENABLE = False
    for raw in ("0", "1", "2"):
        hal.values["fake/right"] = raw
        assert hal.IR.is_triggered(0) is (raw != "1")
    assert hal.open.call_count == 3
    assert not caplog.records


@pytest.mark.parametrize("raw,triggered", [("0", True), ("1", False), ("2", True)])
def test_broken_log_handler_cannot_change_the_safety_read(hal, monkeypatch, raw, triggered):
    for method in ("warning", "info"):
        monkeypatch.setattr(hal._logger, method, Mock(side_effect=OSError("log unavailable")))
    hal.values["fake/right"] = raw
    assert hal.IR.is_triggered(0) is triggered
    assert hal.open.call_count == 1


def test_init_reports_all_three_paths_with_no_extra_reads(hal, caplog):
    assert hal.IR.init() == 0
    assert hal.open.call_count == 3
    assert len(caplog.records) == 3
    assert {record.message.split("side=")[1].split()[0] for record in caplog.records} == {
        "right", "front", "left"}


def test_init_rejects_invalid_values(hal):
    hal.values["fake/right"] = "2"
    assert hal.IR.init() == -1


def test_deinit_discards_only_diagnostic_state(hal, caplog):
    assert not hal.IR.is_triggered(0)
    assert hal.IR.deinit() == 0
    assert not hal._iio_log_state
    assert not hal.IR.is_triggered(0)
    assert "event=initial" in caplog.records[-1].message


def test_http_backend_does_not_start_reading_iio(hal, monkeypatch):
    hal.IR_BACKEND = "http"
    monkeypatch.setattr(hal, "_read_status", lambda: {"ir1": 1, "ir2": 1, "ir3": 0})
    assert hal.IR.is_triggered(0)
    assert not hal.IR.is_triggered(1)
    hal.open.assert_not_called()


def test_real_sensor_filter_releases_after_clear_without_reinitializing(hal, monkeypatch):
    from car_control_modular import sensor_modules
    from car_control_modular.sensor_modules import SensorRuntime, SensorRuntimeConfig

    monkeypatch.setattr(sensor_modules, "time", hal.time)
    runtime = SensorRuntime(SensorRuntimeConfig(
        ir_enable=True, ultrasonic_enable=False, mmwave_enable=False,
        imu_enable=False, imu_fail_soft=True, imu_log_enable=False,
        imu_log_every_sec=1., side_ir_blocks_rotation=True,
        side_ir_confirm_sec=.1, side_ir_release_sec=.2))
    runtime.ir = hal.IR  # No start(): all I/O remains the in-memory fake above.
    hal.values["fake/right"] = "0"
    assert runtime.get_raw_obstacle_status().right
    assert not runtime.get_obstacle_status().right
    hal.clock.now += .11
    assert runtime.get_obstacle_status().right
    hal.values["fake/right"] = "1"
    hal.clock.now += .01
    assert not runtime.get_raw_obstacle_status().right
    assert runtime.get_obstacle_status().right
    hal.clock.now += .21
    assert not runtime.get_obstacle_status().right
    hal.values["fake/right"] = "0"
    assert runtime.get_raw_obstacle_status().right  # A new hazard is immediate.
