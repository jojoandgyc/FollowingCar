"""Bottom-edge recovery across real Astra sampling and the range fusion layer."""
from pathlib import Path
from dataclasses import replace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from car_control_modular import astra_depth
from car_control_modular.astra_depth import AstraDepthConfig, AstraDepthRuntime
from test_depth_raw_geometry_runtime import make_runtime, person


BOX = (294.313720703125, 74.28895568847656, 441.64556884765625, 474.82720947265625)


def setup_chain(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(astra_depth.time, "monotonic", lambda: clock[0])
    sensor = AstraDepthRuntime(AstraDepthConfig())
    sensor._np = np

    class Sensors:
        measurement = None

        def get_astra_target_distance(self, *args, **kwargs):
            self.measurement = sensor.measure_target(*args, **kwargs)
            return self.measurement

    runtime, _unused = make_runtime()
    sensors = Sensors()
    runtime.sensor_runtime = sensors

    def sample(depth, now):
        clock[0] = now
        sensor._latest_depth = (np.full((480, 640), depth, dtype=np.uint16)
                                if np.isscalar(depth) else depth.copy())
        sensor._latest_depth_ts = now - 0.02
        return runtime.get_vision_depth_state(
            640, 480, person(bbox=BOX, capture_timestamp=now - 0.02),
            use_latest_depth=True,
        )

    core = np.zeros((480, 640), dtype=np.uint16)
    core[227:274, 341:351] = 2721
    return runtime, sensors, sample, core


@pytest.mark.parametrize("intermediate_hole", [False, True])
def test_confirmed_torso_recovery_reaches_fusion_without_old_range_bias(monkeypatch, intermediate_hole):
    runtime, sensors, sample, core = setup_chain(monkeypatch)
    assert sample(2137, 100.0).used_distance_m == pytest.approx(2.137)
    if intermediate_hole:
        held = sample(0, 100.05)
        assert held.raw_distance_m is None
        assert held.fusion_mode == "depth_visual"
    first = sample(core, 101.9)
    assert first.used_distance_m is first.raw_distance_m is None
    assert sensors.measurement.confirm_count == 1
    second = sample(core, 102.02)
    assert sensors.measurement.confirm_count == 2
    assert sensors.measurement.jump_confirmation is None
    assert second.raw_distance_m == pytest.approx(2.721)
    assert second.used_distance_m == pytest.approx(2.721)
    assert second.fusion_mode == "depth_radar"
    assert runtime._vision_depth_fusion._pending_far_count == 0
    for now in (102.06, 102.10, 102.14):
        continued = sample(core, now)
        assert sensors.measurement.torso_recovery_status == "continued"
        assert continued.raw_distance_m == pytest.approx(2.721)
        assert continued.used_distance_m == pytest.approx(2.721)


def test_stricter_fusion_jump_guard_is_not_bypassed_by_two_region_claim(monkeypatch):
    runtime, sensors, sample, core = setup_chain(monkeypatch)
    runtime._vision_depth_fusion.config = replace(
        runtime._vision_depth_fusion.config, fresh_far_jump_m=0.4,
    )
    sample(2137, 100.0)
    sample(core, 101.9)
    guarded = sample(core, 102.02)
    assert sensors.measurement.raw_distance_m == pytest.approx(2.721)
    assert sensors.measurement.jump_confirmation is None
    assert guarded.fusion_mode == "depth_radar_jump_pending"
    assert runtime._vision_depth_fusion._pending_far_count == 1
    assert guarded.used_distance_m == pytest.approx(2.137)


def test_unexpired_visual_hold_retains_recovery_smoothing(monkeypatch):
    _runtime, _sensors, sample, _core = setup_chain(monkeypatch)
    sample(2137, 100.0)
    assert sample(0, 100.05).fusion_mode == "depth_visual"
    recovered = sample(2200, 100.10)
    assert recovered.fusion_mode == "depth_radar_recover"
    assert recovered.fusion_radar_distance_m == pytest.approx((2.137 + 2.2) / 2.0)
    assert recovered.used_distance_m == pytest.approx(2.137 + 0.6 * (2.1685 - 2.137))
