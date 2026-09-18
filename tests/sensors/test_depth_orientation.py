"""Mirror policy and the real ingest path, with no device access."""
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from car_control_modular.astra_depth import AstraDepthConfig, AstraDepthRuntime
from car_control_modular.depth_orientation import (
    DepthOrientation, configure_orientation, read_orientation,
)

LOG = logging.getLogger(__name__)


class Stream:
    def __init__(self, mirrored=True, *, ignores_set=False, read_error=False, set_error=False):
        self.mirrored = mirrored
        self.ignores_set, self.read_error, self.set_error = ignores_set, read_error, set_error
        self.writes = []

    def get_mirroring_enabled(self):
        if self.read_error:
            raise RuntimeError("unsupported")
        return self.mirrored

    def set_mirroring_enabled(self, value):
        self.writes.append(value)
        if self.set_error:
            raise RuntimeError("read only")
        if not self.ignores_set:
            self.mirrored = value


def test_sdk_normalization_means_no_software_flip(caplog):
    depth, color = Stream(), Stream()
    with caplog.at_level(logging.INFO):
        plan = configure_orientation(depth, color, LOG)
    assert depth.writes == color.writes == [False]
    assert plan == DepthOrientation(False, False, False)
    raw = np.array([[0, 349, 400, 1500, 65535]], dtype=np.uint16)
    assert plan.normalize(raw) is raw
    assert "software_flip_x=False" in caplog.text


@pytest.mark.parametrize("set_error", [False, True])
def test_known_mirrored_depth_has_exactly_one_lossless_flip(set_error):
    plan = configure_orientation(Stream(ignores_set=True, set_error=set_error), Stream(), LOG)
    raw = np.array([[0, 349, 400, 1500, 65535]], dtype=np.uint16)
    result = plan.normalize(raw)
    np.testing.assert_array_equal(result, [[65535, 1500, 400, 349, 0]])
    assert result.dtype == np.uint16 and result.flags.c_contiguous
    assert raw[0, 0] == 0  # producer image not mutated


def test_failed_setter_with_known_false_is_safe():
    plan = configure_orientation(Stream(False, set_error=True), Stream(False, set_error=True), LOG)
    assert not plan.software_flip_x


@pytest.mark.parametrize("depth,color", [
    (Stream(read_error=True), Stream()),
    (Stream(), Stream(read_error=True)),
    (Stream(), Stream(ignores_set=True)),
    (Stream(None, ignores_set=True), Stream()),
])
def test_unknown_or_mirrored_registration_color_fails_closed(depth, color):
    with pytest.raises(RuntimeError):
        configure_orientation(depth, color, LOG)


def test_readback_after_start_detects_property_reset():
    depth, color = Stream(), Stream()
    assert not configure_orientation(depth, color, LOG).software_flip_x
    depth.mirrored = True  # driver changes its state when starting
    assert read_orientation(depth, color, LOG, stage="started").software_flip_x


@pytest.mark.parametrize("reset_mirror,unreadable", [(False, False), (True, False), (False, True)])
def test_real_start_verifies_after_stream_start_before_reader(monkeypatch, reset_mirror, unreadable):
    from car_control_modular import astra_depth
    depth, color = Stream(), Stream()
    for stream in (depth, color):
        stream.set_video_mode = lambda mode: None
        stream.stop = lambda: None
        stream.close = lambda: None
    def start_depth():
        depth.mirrored = reset_mirror
        depth.read_error = unreadable
    depth.start = start_depth
    device = SimpleNamespace(
        create_color_stream=lambda: color, create_depth_stream=lambda: depth,
        is_image_registration_mode_supported=lambda mode: True,
        set_image_registration_mode=lambda mode: None,
        get_image_registration_mode=lambda: 1,
        get_device_info=lambda: "fake", close=lambda: None,
    )
    fake_openni = SimpleNamespace(
        initialize=lambda path: None, unload=lambda: None,
        Device=SimpleNamespace(open_any=lambda: device), VideoMode=lambda **kw: kw,
        PIXEL_FORMAT_RGB888=1, PIXEL_FORMAT_DEPTH_1_MM=2, IMAGE_REGISTRATION_DEPTH_TO_COLOR=1,
    )
    monkeypatch.setitem(sys.modules, "openni", SimpleNamespace(openni2=fake_openni))
    reader_starts = []
    monkeypatch.setattr(astra_depth.threading, "Thread", lambda **kw: SimpleNamespace(
        start=lambda: reader_starts.append(True), join=lambda **kw: None,
    ))
    sensor = AstraDepthRuntime(AstraDepthConfig())
    try:
        if unreadable:
            with pytest.raises(RuntimeError, match="镜像状态无法读回"):
                sensor.start()
            assert not reader_starts and not sensor._started
            assert sensor._depth_stream is None
        else:
            sensor.start()
            assert sensor._depth_orientation.software_flip_x == reset_mirror
            assert reader_starts == [True]
    finally:
        sensor.close()


def test_reader_caches_and_records_same_normalized_pixels():
    sensor = AstraDepthRuntime(AstraDepthConfig())
    sensor._np = np
    sensor._depth_orientation = DepthOrientation(True, False, True)
    raw = np.array([[0, 400, 1800], [65535, 8000, 1500]], dtype=np.uint16)
    frame = SimpleNamespace(width=3, height=2, get_buffer_as_uint16=lambda: raw.tobytes())
    def read():
        sensor._stop_event.set()  # exactly one iteration; no reader thread
        return frame
    recorded = []
    sensor._depth_stream = SimpleNamespace(read_frame=read)
    sensor._openni2 = SimpleNamespace(wait_for_any_stream=lambda *a: 0)
    sensor.diagnostics = SimpleNamespace(add_depth=lambda stamp, pixels: recorded.append((stamp, pixels)))
    sensor._depth_loop()
    np.testing.assert_array_equal(sensor._latest_depth, raw[:, ::-1])
    assert sensor._depth_history[0][1] is sensor._latest_depth is recorded[0][1]
    assert sensor._depth_history[0][0] == sensor._latest_depth_ts == recorded[0][0]


@pytest.mark.parametrize("left,right", [(40, 200), (240, 400), (440, 600)])
def test_left_center_right_rois_follow_normalized_depth(left, right):
    import time
    raw = np.full((480, 640), 5000, dtype=np.uint16)
    raw[60:420, 640-right:640-left] = 1500  # mirrored sensor output
    sensor = AstraDepthRuntime(AstraDepthConfig(median_window=1))
    sensor._np = np
    sensor._latest_depth = DepthOrientation(True, False, True).normalize(raw)
    sensor._latest_depth_ts = time.monotonic()
    result = sensor.measure_target((left, 60, right, 420), 640, 480, target_id=1, use_latest_depth=True)
    assert result.raw_distance_m == pytest.approx(1.5)


def test_observation_metadata_marks_already_normalized_space():
    import time
    sensor = AstraDepthRuntime(AstraDepthConfig())
    sensor._np = np
    sensor._depth_orientation = DepthOrientation(True, False, True)
    sensor._latest_depth = np.full((480, 640), 1500, dtype=np.uint16)
    sensor._latest_depth_ts = time.monotonic()
    records = []
    sensor.diagnostics = SimpleNamespace(observe=lambda metadata, **kwargs: records.append(metadata))
    sensor.measure_target((220, 80, 420, 400), 640, 480, target_id=1, use_latest_depth=True)
    assert json.loads(json.dumps(records[0]))["orientation"] == sensor._depth_orientation.metadata()
