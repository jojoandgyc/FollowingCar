from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class GstMjpegTeeConfig:
    device: str
    width: int
    height: int
    fps: float
    raw_output: str
    appsink_max_buffers: int = 2
    appsink_queue_buffers: int = 4
    raw_queue_buffers: int = 180


@dataclass(frozen=True)
class GstAppsrcH264WriterConfig:
    output_path: str
    width: int
    height: int
    fps: float
    encoder: str = "mpph264enc"
    bitrate: int = 0
    queue_buffers: int = 4


class GstMjpegTeeCapture:
    """Single-owner MJPEG camera pipeline with raw passthrough plus BGR frames.

    The camera is opened only once by GStreamer. One tee branch stores the
    original MJPEG stream into an AVI container, while the other branch decodes
    frames for Python inference through appsink.
    """

    def __init__(self, config: GstMjpegTeeConfig) -> None:
        self.config = config
        self.pipeline = None
        self.appsink = None
        self.bus = None
        self._gst = None
        self.pipeline_description = ""
        self.actual = {
            "width": int(config.width),
            "height": int(config.height),
            "fps": float(config.fps),
            "fourcc": "MJPG",
        }

    def open(self) -> None:
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except Exception as exc:  # pragma: no cover - board dependency
            raise RuntimeError("PyGObject GStreamer bindings are required for gstreamer_mjpeg_tee") from exc

        self._gst = Gst
        Gst.init(None)
        if self.config.raw_output:
            Path(self.config.raw_output).parent.mkdir(parents=True, exist_ok=True)
        self.pipeline_description = self._build_pipeline_description()
        self.pipeline = Gst.parse_launch(self.pipeline_description)
        self.appsink = self.pipeline.get_by_name("appsink")
        if self.appsink is None:
            raise RuntimeError("GStreamer pipeline did not create appsink")
        self.bus = self.pipeline.get_bus()
        result = self.pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"Failed to start GStreamer pipeline:\n{self.pipeline_description}")
        state_result, _state, _pending = self.pipeline.get_state(5 * Gst.SECOND)
        if state_result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"GStreamer pipeline failed to reach PLAYING:\n{self.pipeline_description}")

    def read(self, timeout_sec: float = 2.0) -> Tuple[bool, Optional[Any]]:
        if self.pipeline is None or self.appsink is None or self._gst is None:
            raise RuntimeError("GStreamer capture is not open")
        self._raise_bus_error_if_any()
        sample = self.appsink.emit("try-pull-sample", int(max(0.0, timeout_sec) * self._gst.SECOND))
        if sample is None:
            self._raise_bus_error_if_any()
            return False, None
        return True, self._sample_to_bgr(sample)

    def read_latest(self, timeout_sec: float = 2.0, max_drain: int = 8) -> Tuple[bool, Optional[Any], int]:
        """Read the newest currently queued appsink frame.

        appsink returns the oldest sample in its queue. For realtime control, an
        older queued frame is worse than skipping work, so after the first
        blocking pull we drain immediately available samples with zero timeout
        and only convert the last one to BGR.
        """
        if self.pipeline is None or self.appsink is None or self._gst is None:
            raise RuntimeError("GStreamer capture is not open")
        self._raise_bus_error_if_any()
        sample = self.appsink.emit("try-pull-sample", int(max(0.0, timeout_sec) * self._gst.SECOND))
        if sample is None:
            self._raise_bus_error_if_any()
            return False, None, 0

        drained = 0
        for _ in range(max(0, int(max_drain))):
            next_sample = self.appsink.emit("try-pull-sample", 0)
            if next_sample is None:
                break
            sample = next_sample
            drained += 1

        return True, self._sample_to_bgr(sample), drained

    def release(self) -> None:
        if self.pipeline is None or self._gst is None:
            return
        Gst = self._gst
        try:
            self.pipeline.send_event(Gst.Event.new_eos())
            if self.bus is not None:
                self.bus.timed_pop_filtered(2 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
            self.appsink = None
            self.bus = None

    def _build_pipeline_description(self) -> str:
        fps = Fraction(float(self.config.fps)).limit_denominator(1001)
        device = _quote_gst_value(str(self.config.device))
        source = (
            f"v4l2src device={device} do-timestamp=true "
            f"! image/jpeg,width={int(self.config.width)},height={int(self.config.height)},"
            f"framerate={int(fps.numerator)}/{int(fps.denominator)} "
        )
        appsink_branch = (
            f"! queue max-size-buffers={int(self.config.appsink_queue_buffers)} leaky=downstream "
            "! jpegdec ! videoconvert ! video/x-raw,format=BGR "
            f"! appsink name=appsink emit-signals=false sync=false "
            f"max-buffers={int(self.config.appsink_max_buffers)} drop=true"
        )
        if not self.config.raw_output:
            return source + appsink_branch

        raw_location = _quote_gst_value(str(self.config.raw_output))
        return (
            source
            + "! tee name=t "
            + f"t. ! queue max-size-buffers={int(self.config.raw_queue_buffers)} leaky=0 "
            + f"! jpegparse ! avimux ! filesink location={raw_location} "
            + "t. "
            + appsink_branch
        )

    def _sample_to_bgr(self, sample) -> Any:
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("numpy is required for GStreamer frame extraction") from exc

        caps = sample.get_caps()
        if caps is None or caps.get_size() <= 0:
            raise RuntimeError("GStreamer appsink sample has no caps")
        structure = caps.get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        fmt = str(structure.get_value("format"))
        if fmt != "BGR":
            raise RuntimeError(f"expected BGR appsink frames, got {fmt!r}")

        buf = sample.get_buffer()
        ok, mapped = buf.map(self._gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("failed to map GStreamer sample buffer")
        try:
            data = np.frombuffer(mapped.data, dtype=np.uint8)
            expected = width * height * 3
            if data.size < expected:
                raise RuntimeError(f"short GStreamer frame: got {data.size} bytes, expected {expected}")
            frame = data[:expected].reshape((height, width, 3)).copy()
        finally:
            buf.unmap(mapped)
        self.actual = {"width": width, "height": height, "fps": float(self.config.fps), "fourcc": "MJPG"}
        return frame

    def _raise_bus_error_if_any(self) -> None:
        if self.bus is None or self._gst is None:
            return
        msg = self.bus.timed_pop_filtered(0, self._gst.MessageType.ERROR | self._gst.MessageType.EOS)
        if msg is None:
            return
        if msg.type == self._gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            raise RuntimeError(f"GStreamer error: {err}; debug={debug}")
        if msg.type == self._gst.MessageType.EOS:
            raise EOFError("GStreamer pipeline reached EOS")


def _quote_gst_value(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class GstAppsrcH264Writer:
    """BGR frame writer backed by GStreamer appsrc and Rockchip MPP H.264."""

    def __init__(self, config: GstAppsrcH264WriterConfig) -> None:
        self.config = config
        self.pipeline = None
        self.appsrc = None
        self.bus = None
        self._gst = None
        self._frame_index = 0
        self._duration_ns = _frame_duration_ns(float(config.fps))
        self.pipeline_description = ""

    def open(self) -> None:
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except Exception as exc:  # pragma: no cover - board dependency
            raise RuntimeError("PyGObject GStreamer bindings are required for gstreamer_mpp_h264") from exc

        self._gst = Gst
        Gst.init(None)
        Path(self.config.output_path).parent.mkdir(parents=True, exist_ok=True)
        self.pipeline_description = self._build_pipeline_description()
        self.pipeline = Gst.parse_launch(self.pipeline_description)
        self.appsrc = self.pipeline.get_by_name("appsrc")
        if self.appsrc is None:
            raise RuntimeError("GStreamer pipeline did not create appsrc")
        self.bus = self.pipeline.get_bus()
        result = self.pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"Failed to start GStreamer writer:\n{self.pipeline_description}")

    def write(self, frame: Any) -> None:
        if self.pipeline is None or self.appsrc is None or self._gst is None:
            self.open()
        self._raise_bus_error_if_any()
        np = _np()
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"expected BGR frame with shape HxWx3, got {arr.shape}")
        height, width = int(arr.shape[0]), int(arr.shape[1])
        if width != int(self.config.width) or height != int(self.config.height):
            raise ValueError(
                f"frame shape {width}x{height} does not match writer "
                f"{int(self.config.width)}x{int(self.config.height)}"
            )
        data = np.ascontiguousarray(arr).tobytes()
        buffer = self._gst.Buffer.new_allocate(None, len(data), None)
        buffer.fill(0, data)
        buffer.pts = int(self._frame_index * self._duration_ns)
        buffer.dts = buffer.pts
        buffer.duration = int(self._duration_ns)
        self._frame_index += 1
        ret = self.appsrc.emit("push-buffer", buffer)
        if ret != self._gst.FlowReturn.OK:
            raise RuntimeError(f"GStreamer writer push-buffer failed: {ret.value_nick}")

    def release(self) -> None:
        if self.pipeline is None or self._gst is None:
            return
        Gst = self._gst
        try:
            if self.appsrc is not None:
                self.appsrc.emit("end-of-stream")
            if self.bus is not None:
                msg = self.bus.timed_pop_filtered(5 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
                if msg is not None and msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    raise RuntimeError(f"GStreamer writer EOS error: {err}; debug={debug}")
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
            self.appsrc = None
            self.bus = None

    def isOpened(self) -> bool:
        if self.pipeline is None:
            self.open()
        return True

    def _build_pipeline_description(self) -> str:
        fps = Fraction(float(self.config.fps)).limit_denominator(1001)
        location = _quote_gst_value(str(self.config.output_path))
        encoder_props = ""
        if int(self.config.bitrate) > 0 and self.config.encoder == "mpph264enc":
            encoder_props = f" bps={int(self.config.bitrate)}"
        return (
            "appsrc name=appsrc is-live=true block=true format=time "
            f"caps=video/x-raw,format=BGR,width={int(self.config.width)},height={int(self.config.height)},"
            f"framerate={int(fps.numerator)}/{int(fps.denominator)} "
            f"! queue max-size-buffers={int(self.config.queue_buffers)} leaky=downstream "
            "! videoconvert "
            "! video/x-raw,format=NV12 "
            f"! {self.config.encoder}{encoder_props} "
            "! h264parse "
            "! mp4mux "
            f"! filesink location={location}"
        )

    def _raise_bus_error_if_any(self) -> None:
        if self.bus is None or self._gst is None:
            return
        msg = self.bus.timed_pop_filtered(0, self._gst.MessageType.ERROR)
        if msg is None:
            return
        err, debug = msg.parse_error()
        raise RuntimeError(f"GStreamer writer error: {err}; debug={debug}")


def _frame_duration_ns(fps: float) -> int:
    return int(round(1_000_000_000 / max(float(fps), 1e-9)))


def _np():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("numpy is required for GStreamer video writing") from exc
    return np
