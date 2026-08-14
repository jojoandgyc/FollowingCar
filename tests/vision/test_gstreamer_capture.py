#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rk_vision.gstreamer_capture import GstMjpegTeeCapture


class _Gst:
    SECOND = 1000


class _Sample:
    def __init__(self, value: str) -> None:
        self.value = value


class _Appsink:
    def __init__(self, samples) -> None:
        self.samples = list(samples)
        self.calls = []

    def emit(self, name: str, timeout_ns: int):
        if name != "try-pull-sample":
            raise AssertionError(f"unexpected signal {name!r}")
        self.calls.append(timeout_ns)
        if not self.samples:
            return None
        return self.samples.pop(0)


def _fake_capture(samples):
    cap = GstMjpegTeeCapture.__new__(GstMjpegTeeCapture)
    cap.pipeline = object()
    cap.appsink = _Appsink(samples)
    cap.bus = None
    cap._gst = _Gst()
    cap._raise_bus_error_if_any = lambda: None
    cap._sample_to_bgr = lambda sample: sample.value
    return cap


def main() -> int:
    cap = _fake_capture([_Sample("old"), _Sample("middle"), _Sample("new")])
    ok, frame, drained = cap.read_latest(timeout_sec=0.5, max_drain=8)
    if not ok or frame != "new" or drained != 2:
        raise AssertionError((ok, frame, drained))
    if cap.appsink.calls != [500, 0, 0, 0]:
        raise AssertionError(f"unexpected pull timeouts: {cap.appsink.calls}")

    cap = _fake_capture([_Sample("old"), _Sample("next"), _Sample("left")])
    ok, frame, drained = cap.read_latest(timeout_sec=0.1, max_drain=1)
    if not ok or frame != "next" or drained != 1:
        raise AssertionError((ok, frame, drained))
    if [s.value for s in cap.appsink.samples] != ["left"]:
        raise AssertionError("max_drain should leave later samples queued")

    cap = _fake_capture([])
    ok, frame, drained = cap.read_latest(timeout_sec=0.0, max_drain=8)
    if ok or frame is not None or drained != 0:
        raise AssertionError((ok, frame, drained))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
