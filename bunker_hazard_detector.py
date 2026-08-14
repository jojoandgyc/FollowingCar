#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Background bunker/pond hazard detection for the follow request script.

The current deployment may need a separate bunker/pond model.  This module
keeps that path isolated so the request script can later switch to a merged
person+bunker model by reading detections from the main model instead.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_BBOX_RE = re.compile(
    r"\[\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*"
    r"\]"
)


@dataclass(frozen=True)
class HazardDetection:
    class_id: int
    class_name: str
    score: float
    bbox: Tuple[float, float, float, float]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class HazardState:
    active: bool = False
    class_id: int = -1
    class_name: str = ""
    score: float = 0.0
    area_ratio: float = 0.0
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    target_count: int = 0
    updated_ts: float = 0.0
    source: str = "none"
    stale: bool = False

    def reason(self) -> str:
        if not self.active:
            return ""
        return (
            f"{self.source}:{self.class_name or self.class_id}:"
            f"score={self.score:.3f},area_ratio={self.area_ratio:.4f}"
        )


def check_hazard_from_dets(
    dets: Iterable[dict],
    frame_area: float,
    class_ids: Sequence[int],
    score_threshold: float,
    area_ratio_stop: float,
    class_names: Optional[Dict[int, str]] = None,
    source: str = "merged",
) -> HazardState:
    if frame_area <= 0:
        return HazardState(source=source, updated_ts=time.time())

    wanted = set(int(x) for x in class_ids)
    names = class_names or {}
    best: Optional[HazardDetection] = None
    target_count = 0

    for det in dets:
        class_id = int(det.get("class_id", -1))
        score = float(det.get("score", 0.0))
        if class_id not in wanted or score < float(score_threshold):
            continue
        x1, y1, x2, y2 = det["bbox"]
        rec = HazardDetection(
            class_id=class_id,
            class_name=names.get(class_id, f"class_{class_id}"),
            score=score,
            bbox=(float(x1), float(y1), float(x2), float(y2)),
        )
        target_count += 1
        if best is None or rec.area > best.area:
            best = rec

    now = time.time()
    if best is None:
        return HazardState(target_count=0, updated_ts=now, source=source)

    ratio = best.area / float(frame_area)
    return HazardState(
        active=ratio >= float(area_ratio_stop),
        class_id=best.class_id,
        class_name=best.class_name,
        score=best.score,
        area_ratio=ratio,
        bbox=best.bbox,
        target_count=target_count,
        updated_ts=now,
        source=source,
    )


class SampleBunkerDetectWrapper:
    """Launch sample_personv8_track and parse bbox output for hazard model."""

    def __init__(
        self,
        model_path: str,
        runtime_base: str,
        binary_path: str = "./sample_personv8_track",
        workdir: Optional[str] = None,
        num_classes: int = 2,
        det_conf: float = 0.75,
        get_frame_timeout_ms: int = 200,
        loop_period_ms: int = 0,
        echo_raw: bool = False,
    ) -> None:
        self.model_path = model_path
        self.runtime_base = runtime_base
        self.binary_path = binary_path
        self.workdir = workdir
        self.num_classes = int(num_classes)
        self.det_conf = float(det_conf)
        self.get_frame_timeout_ms = int(get_frame_timeout_ms)
        self.loop_period_ms = int(loop_period_ms)
        self.echo_raw = bool(echo_raw)
        self._proc: Optional[subprocess.Popen[str]] = None

    def _build_child_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        base = self.runtime_base
        lib_paths = [
            self.workdir or os.getcwd(),
            os.path.join(base, "cvi_rtsp/lib"),
            os.path.join(base, "cvitek_tpu_sdk/lib"),
            os.path.join(base, "tdl_sdk/lib"),
            os.path.join(base, "zkwl-python-sdk/lib"),
            os.path.join(base, "opencv/lib"),
            os.path.join(base, "opencv/lib64"),
            os.path.join(base, "python3/lib"),
            os.path.join(base, "sys_sdk/lib"),
            os.path.join(base, "sys_sdk/lib/3rd"),
        ]
        old_ld = env.get("LD_LIBRARY_PATH", "")
        if old_ld:
            lib_paths.append(old_ld)
        env["LD_LIBRARY_PATH"] = ":".join(lib_paths)
        env["Y8_SINGLE_PERSON"] = "0"
        env["Y8_DETECT_ALL_CLASSES"] = "1"
        env["Y8_NUM_CLASSES"] = str(max(1, self.num_classes))
        env["Y8_DET_CONF"] = f"{self.det_conf:.3f}"
        env["Y8_GET_FRAME_TIMEOUT_MS"] = str(max(0, self.get_frame_timeout_ms))
        env["Y8_LOOP_PERIOD_MS"] = str(max(0, self.loop_period_ms))
        return env

    def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            [self.binary_path, self.model_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=self._build_child_env(),
            cwd=self.workdir,
        )

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                self._proc.send_signal(signal.SIGINT)
                self._proc.wait(timeout=2.0)
            except Exception:
                self._proc.kill()
        self._proc = None

    def iter_frame_dets(self) -> Iterable[List[dict]]:
        if self._proc is None:
            raise RuntimeError("sample bunker wrapper not started")
        assert self._proc.stdout is not None

        for raw_line in self._proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            if self.echo_raw:
                print(line)

            recs: List[dict] = []
            for m in _BBOX_RE.finditer(line):
                recs.append(
                    {
                        "bbox": (
                            float(m.group(1)),
                            float(m.group(2)),
                            float(m.group(3)),
                            float(m.group(4)),
                        ),
                        "class_id": int(float(m.group(5))),
                        "score": float(m.group(6)),
                    }
                )
            yield recs

        code = self._proc.poll()
        if code not in (None, 0):
            raise RuntimeError(f"sample_personv8_track exited with code {code}")


class RKNNBunkerHazardDetector:
    """Run a RKNN YOLO bunker/pond model on frames supplied by the main loop."""

    def __init__(
        self,
        model_path: str,
        class_ids: Sequence[int],
        class_names: Optional[Dict[int, str]] = None,
        score_threshold: float = 0.0,
        area_ratio_stop: float = 0.05,
        num_classes: int = 2,
        det_conf: float = 0.25,
        input_size: int = 640,
        nms_threshold: float = 0.45,
        input_format: str = "RGB",
        box_format: str = "xywh",
        target: str = "rk3588",
        core_mask: str = "auto",
        backend: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.class_ids = tuple(int(x) for x in class_ids)
        self.class_names = class_names or {0: "bunker", 1: "pond"}
        self.score_threshold = float(score_threshold)
        self.area_ratio_stop = float(area_ratio_stop)
        self.num_classes = int(num_classes)
        self.det_conf = float(det_conf)
        self.input_size = int(input_size)
        self.nms_threshold = float(nms_threshold)
        self.input_format = input_format
        self.box_format = box_format
        self.target = target
        self.core_mask = core_mask
        self.backend = backend
        self._detector = None
        self.last_detections: List[dict] = []
        self.last_timing_ms: Dict[str, float] = {}

    def start(self) -> None:
        self.load()

    def load(self) -> None:
        if self._detector is not None:
            return
        from rk_vision.yolo11 import YOLO11Config, YOLO11RKNNDetector

        detector = YOLO11RKNNDetector(
            YOLO11Config(
                model_path=self.model_path,
                input_size=self.input_size,
                conf_threshold=max(float(self.score_threshold), float(self.det_conf)),
                nms_threshold=self.nms_threshold,
                num_classes=self.num_classes,
                input_format=self.input_format,
                output_box_format=self.box_format,
                target=self.target,
                core_mask=self.core_mask,
                backend=self.backend,
            )
        )
        detector.load()
        self._detector = detector

    def stop(self) -> None:
        detector = self._detector
        if detector is not None:
            detector.release()
        self._detector = None

    def detect_state(self, frame: Any, frame_format: str = "BGR") -> HazardState:
        from rk_vision.frames import numpy_from_frame

        self.load()
        assert self._detector is not None
        _arr, width, height, _fmt = numpy_from_frame(frame, frame_format)
        detections = self._detector.detect(frame, frame_format)
        dets = [
            {
                "class_id": int(det.class_id),
                "score": float(det.score),
                "bbox": tuple(float(v) for v in det.bbox),
            }
            for det in detections
        ]
        self.last_detections = dets
        self.last_timing_ms = dict(getattr(self._detector, "last_timing_ms", {}))
        return check_hazard_from_dets(
            dets=dets,
            frame_area=float(max(1, int(width) * int(height))),
            class_ids=self.class_ids,
            score_threshold=self.score_threshold,
            area_ratio_stop=self.area_ratio_stop,
            class_names=self.class_names,
            source="split_rknn",
        )


class BunkerHazardMonitor:
    """Run split bunker/pond model in a background thread and expose state."""

    def __init__(
        self,
        model_path: str,
        frame_width: int,
        frame_height: int,
        runtime_base: str,
        class_ids: Sequence[int],
        class_names: Optional[Dict[int, str]] = None,
        score_threshold: float = 0.0,
        area_ratio_stop: float = 0.05,
        num_classes: int = 2,
        det_conf: float = 0.75,
        get_frame_timeout_ms: int = 200,
        loop_period_ms: int = 0,
        binary_path: str = "./sample_personv8_track",
        workdir: Optional[str] = None,
        restart_delay: float = 0.8,
        active_hold_sec: float = 0.8,
        echo_raw: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model_path = model_path
        self.frame_area = float(max(1, int(frame_width) * int(frame_height)))
        self.runtime_base = runtime_base
        self.class_ids = tuple(int(x) for x in class_ids)
        self.class_names = class_names or {0: "bunker", 1: "pond"}
        self.score_threshold = float(score_threshold)
        self.area_ratio_stop = float(area_ratio_stop)
        self.num_classes = int(num_classes)
        self.det_conf = float(det_conf)
        self.get_frame_timeout_ms = int(get_frame_timeout_ms)
        self.loop_period_ms = int(loop_period_ms)
        self.binary_path = binary_path
        self.workdir = workdir
        self.restart_delay = float(restart_delay)
        self.active_hold_sec = float(active_hold_sec)
        self.echo_raw = bool(echo_raw)
        self.logger = logger or logging.getLogger("BunkerHazard")

        self._state = HazardState(source="split")
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._wrapper: Optional[SampleBunkerDetectWrapper] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="BunkerHazardMonitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        wrapper = self._wrapper
        if wrapper is not None:
            wrapper.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def get_state(self) -> HazardState:
        with self._state_lock:
            state = self._state
        if not state.active:
            return state
        age = time.time() - float(state.updated_ts)
        if age <= max(0.05, self.active_hold_sec):
            return state
        return HazardState(
            active=False,
            class_id=state.class_id,
            class_name=state.class_name,
            score=state.score,
            area_ratio=state.area_ratio,
            bbox=state.bbox,
            target_count=state.target_count,
            updated_ts=state.updated_ts,
            source=state.source,
            stale=True,
        )

    def _set_state(self, state: HazardState) -> None:
        with self._state_lock:
            self._state = state

    def _run(self) -> None:
        restart_count = 0
        while not self._stop_event.is_set():
            wrapper = SampleBunkerDetectWrapper(
                model_path=self.model_path,
                runtime_base=self.runtime_base,
                binary_path=self.binary_path,
                workdir=self.workdir,
                num_classes=self.num_classes,
                det_conf=self.det_conf,
                get_frame_timeout_ms=self.get_frame_timeout_ms,
                loop_period_ms=self.loop_period_ms,
                echo_raw=self.echo_raw,
            )
            self._wrapper = wrapper
            try:
                wrapper.start()
                self.logger.info("split bunker detector started: model=%s restart=%d", self.model_path, restart_count)
                for dets in wrapper.iter_frame_dets():
                    if self._stop_event.is_set():
                        break
                    state = check_hazard_from_dets(
                        dets=dets,
                        frame_area=self.frame_area,
                        class_ids=self.class_ids,
                        score_threshold=self.score_threshold,
                        area_ratio_stop=self.area_ratio_stop,
                        class_names=self.class_names,
                        source="split",
                    )
                    self._set_state(state)
            except Exception as e:
                self.logger.warning("split bunker detector error: %s", e)
                self._set_state(HazardState(source="split", updated_ts=time.time()))
            finally:
                wrapper.stop()
                self._wrapper = None

            if self._stop_event.is_set():
                break
            restart_count += 1
            time.sleep(max(0.0, self.restart_delay))
