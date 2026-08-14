from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

from bunker_hazard_detector import (
    BunkerHazardMonitor,
    HazardState as BunkerHazardState,
    RKNNBunkerHazardDetector,
    check_hazard_from_dets,
)


_RKNN_ENGINES = {"rknn", "rk", "rk3588", "lite", "lite2", "rknnlite"}


@dataclass(frozen=True)
class BunkerHazardRuntimeConfig:
    enabled: bool
    mode: str
    workdir: str
    model_path: str
    engine: str
    class_ids: Sequence[int]
    class_names: Dict[int, str]
    score_threshold: float
    area_ratio_stop: float
    num_classes: int
    rknn_input_size: int
    rknn_nms_threshold: float
    rknn_backend: str
    rknn_core_mask: str
    rknn_input_format: str
    rknn_box_format: str
    sample_det_conf: float
    get_frame_timeout_ms: int
    loop_period_ms: int
    split_restart_delay: float
    split_active_hold_sec: float
    stop_consec_frames: int
    split_echo_raw: bool
    sample_binary: str
    frame_width: int
    frame_height: int
    runtime_base: str
    rknn_target: str


def _resolve_existing_or_workdir_path(path: str, workdir: str) -> str:
    if os.path.isabs(path):
        return path
    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.abspath(os.path.join(workdir, path))


class BunkerHazardRuntime:
    """Own bunker/pond hazard detection and debouncing.

    The runtime returns BunkerHazardState objects.  It does not decide follow
    behavior and does not send motor commands; request_0513 bridges an active
    hazard into the action runtime so the safety stop remains immediate.
    """

    def __init__(
        self,
        config: BunkerHazardRuntimeConfig,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.monitor: Optional[BunkerHazardMonitor] = None
        self.rknn_detector: Optional[RKNNBunkerHazardDetector] = None
        self.last_stop_log_ts = 0.0
        self.active_frames = 0
        self.last_state_ts = 0.0

    def start(self) -> None:
        c = self.config
        if not c.enabled:
            return

        mode = c.mode
        if mode == "split":
            model_abs = _resolve_existing_or_workdir_path(c.model_path, c.workdir)
            if not os.path.exists(model_abs):
                raise RuntimeError(
                    f"沙坑/水坑模型文件不存在: {model_abs}。"
                    f"可设置 BUNKER_MODEL_PATH 或关闭 BUNKER_AVOID_ENABLE。"
                )
            if c.engine in _RKNN_ENGINES:
                self.rknn_detector = RKNNBunkerHazardDetector(
                    model_path=model_abs,
                    class_ids=tuple(c.class_ids),
                    class_names=c.class_names,
                    score_threshold=c.score_threshold,
                    area_ratio_stop=c.area_ratio_stop,
                    num_classes=c.num_classes,
                    det_conf=c.sample_det_conf,
                    input_size=c.rknn_input_size,
                    nms_threshold=c.rknn_nms_threshold,
                    input_format=c.rknn_input_format,
                    box_format=c.rknn_box_format,
                    target=c.rknn_target,
                    core_mask=c.rknn_core_mask,
                    backend=c.rknn_backend,
                )
                self.rknn_detector.start()
                self.logger.info(
                    "沙坑/水坑检测: split RKNN 模型已启动（model=%s, class_ids=%s, area_ratio_stop=%.3f, conf=%.3f）",
                    model_abs,
                    sorted(c.class_ids),
                    c.area_ratio_stop,
                    c.sample_det_conf,
                )
                return

            binary_abs = _resolve_existing_or_workdir_path(c.sample_binary, c.workdir)
            if not os.path.exists(c.workdir):
                raise RuntimeError(f"沙坑/水坑检测工作目录不存在: {c.workdir}")
            if not os.path.exists(binary_abs):
                raise RuntimeError(
                    f"sample_personv8_track 不存在: {binary_abs}。"
                    f"可设置 BUNKER_SAMPLE_BINARY/BUNKER_WORKDIR。"
                )
            self.monitor = BunkerHazardMonitor(
                model_path=model_abs,
                frame_width=c.frame_width,
                frame_height=c.frame_height,
                runtime_base=c.runtime_base,
                class_ids=tuple(c.class_ids),
                class_names=c.class_names,
                score_threshold=c.score_threshold,
                area_ratio_stop=c.area_ratio_stop,
                num_classes=c.num_classes,
                det_conf=c.sample_det_conf,
                get_frame_timeout_ms=c.get_frame_timeout_ms,
                loop_period_ms=c.loop_period_ms,
                binary_path=binary_abs,
                workdir=c.workdir,
                restart_delay=c.split_restart_delay,
                active_hold_sec=c.split_active_hold_sec,
                echo_raw=c.split_echo_raw,
                logger=self.logger,
            )
            self.monitor.start()
            self.logger.info(
                "沙坑/水坑检测: split 独立 sample 模型已启动（model=%s, class_ids=%s, area_ratio_stop=%.3f）",
                model_abs,
                sorted(c.class_ids),
                c.area_ratio_stop,
            )
            return

        if mode == "merged":
            self.logger.info(
                "沙坑/水坑检测: merged 主模型模式（class_ids=%s, area_ratio_stop=%.3f）",
                sorted(c.class_ids),
                c.area_ratio_stop,
            )
            return

        if mode == "off":
            self.logger.info("沙坑/水坑检测: BUNKER_DETECT_MODE=off，已关闭")
            return

        raise RuntimeError(f"未知 BUNKER_DETECT_MODE: {mode!r}")

    def close(self) -> None:
        try:
            if self.monitor is not None:
                self.monitor.stop()
                self.monitor = None
                self.logger.info("沙坑/水坑 split 检测已关闭")
        except Exception as exc:
            self.logger.warning("沙坑/水坑 split 检测关闭出错: %s", exc)
        try:
            if self.rknn_detector is not None:
                self.rknn_detector.stop()
                self.rknn_detector = None
                self.logger.info("沙坑/水坑 split RKNN 检测已关闭")
        except Exception as exc:
            self.logger.warning("沙坑/水坑 split RKNN 检测关闭出错: %s", exc)

    def _confirm_active(self, state: BunkerHazardState) -> Optional[BunkerHazardState]:
        if not state.active:
            self.active_frames = 0
            return None
        if state.updated_ts != self.last_state_ts:
            self.last_state_ts = state.updated_ts
            self.active_frames += 1
        if self.active_frames < max(1, int(self.config.stop_consec_frames)):
            return None
        return state

    def current_split_state(self) -> Optional[BunkerHazardState]:
        c = self.config
        if not c.enabled or c.mode != "split" or self.monitor is None:
            return None
        return self._confirm_active(self.monitor.get_state())

    def check_split_monitor(self) -> Optional[BunkerHazardState]:
        state = self.current_split_state()
        if state is None:
            return None
        self._log_active_state("独立沙坑/水坑模型触发安全停止", state)
        return state

    def check_split_frame(self, frame: Any, frame_format: str = "BGR") -> Optional[BunkerHazardState]:
        c = self.config
        if not c.enabled or c.mode != "split" or self.rknn_detector is None:
            return None

        try:
            state = self.rknn_detector.detect_state(frame, frame_format)
        except Exception as exc:
            now = time.time()
            if now - self.last_stop_log_ts >= 2.0:
                self.last_stop_log_ts = now
                self.logger.warning("沙坑/水坑 split RKNN 检测失败: %s", exc)
            return None

        state = self._confirm_active(state)
        if state is None:
            return None

        timing = getattr(self.rknn_detector, "last_timing_ms", {})
        now = time.time()
        if now - self.last_stop_log_ts >= 0.5:
            self.last_stop_log_ts = now
            self.logger.warning(
                "独立沙坑/水坑 RKNN 模型触发安全停止: %s(class_id=%s, score=%.3f, area_ratio=%.4f >= %.4f, bbox=%s, infer=%.1fms)",
                state.class_name,
                state.class_id,
                state.score,
                state.area_ratio,
                c.area_ratio_stop,
                [round(float(v), 1) for v in state.bbox],
                float(timing.get("total", 0.0)),
            )
        return state

    def check_merged_dets(
        self,
        dets: Iterable[dict],
        frame_area: float,
    ) -> Optional[BunkerHazardState]:
        c = self.config
        if not c.enabled or c.mode != "merged" or not c.class_ids:
            return None
        if frame_area <= 0:
            return None

        state = check_hazard_from_dets(
            dets=dets,
            frame_area=frame_area,
            class_ids=tuple(c.class_ids),
            score_threshold=c.score_threshold,
            area_ratio_stop=c.area_ratio_stop,
            class_names=c.class_names,
            source="merged",
        )
        if not state.active:
            return None

        now = time.time()
        if now - self.last_stop_log_ts >= 0.5:
            self.last_stop_log_ts = now
            self.logger.warning(
                "检测到安全危险类别 %s(class_id=%s, score=%.3f, area_ratio=%.4f >= %.4f, bbox=%s)，立即停止",
                state.class_name,
                state.class_id,
                state.score,
                state.area_ratio,
                c.area_ratio_stop,
                [round(float(v), 1) for v in state.bbox],
            )
        return state

    def _log_active_state(self, prefix: str, state: BunkerHazardState) -> None:
        now = time.time()
        if now - self.last_stop_log_ts < 0.5:
            return
        self.last_stop_log_ts = now
        self.logger.warning(
            "%s: %s(class_id=%s, score=%.3f, area_ratio=%.4f >= %.4f, bbox=%s)",
            prefix,
            state.class_name,
            state.class_id,
            state.score,
            state.area_ratio,
            self.config.area_ratio_stop,
            [round(float(v), 1) for v in state.bbox],
        )
