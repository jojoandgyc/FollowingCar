#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify enabled follow-car peripherals before the control runtime starts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from car_control_modular.config_loader import load_config_to_env


EXIT_FAILED = 1


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}


def _absolute_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _require_accessible_device(label: str, raw_path: str) -> None:
    path = Path(raw_path)
    if not path.exists():
        raise RuntimeError(f"{label}不存在: {path}")
    if not os.access(path, os.R_OK | os.W_OK):
        raise RuntimeError(f"{label}没有读写权限: {path}")
    print(f"外设自检通过: {label}={path}", flush=True)


def _check_ir() -> Callable[[], None]:
    from ir_hal import IR

    if IR.init() != 0:
        raise RuntimeError("三路红外初始化或读取失败")
    try:
        states = {
            "前": bool(IR.is_triggered(IR.IDX_1)),
            "左": bool(IR.is_triggered(IR.IDX_2)),
            "右": bool(IR.is_triggered(IR.IDX_0)),
        }
        print(f"外设自检通过: 三路红外可读取，触发状态={states}", flush=True)
    except Exception:
        IR.deinit()
        raise
    return IR.deinit


def _check_ultrasonic() -> Callable[[], None]:
    from utrasonic_hal import Utrasonic

    if Utrasonic.init() != 0:
        raise RuntimeError("超声波初始化或IIO节点读取失败")
    distance_cm = Utrasonic.get_distance()
    if distance_cm is None:
        print("外设自检通过: 超声波IIO节点可读取，当前没有有效距离回波", flush=True)
    else:
        print(f"外设自检通过: 超声波距离={float(distance_cm):.1f}厘米", flush=True)
    return Utrasonic.deinit


def _check_imu(timeout_sec: float) -> Callable[[], None]:
    from imu_hal import IMU

    if IMU.init() != 0:
        raise RuntimeError("ICM20600 IMU初始化失败")
    deadline = time.monotonic() + max(0.2, timeout_sec)
    snapshot = {"accel": None, "gyro": None}
    while time.monotonic() < deadline:
        snapshot = IMU.poll(min(0.2, max(0.0, deadline - time.monotonic())))
        if snapshot.get("accel") and snapshot.get("gyro"):
            break
    if not snapshot.get("accel") or not snapshot.get("gyro"):
        IMU.deinit()
        raise RuntimeError("ICM20600 IMU已打开，但没有同时收到加速度和陀螺仪样本")
    print("外设自检通过: ICM20600加速度和陀螺仪均有数据", flush=True)
    return IMU.deinit


def _check_astra_camera() -> None:
    from car_control_modular.astra_depth import AstraDepthConfig, AstraDepthRuntime
    import cv2

    runtime = AstraDepthRuntime(
        AstraDepthConfig(
            openni_path=os.environ.get(
                "ASTRA_DEPTH_OPENNI_PATH", "/home/topeet/AstraSDK/arm64_openni2"
            ),
            width=int(os.environ.get("ASTRA_DEPTH_WIDTH", "640")),
            height=int(os.environ.get("ASTRA_DEPTH_HEIGHT", "480")),
            fps=int(os.environ.get("ASTRA_DEPTH_FPS", "30")),
        )
    )
    try:
        runtime.start()
        if not runtime.wait_until_ready(2.0):
            raise RuntimeError("Astra Depth流已打开但没有收到深度帧")
        camera_device = os.environ.get("RKNN_CAMERA_DEVICE", "/dev/video3")
        _require_accessible_device("Astra RGB摄像头节点", camera_device)
        capture = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
        try:
            capture.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*os.environ.get("RKNN_CAMERA_FOURCC", "YUYV")),
            )
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(os.environ.get("RKNN_CAMERA_WIDTH", "640")))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(os.environ.get("RKNN_CAMERA_HEIGHT", "480")))
            capture.set(cv2.CAP_PROP_FPS, float(os.environ.get("RKNN_CAMERA_FPS", "30")))
            if not capture.isOpened():
                raise RuntimeError(f"Astra RGB摄像头无法打开: {camera_device}")
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Astra Depth运行期间RGB流没有收到彩色帧")
        finally:
            capture.release()
        print(
            f"外设自检通过: Astra RGB+Depth={frame.shape[1]}x{frame.shape[0]}，Depth已硬件配准到RGB",
            flush=True,
        )
    finally:
        runtime.close()


def _check_vision_files() -> None:
    if not _env_bool("MODULE_VISION_ENABLE", True):
        return
    model_path = _absolute_path(os.environ.get("VISION_MODEL_PATH", "models/yolo11n_int8_person_val2017.rknn"))
    if not model_path.is_file():
        raise RuntimeError(f"RKNN检测模型不存在: {model_path}")
    if _env_bool("VISION_REID_ENABLE", True):
        reid_path = _absolute_path(os.environ.get("VISION_REID_MODEL_PATH", "models/osnet_x0_25_msmt17_b1.rknn"))
        if not reid_path.is_file():
            raise RuntimeError(f"RKNN ReID模型不存在: {reid_path}")
    if _env_bool("RKNN_CAMERA_ENABLE", True):
        capture_mode = os.environ.get("RKNN_CAMERA_CAPTURE_MODE", "gstreamer_mjpeg").strip().lower()
        camera_device = os.environ.get("RKNN_CAMERA_DEVICE", "/dev/video1")
        if _env_bool("MODULE_ASTRA_DEPTH_ENABLE", False):
            _check_astra_camera()
        elif capture_mode in {"gst", "gstreamer", "gstreamer_mjpeg", "gstreamer_mjpeg_tee"}:
            _require_accessible_device("摄像头节点", camera_device)
            from rk_vision.gstreamer_capture import GstMjpegTeeCapture, GstMjpegTeeConfig

            capture = GstMjpegTeeCapture(
                GstMjpegTeeConfig(
                    device=camera_device,
                    width=int(os.environ.get("RKNN_CAMERA_WIDTH", "1920")),
                    height=int(os.environ.get("RKNN_CAMERA_HEIGHT", "1080")),
                    fps=float(os.environ.get("RKNN_CAMERA_FPS", "30")),
                    raw_output="",
                )
            )
            try:
                capture.open()
                ok, frame = capture.read(timeout_sec=3.0)
                if not ok or frame is None:
                    raise RuntimeError(f"摄像头已打开但没有画面: {camera_device}")
            finally:
                capture.release()
        else:
            _require_accessible_device("摄像头节点", camera_device)
            import cv2

            capture = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
            try:
                if not capture.isOpened():
                    raise RuntimeError(f"摄像头无法打开: {camera_device}")
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"摄像头已打开但没有画面: {camera_device}")
            finally:
                capture.release()
        print("外设自检通过: 摄像头已取得一帧有效画面", flush=True)
    print("外设自检通过: RKNN模型文件完整", flush=True)


def _check_motor_controller() -> None:
    from car_control_modular.mssd_motor import MssdMotorBackend, MssdMotorConfig

    port = os.environ.get("MOTOR_RS485_PORT", "/dev/ttyS0")
    _require_accessible_device("电机RS485串口", port)
    backend = MssdMotorBackend(
        MssdMotorConfig(
            port=port,
            slave_id=int(os.environ.get("MOTOR_RS485_SLAVE_ID", "1")),
            baudrate=int(os.environ.get("MOTOR_RS485_BAUDRATE", "9600")),
            timeout=float(os.environ.get("MOTOR_RS485_TIMEOUT", "0.3")),
            lib_dir=os.environ.get("MOTOR_RS485_LIB_DIR", "/home/topeet/lianzhan"),
            max_target=int(os.environ.get("MOTOR_RS485_MAX_TARGET", "100")),
            percent_limit=int(os.environ.get("MOTOR_PERCENT_LIMIT", "100")),
            left_sign=int(os.environ.get("MOTOR_LEFT_SIGN", "-1")),
            right_sign=int(os.environ.get("MOTOR_RIGHT_SIGN", "1")),
            forward_target_sign=-1 if int(os.environ.get("MOTOR_FORWARD_TARGET_SIGN", "1")) < 0 else 1,
            m1_is_left_wheel=_env_bool("M1_IS_LEFT_WHEEL", False),
            exit_parking_mode_on_arm=_env_bool("MOTOR_EXIT_PARKING_MODE_ON_ARM", False),
            stop_mode=os.environ.get("MOTOR_RS485_STOP_MODE", "normal"),
            stop_zero_delay_sec=max(0.0, float(os.environ.get("MOTOR_RS485_STOP_ZERO_DELAY_SEC", "0.03"))),
            # 预检只验证串口通信和双轮停车，不允许提前接合电子驻车。
            # 正式进程会在传感器初始化完成后执行一次受控的 0A -> 1A -> 5A 锁相。
            parking_current_a=0.0,
            startup_parking_enabled=False,
        )
    )
    try:
        backend.send_stop("startup_preflight")
    finally:
        backend.close()
    print("外设自检通过: 电机控制器通信正常，双轮已清零停车", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="跟随车启动前外设自检")
    parser.add_argument("--config", required=True, help="运行配置文件")
    parser.add_argument("--imu-timeout", type=float, default=1.0, help="等待IMU完整样本的秒数")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_config_to_env(args.config)
    cleanup: list[Callable[[], None]] = []
    try:
        _check_vision_files()
        _check_motor_controller()
        if _env_bool("MODULE_IR_ENABLE", False):
            cleanup.append(_check_ir())
        if _env_bool("MODULE_ULTRASONIC_ENABLE", False):
            cleanup.append(_check_ultrasonic())
        if _env_bool("MODULE_IMU_ENABLE", False):
            cleanup.append(_check_imu(args.imu_timeout))
        if _env_bool("MODULE_MMWAVE_ENABLE", False):
            print("毫米波将在正式进程内执行USB复位和数据验证，并持续保持同一串口句柄", flush=True)
        print("基础外设自检通过，可以进入正式跟随进程", flush=True)
        return 0
    except Exception as exc:
        print(f"外设自检失败: {exc}", file=sys.stderr, flush=True)
        return EXIT_FAILED
    finally:
        for close in reversed(cleanup):
            try:
                close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
