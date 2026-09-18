#!/usr/bin/env python3
"""Camera-only RGB/Depth orientation preview. Never imports motor/controller code."""
import argparse
import logging
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="显式打开相机；请先退出跟随程序并禁用电机电源")
    parser.add_argument("--device", default="/dev/v4l/by-id/usb-Astra_Pro_HD_Camera_Astra_Pro_HD_Camera-video-index0")
    parser.add_argument("--openni-path", default="/home/topeet/AstraSDK/arm64_openni2")
    parser.add_argument("--fourcc", default="YUYV", help="与当前 UVC 配置相同的格式")
    parser.add_argument("--duration", type=float, default=45.0, help="预览秒数，按 q 提前退出")
    args = parser.parse_args()
    if not args.live:
        parser.error("必须显式传入 --live；本工具仅打开相机，不连接串口或发送电机指令")
    if not 0 < args.duration <= 300 or len(args.fourcc) != 4:
        parser.error("duration 必须在 (0, 300]，fourcc 必须是 4 个字符")
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        parser.error("请在小车本机图形桌面终端运行（需要显示预览窗口）")

    import cv2
    import numpy as np
    from car_control_modular.astra_depth import AstraDepthConfig, AstraDepthRuntime

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print("请先关闭跟随程序、禁用电机电源。本工具不会停止其他正在运行的控制程序。", flush=True)
    print("让人依次站在画面左/中/右，静止 2 秒；观察 RGB、Depth、叠加图的人体轮廓是否同侧对齐。", flush=True)
    sensor = AstraDepthRuntime(AstraDepthConfig(openni_path=args.openni_path))
    camera = None
    title = "RGB | normalized Depth | overlay (q: quit)"
    try:
        sensor.start()
        camera = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not camera.isOpened():
            raise RuntimeError(f"无法打开 UVC：{args.device}")
        if not sensor.wait_until_ready(3.0):
            raise RuntimeError("3 秒内未收到 Depth")
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 1440, 360)
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            ok, rgb = camera.read()
            rgb_arrival = time.monotonic()
            if not ok:
                raise RuntimeError("UVC 图像读取失败")
            # Match arrival times for a STATIC scene check, not an exposure-
            # synchronization calibration. Arrays are immutable after ingest.
            with sensor._depth_lock:
                depth, stamp, _ = sensor._aligned_depth_locked(
                    rgb_arrival, reference_timestamp=rgb_arrival,
                )
            if depth is None or rgb_arrival - stamp > .25:
                raise RuntimeError("Depth 缺失或超过 250ms，停止预览")
            valid = (depth >= 350) & (depth <= 5000)
            scaled = (255 * (1 - np.clip(depth.astype(float) / 5000, 0, 1))).astype(np.uint8)
            color = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
            color[~valid] = 0
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
            overlay = cv2.addWeighted(rgb, .55, color, .45, 0)
            panels = [rgb, color, overlay]
            labels = ["RGB (unchanged)", "Depth (normalized ONCE)",
                      f"Overlay arrival delta={(rgb_arrival-stamp)*1000:.0f}ms"]
            for panel, label in zip(panels, labels):
                cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
                for x in (panel.shape[1] // 3, panel.shape[1] * 2 // 3):
                    cv2.line(panel, (x, 35), (x, panel.shape[0]-1), (255, 255, 255), 1)
            cv2.imshow(title, np.hstack(panels))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if camera is not None:
            camera.release()
        sensor.close()
        cv2.destroyAllWindows()
    print("预览结束。只有左/中/右轮廓均对齐，才进行后续低速跟随验证；本工具不自动判定标定通过。")


if __name__ == "__main__":
    main()
