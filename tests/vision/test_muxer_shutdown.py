"""Recorder subprocess signal isolation, without a camera or motors."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from car_control_modular.fast_mjpeg_writer import FastMjpegWriter


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg unavailable")
def test_muxer_owns_separate_session_and_closes_normally(tmp_path):
    writer = FastMjpegWriter(tmp_path / "session.avi", 30, (160,120), cv2)
    child = writer._process.pid
    try:
        assert os.getsid(child) == child
        assert os.getpgid(child) != os.getpgrp()
        for i in range(8):
            writer.write(np.full((120,160,3), i*25, np.uint8))
    finally:
        writer.release()
    assert writer._process.returncode == 0
    capture = cv2.VideoCapture(str(tmp_path / "session.avi"))
    frames = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frames.append(float(image.mean()))
    capture.release()
    assert frames == pytest.approx([i*25 for i in range(8)], abs=3)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg unavailable")
def test_terminal_group_interrupt_does_not_kill_muxer(tmp_path):
    # Launch an isolated fake producer session before sending a group signal;
    # never send SIGINT to pytest, the user's terminal, or real vehicle tasks.
    script = """
import os, signal, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import cv2
import numpy as np
from car_control_modular.fast_mjpeg_writer import FastMjpegWriter
signal.signal(signal.SIGINT, lambda *args: None)
w = FastMjpegWriter(Path(sys.argv[2]),30,(160,120),cv2)
try:
    w.write(np.zeros((120,160,3),np.uint8))
    os.killpg(os.getpgrp(), signal.SIGINT)
    # Synchronous drain will report failure if the muxer received SIGINT.
    for i in range(12): w.write(np.full((120,160,3),100,np.uint8))
finally:
    w.release()
assert w._process.returncode == 0
"""
    result = subprocess.run([sys.executable, "-c", script, str(ROOT), str(tmp_path/"ctrlc.avi")],
                            start_new_session=True, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
