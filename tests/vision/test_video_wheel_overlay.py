"""Capture-time wheel feedback, without any live device reads."""
import csv
import threading
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from car_control_modular.action_runtime import MotionActionRuntime
from car_control_modular.control_types import SteeringFeedback
from car_control_modular.video_recorder import (
    AsyncVideoRecorder, VideoRecorderConfig, VideoFrameOverlay, VideoWheelOverlay,
)


def feedback(**kwargs):
    return replace(SteeringFeedback(timestamp=100., trustworthy=True,
        left_forward_rpm=34., right_forward_rpm=26.,
        left_speed_rpm=34, right_speed_rpm=-26), **kwargs)


def test_recording_snapshot_never_waits_on_feedback_lock():
    runtime = MotionActionRuntime.__new__(MotionActionRuntime)
    runtime._steering_feedback = feedback()
    runtime._steering_feedback_lock = threading.Lock()
    with runtime._steering_feedback_lock:
        # This call intentionally runs with the normal getter's lock held.
        assert runtime.get_recording_feedback() is runtime._steering_feedback


def test_forward_signs_and_measured_not_requested():
    w=VideoWheelOverlay.from_feedback(feedback(),100.05)
    assert w.left_rpm==34 and w.right_rpm==26
    assert w.status=='fresh' and w.age_ms==pytest.approx(50)
    assert 'L +34.0  R +26.0 RPM' in w.label()
    assert 'FRESH AGE 50ms' in w.label()


@pytest.mark.parametrize('fb,ts,status', [
    (None,100.1,'missing'),
    (feedback(),100.151,'stale'),
    (feedback(trustworthy=False),100.05,'untrusted'),
    (feedback(),99.9,'future'),
    (feedback(left_forward_rpm=float('nan')),100.1,'invalid'),
    (feedback(timestamp=float('inf')),100.1,'invalid'),
    (feedback(timestamp=0),100.1,'invalid'),
    (object(),100.1,'invalid'),
])
def test_unavailable_or_old_is_not_fake_zero(fb,ts,status):
    w=VideoWheelOverlay.from_feedback(fb,ts)
    assert w.status==status
    assert status.upper() in w.label()
    if status in ('missing','future','invalid'):
        assert 'L n/a  R n/a' in w.label()
    else:
        assert w.left_rpm==34


def test_every_capture_including_unprocessed_frames_has_snapshot(tmp_path):
    recorder=AsyncVideoRecorder(VideoRecorderConfig(str(tmp_path/'video.avi'),30,
        overlay_wait_sec=0),cv2_module=cv2)
    image=np.zeros((480,640,3),dtype=np.uint8)
    published=feedback()
    assert recorder.submit(image,capture_frame_id=1,monotonic_sec=100.05,wheel_feedback=published)
    published=feedback(timestamp=100.1,left_forward_rpm=60,right_forward_rpm=58)
    assert recorder.submit(image,capture_frame_id=2,monotonic_sec=100.12,wheel_feedback=published)
    assert recorder.submit(image,capture_frame_id=3,monotonic_sec=100.14)
    recorder.close(timeout_sec=3)
    assert recorder.error is None and recorder.written_frames==3
    rows=list(csv.DictReader(Path(recorder.index_path).open()))
    assert all(r['control_frame_id']=='' for r in rows)
    assert [r['wheel_left_forward_rpm'] for r in rows]==['34.000','60.000','']
    assert [r['wheel_right_forward_rpm'] for r in rows]==['26.000','58.000','']
    assert [r['wheel_feedback_status'] for r in rows]==['fresh','fresh','missing']
    assert rows[0]['wheel_feedback_age_ms']=='50.000'
    assert not image.any()  # Recording cannot draw on the vision pipeline input.
    capture=cv2.VideoCapture(str(tmp_path/'video.avi'))
    count=0
    while True:
        ok,frame=capture.read()
        if not ok:break
        assert frame.any()
        count+=1
    capture.release()
    assert count==3


def test_wheel_text_rendered_without_control_metadata():
    recorder=AsyncVideoRecorder(VideoRecorderConfig('unused.avi',30),cv2_module=cv2)
    lines=[]
    recorder._draw_text_box=lambda image,text,**kw: lines.append(text)
    recorder._annotate_frame(np.zeros((480,640,3),dtype=np.uint8),1,1,
        VideoFrameOverlay(),wheels=VideoWheelOverlay.from_feedback(feedback(),100.05))
    assert any(line.startswith('WHEEL L +34.0  R +26.0 RPM') for line in lines)


def test_reverse_sign_preserved():
    w=VideoWheelOverlay.from_feedback(feedback(left_forward_rpm=-6,right_forward_rpm=6),100.02)
    assert 'L -6.0  R +6.0 RPM' in w.label()
