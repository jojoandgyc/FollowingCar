"""Bounded recorder work and nonblocking producers; no live devices."""
import logging
import os
import queue
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from car_control_modular.video_recorder import AsyncVideoRecorder, VideoRecorderConfig, VideoFrameOverlay


def recorder():
    r = AsyncVideoRecorder(VideoRecorderConfig("unused.avi", 30), cv2_module=cv2)
    r._ensure_thread = lambda: None
    return r


@pytest.mark.parametrize("method", ["submit", "update_overlay"])
def test_metadata_contention_never_blocks_producer(method):
    r = recorder()
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    values = []
    class NoCopy:
        def copy(self): raise AssertionError("contention must be rejected before copy")
    def holder():
        with r._overlay_condition:
            entered.set()
            release.wait(3)
    def producer():
        try:
            values.append(r.submit(NoCopy(), capture_frame_id=1) if method == "submit"
                          else r.update_overlay(1, []))
        finally:
            done.set()
    locked = threading.Thread(target=holder)
    worker = threading.Thread(target=producer)
    locked.start()
    try:
        assert entered.wait(1)
        worker.start()
        assert done.wait(1), "producer waited on recorder metadata lock"
        assert values == [False]
    finally:
        release.set()
        locked.join(2)
        if worker.ident is not None: worker.join(2)


def test_drop_does_not_log_from_capture_thread():
    r = recorder()
    r._logger = SimpleNamespace(warning=lambda *a: pytest.fail("synchronous producer logging"))
    for _ in range(100): assert r._drop_frame(1) is False
    assert r.dropped_frames == 100


@pytest.mark.parametrize("failure", ["copy", "queue"])
def test_failed_enqueue_releases_reservation_and_lock(failure):
    r = recorder()
    def full(item): raise queue.Full
    if failure == "queue":
        r._queue = SimpleNamespace(full=lambda: False, put_nowait=full)
    class BadImage:
        def copy(self): raise ValueError("copy failure")
    if failure == "copy":
        with pytest.raises(ValueError): r.submit(BadImage(), capture_frame_id=1)
    else:
        assert not r.submit(np.zeros((10, 10, 3), np.uint8), capture_frame_id=1)
    assert not r._submitted_capture_ids and not r._overlays
    assert r._overlay_condition.acquire(blocking=False)
    r._overlay_condition.release()


def test_long_label_uses_logarithmic_size_queries_and_bounded_cache():
    r = recorder()
    class CountCV:
        calls = 0
        def getTextSize(self, *args):
            self.calls += 1
            return cv2.getTextSize(*args)
    r._cv2 = CountCV()
    text, size, _ = r._text_layout("long diagnostic " * 100, cv2.FONT_HERSHEY_SIMPLEX, .4, 1, 200)
    assert size[0] <= 200 and text.endswith("~")
    assert r._cv2.calls <= 14
    count = r._cv2.calls
    r._text_layout("long diagnostic " * 100, cv2.FONT_HERSHEY_SIMPLEX, .4, 1, 200)
    assert r._cv2.calls == count
    for index in range(600):
        r._text_layout(str(index), cv2.FONT_HERSHEY_SIMPLEX, .4, 1, 200)
    assert len(r._text_layout_cache) == 256


@pytest.mark.parametrize("width", [1, 5, 50, 300])
def test_layout_handles_tiny_regions(width):
    text, size, _ = recorder()._text_layout("WHEEL L +34.0 R +26.0", cv2.FONT_HERSHEY_SIMPLEX, .4, 1, width)
    assert not text or size[0] <= width


def test_layout_cache_never_reuses_old_frame_pixels():
    r = recorder()
    kwargs = dict(x=10, y=30, font=cv2.FONT_HERSHEY_SIMPLEX, scale=.4,
                  foreground=(0,255,0), background=(0,0,0))
    first = np.full((60,300,3), 30, np.uint8)
    second = np.full((60,300,3), 190, np.uint8)
    expected = second.copy()
    r._draw_text_box(first, "CAP 001234", **kwargs)
    r._draw_text_box(second, "CAP 001234", **kwargs)
    recorder()._draw_text_box(expected, "CAP 001234", **kwargs)
    np.testing.assert_array_equal(second, expected)


def test_worker_inplace_annotation_uses_owned_copy_only():
    r = recorder()
    source = np.zeros((120,160,3), np.uint8)
    assert r.submit(source, capture_frame_id=1)
    item = r._queue.get_nowait()
    output = r._annotate_frame(item.image, 1, 1, VideoFrameOverlay(), copy_image=False)
    assert output is item.image and output.any()
    assert not source.any()
    independent = r._annotate_frame(source, 1, 1, VideoFrameOverlay())
    assert independent is not source and not source.any()


@pytest.mark.skipif(not hasattr(os, "getpriority"), reason="Linux priority API unavailable")
def test_priority_change_is_recorder_thread_only():
    main_tid = threading.get_native_id()
    before = os.getpriority(os.PRIO_PROCESS, main_tid)
    class PriorityProbe(AsyncVideoRecorder):
        def _run(self):
            self._lower_worker_priority()
            self.actual = os.getpriority(os.PRIO_PROCESS, threading.get_native_id())
            self._closed.set()
    r = PriorityProbe(VideoRecorderConfig("unused.avi",30,worker_nice=8),cv2_module=cv2)
    r._ensure_thread()
    assert r.close(2)
    assert r.actual >= before
    assert os.getpriority(os.PRIO_PROCESS, main_tid) == before


def test_timing_log_includes_producer_latency_and_metadata_drops(caplog):
    r = recorder()
    r._recording_timings.append((0.,)*7)
    r.submit(np.zeros((10,10,3),np.uint8),capture_frame_id=1)
    with caplog.at_level(logging.INFO):r._log_recording_timings("test")
    assert "submit_ms(avg=" in caplog.text
    assert "metadata_dropped=0" in caplog.text


@pytest.mark.parametrize("alpha", [0., .62, 1.])
@pytest.mark.parametrize("position", [(10,30), (0,8), (240,59)])
def test_cached_glyphs_preserve_original_translucent_pixels(alpha, position):
    r = AsyncVideoRecorder(VideoRecorderConfig("unused.avi",30,overlay_text_alpha=alpha),cv2_module=cv2)
    rng = np.random.default_rng(23)
    expected = rng.integers(0,255,(60,300,3),dtype=np.uint8)
    actual = expected.copy()
    x,y = position
    rendered,(tw,th),baseline = r._text_layout("L +34",cv2.FONT_HERSHEY_SIMPLEX,.4,1,300-x-4)
    left,right = max(0,x-3),min(300,x+tw+5)
    top,bottom = max(0,y-th-5),min(60,y+baseline+5)
    region = expected[top:bottom,left:right]
    layer = region.copy()
    cv2.putText(layer,rendered,(x-left,y-top),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),3,cv2.LINE_8)
    cv2.putText(layer,rendered,(x-left,y-top),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,0),1,cv2.LINE_8)
    cv2.addWeighted(layer,alpha,region,1-alpha,0,dst=region)
    r._draw_text_box(actual,"L +34",x=x,y=y,font=cv2.FONT_HERSHEY_SIMPLEX,scale=.4,
                     foreground=(0,255,0),background=(0,0,0))
    np.testing.assert_array_equal(actual,expected)


def test_glyph_cache_memory_is_bounded_and_contains_no_camera_background():
    r = recorder()
    source = np.full((80,640,3),93,np.uint8)
    for i in range(200):
        r._draw_text_box(source,f"WHEEL {i} "*8,x=5,y=40,font=cv2.FONT_HERSHEY_SIMPLEX,scale=.6,
                         foreground=(0,255,0),background=(0,0,0))
    assert len(r._text_raster_cache) <= 96
    assert r._text_raster_bytes <= 1024*1024
    assert r._text_raster_bytes == sum(ink.nbytes+mask.nbytes for ink,mask in r._text_raster_cache.values())
    for ink,mask in r._text_raster_cache.values():
        assert set(np.unique(ink)) <= {0,255}
        assert set(np.unique(mask)) <= {0,255}
