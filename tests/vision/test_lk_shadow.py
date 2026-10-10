from dataclasses import FrozenInstanceError
import threading
import time

import cv2
import numpy as np
import pytest

from rk_vision.lk_shadow import (
    LKShadowConfig, LKShadowResult, LKShadowSeed, LKShadowTracker, LKShadowWorker,
)


def textured_image():
    rng = np.random.default_rng(93)
    image = np.zeros((240, 320), np.uint8)
    image[50:190, 60:180] = rng.integers(0, 256, (140, 120), dtype=np.uint8)
    return image


def seed(cap=1, ts=10., uid=1, raw=4, bbox=(60, 50, 180, 190)):
    return LKShadowSeed(uid, raw, cap, ts, bbox)


def test_translation_is_measured_from_images_and_preserves_capture_provenance():
    image = textured_image()
    tracker = LKShadowTracker()
    first = tracker.process(image, 1, 10., seed())
    moved = cv2.warpAffine(image, np.float32([[1, 0, 6], [0, 1, -3]]), (320, 240))
    result = tracker.process(moved, 2, 10.05)
    assert first.status == 'seeded'
    assert 8 <= first.input_points <= 64
    assert result.status == 'tracked'
    assert result.bbox == pytest.approx((66, 47, 186, 187), abs=.8)
    assert result.source == 'lk_shadow'
    assert (result.seed_uid, result.seed_raw_track_id, result.seed_capture_id) == (1, 4, 1)
    assert (result.prev_capture_id, result.capture_id) == (1, 2)
    assert (result.seed_capture_timestamp, result.capture_timestamp) == (10., 10.05)
    assert result.inlier_points >= 8 and result.wall_ms > 0 and result.thread_cpu_ms > 0
    with pytest.raises(FrozenInstanceError):
        result.capture_timestamp = 11.


def test_disappearance_invalidates_instead_of_extrapolating():
    tracker = LKShadowTracker()
    tracker.process(textured_image(), 1, 10., seed())
    result = tracker.process(np.zeros((240, 320), np.uint8), 2, 10.05)
    assert result.status == 'lost'
    assert result.bbox is None
    assert tracker.process(textured_image(), 3, 10.1).status == 'no_seed'


def test_identity_raw_change_starts_a_new_aligned_seed():
    tracker = LKShadowTracker()
    image = textured_image()
    tracker.process(image, 1, 10., seed())
    result = tracker.process(image, 2, 10.05, seed(2, 10.05, 7, 19))
    assert result.status == 'seeded'
    assert (result.seed_uid, result.seed_raw_track_id, result.seed_capture_id) == (7, 19, 2)


def test_old_correction_cannot_seed_a_new_image():
    tracker = LKShadowTracker()
    result = tracker.process(textured_image(), 2, 10.05, seed())
    assert result.status == 'seed_mismatch'
    assert result.bbox is None
    assert tracker.process(textured_image(), 3, 10.1).status == 'no_seed'


def test_duplicate_and_backward_capture_do_not_refresh_or_poison_current_state():
    tracker = LKShadowTracker()
    image = textured_image()
    tracker.process(image, 5, 10., seed(5, 10.))
    assert tracker.process(image, 5, 10.01).status == 'out_of_order'
    assert tracker.process(image, 4, 10.02).status == 'out_of_order'
    assert tracker.process(image, 6, 9.99).status == 'out_of_order'
    result = tracker.process(image, 6, 10.05)
    assert result.status == 'tracked' and result.prev_capture_id == 5


def test_stale_gap_seed_expiry_and_new_image_shape_fail_closed():
    image = textured_image()
    tracker = LKShadowTracker(LKShadowConfig(max_seed_age_sec=.1))
    tracker.process(image, 1, 10., seed())
    result = tracker.process(image, 2, 10.11)
    assert result.status == 'seed_expired' and result.bbox is None
    tracker.process(image, 3, 11., seed(3, 11.))
    assert tracker.process(image, 4, 11.3).status == 'stale_gap'
    tracker.process(image, 5, 12., seed(5, 12.))
    assert tracker.process(image[:200], 6, 12.05).reason == 'image_dimensions_changed'


def test_blank_seed_and_invalid_bbox_do_not_start_tracking():
    tracker = LKShadowTracker()
    assert tracker.process(np.zeros((240, 320), np.uint8), 1, 10., seed()).status == 'lost'
    assert tracker.process(textured_image(), 2, 10.05,
        seed(2, 10.05, bbox=(-10, 10, 200, 230))).status == 'invalid_seed'


def test_invalid_capture_types_are_diagnostic_not_exceptions():
    tracker = LKShadowTracker()
    result = tracker.process(textured_image(), None, 10.)
    assert result.status == 'invalid_frame' and result.bbox is None


def test_unreasonable_geometry_is_rejected(monkeypatch):
    tracker = LKShadowTracker()
    image = textured_image()
    tracker.process(image, 1, 10., seed())
    monkeypatch.setattr(cv2, 'estimateAffinePartial2D', lambda before, after, **kw:
        (np.array([[2., 0, 0], [0, 2., 0]]), np.ones((len(before), 1), np.uint8)))
    result = tracker.process(image, 2, 10.05)
    assert result.status == 'rejected_geometry'
    assert result.bbox is None


class BlockingTracker:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.received = []

    def process(self, frame, capture_id, capture_timestamp, **kwargs):
        if not self.received:
            self.started.set()
            assert self.release.wait(2.)
        self.received.append((capture_id, int(frame[0, 0])))
        return LKShadowResult(capture_id, capture_timestamp, 'no_seed', 'mock')


def wait_processed(worker, count):
    end = time.monotonic() + 2.
    while time.monotonic() < end and worker.stats()['processed'] < count:
        time.sleep(.002)
    assert worker.stats()['processed'] >= count


def test_worker_holds_only_latest_pending_and_owns_buffer():
    tracker = BlockingTracker()
    worker = LKShadowWorker(tracker=tracker)
    try:
        first = np.ones((32, 32), np.uint8)
        assert worker.submit(first, 1, 1.)
        assert tracker.started.wait(1.)
        first[:] = 99
        for cap in range(2, 30):
            assert worker.submit(np.full((32, 32), cap, np.uint8), cap, float(cap))
        assert worker.stats()['pending'] == 1
        assert worker.stats()['replaced'] == 27
        assert not worker.submit(first, 5, 50.)
        tracker.release.set()
        wait_processed(worker, 2)
        assert tracker.received == [(1, 1), (29, 29)]
        assert worker.latest_result().capture_id == 29
    finally:
        tracker.release.set()
        assert worker.close()
    assert worker.close()
    assert not worker.submit(first, 30, 30.)


def test_worker_exceptions_are_diagnostics_and_do_not_kill_worker():
    class FailingOnce:
        def process(self, frame, cap, ts, **kwargs):
            if cap == 1:
                raise RuntimeError('synthetic failure')
            return LKShadowResult(cap, ts, 'no_seed', 'mock')
    worker = LKShadowWorker(tracker=FailingOnce())
    try:
        worker.submit(np.zeros((32, 32), np.uint8), 1, 1.)
        wait_processed(worker, 1)
        assert worker.latest_result().status == 'error'
        assert worker.stats()['errors'] == 1
        worker.submit(np.zeros((32, 32), np.uint8), 2, 2.)
        wait_processed(worker, 2)
        assert worker.latest_result().capture_id == 2
    finally:
        assert worker.close()


def test_pending_aligned_seed_has_priority_over_unseeded_frames():
    tracker = BlockingTracker()
    worker = LKShadowWorker(tracker=tracker)
    image = np.zeros((32, 32), np.uint8)
    try:
        worker.submit(image, 1, 1.)
        assert tracker.started.wait(1.)
        assert worker.submit(image, 2, 2., seed(2, 2.))
        assert not worker.submit(image, 3, 3.)
        assert worker.stats()['seed_priority_dropped'] == 1
        assert worker.stats()['pending'] == 1
        # A new correction replaces both the older correction and its image.
        assert worker.submit(image, 4, 4., seed(4, 4.))
        tracker.release.set()
        wait_processed(worker, 2)
        assert [cap for cap, marker in tracker.received] == [1, 4]
    finally:
        tracker.release.set()
        assert worker.close()


def test_shutdown_drops_pending_and_never_publishes_after_close():
    tracker = BlockingTracker()
    worker = LKShadowWorker(tracker=tracker)
    worker.submit(np.zeros((32, 32), np.uint8), 1, 1.)
    assert tracker.started.wait(1.)
    worker.submit(np.zeros((32, 32), np.uint8), 2, 2.)
    assert not worker.close(timeout=0.)
    tracker.release.set()
    assert worker.close()
    assert tracker.received == [(1, 0)]
    assert worker.latest_result() is None


def test_worker_aggregates_unpolled_flow_results_without_retaining_history():
    image = textured_image()
    worker = LKShadowWorker()
    try:
        assert worker.submit(image, 1, 10., seed())
        wait_processed(worker, 1)
        assert worker.submit(image, 2, 10.05)
        wait_processed(worker, 2)
        assert worker.submit(np.zeros_like(image), 3, 10.1)
        wait_processed(worker, 3)
        stats = worker.stats()
        assert stats['processed'] == 3
        assert stats['flow_attempts'] == 2 and stats['flow_successes'] == 1
        assert stats['status_counts']['seeded'] == 1
        assert stats['status_counts']['tracked'] == 1
        assert stats['status_counts']['lost'] == 1
        assert stats['process_wall_ms_total'] >= stats['process_wall_ms_max'] > 0
        assert stats['process_thread_cpu_ms_total'] >= stats['process_thread_cpu_ms_max'] > 0
        stats['status_counts']['tracked'] = 999
        assert worker.stats()['status_counts']['tracked'] == 1
        assert worker.latest_result().capture_id == 3
        assert not hasattr(worker, '_history')
    finally:
        assert worker.close()
