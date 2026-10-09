"""Real control entry point: diagnostic delivery cannot own its mutex."""
import logging
import threading

import pytest

import request_0513_modular as runtime
from test_turn_depth_scheduling import owner, attempt
from test_depth_optimistic_transaction import scene


def test_slow_output_does_not_block_next_control_or_stop(owner, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    errors, observations = [], []
    logger = runtime.logger
    monkeypatch.setattr(logger, "level", logging.INFO)
    owner._control_update_lock = threading.Lock()

    def decide(*args, **kwargs):
        assert owner._control_update_lock.locked()
        logger.info("control_delivery_probe")
        owner.calls.append("decision_finished")

    owner._queue_actions_for_persons_locked = decide

    class SlowHandler(logging.Handler):
        def emit(self, record):
            if record.msg == "control_delivery_probe":
                observations.append((owner._control_update_lock.locked(), tuple(owner.calls)))
                entered.set()
                assert release.wait(2.)

    handler = SlowHandler()
    logger.addHandler(handler)

    def run():
        try:
            attempt(owner)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run, daemon=True)
    try:
        worker.start()
        assert entered.wait(1.)
        assert observations == [(False, ("decision_finished",))]
        assert owner._control_update_lock.acquire(blocking=False)
        try:
            owner._explicit_stop_requested = True
        finally:
            owner._control_update_lock.release()
    finally:
        release.set()
        worker.join(2.)
        logger.removeHandler(handler)
    assert not worker.is_alive() and not errors
    assert owner._explicit_stop_requested


def test_logging_failure_does_not_replace_decision_error_or_leak_control_lock(owner, monkeypatch):
    logger = runtime.logger
    monkeypatch.setattr(logger, "level", logging.INFO)
    owner._control_update_lock = threading.Lock()

    class BrokenHandler(logging.Handler):
        def emit(self, record):
            raise OSError("offline log sink failed")

    def decide(*args, **kwargs):
        logger.info("control_delivery_probe")
        raise ValueError("decision failed")

    owner._queue_actions_for_persons_locked = decide
    handler = BrokenHandler()
    logger.addHandler(handler)
    try:
        with pytest.raises(ValueError, match="decision failed"):
            attempt(owner)
        assert owner._control_update_lock.acquire(blocking=False)
        owner._control_update_lock.release()
    finally:
        logger.removeHandler(handler)


def test_real_depth_commits_before_slow_diagnostic_without_renewing_sample(scene, monkeypatch):
    obj, camera, distance, clock = scene
    stamp = camera._latest_depth_ts
    logger = runtime.logger
    monkeypatch.setattr(logger, "level", logging.INFO)
    states = []

    def consume(*args, **kwargs):
        target = obj._longitudinal_context["person_targets"][0]
        state = distance.get_frame_distance_state(
            640, target, frame_height=480, depth_use_latest=True,
            prepared_depth=kwargs["prepared_depth"])
        states.append(state)
        logger.info("depth_delivery_probe")

    obj._queue_actions_for_persons_locked = consume

    class DelayedHandler(logging.Handler):
        def emit(self, record):
            if record.msg == "depth_delivery_probe":
                assert not obj._control_update_lock._is_owned()
                assert camera._last_accepted_ts == stamp
                assert states[0].sample_timestamp == stamp
                clock[0] += .4

    handler = DelayedHandler()
    logger.addHandler(handler)
    try:
        assert attempt(obj)
    finally:
        logger.removeHandler(handler)
    assert states[0].sample_timestamp == camera._last_accepted_ts == stamp
    assert clock[0]-stamp > .3  # Delivery did not replace capture with wall time.


def test_compute_profile_reports_real_cpu_stages_after_unlock(scene, monkeypatch):
    obj, camera, distance, clock = scene
    logger = runtime.logger
    monkeypatch.setattr(logger, "level", logging.INFO)
    records = []

    class ProfileHandler(logging.Handler):
        def emit(self, record):
            if record.msg.startswith("depth_compute_profile"):
                assert not obj._control_update_lock._is_owned()
                records.append(record.getMessage())

    handler = ProfileHandler()
    logger.addHandler(handler)
    try:
        assert attempt(obj)
    finally:
        logger.removeHandler(handler)
    assert len(records) == 1
    profile = records[0]
    assert "pixel_scan_started=True" in profile
    assert "stage=complete failure_stage=None committed=True" in profile
    assert "compute_cpu_ms=" in profile and "stages_ms=" in profile
    assert "'wall_ms':" in profile and "'cpu_ms':" in profile
