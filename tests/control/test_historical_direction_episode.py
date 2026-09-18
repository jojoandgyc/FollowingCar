"""CAP285 -> 288/290 -> 291 regression. Real controller, no hardware."""
import logging
from types import SimpleNamespace

import pytest

from car_control_modular.control_types import PersonTarget, SensorFrame
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController


@pytest.fixture
def controller(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("car_control_modular.controllers.time.monotonic", lambda: clock[0])
    c = FollowSafetyController(FollowPolicyConfig(
        direction_history_enable=True, lost_confirm_frames=3, search_cooldown=0,
    ))
    c.active_target_id = 1
    c._has_seen_person = True
    c._test_clock = clock
    return c


def _hint(c, *, loss=206, first=204, last=205, stamp=99.9, uid=1):
    return c.note_historical_direction_hint(
        "left", active_target_id=uid, first_capture_frame_id=first,
        last_capture_frame_id=last, selected_capture_frame_ids=(first, last),
        confidence=.64, loss_capture_frame_id=loss, evidence_timestamp=stamp,
    )


def _visible(c, cap=285, *, low_quality=False):
    target = PersonTarget(track_id=1, bbox=(514., 20., 639., 460.), confidence=.9, area=55000.)
    frame = SensorFrame(width=640, height=480, capture_frame_id=cap,
                        capture_timestamp=c._test_clock[0], persons=[target])
    return c.decide(cap, frame, low_quality_visible=low_quality,
                    target_steering_limit_rpm=5 if low_quality else None)


def _missing(c, cap):
    c._test_clock[0] += .05
    return c.decide(cap, SensorFrame(width=640, height=480, capture_frame_id=cap,
                                   capture_timestamp=c._test_clock[0]))


@pytest.mark.parametrize("low_quality", [False, True])
def test_cap285_reacquired_target_invalidates_prior_left_episode(controller, low_quality):
    c = controller
    c._direction_loss_capture_id = 206
    assert _hint(c)
    _visible(c, low_quality=low_quality)
    assert c._historical_direction_hint is None
    assert c._direction_loss_capture_id is None
    for cap in (288, 290):
        d = _missing(c, cap)
        assert d.reason == "lost_history_hold_right"
        assert c._lost_exit_direction is None
    d = _missing(c, 291)
    assert d.reason == "search_right"
    assert [a.kind for a in d.actions] == ["rotate_right"]
    assert c._direction_loss_capture_id == 288


def test_pending_never_consumes_even_a_fresh_same_episode_hint(controller, caplog):
    c = controller
    _visible(c)
    _missing(c, 288)
    assert _hint(c, loss=288, first=286, last=287, stamp=100.)
    with caplog.at_level(logging.INFO):
        c._capture_lost_exit_direction(SensorFrame(capture_frame_id=288, width=640))
    assert c._lost_exit_direction is None
    assert "history_checked=False" in caplog.text
    assert "selected_direction=none" in caplog.text
    _missing(c, 290)
    assert _missing(c, 291).reason == "search_right"


def test_search_entry_rechecks_previously_cached_fallback(controller):
    c = controller
    _visible(c)
    _missing(c, 288)
    _missing(c, 290)
    c._lost_exit_direction = "left"
    c._lost_hint_source = "historical_direction_evidence"
    assert _missing(c, 291).reason == "search_right"


def test_late_worker_result_from_old_loss_is_rejected_even_with_same_uid(controller):
    c = controller
    c._direction_loss_capture_id = 206
    _visible(c)
    _missing(c, 288)
    assert not _hint(c)
    assert c._historical_direction_hint is None


@pytest.mark.parametrize("stamp", [99.0, 100.1, float("nan"), 0.0])
def test_expired_future_or_invalid_evidence_is_rejected(controller, stamp):
    controller._direction_loss_capture_id = 206
    assert not _hint(controller, stamp=stamp)


def test_evidence_expiry_checked_again_at_consumption(controller):
    c = controller
    c._direction_loss_capture_id = 206
    assert _hint(c)
    c._test_clock[0] = 100.7
    c.lost_confirm_frames = 3
    c._capture_lost_exit_direction(SensorFrame(capture_frame_id=210, width=640))
    assert c._historical_direction_hint is None
    assert c._lost_exit_direction is None


def test_fresh_fallback_still_starts_search_without_target_history(controller):
    c = controller
    _missing(c, 206)
    assert _hint(c)
    _missing(c, 207)
    assert _missing(c, 208).reason == "search_left"


def test_detector_only_candidate_cannot_invalidate_hint(controller):
    c = controller
    c._direction_loss_capture_id = 206
    assert _hint(c)
    c.note_direction_classifier_evidence(207, 100., state="visible",
        bbox=(500., 20., 639., 460.), frame_width=640, confidence=.99)
    assert c._historical_direction_hint is not None
    assert c._direction_loss_capture_id == 206


def test_confirmed_candidate_direction_and_active_search_not_overridden(controller):
    c = controller
    _visible(c)
    c.lost_confirm_frames = 3
    c._lost_exit_direction = "left"
    c._lost_hint_source = "search_candidate_last_left"
    c._ensure_search_state(SensorFrame(width=640, capture_frame_id=288))
    assert c.search_direction == "left"
    c._lost_exit_direction = "right"
    c._ensure_search_state(SensorFrame(width=640, capture_frame_id=291))
    assert c.search_direction == "left"  # no live-search reversal/coverage reset


def test_uid_change_rejects_hint(controller):
    controller._direction_loss_capture_id = 206
    assert _hint(controller)
    controller.active_target_id = 2
    assert controller._historical_hint_for_current_target() is None


def test_runtime_drops_pending_worker_after_target_returns(controller):
    import request_0513_modular as runtime
    # Invoke the actual service method on an inert owner; no runtime init.
    owner = SimpleNamespace(_follow_controller=controller,
        _historical_backfill_pending={"loss_capture_frame_id":206, "episode":1})
    runtime.PersonTracker._service_historical_direction_backfill(owner)
    assert owner._historical_backfill_pending is None


def test_runtime_passes_loss_and_original_capture_time_to_controller(controller):
    import request_0513_modular as runtime
    from car_control_modular.historical_direction_backfill import HistoricalDirectionBackfill
    c = controller
    c._direction_loss_capture_id = 206
    owner = SimpleNamespace(
        _follow_controller=c,
        _historical_backfill_pending={"loss_capture_frame_id":206, "episode":1,
                                     "active_target_id":1, "started_at":99.95, "deadline":100.1},
        _historical_backfill=HistoricalDirectionBackfill(),
        _direction_evidence_ring=[SimpleNamespace(
            capture_frame_id=cap, timestamp=stamp, state="visible",
            bbox=(60., 100., 140., 300.), score=.9, frame_width=640, candidate_count=1,
        ) for cap, stamp in [(204, 99.85), (205, 99.9)]],
    )
    runtime.PersonTracker._service_historical_direction_backfill(owner)
    assert owner._historical_backfill_pending is None
    assert c._historical_direction_hint["loss_capture_frame_id"] == 206
    assert c._historical_direction_hint["evidence_timestamp"] == 99.85
    assert c._lost_exit_direction is c.search_direction is None  # evidence only
    c._test_clock[0] = 100.6
    assert c._historical_hint_for_current_target() is None


def test_runtime_schedules_once_per_first_loss_capture(controller, monkeypatch):
    import request_0513_modular as runtime
    monkeypatch.setattr(runtime, "FOLLOW_DIRECTION_HISTORY_ENABLE", True)
    monkeypatch.setattr(runtime, "HISTORICAL_DIRECTION_BACKFILL_ENABLE", True)
    c = controller
    c._direction_loss_capture_id = 288
    owner = SimpleNamespace(
        _follow_controller=c, _vision_control_state="lost_confirming",
        _historical_backfill_pending=None, _historical_backfill_last_start_capture=0,
        _historical_backfill_episode=0, _direction_evidence_ring=[], _capture_metadata_ring=[],
        _service_historical_direction_backfill=lambda: None,
    )
    runtime.PersonTracker._maybe_schedule_historical_direction_backfill(owner, 290, 1)
    assert owner._historical_backfill_pending["loss_capture_frame_id"] == 288
    assert owner._historical_backfill_episode == 1
    owner._historical_backfill_pending = None
    runtime.PersonTracker._maybe_schedule_historical_direction_backfill(owner, 291, 1)
    assert owner._historical_backfill_pending is None  # not a second job for same loss
    assert owner._historical_backfill_episode == 1


def test_time_based_loss_confirmation_does_not_wait_extra_frames(controller):
    c = controller
    # FollowPolicyConfig is mutable; only this test uses time-based confirmation.
    from dataclasses import replace
    c.cfg = replace(c.cfg, lost_confirm_sec=.01)
    _visible(c)
    assert _missing(c, 288).waiting_lost_confirm
    assert _missing(c, 290).reason == "search_right"
