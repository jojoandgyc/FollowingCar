from car_control_modular.historical_direction_backfill import (
    HistoricalDirectionBackfill,
    HistoricalDirectionCandidate,
)
from car_control_modular.control_types import SensorFrame
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController


def _candidate(capture_id, timestamp, center, *, score=0.8, count=1, width=640):
    box_width = 80.0
    x1 = center * width - box_width / 2.0
    return HistoricalDirectionCandidate(
        capture_frame_id=capture_id,
        timestamp=timestamp,
        state="visible",
        bbox=(x1, 100.0, x1 + box_width, 300.0),
        score=score,
        frame_width=width,
        candidate_count=count,
    )


def test_backfill_accepts_unique_contiguous_detector_chain():
    now = 100.0
    backfill = HistoricalDirectionBackfill(max_age_sec=0.7, min_samples=2)
    result = backfill.evaluate(
        [
            _candidate(1980, 99.70, 0.82),
            _candidate(1984, 99.82, 0.74),
            _candidate(1987, 99.90, 0.68),
        ],
        loss_capture_frame_id=1991,
        now=now,
    )

    assert result.direction == "right"
    assert result.selected_capture_frame_ids == (1980, 1984, 1987)
    assert result.confidence <= 0.70


def test_backfill_rejects_competing_detector_frame():
    now = 100.0
    backfill = HistoricalDirectionBackfill(max_age_sec=0.7, min_samples=2)
    result = backfill.evaluate(
        [
            _candidate(1984, 99.82, 0.74, count=2),
            _candidate(1987, 99.90, 0.68),
        ],
        loss_capture_frame_id=1991,
        now=now,
    )

    assert result.direction is None
    assert result.reason == "no_unique_chain"


def test_backfill_rejects_large_capture_gap_instead_of_guessing():
    now = 100.0
    backfill = HistoricalDirectionBackfill(max_age_sec=0.7, min_samples=2, max_capture_gap=2)
    result = backfill.evaluate(
        [
            _candidate(1980, 99.70, 0.82),
            _candidate(1987, 99.90, 0.68),
        ],
        loss_capture_frame_id=1991,
        now=now,
    )

    assert result.direction is None
    assert result.reason == "geometry_chain_rejected"


def test_backfill_rejects_chain_that_jumps_from_last_target_anchor():
    now = 100.0
    backfill = HistoricalDirectionBackfill(max_age_sec=0.7, min_samples=2)
    result = backfill.evaluate(
        [
            _candidate(1984, 99.82, 0.20),
            _candidate(1987, 99.90, 0.24),
        ],
        loss_capture_frame_id=1991,
        now=now,
        anchor_capture_frame_id=1980,
        anchor_center_ratio=0.82,
    )

    assert result.direction is None
    assert result.reason == "geometry_anchor_rejected"


def test_controller_consumes_hint_only_when_main_history_has_no_side(monkeypatch):
    monkeypatch.setattr("car_control_modular.controllers.time.monotonic", lambda: 100.0)
    controller = FollowSafetyController(
        FollowPolicyConfig(direction_history_enable=True, lost_confirm_frames=2)
    )
    controller.active_target_id = 1
    controller._has_seen_person = True
    controller.lost_confirm_frames = 2
    controller._direction_loss_capture_id = 1991
    assert controller.note_historical_direction_hint(
        "right",
        active_target_id=1,
        first_capture_frame_id=1980,
        last_capture_frame_id=1987,
        selected_capture_frame_ids=(1980, 1984, 1987),
        confidence=0.70,
        loss_capture_frame_id=1991, evidence_timestamp=99.70,
    )

    controller._capture_lost_exit_direction(
        SensorFrame(width=640, height=480, capture_frame_id=1991, capture_timestamp=100.0)
    )

    assert controller._lost_exit_direction == "right"
    assert controller._lost_hint_source == "historical_direction_evidence"


def test_controller_keeps_main_history_priority_over_historical_hint(monkeypatch):
    monkeypatch.setattr("car_control_modular.controllers.time.monotonic", lambda: 100.0)
    controller = FollowSafetyController(
        FollowPolicyConfig(direction_history_enable=True, lost_confirm_frames=2)
    )
    controller.active_target_id = 1
    controller._has_seen_person = True
    controller.lost_confirm_frames = 2
    controller._direction_loss_capture_id = 1991
    controller._target_direction_history.record_visible(
        1988,
        99.90,
        target_id=1,
        bbox=(420.0, 100.0, 560.0, 300.0),
        frame_width=640,
        confidence=0.9,
    )
    assert controller.note_historical_direction_hint(
        "left",
        active_target_id=1,
        first_capture_frame_id=1980,
        last_capture_frame_id=1987,
        selected_capture_frame_ids=(1980, 1984, 1987),
        confidence=0.70,
        loss_capture_frame_id=1991, evidence_timestamp=99.70,
    )

    controller._capture_lost_exit_direction(
        SensorFrame(width=640, height=480, capture_frame_id=1991, capture_timestamp=100.0)
    )

    assert controller._lost_exit_direction == "right"
    assert controller._lost_hint_source == "latest_reliable_capture_side"
