"""Real follow writer/producer path with fake serial and fake clock only."""
from car_control_modular.control_types import SteeringFeedback
import request_0513_modular as main
from test_cap296_moving_handoff_execution import setup


def test_periodic_zero_receipts_feed_fresh_confirmed_producer_retirement(monkeypatch):
    runtime, owner, driver, _, clock, state = setup(monkeypatch)
    owner._action_runtime = runtime
    owner._search_handoff_direction = "right"
    owner._search_handoff_started_capture_ts = 9.8
    runtime.get_steering_feedback = lambda: SteeringFeedback(
        timestamp=clock[0]-.01, trustworthy=True, raw_yaw_rate_right_dps=0.,
        left_read_started=clock[0]-.025, left_read_finished=clock[0]-.018,
        right_read_started=clock[0]-.017, right_read_finished=clock[0]-.01)
    # Expired moving geometry forces real speed-zero packets. Fresh Depth
    # remains a separate grant throughout; the first zero cannot prove rest.
    for now in (10.12, 10.18, 10.24):
        clock[0] = now
        runtime._service_follow_wheels()
    assert driver.pairs == [(0, 0)]*3
    settled = runtime._search_handoff_zero_evidence
    assert settled.receipt is runtime.backend.last_speed_receipt
    assert settled.zero_completed_at == 10.12
    assert len(settled.quiet_samples) == 2

    clock[0] = 10.25
    owner._active_capture_frame_id = 2038
    owner._active_capture_timestamp = 10.15
    old_depth = owner._fresh_depth_linear_snapshot(1)
    assert not main.PersonTracker._hold_search_reacquire_brake(owner,
        bbox=(568., 127., 640., 475.), width=640, eligible=True, confirmed=True, raw_track_id=1)
    assert owner._search_handoff_uid is None
    assert owner._fresh_depth_linear_snapshot(1) == old_depth
    assert driver.pairs == [(0, 0)]*3 and not driver.stops

    # Only the normal writer's still-current axes can send motion afterward.
    clock[0] = 10.29
    runtime._service_follow_wheels()
    assert driver.pairs[-1] == (49, -35)
    assert runtime._search_handoff_zero_evidence is None
    state[2] = 10.295
    clock[0] = 10.296
    runtime._service_follow_wheels()
    assert driver.pairs[-1] != (49, -35)  # retired constraint did not extend Depth
