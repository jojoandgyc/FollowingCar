"""Production tracker observation + controller provenance regression."""
from types import SimpleNamespace
import pytest
from test_brake_hold_direction_observation import owner, sample, consume
from car_control_modular.control_types import SensorFrame
from car_control_modular.search_brake_observation import SearchBrakeObservation


def test_pre_stop_left_does_not_restart_after_uid0_center(owner):
    consume(owner,[sample(owner,656,.421)])
    sent = owner._active_capture_timestamp+.03
    consume(owner,[sample(owner,659,.548,uid=0)])
    owner._action_runtime._search_reacquire_brake_sent_at = sent
    owner._finish_search_brake_observation_hold()
    c=owner._follow_controller
    assert c._target_direction_history.latest_visible_evidence() is None
    assert c.last_person_center_x is None
    for cap in (660,662,664):
        frame=SensorFrame(width=640,height=480,capture_frame_id=cap,capture_timestamp=owner.clock)
        decision=c.decide(cap,frame)
        assert all(a.kind not in ("rotate_left","rotate_right","forward") for a in decision.actions)


def test_post_stop_trusted_right_is_preserved(owner):
    consume(owner,[sample(owner,656,.421)])
    sent=owner._active_capture_timestamp+.03
    consume(owner,[sample(owner,660,.72)])
    owner._action_runtime._search_reacquire_brake_sent_at=sent
    owner._finish_search_brake_observation_hold()
    side=owner._follow_controller._target_direction_history.latest_reliable_side()
    assert side.direction == "right" and side.last_visible_capture_frame_id == 660
    assert owner._longitudinal_valid_until == 99.8


def test_delayed_old_geometry_cannot_reinsert_retired_direction(owner):
    r=sample(owner,656,.1); consume(owner,[r])
    c=owner._follow_controller
    c.retire_pre_search_brake_direction(owner._active_capture_timestamp+.01)
    # Even a later completion/capture number cannot reuse a pre-brake image.
    r=sample(owner,660,.1)
    owner._active_capture_timestamp-=1
    obs=owner._rknn_pipeline.tracker.last_identity_observations[0]
    obs["sample_metadata"]["capture_timestamp"]=owner._active_capture_timestamp
    c.note_direction_classifier_evidence(660,owner._active_capture_timestamp,state="missing")
    assert c._target_direction_history.latest_visible_evidence() is None
    assert c._historical_hint_rejection(dict(active_target_id=1,loss_capture_frame_id=660,
        last_capture_frame_id=659,evidence_timestamp=owner._active_capture_timestamp)) is not None


def candidate(cap,stamp,x,uid=0):
    return dict(raw_track_id=1,uid=uid,detector_bbox=(x*640-60,10,x*640+60,470),
        sample_metadata=dict(capture_frame_id=cap,capture_timestamp=stamp,is_fresh=True),
        assignment=dict(best_uid=1,bbox_quality_ok=True))


def test_uid0_center_candidate_only_vetoes_restart_without_assigning_identity():
    guard=SearchBrakeObservation()
    first=candidate(156,10.,.421,1);second=candidate(159,10.15,.548)
    assert not guard.centered_candidate([first],1,156,10.,640,480,10.1)
    assert guard.centered_candidate([second],1,159,10.15,640,480,10.2)
    assert second["uid"] == 0
    assert not guard.centered_candidate([second],1,159,10.15,640,480,10.21)


def test_confirmed_center_does_not_add_observation_wait():
    guard=SearchBrakeObservation()
    guard.centered_candidate([candidate(156,10.,.421,1)],1,156,10.,640,480,10.1)
    assert not guard.centered_candidate([candidate(159,10.15,.548,1)],1,159,10.15,640,480,10.2)


@pytest.mark.parametrize("bad",["jump","raw","excluded","competition","wrong_uid","stale","multiple"])
def test_brake_only_candidate_checks_continuity_and_identity_evidence(bad):
    guard=SearchBrakeObservation()
    first=candidate(156,10.,.421,1)
    guard.centered_candidate([first],1,156,10.,640,480,10.1)
    second=candidate(159,10.15,.548)
    if bad=="jump": second["detector_bbox"]=(500,10,630,470)
    if bad=="raw": second["raw_track_id"]=2
    if bad=="excluded":second["assignment"]["search_excluded"]=True
    if bad=="competition":second["sample_metadata"]["identity_competition"]={"passed":False}
    if bad=="wrong_uid":second["assignment"]["best_uid"]=2
    rows=[second,second] if bad=="multiple" else [second]
    assert not guard.centered_candidate(rows,1,159,10.15,640,480,10.5 if bad=="stale" else 10.2)


def test_consume_records_current_capture_before_release(owner):
    consume(owner,[sample(owner,656,.421)])
    r=sample(owner,660,.72)
    owner._action_runtime._search_reacquire_settling=SimpleNamespace(ready_at=owner.clock-.2)
    owner._action_runtime._search_reacquire_brake_sent_at=100.
    def pending(**kwargs):
        if not kwargs:return True
        assert owner._search_brake_latest_observation == (1,660)
        assert kwargs["capture_timestamp"] == owner._active_capture_timestamp
        return False
    owner._action_runtime.search_reacquire_brake_pending=pending
    def normal(**kwargs):
        assert owner._follow_controller._target_direction_history.latest_reliable_side().direction == "right"
        raise RuntimeError("normal entry")
    owner._clear_longitudinal_context=normal
    with pytest.raises(RuntimeError,match="normal entry"):consume(owner,[r])


def test_real_runtime_and_tracker_hold_uid0_then_retire_old_left(owner, monkeypatch):
    from test_follow_wheel_periodic import setup_periodic
    from test_search_handoff_execution import arm,quiet
    rt,motor_owner,driver,_,clock,_=setup_periodic(monkeypatch)
    clock[0]=100.
    arm(rt,motor_owner,clock)
    owner._action_runtime=rt
    r=sample(owner,654,.421);clock[0]=owner.clock
    consume(owner,[r])  # seed observation before the actual STOP
    rt._service_follow_wheels()
    for t in (100.05,100.10):
        clock[0]=t;rt.get_steering_feedback=lambda:quiet(clock[0])
        rt._service_follow_wheels()
    r=sample(owner,658,.548,uid=0);clock[0]=owner.clock
    owner._rknn_pipeline.tracker.last_identity_observations[0]["assignment"]["best_uid"]=1
    consume(owner,[r])
    assert rt._search_reacquire_brake_request is not None
    assert driver.stops == [1, 0] and driver.pairs == [(0,0)]
    # The observation veto must expire, not rearm on every new central crop.
    # Remaining physically quiet is established by motor ticks throughout.
    for t in (100.50,100.51,100.56,100.60,100.65,100.70):
        clock[0]=t;rt._service_follow_wheels()
    # New image must follow the post-0A quiet confirmation, not just NORMAL.
    r=sample(owner,668,.55,uid=0);clock[0]=owner.clock
    owner._rknn_pipeline.tracker.last_identity_observations[0]["assignment"]["best_uid"]=1
    def normal(**kwargs):
        c=owner._follow_controller
        assert c.search_direction is None and c._lost_exit_direction is None
        assert c._target_direction_history.latest_visible_evidence() is None
        raise RuntimeError("normal entry without old turn")
    owner._clear_longitudinal_context=normal
    with pytest.raises(RuntimeError,match="normal entry without old turn"):consume(owner,[r])
    assert rt._search_reacquire_brake_request is None
    assert owner._current_forward_percent == 0 and owner._longitudinal_valid_until == 99.8
