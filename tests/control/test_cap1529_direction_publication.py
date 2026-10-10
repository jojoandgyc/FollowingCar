"""Historical direction source and negative evidence cross the real main API."""
from types import SimpleNamespace

import pytest
import request_0513_modular as runtime
from car_control_modular.historical_direction_backfill import HistoricalDirectionBackfill
from test_detector_identity_lease import fixture_pipeline
from test_lateral_zero_runtime import owner
from test_cap1532_direction_handoff import scene, frame


def service_owner(*, count=1, source="detector_formal_person_side"):
    received, headings = [], []
    controller = SimpleNamespace(_direction_loss_capture_id=206,
        _lost_exit_direction=None, search_direction=None,
        _target_direction_history=SimpleNamespace(latest_visible_evidence=lambda: None),
        note_historical_direction_hint=lambda *args, **kw: received.append((args, kw)) or True)
    def heading_at(stamp):
        headings.append(stamp)
        return 12.5
    owner = SimpleNamespace(_follow_controller=controller,
        _historical_backfill_pending=dict(loss_capture_frame_id=206, episode=1,
            active_target_id=1, started_at=99.95, deadline=100.),
        _historical_backfill=HistoricalDirectionBackfill(),
        _action_runtime=SimpleNamespace(get_steering_heading_at=heading_at),
        _direction_evidence_ring=[SimpleNamespace(capture_frame_id=cap,
            timestamp=stamp, state="visible", bbox=(60.,100.,140.,300.), score=.9,
            frame_width=640, candidate_count=count, reason=source)
            for cap,stamp in ((204,99.85),(205,99.9))])
    return owner, received, headings


def test_main_service_passes_original_source_count_and_capture_heading(monkeypatch):
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 100.)
    o, received, headings = service_owner()
    runtime.PersonTracker._service_historical_direction_backfill(o)
    assert len(received) == 1
    args, kw = received[0]
    assert args == ("left",)
    assert kw["evidence_timestamp"] == 99.85
    assert kw["loss_capture_frame_id"] == 206
    candidates = kw["association_candidates"]
    assert isinstance(candidates, tuple)
    assert [(x.capture_frame_id,x.timestamp,x.source,x.candidate_count,x.vehicle_yaw_deg)
            for x in candidates] == [
        (204,99.85,"detector_formal_person_side",1,12.5),
        (205,99.9,"detector_formal_person_side",1,12.5)]
    assert headings == [99.85,99.9]  # cached, capture-aligned reads only


@pytest.mark.parametrize("count", [0,2])
def test_empty_or_ambiguous_detection_is_not_rewritten_as_unique(monkeypatch,count):
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 100.)
    o, received, _ = service_owner(count=count)
    runtime.PersonTracker._service_historical_direction_backfill(o)
    assert not received


def test_diagnostic_source_is_not_upgraded_to_formal(monkeypatch):
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 100.)
    o, received, _ = service_owner(source="detector_diagnostic_person_side")
    runtime.PersonTracker._service_historical_direction_backfill(o)
    assert all(c.source == "detector_diagnostic_person_side"
               for c in received[0][1]["association_candidates"])


@pytest.mark.parametrize("reason", [None,"geometry_conflict","competition_conflict"])
def test_real_main_clears_only_negative_direction_evidence_not_uid0(owner,reason):
    _, records = fixture_pipeline(owner)
    owner._update_detector_identity_lease(records,4,99.9,now=100.,stale=False)
    cleared = []
    controller = owner._follow_controller
    controller._historical_direction_hint = {"active_target_id":1}
    controller.clear_historical_direction_hint = cleared.append
    owner._historical_backfill_pending = {"episode":1}
    owner._rknn_pipeline.tracker.associated_position_contradiction = lambda *a: reason
    records[0].reid_uid = 0
    owner._update_detector_identity_lease(records,5,100.01,now=100.02,stale=False)
    assert cleared == ([] if reason is None else [reason])
    assert (owner._historical_backfill_pending is None) == (reason is not None)


def test_explicit_stop_clears_direction_even_for_duplicate_pipeline_output(owner):
    _, records = fixture_pipeline(owner)
    owner._update_detector_identity_lease(records,4,99.9,now=100.,stale=False)
    cleared = []
    owner._follow_controller.clear_historical_direction_hint = cleared.append
    owner._historical_backfill_pending = {"episode":1}
    owner._explicit_stop_requested = True
    owner._update_detector_identity_lease(records,4,99.9,now=100.,stale=False)
    assert cleared == ["explicit_stop"]
    assert owner._historical_backfill_pending is None


@pytest.mark.parametrize("source", ["detector_formal_person_side", "detector_diagnostic_person_side"])
def test_main_delivers_late_chain_after_search_has_started(scene, source):
    c, _, samples = scene
    c.search_state, c.search_direction = "searching", "left"
    c._lost_exit_direction, c._lost_hint_source = "left", "latest_reliable_capture_side"
    c._search_rotation_started_at = 10.1
    o = SimpleNamespace(_follow_controller=c,
        _historical_backfill_pending=dict(loss_capture_frame_id=1529,episode=1,
            active_target_id=1,started_at=10.4,deadline=10.58),
        _historical_backfill=HistoricalDirectionBackfill(),
        _direction_evidence_ring=[SimpleNamespace(capture_frame_id=s.capture_frame_id,
            timestamp=s.timestamp,state=s.state,bbox=s.bbox,score=s.score,
            frame_width=s.frame_width,candidate_count=1,reason=source) for s in samples])
    runtime.PersonTracker._service_historical_direction_backfill(o)
    assert c._historical_direction_hint is not None
    assert o._historical_backfill_pending is None
    c._ensure_search_state(frame())
    assert c.search_direction == ("right" if source == "detector_formal_person_side" else "left")
    assert c._search_rotation_started_at == 10.1


def test_first_backfill_can_start_when_search_already_chose_old_side(scene,monkeypatch):
    c, _, _ = scene
    monkeypatch.setattr(runtime,"FOLLOW_DIRECTION_HISTORY_ENABLE",True)
    monkeypatch.setattr(runtime,"HISTORICAL_DIRECTION_BACKFILL_ENABLE",True)
    c.search_state,c.search_direction = "searching","left"
    c._lost_exit_direction,c._lost_hint_source = "left","latest_reliable_capture_side"
    o = SimpleNamespace(_follow_controller=c,_vision_control_state="searching",
        _historical_backfill_pending=None,_historical_backfill_last_start_capture=0,
        _historical_backfill_episode=0,_direction_evidence_ring=[],_capture_metadata_ring=[],
        _service_historical_direction_backfill=lambda: None)
    runtime.PersonTracker._maybe_schedule_historical_direction_backfill(o,1532,1)
    assert o._historical_backfill_pending["loss_capture_frame_id"] == 1529
    runtime.PersonTracker._maybe_schedule_historical_direction_backfill(o,1535,1)
    assert o._historical_backfill_episode == 1


@pytest.mark.parametrize("stop", ["explicit", "timed_out", "confirmed"])
def test_stopped_or_confirmed_direction_does_not_consume_late_chain(monkeypatch,stop):
    monkeypatch.setattr(runtime.time,"monotonic",lambda:100.)
    o, received, _ = service_owner()
    if stop == "explicit": o._explicit_stop_requested = True
    elif stop == "timed_out": o._follow_controller.search_state = "timed_out"
    else: o._follow_controller._lost_hint_source = "search_candidate_opposite_side"
    runtime.PersonTracker._service_historical_direction_backfill(o)
    assert not received and o._historical_backfill_pending is None


def test_slow_ambiguous_slot_cannot_disappear_from_association_chain(scene):
    c, _, samples = scene
    c.search_state,c.search_direction = "searching","left"
    c._lost_exit_direction,c._lost_hint_source = "left","latest_reliable_capture_side"
    ring = [SimpleNamespace(capture_frame_id=s.capture_frame_id,timestamp=s.timestamp,
        state=s.state,bbox=s.bbox,score=s.score,frame_width=s.frame_width,
        candidate_count=1,reason=s.source,result_age_ms=100.) for s in samples]
    ring[2].result_age_ms,ring[2].candidate_count = 501.,2
    o = SimpleNamespace(_follow_controller=c,
        _historical_backfill_pending=dict(loss_capture_frame_id=1529,episode=1,
            active_target_id=1,started_at=10.4,deadline=10.58),
        _historical_backfill=HistoricalDirectionBackfill(),_direction_evidence_ring=ring)
    runtime.PersonTracker._service_historical_direction_backfill(o)
    assert c._historical_direction_hint is not None  # fallback may ignore the slot
    assert c._historical_direction_hint.get("association") is None
    c._ensure_search_state(frame())
    assert c.search_direction == "left"


@pytest.mark.parametrize("owned",[True,False])
def test_new_raw_hard_conflict_is_checked_but_unrelated_background_cannot_revoke(owner,owned):
    from types import MethodType
    from rk_vision.tracker import DeepSortTracker
    _, records = fixture_pipeline(owner)
    owner._update_detector_identity_lease(records,4,99.9,now=100.,stale=False)
    t = owner._rknn_pipeline.tracker
    t._frame_index = 5
    t.identity_bank = SimpleNamespace(_geometry_revoked_uids={},
        _mapped_geometry_conflicts={25:dict(uid=1)},search_exclusion_for=lambda *a,**kw:None)
    t.last_identity_observations = [dict(raw_track_id=25,frame_index=5,
        sample_metadata=dict(is_fresh=True,capture_frame_id=5,capture_timestamp=100.01),
        assignment=dict(uid=0,mapped_uid=1 if owned else None,best_uid=1,
                        reason="mapped_geometry_reject"))]
    t.associated_position_contradiction = MethodType(DeepSortTracker.associated_position_contradiction,t)
    cleared = []
    owner._follow_controller._historical_direction_hint = dict(active_target_id=1)
    owner._follow_controller.clear_historical_direction_hint = cleared.append
    records[0].reid_uid,records[0].track_id = 0,25
    owner._update_detector_identity_lease(records,5,100.01,now=100.02,stale=False)
    assert cleared == (["geometry_conflict"] if owned else [])
