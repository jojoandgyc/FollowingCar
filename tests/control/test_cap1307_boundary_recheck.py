"""Exercise the real record consumer and zero publisher, no hardware."""
import threading

import pytest

import request_0513_modular as runtime
from test_search_observation_arbitration import owner, _record, NOW


def prepare(owner):
    owner.search_state = owner._follow_controller.search_state = 'none'
    owner._vision_control_state = 'target_visible_depth_valid'
    owner._assignments[3] = dict(uid=0, mapped_uid=7,
        reason='partial_boundary_recheck', identity_control_rejected=True,
        identity_recheck_pending=True, identity_recheck_capture=231,
        identity_recheck_deadline=NOW+.12, bbox_quality_ok=False,
        bbox_quality_tier='reject', bank_updated=False)
    # Keep the real withdrawal path: verify that an existing grant is removed.
    del owner._clear_longitudinal_context
    owner._longitudinal_context_lock = threading.Lock()
    owner._longitudinal_context = {'old_roi': True}
    owner._depth30_linear_snapshot = ('forward',27,7,NOW-.139)
    owner._current_forward_allow_below_min = True


def test_pending_frame_stops_without_feeding_missing_target_to_controller(owner):
    prepare(owner)
    owner._consume_track_records([_record(uid=0)],640,480,'test')
    assert owner._events == [('queue',[runtime.ACTION_STOP],'identity_boundary_recheck')]
    assert owner._use_soft_stop_next
    assert owner._depth30_linear_snapshot is None and owner._longitudinal_context is None
    assert not owner._current_forward_allow_below_min
    assert owner._current_forward_percent == owner._current_rotate_raw_target == 0
    assert owner._follow_controller.active_target_id == 7
    assert owner._follow_controller.search_state == 'none'
    assert not owner._deferred_timeout  # no search-clock extension
    assert 'publish' not in owner._context_events


def test_next_confirmed_frame_uses_normal_control_not_sticky_recheck(owner):
    prepare(owner)
    owner._consume_track_records([_record(uid=0)],640,480,'test')
    owner._assignments[3] = dict(uid=7,bbox_quality_ok=True)
    owner._active_capture_frame_id=233
    owner._events.clear()
    owner._consume_track_records([_record(uid=7)],640,480,'test')
    assert [e[0] for e in owner._events] == ['normal']


@pytest.mark.parametrize('case',['expired','wrong_cap','wrong_uid','confirmed',
    'conflict','search','crowd','prediction','nan_deadline','future_capture','no_pending'])
def test_bad_or_expired_pending_evidence_cannot_skip_loss_handling(owner,case):
    prepare(owner); a=owner._assignments[3]; records=[_record(uid=0)]
    if case=='expired': a['identity_recheck_deadline']=NOW-.001
    if case=='wrong_cap': a['identity_recheck_capture']=230
    if case=='wrong_uid': a['mapped_uid']=8
    if case=='confirmed': a['uid']=7
    if case=='conflict': a['reason']='recent_partial_conflict'
    if case=='search': owner._follow_controller.search_state='searching'
    if case=='crowd': records.append(_record(track=4))
    if case=='prediction':
        from dataclasses import replace
        records=[replace(records[0],time_since_update=1)]
    if case=='nan_deadline': a['identity_recheck_deadline']=float('nan')
    if case=='future_capture': owner._active_capture_timestamp=NOW+.01
    if case=='no_pending': a.pop('identity_recheck_pending')
    assert not owner._hold_for_identity_boundary_recheck(records)
    assert owner._events == []


def test_true_conflict_immediately_returns_to_original_missing_target_flow(owner):
    prepare(owner)
    owner._assignments[3].update(reason='recent_partial_conflict',identity_recheck_pending=False)
    owner._consume_track_records([_record(uid=0)],640,480,'test')
    assert owner._events[0][0:2] == ('normal',[])
    assert owner._depth30_linear_snapshot is None


def test_hazard_has_priority_over_borderline_recheck(owner):
    prepare(owner)
    owner._handle_hazard_safety_state=lambda state: True
    owner._consume_track_records([_record(uid=0)],640,480,'test')
    assert owner._events == []  # no soft-stop overwrite of the hazard action
