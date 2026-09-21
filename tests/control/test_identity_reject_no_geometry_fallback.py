"""Identity rejection is not a recoverable clipped-body observation."""
import pytest
import request_0513_modular as runtime
from test_search_observation_arbitration import owner, _record


@pytest.mark.parametrize("quality",[True,False])
def test_rejected_identity_cannot_authorize_normal_or_limited_yaw(owner,quality):
    owner._assignments[3]=dict(uid=0,mapped_uid=7,bbox_quality_ok=quality,
        identity_control_rejected=True,reason="reacquire_control_verify_reject",
        bbox_quality_reason="reacquire_identity_unverified")
    rec=_record(uid=0)
    owner._consume_track_records([rec],640,480,"test")
    assert not any(e[0]=="limited_yaw" for e in owner._events)
    assert all(not e[1] for e in owner._events if e[0]=="normal")
    result,_=runtime.PersonTracker._active_target_track_record(owner,[rec],7)
    assert result is None


def test_confirmation_grace_cannot_restore_identity_rejection(owner):
    assignment=dict(identity_control_rejected=True)
    rec=_record(uid=0)
    assert runtime.PersonTracker._visual_reacquire_hold_match(
        owner,rec,assignment,person_count=1,width=640,height=480) is None
    assert runtime.PersonTracker._search_geometry_reacquire_id(
        owner,rec,assignment,person_count=1,width=640) is None
