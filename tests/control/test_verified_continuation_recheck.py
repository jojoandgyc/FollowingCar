"""Real record consumer and zero publisher for uncertain torso continuation.

Constructor/hardware endpoints are fakes. The identity recheck, longitudinal
withdrawal and normal/missing-target dispatch paths are the runtime methods.
"""
from dataclasses import replace

import pytest

import request_0513_modular as runtime
from test_cap1307_boundary_recheck import prepare
from test_search_observation_arbitration import owner, _record, NOW


def pending(owner):
    prepare(owner)
    owner._assignments[3].update(reason="verified_continuation_recheck",
        identity_continuation=dict(status="hold", source="partial", pair_cap=414,
                                   reference_cap=230, deadline=NOW+.12))


def test_grey_continuation_withdraws_existing_motion_and_publishes_only_zero(owner):
    pending(owner)
    owner._consume_track_records([_record(uid=0)], 640, 480, "test")
    assert owner._events == [("queue", [runtime.ACTION_STOP], "identity_boundary_recheck")]
    assert owner._use_soft_stop_next
    assert owner._current_forward_percent == owner._current_rotate_raw_target == 0
    assert owner._depth30_linear_snapshot is None
    assert owner._longitudinal_context is None
    assert not owner._current_forward_allow_below_min
    assert owner._follow_controller.active_target_id == 7
    assert owner._follow_controller.search_state == "none"
    assert owner._deferred_timeout == []
    assert "publish" not in owner._context_events


def test_repeated_grey_records_never_extend_deadline_or_reauthorize_depth(owner, monkeypatch):
    pending(owner)
    original = owner._assignments[3]["identity_recheck_deadline"]
    for offset, cap in [(0., 231), (.05, 233), (.10, 235)]:
        monkeypatch.setattr(runtime.time, "monotonic", lambda offset=offset: NOW+offset)
        owner._active_capture_frame_id = cap
        owner._active_capture_timestamp = NOW+offset-.02
        owner._assignments[3]["identity_recheck_capture"] = cap
        owner._consume_track_records([_record(uid=0)], 640, 480, "test")
        assert owner._assignments[3]["identity_recheck_deadline"] == original
        assert owner._depth30_linear_snapshot is None
        assert owner._longitudinal_context is None
    assert all(e == ("queue", [runtime.ACTION_STOP], "identity_boundary_recheck")
               for e in owner._events)
    assert not owner._deferred_timeout
    assert "publish" not in owner._context_events
    owner._events.clear()
    monkeypatch.setattr(runtime.time, "monotonic", lambda: original+.001)
    owner._active_capture_frame_id = 237
    owner._active_capture_timestamp = original-.01
    owner._assignments[3]["identity_recheck_capture"] = 237
    owner._consume_track_records([_record(uid=0)], 640, 480, "test")
    assert owner._events[0][0:2] == ("normal", [])
    assert owner._depth30_linear_snapshot is None
    assert not owner._deferred_timeout


def test_next_new_confirmed_continuation_returns_to_normal_controller(owner):
    pending(owner)
    owner._consume_track_records([_record(uid=0)], 640, 480, "test")
    owner._events.clear()
    owner._active_capture_frame_id = 233
    owner._assignments[3] = dict(uid=7, mapped_uid=7, bbox_quality_ok=True,
        bbox_quality_tier="strong", reason="mapped_verified_continuation", bank_updated=False)
    owner._consume_track_records([_record(uid=7)], 640, 480, "test")
    assert [event[0] for event in owner._events] == ["normal"]
    # A newly confirmed identity alone has not fabricated a depth grant.
    assert owner._depth30_linear_snapshot is None


@pytest.mark.parametrize("case", ["expired", "wrong_cap", "wrong_uid", "predicted",
    "conflict", "search", "crowd", "confirmed_record", "future_capture",
    "nan_deadline", "missing_pending", "not_rejected"])
def test_invalid_recheck_does_not_bypass_normal_loss_path(owner, case):
    pending(owner)
    a = owner._assignments[3]
    records = [_record(uid=0)]
    if case == "expired": a["identity_recheck_deadline"] = NOW-.001
    elif case == "wrong_cap": a["identity_recheck_capture"] = 230
    elif case == "wrong_uid": a["mapped_uid"] = 8
    elif case == "predicted": records = [replace(records[0], time_since_update=1)]
    elif case == "conflict": a["reason"] = "recent_partial_conflict"
    elif case == "search": owner._follow_controller.search_state = "searching"
    elif case == "crowd": records.append(_record(track=4, uid=0))
    elif case == "confirmed_record": records = [_record(uid=7)]
    elif case == "future_capture": owner._active_capture_timestamp = NOW+.01
    elif case == "nan_deadline": a["identity_recheck_deadline"] = float("nan")
    elif case == "missing_pending": a.pop("identity_recheck_pending")
    elif case == "not_rejected": a["identity_control_rejected"] = False
    assert not owner._hold_for_identity_boundary_recheck(records)
    assert owner._events == []
    assert owner._deferred_timeout == []


def test_hazard_takes_priority_over_grey_continuation_soft_stop(owner):
    pending(owner)
    owner._handle_hazard_safety_state = lambda state: True
    owner._consume_track_records([_record(uid=0)], 640, 480, "test")
    assert owner._events == []
