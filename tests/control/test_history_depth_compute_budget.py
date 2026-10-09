"""Skip already unusable history, never predict expiry from a slow scan."""
from copy import deepcopy
from dataclasses import replace

import pytest
import request_0513_modular as runtime

from car_control_modular.astra_depth import AstraDepthRuntime
from car_control_modular.depth_measurement_transaction import RANGING_STATE_FIELDS
from test_depth_optimistic_transaction import scene, prepare
from test_turn_depth_scheduling import owner, context, attempt, NOW


def history(scene, *, roi_age=.310, sample_age=.140):
    obj, camera, distance, clock = scene
    distance.config = replace(distance.config, vision_depth_detector_bbox_max_age_sec=.5)
    obj._longitudinal_context = context(4910, clock[0]-roi_age)
    camera._depth_history.append((clock[0]-sample_age, camera._latest_depth))
    camera._latest_depth_ts = clock[0]


def test_slow_scan_does_not_gate_a_later_fast_historical_sample(scene, monkeypatch):
    obj, camera, distance, clock = scene
    scan = AstraDepthRuntime._select_multiregion_distance
    calls = []
    def costly(private, *args, **kw):
        calls.append(private)
        result = scan(private, *args, **kw)
        clock[0] += .100 if len(calls) == 1 else .020
        return result
    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", costly)
    first, target = prepare(scene)
    assert first.transaction.run().raw_distance_m == pytest.approx(1.8)
    assert distance.commit_prepared_depth(first, target=target)
    clock[0] += .1  # A newer physical frame, not an older accepted duplicate.
    history(scene)
    before = deepcopy({k: getattr(camera, k) for k in RANGING_STATE_FIELDS if hasattr(camera, k)})
    old_grant = obj._depth30_linear_snapshot
    second, _ = prepare(scene)
    assert second.transaction.run().raw_distance_m == pytest.approx(1.8)
    assert second.transaction.reject_reason is None
    assert second.transaction.preflight_remaining_sec == pytest.approx(.04)
    assert len(calls) == 2
    assert not hasattr(camera, "_depth_scan_durations")
    assert obj._depth30_linear_snapshot is old_grant
    # Scheduling statistics cannot mutate temporal filters/confirmation.
    assert before == {k: getattr(camera, k) for k in before}


@pytest.mark.parametrize("roi_age,sample_age,compute", [(.275, .105, .047),
    (.254, .083, .078), (.2794, .1103, .0546)])
def test_real_successful_history_samples_are_not_predicted_to_expire(
        scene, monkeypatch, roi_age, sample_age, compute):
    obj, camera, distance, clock = scene
    history(scene, roi_age=roi_age, sample_age=sample_age)
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC", .5)
    scan = AstraDepthRuntime._select_multiregion_distance
    def measured(private, *args, **kw):
        result = scan(private, *args, **kw)
        clock[0] += compute
        return result
    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance", measured)
    old = obj._depth30_linear_snapshot
    assert attempt(obj)
    assert len(obj.calls) == 1 and obj._depth30_linear_snapshot is old
    assert camera._last_accepted_ts == pytest.approx(NOW-sample_age)


def test_newest_roi_is_not_blocked_by_history_preflight(scene):
    obj, camera, distance, clock = scene
    first, target = prepare(scene)
    assert first.transaction.run().raw_distance_m == pytest.approx(1.8)
    assert distance.commit_prepared_depth(first, target=target)


def test_skipped_history_preserves_pending_confirmation_and_new_roi_can_proceed(scene):
    obj, camera, distance, clock = scene
    history(scene, roi_age=.410)
    camera._pending_jump_count = 2
    camera._pending_jump_distance_m = 2.6
    camera._pending_jump_timestamp = NOW-.2
    pending = (camera._pending_jump_count, camera._pending_jump_distance_m,
               camera._pending_jump_timestamp)
    skipped, target = prepare(scene)
    assert skipped.transaction.run() is None
    assert pending == (camera._pending_jump_count, camera._pending_jump_distance_m,
                       camera._pending_jump_timestamp)
    obj._longitudinal_context = context(4911, NOW-.04)
    current, target = prepare(scene)
    assert current.transaction.run() is not None
    assert distance.commit_prepared_depth(current, target=target)
    assert camera._last_accepted_ts == NOW


def test_history_with_remaining_physical_budget_still_completes(scene):
    obj, camera, distance, clock = scene
    history(scene, roi_age=.255, sample_age=.080)
    item, target = prepare(scene)
    assert item.transaction.run().raw_distance_m == pytest.approx(1.8)
    assert distance.commit_prepared_depth(item, target=target)
    assert camera._last_accepted_ts == pytest.approx(NOW-.080)


def test_500ms_roi_cannot_invent_history_outside_180ms_association(scene, monkeypatch):
    history(scene, roi_age=.410, sample_age=.010)
    monkeypatch.setattr(AstraDepthRuntime, "_select_multiregion_distance",
                        lambda *a, **kw: pytest.fail("no eligible physical sample"))
    item, target = prepare(scene)
    assert item.transaction.run() is None
    assert item.transaction.reject_reason == "history_no_eligible_sample"
    assert not scene[2].commit_prepared_depth(item, target=target)


@pytest.mark.parametrize("failure", ["expiry", "revision", "stopped"])
def test_budget_never_replaces_final_transaction_validation(scene, failure):
    obj, camera, distance, clock = scene
    item, target = prepare(scene)
    assert item.transaction.run() is not None
    if failure == "expiry": clock[0] += .161
    if failure == "revision": camera._measurement_revision += 1
    if failure == "stopped": camera._stop_event.set()
    assert not distance.commit_prepared_depth(item, target=target)
    assert item.transaction.reject_reason == {
        "expiry": "physical_sample_expired_or_future",
        "revision": "measurement_revision_changed", "stopped": "runtime_stopped"}[failure]
    assert camera._last_accepted_ts == 0
