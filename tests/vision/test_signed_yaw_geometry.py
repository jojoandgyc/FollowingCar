"""One physical yaw convention across fast detection and identity recovery."""
from copy import deepcopy
from dataclasses import replace

import pytest

from rk_vision.camera_geometry import horizontal_center_displacement, yaw_image_shift_ratio
from rk_vision.identity_bank import (
    IdentityBank, IdentityBankConfig, PendingHandoff, PendingLateHandoff,
    _pending_observation_continuous,
)
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from test_detector_continuation import full, plan
from test_mapped_identity_geometry import ANCHOR, WRONG, bank_with_anchor, assign


@pytest.mark.parametrize("yaw,delta,residual", [
    (6., -.1, 0.), (-6., .1, 0.), (6., .1, .2), (-6., -.1, .2),
    (6., 0., .1), (0., .1, .1), (None, .1, .1),
])
def test_measured_right_turn_explains_only_left_image_motion(yaw, delta, residual):
    raw, compensated = horizontal_center_displacement(
        current_center=.5+delta, previous_center=.5,
        current_yaw=yaw, previous_yaw=0., camera_hfov_deg=60.,
    )
    assert raw == pytest.approx(abs(delta))
    assert compensated == pytest.approx(residual)
    if yaw is not None:
        assert yaw_image_shift_ratio(0., yaw, 60.) == pytest.approx(-yaw/60.)


@pytest.mark.parametrize("yaw,shift", [(6., -64.), (-6., 64.), (6., 64.), (-6., -64.)])
def test_real_tracker_fast_path_rechecks_physically_signed_motion(yaw, shift):
    tracker = DeepSortTracker(DeepSortTrackerConfig(n_init=1,
        identity_new_confirm_frames=1, identity_update_interval=1, hfov_deg=60.))
    box = (180., 40., 400., 460.)
    for cap in range(1, 5):
        full(tracker, cap, 10.+cap*.05, bbox=box)
    reference = deepcopy(tracker.identity_bank.identities[1].last_strong_observation)
    ctx = dict(capture_frame_id=5, capture_timestamp=10.25, integrated_yaw_deg=yaw)
    moved = (box[0]+shift, box[1], box[2]+shift, box[3])
    proposal = plan(tracker, bbox=moved, ctx=ctx)
    if yaw * shift > 0:
        # Fall back to a full identity check; no fast UID/negative evidence.
        assert proposal is None
        assert tracker.last_detector_continuation_reason == "adjacent_geometry"
        assert not tracker.identity_bank._mapped_geometry_conflicts
        assert tracker._frame_index == 4
    else:
        assert proposal is not None
        # The final commit independently rechecks the same sign, not only plan().
        wrong = replace(proposal, observation=replace(proposal.observation, yaw=-yaw))
        assert tracker.commit_detected_continuation(wrong, now=10.295) is None
        assert tracker._frame_index == 4
        # Build a fresh independent proof after a rejected plan.
        for cap in (5, 6):
            full(tracker, cap, 10.+cap*.05, bbox=box)
        ctx.update(capture_frame_id=7, capture_timestamp=10.35)
        proposal = plan(tracker, cap=7, stamp=10.35, bbox=moved, ctx=ctx)
        rows = tracker.commit_detected_continuation(proposal, now=10.395)
        assert len(rows) == 1 and rows[0].reid_uid == 1
        assert not tracker.control_assignment_for_track(1)["bank_updated"]
    assert tracker.identity_bank.identities[1].last_strong_observation["bbox"] == reference["bbox"]


@pytest.mark.parametrize("yaw,expected", [(24., False), (-24., False), (None, True)])
def test_local_handoff_cannot_choose_smaller_raw_jump(yaw, expected):
    config = IdentityBankConfig(camera_hfov_deg=60.)
    pending = PendingHandoff(1, 10, center_ratio=.5, area=.2,
                            capture_timestamp=10., integrated_yaw_deg=0.)
    current = dict(center_x_ratio=.55, area=.2, capture_timestamp=10.1,
                   integrated_yaw_deg=yaw)
    # Neither known large turn is explained by a five-percent image motion.
    assert _pending_observation_continuous(pending, current, 11, config) is expected


@pytest.mark.parametrize("kind", ["weak", "late"])
@pytest.mark.parametrize("yaw,center,expected", [
    (24., .5, False), (24., .1, True), (-24., .9, True), (-24., .1, False),
])
def test_raw_track_bridge_uses_same_signed_residual(kind, yaw, center, expected):
    bank = IdentityBank(IdentityBankConfig(camera_hfov_deg=60.))
    cls = PendingHandoff if kind == "weak" else PendingLateHandoff
    previous = cls(1, 10, center_ratio=.5, area=.2, area_units="ratio",
                   geometry_source="detector", capture_timestamp=10., integrated_yaw_deg=0.)
    store = bank.pending_weak_handoffs if kind == "weak" else bank.pending_late_handoffs
    store[-1] = previous
    bank._bridge_weak_handoff_observation(track_id=3, candidate_uid=1, frame_index=11,
        sample_metadata=dict(detector_bbox=((center-.08)*640, 70, (center+.08)*640, 400),
                             detector_center_x_ratio=center, detector_area_ratio=.2,
                             capture_timestamp=10.1, integrated_yaw_deg=yaw))
    assert (3 in bank.pending_weak_handoffs) is expected
    if not expected:
        assert store[-1] is previous
        assert not bank.track_to_uid


@pytest.mark.parametrize("new_track", [3, 4])
@pytest.mark.parametrize("yaw,center,expected", [
    (12., .25, True), (-12., .75, True), (12., .75, False), (-12., .25, False),
    (24., .5, False),
])
def test_late_candidate_needs_signed_local_continuity_before_uid(new_track, yaw, center, expected):
    bank = bank_with_anchor()
    bank.config = replace(bank.config, camera_hfov_deg=60.)
    original = deepcopy(bank.identities[1].last_strong_observation)

    def observe(track, frame, x, angle):
        metadata = dict(
            detector_bbox=((x-.08)*640, 70, (x+.08)*640, 400),
            detector_center_x_ratio=x, detector_area_ratio=.2,
            capture_frame_id=frame, capture_timestamp=20.+(frame-300)*.1,
            integrated_yaw_deg=angle, search_reacquire_context_active=True, is_fresh=True,
        )
        return bank._observe_late_search_candidate(
            track_id=track, candidate_uid=1, distance=.05, partial_feature=None,
            match_source="strong", candidate_count=1, bbox_quality_ok=True,
            sample_metadata=metadata, frame_index=frame,
            geometry=bank._handoff_geometry(1, metadata, frame),
        )

    assert observe(3, 300, .5, 0.)[:2] == (0, 1)
    second = observe(new_track, 301, center, yaw)
    assert bool(second and second[0] == 1) is expected
    if expected:
        assert second[1] == 2
        assert bank._reacquire_quarantine.is_held(1)
    else:
        assert second is None or second[:2] == (0, 1)
        assert new_track not in bank.track_to_uid
        assert bank.identities[1].last_strong_observation == original
    assert len(bank.identities[1].features) == 1


def test_wrong_yaw_cannot_hide_recorded_cross_person_jump():
    bank = bank_with_anchor()
    original = deepcopy(bank.identities[1].last_strong_observation)
    delta = ((WRONG[0]+WRONG[2])-(ANCHOR[0]+ANCHOR[2]))/1280.
    # Leftward box displacement AND left chassis turn used to cancel under
    # min(+yaw, -yaw). Physically they add: the old person should move right.
    wrong_yaw = original["integrated_yaw_deg"] + delta*90.
    assert assign(bank, metadata_extra={"integrated_yaw_deg": wrong_yaw}) == 0
    result = bank.last_assignments[1]
    assert result["reason"] == "mapped_geometry_reject"
    assert result["reacquire_geometry"]["yaw_compensated_center_jump_ratio"] == pytest.approx(abs(delta)*2.)
    assert bank.identities[1].last_strong_observation == original
    assert not result["bank_updated"]
