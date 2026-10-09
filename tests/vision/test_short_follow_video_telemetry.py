"""Paired requests and cached plan authority are never shown as wheel ACKs."""
import csv
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from car_control_modular.detector_identity_lease import ValidatedVisualObservation
from car_control_modular.short_follow import ShortFollowConfig, ShortFollowController, ShortFollowObservation
from car_control_modular.video_follow_telemetry import (
    FollowRecordingView, PairedRecordingAuthority, build_follow_snapshot, recording_authority,
)
from car_control_modular.video_recorder import AsyncVideoRecorder, VideoRecorderConfig
from test_video_follow_telemetry import builder_inputs


def paired_recording():
    controller, frame, decision = builder_inputs()
    # Simulate the last startup/legacy result remaining in the old controller.
    controller.last_distance_pid_result.output_rpm = 180.
    short = ShortFollowController(ShortFollowConfig(enabled=True))
    short.activate(1, 99.)
    short.update(ShortFollowObservation(
        uid=1, capture_id=123, capture_timestamp=100., depth_timestamp=100.,
        distance_m=2.0, center_x_ratio=.7,
    ), now=100.05)
    owner = SimpleNamespace(
        _short_follow=short, _follow_controller=controller,
        _validated_visual_observation=ValidatedVisualObservation(
            uid=1, track_id=9, capture=123, timestamp=100.,
            validated_at=100.05, expires_at=100.50, kind='full'),
        _detector_identity_lease=None, _depth30_linear_snapshot=None,
        _depth30_linear_timing=None, search_state='none',
    )
    snapshot = build_follow_snapshot(controller, frame, decision, 'depth30', 100.05,
                                     paired_snapshot=short.snapshot())
    return owner, snapshot, frame, decision


def test_paired_recording_hides_old_pid_and_target_speed_without_mutating_them():
    owner, snapshot, _, _ = paired_recording()
    assert snapshot.control_mode == 'paired'
    assert snapshot.pid_rpm is snapshot.matching_base_rpm is None
    assert snapshot.speed_timestamp is snapshot.target_speed_m_s is snapshot.relative_speed_m_s is None
    assert snapshot.speed_status == 'disabled'
    assert owner._follow_controller.last_distance_pid_result.output_rpm == 180.
    authority = recording_authority(owner)
    assert isinstance(authority, PairedRecordingAuthority)
    view = FollowRecordingView.from_snapshot(snapshot, authority, 100.10)
    plan = owner._short_follow.snapshot().plan
    assert (view.paired_left_rpm, view.paired_right_rpm) == (plan.left_rpm, plan.right_rpm)
    assert (view.authorized_left_rpm, view.authorized_right_rpm) == (plan.left_rpm, plan.right_rpm)
    assert view.authority_status == 'active'
    assert view.authorized_rpm == (plan.left_rpm+plan.right_rpm)/2.
    assert 'PAIR REQ' in view.labels()[0]
    assert 'AUTH PLAN' in view.labels()[2] and 'NOT ACK' in view.labels()[2]
    assert all('PID ' not in label and '180' not in label for label in view.labels())
    assert snapshot.paired_pi_request_rpm == plan.base_request_rpm
    assert view.paired_pi_request_rpm == plan.base_request_rpm
    assert view.paired_p_rpm == plan.p_rpm and view.paired_i_rpm == plan.i_rpm
    assert view.paired_base_rpm == plan.base_rpm
    assert view.paired_speed_cap_rpm == plan.speed_cap_rpm
    assert view.paired_integral_dt_sec == plan.integral_dt_sec
    assert view.paired_limit_reason == plan.limit_reason
    assert 'PI REQ' in view.labels()[5] and 'BASE' in view.labels()[5]
    assert view.paired_motion_kind == view.authority_motion_kind == 'steer_right'


@pytest.mark.parametrize('center,kind', [(.2, 'pivot_left'), (.8, 'pivot_right')])
def test_near_pivot_is_recorded_as_rotation_with_zero_translation(center, kind):
    owner, _, frame, decision = paired_recording()
    plan = owner._short_follow.update(ShortFollowObservation(
        uid=1, capture_id=124, capture_timestamp=100.1, depth_timestamp=100.1,
        distance_m=1.45, center_x_ratio=center,
    ), now=100.11)
    assert plan.pivot
    owner._validated_visual_observation = replace(owner._validated_visual_observation,
        capture=124, timestamp=100.1, validated_at=100.11, expires_at=100.6)
    snapshot = build_follow_snapshot(owner._follow_controller, frame, decision, 'depth30', 100.11,
                                     paired_snapshot=owner._short_follow.snapshot())
    authority = recording_authority(owner)
    view = FollowRecordingView.from_snapshot(snapshot, authority, 100.12)
    assert view.authority_status == 'active'
    assert view.paired_motion_kind == view.authority_motion_kind == kind
    assert view.authorized_left_rpm == -view.authorized_right_rpm != 0
    assert view.authorized_rpm == view.paired_base_rpm == 0
    assert kind.upper() in view.labels()[0] and kind.upper() in view.labels()[2]
    assert 'NOT ACK' in view.labels()[2]
    expired = FollowRecordingView.from_snapshot(snapshot, authority, 100.401)
    assert expired.authority_status == 'expired'
    assert expired.authority_motion_kind == 'stop'
    assert expired.authorized_left_rpm == expired.authorized_right_rpm == 0
    # The prior request stays explicitly identified; it is not a current ACK.
    assert expired.paired_motion_kind == kind


def test_new_pivot_authority_does_not_relabel_previous_arc_request():
    owner, snapshot, _, _ = paired_recording()
    plan = owner._short_follow.update(ShortFollowObservation(
        uid=1, capture_id=124, capture_timestamp=100.1, depth_timestamp=100.1,
        distance_m=1.45, center_x_ratio=.2,
    ), now=100.11)
    assert plan.pivot
    view = FollowRecordingView.from_snapshot(snapshot, recording_authority(owner), 100.12)
    assert view.paired_motion_kind == 'steer_right'
    assert view.authority_motion_kind == 'pivot_left'
    assert view.paired_sequence != view.authority_sequence
    assert view.paired_base_rpm > 0 and view.authorized_rpm == 0


def test_paired_plan_uses_actual_finite_300ms_deadline_not_legacy_display_180ms():
    owner, snapshot, _, _ = paired_recording()
    authority = recording_authority(owner)
    view = FollowRecordingView.from_snapshot(snapshot, authority, 100.25)
    assert view.authority_status == 'active'
    assert view.depth_status == 'fresh'
    assert view.authority_remaining_ms == pytest.approx(50.)
    expired = FollowRecordingView.from_snapshot(snapshot, authority, 100.301)
    assert expired.authority_status == 'expired'
    assert expired.authorized_left_rpm == expired.authorized_right_rpm == 0.
    assert expired.authorized_rpm == 0.


def test_paired_pi_diagnostics_capture_nonzero_integral_without_using_legacy_cache():
    owner, _, frame, decision = paired_recording()
    # At this small, persistent positive error the pure PI is below its cap,
    # so subsequent fresh samples can accumulate real time-weighted I.
    for cap, stamp in ((124, 100.10), (125, 100.20)):
        plan = owner._short_follow.update(ShortFollowObservation(
            uid=1, capture_id=cap, capture_timestamp=stamp, depth_timestamp=stamp,
            distance_m=1.6, center_x_ratio=.5,
        ), now=stamp + .01)
    assert plan.i_rpm > 0 and plan.base_request_rpm == pytest.approx(plan.p_rpm + plan.i_rpm)
    snapshot = build_follow_snapshot(owner._follow_controller, frame, decision, 'depth30', 100.21,
                                     paired_snapshot=owner._short_follow.snapshot())
    view = FollowRecordingView.from_snapshot(snapshot, recording_authority(owner), 100.22)
    assert view.paired_i_rpm == plan.i_rpm
    assert view.paired_integral_dt_sec == pytest.approx(.10)
    assert view.pid_rpm is None
    assert owner._follow_controller.last_distance_pid_result.output_rpm == 180.


def test_paired_authority_respects_shorter_grey_identity_deadline():
    owner, snapshot, _, _ = paired_recording()
    plan = owner._short_follow.snapshot().plan
    owner._validated_visual_observation = replace(owner._validated_visual_observation,
        continuation_sample_timestamp=plan.depth_timestamp, expires_at=100.11)
    authority = recording_authority(owner)
    assert authority.expires_at == 100.11
    view = FollowRecordingView.from_snapshot(snapshot, authority, 100.12)
    assert view.authority_status == 'expired'
    assert owner._short_follow.snapshot().plan is plan
    assert plan.expires_at == pytest.approx(100.30)


@pytest.mark.parametrize('block', ['explicit_stop', 'shutdown', 'brake_hold', 'search',
                                  'wrong_uid', 'identity_rejected', 'motor_fault', 'revoke'])
def test_current_revocation_or_identity_state_cannot_show_old_pair_as_authorized(block):
    owner, snapshot, _, _ = paired_recording()
    if block == 'explicit_stop':
        owner._explicit_stop_requested = True
    elif block == 'shutdown':
        owner._runtime_shutdown_requested = True
    elif block == 'brake_hold':
        owner._brake_hold_active = True
    elif block == 'search':
        owner.search_state = 'searching'
    elif block == 'wrong_uid':
        owner._follow_controller.active_target_id = 2
    elif block == 'identity_rejected':
        owner._validated_visual_observation = False
    elif block == 'motor_fault':
        owner._motor_backend = SimpleNamespace(motion_write_fault='left_ack_failed')
    else:
        owner._short_follow.revoke('manual_stop', 100.08)
    view = FollowRecordingView.from_snapshot(snapshot, recording_authority(owner), 100.1)
    assert view.authority_status != 'active'
    assert view.authorized_left_rpm is None and view.authorized_right_rpm is None
    # The earlier request remains visibly labelled as a request, not execution.
    assert view.paired_left_rpm is not None and 'PAIR REQ' in view.labels()[0]


def test_recording_reads_only_cached_state_and_never_renews_the_plan():
    owner, snapshot, _, _ = paired_recording()
    def forbidden(*args, **kwargs):
        raise AssertionError('recording must not call control, sensor or motor getters')
    owner._fresh_depth_linear_snapshot = forbidden
    owner._should_hard_stop_now = forbidden
    owner._get_obstacle_status = forbidden
    owner._action_runtime = SimpleNamespace(get_steering_feedback=forbidden)
    state = owner._short_follow.snapshot()
    proof = owner._validated_visual_observation
    for now in (100.10, 100.20, 100.40):
        FollowRecordingView.from_snapshot(snapshot, recording_authority(owner), now)
    assert owner._short_follow.snapshot() is state
    assert owner._validated_visual_observation is proof


def test_request_snapshot_and_current_plan_have_distinct_sequence_fields():
    owner, snapshot, _, _ = paired_recording()
    previous_sequence = snapshot.paired_sequence
    current = owner._short_follow.update(ShortFollowObservation(
        uid=1, capture_id=124, capture_timestamp=100.10, depth_timestamp=100.12,
        distance_m=1.7, center_x_ratio=.3,
    ), now=100.15)
    view = FollowRecordingView.from_snapshot(snapshot, recording_authority(owner), 100.16)
    assert view.paired_sequence == previous_sequence
    assert view.authority_sequence == current.sequence != previous_sequence
    assert view.authority_capture_frame_id == 124
    assert view.authorized_left_rpm == current.left_rpm
    assert view.authorized_right_rpm == current.right_rpm
    # PI terms belong to the original observation snapshot, not the newer
    # authorized pair. Recording must not join two different decisions.
    assert view.paired_pi_request_rpm == snapshot.paired_pi_request_rpm
    assert view.paired_p_rpm == snapshot.paired_p_rpm


def test_new_paired_csv_fields_are_appended_and_legacy_columns_remain(tmp_path):
    owner, snapshot, _, _ = paired_recording()
    columns = FollowRecordingView.csv_columns()
    assert columns.index('follow_control_mode') > columns.index('follow_authority_status')
    assert columns.index('follow_paired_pi_request_rpm') > columns.index('follow_authority_capture_frame_id')
    assert columns.index('follow_paired_motion_kind') > columns.index('follow_paired_limit_reason')
    assert columns.index('follow_authority_motion_kind') > columns.index('follow_paired_motion_kind')
    recorder = AsyncVideoRecorder(VideoRecorderConfig(str(tmp_path/'paired.avi'), 30,
                                   overlay_wait_sec=0), cv2_module=cv2)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    assert recorder.submit(image, capture_frame_id=125, monotonic_sec=100.1,
                           follow_snapshot=snapshot, linear_timing=recording_authority(owner))
    assert recorder.close(timeout_sec=3) and recorder.error is None
    with Path(recorder.index_path).open() as handle:
        row = next(csv.DictReader(handle))
    assert row['follow_control_mode'] == 'paired'
    assert row['follow_pid_rpm'] == ''
    assert row['follow_target_speed_m_s'] == ''
    assert row['follow_paired_left_rpm'] != ''
    assert row['follow_authorized_left_rpm'] != ''
    assert float(row['follow_paired_pi_request_rpm']) == pytest.approx(snapshot.paired_pi_request_rpm, abs=.0001)
    assert float(row['follow_paired_p_rpm']) == pytest.approx(snapshot.paired_p_rpm, abs=.0001)
    assert row['follow_paired_limit_reason'] == snapshot.paired_limit_reason
    assert row['follow_paired_motion_kind'] == row['follow_authority_motion_kind'] == 'steer_right'
    assert not image.any()


def test_main_recording_publisher_passes_short_snapshot_without_starting_hardware():
    import request_0513_modular as runtime
    owner, _, frame, decision = paired_recording()
    owner._camera_video_recorder = object()
    runtime.PersonTracker._publish_follow_recording(owner, frame, decision, 'depth30')
    assert owner._video_follow_snapshot.control_mode == 'paired'
    assert owner._video_follow_snapshot.pid_rpm is None
    assert owner._video_follow_snapshot.paired_sequence == owner._short_follow.snapshot().plan.sequence


def test_first_paired_authority_does_not_relabel_an_old_180rpm_pid_snapshot_as_current():
    owner, _, frame, decision = paired_recording()
    old_legacy = build_follow_snapshot(owner._follow_controller, frame, decision, 'vision', 100.05)
    assert old_legacy.pid_rpm == 180.
    view = FollowRecordingView.from_snapshot(old_legacy, recording_authority(owner), 100.1)
    assert view.control_mode == 'paired'
    assert view.pid_rpm is view.matching_base_rpm is view.target_speed_m_s is None
    assert 'PAIR REQ' in view.labels()[0]
    assert all('PID ' not in label for label in view.labels())
    assert view.paired_pi_request_rpm is view.paired_p_rpm is view.paired_i_rpm is None


def test_future_paired_snapshot_never_shows_requests_from_after_recording_time():
    owner, snapshot, _, _ = paired_recording()
    view = FollowRecordingView.from_snapshot(snapshot, recording_authority(owner), 100.04)
    assert view.status == 'future' and view.control_mode == 'paired'
    assert view.paired_left_rpm is view.authorized_left_rpm is None
    assert 'PAIR REQ' in view.labels()[0]
    assert view.paired_pi_request_rpm is view.paired_p_rpm is view.paired_i_rpm is None
