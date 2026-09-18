"""Stable denominators, clock window and units for cross-run metrics."""
import csv
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from tools.follow_metrics import analyze, comparison, distribution, short_speed_drops


def test_approach_metrics_deduplicate_and_leave_old_logs_empty(run):
    assert analyze(run,tail_sec=0)['control']['approach_profile']['distinct_samples']==0
    path=run/'request_0513_modular.log'
    row='longitudinal_approach uid=1 sample_ts=10 mode=catchup correction_rpm=34.5 request_rpm=74 braking_distance_m=0.12'
    path.write_text(path.read_text()+'\n'+'\n'.join([
        '2026-09-16 12:00:00,010 - '+row,
        '2026-09-16 12:00:00,020 - '+row,
        '2026-09-16 12:00:00,030 - longitudinal_approach uid=1 sample_ts=11 mode=decelerating correction_rpm=5 request_rpm=40 braking_distance_m=0.35',
        '2026-09-16 13:00:00,000 - '+row.replace('sample_ts=10','sample_ts=12'),
    ]))
    a=analyze(run,tail_sec=0)['control']['approach_profile']
    assert a['distinct_samples']==2
    assert a['modes']=={'catchup':1,'decelerating':1}
    assert a['correction_rpm']['max']==34.5
    assert a['braking_distance_m']['max']==.35


def test_far_fresh_low_rpm_is_deduplicated_and_does_not_count_near_or_stale(run):
    path=run/'request_0513_modular.log'
    path.write_text(path.read_text()+'\n'+'\n'.join([
        '2026-09-16 12:00:00,010 - longitudinal_motion target=1 sample_ts=10 status=warming_up',
        '2026-09-16 12:00:00,020 - depth_linear_limit uid=1 sample_ts=10 fresh=True distance=2.87 approved_forward_rpm=24',
        '2026-09-16 12:00:00,021 - depth_linear_limit uid=1 sample_ts=10 fresh=True distance=2.87 approved_forward_rpm=24',
        '2026-09-16 12:00:00,030 - depth_linear_limit uid=1 sample_ts=11 fresh=False distance=2.87 approved_forward_rpm=0',
        '2026-09-16 12:00:00,040 - depth_linear_limit uid=1 sample_ts=12 fresh=True distance=1.5 approved_forward_rpm=0',
        '2026-09-16 12:00:00,050 - depth_far_closing_recovery uid=1 sample_ts=13',
        '2026-09-16 12:00:00,051 - depth_far_closing_recovery uid=1 sample_ts=13',
    ]))
    c=analyze(run,tail_sec=0)['control']
    assert c['far_fresh_below_30rpm_distinct_samples']==1
    assert c['far_fresh_below_30rpm_motion_status']=={'warming_up':1}
    assert c['far_closing_measured_ramp_distinct_samples']==1


def test_estimator_status_matches_rounded_timestamp_but_keeps_distinct_depth_count(run):
    path=run/'request_0513_modular.log'
    path.write_text(path.read_text()+'\n'+'\n'.join([
        '2026-09-16 12:00:00,010 - longitudinal_motion target=1 sample_ts=10.123457 status=warming_up',
        '2026-09-16 12:00:00,020 - depth_linear_limit uid=1 sample_ts=10.123456789 fresh=True distance=2.87 approved_forward_rpm=24',
        '2026-09-16 12:00:00,030 - longitudinal_motion_reset uid=1 sample_ts=10.223456789 reason=yaw_limit',
        '2026-09-16 12:00:00,040 - depth_linear_limit uid=1 sample_ts=10.223456789 fresh=True distance=2.87 approved_forward_rpm=24',
    ]))
    c=analyze(run,tail_sec=0)['control']
    assert c['far_fresh_below_30rpm_distinct_samples']==2
    assert c['far_fresh_below_30rpm_motion_status']=={'warming_up':1,'reset:yaw_limit':1}


def test_detached_and_near_stop_metrics_do_not_count_repeats_or_assume_success(run):
    path=run/'request_0513_modular.log'
    path.write_text(path.read_text()+'\n'+'\n'.join([
        '2026-09-16 12:00:00,010 - depth_ff_detached uid=1 sample_ts=10 approved_percent=20',
        '2026-09-16 12:00:00,011 - depth_ff_detached uid=1 sample_ts=10 approved_percent=20',
        '2026-09-16 12:00:00,020 - depth_ff_detached uid=1 sample_ts=11 approved_percent=0',
        '2026-09-16 12:00:00,030 - near_no_matching_stop uid=1 sample_ts=12',
        '2026-09-16 12:00:00,040 - near_no_matching_stop uid=1 sample_ts=12',
    ]))
    c=analyze(run,tail_sec=0)['control']
    assert c['detached_prior_distinct_samples']==2
    assert c['detached_prior_positive_recovery_samples']==1
    assert c['near_no_matching_stop_distinct_samples']==1


def test_feedback_audit_deduplicates_physical_samples_and_reads_startup_config(run):
    path=run/'request_0513_modular.log'
    path.write_text(path.read_text()+'\n'+'\n'.join([
        '2026-09-16 11:59:59,000 - longitudinal_feedback_config kp_rpm_per_m=27 ego_limit_rpm=105 yaw_uncertainty_max_m_s=0.04 disagreement_skew_ms=50 depth_ttl_ms=180',
        '2026-09-16 12:00:00,010 - longitudinal_feedback_audit uid=1 sample_ts=10 compensation=yaw_disagreement_bounded',
        '2026-09-16 12:00:00,020 - longitudinal_feedback_audit uid=1 sample_ts=10 compensation=yaw_disagreement_bounded',
        '2026-09-16 12:00:00,030 - longitudinal_feedback_audit uid=1 sample_ts=10.02 compensation=yaw_disagreement_rejected',
        '2026-09-16 12:00:00,040 - longitudinal_motion target=1 status=ego_speed_out_of_bounds',
    ]))
    c=analyze(run,tail_sec=0)['control']
    assert c['feedback_audited_distinct_samples']==2
    assert c['feedback_compensation_distinct_samples']=={'yaw_disagreement_bounded':1,'yaw_disagreement_rejected':1}
    assert c['estimator_status_records']=={'ego_speed_out_of_bounds':1}
    assert c['feedback_config']['kp_rpm_per_m']==27
    assert c['feedback_config']['ego_limit_rpm']==105


def test_old_logs_do_not_imply_yaw_envelope_was_tested(run):
    c=analyze(run,tail_sec=0)['control']
    assert c['feedback_config']=={}
    assert c['feedback_compensation_distinct_samples']=={}


def test_scheduling_recovery_is_counted_and_losses_are_not_hidden(run):
    path=run/'request_0513_modular.log'
    path.write_text(path.read_text()+'\n'+'\n'.join([
        '2026-09-16 12:00:00,010 - depth_scheduling_recovery uid=1 sample_ts=10 event=start cap_rpm=52',
        '2026-09-16 12:00:00,020 - depth_scheduling_recovery uid=1 sample_ts=10 event=start cap_rpm=52',
        '2026-09-16 12:00:00,030 - depth_speed_recovery_limit uid=1 sample_ts=10 lost_rpm=18 recovery_policy=scheduling',
        '2026-09-16 12:00:00,040 - depth_scheduling_recovery uid=1 sample_ts=10.1 event=advance cap_rpm=55',
        '2026-09-16 12:00:00,050 - depth_scheduling_recovery_rejected uid=1 reason=near_or_distance_change',
    ]))
    c=analyze(run,tail_sec=0)['control']
    assert c['scheduling_recovery_starts']==1
    assert c['scheduling_recovery_distinct_samples']==2
    assert c['scheduling_recovery_rejections']=={'near_or_distance_change':1}
    assert c['recovery_lost_rpm_distinct_samples']['max']==18
    assert c['all_recovery_starts_per_10sec'] > c['recovery_starts_per_10sec']


def test_bias_metrics_are_requested_not_claimed_as_actual_motor_increase(run):
    path=run/'request_0513_modular.log'
    path.write_text(path.read_text()+'\n'+'\n'.join([
        '2026-09-16 12:00:00,010 - longitudinal_bias_trial uid=1 sample_ts=10 applied_bias_rpm=5',
        '2026-09-16 12:00:00,020 - longitudinal_bias_trial uid=1 sample_ts=10 applied_bias_rpm=5',
        '2026-09-16 12:00:00,030 - longitudinal_bias_trial uid=1 sample_ts=10.1 applied_bias_rpm=0',
        '2026-09-16 12:00:00,040 - longitudinal_motion target=1 status=ready decline_policy=stop_suspect_bounded',
    ]))
    c=analyze(run,tail_sec=0)['control']
    assert c['bias_trial_audited_distinct_samples']==2
    assert c['bias_trial_positive_distinct_samples']==1
    assert c['bias_trial_max_requested_rpm_distinct_samples']['mean']==2.5
    assert c['stop_suspect_bounded_records']==1


def test_recovery_pause_completion_and_initial_approval_are_auditable(run):
    path = run/'request_0513_modular.log'
    path.write_text(path.read_text()+'\n'+'\n'.join([
        '2026-09-16 12:00:00,010 - depth_scheduling_recovery uid=1 sample_ts=10 event=start previous_approved_rpm=24 cap_rpm=24 ramp_completed=False',
        '2026-09-16 12:00:00,020 - depth_scheduling_recovery_gap uid=1 original_sample_ts=10 action=pause',
        '2026-09-16 12:00:00,021 - depth_scheduling_recovery_gap uid=1 original_sample_ts=10 action=pause',
        '2026-09-16 12:00:00,030 - depth_scheduling_recovery uid=1 sample_ts=10.12 event=advance previous_approved_rpm=24 cap_rpm=48 ramp_completed=False',
        '2026-09-16 12:00:00,040 - depth_scheduling_recovery uid=1 sample_ts=10.48 event=advance previous_approved_rpm=24 cap_rpm=60 ramp_completed=True',
        '2026-09-16 12:00:00,041 - depth_scheduling_recovery uid=1 sample_ts=10.48 event=advance previous_approved_rpm=24 cap_rpm=60 ramp_completed=True',
        '2026-09-16 12:00:00,050 - depth_scheduling_recovery_gap uid=1 action=discard reason=unsafe_or_reference_expired',
    ]))
    c = analyze(run, tail_sec=0)['control']
    assert c['scheduling_recovery_starts'] == 1
    assert c['scheduling_recovery_paused_distinct_samples'] == 1
    assert c['scheduling_recovery_completed_distinct_samples'] == 1
    assert c['scheduling_recovery_above_initial_approval_samples'] == 2
    assert c['scheduling_recovery_discard_reasons'] == {'unsafe_or_reference_expired': 1}


def test_short_speed_drops_deduplicate_samples_keep_uid_and_classify_zero():
    records=[(0,10,'1',60,'fresh'), (.05,10,'1',20,'held'),
             (.07,9,'1',0,'none'), (.1,10.1,'1',45,'bridge'),
             (.12,10.2,'2',10,'fresh'), (.18,10.3,'1',0,'none'),
             (.5,10.5,'2',0,'none')]
    result=short_speed_drops(records,2)
    assert result['count']==2
    assert result['per_10sec']==10
    assert result['transitions']=={'fresh->bridge':1,'zero':1}
    assert result['drop_rpm']['max']==45


def test_new_matching_and_limit_metrics_with_old_log_unknowns(run):
    path=run/'request_0513_modular.log'
    path.write_text(path.read_text()+'\n'+'\n'.join([
        '2026-09-16 12:00:00,020 - longitudinal_motion target=1 status=ready_capped unbounded_target_rpm=90 matching_cap_rpm=80 matching_rate_limited=True',
        '2026-09-16 12:00:00,120 - longitudinal_motion target=1 status=ready unbounded_target_rpm=70 matching_cap_rpm=80 matching_rate_limited=False',
        '2026-09-16 12:00:00,130 - depth_linear_limit uid=1 sample_ts=11 fresh=True approved_forward_rpm=60 pid_to_approved_loss_rpm=12',
        '2026-09-16 12:00:00,140 - depth_linear_limit uid=1 sample_ts=11 fresh=False approved_forward_rpm=58 pid_to_approved_loss_rpm=14',
        '2026-09-16 12:00:00,160 - depth_linear_limit uid=1 sample_ts=12 fresh=True approved_forward_rpm=45 pid_to_approved_loss_rpm=None',
        '2026-09-16 12:00:00,170 - stale_vision_depth_audit remaining_ms=38',
        '2026-09-16 12:00:00,180 - stale_vision_depth_audit remaining_ms=0',
    ]))
    c=analyze(run,tail_sec=0)['control']
    assert c['matching_fresh_records']==2
    assert c['matching_capped_percent']==50
    assert c['matching_rate_limited_records']==1
    assert c['matching_caps_rpm']['mean']==80
    assert c['pid_to_approved_loss_rpm_distinct_samples']['count']==1
    assert c['pid_to_approved_loss_rpm_distinct_samples']['mean']==14
    assert c['approved_short_speed_drops']['count']==1
    assert c['stale_vision_live_depth_revocations']==1


@pytest.fixture
def run(tmp_path):
    epoch = datetime(2026, 9, 16, 12, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
    fields = ["capture_frame_id", "control_frame_id", "capture_unix_sec", "selected_target_id",
              "target_distance_m", "distance_detail"]
    with (tmp_path / "camera_raw.frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cap, ctrl, t, uid, d in [(1, 1, 0, 1, 1.5), (2, "", .1, "", ""),
                                   (3, 2, .2, 1, ""), (4, 2, .3, 1, ""),
                                   (5, 3, .4, 1, 1.9)]:
            writer.writerow(dict(zip(fields, [cap, ctrl, epoch + t, uid, d, "test"])))
    (tmp_path / "request_0513_modular.log").write_text("\n".join([
        "2026-09-16 11:59:59,900 - LZ30EMA 电机命令: 标签=DRIVE 左轮=30转/分 右轮=-30转/分",
        "2026-09-16 12:00:00,000 - distance_pid actual=1.500m target=1.500m output=30rpm tracking_base=30.00rpm",
        "2026-09-16 12:00:00,010 - Astra depth timeline: sample_ts=10 temporal=new_sample raw=1.5 distance=1.5",
        "2026-09-16 12:00:00,020 - Astra depth timeline: sample_ts=10 temporal=new_sample raw=1.5 distance=1.5",
        "2026-09-16 12:00:00,030 - Astra depth timeline: sample_ts=10.02 temporal=new_sample raw=None distance=1.5",
        "2026-09-16 12:00:00,200 - LZ30EMA 电机命令: 标签=DRIVE 左轮=0转/分 右轮=0转/分",
        "2026-09-16 12:00:00,210 - Astra depth timeline: sample_ts=10.2 temporal=new_sample raw=1.9 distance=1.9",
        "2026-09-16 12:00:00,220 - distance_pid actual=1.900m target=1.500m output=40rpm tracking_base=0.00rpm",
        "2026-09-16 12:00:00,225 - longitudinal_motion_reset reason=yaw_limit sample_ts=10.2",
        "2026-09-16 12:00:00,230 - 编码器转向反馈: 前进归一转速=20.0/24.0RPM 可信=True",
        "2026-09-16 12:00:00,240 - depth_speed_recovery_started distance=1.9m",
        "2026-09-16 12:00:00,241 - depth_speed_recovery_limit requested_percent=40 approved_percent=25",
        "2026-09-16 12:00:00,250 - depth_linear_limit remaining_ms=150 approved=[('forward', 25)]",
        "2026-09-16 12:00:00,260 - longitudinal_motion_bridge reason=yaw_limit origin_ts=10",
        "2026-09-16 12:00:00,270 - depth_recovery_gap action=resume sample_ts=10.2",
        "2026-09-16 12:00:00,800 - distance_pid actual=2.900m target=1.500m output=60rpm tracking_base=40.00rpm",
        "2026-09-16 12:00:00,810 - follow_wheel_tick reason=periodic interval_ms=50.0",
        "2026-09-16 12:00:00,820 - follow_wheel_tick reason=base_reduced interval_ms=10.0",
        "2026-09-16 12:00:00,830 - longitudinal_motion target=1 compensated=True",
        "2026-09-16 12:00:00,840 - depth_measured_recovery resume_cap_rpm=42.0",
    ]), encoding="utf-8")
    return tmp_path


def test_follow20_metrics_distinguish_safety_interrupts_from_regular_ticks(run):
    result = analyze(run, tail_sec=.7)
    assert result["execution"]["follow_tick_reasons"] == {"periodic":1, "base_reduced":1}
    assert result["execution"]["follow_tick_interval_ms"]["mean"] == 30.
    assert result["execution"]["periodic_tick_interval_ms"]["mean"] == 50.
    assert result["control"]["rotation_compensated_records"] == 1
    assert result["control"]["measured_recovery_records"] == 1


def test_metrics_have_stable_denominators_and_do_not_count_unprocessed_csv(run):
    result = analyze(run, tail_sec=0)
    assert result["execution"]["follow_tick_reasons"] == {}
    assert result["execution"]["follow_tick_interval_ms"]["mean"] is None
    assert result["control"]["rotation_compensated_records"] == 0
    distance = result["distance"]
    assert distance["selected_control_frames"] == 3
    assert distance["missing_frames"] == distance["missing_segments"] == 1
    assert distance["missing_percent"] == pytest.approx(100/3)
    assert distance["absolute_error_m"]["mean"] == pytest.approx(.2)
    assert distance["signed_error_m"]["mean"] == pytest.approx(.2)
    assert distance["actual_distance_m"]["mean"] == pytest.approx(1.7)
    assert distance["distance_std_m"] == pytest.approx(.2)
    assert distance["within_tolerance_percent"] == 50
    assert distance["drift_m_s"] == pytest.approx(1, abs=1e-6)
    assert result["control"]["pid_records"] == 2
    assert result["control"]["no_matching_base_percent"] == 50
    assert result["control"]["integral_at_cap_percent"] is None  # old logs unknown


def test_physical_depth_dedup_and_effective_rate(run):
    depth = analyze(run, tail_sec=0)["depth"]
    assert depth["accepted_distinct_samples"] == 2
    assert depth["effective_hz"] == pytest.approx(5)
    assert depth["gaps_over_180ms"] == 1


def test_authorized_depth_rate_excludes_old_observations_and_duplicates(run):
    path=run / 'request_0513_modular.log'
    path.write_text(path.read_text()+ '\n' + '\n'.join([
        "2026-09-16 12:00:00,010 - depth_roi_extension status=extended age_ms=220",
        "2026-09-16 12:00:00,011 - depth_roi_extension status=turning age_ms=221",
        "2026-09-16 12:00:00,012 - depth_linear_limit fresh=True sample_ts=10 approved=[('forward', 20)] approved_forward_rpm=40",
        "2026-09-16 12:00:00,015 - depth_linear_limit fresh=True sample_ts=10 approved=[('forward', 20)]",
        "2026-09-16 12:00:00,110 - depth_linear_limit fresh=False sample_ts=10.1 approved=[('forward', 20)]",
        "2026-09-16 12:00:00,150 - depth_linear_limit fresh=True sample_ts=10.15 approved=[('forward', 0)]",
        "2026-09-16 12:00:00,240 - depth_linear_limit fresh=True sample_ts=10.23 approved=[('forward', 30)]",
    ]))
    d=analyze(run)['depth']
    assert d['roi_extension_events']=={'extended':1,'turning':1}
    assert d['positive_authorized_samples']==2
    assert d['positive_authorized_hz']==pytest.approx(1/.23)
    assert d['positive_authorized_gaps_over_180ms']==1
    assert d['positive_authorized_gap_ms']['max']==pytest.approx(230)
    rpm=analyze(run)['control']['approved_forward_rpm']
    assert rpm['count']==1 and rpm['mean']==40


def test_motor_sign_and_inherited_command_time_weighting(run):
    execution = analyze(run, tail_sec=0)["execution"]
    assert execution["zero_command_percent"] == pytest.approx(50, abs=.001)
    assert execution["time_weighted_command_base_rpm"] == pytest.approx(15, abs=.001)
    assert execution["encoder_base_rpm_sparse_samples"]["mean"] == 22


def test_bridge_and_recovery_logs_are_independent_counters(run):
    control = analyze(run)["control"]
    assert control["motion_reset_reasons"] == {"yaw_limit": 1}
    assert control["bridge_reasons"] == {"yaw_limit": 1}
    assert control["recovery_gap_actions"] == {"resume": 1}
    assert control["recovery_starts"] == control["recovery_limited_records"] == 1


def test_unknown_motor_interval_is_not_assumed_stopped(run):
    (run / "request_0513_modular.log").write_text("", encoding="utf-8")
    result = analyze(run)
    assert result["execution"]["zero_command_percent"] is None
    assert result["control"]["no_matching_base_percent"] is None
    assert result["distance"]["target_m"] is None
    assert result["warnings"]


def test_mixed_setpoints_do_not_silently_choose_one(run):
    with (run / "request_0513_modular.log").open("a") as stream:
        stream.write("\n2026-09-16 12:00:00,300 - distance_pid actual=1.500m target=1.600m output=0rpm tracking_base=0rpm\n")
    assert analyze(run)["distance"]["target_m"] is None
    assert analyze(run, target_distance=1.6)["distance"]["target_m"] == 1.6


def test_outside_cap_range_reports_error(run):
    with pytest.raises(ValueError, match="No captured rows"):
        analyze(run, cap_start=100)


def test_comparison_identical_inputs_has_zero_deltas_and_documents_window(run):
    result = analyze(run)
    diff = comparison(result, result)
    assert all(v["delta"] == (0 if v['current'] is not None else None)
               for v in diff["metrics"].values())
    assert diff["previous_window"] == result["window"]


def test_p95_nearest_rank_and_missing_values():
    assert distribution([])["p95"] is None
    assert distribution(list(range(1, 101)))["p95"] == 95


def test_distance_parking_counters_are_separate_from_safety_holds(run):
    with (run / "request_0513_modular.log").open("a") as stream:
        stream.write("\n2026-09-16 12:00:00,300 - follow_distance_hold_started uid=1\n")
        stream.write("2026-09-16 12:00:00,310 - follow_distance_hold_observe status=wait count=1\n")
        stream.write("2026-09-16 12:00:00,320 - follow_distance_hold_observe status=duplicate count=1\n")
        stream.write("2026-09-16 12:00:00,330 - follow_distance_hold_released held_ms=120 uid=1\n")
        stream.write("2026-09-16 12:00:00,340 - safety_hold_hazard held_ms=900\n")
    control = analyze(run)["control"]
    assert control['distance_parking_events'] == dict(started=1,observe_wait=1,observe_duplicate=1,released=1)
    assert control['distance_parking_release_ms']['mean'] == 120


def test_continuity_events_and_normalized_counts(run):
    with (run / "request_0513_modular.log").open("a") as stream:
        stream.write("\n2026-09-16 12:00:00,300 - depth_ff_fallback uid=1\n")
        stream.write("2026-09-16 12:00:00,301 - depth30_replay_preserve uid=1\n")
        stream.write("2026-09-16 12:00:00,302 - depth_linear_authority snapshot=None temporal_status=duplicate previous_remaining_ms=30\n")
        stream.write("2026-09-16 12:00:00,303 - depth_linear_authority snapshot=None temporal_status=duplicate previous_remaining_ms=-1\n")
    report = analyze(run, tail_sec=0)
    events = report["control"]["continuity_events"]
    assert events == {"depth_ff_fallback": 1, "depth30_replay_preserve": 1,
                      "duplicate_zero_with_live_lease": 1}
    assert report["control"]["motion_resets_per_10sec"] == pytest.approx(25, abs=.001)
    assert report["control"]["recovery_starts_per_10sec"] == pytest.approx(25, abs=.001)


@pytest.mark.parametrize('mode', ['normal', 'emergency', 'free'])
def test_independent_stop_command_ends_forward_dwell(run, mode):
    (run / 'request_0513_modular.log').write_text('\n'.join([
        '2026-09-16 11:59:59,900 - LZ30EMA 电机命令: 标签=FOLLOW20 左轮=30转/分 右轮=-30转/分',
        f'2026-09-16 12:00:00,100 - LZ30EMA 停车命令: 标签=brake 模式={mode} 清零延时=0.000秒',
        '2026-09-16 12:00:00,200 - 进入刹车保持状态: intent only',
        '2026-09-16 12:00:00,300 - LZ30EMA 电机命令: 标签=FOLLOW20 左轮=20转/分 右轮=-20转/分',
    ]))
    result = analyze(run, tail_sec=0)
    assert result['schema_version'] == 2
    e = result['execution']
    assert e['zero_command_percent'] == pytest.approx(50, abs=.001)
    assert e['time_weighted_command_base_rpm'] == pytest.approx(12.5, abs=.001)
    assert e['direct_stop_records'] == 1


def test_stop_before_window_is_inherited_not_last_forward_command(run):
    (run / 'request_0513_modular.log').write_text('\n'.join([
        '2026-09-16 11:59:59,700 - LZ30EMA 电机命令: 左轮=30转/分 右轮=-30转/分',
        '2026-09-16 11:59:59,900 - LZ30EMA 停车命令: 标签=brake 模式=normal',
    ]))
    assert analyze(run)['execution']['zero_command_percent'] == 100


def test_recovery_loss_is_rpm_and_deduplicated_per_uid_physical_sample(run):
    with (run / 'request_0513_modular.log').open('a') as stream:
        stream.write('\n' + '\n'.join([
            '2026-09-16 12:00:00,300 - longitudinal_motion_observation_skipped reason=depth_detector_bbox_stale',
            '2026-09-16 12:00:00,310 - depth_recovery_continuity_restored uid=1 sample_ts=20',
            '2026-09-16 12:00:00,311 - depth_recovery_continuity_restored uid=1 sample_ts=20',
            '2026-09-16 12:00:00,320 - depth_speed_recovery_limit uid=1 sample_ts=20 lost_rpm=40',
            '2026-09-16 12:00:00,321 - depth_speed_recovery_limit uid=1 sample_ts=20 lost_rpm=38',
            '2026-09-16 12:00:00,330 - depth_speed_recovery_limit uid=1 sample_ts=21 lost_rpm=20',
        ]))
    c=analyze(run)['control']
    assert c['skipped_observation_reasons'] == {'depth_detector_bbox_stale':1}
    assert c['recovery_continuity_restored_samples'] == 1
    assert c['recovery_lost_rpm_distinct_samples']['count'] == 2
    assert c['recovery_lost_rpm_distinct_samples']['mean'] == 30


def test_legacy_loss_requires_explicit_startup_scale_and_counts_derivative_reset(run):
    path = run / 'request_0513_modular.log'
    original = path.read_text()
    additions = '\n' + '\n'.join([
        '2026-09-16 12:00:00,300 - depth_speed_recovery_limit uid=1 sample_ts=20 requested_percent=32 approved_percent=12',
        '2026-09-16 12:00:00,310 - longitudinal_motion_observation_skipped reason=depth_detector_bbox_stale derivative_reset=True',
    ])
    path.write_text(original + additions)
    assert analyze(run)['control']['recovery_lost_rpm_distinct_samples']['count'] == 0
    path.write_text('2026-09-16 11:59:59,500 - depth_clock_config forward_max_rpm=200\n' + original + additions)
    control = analyze(run)['control']
    assert control['recovery_lost_rpm_distinct_samples']['mean'] == 40
    assert control['skipped_observation_derivative_resets'] == 1


def test_compensated_continuity_metrics_do_not_count_repeated_ticks_as_new_samples(run):
    path=run/'request_0513_modular.log'
    path.write_text(path.read_text()+'\n'+'\n'.join([
        '2026-09-16 12:00:00,010 - longitudinal_motion target=1 status=ready_capped samples=2',
        '2026-09-16 12:00:00,020 - longitudinal_motion target=1 status=ready samples=3',
        '2026-09-16 12:00:00,030 - longitudinal_motion target=1 status=warming_up samples=1 chain_reset_reason=compensation_mode_change',
        '2026-09-16 12:00:00,040 - longitudinal_motion_bridge reason=warming_up decay_policy=bounded_80rpm_s chain_reset_reason=compensated_gap_unverified',
        '2026-09-16 12:00:00,050 - longitudinal_motion_observation_skipped uid=1 original_sample_ts=10 derivative_reset=True compensated_gap_preserved=False',
        '2026-09-16 12:00:00,060 - longitudinal_motion_observation_skipped uid=1 original_sample_ts=10 derivative_reset=True compensated_gap_preserved=False',
        '2026-09-16 12:00:00,070 - longitudinal_motion_observation_skipped uid=1 original_sample_ts=11 derivative_reset=False compensated_gap_preserved=True',
        '2026-09-16 12:00:00,080 - longitudinal_motion_observation_skipped uid=1 original_sample_ts=11 derivative_reset=False compensated_gap_preserved=True',
    ]))
    c=analyze(run,tail_sec=0)['control']
    assert c['matching_two_sample_percent']==50
    assert c['matching_warmup_capped_records']==1
    assert c['skipped_derivative_reset_distinct_origins']==1
    assert c['compensated_gap_preserved_distinct_origins']==1
    assert c['compensation_chain_reset_reasons']=={'compensation_mode_change':1,'compensated_gap_unverified':1}
    assert c['bridge_decay_policies']['bounded_80rpm_s']==1


def test_actual_wheel_veto_is_separate_from_pid_limit_and_dwell_matches_motor(run):
    p=run/'request_0513_modular.log'
    p.write_text('\n'.join([
        '2026-09-16 12:00:00,000 - LZ30EMA 电机命令: 左轮=0转/分 右轮=0转/分',
        '2026-09-16 12:00:00,001 - visible_wheel_dispatch requested_forward_rpm=(44, 44) applied_forward_rpm=(0, 0) reason=cross_wait_zero depth_fresh=True',
        '2026-09-16 12:00:00,100 - LZ30EMA 电机命令: 左轮=0转/分 右轮=0转/分',
        '2026-09-16 12:00:00,101 - visible_wheel_dispatch requested_forward_rpm=(44, 44) applied_forward_rpm=(0, 0) reason=cross_timeout_zero depth_fresh=True',
        '2026-09-16 12:00:00,200 - LZ30EMA 电机命令: 左轮=5转/分 右轮=-5转/分',
        '2026-09-16 12:00:00,201 - visible_wheel_dispatch requested_forward_rpm=(44, 44) applied_forward_rpm=(5, 5) reason=cross_aligned_resume depth_fresh=True',
        '2026-09-16 12:00:00,210 - depth_boundary_continuity uid=1 sample_ts=10',
        '2026-09-16 12:00:00,220 - depth_boundary_continuity uid=1 sample_ts=10',
    ]))
    r=analyze(run,tail_sec=0);e=r['execution']
    assert e['positive_requested_but_zero_dispatch_records']==2
    assert e['requested_to_dispatched_loss_rpm']['mean']==pytest.approx((44+44+39)/3)
    assert e['zero_command_sec_by_reason']['cross_wait_zero']==pytest.approx(.1,abs=1e-6)
    assert e['zero_command_sec_by_reason']['cross_timeout_zero']==pytest.approx(.1,abs=1e-6)
    assert r['depth']['boundary_continuity_distinct_samples']==1


def test_unmatched_motor_stop_does_not_inherit_an_old_dispatch_reason(run):
    p=run/'request_0513_modular.log'
    p.write_text('\n'.join([
        '2026-09-16 11:59:59,000 - LZ30EMA 电机命令: 左轮=0转/分 右轮=0转/分',
        '2026-09-16 11:59:59,001 - visible_wheel_dispatch requested_forward_rpm=(44, 44) applied_forward_rpm=(0, 0) reason=cross_wait_zero',
        '2026-09-16 11:59:59,900 - LZ30EMA 电机命令: 左轮=0转/分 右轮=0转/分',
    ]))
    e=analyze(run,tail_sec=0)['execution']
    assert e['zero_command_sec_by_reason']=={'unknown':pytest.approx(.4,abs=1e-6)}


def test_window_audit_legacy_unknown_and_recovery_deduplicated(run):
    p=run/'request_0513_modular.log'
    assert analyze(run)['control']['matching_window_percent'] is None
    p.write_text('\n'.join([
        '2026-09-16 12:00:00,010 - longitudinal_motion target=1 status=ready',
        '2026-09-16 12:00:00,020 - longitudinal_motion target=1 status=ready speed_window_ms=160 instant_target_speed=0.4 decline_policy=bounded_far_positive',
        '2026-09-16 12:00:00,030 - longitudinal_motion target=1 status=ready speed_window_ms=0 instant_target_speed=0.2 decline_policy=immediate_near_or_ttc',
        '2026-09-16 12:00:00,040 - longitudinal_motion target=1 status=no_forward_motion decline_policy=stop_evidence',
        '2026-09-16 12:00:00,050 - depth_measured_recovery uid=1 sample_ts=20 distance_policy=far_closing_measured',
        '2026-09-16 12:00:00,051 - depth_measured_recovery uid=1 sample_ts=20 distance_policy=far_closing_measured',
        '2026-09-16 12:00:00,052 - depth_measured_recovery uid=1 sample_ts=21 distance_policy=nonclosing',
    ]))
    c=analyze(run)['control']
    assert c['matching_window_percent']==50
    assert c['matching_window_audited_records']==2
    assert c['far_closing_recovery_distinct_samples']==1
    assert c['decline_policies']=={'bounded_far_positive':1,'immediate_near_or_ttc':1,'stop_evidence':1}


def test_replay_retention_counts_original_evidence_not_ticks(run):
    p=run/'request_0513_modular.log'
    p.write_text('\n'.join([
        '2026-09-16 12:00:00,010 - longitudinal_replay_retained uid=1 origin_ts=20 derivative_reset=False',
        '2026-09-16 12:00:00,020 - longitudinal_replay_retained uid=1 origin_ts=20 derivative_reset=False',
        '2026-09-16 12:00:00,030 - longitudinal_replay_retained uid=1 origin_ts=21 derivative_reset=True',
        '2026-09-16 12:00:00,040 - longitudinal_motion_reset reason=depth_timestamp_missing depth_detail=depth_sample_observation_discarded_fused_radar_hold_hold',
        '2026-09-16 12:00:00,050 - longitudinal_motion_reset reason=yaw_limit depth_detail=depth_multiregion',
    ]))
    c=analyze(run)['control']
    assert c['replay_retained_distinct_origins']==2
    assert c['replay_derivative_retained_distinct_origins']==1
    assert c['replay_hold_reset_records']==1


def test_raw_closing_and_no_matching_diagnostics_keep_distinct_denominators(run):
    p=run/'request_0513_modular.log'
    p.write_text('\n'.join([
        '2026-09-16 12:00:00,010 - longitudinal_approach uid=1 sample_ts=20 mode=distance_only closing_source=raw_depth_window closing_window_ms=100 no_matching_reason=warming_up',
        '2026-09-16 12:00:00,011 - longitudinal_approach uid=1 sample_ts=20 mode=distance_only closing_source=raw_depth_window closing_window_ms=100 no_matching_reason=warming_up',
        '2026-09-16 12:00:00,012 - distance_pid actual=3.0m output=+60rpm tracking_base=0.00rpm uid=1 sample_ts=20 matching_source=none',
        '2026-09-16 12:00:00,020 - longitudinal_motion target=1 sample_ts=21 status=warming_up',
        '2026-09-16 12:00:00,021 - distance_pid actual=3.0m output=+44rpm tracking_base=0.00rpm uid=1 sample_ts=21 matching_source=none',
        '2026-09-16 12:00:00,030 - depth_recovery_continuity_restored uid=1 sample_ts=20 normal_live_continuation=True',
        '2026-09-16 12:00:00,031 - depth_recovery_continuity_restored uid=1 sample_ts=20 normal_live_continuation=True',
    ]))
    c=analyze(run)['control']
    assert c['no_matching_reason_records']=={'warming_up':2}
    assert c['no_matching_base_percent']==100 # 60RPM fallback isn't a human estimate.
    assert c['normal_live_continuation_samples']==1
    assert c['approach_profile']['closing_sources']=={'raw_depth_window':1}
    assert c['approach_profile']['closing_window_ms']['mean']==100


def test_shared_motion_metrics_deduplicate_and_distinguish_zero_from_unknown(run):
    p=run/'request_0513_modular.log'
    assert analyze(run)['control']['shared_motion_window']['distinct_samples']==0
    p.write_text('\n'.join([
        '2026-09-16 12:00:00,010 - longitudinal_shared_window uid=1 sample_ts=20 status=warming_up span_ms=0 range_rate=None window_target_speed=None reset_reason=shared_physical_gap',
        '2026-09-16 12:00:00,011 - longitudinal_shared_window uid=1 sample_ts=20 status=warming_up span_ms=0 range_rate=None window_target_speed=None reset_reason=shared_physical_gap',
        '2026-09-16 12:00:00,020 - longitudinal_shared_window uid=1 sample_ts=21 status=no_forward_motion span_ms=100 range_rate=-0.3 window_target_speed=0.0 reset_reason=None',
    ]))
    c=analyze(run)['control']['shared_motion_window']
    assert c['distinct_samples']==2
    assert c['both_rates_valid_samples']==1
    assert c['statuses']=={'warming_up':1,'no_forward_motion':1}
    assert c['reset_reasons']=={'shared_physical_gap':1}


def test_independent_distance_mode_is_not_mislabeled_estimator_failure(run):
    p=run/'request_0513_modular.log'
    p.write_text('\n'.join([
        '2026-09-16 11:59:59,000 - distance_control_mode matching_enabled=False closure_independent=True depth_ttl_ms=180',
        '2026-09-16 12:00:00,010 - longitudinal_approach uid=1 sample_ts=20 mode=distance_only closing_source=raw_depth_window no_matching_reason=trial_disabled',
        '2026-09-16 12:00:00,011 - distance_pid actual=3.0m output=+60rpm tracking_base=0.00rpm uid=1 sample_ts=20 matching_source=none',
        '2026-09-16 12:00:00,012 - distance_brake_fallback uid=1 sample_ts=21 bound_rpm=107 source=encoder_age_bound',
        '2026-09-16 12:00:00,013 - distance_brake_fallback uid=1 sample_ts=21 bound_rpm=107 source=encoder_age_bound',
    ]))
    c=analyze(run)['control']
    assert c['distance_control_mode']=='distance_only'
    assert c['no_matching_reason_records']=={'trial_disabled':1}
    assert c['no_matching_closing_sources']=={'raw_depth_window':1}
    assert c['stale_encoder_bound_rpm']['count']==1
    assert c['stale_encoder_bound_rpm']['mean']==107
