"""PI metrics retain physical-sample denominators and mode semantics."""
import pytest

from tools.follow_metrics import analyze, comparison
from test_follow_metrics import run


def log(run, *rows):
    (run / "request_0513_modular.log").write_text("\n".join(rows), encoding="utf-8")


def test_explicit_pi_startup_is_not_mislabeled_optional_and_matching_is_diagnostic(run):
    log(run,
        "2026-09-16 11:59:59,000 - distance_control_mode mode=distance_pi kp_per_sec=1.0 ki_per_sec2=.4 integral_max_m_s=.8 memory_ms=350 depth_ttl_ms=180",
        "2026-09-16 12:00:00,010 - distance_pi uid=1 sample_ts=20 status=tracking i_rpm=12 integral_m_s=.163 integral_frozen=False sample_dt_sec=.05 brake_source=raw_relative_motion closure_source=raw_depth_window",
        "2026-09-16 12:00:00,011 - distance_pid actual=1.8m target=1.5m output=32rpm tracking_base=0rpm uid=1 sample_ts=20 matching_source=none",
        "2026-09-16 13:00:00,000 - distance_control_mode matching_enabled=True",
    )
    result = analyze(run)
    control = result["control"]
    assert control["distance_control_mode"] == "distance_pi"
    assert control["no_matching_base_percent"] == 100
    assert control["no_matching_base_interpretation"] == "diagnostic_only_by_design"
    assert control["distance_pi"]["config"]["memory_ms"] == 350
    assert any("intentionally has no matching-speed base" in text for text in result["limitations"])


def test_pi_integral_and_statuses_are_distinct_uid_physical_samples(run):
    row = "distance_pi uid=1 sample_ts=20 status=tracking i_rpm=10 integral_m_s=.13 integral_frozen=False sample_dt_sec=.05 brake_source=raw_relative_motion closure_source=raw_depth_window"
    log(run,
        "2026-09-16 12:00:00,010 - " + row,
        "2026-09-16 12:00:00,011 - " + row,
        "2026-09-16 12:00:00,020 - distance_pi uid=1 sample_ts=21 status=recovering i_rpm=20 integral_m_s=.27 integral_frozen=True sample_dt_sec=0 brake_source=stationary_fallback closure_source=encoder_fallback",
        "2026-09-16 12:00:00,030 - distance_pi uid=2 sample_ts=21 status=tracking i_rpm=0 integral_m_s=0 integral_frozen=False sample_dt_sec=.05 brake_source=raw_relative_motion closure_source=raw_depth_window",
        "2026-09-16 13:00:00,000 - " + row.replace("sample_ts=20", "sample_ts=22"),
    )
    control = analyze(run)["control"]
    pi = control["distance_pi"]
    assert control["distance_control_mode"] == "distance_pi"
    assert pi["distinct_samples"] == 3
    assert pi["integral_rpm"]["mean"] == 10
    assert pi["integral_m_s"]["mean"] == pytest.approx(.4 / 3)
    assert pi["integral_frozen_percent"] == pytest.approx(100 / 3)
    assert pi["statuses"] == {"tracking": 2, "recovering": 1}
    assert pi["brake_sources"] == {"raw_relative_motion": 2, "stationary_fallback": 1}


def test_motion_memory_is_distinct_samples_and_never_a_fresh_origin(run):
    row = 'distance_pi uid=1 sample_ts=20 brake_source=relative_motion_memory motion_origin_ts=19.9 motion_uncertainty_m_s=.2'
    log(run,
        '2026-09-16 12:00:00,010 - '+row,
        '2026-09-16 12:00:00,011 - '+row,
        '2026-09-16 12:00:00,020 - '+row.replace('sample_ts=20', 'sample_ts=20.1').replace('m_s=.2', 'm_s=.4'),
        '2026-09-16 12:00:00,030 - distance_pi uid=1 sample_ts=20.2 brake_source=motion_unknown_bound',
    )
    pi = analyze(run)['control']['distance_pi']
    assert pi['motion_memory_samples'] == 2
    assert pi['motion_memory_distinct_origins'] == 1
    assert pi['motion_memory_uncertainty_m_s']['mean'] == pytest.approx(.3)
    assert pi['brake_sources'] == {'relative_motion_memory': 2, 'motion_unknown_bound': 1}


def test_pause_memory_and_rejections_distinguish_records_from_samples(run):
    log(run,
        "2026-09-16 12:00:00,010 - distance_pi_pause uid=1 sample_ts=20 reason=no_depth memory_retained=True old_lease_may_be_live=False",
        "2026-09-16 12:00:00,011 - distance_pi_pause uid=1 sample_ts=20 reason=no_depth memory_retained=True old_lease_may_be_live=False",
        "2026-09-16 12:00:00,020 - distance_pi_pause uid=1 sample_ts=20 reason=long_gap memory_retained=False old_lease_may_be_live=False",
        "2026-09-16 12:00:00,030 - distance_pi_admission_rejected sample_ts=21 reason=depth_write_budget",
        "2026-09-16 12:00:00,031 - distance_pi_admission_rejected sample_ts=21 reason=depth_write_budget",
    )
    pi = analyze(run)["control"]["distance_pi"]
    assert pi["pause_records"] == 3
    assert pi["pause_memory_retained_records"] == 2
    assert pi["pause_memory_cleared_records"] == 1
    assert pi["pause_distinct_origins"] == 1
    assert pi["admission_rejected_records"] == 2
    assert pi["admission_rejected_distinct_samples"] == 1
    assert pi["admission_reject_reasons"] == {"depth_write_budget": 2}


def test_final_pi_losses_use_rpm_quantum_and_deduplicate_max_sample_loss(run):
    log(run,
        "2026-09-16 12:00:00,010 - distance_pi uid=1 sample_ts=20 status=tracking i_rpm=1",
        "2026-09-16 12:00:00,020 - distance_pi uid=1 sample_ts=21 status=tracking i_rpm=2",
        "2026-09-16 12:00:00,030 - depth_linear_limit uid=1 sample_ts=20 approved_forward_rpm=20 pid_to_approved_loss_rpm=1 forward_scale_rpm=200",
        "2026-09-16 12:00:00,031 - depth_linear_limit uid=1 sample_ts=20 approved_forward_rpm=20 pid_to_approved_loss_rpm=1 forward_scale_rpm=200",
        "2026-09-16 12:00:00,040 - depth_linear_limit uid=1 sample_ts=21 approved_forward_rpm=40 pid_to_approved_loss_rpm=1 forward_scale_rpm=200",
        "2026-09-16 12:00:00,050 - depth_linear_limit uid=1 sample_ts=21 approved_forward_rpm=24 pid_to_approved_loss_rpm=17 forward_scale_rpm=200",
        "2026-09-16 12:00:00,060 - depth_linear_limit uid=2 sample_ts=21 approved_forward_rpm=0 pid_to_approved_loss_rpm=40 forward_scale_rpm=200",
    )
    pi = analyze(run)["control"]["distance_pi"]
    assert pi["quantization_step_rpm"] == 2
    assert pi["positive_final_loss_rpm"]["count"] == 2
    assert pi["positive_final_loss_rpm"]["max"] == 17
    assert pi["quantization_compatible_limited_samples"] == 1
    assert pi["material_limited_samples"] == 1
    assert pi["integral_m_s"]["count"] == 0  # No invented wheel circumference.


def test_pi_without_logged_output_scale_does_not_guess_quantization(run):
    log(run,
        "2026-09-16 12:00:00,010 - distance_pi uid=1 sample_ts=20 status=tracking i_rpm=1",
        "2026-09-16 12:00:00,020 - depth_linear_limit uid=1 sample_ts=20 approved_forward_rpm=20 pid_to_approved_loss_rpm=1",
    )
    pi = analyze(run)["control"]["distance_pi"]
    assert pi["quantization_step_rpm"] is None
    assert pi["quantization_compatible_limited_samples"] is None
    assert pi["material_limited_samples"] is None


def test_explicit_pi_limit_separates_quantization_from_real_integral_reduction(run):
    log(run,
        "2026-09-16 12:00:00,010 - distance_pi_limit uid=1 sample_ts=20 requested_rpm=21 approved_rpm=20 integral_before_m_s=.1 integral_after_m_s=.1 quantization_rpm=2",
        "2026-09-16 12:00:00,020 - distance_pi_limit uid=1 sample_ts=21 requested_rpm=35 approved_rpm=34 integral_before_m_s=.15 integral_after_m_s=.15 quantization_rpm=2",
        "2026-09-16 12:00:00,030 - distance_pi_limit uid=1 sample_ts=21 requested_rpm=35 approved_rpm=32 integral_before_m_s=.15 integral_after_m_s=.12 quantization_rpm=2",
        "2026-09-16 12:00:00,031 - distance_pi_limit uid=1 sample_ts=21 requested_rpm=35 approved_rpm=32 integral_before_m_s=.15 integral_after_m_s=.12 quantization_rpm=2",
    )
    pi = analyze(run)["control"]["distance_pi"]
    assert pi["approval_limit_records"] == 4
    assert pi["approval_limited_distinct_samples"] == 2
    assert pi["approval_limit_classes"] == {"quantization_compatible": 1, "material": 1}
    assert pi["approval_integral_reduced_distinct_samples"] == 1
    assert pi["approval_integral_reduction_m_s"]["max"] == pytest.approx(.03)
    assert pi["post_limit_integral_m_s"]["mean"] == pytest.approx(.11)


def test_comparing_pi_with_legacy_suppresses_matching_availability_grade(run):
    legacy = analyze(run)
    log(run,
        "2026-09-16 11:59:59,000 - distance_control_mode mode=distance_pi",
        "2026-09-16 12:00:00,010 - distance_pid actual=1.8m target=1.5m output=32rpm tracking_base=0rpm uid=1 sample_ts=20 matching_source=none",
    )
    pi = analyze(run)
    metric = comparison(pi, legacy)["metrics"]["control.no_matching_base_percent"]
    assert metric["current"] == 100
    assert metric["delta"] is None
    assert metric["diagnostic_only"] is True
    assert legacy["control"]["no_matching_base_interpretation"] == "matching_availability"
    assert legacy["control"]["distance_pi"]["distinct_samples"] == 0


def test_cache_and_authority_diagnostics_are_windowed_and_deduplicated(run):
    cache = "depth_cache_retained uid=1 anchor_ts=20 reason=depth_detector_bbox_stale"
    suspended = "distance_pi_authority_suspended uid=1 sample_ts=21 reason=no_live_grant_before_pi"
    log(run,
        "2026-09-16 12:00:00,010 - " + cache,
        "2026-09-16 12:00:00,011 - " + cache,
        "2026-09-16 12:00:00,020 - " + suspended,
        "2026-09-16 12:00:00,021 - " + suspended,
        "2026-09-16 12:00:00,022 - distance_pi uid=1 sample_ts=22 status=tracking_gap_no_integral sample_dt_sec=0",
        "2026-09-16 13:00:00,000 - " + cache.replace("anchor_ts=20", "anchor_ts=30"),
        "2026-09-16 13:00:00,001 - " + suspended.replace("sample_ts=21", "sample_ts=31"),
    )
    result = analyze(run)
    assert result["depth"]["cache_retained_records_by_reason"] == {"depth_detector_bbox_stale": 2}
    assert result["depth"]["cache_retained_distinct_anchors"] == 1
    pi = result["control"]["distance_pi"]
    assert pi["authority_suspend_records_by_reason"] == {"no_live_grant_before_pi": 2}
    assert pi["authority_suspend_distinct_origin_reasons"] == 1
    assert pi["statuses"] == {"tracking_gap_no_integral": 1}


def test_launch_diagnostics_distinguish_demand_braking_feedback_and_old_logs(run):
    row = ("distance_pi uid=1 sample_ts=20 status=tracking launch_floor_rpm=120 "
           "total_demand_rpm=120 envelope_rpm=40 ego_rpm=18 feedback_age_ms=25 "
           "software_rise_bypassed=True demand_limit_reason=braking_envelope")
    log(run,
        "2026-09-16 12:00:00,010 - " + row,
        "2026-09-16 12:00:00,011 - " + row,
        "2026-09-16 12:00:00,020 - distance_pi uid=1 sample_ts=21 status=tracking launch_floor_rpm=0 software_rise_bypassed=False",
        "2026-09-16 12:00:00,030 - distance_pi uid=1 sample_ts=22 status=tracking",
        "2026-09-16 13:00:00,010 - " + row.replace("sample_ts=20", "sample_ts=25"),
    )
    pi = analyze(run)["control"]["distance_pi"]
    assert pi["launch_audited_samples"] == 2
    assert pi["launch_requested_samples"] == pi["launch_software_rise_bypassed_samples"] == 1
    assert pi["launch_demand_limit_reasons"] == {"braking_envelope": 1}
    assert pi["launch_total_demand_rpm"]["mean"] == 120
    assert pi["launch_brake_cap_rpm"]["mean"] == 40
    assert pi["launch_feedback_rpm"]["mean"] == 18
    assert pi["launch_feedback_age_ms"]["mean"] == 25
