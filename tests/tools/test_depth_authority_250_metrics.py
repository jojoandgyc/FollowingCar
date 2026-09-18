"""Count short coverage opportunities without claiming they are real motor motion."""
from tools.follow_metrics import analyze, comparison
from test_follow_metrics import run


def test_depth_and_authorization_gaps_split_180_250_without_duplicate_inflation(run):
    rows = []
    for index, sample in enumerate((10., 10.2, 10.45, 10.8, 10.8)):
        prefix = f"2026-09-16 12:00:00,{10 + index:03d} - "
        rows.append(prefix + f"Astra depth timeline: temporal=new_sample sample_ts={sample} raw=2 distance=2")
        rows.append(prefix + f"depth_linear_limit fresh=True sample_ts={sample} approved=[('forward', 20)]")
    (run / "request_0513_modular.log").write_text("\n".join(rows), encoding="utf-8")
    result = analyze(run)
    depth = result["depth"]
    assert depth["accepted_distinct_samples"] == 4
    assert depth["gaps_over_180ms"] == 3
    assert depth["gaps_180_to_250ms"] == 2
    assert depth["gaps_over_250ms"] == 1
    assert depth["positive_authorized_gaps_180_to_250ms"] == 2
    assert depth["positive_authorized_gaps_over_250ms"] == 1
    assert comparison(result, result)["metrics"]["depth.positive_authorized_gaps_over_250ms_per_10sec"]["delta"] == 0


def test_absent_depth_does_not_invent_a_continuation_success(run):
    (run / "request_0513_modular.log").write_text("", encoding="utf-8")
    depth = analyze(run)["depth"]
    assert depth["positive_authorized_gaps_180_to_250ms"] == 0
    assert depth["positive_authorized_gaps_over_250ms"] == 0
    assert depth["continuation_180_to_250"]["allowed_distinct_grants"] == 0


def test_continuation_reads_are_not_counted_as_new_depth_authority(run):
    rows = [
        "depth_forward_continuation uid=1 sample_ts=10 allowed=True reason=margin_ok",
        "depth_forward_continuation uid=1 sample_ts=10 allowed=True reason=margin_ok",
        "depth_forward_continuation uid=1 sample_ts=10 allowed=False reason=braking_margin",
        "depth_late_sample_continuation uid=1 original_sample_ts=10 observation_ts=10.01",
        "depth_late_sample_continuation uid=1 original_sample_ts=10 observation_ts=10.02",
    ]
    (run / "request_0513_modular.log").write_text("\n".join(
        f"2026-09-16 12:00:00,{10 + index:03d} - {row}" for index, row in enumerate(rows)
    ), encoding="utf-8")
    depth = analyze(run)["depth"]
    assert depth["accepted_distinct_samples"] == depth["positive_authorized_samples"] == 0
    continuation = depth["continuation_180_to_250"]
    assert continuation["check_records"] == {"True": 2, "False": 1}
    assert continuation["allowed_distinct_grants"] == 1
    assert continuation["rejected_distinct_grants"] == 1
    assert continuation["late_sample_preserved_distinct_grants"] == 1
