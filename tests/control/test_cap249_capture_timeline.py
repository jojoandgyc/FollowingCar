"""No hardware: historical velocity windows must not become motion leases.

The CAP triples below preserve logged capture timestamps, detector centers and
processing ages from run_20260929_234529_138351_7abb13d5. Boxes are synthetic
associated boxes: this reproduces the time/rate policy, not the full perception
pipeline or resulting vehicle trajectory.
"""
from dataclasses import replace

import pytest

from car_control_modular.control_types import DepthTargetObservation, PersonTarget, SensorFrame
from car_control_modular.outward_trajectory import make_outward_lead
from car_control_modular.visual_steering_evidence import CaptureSteeringEvidence
import car_control_modular.visual_steering_evidence as evidence_module


def sample(cap, stamp, x, *, uid=1, raw=6, braking=False, edge=False):
    box = (0., 50., 2*x*640, 450.) if edge else (x*640-40, 50., x*640+40, 450.)
    observation = DepthTargetObservation(box, uid, raw, cap, stamp,
        source="yolo_braking_only" if braking else "yolo_detector")
    target = PersonTarget(box, uid, .95, (box[2]-box[0])*400,
        depth_observation=None if braking else observation,
        braking_observation=observation if braking else None)
    return target, SensorFrame(width=640, height=480, persons=[target],
        capture_frame_id=cap, capture_timestamp=stamp)


def observe(evidence, cap, stamp, x, *, age=.1, **kwargs):
    target, frame = sample(cap, stamp, x, **kwargs)
    return evidence.observe(target, frame, stamp+age, 66)


REAL_TIME_TRIPLES = [
    ((217,21724.992839,.2475,.2020),(221,21725.192619,.3075,.1822),(224,21725.356571,.3518,.1366)),
    ((241,21726.256533,.3614,.1819),(244,21726.422989,.3834,.1954),(249,21726.658922,.4105,.1578)),
    ((244,21726.422989,.3834,.1954),(249,21726.658922,.4105,.1578),(253,21726.854782,.4538,.1867)),
    ((249,21726.658922,.4105,.1578),(253,21726.854782,.4538,.1867),(256,21727.020916,.4916,.1614)),
    ((256,21727.020916,.4916,.1614),(258,21727.152675,.5181,.2049),(263,21727.388109,.5710,.1546)),
    ((258,21727.152675,.5181,.2049),(263,21727.388109,.5710,.1546),(266,21727.552581,.6504,.1975)),
    ((263,21727.388109,.5710,.1546),(266,21727.552581,.6504,.1975),(271,21727.787423,.7461,.1899)),
    ((286,21728.584053,.8765,.2258),(290,21728.817085,.8952,.1847),(294,21729.018184,.9225,.1345)),
    ((395,21734.274589,.7584,.2326),(399,21734.474858,.7325,.2117),(403,21734.675840,.7087,.1478)),
    ((399,21734.474858,.7325,.2117),(403,21734.675840,.7087,.1478),(407,21734.876569,.6745,.1087)),
    ((420,21735.571306,.6382,.1132),(423,21735.707642,.6616,.1584),(427,21735.941427,.7044,.1413)),
    ((423,21735.707642,.6616,.1584),(427,21735.941427,.7044,.1413),(431,21736.139241,.7412,.1565)),
    ((427,21735.941427,.7044,.1413),(431,21736.139241,.7412,.1565),(434,21736.305223,.7533,.1765)),
    ((463,21737.804437,.6296,.1817),(467,21738.002828,.5805,.1771),(470,21738.171852,.5282,.1333)),
]


@pytest.mark.parametrize("samples", REAL_TIME_TRIPLES, ids=lambda s: f"CAP{s[-1][0]}")
def test_logged_consistent_rate_only_failed_old_history_span(samples, monkeypatch):
    def replay():
        evidence = CaptureSteeringEvidence()
        for cap, stamp, x, age in samples:
            result = observe(evidence, cap, stamp, x, age=age)
        return result
    with monkeypatch.context() as legacy:
        legacy.setattr(evidence_module, "MAX_CAPTURE_HISTORY_SPAN_SEC", .35)
        old = replay()
    current = replay()
    assert old.reason == "rate_inconsistent" and old.rate_dps is None
    assert current.reason == "capture_rate_valid"
    assert current.rate_dps == pytest.approx(
        (samples[-1][2]-samples[-2][2])*66/(samples[-1][1]-samples[-2][1]))
    assert .35 < current.span_sec <= .5
    assert current.capture_timestamp == samples[-1][1]


@pytest.mark.parametrize("spacing", [.175, .2, .25])
def test_newest_age_and_two_capture_intervals_have_independent_bounds(spacing):
    evidence = CaptureSteeringEvidence()
    for cap, x in enumerate([.60,.63,.66], 1):
        result = observe(evidence, cap, 10+(cap-1)*spacing, x, age=.25)
    assert result.reason == "capture_rate_valid"
    assert result.span_sec == pytest.approx(2*spacing)
    assert result.rate_dps == pytest.approx(.03*66/spacing)


def test_skip_stale_continuous_capture_never_renews_or_appends_evidence():
    evidence = CaptureSteeringEvidence()
    observe(evidence, 1, 10., .70)
    accepted = observe(evidence, 2, 10.2, .68)
    before = (tuple(evidence.samples), tuple(evidence.brake_samples), evidence.reliable_anchor)
    stale = observe(evidence, 3, 10.25, .675, age=.26)
    assert stale.reason == "stale_observation_skipped"
    assert stale.detector_x is None and stale.rate_dps is None
    assert stale.first_timestamp == 0 and not stale.outward_consistent
    assert stale.inward_turnaround_rate_dps is None and stale.outward_continuity_rate_dps is None
    assert evidence.last is accepted
    assert (tuple(evidence.samples), tuple(evidence.brake_samples), evidence.reliable_anchor) == before
    fresh = observe(evidence, 4, 10.4, .66, age=.12)
    assert fresh.reason == "capture_rate_valid" and fresh.rate_dps == pytest.approx(-6.6)
    assert [point[0] for point in evidence.samples] == [10.,10.2,10.4]


def test_skipped_frame_does_not_bridge_long_gap_or_receive_later_freshness():
    evidence = CaptureSteeringEvidence()
    observe(evidence, 1, 10., .70)
    observe(evidence, 2, 10.2, .68)
    observe(evidence, 3, 10.25, .675, age=.26)
    duplicate = observe(evidence, 2, 10.2, .68, age=.32)
    assert duplicate.rate_dps is None
    fresh = observe(evidence, 4, 10.451, .655, age=.1)
    assert fresh.reason == "warming" and fresh.rate_dps is None
    assert len(evidence.samples) == len(evidence.brake_samples) == 1


@pytest.mark.parametrize("conflict", ["uid", "raw", "quality", "missing", "source",
                                      "association", "jump", "scale", "mode", "future"])
def test_late_frame_still_breaks_history_for_identity_geometry_or_quality_conflict(conflict):
    evidence = CaptureSteeringEvidence()
    observe(evidence, 1, 10., .70)
    observe(evidence, 2, 10.2, .68)
    kwargs = {"uid": 2} if conflict == "uid" else {"raw": 7} if conflict == "raw" else {}
    if conflict == "mode": kwargs["braking"] = True
    target, frame = sample(3, 10.25, .40 if conflict == "jump" else .675, **kwargs)
    now = 10.51
    if conflict == "quality": target = replace(target, confidence=.2)
    if conflict == "missing": target = replace(target, depth_observation=None)
    if conflict == "source": target = replace(target, depth_observation=replace(target.depth_observation, source="predicted"))
    if conflict == "association": target = replace(target, depth_observation=replace(target.depth_observation, capture_timestamp=10.24))
    if conflict == "scale":
        box = (target.bbox[0]-40, 50, target.bbox[2]+40, 450)
        target = replace(target, bbox=box, depth_observation=replace(target.depth_observation, bbox=box))
    if conflict == "future": now = 10.24
    result = evidence.observe(target, frame, now, 66)
    assert result.reason == "unqualified"
    assert not evidence.samples and not evidence.brake_samples
    assert evidence.reliable_anchor is None


@pytest.mark.parametrize("mode", ["reversal", "gap", "jitter"])
def test_longer_window_does_not_admit_bad_velocity(mode):
    evidence = CaptureSteeringEvidence()
    points = [(10.,.6),(10.2,.62),(10.4,.64)]
    if mode == "reversal": points[-1] = (10.4,.60)
    if mode == "gap": points[-1] = (10.451,.64)
    if mode == "jitter": points[-1] = (10.4,.82)
    for cap, (stamp, x) in enumerate(points, 1):
        result = observe(evidence, cap, stamp, x)
    assert result.rate_dps is None and not result.outward_continuity_rate_dps


def test_cross_quality_inward_braking_uses_same_bounded_history():
    evidence = CaptureSteeringEvidence()
    observe(evidence, 1, 10., .75)
    observe(evidence, 2, 10.2, .70, braking=True)
    result = observe(evidence, 3, 10.4, .65)
    assert result.reason == "braking_continuity:capture_rate_valid"
    assert result.rate_dps == pytest.approx(-16.5)
    assert result.span_sec == pytest.approx(.4)
    assert not result.outward_consistent


def test_braking_only_edge_rate_remains_continuity_only_not_lead():
    evidence = CaptureSteeringEvidence()
    observe(evidence, 1, 10., .27, edge=True)
    observe(evidence, 2, 10.1, .25, edge=True, braking=True)
    observe(evidence, 3, 10.3, .23, edge=True, braking=True)
    result = observe(evidence, 4, 10.5, .21, edge=True, braking=True)
    assert result.rate_dps is None
    assert result.outward_continuity_rate_dps == pytest.approx(-6.6)
    assert make_outward_lead(result, 10.6, hfov=66, deadband=3,
                             release_margin=1.8, enabled=True) is None


def test_turnaround_slopes_do_not_overwrite_bbox_edges():
    evidence = CaptureSteeringEvidence()
    observe(evidence, 1, 10., .27, edge=True)
    observe(evidence, 2, 10.1, .25, edge=True, braking=True)
    observe(evidence, 3, 10.3, .23, edge=True, braking=True)
    result = observe(evidence, 4, 10.5, .25, edge=True, braking=True)
    assert result.rate_dps is None and result.outward_continuity_rate_dps is None
    assert result.inward_turnaround_rate_dps == pytest.approx(6.6)


def test_outward_consumer_accepts_slow_capture_cadence_without_more_rpm_or_ttl():
    evidence = CaptureSteeringEvidence()
    for cap, x in enumerate([.505,.530,.555], 1):
        result = observe(evidence, cap, 10+(cap-1)*.2, x)
    lead = make_outward_lead(result, 10.5, hfov=66, deadband=3,
                             release_margin=1.8, enabled=True)
    assert lead is not None and lead.correction_rpm == 4
    assert lead.capture_timestamp == 10.4 and lead.first_timestamp == 10.
    assert lead.matches(1, 3, 10.4, 10.65)
    assert not lead.matches(1, 3, 10.4, 10.651)
    assert make_outward_lead(result, 10.651, hfov=66, deadband=3,
                             release_margin=1.8, enabled=True) is None
