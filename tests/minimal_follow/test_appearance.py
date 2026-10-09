from minimal_follow.appearance.bank import AppearanceTemplateBank
from minimal_follow.appearance.identity import AppearanceIdentityConfig, AppearanceIdentityPolicy
from minimal_follow.appearance.quality import AppearanceQualityConfig, AppearanceQualityGate
from minimal_follow.appearance.worker import AppearanceResult


class _Frame:
    def copy(self):
        return self


class _Worker:
    def __init__(self):
        self.latest = None
        self.requests = []

    def submit(self, request, *, min_interval_sec):
        self.requests.append((request, min_interval_sec))
        return True

    def poll_latest(self):
        result, self.latest = self.latest, None
        return result


def test_template_bank_normalizes_and_keeps_best_full_match():
    bank = AppearanceTemplateBank(max_full=2)
    assert bank.add((4.0, 0.0), source="full")
    match = bank.match((0.99, 0.01), None)
    assert match.source == "full"
    assert match.score is not None and match.score > 0.98


def test_reacquire_requires_two_fresh_matching_results():
    worker = _Worker()
    policy = AppearanceIdentityPolicy(
        AppearanceIdentityConfig(
            stable_enroll_frames=1,
            reacquire_confirm_results=2,
            full_match_threshold=0.70,
        ),
        AppearanceQualityGate(AppearanceQualityConfig(min_height_px=20, min_area_px=100)),
        worker,
    )
    frame = _Frame()
    bbox = (100.0, 80.0, 220.0, 320.0)

    policy.visible_target(
        frame=frame, bbox=bbox, score=.95, frame_id=1, now=1.0, frame_width=640, frame_height=480,
    )
    assert worker.requests[-1][0].purpose == "enroll"
    assert worker.requests[-1][0].allow_full is True
    assert worker.requests[-1][0].compute_partial is True
    worker.latest = AppearanceResult(1, 1.0, 1.01, "enroll", (1.0, 0.0), None, {})
    policy.visible_target(
        frame=frame, bbox=bbox, score=.95, frame_id=2, now=1.02, frame_width=640, frame_height=480,
    )
    assert policy.enrolled

    policy.target_missing(2.0)
    worker.latest = AppearanceResult(3, 2.01, 2.02, "reacquire", (1.0, 0.0), None, {})
    first = policy.search_candidate(
        frame=frame, bbox=bbox, score=.95, frame_id=3, now=2.03, frame_width=640, frame_height=480,
    )
    assert first.accepted is False and first.reason == "confirming"

    worker.latest = AppearanceResult(4, 2.04, 2.05, "reacquire", (1.0, 0.0), None, {})
    second = policy.search_candidate(
        frame=frame, bbox=bbox, score=.95, frame_id=4, now=2.06, frame_width=640, frame_height=480,
    )
    assert second.accepted is True and second.reason == "confirmed"


def test_partial_gallery_is_a_valid_cache_and_rejects_unenrolled_reacquire():
    worker = _Worker()
    policy = AppearanceIdentityPolicy(
        AppearanceIdentityConfig(stable_enroll_frames=1, reacquire_confirm_results=1),
        AppearanceQualityGate(AppearanceQualityConfig(min_height_px=20, min_area_px=100)),
        worker,
    )
    frame = _Frame()
    edge_bbox = (100.0, 0.0, 220.0, 240.0)

    # Before any trustworthy cache exists, a post-loss candidate cannot take
    # control merely because it is the largest detected person.
    policy.target_missing(1.0)
    pending = policy.search_candidate(
        frame=frame, bbox=edge_bbox, score=.95, frame_id=1, now=1.01,
        frame_width=640, frame_height=480,
    )
    assert pending.accepted is False and pending.state == "enrollment_pending"

    # Edge-clipped enrollment retains a torso template only; it is still a
    # valid identity cache for later partial re-identification.
    policy.visible_target(
        frame=frame, bbox=edge_bbox, score=.95, frame_id=2, now=2.0,
        frame_width=640, frame_height=480,
    )
    assert worker.requests[-1][0].allow_full is False
    assert worker.requests[-1][0].compute_partial is True
    worker.latest = AppearanceResult(2, 2.0, 2.01, "enroll", None, (1.0, 0.0), {})
    policy.visible_target(
        frame=frame, bbox=edge_bbox, score=.95, frame_id=3, now=2.02,
        frame_width=640, frame_height=480,
    )
    assert policy.enrolled
