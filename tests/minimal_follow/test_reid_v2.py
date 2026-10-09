from minimal_follow.reid_v2 import ReidCandidate, ReidConfig, ReidPolicy
from minimal_follow.reid_v2.association import associate
from minimal_follow.reid_v2.profile import TargetProfile
from minimal_follow.reid_v2.worker import ReidResult


class _Frame:
    shape = (480, 640, 3)

    def __getitem__(self, _slice):
        return self

    def copy(self):
        return self


class _Worker:
    def __init__(self):
        self.results = []
        self.requests = []

    def submit(self, request, *, min_interval_sec):
        self.requests.append(request)
        return True

    def poll_latest(self):
        return self.results.pop(0) if self.results else None


def _candidate(x1=100.0, y1=80.0, x2=260.0, y2=400.0):
    return ReidCandidate((x1, y1, x2, y2), 1, (x2 - x1) * (y2 - y1), 0.95)


def test_association_does_not_switch_to_larger_person_far_away():
    target = _candidate()
    larger_other = _candidate(380.0, 40.0, 630.0, 470.0)
    continuous = _candidate(108.0, 84.0, 268.0, 404.0)
    selected = associate([larger_other, continuous], target.bbox, frame_width=640, frame_height=480,
                         min_iou=0.12, max_center_distance_ratio=0.16)
    assert selected == continuous


def test_profile_uses_top_three_mean_not_a_single_accidental_hit():
    profile = TargetProfile(max_full=4, max_torso=2, duplicate_similarity=1.0)
    assert profile.add((1.0, 0.0), source="full", quality=1.0, captured_at=1.0, view_bin=1)
    assert profile.add((0.6, 0.8), source="full", quality=.9, captured_at=2.0, view_bin=1)
    assert profile.add((0.6, -0.8), source="full", quality=.8, captured_at=3.0, view_bin=1)
    match = profile.match((1.0, 0.0), None)
    assert match.source == "full"
    assert match.score is not None and .70 < match.score < .75


def test_reacquire_requires_repeated_fresh_reid_evidence():
    worker = _Worker()
    policy = ReidPolicy(ReidConfig(
        stable_frames=1, min_full_templates=1, min_torso_templates=1,
        full_threshold=.70, torso_threshold=.80, confirm_hits=2, confirm_window=3,
    ), worker)
    frame = _Frame()
    candidate = _candidate()

    first = policy.observe(frame=frame, candidates=[candidate], frame_id=1, now=1.0, frame_width=640, frame_height=480)
    assert first.accepted and first.state == "ENROLLING"
    worker.results.append(ReidResult(1, 1.0, 1.01, "enroll", candidate.bbox, .9, (1.0, 0.0), None, {}))
    locked = policy.observe(frame=frame, candidates=[candidate], frame_id=2, now=1.1, frame_width=640, frame_height=480)
    assert locked.accepted and locked.state == "LOCKED"

    missing = policy.observe(frame=frame, candidates=[], frame_id=3, now=2.0, frame_width=640, frame_height=480)
    assert not missing.accepted and missing.state == "SEARCHING"

    pending = policy.observe(frame=frame, candidates=[candidate], frame_id=4, now=2.01, frame_width=640, frame_height=480)
    assert not pending.accepted and pending.reason == "reid_pending"
    worker.results.append(ReidResult(4, 2.01, 2.02, "reacquire", candidate.bbox, .9, (1.0, 0.0), None, {}))
    confirming = policy.observe(frame=frame, candidates=[candidate], frame_id=5, now=2.03, frame_width=640, frame_height=480)
    assert not confirming.accepted and confirming.reason == "reid_confirming"
    worker.results.append(ReidResult(5, 2.04, 2.05, "reacquire", candidate.bbox, .9, (1.0, 0.0), None, {}))
    confirmed = policy.observe(frame=frame, candidates=[candidate], frame_id=6, now=2.06, frame_width=640, frame_height=480)
    assert confirmed.accepted and confirmed.reason == "reid_confirmed"


def test_search_round_robins_after_one_candidate_uses_its_confirmation_window():
    worker = _Worker()
    policy = ReidPolicy(ReidConfig(
        stable_frames=1, min_full_templates=1, min_torso_templates=1,
        full_threshold=.70, confirm_hits=2, confirm_window=2,
    ), worker)
    frame = _Frame()
    target = _candidate()
    left = _candidate(10.0, 80.0, 170.0, 400.0)
    right = _candidate(400.0, 80.0, 560.0, 400.0)

    policy.observe(frame=frame, candidates=[target], frame_id=1, now=1.0, frame_width=640, frame_height=480)
    worker.results.append(ReidResult(1, 1.0, 1.01, "enroll", target.bbox, .9, (1.0, 0.0), None, {}))
    policy.observe(frame=frame, candidates=[target], frame_id=2, now=1.1, frame_width=640, frame_height=480)
    policy.observe(frame=frame, candidates=[], frame_id=3, now=2.0, frame_width=640, frame_height=480)

    policy.observe(frame=frame, candidates=[left, right], frame_id=4, now=2.01, frame_width=640, frame_height=480)
    assert worker.requests[-1].bbox == left.bbox
    worker.results.append(ReidResult(4, 2.01, 2.02, "reacquire", left.bbox, .9, (0.0, 1.0), None, {}))
    policy.observe(frame=frame, candidates=[left, right], frame_id=5, now=2.03, frame_width=640, frame_height=480)
    worker.results.append(ReidResult(5, 2.03, 2.04, "reacquire", left.bbox, .9, (0.0, 1.0), None, {}))
    policy.observe(frame=frame, candidates=[left, right], frame_id=6, now=2.05, frame_width=640, frame_height=480)
    assert worker.requests[-1].bbox == right.bbox
