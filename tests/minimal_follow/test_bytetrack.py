from minimal_follow.reid_v2 import ByteTrack, ByteTrackConfig, ReidCandidate


def _candidate(x1, y1, x2, y2, score=.9):
    return ReidCandidate((x1, y1, x2, y2), 0, (x2 - x1) * (y2 - y1), score)


def test_bytetrack_keeps_ids_for_two_people_after_small_motion():
    tracker = ByteTrack(ByteTrackConfig(min_confirmed_hits=1, max_lost_frames=3))
    first = tracker.update([_candidate(20, 40, 120, 300), _candidate(350, 40, 450, 300)])
    assert [item.track_id for item in first] == [1, 2]
    second = tracker.update([_candidate(30, 40, 130, 300), _candidate(340, 40, 440, 300)])
    assert [item.track_id for item in second] == [1, 2]


def test_bytetrack_uses_low_confidence_detection_to_keep_an_existing_track():
    tracker = ByteTrack(ByteTrackConfig(min_confirmed_hits=1, max_lost_frames=3))
    tracker.update([_candidate(20, 40, 120, 300, .9)])
    result = tracker.update([_candidate(25, 40, 125, 300, .3)])
    assert len(result) == 1 and result[0].track_id == 1
