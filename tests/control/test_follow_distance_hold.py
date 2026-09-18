"""CAP746→822 parking recovery. Real entry points, no motor/camera hardware."""
from dataclasses import replace
from types import SimpleNamespace
import threading

import pytest
import request_0513_modular as runtime
from car_control_modular.follow_distance_hold import FollowDistanceHold, is_follow_distance_hold
from car_control_modular.control_types import DepthTargetObservation, SteeringFeedback, HazardState, ObstacleState
from test_longitudinal_authority_runtime import owner, _frame, NOW


@pytest.fixture
def parked(owner):
    owner._brake_hold_active = True
    owner._brake_hold_label = 'follow_distance_hold'
    owner._brake_hold_stop_mode = None
    owner._follow_distance_hold = FollowDistanceHold(1, NOW-.15)
    owner.motor_io_lock = threading.RLock()
    owner._action_runtime = SimpleNamespace(hard_stop_check=lambda action: False)
    return owner


def observation(stamp=NOW-.08, distance=1.65):
    frame = _frame(stamp=stamp, distance=distance)
    target = replace(frame.persons[0], depth_observation=DepthTargetObservation(
        target_id=1, raw_track_id=2, bbox=frame.persons[0].bbox,
        capture_frame_id=frame.capture_frame_id, capture_timestamp=frame.capture_timestamp,
        source='yolo_detector'))
    return replace(frame, persons=[target], steering_feedback=SteeringFeedback(
        timestamp=NOW-.01, trustworthy=True))


def observe(owner, frame=None, **kwargs):
    frame = frame or observation()
    return owner._observe_follow_distance_hold(frame, frame.persons[0],
                 fresh=kwargs.get('fresh', True), steerable=kwargs.get('steerable', True),
                 low_quality=kwargs.get('low_quality', False))


def test_two_distinct_safe_depth_frames_release_without_a_rotate_command(parked):
    assert not observe(parked)
    assert parked._brake_hold_active
    assert not observe(parked)  # repeated physical sample is not frame two
    assert observe(parked, observation(stamp=NOW-.04))
    assert not parked._brake_hold_active
    assert parked._depth30_linear_snapshot is None  # no old motor authority revived


def test_safe_replay_does_not_clear_evidence_or_confirm(parked):
    assert not observe(parked)
    f = observation()
    f = replace(f, distance_state=replace(f.distance_state, raw_distance_m=None,
        sample_timestamp=None, observation_timestamp=NOW-.08, temporal_status='duplicate'))
    assert not observe(parked, f, fresh=False)
    assert parked._follow_distance_hold.count == 1
    assert observe(parked, observation(stamp=NOW-.04))


@pytest.mark.parametrize('case', ['hazard','obstacle','brake_latched','target_latched','safety_distance',
    'stale_depth','old_before_stop','stale_rgb','stale_feedback','untrusted_feedback',
    'reverse_feedback','wrong_uid','missing_detector','held_only','too_near','far_jump','weak'])
def test_no_release_on_unsafe_or_unqualified_evidence(parked, case):
    observe(parked)
    f = observation(stamp=NOW-.04)
    if case == 'hazard': f = replace(f, hazard=HazardState(active=True))
    elif case == 'obstacle': f = replace(f, obstacles=ObstacleState(front=True))
    elif case in {'brake_latched','target_latched'}: f = replace(f, distance_state=replace(f.distance_state, **{case:True}))
    elif case == 'safety_distance': f = replace(f, distance_state=replace(f.distance_state, safety_distance_m=.4))
    elif case == 'stale_depth': f = observation(stamp=NOW-.19)
    elif case == 'old_before_stop': f = observation(stamp=NOW-.16)
    elif case == 'stale_rgb': f = replace(f, capture_timestamp=NOW-.3)
    elif case == 'stale_feedback': f = replace(f, steering_feedback=replace(f.steering_feedback,timestamp=NOW-.16))
    elif case == 'untrusted_feedback': f = replace(f, steering_feedback=replace(f.steering_feedback,trustworthy=False))
    elif case == 'reverse_feedback': f = replace(f, steering_feedback=replace(f.steering_feedback,left_forward_rpm=-5))
    elif case == 'wrong_uid': parked._follow_controller.active_target_id = 2
    elif case == 'missing_detector': f = replace(f, persons=[replace(f.persons[0],depth_observation=None)])
    elif case == 'held_only': f = replace(f,distance_state=replace(f.distance_state,raw_distance_m=None))
    elif case == 'too_near': f = observation(stamp=NOW-.04,distance=1.5)
    elif case == 'far_jump': f = observation(stamp=NOW-.04,distance=3.)
    assert not observe(parked,f,low_quality=case=='weak')
    assert parked._brake_hold_active


@pytest.mark.parametrize('case',['unknown','safety','fault','search','explicit','shutdown','mode'])
def test_other_holds_are_not_ordinary_distance_parking(parked,case):
    if case in {'unknown','safety','fault'}: parked._brake_hold_label=case
    elif case=='search': parked.search_state='searching'
    elif case=='explicit': parked._explicit_stop_requested=True
    elif case=='shutdown': parked._runtime_shutdown_requested=True
    elif case=='mode': parked._brake_hold_stop_mode='brake'
    assert not is_follow_distance_hold(parked)
    assert not observe(parked)


@pytest.mark.parametrize('race',['hard_stop','exception','replace_token','expire_during_check'])
def test_recheck_cannot_release_new_safety_event(parked, race, monkeypatch):
    observe(parked)
    def check(action):
        if race=='hard_stop': return True
        if race=='exception': raise RuntimeError('unavailable')
        if race=='replace_token': parked._follow_distance_hold=FollowDistanceHold(1,NOW)
        if race=='expire_during_check': monkeypatch.setattr(runtime.time,'monotonic',lambda:NOW+.3)
        return False
    parked._action_runtime.hard_stop_check=check
    assert not observe(parked,observation(stamp=NOW-.04))
    assert parked._brake_hold_active
