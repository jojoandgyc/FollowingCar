"""250ms detector window, independent 180ms Depth lease. No hardware."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import request_0513_modular as runtime
from car_control_modular.control_types import SteeringFeedback
from car_control_modular.depth_roi_policy import roi_age_status
from test_depth_raw_geometry_runtime import make_runtime, person
from test_lateral_zero_runtime import owner, NOW
from test_longitudinal_authority_runtime import _commit


def feedback(now=100., **changes):
    return replace(SteeringFeedback(timestamp=now-.01, trustworthy=True,
                                    yaw_rate_right_dps=2., raw_yaw_rate_right_dps=3.), **changes)


@pytest.mark.parametrize('age,expected', [(.1,'normal'),(.19,'extended'),(.24,'extended'),(.25,'extended'),(.251,'expired')])
def test_detector_window(age, expected):
    assert roi_age_status(100.-age,100.,.25,feedback())==expected


@pytest.mark.parametrize('fb,expected', [
    (None,'feedback_missing'), (feedback(trustworthy=False),'feedback_missing'),
    (feedback(timestamp=99.8),'feedback_stale'), (feedback(timestamp=100.1),'feedback_stale'),
    (feedback(raw_yaw_rate_right_dps=12),'turning'),
    (feedback(yaw_rate_right_dps=-6),'turning'),
    (feedback(raw_yaw_rate_right_dps=float('nan')),'feedback_invalid'),
])
def test_extended_window_fails_closed(fb, expected):
    assert roi_age_status(99.78,100.,.25,fb)==expected
    assert roi_age_status(99.9,100.,.25,fb)=='normal'  # old path unchanged


def test_legacy_180_setting_does_not_enable_extension():
    assert roi_age_status(99.78,100.,.18,feedback())=='expired'


@pytest.mark.parametrize('yaw,accepted', [(2.,True),(6.,False)])
def test_final_sampling_gate_uses_new_depth_with_old_detector(monkeypatch,yaw,accepted):
    monkeypatch.setattr('car_control_modular.distance_runtime.time.monotonic',lambda:100.)
    obj,sensors=make_runtime(vision_depth_detector_bbox_max_age_sec=.25)
    p=person(capture_timestamp=99.78)
    state=obj.get_vision_depth_state(640,480,p,use_latest_depth=True,capture_timestamp=99.78,
                                   steering_feedback=feedback(raw_yaw_rate_right_dps=yaw))
    assert bool(sensors.calls)==accepted
    if accepted:
        assert state.raw_distance_m==2.
    else:
        assert state.raw_distance_m is None


def test_rgb_window_never_extends_motor_lease(owner,monkeypatch):
    monkeypatch.setattr(runtime,'ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC',.25)
    monkeypatch.setattr(runtime,'ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC',.18)
    _commit(owner,stamp=NOW-.02)
    assert owner._depth30_linear_timing.depth_expires_at==pytest.approx(NOW+.16)
    assert owner._fresh_depth_linear_snapshot(1,now=NOW+.17) is None
    actions,accepted=_commit(owner,stamp=NOW-.20)
    assert not accepted and actions==[]


def test_owner_rechecks_low_yaw_for_extension(owner,monkeypatch):
    monkeypatch.setattr(runtime,'ASTRA_DEPTH_LONGITUDINAL_BBOX_MAX_AGE_SEC',.25)
    owner._action_runtime=SimpleNamespace(get_steering_feedback=lambda:feedback(NOW))
    assert owner._depth_roi_age_allowed(NOW-.22,NOW)
    owner._action_runtime.get_steering_feedback=lambda:feedback(NOW,raw_yaw_rate_right_dps=20)
    assert not owner._depth_roi_age_allowed(NOW-.22,NOW)

