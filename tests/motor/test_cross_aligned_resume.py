"""Fresh forward feedback can end a latched reversal wait, never by timeout."""
from types import SimpleNamespace
import pytest
from car_control_modular.wheel_zero_cross import WheelZeroCrossGuard


def tick(g,t,left,right=0,requested=(44,44),stamp=None,trusted=True):
    f=SimpleNamespace(timestamp=t if stamp is None else stamp,trustworthy=trusted,
        left_forward_rpm=left,right_forward_rpm=right)
    result=g.limit(requested,f,t)
    g.note_sent(result[0],t)
    return result


def test_cap426_439_two_new_aligned_samples_release_then_ramp():
    g=WheelZeroCrossGuard()
    assert tick(g,100,-2)[0]==(0,0)
    assert tick(g,100.1,8)[0]==(0,0)
    assert tick(g,100.12,8,stamp=100.1)[0]==(0,0)
    assert tick(g,100.2,8)==((5,5),'cross_aligned_resume')
    prior=5
    for i in range(1,14):
        pair,reason=tick(g,100.2+i*.05,8)
        assert prior<=pair[0]<=prior+4
        assert pair[0]==pair[1] and pair[0]<=44
        prior=pair[0]
    assert pair==(44,44)


@pytest.mark.parametrize('left,right,trust,stamp',[
    (-2,0,True,None),(8,-2,True,None),(8,0,False,None),
    (8,0,True,99),(float('nan'),0,True,None),(60,0,True,None)])
def test_invalid_or_opposing_feedback_never_releases_even_after_timeout(left,right,trust,stamp):
    g=WheelZeroCrossGuard();tick(g,100,-2)
    for t in (100.3,100.4,100.5):
        assert tick(g,t,left,right,stamp=stamp,trusted=trust)[0]==(0,0)


def test_untrusted_or_opposing_sample_breaks_two_frame_confirmation():
    g=WheelZeroCrossGuard();tick(g,100,-2)
    tick(g,100.1,8)
    tick(g,100.2,8,trusted=False)
    assert tick(g,100.3,8)[0]==(0,0)
    assert tick(g,100.4,8)[1]=='cross_aligned_resume'
    assert tick(g,100.5,-2)[0]==(0,0)


def test_explicit_stop_and_requested_reduction_are_not_delayed_by_ramp():
    g=WheelZeroCrossGuard();tick(g,100,-2);tick(g,100.1,8);tick(g,100.2,8)
    assert tick(g,100.25,8,requested=(2,2))[0]==(2,2)
    assert tick(g,100.3,8,requested=(0,0))[0]==(0,0)
    assert g.resume_signs is None


def test_rotation_and_reverse_keep_quiet_confirmation():
    for request in ((8,-8),(-8,-8)):
        g=WheelZeroCrossGuard();tick(g,100,20,20,requested=request)
        for t in (100.1,100.2,100.4):
            assert tick(g,t,request[0],request[1],requested=request)[0]==(0,0)


def test_ramp_preserves_forward_curvature_and_cannot_jump_after_long_gap():
    g=WheelZeroCrossGuard();tick(g,100,-2);tick(g,100.1,8)
    tick(g,100.2,8)
    pair,_=tick(g,100.3,8,requested=(44,22))
    assert abs(pair[0]-2*pair[1])<=1
    old=pair
    pair,_=tick(g,101,8,requested=(44,22))
    assert pair[0]<=old[0]+8


def test_long_feedback_gap_restarts_aligned_confirmation():
    g=WheelZeroCrossGuard();tick(g,100,-2);tick(g,100.1,8)
    assert tick(g,100.4,8)[0]==(0,0)
    assert tick(g,100.45,8)[1]=='cross_aligned_resume'


def test_motor_deadline_can_cancel_aligned_resume(monkeypatch):
    from test_visible_wheel_continuity import visible_runtime, feedback
    motor,owner,driver,symbols,clock=visible_runtime(monkeypatch)
    for t,l in [(10.,-2),(10.05,8),(10.10,8)]:
        clock[0]=t
        motor.get_steering_feedback=lambda:feedback(clock[0],l,0)
        motor._send_follow_wheel_targets(24,-24,'TEST')
    assert driver.pairs[-1]==(5,-5)
    owner._fresh_depth_linear_snapshot=lambda uid,now=None:None
    clock[0]+=.05
    motor._send_follow_wheel_targets(24,-24,'TEST')
    assert driver.pairs[-1]==(0,0)
