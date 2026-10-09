"""Fake register transport only: 5A NORMAL <-> 0A speed mode."""
from dataclasses import replace
from enum import IntEnum
import sys
from types import SimpleNamespace
import pytest
from test_depth_drive_rpm import make_runtime
from test_follow_wheel_periodic import setup_periodic
from test_near_yaw_park_execution import request_park
from test_visible_wheel_continuity import feedback


def test_normal_speed_zero_and_repark_currents_are_volatile_and_cached():
    r,o,d,s=make_runtime()
    b=r.backend
    b.send_stop('park',mode='normal')
    assert b.parking_current_a==5
    assert d.register_writes==[('right_parking_current',5.,False),('left_parking_current',5.,False)]
    b.send_targets(0,0,'exit_guard_zero')
    assert b.parking_current_a==0
    for _ in range(10):b.send_targets(0,0,'zero_keepalive')
    assert len(d.register_writes)==4
    b.send_targets(7,7,'turn')
    assert len(d.register_writes)==4
    b.send_stop('repark',mode='normal')
    assert b.parking_current_a==5 and len(d.register_writes)==6
    b.refresh_normal_stop('hold')
    assert len(d.register_writes)==6
    assert not any(persist for _,_,persist in d.register_writes)


def test_current_readback_precedes_nonzero_motion():
    r,o,d,s=make_runtime(); b=r.backend
    b.send_stop('park',mode='normal')
    reads=[];old=d.read_register
    def read(name):
        assert d.pairs[-1]==(0,0)
        reads.append(name);return old(name)
    d.read_register=read
    b.send_targets(8,8,'turn')
    assert len(reads)==2 and d.pairs[-1]==(8,8)


@pytest.mark.parametrize('bad', ['mismatch','nan','write_error'])
def test_exit_readback_failure_cannot_send_motion_and_retry_is_not_skipped(bad):
    r,o,d,s=make_runtime(); b=r.backend
    b.send_stop('park',mode='normal');before=list(d.pairs)
    read,write=d.read_register,d.write_register
    if bad=='write_error':
        def fail(*args,**kwargs):raise OSError('simulated')
        d.write_register=fail
    else:d.read_register=lambda name:float('nan') if bad=='nan' else 5.
    with pytest.raises((RuntimeError,OSError)):b.send_targets(8,8,'turn')
    assert d.pairs==before and d.stops[-1]==1 and b._parking_current_uncertain
    d.read_register,d.write_register=read,write
    b.send_targets(8,8,'retry')
    assert b.parking_current_a==0 and not b._parking_current_uncertain


def test_enter_failure_uses_emergency_not_false_normal():
    r,o,d,s=make_runtime();b=r.backend
    d.read_register=lambda name:0.
    with pytest.raises(RuntimeError):b.send_stop('park',mode='normal')
    assert d.stops==[1,1] and not b.normal_zero_hold  # entry EMERGENCY, then failure fallback


def test_zero_current_trial_clears_retained_current_and_never_reparks_at_five(monkeypatch):
    r,o,d,s=make_runtime(); b=r.backend
    b.config=replace(b.config,parking_current_a=0.)
    d.registers.update(left_parking_current=5.,right_parking_current=5.)
    b.parking_current_a=5.
    monkeypatch.setattr('car_control_modular.mssd_motor.time.sleep',lambda _:None)
    b.enable_startup_parking()
    assert b.parking_current_a==0 and all(v==0 for v in d.registers.values())
    d.stops.clear()
    b.send_stop('zero_current_trial',mode='normal',preserve_zero=True)
    assert d.stops==[1,0] and b.normal_zero_hold
    b.refresh_normal_stop('refresh')
    b.send_targets(8,8,'resume')
    b.send_stop('repark',mode='normal')
    assert b.parking_current_a==0 and all(v==0 for v in d.registers.values())
    assert all(value==0 for _,value,_ in d.register_writes)


def test_zero_current_normal_entry_clears_old_five_amp_cache():
    r,o,d,s=make_runtime(); b=r.backend
    b.set_parking_current(5.,persist=False)
    b.config=replace(b.config,parking_current_a=0.)
    b.send_stop('zero_current_trial',mode='normal')
    assert d.stops==[1,0] and b.parking_current_a==0
    assert all(v==0 for v in d.registers.values())


@pytest.mark.parametrize('mode',['emergency','free'])
def test_emergency_free_do_not_wait_for_current_registers(mode):
    r,o,d,s=make_runtime();r.backend.send_stop('safety',mode=mode)
    assert d.register_writes==[]
    assert d.stops==[1 if mode=='emergency' else 2]


def test_explicit_preserved_normal_ignores_zero_keepalive():
    r,o,d,s=make_runtime();b=r.backend
    b.send_stop('cross_park',mode='normal',preserve_zero=True)
    n=len(d.pairs)
    b.send_targets(0,0,'keepalive')
    assert b.parking_current_a==5 and len(d.pairs)==n
    b.send_targets(8,8,'resume')
    assert b.parking_current_a==0


def test_current_switch_delay_cannot_extend_periodic_authority(monkeypatch):
    r,o,d,s,clock,state=setup_periodic(monkeypatch)
    r.backend.send_stop('ordinary_park',mode='normal')
    old=d.read_register
    def read(name):clock[0]=10.3;return old(name)
    d.read_register=read
    r._service_follow_wheels()
    assert d.pairs[-1]==(0,0)
    assert not any(l or rr for l,rr in d.pairs)


def test_five_hundred_ms_park_keeps_five_amps_then_zero_on_release(monkeypatch):
    r,o,d,s,clock,state=setup_periodic(monkeypatch)
    req=request_park(o,clock);r._service_follow_wheels()
    assert r.backend.parking_current_a==5
    for t in [10.01,10.05,10.099,10.1,10.40,10.499]:
        clock[0]=t;r._service_follow_wheels()
        assert r.backend.parking_current_a==5
    clock[0]=10.50
    r._service_follow_wheels()
    assert r.backend.parking_current_a==0
    clock[0]=10.52
    r.get_steering_feedback=lambda:feedback(clock[0],-19,-1)
    assert r.near_yaw_park_motion_ready(req,10.515,clock[0])
    o._near_yaw_park_request=None;o._brake_hold_active=False;o._brake_hold_label='brake'
    state[:2]=[0,7]
    state[2:]=[clock[0]+.15,clock[0]+.15]  # fresh motion, not an expired pre-stop lease
    r._service_follow_wheels()
    assert r.backend.parking_current_a==0 and d.pairs[-1]==(0,0)
    # Current-only release keeps STOP; eligibility is not an immediate reverse-wheel jump.


@pytest.mark.parametrize('startup', [True, False])
def test_real_initialization_preserves_startup_park_until_actual_exit(monkeypatch, startup):
    r,o,d,s=make_runtime(); b=r.backend
    b.driver=None
    b.config=replace(b.config,startup_parking_enabled=startup)
    d.registers.update(left_parking_current=5.,right_parking_current=5.)
    class Modes(IntEnum):
        NORMAL=0
        EMERGENCY=1
        FREE=2
    client=SimpleNamespace(from_serial=lambda *a,**kw:d)
    monkeypatch.setitem(sys.modules,'lz30ema_rs485',SimpleNamespace(LZ30EMAClient=client,StopMode=Modes))
    monkeypatch.setattr(b,'_resolve_lib_dir',lambda:'.')
    monkeypatch.setattr('car_control_modular.mssd_motor.time.sleep',lambda _:None)
    assert b.ensure_driver() is d
    assert b.parking_current_a==(5 if startup else 0)
    if startup:
        assert not b.motion_armed
        assert d.register_writes[-1]==('left_parking_current',5.,True)
    b.send_targets(8,8,'first_real_speed')
    assert b.parking_current_a==0 and d.pairs[-1]==(8,8)
    assert d.register_writes[-1]==('left_parking_current',0.,False)


def test_danger_arriving_during_current_readback_prevents_motion(monkeypatch):
    r,o,d,s,clock,state=setup_periodic(monkeypatch)
    r.backend.send_stop('ordinary_park',mode='normal')
    danger=[False]; read=d.read_register
    r.hard_stop_check=lambda *_:danger[0]
    def incoming_danger(name):
        danger[0]=True
        return read(name)
    d.read_register=incoming_danger
    r._service_follow_wheels()
    assert d.stops[-1]==1
    assert not any(l or rr for l,rr in d.pairs)
