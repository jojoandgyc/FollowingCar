"""500ms actuator ramp is observation delay, never permission to coast blind."""
from dataclasses import replace
import pytest

from car_control_modular.turn_buildup import TurnBuildup
from test_cap837_turn_buildup import image_intent, setup
from test_visible_wheel_continuity import feedback


def observed(yaw=6):
    b = TurnBuildup()
    b.note_sent(1, (90+yaw, 90-yaw), 10., 0.)
    phases = []
    for i in range(1, 13):
        t = 10.+i*.05
        result = b.adjust(90, yaw, image_intent(t, 1200+i), feedback(t, 30, 30),
                          t, 1, limit=10.)
        phases.append((t, result[2]))
        b.note_sent(1, (90+yaw, 90-yaw), t, t-.05)
    return b, phases


def test_production_waits_500ms_and_two_independent_feedbacks():
    b, phases = observed()
    assert all(p == 'buildup_motor_response_wait' for t,p in phases if t < 10.5)
    assert b.started >= 10.55
    assert phases[-1][1] == 'buildup_yaw_only'


def test_at_limit_no_extra_yaw_and_no_common_speed_loss():
    b, _ = observed(10)
    assert b.adjust(90, 10, image_intent(10.65, 1214), feedback(10.65, 30, 30),
                    10.65, 1, limit=10.) == (90, 10, 'buildup_yaw_only')


def test_marginal_visual_pause_does_not_close_or_extend_episode():
    b, _ = observed()
    origin = b.started
    late = replace(image_intent(10.65, 1214), capture_timestamp=10.39)
    assert b.adjust(90, 6, late, feedback(10.65, 30, 30), 10.65, 1, limit=10.)[2] == 'buildup_wait_fresh_visual'
    assert not b.closed and b.started == origin
    b.note_sent(1, (96,84), 10.65, 10.6)
    assert b.adjust(90, 6, image_intent(10.7, 1215), feedback(10.7,30,30),10.7,1,limit=10.)[:2] == (90,8)
    assert b.started == origin
    assert b.adjust(90,6,image_intent(origin+.36,1220),feedback(origin+.36,30,30),
                    origin+.36,1,limit=10.)[2] == 'buildup_timeout'


@pytest.mark.parametrize('change', ['uid','zero','missing_write'])
def test_motor_wait_does_not_survive_lost_command_provenance(change):
    b, _ = observed()
    if change == 'uid':
        b.adjust(90,6,replace(image_intent(10.65,1214),target_id=2),feedback(10.65,30,30),10.65,2,limit=10.)
        assert not b.history and b.response_delay_sec == .5
    elif change == 'zero':
        b.note_sent(1,(0,0),10.65,10.6)
        assert b.started is None
    else:
        b.sync_writer(0.)
        assert not b.history and b.closed


def test_real_writer_default_waits_without_inserting_zeros(monkeypatch, caplog):
    r,o,d,_,clock = setup(monkeypatch)
    # setup's historical comparator explicitly opts into 100ms. Restore the
    # actual production default and drive repeated fresh captures/feedback.
    r._turn_buildup = TurnBuildup()
    with caplog.at_level('INFO'):
        for i in range(15):
            t = 10.+i*.05; clock[0]=t
            o._lateral_intent_store.publish(image_intent(t,1200+i))
            r._send_follow_wheel_targets(92,-72,'FOLLOW20')
            if i < 10: assert r._turn_buildup.started is None
    assert not d.stops and all(p == (92,-72) for p in d.pairs)
    assert 'response_phase=buildup_motor_response_wait' in caplog.text
    assert r._turn_buildup.started >= 10.5
