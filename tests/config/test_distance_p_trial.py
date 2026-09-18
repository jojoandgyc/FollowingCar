import os
from pathlib import Path
import pytest
from car_control_modular.config_loader import load_config_to_env

CONFIG=Path(__file__).resolve().parents[2]/'car_control_modular/config/reid_runtime.ini'


@pytest.mark.parametrize('trial',['24','27','36'])
def test_explicit_p_trial_changes_only_longitudinal_p(monkeypatch,trial):
    env={}
    monkeypatch.setattr(os,'environ',env)
    load_config_to_env(str(CONFIG))
    before=dict(env)
    env.clear()
    env['FOLLOW_DISTANCE_P_TRIAL']=trial
    load_config_to_env(str(CONFIG))
    after=dict(env)
    after.pop('FOLLOW_DISTANCE_P_TRIAL')
    assert after.pop('DISTANCE_PID_KP_RPM_PER_M')==trial
    assert before.pop('DISTANCE_PID_KP_RPM_PER_M')=='24.0'
    assert after==before


@pytest.mark.parametrize('value',['28','nan','0','200','-1'])
def test_invalid_trial_rejected_before_environment_mapping(monkeypatch,value):
    env={'FOLLOW_DISTANCE_P_TRIAL':value}
    monkeypatch.setattr(os,'environ',env)
    with pytest.raises(ValueError,match='FOLLOW_DISTANCE_P_TRIAL'):
        load_config_to_env(str(CONFIG))
    assert env=={'FOLLOW_DISTANCE_P_TRIAL':value}


@pytest.mark.parametrize('trial',['0','5','10'])
def test_bias_trial_changes_only_command_bias_not_p_or_wheel_scale(monkeypatch,trial):
    env={}
    monkeypatch.setattr(os,'environ',env)
    load_config_to_env(str(CONFIG)); before=dict(env)
    env.clear();env['FOLLOW_MATCHING_BIAS_TRIAL']=trial
    load_config_to_env(str(CONFIG));after=dict(env)
    after.pop('FOLLOW_MATCHING_BIAS_TRIAL')
    assert after.pop('DISTANCE_MATCHING_TEST_BIAS_RPM')==trial
    assert before.pop('DISTANCE_MATCHING_TEST_BIAS_RPM')=='0'
    assert before==after


@pytest.mark.parametrize('trial',['-5','20','nan','5.0'])
def test_invalid_bias_trial_is_not_silently_clamped(monkeypatch,trial):
    monkeypatch.setattr(os,'environ',{'FOLLOW_MATCHING_BIAS_TRIAL':trial})
    with pytest.raises(ValueError,match='FOLLOW_MATCHING_BIAS_TRIAL'):
        load_config_to_env(str(CONFIG))
